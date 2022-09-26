from abc import ABC
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from scipy import sparse

from numpy import ndarray
from scipy.sparse import csc_matrix

import scqubits.core.qubit_base as base

import scqubits.core.descriptors as descriptors
import scqubits.utils.spectrum_utils as utils
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.noise import NoisySystem


class NoisySnailmon(NoisySystem, ABC):
    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return ["tphi_1_over_f_flux"]


class Snailmon(base.QubitBaseClass, serializers.Serializable, NoisySnailmon):
    r"""SNAIL

    | [1] Frattini et al., Appl. Phys Lett. 110, 222603 (2017). https://doi.org/10.1063/1.4984142
    | [2] Orlando et al., Physical Review B, 60, 15398 (1999).
          https://link.aps.org/doi/10.1103/PhysRevB.60.15398

    The Superconducting Nonlinear Asymmetric Inductive eLement (SNAIL) qubit consists
    of a superconducting loop of n large Josephson Junctions and one smaller
    junction. It was designed in order to obtain :math:`\textbf{\phi}^3` nonlinearity
    from a dipole element as opposed to previous attempts (such as the Josephson ring
    modulator) that relied on a quadrupole element, and it can best be understood as an
    extension of the flux qubit (first defined in [2]) to n large Josephson junctions
    instead of 2. The greatest number of large junctions that has been experimentally
    realized in a SNAIL qubit is 3. Typically, one assumes
    :math:`E_{J1}=E_{J2}=E_{J3}=E_J` and :math:`E_{J4}=\alpha E_J`.
    In the case of 3 Josephson junctions, the Hamiltonian is given by

    .. math::

        H_\text{snail}=&(n_i-n_{gi})4(E_C)_{ij}(n_j-n_{gj}) \\
                      -&E_J\cos(\phi_1)-E_J\cos(\phi_2-\phi_1)-E_J\cos(\phi_3-\phi_2) \\
                      -&\alpha E_J\cos(2\pi f-\phi_3),

    where :math:`i,j\in\{1,2,3\}` is represented in the charge basis for the three
    degrees of freedom. Initialize with, for example::

        EJ = 35.0
        alpha = 0.6
        snail_qubit = scq.Snail(EJ1 = EJ, EJ2 = EJ, EJ3 = EJ, EJ4 = alpha * EJ,
                                     EC1 = 1.0, EC2 = 1.0, EC3 = 1.0, EC4 = 1.0 / alpha,
                                     ECg1 = 50.0, ECg2 = 50.0, ECg3 = 50.0, ng1 = 0.0,
                                     ng2 = 0.0, ng3 = 0.0, flux = 0.5, ncut = 10).

    Parameters
    __________
    EJ1, EJ2, EJ3, EJ4: float
        Josephson energy of the ith junction
        `EJ1 = EJ2 = EJ3`, with `EJ3 = alpha * EJ1` and `alpha <= 1`
    EC1, EC2, EC3, EC4: float
        charging energy associated with the ith junction, may include parallel shunt
        capacitance
    ECg1, ECg2, ECg3, ECg4: float
        charging energy associated with the capacitive coupling to ground for the
        three islands
    ng1, ng2, ng3: float
        offset charge associated with each island
    flux: float
        magnetic flux through the circuit loop, measured in units of the flux quantum
    ncut: int
        charge number cutoff for the charge `n` on the three islands `n`,
        `n = -ncut, ..., ncut`
        charge number cutoff for the charge on the three islands `n`, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJ1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EJ2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EJ3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EJ4 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC4 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECg1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECg2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECg3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ1: float,
        EJ2: float,
        EJ3: float,
        EJ4: float,
        EC1: float,
        EC2: float,
        EC3: float,
        EC4: float,
        ECg1: float,
        ECg2: float,
        ECg3: float,
        ng1: float,
        ng2: float,
        ng3: float,
        flux: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        # Constant parameters for each Josephson Junction
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ3 = EJ3
        self.EJ4 = EJ4
        # Capacitance connected to each Josephson Junction
        self.EC1 = EC1
        self.EC2 = EC2
        self.EC3 = EC3
        self.EC4 = EC4

        # Capacitance connected to ground
        self.ECg1 = ECg1
        self.ECg2 = ECg2
        self.ECg3 = ECg3

        # offset charges associated with each Josephson Junction
        self.ng1 = ng1
        self.ng2 = ng2
        self.ng3 = ng3
        # flux
        self.flux = flux
        # Truncation dimension
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._image_filename = None

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ1": 887.3,
            "EJ2": 887.3,
            "EJ3": 887.3,
            "EJ4": 117.5,
            "EC1": 0.2873,
            "EC2": 0.2873,
            "EC3": 0.2873,
            "EC4": 1.437,
            "ECg1": 193.7,
            "ECg2": 193.7,
            "ECg3": 193.7,
            "ng1": 0.0,
            "ng2": 0.0,
            "ng3": 0.0,
            "flux": 0.0,
            "ncut": 6,
            "truncated_dim": 6,
        }

    # Construct the Ec matrix, we need this to calculate the kinetic_energy matrix in
    # the Hamiltonian
    def EC_matrix(self) -> ndarray:
        """Returns the charging energy matrix"""
        c1 = 1 / (2 * self.EC1)
        c2 = 1 / (2 * self.EC2)
        c3 = 1 / (2 * self.EC3)
        c4 = 1 / (2 * self.EC4)

        cg1 = 1 / (2 * self.ECg1)
        cg2 = 1 / (2 * self.ECg2)
        cg3 = 1 / (2 * self.ECg3)
        cmat = np.array(
            [
                [c1 + c2 + cg1, -c2, 0],
                [-c2, c2 + c3 + cg2, -c3],
                [0, -c3, c3 + c4 + cg3],
            ]
        )
        ec_matrix = 0.5 * np.linalg.inv(cmat)
        return ec_matrix

    def _evals_calc(
        self, evals_count: int, hamiltonian_mat: csc_matrix = None
    ) -> ndarray:
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals = utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            sigma=0.0,
            which="LM",
            return_eigenvectors=False,
        )
        return np.sort(evals)

    def _esys_calc(
        self, evals_count: int, hamiltonian_mat: csc_matrix = None
    ) -> Tuple[ndarray, ndarray]:
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals, evecs = utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            sigma=0.0,
            which="LM",
            return_eigenvectors=True,
        )
        evals, evecs = utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.ncut + 1) ** 3

    def potential(self, phi1: ndarray, phi2: ndarray, phi3: ndarray) -> ndarray:
        """Returns the value of the potential energy at phi1 and phi2 and phi3,
        disregarding constants."""
        return (
            -self.EJ1 * np.cos(phi1)
            - self.EJ2 * np.cos(phi2 - phi1)
            - self.EJ3 * np.cos(phi3 - phi2)
            - self.EJ4 * np.cos(2.0 * np.pi * self.flux - phi3)
        )

    def kineticmat(self) -> csc_matrix:
        """Return the kinetic energy matrix."""
        ec_mat = self.EC_matrix()
        identity = sparse.identity(2 * self.ncut + 1, format="csc")

        n_op = np.arange(-self.ncut, self.ncut + 1, 1)
        n_op = sparse.diags(n_op).tocsc()

        n1 = sparse.kron(
            sparse.kron(n_op, identity, format="csc"), identity, format="csc"
        )

        n2 = sparse.kron(
            sparse.kron(identity, n_op, format="csc"), identity, format="csc"
        )

        n3 = sparse.kron(
            sparse.kron(identity, identity, format="csc"), n_op, format="csc"
        )

        ng1 = self.ng1 * sparse.kron(
            sparse.kron(identity, identity, format="csc"), identity, format="csc"
        )
        ng2 = self.ng2 * sparse.kron(
            sparse.kron(identity, identity, format="csc"), identity, format="csc"
        )
        ng3 = self.ng3 * sparse.kron(
            sparse.kron(identity, identity, format="csc"), identity, format="csc"
        )

        nvec = np.array([n1 - ng1, n2 - ng2, n3 - ng3])

        return 4 * nvec.T @ ec_mat @ nvec

    def potentialmat(self) -> csc_matrix:
        """Return the potential energy matrix."""
        identity = sparse.identity(2 * self.ncut + 1, format="csc")

        ones_on_diagonal = np.ones((1, 2 * self.ncut))
        e_positive_phi = sparse.diags(ones_on_diagonal, [1]).tocsc()
        e_negative_phi = sparse.diags(ones_on_diagonal, [-1]).tocsc()

        potential_mat = (
            -0.5
            * self.EJ1
            * sparse.kron(
                sparse.kron(e_positive_phi, identity, format="csc"),
                identity,
                format="csc",
            )
        )

        potential_mat += (
            -0.5
            * self.EJ2
            * sparse.kron(
                sparse.kron(e_positive_phi, e_negative_phi, format="csc"),
                identity,
                format="csc",
            )
        )
        potential_mat += (
            -0.5
            * self.EJ3
            * sparse.kron(
                sparse.kron(identity, e_positive_phi, format="csc"),
                e_negative_phi,
                format="csc",
            )
        )
        potential_mat += (
            -0.5
            * self.EJ4
            * sparse.kron(
                sparse.kron(identity, identity, format="csc"),
                np.exp(-1j * 2 * np.pi * self.flux) * e_positive_phi,
                format="csc",
            )
        )
        potential_mat += potential_mat.conjugate().T
        return potential_mat

    def hamiltonian(self) -> csc_matrix:
        """Return Hamiltonian in basis obtained by employing charge basis for both
        degrees of freedom"""
        return self.kineticmat() + self.potentialmat()

    def d_hamiltonian_d_flux(self) -> csc_matrix:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `flux`.
        """
        return (
            2
            * np.pi
            * self.EJ4
            * (
                np.sin(2 * np.pi * self.flux) * self.cos_phi_3_operator()
                - np.cos(2 * np.pi * self.flux) * self.sin_phi_3_operator()
            )
        )

    def _n_operator(self) -> ndarray:
        diag_elements = np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_)
        return np.diag(diag_elements)

    def _exp_i_phi_operator(self) -> ndarray:
        dim = self.hilbertdim()
        off_diag_elements = np.ones(dim - 1, dtype=np.complex_)
        e_iphi_matrix = np.diag(off_diag_elements, k=1)
        return e_iphi_matrix

    def _identity(self) -> ndarray:
        dim = 2 * self.ncut + 1
        return np.eye(dim)

    def n_1_operator(self) -> csc_matrix:
        r"""Return charge number operator conjugate to :math:`\phi_1`"""
        return sparse.kron(
            sparse.kron(self._n_operator(), self._identity(), format="csc"),
            self._identity(),
            format="csc",
        )

    def n_2_operator(self) -> csc_matrix:
        r"""Return charge number operator conjugate to :math:`\phi_2`"""
        return sparse.kron(
            sparse.kron(self._identity(), self._n_operator(), format="csc"),
            self._identity(),
            format="csc",
        )

    def n_3_operator(self) -> csc_matrix:
        r"""Return charge number operator conjugate to :math:`\phi_3`"""
        return sparse.kron(
            sparse.kron(self._identity(), self._identity(), format="csc"),
            self._n_operator(),
            format="csc",
        )

    def exp_i_phi_1_operator(self) -> csc_matrix:
        r"""Return operator :math:`e^{i\phi_1}` in the charge basis."""
        return sparse.kron(
            sparse.kron(self._exp_i_phi_operator(), self._identity(), format="csc"),
            self._identity(),
            format="csc",
        )

    def exp_i_phi_2_operator(self) -> csc_matrix:
        r"""Return operator :math:`e^{i\phi_2}` in the charge basis."""
        return sparse.kron(
            sparse.kron(self._identity(), self._exp_i_phi_operator(), format="csc"),
            self._identity(),
            format="csc",
        )

    def exp_i_phi_3_operator(self) -> csc_matrix:
        r"""Return operator :math:`e^{i\phi_3}` in the charge basis."""
        return sparse.kron(
            sparse.kron(self._identity(), self._identity(), format="csc"),
            self._exp_i_phi_operator(),
            format="csc",
        )

    def cos_phi_1_operator(self) -> csc_matrix:
        """Return operator :math:`\\cos \\phi_1` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_1_operator()
        cos_op += cos_op.T
        return cos_op

    def cos_phi_2_operator(self) -> csc_matrix:
        """Return operator :math:`\\cos \\phi_2` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_2_operator()
        cos_op += cos_op.T
        return cos_op

    def cos_phi_3_operator(self) -> csc_matrix:
        """Return operator :math:`\\cos \\phi_3` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_3_operator()
        cos_op += cos_op.T
        return cos_op

    def sin_phi_1_operator(self) -> csc_matrix:
        """Return operator :math:`\\sin \\phi_1` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_1_operator()
        sin_op += sin_op.conj().T
        return sin_op

    def sin_phi_2_operator(self) -> csc_matrix:
        """Return operator :math:`\\sin \\phi_2` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_2_operator()
        sin_op += sin_op.conj().T
        return sin_op

    def sin_phi_3_operator(self) -> csc_matrix:
        """Return operator :math:`\\sin \\phi_2` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_3_operator()
        sin_op += sin_op.conj().T
        return sin_op
