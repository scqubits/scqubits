import numpy as np
from scipy import sparse
from typing import Any, Dict, Optional, Tuple
import scqubits.core.qubit_base as base
from scqubits.core import descriptors
import scqubits.core.discretization as discretization
from numpy import ndarray
import scqubits.utils.spectrum_utils as utils
import scqubits.io_utils.fileio_serializers as serializers
from scipy.sparse import csc_matrix


class Snail(base.QubitBaseClass, serializers.Serializable):
    r"""SNAIL

    | [1] Frattini et al., Appl. Phys Lett. 110, 222603 (2017). https://doi.org/10.1063/1.4984142
    | [2] Orlando et al., Physical Review B, 60, 15398 (1999).
          https://link.aps.org/doi/10.1103/PhysRevB.60.15398

    The Superconducting Nonlinear Asymmetric Inductive eLement (SNAIL) qubit consists
    of a superconducting loop of n large Josephson Junctions and one smaller
    junction. It was designed in order to obtain :math:`\textbf{\phi}^3` nonlinearity
    from a dipole element as opposed to previous attempts (such as the Josephson ring
    modulator) that relied on a quadrupole element, and it can best be understood as an extension of the flux qubit
    (first defined in [2]) to n large Josephson junctions instead of 2. The greatest number of large junctions
    that has been experimentally realized in a SNAIL qubit is 3. Typically, one assumes
    :math:`E_{J1}=E_{J2}=E_{J3}=E_J` and :math:`E_{J4}=\alpha E_J`. In the case of 3 Josephson Junctions, the
    Hamiltonian is given by

    .. math::

        H_\text{snail}=&(n_i-n_{gi})4(E_C)_{ij}(n_j-n_{gj}) \\
                      -&E_J\cos(\phi_1)-E_J\cos(\phi_2-\phi_1)-E_J\cos(\phi_3-\phi_2) \\
                      -&\alpha E_J\cos(2\pi f-\phi_3),

    where :math:`i,j\in\{1,2,3\}` is represented in the charge basis for the three degrees
    of freedom. Initialize with, for example::

        EJ = 35.0
        alpha = 0.6
        snail_qubit = scq.Snail(EJ1 = EJ, EJ2 = EJ, EJ3 = EJ, EJ4 = alpha * EJ,
                                     ECJ1 = 1.0, ECJ2 = 1.0, ECJ3 = 1.0, ECJ4 = 1.0 / alpha,
                                     ECg1 = 50.0, ECg2 = 50.0, ECg3 = 50.0, ng1 = 0.0,
                                     ng2 = 0.0, ng3 = 0.0, flux = 0.5, ncut = 10).

    Parameters
    __________
    EJ1, EJ2, EJ3, EJ4: float
        Josephson energy of the ith junction
        `EJ1 = EJ2 = EJ3`, with `EJ3 = alpha * EJ1` and `alpha <= 1`
    ECJ1, ECJ2, ECJ3, ECJ4: float
        charging energy associated with the ith junction
    ECg1, ECg2, ECg3, ECg4: float
        charging energy associated with the capacative coupling to ground for the
        three islands
    ng1, ng2, ng3: float
        offset charge associated with island i
    flux: float
        magnetic flux through the circuit loop, measured in units of the flux quantum
    ncut: int
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
    ECJ1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ4 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
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
        ECJ1: float,
        ECJ2: float,
        ECJ3: float,
        ECJ4: float,
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
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ3 = EJ3
        self.EJ4 = EJ4
        self.ECJ1 = ECJ1
        self.ECJ2 = ECJ2
        self.ECJ3 = ECJ3
        self.ECJ4 = ECJ4
        self.ECg1 = ECg1
        self.ECg2 = ECg2
        self.ECg3 = ECg3
        self.ng1 = ng1
        self.ng2 = ng2
        self.ng3 = ng3
        self.flux = flux
        self.ncut = ncut
        self.truncated_dim = truncated_dim

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ1": 1.0,
            "EJ2": 1.0,
            "EJ3": 1.0,
            "EJ4": 0.7,
            "ECJ1": 1,
            "ECJ2": 1,
            "ECJ3": 1,
            "ECJ4": 1 / 0.7,
            "ECg1": 10.0,
            "ECg2": 10.0,
            "ECg3": 10.0,
            "ng1": 0.0,
            "ng2": 0.0,
            "ng3": 0.0,
            "flux": 0.41,
            "ncut": 10,
            "truncated_dim": 10,
        }

    @classmethod
    def create(cls) -> "Snail":
        phi_grid = discretization.Grid1d(-19.0, 19.0, 200)
        init_params = cls.default_params()
        init_params["grid"] = phi_grid
        snail = cls(**init_params)
        snail.widget()
        return snail

    def EC_matrix(self) -> ndarray:
        """Return the charging energy matrix"""
        CJ1 = 1.0 / (2 * self.ECJ1)  # capacitances in units where e is set to 1
        CJ2 = 1.0 / (2 * self.ECJ2)
        CJ3 = 1.0 / (2 * self.ECJ3)
        CJ4 = 1.0 / (2 * self.ECJ4)
        Cg1 = 1.0 / (2 * self.ECg1)
        Cg2 = 1.0 / (2 * self.ECg2)
        Cg3 = 1.0 / (2 * self.ECg3)
        Cmat = np.array(
            [
                [CJ1 + CJ2 + Cg1, -CJ2, 0],
                [-CJ2, CJ2 + CJ3 + Cg2, -CJ3],
                [0, -CJ3, CJ3 + CJ4 + Cg3],
            ]
        )
        return np.linalg.inv(Cmat) / 2.0

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = utils.eigsh_safe(
            hamiltonian_mat, which="SM", k=evals_count, return_eigenvectors=False
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = utils.eigsh_safe(
            hamiltonian_mat, eigvals=(0, evals_count - 1), eigvals_only=False
        )
        evals, evecs = utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.ncut + 1) ** 3

    def potential(self, phi1: ndarray, phi2: ndarray, phi3: ndarray) -> ndarray:
        """Return value of the potential energy at phi1 and phi2 and phi3, disregarding
        constants."""
        return (
            -self.EJ1 * np.cos(phi1)
            - self.EJ2 * np.cos(phi2 - phi1)
            - self.EJ3 * np.cos(phi3 - phi2)
            - self.EJ4 * np.cos(2.0 * np.pi * self.flux - phi3)
        )

    def kineticmat(self) -> csc_matrix:
        """Return the kinetic energy matrix."""
        ec = self.EC_matrix()
        identity = sparse.identity(2 * self.ncut + 1, format="csc")

        n1 = np.arange(-self.ncut, self.ncut + 1, 1)
        n1 = sparse.diags(n1).tocsc()
        n1 = sparse.kron(
            sparse.kron(n1, identity, format="csc"), identity, format="csc"
        )

        n2 = np.arange(-self.ncut, self.ncut + 1, 1)
        n2 = sparse.diags(n2).tocsc()
        n2 = sparse.kron(
            sparse.kron(identity, n2, format="csc"), identity, format="csc"
        )

        n3 = np.arange(-self.ncut, self.ncut + 1, 1)
        n3 = sparse.diags(n3).tocsc()
        n3 = sparse.kron(
            sparse.kron(identity, identity, format="csc"), n3, format="csc"
        )

        nvec = np.array([n1, n2, n3])

        return 4 * nvec.T @ ec @ nvec

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
                sparse.kron(e_positive_phi + e_negative_phi, identity, format="csc"),
                identity,
                format="csc",
            )
        )
        potential_mat += (
            -0.5
            * self.EJ4
            * sparse.kron(
                sparse.kron(identity, identity, format="csc"),
                np.exp(-1j * 2 * np.pi * self.flux) * e_positive_phi
                + np.exp(+1j * 2 * np.pi * self.flux) * e_negative_phi,
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
            * self.EJ2
            * sparse.kron(
                sparse.kron(e_negative_phi, e_positive_phi, format="csc"),
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
            * self.EJ3
            * sparse.kron(
                sparse.kron(identity, e_negative_phi, format="csc"),
                e_positive_phi,
                format="csc",
            )
        )
        return potential_mat

    def hamiltonian(self) -> csc_matrix:
        """Return Hamiltonian in basis obtained by employing charge basis for both
        degrees of freedom"""
        return self.kineticmat() + self.potentialmat()

    def _n_operator(self) -> ndarray:
        diag_elements = np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_)
        return np.diag(diag_elements)

    def _exp_i_phi_operator(self) -> ndarray:
        dim = self.hilbertdim()
        off_diag_elements = np.ones(dim - 1, dtype=np.complex_)
        e_iphi_matrix = np.diag(off_diag_elements, k=1)
        return e_iphi_matrix

    def _identity(self) -> ndarray:
        dim = self.hilbertdim()
        return np.eye(dim)

    def n_1_operator(self) -> csc_matrix:
        r"""Return charge number operator conjugate to :math:`\phi_1`"""
        return sparse.kron(
            sparse.kron(self._n_operator(), self._identity(), format="csc"),
            self._identity,
            format="csc",
        )

    def n_2_operator(self) -> csc_matrix:
        r"""Return charge number operator conjugate to :math:`\phi_2`"""
        return sparse.kron(
            sparse.kron(self._identity(), self._n_operator(), format="csc"),
            self._identity,
            format="csc",
        )

    def n_3_operator(self) -> csc_matrix:
        r"""Return charge number operator conjugate to :math:`\phi_3`"""
        return sparse.kron(
            sparse.kron(self._identity(), self._identity(), format="csc"),
            self._n_operator,
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
