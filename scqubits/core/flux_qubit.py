# flux_qubit.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.noise import NOISE_PARAMS, NoisySystem


# -Flux qubit noise class
class NoisyFluxQubit(NoisySystem, ABC):
    @abstractmethod
    def d_hamiltonian_d_EJ1(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        pass

    @abstractmethod
    def d_hamiltonian_d_EJ2(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        pass

    @abstractmethod
    def d_hamiltonian_d_EJ3(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        pass

    @classmethod
    @abstractmethod
    def supported_noise_channels(cls) -> List[str]:
        pass

    def tphi_1_over_f_cc1(
        self,
        A_noise: float = NOISE_PARAMS["A_cc"],
        i: int = 0,
        j: int = 1,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        Calculate the 1/f dephasing time (or rate) due to critical current noise of
        junction associated with Josephson energy :math:`EJ1`.

        Parameters
        ----------
        A_noise:
            noise strength
        i:
            state index that along with j defines a qubit
        j:
            state index that along with i defines a qubit
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`,
             or rate in inverse units.

        """
        if "tphi_1_over_f_cc1" not in self.supported_noise_channels():
            raise RuntimeError(
                "Critical current noise channel 'tphi_1_over_f_cc1' is not supported in"
                " this system."
            )

        return self.tphi_1_over_f(
            A_noise=A_noise,
            i=i,
            j=j,
            noise_op=self.d_hamiltonian_d_EJ1(),
            esys=esys,
            get_rate=get_rate,
            **kwargs
        )

    def tphi_1_over_f_cc2(
        self,
        A_noise: float = NOISE_PARAMS["A_cc"],
        i: int = 0,
        j: int = 1,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        Calculate the 1/f dephasing time (or rate) due to critical current noise of
        junction associated with Josephson energy :math:`EJ2`.

        Parameters
        ----------
        A_noise:
            noise strength
        i:
            state index that along with j defines a qubit
        j:
            state index that along with i defines a qubit
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
            :math:`T_{\phi}` time or rate:
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.
        """
        if "tphi_1_over_f_cc2" not in self.supported_noise_channels():
            raise RuntimeError(
                "Critical current noise channel 'tphi_1_over_f_cc2' is not supported in"
                " this system."
            )

        return self.tphi_1_over_f(
            A_noise=A_noise,
            i=i,
            j=j,
            noise_op=self.d_hamiltonian_d_EJ2(),
            esys=esys,
            get_rate=get_rate,
            **kwargs
        )

    def tphi_1_over_f_cc3(
        self,
        A_noise: float = NOISE_PARAMS["A_cc"],
        i: int = 0,
        j: int = 1,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        Calculate the 1/f dephasing time (or rate) due to critical current noise of junction associated with
        Josephson energy :math:`EJ3`.

        Parameters
        ----------
        A_noise:
            noise strength
        i:
            state index that along with j defines a qubit
        j:
            state index that along with i defines a qubit
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.
        """
        if "tphi_1_over_f_cc3" not in self.supported_noise_channels():
            raise RuntimeError(
                "Critical current noise channel 'tphi_1_over_f_cc3' is not supported in"
                " this system."
            )

        return self.tphi_1_over_f(
            A_noise=A_noise,
            i=i,
            j=j,
            noise_op=self.d_hamiltonian_d_EJ3(),
            esys=esys,
            get_rate=get_rate,
            **kwargs
        )

    def tphi_1_over_f_cc(
        self,
        A_noise: float = NOISE_PARAMS["A_cc"],
        i: int = 0,
        j: int = 1,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        Calculate the 1/f dephasing time (or rate) due to critical-current noise
        from all three Josephson junctions :math:`EJ1`, :math:`EJ2` and :math:`EJ3`.
        The combined noise is calculated by summing the rates from the individual
        contributions.

        Parameters
        ----------
        A_noise:
            noise strength
        i:
            state index that along with j defines a qubit
        j:
            state index that along with i defines a qubit
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
            decoherence time in units of :math:`2\pi` (system units),
            or rate in inverse units.
        """
        if "tphi_1_over_f_cc" not in self.supported_noise_channels():
            raise RuntimeError(
                "Critical current noise channel 'tphi_1_over_f_cc' is not supported in"
                " this system."
            )

        rate = self.tphi_1_over_f_cc1(
            A_noise=A_noise, i=i, j=j, esys=esys, get_rate=True, **kwargs
        )
        rate += self.tphi_1_over_f_cc2(
            A_noise=A_noise, i=i, j=j, esys=esys, get_rate=True, **kwargs
        )
        rate += self.tphi_1_over_f_cc3(
            A_noise=A_noise, i=i, j=j, esys=esys, get_rate=True, **kwargs
        )
        if get_rate:
            return rate
        else:
            return 1 / rate if rate != 0 else np.inf


# -Flux qubit, both degrees of freedom in charge basis---------------------------------


class FluxQubit(base.QubitBaseClass, serializers.Serializable, NoisyFluxQubit):
    r"""Flux Qubit

    | [1] Orlando et al., Physical Review B, 60, 15398 (1999).
          https://link.aps.org/doi/10.1103/PhysRevB.60.15398

    The original flux qubit as defined in [1], where the junctions are allowed to
    have varying junction energies and capacitances to allow for junction asymmetry.
    Typically, one takes :math:`E_{J1}=E_{J2}=E_J`, and :math:`E_{J3}=\alpha E_J`
    where :math:`0\le \alpha \le 1`. The same relations typically hold for the
    junction capacitances. The Hamiltonian is given by

    .. math::

       H_\text{flux}=&(n_{i}-n_{gi})4(E_\text{C})_{ij}(n_{j}-n_{gj}) \\
                    -&E_{J}\cos\phi_{1}-E_{J}\cos\phi_{2}-\alpha E_{J}\cos(2\pi f
                    + \phi_{1} - \phi_{2}),

    where :math:`i,j\in\{1,2\}` is represented in the charge basis for both degrees
    of freedom. Initialize with, for example::

        EJ = 35.0
        alpha = 0.6
        flux_qubit = scq.FluxQubit(EJ1 = EJ, EJ2 = EJ, EJ3 = alpha*EJ,
                                     ECJ1 = 1.0, ECJ2 = 1.0, ECJ3 = 1.0/alpha,
                                     ECg1 = 50.0, ECg2 = 50.0, ng1 = 0.0, ng2 = 0.0,
                                     flux = 0.5, ncut = 10)

    Parameters
    ----------
    EJ1, EJ2, EJ3: float
        Josephson energy of the ith junction
        `EJ1 = EJ2`, with `EJ3 = alpha * EJ1` and `alpha <= 1`
    ECJ1, ECJ2, ECJ3: float
        charging energy associated with the ith junction
    ECg1, ECg2: float
        charging energy associated with the capacitive coupling to ground for the
        two islands
    ng1, ng2: float
        offset charge associated with island i
    flux: float
        magnetic flux through the circuit loop, measured in units of the flux quantum
    ncut: int
        charge number cutoff for the charge on both islands `n`,  `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    esys_method: 
        method for esys diagonalization, callable or string representation 
    esys_method_options: 
        dictionary with esys diagonalization options 
    evals_method: 
        method for evals diagonalization, callable or string representation 
    evals_method_options: 
        dictionary with evals diagonalization options 
    """

    EJ1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EJ2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EJ3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ3 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECg1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECg2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ1: float,
        EJ2: float,
        EJ3: float,
        ECJ1: float,
        ECJ2: float,
        ECJ3: float,
        ECg1: float,
        ECg2: float,
        ng1: float,
        ng2: float,
        flux: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ3 = EJ3
        self.ECJ1 = ECJ1
        self.ECJ2 = ECJ2
        self.ECJ3 = ECJ3
        self.ECg1 = ECg1
        self.ECg2 = ECg2
        self.ng1 = ng1
        self.ng2 = ng2
        self.flux = flux
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(
            -np.pi / 2, 3 * np.pi / 2, 100
        )  # for plotting in phi_j basis

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ1": 1.0,
            "EJ2": 1.0,
            "EJ3": 0.8,
            "ECJ1": 0.016,
            "ECJ2": 0.016,
            "ECJ3": 0.021,
            "ECg1": 0.83,
            "ECg2": 0.83,
            "ng1": 0.0,
            "ng2": 0.0,
            "flux": 0.4,
            "ncut": 10,
            "truncated_dim": 10,
        }

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc1",
            "tphi_1_over_f_cc2",
            "tphi_1_over_f_cc3",
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            # 'tphi_1_over_f_ng1',
            # 'tphi_1_over_f_ng2',
            # 'tphi_1_over_f_ng',
        ]

    def EC_matrix(self) -> ndarray:
        """Return the charging energy matrix"""
        Cmat = np.zeros((2, 2))
        CJ1 = 1.0 / (2 * self.ECJ1)  # capacitances in units where e is set to 1
        CJ2 = 1.0 / (2 * self.ECJ2)
        CJ3 = 1.0 / (2 * self.ECJ3)
        Cg1 = 1.0 / (2 * self.ECg1)
        Cg2 = 1.0 / (2 * self.ECg2)

        Cmat[0, 0] = CJ1 + CJ3 + Cg1
        Cmat[1, 1] = CJ2 + CJ3 + Cg2
        Cmat[0, 1] = -CJ3
        Cmat[1, 0] = -CJ3

        return np.linalg.inv(Cmat) / 2.0

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(
            hamiltonian_mat,
            subset_by_index=(0, evals_count - 1),
            eigvals_only=True,
            check_finite=False,
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(
            hamiltonian_mat,
            subset_by_index=(0, evals_count - 1),
            eigvals_only=False,
            check_finite=False,
        )
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.ncut + 1) ** 2

    def potential(self, phi1: ndarray, phi2: ndarray) -> ndarray:
        """Return value of the potential energy at phi1 and phi2, disregarding
        constants."""
        return (
            -self.EJ1 * np.cos(phi1)
            - self.EJ2 * np.cos(phi2)
            - self.EJ3 * np.cos(2.0 * np.pi * self.flux + phi1 - phi2)
        )

    def kineticmat(self) -> ndarray:
        """Return the kinetic energy matrix."""
        ECmat = self.EC_matrix()

        kinetic_mat = (
            4.0
            * ECmat[0, 0]
            * np.kron(
                np.matmul(
                    self._n_operator() - self.ng1 * self._identity(),
                    self._n_operator() - self.ng1 * self._identity(),
                ),
                self._identity(),
            )
        )
        kinetic_mat += (
            4.0
            * ECmat[1, 1]
            * np.kron(
                self._identity(),
                np.matmul(
                    self._n_operator() - self.ng2 * self._identity(),
                    self._n_operator() - self.ng2 * self._identity(),
                ),
            )
        )
        kinetic_mat += (
            4.0
            * (ECmat[0, 1] + ECmat[1, 0])
            * np.kron(
                self._n_operator() - self.ng1 * self._identity(),
                self._n_operator() - self.ng2 * self._identity(),
            )
        )
        return kinetic_mat

    def potentialmat(self) -> ndarray:
        """Return the potential energy matrix for the potential."""
        potential_mat = (
            -0.5
            * self.EJ1
            * np.kron(
                self._exp_i_phi_operator() + self._exp_i_phi_operator().T,
                self._identity(),
            )
        )
        potential_mat += (
            -0.5
            * self.EJ2
            * np.kron(
                self._identity(),
                self._exp_i_phi_operator() + self._exp_i_phi_operator().T,
            )
        )
        potential_mat += (
            -0.5
            * self.EJ3
            * (
                np.exp(1j * 2 * np.pi * self.flux)
                * np.kron(self._exp_i_phi_operator(), self._exp_i_phi_operator().T)
            )
        )
        potential_mat += (
            -0.5
            * self.EJ3
            * (
                np.exp(-1j * 2 * np.pi * self.flux)
                * np.kron(self._exp_i_phi_operator().T, self._exp_i_phi_operator())
            )
        )
        return potential_mat

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Return Hamiltonian in the basis obtained by employing charge basis for both
        degrees of freedom or in the eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the basis obtained by employing charge basis for both degrees of freedom.
            If `True`, the energy eigenspectrum is computed, returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns Hamiltonian in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, the Hamiltonian has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, Hamiltonian has dimensions of m x m,
            for m given eigenvectors.
        """
        hamiltonian_mat = self.kineticmat() + self.potentialmat()
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )

    def d_hamiltonian_d_EJ1(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJ1 in the native Hamiltonian basis or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the native Hamiltonian basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -0.5 * np.kron(
            self._exp_i_phi_operator() + self._exp_i_phi_operator().T, self._identity()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ2(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJ2 in the native Hamiltonian basis or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the native Hamiltonian basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -0.5 * np.kron(
            self._identity(), self._exp_i_phi_operator() + self._exp_i_phi_operator().T
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ3(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJ3 in the native Hamiltonian basis or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the native Hamiltonian basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = (
            -0.5
            * (
                np.exp(1j * 2 * np.pi * self.flux)
                * np.kron(self._exp_i_phi_operator(), self._exp_i_phi_operator().T)
            )
        ) + (
            -0.5
            * (
                np.exp(-1j * 2 * np.pi * self.flux)
                * np.kron(self._exp_i_phi_operator().T, self._exp_i_phi_operator())
            )
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_flux(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns the operator representing a derivative of the Hamiltonian with respect to flux
        in the native Hamiltonian basis or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the native Hamiltonian basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = (
            2j
            * np.pi
            * (
                -0.5
                * self.EJ3
                * (
                    np.exp(1j * 2 * np.pi * self.flux)
                    * np.kron(self._exp_i_phi_operator(), self._exp_i_phi_operator().T)
                )
                + 0.5
                * self.EJ3
                * (
                    np.exp(-1j * 2 * np.pi * self.flux)
                    * np.kron(self._exp_i_phi_operator().T, self._exp_i_phi_operator())
                )
            )
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _n_operator(self) -> ndarray:
        diag_elements = np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_)
        return np.diag(diag_elements)

    def _exp_i_phi_operator(self) -> ndarray:
        dim = 2 * self.ncut + 1
        off_diag_elements = np.ones(dim - 1, dtype=np.complex_)
        e_iphi_matrix = np.diag(off_diag_elements, k=-1)
        return e_iphi_matrix

    def _identity(self) -> ndarray:
        dim = 2 * self.ncut + 1
        return np.eye(dim)

    def n_1_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        r"""
        Returns the charge number operator conjugate to :math:`\phi_1` in the charge? or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the charge number operator conjugate to :math:`\phi_1` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns the charge number operator conjugate to :math:`\phi_1` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns the charge number operator conjugate to :math:`\phi_1` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Charge number operator conjugate to :math:`\phi_1` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m, for m given eigenvectors.
        """
        native = np.kron(self._n_operator(), self._identity())
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def n_2_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        r"""
        Returns the charge number operator conjugate to :math:`\phi_2` in the charge? or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the charge number operator conjugate to :math:`\phi_2` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns the charge number operator conjugate to :math:`\phi_2` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns the charge number operator conjugate to :math:`\phi_2` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Charge number operator conjugate to :math:`\phi_2` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m, for m given eigenvectors.
        """
        native = np.kron(self._identity(), self._n_operator())
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def exp_i_phi_1_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        r"""
        Returns operator :math:`e^{i\phi_1}` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`e^{i\phi_1}` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`e^{i\phi_1}` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`e^{i\phi_1}` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`e^{i\phi_1}` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`e^{i\phi_1}` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`e^{i\phi_1}` has dimensions of m x m,
            for m given eigenvectors.
        """
        native = np.kron(self._exp_i_phi_operator(), self._identity())
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def exp_i_phi_2_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        r"""
        Returns operator :math:`e^{i\phi_2}` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`e^{i\phi_2}` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`e^{i\phi_2}` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`e^{i\phi_2}` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`e^{i\phi_2}` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`e^{i\phi_2}` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`e^{i\phi_2}` has dimensions of m x m,
            for m given eigenvectors.
        """
        native = np.kron(self._identity(), self._exp_i_phi_operator())
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def cos_phi_1_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\phi_1` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\phi_1` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\phi_1` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\phi_1` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\phi_1` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\phi_1` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\phi_1` has dimensions of m x m,
            for m given eigenvectors.
        """
        cos_op = 0.5 * self.exp_i_phi_1_operator()
        cos_op += cos_op.T
        native = cos_op
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def cos_phi_2_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\phi_2` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\phi_2` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\phi_2` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\phi_2` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\phi_2` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\phi_2` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\phi_2` has dimensions of m x m,
            for m given eigenvectors.
        """
        cos_op = 0.5 * self.exp_i_phi_2_operator()
        cos_op += cos_op.T
        native = cos_op
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def sin_phi_1_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\sin \\phi_1` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\sin \\phi_1` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\sin \\phi_1` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\sin \\phi_1` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\sin \\phi_1` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\sin \\phi_1` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\sin \\phi_1` has dimensions of m x m,
            for m given eigenvectors.
        """
        sin_op = -1j * 0.5 * self.exp_i_phi_1_operator()
        sin_op += sin_op.conj().T
        native = sin_op
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def sin_phi_2_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\sin \\phi_2` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\sin \\phi_2` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\sin \\phi_1` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\sin \\phi_1` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\sin \\phi_2` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\sin \\phi_2` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\sin \\phi_2` has dimensions of m x m,
            for m given eigenvectors.
        """
        sin_op = -1j * 0.5 * self.exp_i_phi_2_operator()
        sin_op += sin_op.conj().T
        native = sin_op
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def plot_potential(
        self,
        phi_grid: discretization.Grid1d = None,
        contour_vals: ndarray = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        contour_vals:
            specific contours to draw
        **kwargs:
            plot options
        """
        phi_grid = phi_grid or self._default_grid
        x_vals = y_vals = phi_grid.make_linspace()
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5, 5)
        return plot.contours(
            x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs
        )

    def wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        phi_grid: discretization.Grid1d = None,
    ) -> storage.WaveFunctionOnGrid:
        """
        Return a flux qubit wave function in phi1, phi2 basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        which:
            index of desired wave function (default value = 0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count=evals_count)
        else:
            _, evecs = esys
        phi_grid = phi_grid or self._default_grid

        dim = 2 * self.ncut + 1
        state_amplitudes = np.reshape(evecs[:, which], (dim, dim))

        n_vec = np.arange(-self.ncut, self.ncut + 1)
        phi_vec = phi_grid.make_linspace()
        a_1_phi = np.exp(1j * np.outer(phi_vec, n_vec)) / (2 * np.pi) ** 0.5
        a_2_phi = a_1_phi.T
        wavefunc_amplitudes = np.matmul(a_1_phi, state_amplitudes)
        wavefunc_amplitudes = np.matmul(wavefunc_amplitudes, a_2_phi)
        wavefunc_amplitudes = spec_utils.standardize_phases(wavefunc_amplitudes)

        grid2d = discretization.GridSpec(
            np.asarray(
                [
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                ]
            )
        )
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        phi_grid: discretization.Grid1d = None,
        mode: str = "abs",
        zero_calibrate: bool = True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which:
            index of wave function to be plotted (default value = (0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        mode:
            choices as specified in `constants.MODE_FUNC_DICT`
            (default value = 'abs_sqr')
        zero_calibrate:
            if True, colors are adjusted to use zero wavefunction amplitude as the
            neutral color in the palette
        **kwargs:
            plot options
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5, 5)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
