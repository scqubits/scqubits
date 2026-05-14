# lambda_qubit.py
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

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from numpy import ndarray
from scipy.sparse import csc_matrix

import scqubits.core.descriptors as descriptors
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


class Lambda(base.QubitBaseClass, serializers.Serializable):
    r"""Three-level toy qubit defined directly in its energy eigenbasis.

    The model contains exactly three levels with energies
    :math:`E_0 = 0`, :math:`E_1 = E_{01}`, and :math:`E_2 = E_{02}`.
    The native Hamiltonian is therefore diagonal in this basis.

    The charge operator is parameterized by three matrix elements:
    :math:`\langle 0|\hat n|1\rangle = n_{01}`,
    :math:`\langle 1|\hat n|2\rangle = n_{12}`, and
    :math:`\langle 0|\hat n|2\rangle = n_{02}`.

    The phase operator is derived from the commutation relation

    .. math::

        [\hat H/\hbar, \hat\phi] = -i\,8E_C\,\hat n

    assuming kinetic energy :math:`4E_C \hat n^2`.

    Parameters
    ----------
    E_01:
        Transition energy :math:`E_1 - E_0` in frequency units.
    E_02:
        Transition energy :math:`E_2 - E_0` in frequency units.
    n_01:
        Charge matrix element :math:`\langle 0|\hat n|1\rangle`
        in the native three-level basis.
    n_12:
        Charge matrix element :math:`\langle 1|\hat n|2\rangle`
        in the native three-level basis.
    n_02:
        Charge matrix element :math:`\langle 0|\hat n|2\rangle`
        in the native three-level basis.
    s_01:
        Optional system-bath matrix element :math:`\langle 0|\hat s|1\rangle`.
        If all `s_01`, `s_12`, and `s_02` are omitted,
        :meth:`sys_bath_operator` defaults to :meth:`n_operator`.
    s_12:
        Optional system-bath matrix element :math:`\langle 1|\hat s|2\rangle`.
    s_02:
        Optional system-bath matrix element :math:`\langle 0|\hat s|2\rangle`.
    EC:
        Charging energy entering the commutator relation and kinetic term
        normalization.
    truncated_dim:
        Effective dimension used when transforming operators into the energy basis.
        Since this is a strict three-level model, values must satisfy
        :math:`2 \leq` ``truncated_dim`` :math:`\leq 3`.
    id_str:
        Optional string by which this instance can be referred to in
        :class:`HilbertSpace` and `ParameterSweep`. If not provided, an id is
        auto-generated.
    esys_method:
        Method for esys diagonalization, callable or string representation.
    esys_method_options:
        Dictionary with esys diagonalization options.
    evals_method:
        Method for evals diagonalization, callable or string representation.
    evals_method_options:
        Dictionary with evals diagonalization options.
    """

    E_01 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    E_02 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    n_01 = descriptors.WatchedProperty(complex, "QUANTUMSYSTEM_UPDATE")
    n_12 = descriptors.WatchedProperty(complex, "QUANTUMSYSTEM_UPDATE")
    n_02 = descriptors.WatchedProperty(complex, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        E_01: float,
        E_02: float,
        n_01: complex,
        n_12: complex,
        n_02: complex,
        EC: float,
        s_01: Optional[complex] = None,
        s_12: Optional[complex] = None,
        s_02: Optional[complex] = None,
        truncated_dim: int = 3,
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

        self.E_01 = E_01
        self.E_02 = E_02
        self.n_01 = n_01
        self.n_12 = n_12
        self.n_02 = n_02
        self.EC = EC
        self.s_01 = s_01
        self.s_12 = s_12
        self.s_02 = s_02
        self.truncated_dim = truncated_dim
        if self.truncated_dim != 3:
            raise ValueError("`truncated_dim` must be 3 for Lambda.")

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "E_01": 1.0,
            "E_02": 5.0,
            "n_01": 0.1,
            "n_12": 1.0,
            "n_02": 1.0,
            "EC": 1.0,
            "truncated_dim": 3,
        }

    def hilbertdim(self) -> int:
        """Returns the fixed three-level Hilbert space dimension."""
        return 3

    def _energy_eigenvalues_native(self) -> ndarray:
        return np.asarray([0.0, self.E_01, self.E_02], dtype=float)

    def _evals_calc(self, evals_count: int) -> ndarray:
        """Returns lowest `evals_count` eigenvalues analytically.

        Since the Lambda Hamiltonian is diagonal in the native basis, the
        eigenspectrum is obtained without numerical diagonalization.
        """
        evals = self._energy_eigenvalues_native()
        evals_sorted = np.sort(evals)
        return evals_sorted[:evals_count]

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        """Returns lowest `evals_count` eigenpairs analytically.

        Eigenvectors are columns of the identity matrix in native basis,
        reordered to match ascending eigenvalue order.
        """
        evals = self._energy_eigenvalues_native()
        order = np.argsort(evals)
        evals_sorted = evals[order]
        evecs_sorted = np.eye(self.hilbertdim(), dtype=np.complex128)[:, order]
        return evals_sorted[:evals_count], evecs_sorted[:, :evals_count]

    def _n_operator_native(self) -> ndarray:
        n_op = np.zeros((3, 3), dtype=np.complex128)
        n_op[0, 1] = self.n_01
        n_op[1, 0] = np.conjugate(self.n_01)
        n_op[1, 2] = self.n_12
        n_op[2, 1] = np.conjugate(self.n_12)
        n_op[0, 2] = self.n_02
        n_op[2, 0] = np.conjugate(self.n_02)
        return n_op

    def _s_operator_native(self) -> ndarray:
        if self.s_01 is None and self.s_12 is None and self.s_02 is None:
            return self._n_operator_native()
        if self.s_01 is None or self.s_12 is None or self.s_02 is None:
            raise ValueError(
                "Custom system-bath operator requires `s_01`, `s_12`, and `s_02`."
            )
        s_op = np.zeros((3, 3), dtype=np.complex128)
        s_op[0, 1] = self.s_01
        s_op[1, 0] = np.conjugate(self.s_01)
        s_op[1, 2] = self.s_12
        s_op[2, 1] = np.conjugate(self.s_12)
        s_op[0, 2] = self.s_02
        s_op[2, 0] = np.conjugate(self.s_02)
        return s_op

    def n_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> Union[ndarray, csc_matrix]:
        """Returns the charge operator in native or energy basis."""
        native = self._n_operator_native()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def sys_bath_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> Union[ndarray, csc_matrix]:
        """Returns Lambda system-bath operator in native or energy basis.

        If no custom `s_12`/`s_02` are supplied, this returns the same operator
        as :meth:`n_operator`.
        """
        native = self._s_operator_native()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> Union[ndarray, csc_matrix]:
        r"""Returns phase operator from :math:`[\hat H/\hbar,\hat\phi]=-i8E_C\hat n`.

        For off-diagonal elements, this uses
        :math:`\phi_{mn} = -i\,8E_C\,n_{mn}/(E_m - E_n)`.
        Diagonal elements are set to zero.
        """
        evals = self._energy_eigenvalues_native()
        n_native = self._n_operator_native()
        phi_native = np.zeros((3, 3), dtype=np.complex128)

        for m in range(3):
            for n in range(3):
                if m == n:
                    continue
                delta = evals[m] - evals[n]
                if np.isclose(delta, 0.0):
                    raise ValueError(
                        "Cannot derive `phi_operator`: Lambda energies are degenerate."
                    )
                phi_native[m, n] = -1j * 8.0 * self.EC * n_native[m, n] / delta

        return self.process_op(native_op=phi_native, energy_esys=energy_esys)

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> Union[ndarray, csc_matrix]:
        """Returns diagonal Hamiltonian in native or energy basis."""
        native_hamiltonian = np.diag(self._energy_eigenvalues_native())
        return self.process_hamiltonian(
            native_hamiltonian=native_hamiltonian, energy_esys=energy_esys
        )
