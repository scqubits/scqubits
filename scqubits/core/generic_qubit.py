# generic_qubit.py
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

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy as sp

from numpy import ndarray

import scqubits.core.descriptors as descriptors
import scqubits.core.operators as operators
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.utils.spectrum_utils import get_matrixelement_table, order_eigensystem

# -generic qubit (two-level system)----------------------------------------------


class GenericQubit(base.QuantumSystem, serializers.Serializable):
    """Class for a generic qubit (genuine two-level system). Create a class instance
    via::

        GenericQubit(E=4.3)

    Parameters
    ----------
    E:
       qubit energy splitting
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """

    truncated_dim = 2  # type:ignore
    _sys_type: str
    _init_params: list

    E = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(self, E: float, id_str: Optional[str] = None) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.E = E

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {"E": 5.0}

    def hamiltonian(self):
        return 0.5 * self.E * self.sz_operator()

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return 2

    def eigenvals(self, evals_count: int = 2) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(hamiltonian_mat, eigvals_only=True)
        return np.sort(evals)

    def eigensys(self, evals_count: int = 2) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, eigvals_only=False)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def matrixelement_table(self, operator: str) -> ndarray:
        """Returns table of matrix elements for `operator` with respect to the
        eigenstates of the qubit. The operator is given as a string matching a class
        method returning an operator matrix.

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix in
            qubit-internal basis.
        """
        _, evecs = self.eigensys()
        operator_matrix = getattr(self, operator)()
        table = get_matrixelement_table(operator_matrix, evecs)
        return table

    @classmethod
    def create(cls) -> base.QuantumSystem:
        raise NotImplementedError

    def widget(self, params: Dict[str, Any] = None):
        raise NotImplementedError(
            "GenericQubit does not support widget-based " "creation."
        )

    def sx_operator(self):
        return operators.sigma_x()

    def sy_operator(self):
        return operators.sigma_y()

    def sz_operator(self):
        return operators.sigma_z()

    def sp_operator(self):
        return operators.sigma_plus()

    def sm_operator(self):
        return operators.sigma_minus()
