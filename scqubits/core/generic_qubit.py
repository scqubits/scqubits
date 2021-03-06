# generic_qubit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy as sp
from numpy import ndarray

import scqubits.core.operators as operators
import scqubits.core.descriptors as descriptors
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.spectrum_utils import order_eigensystem, get_matrixelement_table


# —generic qubit (two-level system)——————————————————————————————————————————————

class GenericQubit(base.QuantumSystem, serializers.Serializable):
    """Class for a generic qubit (genuine two-level system). Create a class instance via::

        GenericQubit(E=4.3)

    Parameters
    ----------
    E:
       qubit energy splitting
    """
    truncated_dim = 2
    _evec_dtype: type
    _sys_type: str
    _init_params: list

    E = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    sx = staticmethod(operators.sigma_x)
    sy = staticmethod(operators.sigma_y)
    sz = staticmethod(operators.sigma_z)
    sp = staticmethod(operators.sigma_plus)
    sm = staticmethod(operators.sigma_minus)

    def __init__(self, E: float) -> None:
        self.E = E
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {'E': 5.0}

    def hamiltonian(self):
        return 0.5 * self.E * self.sz()

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
        """Returns table of matrix elements for `operator` with respect to the eigenstates of the qubit.
        The operator is given as a string matching a class method returning an operator matrix.

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix in qubit-internal basis.
        """
        _, evecs = self.eigensys()
        operator_matrix = getattr(self, operator)()
        table = get_matrixelement_table(operator_matrix, evecs)
        return table

    def create(cls) -> base.QuantumSystem:
        raise NotImplementedError

    def widget(self, params: Dict[str, Any] = None):
        raise NotImplementedError
