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

from __future__ import annotations

from typing import Any

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
    """Class for a generic qubit (genuine two-level system).

    Create a class instance via::

        GenericQubit(E=4.3)

    Parameters
    ----------
    E:
       qubit energy splitting.
    id_str:
        optional string by which this instance can be referred to in :class:`HilbertSpace`
        and :class:`ParameterSweep`. If not provided, an id is auto-generated.
    """

    truncated_dim = 2
    _sys_type: str
    _init_params: list

    E = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(self, E: float, id_str: str | None = None) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.E = E

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return a default-parameter dict suitable for instantiating the class."""
        return {"E": 5.0}

    def hamiltonian(self):
        r"""Return the qubit Hamiltonian :math:`H = \tfrac{1}{2} E\,\sigma_z`."""
        return 0.5 * self.E * self.sz_operator()

    def hilbertdim(self) -> int:
        """Return the Hilbert space dimension."""
        return 2

    def eigenvals(self, evals_count: int = 2) -> ndarray:
        """Return array of eigenvalues, sorted in ascending order.

        Uses :func:`scipy.linalg.eigh` (Hermitian; ascending real eigenvalues),
        then applies :func:`numpy.sort` to ensure ascending order.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues (default: 2).
        """
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(hamiltonian_mat, eigvals_only=True)
        return np.sort(evals)

    def eigensys(self, evals_count: int = 2) -> tuple[ndarray, ndarray]:
        """Return arrays of eigenvalues and eigenvectors.

        Uses :func:`scipy.linalg.eigh` (Hermitian; ascending real eigenvalues).

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues (default: 2).
        """
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, eigvals_only=False)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def matrixelement_table(self, operator: str) -> ndarray:
        """Return table of matrix elements for ``operator`` with respect to qubit eigenstates.

        Parameters
        ----------
        operator:
            name of a class method (in string form) returning the operator
            matrix in the qubit-internal basis.
        """
        _, evecs = self.eigensys()
        operator_matrix = getattr(self, operator)()
        table = get_matrixelement_table(operator_matrix, evecs)
        return table

    @classmethod
    def create(cls) -> base.QuantumSystem:
        """Not implemented for :class:`GenericQubit`; always raises :exc:`NotImplementedError`."""
        raise NotImplementedError

    def widget(self, params: dict[str, Any] | None = None):
        """Not implemented for :class:`GenericQubit`; always raises :exc:`NotImplementedError`.

        Parameters
        ----------
        params:
            unused; kept for API compatibility with :meth:`QuantumSystem.widget`.
        """
        raise NotImplementedError(
            "GenericQubit does not support widget-based " "creation."
        )

    def sx_operator(self):
        r"""Return the Pauli :math:`\sigma_x` operator."""
        return operators.sigma_x()

    def sy_operator(self):
        r"""Return the Pauli :math:`\sigma_y` operator."""
        return operators.sigma_y()

    def sz_operator(self):
        r"""Return the Pauli :math:`\sigma_z` operator."""
        return operators.sigma_z()

    def sp_operator(self):
        r"""Return the raising operator :math:`\sigma_+ = (\sigma_x + i\sigma_y)/2`."""
        return operators.sigma_plus()

    def sm_operator(self):
        r"""Return the lowering operator :math:`\sigma_- = (\sigma_x - i\sigma_y)/2`."""
        return operators.sigma_minus()
