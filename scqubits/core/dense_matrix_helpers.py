# dense_matrix_helpers.py
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
"""Diagonal-acting helpers used when the circuit's symbolic Hamiltonian is
evaluated against either dense or sparse operator matrices.

The four ``cos``/``sin`` variants are stored in the
``replacement_dict`` passed to :func:`eval` inside
``_hamiltonian_for_harmonic_extended_vars`` (see ``circuit_routines.py``).
``matrix_power_sparse`` provides the ``matrix_power`` entry for the same
dispatch.
"""

from __future__ import annotations

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix


def _cos_dia(x: csc_matrix) -> csc_matrix:
    """Return a sparse diagonal matrix containing ``cos(x.diagonal())``.

    Parameters
    ----------
    x:
        input sparse matrix whose diagonal is used
    """
    return sparse.diags(np.cos(x.diagonal())).tocsc()  # type: ignore[return-value]


def _sin_dia(x: csc_matrix) -> csc_matrix:
    """Return a sparse diagonal matrix containing ``sin(x.diagonal())``.

    Parameters
    ----------
    x:
        input sparse matrix whose diagonal is used
    """
    return sparse.diags(np.sin(x.diagonal())).tocsc()  # type: ignore[return-value]


def _cos_dia_dense(x: ndarray) -> ndarray:
    """Compute the cosine of a dense diagonal matrix.

    Parameters
    ----------
    x:
        input dense diagonal matrix whose diagonal is used
    """
    return np.diag(np.cos(x.diagonal()))


def _sin_dia_dense(x: ndarray) -> ndarray:
    """Compute the sine of a dense diagonal matrix.

    Parameters
    ----------
    x:
        input dense diagonal matrix whose diagonal is used
    """
    return np.diag(np.sin(x.diagonal()))


def matrix_power_sparse(dense_mat: ndarray, n: int) -> csc_matrix:
    """Return the ``n``-th matrix power of ``dense_mat`` computed in sparse form.

    Parameters
    ----------
    dense_mat:
        dense input matrix, converted internally to :class:`scipy.sparse.csc_matrix`
    n:
        non-negative integer exponent

    Returns
    -------
    Sparse matrix :math:`(\\text{dense\\_mat})^n`.
    """
    sparse_mat = sparse.csc_matrix(dense_mat)
    return sparse_mat**n
