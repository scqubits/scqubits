# matrix_helpers.py
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
"""Mixed dense/sparse matrix helpers used as callable replacements when the
circuit's symbolic Hamiltonian is evaluated by
``_hamiltonian_for_harmonic_extended_vars`` (see ``circuit_internals.routines``).

Five helpers populate the ``replacement_dict`` passed to :func:`eval`:

- ``_cos_dia`` / ``_sin_dia`` — sparse-in, sparse-out cos/sin on the diagonal
- ``_cos_dia_dense`` / ``_sin_dia_dense`` — dense-in, dense-out variants
- ``matrix_power_sparse`` — dense-in, sparse-out matrix power

The dispatcher selects between the sparse and dense variants based on the
circuit's ``type_of_matrices`` setting. (B5 plans to replace this
``eval``-driven dispatch with ``sympy.lambdify``, at which point this
module disappears.)
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
