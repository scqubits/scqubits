# discretized_phi_operators.py
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
"""Operators acting on the discretized-phi (real-space) basis of an extended
variable.

Each operator is returned as a :class:`scipy.sparse.csc_matrix`. The
operators are constructed from a :class:`~scqubits.core.discretization.Grid1d`
specification of the discretization grid.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

from scqubits.core import discretization


def _identity_phi(grid: discretization.Grid1d) -> csc_matrix:
    """Return the identity operator in the discretized-phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the identity operator

    Returns
    -------
    identity operator in the discretized phi basis
    """
    pt_count = grid.pt_count
    return sparse.identity(pt_count, format="csc")  # type: ignore[return-value]


def _diag_from_function(
    grid: discretization.Grid1d, values_fn: Callable[[ndarray], ndarray]
) -> csc_matrix:
    """Return a sparse CSC diagonal operator whose diagonal is ``values_fn(grid.make_linspace())``."""
    pt_count = grid.pt_count
    matrix = sparse.dia_matrix((pt_count, pt_count))
    matrix.setdiag(values_fn(grid.make_linspace()))
    return matrix.tocsc()


def _phi_operator(grid: discretization.Grid1d) -> csc_matrix:
    """Return the phi operator in the discretized-phi basis."""
    return _diag_from_function(grid, lambda x: x)


def _i_d_dphi_operator(grid: discretization.Grid1d) -> csc_matrix:
    """Return ``i * d/dphi`` in the discretized-phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the operator

    Returns
    -------
    i*d/dphi operator in the discretized phi basis
    """
    return grid.first_derivative_matrix(prefactor=-1j)


def _i_d2_dphi2_operator(grid: discretization.Grid1d) -> csc_matrix:
    """Return ``i * d^2/dphi^2`` in the discretized-phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the operator

    Returns
    -------
    i*d2/dphi2 operator in the discretized phi basis
    """
    return grid.second_derivative_matrix(prefactor=-1.0)


def _cos_phi(grid: discretization.Grid1d) -> csc_matrix:
    """Return the cos operator in the discretized-phi basis."""
    return _diag_from_function(grid, np.cos)


def _sin_phi(grid: discretization.Grid1d) -> csc_matrix:
    """Return the sin operator in the discretized-phi basis."""
    return _diag_from_function(grid, np.sin)
