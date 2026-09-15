# charge_basis_operators.py
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
"""Operators in the charge (number) basis for periodic-variable subsystems.

The charge basis indexes the integer charges
:math:`n = -n_\\mathrm{cut}, \\dots, n_\\mathrm{cut}`. Every operator
returned here is a :class:`scipy.sparse.csc_matrix` of dimension
``2 * ncut + 1``.
"""

from __future__ import annotations

import numpy as np

from scipy import sparse
from scipy.sparse import csc_matrix


def _identity_theta(ncut: int) -> csc_matrix:
    """Return the identity operator in the charge basis.

    Parameters
    ----------
    ncut:
        charge basis cutoff, ``n = -ncut, ..., ncut``
    """
    dim_theta = 2 * ncut + 1
    return sparse.identity(dim_theta, format="csc")  # type: ignore[return-value]


def _n_theta_operator(ncut: int) -> csc_matrix:
    """Return the charge operator :math:`n` in the charge basis.

    Parameters
    ----------
    ncut:
        charge basis cutoff, ``n = -ncut, ..., ncut``
    """
    dim_theta = 2 * ncut + 1
    diag_elements = np.arange(-ncut, ncut + 1)
    n_theta_matrix = sparse.dia_matrix(  # type: ignore[type-var]
        (diag_elements, [0]), shape=(dim_theta, dim_theta)
    ).tocsc()  # type: ignore[misc]
    return n_theta_matrix  # type: ignore[return-value]


def _exp_i_theta_operator(ncut: int, prefactor: int = 1) -> csc_matrix:
    r"""Operator :math:`e^{i\,\mathrm{prefactor}\,\theta}` in the charge basis.

    Parameters
    ----------
    ncut:
        charge basis cutoff, ``n = -ncut, ..., ncut``
    prefactor:
        integer prefactor multiplying :math:`\theta` in the exponent
    """
    dim_theta = 2 * ncut + 1
    matrix = sparse.dia_matrix(  # type: ignore[type-var]
        (np.ones(dim_theta), [-prefactor]),
        shape=(dim_theta, dim_theta),
    ).tocsc()  # type: ignore[misc]
    return matrix  # type: ignore[return-value]


def _exp_i_theta_operator_conjugate(ncut: int) -> csc_matrix:
    r"""Operator :math:`e^{-i\theta}` in the charge basis.

    Parameters
    ----------
    ncut:
        charge basis cutoff, ``n = -ncut, ..., ncut``
    """
    dim_theta = 2 * ncut + 1
    matrix = sparse.dia_matrix(  # type: ignore[type-var]
        (np.ones(dim_theta), [1]),
        shape=(dim_theta, dim_theta),
    ).tocsc()  # type: ignore[misc]
    return matrix  # type: ignore[return-value]


def _cos_theta(ncut: int) -> csc_matrix:
    """Return the operator :math:`\\cos \\varphi` in the charge basis.

    Parameters
    ----------
    ncut:
        charge basis cutoff, ``n = -ncut, ..., ncut``
    """
    cos_op = 0.5 * (_exp_i_theta_operator(ncut) + _exp_i_theta_operator_conjugate(ncut))
    return cos_op


def _sin_theta(ncut: int) -> csc_matrix:
    """Return the operator :math:`\\sin \\varphi` in the charge basis.

    Parameters
    ----------
    ncut:
        charge basis cutoff, ``n = -ncut, ..., ncut``
    """
    sin_op = (
        -1j
        * 0.5
        * (_exp_i_theta_operator(ncut) - _exp_i_theta_operator_conjugate(ncut))
    )
    return sin_op
