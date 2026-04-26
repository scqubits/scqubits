# sawtooth.py
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
"""Sawtooth junction physics: a smooth periodic potential with sawtooth
current-phase relationship, evaluated as a truncated Fourier series.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

from numpy import ndarray
from scipy.sparse import csc_matrix

# Skewness ``s`` of the sawtooth current-phase relationship. Values close to
# 1 produce a sharp sawtooth shape; ``0.99`` is the original parameter from
# Andras's definition.
SAWTOOTH_SKEWNESS = 0.99

# Number of terms retained in the Fourier-series truncation.
SAWTOOTH_FOURIER_TERMS = 1000


def sawtooth_potential(phi_pts: ndarray):
    """Return the sawtooth-junction potential evaluated at ``phi_pts``.

    The potential is computed from a truncated Fourier series with skewness
    parameter :data:`SAWTOOTH_SKEWNESS` and :data:`SAWTOOTH_FOURIER_TERMS`
    terms:
    :math:`V(\\varphi) = -\\sum_{k=1}^{N} (s+1)(-s)^{k-1}
    \\cos(k\\varphi)/k^2`.

    Parameters
    ----------
    phi_pts:
        phase values at which the potential is evaluated
    """
    s = SAWTOOTH_SKEWNESS
    V = np.zeros_like(phi_pts)
    for idx in range(1, SAWTOOTH_FOURIER_TERMS + 1):
        V += (s + 1) * (-s) ** (idx - 1) * np.cos(idx * phi_pts) / idx**2
    return -V


def sawtooth_operator(x: ndarray | csc_matrix):
    """Apply :func:`sawtooth_potential` to the diagonal of ``x``.

    Parameters
    ----------
    x:
        argument of the sawtooth operator in the Hamiltonian
    """
    diagonal_elements = sawtooth_potential(x.diagonal())

    operator = sp.sparse.dia_matrix(
        (diagonal_elements, 0), shape=(len(diagonal_elements), len(diagonal_elements))
    )
    return operator.tocsc()
