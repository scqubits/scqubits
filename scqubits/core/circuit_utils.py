# circuit-utils.py
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

import re

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
import sympy as sm

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

from scqubits.core import circuit_input
from scqubits.core import discretization as discretization

# Public deprecation shim — `assemble_circuit` and
# `assemble_transformation_matrix` are advertised in `scqubits/__init__.py`
# and may also be imported as `scqubits.core.circuit_utils.<name>` by
# downstream code. Their definitions now live in `circuit_yaml_assembly`.
from scqubits.core.circuit_yaml_assembly import (  # noqa: F401
    assemble_circuit,
    assemble_transformation_matrix,
)
from scqubits.utils.misc import (
    Qobj_to_scipy_csc_matrix,
    flatten_list_recursive,
    is_string_float,
    unique_elements_in_list,
)

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem


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


# def sawtooth_potential(x: float) -> float:
#     """
#     Is the function which returns the potential of a sawtooth junction,
#     i.e. a junction with a sawtooth current phase relationship, only in the discretized phi basis.
#     """
#     x_rel = (x - np.pi) % (2*np.pi) - np.pi
#     return (x_rel)**2/(np.pi)**2 # normalized to have a maximum of 1


def sawtooth_potential(phi_pts: ndarray):
    """Return the sawtooth-junction potential evaluated at ``phi_pts``.

    The potential is computed from a truncated Fourier series with skewness
    parameter :math:`s=0.99` and ``N=1000`` terms:
    :math:`V(\\varphi) = -\\sum_{k=1}^{N} (s+1)(-s)^{k-1}
    \\cos(k\\varphi)/k^2`.

    Parameters
    ----------
    phi_pts:
        phase values at which the potential is evaluated
    """
    # definition from Andras
    skewness = 0.99
    N = 1000
    V = np.zeros_like(phi_pts)
    for idx in range(1, N + 1):
        V += (skewness + 1) * (-skewness) ** (idx - 1) * np.cos(idx * phi_pts) / idx**2
    return -V


def truncation_template(
    system_hierarchy: list, individual_trunc_dim: int = 6, combined_trunc_dim: int = 30
) -> list:
    """Generate a template for truncated subsystem dimensions.

    Used when hierarchical diagonalization is enabled.

    Parameters
    ----------
    system_hierarchy:
        list which sets the system hierarchy
    individual_trunc_dim:
        The default used to set truncation dimension for subsystems which do not
        use hierarchical diagonalization, by default 6
    combined_trunc_dim:
        The default used to set the truncated dim for subsystems which use hierarchical
        diagonalization, by default 30

    Returns
    -------
    The template for setting the truncated dims for the Circuit instance when
    hierarchical diagonalization is used.
    """
    trunc_dims: list[int | list] = []
    for subsystem_hierarchy in system_hierarchy:
        if subsystem_hierarchy == flatten_list_recursive(subsystem_hierarchy):
            trunc_dims.append(individual_trunc_dim)
        else:
            trunc_dims.append(
                [combined_trunc_dim, truncation_template(subsystem_hierarchy)]
            )
    return trunc_dims


def get_trailing_number(input_str: str) -> int:
    """Return the integer trailing the input string.

    For example, ``get_trailing_number("a23")`` returns ``23``.

    Parameters
    ----------
    input_str:
        string which ends with a number; raises :class:`ValueError` if no
        trailing digits are present

    Returns
    -------
    trailing integer
    """
    match = re.search(r"\d+$", input_str)
    if match is None:
        raise ValueError(f"get_trailing_number: {input_str!r} has no trailing digits")
    return int(match.group())


def example_circuit(qubit: str) -> str:
    """Return example input strings for some of the popular qubits.

    The input strings are intended for ``AnalyzeQCircuit`` and
    ``CustomQCircuit``.

    Parameters
    ----------
    qubit:
        "fluxonium" or "transmon" or "zero_pi" or "cos2phi" choosing the respective
        example input strings.
    """

    # example input strings for popular qubits
    inputs_by_qubit_name = dict(
        fluxonium="nodes: 2\nbranches:\nJJ	1,2	Ej	Ecj\nL	1,2	El\nC	1,2	Ec",
        transmon="nodes: 2\nbranches:\nC\t1,2\tEc\nJJ\t1,2\tEj\tEcj\n",
        cos2phi="nodes: 4\nbranches:\nC\t1,3\tEc\nJJ\t1,2\tEj\tEcj\nJJ\t3, "
        "4\tEj\tEcj\nL\t1,4\tEl\nL\t2,3\tEl\n\n",
        zero_pi="nodes: 4\nbranches:\nJJ\t1,2\tEj\tEcj\nL\t2,3\tEl\nJJ\t3,"
        "4\tEj\tEcj\nL\t4,1\tEl\nC\t1,3\tEc\nC\t2,4\tEc\n",
    )

    if qubit in inputs_by_qubit_name:
        return inputs_by_qubit_name[qubit]
    else:
        raise AttributeError("Qubit not available or invalid input.")
