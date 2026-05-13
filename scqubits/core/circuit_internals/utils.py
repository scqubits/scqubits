# utils.py
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

"""Small holdouts from the original ``circuit_utils.py`` junk drawer.

Two real residents remain:

* :func:`truncation_template` — public-API helper for hierarchical
  diagonalization, advertised as ``scqubits.truncation_template``;
* :func:`get_trailing_number` — string-parsing helper used in ~55 places
  across ``scqubits.core``.

The remaining names listed in :data:`__all__` are re-exports kept here so
that ``from scqubits.core.circuit_utils import <name>`` (the legacy
public path) continues to work — see the four-line shim at
``scqubits/core/circuit_utils.py``.
"""

from __future__ import annotations

import re

from scqubits.core.circuit_internals.input import example_circuit  # noqa: F401
from scqubits.core.circuit_internals.matrix_helpers import (  # noqa: F401
    matrix_power_sparse,
)
from scqubits.core.circuit_internals.sympy_helpers import (  # noqa: F401
    is_potential_term,
    keep_terms_for_subsystem,
    round_symbolic_expr,
)
from scqubits.core.circuit_internals.yaml_assembly import (  # noqa: F401
    assemble_circuit,
    assemble_transformation_matrix,
    yaml_like_out_with_pp,
)
from scqubits.utils.misc import flatten_list_recursive

__all__ = [
    "truncation_template",
    "get_trailing_number",
    "example_circuit",
    "assemble_circuit",
    "assemble_transformation_matrix",
    "is_potential_term",
    "keep_terms_for_subsystem",
    "matrix_power_sparse",
    "round_symbolic_expr",
    "yaml_like_out_with_pp",
]


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
