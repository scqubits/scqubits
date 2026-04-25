# branch_metadata.py
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
"""Lightweight branch-type predicates shared by circuit input parsing and
the symbolic / numerical circuit machinery.

Kept dependency-free (no other ``scqubits.core`` imports) so it can be
imported by ``circuit_input.py`` and ``circuit_utils.py`` without
introducing an import cycle.
"""

from __future__ import annotations


def _junction_order(branch_type: str) -> int:
    """Return the order of a JJ branch.

    For order :math:`n`, the energy is given by
    :math:`\\cos(\\varphi) + \\cos(2\\varphi) + \\cdots + \\cos(n\\varphi)`.

    Parameters
    ----------
    branch_type:
        branch type string (expected to contain ``"JJ"``)

    Returns
    -------
    Order of the Josephson junction. Raises :class:`ValueError` when the branch
    type does not correspond to a Josephson junction.
    """
    if "JJ" not in branch_type:
        raise ValueError("The branch is not a JJ branch")
    if len(branch_type) > 2:
        if (
            branch_type[2] == "s"
        ):  # adding "JJs" which is a junction with sawtooth current phase relationship
            return 1
        return int(branch_type[2:])
    else:
        return 1


def _capacitance_variable_for_branch(branch_type: str) -> str:
    """Return the parameter name that stores the capacitance of the branch.

    Parameters
    ----------
    branch_type:
        branch type string (expected to contain ``"C"`` or ``"JJ"``)
    """
    if "C" in branch_type:
        return "EC"
    elif "JJ" in branch_type:
        return "ECJ"
    else:
        raise ValueError("Branch type is not a capacitor or a JJ")
