# sympy_helpers.py
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
"""SymPy-side helpers used across the circuit machinery: symbol-list
construction, potential-term detection, coefficient rounding, and
subsystem-restricted expression filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sm

from numpy import ndarray

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem


def _generate_symbols_list(
    var_str: str, iterable_list: list[int] | ndarray
) -> list[sm.Symbol]:
    """Return symbols whose names are ``var_str + str(iterable)``.

    Parameters
    ----------
    var_str:
        name of the variable which needs to be generated
    iterable_list:
        The list of indices which generates the symbols
    """
    return [sm.symbols(var_str + str(iterable)) for iterable in iterable_list]


def is_potential_term(term: sm.Expr) -> bool:
    """Determines if a given sympy expression term is part of the potential.

    Parameters
    ----------
    term:
        a single term as a Sympy expression

    Returns
    -------
    ``True`` if the term is part of the potential of this instance's Hamiltonian.
    """
    for symbol in term.free_symbols:
        if "θ" in symbol.name or "Φ" in symbol.name:
            return True
    return False


def round_symbolic_expr(expr: sm.Expr, number_of_digits: int) -> sm.Expr:
    """Round all floating-point coefficients in a Sympy expression.

    The expression is first expanded; every :class:`sympy.Float` encountered
    in the resulting tree is replaced by its rounded value.

    Parameters
    ----------
    expr:
        Sympy expression to round
    number_of_digits:
        number of decimal digits to round to

    Returns
    -------
    Rounded Sympy expression.
    """
    rounded_expr = expr.expand()
    for term in sm.preorder_traversal(expr.expand()):
        if isinstance(term, sm.Float):
            rounded_expr = rounded_expr.subs(term, round(term, number_of_digits))
    return rounded_expr


def keep_terms_for_subsystem(
    sym_expr: sm.Expr, subsys: "Subsystem", substitute_zero: bool = False
) -> sm.Expr:
    """Drop terms from ``sym_expr`` not involving ``subsys`` variables.

    If ``substitute_zero`` is ``True``, every free symbol in ``sym_expr`` is
    substituted with zero and the resulting expression is returned.

    Parameters
    ----------
    sym_expr:
        symbolic expression to filter
    subsys:
        subsystem whose ``dynamic_var_indices`` determine the terms to keep
    substitute_zero:
        if ``True``, substitute zero for all free symbols and return that

    Returns
    -------
    Filtered symbolic expression.
    """
    # Local import to avoid a top-level dep on circuit_utils for one helper.
    from scqubits.core.circuit_utils import get_trailing_number

    if substitute_zero:
        for var_sym in sym_expr.free_symbols:
            sym_expr = sym_expr.subs(var_sym, 0)
        return sym_expr
    terms = sym_expr.as_ordered_terms()
    for term in terms:
        var_indices = [
            get_trailing_number(sym_var.name) for sym_var in list(term.free_symbols)
        ]
        if len(set(var_indices) & set(subsys.dynamic_var_indices)) == 0:
            sym_expr = sym_expr - term
    return sym_expr
