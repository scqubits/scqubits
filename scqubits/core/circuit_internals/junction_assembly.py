"""Pipeline that converts symbolic Josephson-junction potential terms
into ``qutip.Qobj`` matrix operators.

The two public entry points -- ``evaluate_matrix_cosine_terms`` and
``evaluate_matrix_sawtooth_terms`` -- take a symbolic expression
involving ``cos(...)`` / ``sin(...)`` or the ``saw(...)`` placeholder
and construct its matrix representation in the appropriate Hilbert
space.  Each takes an explicit ``circuit`` argument
(a ``Subsystem``-or-``Circuit`` instance) to access the cross-mixin
methods needed for per-variable operator construction:
``hierarchical_diagonalization``, ``hilbertdim``, ``return_root_child``,
``identity_wrap_for_hd``, ``_kron_operator``, ``exp_i_operator``,
``_evaluate_symbolic_expr``, and ``subsystems``.
"""

from __future__ import annotations

import functools
import operator as builtin_op

from typing import TYPE_CHECKING, Any

import numpy as np
import qutip as qt
import sympy as sm

from scqubits.core.circuit_internals.sawtooth import sawtooth_potential
from scqubits.core.circuit_internals.utils import get_trailing_number

if TYPE_CHECKING:
    from scqubits.core.circuit_internals._protocols import CircuitProtocol


def extract_junction_phase(term: sm.Expr) -> sm.Expr:
    """Return the Josephson phase expression inside a single ``cos``/``sin`` factor.

    ``term`` is expected to be a single product (one summand of an ordered
    sum) that contains exactly one ``cos`` or ``sin`` factor — i.e. a
    Josephson-junction contribution to the potential, ``E_J cos(phi)`` or
    a derivative ``sin(phi)``. The returned expression is the ``phi``
    argument: a linear combination of node-flux symbols.

    The leading ``1.0 *`` ensures sympy splits a bare ``cos(...)`` into a
    product so ``.args`` always yields the trig factor as one of its
    arguments.
    """
    return [
        arg.args[0] for arg in (1.0 * term).args if arg.has(sm.cos) or arg.has(sm.sin)
    ][0]


def term_has_cos_factor(term: sm.Expr) -> bool:
    """Return ``True`` when ``term`` contains a ``cos`` factor."""
    return any(arg.has(sm.cos) for arg in (1.0 * term).args)


def term_has_sin_factor(term: sm.Expr) -> bool:
    """Return ``True`` when ``term`` contains a ``sin`` factor."""
    return any(arg.has(sm.sin) for arg in (1.0 * term).args)


def assemble_cos_term(op: qt.Qobj) -> qt.Qobj:
    """Build ``cos`` from the matrix exponential ``op = exp(i * arg)``."""
    return (op + op.dag()) * 0.5


def assemble_sin_term(op: qt.Qobj) -> qt.Qobj:
    """Build ``sin`` from the matrix exponential ``op = exp(i * arg)``."""
    return (op - op.dag()) * 0.5 * (-1j)


def build_junction_phase_operator_list(
    circuit: CircuitProtocol,
    junction_phase_expr: sm.Expr,
    var_indices: list[int],
    bare_esys: dict[int, tuple] | None,
) -> list[qt.Qobj]:
    """Build the per-variable ``exp(i * prefactor * var)`` operators for a JJ term.

    Each Josephson term ``cos(sum_k a_k phi_k)`` is realised as a product
    ``prod_k exp(i a_k phi_k)`` (combined with the dagger via
    :func:`assemble_cos_term` / :func:`assemble_sin_term`); this helper
    returns that list of per-variable factors, one per symbol in
    ``junction_phase_expr.free_symbols``.

    Parameters
    ----------
    circuit:
        the ``Subsystem`` or ``Circuit`` instance providing
        ``return_root_child``, ``_kron_operator``, ``exp_i_operator``,
        and ``identity_wrap_for_hd``.
    junction_phase_expr:
        the phase argument extracted by :func:`extract_junction_phase`,
        i.e. a linear combination of node-flux symbols.
    var_indices:
        the variable indices (parsed from each free symbol's name)
        corresponding to ``junction_phase_expr.free_symbols``, used to
        route to the right child subsystem.
    bare_esys:
        optional precomputed dict of bare eigensystems for nested
        subsystems, keyed by subsystem index.
    """
    operator_list = []
    for idx, var_symbol in enumerate(junction_phase_expr.free_symbols):
        prefactor = float(junction_phase_expr.coeff(var_symbol))
        child_circuit = circuit.return_root_child(var_indices[idx])
        operator_bare = child_circuit._kron_operator(
            circuit.exp_i_operator(var_symbol, prefactor), var_indices[idx]
        )
        operator_list.append(
            circuit.identity_wrap_for_hd(
                operator_bare,
                child_circuit,
                bare_esys=bare_esys,
            )
        )
    return operator_list


def evaluate_matrix_cosine_terms(
    circuit: CircuitProtocol,
    junction_potential: sm.Expr,
    bare_esys: dict[int, tuple] | None = None,
) -> qt.Qobj:
    """Evaluate symbolic Josephson cosine/sine terms to a :class:`qutip.Qobj`.

    Parameters
    ----------
    circuit:
        the ``Subsystem`` or ``Circuit`` instance providing
        ``hierarchical_diagonalization``, ``subsystems``, ``hilbertdim``,
        and the per-variable operator construction surface.
    junction_potential:
        symbolic expression containing ``cos`` and/or ``sin`` terms.
    bare_esys:
        optional precomputed dict of bare eigensystems for subsystems.
    """
    if circuit.hierarchical_diagonalization:
        subsystem_list = circuit.subsystems
        identity = qt.tensor(
            [qt.identity(subsystem.truncated_dim) for subsystem in subsystem_list]
        )
    else:
        identity = qt.identity(circuit.hilbertdim())

    junction_potential_matrix = identity * 0

    if (
        isinstance(junction_potential, (int, float))
        or len(junction_potential.free_symbols) == 0
    ):
        return junction_potential_matrix

    for term in junction_potential.as_ordered_terms():
        coefficient = float(list(term.as_coefficients_dict().values())[0])
        junction_phase_expr = extract_junction_phase(term)

        var_indices = [
            get_trailing_number(var_symbol.name)
            for var_symbol in junction_phase_expr.free_symbols
        ]

        # Strip constant offsets from the junction phase and absorb the
        # resulting global phase into the term's coefficient.
        for summand in junction_phase_expr.as_ordered_terms():
            if not summand.free_symbols:
                junction_phase_expr -= summand
                coefficient *= np.exp(float(summand) * 1j)

        operator_list = build_junction_phase_operator_list(
            circuit, junction_phase_expr, var_indices, bare_esys
        )
        term_operator = coefficient * functools.reduce(
            builtin_op.mul,
            operator_list,
        )
        if term_has_cos_factor(term):
            junction_potential_matrix += assemble_cos_term(term_operator)
        elif term_has_sin_factor(term):
            junction_potential_matrix += assemble_sin_term(term_operator)
    return junction_potential_matrix


def evaluate_matrix_sawtooth_terms(
    circuit: CircuitProtocol,
    saw_expr: sm.Expr,
    bare_esys: dict[int, tuple] | None = None,
) -> qt.Qobj:
    """Evaluate symbolic sawtooth-potential terms to a :class:`qutip.Qobj`.

    Parameters
    ----------
    circuit:
        the ``Subsystem`` or ``Circuit`` instance providing
        ``hierarchical_diagonalization``, ``subsystems``, ``hilbertdim``,
        and ``_evaluate_symbolic_expr``.
    saw_expr:
        symbolic expression containing :func:`sawtooth_potential` terms.
    bare_esys:
        optional precomputed dict of bare eigensystems for subsystems.
    """
    if circuit.hierarchical_diagonalization:
        subsystem_list = circuit.subsystems
        identity = qt.tensor(
            [qt.identity(subsystem.truncated_dim) for subsystem in subsystem_list]
        )
    else:
        identity = qt.identity(circuit.hilbertdim())

    saw_potential_matrix = identity * 0

    saw = sm.Function("saw", real=True)
    for saw_term in saw_expr.as_ordered_terms():
        coefficient = float(list(saw_term.as_coefficients_dict().values())[0])
        saw_argument_expr = [
            arg.args[0] for arg in (1.0 * saw_term).args if (arg.has(saw))
        ][0]

        saw_argument_operator = circuit._evaluate_symbolic_expr(
            saw_argument_expr, bare_esys
        )

        # since this operator only works for discretized phi basis
        diagonal_elements = sawtooth_potential(saw_argument_operator.diag())
        saw_potential_matrix += coefficient * qt.qdiags(
            diagonal_elements, 0, dims=saw_potential_matrix.dims
        )

    return saw_potential_matrix
