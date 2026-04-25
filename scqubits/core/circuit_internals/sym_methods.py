from __future__ import annotations

import functools
import itertools
import operator as builtin_op
import re

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray
    from scipy.sparse import csc_matrix

    from scqubits.core.circuit import Subsystem

import numpy as np
import qutip as qt
import sympy as sm

from sympy import latex

try:
    from IPython.display import Latex, display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True


__all__ = [
    "CircuitSymMethods",
]

from abc import ABC

from scqubits.core.circuit_internals.utils import get_trailing_number
from scqubits.core.circuit_internals.sympy_helpers import (
    is_potential_term,
    round_symbolic_expr,
)
from scqubits.utils.misc import (
    check_sync_status_circuit,
    flatten_list_recursive,
    list_intersection,
    unique_elements_in_list,
)


class CircuitSymMethods(ABC):
    """Mixin providing symbolic-Hamiltonian utilities shared by circuit classes.

    Implements helpers for splitting, simplifying and rendering symbolic
    Hamiltonians, as well as building numerical operators from symbolic
    expressions for use by :class:`Circuit` and :class:`Subsystem`.
    """

    # Attributes set by concrete subclasses (Circuit, Subsystem). Declared here so
    # mypy can resolve cross-subclass attribute access in shared methods.
    external_fluxes: list[Any]
    offset_charges: list[Any]
    free_charges: list[Any]
    symbolic_params: dict[Any, Any]
    var_categories: dict[str, list[int]]
    dynamic_var_indices: list[int]
    hierarchical_diagonalization: bool
    ext_basis: Any
    hamiltonian_symbolic: Any
    type_of_matrices: str
    subsystems: list[Any]
    system_hierarchy: list[Any]
    parent: Any
    vars: dict[str, Any]
    _hamiltonian_sym_for_numerics: Any
    _id_str: str
    potential_symbolic: Any
    subsystem_interactions: Any
    get_subsystem_index: Callable[..., Any]
    transformation_matrix: Any
    is_grounded: bool
    closure_branches: list[Any]
    symbolic_circuit: Any
    affine_transformation_matrix: Any
    use_dynamic_flux_grouping: bool
    discretized_phi_range: dict[int, Any]
    cutoff_names: list[str]
    is_purely_harmonic: bool

    # Method stubs declaring methods provided by sibling mixins
    # (CircuitRoutines) when composed into Subsystem/Circuit. Declared under
    # TYPE_CHECKING so that mypy resolves shared-method `self.X` references
    # without affecting runtime behavior.
    if TYPE_CHECKING:

        def _identity_qobj(self) -> qt.Qobj: ...
        def _evaluate_matrix_cosine_terms(
            self,
            junction_potential: sm.Expr,
            bare_esys: dict[int, tuple] | None = ...,
        ) -> qt.Qobj: ...
        def _evaluate_matrix_sawtooth_terms(
            self, saw_expr: sm.Expr, bare_esys: dict[int, tuple] | None = ...
        ) -> qt.Qobj: ...
        def return_root_child(self, var_index: int) -> "Subsystem": ...
        def identity_wrap_for_hd(
            self,
            operator: csc_matrix | ndarray | None,
            child_instance: "Subsystem",
            bare_esys: dict[int, tuple] | None = ...,
        ) -> qt.Qobj: ...
        def get_operator_by_name(
            self,
            operator_name: str,
            power: int | None = ...,
            bare_esys: dict[int, tuple] | None = ...,
        ) -> qt.Qobj: ...

    @staticmethod
    def _contains_trigonometric_terms(hamiltonian: sm.Expr) -> bool:
        """Check if the Hamiltonian contains any trigonometric terms.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian expression to inspect
        """
        trigonometric_operators = [sm.cos, sm.sin, sm.Function("saw", real=True)]
        return any(hamiltonian.atoms(operator) for operator in trigonometric_operators)

    @staticmethod
    def _is_symbol_periodic_charge(sym: sm.Symbol) -> bool:
        """Return True if ``sym`` is a periodic charge symbol of the form ``n<index>``.

        Parameters
        ----------
        sym:
            sympy symbol whose name is tested
        """
        return sym.name[0] == "n" and sym.name[1:].isnumeric()

    @staticmethod
    def _is_symbol_continuous_charge(sym: sm.Symbol) -> bool:
        """Return True if ``sym`` is a continuous charge symbol of the form ``Q<index>``.

        Parameters
        ----------
        sym:
            sympy symbol whose name is tested
        """
        return sym.name[0] == "Q" and sym.name[1:].isnumeric()

    @staticmethod
    def _is_symbol_phase(sym: sm.Symbol) -> bool:
        """Return True if ``sym`` is a phase symbol of the form ``θ<index>``.

        Parameters
        ----------
        sym:
            sympy symbol whose name is tested
        """
        return sym.name[0] == "θ" and sym.name[1:].isnumeric()

    @staticmethod
    def _find_and_categorize_variable_indices(
        hamiltonian: sm.Expr,
    ) -> tuple[set[int], set[int], set[int]]:
        """Categorize variable indices appearing in ``hamiltonian`` by symbol type.

        Returns three sets of trailing-number indices: those of periodic charge
        symbols, continuous charge symbols, and phase symbols, respectively.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian expression to inspect
        """
        periodic_var_indices = set(
            get_trailing_number(symbol.name)
            for symbol in hamiltonian.free_symbols
            if CircuitSymMethods._is_symbol_periodic_charge(symbol)
        )
        extended_var_indices = set(
            get_trailing_number(symbol.name)
            for symbol in hamiltonian.free_symbols
            if CircuitSymMethods._is_symbol_continuous_charge(symbol)
        )
        phase_var_indices = set(
            get_trailing_number(symbol.name)
            for symbol in hamiltonian.free_symbols
            if CircuitSymMethods._is_symbol_phase(symbol)
        )
        return periodic_var_indices, extended_var_indices, phase_var_indices

    # @staticmethod
    def _is_expression_purely_harmonic(self, hamiltonian: sm.Expr) -> bool:
        """Check if the Hamiltonian is purely harmonic.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian expression to inspect
        """
        # if the hamiltonian contains any cos or sin term, return False
        if self._contains_trigonometric_terms(hamiltonian):
            return False
        # if the hamiltonian contains any charge operator of periodic variables, return false
        (
            periodic_charge_variable_index,
            extended_charge_variable_index,
            phase_variable_index,
        ) = self._find_and_categorize_variable_indices(hamiltonian)
        if len(periodic_charge_variable_index) > 0:
            return False
        # if the hamiltonian has any DoF where only its charge or flux operator is present, return False
        if extended_charge_variable_index != phase_variable_index:
            return False
        return True

    def _constants_in_subsys(
        self, H_sys: sm.Expr, constants_list: list[sm.Expr]
    ) -> list[sm.Expr]:
        """Return the constants of ``constants_list`` belonging to a subsystem.

        A constant belongs to the subsystem with Hamiltonian ``H_sys`` if all of
        its free symbols also appear in ``H_sys``.

        Parameters
        ----------
        H_sys:
            subsystem Hamiltonian
        constants_list:
            list of constant terms to be partitioned across subsystems

        Returns
        -------
        list of constants belonging to the subsystem
        """
        constants_subsys_list = []
        subsys_free_symbols = set(H_sys.free_symbols)
        for term in constants_list:
            if set(term.free_symbols) & subsys_free_symbols == set(term.free_symbols):
                constants_subsys_list.append(term)
        return constants_subsys_list

    def _list_of_constants_from_expr(self, expr: sm.Expr) -> list[sm.Expr]:
        """Return the terms of ``expr`` whose free symbols are entirely parameters.

        A term qualifies as a constant if all of its free symbols are external
        fluxes, offset charges, free charges, symbolic parameters, or the
        identity placeholder ``I``.

        Parameters
        ----------
        expr:
            symbolic expression to scan for constant terms

        Returns
        -------
        list of constant terms found in ``expr``
        """
        ordered_terms = expr.as_ordered_terms()
        constants = [
            term
            for term in ordered_terms
            if (
                set(
                    self.external_fluxes
                    + self.offset_charges
                    + self.free_charges
                    + list(self.symbolic_params.keys())
                    + [sm.symbols("I")]
                )
                & set(term.free_symbols)
            )
            == set(term.free_symbols)
        ]
        return constants

    def _sym_subsystem_hamiltonian_and_interactions(
        self,
        hamiltonian: sm.Expr,
        subsys_indices: list,
        non_operator_symbols: list[sm.Symbol],
    ) -> tuple[list[sm.Expr], list[sm.Expr]]:
        """Split ``hamiltonian`` into per-subsystem Hamiltonians and interactions.

        Constant terms are extracted first and re-attached to the subsystem
        whose free symbols contain them (or to the first subsystem if none
        match).

        Parameters
        ----------
        hamiltonian:
            full symbolic Hamiltonian to be split
        subsys_indices:
            list whose entries are (possibly nested) lists of variable indices
            defining each subsystem
        non_operator_symbols:
            symbols that should not be treated as operators when partitioning
            terms (e.g. external fluxes, offset charges, parameters)

        Returns
        -------
        tuple of two lists: per-subsystem Hamiltonians and per-subsystem
        interaction expressions
        """
        systems_sym = []
        interaction_sym = []
        constants = self._list_of_constants_from_expr(hamiltonian)
        hamiltonian = self._remove_constants_from_hamiltonian(hamiltonian, constants)

        for subsys_index_list in subsys_indices:
            subsys_index_list = flatten_list_recursive(subsys_index_list)
            H_sys, H_int = self._find_subsys_hamiltonian(
                hamiltonian, subsys_index_list, non_operator_symbols
            )
            # add the constants that belong to the subsystem
            subsys_const_list = self._constants_in_subsys(H_sys, constants)
            systems_sym.append(H_sys + sum(subsys_const_list))
            # remove the constants that are already added
            constants = [const for const in constants if const not in subsys_const_list]
            interaction_sym.append(H_int)
            hamiltonian -= H_sys + H_int

        if len(constants) > 0:
            systems_sym[0] += sum(constants)

        return systems_sym, interaction_sym

    def _remove_constants_from_hamiltonian(
        self, hamiltonian: sm.Expr, constants: list[sm.Expr]
    ) -> sm.Expr:
        """Subtract each term in ``constants`` from ``hamiltonian``.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian from which constants are subtracted
        constants:
            list of constant terms to remove
        """
        for const in constants:
            hamiltonian -= const
        return hamiltonian

    def _find_subsys_hamiltonian(
        self,
        hamiltonian: sm.Expr,
        subsys_index_list: list[int],
        non_operator_symbols: list[sm.Symbol],
    ) -> tuple[sm.Expr, sm.Expr]:
        """Split ``hamiltonian`` into a subsystem part and an interaction part.

        Terms whose operator indices lie entirely within ``subsys_index_list``
        contribute to the subsystem Hamiltonian; terms with operator indices
        both inside and outside the list contribute to the interaction.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian to be split
        subsys_index_list:
            variable indices defining the subsystem
        non_operator_symbols:
            symbols that should not be treated as operators

        Returns
        -------
        tuple ``(H_sys, H_int)`` of the subsystem Hamiltonian and the
        interaction Hamiltonian
        """
        hamiltonian_terms = hamiltonian.as_ordered_terms()
        H_sys = 0 * sm.symbols("x")
        H_int = 0 * sm.symbols("x")
        for term in hamiltonian_terms:
            term_operator_indices = [
                get_trailing_number(var_sym.name)
                for var_sym in term.free_symbols
                if var_sym not in non_operator_symbols
            ]
            term_operator_indices_unique = unique_elements_in_list(
                term_operator_indices
            )
            if len(set(term_operator_indices_unique) - set(subsys_index_list)) == 0:
                H_sys += term
            if (
                len(set(term_operator_indices_unique) - set(subsys_index_list)) > 0
                and len(set(term_operator_indices_unique) & set(subsys_index_list)) > 0
            ):
                H_int += term
        return H_sys, H_int

    @check_sync_status_circuit
    def _evaluate_symbolic_expr(
        self, sym_expr: sm.Expr, bare_esys: Any = None
    ) -> qt.Qobj:
        """Numerically evaluate a symbolic operator expression.

        Substitutes circuit parameters into ``sym_expr`` and converts each
        term into a matrix using the appropriate operator factory; the
        weighted sum of these matrices is returned.

        Parameters
        ----------
        sym_expr:
            symbolic expression containing operator symbols and parameters
        bare_esys:
            optional cached bare eigensystem data forwarded to identity-wrap
            calls when hierarchical diagonalization is in use
        """
        sym_expr = self._substitute_parameters(sym_expr)
        if sym_expr == 0:
            return 0
        expr_dict = sym_expr.as_coefficients_dict()
        terms = list(expr_dict.keys())
        eval_matrix_list = [
            self._evaluate_term(term, expr_dict[term], bare_esys) for term in terms
        ]
        return sum(eval_matrix_list)

    def _substitute_parameters(self, sym_expr: sm.Expr) -> sm.Expr:
        """Substitute current parameter attribute values into ``sym_expr``.

        External fluxes, offset charges, free charges, and symbolic
        parameters are replaced by the corresponding attribute values of
        ``self``.

        Parameters
        ----------
        sym_expr:
            symbolic expression in which parameters are replaced
        """
        param_symbols = (
            self.external_fluxes
            + self.offset_charges
            + self.free_charges
            + list(self.symbolic_params.keys())
        )
        for param in param_symbols:
            sym_expr = sym_expr.subs(param, getattr(self, param.name))
        return sym_expr

    def _evaluate_term(
        self, term: sm.Expr, coefficient_sympy: sm.Expr, bare_esys: Any
    ) -> qt.Qobj:
        """Evaluate a single symbolic term to its matrix form.

        Each operator factor of ``term`` is converted to a matrix and the
        result is multiplied by the (real) ``coefficient_sympy``.

        Parameters
        ----------
        term:
            symbolic operator product (without coefficient)
        coefficient_sympy:
            scalar coefficient multiplying ``term``
        bare_esys:
            optional cached bare eigensystem data forwarded to identity-wrap
            calls when hierarchical diagonalization is in use
        """
        if term == 1:
            return self._identity_qobj() * float(coefficient_sympy)
        factors = term.as_ordered_factors()
        factor_op_list = [
            self._evaluate_factor(factor, bare_esys) for factor in factors
        ]
        operator_list = self._combine_factors(factor_op_list, bare_esys)
        return functools.reduce(builtin_op.mul, operator_list) * float(
            coefficient_sympy
        )

    def _evaluate_factor(self, factor: sm.Expr, bare_esys: Any) -> Any:
        """Dispatch a single symbolic factor to the right matrix-evaluation routine.

        Cosine/sine factors are routed to the matrix-cosine evaluator,
        sawtooth factors to the sawtooth evaluator, and all remaining
        operator factors to the operator-name lookup.

        Parameters
        ----------
        factor:
            symbolic factor to evaluate
        bare_esys:
            optional cached bare eigensystem data forwarded to identity-wrap
            calls when hierarchical diagonalization is in use
        """
        if any([arg.has(sm.cos) or arg.has(sm.sin) for arg in (1.0 * factor).args]):
            return self._evaluate_matrix_cosine_terms(factor, bare_esys=bare_esys)
        elif any(
            [arg.has(sm.Function("saw", real=True)) for arg in (1.0 * factor).args]
        ):
            return self._evaluate_sawtooth_factor(factor, bare_esys)
        else:
            return self._evaluate_operator_factor(factor)

    def _evaluate_sawtooth_factor(self, factor: sm.Expr, bare_esys: Any) -> Any:
        """Evaluate a sawtooth-function factor to its matrix form.

        When hierarchical diagonalization is active, all symbols in the
        factor must belong to the same subsystem; the resulting operator is
        identity-wrapped onto the full Hilbert space.

        Parameters
        ----------
        factor:
            symbolic factor containing sawtooth operators
        bare_esys:
            optional cached bare eigensystem data forwarded to identity-wrap
            calls when hierarchical diagonalization is in use
        """
        if not self.hierarchical_diagonalization:
            return self._evaluate_matrix_sawtooth_terms(factor, bare_esys=bare_esys)
        index_subsystem = [
            self.return_root_child(get_trailing_number(sym.name))
            for sym in factor.free_symbols
        ]
        if len(set(index_subsystem)) > 1:
            raise Exception(
                "Sawtooth function terms must belong to the same subsystem."
            )
        operator = index_subsystem[0]._evaluate_matrix_sawtooth_terms(factor)
        return self.identity_wrap_for_hd(
            operator, index_subsystem[0], bare_esys=bare_esys
        )

    def _evaluate_operator_factor(self, factor: sm.Expr) -> Any:
        """Evaluate a non-trigonometric operator factor of the form ``op**k``.

        Without hierarchical diagonalization, the matrix is fetched directly
        via :meth:`get_operator_by_name`. Otherwise the owning subsystem is
        looked up and a tuple ``(subsys, operator)`` is returned for later
        identity-wrapping by :meth:`_combine_factors`.

        Parameters
        ----------
        factor:
            symbolic factor whose single free symbol identifies the operator
        """
        power_dict = dict(factor.as_powers_dict())
        free_sym = list(factor.free_symbols)[0]
        if not self.hierarchical_diagonalization:
            return self.get_operator_by_name(free_sym.name, power=power_dict[free_sym])
        subsys = self.return_root_child(get_trailing_number(free_sym.name))
        operator = subsys.get_operator_by_name(
            free_sym.name, power=power_dict[free_sym]
        )
        return (subsys, operator)

    def _combine_factors(self, factor_op_list, bare_esys):
        """Combine per-subsystem operator factors into a flat operator list.

        Operators tagged with the same parent subsystem are multiplied together
        and identity-wrapped via :meth:`identity_wrap_for_hd`; standalone
        operators pass through unchanged.

        Parameters
        ----------
        factor_op_list:
            list whose entries are either operators or
            ``(subsystem, operator)`` tuples
        bare_esys:
            optional cached bare eigensystem data forwarded to identity-wrap
            calls
        """
        operators_per_subsys = {}
        operator_list = []
        for factor_op in factor_op_list:
            if not isinstance(factor_op, tuple):
                operator_list.append(factor_op)
                continue
            subsys, operator = factor_op
            if subsys not in operators_per_subsys:
                operators_per_subsys[subsys] = [operator]
            else:
                operators_per_subsys[subsys].append(operator)
        operator_list += [
            self.identity_wrap_for_hd(
                functools.reduce(builtin_op.mul, operators_per_subsys[subsys]),
                subsys,
                bare_esys=bare_esys,
            )
            for subsys in operators_per_subsys
        ]
        return operator_list

    def _shift_harmonic_oscillator_potential(self, hamiltonian: sm.Expr) -> sm.Expr:
        """Shift extended-variable harmonic-oscillator potentials to flux minima.

        Solves the linear system that eliminates the linear-in-coordinate
        terms of `hamiltonian`, then substitutes the resulting flux offsets and
        rounds residual coefficients via :func:`round_symbolic_expr`.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian (after parameter substitution) to shift
        """
        # shifting the harmonic oscillator potential to the point of external fluxes
        flux_shift_vars = {}
        for var_index in self.var_categories["extended"]:
            flux_shift_vars[var_index] = sm.symbols("Δθ" + str(var_index))
            hamiltonian = hamiltonian.replace(
                sm.symbols(f"θ{var_index}"),
                sm.symbols(f"θ{var_index}") + flux_shift_vars[var_index],
            )  # substituting the flux offset variable offsets to collect the
            # coefficients later
        hamiltonian = hamiltonian.expand()

        flux_shift_equations = [
            hamiltonian.coeff(f"θ{var_index}").subs(
                [(f"θ{i}", 0) for i in self.var_categories["extended"]]
            )
            for var_index in flux_shift_vars.keys()
        ]  # finding the coefficients of the linear terms

        A, b = sm.linear_eq_to_matrix(
            flux_shift_equations, tuple(flux_shift_vars.values())
        )
        flux_shifts = sm.linsolve(
            (A, b), tuple(flux_shift_vars.values())
        )  # solving for the flux offsets

        if len(flux_shifts) != 0:
            flux_shifts = list(list(flux_shifts)[0])
        else:
            flux_shifts = []

        flux_shifts_dict = dict(zip(list(flux_shift_vars.values()), list(flux_shifts)))

        hamiltonian = hamiltonian.subs(
            list(flux_shifts_dict.items())
        )  # substituting the flux offsets to remove the linear terms
        hamiltonian = hamiltonian.subs(
            [(var, 0) for var in flux_shift_vars.values()]
        )  # removing the shift vars from the Hamiltonian
        # remove constants from Hamiltonian
        hamiltonian -= hamiltonian.as_coefficients_dict()[1]
        return round_symbolic_expr(hamiltonian.expand(), 16)
        # * ##########################################################################

    def _generate_sym_potential(self):
        """Return the symbolic potential extracted from the symbolic Hamiltonian.

        Sums the terms identified by :func:`is_potential_term` and rewrites
        ``cosθi`` / ``sinθi`` placeholder symbols into proper sympy ``cos`` /
        ``sin`` expressions on the dynamic variables.
        """
        # and bringing the potential into the same form as for the class Circuit
        potential_symbolic = 0 * sm.symbols("x")
        for term in self.hamiltonian_symbolic.as_ordered_terms():
            if is_potential_term(term):
                potential_symbolic += term
        for i in self.dynamic_var_indices:
            potential_symbolic = (
                potential_symbolic.replace(
                    sm.symbols(f"cosθ{i}"), sm.cos(1.0 * sm.symbols(f"θ{i}"))
                )
                .replace(sm.symbols(f"sinθ{i}"), sm.sin(1.0 * sm.symbols(f"θ{i}")))
                .subs(sm.symbols("I"), 1 / (2 * np.pi))
            )
        return potential_symbolic

    def _is_mat_mul_replacement_necessary(self, term: sm.Expr):
        """Return ``True`` if `term` mixes extended-variable factors via ``*``.

        Such terms must be rendered with matrix multiplication when the term
        is later evaluated on numerical operators.

        Parameters
        ----------
        term:
            symbolic Hamiltonian term to inspect
        """
        return (
            set(self.var_categories["extended"])
            & set([get_trailing_number(str(i)) for i in term.free_symbols])
        ) and "*" in str(term)

    def _replace_mat_mul_operator(self, term: sm.Expr):
        """Render `term` as a string using matrix-multiplication semantics.

        For the discretized basis, ``*`` between charge operators is rewritten
        as ``@``. For the harmonic basis, ``X**n`` on extended-variable
        operators becomes ``matrix_power(X, n)`` and remaining inter-operator
        ``*`` becomes ``@``.

        Parameters
        ----------
        term:
            symbolic Hamiltonian term to render as a Python expression string
        """
        if not self._is_mat_mul_replacement_necessary(term):
            return str(term)

        if self.ext_basis == "discretized":
            term_string = str(term)
            term_var_categories = [
                get_trailing_number(str(i)) for i in term.free_symbols
            ]
            if len(set(term_var_categories) & set(self.var_categories["extended"])) > 1:
                if all(["Q" in var.name for var in term.free_symbols]):
                    term_string = str(term).replace(
                        "*", "@"
                    )  # replacing all the * with @

        elif self.ext_basis == "harmonic":
            # replace ** with np.matrix_power
            if "**" in str(term):
                operators = [
                    match.replace("**", "")
                    for match in re.findall(r"[^*]+\*{2}", str(term), re.MULTILINE)
                ]
                exponents = re.findall(r"(?<=\*{2})\d", str(term), re.MULTILINE)

                new_string_list = []
                for idx, operator in enumerate(operators):
                    if get_trailing_number(operator) in self.var_categories["extended"]:
                        new_string_list.append(
                            f"matrix_power({operator},{exponents[idx]})"
                        )
                    else:
                        new_string_list.append(operator + "**" + exponents[idx])
                term_string = "*".join(new_string_list)
            else:
                term_string = str(term)

            # replace * with @ in the entire term
            if len(term.free_symbols) > 1:
                term_string = re.sub(
                    r"(?<=[^*])\*(?!\*)", "@", term_string, re.MULTILINE
                )
        return term_string

    def _generate_hamiltonian_sym_for_numerics(
        self,
        hamiltonian: sm.Expr | None = None,
        return_exprs: bool = False,
    ):
        """Generate a numerics-ready symbolic Hamiltonian.

        Substitutes identity placeholders for external fluxes / offset charges
        and (in the discretized basis) marks squared momentum operators ``Q**2``
        as ``Qs``. Stores the result in
        :attr:`Circuit._hamiltonian_sym_for_numerics` unless ``return_exprs`` is
        ``True``, in which case ``(hamiltonian, cos_terms)`` is returned.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian to process; defaults to
            ``self.hamiltonian_symbolic.expand()``
        return_exprs:
            if ``True``, return ``(hamiltonian, cos_terms)`` instead of mutating
            instance attributes
        """

        hamiltonian = hamiltonian or (
            self.hamiltonian_symbolic.expand()
        )  # applying expand is critical; otherwise the replacement of p^2 with ps2
        # would not succeed

        if self.ext_basis == "discretized":
            # marking the squared momentum operators with a separate symbol
            for i in self.var_categories["extended"]:
                hamiltonian = hamiltonian.replace(
                    sm.symbols(f"Q{i}") ** 2, sm.symbols("Qs" + str(i))
                )

        # associate an identity matrix with the external flux vars
        for ext_flux in self.external_fluxes:
            hamiltonian = hamiltonian.subs(
                ext_flux, ext_flux * sm.symbols("I") * 2 * np.pi
            )

        # associate an identity matrix with offset and free charge vars
        for charge_var in self.offset_charges + self.free_charges:
            hamiltonian = hamiltonian.subs(charge_var, charge_var * sm.symbols("I"))

        # finding the cosine terms
        cos_terms = sum(
            [term for term in hamiltonian.as_ordered_terms() if "cos" in str(term)]
        )
        if return_exprs:
            return hamiltonian, cos_terms
        setattr(self, "_hamiltonian_sym_for_numerics", hamiltonian)
        setattr(self, "junction_potential", cos_terms)

    def _get_eval_hamiltonian_string(self, H: sm.Expr) -> str:
        """Return the Python expression string for the Hamiltonian in the chosen basis.

        Each operator symbol is rewritten as a call to its ``_operator()``
        accessor and matrix multiplication is expressed via ``@`` per
        :meth:`_replace_mat_mul_operator`.

        Parameters
        ----------
        H:
            symbolic Hamiltonian to render
        """
        expr_dict = H.as_coefficients_dict()
        # removing zero terms
        expr_dict = {key: expr_dict[key] for key in expr_dict if expr_dict[key] != 0}
        terms_list = list(expr_dict.keys())
        coeff_list = list(expr_dict.values())

        H_string = ""
        for idx, term in enumerate(terms_list):
            term_string = f"{coeff_list[idx]}*{self._replace_mat_mul_operator(term)}"
            if float(coeff_list[idx]) > 0:
                term_string = "+" + term_string
            H_string += term_string

        # replace all position, sin and cos operators with methods
        H_string = re.sub(r"(?P<x>(θ\d)|(cosθ\d))", r"\g<x>_operator()", H_string)

        # replace all other operators with methods
        operator_symbols_list = flatten_list_recursive(
            [
                (
                    list(short_op_dict.values())
                    if isinstance(short_op_dict, dict)
                    else short_op_dict
                )
                for short_op_dict in list(self.vars.values())
            ]
        )
        operator_name_list = [symbol.name for symbol in operator_symbols_list]
        for operator_name in operator_name_list:
            if "θ" not in operator_name:
                H_string = H_string.replace(
                    operator_name, operator_name + "_operator()"
                )
        return H_string

    def _qutip_parameter_function_factory(
        self,
        parameter_expr: sm.Expr,
        free_var_func_dict: dict[str, Callable],
        lambdify_func: Callable,
    ) -> Callable:
        """Build a ``parameter_func(t, args)`` for time-dependent qutip terms.

        Each free symbol of ``parameter_expr`` is evaluated via the matching
        callable in ``free_var_func_dict``, and the resulting numerical values
        are passed to ``lambdify_func`` to produce the scalar coefficient at
        time ``t``.

        Parameters
        ----------
        parameter_expr:
            symbolic expression for the time-dependent coefficient
        free_var_func_dict:
            mapping from free-symbol name to a ``(t, args) -> float`` callable
        lambdify_func:
            sympy-lambdified evaluator for ``parameter_expr``
        """

        def parameter_func(t, args):
            return lambdify_func(
                *[
                    free_var_func_dict[sym.name](t, args)
                    for sym in parameter_expr.free_symbols
                ]
            )

        return parameter_func

    @check_sync_status_circuit
    def hamiltonian_for_qutip_dynamics(
        self,
        free_var_func_dict: dict[str, Callable],
        prefactor: float = 1.0,
        extra_terms: str | None = None,
    ) -> tuple[
        list[qt.Qobj | tuple[qt.Qobj, Callable]], sm.Expr, dict[qt.Qobj, sm.Expr]
    ]:
        r"""Return the Hamiltonian in a format suitable for ``qutip.mesolve``.

        Splits the Hamiltonian into time-independent and time-dependent
        contributions. Each free symbol of ``free_var_func_dict`` is treated
        as a time-varying parameter; remaining terms are collected into the
        fixed part. Optional ``extra_terms`` are added before splitting (useful
        e.g. for charge drives on a fluxonium without offset charge).

        Example::

            def flux_t(t, args):
                return 0.5 + 0.02 * np.sin(t * 2)
            free_var_func_dict = {"Φ1": flux_t}
            H = self.hamiltonian_for_qutip_dynamics(
                free_var_func_dict, extra_terms="0.1*ng*Q1"
            )

        Parameters
        ----------
        free_var_func_dict:
            mapping ``{"var": f}`` where ``f(t, args)`` returns the value of
            ``var`` at time ``t``
        prefactor:
            scalar multiplying the Hamiltonian and time-dependent operators
            (e.g. set to ``2*np.pi`` for qutip simulations)
        extra_terms:
            string parsed by sympy containing extra Hamiltonian terms (useful
            for custom drive operators)

        Returns
        -------
        Tuple ``(H_qutip, fixed_hamiltonian, time_dep_terms)`` where
        ``H_qutip`` is the qutip-mesolve-style nested list, ``fixed_hamiltonian``
        is the symbolic time-independent part, and ``time_dep_terms`` is a
        dict mapping each qutip operator to its symbolic coefficient.
        """
        free_var_names = list(free_var_func_dict.keys())
        free_var_symbols = [sm.symbols(sym_name) for sym_name in free_var_names]

        fixed_hamiltonian = 0 * sm.symbols("x")
        time_varying_hamiltonian = []

        # adding extra terms to the Hamiltonian
        if extra_terms:
            extra_terms_sym = sm.parse_expr(extra_terms)
            for extra_sym in extra_terms_sym.free_symbols:
                if (
                    extra_sym not in self.hamiltonian_symbolic.free_symbols
                    and extra_sym not in free_var_symbols
                ):
                    raise Exception(f"{extra_sym.name} is unknown.")
        else:
            extra_terms_sym = 0

        sym_hamiltonian = self._hamiltonian_sym_for_numerics + extra_terms_sym
        sym_hamiltonian = sym_hamiltonian.subs("I", 1).expand()

        expr_dict = sym_hamiltonian.expand().as_coefficients_dict()
        terms = list(expr_dict.keys())
        time_dep_terms: dict[sm.Expr, sm.Expr] = {}

        for term in terms:
            if len(list_intersection(list(term.free_symbols), free_var_symbols)) == 0:
                fixed_hamiltonian = fixed_hamiltonian + term * expr_dict[term]
                continue
            # if the term does have a free variable
            # expand trigonometrically
            should_trig_expand = any(
                [
                    (free_sym in term.free_symbols and term.coeff(free_sym) == 0)
                    for free_sym in free_var_symbols
                ]
            )
            term_expanded = term.expand(trig=should_trig_expand)

            term_expr_dict = term_expanded.as_coefficients_dict()
            terms_in_term = list(term_expr_dict.keys())
            for inner_term in terms_in_term:
                operator_expr, parameter_expr = inner_term.as_independent(
                    *free_var_symbols, as_Mul=True
                )

                if parameter_expr in time_dep_terms:
                    time_dep_terms[parameter_expr] = (
                        operator_expr * expr_dict[term] * term_expr_dict[inner_term]
                        + time_dep_terms[parameter_expr]
                    )
                else:
                    time_dep_terms[parameter_expr] = round_symbolic_expr(
                        operator_expr * expr_dict[term] * term_expr_dict[inner_term], 13
                    )

        for parameter_expr in time_dep_terms:
            # separating the time independent constants
            for sym in parameter_expr.free_symbols:
                if sym not in free_var_symbols:
                    parameter_expr = parameter_expr.subs(sym, getattr(self, sym.name))

            lambdify_func = sm.lambdify(
                list(parameter_expr.free_symbols), parameter_expr, "numpy"
            )
            parameter_func = self._qutip_parameter_function_factory(
                parameter_expr, free_var_func_dict, lambdify_func
            )

            operator_matrix = (
                self._evaluate_symbolic_expr(time_dep_terms[parameter_expr]) * prefactor
            )  # also multiplying the constant to the operator
            if operator_matrix == 0:
                continue
            time_varying_hamiltonian.append([operator_matrix, parameter_func])

        fixed_hamiltonian = fixed_hamiltonian.subs("I", 1)
        return (
            [self._evaluate_symbolic_expr(fixed_hamiltonian) * prefactor]
            + time_varying_hamiltonian,
            fixed_hamiltonian,
            dict((val, key) for key, val in time_dep_terms.items()),
        )

    # ****************************************************************
    # ***** Functions for pretty display of symbolic expressions *****
    # ****************************************************************
    @staticmethod
    def print_expr_in_latex(expr: sm.Expr | list["sm.Equality"]) -> None:
        """Print a sympy expression or a list of equalities in LaTeX.

        Parameters
        ----------
        expr:
            a sympy expressions or a list of equalities
        """
        if isinstance(expr, sm.Expr):
            display(Latex("$ " + sm.printing.latex(expr) + " $"))
        elif isinstance(expr, list):
            equalities_in_latex = "$ "
            for eqn in expr:
                equalities_in_latex += sm.printing.latex(eqn) + r" \\\ "
            equalities_in_latex = equalities_in_latex[:-4] + " $"
            display(Latex(equalities_in_latex))

    def __repr__(self) -> str:
        """Return the circuit's ``id_str`` as the textual representation."""
        return self._id_str

    def _repr_latex_(self):
        """Return a LaTeX/Markdown rendering of the circuit for IPython display.

        Falls back to the plain ``id_str`` if IPython is not available.
        """
        # string to describe the Circuit
        if not _HAS_IPYTHON:
            return self._id_str
        # Hamiltonian string
        H_latex_str = (
            "$H=" + sm.printing.latex(self.sym_hamiltonian(return_expr=True)) + "$"
        )
        # describe the variables
        cutoffs_dict = self.cutoffs_dict()
        var_str = "Operators (flux, charge) - cutoff: "
        if len(self.var_categories["periodic"]) > 0:
            var_str += "  \n Discrete Charge Basis:  "
            for var_index in self.var_categories["periodic"]:
                var_str += (
                    f"$(θ{var_index}, n{var_index}) - {cutoffs_dict[var_index]}$, "
                )

        var_str_discretized = "  \nDiscretized Phi basis:  "
        var_str_harmonic = "  \nHarmonic oscillator basis:  "

        for var_index in self.var_categories["extended"]:
            var_index_basis = self._basis_for_var_index(var_index)
            if var_index_basis == "discretized":
                var_str_discretized += (
                    f"$(θ{var_index}, Q{var_index}) - {cutoffs_dict[var_index]}$, "
                )
            if var_index_basis == "harmonic":
                var_str_harmonic += (
                    f"$(θ{var_index}, Q{var_index}) - {cutoffs_dict[var_index]}$, "
                )

        if var_str_discretized == "  \nDiscretized Phi basis:  ":
            var_str_discretized = ""
        if var_str_harmonic == "  \nHarmonic oscillator basis:  ":
            var_str_harmonic = ""

        display(Latex(H_latex_str))
        display(
            Latex(
                var_str
                + var_str_discretized
                + ("  \n" if var_str_discretized else "")
                + var_str_harmonic
            )
        )
        attr_value = lambda sym: getattr(self, sym.name)
        self._display_symbol_pairs(
            "Symbolic parameters", self.symbolic_params, self.symbolic_params.get
        )
        self._display_symbol_pairs("External fluxes", self.external_fluxes, attr_value)
        self._display_symbol_pairs("Offset charges", self.offset_charges, attr_value)
        self._display_symbol_pairs("Free charges", self.free_charges, attr_value)
        if self.hierarchical_diagonalization:
            display(Latex(f"System hierarchy: {self.system_hierarchy}"))
            display(Latex(f"Truncated Dimensions: {self.subsystem_trunc_dims}"))

    def _display_symbol_pairs(
        self, label: str, symbols, value_fn: Callable[..., Any]
    ) -> None:
        """Display ``label`` followed by ``(symbol, value_fn(symbol))`` pairs.

        No-op when ``symbols`` is empty. Used by ``_repr_latex_`` to render the
        circuit's symbolic parameters, external fluxes, offset/free charges.
        """
        if not symbols:
            return
        line = f"{label} (symbol, default value):  "
        for sym in symbols:
            line += f"$({sym.name}, {value_fn(sym)})$, "
        display(Latex(line))

    def _make_expr_human_readable(self, expr: sm.Expr, float_round: int = 6) -> sm.Expr:
        """Method returns a user readable symbolic expression for the current instance.

        Parameters
        ----------
        expr:
            A symbolic sympy expression
        float_round:
            Number of digits after the decimal to which floats are rounded

        Returns
        -------
        Sympy expression which is simplified to make it human readable.
        """
        expr_modified = expr
        # rounding the decimals in the coefficients
        # citation:
        # https://stackoverflow.com/questions/43804701/round-floats-within-an-expression
        # accepted answer
        for term in sm.preorder_traversal(expr):
            if isinstance(term, sm.Float):
                expr_modified = expr_modified.subs(term, round(term, float_round))

        for var_index in self.dynamic_var_indices:
            # replace sinθ with sin(..) and similarly with cos
            expr_modified = (
                expr_modified.replace(
                    sm.symbols(f"cosθ{var_index}"),
                    sm.cos(1.0 * sm.symbols(f"θ{var_index}")),
                )
                .replace(
                    sm.symbols(f"sinθ{var_index}"),
                    sm.sin(1.0 * sm.symbols(f"θ{var_index}")),
                )
                .replace(
                    (1.0 * sm.symbols(f"θ{var_index}")),
                    (sm.symbols(f"θ{var_index}")),
                )
            )
            # replace Qs with Q^2 etc
            expr_modified = expr_modified.replace(
                sm.symbols("Qs" + str(var_index)), sm.symbols(f"Q{var_index}") ** 2
            )
            expr_modified = expr_modified.replace(
                sm.symbols("ng" + str(var_index)), sm.symbols("n_g" + str(var_index))
            )
            # replace I by 1
            expr_modified = expr_modified.replace(sm.symbols("I"), 1)
        for ext_flux_var in self.external_fluxes:
            # removing 1.0 decimals from flux vars
            expr_modified = expr_modified.replace(1.0 * ext_flux_var, ext_flux_var)
        return expr_modified

    def sym_potential(
        self, float_round: int = 6, print_latex: bool = False, return_expr: bool = False
    ) -> sm.Expr | None:
        """Method prints a user readable symbolic potential for the current instance.

        Parameters
        ----------
        float_round:
            Number of digits after the decimal to which floats are rounded
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
                if set to True, all printing is suppressed and the function will silently
                return the sympy expression
        """
        potential = self._make_expr_human_readable(
            self.potential_symbolic, float_round=float_round
        )

        for external_flux in self.external_fluxes:
            potential = potential.replace(
                external_flux,
                sm.symbols(
                    "(2π" + "Φ_{" + str(get_trailing_number(str(external_flux))) + "})"
                ),
            )

        if return_expr:
            return potential
        if print_latex:
            print(latex(potential))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(potential)
        else:
            print(potential)
        return None

    def sym_hamiltonian(
        self,
        subsystem_index: int | None = None,
        float_round: int = 6,
        print_latex: bool = False,
        return_expr: bool = False,
    ) -> sm.Expr | None:
        """Prints a user readable symbolic Hamiltonian for the current instance.

        Parameters
        ----------
        subsystem_index:
            when set to an index, the Hamiltonian for the corresponding subsystem is
            returned.
        float_round:
            Number of digits after the decimal to which floats are rounded
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
            if set to True, all printing is suppressed and the function will silently
            return the sympy expression
        """
        if subsystem_index is not None:
            if not self.hierarchical_diagonalization:
                raise Exception(
                    "Hierarchical diagonalization was not enabled. Hence there "
                    "are no identified subsystems addressable by "
                    "subsystem_index."
                )
            # start with the raw system hamiltonian
            sym_hamiltonian = self._make_expr_human_readable(
                self.subsystems[subsystem_index].hamiltonian_symbolic.expand(),
                float_round=float_round,
            )
            # create PE symbolic expressions
            sym_hamiltonian_PE = self._make_expr_human_readable(
                self.subsystems[subsystem_index].potential_symbolic.expand(),
                float_round=float_round,
            )
            # obtain the KE of hamiltonian
            pot_symbols = (
                self.external_fluxes
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["extended"]
                ]
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["periodic"]
                ]
            )
            sym_hamiltonian_KE = 0 * sm.Symbol("x")
            for term in sym_hamiltonian.args:
                if term.free_symbols.isdisjoint(pot_symbols):
                    sym_hamiltonian_KE = sm.Add(sym_hamiltonian_KE, term)

            # add a symbolic 2pi
            for external_flux in self.external_fluxes:
                sym_hamiltonian_PE = self._make_expr_human_readable(
                    sym_hamiltonian_PE.replace(
                        external_flux,
                        sm.symbols(
                            "(2π"
                            + "Φ_{"
                            + str(get_trailing_number(str(external_flux)))
                            + "})"
                        ),
                    ),
                    float_round=float_round,
                )
            # substitute free charges
            for free_charge in self.free_charges:
                sym_hamiltonian_PE = sym_hamiltonian_PE.subs(
                    free_charge, getattr(self, free_charge.name)
                )
            # obtain system symbolic hamiltonian by glueing KE and PE
            sym_hamiltonian = sm.Add(
                sym_hamiltonian_KE, sym_hamiltonian_PE, evaluate=False
            )
        else:
            # create KE and PE symbolic expressions
            sym_hamiltonian = self._make_expr_human_readable(
                self.hamiltonian_symbolic.expand(),
                float_round=float_round,
            )
            # substitute free charges
            for free_charge in self.free_charges:
                sym_hamiltonian = sym_hamiltonian.subs(
                    free_charge, getattr(self, free_charge.name)
                )
            pot_symbols = (
                self.external_fluxes
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["extended"]
                ]
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["periodic"]
                ]
            )
            sym_hamiltonian_KE = 0 * sm.Symbol("x")
            for term in sym_hamiltonian.args:
                if term.free_symbols.isdisjoint(pot_symbols):
                    sym_hamiltonian_KE = sm.Add(sym_hamiltonian_KE, term)
            sym_hamiltonian_PE = self._make_expr_human_readable(
                self.potential_symbolic.expand(), float_round=float_round
            )
            # add a 2pi coefficient in front of external fluxes, since the the external
            # fluxes are measured in 2pi numerically
            for external_flux in self.external_fluxes:
                sym_hamiltonian_PE = sym_hamiltonian_PE.replace(
                    external_flux,
                    sm.symbols(
                        "(2π"
                        + "Φ_{"
                        + str(get_trailing_number(str(external_flux)))
                        + "})"
                    ),
                )
            # add the KE and PE and suppress the evaluation
            sym_hamiltonian = sm.Add(
                sym_hamiltonian_KE, sym_hamiltonian_PE, evaluate=False
            )
        if return_expr:
            return sym_hamiltonian
        if print_latex:
            print(latex(sym_hamiltonian))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(sym_hamiltonian)
        else:
            print(sym_hamiltonian)
        return None

    def sym_interaction(
        self,
        subsystem_indices: tuple[int],
        float_round: int = 6,
        print_latex: bool = False,
        return_expr: bool = False,
    ) -> sm.Expr | None:
        """Print the interaction between any set of subsystems for the current instance.
        It would print the interaction terms having operators from all the subsystems
        mentioned in the tuple.

        Parameters
        ----------
        subsystem_indices:
            Tuple of subsystem indices
        float_round:
            Number of digits after the decimal to which floats are rounded
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
            if set to True, all printing is suppressed and the function will silently
            return the sympy expression
        """
        interaction = sm.symbols("x") * 0
        for subsys_index_pair in itertools.combinations(subsystem_indices, 2):
            for term in self.subsystem_interactions[
                min(subsys_index_pair)
            ].as_ordered_terms():
                term_mod = term.subs(
                    [
                        (symbol, 1)
                        for symbol in self.external_fluxes
                        + self.offset_charges
                        + self.free_charges
                        + list(self.symbolic_params.keys())
                        + [sm.symbols("I")]
                    ]
                )
                interaction_var_indices = [
                    self.get_subsystem_index(get_trailing_number(symbol.name))
                    for symbol in term_mod.free_symbols
                ]
                if np.array_equal(
                    np.sort(interaction_var_indices), np.sort(subsystem_indices)
                ):
                    interaction += term
        for external_flux in self.external_fluxes:
            interaction = self._make_expr_human_readable(
                interaction.replace(external_flux, external_flux / (2 * np.pi)),
                float_round=float_round,
            )
            interaction = interaction.replace(
                external_flux,
                sm.symbols(
                    "(2π" + "Φ_{" + str(get_trailing_number(str(external_flux))) + "})"
                ),
            )
        if return_expr:
            return interaction
        if print_latex:
            print(latex(interaction))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(interaction)
        else:
            print(interaction)
        return None

    def operator_names_in_hamiltonian_symbolic(self) -> list[str]:
        """Return the names of all operator symbols in the symbolic Hamiltonian.

        Excludes offset charges (``ng``), external fluxes (``Φ``), and any
        symbolic parameters.
        """
        return [
            symbol.name
            for symbol in self.hamiltonian_symbolic.free_symbols
            if ("ng" not in symbol.name and "Φ" not in symbol.name)
            and symbol not in self.symbolic_params
        ]

    def offset_charge_transformation(self) -> None:
        """Print the offset-charge transformation from node charges to periodic vars.

        Renders the equations relating each ``ng_i`` (offset charge of a
        periodic variable) to a linear combination of node-offset charges
        ``q_ni``, using the inverse of :attr:`transformation_matrix`. Output
        is rendered via LaTeX in IPython, plain-printed otherwise.
        """
        if not hasattr(self, "symbolic_circuit"):
            raise Exception(
                f"{self._id_str} instance is not generated from a SymbolicCircuit instance, and hence does not have any associated branches."
            )
        trans_mat = np.linalg.inv(self.transformation_matrix.T)
        node_offset_charge_vars = [
            sm.symbols(f"q_n{index}")
            for index in range(
                1, len(self.symbolic_circuit.nodes) - self.is_grounded + 1
            )
        ]
        periodic_offset_charge_vars = [
            sm.symbols(f"ng{index}")
            for index in self.symbolic_circuit.var_categories["periodic"]
        ]
        periodic_offset_charge_eqns = []
        for idx, node_var in enumerate(periodic_offset_charge_vars):
            periodic_offset_charge_eqns.append(
                self._make_expr_human_readable(
                    sm.Eq(
                        periodic_offset_charge_vars[idx],
                        np.sum(trans_mat[idx, :] * node_offset_charge_vars),
                    )
                )
            )
        if _HAS_IPYTHON:
            self.print_expr_in_latex(periodic_offset_charge_eqns)
        else:
            print(periodic_offset_charge_eqns)
