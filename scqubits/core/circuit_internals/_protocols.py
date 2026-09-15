"""Type-only protocol declaring the cross-mixin surface of ``Circuit`` /
``Subsystem``.

Every mixin under ``circuit_internals`` references attributes and methods
that are *not* defined locally — they live on a sibling mixin or on the
concrete ``Subsystem`` / ``Circuit`` composition. Historically each mixin
re-declared the cross-mixin surface inside its own ``if TYPE_CHECKING:``
block, which produced ~5 parallel declaration sets that drifted (different
type hints for the same attribute, missing methods, etc.).

``CircuitProtocol`` consolidates the union of those declarations into one
location so:

- A new shared attribute is declared exactly once.
- ``mypy`` cannot accept an inconsistent type for the same attribute
  across mixins (because there is only one declaration).
- A reader looking up "what does ``self.X`` resolve to in mixin Y" has
  one file to read.

Each mixin under ``circuit_internals`` inherits from ``CircuitProtocol``
unconditionally::

    from scqubits.core.circuit_internals._protocols import CircuitProtocol

    class SomeMixin(ABC, CircuitProtocol):
        ...

At runtime ``CircuitProtocol`` is an empty class (its body is gated under
``TYPE_CHECKING``), so the addition to the MRO is functionally a no-op.
``mypy`` sees the full attribute and method surface declared inside the
class body.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal


class CircuitProtocol:
    """Single source of truth for the cross-mixin attribute / method surface.

    Inherit from this in each ``circuit_internals`` mixin so ``mypy``
    resolves shared-method ``self.<x>`` references without requiring
    per-mixin re-declarations.

    No methods on this class have an implementation; it is purely a
    declaration host. ``Subsystem`` / ``Circuit`` provide the actual
    attributes and methods at composition time.
    """

    if TYPE_CHECKING:
        # The imports below are only resolved by ``mypy``; at runtime the
        # entire ``if`` branch is skipped, leaving ``CircuitProtocol`` as
        # an empty class.
        import qutip as qt  # noqa: F401
        import sympy as sm  # noqa: F401

        from numpy import ndarray  # noqa: F401
        from scipy.sparse import csc_matrix  # noqa: F401

        from scqubits.core.circuit import Subsystem  # noqa: F401
        from scqubits.core.hilbert_space import HilbertSpace  # noqa: F401

        # --------------------------------------------------------------
        # Topology / categorization state
        # --------------------------------------------------------------
        hierarchical_diagonalization: bool
        is_child: bool
        is_grounded: bool
        is_purely_harmonic: bool
        parent: Any
        subsystems: list[Any]
        affected_subsystem_indices: list[int]
        var_categories: dict[
            Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]
        ]
        dynamic_var_indices: list[int]
        cutoff_names: list[str]
        discretized_phi_range: dict[int, Any]

        # --------------------------------------------------------------
        # Symbolic state
        # --------------------------------------------------------------
        external_fluxes: list[Any]
        offset_charges: list[Any]
        free_charges: list[Any]
        symbolic_params: dict[Any, Any]
        symbolic_circuit: Any
        hamiltonian_symbolic: Any
        potential_symbolic: Any
        junction_potential: Any
        subsystem_interactions: Any
        subsystem_hamiltonians: dict[int, Any]
        closure_branches: Any
        transformation_matrix: Any
        affine_transformation_matrix: Any
        use_dynamic_flux_grouping: bool

        # --------------------------------------------------------------
        # Numerical / configuration state
        # --------------------------------------------------------------
        ext_basis: Any
        hilbert_space: Any
        truncated_dim: int
        type_of_matrices: str
        operators_by_name: Any
        vars: dict[str, Any]
        system_hierarchy: list[Any]
        subsystem_trunc_dims: list[Any]
        evals_method: Any
        evals_method_options: Any
        esys_method: Any
        esys_method_options: Any

        # --------------------------------------------------------------
        # Internal flags / caches
        # --------------------------------------------------------------
        _frozen: bool
        _bare_eigensystem: Any
        _out_of_sync: bool
        _out_of_sync_with_parent: bool
        _user_changed_parameter: bool
        _user_changed_parameter_dict: dict[str, Any]
        _internal_updates_required: bool
        _operators_by_name: Any
        _bare_eigensys_for_subsystems: list[Any]
        _hamiltonian_sym_for_numerics: Any
        _default_grid_phi: Any
        _id_str: str

        broadcast: Callable[..., Any]

        # --------------------------------------------------------------
        # Methods provided by sibling mixins / the residual
        # ``CircuitRoutines``
        # --------------------------------------------------------------
        def hilbertdim(self) -> int: ...
        def cutoffs_dict(self) -> dict[int, int]: ...
        def discretized_grids_dict_for_vars(self) -> dict[int, Any]: ...
        def exp_i_operator(self, var_symbol: sm.Symbol, prefactor: float) -> Any: ...
        def get_subsystem_index(self, var_index: int) -> int: ...
        def return_root_child(self, var_index: int) -> Subsystem: ...
        def get_osc_param(self, var_index: int, which_param: str = ...) -> float: ...
        def generate_bare_esys(self, *args: Any, **kwargs: Any) -> Any: ...
        def get_ext_basis(self) -> str | list[str]: ...
        def get_operator_by_name(
            self,
            operator_name: str,
            power: int | None = ...,
            bare_esys: dict[int, tuple] | None = ...,
        ) -> qt.Qobj: ...
        def identity_wrap_for_hd(
            self,
            operator: csc_matrix | ndarray | None,
            child_instance: Subsystem,
            bare_esys: dict[int, tuple] | None = ...,
        ) -> qt.Qobj: ...
        def eigensys(self, evals_count: int = ...) -> tuple[ndarray, ndarray]: ...
        def _kron_operator(self, op: Any, var_index: int) -> Any: ...
        def _sparsity_adaptive(self, matrix: Any) -> Any: ...
        def _identity_qobj(self) -> qt.Qobj: ...
        def _identity(self) -> Any: ...
        def _get_cutoff_value(self, var_index: int) -> int: ...
        def _set_vars(self) -> None: ...
        def _set_operators(self) -> dict[str, Callable]: ...
        def _generate_subsystems(self, *args: Any, **kwargs: Any) -> None: ...
        def _update_interactions(self, recursive: bool = ...) -> None: ...
        def _diagonalize_purely_harmonic_hamiltonian(
            self, return_osc_dict: bool = ...
        ) -> Any: ...
        def _set_harmonic_basis_osc_params(
            self, hamiltonian: Any | None = ...
        ) -> Any: ...
        def _evaluate_matrix_cosine_terms(
            self,
            junction_potential: sm.Expr,
            bare_esys: dict[int, tuple] | None = ...,
        ) -> qt.Qobj: ...

        # --------------------------------------------------------------
        # Bridges to ``CircuitSymMethods`` (declared as ``Callable``
        # rather than as defs because they are bound dynamically at
        # composition time).
        # --------------------------------------------------------------
        _find_and_set_sym_attrs: Callable[..., Any]
        _generate_sym_potential: Callable[..., Any]
        _generate_hamiltonian_sym_for_numerics: Callable[..., Any]
        _sym_subsystem_hamiltonian_and_interactions: Callable[..., Any]
        _is_expression_purely_harmonic: Callable[..., Any]
        _evaluate_symbolic_expr: Callable[..., Any]
        _basis_for_var_index: Callable[..., Any]
        _get_eval_hamiltonian_string: Callable[..., Any]
        _is_diagonalization_necessary: Callable[..., Any]
        _operator_from_sym_expr_wrapper: Callable[[sm.Expr], Any]
