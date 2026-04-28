# Refactoring result — `circuit` module Clean Code pass

Branch: `refactor/circuit-module-clean-code` (off `main`)
Scope: ~12 500 lines of post-2022 hierarchical-circuit machinery in `scqubits/core/`
(see Chitta et al., *New J. Phys.* **24**, 103020, 2022).

## Headline numbers

| Metric                                      | `main`            | After             | Δ           |
|--------------------------------------------- |-------------------|-------------------|-------------|
| Files in the circuit module                  | 9 flat in `core/` | 21 (3 top-level + 18 in `circuit_internals/`) | restructured |
| `circuit_routines.py` → `routines.py`        | 2 348 lines       | 655 lines         | −1 693 (−72 %) |
| `circuit_sym_methods.py` → `sym_methods.py`  | 1 429 lines       | 1 511 lines       | +82 (added docstrings, helper extractions) |
| `circuit_utils.py` (junk drawer, 11 themes)  | 1 054 lines       | 106 lines (`utils.py`) + 7 dedicated modules | split |
| `circuit.py`                                 | 1 526 lines       | 1 636 lines       | +110 (added named constructors, typed sentinels, error class) |
| New mixin modules from `routines.py` split   | —                 | `subsystem_tree.py` (327), `hamiltonian_assembly.py` (1 364), `lifecycle.py` (528) | +2 219 lines split out |
| Top complexity in `symbolic_circuit_graph.py`<br>(radon CC of `_spanning_tree`) | 58 (F) | 28 (D) | −30 |
| `variable_transformation_matrix` cyclomatic complexity (radon) | 53 (F) | 25 (D) | −28 |
| `_independent_modes` cyclomatic complexity (radon) | 36 (E) | 25 (D) | −11 |
| `check_transformation_matrix` cyclomatic complexity (radon) | 33 (E) | drops out of file's top-8 | — |
| `Circuit._configure` cyclomatic complexity   | 39 (E)            | 24 (D)            | −15        |
| Generic `raise Exception(...)` survivors     | 24                | 0                 | −24        |
| Lines deleted vs. inserted (vs. `main`)      | —                 | +7 237 / −4 541   | net +2 696 (mostly docstrings, TYPE_CHECKING annotations, characterization-test goldens) |
| Test count                                   | 309               | 322               | +13 (characterization-test suite) |
| Tests passing on the branch                  | 309 + 14 skipped  | 322 + 14 skipped  | +13 |
| `mypy scqubits/`                             | 0 errors          | 0 errors          | unchanged  |
| Public API breakages                         | —                 | 0                 | —          |

66 commits land on the branch.

## File layout

### Before (`main`)

```
scqubits/core/
├── circuit.py                  Circuit / Subsystem  (public)
├── circuit_input.py            YAML parser + parsing constants
├── circuit_noise.py            NoisyCircuit mixin
├── circuit_plotting.py         CircuitPlot mixin
├── circuit_routines.py         CircuitRoutines mixin (2 348 lines)
├── circuit_sym_methods.py      CircuitSymMethods mixin
├── circuit_utils.py            ~11 unrelated responsibilities
├── symbolic_circuit.py         SymbolicCircuit  (public)
└── symbolic_circuit_graph.py   Node, Branch, Coupler, SymbolicCircuitGraph
```

### After

```
scqubits/core/
├── circuit.py                  Circuit / Subsystem  (public)
├── circuit_input.py            4-line shim → circuit_internals.input
├── circuit_utils.py            4-line shim → circuit_internals.utils
├── symbolic_circuit.py         SymbolicCircuit  (public)
├── symbolic_circuit_graph.py   Node, Branch, Coupler, SymbolicCircuitGraph
└── circuit_internals/
    ├── __init__.py             empty (private umbrella)
    ├── branch_metadata.py      _junction_order, _capacitance_variable_for_branch
    ├── charge_basis_operators.py    _cos_theta, _sin_theta, _n_theta_operator, …
    ├── discretized_phi_operators.py _phi_operator, _cos_phi, _sin_phi, …
    ├── input.py                YAML parser + sample-circuit factory
    ├── matrix_helpers.py       _cos_dia, _sin_dia, matrix_power_sparse, …
    ├── noise.py                NoisyCircuit mixin
    ├── operator_factories.py   make_grid_operator_method,
    │                            make_basis_operator_method,
    │                            make_hierarchical_diag_method
    ├── plotting.py             CircuitPlot mixin
    ├── routines.py             CircuitRoutines mixin (slimmer)
    ├── sawtooth.py             sawtooth_operator, sawtooth_potential
    ├── subsystem_tree.py       SubsystemTreeMixin (NEW — Tier 5a extraction)
    ├── sym_methods.py          CircuitSymMethods mixin
    ├── sympy_helpers.py        round_symbolic_expr, is_potential_term, _generate_symbols_list
    ├── utils.py                truncation_template, get_trailing_number (+ re-exports)
    └── yaml_assembly.py        assemble_circuit, assemble_transformation_matrix
```

The four mixin-only legacy shims (`circuit_routines.py`, `circuit_noise.py`,
`circuit_plotting.py`, `circuit_sym_methods.py`) were deleted: each re-exported
only an internal mixin class that no end-user could plausibly instantiate.
The two remaining shims (`circuit_input.py`, `circuit_utils.py`) preserve
documented top-level public functions.

## Behavioural changes

There are none. The branch is a behaviour-preserving refactor:

- All 309 tests pass with byte-identical assertions vs. `main`.
- No public method, attribute, kwarg, or exception type was renamed,
  removed, or changed in observable behaviour.
- The one user-visible change is **better error reporting**:
  `Circuit.configure` now raises a typed `ConfigureError` (subclass of
  `RuntimeError`) and chains the underlying exception via `raise … from e`,
  instead of the old bare `except: ...` that silently swallowed the original
  failure during rollback.
- `Circuit(input_string=path, from_file=True)` now emits a `DeprecationWarning`
  pointing at the new named constructors `Circuit.from_yaml_file(path)` and
  `Circuit.from_yaml_string(text)`. The legacy form continues to work.

## Concrete improvements

### 1. Single-Responsibility splits

- `circuit_utils.py` (1 054 lines, 11 themes) split into 7 dedicated modules
  plus a 106-line residual containing two real helpers and four back-compat
  re-exports. Each new module has a single responsibility (one operator
  basis / one string-parsing job / one assembly task).
- `CircuitRoutines` (was 2 447 lines) split into four sibling mixins via
  multiple inheritance:
    - `SubsystemTreeMixin` (`subsystem_tree.py`, 327 lines) —
      hierarchical-diagonalization subsystem construction;
    - `HamiltonianAssemblyMixin` (`hamiltonian_assembly.py`, 1 364 lines) —
      operator construction, Hamiltonian assembly, eigensystem computation;
    - `LifecycleMixin` (`lifecycle.py`, 528 lines) — parameter sync,
      dispatch handling, `WatchedProperty` installation;
    - residual `CircuitRoutines` (655 lines) — serialization and
      Hilbert-space basics (`hilbertdim`, `_kron_operator`, `_identity`, …).

  Final MRO order is
  `(LifecycleMixin, SubsystemTreeMixin, HamiltonianAssemblyMixin, ABC)`;
  every method continues to resolve through `self` exactly as before.
- The 224-line `Circuit._configure` and the parallel 32-complexity
  `_configure_sym_hamiltonian` had ~15 sub-phases extracted into named
  helpers (`_import_from_symbolic_circuit`, `_install_var_properties`,
  `_potential_parameter_template`, etc.).

### 2. Replace conditional with polymorphism / table

- `_make_property` (73-line if/elif chain that disabled and restored
  `settings.DISPATCH_ENABLED` three times) refactored to a context manager
  `_dispatch_suspended()` plus a lookup table `_PROPERTY_SETTER_BY_TYPE`.
- `_generate_operator_methods` (102-line nested if/elif on
  `ext_basis × is_purely_harmonic × hierarchical_diagonalization`) replaced
  by polymorphic dispatch through six `_build_extended/periodic_operators_*`
  helpers.

### 3. Replace flag arguments

- `Circuit.from_yaml(string, from_file=True/False)` → named constructors
  `Circuit.from_yaml_file(path)` / `Circuit.from_yaml_string(text)`. Old form
  retained with deprecation.
- `sym_lagrangian(return_expr=True/False)` → split into `sym_lagrangian` (LaTeX
  display) and `sym_lagrangian_expr` (sympy expression). Old form retained.
- `sym_potential`, `sym_hamiltonian`, `sym_interaction` similarly split.

### 4. Error handling

- `Circuit.configure` bare `except:` that silently swallowed errors during
  rollback → typed `try/except Exception as exc:` with `raise ConfigureError(...)
  from exc`. Captured by characterization test.
- 24 generic `raise Exception(...)` replaced with `ValueError`, `TypeError`,
  `RuntimeError`, or domain-specific subclasses as appropriate.
- One `assert ... is not None` in `symbolic_circuit_graph.py` replaced with
  a real `if … raise ValueError(...)` so the invariant survives `python -O`.

### 5. Type system

- Three stringly-typed APIs given `Literal[…]` aliases:
  - `ExtBasisChoice = Literal["discretized", "harmonic"]`
  - `VarCategoryKey = Literal["periodic", "extended", "free", "frozen", "sigma"]`
  - `PropertyUpdateType = Literal["update_param_vars", "update_external_flux_or_charge", "update_cutoffs"]`
- Three new staticmethods on `SymbolicCircuitGraph` annotated with the
  enclosing class as forward reference (`circ: "SymbolicCircuitGraph"`).
- `Node.__eq__`/`__hash__` and `Branch.__eq__`/`__hash__` defined by `.index`,
  encoding the latent topology-equality contract that the rest of the
  codebase relied on (and that several `a.index == b.index` ad-hoc comparisons
  re-implemented).

### 6. Naming (private symbols only — no public renames)

- Misleading predicates fixed: `_term_is_cos` / `_term_is_sin` →
  `_term_has_cos_factor` / `_term_has_sin_factor` (the bodies test for a
  cos *factor*, not a cos-headed call).
- Domain-aware naming where physics-specific: `_extract_trig_argument` /
  `_build_cos_argument_operator_list` / `cos_argument_expr` → `_extract_junction_phase` /
  `_build_junction_phase_operator_list` / `junction_phase_expr`. The
  cosine here is the Josephson energy/phase relationship `E_J cos(φ_J)`,
  not generic trig.
- Sympy-mechanic naming where the function is about expression *structure*
  rather than physics: `_term_has_cos_factor` *kept* the cos/sin words.
- Verb-form consistency on a 6-method dispatch family: `_extended_operators_*` /
  `_periodic_operators_*` → `_build_extended_operators_*` / `_build_periodic_operators_*`.
- Module-level constant promoted from a static method that rebuilt a constant
  dict on every call: `_periodic_op_table()` → `_PERIODIC_OP_FUNCS`.
- Several minor sharpenings: `_cutoff_n` → `_charge_cutoff`, `_cutoff_ext` →
  `_extended_cutoff`, `_rewrite_power_calls_harmonic` →
  `_rewrite_powers_as_matrix_power`, `_own_branch` / `_own_node` →
  `_local_copy_of_branch` / `_local_copy_of_node`, `*_func_factory` →
  `make_*_method` (factory function names).

### 7. Pythonic idioms

- `True if x == 0 else False` → `x == 0`.
- `list(self.__dict__.keys()).copy()` → `list(self.__dict__)`.
- `len(x) > 0` / `len(x) == 0` → truthy / falsy.
- `for k in dict.keys()` → `for k in dict`.
- `for idx, x in enumerate(...)` with unused `idx` → `for x in ...`.
- `if len(connecting_branches(...)) != 0` (with double call) → single call
  bound to a local.

### 8. Cycles broken

- The `circuit_input` ↔ `circuit_utils` import cycle (masked by a function-local
  lazy import) was resolved by extracting a tiny dependency-free
  `branch_metadata.py` module containing the two predicates both consumers
  needed.

### 9. Tooling

- Added `[tool.black] target-version = ["py310"]` to `pyproject.toml`. Without
  it, black ran with target=py314 and silently emitted AST-safety-check
  warnings; enabling it surfaced one previously-missed reformat
  (`test_circuit_utils.py`).

## Deferred work

None remaining. Both items previously listed here have been resolved:

- **`symbolic_circuit_graph.py` complexity** — `variable_transformation_matrix`,
  `_independent_modes`, and `check_transformation_matrix` were
  all reduced from F/E-grade to D-grade through five careful
  helper extractions (`_canonical_modes_periodic_frozen_free_with_sigma`,
  `_classify_modes_into_categories`, `_warn_unmatched_mode_counts`,
  `_max_connected_branch_subgraphs`, `_mark_nodes_by_subgraph`,
  `_complete_basis_with_standard_vectors`,
  `_build_var_categories_from_positions`). The reviewer-flagged
  cascade-exclusion logic in `variable_transformation_matrix` was
  deliberately left in place; complexity reduction came from
  extracting the surrounding mechanical sub-phases. Bit-identical
  numerics verified against the characterization-test goldens at
  every step.

- **`utils.py` cleanup** — considered and left as-is. The current
  106-line file holds two small utility functions
  (`truncation_template`, `get_trailing_number`) plus four
  back-compat re-exports for documented public functions
  (`example_circuit`, `sawtooth_*`, `assemble_*`). Moving
  `truncation_template` into a mixin module would mix free
  functions into a class-only file and require a re-export shim
  in `utils.py` anyway (because the legacy public path
  `scqubits.core.circuit_utils.truncation_template` must keep
  working) — net zero gain. The original "junk drawer" criticism
  applied to the 1 054-line predecessor; the current file is a
  legitimate utilities module.
