# Refactoring Plan — `circuit` module (Clean Code pass)

## Context

The `circuit` family in `scqubits/core/` (≈12,500 lines across 9 files) carries
the post-2022 hierarchical-circuit machinery (Chitta et al., *New J. Phys.* 24,
103020, 2022). Several Clean-Code smells have grown into structural problems:

- 70-method `CircuitRoutines` mixin and 35-method `CircuitSymMethods` mixin —
  Single-Responsibility violations.
- `Circuit.configure` rolls back via a bare `except:` and **silently swallows
  the original error** (no re-raise).
- `circuit_utils.py` is a junk drawer with ~11 unrelated responsibilities.
- An import cycle `circuit_input ↔ circuit_utils` is masked by a
  function-local lazy import.
- Two long dispatcher methods (`_make_property` 73 lines,
  `_generate_operator_methods` 102 lines) hide logic behind string-keyed
  if-elif chains with copy-pasted ceremony.
- `_configure` (224 lines) does ~15 distinct things sequentially; the same
  18-attribute `setattr/getattr` block also appears in `from_yaml`.

This plan proposes three batches by ascending risk. **Only Batch A is
executed in this session.** Batches B and C are documented for follow-up.

## Working conventions

- Branch: `refactor/circuit-module-clean-code` off `main`.
- After every change: `isort -w88 --lbt 1 --profile black .` then `black .`
  (order matters; black-then-isort leaves residual diffs).
- Test gate: fast subset
  `pytest -v --pyargs scqubits.tests.test_circuit
  scqubits.tests.test_circuit_utils scqubits.tests.test_circuit_sym_methods`
  must pass after every refactor; full
  `pytest -v --pyargs scqubits --num_cpus=2` must pass before the last commit.
  Consider ways to shorten tests without losing coverage to increase throughput,
  if appropriate.
- Occasionally check with mypy that type annotations are not messed up.
- No public API change: no removed/renamed public methods or kwargs.
  Deprecation shims with `DeprecationWarning` are acceptable.
- `_frozen = True` blocks ad-hoc attribute creation; new attributes must be
  set in `__init__` / `_configure` paths.
- One refactor per commit. Message format:
  `refactor(circuit): <description> [Clean Code: <principle>]`.
- `circuit_noise.py` is out of scope except for trivial import fix-ups.

## Verified findings

Each finding was verified against `main` at commit `1755e27b`. Line numbers
reflect current code. Where the user-prompt's pre-identified description
differed from the actual code, the discrepancy is flagged inline.

**Note on revisions:** the original draft was independently reviewed; three
findings (A6 scope, A7 scope, finding #3 wording) were corrected.
Documented inline.

1. **`circuit.py:418` — `Circuit.__init__` is a kwarg dispatcher.** Validates
   the mutually-exclusive `input_string` / `symbolic_hamiltonian` pair, then
   delegates to `from_yaml` (line 454) or `_from_symbolic_hamiltonian`
   (line 470). **Verified.**

2. **`circuit.py:541` — `from_yaml` is an instance method with `from_file:
   bool` flag argument** that switches "string is a path" vs. "string is YAML
   content." Docstring says "deprecated and will not be supported in future
   releases" but no `DeprecationWarning` is emitted. **Verified.**

3. **`circuit.py:824` — bare `except:` followed by an opaque re-raise.**
   The except block restores `old_*`, re-calls
   `_configure` / `_configure_sym_hamiltonian` with old values, and at
   `circuit.py:850` raises a fresh
   `Exception("Configure failed due to incorrect parameters.")`. Both the
   original error type and the original cause are discarded (no
   `raise ... from e`). The pre-prompt's description matches the structure
   but the original draft of this plan claimed "never raises" — that was
   wrong; line 850 does raise, just badly. Additionally,
   `hasattr(self, "symbolic_circuit")` is checked **four times** in this
   single method (lines 788, 798, 828, 832).

   The behavior change introduced by A1 is therefore narrower than it
   appeared at first reading: today raises a generic `Exception` after
   silent rollback; A1 raises a typed exception with `from e` after the
   same rollback. Callers that pattern-match on `Exception` still match;
   callers that introspect `.__cause__` get more information than before.

4. **`circuit.py:1065` — `_configure` is 224 lines** (1065–1288). Roughly 15
   responsibilities run in sequence: rebuild `symbolic_circuit` if matrices
   changed; copy 18 attributes from `symbolic_circuit`; create cutoff
   `WatchedProperty`s for periodic and extended variables; create properties
   for symbolic params; set discretized-phi range; create properties for
   external fluxes / offset / free charges; switch to dense if 1 variable;
   shift harmonic potential; branch on `hierarchical_diagonalization`.
   **Verified.**

5. **The 18-attribute copy block is duplicated** in `from_yaml`
   (`circuit.py:621–641`) and `_configure` (`circuit.py:1156–1177`). Same
   `required_attributes` list, same `setattr/getattr` loop. The `_configure`
   version adds one item (`is_purely_harmonic`); otherwise identical.

6. **`circuit_routines.py:1692` — `_generate_operator_methods` is 102 lines**
   (1692–1794). Nested if-elif on
   `ext_basis × is_purely_harmonic × hierarchical_diagonalization`. The
   periodic-variable block (1771–1789) repeats the
   hierarchical-vs-non-hierarchical branch a second time. Prime
   polymorphic-extraction candidate (Batch B).

7. **`circuit_routines.py:742` — `_make_property` is 73 lines** (742–814).
   Three near-identical `elif property_update_type == "..."` branches each
   contain the same 5-line dispatch enable/disable dance (lines 774–779,
   784–789, 794–799). Replace with a `_dispatch_suspended()`
   `@contextmanager` and a `{property_update_type: setter_method}` lookup
   table.

8. **`circuit_routines.py:648, 687, 718` — three `_set_property_and_update_*`
   methods** share an identical "propagate to affected subsystems" 5-line
   block (`circuit_routines.py:682–685`, `:712–716`, `:736–740`). The
   `param_vars` variant has additional pre-block work
   (`_mark_all_subsystems_as_affected`); the trailing 5 lines are identical.

9. **Import dependency masked by a function-local lazy import.**
   `circuit_utils.py:33` imports `circuit_input` at module top (used by
   `assemble_circuit` at lines 735–740 for `remove_comments`,
   `remove_branchline`, `strip_empty_lines`, `BRANCHES.parse_string`).
   `circuit_input.py:141` lazy-imports `_junction_order` from
   `circuit_utils` inside `find_jj_order` to avoid the cycle. After A6 the
   cycle becomes one-way (`circuit_utils → circuit_input` only) and the
   lazy import disappears. The `circuit_utils → circuit_input` direction
   remains and is fine.

   `_junction_order` and `_capacitance_variable_for_branch` are also
   imported by `symbolic_circuit_graph.py:17–18` and
   `symbolic_circuit.py:28–29`, plus `tests/test_circuit_utils.py`. A6
   touches **5 files**, not 3 (see the corrected table row below).
   New module renamed `branch_metadata.py` to avoid collision with the
   `BRANCH_TYPES` dict in `circuit_input.py:50`.

10. **`circuit_utils.py` — 1,054 lines spanning ~11 distinct responsibilities:**
    - branch-type metadata (`_junction_order`,
      `_capacitance_variable_for_branch`)
    - sawtooth physics (`sawtooth_operator`, `sawtooth_potential`)
    - hierarchical-diagonalization helpers (`truncation_template`,
      `keep_terms_for_subsystem`)
    - string utility (`get_trailing_number`)
    - discretized-phi operators (`_identity_phi`, `_phi_operator`,
      `_i_d_dphi_operator`, `_i_d2_dphi2_operator`, `_cos_phi`, `_sin_phi`)
    - charge-basis operators (`_identity_theta`, `_n_theta_operator`,
      `_exp_i_theta_operator`, `_exp_i_theta_operator_conjugate`,
      `_cos_theta`, `_sin_theta`)
    - sympy helpers (`_generate_symbols_list`, `is_potential_term`,
      `round_symbolic_expr`)
    - example circuit YAMLs (`example_circuit`)
    - operator-method factories (`grid_operator_func_factory`,
      `hierarchical_diagonalization_func_factory`, `operator_func_factory`)
    - dense/sparse matrix helpers (`_cos_dia`, `_sin_dia`, `_cos_dia_dense`,
      `_sin_dia_dense`, `matrix_power_sparse`)
    - YAML assembly (`yaml_like_out_with_pp`, `assemble_circuit`,
      `assemble_transformation_matrix`)

    Split per responsibility (Batch B).

11. **Three discretized-phi operator builders share an identical 5-line
    pattern** (`circuit_utils.py:201–218` `_phi_operator`, `:251–268`
    `_cos_phi`, `:271–288` `_sin_phi`). Each:
    ```
    pt_count = grid.pt_count
    matrix = sparse.dia_matrix((pt_count, pt_count))
    diag_elements = <fn>(grid.make_linspace())
    matrix.setdiag(diag_elements)
    return matrix.tocsc()
    ```
    Collapsible to a `_diag_from_function(grid, values_fn)` helper.
    `_identity_phi` (`circuit_utils.py:185–198`) does *not* fit this
    pattern — it uses `sparse.identity(pt_count, format="csc")` directly
    and is excluded from A7's scope.

12. **Other long methods worth flagging for Batch B/C:**
    - `symbolic_circuit.py:320` — `configure`, 116 lines (320–435).
    - `circuit_routines.py:1975` — `_hamiltonian_for_harmonic_extended_vars`,
      92 lines (1975–2067). Uses
      `eval(H_LC_str, replacement_dict)` (line 2065) — a string-Hamiltonian-
      eval pattern that deserves separate review.
    - `circuit_routines.py:211` — `_diagonalize_purely_harmonic_hamiltonian`,
      89 lines.
    - `circuit_routines.py:1603` — `_purely_harmonic_operator_func_factory`,
      87 lines.
    - `circuit_routines.py:1894` — `get_operator_by_name`, 75 lines.

## Architectural diagram

### Current

```
    CircuitRoutines (70 methods, 2348 lines)
    CircuitSymMethods (35 methods, 1429 lines)
    CircuitPlot (970 lines)
              │
              ▼
        CircuitABC  ── empty pass-through
              │
       ┌──────┴──────┐
       ▼             ▼
   Subsystem      Circuit
   (CircuitABC,    (CircuitABC,
    QubitBaseClass, QubitBaseClass,
    Serializable,   Serializable,
    DispatchClient, DispatchClient,
    NoisyCircuit)   NoisyCircuit)
       ▲             ▲
       └─── recursion (Subsystem owns Subsystem children) ───┘
```

`Subsystem` and `Circuit` repeat the same five-base inheritance list. The
empty `CircuitABC` could absorb the shared bases (deferred to Batch B).

### Target post-Batch A

The shape of the diagram is unchanged. The deltas are internal:

```
branch_metadata.py (no scqubits.core deps)
    ▲   ▲   ▲   ▲
    │   │   │   │
    │   │   │   └── tests/test_circuit_utils.py
    │   │   └────── symbolic_circuit_graph.py
    │   └────────── symbolic_circuit.py
    │   └────────── circuit_input.py
    └────────────── circuit_utils.py    (still imports circuit_input
                                         for assemble_circuit; that
                                         direction stays)
```

- New module `branch_metadata.py` converts the
  `circuit_input ↔ circuit_utils` cycle into a one-way edge
  (`circuit_utils → circuit_input` survives — `assemble_circuit` still
  calls `remove_comments` etc.) and lets the lazy import in
  `circuit_input.find_jj_order` go away.
- `_make_property` body shrinks from 73 → ~30 lines via context manager +
  lookup table.
- `_set_property_and_update_*` trio shares one extracted helper.
- `from_yaml` and `_configure` share `_import_from_symbolic_circuit` for the
  18-attribute copy.
- `configure`'s bare `except:` becomes typed and re-raises with
  `raise ... from e`.

## Risk-ranked refactor list

| ID | Refactor | Files touched | Risk | Clean-Code principle | Δ lines | Depends on |
|---|---|---|---|---|---|---|
| A1 | Replace bare `except:` and the generic `raise Exception(...)` in `Circuit.configure` with a typed `ConfigureError(...) from e` and narrow the catch to `Exception` | `circuit.py` | low | Error handling (Ch. 7) | +6 / -2 | – |
| A2 | Extract `_dispatch_suspended()` `@contextmanager` and replace 3 inline copies in `_make_property` | `circuit_routines.py` | low | DRY | +8 / -15 | – |
| A3 | Replace 3-branch `if/elif` in `_make_property` with `{property_update_type: setter_method}` lookup using A2 | `circuit_routines.py` | low | Replace conditional with table; SRP | +5 / -28 | A2 |
| A4 | Extract `_propagate_param_to_affected_subsystems` helper used by all three `_set_property_and_update_*` | `circuit_routines.py` | low | DRY | +8 / -15 | – |
| A5 | Extract `_import_from_symbolic_circuit` covering the 18-attribute copy; call from `from_yaml` and `_configure` | `circuit.py` | low | DRY | +8 / -38 | – |
| A6 | Create `branch_metadata.py`; move `_junction_order` and `_capacitance_variable_for_branch`; update all callers; remove the lazy import in `circuit_input.find_jj_order` | `branch_metadata.py` (new), `circuit_input.py`, `circuit_utils.py`, `symbolic_circuit.py`, `symbolic_circuit_graph.py`, `tests/test_circuit_utils.py` | low | Dependency direction | +35 / -35 | – |
| A7 | Add `_diag_from_function(grid, values_fn)` helper; rewrite `_phi_operator`, `_cos_phi`, `_sin_phi` to use it (`_identity_phi` excluded — uses `sparse.identity` directly) | `circuit_utils.py` | low | DRY | +5 / -16 | – |
| A8 | **No-op / narrowed.** Originally framed as "trim multi-paragraph docstrings to one-liners." On inspection no methods touched in A1–A7 carry genuine work-narrative docstrings — the only candidates were three duplicated factory-closure docstrings documenting the `energy_esys` parameter. Those are real parameter spec, not narration; gutting would be a regression. | – | n/a | – | 0 | A1–A7 |

**Estimated cumulative Δ for Batch A: ≈ −105 lines.** Actual outcome:
**−83 lines** measured on production source code only
(`scqubits/core/{circuit,circuit_input,circuit_routines,circuit_utils,symbolic_circuit,symbolic_circuit_graph}.py`),
satisfying the "net negative" requirement.

## Sequencing

### Batch A — executed

A1 → A2 → A3 → A4 → A5 → A6 → A7 (→ A8 no-op).

Each is its own commit. Plus two follow-up commits:
(i) drop leading `_` from upper-case constants `PROPERTY_SETTER_BY_TYPE`
and `SYMBOLIC_CIRCUIT_ATTRIBUTES`,
(ii) drop a now-unused `# type: ignore[return-value]` in
`_diag_from_function` after A7 made it redundant.

### Batch B — plan only, do not execute

- **B1.** Split `circuit_utils.py` along the 11 responsibility groups above.
  New modules: `discretized_phi_operators.py`, `charge_basis_operators.py`,
  `circuit_yaml_assembly.py`, `sympy_helpers.py`, `operator_factories.py`,
  `sawtooth.py`, `dense_helpers.py` (plus `branch_types.py` from A6).
  Orphans (`example_circuit`, `truncation_template`,
  `keep_terms_for_subsystem`) move to wherever their primary consumer
  lives.
- **B2.** Add named constructors `Circuit.from_yaml_file(path, ...)` and
  `Circuit.from_yaml_string(text, ...)`. Mark the `from_file: bool` form
  deprecated via `DeprecationWarning`; keep working until the next minor.
- **B3.** Polymorphic basis extraction: replace the long if-elif in
  `_generate_operator_methods` with small `BasisOperatorFactory` classes —
  `DiscretizedBasis`, `HarmonicBasis`, `PurelyHarmonicBasis`,
  `HierarchicalBasis`, each implementing
  `build(self, var_index, sym_variable) -> Callable`.
- **B4.** Inheritance consolidation: move
  `(QubitBaseClass, Serializable, DispatchClient, NoisyCircuit)` into
  `CircuitABC` so `Subsystem` and `Circuit` only inherit from it. Verify
  MRO equivalence (`Circuit.__mro__`) before/after.
- **B5.** Replace `eval(H_LC_str, replacement_dict)` in
  `_hamiltonian_for_harmonic_extended_vars` with a `sympy.lambdify`-based
  dispatcher.

### Batch C — plan only, do not execute

- **C1.** Transactional `configure()` via context manager
  (`with self._configure_transaction():`) that snapshots, runs, and on
  `Exception` restores **and re-raises** with `raise ... from e`. High
  risk because rollback semantics are subtle; current
  silent-succeed-on-bad-config is observable behavior some callers may
  rely on.
- **C2.** Composite-pattern split of `Circuit`/`Subsystem`: hoist
  recursive-children iteration out of every site that does
  `for subsys in self.subsystems:` into a `walk_subsystems()` helper and
  a small `Composite` ABC.
- **C3.** Noise-method consolidation: dynamically-generated per-channel /
  per-branch methods in `circuit_noise.py` should be replaced with
  explicit `_noise_method_for(channel, branch)` dispatched through a
  registry.

### Batch C additions surfaced by radon / CodeFactor (added during Batch A wrap-up)

Quality checkers (CodeFactor.io connected to the repo, plus local
`radon cc`) flagged hotspots not on the original list:

- **C4.** `symbolic_circuit_graph.SymbolicCircuitGraph._spanning_tree` —
  cyclomatic complexity 58 (radon F). Highest-complexity function in
  the entire circuit module. Decompose into smaller predicates and a
  driver loop.
- **C5.** `symbolic_circuit_graph.SymbolicCircuitGraph.variable_transformation_matrix`
  — complexity 53 (F). Sibling smell to C4; same playbook.
- **C6.** `symbolic_circuit_graph.SymbolicCircuitGraph._independent_modes`
  — complexity 36 (E).
- **C7.** `symbolic_circuit_graph.SymbolicCircuitGraph.check_transformation_matrix`
  — complexity 33 (E).
- **C8.** `circuit.Circuit._configure_sym_hamiltonian` — complexity 32
  (E). Sibling of `_configure` (39); the original plan flagged
  `_configure` but missed this parallel one. Same DRY treatment as A5
  / future B-style decomposition.
- **C9.** `circuit_utils.assemble_circuit` — complexity 28 (D). Will
  be touched anyway by B1's split of `circuit_utils.py`; flag here
  for explicit decomposition.

CodeFactor.io overall: grade A, 32 open issues across 93 files (95.7%
A, 4.3% B). Flagged issue types: *Complexity per Method*, *Lines of
Code per Method*, *Lines of code that are similar to other code*
(duplication). No surprises beyond what radon shows; the items above
constitute the actionable subset.

## Test-coverage assessment

Run before starting Batch A:

```
pytest -v --pyargs scqubits.tests.test_circuit \
    scqubits.tests.test_circuit_utils \
    scqubits.tests.test_circuit_sym_methods \
    --cov=scqubits.core --cov-report=term-missing
```

Expectations from PR #285 / #286: `circuit_utils.py` is well covered by
`test_circuit_utils.py` (≈70%+ of pure helpers). `circuit.py:configure /
_configure` and `circuit_routines.py:_make_property` are likely under 60%.

Per-refactor characterization tests (added **before** the corresponding
refactor when below ~70%):

- **A1** — `test_circuit.py`: configure with a deliberately invalid
  `transformation_matrix`, assert that the new typed exception
  (`ConfigureError`) carries the original error as `.__cause__`. Today
  `Exception` is raised with no chain, so this test pins the post-A1
  behavior (more informative on failure; same pattern-matching surface
  for callers that use `except Exception`).
- **A2/A3** — Unit test that calls a `_make_property`-created setter while
  `settings.DISPATCH_ENABLED = True`, asserts the central-dispatch flag is
  restored after the setter returns. Pins observable behavior independent
  of internal refactor.
- **A4** — Test that mutates a parameter on a parent circuit with two
  extended-variable subsystems and asserts both are propagated.
- **A5** — Existing `test_circuit.py` round-trips already exercise this;
  verify in full pytest run.
- **A6** — Subprocess test that imports `circuit_input` and
  `circuit_utils` in both orders in a fresh interpreter — ensures cycle
  is gone.
- **A7** — Existing `test_circuit_utils.py` tests for the four
  discretized-phi operators cover this; ensure they pass unchanged.

Coverage gate: any Batch-A change reducing per-file coverage in the
touched module is a fail. Add a characterization test before refactoring
rather than dropping coverage.

## Definition of done — Batch A

1. Branch `refactor/circuit-module-clean-code` exists off `main` (and
   `REFACTORING_PLAN.md` is committed there as the first commit).
2. One commit per refactor (A1–A8), in order, with the required message
   format.
3. After every commit:
   - `isort -w88 --lbt 1 --profile black .` then `black .` produce **no
     diff**.
   - Fast subset
     `pytest -v --pyargs scqubits.tests.test_circuit
     scqubits.tests.test_circuit_utils scqubits.tests.test_circuit_sym_methods`
     passes.
4. The full `pytest -v --pyargs scqubits --num_cpus=2` passes on the
   final commit.
5. `mypy scqubits/` reports zero new errors (CI gate from
   `.github/workflows/mypy.yml`).
6. `git diff --shortstat main...refactor/circuit-module-clean-code` shows
   lines deleted > lines added.
7. Final commit body lists, by ID, the refactors completed and any
   deferred items.

## Critical files

- `scqubits/core/circuit.py` — `__init__` (418), `from_yaml` (541),
  `configure` (718), `_configure` (1065).
- `scqubits/core/circuit_routines.py` — `_set_property_and_update_*`
  (648/687/718), `_make_property` (742),
  `_generate_operator_methods` (1692; B3 only).
- `scqubits/core/circuit_input.py` — `find_jj_order` (121–145; A6).
- `scqubits/core/circuit_utils.py` — `_junction_order` (39),
  `_capacitance_variable_for_branch` (114), discretized-phi operators
  (185–288), `round_symbolic_expr` (689), and the rest of the junk
  drawer (B1).
- `scqubits/core/symbolic_circuit.py` — `configure` (320) (Batch B/C
  only).

## Existing utilities to reuse

- `contextlib.contextmanager` from stdlib (no new dep) for A2's
  `_dispatch_suspended()`.
- `scqubits.core.descriptors.WatchedProperty` — already used by
  `_make_property` for the central-dispatch wrapper.
- `scqubits.utils.misc.flatten_list_recursive` /
  `unique_elements_in_list` — already in use across `circuit_utils.py`,
  reuse rather than re-implement.

## Verification (end-to-end)

```bash
git checkout main && git pull
git checkout -b refactor/circuit-module-clean-code
git add REFACTORING_PLAN.md
git commit -m "docs: add circuit-module refactoring plan"
# stop here — wait for user go-ahead before executing Batch A
```

Then for each Batch-A refactor:

```bash
# 1. apply change
isort -w88 --lbt 1 --profile black .
black .
pytest -v --pyargs scqubits.tests.test_circuit \
    scqubits.tests.test_circuit_utils \
    scqubits.tests.test_circuit_sym_methods
git add <touched files>
git commit -m "refactor(circuit): <description> [Clean Code: <principle>]"
```

After A8, run the full suite and `mypy scqubits/`, then push.

## Backlog (out of scope for this session)

- Batches B (B1–B5) and C (C1–C3) per the sequencing above.
- Pre-commit hooks for `isort`/`black`/`mypy`.
- Notebook-based tests via `nbval`.
