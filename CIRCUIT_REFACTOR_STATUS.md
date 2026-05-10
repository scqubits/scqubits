# Circuit subsystem -- refactor status assessment

This document is for a reviewer assessing whether the refactoring work
on the `circuit` subsystem of scqubits is complete. It is **not** a
historical narrative; it is meant to give a fresh reviewer enough
context to form an independent judgement about whether further
refactoring is justified.

## What you are deciding

Whether the circuit subsystem (`scqubits/core/circuit.py`,
`scqubits/core/symbolic_circuit.py`,
`scqubits/core/symbolic_circuit_graph.py`, and everything under
`scqubits/core/circuit_internals/`) can ship in its current shape, or
whether you see one or more concrete code-quality concerns that warrant
another refactoring round.

The answer is allowed to be: yes, ship; yes, ship after item X is
fixed; no, here's a non-trivial structural problem still to address.
What I want from you is the third option only if it's real.

## Branch state

- Working branch: `refactor/symbolic-circuit-graph-architecture`
- Origin: `https://github.com/scqubits/scqubits.git`
- Commits ahead of `origin/main`: 107
- Local push status: synced with `origin/<branch>`

Earlier commits in the branch (~80 of the 107) are pre-existing
type-cleanup / earlier-refactor work that was already validated; the
most recent ~25 commits are the work this assessment focuses on.

## Module shape now

```
scqubits/core/
├── circuit.py                      Circuit, Subsystem, ConfigureError (top-level public surface)
├── symbolic_circuit.py             SymbolicCircuit (stage 1: YAML -> sympy Hamiltonian)
├── symbolic_circuit_graph.py       SymbolicCircuitGraph + Node/Branch/Coupler (graph layer)
└── circuit_internals/
    ├── _protocols.py               CircuitProtocol -- single source of truth for cross-mixin attrs
    ├── routines.py                 CircuitRoutines (Hilbert-space basics, serialization)
    ├── lifecycle.py                LifecycleMixin (param sync, dispatch, update pipeline)
    ├── subsystem_tree.py           SubsystemTreeMixin (HD subsystem construction)
    ├── hamiltonian_assembly.py     HamiltonianAssemblyMixin (operators, Hamiltonian assembly)
    ├── junction_assembly.py        Josephson cos/sin -> matrix evaluation pipeline
    ├── sym_methods.py              CircuitSymMethods (LaTeX, sym transforms)
    ├── plotting.py                 CircuitPlot (wave-function, potential plots)
    ├── noise.py                    NoisyCircuit (T1, T2 channels)
    ├── input.py                    YAML grammar + parsing constants
    ├── yaml_assembly.py            assemble_circuit, assemble_transformation_matrix
    ├── branch_metadata.py          per-branch-type metadata
    ├── operator_factories.py       make_*_method factories for dynamic operator binding
    ├── charge_basis_operators.py   bare per-basis primitives (charge basis)
    ├── discretized_phi_operators.py bare per-basis primitives (discretized phi)
    ├── matrix_helpers.py           cos/sin of diagonal sparse, matrix_power_sparse
    ├── sawtooth.py                 sawtooth-junction model
    ├── sympy_helpers.py            round_symbolic_expr, is_potential_term
    └── utils.py                    truncation_template, get_trailing_number
```

Total circuit-subsystem lines: ~14 300 LOC.

Authoritative reference for module internals:
**`CIRCUIT_DEVELOPER_MANUAL.md`** (1478 lines) at the repo root. It is
the document a maintainer should read before touching this code, and
it is up-to-date with every change in this branch (each commit that
changed code also updated the manual where the code is documented).
The manual was given a comprehensive reviewer-agent audit during this
branch and has had its findings addressed; it is not a draft.

## What changed in this round

In rough phase order. Commit hashes are in `git log
origin/main..HEAD`; `<hash>` references in this document refer to
those commits.

### Phase 0 -- footgun cleanup (5 commits)

Closed five small but real silent-failure paths.

- `5ba27628` `fix(circuit): reject transformation_matrix on symbolic-Hamiltonian path`
  Was silently ignored on `Circuit(symbolic_hamiltonian=...)`; now
  raises `ValueError`. The error message in the same branch already
  named `transformation_matrix` as forbidden -- the validation
  condition just hadn't been updated. **Behaviour change**: a path
  that was a silent no-op now raises. Existing test
  `test_circuit_with_symbolic_hamiltonian` was passing
  `transformation_matrix=` as a no-op; that line was removed
  (eigenvalues asserted afterward unchanged, since the call was
  already silently ignored).

- `17343c41` `refactor(circuit): replace eval() with getattr() in get_osc_param`
  One-line: a string-concat-and-eval was rewritten as `getattr`.

- `cda937c1` `refactor(circuit): replace _clear_unnecessary_attribs string-pattern matching with a registry`
  Was using `"cutoff_n_" in attrib`, `attrib[1:3] == "ng"`, etc. to
  decide what to drop. Now uses `self._dynamic_var_attribs`
  populated by `_install_var_properties`. Plus a regression test that
  registers a name not matching any of the legacy substring
  patterns and asserts it gets cleared.

- `7c75b139` `refactor(circuit): retire vestigial Node.marker`
  Audit confirmed `_mark_nodes_by_subgraph` was dead code (zero
  call sites) and `Node.marker` had no readers outside one test.
  Field, writer, and test all deleted.

- `5ded4d84` `fix(circuit): narrow potential_φ type via isinstance for mypy`
  Followed from `_JJs_terms` getting a return annotation of
  `sm.Expr | int` -- exposed a latent `union-attr` mypy error
  guarded by a runtime `!= 0` check that mypy couldn't narrow.
  Fixed with `isinstance` guard.

### Phase 1 -- safety net (2 commits)

Strengthened the test surface so the bigger Phase 2/3 refactors had
something to check against.

- `9096feaf` `refactor(circuit): pin recomputation contract via SymbolicCircuit._STAGE2_ATTRIBUTES`
  The 18-attribute "what flows from `SymbolicCircuit` to `Circuit`"
  list moved from the consumer side (`Circuit`) to the producer side
  (`SymbolicCircuit._STAGE2_ATTRIBUTES`). Plus
  `TestRecomputationContract` (6 tests) asserting every name in the
  tuple is set on a fresh `Circuit`, and that topology-level
  attributes (branches, nodes, var_categories, ...) survive
  `_configure` reference-identical.

- `e21684e0` `test(circuit): expand characterization fixtures 4 -> 7 + JSON sidecars`
  Added `ungrounded` (sigma-mode coverage), `ml_coupled` (mutual
  inductance), `jj2` (higher-order JJ). Plus JSON sidecars next to
  every `.npy` golden so numerical-change PRs become reviewable
  (sidecar contains first-6 evals, trace, Frobenius norm, etc.; not
  compared, just informational).

### Phase 2 -- static-tooling cleanup (4 commits)

- `10e73ca5` `refactor(circuit): consolidate cross-mixin TYPE_CHECKING blocks into CircuitProtocol`
  Five mixins each had a parallel `if TYPE_CHECKING:` block declaring
  cross-mixin attributes. They had drifted (same attribute typed
  differently across files). Replaced with a single `CircuitProtocol`
  class inherited by all five mixins; the class body is gated under
  `TYPE_CHECKING`, so runtime is a no-op. -295 / +205 LOC. Runtime
  MRO grows by one class (`CircuitProtocol`) appearing once via
  Python deduplication, not five times -- verified at runtime.

- `893f1fa0` `feat(circuit): typed operator() dispatcher for the dynamic <name>_operator methods`
  Per-variable operators are bound at `_configure` time via
  `types.MethodType` and are invisible to mypy / IDE / sphinx-autodoc.
  Added `Circuit.operator(name, *, energy_esys=False)` as a typed
  accessor that resolves to the dynamic method. The dynamic methods
  stay for back-compat. Three new tests pin behavioural equivalence
  vs. the dynamic path.

- `7fa3901f` `feat(circuit): explicit noise-channel registry via NoisyCircuit.channels()`
  Same brittle-name-pattern problem as `_clear_unnecessary_attribs`.
  Was: `Circuit.supported_noise_channels()` walked `self.__dict__`
  and substring-matched `"t1_" / "tphi_1_over_f"`. Now: each
  `_generate_*` helper routes its `setattr` through
  `_register_noise_method`, which also populates
  `self._noise_channels_registry`. New public `channels() -> dict[str,
  Callable]`. Three new tests including a non-tautological parity
  check vs. the legacy `__dict__`-walk algorithm.

- `fccc17a2` `fix(circuit): address Phase 2 reviewer findings`
  Reviewer audit caught: the sed pass for the registry refactor
  missed a multi-line `setattr` block in
  `_generate_t1_flux_bias_line_methods`, dropping
  `t1_flux_bias_line<n>` methods from `supported_noise_channels()`
  for circuits with closure branches. Fixed plus replaced the
  tautological initial parity test with one that actually compares
  to the legacy implementation. Plus `operator()` error-message
  suffix strip and HD-parent-limitation docstring.

### Phase 3 -- architectural refactors (2 commits)

- `9fa55bc2` `refactor(circuit): split JJ pipeline into circuit_internals/junction_assembly.py`
  The Josephson cos/sin -> matrix-evaluation pipeline (8 helpers)
  lifted out of `HamiltonianAssemblyMixin` into a sibling module.
  The mixin keeps two thin wrapper methods
  (`_evaluate_matrix_cosine_terms`, `_evaluate_matrix_sawtooth_terms`)
  because external callers in `CircuitSymMethods` invoke them via
  `self.<method>`. `hamiltonian_assembly.py` shrank from 1392 ->
  1230 lines.

- `b39a0a09` `docs(circuit): fix two stale references missed by Phase 3.1's doc sweep`
  Reviewer audit on `9fa55bc2` caught two stale name references
  (`_build_junction_phase_operator_list`, `_extract_junction_phase`)
  in the manual / module docstring after the helpers had moved.

**Phase 3.2 was deliberately skipped.** The original plan called for
unifying `Circuit._configure` and `Subsystem._configure` via a
template-method, on the framing that they were "two parallel ~50-line
functions with subtle divergences." A design-pass review (with the
Plan agent doing line-by-line decomposition) found the truly-shared
body is **four lines** (`_frozen` toggles + `_set_vars` +
`_set_operators`); the HD-vs-non-HD branches look similar but have
load-bearing content differences (one calls
`_generate_hamiltonian_sym_for_numerics`, the other just `.copy()`s,
because subsystem state was already prepared by the parent; one warns
on `ext_basis="harmonic"` mismatch, the other doesn't; etc.).
Forcing unification would have required either two near-identical
helpers or a flag-soup `_configure_common`. Documented in the manual
at §6.2 so a future contributor doesn't re-attempt. **This is a
load-bearing decision -- if the reviewer disagrees, that's a
substantive challenge to the current state.**

### Tooling

- `aaa04c19` `tools(docstring-lint): add docstring-compliance linter + GitHub workflow`
- `78dd8f2b` `tools(docstring-lint): fix reviewer findings, generalize over-narrow rules`

The repo now has `tools/docstring_lint.py` (single-file, stdlib-only)
that flags placeholder phrases (DOC001), types duplicated into
docstring sections when the signature has the annotation (DOC002,
context-aware), refactor-history phrasing (DOC003), and empty
numpydoc section headers (DOC004). For private (`_`-prefixed)
symbols all issues are warnings, not errors. There is a GitHub
Actions workflow at `.github/workflows/docstring-lint.yml` that runs
the linter in `--compare-to origin/main` mode on PRs touching
`scqubits/**/*.py`, so only docstring issues newly introduced by the
PR can fail the build. 65 unit + integration tests cover every check
plus the compare-to git interaction.

After landing the linter, all 8 originally-flagged DOC002 errors
across `scqubits/` were fixed (`7243883d`). Current baseline at HEAD
is **0 errors, 0 warnings**.

## Quality metrics at HEAD

- Full pytest suite: **356 passed, 14 skipped** (was 338 passed at
  branch start; +18 from new tests added during this work).
- Characterization-test fixtures: 7 (was 4); each pins
  ``Circuit.hamiltonian()`` and ``Circuit.eigenvals(6)`` at
  `rtol=1e-10` against committed `.npy` goldens.
- mypy: clean on every modified file. `mypy scqubits/` succeeds.
- docstring-lint: clean (`python tools/docstring_lint.py scqubits/`
  reports "no new docstring issues").
- Developer's manual: 1478 lines, factually audited by an external
  reviewer agent during this branch, all flagged errors fixed.
- Memory rules established during this session and applicable to
  future work are saved in
  `~/.claude/projects/.../memory/MEMORY.md` (out of repo).

## Things deliberately not done

The reviewer should be aware of these so they don't flag absent items
as oversights:

1. **No `Circuit._configure` / `Subsystem._configure` unification.**
   Reasoning above. Manual §6.2 documents the decision.

2. **No replacement of dill-based serialization.** A previous
   recommendation to introduce JSON-with-schema was retracted: dill
   is required for scqubits multiprocessing on Windows (spawn-based
   process creation cannot pickle bound methods / closures /
   sympy / qutip objects without dill). Manual is silent on this --
   if the reviewer wants the constraint documented, that's a useful
   addition.

3. **No general perf audit beyond the circuit module.** The
   pre-existing perf-fix series in this branch (commits
   `45d0f8eb..85a6ba54`, `dda314e3`, `75cef2ff` -- before the
   refactor work began) cut `Circuit(n=12)` build from ~55 s to
   ~1.4 s. That work is not under review here; it was already
   landed before the assessment scope started.

4. **No `_evaluate_matrix_sawtooth_terms` algorithmic-bug fix.**
   The previous reviewer noted a pre-existing line-460 bug in that
   helper (per-iteration coefficient computed from
   `saw_expr.as_coefficients_dict()` rather than `saw_term`).
   Carried over verbatim into `junction_assembly.py`. Tracked
   informally; not addressed in this branch because it predates the
   refactor work and warrants its own focused investigation.

5. **No deepening of `WatchedProperty` package-level documentation.**
   `circuit_internals/lifecycle.py` and §9 of the manual cover the
   circuit-side flow well; the package-wide `WatchedProperty`
   contract (which other qubit classes also use) is not covered. Out
   of circuit-subsystem scope.

## Specific spots a reviewer should check

To form a defensible "ship" or "don't ship" verdict:

1. **The 4-lines-shared decision (Phase 3.2 skip).** Read
   `Circuit._configure` (`scqubits/core/circuit.py:1247`) and
   `Subsystem._configure` (`scqubits/core/circuit.py:305`)
   side-by-side. Ask: would template-method / shared-base extraction
   buy more clarity than it costs? If yes, this is the strongest
   challenge to the current state.

2. **CircuitProtocol completeness.** Scan
   `circuit_internals/_protocols.py`. Ask: are there cross-mixin
   attributes that mixins reference via `self.<x>` but `CircuitProtocol`
   doesn't declare? Two were added during the session (`is_grounded`,
   `exp_i_operator`) only when mypy errors surfaced their absence.
   A grep audit may find more.

3. **The reverted type tightening.** `CircuitProtocol` declares
   `parent: Any`, `subsystems: list[Any]`, `hilbert_space: Any` --
   the comment explains that tightening surfaces ~9 latent typing
   issues at downstream call sites. Ask: is fixing those issues a
   one-day task that should be done now, or a separate workstream?

4. **Test coverage of the typed `operator()` dispatcher.** The three
   tests in `TestTypedOperatorAccessor` cover transmon (non-HD).
   They do not cover an HD parent (which has empty `operators_by_name`
   and routes to subsystem operators). Ask: is the documented HD
   limitation acceptable, or should the dispatcher route via
   `return_root_child(var_index)` like `get_operator_by_name` does?

5. **Noise-channel registry coverage.** Run
   `python tools/docstring_lint.py scqubits/` and
   `python -m pytest scqubits/tests/test_circuit.py::TestNoiseChannelsRegistry`.
   The parity test asserts the new registry-based
   `supported_noise_channels` matches what the legacy
   `__dict__`-walk algorithm would have returned. The reviewer should
   verify that test isn't tautological (the original was;
   `fccc17a2` fixed it).

6. **Characterization-fixture honesty.** The 7 `.npy` goldens were
   regenerated when the fixtures were added. Verify the `.json`
   sidecars next to each `.npy` look numerically reasonable for the
   circuit topology (e.g., a transmon's first eigenvalue should be
   around the right ballpark for the YAML's `EJ`/`EC`).

7. **The deferred sawtooth coefficient bug.** Read
   `circuit_internals/junction_assembly.py:230` (the
   `_check_returns`-equivalent inside
   `evaluate_matrix_sawtooth_terms`). Ask: is the
   carried-over-verbatim status acceptable, or is this a real
   regression risk?

## What "ship" means here

The branch is meant to land in `main` as one PR (or one squash). The
review question is not "does each commit perfectly stand alone" but
"is the end state of the circuit subsystem in a place that
maintainers can safely build on?" Concrete operationalisations:

- A new contributor reading `CIRCUIT_DEVELOPER_MANUAL.md` should be
  able to find and modify any subsystem in the circuit module.
- Adding a new per-variable property kind should require updating
  exactly one file (`circuit.py`'s `_install_var_properties`); the
  registry takes care of clearing.
- Adding a new attribute on `SymbolicCircuit` that should flow to
  `Circuit` should require updating exactly one tuple
  (`SymbolicCircuit._STAGE2_ATTRIBUTES`); the regression test will
  flag a missing entry.
- Adding a new noise channel should require routing the `setattr`
  through `_register_noise_method`; the parity test will flag an
  unregistered channel.
- Adding a new lifecycle hook does require touching both
  `Circuit._configure` and `Subsystem._configure` -- this is the
  load-bearing exception, documented in §6.2.

If those four out of five "exactly one" promises plus the documented
exception are satisfied to the reviewer's eye, the branch is shippable.

## Pointers

- Branch commits (recent first): `git log --oneline origin/main..HEAD`
- Test gate: `pytest --pyargs scqubits` (~4 min, 356 passed)
- Characterization gate: `pytest scqubits/tests/test_circuit_characterization.py` (~5 s, 19 passed)
- Type gate: `mypy scqubits/` (clean)
- Lint gate: `python tools/docstring_lint.py scqubits/` (clean)
- Developer's manual: `CIRCUIT_DEVELOPER_MANUAL.md`
- Linter docs: `tools/README.md`
- Linter tests: `python -m pytest tools/test_docstring_lint.py` (65 passed)
