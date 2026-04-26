# Architecture of the `circuit` module

This document describes the file and class structure surrounding
`scqubits/core/circuit.py`. It is aimed at someone who has never opened the
module before and wants to understand how a YAML circuit description becomes
a numerical Hamiltonian, an eigenspectrum, and a plot.

The implementation is the post-2022 hierarchical-circuit machinery introduced
in Chitta et al., *New J. Phys.* **24**, 103020 (2022). All symbol names
and the basic algorithm are taken from that paper.

---

## 1. Two-stage pipeline

Custom-circuit analysis runs as a two-stage pipeline. Each stage has a
top-level public class:

```
            ┌──────────────────────┐    ┌──────────────────────┐
   YAML ─►  │  SymbolicCircuit     │ ─► │  Circuit             │ ─► numerics
            │  (sympy expressions, │    │  (numerical operators,│
            │  graph topology)     │    │  Hilbert space)       │
            └──────────────────────┘    └──────────────────────┘
                  symbolic_circuit.py        circuit.py
```

- **`SymbolicCircuit`** (`scqubits/core/symbolic_circuit.py`) parses a YAML
  description into a graph of `Node`s and `Branch`es, builds the symbolic
  Lagrangian and Hamiltonian, finds a coordinate transformation that
  classifies each variable as periodic / extended / free / frozen / sigma,
  and computes the symbolic potential / kinetic / capacitance / inductance
  matrices.
- **`Circuit`** (`scqubits/core/circuit.py`) wraps a `SymbolicCircuit`,
  installs Python attributes for every parameter (`EJ`, `EC`, `Φ`, …) so the
  user can mutate them and trigger automatic re-evaluation via the central
  dispatch system, allocates a Hilbert space, and turns the symbolic
  Hamiltonian into numerical operators (qutip `Qobj` or scipy sparse) that
  can be diagonalized.

A user typically writes:

```python
import scqubits as scq
qubit = scq.Circuit.from_yaml_file("transmon.yaml")
qubit.EJ = 12.0           # mutating any param re-evaluates downstream state
energies = qubit.eigenvals()
```

`Circuit.from_yaml_file` builds the `SymbolicCircuit` and wraps it; the
user thereafter interacts with the wrapper.

---

## 2. File layout around `circuit.py`

```
scqubits/core/
├── circuit.py                    public:  Circuit, Subsystem, ConfigureError, CircuitABC
├── symbolic_circuit.py           public:  SymbolicCircuit
├── symbolic_circuit_graph.py     public:  Node, Branch, Coupler
│                                 internal: SymbolicCircuitGraph (ABC)
│
├── circuit_input.py              4-line back-compat shim → circuit_internals.input
├── circuit_utils.py              4-line back-compat shim → circuit_internals.utils
│
└── circuit_internals/            private umbrella; empty __init__.py
    ├── routines.py                CircuitRoutines mixin       (parameter sync,
    │                                                            Hamiltonian assembly,
    │                                                            evals/esys, …)
    ├── sym_methods.py             CircuitSymMethods mixin     (LaTeX rendering,
    │                                                            symbolic transforms)
    ├── plotting.py                CircuitPlot mixin           (wave-function plots,
    │                                                            potential plots)
    ├── noise.py                   NoisyCircuit mixin          (T1/T2 channels)
    ├── subsystem_tree.py          SubsystemTreeMixin          (HD subsystem
    │                                                            construction)
    │
    ├── input.py                   YAML parser + parsing constants
    ├── yaml_assembly.py           assemble_circuit, assemble_transformation_matrix
    ├── branch_metadata.py         _junction_order, _capacitance_variable_for_branch
    │                               (broken out to break a circular import)
    │
    ├── operator_factories.py      make_grid_operator_method,
    │                               make_basis_operator_method,
    │                               make_hierarchical_diag_method
    ├── charge_basis_operators.py  _cos_theta, _sin_theta, _n_theta_operator,
    │                               _exp_i_theta_operator, …
    ├── discretized_phi_operators.py _phi_operator, _cos_phi, _sin_phi,
    │                                 _i_d_dphi_operator, …
    ├── matrix_helpers.py          _cos_dia, _sin_dia, matrix_power_sparse, …
    ├── sawtooth.py                sawtooth_operator, sawtooth_potential
    │
    ├── sympy_helpers.py           round_symbolic_expr, is_potential_term, …
    └── utils.py                   truncation_template, get_trailing_number
                                    (+ four back-compat re-exports)
```

The two top-level files (`circuit.py`, `symbolic_circuit.py`,
`symbolic_circuit_graph.py`) hold every symbol that is part of the published
public API. Everything inside `circuit_internals/` is implementation detail;
external users should never need to import from it.

---

## 3. Class hierarchy

### 3.1 The symbolic side

```
        ABC
         │
SymbolicCircuitGraph         ← scqubits/core/symbolic_circuit_graph.py
         │
SymbolicCircuit              ← scqubits/core/symbolic_circuit.py
   (also inherits Serializable)
```

- **`Node`**, **`Branch`**, **`Coupler`** are plain data classes living
  alongside `SymbolicCircuitGraph`. `Node.__eq__` / `Branch.__eq__` are
  defined by `.index`, encoding the topology-equality contract that
  spanning-tree construction (which works on a `copy.deepcopy`) relies on.
- **`SymbolicCircuitGraph`** holds the graph-theoretic algorithms: spanning
  trees, closure branches, capacitance / inductance matrices, the
  variable-transformation matrix, and the resulting categorisation of
  variables as periodic / extended / free / frozen / sigma.
- **`SymbolicCircuit`** layers symbolic (sympy) Lagrangian and Hamiltonian
  construction on top of the graph machinery.

### 3.2 The numerical side

```
                                        ABC
                                         │
                              SubsystemTreeMixin             ← circuit_internals/subsystem_tree.py
                                         │
                              CircuitRoutines                ← circuit_internals/routines.py
                              CircuitSymMethods              ← circuit_internals/sym_methods.py
                              CircuitPlot                    ← circuit_internals/plotting.py
                                         │
                                       CircuitABC            ← circuit.py (aggregator only)
                                         │
                            ┌────────────┴─────────────┐
                            │                          │
                       Subsystem                    Circuit          ← circuit.py
                            │                          │
       (each also inherits  │                          │
        QubitBaseClass,     │                          │
        Serializable,       │                          │
        DispatchClient,     │                          │
        NoisyCircuit)
```

- **`SubsystemTreeMixin`** holds the seven-method cluster that decomposes a
  parent's symbolic Hamiltonian into per-subsystem terms, instantiates the
  child `Subsystem` objects, and rebuilds the resulting `HilbertSpace`
  interactions.
- **`CircuitRoutines`** is the bulk of the runtime machinery: parameter
  syncing across the subsystem tree, dispatch wiring, operator construction
  (per-variable `<name>_operator` methods built dynamically and bound via
  `types.MethodType`), Hamiltonian evaluation, and eigenvalue calculation.
  It inherits from `SubsystemTreeMixin`.
- **`CircuitSymMethods`** holds symbolic-side operations the user might want
  to inspect: the symbolic Lagrangian, Hamiltonian, potential, interactions,
  and the LaTeX-rendering helpers that drive `_repr_latex_`.
- **`CircuitPlot`** holds the plotting code (wave functions, potentials).
- **`NoisyCircuit`** is the noise-channels mixin (T1, T2, dielectric loss,
  flux noise, …); it inherits from `noise.NoisySystem` in `scqubits.core.noise`.
- **`CircuitABC`** is a thin aggregator: `class CircuitABC(CircuitRoutines,
  CircuitSymMethods, CircuitPlot)`. It exists only so the two concrete
  classes (`Subsystem`, `Circuit`) have a single named base.

The two concrete classes both inherit:

```python
class Subsystem(CircuitABC, base.QubitBaseClass, Serializable, DispatchClient, NoisyCircuit):
class Circuit  (CircuitABC, base.QubitBaseClass, Serializable, DispatchClient, NoisyCircuit):
```

The MRO order matters because all three of `CircuitRoutines`,
`CircuitSymMethods`, `CircuitPlot`, `NoisyCircuit` are mixin-style classes
that read `self.<various attributes>` set by `__init__`. The order is
fixed; do not reorder without checking `Circuit.__mro__`.

### 3.3 Why so many mixins?

Each mixin owns one concern. Splitting them lets the implementation for one
concern grow (or be rewritten) without disturbing the others. The split
also makes the cross-mixin dependency graph explicit: each mixin declares
the attributes it expects from siblings under `if TYPE_CHECKING:`, so mypy
can verify the contract.

---

## 4. End-to-end flow

### 4.1 Construction

```
Circuit.from_yaml_file(path)
   ├─ reads file, calls Circuit(input_string=text, from_file=False)
   ├─ Circuit.__init__:
   │     ├─ Circuit.from_yaml(text)  ─►  builds SymbolicCircuit  ─► self.symbolic_circuit
   │     │     └─ SymbolicCircuit.__init__:
   │     │            ├─ parse YAML into Node / Branch / Coupler lists
   │     │            ├─ build the graph (spanning tree, closure branches)
   │     │            ├─ find the variable-transformation matrix
   │     │            ├─ classify each variable: var_categories = {periodic, extended, …}
   │     │            └─ build hamiltonian_symbolic, potential_symbolic, …
   │     ├─ self._import_from_symbolic_circuit()
   │     │     └─ copies the 18 attributes that need to live on the Circuit instance
   │     │        (hamiltonian_symbolic, var_categories, external_fluxes, …)
   │     └─ self._configure(...)   ◄── the heavy lifting (next section)
   └─ returns Circuit instance
```

### 4.2 `_configure` (in `circuit.py`)

`_configure` is the single point where every observable piece of the
`Circuit` is wired up. Roughly 15 sub-phases run in sequence:

1. Re-build `symbolic_circuit` if a transformation matrix or closure-branch
   override was passed.
2. Copy 18 attributes from `symbolic_circuit` into `self` (delegated to the
   helper `_import_from_symbolic_circuit`; same helper is reused by
   `from_yaml`).
3. Install `WatchedProperty` descriptors on `self` for every cutoff,
   external flux, offset charge, free charge, and symbolic param so that
   mutating them notifies the central dispatch system (delegated to
   `_install_var_properties`).
4. Decide cutoff defaults (`DEFAULT_PERIODIC_CUTOFF = 5`,
   `DEFAULT_EXTENDED_CUTOFF = 30`).
5. If the system is purely harmonic, run
   `_diagonalize_purely_harmonic_hamiltonian` to decouple the oscillators.
6. If hierarchical diagonalization is requested, run
   `_generate_subsystems` (defined on `SubsystemTreeMixin`); this allocates
   child `Subsystem` instances and a `HilbertSpace`.
7. Either way, build the `<name>_operator` methods and bind them onto
   `self` via `types.MethodType` (delegated to `_set_operators`).

If anything in `configure` fails, the `try/except` block restores the
previous configuration and re-raises a `ConfigureError(...) from
original_exception`.

### 4.3 Numerical evaluation

After construction, the user can:

```python
qubit.hamiltonian()       # build the Hamiltonian as csc_matrix or qutip.Qobj
qubit.eigenvals(...)      # diagonalize
qubit.n_1_operator()      # any per-variable operator, dynamically created
qubit.plot_wavefunction(0)
```

Each numerical operator method is a closure built by one of three factories
in `circuit_internals/operator_factories.py`:

- `make_grid_operator_method` for discretized-phi variables;
- `make_basis_operator_method` for periodic (charge basis) and harmonic
  variables;
- `make_hierarchical_diag_method` for variables resolved via lookup against
  a subsystem's `get_operator_by_name`.

The factories produce a function of signature
`(self, energy_esys=False) -> Qobj | ndarray` and bind it as an instance
method.

The Hamiltonian is assembled in three flavours depending on the basis mix:

- `_hamiltonian_for_harmonic_extended_vars` for harmonic-basis extended vars;
- `_hamiltonian_for_purely_harmonic` for the all-extended-purely-harmonic
  case (closed-form);
- `_evaluate_hamiltonian` for the general case, which substitutes parameter
  values into `_hamiltonian_sym_for_numerics` and evaluates.

The Josephson-junction `cos(φ_J)` terms are handled separately by
`_evaluate_matrix_cosine_terms`, which uses
`_extract_junction_phase` / `_build_junction_phase_operator_list` to build
the per-variable `exp(i * a_k * φ_k)` factors and combines them into
`(O + O†)/2 = cos(O)` and `(O − O†)/(2i) = sin(O)` via `_assemble_cos_term`
/ `_assemble_sin_term`.

### 4.4 Hierarchical diagonalization (HD)

When `system_hierarchy` is non-empty, the parent `Circuit` becomes a tree
of `Subsystem`s. The tree is constructed by `SubsystemTreeMixin`:

```
Circuit (parent)
   ├─ subsystems = [Subsystem₁, Subsystem₂, …]
   │     each Subsystem has
   │       ├─ its own hamiltonian_symbolic = systems_sym[i]
   │       ├─ truncated_dim
   │       ├─ ext_basis (possibly different per subsystem)
   │       └─ subsystems = [...]   ← can recurse
   └─ hilbert_space = HilbertSpace(subsystems)
        with interactions reconstructed from subsystem_interactions[i]
```

`_generate_subsystems` decomposes `self.hamiltonian_symbolic` via
`_sym_subsystem_hamiltonian_and_interactions` (defined on
`CircuitSymMethods`), then either updates existing subsystems
(`_update_existing_subsystems`) or instantiates new ones
(`_create_new_subsystems`). `_update_interactions` rebuilds the
`HilbertSpace.interaction_list` using
`_operator_from_sym_expr_wrapper` to turn each symbolic interaction term
into an evaluable callable.

### 4.5 Parameter mutation and dispatch

Every cutoff / flux / offset-charge / free-charge / symbolic-param attribute
on a `Circuit` or `Subsystem` is a `WatchedProperty` whose setter routes
through `settings.DISPATCH_ENABLED`. The dispatch table is set up by
`_make_property` (in `routines.py`), which:

1. Looks up the right setter method via `_PROPERTY_SETTER_BY_TYPE` (a
   constant `dict[PropertyUpdateType, str]` mapping a `Literal` of update
   kinds to the bound-setter-method name).
2. Wraps the actual setter in a `_dispatch_suspended()` context manager,
   ensuring `settings.DISPATCH_ENABLED` is restored even on exception.

When a user does `qubit.EJ = 12.0`, the chain is:

```
qubit.EJ = 12.0
   → WatchedProperty.__set__
       → _set_property_and_update_param_vars("EJ", 12.0)
           → _propagate_param_to_affected_subsystems(...)
               → for each affected subsystem: subsys._sync_parameters_with_parent()
           → triggers central-dispatch event "QUANTUMSYSTEM_UPDATE"
   ─► next call to qubit.eigenvals() rebuilds the Hamiltonian
```

The `update()` method on `CircuitRoutines` is the entry point that picks up
queued events and refreshes whatever was invalidated.

---

## 5. Public API

The names a user is expected to import directly:

From `scqubits` (the top-level package):

- `Circuit` — the main user-facing class.
- `SymbolicCircuit` — exposed for users who want to build a symbolic circuit
  by hand and pass it in.
- `truncation_template` — helper for setting up hierarchical-diagonalization
  truncation dimensions.
- `assemble_circuit`, `assemble_transformation_matrix` — utilities for
  composing custom circuits programmatically.

From `scqubits.core.symbolic_circuit_graph`:

- `Node`, `Branch`, `Coupler` — used when constructing closure-branch lists
  for `Circuit.configure(closure_branches=...)`.

Everything else under `scqubits.core.circuit_internals.*` is internal.
The leading-underscore convention is enforced: any name beginning with `_`
in any module under `circuit_internals/` is implementation detail, may move
or change, and must not be imported by external code.

### Named constructors

`Circuit` exposes three construction paths:

```python
Circuit.from_yaml_file(path, ...)          # preferred for files
Circuit.from_yaml_string(yaml_text, ...)   # preferred for inline YAML
Circuit(input_string=..., from_file=...)   # legacy; emits DeprecationWarning
                                           # if from_file is passed explicitly
```

The legacy `from_file: bool` flag still works (sentinel-protected) but the
deprecation warning encourages migration to the named constructors.

### Error type

`Circuit.configure(...)` raises `ConfigureError` (a `RuntimeError` subclass
defined in `circuit.py`) if reconfiguration fails. The original cause is
attached via `raise ConfigureError(...) from exc`; the previous configuration
is restored before the error propagates.

---

## 6. Where to look for what

| If you want to change…                                | Edit…                                                           |
|-------------------------------------------------------|-----------------------------------------------------------------|
| The YAML grammar                                      | `circuit_internals/input.py`                                    |
| How a YAML file is composed programmatically          | `circuit_internals/yaml_assembly.py`                            |
| The graph algorithms (spanning tree, transformation)  | `symbolic_circuit_graph.py`                                     |
| Symbolic Lagrangian / Hamiltonian construction        | `symbolic_circuit.py`                                           |
| LaTeX rendering of symbolic state                     | `circuit_internals/sym_methods.py`                              |
| How parameters propagate when mutated                 | `circuit_internals/routines.py` — `_set_property_and_update_*`  |
| How operator methods are built and bound              | `circuit_internals/routines.py` — `_set_operators`, `_build_*`  |
| How JJ `cos(φ)` terms become matrices                 | `circuit_internals/routines.py` — `_evaluate_matrix_cosine_terms` |
| How HD subsystems are constructed                     | `circuit_internals/subsystem_tree.py`                           |
| How wave functions are plotted                        | `circuit_internals/plotting.py`                                 |
| How noise channels are wired                          | `circuit_internals/noise.py`                                    |
| The bare per-basis operators (charge / discretized-φ) | `circuit_internals/charge_basis_operators.py`<br>`circuit_internals/discretized_phi_operators.py` |
| The cosine/sine of a diagonal matrix                  | `circuit_internals/matrix_helpers.py`                           |
| The sawtooth-junction physics                         | `circuit_internals/sawtooth.py`                                 |
| The factory closures for operator methods             | `circuit_internals/operator_factories.py`                       |
| Circuit-input branch-type predicates                  | `circuit_internals/branch_metadata.py`                          |

For test coverage, the corresponding test files are under `scqubits/tests/`:

- `test_circuit.py` — end-to-end tests of `Circuit` and `Subsystem`.
- `test_circuit_utils.py` — unit tests for the bare helpers in
  `circuit_internals/{charge_basis_operators,discretized_phi_operators,
  matrix_helpers,sawtooth,sympy_helpers,utils,branch_metadata,input}.py`.
- `test_circuit_sym_methods.py` — focused tests of `CircuitSymMethods`.
