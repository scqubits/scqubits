# Developer's Manual: the `circuit` module

This document is for developers who need to **maintain, extend, or
debug** the custom-circuits subsystem of scqubits. It is not a user
tutorial — for that, see `scqubits-doc`'s `guide/circuit/`. It assumes
familiarity with superconducting-qubit physics at the level of the
Chitta et al. (2022) paper that the implementation is based on, and
working knowledge of Python, sympy, qutip, and `scipy.sparse`.

> **Reference paper.** Chitta, Zhao, Huang, Mondragon-Shem, Koch,
> *"Computer-aided quantization and numerical analysis of
> superconducting circuits,"*
> *New J. Phys.* **24**, 103020 (2022).
> Throughout, "the paper" refers to that work; equations and section
> numbers cited as e.g. "(Eq. 12)" or "(§4.2)" refer to it.

## Table of contents

1. [Scope and module map](#1-scope-and-module-map)
2. [The two-stage pipeline](#2-the-two-stage-pipeline)
3. [Class hierarchy and mixin composition](#3-class-hierarchy-and-mixin-composition)
4. [The symbolic circuit graph layer](#4-the-symbolic-circuit-graph-layer)
5. [The symbolic circuit layer](#5-the-symbolic-circuit-layer)
6. [The numerical layer (`Circuit`, `Subsystem`)](#6-the-numerical-layer-circuit-subsystem)
7. [Hierarchical diagonalization](#7-hierarchical-diagonalization)
8. [Operator method generation](#8-operator-method-generation)
9. [Parameter lifecycle and central dispatch](#9-parameter-lifecycle-and-central-dispatch)
10. [Noise model](#10-noise-model)
11. [Plotting](#11-plotting)
12. [Symbolic methods (LaTeX, transforms)](#12-symbolic-methods-latex-transforms)
13. [Public API surface](#13-public-api-surface)
14. [Test infrastructure](#14-test-infrastructure)
15. [Performance: hot paths](#15-performance-hot-paths)
16. [Extension recipes](#16-extension-recipes)
17. [Maintenance pitfalls](#17-maintenance-pitfalls)
18. [Where to look for what](#18-where-to-look-for-what)

---

## 1. Scope and module map

The `circuit` module lives at `scqubits/core/`. Layout:

```
scqubits/core/
├── circuit.py                    top-level: Circuit
│                                 module-level (scqubits.core.circuit):
│                                            Subsystem, ConfigureError
├── symbolic_circuit.py           top-level: SymbolicCircuit
├── symbolic_circuit_graph.py     module-level (scqubits.core.symbolic_circuit_graph):
│                                            Node, Branch, Coupler
│                                 internal: SymbolicCircuitGraph (ABC), _AdjacencyIndex
│
├── circuit_input.py              back-compat shim → circuit_internals.input
├── circuit_utils.py              back-compat shim → circuit_internals.utils
│
└── circuit_internals/            private umbrella; empty __init__.py
    ├── routines.py                CircuitRoutines (Hilbert-space basics)
    ├── lifecycle.py               LifecycleMixin (param sync, dispatch)
    ├── subsystem_tree.py          SubsystemTreeMixin (HD subsystem construction)
    ├── hamiltonian_assembly.py    HamiltonianAssemblyMixin (operators + Hamiltonian)
    ├── sym_methods.py             CircuitSymMethods (LaTeX, sym transforms)
    ├── plotting.py                CircuitPlot (wave-function, potential plots)
    ├── noise.py                   NoisyCircuit (T1, T2 channels)
    │
    ├── input.py                   YAML parser + parsing constants
    ├── yaml_assembly.py           assemble_circuit, assemble_transformation_matrix
    ├── branch_metadata.py         _junction_order, _capacitance_variable_for_branch
    │
    ├── operator_factories.py      make_grid_operator_method, make_basis_operator_method,
    │                              make_hierarchical_diag_method
    ├── charge_basis_operators.py  _cos_theta, _sin_theta, _n_theta_operator, ...
    ├── discretized_phi_operators.py  _phi_operator, _cos_phi, _sin_phi, _i_d_dphi_operator
    ├── matrix_helpers.py          _cos_dia, _sin_dia, matrix_power_sparse
    ├── sawtooth.py                sawtooth_operator, sawtooth_potential
    │
    ├── sympy_helpers.py           round_symbolic_expr, is_potential_term, ...
    └── utils.py                   truncation_template, get_trailing_number, ...
```

Total: ~14 000 lines. The three top-level files (`circuit.py`,
`symbolic_circuit.py`, `symbolic_circuit_graph.py`) contain every name
that is part of the published public API. Everything inside
`circuit_internals/` is implementation detail and may be reorganised
between minor releases without notice.

---

## 2. The two-stage pipeline

Custom-circuit analysis runs as a strict two-stage pipeline:

```
            ┌──────────────────────┐    ┌──────────────────────┐
   YAML ─►  │  SymbolicCircuit     │ ─► │  Circuit             │ ─► numerics
            │  (sympy expressions, │    │  (numerical operators,│
            │  graph topology)     │    │  Hilbert space)       │
            └──────────────────────┘    └──────────────────────┘
                  symbolic_circuit.py        circuit.py
```

**Stage 1 — `SymbolicCircuit`.** Inputs: a YAML description of
topology (nodes; capacitances `C`, inductances `L`, Josephson
junctions `JJ`, optional mutual inductances `ML`). Output: a
`SymbolicCircuit` instance whose key attributes are
`hamiltonian_symbolic` (a sympy `Expr`), `var_categories` (the
classification dict, see §4.4), `transformation_matrix` (an ndarray),
and metadata about external fluxes, offset charges, free charges, and
branch parameters.

**Stage 2 — `Circuit`.** Inputs: a `SymbolicCircuit` (or in some
flows, a pre-built sympy Hamiltonian directly via the
`symbolic_hamiltonian=` constructor argument). Output: a `Circuit`
instance whose `hamiltonian()` and `eigenvals()` produce numerical
arrays. Stage 2 work happens inside `Circuit.configure(...)`, which
calls `_configure(...)`; see §6.3.

---

## 3. Class hierarchy and mixin composition

### 3.1 Symbolic side

```python
class SymbolicCircuitGraph(ABC):                # symbolic_circuit_graph.py
    ...

class SymbolicCircuit(serializers.Serializable, SymbolicCircuitGraph):
    # Serializable is the FIRST base — it appears before
    # SymbolicCircuitGraph in MRO.  symbolic_circuit.py
    ...
```

To check the actual MRO at runtime:

```python
[c.__name__ for c in scq.SymbolicCircuit.__mro__]
# → ['SymbolicCircuit', 'Serializable', 'Protocol', 'Generic',
#    'SymbolicCircuitGraph', 'ABC', 'object']
```

`SymbolicCircuitGraph` is a private base class that owns the
graph-theoretic algorithms (spanning trees, closure branches,
capacitance/inductance matrix construction, the
variable-transformation matrix). Several methods one might assume
"belong to `SymbolicCircuit`" actually live on the graph base —
notably `_time_dependent_flux_distribution`,
`_merge_branch_symbols`, `_find_path_to_root`, `_find_loop`. The
`Node` and `Branch` classes (public API for closure-branch user
overrides) live in the same file.

`SymbolicCircuit` is a thin extension that adds the symbolic
Lagrangian and Hamiltonian construction (Eq. 6, Eq. 12 of the paper)
plus YAML-string round-trip support.

### 3.2 Numerical side

`CircuitRoutines` is composed via multiple inheritance from four
sibling mixins. The actual class declarations:

```python
class CircuitRoutines(
    LifecycleMixin,
    SubsystemTreeMixin,
    HamiltonianAssemblyMixin,
    ABC,
):
    ...

class CircuitABC(CircuitRoutines, CircuitSymMethods, CircuitPlot):
    pass

class Subsystem(
    CircuitABC, base.QubitBaseClass, Serializable, DispatchClient, NoisyCircuit
):
    ...

class Circuit(
    CircuitABC, base.QubitBaseClass, Serializable, DispatchClient, NoisyCircuit
):
    ...
```

Python resolves method calls via C3 linearisation of this list — strictly
linear, not parallel. To check the actual order at runtime:

```python
[c.__name__ for c in scq.Circuit.__mro__]
```

What each mixin owns:

| Mixin | Concern |
|---|---|
| `LifecycleMixin` | Parameter sync across the subsystem tree, central-dispatch event reception, `WatchedProperty` setter installation, the `update()` pipeline. |
| `SubsystemTreeMixin` | Hierarchical-diagonalization tree construction (decompose Hamiltonian → child Subsystems, build HilbertSpace interactions). |
| `HamiltonianAssemblyMixin` | Per-variable operator-method construction, JJ cosine/sine matrix evaluation, Hamiltonian assembly, eigenvalue/eigensystem computation. |
| residual `CircuitRoutines` | Hilbert-space basics: `cutoffs_dict`, `hilbertdim`, `_kron_operator`, `_identity`, serialization, bare-eigensystem caching. |
| `CircuitSymMethods` | Internal LaTeX rendering and symbolic-side transforms. (Note: the public `sym_lagrangian`, `sym_hamiltonian`, etc. methods live on `Circuit` itself and call into this mixin.) |
| `CircuitPlot` | Wave-function plots, potential-energy plots, cutoff accessors. |
| `NoisyCircuit` | T1/T2 noise channels (capacitive loss, inductive loss, dielectric, flux noise, …). Sits outside `CircuitABC` because it inherits from `NoisySystem` in `scqubits.core.noise`, which is a separate concern hierarchy. |

### 3.3 Why this many mixins?

Single Responsibility per concern. Each mixin owns one kind of work
(parameter sync, tree construction, operator assembly, etc.) and
declares the cross-mixin attributes / methods it depends on under
`if TYPE_CHECKING:` annotations. Adding a sixth mixin should preserve
the same pattern; see §16.4 for the recipe.

Line count is a side effect, not a design rule. `noise.py` is 1923
lines and `sym_methods.py` is 1520 lines; both own coherent concerns
and should not be trimmed for length.

---

## 4. The symbolic circuit graph layer

Owns: the YAML parser, the `Node`/`Branch`/`Coupler` data structure,
the spanning-tree and closure-branch algorithms, and the
variable-transformation matrix.

### 4.1 YAML grammar

Inputs are processed by `circuit_internals/input.py`:

```yaml
branches:
- [JJ, 1, 2, EJ=10, ECJ=20]    # Josephson junction; EJ, ECJ in GHz
- [L, 2, 3, EL=0.04]           # inductance; EL in GHz
- [C, 1, 3, EC=2]              # capacitance; EC in GHz
- [ML, 0, 1, EML=0.01]         # mutual inductance between branches indexed 0 and 1
```

- **Node IDs** are integers. The validation rule is `min(node_ids) in
  {0, 1}`: lowest-numbered node is either 0 (interpreted as ground)
  or 1 (no ground). Node IDs do not have to be contiguous; gaps are
  allowed.
- **Branch parameters** can be numeric literals (`10`) or named
  symbols with optional default values (`EJ=10` declares `EJ` with
  default 10). The same symbol on a later branch with a non-empty
  value of any kind raises `ValueError` from
  `_merge_branch_symbols`. Re-declaring with an empty value is
  silently allowed (it just references the prior definition).
- **Couplers** (`ML`, magnetically-coupled-`L`) take two branch
  *indices* (positions in `parsed_branches`, not node IDs) plus
  parameters. They produce a `Coupler` object that adds off-diagonal
  terms to the inductance matrix; see §4.2 below.
- The grammar is parsed by `pyparsing`. Constants `BRANCHES`,
  `BRANCH_TYPES`, `JJ_ORDER`, `aux_val`, `prefix_dict`, etc. all
  live in `input.py:__all__` for users who want to extend the
  parser.

### 4.2 The graph data structure

```python
class Node:
    index: int           # integer ID (0 ⇒ ground node)
    branches: list[Branch]   # back-pointers populated by Branch.__init__

class Branch:
    nodes: tuple[Node, Node]
    type: str            # "C", "L", "JJ", "JJ2", "JJs", ...
    parameters: dict     # {"EC": 2.0} for C; {"EJ": 10, "ECJ": 20} for JJ; ...
    index: int           # branch position in declaration order

class Coupler:
    branches: tuple[Branch, Branch]
    type: str            # "ML"
    parameters: dict
    index: int
```

`Branch.parameters` is *not* the constructor's raw `parameters`
list. The dict shape is computed by `Branch._set_parameters` and
depends on `branch_type`:
- `C` → `{"EC": value}`, `L` → `{"EL": value}`.
- `JJ` → `{"EJ": ej, "ECJ": ecj}`. `JJ2` adds `EJ2`; `JJ3` adds `EJ3`;
  in general `JJ<n>` adds `EJ` through `EJ<n>` and one trailing
  `ECJ`. The mapping is encoded in `_junction_order` (in
  `circuit_internals/branch_metadata.py`).
- `JJs` is the sawtooth-junction model (see `circuit_internals/sawtooth.py`).

`Node.__eq__` and `Branch.__eq__` are defined by `.index`. This is
load-bearing: the spanning-tree algorithm runs on a
`copy.deepcopy(self)` and then remaps the resulting `Branch`/`Node`
references back to `self`'s actual instances via
`_local_copy_of_branch` / `_local_copy_of_node`, which use `==` (i.e.,
`.index`) to find the match. Changing the equality semantics silently
breaks remap.

#### Couplers in more detail

A `Coupler("ML", b1, b2, M=...)` represents a *mutual inductance*
between two `L` branches `b1` and `b2`. It contributes an
off-diagonal entry `M` to the inductance matrix. Critically:

- Only `L` branches can be coupled; coupling JJs or capacitors via
  `ML` is rejected at parse time.
- `Coupler` does **not** participate in the spanning-tree
  construction (it has no nodes of its own). Its sole role is to
  modify the symbolic inductance matrix in
  `SymbolicCircuit.generate_symbolic_lagrangian`.
- Variable categorization (§4.4) treats coupled branches the same
  as their underlying L branches; the coupling is invisible to the
  classification.

### 4.3 Spanning tree, closure branches, and `_AdjacencyIndex`

`SymbolicCircuitGraph._spanning_tree(consider_capacitive_loops=False,
use_closure_branches=True)` returns a dict with four keys:

- `list_of_trees: list[list[Branch]]` — one spanning tree per
  connected component of the inductive subgraph (more on this
  below).
- `node_sets_for_trees: list[list[list[Node]]]` — for each tree, the
  BFS-layered node sets (generation 0 = root, generation 1 = its
  neighbours, …).
- `loop_branches_for_trees: list[list[Branch]]` — for each tree, all
  branches with both endpoints in the tree's node set.
- `closure_branches_for_trees: list[list[Branch]]` — the subset of
  loop branches that close superconducting loops (each contributes
  one external flux).

**What "connected component" means here is non-obvious.** With
default `consider_capacitive_loops=False`, capacitor branches are
*removed* before tree construction. A "tree" therefore spans the
`L+JJ`-connected components of the circuit. A circuit whose
capacitive branches connect everything but whose inductive branches
sit on disjoint sub-graphs has *multiple* trees — one per
inductive component.

The algorithm proceeds in three named phases:

1. `_construct_initial_tree_per_component` — greedy spanning tree
   per component, walking BFS layers (built by
   `_build_node_sets_for_trees`) and picking one branch per layer
   transition.
2. `_loop_branches_and_closures_per_component` — for each component,
   compute `(loop_branches, closure_branches)` by inspecting which
   non-tree branches have both endpoints inside the tree's node
   set.
3. `_apply_user_closure_policy` — if the user supplied explicit
   `Branch` closures via `Circuit.configure(closure_branches=...)`,
   override the auto-derived split: tree = `loop_branches \
   user_closures`, closures = `loop_branches ∩ user_closures`.

After `_spanning_tree` returns, `_find_path_to_root` and `_find_loop`
need to walk the tree from a given node to the root. These are
implemented as O(depth) lookups against an `_AdjacencyIndex` (private
class in `symbolic_circuit_graph.py`) built once per spanning-tree
dict and cached on `self._adjacency_cache`. Because a spanning tree
is acyclic, the path between any two nodes is unique — a DFS
returns the same path that any correct algorithm would.

### 4.4 Variable categorization

The paper defines (§2.2):

> *A variable θₑ is **extended** if it describes a direction in
> configuration space in which the potential energy is confining. A
> variable θₚ is **periodic** if it aligns with a direction in which
> the potential energy is periodic. A variable θ_c is **free** if
> ∂_{θ_c} L = 0 (does not appear in the potential). A variable θ_f
> is **frozen** if ∂_{θ̇_f} L = 0 (its time derivative is missing
> from the kinetic energy).*

A fifth category, **sigma**, is the center-of-mass mode that arises
when the circuit is not grounded; its conjugate momentum vanishes and
it is dropped from the dynamics.

`var_categories` is a dict with keys
`Literal["periodic", "extended", "free", "frozen", "sigma"]`,
each mapping to a list of 1-based variable indices. The `Literal`
type is exported as `VarCategoryKey` from `scqubits.core.circuit`
(not from the top-level `scqubits` namespace).

#### Implementation: kernel of incidence on a branch subset

The classification is computed by
`SymbolicCircuitGraph.variable_transformation_matrix()` and rests on
one primitive: `_independent_modes(B)` returns a basis for the
*kernel of the incidence operator restricted to branch subset
`B`* — equivalently, the subspace of node-vectors with no flux
difference across any branch in `B`.

The three canonical mode bases come from three different `B`:

| Mode kind | `B = ` | Physical meaning |
|---|---|---|
| **periodic_modes** | L branches | No flux drop on any inductor ⇒ no inductive (quadratic-in-θ) potential energy ⇒ potential is purely cos/sin in this variable ⇒ confining only mod 2π. |
| **frozen_modes** | non-L branches (= C and JJ) | No flux drop on any C or JJ ⇒ no kinetic energy from those branches ⇒ ∂_{θ̇} L = 0 ⇒ Euler-Lagrange becomes algebraic. |
| **free_modes** | non-C branches (= L and JJ) | No flux drop on any L or JJ ⇒ θ is absent from the potential ⇒ cyclic coordinate ⇒ conjugate momentum conserved. |
| **LC_modes** | JJ branches only | Used as additional candidates when filling out the basis; each JJ pair produces a candidate "loop" mode. |

(Note JJ appears in both the `non-L` and `non-C` subsets — it
contributes both kinetic and potential energy.)

`_independent_modes` is called from two places with **different**
`basisvec_entries`, and the distinction is load-bearing:

- `check_transformation_matrix` uses `basisvec_entries=[1, 0]` —
  one-hot probes, used to test whether a candidate transformation
  fully classifies every mode.
- `variable_transformation_matrix` uses `basisvec_entries=[-1, 1]`
  — bipolar probes, used to actually build the basis vectors that
  become rows of the transformation matrix.

Both share the same algorithm (max-connected subgraph
partitioning + per-component basis assembly); only the entry
values differ.

#### From mode lists to `var_categories`

After building the four mode lists plus the Σ vector (added to
`frozen_modes` if the circuit is non-grounded):

1. **Greedy basis assembly.** Iterate
   `frozen_modes + free_modes + periodic_modes + LC_modes` in that
   order, accepting each vector that strictly increases the rank of
   the current basis. The order is load-bearing for the
   classification — frozen wins over free wins over periodic wins
   over LC.
2. **Standard-basis completion.** Extend to a full-rank basis using
   either the heuristic completion (in
   `_complete_basis_with_standard_vectors`) or the canonical
   identity matrix, per `basis_completion`. The heuristic walks
   `reversed(itertools.combinations(range(n), zeros_count))` to
   generate candidate vectors with `zeros_count` zero-positions
   and the rest ones; this iteration order matches the legacy
   `itertools.permutations` first-occurrence order, so the resulting
   basis is byte-equivalent to pre-refactor output.
3. **Single-pass row classification.** Build a
   `dict[tuple, str]` mapping each accepted basis row to its label
   (precedence: sigma > free > periodic > frozen > rest). Walk the
   completed basis once, dropping each row into the right bucket.
   Rows not in any of the canonical mode lists become "rest" =
   *extended*.
4. **Reorder.** Permute rows so they come out in block order
   `periodic → extended → free → frozen → sigma`. The permutation
   defines `var_categories`.

### 4.5 `_independent_modes` in detail

This is the main mode-construction primitive:

```python
def _independent_modes(
    self,
    branch_subset: list[Branch],
    single_nodes: bool = True,
    basisvec_entries: list[int] | None = None,
) -> list[list[int]]:
```

Algorithm:

1. **Order the nodes** with the ground node (if any) at the end.
   This is the column ordering of the returned basis vectors.
2. **Partition `branch_subset` into max-connected subgraphs** via
   `_max_connected_branch_subgraphs` — a transitive closure on
   `Branch.is_connected`.
3. **Compute per-node subgraph membership** via
   `_compute_subgraph_membership` (returns a list of integer markers
   parallel to the node order — `-1` for subgraphs that touch the
   ground node, `0` for nodes not in any subgraph in `branch_subset`,
   and `k>0` for the *k*-th non-grounded subgraph).
4. **Build basis vectors:**
   - One vector per non-grounded subgraph: marks subgraph members
     with `basisvec_entries[0]` (default `1`), others with
     `basisvec_entries[1]` (default `0`).
   - If `single_nodes=True`, also try one-hot vectors for nodes with
     subgraph index `0`, accepting each that strictly increases rank.
5. **Drop the ground-node column when grounded.** If
   `self.is_grounded`, drop the trailing column corresponding to the
   ground node: `basis = [vec[:-1] for vec in basis]`. For
   non-grounded circuits the column is retained — every node
   degree of freedom contributes to the basis.

---

## 5. The symbolic circuit layer

`SymbolicCircuit` (in `symbolic_circuit.py`, ~1 250 lines) wraps the
graph layer and adds:

- **YAML construction.** The classmethod `SymbolicCircuit.from_yaml(yaml,
  from_file=...)` is the ground-truth factory. The numerical-side
  classmethods `Circuit.from_yaml_file` and `Circuit.from_yaml_string`
  are thin wrappers — both go straight to `Circuit(input_string=...,
  from_file=...)`; they do *not* chain through one another.
- **`SymbolicCircuit.configure(...)`** — separate from
  `Circuit.configure(...)`. Recomputes
  `transformation_matrix`, `var_categories`,
  `external_fluxes`, etc. when the user supplies a new
  `transformation_matrix` or `closure_branches` policy.
- **Symbolic Lagrangian.** `generate_symbolic_lagrangian()` builds
  the full Lagrangian in node variables, then applies the inverse
  variable-transformation to express it in new variables.
- **Symbolic Hamiltonian.** `generate_symbolic_hamiltonian()` does
  the Legendre transform; result is stored on
  `self.hamiltonian_symbolic`.
- **External-flux assignment.** The list `external_fluxes:
  list[Symbol]` has one entry per closure branch.
  `_set_external_fluxes` populates it. The companion routine
  `_time_dependent_flux_distribution` (which lives on
  `SymbolicCircuitGraph`, not `SymbolicCircuit`) extends Kirchhoff's
  loop rule (paper §2.1, Eq. 2) to time-dependent external fluxes;
  the paper itself develops only the time-independent case.

The result of stage 1 is exposed to stage 2 by
`Circuit._import_from_symbolic_circuit`, which copies a list of
attributes (`hamiltonian_symbolic`, `var_categories`,
`external_fluxes`, `offset_charges`, `free_charges`,
`symbolic_params`, `closure_branches`, `transformation_matrix`,
…) onto `self`. The exact list is encoded in the helper's body —
**when adding a new attribute that needs to flow from stage 1 to
stage 2, add it there**. It is not enough to set it on
`SymbolicCircuit`.

---

## 6. The numerical layer (`Circuit`, `Subsystem`)

### 6.1 Construction paths

```python
Circuit.from_yaml_file(path, ...)            # preferred for files
Circuit.from_yaml_string(yaml_text, ...)     # preferred for inline YAML
Circuit(input_string=..., from_file=...)     # legacy; sentinel-protected DeprecationWarning
Circuit(symbolic_hamiltonian=..., ...)       # alternative; pre-built sympy Expr
```

Both named constructors call `cls(input_string=..., from_file=True/
False)` directly — neither chains through the other.

`Circuit.__init__` does, roughly in this order (read the actual
function for the literal sequence — the deprecation handling and
`_frozen` toggle bracket the steps below):

1. Validate the input (must have either `input_string` or
   `symbolic_hamiltonian`, not both).
2. Set core attributes (`is_child=False`, `evals_method`, …).
3. Build a `SymbolicCircuit` via `self.from_yaml(...)` or process the
   user-provided sympy Hamiltonian via
   `self._from_symbolic_hamiltonian(...)`.
4. The chosen path then calls `self.configure(...)`, which calls
   `self._configure(...)`. The rollback contract is at
   `self.configure(...)` — see §6.3.

### 6.2 `Subsystem`

A subsystem is a recursive child of a `Circuit` produced by
hierarchical diagonalisation. Same mixin chain as `Circuit`, except:

- The constructor takes a `parent` and a sliced symbolic Hamiltonian
  rather than a fresh YAML.
- `is_child=True`.
- It defines its own `_configure(...)` (in `circuit.py`),
  which is *much* simpler than `Circuit._configure(...)` — it skips
  YAML parsing and graph-construction, taking those from the parent.

> **Why aren't `Subsystem._configure` and `Circuit._configure` unified?**
> The two methods *look* parallel but their HD-vs-non-HD branches have
> load-bearing content differences (Circuit calls
> `_generate_hamiltonian_sym_for_numerics()` where Subsystem just
> `.copy()`s, because Subsystem's symbolic Hamiltonian was already
> processed by the parent; Circuit warns on `ext_basis="harmonic"`
> mismatch where Subsystem inherits silently; Circuit resets
> `operators_by_name = None` where Subsystem doesn't; etc.).
> The truly-shared body across both methods is roughly four lines
> (`_frozen` toggles + `_set_vars()` + `_set_operators()`). A 2026-05-10
> design-review pass concluded that a template-method extraction would
> require either two near-identical helpers (re-introducing the
> duplication) or a flag-soup `_configure_common` whose body is
> "if circuit_path: ... else: ..." (worse than the current
> independent methods). Keep them independent; the divergence is
> intentional, not accidental.

### 6.3 The `_configure` flow and the rollback contract

`Circuit._configure(...)` is the single point where every observable
piece of state is wired up. Roughly nine sub-phases run in sequence:

1. Rebuild `symbolic_circuit` if a transformation matrix or
   closure-branch override was passed.
2. `_import_from_symbolic_circuit` — copy the attributes from
   `SymbolicCircuit` to self.
3. `_install_var_properties` — install `WatchedProperty` descriptors
   for every cutoff (`cutoff_n_<i>`, `cutoff_ext_<i>`), external flux
   (`Φ<i>`), offset charge (`ng<i>`), free charge (`ng_free_<i>`),
   and symbolic param (`EJ`, `EC`, …).
4. Choose default cutoffs and the discretized-phi range (defaults:
   `DEFAULT_PERIODIC_CUTOFF=5`, `DEFAULT_EXTENDED_CUTOFF=30`,
   half-range `6π`).
5. If purely harmonic, run `_diagonalize_purely_harmonic_hamiltonian`
   to decouple oscillator modes analytically; shift the harmonic
   potential to its minimum.
6. If `hierarchical_diagonalization`, call `_generate_subsystems`
   (defined on `SubsystemTreeMixin`).
7. Build `self.vars` (the operator-symbol dictionary used by the
   factories) via `_set_vars` or `_set_vars_no_hd`.
8. Build the dynamic per-variable operator methods via
   `_set_operators` and bind them with `types.MethodType`.
9. Trigger an `update()` to compute the initial bare eigensystem.

#### The rollback contract

The `try/except` rollback is **not** in `_configure` — it is at the
public boundary, inside `Circuit.configure(...)`. The pattern is:

```python
def configure(self, ...):
    # snapshot prior config
    old_X = self.X
    ...
    try:
        self._configure(...)
    except Exception as exc:
        self.X = old_X
        ...
        self._configure(...)  # restore
        raise ConfigureError(...) from exc
```

**Maintenance invariant:** do not introduce a nested `try/except`
inside `_configure(...)` itself. The rollback assumes flat error
propagation from `_configure` outward to `configure`. A nested
handler that catches and silently logs an exception will leave the
instance in a half-configured state with no rollback.

If you need conditional behavior inside `_configure` based on a
recoverable condition, branch on a predicate rather than catching an
exception. The `ConfigureError` chain (`raise ... from exc`) is the
guarantee that the original cause is preserved for the caller.

---

## 7. Hierarchical diagonalization

Concept (paper §4.2): rather than diagonalising the full Hilbert
space (size = product of all per-variable cutoffs, exponential),
partition variables into subsystems, diagonalise each subsystem in
its own Hilbert space, retain only its low-lying eigenstates, and
assemble a much smaller "dressed" Hilbert space from those reduced
bases.

scqubits expresses this as a tree:

- The user supplies `system_hierarchy` to `Circuit.configure(...)`
  — a nested list of variable indices describing the partition.
  Example: `system_hierarchy=[[1, 3], [2]]` says "subsystem A holds
  variables 1 and 3, subsystem B holds variable 2".
- `subsystem_trunc_dims` — parallel nested list of truncation
  dimensions per subsystem. The helper
  `scqubits.truncation_template(system_hierarchy)` produces a
  reasonable default: 6 for each leaf subsystem, 30 for non-leaf.

Cutoff vs. truncation:
- *Cutoffs* (`cutoff_n_<i>`, `cutoff_ext_<i>`) bound the bare basis
  size for variable `<i>`. They live on the leaf circuit / subsystem
  that owns the variable.
- *Truncation* (`truncated_dim`) bounds the number of low-lying
  eigenstates kept after a subsystem is diagonalised in its bare
  basis. The next level up sees only `truncated_dim` states from
  this subsystem.
- A subsystem's `hilbertdim()` is the product of cutoffs of its
  variables (post-cutoff but pre-truncation). The
  `_check_truncation_indices` helper raises `ValueError` if any
  `truncated_dim >= hilbertdim() - 1`.

The construction flow in `_configure`:

1. `_generate_subsystems` (on `SubsystemTreeMixin`) decomposes the
   symbolic Hamiltonian via `_get_systems_and_interactions`, which
   delegates the heavy symbolic work to
   `_sym_subsystem_hamiltonian_and_interactions` on
   `CircuitSymMethods`.
2. For each top-level group, construct a child `Subsystem`.
3. Build a `~scqubits.HilbertSpace` from the subsystems.
4. `_update_interactions(recursive=...)` rewrites the symbolic
   interaction terms into HilbertSpace `add_interaction` calls. Each
   interaction term is wrapped via `_operator_from_sym_expr_wrapper`
   so it evaluates correctly when the user mutates a parameter.

Subsystems can themselves be hierarchically diagonalised — the tree
is fully recursive. Termination is when a subsystem holds only one
variable, or when `system_hierarchy=None`.

---

## 8. Operator method generation

For a circuit with variables 1, 2, 3, the user expects
`qubit.n1_operator()`, `qubit.θ2_operator()`, `qubit.cosθ1_operator()`,
and so on. These methods are bound dynamically via
`types.MethodType` in `_set_operators`, so they are invisible to
`mypy`, IDE autocomplete, and `sphinx-autodoc`. The typed accessor
`qubit.operator(name, *, energy_esys=False)` (defined on
`HamiltonianAssemblyMixin`) is the recommended way to access them
when static-tooling visibility matters: `qubit.operator("n1")` is
equivalent to `qubit.n1_operator()` but is fully typed.

**Note the naming**: no underscore between the prefix
letter and the digit, no underscore between `cos`/`sin` and the
variable. This is enforced by the regex in
`sym_methods.py`'s `_get_eval_hamiltonian_string`:

```python
re.sub(r"(?P<x>(θ\d)|(cosθ\d))", r"\g<x>_operator()", H_string)
```

The full naming inventory (per `_build_extended_operators_*`,
`_build_periodic_operators_*`, and `_make_purely_harmonic_operator_method`):

- *Periodic charge*: `n1_operator`, `n2_operator`, …
- *Periodic trig*: `cosθ1_operator`, `sinθ1_operator`, …
- *Extended position/momentum*: `θ1_operator`, `Q1_operator`, …
- *Discretized extras*: `Qs1_operator` (charge-squared, used for
  ``Q^2`` terms in the discretized basis).
- *Harmonic-basis extras*: `a1_operator`, `ad1_operator` (annihilation,
  creation), `Nh1_operator` (number).
- *Identity*: `I_operator`.
- *Cross-cutting helper*: `exp_i_operator` — defined on
  `HamiltonianAssemblyMixin` rather than per-variable. Used by
  `junction_assembly.build_junction_phase_operator_list` (§8 below)
  when assembling `cos(φ)` / `sin(φ)` matrix elements for a JJ phase
  that is a linear combination of variables.

These methods are built dynamically during `_configure` by
`_set_operators` and attached via `types.MethodType`. Three factory
functions in `circuit_internals/operator_factories.py` build the
per-variable closures:

| Factory | When used | Returns a method that… |
|---|---|---|
| `make_grid_operator_method(inner_op, index)` | Discretized-phi extended variables | Evaluates `inner_op` on the variable's `Grid1d` (from `discretized_grids_dict_for_vars`), embeds via `_kron_operator`. |
| `make_basis_operator_method(inner_op, index, op_type=None)` | Periodic (charge-basis) and harmonic-basis variables | Runs `inner_op` on `cutoffs_dict()[index]`. For harmonic basis with `op_type` in `{"position", "momentum", "sin", "cos"}`, applies the `osc_lengths`-derived prefactor. |
| `make_hierarchical_diag_method(symbol_name)` | HD child variables | Retrieves the named operator from the parent subsystem via `get_operator_by_name`, converts to `csc_matrix`. |

The dispatch is done by `_generate_operator_methods`, which calls
`_build_extended_operator_methods` and
`_build_periodic_operator_methods`. Each of those calls a
basis-specific helper such as `_build_extended_operators_discretized`
or `_build_extended_operators_harmonic`.

The bare per-basis primitives are split across:

- `circuit_internals/charge_basis_operators.py` — `_cos_theta`,
  `_sin_theta`, `_n_theta_operator`, `_exp_i_theta_operator`,
  `_identity_theta`.
- `circuit_internals/discretized_phi_operators.py` — `_phi_operator`,
  `_cos_phi`, `_sin_phi`, `_i_d_dphi_operator`,
  `_i_d2_dphi2_operator`, `_identity_phi`. The discretized basis
  uses `Grid1d` from `scqubits.core.discretization`; per-variable
  grid spans are read from `discretized_phi_range[var_index]`.
- `circuit_internals/sawtooth.py` — `sawtooth_operator`,
  `sawtooth_potential` (sawtooth-junction model).
- `circuit_internals/matrix_helpers.py` — `_cos_dia` / `_sin_dia` for
  cos/sin of a diagonal sparse matrix; `matrix_power_sparse`.

#### JJ matrix evaluation

A junction contributes `E_J cos(φ_J)` to the potential, where φ_J is
a *linear combination* of node-flux variables. The full pipeline lives
in `circuit_internals/junction_assembly.py` as module-level functions
taking an explicit ``circuit`` argument:

1. `evaluate_matrix_cosine_terms(circuit, junction_potential, ...)`
   iterates the sympy terms.
2. For each, `extract_junction_phase(term)` returns the phase
   expression inside the `cos` or `sin`.
3. `build_junction_phase_operator_list(circuit, phase_expr,
   var_indices, bare_esys)` builds the per-variable
   `exp(i · prefactor · var)` operators.
4. `assemble_cos_term(op)` combines them as `(O + O†)/2`;
   `assemble_sin_term(op)` does `−i(O − O†)/2`.
5. The `term_has_cos_factor` / `term_has_sin_factor` predicates pick
   which assembler to call for each sympy term.

`HamiltonianAssemblyMixin._evaluate_matrix_cosine_terms` and
`_evaluate_matrix_sawtooth_terms` are thin instance-method wrappers
around the `junction_assembly.*` functions, kept on the mixin so
existing callers (notably ``CircuitSymMethods._evaluate_symbolic_expr``)
that invoke ``self._evaluate_matrix_cosine_terms(...)`` keep working.

---

## 9. Parameter lifecycle and central dispatch

Every cutoff / flux / offset-charge / free-charge / symbolic param on
a `Circuit` or `Subsystem` is a `WatchedProperty` (defined in
`scqubits/core/descriptors.py`). The setter routes through
`settings.DISPATCH_ENABLED`.

The setter installation is done by `LifecycleMixin._make_property`
during `_configure`. The descriptor pattern is the load-bearing
piece: `WatchedProperty.__set__` does *not* assign the value
directly. Instead it dispatches through
`_PROPERTY_SETTER_BY_TYPE[update_type]` to one of three named
setter methods on the instance. Each property is registered with
its `update_type` (`"update_param_vars"`,
`"update_external_flux_or_charge"`, or `"update_cutoffs"`), and the
descriptor reads that mapping every time the user assigns. To add
a new lifecycle event kind, add a key to
`_PROPERTY_SETTER_BY_TYPE` and an instance method with the
matching name; the descriptor wires up automatically.

Setter dispatch uses a constant lookup table:

```python
PropertyUpdateType = Literal[
    "update_param_vars",
    "update_external_flux_or_charge",
    "update_cutoffs",
]

_PROPERTY_SETTER_BY_TYPE = {
    "update_param_vars": "_set_property_and_update_param_vars",
    "update_external_flux_or_charge":
        "_set_property_and_update_ext_flux_or_charge",
    "update_cutoffs": "_set_property_and_update_cutoffs",
}
```

Each setter wraps the actual write in a `_dispatch_suspended()`
context manager that temporarily disables central dispatch (so the
write doesn't trigger a recursive update before all child state is
consistent), then restores it. The context manager's `try/finally`
makes the restoration robust to exceptions.

The three setter variants differ in *what they invalidate*:

- `_set_property_and_update_param_vars` (e.g. setting `EJ`,
  `EC`) — symbolic params changed ⇒ Hamiltonian must be re-evaluated.
  Calls `_propagate_param_to_affected_subsystems`.
- `_set_property_and_update_ext_flux_or_charge` (e.g. setting `Φ1`,
  `ng1`) — external flux / offset-charge values changed ⇒
  Hamiltonian re-evaluation but no operator-method rebuild. Also
  propagates to subsystems carrying that flux/charge.
- `_set_property_and_update_cutoffs` (e.g. setting `cutoff_n_1`) —
  cutoff changed ⇒ basis size changed ⇒ operator methods must
  eventually be rebuilt and bare eigensystem invalidated. The
  setter itself only validates the value, writes
  `self._<param_name>`, and calls
  `_propagate_param_to_affected_subsystems` to forward the new
  value to any subsystem that owns that cutoff. Operator-method
  rebuild is triggered later by `update()` (queued via the
  central-dispatch event).

When the user does `qubit.EJ = 12.0`:

```
qubit.EJ = 12.0
   → WatchedProperty.__set__
       → _set_property_and_update_param_vars("EJ", 12.0)
           → _propagate_param_to_affected_subsystems("EJ", 12.0)
               → for each subsystem holding `EJ`:
                     setattr(subsys, "EJ", 12.0)
                     # re-fires WatchedProperty on the child
           → fires central-dispatch event "QUANTUMSYSTEM_UPDATE"
   ─► next call to qubit.eigenvals() rebuilds the Hamiltonian
```

`_sync_parameters_with_parent` is a separate operation, invoked
later by `update()` when the parent's queued event drains; it is
**not** called from `_propagate_param_to_affected_subsystems`.

The `update()` method on `LifecycleMixin` is the entry point that
picks up queued events and refreshes whatever was invalidated; it is
called transparently from `eigenvals()`, `eigensys()`, and
`hamiltonian()` via the `@check_sync_status_circuit` decorator.

---

## 10. Noise model

`circuit_internals/noise.py` (~1900 lines) defines the `NoisyCircuit`
mixin — a subclass of `NoisySystem` (in `scqubits.core.noise`) with
circuit-specific noise channels.

The mixin overrides `supported_noise_channels()` and
`effective_noise_channels()` (the latter filters by topology — e.g.
flux-noise channels are only enabled if the circuit has external
fluxes). The set of channels is computed dynamically based on the
circuit's branches and external fluxes — there is **no** static
`noise_channels` constant.

The dynamically-generated methods are also recorded in an explicit
registry: `NoisyCircuit.channels()` returns
`dict[str, Callable]` mapping each generated noise-method name to
its bound callable. This is the recommended introspection surface
when you want to enumerate channels without walking ``self.__dict__``
or matching names by regex. ``supported_noise_channels()`` is now
sourced from this registry; the registry is populated by
``_register_noise_method``, which every per-channel generator
(``_generate_t1_methods``, ``_generate_overall_*``, etc.) calls in
place of bare ``setattr``.

Channel families:

- **Capacitive losses** — `t1_capacitive` plus per-capacitor variants
  named `t1_capacitive<branch.index>` (no underscore between kind
  and index). Per-branch values are read directly from
  `branch.parameters["EC"]`; there is no separate accessor helper.
- **Inductive losses** — `t1_inductive` plus per-inductor variants
  named `t1_inductive<branch.index>`. Per-branch values come from
  `branch.parameters["EL"]`. The shared per-branch wrapper is
  `_wrapper_t1_inductive_capacitive`, which the dynamic-method
  generator (`_generate_t1_methods`) instantiates once per
  inductor / capacitor and binds with the index-suffixed name.
- **Charge-impedance / dielectric noise** — `t1_charge_impedance`,
  `t1_flux_bias_line`, etc. Standard `NoisySystem` formula
  specialized for the circuit's charge / flux operators.
- **1/f flux noise** — `tphi_1_over_f_flux<n>` is a *per-closure-branch*
  method generated dynamically; one is created for each closure
  branch in `external_fluxes`. The signature follows
  `NoisySystem.tphi_1_over_f`: `(A_noise, i, j, esys=None,
  get_rate=False, **kwargs)`.
- **1/f critical-current noise** — `tphi_1_over_f_cc<branch_idx>`,
  one per JJ.
- **1/f offset-charge noise** — `tphi_1_over_f_ng<charge_idx>`, one
  per offset charge.

Most channel implementations follow the same pattern: build the
matrix element `<i|noise_op|j>` between bare-basis states, then plug
into the standard T1 / Tφ formula. The matrix-element computation
uses the dynamic `<symbol>_operator()` methods generated in §8.

---

## 11. Plotting

`circuit_internals/plotting.py` (~970 lines) defines `CircuitPlot`.
Two main user-facing methods:

- `plot_wavefunction(esys=None, which=...)` — for a non-HD circuit
  with up to 2 extended variables, plots the wavefunction in
  position space (1D line, 2D heatmap, or 3D surface).
- `plot_potential(...)` — plots the potential energy
  V(θ_1, θ_2, …) with all parameters substituted.

The two main internal helpers:

- `_recursive_basis_change(wf, …)` — for HD circuits, recursively
  un-does the eigenbasis rotation imposed by each subsystem so the
  wavefunction can be displayed in original coordinates.
- `_basis_change_harm_osc_to_n` — for harmonic-basis variables,
  converts oscillator-basis wavefunctions back to charge basis.

Cutoff accessors on the mixin: `_charge_cutoff(var_index)` returns
`cutoff_n_<var_index>`; `_extended_cutoff(var_index)` returns
`cutoff_ext_<var_index>`. These hide the attribute-name convention
from the rest of the plotting code.

---

## 12. Symbolic methods (LaTeX, transforms)

`circuit_internals/sym_methods.py` (~1500 lines) defines
`CircuitSymMethods`. Note that the user-visible LaTeX-rendering
methods — `sym_lagrangian`, `sym_hamiltonian`, `sym_potential`,
`sym_interaction`, and their `_expr` siblings — actually live on
`Circuit` itself (in `circuit.py`), not on the mixin. The mixin
provides the lower-level rendering and substitution machinery they
call.

Public-side methods (on `Circuit`):

- `sym_lagrangian(vars_type="node")` — pretty LaTeX of the Lagrangian
  in either `"node"` (default) or `"new"` (post-transformation)
  variables.
- `sym_hamiltonian(...)`, `sym_potential(...)`,
  `sym_interaction(subsys_index, ...)` — analogous for Hamiltonian,
  potential, subsystem interactions.
- `sym_lagrangian_expr(...)`, `sym_hamiltonian_expr(...)`,
  `sym_potential_expr(...)`, `sym_interaction_expr(...)` — variants
  that return the raw sympy `Expr` instead of rendering LaTeX. The
  legacy `return_expr=True` flag on the LaTeX methods is deprecated;
  new code should call the `*_expr` siblings directly.

Internal-use methods worth knowing:

- `_generate_hamiltonian_sym_for_numerics(...)` — bridge between
  `hamiltonian_symbolic` (the pure symbolic form) and
  `_hamiltonian_sym_for_numerics` (the form with identity-matrix
  placeholders ready for numerical substitution). Marks `Q**2` as
  `Qs<i>` in the discretized basis so it can be replaced with
  `_i_d2_dphi2_operator`. Tags external fluxes with `* I * 2π` and
  offset charges with `* I` via a single batched
  `xreplace(dict)`.
- `_potential_energy_symbols()` — returns the list of sympy symbols
  that contribute to the potential.
- `_kinetic_part(expr)` — returns the kinetic component of a
  Hamiltonian expression.
- `_substitute_parameters(...)` — substitutes `self.<param>`
  numeric values into a sympy expression by iterating over
  `symbolic_params` and calling `subs` once per symbol.
- `_replace_mat_mul_operator(term)` — turns a sympy product
  `θ1 * θ2 * Q3` into the Python-string form
  `θ1_operator() @ θ2_operator() @ Q3_operator()` for downstream
  evaluation.
- `_get_eval_hamiltonian_string(H)` — produces the Python-expression
  string that gets `eval()`-ed inside
  `_hamiltonian_for_purely_harmonic`. **The main `_evaluate_hamiltonian`
  flow does NOT use `eval()`** — it does sympy-side substitution via
  `_evaluate_symbolic_expr`. Only the purely-harmonic and
  LC-piece paths route through `eval()`.

---

## 13. Public API surface

### 13.1 Names a user is expected to import

From `scqubits` (the top-level package):

- `Circuit`
- `SymbolicCircuit`
- `truncation_template` — generates a default `subsystem_trunc_dims`
  for a given `system_hierarchy`.
- `assemble_circuit(...)` and `assemble_transformation_matrix(...)` —
  programmatic alternatives to YAML parsing for users who want to
  construct circuits in code. See §13.5.

From `scqubits.core.symbolic_circuit_graph`:

- `Node`, `Branch`, `Coupler` — used when constructing
  closure-branch lists for `Circuit.configure(closure_branches=...)`.

From `scqubits.core.circuit`:

- `Subsystem` — the recursive child class produced by hierarchical
  diagonalisation. Users do not normally instantiate it directly,
  but it appears in `isinstance` checks and type annotations.
- `ConfigureError` — raised by `Circuit.configure(...)` when the
  reconfiguration is rejected; the original cause is preserved as
  `__cause__`. See §6.3.
- `VarCategoryKey` and `ExtBasisChoice` — `Literal` aliases for
  downstream code that wants to type-check its arguments.

(Only `Circuit`, `SymbolicCircuit`, `truncation_template`,
`assemble_circuit`, and `assemble_transformation_matrix` are exposed
at the top-level `scqubits` namespace. `Node`, `Branch`, `Coupler`,
`Subsystem`, `ConfigureError`, and the `Literal` aliases must be
imported from their declaring modules — even though they are part
of the documented user-facing API.)

Everything under `scqubits.core.circuit_internals.*` is implementation
detail. The leading-underscore convention is enforced: any name
beginning with `_` may move or change without notice; do not import
it from external code.

### 13.2 Construction paths

```python
Circuit.from_yaml_file(path, ...)
Circuit.from_yaml_string(yaml_text, ...)
Circuit(input_string=..., from_file=...)     # legacy
Circuit(symbolic_hamiltonian=expr, symbolic_param_dict=..., ...)
```

The last form bypasses YAML parsing entirely. Internally it goes
through `_from_symbolic_hamiltonian` and `_read_symbolic_hamiltonian`
to populate the same attributes that `from_yaml` would. **Limitations
of the symbolic-Hamiltonian path:**

- No `symbolic_circuit` is built. Methods that depend on the graph
  structure (closure-branch detection, time-dependent flux
  distribution, sym_lagrangian, sym_potential decomposition) are
  unavailable.
- `Circuit.configure(...)` rejects `closure_branches=`,
  `use_dynamic_flux_grouping=True`, and `generate_noise_methods=True`
  with `ValueError` on the symbolic-Hamiltonian path. The
  `transformation_matrix=` argument is *silently ignored* on this
  path — the configure call accepts it without raising, but it is
  not consumed by `_configure_sym_hamiltonian`.
- Variable indices and names must already be encoded in the input
  expression (e.g. `θ1`, `Q1`, `n1`, `ng1`).

### 13.3 Lifecycle methods

- `Circuit.configure(transformation_matrix=..., system_hierarchy=...,
  subsystem_trunc_dims=..., closure_branches=..., ext_basis=...)` —
  reconfigure an existing instance. On failure raises
  `ConfigureError` and restores the prior configuration (§6.3).
- `Circuit.update(calculate_bare_esys=True)` — manually trigger a
  refresh. Most callers don't need this; it is invoked transparently
  by eigenvalue methods.

### 13.4 Inherited from `QubitBaseClass`

- `eigenvals(evals_count=...)`, `eigensys(evals_count=...)` — return
  the lowest-energy spectrum.
- `hamiltonian()` — return the full Hamiltonian as `csc_matrix`,
  `Qobj`, or `ndarray` depending on `type_of_matrices` and
  hierarchical-diagonalisation settings.

### 13.5 Programmatic circuit assembly

`assemble_circuit(branches_list, couplers_list=None,
basis_completion="heuristic", ...)` builds a `Circuit` from
already-constructed `Branch` and `Coupler` objects, bypassing the
YAML parser. Useful when generating circuits programmatically (e.g.
sweeping topology in a numerical experiment). The companion
`assemble_transformation_matrix(...)` lets the caller specify a
custom transformation matrix without going through
`SymbolicCircuit.configure`.

Both live in `circuit_internals/yaml_assembly.py` and are re-exported
from `scqubits.core.circuit_utils` for legacy reasons; the canonical
import path is:

```python
from scqubits import assemble_circuit, assemble_transformation_matrix
```

### 13.6 Serialization

`Circuit` and `Subsystem` inherit from `Serializable`. The
`serialize()` / `deserialize()` methods (defined on the residual
`CircuitRoutines`) round-trip via `dill`:

- `serialize()` pickles the entire instance into hex-encoded bytes
  inside an `IOData` object.
- `deserialize(io_data)` reconstructs by reverse pickling.

**Maintenance invariant:** what gets persisted is whatever `dill`
can pickle from `self.__dict__`. The 18-attribute import block in
`_import_from_symbolic_circuit` (§17.3) defines what *must* be set
for a fresh construction. The two lists overlap but are not the
same: the import block is the *recomputation contract*, while
`serialize` captures whatever was on the instance at the time. If
you add a new attribute that should survive a round-trip, ensure it
is set during `_configure` (so it's present at serialize time) and
that it is reachable via the recomputation contract (so a fresh
construction gives the same instance).

### 13.7 Dynamic operator methods

After `_configure` runs, the instance has methods of the form
`<symbol>_operator(energy_esys=False)` for each variable symbol; see
§8 for the naming inventory.

`energy_esys=True` returns the operator in the energy eigenbasis
(rotates by `eigensys()`). `energy_esys=(evals, evecs)` lets the
caller supply a pre-computed eigensystem.

The dynamic methods are bound via `types.MethodType` and are
therefore invisible to `mypy`, IDE autocomplete, and
`sphinx-autodoc`. For static-tooling-friendly access, use the typed
dispatcher:

```python
qubit.operator("n1")                      # equivalent to qubit.n1_operator()
qubit.operator("cosθ1", energy_esys=True) # rotated to energy eigenbasis
```

`Circuit.operator(name, *, energy_esys=...)` raises `AttributeError`
with a list of available names if ``name`` does not match any bound
operator method.

---

## 14. Test infrastructure

Three tiers of tests:

### 14.1 Unit tests

`scqubits/tests/test_circuit.py` (~800 lines, 26 tests). Coverage
includes representative qubit YAMLs (transmon, fluxonium, zero-pi,
cos2phi-qubit, DFC) with hand-checked eigenvalues; `ConfigureError`
behavior; named-constructor compatibility; the `make_branch`
`node_index_offset` parameter; randomized cross-checks of
`variable_transformation_matrix`'s row classification; the
`_AdjacencyIndex` cache; the DFS rewrite of `_find_path_to_root`
(asymptotic test that legacy O(n!) algorithm would not complete);
and the `_clear_unnecessary_attribs` registry behavior.

### 14.2 Characterization tests

`scqubits/tests/test_circuit_characterization.py` is a regression
net that pins the *numerical output* of the `circuit` module against
committed `.npy` golden fixtures. Four representative circuits
(transmon, fluxonium, zero-pi with HD, cos2phi). Each fixture pins
both `Circuit.hamiltonian()` and `Circuit.eigenvals(6)` at
`rtol=1e-10`, plus five lifecycle-dispatch tests (mutating fluxes /
cutoffs / params propagates correctly).

To regenerate after intentional numerical changes:

```bash
SCQUBITS_REGENERATE_GOLDENS=1 python -m pytest \
    --pyargs scqubits.tests.test_circuit_characterization
```

### 14.3 Full suite

`pytest --pyargs scqubits` runs 336+ tests. Several circuit-adjacent
tests live in `test_circuit_utils.py`, `test_circuit_sym_methods.py`,
`test_noise.py`, `test_circuit_plot.py`, and `test_explorer.py`.

---

## 15. Performance: hot paths

For an n-node chain circuit (1 JJ + n−1 capacitors), construction
time is bounded by:

| n | Time |
|---|---|
| 10 | ~1.4 s |
| 12 | ~1.4 s |
| 14 | ~1.9 s |
| 16 | ~2.4 s |

The asymptotic complexity is dominated by sympy's symbolic
manipulation (`subs`, `expand`, `cacheit`) inside
`generate_symbolic_hamiltonian` and `round_symbolic_expr`. The
graph-side algorithms have all been brought to polynomial complexity:

- `_find_path_to_root` and `_find_loop` use O(depth) DFS via
  `_AdjacencyIndex`.
- `_complete_basis_with_standard_vectors` enumerates distinct
  candidates via `itertools.combinations` over zero-positions —
  O(n²) candidates rather than O(n!) permutations.
- `_independent_modes` uses pure return values; rank tests are
  O(1) dict-keyed via the `_AdjacencyIndex` cache.
- `variable_transformation_matrix`'s row classification is a
  single-pass dict lookup rather than five chained O(n²)
  comprehensions.
- `round_symbolic_expr` uses one `xreplace(dict)` rather than per-Float
  `subs` calls; the same batched-substitution pattern is used in
  `_generate_hamiltonian_sym_for_numerics`.

If you need to speed up further, the remaining hot paths are inside
sympy: `Float.__round__` via mpmath `evalf`, sympy's `cacheit`
overhead, and `Expr.subs` walks. These would require either routing
through Python `float` (loses precision for `digits=16` use cases)
or restructuring the Hamiltonian-generation algorithm to do less
symbolic substitution overall.

---

## 16. Extension recipes

### 16.1 Adding a new branch type

Example: a kinetic-inductor branch type `KL` with parameter `EKL`.

1. Update `circuit_internals/input.py`: add `"KL"` to the
   `BRANCHES` pyparsing grammar; add a `BRANCH_KL` pattern; if the
   parameter dict shape differs from `L`, update `process_param`.
2. Update `circuit_internals/branch_metadata.py`: if `KL` is a
   multi-parameter junction-style type, update `_junction_order`. If
   it counts as a capacitor or JJ for cutoff purposes, update
   `_capacitance_variable_for_branch`.
3. Update `Branch._set_parameters` in `symbolic_circuit_graph.py`:
   add an `elif self.type == "KL":` clause assigning
   `{"EKL": parameters[0]}` (or whatever the right shape is).
4. Update `SymbolicCircuit.generate_symbolic_lagrangian`: add a
   branch-type clause that contributes the right Lagrangian term.
5. Decide whether `KL` participates in spanning-tree construction
   (yes for inductive-style branches) and update the relevant filters
   in `_independent_modes` callers (e.g. the
   `_canonical_modes_periodic_frozen_free_with_sigma` mode-list
   construction).
6. Add a unit test that builds a circuit with at least one `KL`
   branch and checks `var_categories` and `eigenvals`.

### 16.2 Adding a new noise channel

1. Add the method to `NoisyCircuit` in
   `circuit_internals/noise.py`:
   ```python
   def t1_my_channel(self, i=1, j=0, T=0.015, esys=None,
                     get_rate=False, **kwargs):
       """Docstring with the noise model formula."""
       # build matrix element <j|noise_op|i>, plug into standard T1 formula
   ```
2. Update `supported_noise_channels()` and / or
   `effective_noise_channels()` if the channel is conditional on
   topology.
3. Add a test in `test_noise.py` building a representative circuit
   and asserting the new method returns a positive finite number.

### 16.3 Adding a new operator basis

Suppose you want a third `ext_basis` choice, e.g. `"wavelet"`.

1. Add `"wavelet"` to the `ExtBasisChoice` Literal in `circuit.py`.
2. Add a `_build_extended_operators_wavelet(self)` method on
   `HamiltonianAssemblyMixin` returning the
   `{<symbol_name>_operator: factory_callable}` dict.
3. Add a clause to the dispatcher
   `_build_extended_operator_methods`:
   ```python
   if self.ext_basis == "wavelet":
       return self._build_extended_operators_wavelet()
   ```
4. If the basis needs new bare-operator primitives, add a new module
   under `circuit_internals/`, e.g. `wavelet_operators.py`.
5. Add a characterization-test fixture using the new basis and pin
   the matrices with `.npy` goldens.

### 16.4 Adding a new mixin

Decide which concern your mixin owns; the existing five
(`Lifecycle`, `SubsystemTree`, `HamiltonianAssembly`, `SymMethods`,
`Plot`) are deliberate splits. Then:

1. Create `circuit_internals/<concern>.py` with a class
   `<Concern>Mixin(ABC)`.
2. Add `if TYPE_CHECKING:` declarations for every cross-mixin
   attribute / method the mixin reads through `self`. Use the
   existing mixins as a model. mypy must be clean.
3. Add the mixin to the inheritance chain. There are two cases:
   - Logically part of the routines stack: add to
     `class CircuitRoutines(MyMixin, LifecycleMixin, ...)` in
     `routines.py`.
   - Parallel to `CircuitSymMethods` / `CircuitPlot`: add to
     `class CircuitABC(CircuitRoutines, ..., MyMixin)` in
     `circuit.py`.
4. Verify `Circuit.__mro__` and `Subsystem.__mro__` contain the new
   mixin in the expected position; check that no existing method
   resolution changes.
5. Run the characterization tests to confirm bit-equivalence.

---

## 17. Maintenance pitfalls

### 17.1 The deepcopy + remap pattern in `_spanning_tree`

`_spanning_tree` runs on `circ_copy = copy.deepcopy(self)`. Mutating
`circ_copy` is safe. The result lists hold references to the *copied*
`Branch` / `Node` instances; `_remap_spanning_tree_to_self`
substitutes each one with the matching instance on `self` (matched
via `==`, which is defined by `.index`). If you change `Node.__eq__` /
`Branch.__eq__`, the remap silently breaks.

### 17.2 `spanning_tree_dict` cache staleness

`_find_path_to_root` and `_find_loop` require the
`spanning_tree_dict` parameter explicitly (no `None` default).
Internal callers pass `self.spanning_tree_dict`. The `_AdjacencyIndex`
cache is keyed by `id(spanning_tree_dict)`; passing a different
(freshly constructed) dict transparently rebuilds the cache.

### 17.3 The recomputation contract: `SymbolicCircuit._STAGE2_ATTRIBUTES`

`Circuit._import_from_symbolic_circuit` iterates a tuple of attribute
names defined as a `ClassVar` on `SymbolicCircuit` itself
(``_STAGE2_ATTRIBUTES``). When you add a new attribute to
`SymbolicCircuit` that the numerical `Circuit` side needs to read,
add the name to that tuple — and **only** that tuple. Do not add it
to `_install_var_properties` (which has a different role: installing
per-variable property descriptors for cutoffs / fluxes / charges,
not bridging stage 1 → stage 2).

A regression test (`TestRecomputationContract`) asserts that for
every canonical YAML fixture, every name in `_STAGE2_ATTRIBUTES` is
present and equal on a fresh `Circuit` instance.

### 17.4 `_frozen` attribute

`Circuit` and `Subsystem` set `self._frozen = True` after
`_configure` to block ad-hoc attribute creation. New attributes must
be set in `__init__` / `_configure` / `_import_from_symbolic_circuit`
before `_frozen` flips. Setting them later raises in `__setattr__`.

### 17.5 The `from_file: bool` deprecation

`Circuit(yaml, from_file=True)` and `Circuit(yaml, from_file=False)`
still work but emit `DeprecationWarning`. The deprecation is
sentinel-protected — passing `from_file` *implicitly* (omitting the
arg) does NOT warn. New code should use `Circuit.from_yaml_file(path)`
or `Circuit.from_yaml_string(text)`.

### 17.6 The `eval`-based assembly is narrow

Two paths in `HamiltonianAssemblyMixin` use `eval()`:

- `_hamiltonian_for_purely_harmonic` — for circuits where every
  extended variable is in the harmonic basis and there are no JJ
  cosines, the Hamiltonian string is built and `eval()`-ed.
- `_purely_harmonic_operator_func_factory` — emits per-variable
  operator method bodies that include `eval()` of an operator
  string.

The *general* `_evaluate_hamiltonian` flow does NOT use `eval()`; it
walks the symbolic Hamiltonian term-by-term with `as_coefficients_dict`
and substitutes operator products via `_evaluate_symbolic_expr` (in
`sym_methods.py`). When debugging numerical-Hamiltonian issues, look
in the right place.

Pitfalls in the `eval()` paths:
- Symbol names that clash with Python builtins (`I` is reserved for
  the identity matrix; `_generate_hamiltonian_sym_for_numerics`
  enforces this).
- Numerical `Float` coefficients must be rounded (via
  `round_symbolic_expr`) before `eval()` runs; otherwise
  scientific-notation forms can mislead the parser.

### 17.7 Characterization-test goldens

Any change that intentionally modifies numerical output (e.g. a bug
fix that corrects an eigenvalue) must regenerate the goldens. The
procedure:

```bash
SCQUBITS_REGENERATE_GOLDENS=1 python -m pytest \
    --pyargs scqubits.tests.test_circuit_characterization
git diff scqubits/tests/characterization_goldens/  # binary diff is opaque
# explain why in the commit message
```

A goldens-regeneration commit should always be its own commit and
explain *why* the numerics changed.

### 17.8 `_clear_unnecessary_attribs` is registry-driven

`Circuit._clear_unnecessary_attribs` (in `circuit.py`, ~L896) is
called between reconfiguration steps to drop stale per-variable
properties (cutoffs, external fluxes, offset/free charges) from
`self.__dict__`. It walks `self._dynamic_var_attribs` — a
`set[str]` populated by `_install_var_properties` as it installs
each property — and deletes the underlying `_<name>` value for
any registered name whose corresponding variable category has
been removed by the current reconfiguration.

When adding a new per-variable property kind, the only place
that needs updating is `_install_var_properties`: register the
attribute name into `self._dynamic_var_attribs` after calling
`_make_property`. `_clear_unnecessary_attribs` picks it up
automatically.

Symbolic-param names (e.g. ``EJ``, ``EC``) are deliberately *not*
registered — those attribs survive reconfiguration and are
managed separately via `self.symbolic_params`.

---

## 18. Where to look for what

| If you want to change… | Edit… |
|---|---|
| YAML grammar | `circuit_internals/input.py` |
| YAML programmatic assembly | `circuit_internals/yaml_assembly.py` |
| Graph algorithms (spanning tree, transformation) | `symbolic_circuit_graph.py` |
| Symbolic Lagrangian / Hamiltonian construction | `symbolic_circuit.py` |
| LaTeX rendering of symbolic state | `circuit_internals/sym_methods.py` |
| Symbolic→numeric Hamiltonian bridge | `circuit_internals/sym_methods.py` (`_generate_hamiltonian_sym_for_numerics`) |
| Per-variable operator method generation | `circuit_internals/hamiltonian_assembly.py` (`_set_operators`, `_build_*`) |
| Operator-method factory closures | `circuit_internals/operator_factories.py` |
| JJ `cos(φ)` → matrix evaluation | `circuit_internals/junction_assembly.py` (`evaluate_matrix_cosine_terms`, `extract_junction_phase`, `assemble_cos_term`, …); `hamiltonian_assembly.py` exposes thin instance-method wrappers |
| HD subsystem construction | `circuit_internals/subsystem_tree.py` |
| Parameter sync / dispatch / WatchedProperty | `circuit_internals/lifecycle.py` and `scqubits/core/descriptors.py` |
| Hilbert-space basics (kron, identity, hilbertdim) | `circuit_internals/routines.py` |
| Wave-function / potential plotting | `circuit_internals/plotting.py` |
| Noise channels | `circuit_internals/noise.py` |
| Bare per-basis operators (charge / discretized-φ) | `circuit_internals/charge_basis_operators.py`, `circuit_internals/discretized_phi_operators.py` |
| Sparse cos/sin of a diagonal matrix | `circuit_internals/matrix_helpers.py` |
| Sawtooth-junction physics | `circuit_internals/sawtooth.py` |
| Branch-type predicates | `circuit_internals/branch_metadata.py` |
| Sympy expression rounding | `circuit_internals/sympy_helpers.py` |

For tests:

| Test concern | File |
|---|---|
| End-to-end Circuit / Subsystem | `scqubits/tests/test_circuit.py` |
| Numerical regression net | `scqubits/tests/test_circuit_characterization.py` |
| Bare circuit-utility helpers | `scqubits/tests/test_circuit_utils.py` |
| `CircuitSymMethods` algebra | `scqubits/tests/test_circuit_sym_methods.py` |
| `NoisyCircuit` channels | `scqubits/tests/test_noise.py` |
| `CircuitPlot` rendering | `scqubits/tests/test_circuit_plot.py` |
