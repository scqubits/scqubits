# QuTiP usage in scqubits — report

Inventory for this repository (Python, packaging, markdown). No Sphinx `.rst` docs were found in the repo; `README.md` does not mention QuTiP by name.

---

## 1. Packaging and project metadata

| Location | Role |
|----------|------|
| `pyproject.toml` | Core dependency: `qutip>=4.3.1` |
| `meta.yaml` | Conda recipe: `qutip >=4.3.1` |
| `AGENTS.md` | Mentions `Qobj`, `qutip_cuquantum.CuQobjEvo`, optional `qutip-cuquantum`, dependency table |

---

## 2. Python modules that import or reference QuTiP

### 2.1 Core (`scqubits/core/`)

| File | Import style | Primary QuTiP role |
|------|--------------|-------------------|
| `hilbert_space.py` | `import qutip as qt` | Composite Hilbert space: Hamiltonians and operators as `qt.Qobj`; `qt.tensor`, `qt.qzero`, `qt.destroy`, `qt.states.basis`, `qt.Qobj`; `qt.settings.core["default_dtype"]` (cuDensity); optional `qutip_cuquantum` + `CuQuantumBackend`; `InteractionTermStr` / `parse_qutip_functions`; `eigenvals` / `eigensys` delegate to `Qobj.eigenenergies` / `eigenstates` when no custom diagonalizer |
| `diag.py` | `from qutip import Qobj, QobjEvo`, `import qutip as q` | All `DIAG_METHODS`: accept `Qobj`, cast via `q` version / data layer; `_convert_evecs_to_qobjs`; `esys_cuquantum` uses `qutip_cuquantum` + `QobjEvo` |
| `spec_lookup.py` | `import qutip as qt`, `from qutip import Qobj` | Dressed-state lookup: `qt.basis`, `qt.tensor`, `qt.qdiags`; states as `qt.Qobj` |
| `qubit_base.py` | `import qutip as qt` | Types and operator APIs using `qt.Qobj` in signatures (matrix elements / dispatch) |
| `noise.py` | `import qutip as qt` | Noise APIs accepting `qt.Qobj` where documented |
| `param_sweep.py` | `from qutip import Qobj` | Parameter sweeps: operators as `Qobj` in APIs |
| `sweeps.py` | `from qutip import Qobj` | Sweep helpers typed with `Qobj` |
| `storage.py` | (no direct `import qutip`) | Uses `QutipEigenstates` from `fileio_qutip`; docstrings refer to qutip kets |
| `circuit_routines.py` | `import qutip as qt` | Circuit numerics: `qt.identity`, `qt.tensor`, `qt.qdiags`, `qt.Qobj` for operators |
| `circuit_sym_methods.py` | `import qutip as qt` | `hamiltonian_for_qutip_dynamics`, `_qutip_parameter_function_factory` for QuTiP time evolution |
| `circuit_noise.py` | `import qutip as qt` | Noise paths checking / using `qt.Qobj` |

Representative **HilbertSpace**-related QuTiP entry points (not exhaustive): `hamiltonian`, `dressed_operator`, `diag_hamiltonian`, `hubbard_operator`, `annihilate`, `eigenvals`, `eigensys`, and code paths gated on `qt.settings.core["default_dtype"]`.

Representative **diag** entry points: `_cast_matrix`, `_convert_evecs_to_qobjs`, and every registered `evals_*` / `esys_*` function in `DIAG_METHODS`, plus `esys_cuquantum` / `evals_cuquantum`.

---

### 2.2 I/O (`scqubits/io_utils/`)

| File | Import style | Role |
|------|--------------|------|
| `fileio_qutip.py` | `import qutip as qt` | `QutipEigenstates`: serialize/deserialize eigenstates as `qt.Qobj` (dims handling for QuTiP 5+) |

---

### 2.3 Utils (`scqubits/utils/`)

| File | Import style | Notable functions / APIs |
|------|--------------|---------------------------|
| `spectrum_utils.py` | `import qutip as qt`, `from qutip import Qobj` | Matrix elements and `identity_wrap`-style helpers using `qt.Qobj`; `convert_evecs_to_ndarray`; overlap / dispatch utilities; `qt.operators.qeye` in composite identity construction |
| `misc.py` | `import qutip as qt` | `qt_ket_to_ndarray`, `Qobj_to_scipy_csc_matrix` (QuTiP version branches); `about` output includes `qt.__version__` |
| `cuquantum_runtime.py` | (no `import qutip`) | Docstring only: “qutip-cuquantum” (no direct QuTiP import) |

---

### 2.4 UI (`scqubits/ui/`)

| File | Import style | Role |
|------|--------------|------|
| `hspace_widget.py` | `from qutip import Qobj` | `isinstance(..., Qobj)` for operator typing in the widget |

---

### 2.5 Tests (`scqubits/tests/`)

| File | Import style | Role |
|------|--------------|------|
| `test_hilbertspace.py` | `import qutip as qt` | Hilbert-space tests using QuTiP |
| `test_circuit.py` | `import qutip as qt` | Includes `test_qutip_dynamics`; `hamiltonian_for_qutip_dynamics`; reference data for qutip evolution |

---

## 3. Related optional stack (not `qutip` package itself)

These use **`qutip_cuquantum`** (and cuQuantum), not the core `qutip` import line, but they are part of the QuTiP ecosystem integration:

- `scqubits/core/hilbert_space.py` — `import qutip_cuquantum as qcu` in cuQuantum branches
- `scqubits/core/diag.py` — `import qutip_cuquantum as qcu` in `esys_cuquantum`
- `scqubits/core/spec_lookup.py` — commented references to `qutip_cuquantum`

---

## 4. QuTiP APIs touched in code (summary)

- **Types / objects:** `Qobj`, `QobjEvo` (diag / cuQuantum path)
- **Algebra & states:** `tensor`, `identity`, `qeye` (via `qt.operators.qeye`), `qzero`, `qdiags`, `destroy`, `basis` / `states.basis`, `.dag()`
- **Settings:** `qt.settings.core["default_dtype"]` (cuDensity path)
- **Solvers (delegation):** `Qobj.eigenenergies`, `Qobj.eigenstates`
- **Version / internals:** `q.__version__`, `q.core.data.dense.Dense` (diag casting)
- **I/O:** constructing `qt.Qobj` from arrays + `dims` in `QutipEigenstates.deserialize`

---

## 5. Files with no direct `qutip` string but QuTiP-coupled

Any module importing **`QutipEigenstates`** from `scqubits.io_utils.fileio_qutip` (e.g. `storage`, `spectrum_utils`, `diag`, `hilbert_space`) is part of the QuTiP eigenstate pipeline even if the file does not `import qutip`.

---

## 6. Counts

- **Python files** with a `qutip` substring match: **17** (under `scqubits/`)
- **Markdown** with `qutip`: **`AGENTS.md`** only (among project docs searched)
- **Config** with `qutip`: **`pyproject.toml`**, **`meta.yaml`**

---

## 7. Standard QuTiP vs cuQuantum/cuDensity — incompatibility scope

When QuTiP’s global dtype/backend is set to **cuDensity** (or Hamiltonians/eigenstates are produced via **qutip-cuquantum**), core QuTiP and extensions may use GPU-backed data layers. Code that assumes **CPU `Dense` / CSR** behavior can then fail or silently diverge.

The following are **excluded** from the “potential incompatibility” list below (per project rules):

| # | Exclusion | Examples in this repo |
|---|-----------|----------------------|
| 1 | **No runtime QuTiP call** — mentions in docstrings, comments, or type-only prose | Parameter docs saying “qutip.Qobj”; narrative in `AGENTS.md` |
| 2 | **`qutip-cuquantum` API** (not standard `qutip`) | `import qutip_cuquantum as qcu`, `CuQuantumBackend`, `CuQobjEvo`, `CuState`, etc. |
| 3 | **Type checking only** | `isinstance(x, Qobj)`, `Union[..., Qobj]` annotations, `callable(operator)` guards without subsequent QuTiP ops on that value in the same block |
| 4 | **`QutipEigenstates` machinery** | `QutipEigenstates` subclass; `_convert_evecs_to_qobjs(..., wrap=True)`; `fileio_qutip` serialize/deserialize; `get_eigenstate_index_maxoverlap` (container type + overlap loop tied to that pipeline) |

Everything in **§8** is a **standard `qutip` / `Qobj` runtime call** (or direct use of `q.core.*` / `q.__version__` for dispatch) that may still run **after** or **beside** cuQuantum-backed objects, unless you prove those code paths are unreachable under cuDensity.

---

## 8. Potential incompatibility sites (after exclusions)

**Mechanisms** that tend to break across dtypes: branching on `matrix.dtype == q.core.data.dense.Dense`; `.full()` / `.data` / `.to("CSR").data_as()`; `Qobj_to_scipy_csc_matrix` / `qt_ket_to_ndarray`; `Qobj.transform`, `.overlap`, `.norm`, `**` on `Qobj`; factory ops `qt.tensor`, `qt.identity`, `qt.qzero`, `qt.destroy`, `qt.basis`, `qt.qdiags`, `qt.operators.qeye`; `QobjEvo` wrapping.

| Location | Code / API (standard QuTiP) | Why it may conflict with cuDensity / GPU-backed `Qobj` |
|----------|----------------------------|--------------------------------------------------------|
| `scqubits/core/diag.py` | `_cast_matrix`: `q.__version__`, `matrix.dtype == q.core.data.dense.Dense`, `.full()`, `.to("CSR").data_as()` | Assumes **Dense vs CSR** data layer; cuDensity/Cu-backed types may not follow this branch matrix or may require different extraction. |
| `scqubits/core/diag.py` | `_convert_evecs_to_qobjs`: `Qobj(...)`, `v.norm()` (wrap into `QutipEigenstates` excluded, but construction/norm are not) | Builds kets and normalizes via QuTiP; may not match expectations if upstream evecs are GPU-only or non-Dense. |
| `scqubits/core/diag.py` | `esys_cuquantum`: `q.QobjEvo(matrix)` | **Standard** `QobjEvo` bridges Hamiltonian into cuQuantum; behavior depends on `matrix`’s dtype when default is cuDensity. |
| `scqubits/core/diag.py` | `esys_cuquantum`: `Qobj(...)` around `to_array()` output | Standard `Qobj` constructor from numpy; usually safe if array is host memory; dims/type must match QuTiP 5 expectations. |
| `scqubits/core/hilbert_space.py` | `hamiltonian` path: `qt.qzero`, `qt.Qobj`, `isinstance(term, qt.Qobj)`, `qt.tensor`-style assembly | Composite Hamiltonian built from QuTiP algebra; mixed dtypes (some terms cu-backed) can break tensor/add rules. |
| `scqubits/core/hilbert_space.py` | `diag_hamiltonian`, `diag_operator`, `hubbard_operator`, `annihilate`: `qt.Qobj(np...)`, `qt.states.basis`, `.dag()`, `qt.destroy` | Standard constructors and ladder ops; output dtype follows global QuTiP settings. |
| `scqubits/core/hilbert_space.py` | `op_in_dressed_eigenbasis`: `spec_utils.identity_wrap`, `id_wrapped_op.transform(dressed_evecs)`, `utils.Qobj_to_scipy_csc_matrix`, `qt.Qobj(...)` | **`.transform`** and CSC conversion assume QuTiP can move data to a scipy-friendly form; dressed evecs from cu solver may be problematic. |
| `scqubits/core/hilbert_space.py` | `standardize_eigenvector_phases`: `Qobj_to_scipy_csc_matrix(evec)` | Same as above on stored eigenvectors. |
| `scqubits/core/spec_lookup.py` | `qt.basis`, `qt.tensor`, `qt.qdiags` | Basis/tensor/diag factories create `Qobj`s; downstream arithmetic with cu-backed states may mismatch. |
| `scqubits/utils/misc.py` | `qt_ket_to_ndarray`: `.data.as_ndarray()` / legacy `.data.toarray()` | Peeks inside `Qobj` data representation; cuDensity may not expose the same interface. |
| `scqubits/utils/misc.py` | `Qobj_to_scipy_csc_matrix`: `.to("csr").data.as_scipy().tocsc()` / legacy `.data.tocsc()` | **High risk**: forces CSR/scipy path; common failure point for non-CPU dtypes. |
| `scqubits/utils/spectrum_utils.py` | `matrix_element`: `operator.data`, `Qobj_to_scipy_csc_matrix` on kets | Direct `.data` access + scipy conversion. |
| `scqubits/utils/spectrum_utils.py` | `get_matrixelement_table`: `Qobj_to_scipy_csc_matrix` | Same. |
| `scqubits/utils/spectrum_utils.py` | `convert_evecs_to_ndarray`: `eigenstate.full()` | Assumes `.full()` is valid and CPU-shaped. |
| `scqubits/utils/spectrum_utils.py` | `convert_matrix_to_qobj`, `convert_opstring_to_qobj`, `convert_operator_to_qobj`: `qt.Qobj(...)` | New `Qobj` from arrays; usually Dense by default unless global dtype overrides. |
| `scqubits/utils/spectrum_utils.py` | `identity_wrap`: `qt.operators.qeye`, `qt.tensor` | Tensor product of identities + subsystem op; dtype propagation across factors. |
| `scqubits/core/noise.py` | After `isinstance(noise_op, qt.Qobj)`: `Qobj_to_scipy_csc_matrix(noise_op)` | Standard QuTiP→scipy path on user-provided `Qobj`; risky if noise op is cu-backed. |
| `scqubits/core/circuit_routines.py` | `qt.identity`, `qt.tensor`, `qt.qdiags`, `qt.Qobj(...)`, `Qobj ** int` | Circuit operator assembly; large tensors with cuDensity may hit unsupported ops. |
| `scqubits/core/circuit_sym_methods.py` | `_evaluate_symbolic_expr` → `qt.Qobj` numerics; `hamiltonian_for_qutip_dynamics` output list for `mesolve` | Intended for **standard** `qutip.mesolve` (see tests); not cuQuantum-specific. |
| `scqubits/tests/test_circuit.py` | `qt.Qobj`, `qt.mesolve` | Regression / reference only; still documents expected **CPU** QuTiP dynamics stack. |

**Not listed as incompatibility hotspots** (per §7): `qubit_base.py`, `param_sweep.py`, `sweeps.py`, `circuit_noise.py`, `ui/hspace_widget.py` — in the current tree they either only **annotate** / **`isinstance`** `Qobj` or delegate to helpers already tabulated above. **`misc.about`** only reads `qt.__version__` (no numerical dtype coupling).

---

*This file lives at the repository root (`qutip-usage-report.md`). Paths in tables are relative to the repository root unless noted.*
