# scqubits — Claude Code Context

## Project overview
scqubits: open-source Python library for simulating superconducting qubits.
Lead developer: Jens Koch (Northwestern). Co-developer: Peter Groszkowski.
Published via conda-forge and PyPI. Docs: https://scqubits.readthedocs.io

## Repository layout
Root: C:\Users\drjen\coding\scqubits\
scqubits/core/       - qubit classes, Hilbert space, operators, sweeps
scqubits/explorer/   - interactive explorer widget
scqubits/io_utils/   - serialization (HDF5, CSV)
scqubits/ui/         - GUI widgets
scqubits/utils/      - plotting, spectrum utilities
scqubits/tests/      - pytest suite

Key files in core/:
  qubit_base.py, transmon.py, fluxonium.py, flux_qubit.py
  zeropi.py, zeropi_full.py, cos2phi_qubit.py, oscillator.py
  hilbert_space.py, param_sweep.py, noise.py, diag.py
  circuit.py, symbolic_circuit.py, central_dispatch.py

Related repo: C:\Users\drjen\coding\scqubits-doc\

## Code standards
- PEP 8, NumPy docstrings, type hints on all public methods
- Python 3.9-3.12

## Critical rules
- Do NOT change public API signatures without flagging it
- Do NOT refactor working numerical code without explicit instruction
- Do NOT run tests automatically unless asked
- central_dispatch.py is delicate - changes affect the whole library
- Never edit scqubits-doc/docs/build/ or _autosummary/ or _generated/

## Doc update workflow
1. Update docstrings in Python source
2. Update narrative RST in scqubits-doc/docs/source/guide/
3. Do NOT re-execute notebooks unless asked

## Session discipline
- One task per session: code OR docs, not both
- Use /compact before switching major tasks

 # scqubits type-annotation project conventions

   ## Goal
   Add and correct type annotations across scqubits, verified by mypy.
   For this project, assume that we will support Python 3.10 and upwards
   (dropping 3.9 from being supported).

   ## Scope rules
   - **Never change runtime behavior.** Annotations only. No refactors, no
     renames, no logic edits. If you believe a logic change is needed,
     stop and ask.
   - **One file per turn** unless explicitly told otherwise.
   - **Diff-first.** Show the proposed diff before writing it. After I
     approve, apply with the edit tool.

   ## Style
   - `from __future__ import annotations` at the top of every file you touch.
   - Prefer PEP 604 unions (`int | None`) over `Optional[int]`.
   - Prefer `collections.abc` ABCs (`Iterable`, `Mapping`, `Callable`) over
     `typing` equivalents.
   - For NumPy:
       - Use `numpy.typing.NDArray[np.float64]` etc., not bare `np.ndarray`.
       - When a function genuinely accepts scalar-or-array, use
         `npt.ArrayLike` for inputs and a precise `NDArray[...]` for outputs.
   - For qutip: use `qutip.Qobj` directly. If qutip stubs are missing, add
     `# type: ignore[attr-defined]` only on the specific failing line.
   - For SciPy sparse: `scipy.sparse.csc_matrix` etc., not the abstract base.

   ## Sources of truth, in priority order
   1. Existing annotations already in the file (don't override without reason).
   2. Docstrings (NumPy-style — read the `Parameters` and `Returns` sections).
   3. The MonkeyType database at `monkeytype.sqlite3` in the repo root.
      Use `monkeytype stub <module>` to see observed signatures.
   4. Tests under `scqubits/tests/` that exercise the function.
   5. Your own inference from the function body.
   Never invent a type from priors alone if (1)–(4) disagree with you.

   ## Forbidden
   - `Any` as a way to silence mypy. If you reach for `Any`, stop and ask.
   - `# type: ignore` without a specific error code in brackets.
   - Removing existing `# type: ignore` comments without checking they're
     actually unused (`warn_unused_ignores` is on).

   ## Verification
   After every edit, run:
       mypy --config-file mypy.ini <path/to/edited/file>
   and report the error delta vs. before the edit.

## Environment

This project uses a Python 3.10 conda environment already active in this terminal 

When suggesting new dependencies:
- Check conda-forge first: `mamba search -c conda-forge <package>`.
- If available on conda-forge, propose `mamba install -c conda-forge <package>`
  and update `environment.yml`.
- Only fall back to `pip install` if the package is not on conda-forge.
  In that case, add it under the `pip:` block in `environment.yml`.
- Never run `pip install` for something that conda-forge already provides.