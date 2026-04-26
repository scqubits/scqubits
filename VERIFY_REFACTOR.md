# How to verify the `circuit`-module refactor preserves behavior

This document lets you independently confirm that the refactor on
`refactor/circuit-module-clean-code` does not change observable
behavior of the `Circuit` module relative to `main`. It runs two
checks; both should pass.

The procedure is self-contained: run the shell blocks in order from the
repo root.  No edits to the repo are made (everything is reverted at
the end).  Total runtime ~5 minutes.

## Why two checks

The refactor branch modifies a handful of test files. Most of those
modifications are unavoidable (e.g. private symbols moved to a new
module, so an import line had to change), but they break the cleanest
possible assurance ("if `pytest` passes, nothing changed"). The two
checks below close the loop:

1. **Check A — Main's tests against HEAD's source.** Take the test
   files exactly as they exist on `main` (with two surgical import-line
   patches forced by privacy / shim cleanup), run them against the
   refactored code. If every assertion main wrote is satisfied by the
   refactored code, observable behavior is preserved for every input
   main probed.

2. **Check B — Independent golden capture on main, byte-compare to
   HEAD.** Generate `Circuit.hamiltonian()` and `Circuit.eigenvals()`
   for four representative circuits on a clean `main` checkout; diff
   those byte-for-byte against the goldens committed on the refactor
   branch. This protects against the concern that the goldens
   themselves were captured mid-refactor.

Together these isolate "did the refactor change observable behavior"
from "did Claude change tests."

## Prerequisites

- Clean working tree (`git status` shows nothing modified).
- Branch `refactor/circuit-module-clean-code` checked out and pushed.
- `main` available locally (verify with `git rev-parse main`).
- `scqubits` installed in editable mode in your active environment
  (recommended: the same conda env you've been using to run pytest).
- About 4 GB free in your system temp directory (golden Hamiltonians
  for the small fixtures total ~1.8 MB; we just need scratch space).

```bash
git status                          # should be clean
git branch --show-current           # should be refactor/circuit-module-clean-code
python -c "import scqubits; print(scqubits.__file__)"   # should resolve to your local checkout
```

## Check A — Main's tests against HEAD's source

### A1. Save HEAD's tests aside, then check out `main`'s tests in place

```bash
# Stash HEAD's test files so we can restore them later. The new
# characterization-test file and its goldens don't exist on main, so
# they'll be left in place by `git checkout` (this is fine).
git checkout main -- scqubits/tests/

# What you should see modified:
git status --short scqubits/tests/
```

Expected: 4 files modified
(`test_circuit.py`, `test_circuit_sym_methods.py`,
`test_circuit_utils.py`, `test_diag.py`).

### A2. Apply the two forced import-line patches

The reason: two of main's tests reach into private symbols that the
refactor moved (`_capacitance_variable_for_branch`,
`CircuitSymMethods`).  These imports cannot be left as-is — they would
fail at collection time. The patches change *only* the import lines;
every test body and every assertion remains byte-identical to main.

Save this script to a temp file, run it once:

```bash
python << 'PYEOF'
"""Patch only the import lines in main's tests so they can collect."""
import re
from pathlib import Path

t1 = Path("scqubits/tests/test_circuit_utils.py")
src = t1.read_text(encoding="utf-8")
new_imports = """from scqubits.core.circuit_internals.branch_metadata import (
    _capacitance_variable_for_branch,
    _junction_order,
)
from scqubits.core.circuit_internals.charge_basis_operators import (
    _cos_theta,
    _exp_i_theta_operator,
    _exp_i_theta_operator_conjugate,
    _identity_theta,
    _n_theta_operator,
    _sin_theta,
)
from scqubits.core.circuit_internals.discretized_phi_operators import (
    _cos_phi,
    _i_d_dphi_operator,
    _identity_phi,
    _phi_operator,
    _sin_phi,
)
from scqubits.core.circuit_internals.matrix_helpers import (
    _cos_dia,
    _cos_dia_dense,
    _sin_dia,
    _sin_dia_dense,
    matrix_power_sparse,
)
from scqubits.core.circuit_internals.sympy_helpers import (
    _generate_symbols_list,
    is_potential_term,
    round_symbolic_expr,
)
from scqubits.core.circuit_utils import (
    example_circuit,
    get_trailing_number,
    sawtooth_operator,
    sawtooth_potential,
    truncation_template,
)"""
src2 = re.sub(
    r"from scqubits\.core\.circuit_utils import \([^)]+\)",
    new_imports, src, count=1,
)
assert src2 != src, "test_circuit_utils.py: import block not found"
t1.write_text(src2, encoding="utf-8")

t2 = Path("scqubits/tests/test_circuit_sym_methods.py")
src = t2.read_text(encoding="utf-8")
src2 = src.replace(
    "from scqubits.core.circuit_sym_methods import CircuitSymMethods",
    "from scqubits.core.circuit_internals.sym_methods import CircuitSymMethods",
)
assert src2 != src, "test_circuit_sym_methods.py: import line not found"
t2.write_text(src2, encoding="utf-8")
print("two import lines patched; test bodies unchanged")
PYEOF
```

### A3. Run pytest

```bash
python -m pytest --pyargs scqubits -q
```

Expected outcome:

```
318 passed, 14 skipped, ... in ~4 minutes
```

If this number is 318 (or higher — the characterization tests sitting
on disk add 13 to whatever main's count is), Check A passes. If
*anything* fails or errors, that's a real signal — read the failure.

### A4. Restore HEAD's tests

Don't skip this step.

```bash
git checkout HEAD -- scqubits/tests/
git status --short scqubits/tests/    # should be empty
```

## Check B — Independent golden capture on main, byte-compare to HEAD

### B1. Save HEAD's goldens to a temp directory

We want to compare them against fresh main-baseline goldens later.

```bash
python << 'PYEOF'
import os, shutil, tempfile
from pathlib import Path

src = Path("scqubits/tests/characterization_goldens")
dst = Path(tempfile.gettempdir()) / "head_goldens"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"saved {len(list(dst.glob('*.npy')))} HEAD goldens to {dst}")
PYEOF
```

### B2. Save the characterization-test file aside (it doesn't exist on `main`)

```bash
python << 'PYEOF'
import shutil, tempfile
from pathlib import Path
src = Path("scqubits/tests/test_circuit_characterization.py")
dst = Path(tempfile.gettempdir()) / "test_circuit_characterization.py"
shutil.copy(src, dst)
print(f"saved {src} to {dst}")
PYEOF
```

### B3. Switch to `main`

Stash anything that might block, then check out `main`.

```bash
git stash push -u -m "verify-refactor-temp" 2>/dev/null
git checkout main
git log --oneline -1                  # confirm you're on main
```

### B4. Drop the characterization test back in, patch it for main's API,
generate goldens against main's source

`main` doesn't have `Circuit.from_yaml_string`, so the test fixtures
must use the legacy `Circuit(yaml, from_file=False)` form on this side.

```bash
python << 'PYEOF'
import os, shutil, tempfile
from pathlib import Path

# Copy the characterization test back into the (now-main) tree.
src = Path(tempfile.gettempdir()) / "test_circuit_characterization.py"
dst = Path("scqubits/tests/test_circuit_characterization.py")
shutil.copy(src, dst)

# Empty goldens dir so the test creates fresh main-baseline goldens.
goldens = Path("scqubits/tests/characterization_goldens")
if goldens.exists():
    shutil.rmtree(goldens)
goldens.mkdir(parents=True)

# Patch the four ``Circuit.from_yaml_string(...)`` calls in the test
# fixtures to use main's legacy ``Circuit(..., from_file=False)`` API.
text = dst.read_text(encoding="utf-8")
for yaml_var in ("TRANSMON_YAML", "FLUXONIUM_YAML", "ZERO_PI_YAML", "COS2PHI_YAML"):
    old = f'scq.Circuit.from_yaml_string({yaml_var}, ext_basis="discretized")'
    new = f'scq.Circuit({yaml_var}, from_file=False, ext_basis="discretized")'
    assert old in text, f"could not find {old}"
    text = text.replace(old, new)
dst.write_text(text, encoding="utf-8")
print("characterization test ported to main API; goldens dir empty")
PYEOF

# Generate goldens — this writes 8 .npy files into characterization_goldens/
SCQUBITS_REGENERATE_GOLDENS=1 python -m pytest \
    --pyargs scqubits.tests.test_circuit_characterization -q
```

Expected outcome:

```
5 passed, 8 skipped in ~10 seconds
```

(The 8 "skipped" are the 4 hamiltonian + 4 eigenvals tests that
generate-and-skip on first run; the 5 "passed" are the
lifecycle-dispatch tests.)

### B5. Save the main-baseline goldens, then compare to HEAD's

```bash
python << 'PYEOF'
import shutil, tempfile
from pathlib import Path
src = Path("scqubits/tests/characterization_goldens")
dst = Path(tempfile.gettempdir()) / "main_goldens"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"saved {len(list(dst.glob('*.npy')))} main-baseline goldens to {dst}")
PYEOF

python << 'PYEOF'
"""Byte-compare main-baseline goldens to HEAD-committed goldens."""
import tempfile
import numpy as np
from pathlib import Path

main_dir = Path(tempfile.gettempdir()) / "main_goldens"
head_dir = Path(tempfile.gettempdir()) / "head_goldens"

mismatches = 0
total = 0
for main_npy in sorted(main_dir.glob("*.npy")):
    head_npy = head_dir / main_npy.name
    a = np.load(main_npy)
    b = np.load(head_npy)
    total += 1
    if a.shape != b.shape:
        print(f"  MISMATCH {main_npy.name}: shape {a.shape} vs {b.shape}")
        mismatches += 1
        continue
    close = np.allclose(a, b, rtol=1e-10, atol=1e-15)
    denom = np.maximum(np.abs(b), 1e-300)
    max_rel = float(np.max(np.abs(a - b) / denom))
    max_abs = float(np.max(np.abs(a - b)))
    if close:
        print(f"  match    {main_npy.name}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
    else:
        print(f"  MISMATCH {main_npy.name}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
        mismatches += 1

print(f"\nResult: {total - mismatches}/{total} goldens match (rtol=1e-10)")
PYEOF
```

Expected output:

```
  match    cos2phi__evals.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    cos2phi__hamiltonian.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    fluxonium__evals.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    fluxonium__hamiltonian.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    transmon__evals.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    transmon__hamiltonian.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    zero_pi_hd__evals.npy: max_abs=0.00e+00, max_rel=0.00e+00
  match    zero_pi_hd__hamiltonian.npy: max_abs=0.00e+00, max_rel=0.00e+00

Result: 8/8 goldens match (rtol=1e-10)
```

Bit-identical (max deviations exactly 0.0) is the expected outcome.
Anything other than 0.0 — even a tiny non-zero `max_rel` — would
indicate that the refactor changed numerics at floating-point
precision; that's worth investigating. The exact-zero result rules
out floating-point reordering as well as algorithmic change.

### B6. Clean up and restore the refactor branch

```bash
# Remove the test file and goldens that were temporarily put back
# (they're not on main; we don't want to commit them by accident).
rm -rf scqubits/tests/characterization_goldens
rm -f  scqubits/tests/test_circuit_characterization.py

# Switch back to the refactor branch
git checkout refactor/circuit-module-clean-code

# Confirm clean state
git log --oneline -1
git status --short        # should be empty
```

## What this proves — and what it doesn't

**Proves:**

- The refactor does not change `Circuit.hamiltonian()` or
  `Circuit.eigenvals(6)` for the four representative circuit fixtures
  (transmon, fluxonium, hierarchically-diagonalized zero-pi, cos2phi)
  at any precision down to the floating-point bit.
- The refactor does not break any of the 309 assertions written in
  main's existing test suite. Every one of them passes against the
  refactored code with only the two forced import-line patches —
  patches whose minimality you can verify by `git diff` between the
  patched files and `main`'s versions.

**Does not prove:**

- Behavior on inputs not covered by main's test suite or by the four
  characterization fixtures. That is an inherent limit of any
  test-based verification — un-tested inputs cannot be proven to be
  preserved.
- Performance. The experiment measures only output equivalence.
- Absence of edge-case behavioral differences in code paths that
  main's tests don't exercise (specific noise-channel combinations,
  certain GUI workflows, parameter-sweep corner cases).

## What if a check fails

If Check A fails:

- Identify which test failed. Compare its body in `git show
  main:scqubits/tests/<file>` to what's currently on disk.  If the
  bodies match, the failure is a real refactor regression, not a
  test-modification artifact.
- Open `git log refactor/circuit-module-clean-code -- <test_file>`
  and walk back through the refactor commits until the test starts
  passing again. The first commit at which it fails is the
  introducing commit.

If Check B fails (any non-zero deviation):

- The refactor changed numerics. Even a small non-zero `max_rel`
  indicates a floating-point ordering change.  Look at commits that
  touched `_evaluate_matrix_cosine_terms`,
  `_hamiltonian_for_harmonic_extended_vars`, or any of the helpers
  feeding them — these are the most likely sources of FP-order shifts.
- The two extractions to look at first:
  `b24cf618` (`HamiltonianAssemblyMixin`) and `033565a6`
  (`SubsystemTreeMixin`).
