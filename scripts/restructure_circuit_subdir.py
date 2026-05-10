#!/usr/bin/env python3
"""One-shot migration: move circuit *support* files into
`scqubits/core/circuit_internals/` (Option B).

Public entry points stay at `scqubits/core/`:
    - circuit.py              (Circuit, Subsystem, CircuitABC, ConfigureError)
    - symbolic_circuit.py     (SymbolicCircuit)
    - symbolic_circuit_graph.py (Node, Branch, Coupler, SymbolicCircuitGraph)

Support files move to `scqubits/core/circuit_internals/`. Long-standing public
modules (the six legacy `circuit_*.py` files plus `symbolic_circuit.py`'s
position is unchanged) get backward-compatibility shims at their old
paths. The eight modules introduced in B1 (branch_metadata,
charge_basis_operators, discretized_phi_operators, matrix_helpers,
operator_factories, sympy_helpers, sawtooth, circuit_yaml_assembly) had
no downstream users before today and get NO shims.

Run from the repo root:
    python scripts/restructure_circuit_subdir.py
Then verify:
    python -m pytest --collect-only --pyargs scqubits
    python -m mypy scqubits/
    python -m pytest --pyargs scqubits.tests.test_circuit \
        scqubits.tests.test_circuit_utils \
        scqubits.tests.test_circuit_sym_methods
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SQ = REPO_ROOT / "scqubits"

# (old_path_relative_to_scqubits, new_path_relative_to_scqubits)
MOVES = [
    # Long-standing support files (B0 / pre-refactor): get shims at old paths.
    ("core/circuit_input.py", "core/circuit_internals/input.py"),
    ("core/circuit_noise.py", "core/circuit_internals/noise.py"),
    ("core/circuit_plotting.py", "core/circuit_internals/plotting.py"),
    ("core/circuit_routines.py", "core/circuit_internals/routines.py"),
    ("core/circuit_sym_methods.py", "core/circuit_internals/sym_methods.py"),
    ("core/circuit_utils.py", "core/circuit_internals/utils.py"),
    # Files introduced during B1 (recent — no downstream users): no shim.
    ("core/branch_metadata.py", "core/circuit_internals/branch_metadata.py"),
    (
        "core/charge_basis_operators.py",
        "core/circuit_internals/charge_basis_operators.py",
    ),
    (
        "core/discretized_phi_operators.py",
        "core/circuit_internals/discretized_phi_operators.py",
    ),
    ("core/matrix_helpers.py", "core/circuit_internals/matrix_helpers.py"),
    ("core/operator_factories.py", "core/circuit_internals/operator_factories.py"),
    ("core/sympy_helpers.py", "core/circuit_internals/sympy_helpers.py"),
    ("core/sawtooth.py", "core/circuit_internals/sawtooth.py"),
    ("core/circuit_yaml_assembly.py", "core/circuit_internals/yaml_assembly.py"),
]

# Long-standing files that need a shim at the OLD path.
SHIMMED_OLD_RELS = {
    "core/circuit_input.py",
    "core/circuit_noise.py",
    "core/circuit_plotting.py",
    "core/circuit_routines.py",
    "core/circuit_sym_methods.py",
    "core/circuit_utils.py",
}


def _module_path(rel: str) -> str:
    """`core/circuit_utils.py` -> `scqubits.core.circuit_utils`."""
    return "scqubits." + rel.replace("/", ".")[:-3]


# Map old dotted path -> new dotted path.
PATH_MAP: dict[str, str] = {_module_path(old): _module_path(new) for old, new in MOVES}

# Sort longest-first so longer prefixes match before shorter ones.
SORTED_KEYS = sorted(PATH_MAP, key=len, reverse=True)

# Directories under the repo root we should NOT walk for import rewrites.
SKIP_DIRS = {".git", "build", "dist", ".venv", "venv", "__pycache__"}


def git_mv_all() -> None:
    """Move each file via `git mv`."""
    pkg_dir = SQ / "core" / "circuit_internals"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    for old_rel, new_rel in MOVES:
        old = SQ / old_rel
        new = SQ / new_rel
        if not old.exists():
            print(f"  SKIP (already moved): {old_rel}")
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "mv", str(old), str(new)], cwd=str(REPO_ROOT))
        print(f"  moved: {old_rel} -> {new_rel}")


def _should_skip(py_file: Path) -> bool:
    """Skip the migration script itself, and anything under SKIP_DIRS."""
    if py_file == Path(__file__).resolve():
        return True
    return any(part in SKIP_DIRS for part in py_file.parts)


def rewrite_imports() -> None:
    """Walk every .py source file and rewrite old import paths."""
    rewrite_count = 0
    for py in REPO_ROOT.rglob("*.py"):
        if _should_skip(py):
            continue
        text = py.read_text(encoding="utf-8")
        new = text
        for old in SORTED_KEYS:
            replacement = PATH_MAP[old]
            # `from <old> import` -> `from <new> import`
            new = re.sub(rf"\bfrom {re.escape(old)}\b", f"from {replacement}", new)
            # `import <old>` (with optional ` as ...`) -> `import <new>`
            new = re.sub(rf"\bimport {re.escape(old)}\b", f"import {replacement}", new)
        if new != text:
            py.write_text(new, encoding="utf-8")
            rewrite_count += 1
            print(f"  rewrote: {py.relative_to(REPO_ROOT)}")
    print(f"  total files rewritten: {rewrite_count}")


SHIM_TEMPLATE = '''# {old_basename}.py — backward-compatibility shim
#
# The real module now lives at ``{new_dotted}``. This file re-exports its
# public names so existing ``from scqubits.core.{old_stem} import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``{new_dotted}``."""

from {new_dotted} import *  # noqa: F401, F403
'''


def write_shims() -> None:
    """Create shim files at SHIMMED_OLD_RELS."""
    for old_rel in SHIMMED_OLD_RELS:
        new_rel = next(n for o, n in MOVES if o == old_rel)
        old_path = SQ / old_rel
        old_basename = Path(old_rel).stem
        new_dotted = _module_path(new_rel)
        old_path.write_text(
            SHIM_TEMPLATE.format(
                old_basename=old_basename,
                old_stem=old_basename,
                new_dotted=new_dotted,
            ),
            encoding="utf-8",
        )
        subprocess.check_call(["git", "add", str(old_path)], cwd=str(REPO_ROOT))
        print(f"  shim: {old_rel} -> {new_dotted}")


PACKAGE_INIT_CONTENT = '''"""Internal sub-package for circuit support modules.

Public entry points (``Circuit``, ``Subsystem``, ``SymbolicCircuit`` etc.)
remain at ``scqubits.core.circuit`` / ``scqubits.core.symbolic_circuit``.
The ``_internals`` suffix signals that the modules within are not part
of the supported public API and may be reorganised without notice.
"""
'''


def write_package_init() -> None:
    """Write `scqubits/core/circuit_internals/__init__.py`."""
    init_path = SQ / "core" / "circuit_internals" / "__init__.py"
    init_path.write_text(PACKAGE_INIT_CONTENT, encoding="utf-8")
    subprocess.check_call(["git", "add", str(init_path)], cwd=str(REPO_ROOT))
    print(f"  wrote: {init_path.relative_to(REPO_ROOT)}")


def main() -> int:
    print("[1/4] git mv'ing files...")
    git_mv_all()
    print("[2/4] writing circuit_internals package __init__.py...")
    write_package_init()
    print("[3/4] rewriting imports across the tree...")
    rewrite_imports()
    print("[4/4] writing backward-compat shims...")
    write_shims()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
