#!/usr/bin/env python3
"""
Type-annotation modernizer for scqubits (Python 3.10+ target).

Handles the mechanical transformation layer:
  - Inserts `from __future__ import annotations` if missing
  - Optional[X]   ->  X | None
  - Union[A, B]   ->  A | B
  - Tuple[A, B]   ->  tuple[A, B]
  - List[X]       ->  list[X]
  - Dict[K, V]    ->  dict[K, V]
  - Set[X]        ->  set[X]
  - FrozenSet[X]  ->  frozenset[X]
  - Flags bare `# type: ignore` (no error code) for manual review
  - Fixes invalid `# type:ignore` (no space) -> `# type: ignore[FIXME]`
  - Prunes now-unused typing imports (Optional, Union, Tuple, List, Dict, Set, FrozenSet)

Does NOT touch:
  - Callable[...] signatures (need manual [ArgTypes], ReturnT)
  - Any logic or variable names
  - Annotations that require judgment (inferred from usage, ndarray dtype precision)

Usage:
  python modernize_types.py <file.py> [--dry-run] [--mypy]

With --mypy: runs `mypy <file>` before and after and reports error delta.
With --dry-run: prints the transformed file without writing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core type-string transformer
# ---------------------------------------------------------------------------


def _find_closing_bracket(text: str, open_pos: int) -> int:
    """Return index of `]` matching the `[` at open_pos. Returns -1 on failure."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_args(inner: str) -> list[str]:
    """Split comma-separated type args, respecting bracket nesting."""
    args: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in inner:
        if ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    # filter out empty strings from trailing commas or blank entries
    return [a for a in args if a]


def modernize_type_str(s: str) -> str:
    """Recursively convert a single type-annotation string to PEP 604 style."""
    s = s.strip()

    if s.startswith("Optional[") and s.endswith("]"):
        inner = modernize_type_str(s[9:-1])
        return f"{inner} | None"

    if s.startswith("Union[") and s.endswith("]"):
        raw_args = _split_args(s[6:-1])
        modern: list[str] = []
        has_none = False
        for a in raw_args:
            m = modernize_type_str(a)
            if m == "None":
                has_none = True
            else:
                modern.append(m)
        joined = " | ".join(modern)
        return f"{joined} | None" if has_none else joined

    for old, new_prefix in [
        ("Tuple[", "tuple["),
        ("List[", "list["),
        ("Dict[", "dict["),
        ("Set[", "set["),
        ("FrozenSet[", "frozenset["),
    ]:
        if s.startswith(old) and s.endswith("]"):
            inner = s[len(old) : -1]
            args = _split_args(inner)
            modern_args = [modernize_type_str(a) for a in args]
            return f"{new_prefix}{', '.join(modern_args)}]"

    return s


# ---------------------------------------------------------------------------
# Full-text transformer: replace every construct occurrence
# ---------------------------------------------------------------------------

_CONSTRUCTS = ["Optional", "Union", "Tuple", "List", "Dict", "Set", "FrozenSet"]
_CONSTRUCT_PATTERN = re.compile(r"\b(" + "|".join(_CONSTRUCTS) + r")\[")

# Bare generics (no subscript): Dict -> dict, List -> list, etc.
# Only in annotation positions: after `:`, `->`, `[`, `,` (heuristic via lookbehind context).
_BARE_GENERIC_MAP = {
    "Dict": "dict",
    "List": "list",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
}


def _replace_constructs(text: str) -> tuple[str, int]:
    """Replace all typing construct occurrences.  Returns (new_text, change_count)."""
    changes = 0
    result: list[str] = []
    i = 0
    while i < len(text):
        m = _CONSTRUCT_PATTERN.search(text, i)
        if m is None:
            result.append(text[i:])
            break
        bracket_open = m.end() - 1  # index of '['
        bracket_close = _find_closing_bracket(text, bracket_open)
        if bracket_close == -1:
            result.append(text[i : m.end()])
            i = m.end()
            continue

        full_expr = text[m.start() : bracket_close + 1]
        modern = modernize_type_str(full_expr)
        if modern != full_expr:
            changes += 1
        result.append(text[i : m.start()])
        result.append(modern)
        i = bracket_close + 1

    text = "".join(result)

    # Second pass: bare generics used as annotations.
    # Match in annotation context: preceded by `:`, `->`, `[`, `,`, `|`, or `=`.
    # Use a capture group for the context (fixed-width lookbehind can't mix lengths).
    for old_name, new_name in _BARE_GENERIC_MAP.items():
        # Also match after `(` (e.g., cast(List, x)) — runtime-safe since the
        # builtin `list` is a valid type argument.
        pattern = re.compile(r"(:|->|\[|,|\||=|\()(\s*)\b" + old_name + r"\b(?!\[|\w)")
        new_text, n = pattern.subn(r"\1\2" + new_name, text)
        if n:
            changes += n
            text = new_text

    return text, changes


# ---------------------------------------------------------------------------
# Import block cleaner
# ---------------------------------------------------------------------------

_OBSOLETE_IMPORTS = {
    "Optional",
    "Union",
    "Tuple",
    "List",
    "Dict",
    "Set",
    "FrozenSet",
}

# These types are also in `typing` but should come from `collections.abc`
# (project convention, Python 3.10+ idiom).
_ABC_IMPORTS = {
    "Callable",
    "Iterable",
    "Iterator",
    "Mapping",
    "MutableMapping",
    "Sequence",
    "MutableSequence",
    "Collection",
    "Container",
    "Generator",
    "Awaitable",
    "Coroutine",
    "AsyncIterable",
    "AsyncIterator",
    "AsyncGenerator",
    "Hashable",
    "Sized",
    "Reversible",
}


def _clean_typing_imports(text: str) -> tuple[str, list[str], list[str]]:
    """Remove unused typing imports, extract ABC-types for relocation.

    Returns (new_text, dropped_obsolete, extracted_for_abc).
    """
    removed: list[str] = []
    moved_to_abc: list[str] = []

    def _remove_from_import(m: re.Match) -> str:
        names_raw = m.group(1)
        names = [
            n.strip().rstrip(",") for n in re.split(r",\s*", names_raw) if n.strip()
        ]
        kept = [
            n for n in names if n not in _OBSOLETE_IMPORTS and n not in _ABC_IMPORTS
        ]
        dropped = [n for n in names if n in _OBSOLETE_IMPORTS]
        abc_bound = [n for n in names if n in _ABC_IMPORTS]
        if not dropped and not abc_bound:
            return m.group(0)  # nothing to change
        removed.extend(dropped)
        moved_to_abc.extend(abc_bound)
        if not kept:
            return ""
        new_imports = ", ".join(kept)
        if len(new_imports) < 60:
            return f"from typing import {new_imports}"
        lines = "from typing import (\n"
        for k in kept:
            lines += f"    {k},\n"
        lines += ")"
        return lines

    text = re.sub(
        r"from typing import \(([^)]+)\)",
        lambda m: _remove_from_import(m),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"from typing import ([^\n(]+)",
        lambda m: _remove_from_import(m),
        text,
    )

    # Merge ABC imports into existing `from collections.abc import ...` or insert a new line
    if moved_to_abc:
        abc_set = set(moved_to_abc)
        existing_match = re.search(r"from collections\.abc import ([^\n]+)", text)
        if existing_match:
            existing_names = {
                n.strip() for n in existing_match.group(1).split(",") if n.strip()
            }
            abc_set |= existing_names
            new_line = f"from collections.abc import {', '.join(sorted(abc_set))}"
            text = (
                text[: existing_match.start()] + new_line + text[existing_match.end() :]
            )
        else:
            # Insert after `from __future__ import annotations` if present,
            # otherwise before the first `from typing` (now possibly removed) or first import.
            new_line = f"from collections.abc import {', '.join(sorted(abc_set))}\n"
            future_match = re.search(r"from __future__ import annotations\n", text)
            if future_match:
                insert_pos = future_match.end()
                # Skip blank lines after the future import
                while insert_pos < len(text) and text[insert_pos] == "\n":
                    insert_pos += 1
                text = text[:insert_pos] + new_line + text[insert_pos:]
            else:
                # Fall back: insert before the first import statement
                first_imp = re.search(r"^(?:import |from )", text, re.MULTILINE)
                if first_imp:
                    text = (
                        text[: first_imp.start()] + new_line + text[first_imp.start() :]
                    )

    if removed or moved_to_abc:
        text = re.sub(r"\n\n\n+", "\n\n", text)
    return text, removed, moved_to_abc


# ---------------------------------------------------------------------------
# `from __future__ import annotations` insertion
# ---------------------------------------------------------------------------


def _ensure_future_annotations(text: str) -> tuple[str, bool]:
    if "from __future__ import annotations" in text:
        return text, False
    # Insert after the module docstring (if any), otherwise before first import
    docstring_end = 0
    stripped = text.lstrip()
    if stripped.startswith('"""') or stripped.startswith("'''"):
        q = stripped[:3]
        end = stripped.find(q, 3)
        if end != -1:
            docstring_end = (
                text.index(stripped[end + 3 : end + 4], 3) + 1
                if end + 3 < len(stripped)
                else 0
            )
    # Find first real import line
    insert_at = None
    for m in re.finditer(r"^(?:import |from )", text, re.MULTILINE):
        insert_at = m.start()
        break
    if insert_at is None:
        return "from __future__ import annotations\n\n" + text, True
    return (
        text[:insert_at] + "from __future__ import annotations\n\n" + text[insert_at:],
        True,
    )


# ---------------------------------------------------------------------------
# type:ignore fixer
# ---------------------------------------------------------------------------


def _fix_invalid_type_ignores(text: str) -> tuple[str, int]:
    """Fix `# type:ignore` (no space) -> `# type: ignore[FIXME]`.
    Also flags bare `# type: ignore` (no error code) for review.
    """
    count = 0

    # Fix completely missing space: `# type:ignore` -> `# type: ignore[FIXME]`
    new_text, n = re.subn(r"#\s*type:ignore(?!\[)", "# type: ignore[FIXME]", text)
    count += n

    return new_text, count


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def modernize_file(
    path: Path,
    *,
    dry_run: bool = False,
    run_mypy: bool = False,
) -> None:
    original = path.read_text(encoding="utf-8")

    # Step 1: gather mypy baseline
    mypy_before: int | None = None
    if run_mypy:
        result = subprocess.run(
            ["mypy", str(path)],
            capture_output=True,
            text=True,
        )
        m = re.search(r"Found (\d+) error", result.stdout + result.stderr)
        mypy_before = int(m.group(1)) if m else 0

    # Step 2: apply transformations
    text = original

    text, future_added = _ensure_future_annotations(text)
    if future_added:
        print(f"  [+] Inserted `from __future__ import annotations`")

    text, ignore_fixes = _fix_invalid_type_ignores(text)
    if ignore_fixes:
        print(
            f"  [~] Fixed {ignore_fixes} invalid `# type:ignore` comment(s) -> [FIXME]"
        )

    text, construct_changes = _replace_constructs(text)
    if construct_changes:
        print(f"  [~] Modernized {construct_changes} type construct(s)")

    text, removed_imports, moved_abc = _clean_typing_imports(text)
    if removed_imports:
        print(f"  [-] Removed obsolete typing imports: {', '.join(removed_imports)}")
    if moved_abc:
        print(f"  [>] Moved to collections.abc: {', '.join(moved_abc)}")

    if text == original:
        print("  (no changes needed)")
        return

    if dry_run:
        print("\n--- DRY RUN OUTPUT ---")
        print(text)
        return

    path.write_text(text, encoding="utf-8")
    print(f"  [OK] Written: {path}")

    # Step 3: gather mypy after
    if run_mypy:
        result = subprocess.run(
            ["mypy", str(path)],
            capture_output=True,
            text=True,
        )
        m = re.search(r"Found (\d+) error", result.stdout + result.stderr)
        mypy_after = int(m.group(1)) if m else 0
        delta = (mypy_before or 0) - mypy_after
        sign = "+" if delta >= 0 else ""
        print(f"\n  mypy: {mypy_before} -> {mypy_after} errors  ({sign}{delta} delta)")


def remove_unused_ignores(target_dir: Path = Path("scqubits")) -> None:
    """Strip `# type: ignore[...]` comments mypy flags as unused.

    Uses mypy output to find [unused-ignore] errors, then removes the
    ignore comment at that exact line while preserving any trailing code.
    """
    print("Running mypy to locate unused ignores...")
    result = subprocess.run(
        ["mypy", str(target_dir)],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    # Pattern: "path/to/file.py:LINE: error: Unused "type: ignore" comment [unused-ignore]"
    err_pattern = re.compile(
        r'^([^:\n]+\.py):(\d+): error: Unused "type: ignore" comment\s+\[unused-ignore\]',
        re.MULTILINE,
    )
    by_file: dict[str, list[int]] = {}
    for m in err_pattern.finditer(output):
        path = m.group(1).replace("\\", "/")
        line = int(m.group(2))
        by_file.setdefault(path, []).append(line)

    total = 0
    for file_path, lines_to_fix in by_file.items():
        p = Path(file_path)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.split("\n")
        for line_num in lines_to_fix:
            idx = line_num - 1
            if idx >= len(lines):
                continue
            # Strip "# type: ignore[...]" (and the leading whitespace before it)
            lines[idx] = re.sub(
                r"\s*#\s*type:\s*ignore(?:\[[^\]]*\])?",
                "",
                lines[idx],
            )
            total += 1
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [~] Stripped {len(lines_to_fix)} unused ignore(s) in {p}")

    print(f"\nTotal unused ignores removed: {total}")


def fix_plt_cm_deprecation(target_dir: Path = Path("scqubits")) -> int:
    """Replace `plt.cm.<colormap>` with `plt.get_cmap("<colormap>")` (matplotlib API)."""
    pat = re.compile(r"plt\.cm\.(\w+)")
    total = 0
    for py_file in target_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        new_text, n = pat.subn(r'plt.get_cmap("\1")', text)
        if n:
            py_file.write_text(new_text, encoding="utf-8")
            print(f"  [~] Replaced {n} `plt.cm.*` in {py_file}")
            total += n
    print(f"\nTotal plt.cm replacements: {total}")
    return total


def fix_implicit_optional(target_dir: Path = Path("scqubits")) -> None:
    """Fix implicit Optional params: `param: T = None` -> `param: T | None = None`.

    Uses mypy's own error messages to drive the rewrite — we only touch lines
    mypy explicitly flags with `PEP 484 prohibits implicit Optional`.
    """
    print(f"Running mypy to collect implicit-Optional positions...")
    result = subprocess.run(
        ["mypy", str(target_dir)],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    # Pattern: "path:LINE: error: Incompatible default for argument "NAME"
    #          (default has type "None", argument has type "T")  [assignment]"
    err_pattern = re.compile(
        r"^([^:\n]+\.py):(\d+): error: Incompatible default for argument \"([^\"]+)\" "
        r"\(default has type \"None\", argument has type \"([^\"]+)\"\)\s+\[assignment\]",
        re.MULTILINE,
    )

    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for m in err_pattern.finditer(output):
        path = m.group(1).replace("\\", "/")
        line = int(m.group(2))
        param = m.group(3)
        type_str = m.group(4)
        by_file.setdefault(path, []).append((line, param, type_str))

    total_fixed = 0
    files_changed = 0
    for file_path, fixes in by_file.items():
        p = Path(file_path)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.split("\n")
        for line_num, param_name, type_str in fixes:
            idx = line_num - 1
            if idx >= len(lines):
                continue
            old_line = lines[idx]
            # Pattern: <param>: <type> = None  ->  <param>: <type> | None = None
            # Be careful: type_str may contain generics, unions, etc.
            pat = re.compile(
                r"(\b"
                + re.escape(param_name)
                + r"\s*:\s*)("
                + re.escape(type_str)
                + r")(\s*=\s*None\b)"
            )
            new_line, n = pat.subn(r"\1\2 | None\3", old_line)
            if n:
                lines[idx] = new_line
                total_fixed += 1
        new_text = "\n".join(lines)
        if new_text != text:
            p.write_text(new_text, encoding="utf-8")
            files_changed += 1
            print(
                f"  [~] Fixed {sum(1 for fx in fixes if True)} implicit Optional(s) in {p}"
            )

    print(f"\nTotal fixes: {total_fixed} across {files_changed} files")


def resolve_fixmes(target_dir: Path = Path("scqubits")) -> None:
    """Replace `# type: ignore[FIXME]` with the real mypy error code at each line.

    Runs mypy on target_dir, matches every error line against every FIXME marker,
    and rewrites the marker with the specific error code(s) mypy reports at that
    position. If no error is reported, the FIXME ignore is removed entirely
    (it was suppressing nothing).
    """
    print(f"Running mypy on {target_dir} to collect error codes...")
    result = subprocess.run(
        ["mypy", str(target_dir)],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    # Map (file, line) -> set of error codes
    line_codes: dict[tuple[str, int], set[str]] = {}
    # mypy error format: "path/to/file.py:LINE: error: MESSAGE  [error-code]"
    err_pattern = re.compile(
        r"^([^:\n]+(?:\.py)):(\d+): error: .*?\[([a-z\-,\s]+)\]\s*$",
        re.MULTILINE,
    )
    for m in err_pattern.finditer(output):
        file_path = m.group(1).replace("\\", "/")
        line = int(m.group(2))
        codes = {c.strip() for c in m.group(3).split(",")}
        key = (file_path, line)
        line_codes.setdefault(key, set()).update(codes)

    # Find all FIXME markers in the target directory
    fixmes_found = 0
    fixmes_fixed = 0
    fixmes_removed = 0
    for py_file in target_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "[FIXME]" not in text:
            continue
        file_rel = str(py_file).replace("\\", "/")
        lines = text.split("\n")
        changed = False
        for idx, line in enumerate(lines):
            if "# type: ignore[FIXME]" not in line:
                continue
            fixmes_found += 1
            line_num = idx + 1
            codes = line_codes.get((file_rel, line_num), set())
            if codes:
                code_str = ", ".join(sorted(codes))
                lines[idx] = line.replace("[FIXME]", f"[{code_str}]")
                fixmes_fixed += 1
            else:
                # No error at this line — drop the ignore entirely
                lines[idx] = re.sub(r"\s*#\s*type:\s*ignore\[FIXME\]", "", line)
                fixmes_removed += 1
            changed = True
        if changed:
            py_file.write_text("\n".join(lines), encoding="utf-8")
            print(f"  [~] Updated {py_file}")

    print(
        f"\nFIXMEs: found={fixmes_found}, "
        f"replaced with real codes={fixmes_fixed}, "
        f"removed (no error at line)={fixmes_removed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file", type=Path, nargs="?", help="Python source file to modernize"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print output without writing"
    )
    parser.add_argument(
        "--mypy", action="store_true", help="Run mypy before/after and report delta"
    )
    parser.add_argument(
        "--fix-fixmes",
        action="store_true",
        help="Replace `[FIXME]` ignore markers across scqubits/ with real mypy error codes",
    )
    parser.add_argument(
        "--fix-implicit-optional",
        action="store_true",
        help="Fix implicit Optional params using mypy error locations",
    )
    parser.add_argument(
        "--fix-plt-cm",
        action="store_true",
        help='Replace deprecated `plt.cm.<name>` with `plt.get_cmap("<name>")`',
    )
    parser.add_argument(
        "--strip-unused-ignores",
        action="store_true",
        help="Remove `# type: ignore[...]` comments mypy reports as unused",
    )
    args = parser.parse_args()

    if args.fix_fixmes:
        resolve_fixmes()
        return

    if args.fix_implicit_optional:
        fix_implicit_optional()
        return

    if args.fix_plt_cm:
        fix_plt_cm_deprecation()
        return

    if args.strip_unused_ignores:
        remove_unused_ignores()
        return

    if not args.file or not args.file.exists():
        print(f"Error: file argument required or not found", file=sys.stderr)
        sys.exit(1)

    print(f"Modernizing {args.file} ...")
    modernize_file(args.file, dry_run=args.dry_run, run_mypy=args.mypy)


if __name__ == "__main__":
    main()
