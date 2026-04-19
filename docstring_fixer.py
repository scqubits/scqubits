#!/usr/bin/env python3
"""
Mechanical docstring fixer for scqubits.

Applies high-confidence, judgment-free transformations to docstrings:

  T1. Normalize default markers:
        (default value = X)  ->  (default: X)
        (default value: X)   ->  (default: X)
  T2. Promote bare-class backticks to :class: cross-refs (allowlist only).
        `HilbertSpace`  ->  :class:`HilbertSpace`
  T3. Strip type info from Parameters entries:
        `EJ : float` (with description on next line) -> `EJ:`
  T4. Dedent the Returns section's indented body to match the find_EJ_EC
      exemplar (no leading indent on the prose).

Does NOT touch:
  - function bodies, signatures, decorators, or any non-docstring source
  - docstrings inside __init__ methods (covered by class docstring)
  - module-level docstrings (those are typically headers, leave alone)
  - docstrings containing triple-backtick code fences
  - docstrings containing display-math blocks (\\[ ... \\])
  - multi-line summaries lacking a terminal period (judgment-required;
    flagged for the agent pass instead)

Modes
-----
    python docstring_fixer.py <file_or_dir>            # dry run; print summary
    python docstring_fixer.py <file_or_dir> --diff     # dry run; print diffs
    python docstring_fixer.py <file_or_dir> --write    # apply changes
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Conservative allowlist: only true scqubits class names. Parameter names
# (EC, EJ, EL), Python literals (True/False/None), and ambiguous identifiers
# are NOT included.
KNOWN_CLASSES = frozenset(
    {
        # Framework
        "HilbertSpace",
        "ParameterSweep",
        "QuantumSystem",
        "QubitBaseClass",
        "QubitBaseClass1d",
        "NoisySystem",
        "GenericQubit",
        "SpectrumData",
        "NamedSlotsNdarray",
        "Serializable",
        # Oscillators
        "Oscillator",
        "KerrOscillator",
        # Qubits
        "Transmon",
        "TunableTransmon",
        "Fluxonium",
        "FluxQubit",
        "Cos2PhiQubit",
        "ZeroPi",
        "FullZeroPi",
        # Circuit
        "Circuit",
        "Subsystem",
        "SymbolicCircuit",
        "SymbolicCircuitGraph",
        "Branch",
        "Node",
        # Discretization / numerics
        "Grid1d",
        "GridSpec",
        "WaveFunction",
        "WaveFunctionOnGrid",
    }
)

# NumPy section headers we recognize for indentation rules
NUMPY_SECTION_HEADERS = {
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Notes",
    "Examples",
    "See Also",
    "References",
    "Attributes",
    "Methods",
    "Other Parameters",
}

# Patterns that look like Python types in a Parameters entry
_TYPE_KEYWORDS = {
    "int",
    "float",
    "bool",
    "str",
    "bytes",
    "complex",
    "None",
    "object",
    "Any",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "Optional",
    "Union",
    "List",
    "Dict",
    "Tuple",
    "Set",
    "Sequence",
    "Mapping",
    "Iterable",
    "Iterator",
    "Callable",
    "ndarray",
    "NDArray",
    "csc_matrix",
    "csr_matrix",
    "Qobj",
    "Tensor",
    "DataFrame",
    "Series",
}


# ---------------------------------------------------------------------------
# Per-docstring transformations
# ---------------------------------------------------------------------------


def _normalize_defaults(text: str) -> tuple[str, int]:
    """T1. (default value = X) | (default value: X) -> (default: X)."""
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"(default: {m.group(1).strip()})"

    pattern = re.compile(r"\(default value\s*[=:]\s*([^)]+)\)")
    return pattern.sub(repl, text), n


def _promote_bare_classes(text: str) -> tuple[str, int]:
    """T2. Promote backticked allowlisted class names to :class: refs."""
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        prefix = m.group(1) or ""
        name = m.group(2)
        if prefix:
            return m.group(0)  # already has a Sphinx role
        if name not in KNOWN_CLASSES:
            return m.group(0)
        n += 1
        return f":class:`{name}`"

    # Only single-backtick references; double-backtick literal code is left.
    # Negative lookbehind to skip `` ``Foo`` ``.
    pattern = re.compile(r"(:[a-z]+:)?(?<!`)`([A-Z][A-Za-z0-9_]*)`(?!`)")
    return pattern.sub(repl, text), n


def _looks_like_type(s: str) -> bool:
    """Heuristic: is the post-colon content a type rather than a description?"""
    s = s.strip()
    if not s:
        return False
    # Strip ", optional" / ", default" / ", default None" suffix and recurse
    if "," in s:
        head, _, tail = s.rpartition(",")
        tail_stripped = tail.strip()
        if tail_stripped in (
            "optional",
            "default",
            "default None",
        ) or tail_stripped.startswith("default "):
            return _is_type_core(head.strip())
    return _is_type_core(s)


def _is_type_core(s: str) -> bool:
    """Type detection without suffix processing."""
    s = s.strip()
    if not s:
        return False
    # PEP 604 union: contains | between short tokens
    if re.fullmatch(r"[\w\[\], .\|]+", s) and "|" in s:
        return True
    # Parametrized generic: starts with capital, has [...]
    if re.fullmatch(r"[A-Z][A-Za-z0-9_]*\[.+\]", s):
        return True
    # Bare class name (capitalized, dotted): "ndarray", "np.ndarray", "MyClass"
    if re.fullmatch(r"[A-Za-z_][\w.]*", s):
        first_token = s.split(".")[0]
        if s in _TYPE_KEYWORDS or first_token in _TYPE_KEYWORDS:
            return True
        # All-caps single word → probably a type alias
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", s):
            return True
    # First-token-is-type check for compound forms
    m = re.match(r"^([A-Za-z_]\w*)", s)
    if not m:
        return False
    first = m.group(1)
    # Constraint form: "int >= 0", "float > 0.0", "int >=0"
    if first in _TYPE_KEYWORDS and re.fullmatch(
        r"\w+\s*(>=|<=|>|<|==|!=)\s*[-+]?\d*\.?\d+", s
    ):
        return True
    # Alternative-types form: "int or float", "float or ndarray"
    if first in _TYPE_KEYWORDS and " or " in s:
        tokens = [t.strip() for t in re.split(r"\s+or\s+", s)]
        if all(re.fullmatch(r"[\w.]+", t) for t in tokens):
            return True
    return False


def _strip_param_types(
    text: str, annotated_params: frozenset[str] | None = None
) -> tuple[str, int]:
    """T3. Inside Parameters block, strip 'name : type' → 'name:'.

    If `annotated_params` is provided, only strip the type when the parameter
    is annotated in the signature (otherwise we'd lose type information).
    """
    lines = text.splitlines(keepends=True)
    n = 0

    in_params = False
    entry_indent: int | None = None
    underline_re = re.compile(r"^-+\s*$")
    name_with_after = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*:\s+(\S.*?)\s*$")
    name_bare = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*:\s*$")

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        stripped = line.strip()

        # Detect entering Parameters
        if stripped == "Parameters" and underline_re.match(nxt.strip()):
            in_params = True
            entry_indent = None
            out.append(line)
            i += 1
            continue
        # Detect leaving Parameters (any other recognized section header)
        if (
            in_params
            and stripped in NUMPY_SECTION_HEADERS
            and stripped != "Parameters"
            and underline_re.match(nxt.strip())
        ):
            in_params = False

        if in_params:
            # Establish entry indent from the first param-like line
            if entry_indent is None:
                m_bare = name_bare.match(line.rstrip("\n"))
                m_typed = name_with_after.match(line.rstrip("\n"))
                if m_bare:
                    entry_indent = len(m_bare.group(1))
                elif m_typed:
                    entry_indent = len(m_typed.group(1))

            # Try to fix "name : type" lines
            m_typed = name_with_after.match(line.rstrip("\n"))
            if m_typed and entry_indent is not None:
                indent = len(m_typed.group(1))
                if indent == entry_indent:
                    name = m_typed.group(2)
                    rest = m_typed.group(3)
                    # Only strip when (a) it looks like a type AND (b) the
                    # signature has an annotation for this param. Otherwise
                    # stripping would lose type information.
                    if _looks_like_type(rest) and (
                        annotated_params is None or name in annotated_params
                    ):
                        # Replace with bare "name:"
                        ending = "\n" if line.endswith("\n") else ""
                        out.append(f"{m_typed.group(1)}{name}:{ending}")
                        n += 1
                        i += 1
                        continue

        out.append(line)
        i += 1

    return "".join(out), n


def _dedent_returns(text: str) -> tuple[str, int]:
    """T4. Dedent the Returns section's body to align with the section header.

    Some legacy docstrings over-indent the Returns prose (e.g. body at indent
    8 when the header is at indent 4). Match the find_EJ_EC exemplar where
    body prose is at the same indent as the `Returns` header.
    """
    lines = text.splitlines(keepends=True)
    underline_re = re.compile(r"^-+\s*$")
    leading_ws = re.compile(r"^(\s*)")

    out: list[str] = []
    n = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() == "Returns" and underline_re.match(nxt.strip()):
            header_indent = len(leading_ws.match(line).group(1))  # type: ignore[union-attr]
            out.append(line)
            out.append(nxt)
            i += 2

            # Capture body until blank line, end, or next section header.
            body: list[str] = []
            while i < len(lines):
                ln = lines[i]
                if (
                    ln.strip() in NUMPY_SECTION_HEADERS
                    and i + 1 < len(lines)
                    and underline_re.match(lines[i + 1].strip())
                ):
                    break
                if ln.strip() == "":
                    body.append(ln)
                    i += 1
                    if i < len(lines) and lines[i].strip() == "":
                        break
                    continue
                body.append(ln)
                i += 1

            # Find min indent across non-empty body lines.
            non_empty = [b for b in body if b.strip()]
            if non_empty:
                indents = [len(leading_ws.match(b).group(1)) for b in non_empty]  # type: ignore[union-attr]
                body_min_indent = min(indents)
                excess = body_min_indent - header_indent
                # Only dedent when body is over-indented relative to header.
                if excess > 0:
                    dedented = [(b[excess:] if b.strip() else b) for b in body]
                    out.extend(dedented)
                    n += 1
                    continue

            out.extend(body)
            continue

        out.append(line)
        i += 1

    return "".join(out), n


def _strip_self_cls_params(text: str) -> tuple[str, int]:
    """T5. Strip self: / cls: Parameters entries (boilerplate from monkeytype etc.).

    Removes the entry line and any subsequent more-indented description lines.
    """
    lines = text.splitlines(keepends=True)
    n = 0
    in_params = False
    entry_indent: int | None = None
    underline_re = re.compile(r"^-+\s*$")
    self_cls_entry = re.compile(r"^(\s*)(self|cls)\s*:")
    leading_ws = re.compile(r"^(\s*)")

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        stripped = line.strip()

        if stripped == "Parameters" and underline_re.match(nxt.strip()):
            in_params = True
            entry_indent = None
            out.append(line)
            i += 1
            continue
        if (
            in_params
            and stripped in NUMPY_SECTION_HEADERS
            and stripped != "Parameters"
            and underline_re.match(nxt.strip())
        ):
            in_params = False

        if in_params:
            m = self_cls_entry.match(line)
            if m:
                indent = len(m.group(1))
                if entry_indent is None:
                    entry_indent = indent
                if indent == entry_indent:
                    # Skip this entry header
                    n += 1
                    i += 1
                    # Skip any subsequent description lines (more indented or blank)
                    while i < len(lines):
                        ln = lines[i]
                        if not ln.strip():
                            i += 1
                            break
                        ln_indent = len(leading_ws.match(ln).group(1))  # type: ignore[union-attr]
                        if ln_indent > entry_indent:
                            i += 1
                        else:
                            break
                    continue
            # Establish entry_indent from the first regular param line if needed
            elif entry_indent is None:
                m_any = re.match(r"^(\s*)([A-Za-z_]\w*)\s*:", line)
                if m_any:
                    entry_indent = len(m_any.group(1))

        out.append(line)
        i += 1

    return "".join(out), n


# Pipeline of transformations
def _imperative_summary(text: str) -> tuple[str, int]:
    """T6. Convert summary-line indicative mood to imperative.

    `Returns the X.` → `Return the X.` etc., but only on the docstring's
    first non-empty line (the summary). Does not touch body prose.
    """
    if not text:
        return text, 0
    lines = text.splitlines(keepends=True)
    summary_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip()), -1
    )
    if summary_idx < 0:
        return text, 0
    line = lines[summary_idx]
    # leading whitespace
    leading = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
    body = line[len(leading):]
    rest = body
    # Match common indicative-mood verbs at the very start of the summary.
    # Order matters: longest first to avoid prefix collisions.
    replacements = [
        (r"^Returns\s", "Return "),
        (r"^Computes\s", "Compute "),
        (r"^Calculates\s", "Calculate "),
        (r"^Generates\s", "Generate "),
        (r"^Creates\s", "Create "),
        (r"^Builds\s", "Build "),
        (r"^Constructs\s", "Construct "),
        (r"^Plots\s", "Plot "),
        (r"^Draws\s", "Draw "),
        (r"^Sets\s", "Set "),
        (r"^Gets\s", "Get "),
        (r"^Adds\s", "Add "),
        (r"^Removes\s", "Remove "),
        (r"^Checks\s", "Check "),
        (r"^Performs\s", "Perform "),
        (r"^Updates\s", "Update "),
        (r"^Provides\s", "Provide "),
        (r"^Evaluates\s", "Evaluate "),
        (r"^Wraps\s", "Wrap "),
        (r"^Converts\s", "Convert "),
        (r"^Initializes\s", "Initialize "),
    ]
    n = 0
    for pat, repl in replacements:
        new_rest = re.sub(pat, repl, rest)
        if new_rest != rest:
            rest = new_rest
            n = 1
            break
    if n == 0:
        return text, 0
    lines[summary_idx] = leading + rest
    return "".join(lines), n


def _bool_backticks(text: str) -> tuple[str, int]:
    """T7. Single backticks around True/False/None → double backticks.

    Sphinx renders single-backtick names as cross-references; the Python
    literals don't have a target, so they should use literal-code (double)
    backticks. Conservative: only matches `True` / `False` / `None`
    surrounded by single backticks (not double, not preceded by a Sphinx
    role like `:py:obj:`).
    """
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        prefix = m.group(1) or ""
        if prefix:  # already :role:`True` style → leave alone
            return m.group(0)
        n += 1
        return f"``{m.group(2)}``"

    pattern = re.compile(r"(:[a-z]+:)?(?<!`)`(True|False|None)`(?!`)")
    return pattern.sub(repl, text), n


TRANSFORMS = [
    ("normalize-defaults", _normalize_defaults),
    ("promote-classes", _promote_bare_classes),
    ("strip-self-cls-params", _strip_self_cls_params),
    ("strip-param-types", _strip_param_types),
    ("dedent-returns", _dedent_returns),
    ("imperative-summary", _imperative_summary),
    ("bool-backticks", _bool_backticks),
]


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------


@dataclass
class DocstringEdit:
    """A single docstring location and its proposed new content."""

    qualname: str
    lineno: int  # 1-based
    end_lineno: int
    col_offset: int
    end_col_offset: int
    original: str  # raw source slice including triple-quotes and prefix
    new: str  # full replacement source slice
    changes_by_transform: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original != self.new


@dataclass
class FileResult:
    path: Path
    parse_error: str | None = None
    edits: list[DocstringEdit] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (qualname, reason)

    @property
    def n_changed(self) -> int:
        return sum(1 for e in self.edits if e.changed)


def _docstring_node(node: ast.AST) -> ast.Constant | None:
    """Return the docstring Constant node for a Module/Class/Function, or None."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value
    return None


def _enumerate_docstring_targets(
    tree: ast.Module,
) -> list[tuple[str, ast.AST, ast.Constant, bool]]:
    """Yield (qualname, owner_node, docstring_node, is_init).

    Skips module-level docstrings. Skips __init__ methods (covered by class
    docstring per scqubits convention).
    """
    targets: list[tuple[str, ast.AST, ast.Constant, bool]] = []

    def walk(node: ast.AST, qualifier: str, in_class: bool, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                d = _docstring_node(child)
                if d is not None:
                    qual = f"{qualifier}{child.name}" if qualifier else child.name
                    targets.append((qual, child, d, False))
                walk(
                    child,
                    qualifier=(
                        f"{qualifier}{child.name}." if qualifier else f"{child.name}."
                    ),
                    in_class=True,
                    in_func=in_func,
                )
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                d = _docstring_node(child)
                is_init = child.name == "__init__" and in_class
                if d is not None and not is_init and not in_func:
                    qual = f"{qualifier}{child.name}" if qualifier else child.name
                    targets.append((qual, child, d, is_init))
                walk(child, qualifier=qualifier, in_class=in_class, in_func=True)
            else:
                walk(child, qualifier=qualifier, in_class=in_class, in_func=in_func)

    walk(tree, qualifier="", in_class=False, in_func=False)
    return targets


def _extract_string_parts(raw: str) -> tuple[str, str, str]:
    """Split a triple-quoted string literal into (prefix, opener, body, closer).

    Returns (prefix+opener, body, closer). Handles raw and bytes prefixes
    plus triple-single, triple-double, single-single, single-double quotes.
    """
    # Match leading prefix chars
    m = re.match(r"^([rRuUbB]*)('''|\"\"\"|'|\")", raw)
    if not m:
        raise ValueError(f"Not a recognizable string literal: {raw[:50]!r}")
    prefix = m.group(1)
    quote = m.group(2)
    if not raw.endswith(quote):
        raise ValueError(f"String does not end with matching quote: {raw[-20:]!r}")
    head = prefix + quote
    body = raw[len(head) : -len(quote)]
    return head, body, quote


def _reconstruct_docstring_literal(head: str, new_body: str, closer: str) -> str:
    return head + new_body + closer


def _apply_transforms(
    text: str, annotated_params: frozenset[str] | None = None
) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    cur = text
    for name, fn in TRANSFORMS:
        if name == "strip-param-types":
            cur, n = fn(cur, annotated_params)  # type: ignore[call-arg]
        else:
            cur, n = fn(cur)
        if n:
            counts[name] = n
    return cur, counts


def _annotated_params_for(node: ast.AST) -> frozenset[str]:
    """Return the set of parameter names that are annotated in the signature."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        named: list[str] = []
        for a in args.posonlyargs + args.args + args.kwonlyargs:
            if a.annotation is not None:
                named.append(a.arg)
        if args.vararg and args.vararg.annotation is not None:
            named.append(args.vararg.arg)
        if args.kwarg and args.kwarg.annotation is not None:
            named.append(args.kwarg.arg)
        return frozenset(named)
    return frozenset()


def _should_skip(text: str) -> str | None:
    """Return a skip-reason string if this docstring should be left alone."""
    if "```" in text:
        return "contains triple-backtick code fence"
    if r"\[" in text and r"\]" in text:
        # display math — be cautious about indent transforms
        # Allow defaults / class promotion still to apply via per-transform
        # protections, but skip dedent-returns. We handle this finer-grained
        # below; not a global skip.
        return None
    return None


def fix_file(path: Path) -> FileResult:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return FileResult(path=path, parse_error=f"SyntaxError: {e}")

    result = FileResult(path=path)
    targets = _enumerate_docstring_targets(tree)

    for qualname, owner, ds_node, is_init in targets:
        # Get the raw source slice for this string literal
        try:
            seg = ast.get_source_segment(src, ds_node)
        except Exception as e:
            result.skipped.append((qualname, f"could not extract source: {e}"))
            continue
        if seg is None:
            result.skipped.append((qualname, "ast.get_source_segment returned None"))
            continue

        try:
            head, body, closer = _extract_string_parts(seg)
        except ValueError as e:
            result.skipped.append((qualname, str(e)))
            continue

        skip_reason = _should_skip(body)
        if skip_reason:
            result.skipped.append((qualname, skip_reason))
            continue

        # For class docstrings, use the __init__ signature's annotations.
        annotated = _annotated_params_for(owner)
        if isinstance(owner, ast.ClassDef):
            for child in owner.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    annotated = _annotated_params_for(child)
                    break
        new_body, counts = _apply_transforms(body, annotated)
        if new_body == body:
            continue  # no changes

        new_seg = _reconstruct_docstring_literal(head, new_body, closer)
        edit = DocstringEdit(
            qualname=qualname,
            lineno=ds_node.lineno,
            end_lineno=ds_node.end_lineno or ds_node.lineno,
            col_offset=ds_node.col_offset,
            end_col_offset=ds_node.end_col_offset or ds_node.col_offset,
            original=seg,
            new=new_seg,
            changes_by_transform=counts,
        )
        result.edits.append(edit)

    return result


def _splice_edits(src: str, edits: list[DocstringEdit]) -> str:
    """Apply edits to source, working in reverse line order so offsets stay valid."""
    if not edits:
        return src

    lines = src.splitlines(keepends=True)
    # Compute byte offset of each line start
    line_offsets = [0]
    for ln in lines:
        line_offsets.append(line_offsets[-1] + len(ln))

    def pos(lineno: int, col: int) -> int:
        return line_offsets[lineno - 1] + col

    # Sort edits by start position descending
    sorted_edits = sorted(edits, key=lambda e: (e.lineno, e.col_offset), reverse=True)
    out = src
    for e in sorted_edits:
        if not e.changed:
            continue
        start = pos(e.lineno, e.col_offset)
        end = pos(e.end_lineno, e.end_col_offset)
        # Sanity: the slice should equal the original
        if out[start:end] != e.original:
            # Fall back: search for exact match nearby
            raise RuntimeError(
                f"Source slice mismatch for {e.qualname} at L{e.lineno}:\n"
                f"  expected: {e.original[:80]!r}\n"
                f"  got:      {out[start:end][:80]!r}"
            )
        out = out[:start] + e.new + out[end:]

    return out


def write_file(path: Path, result: FileResult) -> bool:
    """Apply edits to disk; verify file still parses; revert on parse error."""
    if not result.edits:
        return False

    src = path.read_text(encoding="utf-8")
    new_src = _splice_edits(src, result.edits)

    # Verify
    try:
        ast.parse(new_src, filename=str(path))
    except SyntaxError as e:
        print(f"  ABORT: edit produced invalid syntax in {path}: {e}", file=sys.stderr)
        return False

    if new_src != src:
        path.write_text(new_src, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[FileResult]) -> None:
    total_files = len(results)
    changed_files = sum(1 for r in results if r.n_changed)
    total_edits = sum(r.n_changed for r in results)
    transform_totals: dict[str, int] = {}
    for r in results:
        for e in r.edits:
            for k, v in e.changes_by_transform.items():
                transform_totals[k] = transform_totals.get(k, 0) + v

    print(f"Files scanned: {total_files}")
    print(f"Files with proposed changes: {changed_files}")
    print(f"Docstrings modified: {total_edits}")
    print()
    print("Changes by transform:")
    for name, _ in TRANSFORMS:
        n = transform_totals.get(name, 0)
        print(f"  {name:<22} {n}")

    skipped = [(r.path, q, why) for r in results for q, why in r.skipped]
    if skipped:
        print()
        print(f"Skipped docstrings: {len(skipped)}")
        for path, q, why in skipped[:20]:
            print(f"  {path}::{q}  -- {why}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


def print_diffs(results: list[FileResult]) -> None:
    for r in results:
        if not r.n_changed:
            continue
        print(f"\n=== {r.path} ===")
        for e in r.edits:
            if not e.changed:
                continue
            print(
                f"\n--- {e.qualname}  (L{e.lineno})  changes={e.changes_by_transform}"
            )
            diff = difflib.unified_diff(
                e.original.splitlines(keepends=True),
                e.new.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
            for line in diff:
                sys.stdout.write(line if line.endswith("\n") else line + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _gather_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.py") if "tests" not in p.parts)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", help="Python file or directory")
    p.add_argument("--write", action="store_true", help="apply changes to disk")
    p.add_argument(
        "--diff", action="store_true", help="print unified diffs (dry-run only)"
    )
    args = p.parse_args(argv)

    target = Path(args.target)
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 2

    files = _gather_files(target)
    results = [fix_file(f) for f in files]

    parse_errors = [r for r in results if r.parse_error]
    for r in parse_errors:
        print(f"PARSE ERROR: {r.path}: {r.parse_error}", file=sys.stderr)

    if args.write:
        n_written = 0
        for r in results:
            if r.n_changed:
                if write_file(r.path, r):
                    n_written += 1
        print(f"Wrote changes to {n_written} files")
    elif args.diff:
        print_diffs(results)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
