"""Docstring-compliance linter for scqubits.

Walks one or more Python files (or directories) and checks each function /
class / method docstring against the project's documented rules. Designed to
run both locally (``python tools/docstring_lint.py scqubits/``) and inside
CI (``python tools/docstring_lint.py --compare-to origin/main scqubits/``)
where only docstring issues that are *new* relative to ``main`` should fail
the build.

Rules enforced
--------------

The checks below mirror the project's standing docstring requirements
(see ``CIRCUIT_DEVELOPER_MANUAL.md`` and the developer's memory rules):

- **DOC001 -- placeholder phrases.**  Numpydoc-stub residue strings
  (the literal underscore-wrapped ``type`` / ``description`` markers
  some IDEs insert, plus angle-bracketed unresolved-task markers and
  vague-noun phrases gesturing at output structure) must be filled
  in or removed.  See :class:`PlaceholderPhraseCheck` for the exact
  regexes.
- **DOC002 -- types in docstring Parameters / Returns sections.**  Type
  annotations belong in the function signature, not duplicated into the
  docstring.  In numpydoc Parameters blocks the parameter line should be
  ``name:`` (description on the next indented line); a line of the form
  ``name : type`` or ``name: type`` is flagged.
- **DOC003 -- work-narrative phrases.**  Docstrings should describe the
  current contract, not history of how the code got here.  Phrases that
  describe prior states, reference specific commits / sessions, or
  attribute work to particular authors are flagged.  See
  :data:`_WORK_NARRATIVE_PATTERNS` for the exact list.
- **DOC004 -- empty numpydoc sections.**  A ``Parameters`` /
  ``Returns`` / ``Raises`` header with no body indicates a stub or a
  bad edit; flagged.

Severity
--------

For symbols whose qualified name contains any ``_``-prefixed component
(but not ``__dunder__``), all issues are downgraded to **warnings**
rather than errors -- those symbols are private API and the user has
asked that lack of compliance there be forgivable.  Public symbols
produce **errors**.

Exit status
-----------

- ``0`` -- no errors (warnings are allowed).
- ``1`` -- at least one error.
- ``2`` -- invocation error (bad arguments, missing git ref, etc.).

Comparison mode
---------------

``--compare-to <ref>`` (e.g. ``--compare-to origin/main``) runs the
checks twice: once on the working-tree files, once on the files at the
named git ref.  Issues are matched by ``(file, qualname, check_id)`` --
not by line number, so cosmetic line shifts don't count as "new".  Only
issues present at HEAD and absent at the base ref are reported.  An
issue removed by the diff is silently dropped.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import keyword
import re
import subprocess
import sys

from collections.abc import Iterable, Iterator
from pathlib import Path

# ---------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Issue:
    """One docstring-rule violation in one file.

    ``message`` is a one-line summary; ``snippet`` is the offending
    docstring line(s), and ``hint`` is a concrete suggestion for how to
    fix it.  All three are surfaced in the human-readable output;
    ``--format json`` exposes every field.
    """

    file: str  # path relative to the repo root
    qualname: str  # e.g. ``Circuit.configure`` or ``<module>``
    check_id: str  # e.g. ``DOC001``
    severity: str  # ``"error"`` or ``"warning"``
    line: int  # 1-based line in the current file (informational)
    message: str  # one-line summary of what is wrong
    snippet: str = ""  # the offending docstring line, verbatim
    hint: str = ""  # concrete suggestion for how to fix it
    rule_ref: str = ""  # short pointer to the documented rule

    def key(self) -> tuple[str, str, str, str]:
        """Identity tuple used to compare issues across git refs.

        Excludes ``line`` (line numbers shift across edits) and
        ``severity`` (severity is computed from the symbol's privacy,
        which can flip if a symbol is renamed but the issue is the
        same in both versions for the post-rename name).  Also
        excludes ``snippet`` / ``hint`` / ``rule_ref`` -- those are
        derivative of the message and would create spurious "new
        issue" reports if a check's hint wording were tweaked.

        Returns
        -------
        ``(file, qualname, check_id, message)`` tuple.  ``message`` is
        included so distinct ``DOC001`` issues on the same function
        (one ``type``-marker hit and one ``description``-marker hit,
        say) are tracked as different issues rather than being
        coalesced.
        """
        return (self.file, self.qualname, self.check_id, self.message)


# ---------------------------------------------------------------------
# Privacy detection
# ---------------------------------------------------------------------


def _is_private_qualname(qualname: str) -> bool:
    """Return ``True`` if any component of ``qualname`` is ``_``-prefixed.

    Dunder names (``__init__``, ``__set__``, ...) and the synthetic
    ``"<module>"`` qualname are treated as public.

    Parameters
    ----------
    qualname:
        e.g. ``"Circuit.configure"`` or ``"NoisyCircuit._wrapper_t1_inductive_capacitive"``.
    """
    for part in qualname.split("."):
        if part == "<module>":
            continue
        if part.startswith("__") and part.endswith("__"):
            continue
        if part.startswith("_"):
            return True
    return False


# ---------------------------------------------------------------------
# Numpydoc section parsing
# ---------------------------------------------------------------------

_SECTION_HEADERS = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Notes",
    "Examples",
    "References",
    "See Also",
    "Attributes",
    "Methods",
)


def _iter_numpydoc_sections(
    docstring: str,
) -> Iterator[tuple[str, int, str]]:
    """Yield ``(section_name, start_line, body)`` for every numpydoc section.

    ``start_line`` is 1-based and points at the section header line.
    ``body`` is the full text from the line after the dashes up to (but
    not including) the next section header or end-of-docstring.

    The detector is conservative: it requires a header line ``Parameters``
    immediately followed by a ``----------`` underline line, with the
    same indentation.  Bare prose paragraphs that happen to contain the
    word "Parameters" do not count.
    """
    lines = docstring.splitlines()
    headers: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped not in _SECTION_HEADERS:
            continue
        if i + 1 >= len(lines):
            continue
        underline = lines[i + 1].strip()
        if not underline or underline != "-" * len(underline):
            continue
        if len(underline) < 3:
            continue
        # The header indent must equal the underline indent.
        header_indent = len(line) - len(line.lstrip())
        underline_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
        if header_indent != underline_indent:
            continue
        headers.append((stripped, i))

    for j, (name, start) in enumerate(headers):
        body_start = start + 2  # skip header + underline
        body_end = headers[j + 1][1] if j + 1 < len(headers) else len(lines)
        body = "\n".join(lines[body_start:body_end])
        yield name, start + 1, body


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------


class Check:
    """Base class for one docstring rule."""

    id: str = ""
    description: str = ""
    rule_ref: str = ""

    def run(
        self, docstring: str, _node: ast.AST, _qualname: str
    ) -> Iterator[tuple[int, str, str, str]]:
        """Yield ``(line_offset, message, snippet, hint)`` per issue.

        ``line_offset`` is 1-based within the docstring; the caller
        adds the node's docstring start line to compute the absolute
        file line.  ``snippet`` is the offending docstring line
        (verbatim, lstripped); ``hint`` is a concrete fix suggestion.
        """
        raise NotImplementedError


class PlaceholderPhraseCheck(Check):
    """DOC001 -- flags numpydoc-stub residue placeholders.

    See ``PATTERNS`` below for the exact set of detected strings.
    """

    id = "DOC001"
    description = "docstring contains placeholder phrase"
    rule_ref = "rule: numpydoc-stub placeholders must be filled in or removed"

    # (regex, short_label, hint).  ``short_label`` is what appears in
    # the one-line message; ``hint`` is the concrete fix suggestion.
    # Patterns target template residue produced by numpydoc-stub
    # generators (PyCharm's "Insert Documentation String", VS Code's
    # autoDocstring, the ``pyment`` CLI, etc.) -- they leave behind a
    # consistent set of placeholder strings that the author is
    # supposed to fill in.
    PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
        (
            re.compile(r"_type_"),
            "`_type_` placeholder (numpydoc-stub residue)",
            "Replace `_type_` with the actual type -- or, if the type is "
            "already declared in the function signature, drop the type "
            "line entirely (numpydoc supports description-only entries).",
        ),
        (
            re.compile(r"_description_"),
            "`_description_` placeholder (numpydoc-stub residue)",
            "Replace `_description_` with a one-line description of "
            "what this parameter / return value represents.",
        ),
        (
            re.compile(r"<TODO>", re.IGNORECASE),
            "`<TODO>` placeholder",
            "Resolve the TODO before merging, or move it to an issue "
            "tracker and remove the placeholder from the docstring.",
        ),
        (
            re.compile(r"<FIXME>", re.IGNORECASE),
            "`<FIXME>` placeholder",
            "Address the FIXME before merging, or convert it to a "
            "documented limitation in a `Notes` section.",
        ),
    )

    def run(self, docstring, _node, _qualname):
        for offset, line in enumerate(docstring.splitlines(), start=1):
            for pattern, label, hint in self.PATTERNS:
                if pattern.search(line):
                    yield offset, label, line.strip(), hint
                    break  # one issue per line is enough


# Heuristic for "this looks like a numpydoc parameter declaration with a
# type annotation glued to it".  Matches:
#     name : int
#     name: ndarray | None
#     name : Optional[float]
# but NOT:
#     name:
#         description on the next line
# Allowed indents under a Parameters block: typically 4 or 8 spaces.
_PARAM_TYPE_LINE = re.compile(
    r"^(?P<indent>[ \t]{0,12})(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<rhs>\S.*)$"
)


# Set of Python type names (built-ins + scqubits-relevant) that should be
# recognized as types when they appear bare in a docstring.  Used by both
# ``_looks_like_a_type`` (Parameters RHS) and ``_looks_like_bare_type_line``
# (first body line of a Returns section).
_KNOWN_TYPE_NAMES = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "None",
        "Any",
        "Optional",
        "Union",
        "Callable",
        "Iterator",
        "Iterable",
        "Mapping",
        "Sequence",
        "Generator",
        "Type",
        "Literal",
        "ClassVar",
        "TypeVar",
        "Protocol",
        "ndarray",
        "ArrayLike",
        "DTypeLike",
        "Qobj",
        "Symbol",
        "Expr",
        "Matrix",
        "csc_matrix",
        "csr_matrix",
        "coo_matrix",
        "lil_matrix",
        "Path",
        "PathLike",
        "object",
        "complex",
    }
)


def _is_type_token(tok: str) -> bool:
    """Return ``True`` if ``tok`` is plausibly a Python type identifier.

    A token qualifies as a type identifier if it's either in the curated
    ``_KNOWN_TYPE_NAMES`` set or it has the shape of a CapCamelCase
    class name (starts with an uppercase letter and contains at least
    one lowercase letter).  All-uppercase tokens (``URL``, ``TODO``)
    and bare lowercase words not in the known set fail.
    """
    if not tok:
        return False
    if tok in _KNOWN_TYPE_NAMES:
        return True
    if (
        re.match(r"^[A-Z][a-zA-Z0-9_]*$", tok)
        and not tok.isupper()
        and any(c.islower() for c in tok)
    ):
        return True
    return False


def _looks_like_a_type(rhs: str) -> bool:
    """Return ``True`` when the RHS of ``name : RHS`` is a Python type expression.

    Only flags the RHS if **every** token in it (after splitting on
    type-syntax separators ``|``, ``[``, ``]``, ``,``, ``.`` and
    whitespace) is a recognized type identifier.  This rejects benign
    descriptions that happen to start with a known type name like
    ``"None when not applicable"`` or ``"int input value"``.
    """
    rhs = rhs.strip()
    if not rhs:
        return False
    # Strip off ``", by default ..."`` / ``" = default"`` suffixes; the
    # actual type is the head.
    head = re.split(r",\s+by default|,\s*default[:=]?|\s+=\s+", rhs, maxsplit=1)[
        0
    ].strip()
    if not head:
        return False
    # Strip wrapping quotes around forward references like ``"Foo"``.
    if len(head) >= 2 and head[0] in "\"'" and head[-1] == head[0]:
        head = head[1:-1].strip()
    tokens = [t for t in re.split(r"[\s\|\[\],.]+", head) if t]
    if not tokens:
        return False
    return all(_is_type_token(tok) for tok in tokens)


def _looks_like_bare_type_line(line: str) -> bool:
    """Return ``True`` when ``line`` is a bare type expression (Returns context).

    Stricter than ``_looks_like_a_type`` because a single CapCamelCase
    word in a Returns section is much more likely to be a description
    ("Length", "Frequency") than a custom-class type annotation.

    Rules:
      - Reject if the line ends in ``.`` (sentence).
      - Reject if longer than 60 characters (real type expressions are
        short; longer lines are descriptions).
      - Run ``_looks_like_a_type``.
      - If only one token survives tokenisation, require it to be in
        ``_KNOWN_TYPE_NAMES`` (rejects single CapWords like ``Length``).
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith("."):
        return False
    if len(stripped) > 60:
        return False
    if not _looks_like_a_type(stripped):
        return False
    tokens = [t for t in re.split(r"[\s\|\[\],.]+", stripped) if t]
    if len(tokens) == 1 and tokens[0] not in _KNOWN_TYPE_NAMES:
        return False
    return True


def _annotated_param_names(node: ast.AST) -> set[str]:
    """Return the set of parameter names that have type annotations on ``node``.

    Returns the empty set for modules, classes, or any node that isn't a
    function/method.  For a function/method, walks ``args.posonlyargs``,
    ``args.args``, ``args.kwonlyargs``, ``args.vararg`` (``*args``), and
    ``args.kwarg`` (``**kwargs``), and includes a name iff its
    ``annotation`` slot is not ``None``.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    names: set[str] = set()
    args = node.args
    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        if arg.annotation is not None:
            names.add(arg.arg)
    for special in (args.vararg, args.kwarg):
        if special is not None and special.annotation is not None:
            names.add(special.arg)
    return names


def _has_return_annotation(node: ast.AST) -> bool:
    """Return ``True`` iff ``node`` is a function/method with a ``-> ...`` annotation."""
    return (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.returns is not None
    )


class TypesInDocstringCheck(Check):
    """DOC002 -- flags type annotations duplicated into the docstring.

    Catches both the numpydoc ``name : type`` Parameters form *and* the
    bare-type-on-first-line Returns form.  The rule fires only when
    the function signature *already* carries the annotation -- i.e.
    when the docstring is duplicating information that the type
    system already has.  Functions without signature annotations
    legitimately use the docstring as the type source and are not
    flagged.
    """

    id = "DOC002"
    description = "type annotation in docstring duplicates the function signature"
    rule_ref = "rule: types belong in the function signature, not duplicated into the docstring"

    def run(self, docstring, node, _qualname):
        annotated_params = _annotated_param_names(node)
        has_return_ann = _has_return_annotation(node)
        for section, start_line, body in _iter_numpydoc_sections(docstring):
            if section == "Parameters":
                yield from self._check_parameters(
                    section, start_line, body, annotated_params
                )
            elif section in ("Returns", "Yields"):
                if has_return_ann:
                    yield from self._check_returns(section, start_line, body)

    def _check_parameters(self, section, start_line, body, annotated_params):
        # Indent-aware: only lines at the parameter-declaration indent
        # are real parameter entries.  More-indented lines are
        # description continuations (e.g. ``or: list of options`` as a
        # continuation of a previous param's multi-line description),
        # which can otherwise produce false positives.
        body_lines = body.splitlines()
        param_indent: int | None = None
        for offset, line in enumerate(body_lines, start=1):
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if param_indent is None:
                # First non-blank body line sets the parameter-indent
                # baseline.  Subsequent param lines must match.
                param_indent = indent
            if indent != param_indent:
                continue
            m = _PARAM_TYPE_LINE.match(line)
            if not m:
                continue
            param = m.group("name")
            # Belt-and-suspenders: even if a Python keyword somehow
            # ended up at the right indent (it shouldn't be a real
            # parameter name), don't flag it.
            if keyword.iskeyword(param):
                continue
            # Only flag when the signature already carries the
            # annotation -- otherwise the docstring is legitimately
            # acting as the type source.
            if param not in annotated_params:
                continue
            if not _looks_like_a_type(m.group("rhs")):
                continue
            line_in_doc = start_line + 1 + offset
            type_text = m.group("rhs").strip()
            yield (
                line_in_doc,
                f"parameter `{param}` carries type `{type_text}` in docstring "
                f"(signature already declares the type)",
                line.strip(),
                f"Move the type out of the docstring: change this line to "
                f"`{param}:` (description on the next indented line).  "
                f"The signature's `{param}: {type_text}` annotation is "
                f"the source of truth; Sphinx with napoleon picks the "
                f"type up from there automatically.",
            )

    def _check_returns(self, section, start_line, body):
        # Only invoked when the function has a ``-> ...`` annotation,
        # so a bare-type first line in the section is genuinely
        # duplicating the signature.  First non-blank body line: if it
        # parses as a bare type (one identifier, possibly with
        # ``[...]`` / ``|``), flag it.
        body_lines = body.splitlines()
        for offset, line in enumerate(body_lines, start=1):
            if not line.strip():
                continue
            stripped = line.strip()
            if not _looks_like_bare_type_line(stripped):
                return  # description first -- fine
            line_in_doc = start_line + 1 + offset
            yield (
                line_in_doc,
                f"`{section}` section starts with bare type `{stripped}` "
                f"(signature already declares the return type)",
                stripped,
                f"Drop the bare-type first line and put the description "
                f"directly under the `{section}` header.  The signature's "
                f"`-> {stripped}` annotation is the source of truth; "
                f"Sphinx with napoleon surfaces the type from there.",
            )
            return


# Phrases that indicate the docstring is narrating change history rather
# than describing the current contract.  Case-insensitive, word-boundary
# anchored.  The set is deliberately narrow: each phrase is one that
# clearly signals a backward-looking statement that belongs in a commit
# message or PR description, not in a docstring.
_WORK_NARRATIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwe used to\b",
        r"\bthis was changed\b",
        r"\bin a prior refactor\b",
        r"\bsee commit\b",
        r"\bafter the refactor\b",
        r"\bpreviously this (function|method|class|routine)\b",
        r"\boriginally this (function|method|class|routine)\b",
        r"\bused to (be|live)\b",
        r"\b(was|were) (formerly|previously)\b",
    )
)


class WorkNarrativeCheck(Check):
    """DOC003 -- flags refactor-history narrative inside docstrings."""

    id = "DOC003"
    description = "docstring contains work-narrative / refactor-history phrasing"
    rule_ref = "rule: docstrings describe the current contract, not change history"

    def run(self, docstring, _node, _qualname):
        for offset, line in enumerate(docstring.splitlines(), start=1):
            for pattern in _WORK_NARRATIVE_PATTERNS:
                m = pattern.search(line)
                if m:
                    yield (
                        offset,
                        f"work-narrative phrase `{m.group(0)}`",
                        line.strip(),
                        "Docstrings describe the current contract, not "
                        "history.  Rewrite to describe what this function "
                        "does *now* -- move refactor history to the commit "
                        "message or PR description, and document any "
                        "load-bearing legacy behavior as a forward-looking "
                        'Notes paragraph ("This function preserves X for '
                        'back-compat.").',
                    )
                    break


class EmptyNumpydocSectionCheck(Check):
    """DOC004 -- flags numpydoc section headers with no body."""

    id = "DOC004"
    description = "numpydoc section header with no content"
    rule_ref = "rule: numpydoc section headers must have a body"

    SECTIONS_TO_CHECK = ("Parameters", "Returns", "Yields", "Raises")

    def run(self, docstring, _node, _qualname):
        for section, start_line, body in _iter_numpydoc_sections(docstring):
            if section not in self.SECTIONS_TO_CHECK:
                continue
            if not body.strip():
                yield (
                    start_line,
                    f"`{section}` section header has empty body",
                    f"{section}\n{'-' * len(section)}",
                    f"Either fill in the `{section}` body with a real "
                    f"description, or remove the header.  An empty "
                    f"section header indicates a stub or a half-finished "
                    f"edit that lost the body.",
                )


CHECKS: tuple[Check, ...] = (
    PlaceholderPhraseCheck(),
    TypesInDocstringCheck(),
    WorkNarrativeCheck(),
    EmptyNumpydocSectionCheck(),
)


# ---------------------------------------------------------------------
# AST traversal
# ---------------------------------------------------------------------


def _iter_documented_nodes(
    tree: ast.AST,
) -> Iterator[tuple[str, ast.AST, str, int]]:
    """Yield ``(qualname, node, docstring, docstring_start_line)`` tuples.

    Visits every module / class / function / async-function node that has
    a docstring (i.e. a string literal as its first statement).  Skips
    nodes whose docstring is ``None`` -- those are checked by a separate
    "missing docstring" rule, which is intentionally not included in
    this tool (see CHECKS list).
    """
    qualname_stack: list[str] = []

    def visit(node: ast.AST) -> Iterator[tuple[str, ast.AST, str, int]]:
        if isinstance(node, ast.Module):
            ds = ast.get_docstring(node, clean=False)
            if ds is not None:
                yield "<module>", node, ds, _docstring_lineno(node)
            for child in node.body:
                yield from visit(child)
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualname_stack.append(node.name)
            ds = ast.get_docstring(node, clean=False)
            if ds is not None:
                qualname = ".".join(qualname_stack)
                yield qualname, node, ds, _docstring_lineno(node)
            for child in node.body:
                yield from visit(child)
            qualname_stack.pop()
            return

    yield from visit(tree)


def _docstring_lineno(node: ast.AST) -> int:
    """Return the 1-based line number of the docstring's first line.

    ``ast.get_docstring`` returns the dedented contents but loses the
    location of the source string.  This helper walks the node body to
    find the first statement and returns its ``lineno``.
    """
    body = getattr(node, "body", None)
    if not body:
        return getattr(node, "lineno", 1)
    first = body[0]
    return getattr(first, "lineno", getattr(node, "lineno", 1))


# ---------------------------------------------------------------------
# File-level lint entry points
# ---------------------------------------------------------------------


def lint_source(
    source: str, file_path: str, checks: Iterable[Check] = CHECKS
) -> list[Issue]:
    """Run all ``checks`` over the docstrings in ``source`` (Python text).

    Parameters
    ----------
    source:
        The Python source text to lint.  Caller is responsible for
        decoding from bytes if needed.
    file_path:
        The path to record on each ``Issue`` (kept opaque; the caller
        chooses whether it's relative to the repo root or absolute).
    checks:
        Sequence of ``Check`` instances to run.  Defaults to ``CHECKS``.

    Returns
    -------
    A list of ``Issue`` instances, in the order discovered (file order,
    then check order).
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return [
            Issue(
                file=file_path,
                qualname="<module>",
                check_id="SYNTAX",
                severity="error",
                line=exc.lineno or 1,
                message=f"syntax error: {exc.msg}",
            )
        ]

    issues: list[Issue] = []
    for qualname, node, docstring, docstring_start in _iter_documented_nodes(tree):
        is_private = _is_private_qualname(qualname)
        severity = "warning" if is_private else "error"
        for check in checks:
            for offset, message, snippet, hint in check.run(docstring, node, qualname):
                issues.append(
                    Issue(
                        file=file_path,
                        qualname=qualname,
                        check_id=check.id,
                        severity=severity,
                        line=docstring_start + offset - 1,
                        message=message,
                        snippet=snippet,
                        hint=hint,
                        rule_ref=check.rule_ref,
                    )
                )
    return issues


def lint_path(
    path: Path,
    repo_root: Path,
    checks: Iterable[Check] = CHECKS,
) -> list[Issue]:
    """Recursively lint every ``.py`` file under ``path`` (or just ``path`` if a file)."""
    if path.is_file():
        files = [path] if path.suffix == ".py" else []
    else:
        files = sorted(path.rglob("*.py"))
    issues: list[Issue] = []
    for file in files:
        try:
            source = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(
                Issue(
                    file=str(file.relative_to(repo_root)),
                    qualname="<module>",
                    check_id="SYNTAX",
                    severity="error",
                    line=1,
                    message=f"could not read file: {exc}",
                )
            )
            continue
        rel = str(file.relative_to(repo_root)).replace("\\", "/")
        issues.extend(lint_source(source, rel, checks))
    return issues


# ---------------------------------------------------------------------
# Comparison-against-base mode
# ---------------------------------------------------------------------


def _git(*args: str, repo_root: Path) -> str:
    """Run ``git`` in ``repo_root`` and return stdout (decoded UTF-8)."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _files_at_ref(ref: str, paths: list[Path], repo_root: Path) -> list[str]:
    """Return repo-relative paths of ``.py`` files present at ``ref`` under ``paths``."""
    out = _git("ls-tree", "-r", "--name-only", ref, repo_root=repo_root)
    candidates = [p.replace("\\", "/") for p in out.splitlines() if p.endswith(".py")]
    rel_roots = [
        str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        for p in paths
    ]
    return [
        f for f in candidates if any(f == r or f.startswith(r + "/") for r in rel_roots)
    ]


def _read_file_at_ref(ref: str, file_rel: str, repo_root: Path) -> str | None:
    """Return the UTF-8 source of ``file_rel`` at ``ref``, or ``None`` if missing."""
    try:
        return _git("show", f"{ref}:{file_rel}", repo_root=repo_root)
    except RuntimeError:
        return None


def lint_against_base(
    paths: list[Path], base_ref: str, repo_root: Path
) -> tuple[list[Issue], list[Issue]]:
    """Return ``(new_issues, removed_issues)`` between HEAD and ``base_ref``.

    ``new_issues`` are issues present at HEAD whose ``key()`` is not in
    the base set.  ``removed_issues`` are issues present at the base ref
    whose ``key()`` is not in the HEAD set (informational; reported as
    "fixed" in the human-readable output).
    """
    head_issues = []
    for path in paths:
        head_issues.extend(lint_path(path, repo_root))

    base_issues: list[Issue] = []
    base_files = set(_files_at_ref(base_ref, paths, repo_root))
    for file_rel in sorted(base_files):
        source = _read_file_at_ref(base_ref, file_rel, repo_root)
        if source is None:
            continue
        base_issues.extend(lint_source(source, file_rel))

    base_keys = {issue.key() for issue in base_issues}
    head_keys = {issue.key() for issue in head_issues}

    new_issues = [issue for issue in head_issues if issue.key() not in base_keys]
    removed_issues = [issue for issue in base_issues if issue.key() not in head_keys]
    return new_issues, removed_issues


# ---------------------------------------------------------------------
# Output formatting + CLI
# ---------------------------------------------------------------------


def _format_text(issues: list[Issue], header: str = "") -> str:
    """Render ``issues`` as a multi-line, informative report.

    Each issue is formatted as a 4-line block::

        path/to/file.py:42  ERROR  DOC001  in `Class.method`
          placeholder marker (numpydoc-stub residue)
          > "    <placeholder>"
          fix: Replace the placeholder with the actual content, or remove the line.
          (rule: memory: feedback_docstring_no_placeholder_phrases.md)

    Issues are sorted by (file, line, check_id, qualname) for stable
    diffs across runs.
    """
    if not issues:
        return ""
    out: list[str] = []
    if header:
        out.append(header)
        out.append("")
    for issue in sorted(issues, key=lambda i: (i.file, i.line, i.check_id, i.qualname)):
        sev = issue.severity.upper()
        # Pad severity to 7 chars so the check-id column lines up across
        # ERROR/WARNING.
        sev_padded = f"{sev:7}"
        out.append(
            f"{issue.file}:{issue.line}  {sev_padded}  {issue.check_id}  "
            f"in `{issue.qualname}`"
        )
        out.append(f"  {issue.message}")
        if issue.snippet:
            # Truncate excessively long snippets so the report stays
            # readable; users who want the full context can open the
            # file at the line shown above.  Use plain double-quote
            # wrapping rather than ``repr()`` so non-ASCII characters
            # (Greek letters, mathematical symbols common in scqubits
            # docstrings) survive intact rather than being rendered
            # as ``\u`` escape sequences.
            snippet = issue.snippet
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            # Replace embedded double quotes so the wrapper stays balanced.
            snippet_safe = snippet.replace('"', '\\"')
            out.append(f'  > "{snippet_safe}"')
        if issue.hint:
            # Wrap the hint at ~78 chars for terminal readability.
            wrapped = _wrap(issue.hint, width=78, indent="    ")
            out.append(f"  fix: {wrapped.lstrip()}")
        if issue.rule_ref:
            out.append(f"  (rule: {issue.rule_ref})")
        out.append("")  # blank line between issues
    return "\n".join(out).rstrip()


def _wrap(text: str, width: int, indent: str) -> str:
    """Soft-wrap ``text`` to ``width`` columns; continuation lines get ``indent``."""
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and current_len + added > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added
    if current:
        lines.append(" ".join(current))
    return ("\n" + indent).join(lines)


def _format_json(new: list[Issue], removed: list[Issue]) -> str:
    return json.dumps(
        {
            "new_issues": [dataclasses.asdict(i) for i in new],
            "removed_issues": [dataclasses.asdict(i) for i in removed],
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint Python docstrings for scqubits compliance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to lint (recursive).",
    )
    parser.add_argument(
        "--compare-to",
        metavar="REF",
        help=(
            "Compare against the docstring state at the given git ref "
            "(e.g. 'origin/main'); only docstring issues NEW relative to "
            "the base ref are reported as failures.  Without this flag "
            "all issues at HEAD are reported."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).  "
        "Used to make file paths relative and to invoke git.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (useful for local pre-commit runs).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    paths = [p.resolve() for p in args.paths]

    new_issues: list[Issue]
    removed_issues: list[Issue]
    if args.compare_to:
        try:
            new_issues, removed_issues = lint_against_base(
                paths, args.compare_to, repo_root
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        new_issues = []
        for path in paths:
            new_issues.extend(lint_path(path, repo_root))
        removed_issues = []

    if args.format == "json":
        print(_format_json(new_issues, removed_issues))
    else:
        if removed_issues:
            print(
                _format_text(
                    removed_issues,
                    header=f"FIXED ({len(removed_issues)} issues no longer present at HEAD):",
                )
            )
            print()
        if new_issues:
            header = (
                f"NEW DOCSTRING ISSUES "
                f"({sum(1 for i in new_issues if i.severity == 'error')} errors, "
                f"{sum(1 for i in new_issues if i.severity == 'warning')} warnings):"
            )
            print(_format_text(new_issues, header=header))
        else:
            print("no new docstring issues")

    if args.strict:
        return 1 if new_issues else 0
    return 1 if any(i.severity == "error" for i in new_issues) else 0


if __name__ == "__main__":
    sys.exit(main())
