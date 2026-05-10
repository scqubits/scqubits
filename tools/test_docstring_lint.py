"""Unit tests for ``tools/docstring_lint.py``.

Run with ``python -m pytest tools/test_docstring_lint.py``.

The tests exercise each ``Check`` against hand-crafted docstring
fixtures and assert that the right ``Issue`` instances are produced
(and importantly, that *only* those are produced — no false positives
on benign numpydoc).  ``compare-to`` mode is exercised separately by a
short integration scenario that monkey-patches the git layer.
"""

from __future__ import annotations

import textwrap

import pytest

from docstring_lint import (
    CHECKS,
    EmptyNumpydocSectionCheck,
    Issue,
    PlaceholderPhraseCheck,
    TypesInDocstringCheck,
    WorkNarrativeCheck,
    _is_private_qualname,
    _iter_numpydoc_sections,
    _looks_like_a_type,
    _looks_like_bare_type_line,
    lint_source,
)


# ---------------------------------------------------------------------
# Privacy detection
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "qualname,expected",
    [
        ("Circuit", False),
        ("Circuit.configure", False),
        ("Circuit._configure", True),
        ("_PrivateClass", True),
        ("_PrivateClass.method", True),
        ("PublicClass.__init__", False),  # dunder is public
        ("PublicClass.__set__", False),  # dunder is public
        ("PublicClass._private_method", True),
        ("<module>", False),
        ("PublicClass.__name", True),  # name-mangled (no trailing dunder)
    ],
)
def test_is_private_qualname(qualname: str, expected: bool) -> None:
    assert _is_private_qualname(qualname) is expected


# ---------------------------------------------------------------------
# Numpydoc section parser
# ---------------------------------------------------------------------


def test_iter_numpydoc_sections_finds_parameters_and_returns() -> None:
    docstring = textwrap.dedent(
        """\
        Summary line.

        Parameters
        ----------
        x:
            description

        Returns
        -------
        the result
        """
    )
    sections = list(_iter_numpydoc_sections(docstring))
    names = [s[0] for s in sections]
    assert names == ["Parameters", "Returns"]


def test_iter_numpydoc_sections_ignores_prose_with_section_word() -> None:
    """The word 'Parameters' in a sentence isn't a section header."""
    docstring = "Returns the input parameters as a list."
    assert list(_iter_numpydoc_sections(docstring)) == []


def test_iter_numpydoc_sections_requires_underline() -> None:
    """A header word without the dashed underline isn't a section."""
    docstring = "Parameters\n\nThis is just prose, no underline."
    assert list(_iter_numpydoc_sections(docstring)) == []


# ---------------------------------------------------------------------
# Type-detection heuristics
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "rhs,expected",
    [
        ("int", True),
        ("str", True),
        ("dict[str, int]", True),
        ("Optional[Foo]", True),
        ("ndarray | None", True),
        ("ClassName", True),
        ("int = 5", True),  # default-value type
        ("the input integer", False),
        ("see :func:`foo` for details", False),
        ("a length-N list of floats", False),
        # Regression tests for false positives caught in reviewer audit:
        ("None when not applicable", False),
        ("None or empty", False),
        ("int input value", False),
    ],
)
def test_looks_like_a_type(rhs: str, expected: bool) -> None:
    assert _looks_like_a_type(rhs) is expected


@pytest.mark.parametrize(
    "line,expected",
    [
        ("ndarray", True),
        ("ndarray | None", True),
        ("dict[str, int]", True),
        ("the result", False),
        ("List of floats.", False),
        ("a number", False),
        # Regression tests for false positives caught in reviewer audit:
        # A single CapWord (not in KNOWN_TYPE_NAMES) is more likely a
        # description than a custom-class type annotation.
        ("Length", False),
        ("Frequency", False),
        ("Identifier", False),
        # Multi-word descriptions starting with a CapWord were
        # previously flagged because every token matched ``[A-Za-z_]``.
        ("Frequency in GHz, computed from omega", False),
        ("Hamiltonian, either unchanged in native basis", False),
        ("eigenvalues, eigenvectors as numpy arrays or in form of a", False),
        # But genuine type unions / tuples should still be flagged.
        ("Qobj", True),
        ("ndarray, ndarray", True),
    ],
)
def test_looks_like_bare_type_line(line: str, expected: bool) -> None:
    assert _looks_like_bare_type_line(line) is expected


# ---------------------------------------------------------------------
# Per-check fixtures
# ---------------------------------------------------------------------


def _run_check_on_docstring(check, docstring: str):
    """Helper: feed a docstring through a check and return its 4-tuples."""
    return list(check.run(docstring, None, "Test.method"))


def test_placeholder_phrase_check_flags_type_and_description() -> None:
    docstring = textwrap.dedent(
        """\
        Summary.

        Returns
        -------
        _type_
            _description_
        """
    )
    issues = _run_check_on_docstring(PlaceholderPhraseCheck(), docstring)
    messages = [m for _, m, _, _ in issues]
    assert any("_type_" in m for m in messages)
    assert any("_description_" in m for m in messages)


def test_placeholder_phrase_check_silent_on_clean_docstring() -> None:
    docstring = "A clean summary with no placeholders."
    assert _run_check_on_docstring(PlaceholderPhraseCheck(), docstring) == []


def test_types_in_docstring_check_flags_param_with_type() -> None:
    """DOC002 flags `x : int` only when the signature already annotates `x`."""
    source = textwrap.dedent(
        """\
        def foo(x: int):
            \"\"\"Summary.

            Parameters
            ----------
            x : int
                description of x
            \"\"\"
        """
    )
    issues = lint_source(source, "fixture.py")
    doc002 = [i for i in issues if i.check_id == "DOC002"]
    assert len(doc002) == 1
    assert "`x`" in doc002[0].message
    assert "int" in doc002[0].message
    assert "x : int" in doc002[0].snippet
    assert "signature" in doc002[0].hint.lower()


def test_types_in_docstring_check_silent_when_signature_has_no_annotation() -> None:
    """If the signature has no type annotation for `x`, the docstring is the
    legitimate type source -- DOC002 must NOT flag it."""
    source = textwrap.dedent(
        """\
        def foo(x):
            \"\"\"Summary.

            Parameters
            ----------
            x : int
                description of x
            \"\"\"
        """
    )
    issues = [i for i in lint_source(source, "fixture.py") if i.check_id == "DOC002"]
    assert issues == []


def test_types_in_docstring_check_silent_on_continuation_with_colon() -> None:
    """A continuation line in a previous param's description that happens
    to start with ``word:`` (e.g. ``or: something``) is NOT a parameter
    declaration -- it's at a deeper indent than the actual parameter
    lines.  DOC002 must skip it.

    Regression test for a real false positive caught by reviewer audit.
    """
    source = textwrap.dedent(
        """\
        def foo(evals_count: int):
            \"\"\"Compute eigenvalues.

            Parameters
            ----------
            evals_count:
                number of desired eigenvalues; default: -1,
                signals all eigenvalues
                or: list of specific eigenvalues to include
            \"\"\"
        """
    )
    issues = [i for i in lint_source(source, "fixture.py") if i.check_id == "DOC002"]
    # ``or:`` at a deeper indent than ``evals_count:`` must NOT be flagged.
    assert issues == []


def test_types_in_docstring_check_silent_on_typeless_param() -> None:
    docstring = textwrap.dedent(
        """\
        Summary.

        Parameters
        ----------
        x:
            description of x (no type, on next line)
        """
    )
    assert _run_check_on_docstring(TypesInDocstringCheck(), docstring) == []


def test_types_in_docstring_check_flags_returns_with_bare_type() -> None:
    """DOC002 flags a bare-type Returns line only when the function has a
    return annotation (otherwise the docstring is the type source)."""
    source = textwrap.dedent(
        """\
        def foo() -> object:
            \"\"\"Summary.

            Returns
            -------
            ndarray | None
            \"\"\"
        """
    )
    issues = [i for i in lint_source(source, "fixture.py") if i.check_id == "DOC002"]
    assert len(issues) == 1
    assert "Returns" in issues[0].message
    assert issues[0].snippet == "ndarray | None"


def test_types_in_docstring_check_silent_on_returns_when_no_return_annotation() -> None:
    """If the signature has no `-> ...` return annotation, the docstring's
    bare-type Returns line is the legitimate type source."""
    source = textwrap.dedent(
        """\
        def foo():
            \"\"\"Summary.

            Returns
            -------
            ndarray | None
            \"\"\"
        """
    )
    issues = [i for i in lint_source(source, "fixture.py") if i.check_id == "DOC002"]
    assert issues == []


def test_types_in_docstring_check_silent_on_returns_with_description() -> None:
    """``Returns`` section starting with prose (not a type) is fine."""
    source = textwrap.dedent(
        """\
        def foo() -> object:
            \"\"\"Summary.

            Returns
            -------
            the converted value, or ``None`` if unavailable.
            \"\"\"
        """
    )
    issues = [i for i in lint_source(source, "fixture.py") if i.check_id == "DOC002"]
    assert issues == []


def test_work_narrative_check_flags_we_used_to() -> None:
    docstring = "We used to compute this differently."
    issues = _run_check_on_docstring(WorkNarrativeCheck(), docstring)
    assert len(issues) == 1
    _, message, _, _ = issues[0]
    assert "we used to" in message.lower()


def test_work_narrative_check_flags_see_commit() -> None:
    docstring = "See commit abc123 for context."
    issues = _run_check_on_docstring(WorkNarrativeCheck(), docstring)
    assert len(issues) == 1


def test_work_narrative_check_silent_on_normal_prose() -> None:
    docstring = (
        "Compute the eigenvalues of the Hamiltonian. "
        "Uses the diagonalization routine described in the paper."
    )
    assert _run_check_on_docstring(WorkNarrativeCheck(), docstring) == []


def test_empty_numpydoc_section_check_flags_empty_parameters() -> None:
    docstring = textwrap.dedent(
        """\
        Summary.

        Parameters
        ----------

        Notes
        -----
        Body of notes.
        """
    )
    issues = _run_check_on_docstring(EmptyNumpydocSectionCheck(), docstring)
    assert len(issues) == 1
    _, message, _, _ = issues[0]
    assert "`Parameters`" in message


def test_empty_numpydoc_section_check_silent_on_filled_section() -> None:
    docstring = textwrap.dedent(
        """\
        Summary.

        Parameters
        ----------
        x:
            description
        """
    )
    assert _run_check_on_docstring(EmptyNumpydocSectionCheck(), docstring) == []


# ---------------------------------------------------------------------
# Severity downgrade for private symbols
# ---------------------------------------------------------------------


def test_private_qualname_downgrades_to_warning(tmp_path) -> None:
    """An issue inside a `_`-prefixed function is a warning, not error."""
    source = textwrap.dedent(
        """\
        def _private_helper():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    issues = lint_source(source, "fixture.py")
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].check_id == "DOC001"


def test_public_qualname_remains_error(tmp_path) -> None:
    source = textwrap.dedent(
        """\
        def public_helper():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    issues = lint_source(source, "fixture.py")
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_private_method_inside_public_class_is_private(tmp_path) -> None:
    """`PublicClass._method` is private (has a `_`-prefixed component)."""
    source = textwrap.dedent(
        """\
        class PublicClass:
            def _internal_method(self):
                \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    issues = lint_source(source, "fixture.py")
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_dunder_method_is_public(tmp_path) -> None:
    """`PublicClass.__init__` is public (dunder methods are public)."""
    source = textwrap.dedent(
        """\
        class PublicClass:
            def __init__(self):
                \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    issues = lint_source(source, "fixture.py")
    assert len(issues) == 1
    assert issues[0].severity == "error"


# ---------------------------------------------------------------------
# Issue-key stability across cosmetic edits
# ---------------------------------------------------------------------


def test_issue_key_excludes_line_number(tmp_path) -> None:
    """Two Issues at different lines but same (file, qualname, check, message)
    should compare equal under ``key()``.  This is what makes
    compare-to-base mode robust to line shifts."""
    a = Issue(
        file="x.py",
        qualname="foo",
        check_id="DOC001",
        severity="error",
        line=10,
        message="m",
    )
    b = Issue(
        file="x.py",
        qualname="foo",
        check_id="DOC001",
        severity="warning",  # severity also excluded from key
        line=99,
        message="m",
    )
    assert a.key() == b.key()


# ---------------------------------------------------------------------
# End-to-end on a realistic source string
# ---------------------------------------------------------------------


def test_lint_source_end_to_end_finds_expected_issues() -> None:
    source = textwrap.dedent(
        """\
        \"\"\"Module-level docstring (no issues).\"\"\"


        def public_func(x: int):
            \"\"\"Do a thing.

            Parameters
            ----------
            x : int
                the bad parameter
            \"\"\"


        def _private_func():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    issues = lint_source(source, "fixture.py")
    assert {i.check_id for i in issues} == {"DOC001", "DOC002"}
    # public_func has 1 error (DOC002 type-in-param)
    public_issues = [i for i in issues if i.qualname == "public_func"]
    assert len(public_issues) == 1
    assert public_issues[0].severity == "error"
    assert public_issues[0].check_id == "DOC002"
    # _private_func has 1 warning (DOC001 placeholder, downgraded)
    private_issues = [i for i in issues if i.qualname == "_private_func"]
    assert len(private_issues) == 1
    assert private_issues[0].severity == "warning"
    assert private_issues[0].check_id == "DOC001"


# ---------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------


def test_all_registered_checks_have_required_attrs() -> None:
    """Every check in CHECKS must declare id, description, rule_ref."""
    for check in CHECKS:
        assert check.id, f"{check.__class__.__name__} missing id"
        assert check.description, f"{check.id} missing description"
        assert check.rule_ref, f"{check.id} missing rule_ref"


def test_check_ids_are_unique() -> None:
    ids = [c.id for c in CHECKS]
    assert len(ids) == len(set(ids)), f"duplicate check ids in CHECKS: {ids}"


# ---------------------------------------------------------------------
# Compare-to-base integration
# ---------------------------------------------------------------------


def _git(repo, *args):
    """Run ``git`` in ``repo`` (a Path); raise on non-zero exit."""
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_two_commit_repo(tmp_path, base_text: str, head_text: str):
    """Create a temp git repo with two commits, return ``(repo, file_path)``."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    fpath = repo / "module.py"
    fpath.write_text(base_text, encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-m", "base")
    fpath.write_text(head_text, encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-m", "head")
    return repo, fpath


def test_compare_to_reports_only_new_issues(tmp_path) -> None:
    """End-to-end compare-to mode: an issue introduced at HEAD should be
    reported as new; the unchanged base content should produce nothing."""
    from docstring_lint import lint_against_base

    base_text = textwrap.dedent(
        """\
        def clean_func():
            \"\"\"Clean docstring with no issues.\"\"\"
        """
    )
    head_text = textwrap.dedent(
        """\
        def clean_func():
            \"\"\"Clean docstring with no issues.\"\"\"


        def buggy_func():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    repo, fpath = _make_two_commit_repo(tmp_path, base_text, head_text)
    new_issues, removed_issues = lint_against_base(
        [fpath], "HEAD~1", repo
    )
    assert len(new_issues) == 1
    assert new_issues[0].qualname == "buggy_func"
    assert new_issues[0].check_id == "DOC001"
    assert removed_issues == []


def test_compare_to_reports_removed_issues(tmp_path) -> None:
    """An issue present at the base ref but absent at HEAD must show as
    "removed" (a fix), not as new."""
    from docstring_lint import lint_against_base

    base_text = textwrap.dedent(
        """\
        def buggy_func():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    head_text = textwrap.dedent(
        """\
        def buggy_func():
            \"\"\"Clean docstring now.\"\"\"
        """
    )
    repo, fpath = _make_two_commit_repo(tmp_path, base_text, head_text)
    new_issues, removed_issues = lint_against_base(
        [fpath], "HEAD~1", repo
    )
    assert new_issues == []
    assert len(removed_issues) == 1
    assert removed_issues[0].check_id == "DOC001"


def test_compare_to_robust_to_line_shifts(tmp_path) -> None:
    """An issue that exists in both base and HEAD but has shifted to a
    different line number (because lines were inserted above it) must
    NOT be reported as new -- the issue key excludes ``line``."""
    from docstring_lint import lint_against_base

    base_text = textwrap.dedent(
        """\
        def buggy_func():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    head_text = textwrap.dedent(
        """\
        # added comment 1
        # added comment 2
        # added comment 3


        def buggy_func():
            \"\"\"Has _type_ placeholder.\"\"\"
        """
    )
    repo, fpath = _make_two_commit_repo(tmp_path, base_text, head_text)
    new_issues, removed_issues = lint_against_base(
        [fpath], "HEAD~1", repo
    )
    # Same logical issue at a different line -- not "new".
    assert new_issues == []
    assert removed_issues == []
