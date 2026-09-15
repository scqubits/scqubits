# `tools/`

Maintenance tools that run against the scqubits source tree but are not
part of the published package.

## `docstring_lint.py` --- docstring-compliance linter

Walks Python files and flags docstrings that violate the project's
documented requirements: placeholder phrases (`_type_`,
`_description_`, `<TODO>`), type annotations duplicated into the
docstring (when the signature already declares them), refactor-history
narrative, and empty numpydoc section headers.

### Usage

```bash
# Report every issue at HEAD.  Useful for a one-off audit.
python tools/docstring_lint.py scqubits/

# Compare against a base ref.  Only issues NEW relative to the ref
# are reported; pre-existing issues are silently inherited.  This is
# the mode the GitHub Actions workflow uses.
python tools/docstring_lint.py --compare-to origin/main scqubits/

# Machine-readable output for tooling integration.
python tools/docstring_lint.py --format json scqubits/

# Treat warnings as errors (useful for local pre-commit runs).
python tools/docstring_lint.py --strict scqubits/
```

### Rules

| ID      | What it catches                                        |
|---------|--------------------------------------------------------|
| DOC001  | Placeholder phrases (`_type_`, `_description_`, `<TODO>`, `<FIXME>`, "return shape") |
| DOC002  | Type annotation duplicated into a numpydoc Parameters or Returns section |
| DOC003  | Work-narrative phrasing ("we used to", "see commit", "by Claude", ...) |
| DOC004  | numpydoc Parameters/Returns/Raises/Yields section header with empty body |

### Severity

For symbols whose qualified name contains any `_`-prefixed
component (but not `__dunder__`), all issues are downgraded to
**warnings** rather than errors. Public symbols produce **errors**.
The exit code is non-zero only if at least one error is present
(warnings alone are tolerated).

`--strict` overrides this: warnings count as errors too.

### Design

- Stdlib-only; no third-party dependencies. Runs in any Python
  environment.
- Single file (`docstring_lint.py`); easy to read, easy to extend.
- Each rule is a `Check` subclass that yields
  `(line_offset, message, snippet, hint)` tuples. Adding a new rule
  means appending a `Check` to the `CHECKS` tuple and writing a
  fixture in `test_docstring_lint.py`.
- Issues are identified by `(file, qualname, check_id, message)` --- not
  by line number --- so cosmetic edits don't show up as new issues in
  compare-against-base mode.

### Tests

```bash
python -m pytest tools/test_docstring_lint.py
```

48 tests covering each rule, the privacy-detection logic, the
numpydoc section parser, and an end-to-end smoke run on a synthetic
fixture.

### CI integration

`.github/workflows/docstring-lint.yml` runs the linter on every PR
that touches `scqubits/**/*.py`. The workflow uses
`--compare-to origin/main`, so only docstring issues *newly
introduced* by the PR fail the build. Pre-existing issues at `main`
remain reported in informational output but do not block merging.
