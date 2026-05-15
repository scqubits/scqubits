"""Single import point for optional UI dependencies.

Consolidates the ``_HAS_IPYVUETIFY`` / ``_HAS_IPYTHON`` availability
flags plus the optional module references (``v``, ``display``, etc.)
that the Explorer subsystem reaches for.  Files that previously
duplicated the ``try: import ipyvuetify as v; except ImportError:
_HAS_IPYVUETIFY = False; else: _HAS_IPYVUETIFY = True`` boilerplate
can now ``from scqubits.utils._optional_deps import v, _HAS_IPYVUETIFY``.

When a dependency is missing, the corresponding name is set to
``None`` and the ``_HAS_*`` flag is ``False``.  Call sites that
would crash on the ``None`` value must be gated either by
:class:`scqubits.utils.misc.Required` (the decorator pattern used
throughout this subpackage) or by an explicit ``if`` guard against
the flag.
"""

from __future__ import annotations

from typing import Any

# ipyvuetify + ipywidgets bundle.  The ``Any``-typed forward
# declarations let the same name absorb both the module-object case
# (success branch) and the ``None`` fallback (failure branch); mypy
# would otherwise complain about redefinition.
try:
    import ipyvuetify
    import ipywidgets as _ipywidgets
    import traitlets as _traitlets
except ImportError:
    _HAS_IPYVUETIFY = False
    v: Any = None
    ipywidgets: Any = None
    traitlets: Any = None
else:
    _HAS_IPYVUETIFY = True
    v = ipyvuetify
    ipywidgets = _ipywidgets
    traitlets = _traitlets

# IPython display: same pattern as above.
try:
    from IPython.display import Latex as _Latex
    from IPython.display import display as _display
except ImportError:
    _HAS_IPYTHON = False
    Latex: Any = None
    display: Any = None
else:
    _HAS_IPYTHON = True
    Latex = _Latex
    display = _display
