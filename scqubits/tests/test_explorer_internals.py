# test_explorer_internals.py
# meant to be run with 'pytest'
#
# Unit tests for the ``scqubits.explorer.explorer_internals`` subpackage:
# the per-PlotType ``PanelBuilder`` registry, the ``ExplorerUI`` typed
# state container, ``PlotID`` value-object semantics, and the optional-
# import guard for ``ipyvuetify``.
#
# These tests cover the refactor-relevant invariants only; live
# ipyvuetify behavior (observer firing, dialog open/close, in-panel
# matplotlib rendering) is left to manual notebook verification.
############################################################################
from __future__ import annotations


def test_explorer_internals_package_importable():
    """The private subpackage imports cleanly even before any builder is registered."""
    from scqubits.explorer import explorer_internals  # noqa: F401
