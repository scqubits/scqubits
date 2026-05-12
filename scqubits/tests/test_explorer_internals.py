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

import scqubits as scq

from scqubits.explorer.explorer_internals import PANEL_BUILDERS, PanelBuilder
from scqubits.explorer.explorer_widget import PlotID
from scqubits.ui.gui_defaults import PlotType


def test_explorer_internals_package_importable():
    """The private subpackage imports cleanly even before any builder is registered."""
    from scqubits.explorer import explorer_internals  # noqa: F401


def test_registry_complete():
    """Every migrated ``PlotType`` has a registered ``PanelBuilder``.

    During the phased migration this asserts the migrated subset only;
    once Phase 2 finishes the assertion tightens to "every member of
    ``PlotType`` has a builder".
    """
    assert PlotType.ENERGY_SPECTRUM in PANEL_BUILDERS
    for key in PANEL_BUILDERS:
        assert isinstance(
            key, PlotType
        ), f"PANEL_BUILDERS key {key!r} is not a PlotType member"


def test_panel_builder_protocol_conformance():
    """Every registered class satisfies the ``PanelBuilder`` protocol."""
    for plot_type, builder_cls in PANEL_BUILDERS.items():
        builder = builder_cls()
        assert isinstance(
            builder, PanelBuilder
        ), f"{builder_cls.__name__} does not satisfy the PanelBuilder protocol"
        assert builder.plot_type is plot_type, (
            f"{builder_cls.__name__}.plot_type ({builder.plot_type!r}) "
            f"does not match its registry key ({plot_type!r})"
        )


def test_plot_id_value_object():
    """``PlotID`` exposes ``is_composite``, ``subsys_ids``, and ``is_default_active``."""
    tmon = scq.TunableTransmon(
        EJmax=40.0, EC=0.2, d=0.1, flux=0.0, ng=0.3, ncut=40, truncated_dim=5
    )
    resonator = scq.Oscillator(E_osc=4.5, truncated_dim=4)

    single = PlotID(PlotType.ENERGY_SPECTRUM, [tmon])
    composite = PlotID(PlotType.TRANSITIONS, [tmon, resonator])

    assert not single.is_composite()
    assert composite.is_composite()

    assert single.subsys_ids() == [tmon.id_str]
    assert composite.subsys_ids() == [tmon.id_str, resonator.id_str]

    # ENERGY_SPECTRUM IS a default for TunableTransmon (it's in `common_panels`).
    assert single.is_default_active()
    # TRANSITIONS IS the default for any Composite plot.
    assert composite.is_default_active()
