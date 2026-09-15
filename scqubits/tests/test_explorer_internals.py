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

import dataclasses

import scqubits as scq

from scqubits.explorer.explorer_internals import PANEL_BUILDERS, PanelBuilder
from scqubits.explorer.explorer_internals._state import ExplorerUI
from scqubits.explorer.explorer_widget import PlotID
from scqubits.ui.gui_defaults import PlotType


def test_explorer_internals_package_importable():
    """The private subpackage imports cleanly even before any builder is registered."""
    from scqubits.explorer import explorer_internals  # noqa: F401


def test_registry_complete():
    """Every member of ``PlotType`` has a registered ``PanelBuilder``."""
    missing = set(PlotType) - set(PANEL_BUILDERS)
    assert (
        not missing
    ), f"PlotType members without a builder: {sorted(m.name for m in missing)}"
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


def test_default_panel_selection_for_known_topologies():
    """``PlotID.is_default_active`` matches the ``default_panels`` lookup table.

    Sweep topology: ``(TunableTransmon, Oscillator)``.  Expected default
    panels follow from ``gui_defaults.default_panels`` ---
    ``common_panels`` for the qubit, ``[]`` for the oscillator, and
    ``[TRANSITIONS]`` for the composite system.
    """
    tmon = scq.TunableTransmon(
        EJmax=40.0, EC=0.2, d=0.1, flux=0.0, ng=0.3, ncut=40, truncated_dim=5
    )
    resonator = scq.Oscillator(E_osc=4.5, truncated_dim=4)

    # Qubit: ENERGY_SPECTRUM + WAVEFUNCTIONS are defaults; others are not.
    assert PlotID(PlotType.ENERGY_SPECTRUM, [tmon]).is_default_active()
    assert PlotID(PlotType.WAVEFUNCTIONS, [tmon]).is_default_active()
    assert not PlotID(PlotType.ANHARMONICITY, [tmon]).is_default_active()
    assert not PlotID(PlotType.SELF_KERR, [tmon]).is_default_active()

    # Oscillator: no defaults.
    assert not PlotID(PlotType.ENERGY_SPECTRUM, [resonator]).is_default_active()
    assert not PlotID(PlotType.SELF_KERR, [resonator]).is_default_active()

    # Composite: only TRANSITIONS.
    assert PlotID(PlotType.TRANSITIONS, [tmon, resonator]).is_default_active()
    assert not PlotID(PlotType.CROSS_KERR, [tmon, resonator]).is_default_active()
    assert not PlotID(PlotType.AC_STARK, [tmon, resonator]).is_default_active()


_EXPECTED_EXPLORER_UI_FIELDS = frozenset(
    {
        "add_plot_dialog",
        "sweep_param_dropdown",
        "sweep_value_slider",
        "param_sliders",
        "param_sliders_container",
        "fixed_param_sliders",
        "top_bar",
        "panel_switch_by_plot_id",
        "panel_switches_by_subsys_name",
        "panel_switches",
    }
)


def test_explorer_ui_dataclass_fields():
    """``ExplorerUI`` exposes every field the Explorer writes to during ``__init__``.

    Catches silent drift between ``Explorer.__init__`` and the dataclass:
    if someone adds a new ``self.ui.<foo> = ...`` assignment in
    ``explorer_widget.py``, this test fails until ``foo`` is declared on
    the dataclass.
    """
    actual = {f.name for f in dataclasses.fields(ExplorerUI)}
    assert actual == _EXPECTED_EXPLORER_UI_FIELDS, (
        f"ExplorerUI fields drifted from the expected set.\n"
        f"  Missing from dataclass: {_EXPECTED_EXPLORER_UI_FIELDS - actual}\n"
        f"  Unexpected on dataclass: {actual - _EXPECTED_EXPLORER_UI_FIELDS}"
    )


def test_explorer_ui_default_instance_is_empty():
    """``ExplorerUI()`` constructs successfully with all-default values."""
    ui = ExplorerUI()
    for f in dataclasses.fields(ExplorerUI):
        value = getattr(ui, f.name)
        if f.name in {
            "param_sliders",
            "fixed_param_sliders",
            "panel_switch_by_plot_id",
            "panel_switches_by_subsys_name",
            "panel_switches",
        }:
            assert value == {}, f"{f.name} default is not an empty dict"
        else:
            assert value is None, f"{f.name} default is not None"


def test_optional_deps_module_exposes_expected_names():
    """``scqubits.utils._optional_deps`` is the single source of truth for
    the ipyvuetify / IPython availability flags + module references.

    Catches accidental name drops (or silent renames) that would force
    callers back to per-file ``try/except ImportError`` boilerplate.
    """
    from scqubits.utils import _optional_deps

    for name in (
        "_HAS_IPYVUETIFY",
        "_HAS_IPYTHON",
        "v",
        "ipywidgets",
        "display",
    ):
        assert hasattr(
            _optional_deps, name
        ), f"scqubits.utils._optional_deps missing expected symbol: {name!r}"

    assert isinstance(_optional_deps._HAS_IPYVUETIFY, bool)
    assert isinstance(_optional_deps._HAS_IPYTHON, bool)
