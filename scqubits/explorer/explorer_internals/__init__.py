# explorer_internals — private subpackage carrying the per-PlotType
# PanelBuilder classes and the registry that the Explorer widget uses
# for dispatch.  See ``scqubits/explorer/explorer_widget.py`` for the
# public ``Explorer`` entry point.
"""Private subpackage hosting per-PlotType panel builders and registry."""

from __future__ import annotations

from scqubits.explorer.explorer_internals._base import PanelBuilder
from scqubits.explorer.explorer_internals.energy_spectrum import (
    EnergySpectrumPanelBuilder,
)
from scqubits.ui.gui_defaults import PlotType

__all__ = [
    "PANEL_BUILDERS",
    "PanelBuilder",
    "EnergySpectrumPanelBuilder",
]

# Registry: every ``PlotType`` value gets one entry as plot types are
# migrated from the legacy ``if/elif`` chains in
# ``explorer_widget.build_panel`` / ``explorer_settings.build_settings_ui``.
# Until all are migrated, callers of those methods fall through to the
# legacy chain for any ``PlotType`` not present here.
PANEL_BUILDERS: dict[PlotType, type[PanelBuilder]] = {
    PlotType.ENERGY_SPECTRUM: EnergySpectrumPanelBuilder,
}
