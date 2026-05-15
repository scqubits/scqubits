"""AC_STARK panel: AC Stark shift of one qubit level driven by a single oscillator.

Render function ``display_cross_kerr`` is shared with the
:class:`CrossKerrPanelBuilder`; AC Stark is the
single-oscillator-among-the-pair branch of that function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.ui.gui_custom_widgets as ui

from scqubits.explorer.explorer_internals.cross_kerr import display_cross_kerr
from scqubits.ui.gui_defaults import PlotType

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID

from scqubits.utils._optional_deps import _HAS_IPYVUETIFY, v


class AcStarkPanelBuilder:
    """Builder for the AC_STARK panel (qubit-oscillator pair only)."""

    plot_type: ClassVar[PlotType] = PlotType.AC_STARK

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        return display_cross_kerr(
            sweep=explorer.sweep,
            subsys1=plot_id.subsystems[0],
            subsys2=plot_id.subsystems[1],
            param_slice=param_slice,
            fig_ax=fig_ax,
            which=explorer.settings.ui["kerr"]["ac_stark_ell"].v_model,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        subsys = plot_id.subsystems[0]
        settings.ui["kerr"]["ac_stark_ell"] = ui.InitializedSelect(
            v_model=1,
            items=list(range(1, subsys.truncated_dim)),
            label="qubit level",
        )
        settings.ui["kerr"]["ac_stark_ell"].observe(
            settings.explorer.update_plots, names="v_model"
        )

        return [
            v.Container(
                class_="d-flex flex-column",
                children=[settings.ui["kerr"]["ac_stark_ell"]],
            )
        ]
