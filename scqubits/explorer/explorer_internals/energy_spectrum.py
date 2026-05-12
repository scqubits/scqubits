"""ENERGY_SPECTRUM panel: bare eigenenergies of one subsystem vs the swept parameter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.core.units as units
import scqubits.ui.gui_custom_widgets as ui

from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType

if TYPE_CHECKING:
    from scqubits.core.oscillator import Oscillator
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QubitBaseClass
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID

try:
    import ipyvuetify as v
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True


@rc_context(matplotlib_settings)
def display_bare_spectrum(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass | Oscillator",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
    evals_count: int | None = None,
    subtract_ground: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the bare spectrum of ``subsys`` across the swept parameter."""
    subsys_index = sweep.get_subsys_index(subsys)
    title = "Bare Spectrum: {}\n".format(subsys.id_str)

    evals_count = evals_count or -1

    bare_evals = sweep["bare_evals"]["subsys":subsys_index][param_slice.fixed]  # type: ignore[misc]
    if subtract_ground:
        bare_evals = bare_evals - bare_evals[:, 0, np.newaxis]

    fig, axes = bare_evals[:, 0:evals_count].plot(
        title=title,
        ylabel="energy [{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


class EnergySpectrumPanelBuilder:
    """Builder for the ENERGY_SPECTRUM panel."""

    plot_type: ClassVar[PlotType] = PlotType.ENERGY_SPECTRUM

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        panel_widget = explorer.settings[plot_id]
        return display_bare_spectrum(
            explorer.sweep,
            plot_id.subsystems[0],  # type: ignore[arg-type]
            param_slice,
            fig_ax,
            subtract_ground=panel_widget[1].v_model,
            evals_count=explorer.settings["level_slider"][plot_id].num_value,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        subsys = plot_id.subsystems[0]
        subsys_index = settings.explorer.sweep.get_subsys_index(subsys)
        evals_count = settings.explorer.sweep.subsys_evals_count(subsys_index)
        settings.ui["level_slider"][plot_id] = ui.NumberEntryWidget(
            num_type=int,
            label="Highest level",
            v_min=1,
            v_max=evals_count,
            v_model=evals_count,
            text_kwargs={
                "style_": "min-width: 140px; max-width: 200px;",
                "dense": True,
            },
            slider_kwargs={
                "style_": "min-width: 110px; max-width: 230px",
                "dense": True,
            },
        )
        ui_subtract_ground_switch = v.Switch(
            label="Subtract E₀", v_model=True, width=300
        )
        settings.ui["level_slider"][plot_id].observe(
            settings.explorer.update_plots, names="v_model"
        )
        ui_subtract_ground_switch.observe(
            settings.explorer.update_plots, names="v_model"
        )
        return [
            settings.ui["level_slider"][plot_id].widget(),
            ui_subtract_ground_switch,
        ]
