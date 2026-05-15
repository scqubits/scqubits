"""ANHARMONICITY panel: third-energy-level anharmonicity vs the swept parameter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.core.units as units

from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QubitBaseClass
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID


@rc_context(matplotlib_settings)
def display_anharmonicity(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    """Plot anharmonicity ``E_2 - 2 E_1 + E_0`` of ``subsys`` across the swept parameter."""
    subsys_index = sweep.get_subsys_index(subsys)

    bare_evals = sweep["bare_evals"]["subsys":subsys_index][param_slice.fixed]  # type: ignore[misc]
    anharmonicity = bare_evals[..., 2] - 2 * bare_evals[..., 1] + bare_evals[..., 0]

    title = f"Anharmonicity: {subsys.id_str}"
    fig, axes = anharmonicity.plot(
        title=title,
        ylabel=f"anharmonicity [{units.get_units()}]",
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


class AnharmonicityPanelBuilder:
    """Builder for the ANHARMONICITY panel (no per-plot settings)."""

    plot_type: ClassVar[PlotType] = PlotType.ANHARMONICITY
    slider_invariant: ClassVar[bool] = True

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        return display_anharmonicity(
            explorer.sweep,
            plot_id.subsystems[0],  # type: ignore[arg-type]
            param_slice,
            fig_ax,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        return []
