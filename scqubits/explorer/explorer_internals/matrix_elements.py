"""MATRIX_ELEMENTS panel: matrix-element heatmap at the current parameter slice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.core.constants
import scqubits.ui.gui_custom_widgets as ui

from scqubits.core.qubit_base import QubitBaseClass
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType, mode_dropdown_dict, mode_dropdown_list

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID


@rc_context(matplotlib_settings)
def display_matrixelements(
    sweep: "ParameterSweep",
    operator_name: str,
    subsys: "QubitBaseClass",
    param_slice: "ParameterSlice",
    mode_str: str,
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    """Render the matrix-element heatmap for ``operator_name`` of ``subsys``."""
    subsys_index = sweep.get_subsys_index(subsys)
    evecs = sweep["bare_evecs"][subsys_index][param_slice.all]

    fig, axes = fig_ax
    axes.cla()

    title = f"{subsys.id_str}: matrix elements (fixed)"
    # show3d=False makes plot_matrixelements return tuple[Figure, Axes]
    fig, axes = subsys.plot_matrixelements(  # type: ignore[assignment]
        operator_name,
        evecs,
        evals_count=subsys.truncated_dim,
        mode=mode_str,
        show3d=False,
        show_numbers=False,
        show_colorbar=True,
        fig_ax=fig_ax,
        title=title,
    )
    axes.text(
        0.5,
        1.05,
        f"{scqubits.core.constants.MODE_STR_DICT[mode_str](operator_name)}",
        fontsize=9,
        fontweight=300,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=axes.transAxes,
    )
    return fig, axes


class MatrixElementsPanelBuilder:
    """Builder for the MATRIX_ELEMENTS panel (qubit subsystems only)."""

    plot_type: ClassVar[PlotType] = PlotType.MATRIX_ELEMENTS

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        subsys = plot_id.subsystems[0]
        assert isinstance(subsys, QubitBaseClass), (
            "MATRIX_ELEMENTS panel requires a QubitBaseClass subsystem; "
            f"got {type(subsys).__name__}."
        )
        ui_mode_dropdown, opname_dropdown = explorer.settings[plot_id]
        return display_matrixelements(
            sweep=explorer.sweep,
            subsys=subsys,
            operator_name=opname_dropdown.v_model,
            mode_str=mode_dropdown_dict[ui_mode_dropdown.v_model],
            param_slice=param_slice,
            fig_ax=fig_ax,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        subsys = plot_id.subsystems[0]
        ui_mode_dropdown = ui.InitializedSelect(
            items=mode_dropdown_list,
            label="Plot matrix elements as",
            v_model=mode_dropdown_list[2],
        )
        op_names = subsys.get_operator_names()
        ui_operator_dropdown = ui.InitializedSelect(
            items=op_names, label="Operator", v_model=op_names[0]
        )
        ui_mode_dropdown.observe(settings.explorer.update_plots, names="v_model")
        ui_operator_dropdown.observe(settings.explorer.update_plots, names="v_model")
        return [ui_mode_dropdown, ui_operator_dropdown]
