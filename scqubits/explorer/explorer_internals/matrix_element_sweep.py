"""MATRIX_ELEMENT_SCAN panel: matrix element magnitudes vs the swept parameter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.ui.gui_custom_widgets as ui
import scqubits.utils.plotting as plot

from scqubits import SpectrumData
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType, mode_dropdown_dict, mode_dropdown_list

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID


@rc_context(matplotlib_settings)
def display_matrixelement_sweep(
    sweep: "ParameterSweep",
    operator_name: str,
    subsys: "QubitBaseClass",
    param_slice: "ParameterSlice",
    mode_str: str,
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    """Render the matrix-element sweep curves for ``operator_name`` of ``subsys``."""
    subsys_index = sweep.get_subsys_index(subsys)
    evals = sweep["bare_evals"][subsys_index][param_slice.fixed]
    evecs = sweep["bare_evecs"][subsys_index][param_slice.fixed]

    fig, axes = fig_ax
    axes.cla()

    param_name = param_slice.param_name
    param_vals = sweep.param_info[param_name]
    paramvals_count = len(param_vals)

    specdata = SpectrumData(
        evals,
        {},
        param_slice.param_name,
        param_vals,
        state_table=evecs,
    )

    matelem_table = np.empty(
        shape=(paramvals_count, subsys.truncated_dim, subsys.truncated_dim),
        dtype=np.complex128,
    )
    for index, paramval in enumerate(param_vals):
        evecs = specdata.state_table[index]  # type: ignore[index]
        matelem_table[index] = subsys.matrixelement_table(
            operator_name, evecs=evecs, evals_count=subsys.truncated_dim
        )

    specdata.matrixelem_table = matelem_table

    title = f"{subsys.id_str}: matrix elements (sweep)"
    fig, axes = plot.matelem_vs_paramvals(
        specdata, mode=mode_str, fig_ax=fig_ax, title=title
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


class MatrixElementSweepPanelBuilder:
    """Builder for the MATRIX_ELEMENT_SCAN panel (qubit subsystems only)."""

    plot_type: ClassVar[PlotType] = PlotType.MATRIX_ELEMENT_SCAN
    slider_invariant: ClassVar[bool] = True

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        subsys = plot_id.subsystems[0]
        if not isinstance(subsys, QubitBaseClass):
            raise TypeError(
                "MATRIX_ELEMENT_SCAN panel requires a QubitBaseClass subsystem; "
                f"got {type(subsys).__name__}."
            )
        ui_mode_dropdown, opname_dropdown = explorer.settings[plot_id]
        return display_matrixelement_sweep(
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
