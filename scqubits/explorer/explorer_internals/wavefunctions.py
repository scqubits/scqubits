"""WAVEFUNCTIONS panel: a qubit's bare eigenfunctions at the current parameter slice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits as scq
import scqubits.ui.gui_custom_widgets as ui

from scqubits import settings
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType, mode_dropdown_dict, mode_dropdown_list

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID


@rc_context(matplotlib_settings)
def display_bare_wavefunctions(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
    mode: str = "real",
    which: int | list[int] = -1,
) -> tuple[Figure, Axes]:
    """Plot the bare wavefunctions of ``subsys`` at the current parameter slice."""
    subsys_index = sweep.get_subsys_index(subsys)
    evals = sweep["bare_evals"][subsys_index][param_slice.all]
    evecs = sweep["bare_evecs"][subsys_index][param_slice.all]

    settings.DISPATCH_ENABLED = False
    sweep._update_hilbertspace(sweep, *param_slice.all_values)
    settings.DISPATCH_ENABLED = True

    title = f"Wavefunctions: {subsys.id_str}"
    return subsys.plot_wavefunction(  # type: ignore[attr-defined]
        which=which, esys=(evals, evecs), mode=mode, title=title, fig_ax=fig_ax
    )


class WavefunctionsPanelBuilder:
    """Builder for the WAVEFUNCTIONS panel (qubit subsystems only)."""

    plot_type: ClassVar[PlotType] = PlotType.WAVEFUNCTIONS

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
                "WAVEFUNCTIONS panel requires a QubitBaseClass subsystem; "
                f"got {type(subsys).__name__}.  Check that "
                "``gui_defaults.supported_panels`` is not offering this plot "
                "for a non-qubit subsystem."
            )
        ui_wavefunction_selector, ui_mode_dropdown = explorer.settings[plot_id]
        return display_bare_wavefunctions(
            explorer.sweep,
            subsys,
            param_slice,
            fig_ax,
            mode=mode_dropdown_dict[ui_mode_dropdown.v_model],
            which=ui_wavefunction_selector.v_model,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        subsys = plot_id.subsystems[0]
        if isinstance(subsys, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)):
            ui_wavefunction_selector = ui.InitializedSelect(
                label="Display wavefunctions",
                items=list(range(subsys.truncated_dim)),
                v_model=0,
            )
        else:
            ui_wavefunction_selector = ui.InitializedSelect(
                label="Display wavefunctions",
                multiple=True,
                items=list(range(subsys.truncated_dim)),
                v_model=list(range(subsys.truncated_dim)),
            )
        ui_mode_dropdown = ui.InitializedSelect(
            items=mode_dropdown_list,
            v_model=mode_dropdown_list[0],
            label="Plot amplitude as",
        )
        ui_wavefunction_selector.observe(
            settings.explorer.update_plots, names="v_model"
        )
        ui_mode_dropdown.observe(settings.explorer.update_plots, names="v_model")
        return [ui_wavefunction_selector, ui_mode_dropdown]
