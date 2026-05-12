"""TRANSITIONS panel: dressed-state transition spectrum across the sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.ui.gui_custom_widgets as ui

from scqubits.core.qubit_base import QuantumSystem
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID

try:
    import ipyvuetify as v
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True


@rc_context(matplotlib_settings)
def display_transitions(
    sweep: "ParameterSweep",
    photon_number: int,
    subsys_list: list["QuantumSystem"] | None,
    initial: int | tuple[int, ...],
    sidebands: bool,
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    """Render the dressed-state transition spectrum across the swept parameter."""
    if photon_number == 1:
        title = "Transition Spectrum"
    else:
        title = f"{photon_number}-photon Transitions"
    sliced_sweep = sweep if param_slice.fixed == tuple() else sweep[param_slice.fixed]
    fig, axes = sliced_sweep.plot_transitions(
        subsystems=subsys_list,
        initial=initial,
        photon_number=photon_number,
        title=title,
        sidebands=sidebands,
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")

    info_string = "Highlighted: transitions for "
    for sys in subsys_list or []:
        info_string += sys.id_str + ", "
    info_string += "with sidebands" if sidebands else "no sidebands"
    axes.text(
        0.5,
        1.05,
        info_string,
        fontsize=9,
        fontweight=300,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=axes.transAxes,
    )

    return fig, axes


class TransitionsPanelBuilder:
    """Builder for the TRANSITIONS panel (composite plot, no subsystem restriction)."""

    plot_type: ClassVar[PlotType] = PlotType.TRANSITIONS

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        transitions_ui = explorer.settings["Transitions"]
        initial_state: int | tuple[int, ...]
        if transitions_ui["initial_bare_dressed_toggle"].v_model == "bare":
            initial_state = tuple(
                int(inttext.v_model)
                for inttext in transitions_ui["initial_state_inttexts"]
            )
        else:
            initial_state = int(transitions_ui["initial_dressed_inttext"].v_model)

        subsys_name_tuple = transitions_ui["highlight_selectmultiple"].v_model
        subsys_list: list[QuantumSystem] | None
        if subsys_name_tuple == ():
            subsys_list = None
        else:
            subsys_list = [
                explorer.sweep.subsys_by_id_str(subsys_name)
                for subsys_name in subsys_name_tuple
            ]

        sidebands = transitions_ui["sidebands_switch"].v_model
        photon_number = int(transitions_ui["photons_inttext"].v_model)
        return display_transitions(
            explorer.sweep,
            photon_number,
            subsys_list,
            initial_state,
            sidebands,
            param_slice,
            fig_ax,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        transitions_ui = settings.ui["Transitions"]
        explorer = settings.explorer

        transitions_ui["initial_state_inttexts"] = [
            ui.ValidatedNumberField(
                label=subsys.id_str,
                num_type=int,
                v_min=0,
                v_max=subsys.truncated_dim,
                v_model=0,
                style_="display: inherit; width: 65px;",
                class_="ml-4",
            )
            for subsys in explorer.sweep.hilbertspace
        ]

        transitions_ui["initial_dressed_inttext"] = ui.ValidatedNumberField(
            label="Dressed state",
            class_="ml-4 align-bottom",
            num_type=int,
            v_min=0,
            v_max=explorer.sweep.hilbertspace.dimension,
            v_model=0,
            style_="display: none; width: 65px;",
        )

        transitions_ui["photons_inttext"] = ui.ValidatedNumberField(
            num_type=int,
            class_="ml-3",
            v_model=1,
            v_min=1,
            v_max=5,
            label="Photon number",
            style_="max-width: 120px",
        )
        transitions_ui["highlight_selectmultiple"] = ui.InitializedSelect(
            multiple=True,
            label="",
            items=explorer.subsys_names,
            v_model=[explorer.subsys_names[0]],
            width=185,
        )

        transitions_ui["initial_bare_dressed_toggle"] = v.RadioGroup(
            v_model="bare",
            children=[
                v.Radio(label="by bare product label", value="bare"),
                v.Radio(label="by dressed index", value="dressed"),
            ],
        )

        transitions_ui["sidebands_switch"] = v.Switch(
            label="Show sidebands", v_model=False, width=250
        )

        for inttext in transitions_ui["initial_state_inttexts"]:
            inttext.observe(explorer.update_plots, names="num_value")
        transitions_ui["initial_dressed_inttext"].observe(
            explorer.update_plots, names="num_value"
        )
        transitions_ui["photons_inttext"].observe(
            explorer.update_plots, names="num_value"
        )
        transitions_ui["highlight_selectmultiple"].observe(
            explorer.update_plots, names="v_model"
        )
        transitions_ui["sidebands_switch"].observe(
            explorer.update_plots, names="v_model"
        )
        transitions_ui["initial_bare_dressed_toggle"].observe(
            explorer.bare_dressed_toggle, names="v_model"
        )

        initial_state_selection = ui.flex_row(
            [
                v.Text(children=["Initial state"]),
                transitions_ui["initial_bare_dressed_toggle"],
                *transitions_ui["initial_state_inttexts"],
                transitions_ui["initial_dressed_inttext"],
            ]
        )

        photon_options_selection = v.Container(
            class_="d-flex flex-row",
            children=[
                v.Text(children=["Single vs. multi-photon transitions"]),
                transitions_ui["photons_inttext"],
            ],
        )
        transition_highlighting = v.Container(
            class_="d-flex flex-row",
            children=[
                v.Text(children=["Highlight:"]),
                transitions_ui["highlight_selectmultiple"],
            ],
        )

        return [
            v.Container(
                class_="d-flex flex-column",
                children=[
                    initial_state_selection,
                    photon_options_selection,
                    transitions_ui["sidebands_switch"],
                    transition_highlighting,
                ],
            ),
        ]
