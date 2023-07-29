# explorer_settings.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import itertools

from typing import TYPE_CHECKING, Any, Dict

import scqubits as scq
import scqubits.ui.gui_custom_widgets as ui

from scqubits.ui.gui_defaults import PlotType, mode_dropdown_list
from scqubits.utils import misc as utils


if TYPE_CHECKING:
    from scqubits import Explorer
    from scqubits.explorer.explorer_widget import PlotID

try:
    from IPython.display import HTML, display, notebook
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

try:
    import ipyvuetify as v
    import ipywidgets
    from scqubits.ui.gui_custom_widgets import flex_row
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True


class ExplorerSettings:
    """
    Generates the UI for Explorer settings.

    Parameters
    ----------
    explorer:
        the `Explorer` object of interest

    Attributes
    ----------
    ui:
        dictionary of all UI elements
    """

    @utils.Required(ipyvuetify=_HAS_IPYVUETIFY)
    def __init__(self, explorer: "Explorer"):
        self.explorer = explorer
        self.ui: Dict[str, Any] = {}
        self.ui["level_slider"]: Dict[PlotID, v.VuetifyWidget] = {}
        self.ui["Transitions"]: Dict[str, v.VuetifyWidget] = {}
        self.ui["kerr"]: Dict[str, v.VuetifyWidget] = {}

        for plot_id in self.explorer.ui["panel_switch_by_plot_id"].keys():
            self.ui[plot_id] = self.build_settings_ui(plot_id)

        self.ui["dialogs"] = {
            plot_id: v.Dialog(
                v_model=False,
                width="40%",
                children=[
                    v.Card(
                        children=[
                            v.Toolbar(
                                children=[
                                    v.ToolbarTitle(
                                        children=[f"Plot settings: {str(plot_id)}"]
                                    )
                                ],
                                color="deep-purple accent-4",
                                dark=True,
                            ),
                            v.CardText(children=[ui.flex_row(self.ui[plot_id])]),
                        ]
                    )
                ],
            )
            for plot_id in self.explorer.ui["panel_switch_by_plot_id"].keys()
        }

    def __getitem__(self, item):
        return self.ui[item]

    def build_settings_ui(self, plot_id: "PlotID"):
        subsys = plot_id.subsystems
        plot_type = plot_id.plot_type

        if plot_type is PlotType.ENERGY_SPECTRUM:
            subsys = subsys[0]
            subsys_index = self.explorer.sweep.get_subsys_index(subsys)
            evals_count = self.explorer.sweep.subsys_evals_count(subsys_index)
            self.ui["level_slider"][plot_id] = ui.NumberEntryWidget(
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
                label="Subtract E\u2080", v_model=True, width=300
            )
            self.ui["level_slider"][plot_id].observe(
                self.explorer.update_plots, names="v_model"
            )
            ui_subtract_ground_switch.observe(
                self.explorer.update_plots, names="v_model"
            )
            return [
                self.ui["level_slider"][plot_id].widget(),
                ui_subtract_ground_switch,
            ]

        if plot_type is PlotType.WAVEFUNCTIONS:
            subsys = subsys[0]
            if isinstance(
                subsys, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)  # scq.Bifluxon
            ):
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
                self.explorer.update_plots, names="v_model"
            )
            ui_mode_dropdown.observe(self.explorer.update_plots, names="v_model")
            return [ui_wavefunction_selector, ui_mode_dropdown]

        if plot_type is PlotType.MATRIX_ELEMENTS:
            subsys = subsys[0]
            ui_mode_dropdown = ui.InitializedSelect(
                items=mode_dropdown_list,
                label="Plot matrix elements as",
                v_model=mode_dropdown_list[2],
            )
            op_names = subsys.get_operator_names()
            ui_operator_dropdown = ui.InitializedSelect(
                items=op_names, label="Operator", v_model=op_names[0]
            )
            ui_mode_dropdown.observe(self.explorer.update_plots, names="v_model")
            ui_operator_dropdown.observe(self.explorer.update_plots, names="v_model")
            return [ui_mode_dropdown, ui_operator_dropdown]

        if plot_type is PlotType.MATRIX_ELEMENT_SCAN:
            subsys = subsys[0]
            ui_mode_dropdown = ui.InitializedSelect(
                items=mode_dropdown_list,
                label="Plot matrix elements as",
                v_model=mode_dropdown_list[2],
            )
            op_names = subsys.get_operator_names()
            ui_operator_dropdown = ui.InitializedSelect(
                items=op_names, label="Operator", v_model=op_names[0]
            )
            ui_mode_dropdown.observe(self.explorer.update_plots, names="v_model")
            ui_operator_dropdown.observe(self.explorer.update_plots, names="v_model")
            return [ui_mode_dropdown, ui_operator_dropdown]

        if plot_type is PlotType.TRANSITIONS:
            self.ui["Transitions"]["initial_state_inttexts"] = [
                ui.ValidatedNumberField(
                    label=subsys.id_str,
                    num_type=int,
                    v_min=0,
                    v_max=subsys.truncated_dim,
                    v_model=0,
                    style_="display: inherit; width: 65px;",
                    class_="ml-4",
                )
                for subsys in self.explorer.sweep.hilbertspace
            ]

            self.ui["Transitions"]["initial_dressed_inttext"] = ui.ValidatedNumberField(
                label="Dressed state",
                class_="ml-4 align-bottom",
                num_type=int,
                v_min=0,
                v_max=self.explorer.sweep.hilbertspace.dimension,
                v_model=0,
                style_="display: none; width: 65px;",
            )

            self.ui["Transitions"]["photons_inttext"] = ui.ValidatedNumberField(
                num_type=int,
                class_="ml-3",
                v_model=1,
                v_min=1,
                v_max=5,
                label="Photon number",
                style_="max-width: 120px",
            )
            self.ui["Transitions"]["highlight_selectmultiple"] = ui.InitializedSelect(
                multiple=True,
                label="",
                items=self.explorer.subsys_names,
                v_model=[self.explorer.subsys_names[0]],
                width=185,
            )

            self.ui["Transitions"]["initial_bare_dressed_toggle"] = v.RadioGroup(
                v_model="bare",
                children=[
                    v.Radio(label="by bare product label", value="bare"),
                    v.Radio(label="by dressed index", value="dressed"),
                ],
            )

            self.ui["Transitions"]["sidebands_switch"] = v.Switch(
                label="Show sidebands", v_model=False, width=250
            )
            for inttext in self.ui["Transitions"]["initial_state_inttexts"]:
                inttext.observe(self.explorer.update_plots, names="num_value")
            self.ui["Transitions"]["initial_dressed_inttext"].observe(
                self.explorer.update_plots, names="num_value"
            )
            self.ui["Transitions"]["photons_inttext"].observe(
                self.explorer.update_plots, names="num_value"
            )
            self.ui["Transitions"]["highlight_selectmultiple"].observe(
                self.explorer.update_plots, names="v_model"
            )
            self.ui["Transitions"]["sidebands_switch"].observe(
                self.explorer.update_plots, names="v_model"
            )
            self.ui["Transitions"]["initial_bare_dressed_toggle"].observe(
                self.explorer.bare_dressed_toggle, names="v_model"
            )

            initial_state_selection = ui.flex_row(
                [
                    v.Text(children=["Initial state"]),
                    self.ui["Transitions"]["initial_bare_dressed_toggle"],
                    *self.ui["Transitions"]["initial_state_inttexts"],
                    self.ui["Transitions"]["initial_dressed_inttext"],
                ]
            )

            photon_options_selection = v.Container(
                class_="d-flex flex-row",
                children=[
                    v.Text(children=["Single vs. multi-photon transitions"]),
                    self.ui["Transitions"]["photons_inttext"],
                ],
            )
            transition_highlighting = v.Container(
                class_="d-flex flex-row",
                children=[
                    v.Text(children=["Highlight:"]),
                    self.ui["Transitions"]["highlight_selectmultiple"],
                ],
            )

            return [
                v.Container(
                    class_="d-flex flex-column",
                    children=[
                        initial_state_selection,
                        photon_options_selection,
                        self.ui["Transitions"]["sidebands_switch"],
                        transition_highlighting,
                    ],
                ),
            ]

        if plot_type is PlotType.SELF_KERR and isinstance(subsys[0], scq.Oscillator):
            return []

        if plot_type is PlotType.SELF_KERR and not isinstance(
            subsys[0], scq.Oscillator
        ):
            subsys = subsys[0]
            ui_kerr_selector = ui.InitializedSelect(
                label="Selected pair of levels",
                items=list(
                    itertools.combinations_with_replacement(
                        list(range(subsys.truncated_dim)), 2
                    )
                ),
                v_model=[[1, 1], [1, 2], [2, 2]],
                multiple=True,
            )
            ui_kerr_selector.observe(self.explorer.update_plots, names="v_model")
            return [ui_kerr_selector]

        if plot_type is PlotType.CROSS_KERR:
            return []

        if plot_type is PlotType.AC_STARK:
            self.ui["kerr"]["ac_stark_ell"] = ui.InitializedSelect(
                v_model=1,
                items=list(range(1, subsys[0].truncated_dim)),
                label="qubit level",
            )
            self.ui["kerr"]["ac_stark_ell"].observe(
                self.explorer.update_plots, names="v_model"
            )

            return [
                v.Container(
                    class_="d-flex flex-column",
                    children=[self.ui["kerr"]["ac_stark_ell"]],
                )
            ]

        return []
