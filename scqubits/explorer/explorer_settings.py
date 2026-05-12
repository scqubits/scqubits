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

from __future__ import annotations

import itertools

from typing import TYPE_CHECKING, Any

import scqubits as scq
import scqubits.ui.gui_custom_widgets as ui

from scqubits.explorer.explorer_internals import PANEL_BUILDERS
from scqubits.ui.gui_defaults import PlotType, mode_dropdown_list
from scqubits.utils import misc as utils

if TYPE_CHECKING:
    from scqubits import Explorer
    from scqubits.explorer.explorer_widget import PlotID

try:
    from IPython.display import HTML, display
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
    """Generates the UI for Explorer settings.

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
        self.ui: dict[str, Any] = {}
        self.ui["level_slider"] = {}
        self.ui["Transitions"] = {}
        self.ui["kerr"] = {}

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
        # Registry-driven dispatch for plot types migrated into
        # ``explorer_internals`` (see PANEL_BUILDERS).  Anything not in
        # the registry falls through to the legacy ``if/elif`` chain
        # below; the chain shrinks one entry per phase 2 commit.
        builder_cls = PANEL_BUILDERS.get(plot_id.plot_type)
        if builder_cls is not None:
            return builder_cls().build_settings_ui(self, plot_id)

        # `subsys` is reassigned from list to single element in many branches below;
        # use `Any` to avoid narrowing-error churn without changing runtime behavior.
        subsys: Any = plot_id.subsystems
        plot_type = plot_id.plot_type

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
