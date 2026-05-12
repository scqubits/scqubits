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
