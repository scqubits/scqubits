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

from typing import TYPE_CHECKING, Any

import scqubits.ui.gui_custom_widgets as ui

from scqubits.explorer.explorer_internals import PANEL_BUILDERS
from scqubits.utils import misc as utils

if TYPE_CHECKING:
    from scqubits import Explorer
    from scqubits.explorer.explorer_widget import PlotID

from scqubits.ui._optional_deps import _HAS_IPYTHON, _HAS_IPYVUETIFY, v


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

        for plot_id in self.explorer.ui.panel_switch_by_plot_id.keys():
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
            for plot_id in self.explorer.ui.panel_switch_by_plot_id.keys()
        }

    def __getitem__(self, item):
        return self.ui[item]

    def build_settings_ui(self, plot_id: "PlotID"):
        # Registry-driven dispatch: every ``PlotType`` has a
        # corresponding ``PanelBuilder`` under ``explorer_internals``.
        # See ``PANEL_BUILDERS`` for the mapping.
        builder_cls = PANEL_BUILDERS.get(plot_id.plot_type)
        if builder_cls is None:
            raise NotImplementedError(
                f"No PanelBuilder is registered for plot type {plot_id.plot_type!r}."
            )
        return builder_cls().build_settings_ui(self, plot_id)
