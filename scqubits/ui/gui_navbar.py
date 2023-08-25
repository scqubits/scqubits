# gui_navbar.py
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

from typing import Tuple

import scqubits.ui.gui_custom_widgets as ui
import scqubits.utils.misc as utils

from scqubits.ui import gui_defaults as gui_defaults
from scqubits.ui.gui_custom_widgets import flex_column
from scqubits.ui.gui_defaults import NAV_COLOR

try:
    import ipyvuetify as v
    import ipywidgets
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True

try:
    from IPython.display import display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True


@utils.Required(ipyvuetify=_HAS_IPYVUETIFY, IPython=_HAS_IPYTHON)
def create_navbar() -> Tuple[v.Card, dict]:
    # Navigation bar elements are:
    # CHOOSE_QUBIT
    # CHOOSE_PLOT
    # AUTO_UPDATING
    # DO_UPDATE

    icons = gui_defaults.icons

    navbar_elements = {}

    navbar_elements["CHOOSE_QUBIT"] = v.Select(
        v_model=gui_defaults.supported_qubits[0],
        items=gui_defaults.supported_qubits,
        filled=True,
        label="Qubit",
        dense=True,
    )

    navbar_elements["CHOOSE_PLOT"] = v.BtnToggle(
        v_model=0,
        mandatory=True,
        class_="p-0 mx-0 my-2",
        color=NAV_COLOR,
        children=[
            flex_column(
                [
                    ui.vTooltipBtn(
                        color=NAV_COLOR,
                        class_="my-4",
                        children=[icons[icon_filename]],
                        tooltip=plot_type_name,
                        width=50,
                        elevation=0,
                    )
                    for plot_type_name, icon_filename in gui_defaults.gui_plot_choice_dict.items()
                ],
                class_="px-0",
                style_=f"background-color: {NAV_COLOR}",
            )
        ],
    )

    navbar_elements["AUTO_UPDATING"] = v.Switch(
        v_model=True,
        width=150,
        label="Auto update",
    )

    navbar_elements["DO_UPDATE"] = v.Btn(
        children=[v.Icon(children=["mdi-refresh"])],
        fab=True,
        color="orange",
        small=True,
        disabled=True,
        elevation="0",
        class_="px-2",
    )

    navbar_elements["HEADER"] = v.Sheet(
        class_="d-flex flex-row m-0 pt-3",
        style_="padding-right: 50%",
        height=70,
        flat=True,
        width="100%",
        color=NAV_COLOR,
        children=[
            v.Card(
                class_="p-2 mx-4",
                color=NAV_COLOR,
                elevation=0,
                children=[icons["scq-logo.png"]],
            ),
            navbar_elements["CHOOSE_QUBIT"],
        ],
    )

    nav_drawer = v.NavigationDrawer(
        v_model="drawer",
        permanent=True,
        elevation="0",
        color=NAV_COLOR,
        floating=True,
        width=90,
        height=800,
        children=[
            v.List(
                nav=True,
                dense=True,
                children=[
                    v.ListItem(
                        color=NAV_COLOR,
                        class_="align-items-bottom p-0 m-0",
                        children=[navbar_elements["CHOOSE_PLOT"]],
                    )
                ],
            )
        ],
    )

    return nav_drawer, navbar_elements
