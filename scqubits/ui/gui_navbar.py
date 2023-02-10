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

import base64, os

from typing import Any, Dict, List, Optional, Tuple, Union


from scqubits.ui.gui_defaults import NAV_COLOR
from scqubits.ui import gui_defaults as gui_defaults
from scqubits.ui.gui_setup import flex_column
import scqubits.ui.gui_custom_widgets as ui

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


def create_navbar() -> Tuple[v.Card, dict]:
    # Navigation bar elements are:
    # CHOOSE_QUBIT
    # CHOOSE_PLOT
    # AUTO_UPDATING
    # DO_UPDATE

    def update_mini(widget, event, data):
        nav_drawer.mini_variant = not nav_drawer.mini_variant

    icons = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")

    file_names = ["En.png", "psi.png", "Me.png", "MeS.png", "T1.png", "qubit.png"]
    for name in file_names:
        full_path = os.path.join(path, name)
        file = open(full_path, "rb")
        image = file.read()
        image_base64 = base64.b64encode(image).decode("ascii")
        icons[name] = v.Img(src=f"data:image/png;base64,{image_base64}", width=50)

    navbar_elements = {}

    navbar_elements["CHOOSE_QUBIT"] = v.Select(
        v_model=gui_defaults.supported_qubits[0],
        items=gui_defaults.supported_qubits,
        filled=True,
        label="Qubit",
        # class_="pl-2 my-0 py-0",
        # dense=True,
        # style_="min-width: 200px; max-width: 500px",
    )

    navbar_elements["CHOOSE_PLOT"] = v.BtnToggle(
        v_model=0,
        mandatory=True,
        class_="p-0 mx-0 my-2",
        color=NAV_COLOR,
        children=[
            flex_column(
                [
                    ui.vBtn(
                        color=NAV_COLOR,
                        class_="my-4",
                        children=[icons[name]],
                        width=50,
                        elevation=0,
                    )
                    for name in file_names
                    if name != "qubit.png"
                ],
                class_="px-0",
                style_=f"background-color: {NAV_COLOR}",
            )
        ],
    )

    navbar_elements["AUTO_UPDATING"] = v.Switch(
        v_model=True,
        width=150,
        label="Auto\nupdate",
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

    navbar_elements["HEADER"] = v.Card(
        class_="d-flex flex-row align-left p-0 m-0",
        height=70,
        width="100%",
        flat=True,
        color=NAV_COLOR,
        children=[
            v.Container(
                class_="p-1",
                width=60,
                color=NAV_COLOR,
                children=[icons["qubit.png"]]
            ),
            navbar_elements["CHOOSE_QUBIT"]
        ]
    )

    nav_drawer = v.NavigationDrawer(
        v_model="drawer",
        permanent=True,
        mini_variant=True,
        mini_variant_width=90,
        elevation="0",
        color=NAV_COLOR,
        width="40%",
        children=[
            v.List(
                nav=True,
                dense=True,
                children=[
                    # v.ListItem(
                    #     link=True,
                    #     class_="align-items-bottom",
                    #     children=[
                    #         ui.vBtn(
                    #             onclick=update_mini,
                    #             color=NAV_COLOR,
                    #             elevation=0,
                    #             width=50,
                    #             height=70,
                    #             children=[icons["qubit.png"]],
                    #         ),
                    #         navbar_elements["CHOOSE_QUBIT"],
                    #     ],
                    # ),
                    v.ListItem(
                        color=NAV_COLOR,
                        class_="align-items-bottom p-0 m-0",
                        children=[navbar_elements["CHOOSE_PLOT"]],
                    )
                    # v.ListItem(
                    #     color=NAV_COLOR,
                    #     class_="align-items-bottom p-0 m-0",
                    #     children=[
                    #         v.Container(
                    #             class_="d-flex flex-column flex-start align-center p-0",
                    #             children=[
                    #                 v.Text(children=["manual updating"]),
                    #                 navbar_elements["TOGGLE_MANUAL_UPDATING"],
                    #                 navbar_elements["DO_UPDATE"],
                    #             ],
                    #         )
                    #     ],
                    # )
                    # v.ListItem(
                    #     class_="align-items-bottom",
                    #     children=[
                    #         self.v_nav_btn_sweep_conf,
                    #         self.v_nav_panel_sweep_conf,
                    #     ],
                    # ),
                    # v.ListItem(
                    #     class_="align-items-bottom",
                    #     children=[
                    #         self.v_nav_plot_settings_btn,
                    #         self.v_nav_plot_settings_panel,
                    #     ],
                    # ),
                ],
            )
        ],
    )

    nav_drawer.on_event("update:miniVariant", update_mini)

    return nav_drawer, navbar_elements
