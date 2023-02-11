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

import base64
import os

from typing import Tuple


from scqubits.ui.gui_defaults import NAV_COLOR
from scqubits.ui import gui_defaults as gui_defaults
from scqubits.ui.gui_custom_widgets import flex_column
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
        class_="d-flex flex-row m-0",
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
                children=[icons["qubit.png"]],
            ),
            navbar_elements["CHOOSE_QUBIT"],
        ],
    )

    nav_drawer = v.NavigationDrawer(
        v_model="drawer",
        permanent=True,
        mini_variant=True,
        mini_variant_width=90,
        elevation="0",
        color=NAV_COLOR,
        floating=True,
        width="40%",
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
        ]
    )

    nav_drawer.on_event("update:miniVariant", update_mini)

    return nav_drawer, navbar_elements
