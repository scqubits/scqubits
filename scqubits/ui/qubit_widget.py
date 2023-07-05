# qubit_widget.py
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

from typing import Any, Callable, Dict, Optional

import scqubits.core.units as units
import scqubits.utils.misc as utils


try:
    import ipyvuetify
    import ipywidgets
    from scqubits.ui.gui_custom_widgets import ValidatedNumberField
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
def create_widget(
    callback_func: Callable,
    init_params: Dict[str, Any],
    image_filename: Optional[str] = None,
) -> None:
    """
    Displays ipyvuetify for initialization of a QuantumSystem object.

    Parameters
    ----------
    callback_func:
        callback_function depends on all the parameters provided as keys (str) in the
        parameter_dict, and is called upon changes of values inside the widgets
    init_params:
        names and values of initialization parameters
    image_filename:
        file name for circuit image to be displayed alongside the qubit
    """
    widgets = {}
    box_list = []
    for name, value in init_params.items():
        label_str = name
        # NOTE: This will break if names of energy parameters in future qubits
        # do not start with 'E'
        if name[0] == "E":
            label_str += f" [{units.get_units()}]"
        elif name == "flux":
            label_str += r" [Φ₀]"

        widgets[name] = ValidatedNumberField(
            v_model=value,
            num_type=type(value),
            placeholder=f"enter appropriate value for {label_str}",
            label=label_str,
            name=name,
            outlined=True,
            filled=True,
            dense=True,
            style_="width: 30%;",
            class_="ml-2 py-0",
        )
        widgets[name].observe(callback_func, names="v_model")
        box_list.append(widgets[name])

    if image_filename:
        file = open(image_filename, "rb")
        image = file.read()

        image_widget = ipyvuetify.Container(
            children=[ipywidgets.Image(value=image, format="jpg")],
            class_="ml-3 mt-0 pt-0",
            style_="width: 65%; max-width:1000px",
        )
        ui_widget = ipyvuetify.Row(
            children=[
                ipyvuetify.Row(
                    children=box_list,
                    style_="max-width: 22%; align-content: flex-start;",
                ),
                image_widget,
            ],
            class_="ml-3 mt-4",
        )

    else:
        ui_widget = ipyvuetify.Row(children=box_list, align_items="flex-start")

    display(ui_widget)
