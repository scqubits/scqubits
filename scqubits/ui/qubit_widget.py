# qubit_widget.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


try:
    import ipywidgets
except ImportError:
    _HAS_IPYWIDGETS = False
else:
    _HAS_IPYWIDGETS = True

try:
    from IPython.display import display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

import scqubits.utils.misc as utils
import scqubits.core.units as units
import scqubits.settings as settings


@utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
def create_widget(callback_func, init_params, image_filename=None):
    """
    Displays ipywidgets for initialization of a QuantumSystem object.

    Parameters
    ----------
    callback_func: function
        callback_function depends on all the parameters provided as keys (str) in the parameter_dict, and is called upon
        changes of values inside the widgets
    init_params: {str: value, str: value, ...}
        names and values of initialization parameters
    image_filename: str, optional
        file name for circuit image to be displayed alongside the qubit
    Returns
    -------

    """
    widgets = {}
    box_list = []
    for name, value in init_params.items():
        label_str = name
        # NOTE: This will break if names of energy parameters in future qubits do not start with 'E'
        if name[0] == 'E':
            label_str += ' [' + units.get_units() + ']'
        elif name == 'flux':
            label_str += r' [$\Phi_0$]'
        label = ipywidgets.Label(value=label_str)
        if isinstance(value, float):
            enter_widget = ipywidgets.FloatText
        else:
            enter_widget = ipywidgets.IntText

        widgets[name] = enter_widget(value=value, description='', disabled=False, layout=ipywidgets.Layout(width='150px'))
        box_list.append(ipywidgets.HBox([label, widgets[name]], layout=ipywidgets.Layout(justify_content='flex-end')))

    if image_filename:
        file = open(image_filename, "rb")
        image = file.read()
        image_widget = ipywidgets.Image(value=image, format='jpg', layout=ipywidgets.Layout(width='700px'))
        ui_widget = ipywidgets.HBox([ipywidgets.VBox(box_list), ipywidgets.VBox([image_widget])])
    else:
        ui_widget = ipywidgets.VBox(box_list)

    out = ipywidgets.interactive_output(callback_func, widgets)
    display(ui_widget, out)
