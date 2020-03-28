# ui_base.py
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


@utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
def create_widget(callback_func, init_params, image_filename=None):
    """

    Parameters
    ----------
    callback_func: function
        callback_function depends on all the parameters provided as keys (str) in the parameter_dict, and is
    init_params: {str: value, str: value, ...}
        names and values of initialization parameters
    image_filename: str, optional
        file name for circuit image to be displayed alongside the qubit
    Returns
    -------

    """
    widgets = {}
    for name, value in init_params.items():
        if isinstance(value, float):
            widgets[name] = ipywidgets.FloatText(
                value=value,
                description=name,
                disabled=False
            )
        elif isinstance(value, int):
            widgets[name] = ipywidgets.IntText(
                value=value,
                min=0,
                description=name,
                disabled=False
            )

    if image_filename:
        file = open(image_filename, "rb")
        image = file.read()
        image_widget = ipywidgets.Image(
            value=image,
            format='png',
            width=600
        )
        display(image_widget)

    user_interface = ipywidgets.interactive(callback_func, **widgets)
    display(user_interface)
