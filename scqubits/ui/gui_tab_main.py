# gui_tab_main.py
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

import math
import warnings
from distutils.version import StrictVersion
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend as get_matplotlib_backend
from matplotlib.figure import Axes, Figure

import scqubits as scq
import scqubits.ui.gui_custom_widgets as ui
import scqubits.ui.gui_defaults as gui_defaults
import scqubits.ui.gui_navbar as gui_navbar
import scqubits.utils.misc as utils
from scqubits.core.discretization import Grid1d
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import NAV_COLOR
from scqubits.ui.gui_setup import (
    flex_column,
    flex_row,
    init_dict_v_noise_params,
    init_dict_v_plot_options,
    init_filename_textfield,
    init_qubit_params_widgets_dict,
    init_ranges_widgets_dict,
    init_save_btn,
)

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


def create_tab_main(plot_option_elements, qubit_params_elements) -> v.Sheet:
    sheet = v.Sheet(
        class_="d-flex d-row px-2 mx-1",
        style_="height: 280px, max_height: 280px",
        children=[v.Container(
            class_="d-flex flex-row mx-0 px-0",
            style_="transform: scale(0.9)",
            children=[
                v.Container(
                    # style_="transform: scale(0.85)",
                    class_="d-flex align-start flex-column pb-0",
                    style_="width: 50%",
                    children=plot_option_elements,
                ),
                v.Container(
                    style_="max-height: 350px",
                    # style_="transform: scale(0.85); max-height: 350px",
                    class_="d-flex align-start flex-column flex-wrap flex-align-content-start overflow-auto",
                    children=qubit_params_elements,
                ),
            ],
        )],
    )
    return sheet

#
# def plot_options() -> dict:
#     return {
#         0: self.energy_spectrum_options_widgets(),
#         1: self.wavefunctions_options_widgets(),
#         2: self.matelem_scan_options_widgets(),
#         3: self.matelem_options_widgets(),
#         4: self.coherence_times_options_widgets()
#     }
#
#
# def qubit_params() -> list:
#     widget_list = []
#
#     for param_widget in self.dict_v_qubit_params.values():
#         if hasattr(param_widget, "widget"):
#             widget_list.append(param_widget.widget())
#         else:
#             widget_list.append(param_widget)
#     return widget_list