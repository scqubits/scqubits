# gui.py
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


import inspect
from cProfile import label
from itertools import dropwhile
from re import I
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Axes, Figure
from tables import Description

from scqubits.ui.gui_defaults import WIDTH

try:
    from ipywidgets import (Box, Button, Checkbox, Dropdown, FloatRangeSlider,
                            FloatSlider, FloatText, HBox, Image, IntSlider,
                            IntText, Label, Layout, Output, SelectMultiple,
                            Tab, Text, ToggleButtons, VBox)
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

from pathlib import Path

import gui_defaults

import scqubits as scq
import scqubits.utils.misc as utils
from scqubits.core.qubit_base import QubitBaseClass


class GUI_V2:

    # Handle to the most recently generated Figure, Axes tuple
    fig_ax: Optional[Tuple[Figure, Axes]] = None

    def __repr__(self):
        return ""

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def __init__(self):
        scq.settings.PROGRESSBAR_DISABLED = False

        # Display Elements
        self.fig: Figure
        self.plot_output: Output = Output(
            layout=Layout(width=gui_defaults.WIDTH, height=gui_defaults.PLOT_HEIGHT)
        )
        self.tab_widget: Tab = Tab(
            layout=Layout(width=gui_defaults.WIDTH, height=gui_defaults.TAB_HEIGHT)
        )

        self.active_qubit: QubitBaseClass

        self.active_defaults: Dict[str, Any] = {}
        self.qubit_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_param_ranges_widgets: Dict[str, Union[IntText, FloatText]] = {}
        self.qubit_scan_params: Dict[str, Union[int, float, None]] = {}

        self.qubit_and_plot_ToggleButtons: Dict[str, ToggleButtons] = {}
        self.manual_update_and_save_widgets: Dict[str, Union[Checkbox, Button]] = {}
        self.qubit_plot_options_widgets: Dict[
            str,
            Union[
                Image,
                Button,
                Text,
                Dropdown,
                FloatSlider,
                IntSlider,
                FloatSlider,
                SelectMultiple,
                Checkbox,
            ],
        ] = {}
        self.qubit_params_widgets: Dict[str, Union[IntSlider, FloatSlider]] = {}

        # ------------------------------------------------------------------------------

        self.initialize_qubit_and_plot_ToggleButtons()
        self.initialize_manual_update_and_save_widgets_dict()
        starting_qubit = self.qubit_and_plot_ToggleButtons[
            "qubit_buttons"
        ].get_interact_value()

        self.set_qubit(starting_qubit)

        self.current_plot_option_refresh = self.get_plot_option_refresh()

        self.initialize_display()

        self.initialize_observe()

        self.current_plot_option_refresh(None)

    # Initialization Methods -----------------------------------------------------------
    def initialize_qubit_and_plot_ToggleButtons(self) -> None:
        """Creates all the ToggleButtons that controls
        which qubit or plot the user can choose from.
        """
        self.qubit_and_plot_ToggleButtons = {
            "qubit_buttons": ToggleButtons(
                options=gui_defaults.supported_qubits, description="Qubits:"
            ),
            "plot_buttons": ToggleButtons(
                options=gui_defaults.plot_choices,
                description="Plot:",
                button_style="info",
            ),
        }

    def initialize_manual_update_and_save_widgets_dict(self):
        self.manual_update_and_save_widgets = {
            "manual_update_checkbox": Checkbox(
                value=False, description="Manual Update", disabled=False, indent=False
            ),
            "update_button": Button(description="Update", disabled=True),
            "save_button": Button(icon="save", layout=Layout(width="35px")),
            "filename_text": Text(
                value=str(Path.cwd().joinpath("plot.pdf")),
                description="",
                disabled=False,
                layout=Layout(width=gui_defaults.TEXT_WIDTH),
            ),
        }

    def set_qubit(self, qubit_name: str) -> None:
        """Sets up the chosen qubit to be the active qubit
        and updates the defaults and widgets accordingly.

        Parameters
        ----------
        qubit_name:
        """
        self.active_defaults = gui_defaults.qubit_defaults[qubit_name]
        self.initialize_qubit(qubit_name)
        self.initialize_qubit_params_dicts()
        self.initialize_qubit_plot_options_widgets_dict()
        self.initialize_qubit_params_widgets_dict()
        self.initialize_qubit_param_ranges_widgets_dict()

    def initialize_qubit(self, qubit_name: str) -> None:
        """Initializes self.active_qubit to the user's choice
        using the chosen qubit's default parameters.

        Parameters
        ----------
        qubit_name:
        """
        QubitClass = getattr(scq, qubit_name)
        init_params = QubitClass.default_params()

        if qubit_name == "ZeroPi" or qubit_name == "FullZeroPi":
            init_params["grid"] = scq.Grid1d(
                min_val=gui_defaults.grid_defaults["grid_min_val"],
                max_val=gui_defaults.grid_defaults["grid_max_val"],
                pt_count=gui_defaults.grid_defaults["grid_pt_count"],
            )

        self.active_qubit = QubitClass(**init_params)

    def initialize_qubit_params_dicts(self) -> None:
        """
        Initializes qubit_params and qubit_scan_params.
        Note that qubit_scan_params will be used to create the
        dropdown options.
        """
        self.qubit_params.clear()
        self.qubit_scan_params.clear()
        self.qubit_params = self.active_qubit.default_params()
        if "truncated_dim" in self.qubit_params.keys():
            del self.qubit_params["truncated_dim"]

        for param_name, param_val in self.qubit_params.items():
            if "cut" in param_name:
                pass
            else:
                self.qubit_scan_params[param_name] = param_val

    def initialize_qubit_plot_options_widgets_dict(self) -> None:
        """
        Creates all the widgets that will be used for general plotting options.
        """
        std_layout = Layout(width=gui_defaults.SLIDER_WIDTH)

        operator_dropdown_list = self.get_operators()
        scan_dropdown_list = self.qubit_scan_params.keys()
        mode_dropdown_list = [
            ("Re(·)", "real"),
            ("Im(·)", "imag"),
            ("|·|", "abs"),
            (u"|\u00B7|\u00B2", "abs_sqr"),
        ]
        file = open(self.active_qubit._image_filename, "rb")
        image = file.read()

        self.qubit_plot_options_widgets = {
            "qubit_info_image_widget": Image(
                value=image, format="jpg", layout=Layout(width=gui_defaults.IMAGE_WIDTH)
            ),
            "scan_dropdown": Dropdown(
                options=scan_dropdown_list,
                value=self.active_defaults["scan_param"],
                description="Scan over",
                disabled=False,
                layout=std_layout,
            ),
            "mode_dropdown": Dropdown(
                options=mode_dropdown_list,
                description="Plot as:",
                disabled=False,
                layout=std_layout,
            ),
            "operator_dropdown": Dropdown(
                options=operator_dropdown_list,
                value=self.active_defaults["operator"],
                description="Operator",
                disabled=False,
                layout=std_layout,
            ),
            "scan_range_slider": FloatRangeSlider(
                min=self.active_defaults[self.active_defaults["scan_param"]]["min"],
                max=self.active_defaults[self.active_defaults["scan_param"]]["max"],
                value=[
                    self.active_defaults[self.active_defaults["scan_param"]]["min"],
                    self.active_defaults[self.active_defaults["scan_param"]]["max"],
                ],
                step=0.05,
                description="{} range".format(self.active_defaults["scan_param"]),
                continuous_update=False,
                layout=std_layout,
            ),
            "eigenvalue_state_slider": IntSlider(
                min=1,
                max=10,
                value=7,
                description="Highest state",
                continuous_update=False,
                layout=std_layout,
            ),
            "matrix_element_state_slider": IntSlider(
                min=1,
                max=6,
                value=4,
                description="Highest state",
                continuous_update=False,
                layout=std_layout,
            ),
            "wavefunction_single_state_selector": IntSlider(
                min=0,
                max=10,
                value=0,
                description="State no.",
                continuous_update=False,
                layout=std_layout,
            ),
            "wavefunction_scale_slider": FloatSlider(
                min=0.1,
                max=4,
                value=self.active_defaults["scale"],
                disabled=True,
                description="\u03c8 ampl.",
                continuous_update=False,
                layout=std_layout,
            ),
            "wavefunction_multi_state_selector": SelectMultiple(
                options=range(0, 10),
                value=[0, 1, 2, 3, 4],
                description="States",
                disabled=False,
                continuous_update=False,
                layout=std_layout,
            ),
            "show_numbers_checkbox": Checkbox(
                value=False, description="Show values", disabled=False
            ),
            "show3d_checkbox": Checkbox(
                value=True, description="Show 3D", disabled=False
            ),
            "subtract_ground_checkbox": Checkbox(
                value=False, description="Subtract E\u2080", disabled=False
            ),
            "manual_scale_checkbox": Checkbox(
                value=False, description="Manual Scaling", disabled=False
            ),
        }

    def initialize_qubit_params_widgets_dict(self) -> None:
        """Creates all the widgets that will be used
        for changing the parameter values for the specified qubit.
        """
        self.qubit_params_widgets.clear()
        self.qubit_param_ranges_widgets.clear()
        std_layout = Layout(width=gui_defaults.SLIDER_WIDTH)

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            grid_min = gui_defaults.grid_defaults["grid_min_val"]
            grid_max = gui_defaults.grid_defaults["grid_max_val"]
            self.qubit_params_widgets["grid"] = FloatRangeSlider(
                min=-12 * np.pi,
                max=12 * np.pi,
                value=[grid_min, grid_max],
                step=0.05,
                description="Grid range",
                continuous_update=False,
                layout=std_layout,
            )

        for param_name, param_val in self.qubit_params.items():
            if isinstance(param_val, int):
                kwargs = (
                    self.active_defaults.get(param_name) or self.active_defaults["int"]
                )
                self.qubit_params_widgets[param_name] = IntSlider(
                    **kwargs,
                    value=param_val,
                    description="{}:".format(param_name),
                    continuous_update=False,
                    layout=std_layout
                )
            else:
                kwargs = (
                    self.active_defaults.get(param_name)
                    or self.active_defaults["float"]
                )
                self.qubit_params_widgets[param_name] = FloatSlider(
                    **kwargs,
                    value=param_val,
                    step=0.01,
                    description="{}:".format(param_name),
                    continuous_update=False,
                    layout=std_layout
                )

    def initialize_qubit_param_ranges_widgets_dict(self) -> None:
        self.qubit_param_ranges_widgets.clear()
        param_range_text_layout = Layout(width=gui_defaults.RANGE_TEXT_WIDTH)

        for param_name, param_widget in self.qubit_params_widgets.items():
            param_min_text = None
            param_max_text = None

            if isinstance(param_widget, IntSlider):
                param_min_text = IntText(
                    value=param_widget.min,
                    description="min=",
                    layout=param_range_text_layout,
                )
                param_max_text = IntText(
                    value=param_widget.max,
                    description="max=",
                    layout=param_range_text_layout,
                )
            else:
                param_min_text = FloatText(
                    value=param_widget.min,
                    description="min=",
                    step=0.01,
                    layout=param_range_text_layout,
                )
                param_max_text = FloatText(
                    value=param_widget.max,
                    description="max=",
                    step=0.01,
                    layout=param_range_text_layout,
                )

            self.qubit_param_ranges_widgets[param_name] = {
                "min": param_min_text,
                "max": param_max_text,
            }

    def initialize_tab_widget(self) -> None:
        qubit_plot_tab = self.qubit_plot_layout()
        param_ranges_tab = self.qubit_param_ranges_layout()
        qubit_info_tab = self.qubit_info_layout()

        tab_titles = ["Qubit Plot", "Ranges", "Qubit Info"]
        self.tab_widget.children = [qubit_plot_tab, param_ranges_tab, qubit_info_tab]

        for title_index in range(len(self.tab_widget.children)):
            self.tab_widget.set_title(title_index, tab_titles[title_index])

    def initialize_display(self) -> None:
        qubit_and_plot_choice_display = self.qubit_and_plot_ToggleButtons_layout()
        self.initialize_tab_widget()
        manual_update_display = self.manual_update_and_save_layout()

        display(
            qubit_and_plot_choice_display,
            self.tab_widget,
            manual_update_display,
            self.plot_output,
        )

    def initialize_observe(self) -> None:
        self.qubit_and_plot_ToggleButtons["qubit_buttons"].observe(
            self.qubit_change, names="value"
        )
        self.qubit_and_plot_ToggleButtons["plot_buttons"].observe(
            self.plot_option_layout_refresh, names="value"
        )
        self.manual_update_and_save_widgets["manual_update_checkbox"].observe(
            self.manual_update_checkbox, names="value"
        )
        self.manual_update_and_save_widgets["update_button"].on_click(
            self.manual_update_button_onclick
        )
        self.manual_update_and_save_widgets["save_button"].on_click(
            self.save_button_clicked_action
        )
        self.qubit_plot_options_widgets["manual_scale_checkbox"].observe(
            self.manual_scale_tf, names="value"
        )
        self.observe_param_ranges()
        self.observe_plot()

    # Retrieval Methods------------------------------------------------------------------
    def get_operators(self) -> List[str]:
        """Returns a list of operators for the active_qubit.
        Note that this list omits any operators that start with "_".

        Returns
        -------
        List[ str ]
        """
        operator_list = []
        for name, val in inspect.getmembers(self.active_qubit):
            if "operator" in name and name[0] != "_":
                operator_list.append(name)
        return operator_list

    def get_current_values(self) -> Dict[str, Union[int, float]]:
        current_values_dict = {}
        for param_name, widget in self.qubit_params_widgets.items():
            current_values_dict[param_name] = widget.get_interact_value()
        return current_values_dict

    def get_plot_option_refresh(self):
        current_plot_option = self.qubit_and_plot_ToggleButtons[
            "plot_buttons"
        ].get_interact_value()

        if current_plot_option == "Energy spectrum":
            plot_option_refresh = self.evals_vs_paramvals_plot_refresh
        elif current_plot_option == "Wavefunctions":
            plot_option_refresh = self.wavefunctions_plot_refresh
        elif current_plot_option == "Matrix element scan":
            plot_option_refresh = self.matelem_vs_paramvals_plot_refresh
        elif current_plot_option == "Matrix elements":
            plot_option_refresh = self.matrixelements_plot_refresh

        return plot_option_refresh

    # Observe Methods-------------------------------------------------------------------
    def observe_param_ranges(self):
        for text_widgets in self.qubit_param_ranges_widgets.values():
            text_widgets["min"].observe(self.param_ranges_update, names="value")
            text_widgets["max"].observe(self.param_ranges_update, names="value")

    def observe_plot(self):
        self.qubit_plot_options_widgets["scan_dropdown"].observe(
            self.scan_dropdown_refresh, names="value"
        )

        for widget in self.qubit_params_widgets.values():
            widget.observe(self.current_plot_option_refresh, names="value")
        for widget in self.qubit_plot_options_widgets.values():
            widget.observe(self.current_plot_option_refresh, names="value")

    def unobserve_plot(self):
        self.qubit_plot_options_widgets["scan_dropdown"].unobserve(
            self.scan_dropdown_refresh, names="value"
        )

        for widget in self.qubit_params_widgets.values():
            widget.unobserve(self.current_plot_option_refresh, names="value")
        for widget in self.qubit_plot_options_widgets.values():
            widget.unobserve(self.current_plot_option_refresh, names="value")

    # Eventhandler Methods -------------------------------------------------------------
    def qubit_change(self, change) -> None:
        if not self.manual_update_and_save_widgets[
            "manual_update_checkbox"
        ].get_interact_value():
            self.unobserve_plot()
        self.set_qubit(change["new"])
        self.initialize_tab_widget()
        self.observe_param_ranges()
        if not self.manual_update_and_save_widgets[
            "manual_update_checkbox"
        ].get_interact_value():
            self.observe_plot()
            self.current_plot_option_refresh(None)

    def scan_dropdown_refresh(self, change) -> None:
        if not self.manual_update_and_save_widgets[
            "manual_update_checkbox"
        ].get_interact_value():
            self.unobserve_plot()
        self.qubit_params_widgets[change.old].disabled = False
        self.qubit_params_widgets[change.new].disabled = True

        self.qubit_plot_options_widgets["scan_range_slider"].min = self.active_defaults[
            change.new
        ]["min"]
        self.qubit_plot_options_widgets["scan_range_slider"].max = self.active_defaults[
            change.new
        ]["max"]
        self.qubit_plot_options_widgets["scan_range_slider"].value = [
            self.active_defaults[change.new]["min"],
            self.active_defaults[change.new]["max"],
        ]

        self.qubit_plot_options_widgets[
            "scan_range_slider"
        ].description = "{} range".format(change.new)
        if not self.manual_update_and_save_widgets[
            "manual_update_checkbox"
        ].get_interact_value():
            self.observe_plot()
            self.current_plot_option_refresh(None)

    def plot_option_layout_refresh(self, change) -> None:
        if not self.manual_update_and_save_widgets[
            "manual_update_checkbox"
        ].get_interact_value():
            self.unobserve_plot()
        self.current_plot_option_refresh = self.get_plot_option_refresh()
        new_plot_option = self.plot_option_layout()

        self.tab_widget.children[0].children[0].children = tuple(
            new_plot_option.children
        )

        if not self.manual_update_and_save_widgets[
            "manual_update_checkbox"
        ].get_interact_value():
            self.observe_plot()
            self.current_plot_option_refresh(None)

    def manual_scale_tf(self, change) -> None:
        if change["new"]:
            self.qubit_plot_options_widgets[
                "wavefunction_scale_slider"
            ].disabled = False
        else:
            self.qubit_plot_options_widgets["wavefunction_scale_slider"].disabled = True

    def manual_update_checkbox(self, change) -> None:
        if change["new"]:
            self.manual_update_and_save_widgets["update_button"].disabled = False
            self.unobserve_plot()
        else:
            self.manual_update_and_save_widgets["update_button"].disabled = True
            self.observe_plot()
            self.current_plot_option_refresh(None)

    def manual_update_button_onclick(self, change) -> None:
        self.current_plot_option_refresh(None)

    def param_ranges_update(self, change) -> None:
        scan_dropdown_value = self.qubit_plot_options_widgets[
            "scan_dropdown"
        ].get_interact_value()

        for param_name, text_widgets in self.qubit_param_ranges_widgets.items():
            param_widget = self.qubit_params_widgets[param_name]

            new_min = text_widgets["min"].get_interact_value()
            new_max = text_widgets["max"].get_interact_value()
            param_widget.min = new_min
            param_widget.max = new_max

            if param_name == scan_dropdown_value:
                self.qubit_plot_options_widgets["scan_range_slider"].min = new_min
                self.qubit_plot_options_widgets["scan_range_slider"].max = new_max

    def save_button_clicked_action(self, change):
        self.fig.savefig(self.manual_update_and_save_widgets["filename_text"].value)

    def evals_vs_paramvals_plot_refresh(self, change) -> None:
        value_dict = {
            "scan_value": self.qubit_plot_options_widgets[
                "scan_dropdown"
            ].get_interact_value(),
            "scan_range": self.qubit_plot_options_widgets[
                "scan_range_slider"
            ].get_interact_value(),
            "subtract_ground_tf": self.qubit_plot_options_widgets[
                "subtract_ground_checkbox"
            ].get_interact_value(),
            "eigenvalue_state_value": self.qubit_plot_options_widgets[
                "eigenvalue_state_slider"
            ].get_interact_value(),
            **self.get_current_values(),
        }

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            del value_dict["grid"]
            grid_min, grid_max = self.qubit_params_widgets["grid"].get_interact_value()
            value_dict["grid_min_val"] = grid_min
            value_dict["grid_max_val"] = grid_max
            value_dict["grid_pt_count"] = gui_defaults.grid_defaults["grid_pt_count"]

        self.evals_vs_paramvals_plot(**value_dict)

    def wavefunctions_plot_refresh(self, change) -> None:
        value_dict = {
            "mode_value": self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].get_interact_value(),
            **self.get_current_values(),
        }

        if isinstance(self.active_qubit, scq.FullZeroPi):
            self.plot_output.clear_output()
            return
        elif isinstance(
            self.active_qubit, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)
        ):
            value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.qubit_plot_options_widgets[
                "wavefunction_single_state_selector"
            ].get_interact_value()

            if isinstance(self.active_qubit, scq.ZeroPi):
                del value_dict["grid"]
                grid_min, grid_max = self.qubit_params_widgets[
                    "grid"
                ].get_interact_value()
                value_dict["grid_min_val"] = grid_min
                value_dict["grid_max_val"] = grid_max
                value_dict["grid_pt_count"] = gui_defaults.grid_defaults[
                    "grid_pt_count"
                ]
        else:
            value_dict["scale_value"] = self.qubit_plot_options_widgets[
                "wavefunction_scale_slider"
            ].get_interact_value()
            value_dict["eigenvalue_states"] = self.qubit_plot_options_widgets[
                "wavefunction_multi_state_selector"
            ].get_interact_value()

        self.wavefunctions_plot(**value_dict)

    def matelem_vs_paramvals_plot_refresh(self, change) -> None:
        value_dict = {
            "scan_value": self.qubit_plot_options_widgets[
                "scan_dropdown"
            ].get_interact_value(),
            "scan_range": self.qubit_plot_options_widgets[
                "scan_range_slider"
            ].get_interact_value(),
            "operator_value": self.qubit_plot_options_widgets[
                "operator_dropdown"
            ].get_interact_value(),
            "matrix_element_state_value": self.qubit_plot_options_widgets[
                "matrix_element_state_slider"
            ].get_interact_value(),
            "mode_value": self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].get_interact_value(),
            **self.get_current_values(),
        }

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            del value_dict["grid"]
            grid_min, grid_max = self.qubit_params_widgets["grid"].get_interact_value()
            value_dict["grid_min_val"] = grid_min
            value_dict["grid_max_val"] = grid_max
            value_dict["grid_pt_count"] = gui_defaults.grid_defaults["grid_pt_count"]

        self.matelem_vs_paramvals_plot(**value_dict)

    def matrixelements_plot_refresh(self, change):
        value_dict = {
            "operator_value": self.qubit_plot_options_widgets[
                "operator_dropdown"
            ].get_interact_value(),
            "eigenvalue_state_value": self.qubit_plot_options_widgets[
                "eigenvalue_state_slider"
            ].get_interact_value(),
            "mode_value": self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].get_interact_value(),
            "show_numbers_tf": self.qubit_plot_options_widgets[
                "show_numbers_checkbox"
            ].get_interact_value(),
            "show3d_tf": self.qubit_plot_options_widgets[
                "show3d_checkbox"
            ].get_interact_value(),
            **self.get_current_values(),
        }

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            del value_dict["grid"]
            grid_min, grid_max = self.qubit_params_widgets["grid"].get_interact_value()
            value_dict["grid_min_val"] = grid_min
            value_dict["grid_max_val"] = grid_max
            value_dict["grid_pt_count"] = gui_defaults.grid_defaults["grid_pt_count"]

        self.matrixelements_plot(**value_dict)

    # Layout Methods ------------------------------------------------------------------
    def qubit_and_plot_ToggleButtons_layout(self) -> VBox:
        qubit_choice_hbox = HBox([self.qubit_and_plot_ToggleButtons["qubit_buttons"]])
        plot_choice_hbox = HBox([self.qubit_and_plot_ToggleButtons["plot_buttons"]])

        qubit_and_plot_choice_vbox = VBox(
            [qubit_choice_hbox, plot_choice_hbox],
            layout=Layout(width=gui_defaults.WIDTH),
        )

        return qubit_and_plot_choice_vbox

    def manual_update_and_save_layout(self) -> HBox:
        manual_update_and_save_HBox = HBox(
            [
                self.manual_update_and_save_widgets["manual_update_checkbox"],
                self.manual_update_and_save_widgets["update_button"],
                self.manual_update_and_save_widgets["save_button"],
                self.manual_update_and_save_widgets["filename_text"],
            ],
            layout=Layout(width=gui_defaults.WIDTH, justify_content="space-between"),
        )

        return manual_update_and_save_HBox

    def qubit_param_ranges_layout(self) -> HBox:
        param_range_hbox_layout = Layout(
            width=gui_defaults.RANGE_HBOX_WIDTH, justify_content="space-between"
        )
        HBox_layout = Layout(
            display="flex",
            flex_flow="column wrap",
            object_fit="contain",
            width="100%",
            height=gui_defaults.WRAP_HEIGHT,
        )
        qubit_param_ranges_grid = HBox(layout=HBox_layout)

        for param_name, text_widgets in self.qubit_param_ranges_widgets.items():
            param_range_HBox = HBox(layout=param_range_hbox_layout)
            param_label = Label(value=param_name + ":")
            param_range_HBox.children += (
                param_label,
                HBox([text_widgets["min"], text_widgets["max"]]),
            )

            qubit_param_ranges_grid.children += (param_range_HBox,)

        return qubit_param_ranges_grid

    def qubit_plot_layout(self) -> HBox:
        plot_option_vbox = self.plot_option_layout()
        qubit_params_grid = self.qubit_params_grid_layout()

        qubit_plot_layout = HBox([plot_option_vbox, qubit_params_grid])

        return qubit_plot_layout

    def qubit_info_layout(self) -> Box:
        qubit_info_box = Box(layout=Layout(justify_content="center"))
        qubit_info_box.children = [
            self.qubit_plot_options_widgets["qubit_info_image_widget"]
        ]

        return qubit_info_box

    def plot_option_layout(self) -> VBox:
        current_plot_option = self.qubit_and_plot_ToggleButtons[
            "plot_buttons"
        ].get_interact_value()
        VBox_layout = Layout(object_fit="contain", width="50%")
        plot_option_vbox = VBox(layout=VBox_layout)

        if current_plot_option == "Energy spectrum":
            plot_option_vbox.children += self.energy_scan_layout()
        elif current_plot_option == "Wavefunctions":
            plot_option_vbox.children += self.wavefunctions_layout()
        elif current_plot_option == "Matrix element scan":
            plot_option_vbox.children += self.matelem_scan_layout()
        else:
            plot_option_vbox.children += self.matelem_layout()

        return plot_option_vbox

    def qubit_params_grid_layout(self) -> HBox:
        HBox_layout = Layout(
            display="flex",
            flex_flow="column wrap",
            object_fit="contain",
            width="100%",
            height=gui_defaults.WRAP_HEIGHT,
        )
        qubit_params_grid = HBox(
            children=list(self.qubit_params_widgets.values()), layout=HBox_layout
        )

        return qubit_params_grid

    def energy_scan_layout(
        self,
    ) -> Tuple[Dropdown, FloatRangeSlider, Checkbox, IntSlider]:
        """Returns a tuple for the evals_vs_paramvals layout

        Returns
        -------
        Tuple
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = True

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["scan_range_slider"],
            self.qubit_plot_options_widgets["subtract_ground_checkbox"],
            self.qubit_plot_options_widgets["eigenvalue_state_slider"],
        )

        return plot_options_widgets_tuple

    def matelem_scan_layout(
        self,
    ) -> Tuple[Dropdown, Dropdown, FloatRangeSlider, IntSlider, Dropdown]:
        """Returns an interactive for the matelem_vs_paramvals plot

        Returns
        -------
        widgets.interactive
        """
        self.qubit_plot_options_widgets["mode_dropdown"].value = self.active_defaults[
            "mode_matrixelem"
        ]
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = True

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["scan_range_slider"],
            self.qubit_plot_options_widgets["matrix_element_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
        )

        return plot_options_widgets_tuple

    def wavefunctions_layout(self):
        """Returns an interactive for the wavefunction plot

        Returns
        -------
        widgets.interactive
        """
        if isinstance(self.active_qubit, scq.FullZeroPi):
            plot_options_widgets_tuple = (Label(value="This not Implemented"),)
        else:
            self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].value = self.active_defaults["mode_wavefunc"]
            self.qubit_params_widgets[
                self.qubit_plot_options_widgets["scan_dropdown"].value
            ].disabled = False

            if isinstance(
                self.active_qubit, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)
            ):
                which_widget = self.qubit_plot_options_widgets[
                    "wavefunction_single_state_selector"
                ]
            else:
                which_widget = self.qubit_plot_options_widgets[
                    "wavefunction_multi_state_selector"
                ]

            if isinstance(
                self.active_qubit, (scq.ZeroPi, scq.FluxQubit, scq.Cos2PhiQubit)
            ):
                plot_options_widgets_tuple = (
                    which_widget,
                    self.qubit_plot_options_widgets["mode_dropdown"],
                )
            else:
                plot_options_widgets_tuple = (
                    which_widget,
                    self.qubit_plot_options_widgets["mode_dropdown"],
                    self.qubit_plot_options_widgets["manual_scale_checkbox"],
                    self.qubit_plot_options_widgets["wavefunction_scale_slider"],
                )

        return plot_options_widgets_tuple

    def matelem_layout(self):
        """Returns an interactive for the matrix elements plot.

        Returns
        -------
        widgets.interactive
        """
        self.qubit_plot_options_widgets["mode_dropdown"].value = self.active_defaults[
            "mode_matrixelem"
        ]
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = False

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["eigenvalue_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
            self.qubit_plot_options_widgets["show_numbers_checkbox"],
            self.qubit_plot_options_widgets["show3d_checkbox"],
        )

        return plot_options_widgets_tuple

    # Plot functions------------------------------------------------------------------
    def evals_vs_paramvals_plot(
        self,
        scan_value: str,
        scan_range: Tuple[float, float],
        eigenvalue_state_value: int,
        subtract_ground_tf: bool,
        **params: Union[Tuple[float, float], float, int]
    ) -> None:
        """This is the method associated with qubit_plot_interactive that allows for
        us to interact with plot_evals_vs_paramvals().

        Parameters
        ----------
        scan_value:
            Current value of the scan parameter dropdown.

        scan_range:
            Sets the interval [ min, max ] through
            which plot_evals_vs_paramvals() will plot over.

        eigenvalue_state_value:
            The number of states/eigenvalues that will be plotted.

        subtract_ground_tf:
            Determines whether we subtract away the ground energy or not.
            Initially set to False.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        scan_min, scan_max = scan_range
        self.active_qubit.set_params(**params)
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        with self.plot_output:
            plt.cla()
            self.fig, ax = self.active_qubit.plot_evals_vs_paramvals(
                scan_value,
                np_list,
                evals_count=eigenvalue_state_value,
                subtract_ground=subtract_ground_tf,
            )
            self.fig.canvas.header_visible = False
            self.fig.set_size_inches(*gui_defaults.FIG_SIZE_INCHES)
            self.plot_output.clear_output()
            plt.close(1)
            plt.tight_layout()
            plt.show()
        GUI_V2.fig_ax = self.fig, ax

    def wavefunctions_plot(
        self,
        eigenvalue_states: Union[List[int], int],
        mode_value: str,
        scale_value: Optional[float],
        **params: Union[Tuple[float, float], float, int]
    ) -> None:
        """This is the method associated with qubit_plot_interactive that allows for
        us to interact with plot_wavefunction(). Namely, this method is for
        the qubits that have an option for scaling the wavefunction amplitudes.

        Parameters
        ----------
        eigenvalue_states:
            The number of states to be plotted

        mode_value:
            Current value of the mode (e.g. real, imaginary, etc.)

        manual_scale_tf:

        scale_value:

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        self.active_qubit.set_params(**params)
        with self.plot_output:
            plt.cla()
            if isinstance(
                self.active_qubit, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)
            ):
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states, mode=mode_value
                )
            else:
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states, mode=mode_value, scaling=scale_value
                )
            self.fig.canvas.header_visible = False
            self.fig.set_size_inches(*gui_defaults.FIG_SIZE_INCHES)
            self.plot_output.clear_output()
            plt.close(1)
            plt.tight_layout()
            plt.show()
        GUI_V2.fig_ax = self.fig, ax

    def matelem_vs_paramvals_plot(
        self,
        operator_value: str,
        scan_value: str,
        scan_range: Tuple[float, float],
        matrix_element_state_value: int,
        mode_value: str,
        **params: Union[Tuple[float, float], float, int]
    ) -> None:
        """This is the method associated with qubit_plot_interactive that allows for
        us  to interact with plot_matelem_vs_paramvals().

        Parameters
        ----------
        operator_value:
            Current value of the operator dropdown.

        scan_value:
            Current value of the scan parameter dropdown.

        scan_range:
            Sets the interval [ min, max ] through
            which plot_matelem_vs_paramvals() will plot over.

        matrix_element_state_value:
            The number of states/elements that will be shown.

        mode_value:
            Current value of the mode (e.g. real, imaginary, etc.)

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        scan_min, scan_max = scan_range
        self.active_qubit.set_params(**params)
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        with self.plot_output:
            plt.cla()
            self.fig, ax = self.active_qubit.plot_matelem_vs_paramvals(
                operator_value,
                scan_value,
                np_list,
                select_elems=matrix_element_state_value,
                mode=mode_value,
            )
            self.fig.canvas.header_visible = False
            self.fig.set_size_inches(*gui_defaults.FIG_SIZE_INCHES)
            self.plot_output.clear_output()
            plt.close(1)
            plt.tight_layout()
            plt.show()
        GUI_V2.fig_ax = self.fig, ax

    def matrixelements_plot(
        self,
        operator_value: str,
        eigenvalue_state_value: int,
        mode_value: str,
        show_numbers_tf: bool,
        show3d_tf: bool,
        **params: Union[Tuple[float, float], float, int]
    ):
        """This is the method associated with qubit_plot_interactive that allows for
        us to interact with plot_matrixelements().

        Parameters
        ----------
        operator_value:
            Current value of the operator dropdown.

        eigenvalue_state_value:
            The number of states/eigenvalues that will be plotted

        mode_value:
            Current value of the mode (e.g. real, imaginary, etc.)

        show_numbers_tf:
            Determines whether the numerical values will be shown in the 2D plot.
            Initially set to False.

        show3d_tf:
            Determines whether a 3D version of the 2D plot will be shown.
            Initially set to True.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        self.active_qubit.set_params(**params)
        with self.plot_output:
            plt.cla()
            self.fig, ax = self.active_qubit.plot_matrixelements(
                operator_value,
                evals_count=eigenvalue_state_value,
                mode=mode_value,
                show_numbers=show_numbers_tf,
                show3d=show3d_tf,
            )
            self.fig.canvas.header_visible = False
            self.fig.set_size_inches(*gui_defaults.FIG_SIZE_INCHES)
            self.plot_output.clear_output()
            plt.close(1)
            plt.show()
        GUI_V2.fig_ax = self.fig, ax
