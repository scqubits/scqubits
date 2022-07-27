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

import math
import warnings

from distutils.version import StrictVersion

from pathlib import Path
from tkinter.tix import Select
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import get_backend as get_matplotlib_backend
from matplotlib.figure import Axes, Figure

from scqubits.core.discretization import Grid1d
from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.zeropi import ZeroPi
from scqubits.core.zeropi_full import FullZeroPi

try:
    from ipywidgets import (
        Box,
        Button,
        Checkbox,
        Dropdown,
        FloatRangeSlider,
        FloatSlider,
        FloatText,
        HBox,
        HTML,
        Image,
        IntSlider,
        IntText,
        Label,
        Layout,
        Output,
        SelectMultiple,
        Tab,
        Text,
        ToggleButtons,
        VBox,
    )
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

MATPLOTLIB_WIDGET_BACKEND = "module://ipympl.backend_nbagg"
_HAS_WIDGET_BACKEND = get_matplotlib_backend() == MATPLOTLIB_WIDGET_BACKEND

import scqubits as scq
import scqubits.ui.gui_defaults as gui_defaults
import scqubits.core.noise as noise
import scqubits.utils.misc as utils

from scqubits.core.qubit_base import QubitBaseClass


class GUI:
    # Handle to the most recently generated Figure, Axes tuple
    fig_ax: Optional[Tuple[Figure, Axes]] = None

    def __repr__(self):
        return ""

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def __init__(self):
        scq.settings.PROGRESSBAR_DISABLED = False
        scq.settings.T1_DEFAULT_WARNING = False
        if _HAS_WIDGET_BACKEND and StrictVersion(
            matplotlib.__version__
        ) < StrictVersion("3.5.1"):
            warnings.warn(
                "The widget backend requires Matplotlib >=3.5.1 for proper functioning",
                UserWarning,
            )

        # Display Elements
        self.fig: Figure
        ax: Axes
        self.plot_output: Output = Output(layout={"width": "100%"})
        self.tab_widget: Tab = Tab(layout=Layout(width="95%"))

        self.active_qubit: QubitBaseClass
        self.manual_update_bool: bool = False
        self.plot_change_bool: bool = True

        self.active_defaults: Dict[str, Any] = {}
        self.qubit_params: Dict[str, Union[int, float, None]] = {}
        self.ranges_widgets: Dict[str, Dict[str, Union[IntText, FloatText]]] = {}
        self.qubit_scan_params: Dict[str, Union[int, float, None]] = {}
        self.noise_param_widgets: Dict[str, Union[FloatText, IntText]] = {}

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
        self.qubit_params_widgets: Dict[
            str, Union[IntSlider, FloatSlider, FloatRangeSlider]
        ] = {}

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
                value="Wavefunctions",
            ),
        }

    def initialize_manual_update_and_save_widgets_dict(self) -> None:
        """Creates all the widgets associated with manually updating and
        saving plots.
        """
        self.manual_update_and_save_widgets = {
            "manual_update_checkbox": Checkbox(
                value=False,
                description="Manual Update",
                disabled=False,
                indent=False,
                layout=Layout(width="125px"),
            ),
            "update_button": Button(
                description="Update", disabled=True, layout=Layout(width="100px")
            ),
            "save_button": Button(icon="save", layout=Layout(width="35px")),
            "filename_text": Text(
                value=str(Path.cwd().joinpath("plot.pdf")),
                description="",
                disabled=False,
                layout=Layout(width="500px"),
            ),
        }

    def initialize_noise_param_widgets(self) -> None:
        """Creates all the widgets associated with coherence times plots
        """
        self.noise_param_widgets.clear()
        noise_params = ["T", "omega_low", "omega_high", "t_exp"]
        noise_channels = self.active_qubit.supported_noise_channels()

        if "tphi_1_over_f_flux" in noise_channels:
            noise_params.append("A_flux")
        if "tphi_1_over_f_cc" in noise_channels:
            noise_params.append("A_cc")
        if "tphi_1_over_f_ng" in noise_channels:
            noise_params.append("A_ng")
        if (
            "t1_charge_impedance" in noise_channels
            or "t1_flux_bias_line" in noise_channels
        ):
            noise_params.append("R_0")
        if "t1_flux_bias_line" in noise_channels:
            noise_params.append("M")
        if "t1_quasiparticle_tunneling" in noise_channels:
            noise_params.append("x_qp")
            noise_params.append("Delta")

        for noise_param in noise_params:
            self.noise_param_widgets[noise_param] = FloatText(
                value=noise.NOISE_PARAMS[noise_param],
                disalbed=False,
                description=noise_param,
                step=0.001,
            )

    def set_qubit(self, qubit_name: str) -> None:
        """Sets up the chosen qubit to be the active qubit
        and updates the activedefaults and widget dictionaries
        accordingly.
        """
        if qubit_name in gui_defaults.slow_qubits:
            scq.settings.PROGRESSBAR_DISABLED = False
        else:
            scq.settings.PROGRESSBAR_DISABLED = True

        self.active_defaults = gui_defaults.qubit_defaults[qubit_name]
        self.initialize_qubit(qubit_name)
        self.initialize_noise_param_widgets()
        self.initialize_qubit_params_dicts()
        self.initialize_qubit_plot_options_widgets_dict()
        self.initialize_qubit_params_widgets_dict()
        self.initialize_ranges_widgets_dict()

    def initialize_qubit(self, qubit_name: str) -> None:
        """Initializes self.active_qubit to the user's choice
        using the chosen qubit's default parameters.
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
        scan parameter dropdown options.
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
        """Creates all the widgets that will be used for general plotting options."""
        std_layout = Layout(width="95%")

        current_qubit = self.qubit_and_plot_ToggleButtons[
            "qubit_buttons"
        ].get_interact_value()
        operator_dropdown_list = self.active_qubit.get_operator_names()
        scan_dropdown_list = self.qubit_scan_params.keys()
        noise_channel_list = self.active_qubit.supported_noise_channels()
        file = open(self.active_qubit._image_filename, "rb")
        image = file.read()

        self.qubit_plot_options_widgets = {
            "qubit_info_image_widget": Image(
                value=image, format="jpg", layout=Layout(width="80%")
            ),
            "scan_dropdown": Dropdown(
                options=scan_dropdown_list,
                value=self.active_defaults["scan_param"],
                description="Scan over",
                disabled=False,
                layout=std_layout,
            ),
            "mode_dropdown": Dropdown(
                options=gui_defaults.mode_dropdown_list,
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
            "noise_channel_multi-select": SelectMultiple(
                options=noise_channel_list,
                value=noise_channel_list,
                description="Noise Channels",
                disabled=False,
                layout=std_layout,
            ),
            "highest_state_slider": IntSlider(
                min=1,
                max=10,
                value=5,
                continuous_update=False,
                layout=std_layout,
                description="Highest State",
            ),
            "show_numbers_checkbox": Checkbox(
                value=False, description="Show values", disabled=False, indent=False
            ),
            "show3d_checkbox": Checkbox(
                value=True, description="Show 3D", disabled=False, indent=False
            ),
            "subtract_ground_checkbox": Checkbox(
                value=True,
                description="Subtract E\u2080",
                disabled=False,
                indent=False,
            ),
            "coherence_scale_text": FloatText(
                value=1.0,
                disabled=False,
                description="Scale",
                step=0.01,
                layout=std_layout,
            ),
            "i_text": IntText(
                value=1, disabled=False, description="i", step=1, layout=std_layout
            ),
            "j_text": IntText(
                value=0, disabled=False, description="j", step=1, layout=std_layout
            ),
            "t1_checkbox": Checkbox(
                value=False,
                description="Effective T1",
                disabled=False,
                indent=False,
                layout=std_layout,
            ),
            "t2_checkbox": Checkbox(
                value=False,
                description="Effective T2",
                disabled=False,
                indent=False,
                layout=std_layout,
            ),
        }

        if current_qubit in ["Transmon", "TunableTransmon", "Fluxonium"]:
            self.qubit_plot_options_widgets["manual_scale_checkbox"] = Checkbox(
                value=False, description="Manual Scaling", disabled=False, indent=False
            )
            self.qubit_plot_options_widgets["multi_state_selector"] = SelectMultiple(
                options=range(0, 10),
                value=[0, 1, 2, 3, 4],
                description="States",
                disabled=False,
                continuous_update=False,
                layout=std_layout,
            )
            self.qubit_plot_options_widgets["wavefunction_scale_slider"] = FloatSlider(
                min=0.1,
                max=4,
                value=self.active_defaults["scale"],
                disabled=True,
                description="\u03c8 ampl.",
                continuous_update=False,
                layout=std_layout,
            )
        else:
            self.qubit_plot_options_widgets["wavefunction_state_slider"] = IntSlider(
                min=0,
                max=9,
                value=5,
                continuous_update=False,
                layout=std_layout,
                description="State No.",
            )

        if current_qubit in gui_defaults.paramvals_from_papers.keys():
            common_params_dropdown_list = ["Manual"]
            common_params_dropdown_list.extend(
                gui_defaults.paramvals_from_papers[current_qubit].keys()
            )
            self.qubit_plot_options_widgets["common_params_dropdown"] = Dropdown(
                options=common_params_dropdown_list, disabled=False, layout=std_layout
            )
        else:
            self.qubit_plot_options_widgets["common_params_dropdown"] = Label(
                value="None"
            )

        self.qubit_plot_options_widgets["link_HTML"] = HTML(value="")

    def initialize_qubit_params_widgets_dict(self) -> None:
        """Creates all the widgets associated with the parameters of the
        chosen qubit.
        """
        self.qubit_params_widgets.clear()
        std_layout = Layout(width="45%")

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

    def initialize_ranges_widgets_dict(self) -> None:
        """Creates all the widgets associated with changing the ranges of
        certain qubit plot options as well as all of the qubit's parameters.
        """
        self.ranges_widgets.clear()
        range_text_layout = Layout(width="45%")
        total_dict = {**self.qubit_plot_options_widgets, **self.qubit_params_widgets}

        for widget_name, widget in total_dict.items():
            if widget_name == "noise_channel_multi-select":
                continue

            widget_min_text = None
            widget_max_text = None

            if isinstance(widget, IntSlider):
                widget_min_text = IntText(
                    value=widget.min, description="min=", layout=range_text_layout,
                )
                widget_max_text = IntText(
                    value=widget.max, description="max=", layout=range_text_layout,
                )
            elif isinstance(widget, FloatSlider):
                widget_min_text = FloatText(
                    value=widget.min,
                    description="min=",
                    step=0.01,
                    layout=range_text_layout,
                )
                widget_max_text = FloatText(
                    value=widget.max,
                    description="max=",
                    step=0.01,
                    layout=range_text_layout,
                )
            elif isinstance(widget, SelectMultiple):
                min_val = widget.options[0]
                max_val = widget.options[-1]

                widget_min_text = IntText(
                    value=min_val, description="min=", layout=range_text_layout,
                )
                widget_max_text = IntText(
                    value=max_val, description="max=", layout=range_text_layout,
                )
            else:
                continue

            self.ranges_widgets[widget_name] = {
                "min": widget_min_text,
                "max": widget_max_text,
            }

        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            widget_min_text = FloatText(
                value=self.active_qubit._default_grid.min_val,
                description="min=",
                step=0.01,
                layout=range_text_layout,
            )
            widget_max_text = FloatText(
                value=self.active_qubit._default_grid.max_val,
                description="max=",
                step=0.01,
                layout=range_text_layout,
            )
            self.ranges_widgets["wavefunction_domain_slider"] = {
                "min": widget_min_text,
                "max": widget_max_text,
            }

    def initialize_tab_widget(self) -> None:
        """Creates each of the tabs in self.tab_widget"""
        qubit_plot_tab = self.qubit_plot_layout()
        param_ranges_tab = self.ranges_layout()
        qubit_info_tab = self.qubit_info_layout()
        common_qubit_params_tab = self.common_qubit_params_layout()
        noise_param_tab = self.noise_params_layout()

        tab_titles = [
            "Main",
            "Qubit info",
            "Literature params",
            "Param ranges",
            "Noise Params",
        ]
        self.tab_widget.children = [
            qubit_plot_tab,
            qubit_info_tab,
            common_qubit_params_tab,
            param_ranges_tab,
            noise_param_tab,
        ]

        for title_index in range(len(self.tab_widget.children)):
            self.tab_widget.set_title(title_index, tab_titles[title_index])

    def initialize_display(self) -> None:
        """Creates the components of the GUI and displays all these components."""
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
        """Links all the necessary widgets to their desired function."""
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
        self.observe_coherence_elements()
        self.observe_ranges()
        self.observe_plot_elements()
        self.observe_plot_refresh()

    # Retrieval Methods------------------------------------------------------------------
    def get_current_values(self) -> Dict[str, Union[int, float]]:
        """Obtains the current values from each of the qubit parameter
        sliders.

        Returns
        -------
            Dictionary of the current value for each of the qubit's parameters
        """
        current_values_dict = {}
        for param_name, widget in self.qubit_params_widgets.items():
            current_values_dict[param_name] = widget.get_interact_value()
        return current_values_dict

    def get_plot_option_refresh(self) -> Callable[[Any], None]:
        """Obtains the current plot option

        Returns
        -------
            Method pertaining to refreshing the current plot option.
        """
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
        elif current_plot_option == "Coherence times":
            plot_option_refresh = self.coherence_vs_paramvals_plot_refresh

        return plot_option_refresh

    def update_params(self):
        """Uses the current parameter values to set the parameters of the
        active qubit.
        """
        current_values = self.get_current_values()

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            del current_values["grid"]
            grid_min, grid_max = self.qubit_params_widgets["grid"].get_interact_value()
            current_values["grid_min_val"] = grid_min
            current_values["grid_max_val"] = grid_max
            current_values["grid_pt_count"] = gui_defaults.grid_defaults[
                "grid_pt_count"
            ]

        self.active_qubit.set_params(**current_values)

    def check_ranges(
        self,
        new_min: Union[int, float],
        new_max: Union[int, float],
        widget_name: str,
        text_widget: Dict[str, Union[IntText, FloatText]],
        changed_widget_key: str,
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """Adjusts range values so that they make sense/are viable.

        Parameters
        ----------
        new_min:
            The current value of the minimum IntText/FloatText
        new_max:
           The current value of the maximum IntText/FloatText
        widget_name:
            The name of the corresponding parameter/qubit plot option
        text_widget:
            A dictionary that contains the minimum and maximum
            IntText/FloatText widgets
        changed_widget_key:
            A string that encodes whether the minimum or maximum IntText/FloatText
            widget has been changed

        Returns
        -------
            A tuple containing the viable minimum and maximum values for the
            corresponding parameter/qubit plot option widget
        """
        if new_min <= 0 or ("cut" in widget_name and new_min == 1):
            if widget_name == "highest_state_slider":
                new_min = 1
            elif widget_name == "wavefunction_domain_slider" or widget_name == "flux":
                pass
            elif widget_name == "wavefunction_scale_slider":
                new_min = text_widget["min"].step
            elif widget_name in self.qubit_params_widgets.keys():
                if "cut" in widget_name:
                    new_min = 2
                else:
                    new_min = self.active_defaults[widget_name]["min"]
            else:
                new_min = 0
        if new_max <= new_min:
            if (widget_name == "highest_state_slider" and new_min == 1) or (
                widget_name != "wavefunction_domain_slider" and new_min == 0
            ):
                new_max = new_min + text_widget["min"].step
            elif changed_widget_key == "min=":
                new_min = new_max - text_widget["max"].step
            else:
                new_max = new_min + text_widget["min"].step
        return new_min, new_max

    def update_range_values(
        self,
        new_min: Union[int, float],
        new_max: Union[int, float],
        widget_name: str,
        text_widget: Dict[str, Union[IntText, FloatText]],
    ):
        """Adjusts the values of the IntText/FloatText widgets

        Parameters
        ----------
        new_min:
            The current value of the minimum IntText/FloatText
        new_max:
           The current value of the maximum IntText/FloatText
        widget_name:
            The name of the corresponding parameter/qubit plot option
        text_widget:
            A dictionary that contains the minimum and maximum
            IntText/FloatText widgets
        """
        text_widget["min"].value = new_min
        text_widget["max"].value = new_max

        if widget_name in self.qubit_plot_options_widgets.keys():
            widget = self.qubit_plot_options_widgets[widget_name]
        elif widget_name in self.qubit_params_widgets.keys():
            widget = self.qubit_params_widgets[widget_name]
        else:
            widget = None

        if isinstance(widget, SelectMultiple):
            current_values = list(widget.value)
            new_values = []
            widget.options = range(new_min, new_max + 1)
            for value in current_values:
                if value in widget.options:
                    new_values.append(value)
            if len(new_values) == 0:
                new_values.append(widget.options[0])
            widget.value = new_values
        elif widget is None:
            pass
        else:
            widget.min = new_min
            widget.max = new_max

    # Observe Methods-------------------------------------------------------------------
    def observe_ranges(self) -> None:
        for text_widgets in self.ranges_widgets.values():
            text_widgets["min"].observe(self.ranges_update, names="value")
            text_widgets["max"].observe(self.ranges_update, names="value")

    def unobserve_ranges(self) -> None:
        for text_widgets in self.ranges_widgets.values():
            text_widgets["min"].unobserve(self.ranges_update, names="value")
            text_widgets["max"].unobserve(self.ranges_update, names="value")

    def observe_plot_refresh(self) -> None:
        if self.manual_update_bool:
            return
        qubit_plot_options_blacklist = [
            "qubit_info_image_widget",
            "common_params_dropdown",
            "link_HTML",
        ]
        total_dict = {
            **self.qubit_params_widgets,
            **self.qubit_plot_options_widgets,
            **self.noise_param_widgets,
        }
        for widget_name, widget in total_dict.items():
            if widget_name not in qubit_plot_options_blacklist:
                widget.observe(self.plot_refresh, names="value")

    def unobserve_plot_refresh(self) -> None:
        if self.manual_update_bool:
            return
        qubit_plot_options_blacklist = [
            "qubit_info_image_widget",
            "common_params_dropdown",
            "link_HTML",
        ]
        total_dict = {
            **self.qubit_params_widgets,
            **self.qubit_plot_options_widgets,
            **self.noise_param_widgets,
        }
        for widget_name, widget in total_dict.items():
            if widget_name not in qubit_plot_options_blacklist:
                widget.unobserve(self.plot_refresh, names="value")

    def observe_plot_elements(self) -> None:
        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.qubit_plot_options_widgets["manual_scale_checkbox"].observe(
                self.manual_scale_tf, names="value"
            )
        self.qubit_plot_options_widgets["scan_dropdown"].observe(
            self.scan_dropdown_refresh, names="value"
        )
        self.qubit_plot_options_widgets["common_params_dropdown"].observe(
            self.common_params_dropdown_link_refresh, names="value"
        )
        self.qubit_plot_options_widgets["common_params_dropdown"].observe(
            self.common_params_dropdown_params_refresh, names="value"
        )
        self.qubit_plot_options_widgets["t1_checkbox"].observe(
            self.t1_t2_checkbox_update, names="value"
        )
        self.qubit_plot_options_widgets["t2_checkbox"].observe(
            self.t1_t2_checkbox_update, names="value"
        )

        for widget_name, widget in self.qubit_params_widgets.items():
            if "cut" in widget_name:
                widget.observe(self.adjust_state_widgets, names="value")
            widget.observe(self.common_params_dropdown_value_refresh, names="value")

    def unobserve_plot_elements(self) -> None:
        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.qubit_plot_options_widgets["manual_scale_checkbox"].unobserve(
                self.manual_scale_tf, names="value"
            )
        self.qubit_plot_options_widgets["scan_dropdown"].unobserve(
            self.scan_dropdown_refresh, names="value"
        )
        self.qubit_plot_options_widgets["common_params_dropdown"].unobserve(
            self.common_params_dropdown_link_refresh, names="value"
        )
        self.qubit_plot_options_widgets["common_params_dropdown"].unobserve(
            self.common_params_dropdown_params_refresh, names="value"
        )
        self.qubit_plot_options_widgets["t1_checkbox"].unobserve(
            self.t1_t2_checkbox_update, names="value"
        )
        self.qubit_plot_options_widgets["t2_checkbox"].unobserve(
            self.t1_t2_checkbox_update, names="value"
        )

        for widget_name, widget in self.qubit_params_widgets.items():
            if "cut" in widget_name:
                widget.unobserve(self.adjust_state_widgets, names="value")
            widget.unobserve(self.common_params_dropdown_value_refresh, names="value")

    def observe_coherence_elements(self) -> None:
        self.qubit_plot_options_widgets["coherence_scale_text"].observe(
            self.coherence_text, names="value"
        )
        self.qubit_plot_options_widgets["i_text"].observe(
            self.coherence_text, names="value"
        )
        self.qubit_plot_options_widgets["j_text"].observe(
            self.coherence_text, names="value"
        )

        for widget in self.noise_param_widgets.values():
            widget.observe(self.coherence_text, names="value")

    def unobserve_coherence_elements(self) -> None:
        self.qubit_plot_options_widgets["coherence_scale_text"].unobserve(
            self.coherence_text, names="value"
        )
        self.qubit_plot_options_widgets["i_text"].unobserve(
            self.coherence_text, names="value"
        )
        self.qubit_plot_options_widgets["j_text"].unobserve(
            self.coherence_text, names="value"
        )

        for widget in self.noise_param_widgets.values():
            widget.unobserve(self.coherence_text, names="value")

    # Eventhandler Methods -------------------------------------------------------------
    def qubit_change(self, change) -> None:
        self.plot_change_bool = True
        self.plot_output.clear_output()
        new_qubit = change["new"]
        self.unobserve_ranges()
        self.unobserve_coherence_elements()
        self.unobserve_plot_elements()
        self.unobserve_plot_refresh()
        self.manual_update_and_save_widgets["manual_update_checkbox"].unobserve(
            self.manual_update_checkbox, names="value"
        )
        if new_qubit in gui_defaults.slow_qubits:
            self.manual_update_and_save_widgets["manual_update_checkbox"].value = True
            self.manual_update_and_save_widgets["update_button"].disabled = False
            self.manual_update_bool = True
        else:
            self.manual_update_and_save_widgets["manual_update_checkbox"].value = False
            self.manual_update_and_save_widgets["update_button"].disabled = True
            self.manual_update_bool = False
        self.manual_update_and_save_widgets["manual_update_checkbox"].observe(
            self.manual_update_checkbox, names="value"
        )
        self.set_qubit(new_qubit)
        self.initialize_tab_widget()
        self.observe_ranges()
        self.observe_coherence_elements()
        self.observe_plot_elements()
        self.observe_plot_refresh()
        self.plot_refresh(None)

    def scan_dropdown_refresh(self, change) -> None:
        self.qubit_params_widgets[change.old].disabled = False
        self.qubit_params_widgets[change.new].disabled = True

    def plot_option_layout_refresh(self, change) -> None:
        self.plot_change_bool = True
        self.plot_output.clear_output()
        self.unobserve_ranges()
        self.unobserve_plot_elements()
        self.unobserve_plot_refresh()
        self.current_plot_option_refresh = self.get_plot_option_refresh()
        new_plot_option = self.plot_option_layout()

        self.tab_widget.children[0].children[0].children = tuple(
            new_plot_option.children
        )
        self.observe_ranges()
        self.observe_plot_elements()
        self.observe_plot_refresh()
        self.plot_refresh(None)

    def manual_scale_tf(self, change) -> None:
        if change["new"]:
            self.qubit_plot_options_widgets[
                "wavefunction_scale_slider"
            ].disabled = False
        else:
            self.qubit_plot_options_widgets["wavefunction_scale_slider"].disabled = True

    def coherence_text(self, change) -> None:
        self.unobserve_coherence_elements()
        self.unobserve_plot_refresh()
        widget_key = change["owner"].description
        widget = None

        if widget_key == "Scale":
            widget_key = "coherence_scale_text"
        elif widget_key == "i":
            widget_key = "i_text"
        elif widget_key == "j":
            widget_key = "j_text"

        if widget_key in self.qubit_plot_options_widgets.keys():
            widget = self.qubit_plot_options_widgets[widget_key]
        else:
            widget = self.noise_param_widgets[widget_key]

        if change["new"] <= 0:
            if widget_key == "i_text":
                widget.value = 1
            elif widget_key == "j_text":
                widget.value = 0
            else:
                widget.value = widget.step

        i_text_widget = self.qubit_plot_options_widgets["i_text"]
        j_text_widget = self.qubit_plot_options_widgets["j_text"]
        if i_text_widget.get_interact_value() <= j_text_widget.get_interact_value():
            if widget_key == "i_text":
                j_text_widget.value = i_text_widget.value - i_text_widget.step
            else:
                i_text_widget.value = j_text_widget.value + j_text_widget.step
        self.observe_coherence_elements()
        self.observe_plot_refresh()

    def t1_t2_checkbox_update(self, change):
        self.plot_output.clear_output()
        self.plot_change_bool = True

    def manual_update_checkbox(self, change) -> None:
        if change["new"]:
            self.plot_output.clear_output()
            self.manual_update_and_save_widgets["update_button"].disabled = False
            self.unobserve_plot_refresh()
            self.manual_update_bool = True
        else:
            self.manual_update_and_save_widgets["update_button"].disabled = True
            self.manual_update_bool = False
            self.observe_plot_refresh()
            if len(self.plot_output.outputs) == 0:
                self.plot_change_bool = True
            self.plot_refresh(None)

    def manual_update_button_onclick(self, change) -> None:
        self.update_params()
        if len(self.plot_output.outputs) == 0:
            self.plot_change_bool = True
        self.current_plot_option_refresh(None)

    def common_params_dropdown_value_refresh(self, change) -> None:
        current_qubit = self.qubit_and_plot_ToggleButtons[
            "qubit_buttons"
        ].get_interact_value()
        current_dropdown_value = self.qubit_plot_options_widgets[
            "common_params_dropdown"
        ].get_interact_value()

        if current_qubit not in gui_defaults.paramvals_from_papers.keys():
            return
        if current_dropdown_value != "Manual":
            for param_name, param_val in gui_defaults.paramvals_from_papers[
                current_qubit
            ][current_dropdown_value]["params"].items():
                if (
                    self.qubit_params_widgets[param_name].get_interact_value()
                    != param_val
                ):
                    self.qubit_plot_options_widgets[
                        "common_params_dropdown"
                    ].value = "Manual"

    def common_params_dropdown_params_refresh(self, change) -> None:
        current_qubit = self.qubit_and_plot_ToggleButtons[
            "qubit_buttons"
        ].get_interact_value()
        current_dropdown_value = self.qubit_plot_options_widgets[
            "common_params_dropdown"
        ].get_interact_value()

        if current_dropdown_value == "Manual":
            return
        self.unobserve_ranges()
        self.unobserve_plot_refresh()
        self.unobserve_plot_elements()

        params = gui_defaults.paramvals_from_papers[current_qubit][
            current_dropdown_value
        ]["params"]
        for param_name, param_val in params.items():
            param_max = self.ranges_widgets[param_name]["max"].get_interact_value()
            param_min = self.ranges_widgets[param_name]["min"].get_interact_value()

            if param_val < param_min:
                self.ranges_widgets[param_name]["min"].value = self.active_defaults[
                    param_name
                ]["min"]
                self.qubit_params_widgets[param_name].min = self.active_defaults[
                    param_name
                ]["min"]
            if param_val > param_max:
                self.ranges_widgets[param_name]["max"].value = (
                    np.ceil(param_val / 10) * 10
                )
                self.qubit_params_widgets[param_name].max = np.ceil(param_val / 10) * 10

            self.qubit_params_widgets[param_name].value = param_val
        self.observe_ranges()
        self.observe_plot_refresh()
        self.observe_plot_elements()
        self.plot_refresh(None)

    def adjust_state_widgets(self, change) -> None:
        self.unobserve_ranges()
        self.unobserve_plot_refresh()
        self.update_params()
        hilbertdim = self.active_qubit.hilbertdim()
        wavefunction_state_slider_text = self.ranges_widgets["highest_state_slider"]

        if wavefunction_state_slider_text["max"].get_interact_value() >= hilbertdim - 1:
            new_min = wavefunction_state_slider_text["min"].get_interact_value()
            new_max = hilbertdim - 2
            new_min, new_max = self.check_ranges(
                new_min,
                new_max,
                "highest_state_slider",
                wavefunction_state_slider_text,
                "min=",
            )
            self.update_range_values(
                new_min, new_max, "highest_state_slider", wavefunction_state_slider_text
            )

        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            multi_state_selector_text = self.ranges_widgets["multi_state_selector"]

            if multi_state_selector_text["max"].get_interact_value() >= hilbertdim - 2:
                new_min = multi_state_selector_text["min"].get_interact_value()
                new_max = hilbertdim - 3
                new_min, new_max = self.check_ranges(
                    new_min,
                    new_max,
                    "multi_state_selector",
                    multi_state_selector_text,
                    "min=",
                )
                self.update_range_values(
                    new_min, new_max, "multi_state_selector", multi_state_selector_text
                )
        else:
            wavefunction_state_slider_text = self.ranges_widgets[
                "wavefunction_state_slider"
            ]

            if (
                wavefunction_state_slider_text["max"].get_interact_value()
                >= hilbertdim - 2
            ):
                new_min = wavefunction_state_slider_text["min"].get_interact_value()
                new_max = hilbertdim - 3
                new_min, new_max = self.check_ranges(
                    new_min,
                    new_max,
                    "wavefunction_state_slider",
                    wavefunction_state_slider_text,
                    "min=",
                )
                self.update_range_values(
                    new_min,
                    new_max,
                    "wavefunction_state_slider",
                    wavefunction_state_slider_text,
                )
        self.observe_ranges()
        self.observe_plot_refresh()

    def ranges_update(self, change) -> None:
        self.unobserve_ranges()
        self.unobserve_plot_refresh()
        for widget_name, text_widgets in self.ranges_widgets.items():
            new_min = text_widgets["min"].get_interact_value()
            new_max = text_widgets["max"].get_interact_value()
            changed_widget_key = change["owner"].description

            new_min, new_max = self.check_ranges(
                new_min, new_max, widget_name, text_widgets, changed_widget_key
            )

            self.update_range_values(new_min, new_max, widget_name, text_widgets)
        self.observe_ranges()
        self.observe_plot_refresh()
        self.adjust_state_widgets(None)
        self.plot_refresh(None)

    def save_button_clicked_action(self, change) -> None:
        self.fig.savefig(self.manual_update_and_save_widgets["filename_text"].value)

    def plot_refresh(self, change):
        self.update_params()

        if not self.manual_update_bool:
            self.current_plot_option_refresh(None)

    def common_params_dropdown_link_refresh(self, change) -> None:
        current_qubit = self.qubit_and_plot_ToggleButtons[
            "qubit_buttons"
        ].get_interact_value()
        current_dropdown_value = self.qubit_plot_options_widgets[
            "common_params_dropdown"
        ].get_interact_value()

        if current_dropdown_value == "Manual":
            self.qubit_plot_options_widgets["link_HTML"].value = ""
        else:
            link = gui_defaults.paramvals_from_papers[current_qubit][
                current_dropdown_value
            ]["link"]
            self.qubit_plot_options_widgets["link_HTML"].value = (
                "<a href=" + link + " target='_blank'>" + link + "</a>"
            )

    def evals_vs_paramvals_plot_refresh(self, change) -> None:
        scan_dropdown_value = self.qubit_plot_options_widgets[
            "scan_dropdown"
        ].get_interact_value()
        scan_slider = self.qubit_params_widgets[scan_dropdown_value]

        value_dict = {
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.min, scan_slider.max),
            "subtract_ground_tf": self.qubit_plot_options_widgets[
                "subtract_ground_checkbox"
            ].get_interact_value(),
            "eigenvalue_state_value": self.qubit_plot_options_widgets[
                "highest_state_slider"
            ].get_interact_value(),
        }

        self.evals_vs_paramvals_plot(**value_dict)

    def wavefunctions_plot_refresh(self, change) -> None:
        value_dict = {
            "mode_value": self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].get_interact_value(),
        }

        if isinstance(self.active_qubit, scq.FullZeroPi):
            return
        elif isinstance(
            self.active_qubit, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)
        ):
            value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.qubit_plot_options_widgets[
                "wavefunction_state_slider"
            ].get_interact_value()
        else:
            manual_scale_tf_value = self.qubit_plot_options_widgets[
                "manual_scale_checkbox"
            ].get_interact_value()

            if manual_scale_tf_value:
                value_dict["scale_value"] = self.qubit_plot_options_widgets[
                    "wavefunction_scale_slider"
                ].get_interact_value()
            else:
                value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.qubit_plot_options_widgets[
                "multi_state_selector"
            ].get_interact_value()
            value_dict["phi_grid"] = Grid1d(
                min_val=self.ranges_widgets["wavefunction_domain_slider"][
                    "min"
                ].get_interact_value(),
                max_val=self.ranges_widgets["wavefunction_domain_slider"][
                    "max"
                ].get_interact_value(),
                pt_count=self.active_qubit._default_grid.pt_count,
            )

        self.wavefunctions_plot(**value_dict)

    def matelem_vs_paramvals_plot_refresh(self, change) -> None:
        scan_dropdown_value = self.qubit_plot_options_widgets[
            "scan_dropdown"
        ].get_interact_value()
        scan_slider = self.qubit_params_widgets[scan_dropdown_value]

        value_dict = {
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.min, scan_slider.max),
            "operator_value": self.qubit_plot_options_widgets[
                "operator_dropdown"
            ].get_interact_value(),
            "matrix_element_state_value": self.qubit_plot_options_widgets[
                "highest_state_slider"
            ].get_interact_value(),
            "mode_value": self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].get_interact_value(),
        }

        self.matelem_vs_paramvals_plot(**value_dict)

    def matrixelements_plot_refresh(self, change) -> None:
        value_dict = {
            "operator_value": self.qubit_plot_options_widgets[
                "operator_dropdown"
            ].get_interact_value(),
            "eigenvalue_state_value": self.qubit_plot_options_widgets[
                "highest_state_slider"
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
        }

        self.matrixelements_plot(**value_dict)

    def coherence_vs_paramvals_plot_refresh(self, change) -> None:
        t1_effective_tf = self.qubit_plot_options_widgets[
            "t1_checkbox"
        ].get_interact_value()
        t2_effective_tf = self.qubit_plot_options_widgets[
            "t2_checkbox"
        ].get_interact_value()
        scan_dropdown_value = self.qubit_plot_options_widgets[
            "scan_dropdown"
        ].get_interact_value()
        scan_slider = self.qubit_params_widgets[scan_dropdown_value]
        noise_channel_options = list(
            self.qubit_plot_options_widgets[
                "noise_channel_multi-select"
            ].get_interact_value()
        )
        common_noise_options = {
            "i": self.qubit_plot_options_widgets["i_text"].get_interact_value(),
            "j": self.qubit_plot_options_widgets["j_text"].get_interact_value(),
            "T": self.noise_param_widgets["T"].get_interact_value(),
        }
        noise_params_dict = {}

        for noise_param in self.noise_param_widgets.keys():
            if noise_param != "T":
                param_val = self.noise_param_widgets[noise_param].get_interact_value()
                noise_params_dict[noise_param] = param_val

        noise_channels = {"t1_eff": [], "t2_eff": [], "coherence_times": []}
        for noise_channel in noise_channel_options:
            if "tphi_1_over_f" in noise_channel:
                tphi_dict = {
                    "omega_low": noise_params_dict["omega_low"],
                    "omega_high": noise_params_dict["omega_high"],
                    "t_exp": noise_params_dict["t_exp"],
                }
                if "flux" in noise_channel:
                    tphi_dict["A_noise"] = noise_params_dict["A_flux"]
                elif "cc" in noise_channel:
                    tphi_dict["A_noise"] = noise_params_dict["A_cc"]
                elif "ng" in noise_channel:
                    tphi_dict["A_noise"] = noise_params_dict["A_ng"]

                noise_channels["t2_eff"].append((noise_channel, tphi_dict))
                noise_channels["coherence_times"].append((noise_channel, tphi_dict))
            elif noise_channel == "t1_flux_bias_line":
                noise_channels["t1_eff"].append(
                    (
                        noise_channel,
                        dict(M=noise_params_dict["M"], Z=noise_params_dict["R_0"]),
                    )
                )
                noise_channels["t2_eff"].append(
                    (
                        noise_channel,
                        dict(M=noise_params_dict["M"], Z=noise_params_dict["R_0"]),
                    )
                )
                noise_channels["coherence_times"].append(
                    (
                        noise_channel,
                        dict(M=noise_params_dict["M"], Z=noise_params_dict["R_0"]),
                    )
                )
            elif noise_channel == "t1_charge_impedance":
                noise_channels["coherence_times"].append(
                    (noise_channel, dict(Z=noise_params_dict["R_0"]))
                )
            elif noise_channel == "t1_quasiparticle_tunneling":
                noise_channels["t1_eff"].append(
                    (
                        noise_channel,
                        dict(
                            x_qp=noise_params_dict["x_qp"],
                            Delta=noise_params_dict["Delta"],
                        ),
                    )
                )
                noise_channels["t2_eff"].append(
                    (
                        noise_channel,
                        dict(
                            x_qp=noise_params_dict["x_qp"],
                            Delta=noise_params_dict["Delta"],
                        ),
                    )
                )
                noise_channels["coherence_times"].append(
                    (
                        noise_channel,
                        dict(
                            x_qp=noise_params_dict["x_qp"],
                            Delta=noise_params_dict["Delta"],
                        ),
                    )
                )
            else:
                noise_channels["t1_eff"].append(noise_channel)
                noise_channels["t2_eff"].append(noise_channel)
                noise_channels["coherence_times"].append(noise_channel)

        value_dict = {
            "t1_effective_tf": t1_effective_tf,
            "t2_effective_tf": t2_effective_tf,
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.min, scan_slider.max),
            "noise_channels": noise_channels,
            "scale": self.qubit_plot_options_widgets[
                "coherence_scale_text"
            ].get_interact_value(),
            "common_noise_options": common_noise_options,
        }

        self.coherence_vs_paramvals_plot(**value_dict)

    # Layout Methods ------------------------------------------------------------------
    def qubit_and_plot_ToggleButtons_layout(self) -> VBox:
        qubit_choice_hbox = HBox([self.qubit_and_plot_ToggleButtons["qubit_buttons"]])
        plot_choice_hbox = HBox([self.qubit_and_plot_ToggleButtons["plot_buttons"]])

        qubit_and_plot_choice_vbox = VBox([qubit_choice_hbox, plot_choice_hbox])

        return qubit_and_plot_choice_vbox

    def manual_update_and_save_layout(self) -> HBox:
        manual_update_HBox = HBox(
            [
                self.manual_update_and_save_widgets["manual_update_checkbox"],
                self.manual_update_and_save_widgets["update_button"],
            ],
            layout=Layout(justify_content="flex-start"),
        )

        save_HBox = HBox(
            [
                self.manual_update_and_save_widgets["save_button"],
                self.manual_update_and_save_widgets["filename_text"],
            ],
            layout=Layout(justify_content="flex-end"),
        )

        manual_update_and_save_HBox = HBox(
            [manual_update_HBox, save_HBox],
            layout=Layout(width="95%", justify_content="space-between"),
        )

        return manual_update_and_save_HBox

    def ranges_layout(self) -> HBox:
        range_hbox_layout = Layout(width="50%", justify_content="flex-end")
        grid_hbox_layout = Layout(
            display="flex", flex_flow="row wrap", object_fit="contain", width="100%"
        )
        ranges_grid_hbox = HBox(layout=grid_hbox_layout)

        for widget_name, text_widgets in self.ranges_widgets.items():
            if widget_name == "highest_state_slider":
                widget_name = "Highest State"
            elif widget_name == "multi_state_selector":
                widget_name = "States"
            elif widget_name == "wavefunction_state_slider":
                widget_name = "State No."
            elif widget_name == "wavefunction_scale_slider":
                widget_name = "Scale"
            elif widget_name == "wavefunction_domain_slider":
                widget_name = "Wavefunction"

            range_hbox = HBox(layout=range_hbox_layout)
            widget_label = Label(
                value=widget_name + ":", layout=Layout(justify_content="flex-end")
            )
            range_hbox.children += (
                widget_label,
                HBox(
                    [text_widgets["min"], text_widgets["max"]],
                    layout=Layout(width="80%"),
                ),
            )

            ranges_grid_hbox.children += (range_hbox,)

        return ranges_grid_hbox

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

    def common_qubit_params_layout(self) -> HBox:
        dropdown_box = Box(
            [self.qubit_plot_options_widgets["common_params_dropdown"]],
            layout=Layout(width="50%"),
        )
        link_box = Box(
            [self.qubit_plot_options_widgets["link_HTML"]],
            layout=Layout(display="flex", width="50%"),
        )
        common_qubit_params_HBox = HBox(
            [dropdown_box, link_box], layout=Layout(width="100%")
        )
        return common_qubit_params_HBox

    def noise_params_layout(self) -> HBox:
        HBox_layout = Layout(display="flex", object_fit="contain", width="100%")
        noise_params_grid = HBox(layout=HBox_layout)

        params_size = len(self.noise_param_widgets)
        if params_size > 6:
            left_right_HBox_layout = Layout(
                display="flex",
                flex_flow="column nowrap",
                object_fit="contain",
                width="50%",
            )
            left_HBox = HBox(layout=left_right_HBox_layout)
            right_HBox = HBox(layout=left_right_HBox_layout)

            counter = 1
            for noise_param_widget in self.noise_param_widgets.values():
                noise_param_widget.layout.width = "50%"
                if params_size % 2 == 0:
                    if counter <= params_size / 2:
                        left_HBox.children += (noise_param_widget,)
                    else:
                        right_HBox.children += (noise_param_widget,)
                else:
                    if counter <= params_size / 2 + 1:
                        left_HBox.children += (noise_param_widget,)
                    else:
                        right_HBox.children += (noise_param_widget,)
                counter += 1

            noise_params_grid.children += (
                left_HBox,
                right_HBox,
            )
        else:
            noise_params_grid.layout.flex_flow = "column nowrap"
            noise_params_grid.children = list(self.noise_param_widgets.values())

        return noise_params_grid

    def plot_option_layout(self) -> VBox:
        current_plot_option = self.qubit_and_plot_ToggleButtons[
            "plot_buttons"
        ].get_interact_value()
        VBox_layout = Layout(width="35%")
        plot_option_vbox = VBox(layout=VBox_layout)

        if current_plot_option == "Energy spectrum":
            plot_option_vbox.children += self.energy_scan_layout()
        elif current_plot_option == "Wavefunctions":
            plot_option_vbox.children += self.wavefunctions_layout()
        elif current_plot_option == "Matrix element scan":
            plot_option_vbox.children += self.matelem_scan_layout()
        elif current_plot_option == "Matrix elements":
            plot_option_vbox.children += self.matelem_layout()
        elif current_plot_option == "Coherence times":
            plot_option_vbox.children += self.coherence_times_layout()

        return plot_option_vbox

    def qubit_params_grid_layout(self) -> HBox:
        HBox_layout = Layout(display="flex", object_fit="contain", width="65%")
        qubit_params_grid = HBox(layout=HBox_layout)

        params_size = len(self.qubit_params_widgets)
        if params_size > 6:
            left_right_HBox_layout = Layout(
                display="flex",
                flex_flow="column nowrap",
                object_fit="contain",
                width="50%",
            )
            left_HBox = HBox(layout=left_right_HBox_layout)
            right_HBox = HBox(layout=left_right_HBox_layout)

            counter = 1
            for param_slider in self.qubit_params_widgets.values():
                param_slider.layout.width = "95%"
                if params_size % 2 == 0:
                    if counter <= params_size / 2:
                        left_HBox.children += (param_slider,)
                    else:
                        right_HBox.children += (param_slider,)
                else:
                    if counter <= params_size / 2 + 1:
                        left_HBox.children += (param_slider,)
                    else:
                        right_HBox.children += (param_slider,)
                counter += 1

            qubit_params_grid.children += (
                left_HBox,
                right_HBox,
            )
        else:
            qubit_params_grid.layout.flex_flow = "column nowrap"
            qubit_params_grid.children = list(self.qubit_params_widgets.values())

        return qubit_params_grid

    def energy_scan_layout(self) -> Tuple[Dropdown, Checkbox, IntSlider]:
        """Creates the children for energy scan layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = True

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["subtract_ground_checkbox"],
            self.qubit_plot_options_widgets["highest_state_slider"],
        )

        return plot_options_widgets_tuple

    def matelem_scan_layout(self) -> Tuple[Dropdown, Dropdown, IntSlider, Dropdown]:
        """Creates the children for matrix elements scan layout.

        Returns
        -------
            Tuple of plot options widgets
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
            self.qubit_plot_options_widgets["highest_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
        )

        return plot_options_widgets_tuple

    def wavefunctions_layout(
        self,
    ) -> Union[
        Tuple[Label],
        Tuple[Union[IntSlider, SelectMultiple], Dropdown],
        Tuple[Union[IntSlider, SelectMultiple], Dropdown, Checkbox, IntSlider],
    ]:
        """Creates the children for the wavefunctions layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        if isinstance(self.active_qubit, scq.FullZeroPi):
            plot_options_widgets_tuple = (Label(value="Not implemented"),)
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
                    "wavefunction_state_slider"
                ]
            else:
                which_widget = self.qubit_plot_options_widgets["multi_state_selector"]

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

    def matelem_layout(
        self,
    ) -> Tuple[Dropdown, IntSlider, Dropdown, Checkbox, Checkbox]:
        """Creates the children for matrix elements layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_plot_options_widgets["mode_dropdown"].value = self.active_defaults[
            "mode_matrixelem"
        ]
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = False

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["highest_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
            self.qubit_plot_options_widgets["show_numbers_checkbox"],
            self.qubit_plot_options_widgets["show3d_checkbox"],
        )

        return plot_options_widgets_tuple

    def coherence_times_layout(self,) -> Tuple[Dropdown, SelectMultiple]:
        """Creates the children for matrix elements layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = True
        text_VBox = VBox(
            [
                self.qubit_plot_options_widgets["coherence_scale_text"],
                self.qubit_plot_options_widgets["i_text"],
                self.qubit_plot_options_widgets["j_text"],
            ],
            layout=Layout(width="50%"),
        )
        checkbox_VBox = VBox(
            [
                self.qubit_plot_options_widgets["t1_checkbox"],
                self.qubit_plot_options_widgets["t2_checkbox"],
            ],
            layout=Layout(width="35%"),
        )

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["noise_channel_multi-select"],
            HBox(
                [text_VBox, checkbox_VBox],
                layout=Layout(
                    display="flex", justify_content="space-between", width="95%"
                ),
            ),
        )

        return plot_options_widgets_tuple

    # Plot functions------------------------------------------------------------------
    def evals_vs_paramvals_plot(
        self,
        scan_value: str,
        scan_range: Tuple[float, float],
        eigenvalue_state_value: int,
        subtract_ground_tf: bool,
    ) -> None:
        """This method will refresh the energy vs paramvals plot using the current
        values of the plot options widgets as well as the qubit param widgets.

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
        """
        if not _HAS_WIDGET_BACKEND:
            self.plot_output.clear_output(wait=True)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        if self.plot_change_bool:
            self.fig, ax = self.active_qubit.plot_evals_vs_paramvals(
                scan_value,
                np_list,
                evals_count=eigenvalue_state_value,
                subtract_ground=subtract_ground_tf,
            )
            self.plot_change_bool = False

            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
                with self.plot_output:
                    plt.show()
        else:
            self.fig.axes[0].clear()
            self.active_qubit.plot_evals_vs_paramvals(
                scan_value,
                np_list,
                evals_count=eigenvalue_state_value,
                subtract_ground=subtract_ground_tf,
                fig_ax=(self.fig, self.fig.axes[0]),
            )
        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes[0]

    def wavefunctions_plot(
        self,
        eigenvalue_states: Union[List[int], int],
        mode_value: str,
        scale_value: Optional[float] = None,
        phi_grid: Optional[Grid1d] = None,
    ) -> None:
        """This method will refresh the wavefunctions plot using the current
        values of the plot options widgets as well as the qubit param widgets.

        Parameters
        ----------
        eigenvalue_states:
            The number of states to be plotted
        mode_value:
            Current value of the mode (e.g. real, imaginary, etc.)
        scale_value:
            The current value for the wavefunction scale
        phi_grid:
            Specifies the domain over which the wavefunction will be plotted.
        """
        if not _HAS_WIDGET_BACKEND:
            self.plot_output.clear_output(wait=True)

        if isinstance(self.active_qubit, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)):
            if self.plot_change_bool:
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states, mode=mode_value,
                )
                self.plot_change_bool = False

                if _HAS_WIDGET_BACKEND:
                    self.fig.canvas.header_visible = False
                    self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
                    with self.plot_output:
                        plt.show()
            else:
                self.fig.axes[0].clear()
                self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states,
                    mode=mode_value,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
        else:
            self.plot_output.outputs = tuple(
                elem
                for elem in self.plot_output.outputs
                if "Label" not in elem["data"]["text/plain"]
            )
            if len(eigenvalue_states) == 0:
                if _HAS_WIDGET_BACKEND:
                    self.fig.axes[0].clear()
                error_label = Label(value="Please select at least one state.")
                with self.plot_output:
                    display(error_label)
                return
            if self.plot_change_bool:
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states,
                    mode=mode_value,
                    scaling=scale_value,
                    phi_grid=phi_grid,
                )
                self.plot_change_bool = False

                if _HAS_WIDGET_BACKEND:
                    self.fig.canvas.header_visible = False
                    self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
                    with self.plot_output:
                        plt.show()
            else:
                self.fig.axes[0].clear()
                self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states,
                    mode=mode_value,
                    scaling=scale_value,
                    phi_grid=phi_grid,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )

        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes[0]

    def matelem_vs_paramvals_plot(
        self,
        operator_value: str,
        scan_value: str,
        scan_range: Tuple[float, float],
        matrix_element_state_value: int,
        mode_value: str,
    ) -> None:
        """This method will refresh the matrix elements vs paramvals plot using the
        current values of the plot options widgets as well as the qubit param widgets.

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
        """
        if not _HAS_WIDGET_BACKEND:
            self.plot_output.clear_output(wait=True)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        if self.plot_change_bool:
            self.fig, ax = self.active_qubit.plot_matelem_vs_paramvals(
                operator_value,
                scan_value,
                np_list,
                select_elems=matrix_element_state_value,
                mode=mode_value,
            )
            self.plot_change_bool = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
                with self.plot_output:
                    plt.show()
        else:
            self.fig.axes[0].clear()
            self.active_qubit.plot_matelem_vs_paramvals(
                operator_value,
                scan_value,
                np_list,
                select_elems=matrix_element_state_value,
                mode=mode_value,
                fig_ax=(self.fig, self.fig.axes[0]),
            )
        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes

    def matrixelements_plot(
        self,
        operator_value: str,
        eigenvalue_state_value: int,
        mode_value: str,
        show_numbers_tf: bool,
        show3d_tf: bool,
    ):
        """This method will refresh the matrix elements plot using the current
        values of the plot options widgets as well as the qubit param widgets.

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
        """
        if not _HAS_WIDGET_BACKEND:
            self.plot_output.clear_output(wait=True)
        if self.plot_change_bool:
            self.fig, ax = self.active_qubit.plot_matrixelements(
                operator_value,
                evals_count=eigenvalue_state_value,
                mode=mode_value,
                show_numbers=show_numbers_tf,
                show3d=show3d_tf,
            )
            self.plot_change_bool = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
                with self.plot_output:
                    plt.show()
        else:
            self.fig.delaxes(self.fig.axes[2])
            self.fig.delaxes(self.fig.axes[1])
            self.fig.add_subplot(122)
            self.fig.axes[0].clear()
            self.active_qubit.plot_matrixelements(
                operator_value,
                evals_count=eigenvalue_state_value,
                mode=mode_value,
                show_numbers=show_numbers_tf,
                show3d=show3d_tf,
                fig_ax=(self.fig, (self.fig.axes[0], self.fig.axes[1])),
            )
            if _HAS_WIDGET_BACKEND:
                self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)

        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes

    def coherence_vs_paramvals_plot(
        self,
        scan_value: str,
        scan_range: Tuple[float, float],
        noise_channels: Dict[str, List[Union[str, Tuple[str, Dict[str, float]]]]],
        scale: float,
        common_noise_options: Dict[str, Union[int, float]],
        t1_effective_tf: bool,
        t2_effective_tf: bool,
    ) -> None:
        """This method will refresh the coherence vs paramvals plot using the current
        values of the plot options widgets as well as the qubit param widgets.

        Parameters
        ----------
        scan_value:
            Current value of the scan parameter dropdown.
        scan_range:
            Sets the interval [ min, max ] through
            which plot_evals_vs_paramvals() will plot over.
        noise_channels:
            List of noise channels to be displayed
        """
        if not _HAS_WIDGET_BACKEND:
            self.plot_output.clear_output(wait=True)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        if len(noise_channels) == 0:
            error_label = Label(value="Please select at least one noise channel.")
            display(error_label)
            return
        if self.plot_change_bool:
            if not t1_effective_tf and not t2_effective_tf:
                self.fig, ax = self.active_qubit.plot_coherence_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["coherence_times"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                )
            elif t1_effective_tf and not t2_effective_tf:
                self.fig, ax = self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                )
            elif not t1_effective_tf and t2_effective_tf:
                self.fig, ax = self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                )
            else:
                self.fig, ax = plt.subplots(nrows=1, ncols=2)
                self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, ax[0]),
                )
                self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, ax[1]),
                )
            self.plot_change_bool = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
                with self.plot_output:
                    plt.show()
        else:
            for ax in self.fig.axes:
                ax.clear()
            if not t1_effective_tf and not t2_effective_tf:
                axes = np.array(self.fig.axes).reshape(
                    math.ceil(len(self.fig.axes) / 2), 2
                )
                self.active_qubit.plot_coherence_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["coherence_times"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, axes),
                )
            elif t1_effective_tf and not t2_effective_tf:
                self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
            elif not t1_effective_tf and t2_effective_tf:
                self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
            else:
                self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
                self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    scale=scale,
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[1]),
                )
        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes
