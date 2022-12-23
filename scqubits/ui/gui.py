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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import get_backend as get_matplotlib_backend
from matplotlib.figure import Axes, Figure

import scqubits as scq
import scqubits.ui.gui_defaults as gui_defaults
import scqubits.ui.custom_ipyvuetify as ui
import scqubits.core.noise as noise
import scqubits.utils.misc as utils

from scqubits.core.discretization import Grid1d
from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.zeropi import ZeroPi
from scqubits.core.zeropi_full import FullZeroPi
from scqubits.settings import matplotlib_settings


try:
    import ipywidgets
    import ipyvuetify as v
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


class GUI:
    # Handle to the most recently generated Figure, Axes tuple
    fig_ax: Optional[Tuple[Figure, Axes]] = None

    autoconnect_blacklist = [
        "qubit_info_image_widget",
        "common_params_dropdown",
        "scan_dropdown",
        "link_HTML",
        "plot_buttons",
    ]

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
        self.plot_output: Output = ipywidgets.Output()
        self.tab_widget: v.Tabs = v.Tabs(
            children=[], elevation="0", background_color="#f9fbff"
        )

        self.active_qubit: QubitBaseClass
        self.manual_update_bool = False
        self.plot_change_bool = True

        self.active_defaults: Dict[str, Any] = {}
        self.qubit_params: Dict[str, Union[int, float, None]] = {}
        self.ranges_widgets: Dict[str, Dict[str, Union[IntText, FloatText]]] = {}
        self.qubit_scan_params: Dict[str, Union[int, float, None]] = {}
        self.noise_param_widgets: Dict[str, Union[FloatText, IntText]] = {}

        self.qubit_and_plot_choice_widgets: Dict[str, ToggleButtons] = {}
        self.manual_update_and_save_widgets: Dict[str, Union[Checkbox, Button]] = {}
        self.qubit_plot_options_widgets: Dict[str, Any] = {}
        self.qubit_params_widgets: Dict[str, Any] = {}
        self.qubit_params_widgets_txt: Dict[str, Any] = {}

        # ------------------------------------------------------------------------------

        self.initialize_qubit_and_plot_ToggleButtons()
        self.initialize_manual_update_and_save_widgets_dict()
        starting_qubit = self.qubit_and_plot_choice_widgets["qubit_dropdown"].v_model

        self.set_qubit(starting_qubit)

        self.current_plot_option_refresh = self.get_plot_option_refresh()

        self.initialize_display()

        self.initialize_observe()

        self.current_plot_option_refresh()

    # Initialization Methods -----------------------------------------------------------

    def initialize_qubit_and_plot_ToggleButtons(self) -> None:
        """Creates all the ToggleButtons that controls
        which qubit or plot the user can choose from.
        """
        self.qubit_and_plot_choice_widgets = {
            "qubit_dropdown": v.Select(
                v_model=gui_defaults.supported_qubits[0],
                items=gui_defaults.supported_qubits,
                filled=True,
                label="Qubit",
                class_="pl-2",
                style_="min-width: 200px",
            ),
            "plot_buttons": v.BtnToggle(
                v_model=0,
                mandatory=True,
                style_="background: #f9fbff",
                children=[
                    v.Col(
                        style_="background: #f9fbff",
                        children=[
                            v.Btn(
                                children=[plot_choice],
                                text=True,
                            )
                            for plot_choice in gui_defaults.plot_choices
                        ],
                    )
                ],
            ),
        }

    #
    def initialize_manual_update_and_save_widgets_dict(self) -> None:
        """Creates all the widgets associated with manually updating and
        saving plots.
        """
        self.manual_update_and_save_widgets = {
            "manual_update_checkbox": v.Switch(
                v_model=False,
                class_="px-4",
                label="Manual refresh",
            ),
            "update_button": v.Btn(
                children=[v.Icon(children=["mdi-refresh"])],
                fab=True,
                color="primary",
                small=True,
                disabled=True,
                elevation="0",
                class_="px-2",
            ),
            "save_button": v.Btn(
                class_="ml-5 pmr-2",
                height=40,
                width=40,
                min_width=40,
                children=[v.Icon(children=["mdi-download"])],
            ),
            "filename_text": v.TextField(
                class_="ml-3 pl-3",
                style_="max-width: 600px",
                v_model=str(Path.cwd().joinpath("plot.pdf")),
                label="Save As",
            ),
        }

    #
    def initialize_noise_param_widgets(self) -> None:
        """Creates all the widgets associated with coherence times plots"""
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
            self.noise_param_widgets[noise_param] = ui.FloatTextField(
                v_model=noise.NOISE_PARAMS[noise_param],
                name=noise_param,
                label=noise_param,
                step=0.001,
                style_="max-width: 180px",
            )

    def set_qubit(self, qubit_name: str) -> None:
        """Sets up the chosen qubit to be the active qubit
        and updates the active defaults and widget dictionaries
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
        self.initialize_qubit_params_widgets_dict()
        self.initialize_qubit_plot_options_widgets_dict()

        self.initialize_ranges_widgets_dict()

    def initialize_qubit(self, qubit_name: str) -> None:
        """Initializes self.active_qubit to the user's choice
        using the chosen qubit's default parameters.
        """
        QubitClass = getattr(scq, qubit_name)
        init_params = QubitClass.default_params()

        if qubit_name == "ZeroPi" or qubit_name == "FullZeroPi":
            init_params["grid"] = Grid1d(-7 * np.pi, 7 * np.pi, 200)
        self.active_qubit = QubitClass(**init_params)

    #
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
        current_qubit = self.qubit_and_plot_choice_widgets["qubit_dropdown"].v_model
        operator_dropdown_list = self.active_qubit.get_operator_names()
        scan_dropdown_list = list(self.qubit_scan_params.keys())
        noise_channel_list = self.active_qubit.supported_noise_channels()
        file = open(self.active_qubit._image_filename, "rb")
        image = file.read()

        self.qubit_plot_options_widgets = {
            "qubit_info_image_widget": ipywidgets.Image(
                value=image, format="jpg", layout=ipywidgets.Layout(width="100%")
            ),
            "scan_dropdown": v.Select(
                items=scan_dropdown_list,
                outlined=True,
                v_model=self.active_defaults["scan_param"],
                label="Scan over",
            ),
            "mode_dropdown": v.Select(
                v_model=gui_defaults.mode_dropdown_list[0],
                items=[item[0] for item in gui_defaults.mode_dropdown_list],
                label="Plot as:",
            ),
            "operator_dropdown": v.Select(
                items=operator_dropdown_list,
                v_model=self.active_defaults["operator"],
                label="Operator",
            ),
            "noise_channel_multi-select": v.Select(
                multiple=True,
                chips=True,
                items=noise_channel_list,
                v_model=noise_channel_list,
                height=100,
                label="Noise channels",
            ),
            "highest_state_slider": v.Slider(
                thumb_label="always",
                thumb_size="24",
                min=1,
                max=10,
                style_="min-width: 200px",
                v_model=5,
                label="Max level",
            ),
            "show_numbers_checkbox": v.Switch(v_model=False, label="Show values"),
            "show3d_checkbox": v.Switch(v_model=True, label="Show 3D"),
            "subtract_ground_checkbox": v.Switch(
                v_model=True,
                label="Subtract E\u2080",
            ),
            "i_text": ui.IntTextField(v_model=1, name="i", on_change=self.plot_refresh),
            "j_text": ui.IntTextField(v_model=0, name="j", on_change=self.plot_refresh),
            "t1_checkbox": v.Switch(
                v_model=False,
                label="Effective T1",
            ),
            "t2_checkbox": v.Switch(
                v_model=False,
                label="Effective T2",
            ),
        }

        if current_qubit in ["Transmon", "TunableTransmon", "Fluxonium"]:
            self.qubit_plot_options_widgets["manual_scale_checkbox"] = v.Switch(
                v_model=False, label="Manual Scaling"
            )
            self.qubit_plot_options_widgets["multi_state_selector"] = v.Select(
                multiple=True,
                items=list(range(0, 10)),
                v_model=[0, 1, 2, 3, 4],
                label="States",
            )
            self.qubit_plot_options_widgets["wavefunction_scale_slider"] = v.Slider(
                min=0.1,
                max=4,
                v_model=self.active_defaults["scale"],
                disabled=True,
                label="\u03c8 ampl.",
            )
        else:
            self.qubit_plot_options_widgets["wavefunction_state_slider"] = v.Slider(
                min=0,
                max=9,
                width=220,
                v_model=5,
                label="State no.",
            )

        if current_qubit in gui_defaults.paramvals_from_papers.keys():
            common_params_dropdown_list = ["User specified"]
            common_params_dropdown_list.extend(
                gui_defaults.paramvals_from_papers[current_qubit].keys()
            )
            self.qubit_plot_options_widgets["common_params_dropdown"] = v.Select(
                class_="py-5",
                style_="max-width: 400px;",
                label="Select qubit parameters",
                items=common_params_dropdown_list,
                v_model=common_params_dropdown_list[0],
            )
        else:
            self.qubit_plot_options_widgets["common_params_dropdown"] = v.Text(
                children=["None"]
            )

        self.qubit_plot_options_widgets["link_HTML"] = v.Text(children=[""])

    def initialize_qubit_params_widgets_dict(self) -> None:
        """Creates all the widgets associated with the parameters of the
        chosen qubit.
        """
        self.qubit_params_widgets.clear()

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            grid_min = self.active_qubit.grid.min_val
            grid_max = self.active_qubit.grid.max_val
            self.qubit_params_widgets["grid"] = v.RangeSlider(
                min=-12 * np.pi,
                max=12 * np.pi,
                v_model=[grid_min, grid_max],
                step=0.05,
                label="Grid range",
            )

        for param_name, param_val in self.qubit_params.items():
            if isinstance(param_val, int):
                kwargs = (
                    self.active_defaults.get(param_name) or self.active_defaults["int"]
                )
                (
                    self.qubit_params_widgets[param_name],
                    self.qubit_params_widgets_txt[param_name],
                ) = ui.make_slider_textfield(
                    **kwargs,
                    style_="max-width: 200px",
                    v_model=param_val,
                    label=f"{param_name}",
                )
            else:
                kwargs = (
                    self.active_defaults.get(param_name)
                    or self.active_defaults["float"]
                )
                (
                    self.qubit_params_widgets[param_name],
                    self.qubit_params_widgets_txt[param_name],
                ) = ui.make_slider_textfield(
                    **kwargs,
                    style_="max-width: 200px",
                    v_model=param_val,
                    step=0.01,
                    label=f"{param_name}",
                )

    def initialize_ranges_widgets_dict(self) -> None:
        """Creates all the widgets associated with changing the ranges of
        certain qubit plot options as well as all of the qubit's parameters.
        """
        self.ranges_widgets.clear()
        total_dict = {**self.qubit_plot_options_widgets, **self.qubit_params_widgets}

        for widget_name, widget in total_dict.items():
            if widget_name == "noise_channel_multi-select":
                continue

            widget_min_text = None
            widget_max_text = None

            if isinstance(widget, v.RangeSlider):
                widget_min_text = ui.IntTextField(
                    v_model=widget.min, label="min", name="min"
                )
                widget_max_text = ui.IntTextField(
                    v_model=widget.max, label="max", name="max"
                )
            elif isinstance(widget, (v.RangeSlider)) and isinstance(
                widget.v_model, float
            ):
                widget_min_text = ui.FloatTextField(
                    v_model=widget.min,
                    step=0.01,
                    label="min",
                )
                widget_max_text = ui.FloatTextField(
                    v_model=widget.max,
                    step=0.01,
                    label="max",
                )
            elif isinstance(widget, v.Select) and widget.multiple:
                min_val = widget.items[0]
                max_val = widget.items[-1]

                widget_min_text = ui.IntTextField(
                    v_model=min_val,
                    name="min",
                    label="min",
                )
                widget_max_text = ui.IntTextField(
                    v_model=max_val,
                    name="max",
                    label="max",
                )
            else:
                continue

            self.ranges_widgets[widget_name] = {
                "min": widget_min_text,
                "max": widget_max_text,
            }

        if isinstance(
            self.active_qubit,
            (scq.Transmon, scq.TunableTransmon, scq.Fluxonium, scq.FluxQubit),
        ):
            widget_min_text = ui.FloatTextField(
                v_model=self.active_qubit._default_grid.min_val, label="min", step=0.01
            )
            widget_max_text = ui.FloatTextField(
                v_model=self.active_qubit._default_grid.max_val, label="max", step=0.01
            )
            self.ranges_widgets["phi"] = {
                "min": widget_min_text,
                "max": widget_max_text,
            }
        elif isinstance(self.active_qubit, scq.ZeroPi):
            widget_min_text = ui.FloatTextField(
                v_model=self.active_qubit._default_grid.min_val, label="min", step=0.01
            )
            widget_max_text = ui.FloatTextField(
                v_model=self.active_qubit._default_grid.max_val, label="max", step=0.01
            )
            self.ranges_widgets["theta"] = {
                "min": widget_min_text,
                "max": widget_max_text,
            }
        elif isinstance(self.active_qubit, scq.Cos2PhiQubit):
            default_grids = {
                "phi": self.active_qubit._default_phi_grid,
                "theta": self.active_qubit._default_theta_grid,
                "zeta": self.active_qubit._default_zeta_grid,
            }
            for param, param_grid in default_grids.items():
                widget_min_text = ui.FloatTextField(
                    v_model=param_grid.min_val, label="min", step=0.01
                )
                widget_max_text = ui.FloatTextField(
                    v_model=param_grid.max_val, label="max", step=0.01
                )
                self.ranges_widgets[param] = {
                    "min": widget_min_text,
                    "max": widget_max_text,
                }

    def initialize_tab_widget(self) -> None:
        """Creates each of the tabs in self.tab_widget"""

        qubit_plot_tab = v.Sheet(
            class_="d-flex d-row",
            style_="max_height: 600px",
            children=[self.qubit_plot_layout()],
        )
        param_ranges_tab = v.Sheet(children=[self.ranges_layout()])
        qubit_info_tab = v.Sheet(children=[self.qubit_info_layout()])
        common_qubit_params_tab = v.Sheet(children=[self.common_qubit_params_layout()])
        noise_param_tab = v.Sheet(children=[self.noise_params_layout()])

        self.tab_widget.children = [
            v.Tab(children=["Main"]),
            v.TabItem(key="Main", children=[qubit_plot_tab]),
            v.Tab(children=["Qubit info"]),
            v.TabItem(key="Qubit info", children=[qubit_info_tab]),
            v.Tab(children=["Literature params"]),
            v.TabItem(key="Literature params", children=[common_qubit_params_tab]),
            v.Tab(children=["Param ranges"]),
            v.TabItem(key="Param ranges", children=[param_ranges_tab]),
            v.Tab(children=["Noise params"]),
            v.TabItem(key="Noise params", children=[noise_param_tab]),
        ]
        self.tab_widget.v_model = self.tab_widget.v_model or "Main"
        self.tab_widget.align_with_title = True

    def initialize_display(self) -> None:
        """Creates the components of the GUI and displays all these components."""
        qubit_and_plot_choice_display = self.qubit_and_plot_ToggleButtons_layout()
        self.initialize_tab_widget()
        manual_update_widget, save_widget = self.manual_update_and_save_layout()

        display(
            v.Container(
                class_="d-flex ml-0 pl-0 mr-0 pr-0",
                children=[
                    v.Container(
                        class_="d-flex pl-0 pr-0",
                        children=[
                            v.Card(
                                color="#f9fbff",
                                class_="d-flex flex-column ml-0 pl-0 mr-2 pr-2",
                                style_="max-width: 240px",
                                children=[
                                    v.CardTitle(children=["scqubits.GUI"]),
                                    v.Divider(),
                                    v.CardActions(
                                        class_="d-flex flex-column",
                                        children=qubit_and_plot_choice_display
                                        + [manual_update_widget],
                                    ),
                                ],
                                elevation="1",
                            ),
                            v.Container(
                                class_="d-flex flex-column align-center ml-5 py-0 my-0",
                                children=[
                                    self.tab_widget,
                                    v.Divider(),
                                    self.plot_output,
                                    save_widget,
                                ],
                            ),
                        ],
                    )
                ],
            )
        )

    #
    def initialize_observe(self) -> None:
        """Links all the necessary widgets to their desired function."""
        self.qubit_and_plot_choice_widgets["qubit_dropdown"].on_event(
            "change", self.qubit_change
        )

        self.qubit_and_plot_choice_widgets["plot_buttons"].on_event(
            "change", self.plot_option_layout_refresh
        )

        self.manual_update_and_save_widgets["manual_update_checkbox"].on_event(
            "change", self.manual_update_checkbox
        )
        self.manual_update_and_save_widgets["update_button"].on_event(
            "click", self.manual_update_button_onclick
        )
        self.manual_update_and_save_widgets["save_button"].on_event(
            "click", self.save_button_clicked_action
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
            current_values_dict[param_name] = widget.v_model
        return current_values_dict

    #
    def get_plot_option_refresh(self) -> Callable[[Any], None]:
        """Obtains the current plot option

        Returns
        -------
            Method pertaining to refreshing the current plot option.
        """
        current_plot_option = self.qubit_and_plot_choice_widgets["plot_buttons"].v_model
        print("CPO ", current_plot_option)

        if current_plot_option == 0:
            return self.evals_vs_paramvals_plot_refresh
        if current_plot_option == 1:
            return self.wavefunctions_plot_refresh
        elif current_plot_option == 2:
            return self.matelem_vs_paramvals_plot_refresh
        elif current_plot_option == 3:
            return self.matrixelements_plot_refresh
        elif current_plot_option == 4:
            return self.coherence_vs_paramvals_plot_refresh

        raise Exception("Internal GUI exception:", current_plot_option)

    def update_params(self):
        """Uses the current parameter values to set the parameters of the
        active qubit.
        """
        current_values = self.get_current_values()

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            del current_values["grid"]
            grid_min, grid_max = self.qubit_params_widgets["grid"].v_model
            current_values["grid_min_val"] = grid_min
            current_values["grid_max_val"] = grid_max
            current_values["grid_pt_count"] = self.active_qubit.grid.pt_count

        self.active_qubit.set_params(**current_values)

    def check_ranges(
        self,
        new_min: Union[int, float],
        new_max: Union[int, float],
        widget_name: str,
        text_widget: dict,
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
            elif widget_name in ("phi", "theta", "zeta", "flux", "grid"):
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
            if changed_widget_key == "min":
                new_min = new_max - text_widget["max"].step
            else:
                new_max = new_min + text_widget["min"].step
        return new_min, new_max

    def update_range_values(
        self,
        new_min: Union[int, float],
        new_max: Union[int, float],
        widget_name: str,
        text_widget: dict,
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
        text_widget["min"].v_model = new_min
        text_widget["max"].v_model = new_max

        if widget_name in self.qubit_plot_options_widgets.keys():
            widget = self.qubit_plot_options_widgets[widget_name]
        elif widget_name in self.qubit_params_widgets.keys():
            widget = self.qubit_params_widgets[widget_name]
        else:
            widget = None

        if isinstance(widget, SelectMultiple):
            current_values = list(widget.v_model)
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
            text_widgets["min"].on_event("change", self.ranges_update)
            text_widgets["max"].on_event("change", self.ranges_update)

    def unobserve_ranges(self) -> None:
        for text_widgets in self.ranges_widgets.values():
            text_widgets["min"].unobserve(self.ranges_update)
            text_widgets["max"].unobserve(self.ranges_update)

    def observe_plot_refresh(self) -> None:
        if self.manual_update_bool:
            return

        total_dict = {
            **self.qubit_params_widgets,
            **self.qubit_plot_options_widgets,
            **self.noise_param_widgets,
        }
        for widget_name, widget in total_dict.items():
            if widget_name not in self.autoconnect_blacklist:
                widget.on_event("change", self.plot_refresh)

    def unobserve_plot_refresh(self) -> None:
        if self.manual_update_bool:
            return
        total_dict = {
            **self.qubit_params_widgets,
            **self.qubit_plot_options_widgets,
            **self.noise_param_widgets,
        }
        for widget_name, widget in total_dict.items():
            if widget_name not in self.autoconnect_blacklist:
                widget.unobserve(self.plot_refresh)

    def observe_plot_elements(self) -> None:
        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.qubit_plot_options_widgets["manual_scale_checkbox"].on_event(
                "change", self.manual_scale_tf
            )

        self.qubit_plot_options_widgets["scan_dropdown"].on_event(
            "change", self.scan_dropdown_refresh
        )

        self.qubit_plot_options_widgets["common_params_dropdown"].on_event(
            "change",
            function_sequence(
                self.common_params_dropdown_link_refresh,
                self.common_params_dropdown_params_refresh,
            ),
        )

        self.qubit_plot_options_widgets["common_params_dropdown"].on_event(
            "change", self.common_params_dropdown_params_refresh
        )

        self.qubit_plot_options_widgets["t1_checkbox"].on_event(
            "change", self.plot_change_bool_update
        )

        self.qubit_plot_options_widgets["t2_checkbox"].on_event(
            "change", self.plot_change_bool_update
        )

        self.qubit_plot_options_widgets["show3d_checkbox"].on_event(
            "change", self.plot_change_bool_update
        )

        self.qubit_plot_options_widgets["noise_channel_multi-select"].on_event(
            "change", self.plot_change_bool_update
        )

        for widget_name, widget in self.qubit_params_widgets.items():
            if "cut" in widget_name:
                widget.on_event("change", self.adjust_state_widgets)
            widget.on_event("change", self.common_params_dropdown_value_refresh)

    def unobserve_plot_elements(self) -> None:
        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.qubit_plot_options_widgets["manual_scale_checkbox"].unobserve(
                self.manual_scale_tf
            )
        self.qubit_plot_options_widgets["scan_dropdown"].unobserve(
            self.scan_dropdown_refresh
        )
        self.qubit_plot_options_widgets["common_params_dropdown"].unobserve(
            self.common_params_dropdown_link_refresh
        )
        self.qubit_plot_options_widgets["common_params_dropdown"].unobserve(
            self.common_params_dropdown_params_refresh
        )
        self.qubit_plot_options_widgets["t1_checkbox"].unobserve(
            self.plot_change_bool_update
        )
        self.qubit_plot_options_widgets["t2_checkbox"].unobserve(
            self.plot_change_bool_update
        )
        self.qubit_plot_options_widgets["show3d_checkbox"].unobserve(
            self.plot_change_bool_update
        )
        self.qubit_plot_options_widgets["noise_channel_multi-select"].unobserve(
            self.plot_change_bool_update
        )

        for widget_name, widget in self.qubit_params_widgets.items():
            if "cut" in widget_name:
                widget.unobserve(self.adjust_state_widgets)
            widget.unobserve(self.common_params_dropdown_value_refresh)

    def observe_coherence_elements(self) -> None:
        self.qubit_plot_options_widgets["i_text"].on_event(
            "change", self.coherence_text
        )
        self.qubit_plot_options_widgets["j_text"].on_event(
            "change", self.coherence_text
        )

        for widget in self.noise_param_widgets.values():
            widget.on_event("change", self.coherence_text)

    def unobserve_coherence_elements(self) -> None:
        self.qubit_plot_options_widgets["i_text"].unobserve(self.coherence_text)
        self.qubit_plot_options_widgets["j_text"].unobserve(self.coherence_text)

        for widget in self.noise_param_widgets.values():
            widget.unobserve(self.coherence_text)

    # Eventhandler Methods -------------------------------------------------------------
    def qubit_change(self, sender, event, value) -> None:
        self.plot_change_bool = True
        self.plot_output.clear_output()
        new_qubit = sender.v_model

        self.unobserve_ranges()
        self.unobserve_coherence_elements()
        self.unobserve_plot_elements()
        self.unobserve_plot_refresh()
        self.manual_update_and_save_widgets["manual_update_checkbox"].unobserve(
            self.manual_update_checkbox
        )

        if new_qubit in gui_defaults.slow_qubits:
            self.manual_update_and_save_widgets["manual_update_checkbox"].v_model = True
            self.manual_update_and_save_widgets["update_button"].disabled = False
            self.manual_update_bool = True
        else:
            self.manual_update_and_save_widgets[
                "manual_update_checkbox"
            ].v_model = False
            self.manual_update_and_save_widgets["update_button"].disabled = True
            self.manual_update_bool = False

        self.manual_update_and_save_widgets["manual_update_checkbox"].on_event(
            "change", self.manual_update_checkbox
        )

        self.set_qubit(new_qubit)
        self.initialize_tab_widget()
        self.observe_ranges()
        self.observe_coherence_elements()
        self.observe_plot_elements()
        self.observe_plot_refresh()
        self.plot_refresh()

    def scan_dropdown_refresh(self, sender, event, value) -> None:

        new = sender.v_model
        for param in self.qubit_plot_options_widgets["scan_dropdown"].items:
            self.qubit_params_widgets[param].disabled = False
        self.qubit_params_widgets[new].disabled = True
        self.plot_refresh()

    def plot_option_layout_refresh(self, sender, event, value) -> None:

        self.plot_change_bool = True

        self.plot_output.clear_output()
        self.unobserve_ranges()
        self.unobserve_plot_elements()
        self.unobserve_plot_refresh()

        refresher_func = self.get_plot_option_refresh()
        self.initialize_tab_widget()

        self.observe_ranges()
        self.observe_plot_elements()
        self.observe_plot_refresh()
        refresher_func()
        # self.plot_refresh()

    def manual_scale_tf(self, *args) -> None:
        sender = args[0]
        if sender.v_model:
            self.qubit_plot_options_widgets[
                "wavefunction_scale_slider"
            ].disabled = False
        else:
            self.qubit_plot_options_widgets["wavefunction_scale_slider"].disabled = True

    def coherence_text(self, *args) -> None:
        sender = args[0]
        self.unobserve_coherence_elements()
        self.unobserve_plot_refresh()
        isNoiseParamChange = None

        if sender in self.noise_param_widgets:
            isNoiseParamChange = True
            widget = sender
        else:
            isNoiseParamChange = False
            i_text_widget = self.qubit_plot_options_widgets["i_text"]
            j_text_widget = self.qubit_plot_options_widgets["j_text"]

        if sender.v_model <= 0:
            if isNoiseParamChange:
                if widget in self.noise_param_widgets:
                    widget.v_model = widget.step
            else:
                if i_text_widget.v_model <= 0:
                    i_text_widget.v_model = 0
                if j_text_widget.v_model <= 0:
                    j_text_widget.v_model = 0

        if not isNoiseParamChange:
            if i_text_widget.v_model == j_text_widget.v_model:
                i_text_widget.v_model = j_text_widget.v_model + j_text_widget.step
        self.observe_coherence_elements()
        self.observe_plot_refresh()

    def plot_change_bool_update(self, *args):
        self.plot_output.clear_output()
        self.plot_change_bool = True

    def manual_update_checkbox(self, *args) -> None:
        sender = args[0]
        if sender.v_model:
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
            self.plot_refresh()

    def manual_update_button_onclick(self, *args) -> None:
        self.update_params()
        if len(self.plot_output.outputs) == 0:
            self.plot_change_bool = True
        self.current_plot_option_refresh()

    def common_params_dropdown_value_refresh(self, *args) -> None:
        current_qubit = self.qubit_and_plot_choice_widgets["qubit_dropdown"].v_model
        current_dropdown_value = self.qubit_plot_options_widgets[
            "common_params_dropdown"
        ].v_model

        if current_qubit not in gui_defaults.paramvals_from_papers.keys():
            return
        if current_dropdown_value != "Manual":
            for param_name, param_val in gui_defaults.paramvals_from_papers[
                current_qubit
            ][current_dropdown_value]["params"].items():
                if self.qubit_params_widgets[param_name].v_model != param_val:
                    self.qubit_plot_options_widgets[
                        "common_params_dropdown"
                    ].v_model = "Manual"

    def common_params_dropdown_params_refresh(self, *args) -> None:
        current_qubit = self.qubit_and_plot_choice_widgets["qubit_dropdown"].v_model
        current_dropdown_value = self.qubit_plot_options_widgets[
            "common_params_dropdown"
        ].v_model

        if current_dropdown_value == "Manual":
            return
        self.unobserve_ranges()
        self.unobserve_plot_refresh()
        self.unobserve_plot_elements()

        params = gui_defaults.paramvals_from_papers[current_qubit][
            current_dropdown_value
        ]["params"]
        for param_name, param_val in params.items():
            param_max = self.ranges_widgets[param_name]["max"].v_model
            param_min = self.ranges_widgets[param_name]["min"].v_model

            if param_val < param_min:
                self.ranges_widgets[param_name]["min"].v_model = self.active_defaults[
                    param_name
                ]["min"]
                self.qubit_params_widgets[param_name].min = self.active_defaults[
                    param_name
                ]["min"]
            if param_val > param_max:
                self.ranges_widgets[param_name]["max"].v_model = (
                    np.ceil(param_val / 10) * 10
                )
                self.qubit_params_widgets[param_name].max = np.ceil(param_val / 10) * 10

            self.qubit_params_widgets[param_name].v_model = param_val
        self.observe_ranges()
        self.observe_plot_refresh()
        self.observe_plot_elements()
        self.plot_refresh()

    def adjust_state_widgets(self, *args) -> None:
        self.unobserve_ranges()
        self.unobserve_plot_refresh()
        self.update_params()
        hilbertdim = self.active_qubit.hilbertdim()
        wavefunction_state_slider_text = self.ranges_widgets["highest_state_slider"]

        if wavefunction_state_slider_text["max"].v_model >= hilbertdim - 1:
            new_min = wavefunction_state_slider_text["min"].v_model
            new_max = hilbertdim - 2
            new_min, new_max = self.check_ranges(
                new_min,
                new_max,
                "highest_state_slider",
                wavefunction_state_slider_text,
                "min",
            )
            self.update_range_values(
                new_min, new_max, "highest_state_slider", wavefunction_state_slider_text
            )

        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            multi_state_selector_text = self.ranges_widgets["multi_state_selector"]

            if multi_state_selector_text["max"].v_model >= hilbertdim - 2:
                new_min = multi_state_selector_text["min"].v_model
                new_max = hilbertdim - 3
                new_min, new_max = self.check_ranges(
                    new_min,
                    new_max,
                    "multi_state_selector",
                    multi_state_selector_text,
                    "min",
                )
                self.update_range_values(
                    new_min, new_max, "multi_state_selector", multi_state_selector_text
                )
        else:
            wavefunction_state_slider_text = self.ranges_widgets[
                "wavefunction_state_slider"
            ]

            if wavefunction_state_slider_text["max"].v_model >= hilbertdim - 2:
                new_min = wavefunction_state_slider_text["min"].v_model
                new_max = hilbertdim - 3
                new_min, new_max = self.check_ranges(
                    new_min,
                    new_max,
                    "wavefunction_state_slider",
                    wavefunction_state_slider_text,
                    "min",
                )
                self.update_range_values(
                    new_min,
                    new_max,
                    "wavefunction_state_slider",
                    wavefunction_state_slider_text,
                )
        self.observe_ranges()
        self.observe_plot_refresh()

    def ranges_update(self, *args) -> None:
        sender = args[0]

        self.unobserve_ranges()
        self.unobserve_plot_refresh()
        for widget_name, text_widgets in self.ranges_widgets.items():
            new_min = text_widgets["min"].v_model
            new_max = text_widgets["max"].v_model
            changed_widget_key = sender

            new_min, new_max = self.check_ranges(
                new_min, new_max, widget_name, text_widgets, changed_widget_key
            )

            self.update_range_values(new_min, new_max, widget_name, text_widgets)
        self.observe_ranges()
        self.observe_plot_refresh()
        self.adjust_state_widgets()
        self.plot_refresh()

    def save_button_clicked_action(self, *args) -> None:
        self.fig.savefig(self.manual_update_and_save_widgets["filename_text"].value)

    def plot_refresh(self, *args):
        self.update_params()

        if not self.manual_update_bool:
            print("UPDATE VIA ", self.current_plot_option_refresh)
            self.current_plot_option_refresh()

    def common_params_dropdown_link_refresh(self, *args) -> None:
        current_qubit = self.qubit_and_plot_choice_widgets["qubit_dropdown"].v_model
        current_dropdown_value = self.qubit_plot_options_widgets[
            "common_params_dropdown"
        ].v_model

        if current_dropdown_value == "User specified":
            self.qubit_plot_options_widgets["link_HTML"].children = [""]
        else:
            print("REACHED 1193")
            link = gui_defaults.paramvals_from_papers[current_qubit][
                current_dropdown_value
            ]["link"]
            self.qubit_plot_options_widgets["link_HTML"].children = [
                "<a href=" + link + " target='_blank'>" + link + "</a>"
            ]

    def evals_vs_paramvals_plot_refresh(self, *args) -> None:
        scan_dropdown_value = self.qubit_plot_options_widgets["scan_dropdown"].v_model
        scan_slider = self.qubit_params_widgets[scan_dropdown_value]

        value_dict = {
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.min, scan_slider.max),
            "subtract_ground_tf": self.qubit_plot_options_widgets[
                "subtract_ground_checkbox"
            ].v_model,
            "eigenvalue_state_value": self.qubit_plot_options_widgets[
                "highest_state_slider"
            ].v_model,
        }

        self.evals_vs_paramvals_plot(**value_dict)

    @matplotlib.rc_context(matplotlib_settings)
    def wavefunctions_plot_refresh(self, *args) -> None:
        value_dict = {
            "mode_value": self.qubit_plot_options_widgets["mode_dropdown"].v_model,
        }

        if isinstance(self.active_qubit, scq.FullZeroPi):
            return
        elif isinstance(
            self.active_qubit, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)
        ):
            value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.qubit_plot_options_widgets[
                "wavefunction_state_slider"
            ].v_model
        else:
            manual_scale_tf_value = self.qubit_plot_options_widgets[
                "manual_scale_checkbox"
            ].v_model

            if manual_scale_tf_value:
                value_dict["scale_value"] = self.qubit_plot_options_widgets[
                    "wavefunction_scale_slider"
                ].v_model
            else:
                value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.qubit_plot_options_widgets[
                "multi_state_selector"
            ].v_model

        if isinstance(
            self.active_qubit,
            (scq.Transmon, scq.TunableTransmon, scq.Fluxonium, scq.FluxQubit),
        ):
            value_dict["phi_grid"] = Grid1d(
                min_val=self.ranges_widgets["phi"]["min"].v_model,
                max_val=self.ranges_widgets["phi"]["max"].v_model,
                pt_count=self.active_qubit._default_grid.pt_count,
            )
        elif isinstance(self.active_qubit, scq.ZeroPi):
            value_dict["theta_grid"] = Grid1d(
                min_val=self.ranges_widgets["theta"]["min"].v_model,
                max_val=self.ranges_widgets["theta"]["max"].v_model,
                pt_count=self.active_qubit._default_grid.pt_count,
            )
        elif isinstance(self.active_qubit, scq.Cos2PhiQubit):
            value_dict["phi_grid"] = Grid1d(
                min_val=self.ranges_widgets["phi"]["min"].v_model,
                max_val=self.ranges_widgets["phi"]["max"].v_model,
                pt_count=self.active_qubit._default_phi_grid.pt_count,
            )
            value_dict["theta_grid"] = Grid1d(
                min_val=self.ranges_widgets["theta"]["min"].v_model,
                max_val=self.ranges_widgets["theta"]["max"].v_model,
                pt_count=self.active_qubit._default_theta_grid.pt_count,
            )
        self.wavefunctions_plot(**value_dict)

    def matelem_vs_paramvals_plot_refresh(self, *args) -> None:
        scan_dropdown_value = self.qubit_plot_options_widgets["scan_dropdown"].v_model
        scan_slider = self.qubit_params_widgets[scan_dropdown_value]

        value_dict = {
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.min, scan_slider.max),
            "operator_value": self.qubit_plot_options_widgets[
                "operator_dropdown"
            ].v_model,
            "matrix_element_state_value": self.qubit_plot_options_widgets[
                "highest_state_slider"
            ].v_model,
            "mode_value": self.qubit_plot_options_widgets["mode_dropdown"].v_model,
        }

        self.matelem_vs_paramvals_plot(**value_dict)

    def matrixelements_plot_refresh(self, *args) -> None:
        value_dict = {
            "operator_value": self.qubit_plot_options_widgets[
                "operator_dropdown"
            ].v_model,
            "eigenvalue_state_value": self.qubit_plot_options_widgets[
                "highest_state_slider"
            ].v_model,
            "mode_value": self.qubit_plot_options_widgets["mode_dropdown"].v_model,
            "show_numbers_tf": self.qubit_plot_options_widgets[
                "show_numbers_checkbox"
            ].v_model,
            "show3d_tf": self.qubit_plot_options_widgets["show3d_checkbox"].v_model,
        }
        self.matrixelements_plot(**value_dict)

    def coherence_vs_paramvals_plot_refresh(self, *args) -> None:
        t1_effective_tf = self.qubit_plot_options_widgets["t1_checkbox"].v_model
        t2_effective_tf = self.qubit_plot_options_widgets["t2_checkbox"].v_model
        scan_dropdown_value = self.qubit_plot_options_widgets["scan_dropdown"].v_model
        scan_slider = self.qubit_params_widgets[scan_dropdown_value]
        noise_channel_options = list(
            self.qubit_plot_options_widgets["noise_channel_multi-select"].v_model
        )
        common_noise_options = {
            "i": self.qubit_plot_options_widgets["i_text"].v_model,
            "j": self.qubit_plot_options_widgets["j_text"].v_model,
            "T": self.noise_param_widgets["T"].v_model,
        }
        noise_params_dict = {}

        for noise_param in self.noise_param_widgets.keys():
            if noise_param != "T":
                param_val = self.noise_param_widgets[noise_param].v_model
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
            "common_noise_options": common_noise_options,
        }

        self.coherence_vs_paramvals_plot(**value_dict)

    # Layout Methods ------------------------------------------------------------------
    def qubit_and_plot_ToggleButtons_layout(self) -> list:
        return [
            self.qubit_and_plot_choice_widgets["qubit_dropdown"],
            self.qubit_and_plot_choice_widgets["plot_buttons"],
        ]

    def manual_update_and_save_layout(self) -> tuple:
        manual_update_HBox = v.Container(
            class_="d-flex flex-row flex-start align-center",
            children=[
                self.manual_update_and_save_widgets["manual_update_checkbox"],
                self.manual_update_and_save_widgets["update_button"],
            ],
        )

        save_HBox = v.Container(
            class_="d-flex flex-row",
            children=[
                self.manual_update_and_save_widgets["save_button"],
                self.manual_update_and_save_widgets["filename_text"],
            ],
        )

        return manual_update_HBox, save_HBox

    def ranges_layout(self) -> v.Container:
        ranges_grid_hbox = v.Container(class_="d-flex flex-row", children=[])

        for widget_name, text_widgets in self.ranges_widgets.items():
            if widget_name == "highest_state_slider":
                widget_name = "Highest State"
            elif widget_name == "multi_state_selector":
                widget_name = "States"
            elif widget_name == "wavefunction_state_slider":
                widget_name = "State no."
            elif widget_name == "wavefunction_scale_slider":
                widget_name = "Scale"
            elif widget_name == "wavefunction_domain_slider":
                widget_name = "Wavefunction"

            range_hbox = v.Container(class_="d-flex flex-row", children=[])
            widget_label = v.Label(children=[widget_name])
            range_hbox.children += (
                widget_label,
                v.Row(children=[text_widgets["min"], text_widgets["max"]]),
            )

            ranges_grid_hbox.children += (range_hbox,)

        return ranges_grid_hbox

    def qubit_plot_layout(self) -> v.Container:
        plot_option_list = self.plot_option_layout()
        qubit_params_grid = self.qubit_params_grid_layout()

        qubit_plot_layout = v.Container(
            class_="d-flex flex-row ml-0 mr-0",
            children=[
                v.Container(
                    style_="transform: scale(0.85)",
                    class_="d-flex align-start flex-column",
                    children=plot_option_list,
                ),
                v.Container(
                    style_="transform: scale(0.85); max-height: 400px",
                    class_="d-flex align-start flex-column flex-wrap flex-align-content-start overflow-auto",
                    children=qubit_params_grid,
                ),
            ],
        )

        return qubit_plot_layout

    def qubit_info_layout(self) -> v.Container:
        qubit_info_box = v.Container(
            class_="py-5",
            children=[self.qubit_plot_options_widgets["qubit_info_image_widget"]],
        )

        return qubit_info_box

    def common_qubit_params_layout(self) -> v.Row:
        dropdown_box = v.Container(
            class_="d-flex flex-row py-5",
            children=[self.qubit_plot_options_widgets["common_params_dropdown"]],
        )

        print(self.qubit_plot_options_widgets["link_HTML"])
        common_qubit_params_HBox = v.Container(
            class_="d-flex flex-column",
            children=[dropdown_box, self.qubit_plot_options_widgets["link_HTML"]],
        )
        return common_qubit_params_HBox

    def noise_params_layout(self) -> v.Container:
        noise_params_grid = v.Container(
            class_="d-flex flex-column flex-wrap",
            children=[],
            style_="max-width: 400px",
        )
        noise_params_grid.children = list(self.noise_param_widgets.values())
        return noise_params_grid

    def plot_option_layout(self) -> list:
        current_plot_option = self.qubit_and_plot_choice_widgets["plot_buttons"].v_model

        if current_plot_option == 0:
            return self.energy_scan_layout()
        elif current_plot_option == 1:
            return self.wavefunctions_layout()
        elif current_plot_option == 2:
            return self.matelem_scan_layout()
        elif current_plot_option == 3:
            return self.matelem_layout()
        elif current_plot_option == 4:
            return self.coherence_times_layout()

        raise Exception("Internal GUI error: ", current_plot_option)

    def qubit_params_grid_layout(self) -> list:
        return [
            v.Col(
                dense=True,
                children=[
                    self.qubit_params_widgets[name],
                    self.qubit_params_widgets_txt[name],
                ],
                class_="d-flex align-baseline pb-0 pt-0 pl-0 pr-0",
            )
            for name in self.qubit_params_widgets.keys()
        ]

    def energy_scan_layout(self) -> tuple:
        """Creates the children for energy scan layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].v_model
        ].disabled = True

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["scan_dropdown"],
            v.Container(
                class_="d-flex flex-column",
                children=[
                    self.qubit_plot_options_widgets["highest_state_slider"],
                    self.qubit_plot_options_widgets["subtract_ground_checkbox"],
                ],
            ),
        )
        return plot_options_widgets_tuple

    def matelem_scan_layout(self) -> tuple:
        """Creates the children for matrix elements scan layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_plot_options_widgets["mode_dropdown"].v_model = self.active_defaults[
            "mode_matrixelem"
        ]
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].v_model
        ].disabled = True

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["highest_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
        )

        return plot_options_widgets_tuple

    def wavefunctions_layout(self) -> Any:
        """Creates the children for the wavefunctions layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        if isinstance(self.active_qubit, scq.FullZeroPi):
            plot_options_widgets_tuple = (v.Text(children=["Not implemented"]),)
        else:
            self.qubit_plot_options_widgets[
                "mode_dropdown"
            ].v_model = self.active_defaults["mode_wavefunc"]
            self.qubit_params_widgets[
                self.qubit_plot_options_widgets["scan_dropdown"].v_model
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
    ) -> tuple:
        """Creates the children for matrix elements layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_plot_options_widgets["mode_dropdown"].v_model = self.active_defaults[
            "mode_matrixelem"
        ]
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].v_model
        ].disabled = False

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["highest_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
            self.qubit_plot_options_widgets["show_numbers_checkbox"],
            self.qubit_plot_options_widgets["show3d_checkbox"],
        )

        return plot_options_widgets_tuple

    def coherence_times_layout(
        self,
    ) -> tuple:
        """Creates the children for matrix elements layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].v_model
        ].disabled = True
        self.qubit_plot_options_widgets["i_text"].style_ = "max-width: 40px"
        self.qubit_plot_options_widgets["j_text"].style_ = "max-width: 40px"
        # self.qubit_plot_options_widgets["t2_checkbox"].style_ = "width: 45%"
        # self.qubit_plot_options_widgets["t1_checkbox"].style_ = "width: 45%"
        text_HBox = v.Container(
            class_="d-flex flex-row justify-space-between align-center",
            children=[
                v.Text(children=["Transitions from"]),
                self.qubit_plot_options_widgets["i_text"],
                v.Text(children=["to"], style_="width: 50px"),
                self.qubit_plot_options_widgets["j_text"],
            ],
        )
        checkbox_HBox = v.Container(
            children=[
                self.qubit_plot_options_widgets["t1_checkbox"],
                self.qubit_plot_options_widgets["t2_checkbox"],
            ],
            class_="d-flex flex-column",
        )

        plot_options_widgets_tuple = (
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["noise_channel_multi-select"],
            v.Container(
                class_="flex-column",
                children=[text_HBox, checkbox_HBox],
                style_="width: 95%",
            ),
        )

        return plot_options_widgets_tuple

    # Plot functions------------------------------------------------------------------

    @matplotlib.rc_context(matplotlib_settings)
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
        self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes[0]

    @matplotlib.rc_context(matplotlib_settings)
    def wavefunctions_plot(
        self,
        eigenvalue_states: Union[List[int], int],
        mode_value: str,
        scale_value: Optional[float] = None,
        phi_grid: Optional[Grid1d] = None,
        theta_grid: Optional[Grid1d] = None,
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

        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
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
        else:
            if isinstance(self.active_qubit, scq.FluxQubit):
                grid_dict = {"phi_grid": phi_grid}
            elif isinstance(self.active_qubit, scq.ZeroPi):
                grid_dict = {"theta_grid": theta_grid}
            elif isinstance(self.active_qubit, scq.Cos2PhiQubit):
                grid_dict = {"phi_grid": phi_grid, "theta_grid": theta_grid}

            if self.plot_change_bool:
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states, mode=mode_value, **grid_dict
                )
                self.plot_change_bool = False

                if _HAS_WIDGET_BACKEND:
                    self.fig.canvas.header_visible = False
                    with self.plot_output:
                        plt.show()
            else:
                self.fig.delaxes(self.fig.axes[1])
                self.fig.axes[0].clear()
                self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states,
                    mode=mode_value,
                    **grid_dict,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )

        self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
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

        self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
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
                with self.plot_output:
                    plt.show()
        else:
            if show3d_tf:
                self.fig.delaxes(self.fig.axes[2])
                self.fig.delaxes(self.fig.axes[1])
                self.fig.add_subplot(122)
                axes = (self.fig.axes[0], self.fig.axes[1])
                axes[0].clear()
            else:
                self.fig.delaxes(self.fig.axes[1])
                self.fig.delaxes(self.fig.axes[0])
                self.fig.add_subplot(111)
                axes = self.fig.axes[0]
            self.active_qubit.plot_matrixelements(
                operator_value,
                evals_count=eigenvalue_state_value,
                mode=mode_value,
                show_numbers=show_numbers_tf,
                show3d=show3d_tf,
                fig_ax=(self.fig, axes),
            )

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

        self.plot_output.outputs = tuple(
            elem
            for elem in self.plot_output.outputs
            # if "Label" not in elem["data"]["text/plain"]
        )
        if len(noise_channels["coherence_times"]) == 0:
            if _HAS_WIDGET_BACKEND:
                self.fig.axes[0].clear()
            error_label = Label(value="Please select at least one noise channel.")
            with self.plot_output:
                display(error_label)
            return

        if self.plot_change_bool:
            if not t1_effective_tf and not t2_effective_tf:
                self.fig, ax = self.active_qubit.plot_coherence_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["coherence_times"],
                    common_noise_options=common_noise_options,
                )
            elif t1_effective_tf and not t2_effective_tf:
                self.fig, ax = self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    common_noise_options=common_noise_options,
                )
            elif not t1_effective_tf and t2_effective_tf:
                self.fig, ax = self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    common_noise_options=common_noise_options,
                )
            else:
                self.fig, ax = plt.subplots(nrows=1, ncols=2)
                self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, ax[0]),
                )
                self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, ax[1]),
                )
            self.plot_change_bool = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                with self.plot_output:
                    plt.show()
        else:
            for ax in self.fig.axes:
                ax.clear()
            if not t1_effective_tf and not t2_effective_tf:
                if len(noise_channels["coherence_times"]) > 1:
                    axes = np.array(self.fig.axes).reshape(
                        math.ceil(len(self.fig.axes) / 2), 2
                    )
                else:
                    axes = self.fig.axes[0]
                self.active_qubit.plot_coherence_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["coherence_times"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, axes),
                )
            elif t1_effective_tf and not t2_effective_tf:
                self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
            elif not t1_effective_tf and t2_effective_tf:
                self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
            else:
                self.active_qubit.plot_t1_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t1_eff"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[0]),
                )
                self.active_qubit.plot_t2_effective_vs_paramvals(
                    param_name=scan_value,
                    param_vals=np_list,
                    noise_channels=noise_channels["t2_eff"],
                    common_noise_options=common_noise_options,
                    fig_ax=(self.fig, self.fig.axes[1]),
                )
        self.fig.set_figwidth(gui_defaults.FIG_WIDTH_INCHES)
        if not _HAS_WIDGET_BACKEND:
            plt.close("all")
            with self.plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes


def function_sequence(func1, func2, *args):
    all_funcs = [func1, func2, *args]

    def func_sequentially():
        func1()
        func2()
        for func in args:
            func()

    return func_sequentially
