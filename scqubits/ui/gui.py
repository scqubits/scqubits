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

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Axes, Figure

import scqubits as scq
import scqubits.ui.gui_custom_widgets as ui
import scqubits.ui.gui_defaults as gui_defaults
import scqubits.ui.gui_navbar as gui_navbar
import scqubits.utils.misc as utils

from scqubits.core.discretization import Grid1d
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_custom_widgets import flex_column, flex_row
from scqubits.ui.gui_defaults import NAV_COLOR
from scqubits.ui.gui_setup import (
    init_dict_v_noise_params,
    init_dict_v_plot_options,
    init_filename_textfield,
    init_qubit_params_widgets_dict,
    init_ranges_widgets_dict,
    init_save_btn,
)
from scqubits.utils.misc import _HAS_WIDGET_BACKEND

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


QUBITS_WITH_GRID_INIT = (scq.ZeroPi, scq.FullZeroPi)  #  scq.Bifluxon
QUBITS_WITH_PHI_GRID = (
    scq.Transmon,
    scq.TunableTransmon,
    scq.Fluxonium,
    scq.FluxQubit,
    scq.Cos2PhiQubit,
)
QUBITS_WITH_THETA_GRID = (scq.ZeroPi, scq.Cos2PhiQubit)  # scq.Bifluxon,
QUBITS_WITHOUT_WAVEFUNCTION_PLOT = (scq.FullZeroPi,)  # scq.Snailmon
QUBITS_WITH_2D_WAVEFUNCTION_PLOT = (
    scq.FluxQubit,
    scq.ZeroPi,
    scq.Cos2PhiQubit,
    # scq.Bifluxon,
)


class GUI:
    """Generates the GUI for scqubits, handling single-qubit properties"""

    fig_ax: Optional[
        Tuple[Figure, Axes]
    ] = None  # Handle to the most recently generated Figure, Axes tuple

    no_plot_refresh_widgets = [
        "info_panel",
        "literature_params",
        "link_HTML",
    ]

    @utils.Required(ipyvuetify=_HAS_IPYVUETIFY, IPython=_HAS_IPYTHON)
    def __init__(self):
        utils.check_matplotlib_compatibility()

        # scq.settings.PROGRESSBAR_DISABLED = False
        scq.settings.T1_DEFAULT_WARNING = False

        self.plot_renewal_requested = True

        self.active_defaults: Dict[str, Any] = {}
        self.qubit_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_scan_params: Dict[str, Union[int, float, None]] = {}

        # Display Elements
        self.fig: Optional[Figure] = None
        self.v_plot_output = ipywidgets.Output(
            layout=ipywidgets.Layout(overflow="hidden")
        )
        self.v_tabs: v.Tabs = v.Tabs(children=[], background_color=NAV_COLOR, height=35)
        self.v_main_tab_container = v.Container()

        self.navbar, self.navbar_elements = gui_navbar.create_navbar()

        self.v_save_btn = init_save_btn()
        self.v_save_filename = init_filename_textfield()

        self.dict_v_ranges: Dict[str, Dict[str, ui.ValidatedNumberField]] = {}
        self.dict_v_noise_params: Dict[str, ui.ValidatedNumberField] = {}
        self.dict_v_plot_options: Dict[str, Any] = {}
        self.dict_v_qubit_params: Dict[str, Any] = {}

        # ------------------------------------------------------------------------------

        starting_qubit = self.navbar_elements["CHOOSE_QUBIT"].v_model
        self.set_qubit_and_init_qubit_widgets(starting_qubit)

        self.init_display()
        self.init_observe()
        self.refresh_current_plot(change=None)

    # Initialization Methods -----------------------------------------------------------
    def __repr__(self):
        return ""

    def set_qubit_and_init_qubit_widgets(self, qubit_name: str) -> None:
        """Sets up the chosen qubit to be the active qubit
        and updates the active defaults and widget dictionaries
        accordingly.
        """
        if qubit_name in gui_defaults.slow_qubits:
            scq.settings.PROGRESSBAR_DISABLED = False
        else:
            scq.settings.PROGRESSBAR_DISABLED = True

        self.active_defaults = gui_defaults.qubit_defaults[qubit_name]

        QubitClass = getattr(scq, qubit_name)
        init_params = QubitClass.default_params()

        if QubitClass in QUBITS_WITH_GRID_INIT:
            init_params["grid"] = Grid1d(-7 * np.pi, 7 * np.pi, 200)
        self.active_qubit = QubitClass(**init_params)

        self.active_qubit.truncated_dim = self.active_qubit.hilbertdim()
        self.set_qubit_params()

        self.dict_v_noise_params = init_dict_v_noise_params(self.active_qubit)
        self.dict_v_qubit_params = init_qubit_params_widgets_dict(
            qubit=self.active_qubit,
            qubit_params=self.qubit_params,
            defaults=self.active_defaults,
        )
        self.dict_v_plot_options = init_dict_v_plot_options(
            self.active_qubit,
            self.active_defaults,
            scan_params=list(self.qubit_scan_params.keys()),
        )
        self.dict_v_ranges = init_ranges_widgets_dict(
            qubit=self.active_qubit,
            dict_v_qubit_params=self.dict_v_qubit_params,
            dict_v_plot_options=self.dict_v_plot_options,
        )

    def set_qubit_params(self) -> None:
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

    def init_tab_widget(self) -> None:
        """Creates each of the tabs in self.tab_widget"""

        self.v_main_tab_container = v.Container(
            class_="d-flex d-row px-2 mx-1",
            dense=True,
            children=[self.main_tab_widgets()],
        )
        param_ranges_tab = v.Sheet(
            class_="d-flex flex-column flex-wrap overflow-auto",
            dense=True,
            style_="max-height: 500px;",
            children=self.param_ranges_tab_widgets(),
        )
        qubit_info_sheet = v.Sheet(children=[self.qubit_info_tab()])
        literature_tab = v.Sheet(
            class_="d-flex flex-column", children=self.literature_params_tab_widgets()
        )
        noise_param_tab = v.Sheet(
            class_="d-flex flex-column flex-wrap overflow-auto",
            children=[self.noise_params_tab_widgets()],
        )

        self.v_tabs.children = [
            v.Tab(children=["Main"]),
            v.TabItem(key="Main", children=[self.v_main_tab_container]),
            v.Tab(children=["Qubit info"]),
            v.TabItem(key="Qubit info", children=[qubit_info_sheet]),
            v.Tab(children=["Literature params"]),
            v.TabItem(key="Literature params", children=[literature_tab]),
            v.Tab(children=["Param ranges"]),
            v.TabItem(key="Param ranges", children=[param_ranges_tab]),
            v.Tab(children=["Noise params"]),
            v.TabItem(key="Noise params", children=[noise_param_tab]),
        ]
        self.v_tabs.v_model = self.v_tabs.v_model or "Main"
        self.v_tabs.align_with_title = True

    def init_display(self) -> None:
        """Creates the components of the GUI and displays all these components."""

        self.init_tab_widget()

        save_widget = flex_row([self.v_save_btn, self.v_save_filename], class_="p-0")
        update_widget = flex_column(
            [self.navbar_elements["AUTO_UPDATING"], self.navbar_elements["DO_UPDATE"]],
            class_="px-0",
            style_="transform: scale(0.9); width: 150px; margin-left: 50px; margin-right: 0px;",
        )

        display(
            v.Container(
                id="outermost_cntnr",
                class_="ml-0 pl-0 mr-0 pr-0",
                children=[
                    self.navbar_elements["HEADER"],
                    v.Container(
                        id="below_header_cntnr",
                        class_="d-flex flex-row pl-0 pr-0 mt-0 pt-0 ml-0",
                        children=[
                            self.navbar,
                            v.Container(
                                id="below_tabs_header_cntnr",
                                class_="d-flex flex-column align-center ml-0 px-0 mx-0 my-0 py-0",
                                children=[
                                    self.v_tabs,
                                    v.Card(
                                        id="update_and_display_card",
                                        class_="d-flex flex-row p-0",
                                        style_="margin-left: 24px; margin-top: 5px",
                                        width="100%",
                                        children=[
                                            update_widget,
                                            flex_column(
                                                [self.v_plot_output],
                                                class_="p-0 m-0",
                                                id="matplotlib_cntnr",
                                            ),
                                        ],
                                    ),
                                    save_widget,
                                ],
                            ),
                        ],
                    ),
                ],
            )
        )

    def init_observe(self) -> None:
        """Links all the necessary widgets to their desired function."""
        self.navbar_elements["CHOOSE_QUBIT"].observe(
            self.change_active_qubit, names="v_model"
        )
        self.navbar_elements["CHOOSE_PLOT"].observe(
            self.change_plot_type, names="v_model"
        )
        self.navbar_elements["AUTO_UPDATING"].observe(
            self.toggle_auto_updating, names="v_model"
        )
        self.navbar_elements["DO_UPDATE"].on_event(
            "click", self.manual_update_button_onclick
        )
        self.v_save_btn.on_event("click", self.save_button_clicked_action)

        self.observe_all()

    def observe_all(self):
        """Switch on monitoring of all widgets in the GUI."""
        self.observe_coherence_widgets()
        self.observe_range_widgets()
        self.observe_plot_option_widgets()
        self.activate_auto_plot_refresh()

    def unobserve_all(self):
        """Switch off monitoring of all widgets in the GUI and stop automatic refreshing of the plot."""
        self.deactivate_auto_plot_refresh()
        self.unobserve_coherence_widgets()
        self.unobserve_range_widgets()
        self.unobserve_plot_options_widget()

    # Retrieval Methods------------------------------------------------------------------
    def get_current_values(self) -> Dict[str, Union[int, float]]:
        """Obtains the current values from each of the qubit parameter
        sliders.

        Returns
        -------
            Dictionary of the current value for each of the qubit's parameters
        """
        current_values_dict = {}
        for param_name, widget in self.dict_v_qubit_params.items():
            if not hasattr(widget, "valid_entry") or widget.is_entry_valid():
                current_values_dict[param_name] = (
                    widget.num_value if hasattr(widget, "num_value") else widget.v_model
                )
        return current_values_dict

    def refresh_current_plot(self, change=None) -> None:
        """Obtains the current plot option

        Returns
        -------
            Method pertaining to refreshing the current plot option.
        """
        current_plot_option = self.navbar_elements["CHOOSE_PLOT"].v_model

        if current_plot_option == 0:
            self.evals_vs_paramvals_plot_refresh()
            return
        if current_plot_option == 1:
            self.wavefunctions_plot_refresh()
            return
        if current_plot_option == 2:
            self.matrixelements_plot_refresh()
            return
        if current_plot_option == 3:
            self.matelem_vs_paramvals_plot_refresh()
            return
        if current_plot_option == 4:
            self.coherence_vs_paramvals_plot_refresh()
            return

        raise Exception("Internal GUI exception:", current_plot_option)

    def update_params(self):
        """Uses the current parameter values to set the parameters of the
        active qubit.
        """
        current_values = self.get_current_values()

        if isinstance(self.active_qubit, QUBITS_WITH_GRID_INIT):
            del current_values["grid"]
            grid_min, grid_max = self.dict_v_qubit_params["grid"].v_model
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
        if new_max < new_min:
            if changed_widget_key == "min":
                new_min = new_max
            else:
                new_max = new_min
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

        if widget_name in self.dict_v_plot_options.keys():
            widget = self.dict_v_plot_options[widget_name]
        elif widget_name in self.dict_v_qubit_params.keys():
            widget = self.dict_v_qubit_params[widget_name]
        else:
            widget = None

        if isinstance(widget, v.Select) and widget.multiple:
            current_values = list(widget.v_model)
            new_values = []
            widget.items = list(range(int(new_min), int(new_max + 1)))
            for value in current_values:
                if value in widget.items:
                    new_values.append(value)
            widget.v_model = new_values
        elif widget is None:
            pass
        elif hasattr(widget, "v_min"):
            widget.v_min = new_min
            widget.v_max = new_max
        else:
            widget.min = new_min
            widget.max = new_max

    # Observe Methods-------------------------------------------------------------------
    def observe_range_widgets(self) -> None:
        for text_widgets in self.dict_v_ranges.values():
            text_widgets["min"].observe(self.ranges_update, names="num_value")
            text_widgets["max"].observe(self.ranges_update, names="num_value")

    def unobserve_range_widgets(self) -> None:
        for text_widgets in self.dict_v_ranges.values():
            text_widgets["min"].unobserve(self.ranges_update, names="num_value")
            text_widgets["max"].unobserve(self.ranges_update, names="num_value")

    def activate_auto_plot_refresh(self) -> None:
        total_dict = {
            **self.dict_v_qubit_params,
            **self.dict_v_plot_options,
            **self.dict_v_noise_params,
        }
        for widget_name, widget in total_dict.items():
            if widget_name not in self.no_plot_refresh_widgets:
                widget.observe(self.plot_refresh, names="v_model")

    def deactivate_auto_plot_refresh(self) -> None:
        total_dict = {
            **self.dict_v_qubit_params,
            **self.dict_v_plot_options,
            **self.dict_v_noise_params,
        }
        for widget_name, widget in total_dict.items():
            if widget_name not in self.no_plot_refresh_widgets:
                try:
                    widget.unobserve(self.plot_refresh, names="v_model")
                except ValueError:
                    pass

    def observe_plot_option_widgets(self) -> None:
        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.dict_v_plot_options["manual_wf_scaling"].observe(
                self.toggle_manual_wf_scaling, names="v_model"
            )

        self.dict_v_plot_options["scan_param"].observe(
            self.set_new_scan_param, names="v_model"
        )
        self.dict_v_plot_options["literature_params"].observe(
            self.set_literature_params_and_refresh_plot, names="v_model"
        )

        self.dict_v_plot_options["literature_params"].observe(
            self.literature_url_refresh, names="v_model"
        )
        self.dict_v_plot_options["t1_checkbox"].observe(
            self.clear_plot, names="v_model"
        )

        self.dict_v_plot_options["t2_checkbox"].observe(
            self.clear_plot, names="v_model"
        )
        self.dict_v_plot_options["show3d_matelem"].observe(
            self.clear_plot, names="v_model"
        )
        self.dict_v_plot_options["noise_channel_multiselect"].observe(
            self.clear_plot, names="v_model"
        )

        for widget_name, widget in self.dict_v_qubit_params.items():
            if "cut" in widget_name:
                widget.observe(self.adjust_state_widgets, names="v_model")
            widget.observe(self.check_user_override_literature_params, names="v_model")

    def unobserve_plot_options_widget(self) -> None:
        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.dict_v_plot_options["manual_wf_scaling"].unobserve(
                self.toggle_manual_wf_scaling, names="v_model"
            )

        self.dict_v_plot_options["scan_param"].unobserve(
            self.set_new_scan_param, names="v_model"
        )
        self.dict_v_plot_options["literature_params"].unobserve(
            self.literature_url_refresh, names="v_model"
        )
        self.dict_v_plot_options["literature_params"].unobserve(
            self.set_literature_params_and_refresh_plot, names="v_model"
        )
        self.dict_v_plot_options["t1_checkbox"].unobserve(
            self.clear_plot, names="v_model"
        )
        self.dict_v_plot_options["t2_checkbox"].unobserve(
            self.clear_plot, names="v_model"
        )
        self.dict_v_plot_options["show3d_matelem"].unobserve(
            self.clear_plot, names="v_model"
        )
        self.dict_v_plot_options["noise_channel_multiselect"].unobserve(
            self.clear_plot, names="v_model"
        )

        for widget_name, widget in self.dict_v_qubit_params.items():
            if "cut" in widget_name:
                widget.unobserve(self.adjust_state_widgets, names="v_model")
            widget.unobserve(
                self.check_user_override_literature_params, names="v_model"
            )

    def observe_coherence_widgets(self) -> None:
        self.dict_v_plot_options["i_text"].observe(
            self.check_coherence_params_bounds, names="num_value"
        )
        self.dict_v_plot_options["j_text"].observe(
            self.check_coherence_params_bounds, names="num_value"
        )

        for widget in self.dict_v_noise_params.values():
            widget.observe(self.check_coherence_params_bounds, names="num_value")

    def unobserve_coherence_widgets(self) -> None:
        self.dict_v_plot_options["i_text"].unobserve(
            self.check_coherence_params_bounds, names="num_value"
        )
        self.dict_v_plot_options["j_text"].unobserve(
            self.check_coherence_params_bounds, names="num_value"
        )

        for widget in self.dict_v_noise_params.values():
            widget.unobserve(self.check_coherence_params_bounds, names="num_value")

    # Eventhandler Methods -------------------------------------------------------------
    def change_active_qubit(self, change) -> None:
        self.clear_plot()
        self.unobserve_all()

        new_qubit = change["new"]
        if new_qubit in gui_defaults.slow_qubits:
            self.navbar_elements["AUTO_UPDATING"].v_model = False
            self.auto_updating_off()
        else:
            self.navbar_elements["AUTO_UPDATING"].v_model = True
            self.auto_updating_on()

        self.set_qubit_and_init_qubit_widgets(new_qubit)

        self.init_tab_widget()
        self.change_plot_options()
        if self.auto_updating:
            self.refresh_current_plot()
        self.observe_all()

    def set_new_scan_param(self, change) -> None:
        new_scan_param = change["new"]
        for param in self.dict_v_plot_options["scan_param"].items:
            self.dict_v_qubit_params[param].disabled = False
        self.dict_v_qubit_params[new_scan_param].disabled = True
        self.plot_refresh(change=None)

    def change_plot_type(self, change) -> None:
        self.clear_plot()

        self.unobserve_all()
        self.change_plot_options()
        self.observe_all()

        if self.auto_updating:
            self.plot_refresh(change=None)

    def toggle_manual_wf_scaling(self, change) -> None:
        if change["new"]:
            self.dict_v_plot_options["wavefunction_scale_slider"].disabled = False
        else:
            self.dict_v_plot_options["wavefunction_scale_slider"].disabled = True

    def check_coherence_params_bounds(self, change) -> None:
        if change["owner"].error:
            return

        self.unobserve_coherence_widgets()
        self.deactivate_auto_plot_refresh()

        widget_key = change["owner"].label

        if widget_key in self.dict_v_noise_params:
            widget = change["owner"]
            if int(change["new"]) <= 0:
                widget.v_model = widget.step
        else:
            i_text_widget = self.dict_v_plot_options["i_text"]
            j_text_widget = self.dict_v_plot_options["j_text"]

            if i_text_widget.num_value <= 0:
                i_text_widget.v_model = 0
            if j_text_widget.num_value <= 0:
                j_text_widget.v_model = 0
            if i_text_widget.v_model == j_text_widget.v_model:
                i_text_widget.v_model = j_text_widget.num_value + 1

        self.observe_coherence_widgets()
        self.activate_auto_plot_refresh()

    def clear_plot(self, *args):
        self.v_plot_output.clear_output()
        self.plot_renewal_requested = True

    def auto_updating_off(self):
        self.navbar_elements["DO_UPDATE"].disabled = False
        self.deactivate_auto_plot_refresh()

    def auto_updating_on(self):
        self.navbar_elements["DO_UPDATE"].disabled = True
        self.activate_auto_plot_refresh()

    @property
    def auto_updating(self):
        return self.navbar_elements["AUTO_UPDATING"].v_model

    def toggle_auto_updating(self, change) -> None:
        if change["new"]:
            self.auto_updating_on()
        else:
            self.auto_updating_off()

    def manual_update_button_onclick(self, widget, event, data) -> None:
        self.update_params()
        if len(self.v_plot_output.outputs) == 0:
            self.plot_renewal_requested = True
        self.refresh_current_plot(change=None)

    def check_user_override_literature_params(self, change) -> None:
        current_qubit = self.navbar_elements["CHOOSE_QUBIT"].v_model
        current_dropdown_value = self.dict_v_plot_options["literature_params"].v_model

        if current_qubit not in gui_defaults.paramvals_from_papers.keys():
            return
        if current_dropdown_value != "User specified":
            for param_name, param_val in gui_defaults.paramvals_from_papers[
                current_qubit
            ][current_dropdown_value]["params"].items():
                if self.dict_v_qubit_params[param_name].num_value != param_val:
                    self.dict_v_plot_options[
                        "literature_params"
                    ].v_model = "User specified"
                    return

    def set_literature_params_and_refresh_plot(self, change) -> None:
        current_qubit = self.navbar_elements["CHOOSE_QUBIT"].v_model
        current_dropdown_value = self.dict_v_plot_options["literature_params"].v_model

        if current_dropdown_value == "User specified":
            return

        self.unobserve_all()
        self.literature_url_refresh(None)

        params = gui_defaults.paramvals_from_papers[current_qubit][
            current_dropdown_value
        ]["params"]
        for param_name, param_val in params.items():
            param_max = self.dict_v_ranges[param_name]["max"].v_model
            param_min = self.dict_v_ranges[param_name]["min"].v_model

            if param_val < param_min:
                self.dict_v_ranges[param_name]["min"].v_model = self.active_defaults[
                    param_name
                ]["min"]
                self.dict_v_qubit_params[param_name].v_min = self.active_defaults[
                    param_name
                ]["min"]
            if param_val > param_max:
                self.dict_v_ranges[param_name]["max"].v_model = (
                    np.ceil(param_val / 10) * 10
                )
                self.dict_v_qubit_params[param_name].v_max = (
                    np.ceil(param_val / 10) * 10
                )

            self.dict_v_qubit_params[param_name].v_model = param_val

        self.observe_all()
        self.plot_refresh(change=None)

    def adjust_state_widgets(self, *args) -> None:
        self.unobserve_all()
        self.update_params()
        hilbertdim = self.active_qubit.hilbertdim()
        wavefunction_state_slider_text = self.dict_v_ranges["highest_state"]

        if wavefunction_state_slider_text["max"].num_value >= hilbertdim - 1:
            new_max = max(hilbertdim - 2, 0)
            new_min = wavefunction_state_slider_text["min"].num_value
            new_min, new_max = self.check_ranges(
                new_min,
                new_max,
                "highest_state",
                wavefunction_state_slider_text,
                "min",
            )
            self.update_range_values(
                new_min, new_max, "highest_state", wavefunction_state_slider_text
            )

        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            multi_state_selector_text = self.dict_v_ranges["multi_state_selector"]
            if multi_state_selector_text["max"].num_value >= hilbertdim - 2:
                new_min = multi_state_selector_text["min"].num_value
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
            wavefunction_state_slider_text = self.dict_v_ranges[
                "wavefunction_state_slider"
            ]

            if wavefunction_state_slider_text["max"].num_value >= hilbertdim - 2:
                new_min = wavefunction_state_slider_text["min"].num_value
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
        self.observe_all()

    def ranges_update(self, change) -> None:
        self.unobserve_all()
        for widget_name, text_widgets in self.dict_v_ranges.items():
            new_min = text_widgets["min"].num_value
            new_max = text_widgets["max"].num_value
            changed_widget_key = change["owner"].label

            new_min, new_max = self.check_ranges(
                new_min, new_max, widget_name, text_widgets, changed_widget_key
            )

            self.update_range_values(new_min, new_max, widget_name, text_widgets)

        for widget in self.dict_v_qubit_params.values():
            widget.update_slider()

        self.observe_all()
        self.adjust_state_widgets()
        self.plot_refresh(change=None)

    def save_button_clicked_action(self, widget, event, data) -> None:
        self.fig.savefig(self.v_save_filename.v_model)

    def plot_refresh(self, change, *args):
        self.update_params()

        if self.auto_updating:
            self.refresh_current_plot(change=None)

    def literature_url_refresh(self, change) -> None:
        current_qubit = self.navbar_elements["CHOOSE_QUBIT"].v_model
        current_dropdown_value = self.dict_v_plot_options["literature_params"].v_model

        if current_dropdown_value == "User specified":
            self.dict_v_plot_options["link_HTML"].children = [""]
        else:
            link = gui_defaults.paramvals_from_papers[current_qubit][
                current_dropdown_value
            ]["link"]
            self.dict_v_plot_options["link_HTML"].tag = "a"
            self.dict_v_plot_options["link_HTML"].attributes = {
                "href": link,
                "target": "_blank",
            }
            self.dict_v_plot_options["link_HTML"].children = [link]

    def evals_vs_paramvals_plot_refresh(self) -> None:
        scan_dropdown_value = self.dict_v_plot_options["scan_param"].v_model
        scan_slider = self.dict_v_qubit_params[scan_dropdown_value]

        value_dict = {
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.v_min, scan_slider.v_max),
            "subtract_ground_tf": self.dict_v_plot_options["subtract_ground"].v_model,
            "eigenvalue_state_value": self.dict_v_plot_options["highest_state"].v_model,
        }

        self.evals_vs_paramvals_plot(**value_dict)

    @matplotlib.rc_context(matplotlib_settings)
    def wavefunctions_plot_refresh(self) -> None:
        value_dict = {
            "mode_value": gui_defaults.mode_dropdown_dict[
                self.dict_v_plot_options["amplitude_mode"].v_model
            ],
        }

        if isinstance(self.active_qubit, QUBITS_WITHOUT_WAVEFUNCTION_PLOT):
            return
        # Remainder of block only for qubits that support wavefunction plots
        elif isinstance(self.active_qubit, QUBITS_WITH_2D_WAVEFUNCTION_PLOT):
            value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.dict_v_plot_options[
                "wavefunction_state_slider"
            ].v_model
        else:  # qubit with 1d wavefunction
            manual_scale_tf_value = self.dict_v_plot_options[
                "manual_wf_scaling"
            ].v_model

            if manual_scale_tf_value:
                value_dict["scale_value"] = self.dict_v_plot_options[
                    "wavefunction_scale_slider"
                ].v_model
            else:
                value_dict["scale_value"] = None
            value_dict["eigenvalue_states"] = self.dict_v_plot_options[
                "multi_state_selector"
            ].v_model

        if isinstance(self.active_qubit, QUBITS_WITH_PHI_GRID):
            value_dict["phi_grid"] = Grid1d(
                min_val=self.dict_v_ranges["phi"]["min"].num_value,
                max_val=self.dict_v_ranges["phi"]["max"].num_value,
                pt_count=self.active_qubit._default_grid.pt_count,
            )
        if isinstance(self.active_qubit, QUBITS_WITH_THETA_GRID):
            value_dict["theta_grid"] = Grid1d(
                min_val=self.dict_v_ranges["theta"]["min"].num_value,
                max_val=self.dict_v_ranges["theta"]["max"].num_value,
                pt_count=self.active_qubit._default_grid.pt_count,
            )
        self.wavefunctions_plot(**value_dict)

    def matelem_vs_paramvals_plot_refresh(self) -> None:
        scan_dropdown_value = self.dict_v_plot_options["scan_param"].v_model
        scan_slider = self.dict_v_qubit_params[scan_dropdown_value]

        value_dict = {
            "scan_value": scan_dropdown_value,
            "scan_range": (scan_slider.v_min, scan_slider.v_max),
            "operator_value": self.dict_v_plot_options["operator_choice"].v_model,
            "matrix_element_state_value": self.dict_v_plot_options[
                "highest_state"
            ].v_model,
            "mode_value": gui_defaults.mode_dropdown_dict[
                self.dict_v_plot_options["amplitude_mode"].v_model
            ],
        }

        self.matelem_vs_paramvals_plot(**value_dict)

    def matrixelements_plot_refresh(self) -> None:
        value_dict = {
            "operator_value": self.dict_v_plot_options["operator_choice"].v_model,
            "eigenvalue_state_value": self.dict_v_plot_options["highest_state"].v_model,
            "mode_value": gui_defaults.mode_dropdown_dict[
                self.dict_v_plot_options["amplitude_mode"].v_model
            ],
            "show_numbers_tf": self.dict_v_plot_options["show_matelem_numbers"].v_model,
            "show3d_tf": self.dict_v_plot_options["show3d_matelem"].v_model,
        }
        self.matrixelements_plot(**value_dict)

    def coherence_vs_paramvals_plot_refresh(self) -> None:
        t1_effective_tf = self.dict_v_plot_options["t1_checkbox"].v_model
        t2_effective_tf = self.dict_v_plot_options["t2_checkbox"].v_model
        scan_dropdown_value = self.dict_v_plot_options["scan_param"].v_model
        scan_slider = self.dict_v_qubit_params[scan_dropdown_value]
        noise_channel_options = list(
            self.dict_v_plot_options["noise_channel_multiselect"].v_model
        )
        common_noise_options = {
            "i": self.dict_v_plot_options["i_text"].num_value,
            "j": self.dict_v_plot_options["j_text"].num_value,
            "T": self.dict_v_noise_params["T"].num_value,
        }
        noise_params_dict = {}

        for noise_param in self.dict_v_noise_params.keys():
            if noise_param != "T":
                param_val = self.dict_v_noise_params[noise_param].num_value
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
            "scan_range": (scan_slider.v_min, scan_slider.v_max),
            "noise_channels": noise_channels,
            "common_noise_options": common_noise_options,
        }

        self.coherence_vs_paramvals_plot(**value_dict)

    # Layout Methods ------------------------------------------------------------------

    def param_ranges_tab_widgets(self) -> list:
        ranges_widgets = []

        for widget_name, text_widgets in self.dict_v_ranges.items():
            if widget_name == "highest_state":
                widget_name = "Max level"
            elif widget_name == "multi_state_selector":
                widget_name = "States"
            elif widget_name == "wavefunction_state_slider":
                widget_name = "State no."
            elif widget_name == "wavefunction_scale_slider":
                widget_name = "Scale"
            elif widget_name == "wavefunction_domain_slider":
                widget_name = "Wavefunction"

            ranges_widgets += [
                v.Container(
                    class_="d-flex flex-row align-center",
                    style_="max-width: 290px; transform: scale(0.9)",
                    dense=True,
                    children=[
                        v.Text(class_="pr-3", children=[widget_name]),
                        text_widgets["min"],
                        text_widgets["max"],
                    ],
                )
            ]
        return ranges_widgets

    def main_tab_widgets(self) -> v.Container:
        self.plot_options_dict = self.plot_options_widgets()
        qubit_params_grid = self.qubit_params_widgets()

        main_tab = v.Container(
            id="main_tab_cntnr",
            class_="d-flex flex-row mx-0 px-0",
            width="100%",
            children=[
                v.Container(
                    id="plot_options_cntnr",
                    class_="d-flex align-start flex-column pb-0",
                    style_="width: 48%; transform: scale(0.9)",
                    children=self.plot_options_dict[0],
                ),
                v.Divider(vertical=True),
                v.Container(
                    id="qubit_params_cntnr",
                    style_="width: 100%; max-height: 350px; transform: scale(0.9)",
                    class_="d-flex align-start flex-column flex-wrap flex-align-content-start overflow-auto mx-1 px-2",
                    children=qubit_params_grid,
                ),
            ],
        )
        return main_tab

    def change_plot_options(self):
        current_plot_option = self.navbar_elements["CHOOSE_PLOT"].v_model
        self.v_main_tab_container.children[0].children[
            0
        ].children = self.plot_options_dict[current_plot_option]

        self.dict_v_plot_options["amplitude_mode"].v_model = (
            "Re(·)" if current_plot_option < 2 else "|·|"
        )
        if current_plot_option in gui_defaults.gui_sweep_plots:
            self.dict_v_qubit_params[
                self.dict_v_plot_options["scan_param"].v_model
            ].disabled = True
        else:
            self.dict_v_qubit_params[
                self.dict_v_plot_options["scan_param"].v_model
            ].disabled = False

    def qubit_info_tab(self) -> v.Container:
        qubit_info_box = v.Container(
            class_="py-5",
            style_="transform: scale(0.9)",
            children=[self.dict_v_plot_options["info_panel"]],
        )

        return qubit_info_box

    def literature_params_tab_widgets(self) -> list:
        return [
            self.dict_v_plot_options["literature_params"],
            self.dict_v_plot_options["link_HTML"],
        ]

    def noise_params_tab_widgets(self) -> v.Container:
        noise_params_grid = v.Container(
            class_="d-flex flex-column flex-wrap py-3",
            children=[],
            style_="max-height: 400px; transform: scale(0.9)",
        )
        noise_params_grid.children = list(self.dict_v_noise_params.values())
        return noise_params_grid

    def plot_options_widgets(self) -> Dict[int, tuple]:
        plot_options_for_active_qubit = {
            0: self.energy_spectrum_options_widgets(),
            1: self.wavefunctions_options_widgets(),
            2: self.matelem_options_widgets(),
            3: self.matelem_scan_options_widgets(),
            4: self.coherence_times_options_widgets(),
        }
        return plot_options_for_active_qubit

    def qubit_params_widgets(self) -> list:
        widget_list = []

        for param_widget in self.dict_v_qubit_params.values():
            if hasattr(param_widget, "widget"):
                widget_list.append(param_widget.widget())
            else:
                widget_list.append(param_widget)
        return widget_list

    def energy_spectrum_options_widgets(self) -> tuple:
        """Creates the children for energy scan layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.dict_v_qubit_params[
            self.dict_v_plot_options["scan_param"].v_model
        ].disabled = True

        plot_options_widgets_tuple = (
            self.dict_v_plot_options["scan_param"],
            flex_column(
                [
                    self.dict_v_plot_options["highest_state"],
                    self.dict_v_plot_options["subtract_ground"],
                ]
            ),
        )
        return plot_options_widgets_tuple

    def matelem_scan_options_widgets(self) -> tuple:
        """Creates the children for matrix elements scan layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.dict_v_plot_options["amplitude_mode"].v_model = self.active_defaults[
            "mode_matrixelem"
        ]
        self.dict_v_qubit_params[
            self.dict_v_plot_options["scan_param"].v_model
        ].disabled = True

        plot_options_widgets_tuple = (
            self.dict_v_plot_options["operator_choice"],
            self.dict_v_plot_options["scan_param"],
            self.dict_v_plot_options["highest_state"],
            self.dict_v_plot_options["amplitude_mode"],
        )

        return plot_options_widgets_tuple

    def wavefunctions_options_widgets(self) -> Any:
        """Creates the children for the wavefunctions layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        if isinstance(self.active_qubit, QUBITS_WITHOUT_WAVEFUNCTION_PLOT):
            plot_options_widgets_tuple = (
                v.Text(children=["Not implemented (configuration space >2d)"]),
            )
        else:
            self.dict_v_plot_options["amplitude_mode"].v_model = self.active_defaults[
                "mode_wavefunc"
            ]
            self.dict_v_qubit_params[
                self.dict_v_plot_options["scan_param"].v_model
            ].disabled = False

            if isinstance(self.active_qubit, QUBITS_WITH_2D_WAVEFUNCTION_PLOT):
                which_widget = self.dict_v_plot_options["wavefunction_state_slider"]
                plot_options_widgets_tuple = (
                    which_widget,
                    self.dict_v_plot_options["amplitude_mode"],
                )
            else:  # 1d wavefunction plot
                which_widget = self.dict_v_plot_options["multi_state_selector"]
                plot_options_widgets_tuple = (
                    which_widget,
                    self.dict_v_plot_options["amplitude_mode"],
                    self.dict_v_plot_options["manual_wf_scaling"],
                    self.dict_v_plot_options["wavefunction_scale_slider"],
                )

        return plot_options_widgets_tuple

    def matelem_options_widgets(
        self,
    ) -> tuple:
        """Creates the children for matrix elements layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.dict_v_plot_options["amplitude_mode"].v_model = self.active_defaults[
            "mode_matrixelem"
        ]
        self.dict_v_qubit_params[
            self.dict_v_plot_options["scan_param"].v_model
        ].disabled = False

        plot_options_widgets_tuple = (
            self.dict_v_plot_options["operator_choice"],
            self.dict_v_plot_options["highest_state"],
            self.dict_v_plot_options["amplitude_mode"],
            self.dict_v_plot_options["show_matelem_numbers"],
            self.dict_v_plot_options["show3d_matelem"],
        )

        return plot_options_widgets_tuple

    def coherence_times_options_widgets(
        self,
    ) -> tuple:
        """Creates the children for matrix elements layout.

        Returns
        -------
            Tuple of plot options widgets
        """
        self.dict_v_qubit_params[
            self.dict_v_plot_options["scan_param"].v_model
        ].disabled = True
        self.dict_v_plot_options["i_text"].style_ = "max-width: 50px"
        self.dict_v_plot_options["j_text"].style_ = "max-width: 50px"
        text_HBox = v.Container(
            class_="d-flex flex-row align-bottom py-0",
            children=[
                v.Text(class_="pr-3", children=["Transitions between"]),
                self.dict_v_plot_options["i_text"],
                v.Text(class_="px-3 pt-2", children=["and"]),
                self.dict_v_plot_options["j_text"],
            ],
            dense=True,
        )
        checkbox_HBox = flex_column(
            [
                self.dict_v_plot_options["t1_checkbox"],
                self.dict_v_plot_options["t2_checkbox"],
            ],
            dense=True,
            class_="py-0 my-0",
        )

        plot_options_widgets_tuple = (
            self.dict_v_plot_options["scan_param"],
            self.dict_v_plot_options["noise_channel_multiselect"],
            flex_column([text_HBox, checkbox_HBox], class_="py-0"),
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
            self.v_plot_output.clear_output(wait=True)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        if self.plot_renewal_requested:
            self.fig, ax = self.active_qubit.plot_evals_vs_paramvals(
                scan_value,
                np_list,
                evals_count=eigenvalue_state_value,
                subtract_ground=subtract_ground_tf,
            )
            self.plot_renewal_requested = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                with self.v_plot_output:
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
            with self.v_plot_output:
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
            self.v_plot_output.clear_output(wait=True)

        if isinstance(
            self.active_qubit, (scq.Transmon, scq.TunableTransmon, scq.Fluxonium)
        ):
            self.v_plot_output.outputs = tuple(
                elem
                for elem in self.v_plot_output.outputs
                if "data" in elem and "Label" not in elem["data"]["text/plain"]
            )
            if len(eigenvalue_states) == 0:
                if _HAS_WIDGET_BACKEND:
                    self.fig.axes[0].clear()
                error_label = v.Text(children=["Must select at least one state."])
                with self.v_plot_output:
                    display(error_label)
                return
            if self.plot_renewal_requested:
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states,
                    mode=mode_value,
                    scaling=scale_value,
                    phi_grid=phi_grid,
                )
                self.plot_renewal_requested = False
                if _HAS_WIDGET_BACKEND:
                    self.fig.canvas.header_visible = False
                    with self.v_plot_output:
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
            elif isinstance(self.active_qubit, (scq.ZeroPi)):  # , scq.Bifluxon)):
                grid_dict = {"theta_grid": theta_grid}
            elif isinstance(self.active_qubit, scq.Cos2PhiQubit):
                grid_dict = {"phi_grid": phi_grid, "theta_grid": theta_grid}

            if self.plot_renewal_requested:
                self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
                    which=eigenvalue_states, mode=mode_value, **grid_dict
                )
                self.plot_renewal_requested = False

                if _HAS_WIDGET_BACKEND:
                    self.fig.canvas.header_visible = False
                    with self.v_plot_output:
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
            with self.v_plot_output:
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
            self.v_plot_output.clear_output(wait=True)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        if self.plot_renewal_requested:
            self.fig, ax = self.active_qubit.plot_matelem_vs_paramvals(
                operator_value,
                scan_value,
                np_list,
                select_elems=matrix_element_state_value,
                mode=mode_value,
            )
            self.plot_renewal_requested = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                with self.v_plot_output:
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
            with self.v_plot_output:
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
            self.v_plot_output.clear_output(wait=True)
        if self.plot_renewal_requested:
            self.fig, ax = self.active_qubit.plot_matrixelements(
                operator_value,
                evals_count=eigenvalue_state_value,
                mode=mode_value,
                show_numbers=show_numbers_tf,
                show3d=show3d_tf,
            )
            self.plot_renewal_requested = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                with self.v_plot_output:
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
            with self.v_plot_output:
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
            self.v_plot_output.clear_output(wait=True)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])

        self.v_plot_output.outputs = tuple(elem for elem in self.v_plot_output.outputs)
        if len(noise_channels["coherence_times"]) == 0:
            if _HAS_WIDGET_BACKEND:
                self.fig.axes[0].clear()
            error_label = v.Text(children=["Must select at least one noise channel."])
            with self.v_plot_output:
                display(error_label)
            return

        if self.plot_renewal_requested:
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
            self.plot_renewal_requested = False
            if _HAS_WIDGET_BACKEND:
                self.fig.canvas.header_visible = False
                with self.v_plot_output:
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
            with self.v_plot_output:
                display(self.fig)
        GUI.fig_ax = self.fig, self.fig.axes
