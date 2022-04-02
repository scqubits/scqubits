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


from cProfile import label
import inspect

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Axes, Figure
from tables import Description

try:
    from ipywidgets import (
        Image,
        Button,
        Dropdown,
        FloatRangeSlider,
        FloatSlider,
        SelectMultiple,
        Checkbox,
        Text,
        ToggleButtons,
        HBox,
        IntSlider,
        Box,
        Tab,
        Label,
        Layout,
        VBox,
        interactive,
        widgets,
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

from pathlib import Path

import qubit_defaults

import scqubits as scq
import scqubits.utils.misc as utils

from scqubits.core.qubit_base import QubitBaseClass


class GUI_REFACTORED:

    # Handle to the most recently generated Figure, Axes tuple
    fig_ax: Optional[Tuple[Figure, Axes]] = None

    def __repr__(self):
        return ""

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def __init__(self):
        scq.settings.PROGRESSBAR_DISABLED = False
        self.active_defaults: Dict[str, Any] = {}
        self.fig: Figure
        self.qubit_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_scan_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_plot_options_widgets: Dict[str, Union[Image, Button, Text, Dropdown, FloatSlider, IntSlider, FloatSlider, SelectMultiple, Checkbox]] = {}
        self.qubit_and_plot_ToggleButtons: Dict[str, ToggleButtons] = {}
        self.qubit_params_widgets: Dict[str, Union[IntSlider, FloatSlider]] = {}
        self.active_qubit: str
        self.out = widgets.Output()

        self.set_qubit("Transmon")
        self.create_qubit_and_plot_ToggleButtons()

        self.qubit_and_plot_choice_display = self.create_qubit_and_plot_ToggleButtons_layout()
        self.tab_widget = self.create_tab_layout()
        display(self.qubit_and_plot_choice_display, self.tab_widget, self.out)

        self.observe_qubit_plot_widgets()
        self.update_evals_vs_paramvals_plot()

    # Initialization Methods ----------------------------------------------------------
    def set_qubit(self, qubit_name: str) -> None:
        """Sets up the chosen qubit to be the active qubit
        and updates the defaults and widgets accordingly.

        Parameters
        ----------
        qubit_name:
        """
        self.active_defaults = qubit_defaults.qubit_defaults[qubit_name]
        self.initialize_qubit(qubit_name)
        self.create_params_dict()
        self.create_plot_settings_widgets()
        self.create_qubit_params_widgets()
    
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
                min_val=qubit_defaults.grid_defaults["grid_min_val"],
                max_val=qubit_defaults.grid_defaults["grid_max_val"],
                pt_count=qubit_defaults.grid_defaults["grid_pt_count"],
            )
        self.qubit_current_params = init_params

        self.active_qubit = QubitClass(**self.qubit_current_params)

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
    
    def observe_qubit_plot_widgets(self):
        for widget in self.qubit_params_widgets.values():
            widget.observe(self.update_evals_vs_paramvals_plot, names = "value")
        for widget in self.qubit_plot_options_widgets.values():
            widget.observe(self.update_evals_vs_paramvals_plot, names = "value")

    def unobserve_qubit_plot_widgets(self):
        for widget in self.qubit_params_widgets.values():
            widget.unobserve(self.update_evals_vs_paramvals_plot, names="value")
        for widget in self.qubit_plot_options_widgets.values():
            widget.unobserve(self.update_evals_vs_paramvals_plot, names="value")

    # Widget EventHandler Methods ------------------------------------------------------
    def scan_dropdown_eventhandler(self, change):
        self.unobserve_qubit_plot_widgets()
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
        self.observe_qubit_plot_widgets()

    def qubit_buttons_eventhandler(self, change):
        self.qubit_change = True
        self.set_qubit(change['new'])
        self.tab_widget.children = self.create_tab_layout().children

    def save_button_clicked_action(self, *args):
        self.fig.savefig(self.qubit_plot_options_widgets["filename_text"].value)

    def plot_option_refresh(self, change):
        tab_widget_list = list(self.tab_widget.children)
        plot_options_and_qubit_params_list = list(tab_widget_list[0].children)
        plot_options_and_qubit_params_list[0] = self.create_plot_option_layout()
        plot_options_and_qubit_params_tuple = tuple(plot_options_and_qubit_params_list)
        tab_widget_list[0].children = plot_options_and_qubit_params_tuple

    def plot_option_changes(self, change):
        pass

    def update_evals_vs_paramvals_plot(self, change = None):
        value_dict = {
            'scan_value': self.qubit_plot_options_widgets["scan_dropdown"].get_interact_value(),
            'scan_range': self.qubit_plot_options_widgets["scan_range_slider"].get_interact_value(),
            'subtract_ground_tf': self.qubit_plot_options_widgets[
                "subtract_ground_checkbox"
            ].get_interact_value(),
            'eigenvalue_state_value': self.qubit_plot_options_widgets[
                "eigenvalue_state_slider"
            ].get_interact_value(),
            **self.get_current_values()
            }
        self.evals_vs_paramvals_plot(**value_dict)
        print("Hello")

    # ---------------------------------

    def update_qubit_params(self, **params):
        self.qubit_current_params.update(params)
        self.active_qubit.set_params(**self.qubit_current_params)

    def update_grid_qubit_params(self, **params):
        grid_min, grid_max = params["grid"]
        updated_grid = scq.Grid1d(
            min_val=grid_min,
            max_val=grid_max,
            pt_count=self.grid_defaults["grid_pt_count"],
        )
        params.update({"grid": updated_grid})
        self.qubit_current_params.update(params)
        del params["grid"]
        params["grid_min_val"] = grid_min
        params["grid_max_val"] = grid_max
        params["grid_pt_count"] = self.grid_defaults["grid_pt_count"]
        self.active_qubit.set_params(**params)

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
        self.update_qubit_params(**params)
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        with self.out:
            plt.cla()
            self.fig, ax = self.active_qubit.plot_evals_vs_paramvals(
                scan_value,
                np_list,
                evals_count=eigenvalue_state_value,
                subtract_ground=subtract_ground_tf,
            )
            self.fig.canvas.header_visible = False
            self.out.clear_output()
            plt.close(1)
            plt.show()
        GUI_REFACTORED.fig_ax = self.fig, ax

    def grid_evals_vs_paramvals_plot(
        self,
        scan_value: str,
        scan_range: Tuple[float, float],
        eigenvalue_state_value: int,
        subtract_ground_tf: bool,
        **params: Union[Tuple[float, float], float, int]
    ) -> None:
        """This is the method associated with qubit_plot_interactive that allows for
        us to interact with plot_evals_vs_paramvals(). Namely, this method is for the
        qubits that require a grid option.

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
        if not self.gui_active:
            return None
        self.update_grid_qubit_params(**params)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        self.fig, ax = self.active_qubit.plot_evals_vs_paramvals(
            scan_value,
            np_list,
            evals_count=eigenvalue_state_value,
            subtract_ground=subtract_ground_tf,
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    # Methods for create_GUI -----------------------------------------------------------
    def energy_scan_option(self):
        """Returns an interactive for the evals_vs_paramvals

        Returns
        -------
        widgets.interactive
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = True

        qubit_plot_widgets_tuple = (
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["scan_range_slider"],
            self.qubit_plot_options_widgets["subtract_ground_checkbox"],
            self.qubit_plot_options_widgets["eigenvalue_state_slider"],
        )

        return qubit_plot_widgets_tuple

    def matelem_scan_option(self):
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

        qubit_plot_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["scan_dropdown"],
            self.qubit_plot_options_widgets["scan_range_slider"],
            self.qubit_plot_options_widgets["wavefunction_scale_slider"],
            self.qubit_plot_options_widgets["matrix_element_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
        )

        return qubit_plot_widgets_tuple

    def wavefunctions_option(self):
        """Returns an interactive for the wavefunction plot

        Returns
        -------
        widgets.interactive
        """
        if isinstance(self.active_qubit, scq.FullZeroPi):
            qubit_plot_widgets_tuple = None
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

            if isinstance(self.active_qubit, scq.ZeroPi):
                interactive_choice = self.grid_wavefunction_plot
            elif isinstance(self.active_qubit, (scq.FluxQubit, scq.Cos2PhiQubit)):
                interactive_choice = self.wavefunction_plot
            else:
                interactive_choice = self.scaled_wavefunction_plot  # type:ignore

            if interactive_choice == self.scaled_wavefunction_plot:
                qubit_plot_widgets_tuple = (
                    which_widget,
                    self.qubit_plot_options_widgets["mode_dropdown"],
                    self.qubit_plot_options_widgets["manual_scale_checkbox"],
                    self.qubit_plot_options_widgets["wavefunction_scale_slider"],
                )
            else:
                qubit_plot_widgets_tuple = (
                    which_widget,
                    self.qubit_plot_options_widgets["mode_dropdown"],
                )

        return qubit_plot_widgets_tuple

    def matelem_option(self):
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

        qubit_plot_widgets_tuple = (
            self.qubit_plot_options_widgets["operator_dropdown"],
            self.qubit_plot_options_widgets["eigenvalue_state_slider"],
            self.qubit_plot_options_widgets["mode_dropdown"],
            self.qubit_plot_options_widgets["show_numbers_checkbox"],
            self.qubit_plot_options_widgets["show3d_checkbox"],
        )

        return qubit_plot_widgets_tuple
    
    def qubit_plot(self, qubit_value: str, qubit_info: bool, plot_value: str) -> None:
        """Sets up and displays qubit_plot_interactive.

        Parameters
        ----------
        qubit_value:
            Current qubit chosen.

        qubit_info:
            Decides whether or not the image corresponding
            to the qubit is shown.

        plot_value:
            Current plot option chosen
        """
        if qubit_value in qubit_defaults.slow_qubits:
            scq.settings.PROGRESSBAR_DISABLED = False
        else:
            scq.settings.PROGRESSBAR_DISABLED = True

        self.set_qubit(qubit_value)
        self.display_qubit_info(qubit_info)
        qubit_plot_interactive = self.create_qubit_plot_interactive(plot_value)
        self.display_qubit_plot_interactive(qubit_plot_interactive)

    def display_qubit_plot_interactive(
        self, qubit_plot_interactive: widgets.interactive
    ) -> None:
        """Organizes the output for qubit_plot_interactive and displays it.

        Parameters
        ----------
        qubit_plot_interactive:
        """
        if qubit_plot_interactive is None:
            display("FullZeroPi currently does not have Wavefunctions implemented.")
            return None

        output = qubit_plot_interactive.children[-1]
        output.layout = Layout(align_items="center")
        widget_columns = self.create_plot_option_columns(qubit_plot_interactive)
        qubit_plot_interactive.children = (
            widgets.HBox(widget_columns, layout=Layout(margin="2px"), box_style="info"),
            widgets.HBox(
                [
                    self.qubit_plot_options_widgets["save_button"],
                    self.qubit_plot_options_widgets["filename_text"],
                ],
                layout=Layout(margin="2px", justify_content="flex-end"),
            ),
            output,
        )
        display(qubit_plot_interactive)

    # Create Methods -------------------------------------------------------------------
    def create_params_dict(self) -> None:
        """
        Initializes qubit_params and qubit_scan_params.
        Note that qubit_scan_params will be used to create the
        dropdown options.
        """
        self.qubit_params.clear()
        self.qubit_scan_params.clear()
        self.qubit_params = dict(self.qubit_current_params)
        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            self.qubit_params["grid"] = None
        if "truncated_dim" in self.qubit_params.keys():
            del self.qubit_params["truncated_dim"]

        for param_name, param_val in self.qubit_params.items():
            if "cut" in param_name or "grid" in param_name:
                pass
            else:
                self.qubit_scan_params[param_name] = param_val

    def create_plot_settings_widgets(self) -> None:
        """
        Creates all the widgets that will be used for general plotting options.
        """
        self.qubit_plot_options_widgets = {}
        std_layout = Layout(width="300px")

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
                value=image, format="jpg", layout=Layout(width="700px")
            ),
            "save_button": Button(
                icon="save", layout=Layout(width="35px")
            ),
            "filename_text": Text(
                value=str(Path.cwd().joinpath("plot.pdf")),
                description="",
                disabled=False,
                layout=Layout(width="500px"),
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
        self.qubit_plot_options_widgets["save_button"].on_click(
            self.save_button_clicked_action
        )
        self.qubit_plot_options_widgets["scan_dropdown"].observe(
            self.scan_dropdown_eventhandler, names="value"
        )

    def create_qubit_params_widgets(self) -> None:
        """Creates all the widgets that will be used
        for changing the parameter values for the specified qubit.
        """
        # We need to clear qubit_params_widgets since the previous widgets from the
        # old qubit will still be initialized otherwise.
        self.qubit_params_widgets.clear()
        std_layout = Layout(width="300px")

        for param_name, param_val in self.qubit_params.items():
            if param_name == "grid":
                grid_min = self.qubit_current_params["grid"].min_val
                grid_max = self.qubit_current_params["grid"].max_val
                self.qubit_params_widgets[param_name] = FloatRangeSlider(
                    min=-12 * np.pi,
                    max=12 * np.pi,
                    value=[grid_min, grid_max],
                    step=0.05,
                    description="Grid range",
                    continuous_update=False,
                    layout=std_layout,
                )
            elif isinstance(param_val, int):
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

    def create_qubit_and_plot_ToggleButtons(self) -> None:
        """Creates all the widgets that controls
        which qubit or plot the user can choose from.
        """
        self.qubit_and_plot_ToggleButtons = {
            "qubit_buttons": ToggleButtons(
                options=qubit_defaults.supported_qubits,
                description="Qubits:",
                layout=Layout(width="800px"),
            ),
            "plot_buttons": ToggleButtons(
                options=qubit_defaults.plot_choices,
                description="Plot:",
                button_style="info",
            )
        }
        self.qubit_and_plot_ToggleButtons["qubit_buttons"].observe(
            self.qubit_buttons_eventhandler, names="value"
        )
        self.qubit_and_plot_ToggleButtons["plot_buttons"].observe(
            self.plot_option_refresh, names="value"
        )

    def create_qubit_and_plot_ToggleButtons_layout(self) -> VBox:
        qubit_choice_hbox = HBox(
            [
                self.qubit_and_plot_ToggleButtons["qubit_buttons"]
            ]
        )
        plot_choice_hbox = HBox(
            [
                self.qubit_and_plot_ToggleButtons["plot_buttons"]
            ]
        )

        qubit_and_plot_choice_vbox = VBox(
            [
                qubit_choice_hbox, plot_choice_hbox
            ]
        )

        return qubit_and_plot_choice_vbox

    def create_plot_option_layout(self) -> VBox:
        current_plot_option = self.qubit_and_plot_ToggleButtons["plot_buttons"].get_interact_value()
        plot_option_vbox = VBox()
        
        if current_plot_option == "Energy spectrum":
            plot_option_vbox.children += self.energy_scan_option()
        elif current_plot_option == "Wavefunctions":
            plot_option_vbox.children += self.wavefunctions_option()
        elif current_plot_option == "Matrix element scan":
            plot_option_vbox.children += self.matelem_scan_option()
        else:
            plot_option_vbox.children += self.matelem_option()
        
        return plot_option_vbox
    
    def create_qubit_params_grid(self):
        box_layout = Layout(display='flex',
                    flex_flow='column wrap',
                    align_items='stretch',
                    width='100%',
                    height = '250px')
        qubit_params_grid = Box( 
            children = list(self.qubit_params_widgets.values()),
            layout = box_layout      
        )

        return qubit_params_grid

    def create_qubit_plot_layout(self):
        plot_option_vbox = self.create_plot_option_layout()
        plot_option_vbox.observe(self.plot_option_changes)
        qubit_params_grid = self.create_qubit_params_grid()
        qubit_params_grid.observe(self.plot_option_changes)

        qubit_plot_layout = HBox(
            [plot_option_vbox, qubit_params_grid]
        )
        
        return qubit_plot_layout
    
    def create_tab_layout(self):
        qubit_plot_options_display = self.create_qubit_plot_layout()
        qubit_plot_options_change = widgets.Label(value = "TODO")
        image_box = Box(layout=Layout(justify_content="center"))
        image_box.children = [
            self.qubit_plot_options_widgets["qubit_info_image_widget"]
        ]
        tab = Tab()
        tab_titles = ["Qubit Plot", "Change Plot Ranges", "Qubit Info"]
        tab.children = [qubit_plot_options_display, qubit_plot_options_change, image_box]
        for i in range(len(tab.children)):
            tab.set_title(i, tab_titles[i])

        return tab

