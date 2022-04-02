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

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Axes, Figure

try:
    from ipywidgets import (
        AppLayout,
        HBox,
        IntSlider,
        Label,
        Layout,
        Text,
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

import scqubits as scq
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
        global_defaults = {
            "mode_wavefunc": "real",
            "mode_matrixelem": "abs",
            "ng": {"min": 0, "max": 1},
            "flux": {"min": 0, "max": 1},
            "EJ": {"min": 1e-10, "max": 70},
            "EC": {"min": 1e-10, "max": 5},
            "int": {"min": 1, "max": 30},
            "float": {"min": 0, "max": 30},
        }
        transmon_defaults = {
            **global_defaults,
            "scan_param": "ng",
            "operator": "n_operator",
            "ncut": {"min": 10, "max": 50},
            "scale": 1,
            "num_sample": 150,
        }
        tunabletransmon_defaults = {
            **global_defaults,
            "scan_param": "flux",
            "operator": "n_operator",
            "EJmax": global_defaults["EJ"],
            "d": {"min": 0, "max": 1},
            "ncut": {"min": 10, "max": 50},
            "scale": 1,
            "num_sample": 150,
        }
        fluxonium_defaults = {
            **global_defaults,
            "scan_param": "flux",
            "operator": "n_operator",
            "EC": {"min": 1e-2, "max": 5},
            "EL": {"min": 1e-10, "max": 2},
            "cutoff": {"min": 10, "max": 120},
            "scale": 1,
            "num_sample": 150,
        }
        fluxqubit_defaults = {
            **global_defaults,
            "scan_param": "flux",
            "operator": "n_1_operator",
            "ncut": {"min": 5, "max": 30},
            "EJ1": global_defaults["EJ"],
            "EJ2": global_defaults["EJ"],
            "EJ3": global_defaults["EJ"],
            "ECJ1": global_defaults["EC"],
            "ECJ2": global_defaults["EC"],
            "ECJ3": global_defaults["EC"],
            "ECg1": global_defaults["EC"],
            "ECg2": global_defaults["EC"],
            "ng1": global_defaults["ng"],
            "ng2": global_defaults["ng"],
            "scale": None,
            "num_sample": 100,
        }
        zeropi_defaults = {
            **global_defaults,
            "scan_param": "flux",
            "operator": "n_theta_operator",
            "ncut": {"min": 5, "max": 50},
            "EL": {"min": 1e-10, "max": 3},
            "ECJ": {"min": 1e-10, "max": 30},
            "dEJ": {"min": 0, "max": 1},
            "dCJ": {"min": 0, "max": 1},
            "scale": None,
            "num_sample": 50,
        }
        fullzeropi_defaults = {
            **global_defaults,
            "scan_param": "flux",
            "operator": "n_theta_operator",
            "ncut": {"min": 5, "max": 50},
            "EL": {"min": 1e-10, "max": 3},
            "ECJ": {"min": 1e-10, "max": 30},
            "dEJ": {"min": 0, "max": 1},
            "dCJ": {"min": 0, "max": 1},
            "dEL": {"min": 0, "max": 1},
            "dC": {"min": 0, "max": 1},
            "zeropi_cutoff": {"min": 5, "max": 30},
            "zeta_cutoff": {"min": 5, "max": 30},
            "scale": None,
            "num_sample": 50,
        }
        cos2phiqubit_defaults = {
            **global_defaults,
            "scan_param": "flux",
            "operator": "phi_operator",
            "EL": {"min": 1e-10, "max": 5},
            "ECJ": {"min": 1e-10, "max": 30},
            "dEJ": {"min": 0, "max": 0.99},
            "dL": {"min": 0, "max": 0.99},
            "dCJ": {"min": 0, "max": 0.99},
            "ncut": {"min": 5, "max": 50},
            "zeta_cut": {"min": 10, "max": 50},
            "phi_cut": {"min": 5, "max": 30},
            "scale": None,
            "num_sample": 50,
        }
        self.qubit_defaults = {
            "Transmon": transmon_defaults,
            "TunableTransmon": tunabletransmon_defaults,
            "Fluxonium": fluxonium_defaults,
            "FluxQubit": fluxqubit_defaults,
            "ZeroPi": zeropi_defaults,
            "FullZeroPi": fullzeropi_defaults,
            "Cos2PhiQubit": cos2phiqubit_defaults,
        }
        self.grid_defaults = {
            "grid_min_val": -6 * np.pi,
            "grid_max_val": 6 * np.pi,
            "grid_pt_count": 50,
        }
        self.plot_choices = [
            "Energy spectrum",
            "Wavefunctions",
            "Matrix element scan",
            "Matrix elements",
        ]
        self.supported_qubits = [
            "Transmon",
            "TunableTransmon",
            "Fluxonium",
            "FluxQubit",
            "ZeroPi",
            "FullZeroPi",
            "Cos2PhiQubit",
        ]
        self.gui_active = True
        self.qubit_change = True
        self.slow_qubits = ["FluxQubit", "ZeroPi", "FullZeroPi", "Cos2PhiQubit"]
        self.active_defaults: Dict[str, Any] = {}
        self.fig: Figure
        self.qubit_current_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_base_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_scan_params: Dict[str, Union[int, float, None]] = {}
        self.qubit_plot_options_widgets: Dict[widgets] = {}
        self.qubit_and_plot_choice_widgets: Dict[widgets] = {}
        self.qubit_params_widgets: Dict[widgets] = {}
        self.active_qubit: QubitBaseClass

        self.set_qubit("Transmon")
        self.create_qubit_and_plot_choice_widgets()

        qubit_and_plot_choice_display, plot_display = self.create_GUI()
        display(qubit_and_plot_choice_display, plot_display)

    # Initialization Methods ----------------------------------------------------------
    def initialize_qubit(self, qubit_name: str) -> None:
        """Initializes self.active_qubit to the user's choice
        using the chosen qubit's default parameters.

        Parameters
        ----------
        qubit_name:
        """
        QubitClass = getattr(scq, qubit_name)
        if self.qubit_change:
            init_params = QubitClass.default_params()

            if qubit_name == "ZeroPi" or qubit_name == "FullZeroPi":
                init_params["grid"] = scq.Grid1d(
                    min_val=self.grid_defaults["grid_min_val"],
                    max_val=self.grid_defaults["grid_max_val"],
                    pt_count=self.grid_defaults["grid_pt_count"],
                )
            self.qubit_current_params = init_params
            self.qubit_change = False

        self.active_qubit = QubitClass(**self.qubit_current_params)

    def set_qubit(self, qubit_name: str) -> None:
        """Sets up the chosen qubit to be the active qubit
        and updates the defaults and widgets accordingly.

        Parameters
        ----------
        qubit_name:
        """
        self.active_defaults = self.qubit_defaults[qubit_name]
        self.initialize_qubit(qubit_name)
        self.create_params_dict()
        self.create_plot_settings_widgets()
        self.create_qubit_params_widgets()

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

    # Widget EventHandler Methods ------------------------------------------------------
    def scan_dropdown_eventhandler(self, change):
        self.gui_active = False
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

        self.gui_active = True
        self.qubit_plot_options_widgets[
            "scan_range_slider"
        ].description = "{} range".format(change.new)

    def qubit_buttons_eventhandler(self, change):
        self.qubit_change = True

    def save_button_clicked_action(self, *args):
        self.fig.savefig(self.qubit_plot_options_widgets["filename_text"].value)

    # Methods for qubit_plot_interactive -----------------------------------------------
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
        if not self.gui_active:
            return None
        scan_min, scan_max = scan_range
        self.update_qubit_params(**params)
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        self.fig, ax = self.active_qubit.plot_evals_vs_paramvals(
            scan_value,
            np_list,
            evals_count=eigenvalue_state_value,
            subtract_ground=subtract_ground_tf,
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

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
        if not self.gui_active:
            return None
        scan_min, scan_max = scan_range
        self.update_qubit_params(**params)
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        self.fig, ax = self.active_qubit.plot_matelem_vs_paramvals(
            operator_value,
            scan_value,
            np_list,
            select_elems=matrix_element_state_value,
            mode=mode_value,
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    def grid_matelem_vs_paramvals_plot(
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
        Namely, this method is for the qubits that require a grid option.

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
        if not self.gui_active:
            return None
        self.update_grid_qubit_params(**params)
        scan_min, scan_max = scan_range
        np_list = np.linspace(scan_min, scan_max, self.active_defaults["num_sample"])
        self.fig, ax = self.active_qubit.plot_matelem_vs_paramvals(
            operator_value,
            scan_value,
            np_list,
            select_elems=matrix_element_state_value,
            mode=mode_value,
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    def scaled_wavefunction_plot(
        self,
        eigenvalue_states: Union[List[int], int],
        mode_value: str,
        manual_scale_tf: bool,
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
        if manual_scale_tf:
            self.qubit_plot_options_widgets[
                "wavefunction_scale_slider"
            ].disabled = False
        else:
            self.qubit_plot_options_widgets["wavefunction_scale_slider"].disabled = True
            scale_value = None

        self.update_qubit_params(**params)
        self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
            which=eigenvalue_states, mode=mode_value, scaling=scale_value
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    def wavefunction_plot(
        self,
        eigenvalue_states: Union[List[int], int],
        mode_value: str,
        **params: Union[Tuple[float, float], float, int]
    ) -> None:
        """This is the method associated with qubit_plot_interactive that allows for
        us  to interact with plot_wavefunction().

        Parameters
        ----------
        eigenvalue_states:
            The number of states to be plotted

        mode_value:
            Current value of the mode (e.g. real, imaginary, etc.)

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        self.update_qubit_params(**params)
        self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
            which=eigenvalue_states, mode=mode_value
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    def grid_wavefunction_plot(
        self,
        eigenvalue_states: Union[List[int], int],
        mode_value: str,
        **params: Union[Tuple[float, float], float, int]
    ) -> None:
        """This is the method associated with qubit_plot_interactive that allows for
        us  to interact with plot_wavefunction(). Namely, this method is for the
        qubits that require a grid option.

        Parameters
        ----------
        eigenvalue_states:
            The number of states to be plotted

        mode_value:
            Current value of the mode (e.g. real, imaginary, etc.)

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        self.update_grid_qubit_params(**params)
        self.fig, ax = self.active_qubit.plot_wavefunction(  # type:ignore
            which=eigenvalue_states, mode=mode_value
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

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
        self.update_qubit_params(**params)
        self.fig, ax = self.active_qubit.plot_matrixelements(
            operator_value,
            evals_count=eigenvalue_state_value,
            mode=mode_value,
            show_numbers=show_numbers_tf,
            show3d=show3d_tf,
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    def grid_matrixelements_plot(
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
        Namely, this method is for the qubits that require a grid option.

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
        self.update_grid_qubit_params(**params)
        self.fig, ax = self.active_qubit.plot_matrixelements(
            operator_value,
            evals_count=eigenvalue_state_value,
            mode=mode_value,
            show_numbers=show_numbers_tf,
            show3d=show3d_tf,
        )
        GUI.fig_ax = self.fig, ax
        plt.show()

    # Methods for create_GUI -----------------------------------------------------------
    def display_qubit_info(self, qubit_info: bool) -> None:
        """Displays the image that corresponds to the current qubit.

        Parameters
        ----------
        qubit_info: bool
        """
        if qubit_info:
            image_box = widgets.Box(layout=Layout(justify_content="center"))
            image_box.children = [
                self.qubit_plot_options_widgets["qubit_info_image_widget"]
            ]
            display(image_box)

    def energy_scan_interactive(self) -> widgets.interactive:
        """Returns an interactive for the evals_vs_paramvals

        Returns
        -------
        widgets.interactive
        """
        self.qubit_params_widgets[
            self.qubit_plot_options_widgets["scan_dropdown"].value
        ].disabled = True

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            interactive_choice = self.grid_evals_vs_paramvals_plot
        else:
            interactive_choice = self.evals_vs_paramvals_plot

        qubit_plot_interactive = widgets.interactive(
            interactive_choice,
            scan_value=self.qubit_plot_options_widgets["scan_dropdown"],
            scan_range=self.qubit_plot_options_widgets["scan_range_slider"],
            subtract_ground_tf=self.qubit_plot_options_widgets[
                "subtract_ground_checkbox"
            ],
            eigenvalue_state_value=self.qubit_plot_options_widgets[
                "eigenvalue_state_slider"
            ],
            **self.qubit_params_widgets
        )

        return qubit_plot_interactive

    def matelem_scan_interactive(self) -> widgets.interactive:
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

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            interactive_choice = self.grid_matelem_vs_paramvals_plot
        else:
            interactive_choice = self.matelem_vs_paramvals_plot

        qubit_plot_interactive = widgets.interactive(
            interactive_choice,
            operator_value=self.qubit_plot_options_widgets["operator_dropdown"],
            scan_value=self.qubit_plot_options_widgets["scan_dropdown"],
            scan_range=self.qubit_plot_options_widgets["scan_range_slider"],
            matrix_element_state_value=self.qubit_plot_options_widgets[
                "matrix_element_state_slider"
            ],
            mode_value=self.qubit_plot_options_widgets["mode_dropdown"],
            **self.qubit_params_widgets
        )

        return qubit_plot_interactive

    def wavefunction_interactive(self) -> widgets.interactive:
        """Returns an interactive for the wavefunction plot

        Returns
        -------
        widgets.interactive
        """
        if isinstance(self.active_qubit, scq.FullZeroPi):
            qubit_plot_interactive = None
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
                qubit_plot_interactive = widgets.interactive(
                    interactive_choice,
                    eigenvalue_states=which_widget,
                    mode_value=self.qubit_plot_options_widgets["mode_dropdown"],
                    manual_scale_tf=self.qubit_plot_options_widgets[
                        "manual_scale_checkbox"
                    ],
                    scale_value=self.qubit_plot_options_widgets[
                        "wavefunction_scale_slider"
                    ],
                    **self.qubit_params_widgets
                )
            else:
                qubit_plot_interactive = widgets.interactive(
                    interactive_choice,
                    eigenvalue_states=which_widget,
                    mode_value=self.qubit_plot_options_widgets["mode_dropdown"],
                    **self.qubit_params_widgets
                )

        return qubit_plot_interactive

    def matelem_interactive(self) -> widgets.interactive:
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

        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            interactive_choice = self.grid_matrixelements_plot
        else:
            interactive_choice = self.matrixelements_plot

        qubit_plot_interactive = widgets.interactive(
            interactive_choice,
            operator_value=self.qubit_plot_options_widgets["operator_dropdown"],
            eigenvalue_state_value=self.qubit_plot_options_widgets[
                "eigenvalue_state_slider"
            ],
            mode_value=self.qubit_plot_options_widgets["mode_dropdown"],
            show_numbers_tf=self.qubit_plot_options_widgets["show_numbers_checkbox"],
            show3d_tf=self.qubit_plot_options_widgets["show3d_checkbox"],
            **self.qubit_params_widgets
        )

        return qubit_plot_interactive
    
    # def coherence_scan_interactive(self) -> widgets.interactive:
    #     """
        
    #     Returns
    #     -------
    #     widgets.interactive
    #     """

    #     pass

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
        if qubit_value in self.slow_qubits:
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
        """Initializes qubit_base_params and qubit_scan_params.
        Note that qubit_scan_params will be used to create the
        dropdown options.
        """
        self.qubit_base_params.clear()
        self.qubit_scan_params.clear()
        self.qubit_base_params = dict(self.qubit_current_params)
        if isinstance(self.active_qubit, (scq.ZeroPi, scq.FullZeroPi)):
            self.qubit_base_params["grid"] = None
        if "truncated_dim" in self.qubit_base_params.keys():
            del self.qubit_base_params["truncated_dim"]

        for param_name, param_val in self.qubit_base_params.items():
            if "cut" in param_name or "grid" in param_name:
                pass
            else:
                self.qubit_scan_params[param_name] = param_val

    def create_plot_settings_widgets(self):
        """Creates all the widgets that will be used for general plotting options."""
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
            "qubit_info_image_widget": widgets.Image(
                value=image, format="jpg", layout=Layout(width="700px")
            ),
            "save_button": widgets.Button(
                icon="save", layout=widgets.Layout(width="35px")
            ),
            "filename_text": widgets.Text(
                value=str(Path.cwd().joinpath("plot.pdf")),
                description="",
                disabled=False,
                layout=Layout(width="500px"),
            ),
            "scan_dropdown": widgets.Dropdown(
                options=scan_dropdown_list,
                value=self.active_defaults["scan_param"],
                description="Scan over",
                disabled=False,
                layout=std_layout,
            ),
            "mode_dropdown": widgets.Dropdown(
                options=mode_dropdown_list,
                description="Plot as:",
                disabled=False,
                layout=std_layout,
            ),
            "operator_dropdown": widgets.Dropdown(
                options=operator_dropdown_list,
                value=self.active_defaults["operator"],
                description="Operator",
                disabled=False,
                layout=std_layout,
            ),
            "scan_range_slider": widgets.FloatRangeSlider(
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
            "eigenvalue_state_slider": widgets.IntSlider(
                min=1,
                max=10,
                value=7,
                description="Highest state",
                continuous_update=False,
                layout=std_layout,
            ),
            "matrix_element_state_slider": widgets.IntSlider(
                min=1,
                max=6,
                value=4,
                description="Highest state",
                continuous_update=False,
                layout=std_layout,
            ),
            "wavefunction_single_state_selector": widgets.IntSlider(
                min=0,
                max=10,
                value=0,
                description="State no.",
                continuous_update=False,
                layout=std_layout,
            ),
            "wavefunction_scale_slider": widgets.FloatSlider(
                min=0.1,
                max=4,
                value=self.active_defaults["scale"],
                description="\u03c8 ampl.",
                continuous_update=False,
                layout=std_layout,
            ),
            "wavefunction_multi_state_selector": widgets.SelectMultiple(
                options=range(0, 10),
                value=[0, 1, 2, 3, 4],
                description="States",
                disabled=False,
                continuous_update=False,
                layout=std_layout,
            ),
            "show_numbers_checkbox": widgets.Checkbox(
                value=False, description="Show values", disabled=False
            ),
            "show3d_checkbox": widgets.Checkbox(
                value=True, description="Show 3D", disabled=False
            ),
            "subtract_ground_checkbox": widgets.Checkbox(
                value=False, description="Subtract E\u2080", disabled=False
            ),
            "manual_scale_checkbox": widgets.Checkbox(
                value=False, description="Manual Scaling", disabled=False
            ),
        }
        self.qubit_plot_options_widgets["save_button"].on_click(
            self.save_button_clicked_action
        )
        self.qubit_plot_options_widgets["scan_dropdown"].observe(
            self.scan_dropdown_eventhandler, names="value"
        )

    def create_qubit_params_widgets(self):
        """Creates all the widgets that will be used
        for changing the parameter values for the specified qubit.
        """
        # We need to clear qubit_params_widgets since the previous widgets from the
        # old qubit will still be initialized otherwise.
        self.qubit_params_widgets.clear()

        for param_name, param_val in self.qubit_base_params.items():
            if param_name == "grid":
                grid_min = self.qubit_current_params["grid"].min_val
                grid_max = self.qubit_current_params["grid"].max_val
                self.qubit_params_widgets[param_name] = widgets.FloatRangeSlider(
                    min=-12 * np.pi,
                    max=12 * np.pi,
                    value=[grid_min, grid_max],
                    step=0.05,
                    description="Grid range",
                    continuous_update=False,
                    layout=Layout(width="300px"),
                )
            elif isinstance(param_val, int):
                kwargs = (
                    self.active_defaults.get(param_name) or self.active_defaults["int"]
                )
                self.qubit_params_widgets[param_name] = widgets.IntSlider(
                    **kwargs,
                    value=param_val,
                    description="{}:".format(param_name),
                    continuous_update=False,
                    layout=Layout(width="300px")
                )
            else:
                kwargs = (
                    self.active_defaults.get(param_name)
                    or self.active_defaults["float"]
                )
                self.qubit_params_widgets[param_name] = widgets.FloatSlider(
                    **kwargs,
                    value=param_val,
                    step=0.01,
                    description="{}:".format(param_name),
                    continuous_update=False,
                    layout=Layout(width="300px")
                )

    def create_qubit_and_plot_choice_widgets(self):
        """Creates all the widgets that controls
        which qubit or plot the user can choose from.
        """
        self.qubit_and_plot_choice_widgets = {
            "qubit_buttons": widgets.ToggleButtons(
                options=self.supported_qubits,
                description="Qubits:",
                layout=widgets.Layout(width="800px"),
            ),
            "plot_buttons": widgets.ToggleButtons(
                options=self.plot_choices,
                description="Plot:",
                button_style="info",
            ),
            "show_qubitinfo_checkbox": widgets.Checkbox(
                value=False, description="qubit info", disabled=False
            ),
        }
        self.qubit_and_plot_choice_widgets["qubit_buttons"].observe(
            self.qubit_buttons_eventhandler, names="value"
        )

    def create_plot_option_columns(
        self, qubit_plot_interactive: widgets.interactive
    ) -> List[widgets.VBox]:
        """Organizes the widgets in qubit_plot_interactive into columns.
        The first column will always contain the widgets that correspond to
        plotting options, whereas the remaining columns will contain the
        widgets that control the qubit parameters.

        Parameters
        ----------
        qubit_plot_interactive:

        Returns
        -------
        List[ widgets.VBox ]
        """
        widgets_per_column = 7
        base_index = (len(qubit_plot_interactive.children) - 1) - len(
            self.qubit_base_params
        )
        initial_index = base_index
        end_index = base_index + widgets_per_column
        widget_list = [VBox([*qubit_plot_interactive.children[0:base_index]])]

        while end_index < len(qubit_plot_interactive.children):
            widget_list.append(
                VBox([*qubit_plot_interactive.children[initial_index:end_index]])
            )
            initial_index += widgets_per_column
            end_index += widgets_per_column
        widget_list.append(VBox([*qubit_plot_interactive.children[initial_index:-1]]))

        return widget_list

    def create_qubit_plot_interactive(self, plot_value: str) -> widgets.interactive:
        """Creates the qubit_plot_interactive that corresponds to the
        selected qubit and plot option.

        Parameters
        ----------
        plot_value:
            Current plot option chosen (e.g. Energy Spectrum)

        Returns
        -------
        widgets.interactive

        """
        if plot_value == "Energy spectrum":
            return self.energy_scan_interactive()
        elif plot_value == "Matrix element scan":
            return self.matelem_scan_interactive()
        elif plot_value == "Wavefunctions":
            return self.wavefunction_interactive()
        elif plot_value == "Matrix elements":
            return self.matelem_interactive()

    def create_GUI(self) -> Tuple[widgets.VBox, widgets.interactive_output]:
        """Creates the two main components of the GUI: the qubit and plot option
        buttons and the interactive_output that connects the buttons with
        the main qubit plot.

        Returns
        -------
        Tuple[ widgets.VBox, widgets.interactive_output ]

        """
        qubit_choice_hbox = widgets.HBox(
            [
                self.qubit_and_plot_choice_widgets["qubit_buttons"],
                self.qubit_and_plot_choice_widgets["show_qubitinfo_checkbox"],
            ]
        )
        plot_choice_hbox = widgets.HBox(
            [self.qubit_and_plot_choice_widgets["plot_buttons"]]
        )

        qubit_and_plot_choice_widgets = widgets.VBox(
            [qubit_choice_hbox, plot_choice_hbox]
        )

        qubit_and_plot_choice_interactive = widgets.interactive_output(
            self.qubit_plot,
            {
                "qubit_value": self.qubit_and_plot_choice_widgets["qubit_buttons"],
                "qubit_info": self.qubit_and_plot_choice_widgets[
                    "show_qubitinfo_checkbox"
                ],
                "plot_value": self.qubit_and_plot_choice_widgets["plot_buttons"],
            },
        )
        qubit_and_plot_choice_interactive.layout.width = "975px"

        return qubit_and_plot_choice_widgets, qubit_and_plot_choice_interactive
