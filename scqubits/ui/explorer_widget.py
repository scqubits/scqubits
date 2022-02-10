# explorer_widget.py
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


import itertools
import warnings
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple
from distutils.version import StrictVersion

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import get_backend as get_matplotlib_backend

import scqubits as scq
from scqubits.core.param_sweep import ParameterSlice
from scqubits.explorer import explorer_panels as panels
from scqubits.ui.gui_defaults import (
    composite_panel_names,
    default_panels,
    mode_dropdown_list,
    subsys_panel_names,
)
from scqubits.utils import misc as utils

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep

try:
    from IPython.display import display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

try:
    import ipywidgets
    from ipywidgets import (
        BoundedIntText,
        Button,
        ButtonStyle,
        Checkbox,
        IntSlider,
        Label,
        Output,
        SelectMultiple,
        Tab,
        ToggleButtons,
        interactive_output,
        ToggleButton,
        Layout,
        Select,
        SelectionSlider,
        Dropdown,
        HBox,
        VBox,
        HTML,
        FloatSlider,
    )
except ImportError:
    _HAS_IPYWIDGETS = False
else:
    _HAS_IPYWIDGETS = True


SEP = " | "
MATPLOTLIB_WIDGET_BACKEND = "module://ipympl.backend_nbagg"


# l400px = Layout(width="400px", justify_content="space-between")
# l800px = Layout(width="800px", justify_content="space-between")
# l900px = Layout(
#     width="900px",
#     align="top",
#     justify_content="space-between",
#     border="1px solid lightgrey",
#     padding="15px 15px 15px 15px",
# )


def width(pixels: int, justify_content: Optional[str] = None) -> Layout:
    if justify_content:
        return Layout(width=str(pixels) + "px", justify_content=justify_content)
    return Layout(width=str(pixels) + "px")


def boxed(pixels: int = 900) -> Layout:
    return Layout(
        width=str(pixels) + "px",
        align="top",
        justify_content="space-between",
        border="1px solid lightgrey",
        padding="15px 15px 15px 15px",
    )


@utils.Required(ipywidgets=_HAS_IPYWIDGETS)
class ExplorerSetup(ipywidgets.VBox):
    """Class for setup of Explorer."""

    def __init__(self, sweep: scq.ParameterSweep):
        """Set up all widget GUI elements and class attributes."""
        super().__init__()
        self._has_widget_backend = get_matplotlib_backend() == MATPLOTLIB_WIDGET_BACKEND
        if self._has_widget_backend and StrictVersion(
            matplotlib.__version__
        ) < StrictVersion("3.5.1"):
            warnings.warn(
                "The widget backend requires Matplotlib >=3.5.1 for proper "
                "functioning",
                UserWarning,
            )
        # else:
        #     plt.ioff()

        # self.out = ipywidgets.Output()

        self.sweep = sweep
        self.subsys_names = [subsys.id_str for subsys in self.sweep.hilbertspace]
        self.subsys_types = {
            subsys.id_str: type(subsys).__name__ for subsys in self.sweep.hilbertspace
        }
        self.subsys_types["Composite"] = "Composite"  # for use in default_panels

        self.panel_count = 0
        self.ncols = 2
        self.nrows = 1

        self.fig: Figure
        self.axes_table: List[List[Axes]]

        px = 1 / plt.rcParams['figure.dpi']
        if self._has_widget_backend:
            self.figwidth = 750 * px
            self.figheight = 260 * px
        else:
            self.figwidth = 550 * px  # compensate for inline backend's scaling
            self.figheight = 190 * px

        # == GUI elements =========================================================
        self.ui_hbox: Dict[str, HBox] = {}
        self.ui_vbox: Dict[str, VBox] = {}

        # == Choose panels tab ====================================================
        self.ui_subsys_dropdown = Dropdown(options=self.subsys_names, layout=width(185))
        self.ui_subsys_dropdown.observe(self.on_subsys_change, "value")

        self.ui_panels_checkboxes: Dict[str, Dict[str, Checkbox]] = {}
        for subsys_name in self.subsys_names:
            self.ui_panels_checkboxes[subsys_name] = {
                panel_name: Checkbox(
                    value=self.get_toggle_value_default(subsys_name, panel_name),
                    description=panel_name,
                    layout=width(185),
                    style={'description_width': 'initial'}
                )
                for panel_name in subsys_panel_names
            }
        self.ui_panels_checkboxes["Composite"] = {
            panel_name: Checkbox(
                value=self.get_toggle_value_default("Composite", panel_name),
                description=panel_name,
                layout=width(185),
                style={'description_width': 'initial'}
            )
            for panel_name in composite_panel_names
        }

        for subsys_name in self.subsys_names:
            self.ui_vbox[subsys_name] = VBox(
                [
                    self.ui_panels_checkboxes[subsys_name][panel_name]
                    for panel_name in subsys_panel_names
                ]
            )
        self.ui_vbox["current_subsys"] = VBox(
            children=self.ui_vbox[self.ui_subsys_dropdown.value].children
        )

        self.ui_vbox["Composite"] = VBox(
            [
                self.ui_panels_checkboxes["Composite"][panel_name]
                for panel_name in composite_panel_names
            ]
        )

        for _, checkbox_dict in self.ui_panels_checkboxes.items():
            for checkbox in checkbox_dict.values():
                checkbox.observe(self.on_toggle_event, "value")

        self.strings_to_panel_checkboxes = {}
        for name in self.ui_panels_checkboxes.keys():
            for panel_name in self.ui_panels_checkboxes[name].keys():
                string_id = name + SEP + panel_name
                checkbox = self.ui_panels_checkboxes[name][panel_name]
                self.strings_to_panel_checkboxes[string_id] = checkbox

        self.ui_vbox["choose_panels"] = VBox(
            [
                self.ui_subsys_dropdown,
                HBox(
                    [self.ui_vbox["current_subsys"], self.ui_vbox["Composite"]],
                    layout=width(400, justify_content="space-between"),
                ),
            ],
        )

        self.ui_panels_list = Select(
            options=self.selected_as_strings(),
            rows=8,
        )
        self.ui_delete_btn = Button(icon="trash", layout=width(35))
        self.ui_delete_btn.on_click(self.delete_panel)

        self.ui_vbox["panels_list"] = VBox(
            [
                # HTML(value="<b>Selected panels</b>"),
                HBox([self.ui_panels_list, self.ui_delete_btn]),
            ]
        )

        self.ui_hbox["panels"] = HBox(
            [self.ui_vbox["choose_panels"], self.ui_vbox["panels_list"]],
            layout=width(800, justify_content="space-between"),
        )

        # == Parameters tab ======================================================
        self.ui_sweep_param_dropdown = Dropdown(
            options=self.sweep.param_info.keys(), layout=width(150)
        )
        self.ui_sweep_param_dropdown.observe(self.update_fixed_sliders, "value")

        self.ui_sweep_value_slider = SelectionSlider(
            description=self.ui_sweep_param_dropdown.value,
            options=self.sweep.param_info[self.ui_sweep_param_dropdown.value],
            continuous_update=False,
            layout=width(215),
            style={"description_width": "initial"},
        )
        self.ui_sweep_value_slider.observe(self.update_plots, "value")

        self.ui_fixed_param_sliders = None
        self.ui_vbox["fixed_param_sliders"] = VBox([])
        self.update_fixed_sliders(None)

        self.ui_vbox["parameters"] = VBox(
            [
                VBox(
                    [HTML("<br>Active sweep parameter"), self.ui_sweep_param_dropdown],
                    layout=width(185),
                ),
                VBox([HTML("<br>Sample value"), self.ui_sweep_value_slider]),
                HTML("<br>"),
                self.ui_vbox["fixed_param_sliders"],
            ],
            layout=width(240),
        )

        self.ui_vbox["panels_select"] = VBox(
            [self.ui_hbox["panels"]],
            layout=width(900),
        )

        # self.ui_vbox["parameters"] = VBox(
        #     [self.ui_vbox["parameters"]],
        #     layout=width(900),
        # )

        # == Panel settings ========================================================
        self.ui_subsys_panel_settings = {
            subsys_name: {
                panel_name: self.create_ui_settings_subsys(subsys_index, panel_name)
                for panel_name in subsys_panel_names
            }
            for subsys_index, subsys_name in enumerate(self.subsys_names)
        }

        self.ui_transitions = {}
        self.ui_composite_panel_settings = {
            "Composite": {
                panel_name: self.create_ui_settings_composite(panel_name)
                for panel_name in composite_panel_names
            }
        }

        self.ui_settings = {
            **self.ui_subsys_panel_settings,
            **self.ui_composite_panel_settings,
        }

        # ui_settings_btn = Button(icon="toggle-on", layout=width(35))
        # ui_settings_btn.style.button_color = "white"
        # ui_settings_btn.on_click(self.toggle_settings_ui)

        # TODO: the following dropdown needs to be adjusted in options whenever panels
        # are added/deleted

        self.ui_panels_choice_dropdown = Dropdown(
            options=self.get_panels_list(), layout=width(250)
        )
        self.ui_panels_choice_dropdown.observe(self.activate_settings, "value")
        subsys_name, panel_name = self.ui_panels_choice_dropdown.value.split(SEP)
        self.ui_hbox["panel_settings"] = HBox(
            children=self.ui_settings[subsys_name][panel_name]
        )

        self.ui_hbox["panel_choice"] = HBox([self.ui_panels_choice_dropdown])

        self.ui_vbox["settings"] = VBox(
            [
                HBox([self.ui_hbox["panel_choice"]]),
                self.ui_hbox["panel_settings"],
            ],
            layout=width(900),
        )

        if self._has_widget_backend:
            self.update_layout_and_plots(None)
            self.out = self.fig.canvas
        else:
            self.out = Output(layout=width(750))
            self.update_layout_and_plots(None)
            with self.out:
                self.out.clear_output(wait=True)
                self.fig.tight_layout()
                display(self.fig)
        self.ui_hbox["main_display"] = HBox(
            [self.ui_vbox["parameters"], self.out]
        )

        # == Main Tab widget =======================================================
        self.ui_main_tab = Tab(
            children=[
                self.ui_vbox["panels_select"],
                self.ui_vbox["settings"],
                # self.ui_vbox["parameters"],
            ]
        )
        self.ui_main_tab.set_title(0, "Choose panels")
        self.ui_main_tab.set_title(1, "Panel settings")
        # self.ui_main_tab.set_title(2, "Sweep parameters")

        self.children = [self.ui_main_tab, self.ui_hbox["main_display"]]  # self.out]


    # def toggle_settings_ui(self, btn):
    #     if btn.icon == "toggle-off":
    #         btn.icon = "toggle-on"
    #         self.ui_hbox["panel_settings"].layout.display = "inherit"
    #         self.ui_hbox["panel_choice"].layout.display = "inherit"
    #     else:
    #         btn.icon = "toggle-off"
    #         self.ui_hbox["panel_settings"].layout.display = "none"
    #         self.ui_hbox["panel_choice"].layout.display = "none"

    # def toggle_panels_ui(self, btn):
    #     if btn.icon == "toggle-off":
    #         btn.icon = "toggle-on"
    #         self.ui_hbox["panels"].layout.display = "inherit"
    #     else:
    #         btn.icon = "toggle-off"
    #         self.ui_hbox["panels"].layout.display = "none"

    def display_panel(
        self,
        full_panel_name: str,
        param_slice: ParameterSlice,
        fig_ax: Tuple[Figure, Axes],
    ):
        subsys_name, panel_name = full_panel_name.split(SEP)

        if subsys_name == "Composite":
            subsys = None
        else:
            subsys = self.sweep.subsys_by_id_str(subsys_name)

        if panel_name == "Energy spectrum":
            panels.display_bare_spectrum(
                self.sweep,
                subsys,
                param_slice,
                fig_ax,
                subtract_ground=self.ui_subsys_panel_settings[subsys_name][panel_name][
                    1
                ].value,
                evals_count=self.ui_subsys_panel_settings[subsys_name][panel_name][
                    0
                ].value,
            )
        if panel_name == "Wavefunctions":
            panels.display_bare_wavefunctions(self.sweep, subsys, param_slice, fig_ax)
        if panel_name == "Matrix elements":
            (
                opname_dropdown,
                matrixscan_toggle,
                mode_dropdown,
            ) = self.ui_subsys_panel_settings[subsys_name][panel_name]
            if matrixscan_toggle.value == "fixed":
                panels.display_matrixelements(
                    sweep=self.sweep,
                    subsys=subsys,
                    operator_name=opname_dropdown.value,
                    mode_str=mode_dropdown.value,
                    param_slice=param_slice,
                    fig_ax=fig_ax,
                )
            else:
                panels.display_matrixelement_sweep(
                    sweep=self.sweep,
                    subsys=subsys,
                    operator_name=opname_dropdown.value,
                    mode_str=mode_dropdown.value,
                    param_slice=param_slice,
                    fig_ax=fig_ax,
                )
        if panel_name == "Anharmonicity":
            panels.display_anharmonicity(self.sweep, subsys, param_slice, fig_ax)
        if panel_name == "Dispersion":
            pass
        if panel_name == "Transitions":
            if self.ui_transitions["initial_dressed_inttext"].disabled:
                initial_state = tuple(
                    inttext.value
                    for inttext in self.ui_transitions["initial_state_inttexts"]
                )
            else:
                initial_state = self.ui_transitions["initial_dressed_inttext"].value

            # if self.ui_transitions["final_dressed_inttext"].disabled:
            #     final_state = tuple(
            #         inttext.value
            #         for inttext in self.ui_transitions["final_state_inttexts"]
            #     )
            # else:
            #     final_state = self.ui_transitions["final_dressed_inttext"].value

            subsys_name_tuple = self.ui_transitions["highlight_selectmultiple"].value
            if subsys_name_tuple == ():
                subsys_list = None
            else:
                subsys_list = [
                    self.sweep.subsys_by_id_str(subsys_name)
                    for subsys_name in subsys_name_tuple
                ]

            sidebands = self.ui_transitions["sidebands_checkbox"].value
            photon_number = self.ui_transitions["photons_inttext"].value
            panels.display_transitions(
                self.sweep,
                photon_number,
                subsys_list,
                initial_state,
                sidebands,
                param_slice,
                fig_ax,
            )
            # return panels.display_n_photon_qubit_transitions
        if panel_name == "Dispersive":
            pass
        if panel_name == "Custom data":
            pass

    @property
    def all_selected(self):
        return {
            name: [
                panel
                for panel in self.ui_panels_checkboxes[name].keys()
                if self.ui_panels_checkboxes[name][panel].value
            ]
            for name in self.ui_panels_checkboxes.keys()
        }

    def selected_as_strings(self):
        all_selected = self.all_selected
        selected = []
        for name in all_selected.keys():
            for panel in all_selected[name]:
                selected.append(name + SEP + panel)
        return selected

    def create_sliders(self):
        sliders = [
            SelectionSlider(
                description=param_name,
                options=param_array,
                continuous_update=False,
                layout=width(215),
                style={"description_width": "initial"},
            )
            for param_name, param_array in self.sweep.param_info.items()
            if param_name != self.ui_sweep_param_dropdown.value
        ]
        for slider in sliders:
            slider.observe(self.update_plots, "value")
        return sliders

    @property
    def fixed_params(self):
        sliders = self.ui_fixed_param_sliders
        return {slider.description: slider.value for slider in sliders}

    def on_toggle_event(self, change):
        self.ui_panels_list.options = self.selected_as_strings()
        self.ui_panels_choice_dropdown.options = self.selected_as_strings()
        self.update_layout_and_plots(change)

    def on_subsys_change(self, change):
        self.ui_vbox["current_subsys"].children = self.ui_vbox[
            self.ui_subsys_dropdown.value
        ].children

    def activate_settings(self, change):
        subsys_name, panel_name = self.ui_panels_choice_dropdown.value.split(SEP)
        self.ui_hbox["panel_settings"].children = [
            *self.ui_settings[subsys_name][panel_name]
        ]

    def delete_panel(self, change):
        btn_string = self.ui_panels_list.value
        toggle_btn = self.strings_to_panel_checkboxes[btn_string]
        toggle_btn.value = False  # this triggers an on_toggle_event

    def get_toggle_value_default(self, subsys_name, panel_name):
        sys_type = self.subsys_types[subsys_name]
        return panel_name in default_panels[sys_type]

    def get_panels_list(self):
        panels_list: List[str] = []
        for subsys_name, btn_dict in self.ui_panels_checkboxes.items():
            for btn_name, btn in btn_dict.items():
                if btn.value:
                    panels_list.append(subsys_name + SEP + btn_name)
        return panels_list

    def update_fixed_sliders(self, change):
        self.ui_fixed_param_sliders = self.create_sliders()
        self.ui_vbox["fixed_param_sliders"].children = [
            HTML("Fixed parameter(s)"),
            *self.ui_fixed_param_sliders,
        ]
        self.ui_sweep_value_slider.description = self.ui_sweep_param_dropdown.value
        self.ui_sweep_value_slider.options = self.sweep.param_info[
            self.ui_sweep_param_dropdown.value
        ]

    def bare_dressed_toggle(self, change):
        if self.ui_transitions["initial_bare_dressed_toggle"].value == "bare":
            self.ui_transitions["initial_dressed_inttext"].disabled = True
            for inttext in self.ui_transitions["initial_state_inttexts"]:
                inttext.disabled = False
        else:
            self.ui_transitions["initial_dressed_inttext"].disabled = False
            for inttext in self.ui_transitions["initial_state_inttexts"]:
                inttext.disabled = True
        # if self.ui_transitions["final_bare_dressed_toggle"].value == "bare":
        #     self.ui_transitions["final_dressed_inttext"].disabled = True
        #     for inttext in self.ui_transitions["final_state_inttexts"]:
        #         inttext.disabled = False
        # else:
        #     self.ui_transitions["final_dressed_inttext"].disabled = False
        #     for inttext in self.ui_transitions["final_state_inttexts"]:
        #         inttext.disabled = True
        self.update_plots(change)

    def fig_ax_by_index(self, index):
        row_index = index // self.ncols
        col_index = index % self.ncols
        return self.fig, self.axes_table[row_index, col_index]

    @property
    def parameter_slice(self):
        return ParameterSlice(
            self.ui_sweep_param_dropdown.value,
            self.ui_sweep_value_slider.value,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

    def update_layout_and_plots(self: "ExplorerSetup", change):
        panels = self.get_panels_list()

        nrows = len(panels) // self.ncols
        if len(panels) % self.ncols != 0:
            nrows += 1

        # with self.out:
        plt.ioff()
        if not hasattr(self, "fig"):
            self.fig, self.axes_table = plt.subplots(
                ncols=self.ncols,
                nrows=nrows,
                figsize=(self.figwidth, self.figheight * nrows),
                squeeze=False,
            )
            self.fig.canvas.toolbar_position = "right"
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
        else:
            self.fig.clear()
            self.fig.set_size_inches(self.figwidth, self.figheight * nrows)
            self.axes_table = self.fig.subplots(
                ncols=self.ncols,
                nrows=nrows,
                squeeze=False,
            )

        if len(panels) % self.ncols != 0:
            for col in range(1, self.ncols):
                self.axes_table[-1, col].remove()
        self.panel_count = len(panels)
        plt.ion()
        self.update_plots(None)

    def update_plots(self: "ExplorerSetup", change):
        param_val = self.ui_sweep_value_slider.value
        panels = self.get_panels_list()

        param_slice = ParameterSlice(
            self.ui_sweep_param_dropdown.value,
            param_val,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )
        for axes in self.axes_table.flatten():
            for item in axes.lines + axes.collections:
                item.remove()
            axes.set_prop_cycle(None)
            axes.relim()
            axes.autoscale_view()

        for index, full_panel_name in enumerate(panels):
            self.display_panel(
                full_panel_name,
                param_slice=param_slice,
                fig_ax=self.fig_ax_by_index(index),
            )

        if not self._has_widget_backend:
            with self.out:
                self.out.clear_output(wait=True)
                self.fig.tight_layout()
                display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.fig.tight_layout()

    def create_ui_settings_subsys(self, subsys_index: int, panel_name: str):
        if panel_name == "Energy spectrum":
            evals_count = self.sweep.subsys_evals_count(subsys_index)
            ui_level_slider = IntSlider(
                description="Highest level",
                min=1,
                max=evals_count,
                value=evals_count,
                continuous_update=False,
                layout=width(300),
            )
            ui_subtract_ground_checkbox = Checkbox(
                description="subtract lowest energy", value=True, layout=width(300)
            )
            ui_level_slider.observe(self.update_plots, "value")
            ui_subtract_ground_checkbox.observe(self.update_plots, "value")
            return [ui_level_slider, ui_subtract_ground_checkbox]

        if panel_name == "Matrix elements":
            subsys = self.sweep.get_subsys(subsys_index)
            ui_mode_dropdown = Dropdown(
                options=mode_dropdown_list,
                description="Plot as:",
            )
            ui_matrixscan_toggle = ToggleButtons(options=["fixed", "sweep"])
            ui_matrixscan_toggle.style.button_width = "55px"
            ui_operator_dropdown = Dropdown(
                options=subsys.get_operator_names(),
                description="Operator",
            )
            ui_mode_dropdown.observe(self.update_plots, "value")
            ui_operator_dropdown.observe(self.update_plots, "value")
            ui_matrixscan_toggle.observe(self.update_layout_and_plots, "value")
            return [ui_operator_dropdown, ui_matrixscan_toggle, ui_mode_dropdown]

        return [HBox()]

    def create_ui_settings_composite(self, panel_name: str):
        if panel_name == "Transitions":
            self.ui_transitions["initial_state_inttexts"] = [
                BoundedIntText(
                    description="",
                    min=0,
                    max=subsys.truncated_dim,
                    value=0,
                    continuous_update=False,
                    layout=width(35),
                )
                for subsys in self.sweep.hilbertspace
            ]
            self.ui_transitions["initial_dressed_inttext"] = BoundedIntText(
                description="",
                min=0,
                max=self.sweep.hilbertspace.dimension,
                value=0,
                continuous_update=False,
                layout=width(35),
                disabled=True,
            )
            # self.ui_transitions["final_state_inttexts"] = [
            #     BoundedIntText(
            #         description="",
            #         min=0,
            #         max=subsys.truncated_dim,
            #         value=0,
            #         continuous_update=False,
            #         layout=width(35),
            #     )
            #     for subsys in self.sweep.hilbertspace
            # ]
            # self.ui_transitions["final_dressed_inttext"] = BoundedIntText(
            #     description="",
            #     min=0,
            #     max=self.sweep.hilbertspace.dimension,
            #     value=1,
            #     continuous_update=False,
            #     layout=width(35),
            #     disabled=True,
            # )
            # self.ui_transitions["final_state_inttexts"][0].value = 1

            self.ui_transitions["photons_inttext"] = BoundedIntText(
                value=1, min=1, max=5, description="", layout=width(35)
            )
            self.ui_transitions["highlight_selectmultiple"] = SelectMultiple(
                description="",
                options=self.subsys_names,
                value=[self.subsys_names[0]],
                rows=4,
                layout=width(185),
            )

            self.ui_transitions["initial_bare_dressed_toggle"] = ToggleButtons(
                options=["bare", "dressed"],
                value="bare",
                description="",
                disable=False,
            )
            self.ui_transitions[
                "initial_bare_dressed_toggle"
            ].style.button_width = "45px"

            # self.ui_transitions["final_bare_dressed_toggle"] = ToggleButtons(
            #     options=["bare", "dressed"],
            #     value="bare",
            #     description="",
            #     disable=False,
            # )
            # self.ui_transitions["final_bare_dressed_toggle"].style.button_width = "45px"

            self.ui_transitions["sidebands_checkbox"] = Checkbox(
                description="show sidebands", value=False, layout=width(250)
            )
            for inttext in self.ui_transitions["initial_state_inttexts"]:
                inttext.observe(self.update_plots, "value")
            self.ui_transitions["initial_dressed_inttext"].observe(
                self.update_plots, "value"
            )
            self.ui_transitions["photons_inttext"].observe(self.update_plots, "value")
            self.ui_transitions["highlight_selectmultiple"].observe(
                self.update_plots, "value"
            )
            self.ui_transitions["sidebands_checkbox"].observe(
                self.update_plots, "value"
            )
            self.ui_transitions["initial_bare_dressed_toggle"].observe(
                self.bare_dressed_toggle, "value"
            )
            # self.ui_transitions["final_bare_dressed_toggle"].observe(
            #     self.bare_dressed_toggle, "value"
            # )

            return [
                VBox(
                    [
                        HBox(
                            [
                                Label("Initial state "),
                                *self.ui_transitions["initial_state_inttexts"],
                                self.ui_transitions["initial_bare_dressed_toggle"],
                                self.ui_transitions["initial_dressed_inttext"],
                            ],
                            layout=Layout(width="400px", justify_content="flex-end"),
                        ),
                        # HBox(
                        #     [
                        #         Label("Final state "),
                        #         *self.ui_transitions["final_state_inttexts"],
                        #         self.ui_transitions["final_bare_dressed_toggle"],
                        #         self.ui_transitions["final_dressed_inttext"],
                        #     ],
                        #     layout=Layout(width="400px", justify_content="flex-end"),
                        # ),
                        HBox(
                            [
                                Label("photons"),
                                self.ui_transitions["photons_inttext"],
                                self.ui_transitions["sidebands_checkbox"],
                            ],
                            layout=Layout(width="400px", justify_content="flex-end"),
                        ),
                    ]
                ),
                VBox(
                    [
                        HBox(
                            [
                                Label("Highlight:"),
                                self.ui_transitions["highlight_selectmultiple"],
                            ],
                            layout=Layout(width="400px", justify_content="flex-end"),
                        ),
                        # HBox(
                        #     [
                        #         self.ui_transitions["sidebands_checkbox"],
                        #     ],
                        #     layout=Layout(width="400px", justify_content="flex-end"),
                        # ),
                    ]
                ),
            ]
        return [HBox()]
