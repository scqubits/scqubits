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


import functools
import itertools
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits as scq
from scqubits.core.param_sweep import ParameterSlice
from scqubits.explorer import explorer_panels as panels
from scqubits.ui.gui_defaults import (
    composite_panel_names,
    default_panels,
    subsys_panel_names,
)
from scqubits.utils import misc as utils

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep
    from scqubits.core.qubit_base import QuantumSystem

try:
    from IPython.display import display, clear_output
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

try:
    from ipywidgets import (
        BoundedIntText,
        Button,
        ButtonStyle,
        Checkbox,
        IntSlider,
        Label,
        Output,
        SelectMultiple,
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


def boxed(pixels: int) -> Layout:
    return Layout(
        width="900px",
        align="top",
        justify_content="space-between",
        border="1px solid lightgrey",
        padding="15px 15px 15px 15px",
    )


# class Panel:
#     def __init__(
#         self,
#         explorer: "ExplorerSetup",
#         subsys: Union[None, "QuantumSystem"],
#         full_panel_name: str,
#     ):
#         self.explorer = explorer
#         self.subsys = subsys
#         self.full_panel_name = full_panel_name
#         self.subsys_name, self.panel_name = full_panel_name.split(SEP)
#
#     @property
#     def plot_func(self):
#         if self.panel_name == "Energy spectrum":
#             return functools.partial(
#                 panels.display_bare_spectrum,
#                 subsys=self.subsys,
#                 subtract_ground=self.explorer.ui_subsys_panel_settings[subsys_name][
#                     panel_name
#                 ][1],
#             )
#         if self.panel_name == "Wavefunctions":
#             return functools.partial(
#                 panels.display_bare_wavefunctions,
#                 subsys=self.subsys,
#             )
#         if self.panel_name == "Matrix elements":
#             return None
#         if self.panel_name == "Anharmoncity":
#             return functools.partial(panels.display_anharmonicity, subsys=self.subsys)
#         if self.panel_name == "Dispersion":
#             return None
#         if self.panel_name == "Transitions":
#             return None
#             # return panels.display_n_photon_qubit_transitions
#         if self.panel_name == "Dispersive":
#             return None
#         if self.panel_name == "Custom data":
#             return None


class ExplorerSetup:
    """Class for setup of Explorer."""

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS)
    def __init__(self, sweep: scq.ParameterSweep):
        """Set up all widget GUI elements and class attributes."""
        self.sweep = sweep
        self.subsys_names = [subsys.id_str for subsys in self.sweep.hilbertspace]
        self.subsys_types = {
            subsys.id_str: type(subsys).__name__ for subsys in self.sweep.hilbertspace
        }
        self.subsys_types["Composite"] = "Composite"  # for use in default_panels

        self.panel_count = 0
        self.ncols = 2

        plt.ioff()
        self.fig = plt.figure()
        self.axes_table = self.fig.subplots(1, 2)
        # == GUI elements =========================================================
        self.ui_subsys_dropdown = Dropdown(options=self.subsys_names, layout=width(185))
        self.ui_subsys_dropdown.observe(self.on_subsys_change, "value")

        self.ui_panels_btns: Dict[str, Dict[str, ToggleButton]] = {}
        for subsys_name in self.subsys_names:
            self.ui_panels_btns[subsys_name] = {
                btn_name: ToggleButton(
                    value=self.get_toggle_value_default(subsys_name, btn_name),
                    description=btn_name,
                    layout=width(185),
                )
                for btn_name in subsys_panel_names
            }
        self.ui_panels_btns["Composite"] = {
            btn_name: ToggleButton(
                value=self.get_toggle_value_default("Composite", btn_name),
                description=btn_name,
                layout=width(185),
            )
            for btn_name in composite_panel_names
        }

        self.ui_hbox: Dict[str, HBox] = {}
        self.ui_vbox: Dict[str, VBox] = {}

        for subsys_name in self.subsys_names:
            self.ui_vbox[subsys_name] = VBox(
                [
                    self.ui_panels_btns[subsys_name][btn_name]
                    for btn_name in subsys_panel_names
                ]
            )
        self.ui_vbox["current_subsys"] = VBox(
            children=self.ui_vbox[self.ui_subsys_dropdown.value].children
        )

        self.ui_vbox["Composite"] = VBox(
            [
                self.ui_panels_btns["Composite"][btn_name]
                for btn_name in composite_panel_names
            ]
        )

        for _, btn_dict in self.ui_panels_btns.items():
            for btn in btn_dict.values():
                btn.observe(self.on_toggle_event, "value")

        self.strings_to_panel_btns = {}
        for name in self.ui_panels_btns.keys():
            for btn_name in self.ui_panels_btns[name].keys():
                string_id = name + SEP + btn_name
                btn = self.ui_panels_btns[name][btn_name]
                self.strings_to_panel_btns[string_id] = btn

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
        # self.ui_panels_list.observe(self.activate_settings, "value")
        self.ui_delete_btn = Button(icon="trash", layout=width(35))
        self.ui_delete_btn.on_click(self.delete_panel)

        self.ui_vbox["panels_list"] = VBox(
            [
                # HTML(value="<b>Selected panels</b>"),
                HBox([self.ui_panels_list, self.ui_delete_btn]),
            ]
        )

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
        self.ui_sweep_value_slider.observe(self.update_explorer, "value")

        self.ui_fixed_param_sliders = None
        self.ui_vbox["fixed_param_sliders"] = VBox([])
        self.update_fixed_sliders(None)

        ui_panels_btn = Button(icon="toggle-on", layout=width(35))
        ui_panels_btn.style.button_color = "white"
        ui_panels_btn.on_click(self.toggle_panels_ui)
        self.ui_hbox["panels"] = HBox(
            [self.ui_vbox["choose_panels"], self.ui_vbox["panels_list"]],
            layout=width(800, justify_content="space-between"),
        )

        self.ui_hbox["parameters"] = HBox(
            [
                VBox([HTML("Active sweep parameter"), self.ui_sweep_param_dropdown],
                     layout=width(185)),
                VBox([HTML("Sample value"), self.ui_sweep_value_slider]),
                self.ui_vbox["fixed_param_sliders"],
            ],
            layout=width(800) #, justify_content="space-between"),
        )

        self.ui_vbox["panels_select"] = VBox(
            [
                HBox([ui_panels_btn, HTML(value="<b>Select plot " "panels</b>")]),
                self.ui_hbox["panels"],
            ],
            layout=boxed(900),
        )

        self.ui_vbox["parameters"] = VBox(
            [HTML(value="<b>Parameters</b>"), self.ui_hbox["parameters"]],
            layout=boxed(900),
        )

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

        ui_settings_btn = Button(icon="toggle-on", layout=width(35))
        ui_settings_btn.style.button_color = "white"
        ui_settings_btn.on_click(self.toggle_settings_ui)

        # TODO: the following dropdown needs to be adjusted in options whenever panels
        # are added/deleted

        self.ui_panels_choice_dropdown = Dropdown(options=self.get_panels_list(),
                                                  layout=width(250))
        self.ui_panels_choice_dropdown.observe(self.activate_settings, "value")
        subsys_name, panel_name = self.ui_panels_choice_dropdown.value.split(SEP)
        self.ui_hbox["panel_settings"] = HBox(
            children=self.ui_settings[subsys_name][panel_name]
        )

        self.ui_hbox["panel_choice"] = HBox([self.ui_panels_choice_dropdown])

        self.ui_vbox["settings"] = VBox(
            [
                HBox(
                    [
                        ui_settings_btn,
                        HTML("<b>Settings &nbsp; </b>"),
                        self.ui_hbox["panel_choice"],
                    ]
                ),
                self.ui_hbox["panel_settings"],
            ],
            layout=boxed(900),
        )

        self.out = Output()
        self.ui_vbox["explorer"] = VBox(children=[self.out])
        self.gui_box = VBox(
            [
                HTML(value="<h3>Explorer</h3>"),
                self.ui_vbox["panels_select"],
                self.ui_vbox["settings"],
                self.ui_vbox["parameters"],
                HTML(value="<br>"),
                self.ui_vbox["explorer"],
            ]
        )
        display(self.gui_box)
        self.update_explorer(None)

    def toggle_settings_ui(self, btn):
        if btn.icon == "toggle-off":
            btn.icon = "toggle-on"
            self.ui_hbox["panel_settings"].layout.display = "inherit"
            self.ui_hbox["panel_choice"].layout.display = "inherit"
        else:
            btn.icon = "toggle-off"
            self.ui_hbox["panel_settings"].layout.display = "none"
            self.ui_hbox["panel_choice"].layout.display = "none"

    def toggle_panels_ui(self, btn):
        if btn.icon == "toggle-off":
            btn.icon = "toggle-on"
            self.ui_hbox["panels"].layout.display = "inherit"
        else:
            btn.icon = "toggle-off"
            self.ui_hbox["panels"].layout.display = "none"

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
            pass
        if panel_name == "Anharmonicity":
            panels.display_anharmonicity(subsys=subsys)
        if panel_name == "Dispersion":
            pass
        if panel_name == "Transitions":
            initial_state = tuple(
                inttext.value
                for inttext in self.ui_transitions["initial_state_inttexts"]
            )
            subsys_name_tuple = self.ui_transitions["highlight_selectmultiple"].value
            if subsys_name_tuple == ():
                subsys_list = None
            else:
                subsys_list = [self.sweep.subsys_by_id_str(subsys_name) for
                               subsys_name in subsys_name_tuple]

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
                for panel in self.ui_panels_btns[name].keys()
                if self.ui_panels_btns[name][panel].value
            ]
            for name in self.ui_panels_btns.keys()
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
            slider.observe(self.update_explorer, "value")
        return sliders

    @property
    def fixed_params(self):
        sliders = self.ui_fixed_param_sliders
        return {slider.description: slider.value for slider in sliders}

    def on_toggle_event(self, change):
        self.ui_panels_list.options = self.selected_as_strings()
        self.update_explorer(change)

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
        toggle_btn = self.strings_to_panel_btns[btn_string]
        toggle_btn.value = False  # this triggers an on_toggle_event

    def get_toggle_value_default(self, subsys_name, panel_name):
        sys_type = self.subsys_types[subsys_name]
        return panel_name in default_panels[sys_type]

    def get_panels_list(self):
        panels_list: List[str] = []
        for subsys_name, btn_dict in self.ui_panels_btns.items():
            for btn_name, btn in btn_dict.items():
                if btn.value:
                    panels_list.append(subsys_name + SEP + btn_name)
        return panels_list

    def update_fixed_sliders(self, change):
        self.ui_fixed_param_sliders = self.create_sliders()
        self.ui_vbox["fixed_param_sliders"].children = [
            HTML("Fixed parameters"),
            *self.ui_fixed_param_sliders,
        ]
        self.ui_sweep_value_slider.description = self.ui_sweep_param_dropdown.value
        self.ui_sweep_value_slider.options = self.sweep.param_info[
            self.ui_sweep_param_dropdown.value
        ]

    # def start_explorer(self):
    #     # Trigger first update
    #     self.update_explorer(None)
    #     # explorer_display = interactive_output(
    #     #     self.update_explorer,
    #     #     controls={"param_val": self.ui_sweep_value_slider}
    #     # {
    #     #     "param_val": self.param_slider,
    #     #     "photonnumber": self.photon_slider,
    #     #     "initial_index": self.initial_slider,
    #     #     "primary_subsys_index": self.primary_subsys_dropdown,
    #     #     "secondary_subsys_index": self.secondary_subsys_dropdown,
    #     # },
    #     # )
    #     # if self.out is None:
    #     #     self.out = HBox([explorer_display])
    #     #     display(self.out)
    #     # else:
    #     #     self.out.children = [explorer_display]

    def fig_ax_by_index(self, index):
        row_index = index // self.ncols
        col_index = index % self.ncols
        return self.fig, self.axes_table[row_index, col_index]

    # def clear_lines(self, axes: Axes):
    #     for line in axes.lines:
    #         line.remove()
    @property
    def parameter_slice(self):
        return ParameterSlice(
            self.ui_sweep_param_dropdown.value,
            self.ui_sweep_value_slider.value,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

    def update_explorer(self: "ExplorerSetup", change):
        param_val = self.ui_sweep_value_slider.value
        panels = self.get_panels_list()

        nrows = len(panels) // self.ncols
        if len(panels) % self.ncols != 0:
            nrows += 1

        if True:  # len(panels) != self.panel_count:
            self.fig.clear()
            self.panel_count = len(panels)
            self.figsize = (9, 2.75 * nrows)
            self.fig.set_size_inches(self.figsize)
            self.axes_table = self.fig.subplots(ncols=self.ncols, nrows=nrows)
            if len(panels) % self.ncols != 0:
                for col in range(1, self.ncols):
                    self.axes_table[-1, col].remove()
        # else:
        #     for axes in itertools.chain(*self.axes_table):
        #         axes.lines = []
        #         axes.collections = []
        #         axes.set_prop_cycle(None)
        #         axes.autoscale()

        param_slice = ParameterSlice(
            self.ui_sweep_param_dropdown.value,
            param_val,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

        for index, full_panel_name in enumerate(panels):
            self.display_panel(
                full_panel_name,
                param_slice=param_slice,
                fig_ax=self.fig_ax_by_index(index),
            )

        with self.out:
            clear_output(wait=True)
            self.fig.tight_layout()
            display(self.fig)

        # initial_bare_list = [0] * len(self.sweep.hilbertspace)
        # initial_bare_list[primary_subsys_index] = initial_index
        # initial_bare = tuple(initial_bare_list)

        # energy_ground = self.sweep[param_slice.all].energy_by_dressed_index(0)
        # energy_initial = (
        #     self.sweep[param_slice.fixed].energy_by_bare_index(initial_bare)
        #     - energy_ground
        # )

        # qbt_subsys = self.sweep.get_subsys(primary_subsys_index)
        # assert isinstance(qbt_subsys, QubitBaseClass1d), (
        #     "Unsupported qubit. Explorer currently only accepts 1d qubits."
        # )
        #
        # row_count = 3
        # column_count = 2
        # fig, axes_table = plt.subplots(
        #     ncols=column_count, nrows=row_count, figsize=self.figsize
        # )
        # axes_array_flattened = np.asarray(axes_table).flatten()
        #
        # Panel 1 ----------------------------------
        # panels.display_bare_spectrum(self.sweep, qbt_subsys, param_slice, fig_ax(0))
        # #
        # # # Panels 2 and 6----------------------------
        # panels.display_bare_wavefunctions(
        #     self.sweep, qbt_subsys, param_slice, fig_ax(1)
        # )
        # #     panels.display_charge_matrixelems(
        # #         self.sweep,
        # #         initial_bare,
        # #         primary_subsys_index,
        # #         param_val,
        # #         fig_ax(5),
        # #     )
        # #
        # # # Panel 3 ----------------------------------
        #
        # panels.display_anharmonicity(self.sweep, qbt_subsys, param_slice, fig_ax(2))
        #
        # panels.display_n_photon_qubit_transitions(
        #     self.sweep, photonnumber, qbt_subsys, initial_bare, param_slice, fig_ax(3)
        # )
        #
        # # # Panel 5 ----------------------------------
        # # panels.display_kerrlike(
        # #     self.sweep,
        # #     primary_subsys_index,
        # #     secondary_subsys_index,
        # #     param_val,
        # #     fig_ax(4),
        # # )
        # return self.fig, self.axes_table

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
            ui_level_slider.observe(self.update_explorer, None)
            ui_subtract_ground_checkbox.observe(self.update_explorer, None)
            return [ui_level_slider, ui_subtract_ground_checkbox]
        else:
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
            self.ui_transitions["final_state_inttexts"] = [
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
            self.ui_transitions["final_state_inttexts"][0].value = 1

            self.ui_transitions["photons_inttext"] = BoundedIntText(
                value=1, min=1, max=5, description="", layout=width(35)
            )
            self.ui_transitions["highlight_selectmultiple"] = SelectMultiple(
                description="",
                options=self.subsys_names,
                value=[self.subsys_names[0]],
                layout=width(185),
            )

            self.ui_transitions["sidebands_checkbox"] = Checkbox(
                description="show sidebands", value=False, layout=width(250)
            )
            for inttext in (self.ui_transitions["initial_state_inttexts"] +
                            self.ui_transitions["final_state_inttexts"]):
                inttext.observe(self.update_explorer, "value")
            self.ui_transitions["photons_inttext"].observe(
                self.update_explorer, "value"
            )
            self.ui_transitions["highlight_selectmultiple"].observe(
                self.update_explorer, "value"
            )
            self.ui_transitions["sidebands_checkbox"].observe(
                self.update_explorer, "value"
            )

            return [
                VBox(
                    [
                        HBox(
                            [
                                Label("Initial state (bare)"),
                                *self.ui_transitions["initial_state_inttexts"],
                            ],
                            layout=Layout(width="265px", justify_content="flex-end"),
                        ),
                        HBox(
                            [
                                Label("Final state (bare)"),
                                *self.ui_transitions["final_state_inttexts"],
                            ],
                            layout=Layout(width="265px", justify_content="flex-end"),
                        ),
                        HBox(
                            [Label("photons"), self.ui_transitions["photons_inttext"]],
                            layout=Layout(width="265px", justify_content="flex-end"),
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
                        HBox(
                            [
                                self.ui_transitions["sidebands_checkbox"],
                            ],
                            layout=Layout(width="400px", justify_content="flex-end"),
                        ),
                    ]
                ),
            ]
        return [HBox()]
