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
from typing import Dict, List, TYPE_CHECKING, Tuple, Union

import scqubits as scq
from scqubits.ui.gui_defaults import (
    default_panels,
    subsys_panel_names,
    composite_panel_names,
)
from scqubits.explorer import explorer_panels as panels
from scqubits.utils import misc as utils


if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep
    from scqubits.core.qubit_base import QuantumSystem

try:
    from IPython.display import display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

try:
    from ipywidgets import (
        Button,
        ButtonStyle,
        ToggleButton,
        Layout,
        Select,
        SelectionSlider,
        Dropdown,
        HBox,
        VBox,
        HTML,
    )
except ImportError:
    _HAS_IPYWIDGETS = False
else:
    _HAS_IPYWIDGETS = True


SEP = " | "


l185px = Layout(width="185px")
l215px = Layout(width="215px")
l35px = Layout(width="35px")
l400px = Layout(width="400px", justify_content="space-between")
l800px = Layout(width="800px", justify_content="space-between")
l900px = Layout(
    width="900px",
    align="top",
    justify_content="space-between",
    border="1px solid lightgrey",
    padding="20px 20px 20px 20px",
)


class Panel:
    def __init__(
        self, subsys: Union[None, "QuantumSystem"], panel_type: str, options=None
    ):
        self.subsys = subsys
        self.panel_type = panel_type
        self.options = options

    def plot_func(self):
        if self.panel_type == "Energy spectrum":
            return functools.partial(panels.display_bare_spectrum, subsys=self.subsys)
        if self.panel_type == "Wavefunctions":
            return functools.partial(
                panels.display_bare_wavefunctions, subsys=self.subsys
            )
        if self.panel_type == "Matrix elements":
            pass
        if self.panel_type == "Anharmoncity":
            return functools.partial(panels.display_anharmonicity, subsys=self.subsys)
        if self.panel_type == "Dispersion":
            pass
        if self.panel_type == "Transitions":
            return panels.display_n_photon_qubit_transitions
        if self.panel_type == "Dispersive":
            pass
        if self.panel_type == "Custom data":
            pass


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

        # == GUI elements =========================================================
        self.ui_subsys_dropdown = Dropdown(options=self.subsys_names, layout=l185px)
        self.ui_subsys_dropdown.observe(self.on_subsys_change, "value")

        self.ui_panels_btns: Dict[str, Dict[str, ToggleButton]] = {}
        for subsys_name in self.subsys_names:
            self.ui_panels_btns[subsys_name] = {
                btn_name: ToggleButton(
                    value=self.get_toggle_value_default(subsys_name, btn_name),
                    description=btn_name,
                    layout=l185px,
                )
                for btn_name in subsys_panel_names
            }
        self.ui_panels_btns["Composite"] = {
            btn_name: ToggleButton(
                value=self.get_toggle_value_default("Composite", btn_name),
                description=btn_name,
                layout=l185px,
            )
            for btn_name in composite_panel_names
        }

        self.ui_hbox: Dict[str, HBox] = {}
        self.ui_vbox: Dict[str, VBox] = {}

        # ui_vbox[<subsys_name>]
        # contains all subsys related toggle buttons
        for subsys_name in self.subsys_names:
            self.ui_vbox[subsys_name] = VBox(
                [
                    self.ui_panels_btns[subsys_name][btn_name]
                    for btn_name in subsys_panel_names
                ]
            )
        self.ui_vbox["current_subsys"] = self.ui_vbox[self.ui_subsys_dropdown.value]

        # ui_vbox["Composite"]
        # contains all composite related toggle buttons
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
                HTML(value="<b>Select plot panels</b>"),
                self.ui_subsys_dropdown,
                HBox(
                    [self.ui_vbox["current_subsys"], self.ui_vbox["Composite"]],
                    layout=l400px,
                ),
            ],
        )

        self.ui_panels_list = Select(options=self.selected_as_strings(), rows=10)
        self.ui_panels_list.observe(self.activate_settings, "value")
        self.ui_delete_btn = Button(icon="trash", layout=l35px)
        self.ui_delete_btn.on_click(self.delete_panel)

        self.ui_explorer_btn = Button(
            icon="external-link", description="Start Explorer"
        )
        self.ui_explorer_btn.on_click(self.start_explorer)

        self.ui_vbox["panels_list"] = VBox(
            [
                HTML(value="<b>Selected panels</b>"),
                HBox([self.ui_panels_list, self.ui_delete_btn]),
            ]
        )

        self.ui_sweep_param_dropdown = Dropdown(
            options=self.sweep.param_info.keys(), layout=l215px
        )
        self.ui_sweep_param_dropdown.observe(self.update_fixed_sliders, "value")

        self.ui_sweep_value_slider = SelectionSlider(
            description=self.ui_sweep_param_dropdown.value,
            options=self.sweep.param_info[self.ui_sweep_param_dropdown.value],
            layout=l215px,
            style={"description_width": "initial"},
        )

        self.ui_fixed_param_sliders = None
        self.ui_vbox["fixed_param_sliders"] = VBox([])
        self.update_fixed_sliders(caller=None)

        self.ui_hbox["panels"] = HBox(
            [self.ui_vbox["choose_panels"], self.ui_vbox["panels_list"]],
            layout=l800px,
        )

        self.ui_hbox["parameters"] = HBox(
            [
                VBox([HTML("Active sweep parameter"), self.ui_sweep_param_dropdown]),
                VBox([HTML("Sample value"), self.ui_sweep_value_slider]),
                self.ui_vbox["fixed_param_sliders"],
            ],
            layout=l800px,
        )

        self.ui_vbox["main"] = VBox(
            [self.ui_hbox["panels"], HTML(value="<br>"), self.ui_hbox["parameters"]],
            layout=l900px
        )

        self.ui_hbox["panel_settings"] = HBox([])
        self.activate_settings(None)

        self.ui_vbox["settings"] = VBox(
            [
                self.ui_hbox["panel_settings"],
            ],
            layout=l900px,
        )

        self.ui_vbox["explorer"] = VBox([])
        self.explorer = None

        self.gui_box = VBox(
            [
                HTML(value="<h3>Explorer Setup</h3>"),
                self.ui_vbox["main"],
                HTML(value="<br>"),
                self.ui_vbox["settings"],
                HTML(value="<br>"),
                self.ui_explorer_btn,
                self.ui_vbox["explorer"],
            ]
        )
        display(self.gui_box)

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
                selected += [name + SEP + panel]
        return selected

    def create_sliders(self):
        sliders = [
            SelectionSlider(
                description=param_name,
                options=param_array,
                layout=l215px,
                style={"description_width": "initial"},
            )
            for param_name, param_array in self.sweep.param_info.items()
            if param_name != self.ui_sweep_param_dropdown.value
        ]
        return sliders

    @property
    def fixed_params(self):
        sliders = self.ui_fixed_param_sliders
        return {slider.description: slider.value for slider in sliders}

    def on_toggle_event(self, change):
        self.ui_panels_list.options = self.selected_as_strings()

    def on_subsys_change(self, change):
        self.ui_vbox["current_subsys"].children = [
            self.ui_vbox[self.ui_subsys_dropdown.value],
        ]

    def activate_settings(self, caller):
        self.ui_hbox["panel_settings"].children = [
            HTML(
                value="<b class='fas fa-cog'>&ensp; Settings: {}</b>".format(
                    self.ui_panels_list.value
                )
            )
        ]

    def delete_panel(self, caller):
        btn_string = self.ui_panels_list.value
        toggle_btn = self.strings_to_panel_btns[btn_string]
        toggle_btn.value = False

    def get_toggle_value_default(self, subsys_name, panel_name):
        sys_type = self.subsys_types[subsys_name]
        return panel_name in default_panels[sys_type]

    def get_panels_list(self):
        panels_list: List[Panel] = []
        for subsys_name, btn_dict in self.ui_panels_btns.items():
            if subsys_name == "Composite":
                subsys = None
            else:
                subsys = self.sweep.subsys_by_id_str(subsys_name)

            for btn_name, btn in btn_dict.items():
                if btn.value:
                    panel = Panel(subsys=subsys, panel_type=btn_name)
                    panels_list.append(panel)
        return panels_list

    def update_fixed_sliders(self, caller):
        self.ui_fixed_param_sliders = self.create_sliders()
        self.ui_vbox["fixed_param_sliders"].children = [
            HTML("Fixed parameters"),
            *self.ui_fixed_param_sliders,
        ]
        self.ui_sweep_value_slider.description = self.ui_sweep_param_dropdown.value
        self.ui_sweep_value_slider.options = self.sweep.param_info[
            self.ui_sweep_param_dropdown.value
        ]

    def start_explorer(self, caller):
        from scqubits.explorer.explorer import Explorer2

        self.explorer = Explorer2(
            self.sweep,
            sweep_param=self.ui_sweep_param_dropdown.value,
            fixed_params=self.fixed_params,
            panels=self.get_panels_list(),
        )

        self.ui_vbox["explorer"].children = [
            self.explorer.user_interface,
            self.explorer.out,
        ]
