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


import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
from distutils.version import StrictVersion

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import get_backend as get_matplotlib_backend

import scqubits as scq
from scqubits.core.qubit_base import QubitBaseClass
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
except ImportError:
    _HAS_IPYWIDGETS = False
else:
    _HAS_IPYWIDGETS = True
    from ipywidgets import (
        BoundedIntText,
        Button,
        Checkbox,
        IntSlider,
        Label,
        Output,
        SelectMultiple,
        Tab,
        ToggleButtons,
        Layout,
        Select,
        SelectionSlider,
        Dropdown,
        HBox,
        VBox,
        HTML,
        FloatSlider,
    )


SEP = " | "
MATPLOTLIB_WIDGET_BACKEND = "module://ipympl.backend_nbagg"
_HAS_WIDGET_BACKEND = get_matplotlib_backend() == MATPLOTLIB_WIDGET_BACKEND


@utils.Required(ipywidgets=_HAS_IPYWIDGETS)
def width(pixels: int, justify_content: Optional[str] = None) -> Layout:
    if justify_content:
        return Layout(width=str(pixels) + "px", justify_content=justify_content)
    return Layout(width=str(pixels) + "px")


@utils.Required(ipywidgets=_HAS_IPYWIDGETS)
def boxed(pixels: int = 900) -> Layout:
    return Layout(
        width=str(pixels) + "px",
        align="top",
        border="1px solid lightgrey",
        padding="10px 10px 10px 10px",
    )


class Explorer:
    """
    Generates the UI for exploring `ParameterSweep` objects.

    Parameters
    ----------
    sweep:
        the `ParameterSweep` object to be visualized.
    """

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS)
    def __init__(self, sweep: scq.ParameterSweep):
        """Set up all widget GUI elements and class attributes."""
        self.gui_display = ipywidgets.VBox()
        self.gui_display.layout.width = "100%"
        self.gui_display.layout.height = "1000px"
        if _HAS_WIDGET_BACKEND and StrictVersion(
            matplotlib.__version__
        ) < StrictVersion("3.5.1"):
            warnings.warn(
                "The widget backend requires Matplotlib >=3.5.1 for proper functioning",
                UserWarning,
            )

        self.sweep = sweep

        self.subsys_names = [subsys.id_str for subsys in self.sweep.hilbertspace]
        self.subsys_types = {
            subsys.id_str: type(subsys).__name__ for subsys in self.sweep.hilbertspace
        }
        self.subsys_types["Composite"] = "Composite"  # for use in default_panels

        self.ncols = 2  # number of columns used for axes in the figure display
        self.fig, self.axes_table = self.build_figure_and_axes_table()
        self.figwidth: float
        self.figheight: float

        # == GUI elements =========================================================
        self.ui_hbox: Dict[str, HBox] = {}
        self.ui_vbox: Dict[str, VBox] = {}
        self.ui: Dict[str, Any] = {}

        # +--------+---------+
        # | TAB1   |  TAB2   |
        # +--------+---------+self.ui_main_tab-----------------------------------------+
        # |                                                                            |
        # |                          self.ui_hbox["panels_select_tab"]                 |
        # |                                   - or -                                   |
        # |                          self.ui_vbox["panel_settings_tab"]                |
        # |                                                                            |
        # +----------------+-----------------------------------------------------------+
        # |      self.ui_hbox["main_display"]                                          |
        # |                |                                                           |
        # |                |                                                           |
        # |                |                                                           |
        # | self.ui_vbox   |         self["ui_figure_display"] = self.fig.canvas       |
        # | ["parameters-  |                          - or -                           |
        # |   panel_left"] |                                   = Output()              |
        # |                |                                                           |
        # |                |                                                           |
        # |                |                                                           |
        # +----------------+-----------------------------------------------------------+

        self.ui_main_tab = self.build_ui_main_tab()
        self.ui_hbox["main_display"] = self.build_ui_main_display()
        self.gui_display.children = [self.ui_main_tab, self.ui_hbox["main_display"]]
        display(self.gui_display)

    def build_figure_and_axes_table(self) -> Tuple[Figure, np.ndarray]:
        # the %inline and %widget backends somehow scale differently; try to compensate
        if _HAS_WIDGET_BACKEND:
            self.figwidth = 6.5 * 1.2
            self.figheight = 2.75 * 1.2
        else:
            self.figwidth = 6.5
            self.figheight = 2.75

        plt.ioff()
        fig = plt.figure(figsize=(self.figwidth, self.figheight))
        fig.canvas.toolbar_position = "right"
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False

        plt.ion()

        axes_table = np.array([])
        return fig, axes_table

    # +--------+---------+
    # | TAB1   |  TAB2   |
    # +--------+---------+self.ui_main_tab-----------------------------------------+
    # |                                                                            |
    # |                          self.ui_hbox["panels_select_tab"]                 |
    # |                                   - or -                                   |
    # |                          self.ui_vbox["panel_settings_tab"]                |
    # |                                                                            |
    # +----------------+-----------------------------------------------------------+
    def build_ui_main_tab(self) -> Tab:
        self.ui_hbox["panels_select_tab"] = self.build_ui_panels_select_tab()
        self.ui_vbox["panel_settings_tab"] = self.build_ui_settings_tab()

        main_tab = Tab(
            children=[
                self.ui_hbox["panels_select_tab"],
                self.ui_vbox["panel_settings_tab"],
            ]
        )
        main_tab.set_title(0, "Choose panels")
        main_tab.set_title(1, "Panel settings")
        return main_tab

    # +----------------+-----------------------------------------------------------+
    # |      self.ui_hbox["main_display"]                                          |
    # |                |                                                           |
    # |                |                                                           |
    # |                |                                                           |
    # | self.ui_vbox   |         self["ui_figure_display"] = self.fig.canvas       |
    # | ["parameters-  |                          - or -                           |
    # |   panel_left"] |                                   = Output()              |
    # |                |                                                           |
    # |                |                                                           |
    # |                |                                                           |
    # +----------------+-----------------------------------------------------------+
    def build_ui_main_display(self) -> HBox:
        self.ui_vbox["parameters_panel_left"] = self.build_ui_parameters_panel_left()
        self.ui["figure_display"] = self.build_ui_figure_display()
        self.update_layout_and_plots(None)
        return HBox([self.ui_vbox["parameters_panel_left"], self.ui["figure_display"]])

    # +--------+
    # | TAB1   |
    # +-----------------------------------------------------------------------------+
    # |  subsys_dropdown      html_label           +-------------------------+  BTN |
    # |                                            |                         |      |
    # |  checkbox             checkbox             |    panels_list          |      |
    # |  checkbox             checkbox             |                         |      |
    # |    ...                   ...               +-------------------------+      |
    # +-----------------------------------------------------------------------------+
    def build_ui_panels_select_tab(self) -> HBox:
        self.ui["subsys_dropdown"] = Dropdown(
            options=self.subsys_names, layout=width(165)
        )
        self.ui["subsys_dropdown"].observe(self.on_subsys_change, "value")

        ui_panels_checkboxes: Dict[str, Dict[str, Checkbox]] = {}
        for subsys_name in self.subsys_names:
            ui_panels_checkboxes[subsys_name] = {
                panel_name: Checkbox(
                    value=self.get_toggle_value_default(subsys_name, panel_name),
                    description=panel_name,
                    layout=width(185),
                    style={"description_width": "initial"},
                )
                for panel_name in subsys_panel_names
            }
        ui_panels_checkboxes["Composite"] = {
            panel_name: Checkbox(
                value=self.get_toggle_value_default("Composite", panel_name),
                description=panel_name,
                layout=width(185),
                style={"description_width": "initial"},
            )
            for panel_name in composite_panel_names
        }
        self.ui["panels_checkboxes"] = ui_panels_checkboxes

        for subsys_name in self.subsys_names:
            self.ui_vbox[subsys_name] = VBox(
                [
                    self.ui["panels_checkboxes"][subsys_name][panel_name]
                    for panel_name in subsys_panel_names
                ]
            )

        self.ui_vbox["current_subsys"] = VBox(
            children=self.ui_vbox[self.ui["subsys_dropdown"].value].children
        )

        self.ui_vbox["Composite"] = VBox(
            [
                self.ui["panels_checkboxes"]["Composite"][panel_name]
                for panel_name in composite_panel_names
            ],
        )

        for _, checkbox_dict in self.ui["panels_checkboxes"].items():
            for checkbox in checkbox_dict.values():
                checkbox.observe(self.on_toggle_event, "value")

        self.ui["strings_to_panel_checkboxes"] = {}
        for name in self.ui["panels_checkboxes"].keys():
            for panel_name in self.ui["panels_checkboxes"][name].keys():
                string_id = name + SEP + panel_name
                checkbox = self.ui["panels_checkboxes"][name][panel_name]
                self.ui["strings_to_panel_checkboxes"][string_id] = checkbox

        html_label = HTML(
            """<p style="border:1px; border-style:solid; border-color:lightgrey; 
                         padding-left: 1em;"> 
                &nbsp;Multi-system 
                </p>"""
        )
        self.ui_hbox["choose_panels_checkboxes_group"] = HBox(
            [
                VBox([self.ui["subsys_dropdown"], self.ui_vbox["current_subsys"]]),
                VBox([html_label, self.ui_vbox["Composite"]]),
            ],
            layout=width(400, justify_content="space-between"),
        )

        self.ui["panels_list"] = Select(options=self.selected_as_strings(), rows=8)
        self.ui["delete_btn"] = Button(icon="trash", layout=width(35))
        self.ui["delete_btn"].on_click(self.delete_panel)

        self.ui_vbox["panels_list_group"] = VBox(
            [HBox([self.ui["panels_list"], self.ui["delete_btn"]])]
        )

        self.ui_hbox["panels_select_tab_content"] = HBox(
            [
                self.ui_hbox["choose_panels_checkboxes_group"],
                self.ui_vbox["panels_list_group"],
            ],
            layout=width(800, justify_content="space-between"),
        )
        return self.ui_hbox["panels_select_tab_content"]

    #          +--------+
    #          |  TAB2  |
    # +-----------------------------------------------------------------------------+
    # |  panel_choice ---------+                                                    |
    # |  panels_choice_dropdown|                                                    |
    # |                                                                             |
    # |  panel_settings ------------------------------+                             |
    # |  ui["settings"][subsys_name][panel_name]                                    |
    # |                                                                             |
    # +-----------------------------------------------------------------------------+
    def build_ui_settings_tab(self) -> VBox:
        self.ui["subsys_panel_settings"] = {
            subsys_name: {
                panel_name: self.build_ui_settings_subsys(subsys_index, panel_name)
                for panel_name in subsys_panel_names
            }
            for subsys_index, subsys_name in enumerate(self.subsys_names)
        }

        self.ui["transitions"] = {}
        self.ui["composite_panel_settings"] = {
            "Composite": {
                panel_name: self.build_ui_settings_composite(panel_name)
                for panel_name in composite_panel_names
            }
        }

        self.ui["settings"] = {
            **self.ui["subsys_panel_settings"],
            **self.ui["composite_panel_settings"],
        }

        self.ui["panels_choice_dropdown"] = Dropdown(
            options=self.get_panels_list(), layout=width(250)
        )
        self.ui["panels_choice_dropdown"].observe(self.activate_settings, "value")

        if self.ui["panels_choice_dropdown"].value:
            subsys_name, panel_name = self.ui["panels_choice_dropdown"].value.split(SEP)
            self.ui_hbox["panel_settings"] = HBox(
                children=self.ui["settings"][subsys_name][panel_name]
            )
        else:
            self.ui_hbox["panel_settings"] = HBox([])

        self.ui_hbox["panel_choice"] = HBox([self.ui["panels_choice_dropdown"]])

        return VBox(
            [
                self.ui_hbox["panel_choice"],
                HTML("<br>"),
                self.ui_hbox["panel_settings"],
            ],
            layout=Layout(width="100%"),  # width(900),
        )

    # +--parameters_panel_left-----+
    # |                            |
    # |  Active Sweep Parameter    |
    # |  "sweep_param_dropdown"    |
    # |                            |
    # |  Sample Value              |
    # |  "sweep_value_slider"      |
    # |                            |
    # |  Fixed Parameter(s)        |
    # |  "fixed_param_sliders"     |
    # |     ...                    |
    # |                            |
    # +----------------------------+
    def build_ui_parameters_panel_left(self) -> VBox:
        self.ui["sweep_param_dropdown"] = Dropdown(
            options=self.sweep.param_info.keys(), layout=width(150)
        )
        self.ui["sweep_param_dropdown"].observe(self.update_fixed_sliders, "value")

        self.ui["sweep_value_slider"] = SelectionSlider(
            description=self.ui["sweep_param_dropdown"].value,
            options=self.sweep.param_info[self.ui["sweep_param_dropdown"].value],
            continuous_update=False,
            style={"description_width": "initial"},
        )
        self.ui["sweep_value_slider"].layout.object_fit = "contain"
        self.ui["sweep_value_slider"].layout.width = "95%"
        self.ui["sweep_value_slider"].observe(self.update_plots, "value")

        self.ui["fixed_param_sliders"] = None
        self.ui_vbox["fixed_param_sliders"] = VBox([])
        self.update_fixed_sliders(None)

        return VBox(
            [
                VBox(
                    [
                        HTML("<br>Active sweep parameter"),
                        self.ui["sweep_param_dropdown"],
                    ],
                    layout=width(185),
                ),
                VBox([HTML("<br>Sample value"), self.ui["sweep_value_slider"]]),
                HTML("<br>"),
                self.ui_vbox["fixed_param_sliders"],
            ],
            layout=boxed(260),
        )

    def build_ui_figure_display(self):
        if _HAS_WIDGET_BACKEND:
            out = self.fig.canvas
            self.fig.tight_layout()
        else:
            out = Output(layout=width(750))
            out.layout.object_fit = "contain"
            out.layout.width = "100%"
            with out:
                out.clear_output(wait=True)
                self.fig.tight_layout()
                display(self.fig)
        return out

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
            panel_widget = self.ui["subsys_panel_settings"][subsys_name][panel_name]
            panels.display_bare_spectrum(
                self.sweep,
                subsys,
                param_slice,
                fig_ax,
                subtract_ground=panel_widget[1].value,
                evals_count=panel_widget[0].value,
            )
        elif panel_name == "Wavefunctions" and isinstance(subsys, QubitBaseClass):
            panels.display_bare_wavefunctions(self.sweep, subsys, param_slice, fig_ax)
        elif panel_name == "Matrix elements" and isinstance(subsys, QubitBaseClass):
            panel_widgets = self.ui["subsys_panel_settings"][subsys_name][panel_name]
            (
                opname_dropdown,
                matrixscan_toggle,
                mode_dropdown,
            ) = panel_widgets
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
        elif panel_name == "Anharmonicity":
            panels.display_anharmonicity(self.sweep, subsys, param_slice, fig_ax)
        elif panel_name == "Transitions":
            if self.ui["transitions"]["initial_dressed_inttext"].disabled:
                initial_state = tuple(
                    inttext.value
                    for inttext in self.ui["transitions"]["initial_state_inttexts"]
                )
            else:
                initial_state = self.ui["transitions"]["initial_dressed_inttext"].value

            subsys_name_tuple = self.ui["transitions"]["highlight_selectmultiple"].value
            if subsys_name_tuple == ():
                subsys_list = None
            else:
                subsys_list = [
                    self.sweep.subsys_by_id_str(subsys_name)
                    for subsys_name in subsys_name_tuple
                ]

            sidebands = self.ui["transitions"]["sidebands_checkbox"].value
            photon_number = self.ui["transitions"]["photons_inttext"].value
            panels.display_transitions(
                self.sweep,
                photon_number,
                subsys_list,
                initial_state,
                sidebands,
                param_slice,
                fig_ax,
            )
        elif panel_name == "Self-Kerr":
            panels.display_self_kerr(
                sweep=self.sweep,
                subsys=subsys,
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        elif panel_name == "Cross-Kerr, ac-Stark":
            panels.display_cross_kerr(
                sweep=self.sweep,
                subsys1=self.sweep.get_subsys(0),
                subsys2=self.sweep.get_subsys(1),
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        elif panel_name == "Custom data":
            pass

    @property
    def all_selected(self) -> Dict[str, Checkbox]:
        """Returns a dictionary labeling all selected checkboxes by their names."""
        return {
            name: [
                panel
                for panel in self.ui["panels_checkboxes"][name].keys()
                if self.ui["panels_checkboxes"][name][panel].value
            ]
            for name in self.ui["panels_checkboxes"].keys()
        }

    def selected_as_strings(self) -> List[str]:
        """Returns a list of strings capturing the names of all panels selected via
        the checkboxes."""
        all_selected = self.all_selected
        selected = []
        for name in all_selected.keys():
            for panel in all_selected[name]:
                selected.append(name + SEP + panel)
        return selected

    def create_sliders(self) -> List[SelectionSlider]:
        """Returns a list of selection sliders, one for each parameter that is part
        of the underlying ParameterSweep object."""
        sliders = [
            SelectionSlider(
                description=param_name,
                options=param_array,
                continuous_update=False,
                layout=Layout(width="95%", object_fit="contain"),
                style={"description_width": "initial"},
            )
            for param_name, param_array in self.sweep.param_info.items()
            if param_name != self.ui["sweep_param_dropdown"].value
        ]
        for slider in sliders:
            slider.observe(self.update_plots, "value")
        return sliders

    @property
    def fixed_params(self) -> Dict[str, float]:
        sliders = self.ui["fixed_param_sliders"]
        return {slider.description: slider.value for slider in sliders}

    def on_toggle_event(self, change):
        self.ui["panels_list"].options = self.selected_as_strings()
        self.ui["panels_choice_dropdown"].options = self.selected_as_strings()
        self.update_layout_and_plots(change)

    def on_subsys_change(self, change):
        self.ui_vbox["current_subsys"].children = self.ui_vbox[
            self.ui["subsys_dropdown"].value
        ].children

    def activate_settings(self, change):
        if self.ui["panels_choice_dropdown"].value:
            subsys_name, panel_name = self.ui["panels_choice_dropdown"].value.split(SEP)
            self.ui_hbox["panel_settings"].children = [
                *self.ui["settings"][subsys_name][panel_name]
            ]

    def delete_panel(self, change):
        btn_string = self.ui["panels_list"].value
        toggle_btn = self.ui["strings_to_panel_checkboxes"][btn_string]
        toggle_btn.value = False  # this triggers an on_toggle_event

    def get_toggle_value_default(self, subsys_name, panel_name):
        sys_type = self.subsys_types[subsys_name]
        return panel_name in default_panels[sys_type]

    def get_panels_list(self):
        panels_list: List[str] = []
        for subsys_name, btn_dict in self.ui["panels_checkboxes"].items():
            for btn_name, btn in btn_dict.items():
                if btn.value:
                    panels_list.append(subsys_name + SEP + btn_name)
        return panels_list

    def update_fixed_sliders(self, change):
        self.ui["fixed_param_sliders"] = self.create_sliders()
        self.ui_vbox["fixed_param_sliders"].children = [
            HTML("Fixed parameter(s)"),
            *self.ui["fixed_param_sliders"],
        ]
        self.ui["sweep_value_slider"].description = self.ui[
            "sweep_param_dropdown"
        ].value
        self.ui["sweep_value_slider"].options = self.sweep.param_info[
            self.ui["sweep_param_dropdown"].value
        ]

    def bare_dressed_toggle(self, change):
        if self.ui["transitions"]["initial_bare_dressed_toggle"].value == "bare":
            self.ui["transitions"]["initial_dressed_inttext"].disabled = True
            for inttext in self.ui["transitions"]["initial_state_inttexts"]:
                inttext.disabled = False
        else:
            self.ui["transitions"]["initial_dressed_inttext"].disabled = False
            for inttext in self.ui["transitions"]["initial_state_inttexts"]:
                inttext.disabled = True
        self.update_plots(change)

    def fig_ax_by_index(self, index):
        row_index = index // self.ncols
        col_index = index % self.ncols
        return self.fig, self.axes_table[row_index, col_index]

    @property
    def parameter_slice(self):
        return ParameterSlice(
            self.ui["sweep_param_dropdown"].value,
            self.ui["sweep_value_slider"].value,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

    def update_layout_and_plots(self: "Explorer", change):
        panels = self.get_panels_list()

        nrows = len(panels) // self.ncols
        if len(panels) % self.ncols != 0:
            nrows += 1

        for axes in self.fig.axes:
            self.fig.delaxes(axes)

        plt.ioff()
        if len(panels) > 0:
            self.axes_table = self.fig.subplots(
                ncols=self.ncols,
                nrows=nrows,
                squeeze=False,
            )
            self.fig.set_size_inches(1.1 * self.figwidth, self.figheight * nrows)

        unfilled_cols_in_last_row = (self.ncols - len(panels) % self.ncols) % self.ncols
        if unfilled_cols_in_last_row != 0:
            for col in range(self.ncols - unfilled_cols_in_last_row, self.ncols):
                self.axes_table[-1, col].remove()
        # self.panel_count = len(panels)
        self.update_plots(None)
        plt.ion()

        if not _HAS_WIDGET_BACKEND:
            with self.ui["figure_display"]:
                self.ui["figure_display"].clear_output(wait=True)
                self.fig.tight_layout()
                display(self.fig)

    def update_plots(self: "Explorer", change):
        if not hasattr(self, "fig"):
            return

        param_val = self.ui["sweep_value_slider"].value
        panels = self.get_panels_list()

        param_slice = ParameterSlice(
            self.ui["sweep_param_dropdown"].value,
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

        if not _HAS_WIDGET_BACKEND:
            with self.ui["figure_display"]:
                self.ui["figure_display"].clear_output(wait=True)
                self.fig.tight_layout()
                display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.fig.tight_layout()

    def build_ui_settings_subsys(self, subsys_index: int, panel_name: str):
        subsys = self.sweep.get_subsys(subsys_index)

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

        if panel_name == "Wavefunctions":
            if isinstance(subsys, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)):
                self.ui["wavefunction_selector"] = Select(
                    options=list(range(subsys.truncated_dim)), rows=6
                )
            else:
                self.ui["wavefunction_selector"] = SelectMultiple(
                    options=list(range(subsys.truncated_dim)), rows=6
                )
            self.ui["mode_dropdown"] = Dropdown(
                options=mode_dropdown_list,
                description="Plot as:",
            )
            return [self.ui["wavefunction_selector"], self.ui["mode_dropdown"]]

        if panel_name == "Matrix elements":
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

    def build_ui_settings_composite(self, panel_name: str):
        if panel_name == "Transitions":
            self.ui["transitions"]["initial_state_inttexts"] = [
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

            self.ui["transitions"]["initial_dressed_inttext"] = BoundedIntText(
                description="",
                min=0,
                max=self.sweep.hilbertspace.dimension,
                value=0,
                continuous_update=False,
                layout=width(35),
                disabled=True,
            )

            self.ui["transitions"]["photons_inttext"] = BoundedIntText(
                value=1, min=1, max=5, description="", layout=width(35)
            )
            self.ui["transitions"]["highlight_selectmultiple"] = SelectMultiple(
                description="",
                options=self.subsys_names,
                value=[self.subsys_names[0]],
                rows=4,
                layout=width(185),
            )

            self.ui["transitions"]["initial_bare_dressed_toggle"] = ToggleButtons(
                options=["bare", "dressed"],
                value="bare",
                description="",
                disable=False,
            )
            self.ui["transitions"][
                "initial_bare_dressed_toggle"
            ].style.button_width = "45px"

            self.ui["transitions"]["sidebands_checkbox"] = Checkbox(
                description="show sidebands", value=False, layout=width(250)
            )
            for inttext in self.ui["transitions"]["initial_state_inttexts"]:
                inttext.observe(self.update_plots, "value")
            self.ui["transitions"]["initial_dressed_inttext"].observe(
                self.update_plots, "value"
            )
            self.ui["transitions"]["photons_inttext"].observe(
                self.update_plots, "value"
            )
            self.ui["transitions"]["highlight_selectmultiple"].observe(
                self.update_plots, "value"
            )
            self.ui["transitions"]["sidebands_checkbox"].observe(
                self.update_plots, "value"
            )
            self.ui["transitions"]["initial_bare_dressed_toggle"].observe(
                self.bare_dressed_toggle, "value"
            )

            initial_state_selection = HBox(
                [
                    Label("Initial state "),
                    *self.ui["transitions"]["initial_state_inttexts"],
                    self.ui["transitions"]["initial_bare_dressed_toggle"],
                    self.ui["transitions"]["initial_dressed_inttext"],
                ],
                layout=Layout(width="400px", justify_content="flex-end"),
            )
            photon_options_selection = HBox(
                [
                    Label("photons"),
                    self.ui["transitions"]["photons_inttext"],
                    self.ui["transitions"]["sidebands_checkbox"],
                ],
                layout=Layout(width="400px", justify_content="flex-end"),
            )
            transition_highlighting = HBox(
                [
                    Label("Highlight:"),
                    self.ui["transitions"]["highlight_selectmultiple"],
                ],
                layout=Layout(width="400px", justify_content="flex-end"),
            )

            return [
                initial_state_selection,
                VBox([photon_options_selection, transition_highlighting]),
            ]
