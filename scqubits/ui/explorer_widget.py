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
from distutils.version import StrictVersion
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend as get_matplotlib_backend
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits as scq
import scqubits.ui.gui_custom_widgets as ui
from scqubits.core.param_sweep import ParameterSlice
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.explorer import explorer_panels as panels
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import (NAV_COLOR, composite_panel_names,
                                      default_panels, mode_dropdown_dict,
                                      mode_dropdown_list, subsys_panel_names)
from scqubits.ui.gui_setup import flex_row
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
    import ipyvuetify as v
    import ipywidgets
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True


SEP = " | "
MATPLOTLIB_WIDGET_BACKEND = "module://ipympl.backend_nbagg"
_HAS_WIDGET_BACKEND = get_matplotlib_backend() == MATPLOTLIB_WIDGET_BACKEND


@utils.Required(ipyvuetify=_HAS_IPYVUETIFY)
def width(pixels: int, justify_content: Optional[str] = None) -> str:
    style_str = f"width: {pixels}px;"
    if justify_content:
        style_str += f" justify_content: {justify_content}"
    return style_str


class Explorer:
    """
    Generates the UI for exploring `ParameterSweep` objects.

    Parameters
    ----------
    sweep:
        the `ParameterSweep` object to be visualized.
    """

    pass

    @utils.Required(ipyvuetify=_HAS_IPYVUETIFY)
    def __init__(self, sweep: scq.ParameterSweep):
        """Set up all widget GUI elements and class attributes."""

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
        self.ui_hbox: Dict[str, v.VuetifyWidget] = {}
        self.ui_vbox: Dict[str, v.VuetifyWidget] = {}
        self.ui: Dict[str, Any] = {}
        self.ui_level_slider = {}
        self.gui_display = v.Container(
            class_="d-flex flex-column",
            style_="width: 100%; height: 1000px",
            children=[],
        )

        self.build_navi_bar_content()
        self.ui_vbox["panel_settings_dialog"] = self.build_ui_settings_dialog()
        self.ui_vbox["parameters_panel_left"] = self.build_ui_parameters_panel_left()

        self.ui["figure_display"] = self.build_ui_figure_display()

        self.v_nav_btn_sweep_conf = ui.IconButton(
            icon_name="mdi-chart-bell-curve", onclick=self.toggle_mini, color=NAV_COLOR
        )
        self.v_nav_panel_sweep_conf = ui.NavbarElement(
            header="Sweep configure", children=self.ui_vbox["parameters_panel_left"]
        )

        self.v_nav_btn_composite = ui.IconButton(
            icon_name="mdi-apache-kafka", onclick=self.toggle_mini, color=NAV_COLOR
        )
        self.v_nav_plot_panel_composite = ui.NavbarElement(
            header="Composite plots", children=[self.ui_vbox["Composite"]]
        )

        self.v_nav_btn_subsystem = ui.IconButton(
            icon_name="mdi-apps", onclick=self.toggle_mini, color=NAV_COLOR
        )
        self.v_nav_panel_subsystem = ui.NavbarElement(
            header="Subsystem plots",
            children=[self.ui["subsys_dropdown"], self.ui_vbox["current_subsys"]],
        )

        self.v_nav_plot_settings_btn = ui.IconButton(
            icon_name="mdi-tune", onclick=self.bring_up_dialog, color=NAV_COLOR
        )
        self.v_nav_plot_settings_panel = ui.NavbarElement(
            header="Plot settings", children=[self.ui_vbox["panels_list_group"]]
        )

        self.btn_panel_map = {
            self.v_nav_btn_sweep_conf: self.v_nav_panel_sweep_conf,
            self.v_nav_btn_composite: self.v_nav_plot_panel_composite,
            self.v_nav_btn_subsystem: self.v_nav_panel_subsystem,
            self.v_nav_plot_settings_btn: self.v_nav_plot_settings_panel,
        }

        self.navi_bar = v.NavigationDrawer(
            v_model="drawer",
            permanent=True,
            mini_variant=True,
            mini_variant_width=70,
            min_width=390,
            width=390,
            elevation="2",
            color=NAV_COLOR,
            children=[
                v.List(
                    children=[
                        v.ListItem(
                            class_="align-items-bottom",
                            children=[
                                self.v_nav_btn_composite,
                                self.v_nav_plot_panel_composite,
                            ],
                        ),
                        v.ListItem(
                            class_="align-items-bottom",
                            children=[
                                self.v_nav_btn_subsystem,
                                self.v_nav_panel_subsystem,
                            ],
                        ),
                        v.ListItem(
                            class_="align-items-bottom",
                            children=[
                                self.v_nav_btn_sweep_conf,
                                self.v_nav_panel_sweep_conf,
                            ],
                        ),
                        v.ListItem(
                            class_="align-items-bottom",
                            children=[
                                self.v_nav_plot_settings_btn,
                                self.v_nav_plot_settings_panel,
                            ],
                        ),
                    ]
                )
            ],
        )

        self.update_layout_and_plots(None)

        self.full_ui = flex_row(
            [
                self.navi_bar,
                v.Container(
                    class_="d-flex flex-row overflow-auto",
                    children=[self.ui["figure_display"]],
                ),
            ],
            class_="px-0",
        )

        display(self.full_ui)

    def toggle_mini(self, widget, event, data):
        require_panel_uncollapse = self.navi_bar.mini_variant

        panel = self.btn_panel_map[widget]
        if require_panel_uncollapse:
            from datetime import datetime

            panel.v_model = 0
        else:
            for panel in self.btn_panel_map.values():
                panel.v_model = None

        self.navi_bar.mini_variant = not self.navi_bar.mini_variant

    @matplotlib.rc_context(matplotlib_settings)
    def build_figure_and_axes_table(self) -> Tuple[Figure, np.ndarray]:
        # the %inline and %widget backends somehow scale differently; try to compensate
        self.figwidth = 6.4
        self.figheight = 2.6

        plt.ioff()
        fig = plt.figure(figsize=(self.figwidth, self.figheight))
        fig.canvas.toolbar_position = "right"
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False

        plt.ion()

        axes_table = np.array([])
        return fig, axes_table

    def bring_up_dialog(self, *args):
        self.ui_vbox["panel_settings_dialog"].v_model = True
        display(self.ui_vbox["panel_settings_dialog"])

    def build_navi_bar_content(self) -> None:
        self.ui["subsys_dropdown"] = ui.InitSelect(
            items=self.subsys_names,
            label="Subsystem",
            filled=True,
            outlined=True,
            height=40,
        )
        self.ui["subsys_dropdown"].observe(self.on_subsys_change, names="v_model")

        ui_panels_switches: Dict[str, Dict[str, v.Switch]] = {}
        for subsys_name in self.subsys_names:
            ui_panels_switches[subsys_name] = {
                panel_name: v.Switch(
                    v_model=self.get_toggle_value_default(subsys_name, panel_name),
                    label=panel_name,
                    dense=True,
                    width=185,
                )
                for panel_name in subsys_panel_names
            }
        ui_panels_switches["Composite"] = {
            panel_name: v.Switch(
                v_model=self.get_toggle_value_default("Composite", panel_name),
                label=panel_name,
                dense=True,
                width=185,
            )
            for panel_name in composite_panel_names
        }
        self.ui["panels_switches"] = ui_panels_switches

        for subsys_name in self.subsys_names:
            self.ui_vbox[subsys_name] = v.Container(
                class_="d-flex flex-column",
                dense=True,
                children=[
                    self.ui["panels_switches"][subsys_name][panel_name]
                    for panel_name in subsys_panel_names
                ],
            )

        self.ui_vbox["current_subsys"] = v.Container(
            class_="d-flex flex-column pt-0",
            children=self.ui_vbox[self.ui["subsys_dropdown"].v_model].children,
        )

        self.ui_vbox["Composite"] = v.Container(
            class_="d-flex flex-column pt-0",
            children=[
                self.ui["panels_switches"]["Composite"][panel_name]
                for panel_name in composite_panel_names
            ],
        )

        for _, widget_dict in self.ui["panels_switches"].items():
            for widget in widget_dict.values():
                widget.observe(self.on_toggle_event, names="v_model")

        self.ui["strings_to_panel_switches"] = {}
        for name in self.ui["panels_switches"].keys():
            for panel_name in self.ui["panels_switches"][name].keys():
                string_id = panel_name + SEP + name
                switch = self.ui["panels_switches"][name][panel_name]
                self.ui["strings_to_panel_switches"][string_id] = switch

        self.ui["panels_list"] = ui.InitSelect(
            items=self.selected_as_strings(), filled=True, dense=True
        )
        self.ui["delete_btn"] = ui.IconButton("mdi-delete", class_="ml-2")
        self.ui["delete_btn"].on_event("click", self.delete_panel)
        self.ui["edit_btn"] = ui.IconButton("mdi-pencil")

        self.ui["edit_btn"].on_event("click", self.bring_up_dialog)

        self.ui_vbox["panels_list_group"] = v.Container(
            class_="d-flex flex-column mx-0 my-0 px-0 py-0",
            children=[
                self.ui["panels_list"],
                v.Container(
                    class_="d-flex flex-row mx-0 my-0 px-0 py-0",
                    children=[self.ui["edit_btn"], self.ui["delete_btn"]],
                ),
            ],
        )

    def build_ui_settings_dialog(self) -> v.Dialog:
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

        self.ui["panels_choice_dropdown"] = ui.InitSelect(
            items=self.get_panels_list(), width=250
        )
        self.ui["panels_choice_dropdown"].observe(
            self.activate_settings, names="v_model"
        )

        if self.ui["panels_choice_dropdown"].v_model:
            panel_name, subsys_name = self.ui["panels_choice_dropdown"].v_model.split(
                SEP
            )
            self.ui_hbox["panel_settings"] = v.Container(
                class_="d-flex flex-row",
                children=self.ui["settings"][subsys_name][panel_name],
            )
        else:
            self.ui_hbox["panel_settings"] = v.Container(
                class_="d-flex flex-row", children=[]
            )

        self.ui_hbox["panel_choice"] = v.Container(
            class_="d-flex flex-row", children=[self.ui["panels_choice_dropdown"]]
        )

        return v.Dialog(
            v_model="dialog",
            width=800,
            children=[
                v.Card(
                    children=[
                        v.CardTitle(children=["Plot settings"]),
                        v.CardText(
                            children=[
                                v.Container(
                                    class_="d-flex flex-column",
                                    children=[
                                        self.ui_hbox["panel_choice"],
                                        v.Spacer(),
                                        self.ui_hbox["panel_settings"],
                                    ],
                                    style_="width: 100%",
                                )
                            ]
                        ),
                    ]
                )
            ],
        )

    @property
    def fixed_param(self):
        return self.ui["sweep_param_dropdown"].v_model

    @property
    def param_vals(self):
        return self.sweep.param_info[self.fixed_param]

    def build_ui_parameters_panel_left(self) -> List[v.VuetifyWidget]:
        self.ui["sweep_param_dropdown"] = ui.InitSelect(
            label="Sweeping over", items=list(self.sweep.param_info.keys()), width=150
        )
        self.ui["sweep_param_dropdown"].observe(
            self.update_fixed_sliders, names="v_model"
        )

        self.ui["sweep_value_slider"] = ui.DiscreteSetSlider(
            label="",  # f"Sample point: {self.fixed_param}",  # = " {self.current},
            param_vals=self.param_vals,
            filled=False,
        )

        self.ui["sweep_value_slider"].observe(self.update_plots, names="v_model")

        self.ui["fixed_param_sliders"] = None
        self.ui_vbox["fixed_param_sliders"] = v.Container(
            class_="d-flex flex-column", children=[]
        )
        self.update_fixed_sliders(None)

        return [
            self.ui["sweep_param_dropdown"],
            v.Spacer(),
            v.ListItemTitle(style_="font-weight: 300", children=["Fixed parameter(s)"]),
            self.ui["sweep_value_slider"],
            *self.ui["fixed_param_sliders"],
        ]

    @matplotlib.rc_context(matplotlib_settings)
    def build_ui_figure_display(self) -> ipywidgets.Output:
        if _HAS_WIDGET_BACKEND:
            out = self.fig.canvas
            self.fig.tight_layout()
        else:
            out = ipywidgets.Output(width=750)
            out.layout.object_fit = "contain"
            out.layout.width = "100%"
            with out:
                out.clear_output(wait=True)
                self.fig.tight_layout()
                display(self.fig)
        return out

    @matplotlib.rc_context(matplotlib_settings)
    def display_panel(
        self,
        full_panel_name: str,
        param_slice: ParameterSlice,
        fig_ax: Tuple[Figure, Axes],
    ):
        panel_name, subsys_name = full_panel_name.split(SEP)

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
                subtract_ground=panel_widget[1].v_model,
                evals_count=int(self.ui_level_slider[full_panel_name].v_model),
            )
        elif panel_name == "Wavefunctions" and isinstance(subsys, QubitBaseClass):
            ui_wavefunction_selector, ui_mode_dropdown = self.ui["subsys_panel_settings"][subsys_name][panel_name]
            panels.display_bare_wavefunctions(
                self.sweep,
                subsys,
                param_slice,
                fig_ax,
                mode=mode_dropdown_dict[ui_mode_dropdown.v_model],
                which=ui_wavefunction_selector.v_model,
            )
        elif panel_name == "Matrix elements" and isinstance(subsys, QubitBaseClass):
            panel_widgets = self.ui["subsys_panel_settings"][subsys_name][panel_name]
            (
                opname_dropdown,
                matrixscan_toggle,
                mode_dropdown,
            ) = panel_widgets
            if matrixscan_toggle.v_model == "fixed":
                panels.display_matrixelements(
                    sweep=self.sweep,
                    subsys=subsys,
                    operator_name=opname_dropdown.v_model,
                    mode_str=mode_dropdown.v_model,
                    param_slice=param_slice,
                    fig_ax=fig_ax,
                )
            else:
                panels.display_matrixelement_sweep(
                    sweep=self.sweep,
                    subsys=subsys,
                    operator_name=opname_dropdown.v_model,
                    mode_str=mode_dropdown.v_model,
                    param_slice=param_slice,
                    fig_ax=fig_ax,
                )
        elif panel_name == "Anharmonicity":
            panels.display_anharmonicity(self.sweep, subsys, param_slice, fig_ax)
        elif panel_name == "Transitions":
            if self.ui["transitions"]["initial_dressed_inttext"].disabled:
                initial_state = tuple(
                    inttext.v_model
                    for inttext in self.ui["transitions"]["initial_state_inttexts"]
                )
            else:
                initial_state = self.ui["transitions"][
                    "initial_dressed_inttext"
                ].v_model

            subsys_name_tuple = self.ui["transitions"][
                "highlight_selectmultiple"
            ].v_model
            if subsys_name_tuple == ():
                subsys_list = None
            else:
                subsys_list = [
                    self.sweep.subsys_by_id_str(subsys_name)
                    for subsys_name in subsys_name_tuple
                ]

            sidebands = self.ui["transitions"]["sidebands_switch"].v_model
            photon_number = self.ui["transitions"]["photons_inttext"].v_model
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
    def all_selected(self) -> Dict[str, v.Switch]:
        """Returns a dictionary labeling all selected switches by their names."""
        return {
            name: [
                panel
                for panel in self.ui["panels_switches"][name].keys()
                if self.ui["panels_switches"][name][panel].v_model
            ]
            for name in self.ui["panels_switches"].keys()
        }

    def selected_as_strings(self) -> List[str]:
        """Returns a list of strings capturing the names of all panels selected via
        the switches."""
        all_selected = self.all_selected
        selected = []
        for name in all_selected.keys():
            for panel in all_selected[name]:
                selected.append(panel + SEP + name)
        return selected

    def create_sliders(self) -> List[v.VuetifyWidget]:
        """Returns a list of selection sliders, one for each parameter that is part
        of the underlying ParameterSweep object."""
        sliders = [
            ui.InitSelect(
                label=param_name,
                items=param_array,
            )
            for param_name, param_array in self.sweep.param_info.items()
            if param_name != self.ui["sweep_param_dropdown"].v_model
        ]
        for slider in sliders:
            slider.observe(self.update_plots, names="v_model")
        return sliders

    @property
    def fixed_params(self) -> Dict[str, float]:
        sliders = self.ui["fixed_param_sliders"]
        return {slider.description: slider.v_model for slider in sliders}

    def on_toggle_event(self, change):
        self.ui["panels_list"].options = self.selected_as_strings()
        self.ui["panels_choice_dropdown"].options = self.selected_as_strings()
        self.update_layout_and_plots(change)

    def on_subsys_change(self, change):
        self.ui_vbox["current_subsys"].children = self.ui_vbox[
            self.ui["subsys_dropdown"].v_model
        ].children

    def activate_settings(self, change):
        if self.ui["panels_choice_dropdown"].v_model:
            panel_name, subsys_name = self.ui["panels_choice_dropdown"].v_model.split(
                SEP
            )
            self.ui_hbox["panel_settings"].children = [
                *self.ui["settings"][subsys_name][panel_name]
            ]

    def delete_panel(self, *args):
        btn_string = self.ui["panels_list"].v_model
        toggle_btn = self.ui["strings_to_panel_switches"][btn_string]
        toggle_btn.v_model = False  # this triggers an on_toggle_event

    def get_toggle_value_default(self, subsys_name, panel_name):
        sys_type = self.subsys_types[subsys_name]
        return panel_name in default_panels[sys_type]

    def get_panels_list(self):
        panels_list: List[str] = []
        for subsys_name, btn_dict in self.ui["panels_switches"].items():
            for btn_name, btn in btn_dict.items():
                if btn.v_model:
                    panels_list.append(btn_name + SEP + subsys_name)
        return panels_list

    def update_fixed_sliders(self, change):
        self.ui["fixed_param_sliders"] = self.create_sliders()
        self.ui_vbox["fixed_param_sliders"].children = [
            *self.ui["fixed_param_sliders"],
        ]
        self.ui["sweep_value_slider"].label = self.ui["sweep_param_dropdown"].v_model
        self.ui["sweep_value_slider"].options = self.sweep.param_info[
            self.ui["sweep_param_dropdown"].v_model
        ]

    def bare_dressed_toggle(self, change):
        if self.ui["transitions"]["initial_bare_dressed_toggle"].v_model == "bare":
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
            self.ui["sweep_param_dropdown"].v_model,
            self.ui["sweep_value_slider"].current_value(),
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

    @matplotlib.rc_context(matplotlib_settings)
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
            self.fig.set_size_inches(1.1 * self.figwidth, 0.9 * self.figheight * nrows)

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

    @matplotlib.rc_context(matplotlib_settings)
    def update_plots(self: "Explorer", change):
        if not hasattr(self, "fig"):
            return

        param_val = self.ui["sweep_value_slider"].current_value()
        panels = self.get_panels_list()

        param_slice = ParameterSlice(
            self.ui["sweep_param_dropdown"].v_model,
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
        full_panel_name = panel_name + SEP + subsys.id_str

        if panel_name == "Energy spectrum":
            evals_count = self.sweep.subsys_evals_count(subsys_index)
            self.ui_level_slider[full_panel_name] = ui.NumberEntryWidget(
                num_type=int,
                label="Highest level",
                v_min=1,
                v_max=evals_count,
                v_model=evals_count,
            )
            ui_subtract_ground_switch = v.Switch(
                label="Subtract E\u2080", v_model=True, width=300
            )
            self.ui_level_slider[full_panel_name].observe(
                self.update_plots, names="v_model"
            )
            ui_subtract_ground_switch.observe(self.update_plots, names="v_model")
            return [
                self.ui_level_slider[full_panel_name].widget(),
                ui_subtract_ground_switch,
            ]

        if panel_name == "Wavefunctions":
            if isinstance(subsys, (scq.FluxQubit, scq.ZeroPi, scq.Cos2PhiQubit)):
                ui_wavefunction_selector = ui.InitSelect(
                    label="Display wavefunctions",
                    items=list(range(subsys.truncated_dim)),
                    v_model=0
                )
            else:
                ui_wavefunction_selector = ui.InitSelect(
                    label="Display wavefunctions",
                    multiple=True,
                    items=list(range(subsys.truncated_dim)),
                    v_model=list(range(5))
                )
            ui_mode_dropdown = ui.InitSelect(
                items=mode_dropdown_list,
                v_model=mode_dropdown_list[0],
                label="Plot amplitude as",
            )
            ui_wavefunction_selector.observe(self.update_plots, names="v_model")
            ui_mode_dropdown.observe(self.update_plots, names="v_model")
            return [ui_wavefunction_selector, ui_mode_dropdown]

        if panel_name == "Matrix elements":
            ui_mode_dropdown = ui.InitSelect(
                items=list(zip(mode_dropdown_dict.keys(), mode_dropdown_dict.values())),
                label="Plot matrix elements as",
                v_model="abs",
            )
            ui_matrixscan_toggle = v.Switch(
                items=["fixed", "sweep"], style_="width: 55px", v_model=None
            )
            ui_operator_dropdown = ui.InitSelect(
                items=subsys.get_operator_names(), label="Operator", v_model=None
            )
            ui_mode_dropdown.observe(self.update_plots, names="v_model")
            ui_operator_dropdown.observe(self.update_plots, names="v_model")
            ui_matrixscan_toggle.observe(self.update_layout_and_plots, names="v_model")
            return [ui_mode_dropdown, ui_operator_dropdown, ui_matrixscan_toggle]

        return []

    def build_ui_settings_composite(self, panel_name: str):
        if panel_name == "Transitions":
            self.ui["transitions"]["initial_state_inttexts"] = [
                ui.ValidatedNumberField(
                    label="",
                    num_type=int,
                    v_min=0,
                    v_max=subsys.truncated_dim,
                    v_model=0,
                    width=35,
                )
                for subsys in self.sweep.hilbertspace
            ]

            self.ui["transitions"]["initial_dressed_inttext"] = ui.ValidatedNumberField(
                label="",
                num_type=int,
                v_min=0,
                v_max=self.sweep.hilbertspace.dimension,
                v_model=0,
                width=35,
                disabled=True,
            )

            self.ui["transitions"]["photons_inttext"] = ui.ValidatedNumberField(
                num_type=int, v_model=1, v_min=1, v_max=5, label="", width=35
            )
            self.ui["transitions"]["highlight_selectmultiple"] = ui.InitSelect(
                multiple=True,
                label="",
                items=self.subsys_names,
                v_model=[self.subsys_names[0]],
                width=185,
            )

            self.ui["transitions"]["initial_bare_dressed_toggle"] = v.Switch(
                items=["bare", "dressed"],
                v_model="bare",
                disable=False,
            )
            self.ui["transitions"]["initial_bare_dressed_toggle"].style_ = "width: 45px"

            self.ui["transitions"]["sidebands_switch"] = v.Switch(
                label="show sidebands", v_model=True, width=250
            )
            for inttext in self.ui["transitions"]["initial_state_inttexts"]:
                inttext.observe(self.update_plots, names="num_value")
            self.ui["transitions"]["initial_dressed_inttext"].observe(
                self.update_plots, names="num_value"
            )
            self.ui["transitions"]["photons_inttext"].observe(
                self.update_plots, names="num_value"
            )
            self.ui["transitions"]["highlight_selectmultiple"].observe(
                self.update_plots, names="v_model"
            )
            self.ui["transitions"]["sidebands_switch"].observe(
                self.update_plots, names="v_model"
            )
            self.ui["transitions"]["initial_bare_dressed_toggle"].observe(
                self.bare_dressed_toggle, names="v_model"
            )

            initial_state_selection = v.Container(
                class_="d-flex flex-row justify-end",
                children=[
                    v.Text(children=["Initial state "]),
                    *self.ui["transitions"]["initial_state_inttexts"],
                    self.ui["transitions"]["initial_bare_dressed_toggle"],
                    self.ui["transitions"]["initial_dressed_inttext"],
                ],
                style_="width: 400px",
            )
            photon_options_selection = v.Container(
                class_="d-flex flex-row justify-end",
                children=[
                    v.Text(children=["photons"]),
                    self.ui["transitions"]["photons_inttext"],
                    self.ui["transitions"]["sidebands_switch"],
                ],
                style_="width: 400px",
            )
            transition_highlighting = v.Container(
                class_="d-flex flex-row justify-end",
                children=[
                    v.Text(children=["Highlight:"]),
                    self.ui["transitions"]["highlight_selectmultiple"],
                ],
                style_="width: 400px",
            )

            return [
                initial_state_selection,
                v.Container(
                    class_="d-flex flex-column",
                    children=[photon_options_selection, transition_highlighting],
                ),
            ]
