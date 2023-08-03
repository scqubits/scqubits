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

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits as scq

import scqubits.ui.gui_custom_widgets as ui
import scqubits.ui.gui_defaults as gui_defaults
from scqubits.core.param_sweep import ParameterSlice
from scqubits.core.qubit_base import QuantumSystem, QubitBaseClass
from scqubits.explorer import explorer_panels as panels
from scqubits.explorer.explorer_settings import ExplorerSettings
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import (
    NAV_COLOR,
    PlotType,
    default_panels,
    mode_dropdown_dict,
    supported_panels,
)

from scqubits.utils import misc as utils
from scqubits.utils.misc import _HAS_WIDGET_BACKEND

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep

try:
    from IPython.display import HTML, display, notebook
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

try:
    import ipyvuetify as v
    import ipywidgets
    from scqubits.ui.gui_custom_widgets import flex_row
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True


class PlotID:
    """Class for storing plot identifiers. Used for plot panel selection."""

    SEP = " | "

    def __init__(self, plot_type: PlotType, subsystems: List[QuantumSystem]):
        self.plot_type: PlotType = plot_type
        self.subsystems: List[QuantumSystem] = subsystems

    def __repr__(self):
        return f"PlotID({self.plot_type}, {self.subsystems})"

    def __str__(self):
        subsys_names = self.subsys_ids()
        if len(subsys_names) == 1:
            return f"{self.plot_type.value}{self.SEP}{subsys_names[0]}"
        return f"{self.plot_type.value}{self.SEP}{','.join(subsys_names)}"

    def is_composite(self) -> bool:
        return len(self.subsystems) > 1

    def subsys_ids(self) -> List[str]:
        return [subsys.id_str for subsys in self.subsystems]

    def is_default_active(self) -> bool:
        if self.is_composite():
            return self.plot_type in default_panels["Composite"]
        subsys_type_str = type(self.subsystems[0]).__name__
        return self.plot_type in default_panels[subsys_type_str]


class Explorer:
    """
    Generates the UI for exploring `ParameterSweep` objects.

    Parameters
    ----------
    sweep:
        the `ParameterSweep` object to be visualized.
    ncols:
        number of columns used in plot panel display

    Attributes
    ----------
    sweep:
        the `ParameterSweep` object to be visualized.
    ncols:
        number of columns used in plot panel display
    subsystems:
        list of subsystems in the `ParameterSweep` object
    subsys_names:
        list of subsystem names
    ui:
        dictionary of all UI elements
    settings: ExplorerSettings
        settings for the Explorer

    Plot panels are labeled by PlotID instances where supported plot names are
    - "Energy spectrum"
    - "Wavefunctions" (only for subsystems who inherit from `QubitBaseClass`)
    - "Matrix elements"
    - "Anharmonicity" (for subsystems who inherit from `QubitBaseClass`)
    - "Transitions"
    - "Self-Kerr"
    - "Cross-Kerr"
    - "ac Stark"
    """

    @utils.Required(ipyvuetify=_HAS_IPYVUETIFY)
    def __init__(self, sweep: scq.ParameterSweep, ncols: int = 2):
        """Set up all widget GUI elements and class attributes."""
        self.sweep = sweep
        self.ncols = ncols  # number of columns used for axes in the figure display

        self.subsystems: List[QuantumSystem] = self.sweep.hilbertspace.subsystem_list
        self.subsys_names: List[str] = [subsys.id_str for subsys in self.subsystems]

        utils.check_matplotlib_compatibility()

        # == GUI elements =========================================================
        self.ui: Dict[str, Any] = {}
        self.build_panel_switches()
        self.ui["add_plot_dialog"] = self.build_ui_add_plot_dialog()

        self.ui["sweep_param_dropdown"] = ui.InitializedSelect(
            class_="px-2",
            style_="max-width: 200px;",
            label="Active Sweep Parameter",
            items=list(self.sweep.param_info.keys()),
        )
        self.ui["sweep_param_dropdown"].observe(
            self.update_parameter_sliders, names="v_model"
        )

        # self.ui["sweep_value_slider"] = ui.DiscreteSetSlider(
        #     param_name=self.ui["sweep_param_dropdown"].v_model,
        #     param_vals=self.param_vals,
        #     filled=False,
        #     class_="px-3",
        #     style_="max-width: 300px; padding-top: 10px",
        # )
        #
        #
        # self.ui["sweep_value_slider"].observe(self.update_plots, names="v_model")

        self.ui["param_sliders"] = self.create_sliders()
        self.update_parameter_sliders(None)
        self.ui["top_bar"] = v.Sheet(
            class_="d-flex flex-row m-0 pt-3 align-bottom",
            height=70,
            flat=True,
            width="100%",
            color=NAV_COLOR,
            children=[
                v.Card(
                    class_="p-2 mx-4",
                    color=NAV_COLOR,
                    elevation=0,
                    children=[gui_defaults.icons["scq-logo.png"]],
                ),
                self.ui["sweep_param_dropdown"],
                self.ui["param_sliders_container"],
            ],
        )

        self.plot_collection = ui.PlotPanelCollection(
            ncols=self.ncols,
            plot_choice_dialog=self.show_plot_choice_dialog,
            toggle_switches_by_plot_id=self.ui["panel_switch_by_plot_id"],
            plot_settings_dialog=self.plot_settings_dialog,
        )

        self.settings = ExplorerSettings(self)

        self.explorer_display = v.Container(
            class_="d-flex flex-column mx-0 px-0",
            children=[
                self.ui["top_bar"],
                self.plot_collection.show(),
                self.ui["add_plot_dialog"],
                *self.settings["dialogs"].values(),
            ],
        )

        self.create_initial_plot_panels()
        display(self.explorer_display)

    @property
    def fixed_param(self):
        """Return the currently selected sweep parameter."""
        return self.ui["sweep_param_dropdown"].v_model

    @property
    def param_vals(self):
        """Return the selected value of the current sweep parameter."""
        return self.sweep.param_info[self.fixed_param]

    def show_plot_choice_dialog(self, *args) -> None:
        self.ui["add_plot_dialog"].v_model = True

    def close_plot_choice_dialog(self, *args) -> None:
        self.ui["add_plot_dialog"].v_model = False

    def plot_settings_dialog(self, plot_id: PlotID) -> None:
        self.settings["dialogs"][plot_id].v_model = True

    def update_switches(self) -> None:
        for switch in self.ui["panel_switch_by_plot_id"].values():
            switch.v_model = switch.ref in self.plot_collection.id_list()

    def build_panel_switches(self) -> None:
        # The panel switches reflect the set of panels that are currently displayed
        ui_panel_switch_by_plot_id: Dict[PlotID, ui.LinkedSwitch] = {}
        ui_panel_switches_by_subsys_name: Dict[str, List[ui.LinkedSwitch]] = {
            "Composite": []
        }

        # Add default panels for individual subsystems

        for subsys in self.subsystems:
            ui_panel_switches_by_subsys_name[subsys.id_str] = []
            subsys_type_str = type(subsys).__name__
            for plot_type in supported_panels[subsys_type_str]:
                plot_id = PlotID(plot_type, [subsys])
                ui_panel_switch_by_plot_id[plot_id] = ui.LinkedSwitch(
                    v_model=plot_id.is_default_active(),
                    label=plot_type.value,
                    ref=plot_id,
                    dense=True,
                    width=185,
                )
                ui_panel_switches_by_subsys_name[subsys.id_str] += [
                    ui_panel_switch_by_plot_id[plot_id]
                ]

        # Add composite panels

        plot_id = PlotID(PlotType.TRANSITIONS, self.subsystems)
        ui_panel_switch_by_plot_id[plot_id] = ui.LinkedSwitch(
            v_model=plot_id.is_default_active(),
            label="Transitions",
            ref=plot_id,
            dense=True,
            width=185,
        )
        ui_panel_switches_by_subsys_name["Composite"] += [
            ui_panel_switch_by_plot_id[plot_id]
        ]

        for subsys1, subsys2 in itertools.combinations(self.subsystems, 2):
            is_oscillator1 = isinstance(subsys1, scq.Oscillator)
            is_oscillator2 = isinstance(subsys2, scq.Oscillator)
            if any([is_oscillator1, is_oscillator2]) and not all(
                [is_oscillator1, is_oscillator2]
            ):
                plot_type = PlotType.AC_STARK
            else:
                plot_type = PlotType.CROSS_KERR

            plot_id = PlotID(plot_type, [subsys1, subsys2])

            ui_panel_switch_by_plot_id[plot_id] = ui.LinkedSwitch(
                v_model=plot_id.is_default_active(),
                label=f"{plot_type.value}: {subsys1.id_str}, {subsys2.id_str}",
                ref=plot_id,
                dense=True,
                width=185,
            )
            ui_panel_switches_by_subsys_name["Composite"] += [
                ui_panel_switch_by_plot_id[plot_id]
            ]

        self.ui["panel_switch_by_plot_id"] = ui_panel_switch_by_plot_id
        self.ui["panel_switches_by_subsys_name"] = ui_panel_switches_by_subsys_name

        self.ui["panel_switches"] = {}
        for subsys_name in self.ui["panel_switches_by_subsys_name"].keys():
            self.ui["panel_switches"][subsys_name] = v.Container(
                class_="d-flex flex-column",
                dense=True,
                children=self.ui["panel_switches_by_subsys_name"][subsys_name],
            )

        for switch in self.ui["panel_switch_by_plot_id"].values():
            switch.observe(self.on_toggle_event, names="v_model")

    def build_ui_add_plot_dialog(self) -> v.Dialog:
        return v.Dialog(
            v_model=False,
            width="75%",
            children=[
                v.Card(
                    children=[
                        v.Toolbar(
                            children=[v.ToolbarTitle(children=["Choose Plot"])],
                            color="deep-purple accent-4",
                            dark=True,
                        ),
                        ui.flex_row(
                            [
                                ui.flex_column(
                                    [
                                        v.CardTitle(children="Composite-system plots"),
                                        self.ui["panel_switches"]["Composite"],
                                    ]
                                ),
                                v.Divider(vertical=True),
                                ui.flex_column(
                                    [
                                        v.CardTitle(
                                            children="Subsystem-specific plots"
                                        ),
                                        ui.flex_row(
                                            [
                                                ui.flex_column(
                                                    [
                                                        v.CardTitle(
                                                            style_="font-weight: normal;",
                                                            children=subsys_name,
                                                        ),
                                                        self.ui["panel_switches"][
                                                            subsys_name
                                                        ],
                                                    ]
                                                )
                                                for subsys_name in self.subsys_names
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        v.CardActions(
                            class_="justify-center",
                            children=[
                                ui.LinkedButton(
                                    onclick=self.close_plot_choice_dialog,
                                    children=["FINISH"],
                                )
                            ],
                        ),
                    ]
                )
            ],
        )

    @matplotlib.rc_context(matplotlib_settings)
    def build_panel(
        self,
        plot_id: PlotID,
        param_slice: ParameterSlice,
        fig_ax: Tuple[Figure, Axes],
    ):
        if plot_id.plot_type is PlotType.ENERGY_SPECTRUM:
            panel_widget = self.settings[plot_id]
            return panels.display_bare_spectrum(
                self.sweep,
                plot_id.subsystems[0],
                param_slice,
                fig_ax,
                subtract_ground=panel_widget[1].v_model,
                evals_count=self.settings["level_slider"][plot_id].num_value,
            )
        elif plot_id.plot_type is PlotType.WAVEFUNCTIONS and isinstance(
            plot_id.subsystems[0], QubitBaseClass
        ):
            ui_wavefunction_selector, ui_mode_dropdown = self.settings[plot_id]
            return panels.display_bare_wavefunctions(
                self.sweep,
                plot_id.subsystems[0],
                param_slice,
                fig_ax,
                mode=mode_dropdown_dict[ui_mode_dropdown.v_model],
                which=ui_wavefunction_selector.v_model,
            )
        elif plot_id.plot_type is PlotType.MATRIX_ELEMENTS and isinstance(
            plot_id.subsystems[0], QubitBaseClass
        ):
            ui_mode_dropdown, opname_dropdown = self.settings[plot_id]
            return panels.display_matrixelements(
                sweep=self.sweep,
                subsys=plot_id.subsystems[0],
                operator_name=opname_dropdown.v_model,
                mode_str=mode_dropdown_dict[ui_mode_dropdown.v_model],
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        elif plot_id.plot_type is PlotType.MATRIX_ELEMENT_SCAN and isinstance(
            plot_id.subsystems[0], QubitBaseClass
        ):
            ui_mode_dropdown, opname_dropdown = self.settings[plot_id]
            return panels.display_matrixelement_sweep(
                sweep=self.sweep,
                subsys=plot_id.subsystems[0],
                operator_name=opname_dropdown.v_model,
                mode_str=mode_dropdown_dict[ui_mode_dropdown.v_model],
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        elif plot_id.plot_type is PlotType.ANHARMONICITY:
            return panels.display_anharmonicity(
                self.sweep, plot_id.subsystems[0], param_slice, fig_ax
            )
        elif plot_id.plot_type is PlotType.TRANSITIONS:
            if (
                self.settings["Transitions"]["initial_bare_dressed_toggle"].v_model
                == "bare"
            ):
                initial_state = tuple(
                    int(inttext.v_model)
                    for inttext in self.settings["Transitions"][
                        "initial_state_inttexts"
                    ]
                )
            else:
                initial_state = int(
                    self.settings["Transitions"]["initial_dressed_inttext"].v_model
                )

            subsys_name_tuple = self.settings["Transitions"][
                "highlight_selectmultiple"
            ].v_model
            if subsys_name_tuple == ():
                subsys_list = None
            else:
                subsys_list = [
                    self.sweep.subsys_by_id_str(subsys_name)
                    for subsys_name in subsys_name_tuple
                ]

            sidebands = self.settings["Transitions"]["sidebands_switch"].v_model
            photon_number = int(self.settings["Transitions"]["photons_inttext"].v_model)
            return panels.display_transitions(
                self.sweep,
                photon_number,
                subsys_list,
                initial_state,
                sidebands,
                param_slice,
                fig_ax,
            )
        elif plot_id.plot_type is PlotType.SELF_KERR:
            if self.settings[plot_id]:  # has settings, so must be qubit-mode self-Kerr
                ui_state_selection = self.settings[plot_id][0]
                which = ui_state_selection.v_model
                return panels.display_qubit_self_kerr(
                    sweep=self.sweep,
                    subsys=plot_id.subsystems[0],
                    param_slice=param_slice,
                    fig_ax=fig_ax,
                    which=which,
                )

            return panels.display_self_kerr(
                sweep=self.sweep,
                subsys=plot_id.subsystems[0],
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        elif plot_id.plot_type is PlotType.CROSS_KERR:
            return panels.display_cross_kerr(
                sweep=self.sweep,
                subsys1=plot_id.subsystems[0],
                subsys2=plot_id.subsystems[1],
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        elif plot_id.plot_type is PlotType.AC_STARK:
            return panels.display_cross_kerr(
                sweep=self.sweep,
                subsys1=plot_id.subsystems[0],
                subsys2=plot_id.subsystems[1],
                param_slice=param_slice,
                fig_ax=fig_ax,
                which=self.settings.ui["kerr"]["ac_stark_ell"].v_model,
            )
        raise NotImplementedError(f"Plot type {plot_id} not implemented.")

    @property
    def active_switches_by_plot_id(self) -> Dict[PlotID, "ui.LinkedSwitch"]:
        """Returns a dictionary labeling all selected switches by their plot_id names."""
        return {
            plot_id: switch
            for plot_id, switch in self.ui["panel_switch_by_plot_id"].items()
            if switch.v_model
        }

    @property
    def selected_plot_id_list(self) -> List[PlotID]:
        """Returns a list of strings capturing the names of all panels selected via
        the switches."""
        return list(self.active_switches_by_plot_id.keys())

    def create_sliders(self) -> Dict[str, "v.VuetifyWidget"]:
        """Returns a list of selection sliders, one for each parameter that is part
        of the underlying ParameterSweep object."""
        slider_by_name = {
            param_name: ui.DiscreteSetSlider(
                param_name=param_name,
                param_vals=param_array.tolist(),
                filled=False,
                class_="px-3",
                style_="max-width: 300px; padding-top: 10px",
            )
            for param_name, param_array in self.sweep.param_info.items()
        }
        for slider in slider_by_name.values():
            slider.observe(self.update_plots, names="v_model")
        return slider_by_name

    @property
    def fixed_params(self) -> Dict[str, float]:
        sliders = self.ui["fixed_param_sliders"]
        return {
            param_name: slider.current_value() for param_name, slider in sliders.items()
        }

    @property
    def axes_list(self) -> List[Axes]:
        return self.plot_collection.axes_list()

    @matplotlib.rc_context(matplotlib_settings)
    def create_initial_plot_panels(self):
        for plot_id in self.selected_plot_id_list:
            param_slice = self.parameter_slice
            fig = matplotlib.pyplot.figure()
            axes = fig.subplots()
            fig.set_figwidth(4)
            fig.set_figheight(4)
            fig, axes = self.build_panel(
                plot_id, param_slice=param_slice, fig_ax=(fig, axes)
            )
            self.plot_collection.new_plot_panel(plot_id, fig, axes)

    @matplotlib.rc_context(matplotlib_settings)
    def on_toggle_event(self, change):
        toggled_panel_id = change["owner"].ref
        if change["new"]:
            param_slice = self.parameter_slice

            fig = matplotlib.pyplot.figure()
            axes = fig.subplots()
            fig.set_figwidth(4)
            fig.set_figheight(4)
            fig, axes = self.build_panel(
                toggled_panel_id, param_slice=param_slice, fig_ax=(fig, axes)
            )
            self.plot_collection.new_plot_panel(toggled_panel_id, fig, axes)

        else:
            self.plot_collection.close_panel_by_id(toggled_panel_id)

    def update_parameter_sliders(self, change):
        current_sweep_param = self.ui["sweep_param_dropdown"].v_model
        self.ui["fixed_param_sliders"] = self.ui["param_sliders"].copy()
        self.ui["fixed_param_sliders"].pop(current_sweep_param)

        self.ui["sweep_value_slider"] = self.ui["param_sliders"][current_sweep_param]

        self.ui["param_sliders_container"] = ui.flex_row(
            [
                self.ui["sweep_value_slider"],
                v.Text(children="Fixed:" if self.ui["fixed_param_sliders"] else ""),
                *self.ui["fixed_param_sliders"].values(),
            ]
        )

        if "top_bar" in self.ui:
            self.ui["top_bar"].children = self.ui["top_bar"].children[:-1] + [
                self.ui["param_sliders_container"]
            ]
            self.update_plots(None)

    def bare_dressed_toggle(self, change):
        if (
            self.settings.ui["Transitions"]["initial_bare_dressed_toggle"].v_model
            == "bare"
        ):
            self.settings.ui["Transitions"][
                "initial_dressed_inttext"
            ].style_ = "display: none; max-width: 100px;"
            for inttext in self.settings.ui["Transitions"]["initial_state_inttexts"]:
                inttext.style_ = "display: inline-block; max-width: 100px;"
        else:
            self.settings.ui["Transitions"][
                "initial_dressed_inttext"
            ].style_ = "display: inherit; max-width: 100px;"
            for inttext in self.settings.ui["Transitions"]["initial_state_inttexts"]:
                inttext.style_ = "display: none; max-width: 80px;"
        self.update_plots(change)

    @property
    def parameter_slice(self):
        return ParameterSlice(
            self.ui["sweep_param_dropdown"].v_model,
            self.ui["sweep_value_slider"].current_value(),
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

    @matplotlib.rc_context(matplotlib_settings)
    def update_plots(self: "Explorer", change):
        param_val = self.ui["sweep_value_slider"].current_value()
        panel_ids = self.selected_plot_id_list

        param_slice = ParameterSlice(
            self.ui["sweep_param_dropdown"].v_model,
            param_val,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

        for axes in self.axes_list:
            for item in axes.lines + axes.collections + axes.texts:
                item.remove()
            axes.set_prop_cycle(None)
            axes.relim()
            axes.autoscale_view()

        for index, panel_id in enumerate(panel_ids):
            fig = self.plot_collection.panel_by_id[panel_id].fig
            ax = self.plot_collection.panel_by_id[panel_id].axes
            output_widget = self.plot_collection.panel_by_id[panel_id].output
            self.build_panel(panel_id, param_slice=param_slice, fig_ax=(fig, ax))
            ax.title.set_text("")  # title is instead displayed in card header
            if not _HAS_WIDGET_BACKEND:
                with output_widget:
                    output_widget.clear_output(wait=True)
                    display(fig)
            else:
                fig.canvas.draw_idle()
