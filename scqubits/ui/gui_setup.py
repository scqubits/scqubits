# gui_setup.py
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

from pathlib import Path
from typing import Dict, Any

import ipyvuetify as v
import ipywidgets
import numpy as np

import scqubits as scq
import scqubits.core.noise as noise
import scqubits.ui.gui_custom_widgets as ui
from scqubits.core.qubit_base import QubitBaseClass

from scqubits.ui import gui_defaults as gui_defaults, gui_custom_widgets as ui


def init_qubit_dropdown():
    return v.Select(
        v_model=gui_defaults.supported_qubits[0],
        items=gui_defaults.supported_qubits,
        filled=True,
        label="Qubit",
        class_="pl-2",
        style_="min-width: 200px",
    )


def init_plot_choice_buttons():
    return v.BtnToggle(
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
    )


def init_manual_update_switch():
    return v.Switch(
        v_model=False,
        class_="px-4",
        label="Manual refresh",
    )


def init_manual_update_btn():
    return v.Btn(
        children=[v.Icon(children=["mdi-refresh"])],
        fab=True,
        color="orange",
        small=True,
        disabled=True,
        elevation="0",
        class_="px-2",
    )


def init_save_btn():
    return v.Btn(
        class_="ml-5 pmr-2",
        height=40,
        width=40,
        min_width=40,
        children=[v.Icon(children=["mdi-download"])],
    )


def init_manual_update_widget(switch, button):
    return v.Container(
        class_="d-flex flex-row flex-start align-center",
        children=[switch, button],
    )


def flex_row(*widgets):
    return v.Container(
        class_="d-flex flex-row",
        children=widgets,
    )


def flex_column(*widgets):
    return v.Container(
        class_="d-flex flex-column",
        children=widgets,
    )


def init_filename_textfield():
    return v.TextField(
        class_="ml-3 pl-3",
        style_="max-width: 600px",
        v_model=str(Path.cwd().joinpath("plot.pdf")),
        label="Save As",
    )


def init_noise_param_floattextfield(noise_param: str):
    return ui.FloatTextField(
        v_model=noise.NOISE_PARAMS[noise_param],
        name=noise_param,
        label=noise_param,
        step=0.001,
        style_="max-width: 180px",
    )


def init_dict_v_plot_options(
    active_qubit, active_defaults, scan_params
) -> Dict[str, v.VuetifyWidget]:
    """Creates all the widgets that will be used for general plotting options."""
    # current_qubit = self.v_qubit_choice.v_model
    operator_names = active_qubit.get_operator_names()
    noise_channels = active_qubit.supported_noise_channels()

    file = open(active_qubit._image_filename, "rb")
    image = file.read()

    dict_v_plot_options = {
        "info_panel": ipywidgets.Image(
            value=image, format="jpg", layout=ipywidgets.Layout(width="100%")
        ),
        "scan_param": v.Select(
            items=scan_params,
            outlined=True,
            v_model=active_defaults["scan_param"],
            label="Scan over",
        ),
        "amplitude_mode": v.Select(
            v_model=gui_defaults.mode_dropdown_list[0],
            items=gui_defaults.mode_dropdown_list,
            label="Plot as:",
        ),
        "operator_choice": v.Select(
            items=operator_names,
            v_model=active_defaults["operator"],
            label="Operator",
        ),
        "noise_channel_multiselect": v.Select(
            multiple=True,
            dense=True,
            items=noise_channels,
            v_model=noise_channels,
            single_line=True,
            label="Noise channels",
        ),
        "highest_state": v.Slider(
            thumb_label="always",
            thumb_size="24",
            min=1,
            max=10,
            style_="min-width: 200px",
            v_model=5,
            label="Max level",
        ),
        "show_matelem_numbers": v.Switch(v_model=False, label="Show values"),
        "show3d_matelem": v.Switch(v_model=True, label="Show 3D"),
        "subtract_ground": v.Switch(
            v_model=True,
            label="Subtract E\u2080",
        ),
        "i_text": ui.IntTextField(v_model=1, name="i"),
        "j_text": ui.IntTextField(v_model=0, name="j"),
        "t1_checkbox": v.Switch(
            v_model=False,
            label="Effective T1",
        ),
        "t2_checkbox": v.Switch(
            v_model=False,
            label="Effective T2",
        ),
    }

    if active_qubit._sys_type in ("Transmon", "TunableTransmon", "Fluxonium"):
        dict_v_plot_options["manual_wf_scaling"] = v.Switch(
            v_model=False, label="Manual Scaling"
        )
        dict_v_plot_options["multi_state_selector"] = v.Select(
            multiple=True,
            items=list(range(0, 10)),
            v_model=[0, 1, 2, 3, 4],
            label="States",
        )
        dict_v_plot_options["wavefunction_scale_slider"] = v.Slider(
            min=0.1,
            max=4.0,
            step=0.05,
            v_model=active_defaults["scale"],
            disabled=True,
            label="\u03c8 ampl.",
            style_="width: 200px",
        )
    else:
        dict_v_plot_options["wavefunction_state_slider"] = v.Slider(
            min=0,
            max=9,
            v_model=5,
            style_="width: 200px",
            thumb_label="always",
            thumb_size="24",
            label="State no.",
        )

    literature_params_dropdown_list = ["User specified"]
    if active_qubit._sys_type in gui_defaults.paramvals_from_papers.keys():
        literature_params_dropdown_list.extend(
            gui_defaults.paramvals_from_papers[active_qubit._sys_type].keys()
        )
    dict_v_plot_options["literature_params"] = v.Select(
        class_="pl-3 pt-5 mb-0 pb-0",
        height=50,
        label="Select qubit parameters",
        items=literature_params_dropdown_list,
        v_model=literature_params_dropdown_list[0],
    )
    dict_v_plot_options["link_HTML"] = v.Html(
        class_="pl-3 pb-5 pt-0 mt-0", tag="a", children=[""]
    )

    return dict_v_plot_options


def init_dict_v_noise_params(active_qubit) -> Dict[str, v.VuetifyWidget]:
    """Creates all the widgets associated with coherence times plots"""
    dict_v_noise_params = {}
    noise_params = ["T", "omega_low", "omega_high", "t_exp"]
    noise_channels = active_qubit.supported_noise_channels()

    if "tphi_1_over_f_flux" in noise_channels:
        noise_params.append("A_flux")
    if "tphi_1_over_f_cc" in noise_channels:
        noise_params.append("A_cc")
    if "tphi_1_over_f_ng" in noise_channels:
        noise_params.append("A_ng")
    if "t1_charge_impedance" in noise_channels or "t1_flux_bias_line" in noise_channels:
        noise_params.append("R_0")
    if "t1_flux_bias_line" in noise_channels:
        noise_params.append("M")
    if "t1_quasiparticle_tunneling" in noise_channels:
        noise_params.append("x_qp")
        noise_params.append("Delta")

    for noise_param in noise_params:
        dict_v_noise_params[noise_param] = init_noise_param_floattextfield(noise_param)

    return dict_v_noise_params


def init_qubit_params_widgets_dict(
    qubit: QubitBaseClass,
    qubit_params: Dict[str, float],
    defaults: Dict[str, Any],
) -> Dict[str, v.VuetifyWidget]:
    """Creates all the widgets associated with the parameters of the
    chosen qubit.
    """
    dict_v_qubit_params = {}

    for param_name, param_val in qubit_params.items():
        if isinstance(param_val, int):
            kwargs = defaults.get(param_name) or defaults["int"]

            dict_v_qubit_params[param_name] = ui.NumberEntryWidget(
                num_type=int,
                label=f"{param_name}",
                v_model=param_val,
                style_="max-width: 250px",
                **kwargs,
            )
        else:
            kwargs = defaults.get(param_name) or defaults["float"]
            dict_v_qubit_params[param_name] = ui.NumberEntryWidget(
                num_type=float,
                label=f"{param_name}",
                step=0.01,
                v_model=param_val,
                style_="max-width: 250px",
                **kwargs,
            )
    if isinstance(qubit, (scq.ZeroPi, scq.FullZeroPi)):
        grid_min = qubit.grid.min_val
        grid_max = qubit.grid.max_val
        dict_v_qubit_params["grid"] = v.RangeSlider(
            min=-12 * np.pi,
            max=12 * np.pi,
            v_model=[grid_min, grid_max],
            step=0.05,
            thumb_label=True,
            thumb_size="24",
            style_="width: 200px",
            label="Grid range",
        )
    return dict_v_qubit_params


def init_ranges_widgets_dict(
    qubit, dict_v_plot_options, dict_v_qubit_params
) -> Dict[str, Any]:
    """Creates all the widgets associated with changing the ranges of
    certain qubit plot options as well as all of the qubit's parameters.
    """
    dict_v_ranges = {}
    total_dict = {
        **dict_v_plot_options,
        **dict_v_qubit_params,
    }

    for widget_name, widget in total_dict.items():
        if widget_name == "noise_channel_multiselect":
            continue

        widget_min_text = None
        widget_max_text = None

        if isinstance(widget, (v.Slider, ui.NumberEntryWidget)) and isinstance(
            widget.v_model, int
        ):
            widget_min_text = ui.IntTextField(
                v_model=widget.min,
                label="min",
                name="min",
                style_="width: 80px",
                class_="mp-3",
            )
            widget_max_text = ui.IntTextField(
                v_model=widget.max,
                label="max",
                name="max",
                style_="width: 80px",
                class_="px-3",
            )
        elif isinstance(
            widget, (v.Slider, v.RangeSlider, ui.NumberEntryWidget)
        ) and isinstance(widget.v_model, float):
            widget_min_text = ui.FloatTextField(
                v_model=widget.min,
                step=0.01,
                label="min",
                style_="width: 80px",
                class_="px-3",
            )
            widget_max_text = ui.FloatTextField(
                v_model=widget.max,
                step=0.01,
                label="max",
                style_="width: 80px",
                class_="px-3",
            )
        elif isinstance(widget, v.Select) and widget.multiple:
            min_val = widget.items[0]
            max_val = widget.items[-1]

            widget_min_text = ui.IntTextField(
                v_model=min_val,
                name="min",
                label="min",
                style_="width: 80px",
                class_="px-3",
            )
            widget_max_text = ui.IntTextField(
                v_model=max_val,
                name="max",
                label="max",
                style_="width: 80px",
                class_="px-3",
            )
        else:
            continue

        dict_v_ranges[widget_name] = {
            "min": widget_min_text,
            "max": widget_max_text,
        }

    if isinstance(
        qubit,
        (scq.Transmon, scq.TunableTransmon, scq.Fluxonium, scq.FluxQubit),
    ):
        widget_min_text = ui.FloatTextField(
            v_model=qubit._default_grid.min_val,
            label="min",
            step=0.01,
            style_="width: 80px",
            class_="px-3",
        )
        widget_max_text = ui.FloatTextField(
            v_model=qubit._default_grid.max_val,
            label="max",
            step=0.01,
            style_="width: 80px",
            class_="px-3",
        )
        dict_v_ranges["phi"] = {
            "min": widget_min_text,
            "max": widget_max_text,
        }
    elif isinstance(qubit, scq.ZeroPi):
        widget_min_text = ui.FloatTextField(
            v_model=qubit._default_grid.min_val,
            label="min",
            step=0.01,
            style_="width: 80px",
            class_="px-3",
        )
        widget_max_text = ui.FloatTextField(
            v_model=qubit._default_grid.max_val,
            label="max",
            step=0.01,
            style_="width: 80px",
            class_="px-3",
        )
        dict_v_ranges["theta"] = {
            "min": widget_min_text,
            "max": widget_max_text,
        }
    elif isinstance(qubit, scq.Cos2PhiQubit):
        default_grids = {
            "phi": qubit._default_phi_grid,
            "theta": qubit._default_theta_grid,
            "zeta": qubit._default_zeta_grid,
        }
        for param, param_grid in default_grids.items():
            widget_min_text = ui.FloatTextField(
                v_model=param_grid.min_val, label="min", step=0.01
            )
            widget_max_text = ui.FloatTextField(
                v_model=param_grid.max_val, label="max", step=0.01
            )
            dict_v_ranges[param] = {
                "min": widget_min_text,
                "max": widget_max_text,
            }
    return dict_v_ranges
