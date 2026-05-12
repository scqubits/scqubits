# explorer_panels.py
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

from __future__ import annotations

import itertools

from typing import TYPE_CHECKING

import numpy as np

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits
import scqubits.core.constants
import scqubits.core.units as units
import scqubits.utils.plotting as plot

from scqubits import SpectrumData, settings
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.oscillator import Oscillator
from scqubits.settings import matplotlib_settings
from scqubits.utils.misc import tuple_to_short_str

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QuantumSystem, QubitBaseClass


@rc_context(matplotlib_settings)
def display_anharmonicity(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    subsys_index = sweep.get_subsys_index(subsys)

    bare_evals = sweep["bare_evals"]["subsys":subsys_index][param_slice.fixed]  # type: ignore[misc]
    anharmonicity = bare_evals[..., 2] - 2 * bare_evals[..., 1] + bare_evals[..., 0]

    title = "Anharmonicity: {}".format(subsys.id_str)
    fig, axes = anharmonicity.plot(
        title=title,
        ylabel="anharmonicity [{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


@rc_context(matplotlib_settings)
def display_transitions(
    sweep: "ParameterSweep",
    photon_number: int,
    subsys_list: list["QuantumSystem"],
    initial: int | tuple[int, ...],
    sidebands: bool,
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    if photon_number == 1:
        title = r"Transition Spectrum"
    else:
        title = r"{}-photon Transitions".format(photon_number)
    sliced_sweep = sweep if param_slice.fixed == tuple() else sweep[param_slice.fixed]
    fig, axes = sliced_sweep.plot_transitions(
        subsystems=subsys_list,
        initial=initial,
        photon_number=photon_number,
        title=title,
        sidebands=sidebands,
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")

    info_string = "Highlighted: transitions for "
    for sys in subsys_list:
        info_string += sys.id_str + ", "
    info_string += "with sidebands" if sidebands else "no sidebands"
    axes.text(
        0.5,
        1.05,
        info_string,
        fontsize=9,
        fontweight=300,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=axes.transAxes,
    )

    return fig, axes


@rc_context(matplotlib_settings)
def display_cross_kerr(
    sweep: "ParameterSweep",
    subsys1: "QuantumSystem",
    subsys2: "QuantumSystem",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
    which: int | tuple[int, int] | None = None,
) -> tuple[Figure, Axes]:
    subsys1_index = sweep.get_subsys_index(subsys1)
    subsys2_index = sweep.get_subsys_index(subsys2)
    type_list = [type(sys) for sys in [subsys1, subsys2]]
    label_list: list[int | tuple[int, int]] | list[str]
    if type_list.count(Oscillator) == 1:
        title = f"AC Stark: {subsys1.id_str} + {subsys2.id_str}"
        ylabel = rf"AC Stark Shift $\chi^{{{subsys1_index},{subsys2_index}}}_\ell$"
        level = which or 1
        kerr_data = sweep["chi"][subsys1_index, subsys2_index]
        if param_slice.fixed != tuple():
            kerr_data = kerr_data[param_slice.fixed]
        label_list = [level]
        kerr_datasets = [kerr_data[..., level]]
    elif type_list.count(Oscillator) == 2:
        title = r"Cross-Kerr: {} - {}".format(subsys1.id_str, subsys2.id_str)
        ylabel = r"Kerr Coefficient $K_{{{},{}}}$".format(subsys1_index, subsys2_index)
        level_pairs = [(1, 1)]
        kerr_data = sweep["kerr"][subsys1_index, subsys2_index]
        if param_slice.fixed != tuple():
            kerr_data = kerr_data[param_slice.fixed]
        label_list = []
        kerr_datasets = [
            kerr_data[..., level1, level2] for level1, level2 in level_pairs
        ]
    else:
        title = "Cross-Kerr: {} \u2194 {}".format(subsys1.id_str, subsys2.id_str)
        ylabel = r"Kerr Coefficient $\Lambda^{{{},{}}}_{{ll'}}$".format(
            subsys1_index, subsys2_index
        )
        level_pairs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
        kerr_data = sweep["kerr"][subsys1_index, subsys2_index]
        if param_slice.fixed != tuple():
            kerr_data = kerr_data[param_slice.fixed]
        label_list = [tuple_to_short_str(pair) for pair in level_pairs]
        kerr_datasets = [
            kerr_data[..., level1, level2] for level1, level2 in level_pairs
        ]

    kerr_datasets_arr = np.asarray(kerr_datasets).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets_arr, kerr_data.param_info)
    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=label_list if label_list else None,
        ylabel=ylabel + "[{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


@rc_context(matplotlib_settings)
def display_qubit_self_kerr(
    sweep: "ParameterSweep",
    subsys: "QuantumSystem",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
    which: list[tuple[int, int]] | None = None,
) -> tuple[Figure, Axes]:
    subsys_index = sweep.get_subsys_index(subsys)
    title = r"Self-Kerr: {}".format(subsys.id_str)
    ylabel = r"Kerr coefficient $\Lambda^{{{},{}}}_{{ll}}$".format(
        subsys_index, subsys_index
    )

    kerr_data = sweep["kerr"][subsys_index, subsys_index]
    if param_slice.fixed != tuple():
        kerr_data = kerr_data[param_slice.fixed]

    if not which:
        level_pairs = list(itertools.combinations(list(range(subsys.truncated_dim)), 2))
    else:
        level_pairs = which

    kerr_datasets = []
    for pair in level_pairs:
        kerr_datasets.append(kerr_data[..., pair[0], pair[1]])
    kerr_datasets_arr = np.asarray(kerr_datasets).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets_arr, kerr_data.param_info)

    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=level_pairs,
        ylabel=ylabel + "[{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


@rc_context(matplotlib_settings)
def display_self_kerr(
    sweep: "ParameterSweep",
    subsys: scqubits.Oscillator | scqubits.KerrOscillator,
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    subsys_index = sweep.get_subsys_index(subsys)
    title = r"Self-Kerr: {}".format(subsys.id_str)

    ylabel = "Kerr coefficient $K_{{{}}}$".format(subsys_index)
    kerr_data = sweep["kerr"][subsys_index, subsys_index]
    if param_slice.fixed != tuple():
        kerr_data = kerr_data[param_slice.fixed]

    fig, axes = kerr_data.plot(
        title=title,
        ylabel=ylabel + "[{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes
