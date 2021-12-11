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

from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scqubits import settings
from scqubits.core.namedslots_array import NamedSlotsNdarray

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QubitBaseClass, QubitBaseClass1d, QuantumSystem
    from scqubits.core.oscillator import Oscillator

import scqubits.core.units as units


def display_bare_spectrum(
    sweep: "ParameterSweep",
    subsys: Union["QubitBaseClass", "Oscillator"],
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
    evals_count: int = None,
    subtract_ground: bool = False,
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    title = "bare spectrum: {}".format(subsys.id_str)

    evals_count = evals_count or -1

    bare_evals = sweep["bare_evals"]["subsys":subsys_index][param_slice.fixed]
    if subtract_ground:
        bare_evals = bare_evals - bare_evals[:, 0, np.newaxis]
    fig, axes = bare_evals[:, 0:evals_count].plot(  # type:ignore
        title=title,
        ylabel="energy [{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")


def display_anharmonicity(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass1d",
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)

    bare_evals = sweep["bare_evals"]["subsys":subsys_index][param_slice.fixed]
    anharmonicity = bare_evals[..., 2] - 2 * bare_evals[..., 1] + bare_evals[..., 0]

    title = "anharmonicity: {}".format(subsys.id_str)
    fig, axes = anharmonicity.plot(  # type:ignore
        title=title,
        ylabel="anharmonicity [{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")


def display_bare_wavefunctions(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass1d",
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)

    evals = sweep["bare_evals"][subsys_index][param_slice.all]
    evecs = sweep["bare_evecs"][subsys_index][param_slice.all]

    settings.DISPATCH_ENABLED = False
    sweep._update_hilbertspace(sweep, *param_slice.all_values)
    settings.DISPATCH_ENABLED = True

    title = "wavefunctions: {}".format(subsys.id_str)
    __ = subsys.plot_wavefunction(
        which=-1, esys=(evals, evecs), title=title, fig_ax=fig_ax
    )


def display_dressed_spectrum(
    sweep: "ParameterSweep",
    initial_bare: Tuple[int, ...],
    final_bare: Tuple[int, ...],
    energy_initial: float,
    energy_final: float,
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    energy_difference = energy_final - energy_initial
    title = r"{} $\rightarrow$ {}: {:.4f} {}".format(
        initial_bare, final_bare, energy_difference, units.get_units()
    )
    fig, axes = sweep[param_slice.fixed].plot_transitions(title=title, fig_ax=fig_ax)
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    axes.scatter(
        [param_slice.param_val] * 2, [energy_initial, energy_final], s=40, c="gray"
    )


def display_n_photon_qubit_transitions(
    sweep: "ParameterSweep",
    photonnumber: int,
    subsys: "QubitBaseClass",
    initial_bare: Tuple[int, ...],
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    title = r"{}-photon {} transitions".format(photonnumber, subsys.id_str)
    fig, axes = sweep[param_slice.fixed].plot_transitions(
        subsystems=[subsys],
        initial=initial_bare,
        photon_number=photonnumber,
        title=title,
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")


def display_transitions(
    sweep: "ParameterSweep",
    photonnumber: int,
    subsys_list: List["QuantumSystem"],
    initial_bare: Tuple[int, ...],
    sidebands: bool,
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    if len(subsys_list) == 1:
        title = r"{}-photon {} transitions".format(photonnumber, subsys_list[0].id_str)
    else:
        title = r"{}-photon transitions".format(photonnumber)
    fig, axes = sweep[param_slice.fixed].plot_transitions(
        subsystems=subsys_list,
        initial=initial_bare,
        photon_number=photonnumber,
        title=title,
        sidebands=sidebands,
        coloring=coloring,
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")


def display_chi_01(
    sweep: "ParameterSweep",
    subsys1_index: int,
    subsys2_index: int,
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    chi_data = sweep["chi"][subsys1_index, subsys2_index]
    title = r"$\chi_{01}$" + " = {:.4f}{}".format(
        chi_data[float(param_val), 1], units.get_units()
    )
    fig, axes = chi_data[:, 1].plot(title=title, fig_ax=fig_ax)
    axes.axvline(param_val, color="gray", linestyle=":")


def display_kerrs(
    sweep: "ParameterSweep",
    subsys1_index: int,
    subsys2_index: int,
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    if subsys1_index == subsys2_index:
        return None
    kerr_data = sweep["kerr"][subsys1_index, subsys2_index]
    title = r"Kerr $ij=01,10,11$"

    kerr_datasets = np.asarray(
        [kerr_data[:, 0, 1], kerr_data[:, 1, 0], kerr_data[:, 1, 1]]
    ).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets, kerr_data.param_info)
    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=["01", "10", "11"],
        ylabel=r"Kerr coefficient $\Lambda^{qq'}_{ll'}$"
        + "[{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_val, color="gray", linestyle=":")


def display_kerrlike(
    sweep: "ParameterSweep",
    subsys1_index: int,
    subsys2_index: int,
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys2 = sweep.get_subsys(subsys2_index)

    if subsys2 in sweep.osc_subsys_list:
        display_chi_01(sweep, subsys1_index, subsys2_index, param_val, fig_ax)
    else:
        display_kerrs(sweep, subsys1_index, subsys2_index, param_val, fig_ax)


def display_charge_matrixelems(
    sweep: "ParameterSweep",
    initial_bare: Tuple[int, ...],
    subsys_index: int,
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys = sweep.get_subsys(subsys_index)
    bare_qbt_initial = initial_bare[subsys_index]
    title = r"charge matrix elements for {}".format(subsys.id_str)
    charge_matrixelems = np.abs(
        sweep["n_operator qubit " + str(subsys_index)][:, bare_qbt_initial, :]
    )
    indices = range(charge_matrixelems.shape[1])
    fig, axes = charge_matrixelems.plot(
        title=title,
        ylabel=r"$|\langle i |n| j \rangle|$",
        label_list=["{},{}".format(bare_qbt_initial, fin) for fin in indices],
        fig_ax=fig_ax,
    )
    axes.axvline(param_val, color="gray", linestyle=":")
