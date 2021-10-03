# explorer_panels.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# arXiv:2107.08552 (2021). https://arxiv.org/abs/2107.08552
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scqubits import settings
from scqubits.core.namedslots_array import NamedSlotsNdarray

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep
    from scqubits.core.qubit_base import QubitBaseClass, QubitBaseClass1d
    from scqubits.core.oscillator import Oscillator

import scqubits.core.units as units


def display_bare_spectrum(
    sweep: "ParameterSweep",
    subsys: Union["QubitBaseClass", "Oscillator"],
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    title = "bare spectrum: {}".format(subsys.id_str)

    fig, axes = sweep["bare_evals"]["subsys":subsys_index].plot(  # type:ignore
        title=title, ylabel="energy [{}]".format(units.get_units()), fig_ax=fig_ax
    )
    axes.axvline(param_val, color="gray", linestyle=":")


def display_anharmonicity(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass1d",
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    title = "anharmonicity: {}".format(subsys.id_str)

    bare_evals = sweep["bare_evals"]["subsys":subsys_index]
    anharmonicity = bare_evals[..., 2] - 2 * bare_evals[..., 1] + bare_evals[..., 0]
    fig, axes = anharmonicity.plot(  # type:ignore
        title=title,
        ylabel="anharmonicity [{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_val, color="gray", linestyle=":")


def display_bare_wavefunctions(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass1d",
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    title = "wavefunctions: {}".format(subsys.id_str)
    evals = sweep["bare_evals"][subsys_index][float(param_val)]
    evecs = sweep["bare_evecs"][subsys_index][float(param_val)]
    settings.DISPATCH_ENABLED = False
    sweep._update_hilbertspace(sweep, param_val)
    settings.DISPATCH_ENABLED = True
    __ = subsys.plot_wavefunction(
        which=-1, esys=(evals, evecs), title=title, fig_ax=fig_ax
    )


def display_dressed_spectrum(
    sweep: "ParameterSweep",
    initial_bare: Tuple[int, ...],
    final_bare: Tuple[int, ...],
    energy_initial: float,
    energy_final: float,
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    energy_difference = energy_final - energy_initial
    title = r"{} $\rightarrow$ {}: {:.4f} {}".format(
        initial_bare, final_bare, energy_difference, units.get_units()
    )
    fig, axes = sweep.plot_transitions(title=title, fig_ax=fig_ax)
    axes.axvline(param_val, color="gray", linestyle=":")
    axes.scatter([param_val] * 2, [energy_initial, energy_final], s=40, c="gray")


def display_n_photon_qubit_transitions(
    sweep: "ParameterSweep",
    photonnumber: int,
    subsys: "QubitBaseClass",
    initial_bare: Tuple[int, ...],
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    title = r"{}-photon {} transitions".format(photonnumber, subsys.id_str)
    fig, axes = sweep[:].plot_transitions(
        subsystems=[subsys],
        initial=initial_bare,
        photon_number=photonnumber,
        title=title,
        fig_ax=fig_ax,
    )
    axes.axvline(param_val, color="gray", linestyle=":")


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
