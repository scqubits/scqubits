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
    title = "bare spectrum: subsystem {} ({})".format(subsys_index, subsys._sys_type)

    fig, axes = sweep["bare_evals"]["subsys":subsys_index].plot(  # type:ignore
        title=title, fig_ax=fig_ax
    )
    axes.axvline(param_val, color="gray", linestyle=":")


def display_bare_wavefunctions(
    sweep: "ParameterSweep",
    subsys: "QubitBaseClass1d",
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    title = "wavefunctions: subsystem {} ({})".format(subsys_index, subsys._sys_type)
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
    title = r"{}-photon qubit transitions".format(photonnumber)
    fig, axes = sweep.plot_transitions(
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


def display_charge_matrixelems(
    sweep: "ParameterSweep",
    initial_bare: Tuple[int, ...],
    subsys_index: int,
    param_val: float,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys = sweep.get_subsys(subsys_index)
    bare_qbt_initial = initial_bare[subsys_index]
    title = r"charge matrix elements for {} [{}]".format(
        type(subsys).__name__, subsys_index
    )
    charge_matrixelems = np.abs(
        sweep["n_operator qubit " + str(subsys_index)][:, bare_qbt_initial, :]
    )
    indices = range(charge_matrixelems.shape[1])
    fig, axes = charge_matrixelems.plot(
        title=title,
        ylabel=r"$|\langle i |n| j \rangle|$",
        label_list=["{},{}".format(ini, fin) for fin in indices for ini in indices],
        fig_ax=fig_ax,
    )
    axes.axvline(param_val, color="gray", linestyle=":")
