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

from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.core.units as units
import scqubits.utils.plotting as plot

from scqubits import SpectrumData, settings
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.oscillator import Oscillator
from scqubits.utils.misc import tuple_to_short_str

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QubitBaseClass, QubitBaseClass1d, QuantumSystem


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


def display_matrixelements(
    sweep: "ParameterSweep",
    operator_name: str,
    subsys: "QubitBaseClass1d",
    param_slice: "ParameterSlice",
    mode_str: str,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    evecs = sweep["bare_evecs"][subsys_index][param_slice.all]

    fig, axes = fig_ax
    axes.cla()

    title = "{}: {}".format(subsys.id_str, operator_name)
    fig, axes = subsys.plot_matrixelements(
        operator_name,
        evecs,
        evals_count=subsys.truncated_dim,
        mode=mode_str,
        show3d=False,
        show_numbers=True,
        show_colorbar=False,
        fig_ax=fig_ax,
        title=title,
    )


def display_matrixelement_sweep(
    sweep: "ParameterSweep",
    operator_name: str,
    subsys: "QubitBaseClass1d",
    param_slice: "ParameterSlice",
    mode_str: str,
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    evals = sweep["bare_evals"][subsys_index][param_slice.fixed]
    evecs = sweep["bare_evecs"][subsys_index][param_slice.fixed]

    fig, axes = fig_ax
    axes.cla()

    param_name = param_slice.param_name
    param_vals = sweep.param_info[param_name]
    paramvals_count = len(param_vals)

    specdata = SpectrumData(
        evals,
        {},
        param_slice.param_name,
        param_vals,
        state_table=evecs,
    )

    matelem_table = np.empty(
        shape=(paramvals_count, subsys.truncated_dim, subsys.truncated_dim),
        dtype=np.complex_,
    )
    for index, paramval in enumerate(param_vals):
        evecs = specdata.state_table[index]
        matelem_table[index] = subsys.matrixelement_table(
            operator_name, evecs=evecs, evals_count=subsys.truncated_dim
        )

    specdata.matrixelem_table = matelem_table

    title = "{}: {}".format(subsys.id_str, operator_name)
    fig, axes = plot.matelem_vs_paramvals(
        specdata, mode=mode_str, fig_ax=fig_ax, title=title
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


def display_transitions(
    sweep: "ParameterSweep",
    photon_number: int,
    subsys_list: List["QuantumSystem"],
    initial: Union[int, Tuple[int, ...]],
    sidebands: bool,
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    if len(subsys_list) == 1:
        title = r"{}-photon {} transitions".format(photon_number, subsys_list[0].id_str)
    else:
        title = r"{}-photon transitions".format(photon_number)
    fig, axes = sweep[param_slice.fixed].plot_transitions(
        subsystems=subsys_list,
        initial=initial,
        photon_number=photon_number,
        title=title,
        sidebands=sidebands,
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")


def display_cross_kerr(
    sweep: "ParameterSweep",
    subsys1: "QuantumSystem",
    subsys2: "QuantumSystem",
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys1_index = sweep.get_subsys_index(subsys1)
    subsys2_index = sweep.get_subsys_index(subsys2)
    type_list = [type(sys) for sys in [subsys1, subsys2]]
    if type_list.count(Oscillator) == 1:
        title = r"ac Stark: {} <-> {}".format(subsys1.id_str, subsys2.id_str)
        ylabel = r"ac Stark shift $\chi^{{{},{}}}_l$".format(
            subsys1_index, subsys2_index
        )
        level_pairs = [(1, 1), (2, 2)]
        kerr_data = sweep["chi"][subsys1_index, subsys2_index][param_slice.fixed]
        label_list = [tuple_to_short_str(pair) for pair in level_pairs]
    elif type_list.count(Oscillator) == 2:
        title = r"cross-Kerr: {} <-> {}".format(subsys1.id_str, subsys2.id_str)
        ylabel = r"Kerr coefficient $K_{{},{}}$".format(subsys1_index, subsys2_index)
        level_pairs = [(1, 1)]
        kerr_data = sweep["kerr"][subsys1_index, subsys2_index][param_slice.fixed]
        label_list = []
    else:
        title = "cross-Kerr: {} \u2194 {}".format(subsys1.id_str, subsys2.id_str)
        ylabel = r"Kerr coefficient $\Lambda^{{{},{}}}_{{ll'}}$".format(
            subsys1_index, subsys2_index
        )
        level_pairs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
        kerr_data = sweep["kerr"][subsys1_index, subsys2_index][param_slice.fixed]
        label_list = [tuple_to_short_str(pair) for pair in level_pairs]

    kerr_datasets = []
    for level1, level2 in level_pairs:
        kerr_datasets.append(kerr_data[..., level1, level2])

    kerr_datasets = np.asarray(kerr_datasets).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets, kerr_data.param_info)
    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=label_list if label_list else None,
        ylabel=ylabel + "[{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")


def display_self_kerr(
    sweep: "ParameterSweep",
    subsys: "QuantumSystem",
    param_slice: "ParameterSlice",
    fig_ax: Tuple[Figure, Axes],
) -> None:
    subsys_index = sweep.get_subsys_index(subsys)
    title = r"self-Kerr: {}".format(subsys.id_str)
    if isinstance(subsys, Oscillator):
        ylabel = "Kerr coefficient $K_{{{}}}$".format(subsys_index)
    else:
        ylabel = r"Kerr coefficient $\Lambda^{{{},{}}}_{{ll'}}$".format(
            subsys_index, subsys_index
        )

    kerr_data = sweep["kerr"][subsys_index, subsys_index][param_slice.fixed]

    level_pairs = [(1, 1), (2, 2)]

    kerr_datasets = []
    for level1, level2 in level_pairs:
        kerr_datasets.append(kerr_data[..., level1, level2])
    kerr_datasets = np.asarray(kerr_datasets).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets, kerr_data.param_info)

    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=["11", "22"],
        ylabel=ylabel + "[{}]".format(units.get_units()),
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
