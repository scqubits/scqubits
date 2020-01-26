# sweep_generators.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np
from tqdm.notebook import tqdm

import scqubits.core.sweep_observables as obs
from scqubits.core.storage import SpectrumData
from scqubits.settings import TQDM_KWARGS


def generate_chi_sweep(sweep):
    """Generate data for the AC Stark shift chi as a function of the sweep parameter"""
    osc_subsys_list = sweep._hilbertspace.osc_subsys_list
    qbt_subsys_list = sweep._hilbertspace.qbt_subsys_list

    for osc_index, osc_subsys in osc_subsys_list:
        for qbt_index, qubit_subsys in qbt_subsys_list:
            sweep.compute_custom_data_sweep('chi_osc{}_qbt{}'.format(osc_index, qbt_index), obs.dispersive_chi,
                                            qubit_subsys=qubit_subsys, osc_subsys=osc_subsys, chi_indices=(1, 0))


def generate_charge_matrixelem_sweep(sweep):
    """Generate data for the charge matrix elements as a function of the sweep parameter"""
    for qbt_index, subsys in sweep._hilbertspace.qbt_subsys_list:
        if type(subsys).__name__ in ['Transmon', 'Fluxonium']:
            sweep.compute_custom_data_sweep('n_op_qbt{}'.format(qbt_index), obs.qubit_matrixelement,
                                            qubit_subsys=subsys, qubit_operator=subsys.n_operator())


# **********************************************************************************************************************


def get_difference_spectrum(sweep, initial_state_ind=0, lookup=None):
    """Takes spectral data of energy eigenvalues and subtracts the energy of a select state, given by its state
    index.

    Parameters
    ----------
    sweep: ParameterSweep
    initial_state_ind: int or (i1, i2, ...)
        index of the initial state whose energy is supposed to be subtracted from the spectral data
    lookup: SpectrumLookup, optional

    Returns
    -------
    SpectrumData object
    """
    lookup = lookup or sweep.lookup
    param_count = sweep.param_count
    evals_count = sweep.evals_count
    diff_eigenenergy_table = np.empty(shape=(param_count, evals_count))

    for param_index in tqdm(range(param_count), desc="difference spectrum", **TQDM_KWARGS):
        eigenenergies = sweep.dressed_specdata.energy_table[param_index]
        if isinstance(initial_state_ind, int):
            eigenenergy_index = initial_state_ind
        else:
            eigenenergy_index = lookup.dressed_index(initial_state_ind, param_index)
        diff_eigenenergies = eigenenergies - eigenenergies[eigenenergy_index]
        diff_eigenenergy_table[param_index] = diff_eigenenergies
    return SpectrumData(diff_eigenenergy_table, sweep._hilbertspace.__dict__, sweep.param_name, sweep.param_vals)


def generate_target_states_list(sweep, initial_state_labels):
    """Based on a bare state label (i1, i2, ...)  with i1 being the excitation level of subsystem 1, i2 the
    excitation level of subsystem 2 etc., generate a list of new bare state labels. These bare state labels
    correspond to target states reached from the given initial one by single-photon qubit transitions. These
    are transitions where one of the qubit excitation levels increases at a time. There are no changes in
    oscillator photon numbers.

    Parameters
    ----------
    sweep: ParameterSweep
    initial_state_labels: tuple(int1, int2, ...)
        bare-state labels of the initial state whose energy is supposed to be subtracted from the spectral data

    Returns
    -------
    list of tuple"""
    target_states_list = []
    for subsys_index, qbt_subsys in sweep._hilbertspace.qbt_subsys_list:   # iterate through qubit subsystems
        initial_qbt_state = initial_state_labels[subsys_index]
        for state_label in range(initial_qbt_state + 1, qbt_subsys.truncated_dim):
            # for given qubit subsystem, generate target labels by increasing that qubit excitation level
            target_labels = list(initial_state_labels)
            target_labels[subsys_index] = state_label
            target_states_list.append(tuple(target_labels))
    return target_states_list


def get_n_photon_qubit_spectrum(sweep, photonnumber, initial_state_labels, lookup=None):
    """
    Extracts energies for transitions among qubit states only, while all oscillator subsystems maintain their
    excitation level.

    Parameters
    ----------
    sweep: ParameterSweep
    photonnumber: int
        number of photons used in transition
    initial_state_labels: tuple(int1, int2, ...)
        bare-state labels of the initial state whose energy is supposed to be subtracted from the spectral data
    lookup: SpectrumLookup, optional

    Returns
    -------
    SpectrumData object
    """
    lookup = lookup or sweep.lookup

    target_states_list = generate_target_states_list(sweep, initial_state_labels)
    difference_energies_table = []

    for param_index in range(sweep.param_count):
        difference_energies = []
        initial_energy = lookup.energy_bare_index(initial_state_labels, param_index)
        for target_labels in target_states_list:
            target_energy = lookup.energy_bare_index(target_labels, param_index)
            if target_energy is None or initial_energy is None:
                difference_energies.append(np.NaN)
            else:
                difference_energies.append((target_energy - initial_energy) / photonnumber)
        difference_energies_table.append(difference_energies)
    return target_states_list, SpectrumData(np.asarray(difference_energies_table), sweep._hilbertspace.__dict__,
                                            sweep.param_name, sweep.param_vals)
