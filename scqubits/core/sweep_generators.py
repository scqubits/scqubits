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

import scqubits.core.storage as storage
import scqubits.core.sweep_observables as observable
import scqubits.settings as settings
import scqubits.utils.spectrum_utils as spec_utils

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def compute_custom_data_sweep(sweep, func, **kwargs):
    """Method for computing custom data as a function of the external parameter, calculated via the function `func`.

    Parameters
    ----------
    sweep: ParameterSweep
    func: function
        signature: `func(parametersweep, param_value, **kwargs)`, specifies how to calculate the data
    **kwargs: optional
        other parameters to be included in func

    Returns
    -------
    ndarray
    """
    return np.asarray([func(sweep, param_index, **kwargs)
                       for param_index in tqdm(range(sweep.param_count), desc='data sweep', leave=False,
                                               disable=settings.PROGRESSBAR_DISABLED)])


def generate_chi_sweep(sweep):
    """Generate data for the AC Stark shift chi as a function of the sweep parameter

    Parameters
    ----------
    sweep: ParameterSweep

    Returns
    -------
    dict
        (osc_index, qbt_index) -> ndararray of chi values
    """
    osc_subsys_list = sweep.osc_subsys_list
    qbt_subsys_list = sweep.qbt_subsys_list

    data_dict = {}
    for (osc_index, osc_subsys) in osc_subsys_list:
        for (qbt_index, qubit_subsys) in qbt_subsys_list:
            data_dict[(osc_index, qbt_index)] = sweep.new_datastore(
                chi=compute_custom_data_sweep(sweep, observable.dispersive_chi, qubit_subsys=qubit_subsys,
                                              osc_subsys=osc_subsys, chi_indices=(1, 0))
            )
    return data_dict


def generate_charge_matrixelem_sweep(sweep):
    """Generate data for the charge matrix elements as a function of the sweep parameter

    Parameters
    ----------
    sweep: ParameterSweep

    Returns
    -------
    dict
        (osc_index, qbt_index) -> ndararray of chi values
    """
    data_dict = dict()
    for qbt_index, subsys in sweep.qbt_subsys_list:
        if type(subsys).__name__ in ['Transmon', 'Fluxonium']:
            data = compute_custom_data_sweep(sweep, observable.qubit_matrixelement, qubit_subsys=subsys,
                                             qubit_operator=subsys.n_operator())
            datastore = sweep.new_datastore(matrixelem_table=data)
            data_dict[(qbt_index, subsys)] = datastore
    return data_dict


def generate_diffspec_sweep(sweep, initial_state_ind=0):
    """Takes spectral data of energy eigenvalues and subtracts the energy of a select state, given by its state
    index.

    Parameters
    ----------
    sweep: ParameterSweep
    initial_state_ind: int or (i1, i2, ...)
        index of the initial state whose energy is supposed to be subtracted from the spectral data

    Returns
    -------
    SpectrumData
    """
    lookup = sweep.lookup
    param_count = sweep.param_count
    evals_count = sweep.evals_count
    diff_eigenenergy_table = np.empty(shape=(param_count, evals_count))

    for param_index in tqdm(range(param_count), desc="difference spectrum", leave=False,
                            disable=settings.PROGRESSBAR_DISABLED):
        eigenenergies = sweep.dressed_specdata.energy_table[param_index]
        if isinstance(initial_state_ind, int):
            eigenenergy_index = initial_state_ind
        else:
            eigenenergy_index = lookup.dressed_index(initial_state_ind, param_index)
        diff_eigenenergies = eigenenergies - eigenenergies[eigenenergy_index]
        diff_eigenenergy_table[param_index] = diff_eigenenergies
    return storage.SpectrumData(diff_eigenenergy_table, sweep.system_params, sweep.param_name, sweep.param_vals)


def generate_qubit_transitions_sweep(sweep, photonnumber, initial_state_labels):
    """
    Extracts energies for transitions among qubit states only, while all oscillator subsys_list maintain their
    excitation level.

    Parameters
    ----------
    sweep: ParameterSweep
    photonnumber: int
        number of photons used in transition
    initial_state_labels: tuple(int1, int2, ...)
        bare-state labels of the initial state whose energy is supposed to be subtracted from the spectral data

    Returns
    -------
    list, SpectrumData
        list of transition target states, spectrum data
    """
    lookup = sweep.lookup

    target_states_list = spec_utils.generate_target_states_list(sweep, initial_state_labels)
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

    data = np.asarray(difference_energies_table)
    specdata = storage.SpectrumData(data, sweep.system_params, sweep.param_name, sweep.param_vals)
    return target_states_list, specdata
