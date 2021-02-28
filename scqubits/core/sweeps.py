# sweeps.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import functools
import itertools

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import scqubits.core.storage as storage
import scqubits.core.sweep_observables as observable
import scqubits.settings as settings
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def generator(sweep: "ParameterSweepBase", func: callable, **kwargs) -> np.ndarray:
    """Method for computing custom data as a function of the external parameter,
    calculated via the function `func`.

    Parameters
    ----------
    sweep:
        ParameterSweep object containing HilbertSpace and spectral information
    func:
        signature: `func(parametersweep, [paramindex_tuple, paramvals_tuple,
        **kwargs])`, specifies how to calculate the data for a single choice of
        parameter(s)
    **kwargs:
        keyword arguments to be included in func

    Returns
    -------
        array of custom data
    """
    reduced_parameters = sweep.parameters.create_sliced(sweep._current_param_indices,
                                                        remove_fixed=False)
    total_count = np.prod(reduced_parameters.counts)

    def func_effective(paramindex_tuple: Tuple[int], params, **kw) -> Any:
        paramvals_tuple = params[paramindex_tuple]
        return func(sweep, paramindex_tuple=paramindex_tuple,
                    paramvals_tuple=paramvals_tuple, **kw)

    data_array = list(
        tqdm(
            map(
                functools.partial(
                    func_effective,
                    params=reduced_parameters,
                    **kwargs,
                ),
                itertools.product(*reduced_parameters.ranges),
            ),
            total=total_count,
            desc="sweeping " + func.__name__,
            leave=False,
            disable=settings.PROGRESSBAR_DISABLED,
        )
    )
    return np.asarray(data_array)


def dispersive(sweep: "ParameterSweep"):
    """
    In the dispersive limit, a system of oscillators and qubits has the effective
    Hamiltonian (to leading order in any coupling)
    :math:`H = \sum_s (\omega_s + \sum_q\chi^{sq}_0)a_s^\dag a_s
    + \sum_q \sum_l (\epsilon^q_l + \Delta\epsilon^q_l) |l_q\rangle \langle l_q|
    + \sum_s \sum_{q,l} \chi^{sq}_l a_s^\dag a_s |l_q\rangle \langle l_q|
    + \sum_s \frac{1}{2}\kappa_s (a_s^\dag a_s -1)a_s^\dag a_s
    + \sum_{s\not=s'} k_{ss'} a_s^\dag a_s (a_{s'}^\dag a_{s'}
    + \sum_{q\not=q'}\sum_{l,l'} \Lambda^{qq'}_{ll'} |l_q l'_{q'}\rangle \langle l_q
    l'_{q'}\rangle`.
    Using dressed eigenenergies labeled by the bare product states,
    :math:`E(n_1, n_2, \ldots; l_1, l_2, \lddots)`. Here, dispersive shifts are
    calcultated from exact eigenvalues vie
    :math:`\chi^{(s,q)}_l = E(`
    """
    return None


# def generate_chi_sweep(sweep):
#     """Generate data for the AC Stark shift chi as a function of the sweep parameter
#
#     Parameters
#     ----------
#     sweep: ParameterSweep
#
#     Returns
#     -------
#     dict
#         (osc_index, qbt_index) -> ndararray of chi values
#     """
#     osc_subsys_list = sweep.osc_subsys_list
#     qbt_subsys_list = sweep.qbt_subsys_list
#
#     data_dict = {}
#     for (osc_index, osc_subsys) in osc_subsys_list:
#         for (qbt_index, qubit_subsys) in qbt_subsys_list:
#             data_dict[(osc_index, qbt_index)] = sweep.new_datastore(
#                 chi=compute_custom_data_sweep(
#                     sweep,
#                     observable.dispersive_chi,
#                     qubit_subsys=qubit_subsys,
#                     osc_subsys=osc_subsys,
#                     chi_indices=(1, 0),
#                 )
#             )
#     return data_dict
#
#
# def chi_dispersive(sweep):
#     """Generate data for the AC Stark shift chi as a function of the sweep parameter
#
#     Parameters
#     ----------
#     sweep: ParameterSweep
#
#     Returns
#     -------
#     dict
#         (osc_index, qbt_index) -> ndararray of chi values
#     """
#     osc_subsys_list = sweep.osc_subsys_list
#     qbt_subsys_list = sweep.qbt_subsys_list
#
#     data_dict = {}
#     for (osc_index, osc_subsys) in osc_subsys_list:
#         for (qbt_index, qubit_subsys) in qbt_subsys_list:
#             data_dict[(osc_index, qbt_index)] = sweep.new_datastore(
#                 chi=compute_custom_data_sweep(
#                     sweep,
#                     observable.dispersive_chi,
#                     qubit_subsys=qubit_subsys,
#                     osc_subsys=osc_subsys,
#                     chi_indices=(1, 0),
#                 )
#             )
#     return data_dict
#
#
# def generate_charge_matrixelem_sweep(sweep):
#     """Generate data for the charge matrix elements as a function of the sweep parameter
#
#     Parameters
#     ----------
#     sweep: ParameterSweep
#
#     Returns
#     -------
#     dict
#         (osc_index, qbt_index) -> ndararray of chi values
#     """
#     data_dict = dict()
#     for qbt_index, subsys in sweep.qbt_subsys_list:
#         if type(subsys).__name__ in ["Transmon", "Fluxonium"]:
#             data = compute_custom_data_sweep(
#                 sweep,
#                 observable.qubit_matrixelement,
#                 qubit_subsys=subsys,
#                 qubit_operator=subsys.n_operator(),
#             )
#             datastore = sweep.new_datastore(matrixelem_table=data)
#             data_dict[(qbt_index, subsys)] = datastore
#     return data_dict


# def generate_diffspec_sweep(sweep, initial_state_ind=0):
#     """Takes spectral data of energy eigenvalues and subtracts the energy of a select
#     state, given by its state index.
#
#     Parameters
#     ----------
#     sweep: ParameterSweep
#     initial_state_ind: int or (i1, i2, ...)
#         index of the initial state whose energy is supposed to be subtracted from the
#         spectral data
#
#     Returns
#     -------
#     SpectrumData
#     """
#     lookup = sweep.lookup
#     param_count = sweep.param_count
#     evals_count = sweep.evals_count
#     diff_eigenenergy_table = np.empty(shape=(param_count, evals_count))
#
#     for param_index in tqdm(
#         range(param_count),
#         desc="difference spectrum",
#         leave=False,
#         disable=settings.PROGRESSBAR_DISABLED,
#     ):
#         eigenenergies = sweep.dressed_specdata.energy_table[param_index]
#         if isinstance(initial_state_ind, int):
#             eigenenergy_index = initial_state_ind
#         else:
#             eigenenergy_index = lookup.dressed_index(initial_state_ind, param_index)
#         diff_eigenenergies = eigenenergies - eigenenergies[eigenenergy_index]
#         diff_eigenenergy_table[param_index] = diff_eigenenergies
#     return storage.SpectrumData(
#         diff_eigenenergy_table, sweep.system_params, sweep.param_name, sweep.param_vals
#     )


# def is_single_qubit_transition(
#     state_index1: Tuple[int, ...],
#     state_index2: Tuple[int, ...],
#     hilbertspace: "HilbertSpace",
# ) -> bool:
#     qubits_involved = 0
#     for subsys_index, (i1, i2) in enumerate(zip(state_index1, state_index2)):
#         if isinstance(hilbertspace[subsys_index], Oscillator) and i1 != i2:
#             return False
#         elif i1 != i2 and qubits_involved == 0:
#             qubits_involved += 1
#         else:
#             return False
#     return qubits_involved == 1
#
#
# def qubit_transitions(
#     sweep: "ParameterSweep",
#     photonnumber: int = 1,
#     initial_state: Optional[Tuple[int, ...]] = None,
# ) -> Dict[Tuple[int, ...], SpectrumData]:
#     """
#     Extracts energies for single-qubit transitions.
#
#     Parameters
#     ----------
#     sweep:
#     photonnumber:
#         number of photons used in transition
#     initial_state:
#         bare-state labels of the initial state whose energy is to be subtracted
#         from the final-state energies
#
#     Returns
#     -------
#         list of transition target states, spectrum data
#     """
#     initial_state = initial_state or (0,) * len(sweep._hilbertspace)
#
#     subsys_ranges = [
#         range(subsys_dim) for subsys_dim in sweep._hilbertspace.subsystem_dims
#     ]
#
#     target_states_list = [
#         state_index
#         for state_index in itertools.product(*subsys_ranges)
#         if is_single_qubit_transition(state_index, initial_state, self._hilbertspace)
#     ]
#
#     difference_energies_table = []
#
#     for param_index in range(sweep.param_count):
#         difference_energies = []
#         initial_energy = lookup.energy_bare_index(initial_state_labels, param_index)
#         for target_labels in target_states_list:
#             target_energy = lookup.energy_bare_index(target_labels, param_index)
#             if target_energy is None or initial_energy is None:
#                 difference_energies.append(np.NaN)
#             else:
#                 difference_energies.append(
#                     (target_energy - initial_energy) / photonnumber
#                 )
#         difference_energies_table.append(difference_energies)
#
#     data = np.asarray(difference_energies_table)
#     specdata = storage.SpectrumData(
#         data, sweep.system_params, sweep.param_name, sweep.param_vals
#     )
#     return target_states_list, specdata
