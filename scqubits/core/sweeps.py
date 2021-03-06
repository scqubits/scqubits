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

from scqubits import Oscillator
from scqubits.core.namedslots_array import NamedSlotsNdarray

if TYPE_CHECKING:
    from scqubits import HilbertSpace, Oscillator, SpectrumData
    from scqubits.core.qubit_base import QubitBaseClass
    from scqubits.core.param_sweep import ParameterSweep, ParameterSweepBase

    QuantumSys = Union[QubitBaseClass, Oscillator]


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
    reduced_parameters = sweep.parameters.create_sliced(
        sweep._current_param_indices, remove_fixed=False
    )
    total_count = np.prod(reduced_parameters.counts)

    def func_effective(paramindex_tuple: Tuple[int], params, **kw) -> Any:
        paramvals_tuple = params[paramindex_tuple]
        return func(
            sweep,
            paramindex_tuple=paramindex_tuple,
            paramvals_tuple=paramvals_tuple,
            **kw,
        )

    if hasattr(func, "__name__"):
        func_name = func.__name__
    else:
        func_name = ""

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
            desc="sweeping " + func_name,
            leave=False,
            disable=settings.PROGRESSBAR_DISABLED,
        )
    )
    data_array = np.asarray(data_array)
    return data_array.reshape(reduced_parameters.counts)


def dispersive(sweep: "ParameterSweep"):
    """
    In the dispersive limit, a system of oscillators and qubits has the effective
    Hamiltonian (to leading order in any coupling)
    :math:`H = \sum_s (\omega_s + \chi^{s,\Sigma}_0)a_s^\dag a_s
    + \sum_q \sum_l (\epsilon^q_l + \Delta\epsilon^q_l) |l_q\rangle \langle l_q|
    + \sum_s \sum_{q,l>0} \chi^{sq}_l a_s^\dag a_s |l_q\rangle \langle l_q|
    + \sum_s \kappa_s a_s^\dag a_s^\dag  a_s a_s
    + \sum_{s\not=s'} K_{ss'} a_s^\dag a_s a_{s'}^\dag a_{s'}
    + \sum_{q\not=q'}\sum_{l,l'} \Lambda^{qq'}_{ll'} |l_q l'_{q'}\rangle \langle l_q
    l'_{q'}\rangle`.
    Using dressed eigenenergies labeled by the bare product states,
    :math:`E(n_1, n_2, \ldots; l_1, l_2, \lddots)` the dispersive parameters are
    computed from the exact eigenvalues via

    :math:`\chi^{(s,\Sigma)}_0 = E(\vec{s}=\hat{e}_s,\vec{l}=\vec{o}) - E(\vec{
    s}=\vec{o},\vec{l}=\vec{o}) - \omega_s`
    :math:`\Delte_epsilon^q_l = E(\vec{s}=\vec{o},\vec{l}=\hat{e}_q) - E(\vec{
    s}=\vec{o},\vec{l}=\vec{o}) - \epsilon^q_l`
    :math:`\chi^{sq}_l  = E(\vec{s}=\vec{o},\vec{l}=\hat{e}_q) - E(\vec{
    s}=\vec{o},\vec{l}=\vec{o}) - \epsilon^q_l`
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
