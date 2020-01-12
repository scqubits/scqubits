# sweep_plotting.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import copy
import numpy as np

import scqubits.utils.plotting as plot
from scqubits.settings import DEFAULT_ENERGY_UNITS


def bare_spectrum(sweep, subsys, which=-1, **kwargs):
    """
    Plots energy spectrum of bare system `subsys` for given parameter sweep `sweep`.

    Parameters
    ----------
    sweep: ParameterSweep
    subsys: QuantumSystem
    which: int or list(int), optional
        default: -1, signals to plot all wavefunctions within the truncated Hilbert space;
        int>0: plot wavefunctions 0..int-1; list(int) plot specific wavefunctions
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    fig, axes
    """
    subsys_index = sweep.hilbertspace.index(subsys)
    specdata = sweep.bare_specdata_list[subsys_index]
    if which is None:
        which = subsys.truncated_dim
    return specdata.plot_evals_vs_paramvals(which=which, **kwargs)


def dressed_spectrum(sweep, **kwargs):
    """
    Plots energy spectrum of dressed system

    Parameters
    ----------
    sweep: ParameterSweep
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    fig, axes
    """
    if 'ymax' not in kwargs:
        kwargs['ymax'] = min(15, (np.max(sweep.dressed_specdata.energy_table) -
                                  np.min(sweep.dressed_specdata.energy_table)))
    return sweep.dressed_specdata.plot_evals_vs_paramvals(subtract_ground=True, **kwargs)


def difference_spectrum(sweep, initial_state_ind=0, **kwargs):
    """
    Plots a transition energy spectrum with reference to the given initial_state_ind, obtained by taking energy
    differences of the eigenenergy spectrum.

    Parameters
    ----------
    sweep: ParameterSweep
    initial_state_ind: int
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    return sweep.get_difference_spectrum(initial_state_ind).plot_evals_vs_paramvals(**kwargs)


def n_photon_qubit_spectrum(sweep, photonnumber, initial_state_labels, **kwargs):
    """
    Plots the n-photon qubit transition spectrum.

    Parameters
    ----------
    sweep: ParameterSweep
    photonnumber: int
        number of photons used in the transition
    initial_state_labels: tuple(int1, int2, ...)
        bare state index of the initial state for the transitions
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    label_list, specdata = sweep.get_n_photon_qubit_spectrum(photonnumber, initial_state_labels)
    return specdata.plot_evals_vs_paramvals(label_list=label_list, **kwargs)


def bare_wavefunction(sweep, param_val, subsys, which=-1, phi_count=None, **kwargs):
    """
    Plot bare wavefunctions for given parameter value and subsystem.

    Parameters
    ----------
    sweep: ParameterSweep
    param_val: float
        value of the external parameter
    subsys: QuantumSystem
    which: int or list(int), optional
        default: -1, signals to plot all wavefunctions; int>0: plot wavefunctions 0..int-1; list(int) plot specific
        wavefunctions
    phi_count: int, optional
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    fig, axes
    """
    subsys_index = sweep.hilbertspace.index(subsys)
    sweep.update_hilbertspace(param_val)

    param_index = np.searchsorted(sweep.param_vals, param_val)

    evals = sweep.bare_specdata_list[subsys_index].energy_table[param_index]
    evecs = sweep.bare_specdata_list[subsys_index].state_table[param_index]
    return subsys.plot_wavefunction(esys=(evals, evecs), which=which, mode='real', phi_count=phi_count, **kwargs)


def chi(sweep, qbt_index, osc_index, **kwargs):
    """
    Plot dispersive shifts chi_j for a given pair of qubit and oscillator.

    Parameters
    ----------
    sweep: ParameterSweep
    qbt_index: int
        index of the qubit system within the underlying HilbertSpace
    osc_index: int
        index of the oscillator system within the underlying HilbertSpace
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    data_key = 'chi_osc{}_qbt{}'.format(osc_index, qbt_index)
    ydata = sweep.sweep_data[data_key]
    xdata = sweep.param_vals
    kwargs['xlabel'] = kwargs.get('xlabel') or sweep.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$\chi_j$' + DEFAULT_ENERGY_UNITS
    state_count = ydata.shape[1]
    label_list = list(range(state_count))
    return plot.data_vs_paramvals(xdata, ydata, label_list=label_list, **kwargs)


def chi_01(sweep, qbt_index, osc_index, param_index=0, **kwargs):
    """
    Plot the dispersive shift chi01 for a given pair of qubit and oscillator.

    Parameters
    ----------
    sweep: ParameterSweep
    qbt_index: int
        index of the qubit system within the underlying HilbertSpace
    osc_index: int
        index of the oscillator system within the underlying HilbertSpace
    param_index: int, optional
        index of the external parameter to be used
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    data_key = 'chi_osc{}_qbt{}'.format(osc_index, qbt_index)
    ydata = sweep.sweep_data[data_key]
    xdata = sweep.param_vals
    kwargs['xlabel'] = kwargs.get('xlabel') or sweep.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$\chi_{{01}}$ [{}]'.format(DEFAULT_ENERGY_UNITS)
    kwargs['title'] = kwargs.get('title') or r'$\chi_{{01}}=${:.4f} {}'.format(ydata[param_index], DEFAULT_ENERGY_UNITS)
    return plot.data_vs_paramvals(xdata, ydata, label_list=None, **kwargs)


def charge_matrixelem(sweep, qbt_index, initial_state_idx=0, **kwargs):
    """

    Parameters
    ----------
    sweep: ParameterSweep
    qbt_index: int
        index of the qubit system within the underlying HilbertSpace
    initial_state_idx: int
        index of initial state 
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    data_key = 'n_op_qbt{}'.format(qbt_index)
    specdata = copy.deepcopy(sweep.bare_specdata_list[qbt_index])
    specdata.matrixelem_table = sweep.sweep_data[data_key]
    kwargs['xlabel'] = kwargs.get('xlabel') or sweep.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$|\langle i |n| j \rangle|$'
    label_list = [(initial_state_idx, final_idx) for final_idx in range(sweep.hilbertspace[qbt_index].truncated_dim)]
    return plot.matelem_vs_paramvals(specdata, select_elems=label_list, mode='abs', **kwargs)
