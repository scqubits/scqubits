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


import numpy as np

import scqubits.utils.plotting as plot
from scqubits.settings import DEFAULT_ENERGY_UNITS


def bare_spectrum(sweep, subsys, which=-1, title=None, fig_ax=None):
    """
    Plots energy spectrum of bare system `subsys` for given parameter sweep `sweep`.

    Parameters
    ----------
    sweep: ParameterSweep
    subsys: QuantumSystem
    which: None or int or list(int)
        default: -1, signals to plot all wavefunctions within the truncated Hilbert space;
        int>0: plot wavefunctions 0..int-1; list(int) plot specific wavefunctions
    title: str, optional
        plot title
    fig_ax: Figure, Axes

    Returns
    -------
    fig, axes
    """
    subsys_index = sweep.hilbertspace.index(subsys)
    specdata = sweep.bare_specdata_list[subsys_index]
    if which is None:
        which = subsys.truncated_dim
    return specdata.plot_evals_vs_paramvals(which=which, title=title, fig_ax=fig_ax)


def dressed_spectrum(sweep, title=None, fig_ax=None):
    """
    Plots energy spectrum of dressed system

    Returns
    -------
    fig, axes
    """
    ymax = np.max(sweep.dressed_specdata.energy_table) - np.min(sweep.dressed_specdata.energy_table)
    return sweep.dressed_specdata.plot_evals_vs_paramvals(subtract_ground=True, ymax=min(15, ymax),
                                                          title=title, fig_ax=fig_ax)


def difference_spectrum(sweep, initial_state_ind=0):
    return sweep.get_difference_spectrum(initial_state_ind).plot_evals_vs_paramvals()


def n_photon_qubit_spectrum(sweep, photonnumber, osc_subsys_list, initial_state_ind=0, title=None, fig_ax=None):
    label_array, specdata = sweep.get_n_photon_qubit_spectrum(photonnumber, osc_subsys_list, initial_state_ind)
    return specdata.plot_evals_vs_paramvals(title=title, label_list=label_array.tolist(), fig_ax=fig_ax)


def bare_wavefunction(sweep, param_val, subsys, which=-1, phi_count=None, title=None, fig_ax=None):
    """
    Plot bare wavefunctions for given parameter value and subsystem.

    Parameters
    ----------
    sweep: ParameterSweep
    param_val: float
        value of the external parameter
    subsys: QuantumSystem
    which: int or list(int)
        default: -1, signals to plot all wavefunctions; int>0: plot wavefunctions 0..int-1; list(int) plot specific
        wavefunctions
    phi_count: None or int
    title: str, optional
    fig_ax: Figure, Axes

    Returns
    -------
    fig, axes
    """
    subsys_index = sweep.hilbertspace.index(subsys)
    sweep.update_hilbertspace(param_val)

    param_index = np.searchsorted(sweep.param_vals, param_val)

    evals = sweep.bare_specdata_list[subsys_index].energy_table[param_index]
    evecs = sweep.bare_specdata_list[subsys_index].state_table[param_index]
    return subsys.plot_wavefunction(esys=(evals, evecs), which=which, mode='real', phi_count=phi_count,
                                    title=title, fig_ax=fig_ax)


def chi(sweep, qbt_index, osc_index, title=None, fig_ax=None):
    data_key = 'chi_osc{}_qbt{}'.format(osc_index, qbt_index)
    ydata = sweep.sweep_data[data_key]
    xdata = sweep.param_vals
    xlabel = sweep.param_name
    ylabel = r'$\chi_j$' + DEFAULT_ENERGY_UNITS
    state_count = ydata.shape[1]
    label_list = list(range(state_count))
    return plot.data_vs_paramvals(xdata, ydata, xlim=None, ymax=None, xlabel=xlabel, ylabel=ylabel, title=title,
                                  label_list=label_list, fig_ax=fig_ax)


def chi_01(sweep, qbt_index, osc_index, param_index=0, fig_ax=None):
    data_key = 'chi_osc{}_qbt{}'.format(osc_index, qbt_index)
    ydata = sweep.sweep_data[data_key]
    xdata = sweep.param_vals
    xlabel = sweep.param_name
    ylabel = r'$\chi_{{01}}$ [{}]'.format(DEFAULT_ENERGY_UNITS)
    title = r'$\chi_{{01}}=${:.4f} {}'.format(ydata[param_index], DEFAULT_ENERGY_UNITS)
    return plot.data_vs_paramvals(xdata, ydata, xlim=None, ymax=None, xlabel=xlabel, ylabel=ylabel, title=title,
                                  label_list=None, fig_ax=fig_ax)
