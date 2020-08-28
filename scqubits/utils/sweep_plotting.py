#processed sweep_plotting.py
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

import scqubits.core.sweep_generators as sweep_gen
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot


def bare_spectrum(sweep, subsys, which=-1, **kwargs):
    """
    Plots energy spectrum of bare system `subsys` for given ParameterSweep `sweep`.

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
    subsys_index = sweep.get_subsys_index(subsys)
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
    return sweep.dressed_specdata.plot_evals_vs_paramvals(subtract_ground=True,
                                                          **defaults.dressed_spectrum(sweep, **kwargs))


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
    return sweep_gen.generate_diffspec_sweep(sweep, initial_state_ind).plot_evals_vs_paramvals(**kwargs)


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
    label_list, specdata = sweep_gen.generate_qubit_transitions_sweep(sweep, photonnumber, initial_state_labels)
    return specdata.plot_evals_vs_paramvals(label_list=label_list, **kwargs)


def bare_wavefunction(sweep, param_val, subsys, which=-1, phi_grid=None, **kwargs):
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
    phi_grid: Grid1d, optional
        used for setting a custom grid for phi; if None use self._default_grid
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    fig, axes
    """
    subsys_index = sweep.get_subsys_index(subsys)
    sweep.update_hilbertspace(param_val)
    param_index = np.searchsorted(sweep.param_vals, param_val)
    evals = sweep.bare_specdata_list[subsys_index].energy_table[param_index]
    evecs = sweep.bare_specdata_list[subsys_index].state_table[param_index]
    return subsys.plot_wavefunction(esys=(evals, evecs), which=which, mode='real', phi_grid=phi_grid, **kwargs)


def chi(datastore, **kwargs):
    """
    Plot dispersive shifts chi_j for a given pair of qubit and oscillator.

    Parameters
    ----------
    datastore: DataStore
        contains sweep data for the dispersive shift, stored as datastore.chi
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    ydata = datastore.chi
    xdata = datastore.param_vals
    state_count = ydata.shape[1]
    label_list = list(range(state_count))
    return plot.data_vs_paramvals(xdata, ydata, label_list=label_list, **defaults.chi(datastore.param_name, **kwargs))


def chi_01(datastore, param_index=0, **kwargs):
    """
    Plot the dispersive shift chi01 for a given pair of qubit and oscillator.

    Parameters
    ----------
    datastore: DataStore
    param_index: int, optional
        index of the external parameter to be used
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    ydata = datastore.chi
    xdata = datastore.param_vals
    yval = ydata[param_index]
    return plot.data_vs_paramvals(xdata, ydata, label_list=None, **defaults.chi01(datastore.param_name, yval, **kwargs))


def charge_matrixelem(datastore, qbt_index_subsys, initial_state_idx=0, **kwargs):
    """

    Parameters
    ----------
    datastore: DataStore
    qbt_index_subsys: tuple(int, QuantumSystem)
        index of the qubit system within the underlying HilbertSpace, and qubit object
    initial_state_idx: int
        index of initial state
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
    """
    (qbt_index, qbt_subsys) = qbt_index_subsys
    label_list = [(initial_state_idx, final_idx) for final_idx in range(qbt_subsys.truncated_dim)]
    return plot.matelem_vs_paramvals(datastore, select_elems=label_list, mode='abs',
                                     **defaults.charge_matrixelem(datastore.param_name, **kwargs))
