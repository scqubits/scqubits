# plot_defaults.py
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

import scqubits
import scqubits.core.constants as constants
import scqubits.core.units as units


NAME_REPLACEMENTS = {
    'ng': r'$n_g$',
    'ng1': r'$n_{g1}$',
    'ng2': r'$n_{g2}$',
    'EJ': r'$E_J$',
    'EJ1': r'$E_{J1}$',
    'EJ2': r'$E_{J2}$',
    'EJ3': r'$E_{J3}$',
    'EC': r'$E_C$',
    'EL': r'$E_L$',
    'flux': r'$\Phi_{ext}/\Phi_0$'
}


def recast_name(raw_name):
    if raw_name in NAME_REPLACEMENTS:
        return NAME_REPLACEMENTS[raw_name]
    return raw_name


def set_scaling(qubit, scaling, potential_vals=None):
    """
    Sets the scaling parameter for 1d wavefunctions

    Parameters
    ----------
    qubit: QuantumSystem
    scaling: float
    potential_vals: ndarray

    Returns
    -------
    float
        scaling factor
    """
    if scaling is None:
        if isinstance(qubit, scqubits.Transmon):
            scaling = 0.2 * qubit.EJ
        elif isinstance(qubit, scqubits.Fluxonium):
            scaling = 0.125 * (np.max(potential_vals) - np.min(potential_vals))
    return scaling


def wavefunction1d_discrete(mode=None):
    """Plot defaults for plotting.wavefunction1d_discrete.

    Parameters
    ----------
    mode: str
        amplitude modifier, needed to give the correct default y label"""
    ylabel = r'$\psi_j(n)$'
    if mode:
        ylabel = constants.MODE_STR_DICT[mode](ylabel)
    return {
        'xlabel': 'n',
        'ylabel': ylabel
    }


def wavefunction2d():
    """Plot defaults for plotting.wavefunction2d"""
    return {
        'figsize': (8, 3)
    }


def contours(x_vals, y_vals):
    """Plot defaults for plotting.contours"""
    aspect_ratio = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
    figsize = (8, 8 * aspect_ratio)
    return {
        'figsize': figsize
    }


def matrix():
    """Plot defaults for plotting.matrix"""
    return {
        'figsize': (10, 5)
    }


def evals_vs_paramvals(specdata, **kwargs):
    """Plot defaults for plotting.evals_vs_paramvals"""
    kwargs['xlabel'] = kwargs.get('xlabel') or recast_name(specdata.param_name)
    kwargs['ylabel'] = kwargs.get('ylabel') or 'energy [{}]'.format(units.get_units())
    return kwargs


def matelem_vs_paramvals(specdata):
    """Plot defaults for plotting.matelem_vs_paramvals"""
    return {
        'xlabel': recast_name(specdata.param_name),
        'ylabel': 'matrix element'
    }


def dressed_spectrum(sweep, **kwargs):
    """Plot defaults for sweep_plotting.dressed_spectrum"""
    if 'ylim' not in kwargs:
        kwargs['ymax'] = kwargs.get('ymax') or min(15, (np.max(sweep.dressed_specdata.energy_table) -
                                                        np.min(sweep.dressed_specdata.energy_table)))
    kwargs['xlabel'] = kwargs.get('xlabel') or recast_name(sweep.param_name)
    kwargs['ylabel'] = kwargs.get('ylabel') or r'energy [{}]'.format(units.get_units())
    return kwargs


def chi(sweep, **kwargs):
    """Plot defaults for sweep_plotting.chi"""
    kwargs['xlabel'] = kwargs.get('xlabel') or recast_name(sweep.param_name)
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$\chi_j$ [{}]'.format(units.get_units())
    return kwargs


def chi01(param_name, yval, **kwargs):
    """Plot defaults for sweep_plotting.chi01"""
    kwargs['xlabel'] = kwargs.get('xlabel') or recast_name(param_name)
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$\chi_{{01}}$ [{}]'.format(units.get_units())
    kwargs['title'] = kwargs.get('title') or r'$\chi_{{01}}=${:.4f} {}'.format(yval, units.get_units())
    return kwargs


def charge_matrixelem(param_name, **kwargs):
    """Plot defaults for sweep_plotting.charge_matrixelem"""
    kwargs['xlabel'] = kwargs.get('xlabel') or recast_name(param_name)
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$|\langle i |n| j \rangle|$'
    return kwargs


# supported keyword arguments for plotting and sweep_plotting functions
SPECIAL_PLOT_OPTIONS = [
    'fig_ax',
    'figsize',
    'filename',
    'grid',
    'x_range',
    'y_range',
    'ymax',
]
