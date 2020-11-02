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


def set_wavefunction_scaling(wavefunctions, potential_vals):
    """
    Sets the scaling parameter for 1d wavefunctions

    Parameters
    ----------
    wavefunctions: list of WaveFunction
        list of all WaveFunction objects to be included in plot and scaling
    potential_vals: ndarray

    Returns
    -------
    float
      scaling factor
    """
    # Do not attempt to scale down amplitudes to very small energy spacings, i.e. if energy spacing is smaller than
    # y_range * Y_RANGE_THRESHOLD_FRACTION, then do not apply additional downscaling
    Y_RANGE_THRESHOLD_FRACTION = 1 / 15

    # If energy spacing is used for scaling, fill no more than this  fraction of the spacing.
    FILLING_FRACTION = 1.0

    # Largest allowed wavefunction amplitude range as fraction of y_range.
    MAX_AMPLITUDE_FRACTION = 1 / 7

    # Amplitude threshold for applying any scaling at all. Note that the imaginary part of a wavefunction may be
    # nominally 0; do not scale up in that case.
    PRECISION_THRESHOLD = 1E-8

    wavefunc_count = len(wavefunctions)
    energies = [wavefunc.energy for wavefunc in wavefunctions]

    y_min = np.min(potential_vals)
    y_max = max([np.max(potential_vals), np.max(energies)])
    y_range = y_max - y_min

    amplitudes = np.asarray([wavefunc.amplitudes for wavefunc in wavefunctions])

    def amplitude_mins():
        return np.apply_along_axis(func1d=np.min, axis=1, arr=amplitudes)

    def amplitude_maxs():
        return np.apply_along_axis(func1d=np.max, axis=1, arr=amplitudes)

    def max_amplitude_range():
        return np.max(amplitude_maxs() - amplitude_mins())

    if max_amplitude_range() < PRECISION_THRESHOLD:  # amplitude likely just zero (e.g., mode='imag'); do not scale up
        return 1
    else:
        scale_factor = y_range * MAX_AMPLITUDE_FRACTION / max_amplitude_range()  # set amplitudes to largest acceptable
        amplitudes *= scale_factor

        if wavefunc_count == 1:
            return scale_factor

        amplitude_fillings = np.pad(np.abs(amplitude_mins()), [0, 1]) + np.pad(np.abs(amplitude_maxs()), [1, 0])
        amplitude_fillings = amplitude_fillings[1:-1]

        energy_spacings = np.pad(energies, [0, 1]) - np.pad(energies, [1, 0])
        energy_spacings = energy_spacings[1:-1]

        for energy_gap, amplitude_filling in zip(energy_spacings, amplitude_fillings):
            if energy_gap > y_range * Y_RANGE_THRESHOLD_FRACTION:
                if amplitude_filling > energy_gap * FILLING_FRACTION:
                    scale_factor *= energy_gap * FILLING_FRACTION / amplitude_filling
                    amplitudes *= energy_gap * FILLING_FRACTION / amplitude_filling
                    amplitude_fillings *= energy_gap * FILLING_FRACTION / amplitude_filling
        return scale_factor


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
