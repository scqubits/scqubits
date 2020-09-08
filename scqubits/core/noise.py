# noise.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import scipy.constants
import scqubits.utils.plotting as plotting
import scqubits.core.units as units
import scqubits.settings as settings

# Helpers for units conversion


def calc_therm_ratio(omega, T, omega_in_standard_units=False):
    r"""Returns the ratio
    
    :math:`\beta \omega = \frac{\hbar \omega}{k_B T}`

    after converting `\omega` from system units, to standard units.

    Parameters
    ----------
    omega: float
        angular frequency in system units
    T: float 
        temperature in Kelvin
    omega_in_standard_units: bool 
        is omega given in standard units (i.e. Hz)

    Returns
    -------
    float
    """
    omega = units.to_standard_units(omega) if not omega_in_standard_units else omega
    return (sp.constants.hbar * omega) / (sp.constants.k * T)


def convert_eV_to_Hz(val):
    r"""
    Convert a value in electron volts to Hz.

    Parameters
    ----------
    val: number 
        number in electron volts

    Returns
    -------
    float
        number in Hz
    """
    return val * sp.constants.e / sp.constants.h


# Default values of various noise constants and parameters.
NOISE_PARAMS = {
    'A_flux': 1e-6,   # Flux noise strength. Units: Phi_0
    'A_ng': 1e-4,     # Charge noise strength. Units of charge e
    'A_cc': 1e-7,     # Critical current noise strength. Units of critical current I_c
    'omega_low': 1e-9 * 2 * np.pi,   # Low frequency cutoff. Units: 2pi GHz
    'omega_high': 3 * 2 * np.pi,     # High frequency cutoff. Units: 2pi GHz
    'Delta': 3.4e-4,  # Superconducting gap for aluminum (at T=0). Units: eV
    'x_qp': 3e-6,     # Quasiparticles density (see for example Pol et al 2014)
    't_exp': 1e4,     # Measurement time. Units: ns
    'R_0': 50,        # Characteristic impedance of a transmission line. Units: Ohms
    'T': 0.015,       # Typical temperature for a superconducting circuit experiment. Units: K
    'M': 400,         # Mutual inductance between qubit and a flux line. Units: \Phi_0 / Ampere
    'R_k': sp.constants.h / sp.constants.e**2.0,  # Normal quantum resistance, aka Klitzing constant.
                                                  # Note, in some papers a superconducting quantum
                                                  # resistance is used, and defined as: h/(2e)^2
}


class NoisySystem:

    def effective_noise_channels(self):
        """Return a list of noise channels that are used when calculating the effective noise 
        (i.e. via `t1_effective` and `t2_effective`. 
        """
        return self.supported_noise_channels()

    def plot_coherence_vs_paramvals(self, param_name, param_vals, noise_channels=None, common_noise_options=None,
                                    spectrum_data=None, scale=1, num_cpus=settings.NUM_CPUS, **kwargs):
        r"""
        Show plots of coherence for various channels supported by the qubit as they vary as a function of a
        changing parameter. 

        For example, assuming `qubit` is a qubit object with `flux` being one of its parameters, one can 
        see how coherence due to various noise channels vary as the `flux` changes::

            qubit.plot_coherence_vs_paramvals(param_name='flux',
                                              param_vals=np.linspace(-0.5, 0.5, 100), 
                                              scale=1e-3, ylabel=r"$\mu s$");


        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        noise_channels: None or str or list(str) or list(tuple(str, dict))
            channels to be plotted, if None then noise channels given by `supported_noise_channels` are used
        common_noise_options: dict
            common options used when calculating coherence times
        spectrum_data: SpectrumData
            spectral data used during noise calculations 
        scale: float
            a number that all data is multiplied by before being plotted
        num_cpus: int
            number of cores to be used for computation

        Returns
        -------
        Figure, Axes

        """
        common_noise_options = {} if common_noise_options is None else common_noise_options

        # if we're not told what channels to consider, just use the supported list
        noise_channels = self.supported_noise_channels() if noise_channels is None else noise_channels

        # if we only have a single noise channel to consider (and hence are given a str), put it into a one element list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        if spectrum_data is None:

            # We have to figure out the largest energy level involved in the calculations, to know how many levels we
            # need from the diagonalization.
            # This may be hidden in noise-channel-specific options, so have to search through those, if any were given.
            max_level = max(common_noise_options.get('i', 1), common_noise_options.get('j', 1))
            for noise_channel in noise_channels:
                if isinstance(noise_channel, tuple):
                    opts = noise_channel[1]
                    max_level = max(max_level, opts.get('i', 1), opts.get('j', 1))

            spectrum_data = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=max_level+1,
                                                           subtract_ground=True, get_eigenstates=True, filename=None,
                                                           num_cpus=num_cpus)

        # figure out how many plots we need to produce
        plot_grid = (1, 1) if len(noise_channels) == 1 else (math.ceil(len(noise_channels)/2), 2)

        # figure out how large the figure should be, based on how many plots we have.
        # We currently assume 2 plots per row
        figsize = kwargs.get('figsize', (4, 3) if plot_grid == (1, 1) else (8, 3*plot_grid[0]))

        # If axes was given in fig_as, it should support the plot structure consistent with plot_grid,
        # otherwise the plotting routine below, will fail
        fig, axes = kwargs.get('fig_ax') or plt.subplots(*plot_grid, figsize=figsize)

        plotting_options = {'ylabel': units.get_units_time_label(),
                            'xlabel': param_name,
                            'yscale': 'log',
                            'grid': True,
                            }
        plotting_options.update({k: v for (k, v) in kwargs.items() if k not in ['fig_ax', 'figsize']})

        # remember current value of param_name
        current_val = getattr(self, param_name)

        for n, noise_channel in enumerate(noise_channels):

            # noise_channel is a string representing the noise method
            if isinstance(noise_channel, str):

                noise_channel_method = noise_channel

                # calculate the noise over the full param span in param_vals
                noise_vals = [scale * getattr(self.set_and_return(param_name, v), noise_channel_method)(
                    esys=(spectrum_data.energy_table[v_i, :], spectrum_data.state_table[v_i]),
                    **common_noise_options)
                    for v_i, v in enumerate(param_vals)]

            # noise_channel is a tuple representing the noise method and default options
            elif isinstance(noise_channel, tuple):

                noise_channel_method = noise_channel[0]

                options = common_noise_options.copy()
                # Some of the channel-specific options may be in conflict with the common options options.
                # In such a case, we let the channel-specific options take priority.
                options.update(noise_channel[1])

                # calculate the noise over the full param span in param_vals
                noise_vals = [scale * getattr(self.set_and_return(param_name, v), noise_channel_method)(
                              esys=(spectrum_data.energy_table[v_i, :], spectrum_data.state_table[v_i]),
                              **options)
                              for v_i, v in enumerate(param_vals)]

            else:
                raise ValueError(
                    "The `noise_channels` argument should be one of {str, list of str, or list of tuples}.")

            ax = axes.ravel()[n] if len(noise_channels) > 1 else axes
            plotting_options['fig_ax'] = fig, ax
            plotting_options['title'] = noise_channel_method
            plotting.data_vs_paramvals(param_vals, noise_vals, **plotting_options)

        if len(noise_channels) > 1 and len(noise_channels) % 2:
            axes.ravel()[-1].set_axis_off()

        # Set the parameter we varied to its initial value
        setattr(self, param_name, current_val)

        fig.tight_layout()

        return fig, axes

    def plot_t1_effective_vs_paramvals(self, param_name, param_vals, noise_channels=None, common_noise_options=None,
                                       spectrum_data=None, scale=1, num_cpus=settings.NUM_CPUS, **kwargs):
        r"""
        Plot effective :math:`T_1` coherence as it varies as a function of changing parameter.

        The effective :math:`T_1` is calculated by considering a variety of depolarizing noise channels, 
        according to the formula:

        .. math::
            \frac{1}{T_{1}^{\rm eff}} = \frac{1}{2} \sum_k \frac{1}{T_{1}^{k}}

        where :math:`k` runs over the channels that can contribute to the effective noise. 
        By default all the depolarizing noise channels given by the method `effective_noise_channels` are 
        included.

        For example, assuming `qubit` is a qubit object with `flux` being one of its parameters, one can 
        see how the effective :math:`T_1` varies as the `flux` changes::

            qubit.plot_t1_effective_vs_paramvals(param_name='flux',
                                                 param_vals=np.linspace(-0.5, 0.5, 100), 
                                                );

        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        noise_channels: None or str or list(str) or list(tuple(str, dict))
            channels to be plotted, if None then noise channels given by `supported_noise_channels` are used
        common_noise_options: dict
            common options used when calculating coherence times
        spectrum_data: SpectrumData
            spectral data used during noise calculations 
        scale: float
            a number that all data is multiplied by before being plotted
        num_cpus: int
            number of cores to be used for computation

        Returns
        -------
        Figure, Axes

        """
        common_noise_options = {} if common_noise_options is None else common_noise_options

        # If we're not given channels to consider, just use the effective noise channel list that
        # correspond to t1 processes
        noise_channels = [channel for channel in self.effective_noise_channels()
                          if channel.startswith('t1')] if noise_channels is None else noise_channels

        # if we only have a single noise channel to consider (and hence are given a str), put it into a one element list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        if spectrum_data is None:

            # We have to figure out the largest energy level involved in the calculations, to know how many levels we
            # need from the diagonalization.
            # This may be hidden in noise-channel-specific options, so have to search through those, if any were given.
            max_level = max(common_noise_options.get('i', 1), common_noise_options.get('j', 1))
            for noise_channel in noise_channels:
                if isinstance(noise_channel, tuple):
                    opts = noise_channel[1]
                    max_level = max(max_level, opts.get('i', 1), opts.get('j', 1))

            spectrum_data = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=max_level+1,
                                                           subtract_ground=True, get_eigenstates=True, filename=None,
                                                           num_cpus=num_cpus)

        # remember current value of param_name
        current_val = getattr(self, param_name)

        # calculate the noise over the full param span in param_vals
        noise_vals = [scale * self.set_and_return(param_name, v).t1_effective(
            noise_channels=noise_channels,
            common_noise_options=common_noise_options,
            esys=(spectrum_data.energy_table[v_i, :], spectrum_data.state_table[v_i])
        )
            for v_i, v in enumerate(param_vals)]

        # Set the parameter we varied to its initial value
        setattr(self, param_name, current_val)

        plotting_options = {'fig_ax': plt.subplots(1),
                            'title': 't1_effective',
                            'ylabel': units.get_units_time_label(),
                            'xlabel': param_name,
                            'yscale': 'log',
                            'grid': True,
                            }
        plotting_options.update(kwargs)
        fig, axes = plotting.data_vs_paramvals(param_vals, noise_vals, **plotting_options)

        fig.tight_layout()

        return fig, axes

    def plot_t2_effective_vs_paramvals(self, param_name, param_vals, noise_channels=None, common_noise_options=None,
                                       spectrum_data=None, scale=1, num_cpus=settings.NUM_CPUS, **kwargs):
        r"""
        Plot effective :math:`T_2` coherence as it varies as a function of changing parameter.

        The effective :math:`T_2` is calculated from both pure dephasing channels, as well as 
        depolarization channels, according to the formula:

        .. math::
            \frac{1}{T_{2}^{\rm eff}} = \sum_k \frac{1}{T_{\phi}^{k}} +  \frac{1}{2} \sum_j \frac{1}{T_{1}^{j}}

        where :math:`k` (:math:`j`) run over the relevant pure dephasing (depolariztion) channels that 
        can contribute to the effective noise. 
        By default all noise channels given by the method `effective_noise_channels` are included.

        For example, assuming `qubit` is a qubit object with `flux` being one of its parameters, one can 
        see how the effective :math:`T_2` varies as the `flux` changes::

            qubit.plot_t2_effective_vs_paramvals(param_name='flux',
                                                 param_vals=np.linspace(-0.5, 0.5, 100), 
                                                );

        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        noise_channels: None or str or list(str) or list(tuple(str, dict))
            channels to be plotted, if None then noise channels given by `supported_noise_channels` are used
        common_noise_options: dict
            common options used when calculating coherence times
        spectrum_data: SpectrumData
            spectral data used during noise calculations 
        scale: float
            a number that all data is multiplied by before being plotted
        num_cpus: int
            number of cores to be used for computation

        Returns
        -------
        Figure, Axes

        """
        common_noise_options = {} if common_noise_options is None else common_noise_options

        # If we're not given channels to consider, just use ones from the effective noise channel list
        noise_channels = [channel for
                          channel in self.effective_noise_channels()] if noise_channels is None else noise_channels

        # if we only have a single noise channel to consider (and hence are given a str), put it into a one element list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        if spectrum_data is None:

            # We have to figure out the largest energy level involved in the calculations, to know how many levels we
            # need from the diagonalization.
            # This may be hidden in noise-channel-specific options, so have to search through those, if any were given.
            max_level = max(common_noise_options.get('i', 1), common_noise_options.get('j', 1))
            for noise_channel in noise_channels:
                if isinstance(noise_channel, tuple):
                    opts = noise_channel[1]
                    max_level = max(max_level, opts.get('i', 1), opts.get('j', 1))

            spectrum_data = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=max_level+1,
                                                           subtract_ground=True, get_eigenstates=True, filename=None,
                                                           num_cpus=num_cpus)

        # remember current value of param_name
        current_val = getattr(self, param_name)

        # calculate the noise over the full param span in param_vals
        noise_vals = [scale * self.set_and_return(param_name, v).t2_effective(
            noise_channels=noise_channels,
            common_noise_options=common_noise_options,
            esys=(spectrum_data.energy_table[v_i, :], spectrum_data.state_table[v_i])
        )
            for v_i, v in enumerate(param_vals)]

        # Set the parameter we varied to its initial value
        setattr(self, param_name, current_val)

        plotting_options = {'fig_ax': plt.subplots(1),
                            'title': 't2_effective',
                            'ylabel': units.get_units_time_label(),
                            'xlabel': param_name,
                            'yscale': 'log',
                            'grid': True,
                            }
        plotting_options.update(kwargs)
        fig, axes = plotting.data_vs_paramvals(param_vals, noise_vals, **plotting_options)

        fig.tight_layout()

        return fig, axes

    def _effective_rate(self, noise_channels, common_noise_options, esys, noise_type):
        """
        Helper method used when calculating the effective rates by methods `t1_effective` and `t2_effective`.

        Parameters
        ----------
        noise_channels: None or str or list(str) or list(tuple(str, dict))
            channels to be plotted, if None then noise channels given by `supported_noise_channels` are used
        common_noise_options: dict
            common options used when calculating coherence times
        esys: tuple(evals, evecs)
            spectral data used during noise calculations 
        noise_type: str
            type of noise, one of 'tphi' or 't1'

        Returns
        -------
        coherence rate: float

        """
        rate = 0.0

        for n, noise_channel in enumerate(noise_channels):

            # noise_channel is a string representing the noise method
            if isinstance(noise_channel, str):

                noise_channel_method = noise_channel

                # If dealing with a tphi noise type, the contribution of a t1 process to the dephasing rate its halved.
                scale_factor = 0.5 if noise_type == 'tphi' and noise_channel_method.startswith("t1") else 1

                options = common_noise_options.copy()
                # We need to make sure we calculate a rate
                options['get_rate'] = True

                # calculate the noise over the full param span in param_vals
                rate += scale_factor * getattr(self, noise_channel_method)(esys=esys, **options)

            # noise_channel is a tuple representing the noise method and default options
            elif isinstance(noise_channel, tuple):

                noise_channel_method = noise_channel[0]

                # If dealing with a tphi noise type, the contribution of a t1 process to the dephasing rate its halved.
                scale_factor = 0.5 if noise_type == 'tphi' and noise_channel_method.startswith("t1") else 1

                options = common_noise_options.copy()
                # Some of the channel-specific options may be in conflict with the common options options.
                # In such a case, we let the channel-specific options take priority.
                options.update(noise_channel[1])
                # We need to make sure we calculate a rate
                options['get_rate'] = True

                # calculate the noise over the full param span in param_vals
                rate += scale_factor * getattr(self, noise_channel_method)(esys=esys, **options)

            else:
                raise ValueError(
                    "The `noise_channels` argument should be one of {str, list of str, or list of tuples}.")

        return rate

    def t1_effective(self, noise_channels=None, common_noise_options=None, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the effective :math:`T_1` time (or rate). 

        The effective :math:`T_1` is calculated by considering a variety of depolarizing noise channels, 
        according to the formula:

        .. math::
            \frac{1}{T_{1}^{\rm eff}} = \frac{1}{2} \sum_k \frac{1}{T_{1}^{k}}

        where :math:`k` runs over the channels that can contribute to the effective noise. 
        By default all the depolarizing noise channels given by the method `effective_noise_channels` are 
        included. Users can also provide specific noise channels, with selected options, to be included 
        in the effective :math:`T_1` calculation. For example, assuming `qubit` is a qubit object, can can execute:: 

            tune_tmon.t1_effective(noise_channels=['t1_charge_impedance', 't1_flux_bias_line'], 
                           common_noise_options=dict(T=0.050))

        Parameters
        ----------
        noise_channels: None or str or list(str) or list(tuple(str, dict))
            channels to be plotted, if None then noise channels given by `supported_noise_channels` are used
        common_noise_options: dict
            common options used when calculating coherence times
        esys: tuple(evals, evecs)
            spectral data used during noise calculations 
        get_rate: bool
            get rate or time


        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.


        """
        common_noise_options = {} if common_noise_options is None else common_noise_options

        # If we're not given channels to consider, just use the effective noise channel list that
        # correspond to t1 processes
        noise_channels = [channel for channel in self.effective_noise_channels()
                          if channel.startswith('t1')] if noise_channels is None else noise_channels

        # If we're given only a single channel as a string, make it a one entry list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        # Do a sanity check; if we're given a tphi channel, raise an exception
        for noise_channel in noise_channels:
            channel = noise_channel[0] if isinstance(noise_channel, tuple) else noise_channel
            if not channel.startswith("t1"):
                raise ValueError("Only t1 channels can contribute to effective t1 noise.")

        if esys is None:
            # We have to figure out the largest energy level involved in the calculations, to know how many levels we
            # need from the diagonalization.
            # This may be hidden in noise-channel-specific options, so have to search through those, if any were given.
            max_level = max(common_noise_options.get('i', 1), common_noise_options.get('j', 1))
            for noise_channel in noise_channels:
                if isinstance(noise_channel, tuple):
                    opts = noise_channel[1]
                    max_level = max(max_level, opts.get('i', 1), opts.get('j', 1))

            esys = self.eigensys(evals_count=max_level+1)

        rate = self._effective_rate(noise_channels=noise_channels, common_noise_options=common_noise_options, esys=esys,
                                    noise_type='t1')
        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def t2_effective(self, noise_channels=None, common_noise_options=None, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the effective :math:`T_2` time (or rate). 

        The effective :math:`T_2` is calculated by considering a variety of pure dephasing and depolarizing 
        noise channels, according to the formula:

        .. math::
            \frac{1}{T_{2}^{\rm eff}} = \sum_k \frac{1}{T_{\phi}^{k}} +  \frac{1}{2} \sum_j \frac{1}{T_{1}^{j}}, 

        where :math:`k` (:math:`j`) run over the relevant pure dephasing (depolariztion) channels that can contribute to 
        the effective noise. By default all the noise channels given by the method 
        `effective_noise_channels` are included. Users can also provide specific noise channels, with 
        selected options, to be included in the effective :math:`T_2` calculation. For example, assuming 
        `qubit` is a qubit object, can can execute:: 

            qubit.t2_effective(noise_channels=['t1_flux_bias_line', 't1_capacitive',
                                               ('tphi_1_over_f_flux', dict(A_noise=3e-6))], 
                               common_noise_options=dict(T=0.050))

        Parameters
        ----------
        noise_channels: None or str or list(str) or list(tuple(str, dict))
            channels to be plotted, if None then noise channels given by `supported_noise_channels` are used
        common_noise_options: dict
            common options used when calculating coherence times
        esys: tuple(evals, evecs)
            spectral data used during noise calculations 
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """
        common_noise_options = {} if common_noise_options is None else common_noise_options

        # If we're not given channels to consider, just use ones from the effective noise channels list
        noise_channels = [channel for
                          channel in self.effective_noise_channels()] if noise_channels is None else noise_channels

        # If we're given only a single channel as a string, make it a one entry list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        if esys is None:
            # We have to figure out the largest energy level involved in the calculations, to know how many levels we need
            # from the diagonalization.
            # This may be hidden in noise-channel-specific options, so have to search through those, if any were given.
            max_level = max(common_noise_options.get('i', 1), common_noise_options.get('j', 1))
            for noise_channel in noise_channels:
                if isinstance(noise_channel, tuple):
                    opts = noise_channel[1]
                    max_level = max(max_level, opts.get('i', 1), opts.get('j', 1))

            esys = self.eigensys(evals_count=max_level+1)

        rate = self._effective_rate(noise_channels=noise_channels, common_noise_options=common_noise_options, esys=esys,
                                    noise_type='tphi')

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def tphi_1_over_f(self, A_noise, i, j, noise_op, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the 1/f dephasing time (or rate) due to  arbitrary noise source. 

        We assume that the qubit energies (or the passed in eigenspectrum) has units 
        of frequency (and *not* angular frequency). 

        Parameters
        ----------
        A_noise: float
            noise strength
        i: int >=0
            state index that along with j defines a qubit
        j: int >=0
            state index that along with i defines a qubit
        noise_op: operator (ndarray)
            noise operator, typically Hamiltonian derivative w.r.t. noisy parameter
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.


        """
        # Sanity check
        if i == j or i < 0 or j < 0:
            raise ValueError("Level indices 'i' and 'j' must be different, and i,j>=0")

        p = {key: NOISE_PARAMS[key] for key in ['omega_low', 'omega_high', 't_exp']}
        p.update(kwargs)

        evals, evecs = self.eigensys(evals_count=max(j, i)+1) if esys is None else esys

        if isinstance(noise_op, np.ndarray):  # Check if the operator is given in dense form
            # if so, use numpy's vdot and dot
            rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, i])) -
                          np.vdot(evecs[:, j], np.dot(noise_op, evecs[:, j])))
        else:  # Else, we have a sparse operator, use it's own dot method.
            rate = np.abs(np.vdot(evecs[:, i], noise_op.dot(evecs[:, i])) -
                          np.vdot(evecs[:, j], noise_op.dot(evecs[:, j])))

        rate *= A_noise * np.sqrt(2 * np.abs(np.log(p['omega_low'] * p['t_exp'])))

        # We assume that the system energies are given in units of frequency and
        # not the angular frequency, hence we have to multiply by `2\pi`
        rate *= 2 * np.pi

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def tphi_1_over_f_flux(self, A_noise=NOISE_PARAMS['A_flux'], i=0, j=1, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the 1/f dephasing time (or rate) due to flux noise.

        Parameters
        ----------
        A_noise: float
            noise strength
        i: int >=0
            state index that along with j defines a qubit
        j: int >=0
            state index that along with i defines a qubit
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """

        if 'tphi_1_over_f_flux' not in self.supported_noise_channels():
            raise RuntimeError("Flux noise channel 'tphi_1_over_f_flux' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_flux(),
                                  esys=esys, get_rate=get_rate, **kwargs)

    def tphi_1_over_f_cc(self, A_noise=NOISE_PARAMS['A_cc'], i=0, j=1, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the 1/f dephasing time (or rate) due to critical current noise.

        Parameters
        ----------
        A_noise: float
            noise strength
        i: int >=0
            state index that along with j defines a qubit
        j: int >=0
            state index that along with i defines a qubit
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """

        if 'tphi_1_over_f_cc' not in self.supported_noise_channels():
            raise RuntimeError("Critical current noise channel 'tphi_1_over_f_cc' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_EJ(),
                                  esys=esys, get_rate=get_rate, **kwargs)

    def tphi_1_over_f_ng(self, A_noise=NOISE_PARAMS['A_ng'], i=0, j=1, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the 1/f dephasing time (or rate) due to charge noise.

        Parameters
        ----------
        A_noise: float
            noise strength
        i: int >=0
            state index that along with j defines a qubit
        j: int >=0
            state index that along with i defines a qubit
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time


        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """
        if 'tphi_1_over_f_ng' not in self.supported_noise_channels():
            raise RuntimeError("Charge noise channel 'tphi_1_over_f_ng' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_ng(),
                                  esys=esys, get_rate=get_rate, **kwargs)

    def t1(self, i, j, noise_op, spectral_density, total=True, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the transition time (or rate) using Fermi's Golden Rule due to a noise channel with
        a spectral density `spectral_density` and system noise operator `noise_op`. Mathematically, it reads:

        .. math::

            \frac{1}{T_1} = \frac{1}{\hbar^2} |\langle i| A_{\rm noise} | j \rangle|^2 S(\omega)

        We assume that the qubit energies (or the passed in eigenspectrum) has units 
        of frequency (and *not* angular frequency). 

        The `spectral_density` argument should be a callable object (typically a function) of one argument, 
        which is assumed to be an angular frequency (in the units currently set as system units. 

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        noise_op: operator (ndarray)
            noise operator
        spectral_density: callable object 
            defines a spectral density, must take one argument: `omega`
            (assumed to be in units of `2 \pi * <system units>`)
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time


        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.


        """
        # Sanity check
        if i == j or i < 0 or j < 0:
            raise ValueError("Level indices 'i' and 'j' must be different, and i,j>=0")

        evals, evecs = self.eigensys(evals_count=max(i, j)+1) if esys is None else esys

        # We assume that the energies in `evals` are given in the units of frequency and *not*
        # angular frequency. The function `spectral_density` is assumed to take as a parameter an
        # angular frequency, hence we have to convert.
        omega = 2 * np.pi * (evals[i]-evals[j])

        s = spectral_density(omega) + spectral_density(-omega) if total else spectral_density(omega)

        if isinstance(noise_op, np.ndarray):  # Check if the operator is given in dense form
            # if so, use numpy's vdot and dot
            rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, j])))**2 * s
        else:  # Else, we have a sparse operator, use it's own dot method.
            rate = np.abs(np.vdot(evecs[:, i], noise_op.dot(evecs[:, j])))**2 * s

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def t1_capacitive(self, i=1, j=0, Q_cap=None, T=NOISE_PARAMS['T'],  total=True,
                           esys=None, get_rate=False, **kwargs):
        r"""
        :math:`T_1` due to dielectric dissipation in the Jesephson junction capacitances. 

        References:  Nguyen et al (2019), Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Q_cap: numeric or callable
            capacitive quality factor; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """

        if 't1_capacitive' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_capacitive' is not supported in this system.")

        if Q_cap is None:
            # See Smith et al (2020)
            def q_cap_fun(omega): return 1e6 * (2 * np.pi * 6e9 / np.abs(units.to_standard_units(omega)))**0.7
        elif callable(Q_cap):  # Q_cap is a function of omega
            q_cap_fun = Q_cap
        else:  # Q_cap is given as a number
            def q_cap_fun(omega): return Q_cap

        def spectral_density(omega):
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * 8 * self.EC / q_cap_fun(omega) * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
            s *= 2 * np.pi  # We assume that system energies are given in units of frequency
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total,
                       esys=esys, get_rate=get_rate, **kwargs)

    def t1_charge_impedance(self, i=1, j=0, Z=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'], total=True,
                            esys=None, get_rate=False, **kwargs):
        r"""Noise due to charge coupling to an impedance (such as a transmission line).

        References: Schoelkopf et al (2003), Ithier et al (2005)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Z: float or callable
            impedance; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """

        if 't1_charge_impedance' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_charge_impedance' is not supported in this system.")

        Z_fun = Z if callable(Z) else lambda omega: Z

        def spectral_density(omega):
            # Note, our definition of Q_c is different from Zhang et al (2020) by a factor of 2
            Q_c = NOISE_PARAMS['R_k']/(8*np.pi * complex(Z_fun(omega)).real)
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * omega / Q_c * (1/np.tanh(0.5*therm_ratio)) / (1 + np.exp(-therm_ratio))
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total, esys=esys,
                       get_rate=get_rate, **kwargs)

    def t1_flux_bias_line(self, i=1, j=0, M=NOISE_PARAMS['M'],  Z=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'],
                          total=True,  esys=None, get_rate=False, **kwargs):
        r"""Noise due to a bias flux line. 
        
        References: Koch et al (2007), Groszkowski et al (2018)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        M: float
            Inductance in units of \Phi_0 / Ampere
        Z: complex, float or callable
            A complex impedance; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time


        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """

        if 't1_flux_bias_line' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_flux_bias_line' is not supported in this system.")

        Z_fun = Z if callable(Z) else lambda omega: Z

        def spectral_density(omega, Z=Z):
            """
            Our definitions assume that the noise_op is dH/dflux.
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * (2 * np.pi)**2 * M**2 * omega * sp.constants.hbar / complex(Z_fun(omega)).real  \
                * (1/np.tanh(0.5*therm_ratio)) / (1 + np.exp(-therm_ratio))
            # We assume that system energies are given in units of frequency
            # and that the noise operator to be used with this `spectral_density` is dH/dflux.
            # Hence we have to convert  2 powers of frequency to standard units
            s *= (units.to_standard_units(1))**2.0
            return s

        noise_op = self.d_hamiltonian_d_flux()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total, esys=esys,
                       get_rate=get_rate, **kwargs)

    def t1_inductive(self, i=1, j=0, Q_ind=None, T=NOISE_PARAMS['T'],  total=True,
                          esys=None, get_rate=False, **kwargs):
        r"""
        :math:`T_1` due to inductive dissipation in a superinductor.  

        References: Nguyen et al (2019), Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Q_ind: float or callable
            inductive quality factor; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

        """

        if 't1_inductive' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_inductive' is not supported in this system.")

        if Q_ind is None:
            # See Smith et al (2020)
            def q_ind_fun(omega):
                therm_ratio = abs(calc_therm_ratio(omega, T))
                therm_ratio_500MHz = calc_therm_ratio(2 * np.pi * 500e6, T, omega_in_standard_units=True)
                return 500e6 * (sp.special.kv(0, 1/2 * therm_ratio_500MHz) * np.sinh(1/2 * therm_ratio_500MHz)) \
                    / (sp.special.kv(0, 1/2 * therm_ratio) * np.sinh(1/2 * therm_ratio))   \

        elif callable(Q_ind):  # Q_ind is a function of omega
            q_ind_fun = Q_ind

        else:  # Q_ind is given as a number
            def q_ind_fun(omega): return Q_ind

        def spectral_density(omega):
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * self.EL / q_ind_fun(omega) * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
            s *= 2 * np.pi  # We assume that system energies are given in units of frequency
            return s

        noise_op = self.phi_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total,
                       esys=esys, get_rate=get_rate, **kwargs)

    def t1_quasiparticle_tunneling(self, i=1, j=0, Y_qp=None, x_qp=NOISE_PARAMS['x_qp'], T=NOISE_PARAMS['T'], Delta=NOISE_PARAMS['Delta'],
                                   total=True,  esys=None, get_rate=False, **kwargs):
        r"""Noise due to quasiparticle tunneling across a Josephson junction.

        References: Smith et al (2020), Catelani et al (2011), Pop et al (2014). 


        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Y_qp float or callable
            complex admittance; a fixed value or function of `omega`
        x_qp: float
            quasiparticle density (in units of eV)
        T: float
            temperature in Kelvin
        Delta: float 
            superconducting gap (in units of eV)
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tuple(ndarray, ndarray)
            evals, evecs tuple
        get_rate: bool
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.
        """

        if 't1_quasiparticle_tunneling' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_quasiparticle_tunneling' is not supported in this system.")

        if Y_qp is None:
            def y_qp_fun(omega):

                # Note that y_qp_fun is always symmetric in omega, i.e. In Smith et al 2020,
                # we essentially have something proportional to sinh(omega)/omega
                omega = abs(omega)

                Delta_in_Hz = convert_eV_to_Hz(Delta)
                omega_in_Hz = units.to_standard_units(omega) / (2*np.pi)
                EJ_in_Hz = units.to_standard_units(self.EJ)

                therm_ratio = calc_therm_ratio(omega, T)
                Delta_over_T = calc_therm_ratio(2 * np.pi * Delta_in_Hz, T, omega_in_standard_units=True)

                re_y_qp = np.sqrt(2/np.pi) * (8 / NOISE_PARAMS['R_k']) * (EJ_in_Hz / Delta_in_Hz)  \
                    * (2 * Delta_in_Hz/omega_in_Hz)**(3/2) * x_qp * np.sqrt(1/2 * therm_ratio) \
                    * sp.special.kv(0, 1/2 * abs(therm_ratio)) * np.sinh(1/2 * therm_ratio)

                return re_y_qp

        elif callable(Y_qp):  # Y_qp is a function of omega
            y_qp_fun = Y_qp

        else:  # Y_qp is given as a number
            def y_qp_fun(omega): return Y_qp

        def spectral_density(omega):
            """Eq. 38 in Catalani et al (2011). """
            therm_ratio = calc_therm_ratio(omega, T)
            return omega * NOISE_PARAMS['R_k'] / np.pi * complex(y_qp_fun(omega)).real  \
                * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))

        # In some literature the operator sin(phi/2) is used instead. 
        # This depends on whether one groups the flux with the quadratic or the cos term 
        # in the potential. 
        noise_op = self.cos_phi_operator(alpha=0.5)

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total,
                       esys=esys, get_rate=get_rate, **kwargs)
