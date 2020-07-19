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
def calc_therm_ratio(omega, T):
    """Returns the ratio 
    
    :math:`\beta \omega = \frac{\hbar \omega}{k_B T}`

    after converting `\omega` from system units, to standard units.

    Parameters
    ----------
    omega: float
        angular frequency in system units
    T: float 
        temperature in Kelvin

    Parameters
    ----------
    float

    """
    return (sp.constants.hbar * units.to_standard_units(omega)) / (sp.constants.k * T)


# Default values of various noise constants and parameters.
NOISE_PARAMS = {
    'A_flux': 1e-6,   # Flux noise strength. Units: Phi_0
    'A_ng': 1e-4,     # Charge noise strength. Units of charge e
    'A_cc': 1e-7,     # Critical current noise strength. Units of critical current I_c
    'omega_low': 1e-9 * 2 * np.pi,   # Low frequency cutoff. Units: 2pi GHz
    'omega_high': 3 * 2 * np.pi,     # High frequency cutoff. Units: 2pi GHz
    'Delta': 3.4e-4,  # Superconducting gap for aluminum (at T=0). Units: eV
    'x_qp': 1e-8,     # Quasiparticles density. # TODO add reference
    't_exp': 1e4,     # Measurement time. Units: ns
    'R_0': 50,        # Characteristic impedance of a transmission line. Units: Ohms
    'T': 0.015,       # Typical temperature for a superconducting circuit experiment. Units: K
    'M': 400,         # Mutual inductance between qubit and a flux line. Units: \Phi_0 / Ampere
    'R_k': sp.constants.h / sp.constants.e**2.0,  # Superconducting quantum resistance, aka Klitzing constant.
                                                  # Note, in some papers quantum resistance is defined as: h/(2e)^2
}


class NoisySystem:

    def plot_coherence_vs_paramvals(self, param_name, param_vals, noise_channels=None, common_noise_options={}, 
            spec_data=None, scale=1, num_cpus=settings.NUM_CPUS, **kwargs):
        r"""
        Show plots of various noise channels supported by the qubit.  

        Parameters
        ----------
        noise_channels: str or list(str) or list(tuple(str, dict))

        """
        # if we're not told what channels to consider, just use the supported list
        noise_channels = self.supported_noise_channels() if noise_channels is None else noise_channels

        # if we only have a single noise channel to consider (and hence are given a str), put it into a one element list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        if spec_data is None: 

            # We have to figure out the largest energy level involved in the calculations, to know how many levels we need
            # from the diagonalization.
            # This may be hidden in noise-channel-specific options, so have to search through those, if any were given.
            max_level = max(common_noise_options.get('i', 1), common_noise_options.get('j', 1))
            for noise_channel in noise_channels:
                if isinstance(noise_channel, tuple):
                    opts = noise_channel[1]
                    max_level = max(max_level, opts.get('i', 1), opts.get('j', 1))

            spec_data = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=max_level+1,
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

        # remember current value of param_name
        current_val = getattr(self, param_name)

        for n, noise_channel in enumerate(noise_channels):

            # noise_channel is a string representing the noise method
            if isinstance(noise_channel, str):

                # calculate the noise over the full param span in param_vals
                noise_vals = [scale * getattr(self.set_and_return(param_name, v), noise_channel)(
                             esys=(spec_data.energy_table[v_i, :], spec_data.state_table[v_i]), 
                             **common_noise_options)
                             for v_i, v in enumerate(param_vals)]

                ax = axes.ravel()[n] if len(noise_channels) > 1 else axes
                ax.plot(param_vals, noise_vals, **plotting._extract_kwargs_options(kwargs, 'plot'))
                ax.set_title(noise_channel)

                ax.set_xlabel(param_name)
                ax.set_ylabel(units.get_units_time_label())
                ax.set_yscale("log")
                ax.grid(True)

                plotting._process_options(fig, ax, **kwargs)

            # noise_channel is a tuple representing the noise method and default options
            elif isinstance(noise_channel, tuple):

                nc, opts = noise_channel

                # Some of the options for this noise channel, may be in conflict with the common options that the user
                # specified. In such cases, we let the noise-channel-specific options take priority.
                options=common_noise_options.copy()
                for key, val in opts.items():
                    options[key]=val

                # calculate the noise over the full param span in param_vals
                noise_vals = [scale * getattr(self.set_and_return(param_name, v), nc)(
                              esys=(spec_data.energy_table[v_i, :], spec_data.state_table[v_i]), 
                              **options)
                              for v_i, v in enumerate(param_vals)]

                ax = axes.ravel()[n] if len(noise_channels) > 1 else axes
                ax.plot(param_vals, noise_vals, **plotting._extract_kwargs_options(kwargs, 'plot'))
                ax.set_title(nc)

                ax.set_xlabel(param_name)
                ax.set_ylabel(units.get_units_time_label())
                ax.set_yscale("log")
                ax.grid(True)

                plotting._process_options(fig, ax, **kwargs)

            else:
                raise ValueError("The `noise_channels` argument should be one of {str, list of str, or list of tuples}.")

        # Set the parameter we varied to its initial value
        setattr(self, param_name, current_val)

        fig.tight_layout()

        return fig, axes

    def _effective_rate(self, noise_channels, common_noise_options, esys, noise_type):
        """
        Helper method used when calculating the effective rates related to t1_effective and tphi_effecive.
        """
        rate = 0.0

        for n, noise_channel in enumerate(noise_channels):

            # noise_channel is a string representing the noise method
            if isinstance(noise_channel, str):

                # If dealing with a tphi noise type, the contribution of a t1 process to the dephasing rate its halved.
                scale_factor = 0.5 if noise_type == 'tphi' and noise_channel.startswith("t1") else 1

                noise_channel_method=noise_channel

                options=common_noise_options.copy()
                # We need to make sure we calculate a rate
                options['get_rate']=True

                # calculate the noise over the full param span in param_vals
                rate += scale_factor * getattr(self, noise_channel_method)(esys=esys, **options)

            # noise_channel is a tuple representing the noise method and default options
            elif isinstance(noise_channel, tuple):

                # If dealing with a tphi noise type, the contribution of a t1 process to the dephasing rate its halved.
                scale_factor = 0.5 if noise_type == 'tphi' and noise_channel.startswith("t1") else 1

                noise_channel_method=noise_channel[0]

                options=common_noise_options.copy()
                # Some of the channel-specific options may be in conflict with the common options options.
                # In such a case, we let the channel-specific options take priority.
                options.update(noise_channel[1])
                # We need to make sure we calculate a rate
                options['get_rate']=True

                # calculate the noise over the full param span in param_vals
                rate += scale_factor * getattr(self, noise_channel_method)(esys=esys, **options)

            else:
                raise ValueError("The `noise_channels` argument should be one of {str, list of str, or list of tuples}.")

        return rate

    def t1_effective(self, noise_channels=None, common_noise_options={}, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the effective t1 time (or rate). 

        Parameters
        ----------
        noise_channels: str or list(str) or list(tuple(str, dict))
            noise channels that should contribute to t1_effective

        """
        # If we're not given channels to consider, just use the supported list that
        # correspond to t1 processes
        noise_channels = [channel for channel in self.supported_noise_channels()
                          if channel.startswith('t1')] if noise_channels is None else noise_channels

        # If we're given only a single channel as a string, make it a one entry list
        noise_channels = [noise_channels] if isinstance(noise_channels, str) else noise_channels

        # Do a sanity check; if we're given a tphi channel, raise an exception
        for noise_channel in noise_channels:
            channel = noise_channel[0] if isinstance(noise_channel, tuple) else noise_channel
            if not channel.startswith("t1"):
                raise ValueError("Only t1 channels can contribute to effective t1 noise.")

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

        rate=self._effective_rate(noise_channels=noise_channels, 
                                            common_noise_options=common_noise_options,
                                            esys=esys,
                                            noise_type='t1')
        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def tphi_effective(self, noise_channels=None, common_noise_options={}, esys=None, get_rate=False, **kwargs):
        r"""
        Calculate the effective tphi time (or rate). 

        Parameters
        ----------
        noise_channels: str or list(str) or list(tuple(str, dict))
            noise channels that should contribute to t1_effective

        """

        # If we're not given channels to consider, just use ones fromthe supported list
        noise_channels = [channel for 
                channel in self.supported_noise_channels()] if noise_channels is None else noise_channels

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

        rate=self._effective_rate(noise_channels=noise_channels, 
                                            common_noise_options=common_noise_options,
                                            esys=esys,
                                            noise_type='tphi')

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def tphi_1_over_f(self, A_noise, i, j, noise_op, esys=None, get_rate=False, **params):
        r"""
        Calculate the 1/f dephasing time (or rate) due to arbitrary noise source. 

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
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        float
        """
        # Sanity check
        if i == j or i < 0 or j < 0:
            raise ValueError("Level indices 'i' and 'j' must be different, and i,j>=0")

        p = {key: NOISE_PARAMS[key] for key in ['omega_low', 'omega_high', 't_exp']}
        p.update(params)

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

    def tphi_1_over_f_flux(self, A_noise=NOISE_PARAMS['A_flux'], i=0, j=1, esys=None, get_rate=False, **params):
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
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        float
        """

        if 'tphi_1_over_f_flux' not in self.supported_noise_channels():
            raise RuntimeError("Flux noise channel 'tphi_1_over_f_flux' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_flux(),
                                  esys=esys, get_rate=get_rate, **params)

    def tphi_1_over_f_cc(self, A_noise=NOISE_PARAMS['A_cc'], i=0, j=1, esys=None, get_rate=False, **params):
        """
        Calculate the 1/f dephasing time (or rate) due to critical current noise.

        Parameters
        ----------
        A_noise: float
            noise strength
        i: int >=0
            state index that along with j defines a qubit
        j: int >=0
            state index that along with i defines a qubit
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        float
        """

        if 'tphi_1_over_f_cc' not in self.supported_noise_channels():
            raise RuntimeError("Critical current noise channel 'tphi_1_over_f_cc' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_EJ(),
                                  esys=esys, get_rate=get_rate, **params)

    def tphi_1_over_f_ng(self, A_noise=NOISE_PARAMS['A_ng'], i=0, j=1, esys=None, get_rate=False, **params):
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
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        float
        """
        if 'tphi_1_over_f_ng' not in self.supported_noise_channels():
            raise RuntimeError("Charge noise channel 'tphi_1_over_f_ng' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_ng(),
                                  esys=esys, get_rate=get_rate, **params)

    def t1(self, i, j, noise_op, spectral_density, total=True, esys=None, get_rate=False, **params):
        r"""
        Calculate the transition time (or rate) using Fermi's Golden Rule due to a noise channel with
        a spectral density `spectral_density` and system noise operator `noise_op`. Mathematically, it reads:

        .. math::

            \frac{1}{T_1} = \frac{1}{\hbar^2} |\langle i| A_{\rm noise} | j \rangle|^2 S(\omega)

        We assume that the qubit energies (or the passed in eigenspectrum) has units 
        of frequency (and *not* angular frequency). 

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        noise_op: operator (ndarray)
            noise operator
        spectral_density: callable object 
            defines a spectral density, must take one argument: `omega` (assumed to be in units of `2 \pi * <system units>`)
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        decay rate or time: float
            decay rate in units of `2\pi * <system units>`. Alternatively, 1/rate can be returned. 

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

    def t1_capacitive_loss(self, i=1, j=0, EC=None, Q_cap=None, T=NOISE_PARAMS['T'],  total=True,
                           esys=None, get_rate=False, **params):
        r"""
        Loss due to dielectric dissipation in the Jesephson junction capacitances. 

        References: Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        EC: float
            capacitive energy (in frequency units)
        Q_cap: numeric or callable
            capacitive quality factor; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        decay rate or time: float
            decay rate in units of `2\pi * <system units>`. Alternatively, 1/rate can be returned. 
        """

        if 't1_capacitive_loss' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_capacitive_loss' is not supported in this system.")

        EC = self.EC if EC is None else EC

        if Q_cap is None:
            # See Smith et al (2020)
            q_cap_fun=lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(units.to_standard_units(omega)))**0.7
        elif callable(Q_cap): # Q_cap is a function of omega
            q_cap_fun=Q_cap
        else: # Q_cap is given as a number
            q_cap_fun=lambda omega: Q_cap
            
        def spectral_density(omega):
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * 8 * EC / q_cap_fun(omega) * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
            s *= 2 * np.pi  # We assume that system energies are given in units of frequency
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total,
                       esys=esys, get_rate=get_rate, **params)

    def t1_inductive_loss(self, i=1, j=0, EL=None, Q_ind=None, T=NOISE_PARAMS['T'],  total=True,
                          esys=None, get_rate=False, **params):
        r"""
        Loss due to inductive dissipation in a superinductor.  

        TODO check factor of 1/2 definition in EL; should be the same in all qubits, otherwise we'll get this wrong.
            In that case, could overwrite it for some qubits. 

        References: Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        EL: float
            inductive energy (in frequency units)
        Q_ind: float or callable
            inductive quality factor; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        decay rate or time: float
            decay rate in units of `2\pi * <system units>`. Alternatively, 1/rate can be returned. 
        """

        if 't1_inductive_loss' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_inductive_loss' is not supported in this system.")

        EL = self.EL if EL is None else EL

        if Q_ind is None:
            # See Smith et al (2020)
            q_ind_fun = lambda omega: 500e6
        elif callable(Q_ind): # Q_ind is a function of omega
            q_ind_fun=Q_ind
        else: # Q_ind is given as a number
            q_ind_fun=lambda omega: Q_cap

        def spectral_density(omega):
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * EL / q_ind_fun(omega) * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
            s *= 2 * np.pi  # We assume that system energies are given in units of frequency
            return s

        noise_op = self.phi_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total,
                       esys=esys, get_rate=get_rate, **params)

    def t1_charge_impedance(self, i=1, j=0, Z=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'], total=True,
                            esys=None, get_rate=False, **params):
        r"""Noise due to charge coupling to an impedance (such as a transmission line).

        References: Clerk et al (2010), also Zhang et al (2020) - note different definition of R_k (i.e. their R_q)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Z: float or callable
            potentially complex impedance; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        decay rate or time: float
            decay rate in units of `2\pi * <system units>`. Alternatively, 1/rate can be returned. 
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
                       get_rate=get_rate, **params)

    def t1_flux_bias_line(self, i=1, j=0, M=NOISE_PARAMS['M'],  Z=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'],
                          total=True,  esys=None, get_rate=False, **params):
        r"""Noise due to a bias flux line.
        
        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        M: float
            Impedance in units of \Phi_0 / Ampere
        Z: float or callable
            potentially complex impedance; a fixed value or function of `omega`
        T: float
            temperature in Kelvin
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        decay rate or time: float
            decay rate in units of `2\pi * <system units>`. Alternatively, 1/rate can be returned. 
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
            # (TODO this is ugly; what's a cleaner way to do this? )
            s *= (units.to_standard_units(1))**2.0
            return s

        noise_op = self.d_hamiltonian_d_flux()

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density, total=total, esys=esys,
                       get_rate=get_rate, **params)

    def t1_quasiparticle_tunneling(self, i=1, j=0, Y_qp=None, Delta=NOISE_PARAMS['Delta'], x_qp=NOISE_PARAMS['x_qp'],
                                   T=NOISE_PARAMS['T'], total=True,  esys=None, get_rate=False, **params):
        """Noise due quasiparticle tunneling across a Josephson junction.

        References: Smith et al (2020)

        TODO 
            - Careful about correctness/applicability of this. Seems this strongly depends on admitance each junction sees.
            - Need to check the factor of 1/2 in the operator
        """

        if 't1_quasiparticle_tunneling' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_quasiparticle_tunneling' is not supported in this system.")

        if Y_qp is None:
            # TODO implement a fancy omega-dependent function; how does it differ for different qubits?
            # Namely, should we calculate based on each qubit's topology (i.e. how is are the junctions shunted)?
            y_qp_fun = lambda omega: 1000  # dummy for now
        elif callable(Y_qp): # Y_qp is a function of omega
            y_qp_fun = Y_qp
        else: # Y_qp is given as a number
            y_qp_fun = lambda omega: Y_qp

        def spectral_density(omega, Y_qp=Y_qp):
            """
            Our definitions assume that the noise_op is dH/dflux.
            TODO finish this
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s = NOISE_PARAMS['R_k'] * y_qp_fun(omega) * omega / np.pi * (1/np.tanh(0.5*therm_ratio)) / (1 + np.exp(-therm_ratio))
            return s

        noise_op = self.sin_phi_operator(alpha=0.5)

        return self.t1(i=i, j=j, noise_op=noise_op, spectral_density=spectral_density,  total=total,
                       esys=esys, get_rate=get_rate, **params)
