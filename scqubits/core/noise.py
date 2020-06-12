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

import numpy as np
import scipy as sp
import scipy.constants
import scqubits.utils.misc as utils


# Helpers for units conversion
def temp_to_ghz(T):
    """Converts temperature in Kelvin to GHz"""
    return sp.constants.k * T / sp.constants.h * 1e-9


def temp_to_mhz(T):
    """Converts temperature in Kelvin to MHz"""
    return sp.constants.k * T / sp.constants.h * 1e-6


def temp_to_khz(T):
    """Converts temperature in Kelvin to kHz"""
    return sp.constants.k * T / sp.constants.h * 1e-3


def ghz_to_temp(freq):
    """Converts frequency in GHz to Kelvin"""
    return freq * sp.constants.h / sp.constants.k * 1e9


def mhz_to_temp(freq):
    """Converts frequency in MHz to Kelvin"""
    return freq * sp.constants.h / sp.constants.k * 1e6


def khz_to_temp(freq):
    """Converts frequency in MHz to Kelvin"""
    return freq * sp.constants.h / sp.constants.k * 1e3


# Default values of various noise constants and parameters.
NOISE_PARAMS = {
    # Units: Phi_0
    'A_flux': 1e-6,

    # Units of charge e
    'A_ng': 1e-4,

    # units of critical current i_c
    'A_cc': 1e-7,

    # Units: 2pi GHz
    'omega_low': 1e-9 * 2 * np.pi,
    # Units: 2pi GHz
    'omega_high': 3 * 2 * np.pi,

    # Capacitive quality factor, see Smith et al. npj Quant. Info. 6:8, (2020)
    # TODO: This assumes specific units! Need to have a better way...
    'Q_cap': lambda energy: 1e6 * (6 / np.abs(energy))**0.7,

    # Capacitive quality factor, see Smith et al. npj Quant. Info. 6:8, (2020)
    # TODO: Use full functional, freq-dependent form
    'Q_ind': 500e6,

    # Units: ns
    't_exp': 1e4,

    # Units: Ohms
    'R_0': 50,

    # Units: GHz
    'T': temp_to_ghz(0.015),

    # Mutual inductance. Units: \Phi_0 / Amperes
    'M': 1000,

    # Superconducting quantum of resistance (~6.5 k\Omega)
    'R_q': sp.constants.h/(2*sp.constants.e)**2.0,
}


class NoisySystem:

    def tphi_1_over_f(self, A_noise, i, j, noise_op, esys=None, get_rate=False, **params):
        """
        Calculate the 1/f dephasing time (or rate) due to arbitrary noise source. 

        Assumptions:
        ----------
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

        p = {key: NOISE_PARAMS[key] for key in ['omega_low', 'omega_high', 't_exp']}
        p.update(params)

        evals, evecs = self.eigensys(max(j, i)+1) if esys is None else esys

        rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, i])) -
                np.vdot(evecs[:, j], np.dot(noise_op, evecs[:, j])))

        rate *= A_noise * np.sqrt(2 * np.abs(np.log(p['omega_low'] * p['t_exp'])))

        # We assume that the system energies are given in frequency and not
        # angular frequency
        rate *= 2 * np.pi

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def tphi_1_over_f_flux(self, A_noise=NOISE_PARAMS['A_flux'], i=0, j=1, esys=None, get_rate=False, **params):
        """
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
        """
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

    def t1(self, i, j, noise_op, spec_dens, total=False, esys=None, get_rate=False, **params):
        """
        Calculate the transition time (or rate) using Fermi's Golden Rule due to a noise channel with
        a spectral density `spec_dens` and system noise operator `noise_op`. Mathematically, we have:

        :math:` \frac{1}{T_1} =  |\langle i| A_{\rm noise} | j \rangle|^2 S(energy)

        noting that we abosorbe `hbar` into `noise_op` variable (i.e.: `noise_op = A_{\rm noise}/\hbar`).

        Assumptions:
        ----------
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
        spec_dens: callable object 
            defines a spectral desnity, must take one argument: energy (in frequency units)
        total: bool
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys: tupple(ndarray, ndarray)
            evals, evecs tupple
        get_rate: bool
            get rate or time

        Returns
        -------
        float

        """
        if i == j or i < 0 or j < 0:
            raise ValueError("Level indices 'i' and 'j' must be different, and i,j>=0")

        evals, evecs = self.eigensys(max(i, j)+1) if esys is None else esys
        # We assume that the energies are provided in the units of frequency and *not*
        # angular frequency.
        energy = (evals[i]-evals[j])
        s = spec_dens(energy) + spec_dens(-energy) if total else spec_dens(energy)

        rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, j])))**2 * s

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def t1_capacitive_loss(self, i, j, EC=None, Q_cap=NOISE_PARAMS['Q_cap'], T=NOISE_PARAMS['T'],  total=False,
                            esys=None, get_rate=False, **params):

        if 't1_capacitive_loss' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_capacitive_loss' is not supported in this system.")

        # We assume EC is given in the units of frequency
        EC = self.EC if EC is None else EC

        def spec_dens(energy):
            q_cap = Q_cap(energy) if callable(Q_cap) else Q_cap
            s = 2 * 8 * EC / q_cap * (1/np.tanh(np.abs(energy) / (2 * T))) / (1 + np.exp(-energy/T))
            s *= 2 * np.pi  # We assume the units of EC are given as frequency
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, total=total,
                esys=esys, get_rate=get_rate, **params)

    def t1_inductive_loss(self, i, j, EL=None, Q_ind=NOISE_PARAMS['Q_ind'], T=NOISE_PARAMS['T'],  total=False,
                            esys=None, get_rate=False, **params):
        """
        TODO check factor of 1/2 definition in EL; should be the same in all qubits, otherwise we'll get this wrong.
        """

        if 't1_inductive_loss' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_inductive_loss' is not supported in this system.")

        # We assume EC is given in the units of frequency
        EL = self.EL if EL is None else EL

        def spec_dens(energy):
            q_ind = Q_ind(energy) if callable(Q_ind) else Q_ind
            s = 2 * EL / q_ind * (1/np.tanh(np.abs(energy) / (2 * T))) / (1 + np.exp(-energy/T))
            s *= 2 * np.pi  # We assume the units of EC are given as frequency
            return s

        noise_op = self.phi_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, total=total,
                esys=esys, get_rate=get_rate, **params)

    def t1_tran_line(self, i, j, Z=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'], total=False,
                      esys=None, get_rate=False, **params):
        """Noise due to a capacitive coupling to a transmission line. 

        TODO: update the form here; want to use capacitance ratio to define how strongly
        the line is coupled?
        """
        if 't1_tran_line' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_tran_line' is not supported in this system.")

        def spec_dens(energy):
            Q_c = NOISE_PARAMS['R_q']/(16*np.pi*Z)
            s = energy/Q_c * (1/np.tanh(energy / (2 * T))) / (1 + np.exp(-energy/T))
            s *= 2 * np.pi  # We assume the units of EC are given as frequency
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys,
                       get_rate=get_rate, total=False, **params)


    def t1_flux_bias_line(self, i, j, M=NOISE_PARAMS['M'],  Z=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'],
                           total=False,  esys=None, get_rate=False, **params):
        """Noise due to a bias flux line.
        """

        if 't1_bias_flux_line' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_bias_flux_line' is not supported in this system.")

        def spec_dens(omega):
            return M**2 * 2 * sp.constants.hbar * omega / Z  \
                * (1 + 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T)))

        noise_op = self.d_hamiltonian_d_flux()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)


    # def t1_tran_line(self, i, j, Z=NOISE_PARAMS['R_0'], kbT=NOISE_PARAMS['kbT'], esys=None, get_rate=False, **params):
        # """Noise due to a capacitive coupling to a transmission line.

        # TODO Could be more rigorous in splitting up terms to S(omega) and prefactor before
            # passing to self.t1()
        # """
        # if 't1_tran_line' not in self.supported_noise_channels():
            # raise RuntimeError("Noise channel 't1_tran_line' is not supported in this system.")

        # def spec_dens(omega):
            # Q_c = NOISE_PARAMS['R_q']/(16*np.pi*Z)
            # return omega/Q_c * 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T))

        # noise_op = self.n_operator()

        # return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)

    # def t1_dielectric_loss(self, i, j, tan_delta=NOISE_PARAMS['tan_delta'], T=NOISE_PARAMS['T'],  total=False,
                            # esys=None, get_rate=False, **params):
        # """Noise due to a capacitive dielectric loss.

        # TODO double check formula, tan_delta in particular
        # TODO rewrite this in terms of the charge operator (why does Maryland always use phi??)
        # TODO Cleanup up needed: Could be more rigorous in splitting up terms to S(omega) and prefactor before
            # passing to self.t1()
        # """

        # if 't1_dielectric_loss' not in self.supported_noise_channels():
            # raise RuntimeError("Noise channel 't1_dielectric_loss' is not supported in this system.")

        # def spec_dens(omega):
            # return sp.constants.hbar * omega**2 * tan_delta / (8 * self.EC)  \
                # * (1 + 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T)))

        # noise_op = self.phi_operator()

        # return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)
