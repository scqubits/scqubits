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
# import scipy.constants
import scqubits.utils.misc as utils


# Default values of various noise constants and parameters.
CONSTANTS = {
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

    # loss tangent
    'tan_delta': 8e-6,

    # Units: ns
    't_exp': 1e4,

    # Units: Ohms
    'R_0': 50,

    # Units: Kelvin
    'T': 0.015,

    # Mutual inductance. Units: \Phi_0 / Amperes
    'M': 1000,

    # Superconducting quantum of resistance (~6.5 k\Omega)
    'R_q': sp.constants.h/(2*sp.constants.e)**2.0,
}


class NoisySystem:

    def _supported_noise_channels(self):
        """
        Every noisy system should implement this method which returns a list of supported noise channels.
        """
        raise NotImplemented("_supported_noise_channels() method is not implemented")

    def tphi_1_over_f(self, A_noise, i, j, noise_op, esys=None, get_rate=False, **params):
        """
        TODO fix units issues 
        """

        p = {key: CONSTANTS[key] for key in ['omega_low', 'omega_high', 't_exp']}
        p.update(params)

        evals, evecs = self.eigensys(max(j, i)+1) if esys is None else esys

        rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, i])) -
                np.vdot(evecs[:, j], np.dot(noise_op, evecs[:, j])))

        rate *= A_noise * np.sqrt(2 * np.abs(np.log(p['omega_low'] * p['t_exp'])))

        # TODO: figure out how to deal with units in a better way
        # for now assume units of noise_op are in [frequency units]/[noise type units]
        # (i.e. from  \partial H / \partial \lambda)
        # hence we have to multiply by 2\pi in order to get the 1/rate to give us time.
        rate *= 2 * np.pi

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def tphi_1_over_f_flux(self, A_noise=CONSTANTS['A_flux'], i=0, j=1, esys=None, get_rate=False, **params):

        if 'tphi_1_over_f_flux' not in self._supported_noise_channels():
            raise RuntimeError("Flux noise channel 'tphi_1_over_f_flux' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_flux(),
                                            esys=esys, get_rate=get_rate, **params)

    def tphi_1_over_f_cc(self, A_noise=CONSTANTS['A_cc'], i=0, j=1, esys=None, get_rate=False, **params):

        if 'tphi_1_over_f_cc' not in self._supported_noise_channels():
            raise RuntimeError("Critical current noise channel 'tphi_1_over_f_cc' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_EJ(),
                                            esys=esys, get_rate=get_rate, **params)

    def tphi_1_over_f_ng(self, A_noise=CONSTANTS['A_ng'], i=0, j=1, esys=None, get_rate=False, **params):

        if 'tphi_1_over_f_ng' not in self._supported_noise_channels():
            raise RuntimeError("Charge noise channel 'tphi_1_over_f_ng' is not supported in this system.")

        return self.tphi_1_over_f(A_noise=A_noise, i=i, j=j, noise_op=self.d_hamiltonian_d_ng(),
                                            esys=esys, get_rate=get_rate, **params)

    def t1(self, i, j, noise_op, spec_dens, esys=None, get_rate=False, **params):
        """
        TODO fix units issues 
        """

        evals, evecs = self.eigensys(max(i, j)+1) if esys is None else esys
        omega = 2 * np.pi * (evals[i]-evals[j])

        rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, j])))**2 \
                    * spec_dens(omega * 1e9)  # UNITS HACK TEMPORARY #####

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def t1_dielectric_loss(self, i, j, tan_delta=CONSTANTS['tan_delta'], T=CONSTANTS['T'], esys=None,
                            get_rate=False, **params):
        """Noise due to a capacitive dielectric loss.

        TODO double check formula, tan_delta in particular
        TODO rewrite this in terms of the charge operator (why does everyone use phi??)
        """

        if 't1_dielectric_loss' not in self._supported_noise_channels():
            raise RuntimeError("Noise channel 't1_dielectric_loss' is not supported in this system.")

        def spec_dens(omega):
            return sp.constants.hbar * omega**2 * tan_delta / (8 * self.EC)  \
                * 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T))

        noise_op = self.phi_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)

    def t1_flux_bias_line(self, i, j, M=CONSTANTS['M'],  Z=CONSTANTS['R_0'], T=CONSTANTS['T'], esys=None,
                            get_rate=False, **params):
        """Noise due to a bias flux line.
        """

        if 't1_bias_flux_line' not in self._supported_noise_channels():
            raise RuntimeError("Noise channel 't1_bias_flux_line' is not supported in this system.")

        def spec_dens(omega):
            return M**2 * 2 * sp.constants.hbar * omega / Z  \
                * (1 + 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T)))

        noise_op = self.d_hamiltonian_d_flux()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)

    def t1_tran_line(self, i, j, Z=CONSTANTS['R_0'], T=CONSTANTS['T'], esys=None, get_rate=False, **params):
        """Noise due to a capacitive coupling to a transmission line. 

        TODO Could be more rigorous in splitting up terms to S(omega) and prefactor before
            passing to self.t1()
        """
        if 't1_tran_line' not in self._supported_noise_channels():
            raise RuntimeError("Noise channel 't1_tran_line' is not supported in this system.")

        def spec_dens(omega):
            Q_c = CONSTANTS['R_q']/(16*np.pi*Z)
            return omega/Q_c * 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T))

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)

    # def t1_tran_line(self, i, j, Z=CONSTANTS['R_0'], T=CONSTANTS['T'], esys=None, get_rate=False, **params):
        # """Noise due to a capacitive coupling to a transmission line.

        # TODO Could be more rigorous in splitting up terms to S(omega) and prefactor before
            # passing to self.t1()
        # """
        # if 't1_tran_line' not in self._supported_noise_channels():
            # raise RuntimeError("Noise channel 't1_tran_line' is not supported in this system.")

        # def spec_dens(omega):
            # Q_c = CONSTANTS['R_q']/(16*np.pi*Z)
            # return omega/Q_c * 1/np.tanh(sp.constants.hbar * omega / (2 * sp.constants.k * T))

        # noise_op = self.n_operator()

        # return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)
