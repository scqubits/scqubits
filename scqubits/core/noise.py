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
import scqubits.core.units as units


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
    # Units: Phi_0
    'A_flux': 1e-6,

    # Units of charge e
    'A_ng': 1e-4,

    # units of critical current I_c
    'A_cc': 1e-7,

    # Units: 2pi GHz
    'omega_low': 1e-9 * 2 * np.pi,
    # Units: 2pi GHz
    'omega_high': 3 * 2 * np.pi,

    # Capacitive quality factor, see Smith et al. npj Quant. Info. 6:8, (2020)
    'Q_cap': lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(units.to_standard_units(omega)))**0.7,

    # Capacitive quality factor, see Smith et al. npj Quant. Info. 6:8, (2020)
    # TODO: Use full functional, freq-dependent form
    'Q_ind': 500e6,

    # Units: ns
    't_exp': 1e4,

    # Units: Ohms
    'R_0': 50,

    # Units: K
    'T': 0.015,

    # Mutual inductance between qubit and a flux line. Units: \Phi_0 / Amperes
    'M': 400,

    # Superconducting quantum of resistance (~6.5 k\Omega)
    # Note, in some papers this is defined as: h/e^2 
    # and called the Klitzing constant. 
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
        # Sanity check
        if i == j or i < 0 or j < 0:
            raise ValueError("Level indices 'i' and 'j' must be different, and i,j>=0")

        p = {key: NOISE_PARAMS[key] for key in ['omega_low', 'omega_high', 't_exp']}
        p.update(params)

        evals, evecs = self.eigensys(evals_count=max(j, i)+1) if esys is None else esys

        rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, i])) -
                np.vdot(evecs[:, j], np.dot(noise_op, evecs[:, j])))

        rate *= A_noise * np.sqrt(2 * np.abs(np.log(p['omega_low'] * p['t_exp'])))

        # We assume that the system energies are given in units of frequency and
        # not the angular frequency, hence we have to multiply by `2\pi`
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
        a spectral density `spec_dens` and system noise operator `noise_op`. Mathematically, it reads:

        :math:` \frac{1}{T_1} = \frac{1}{\hbar^2} |\langle i| A_{\rm noise} | j \rangle|^2 S(energy)

        Here we calculate

        :math:` \frac{1}{T_1} = |\langle i| noise_op noise} | j \rangle|^2 spec_dens(energy)

        Hence the units and prefactors have to be appropriately absorbed into function arguments. 

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
        # angular frequency. The function `spec_dens` is assumed to take as a parameter an
        # angular frequency, hence we have to convert.
        omega = 2 * np.pi * (evals[i]-evals[j])

        s = spec_dens(omega) + spec_dens(-omega) if total else spec_dens(omega)

        rate = np.abs(np.vdot(evecs[:, i], np.dot(noise_op, evecs[:, j])))**2 * s

        if get_rate:
            return rate
        else:
            return 1/rate if rate != 0 else np.inf

    def t1_capacitive_loss(self, i, j, EC=None, Q_cap=NOISE_PARAMS['Q_cap'], T=NOISE_PARAMS['T'],  total=False,
                            esys=None, get_rate=False, **params):
        """
        Loss due to dielectric dissipation in the Jesephson junction capacitances. 

        Reference: Smith et al (2020)
        """

        if 't1_capacitive_loss' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_capacitive_loss' is not supported in this system.")

        EC = self.EC if EC is None else EC

        def spec_dens(omega, Q_cap=Q_cap):
            q_cap = Q_cap(omega) if callable(Q_cap) else Q_cap
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * 8 * EC / q_cap * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
            s *= 2 * np.pi  # We assume that system energies are given in units of frequency 
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, total=total,
                esys=esys, get_rate=get_rate, **params)

    def t1_inductive_loss(self, i, j, EL=None, Q_ind=NOISE_PARAMS['Q_ind'], T=NOISE_PARAMS['T'],  total=False,
                            esys=None, get_rate=False, **params):
        """
        TODO check factor of 1/2 definition in EL; should be the same in all qubits, otherwise we'll get this wrong.
            In that case, could overwrite it for some qubits. 

        Reference: Smith et al (2020)
        """

        if 't1_inductive_loss' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_inductive_loss' is not supported in this system.")

        EL = self.EL if EL is None else EL

        def spec_dens(omega, Q_ind=Q_ind):
            q_ind = Q_ind(omega) if callable(Q_ind) else Q_ind
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * EL / q_ind * (1/np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
            s *= 2 * np.pi  # We assume that system energies are given in units of frequency 
            return s

        noise_op = self.phi_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, total=total,
                esys=esys, get_rate=get_rate, **params)


    def t1_charge_impedance(self, i, j, ReZ=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'], total=False,
                      esys=None, get_rate=False, **params):
        """Noise due to charge coupling to an impedance (such as a transmission line).

        Reference: Clerk et al (2010), also Zhang et al (2020) - note different definition of R_q
        """
        if 't1_charge_impedance' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_charge_impedance' is not supported in this system.")

        def spec_dens(omega, ReZ=ReZ):
            ReZ = ReZ(omega) if callable(ReZ) else ReZ
            # Note, our definition of R_q is different from Zhang et al (2020) by a factor of 1/4
            Q_c = NOISE_PARAMS['R_q']/(2*np.pi*ReZ)
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * omega / Q_c * (1/np.tanh(0.5*therm_ratio)) / (1 + np.exp(-therm_ratio))
            return s

        noise_op = self.n_operator()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys,
                       get_rate=get_rate, total=False, **params)

    def t1_flux_bias_line(self, i, j, M=NOISE_PARAMS['M'],  ReZ=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'],
                           total=False,  esys=None, get_rate=False, **params):
        """Noise due to a bias flux line.
        """

        if 't1_flux_bias_line' not in self.supported_noise_channels():
            raise RuntimeError("Noise channel 't1_flux_bias_line' is not supported in this system.")

        def spec_dens(omega, ReZ=ReZ):
            """
            Our definitions assume that the noise_op is dH/dflux.
            """
            ReZ = ReZ(omega) if callable(ReZ) else ReZ
            therm_ratio = calc_therm_ratio(omega, T)
            s = 2 * (2 * np.pi)**2 * M**2 * omega * sp.constants.hbar / ReZ  \
                    * (1/np.tanh(0.5*therm_ratio)) / (1 + np.exp(-therm_ratio))
            # We assume that system energies are given in units of frequency
            # and that the noise operator to be used with this `spec_dens` is dH/dflux.
            # Hence we have to convert  2 powers of frequency to standard units
            # (TODO this is ugly; what's a cleaner way to do this? )
            s *= (units.to_standard_units(1))**2.0
            return s

        noise_op = self.d_hamiltonian_d_flux()

        return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)

    # TODO add quasi particle tunneling; depends on admitance each junction sees. 
    # In principle this can vary quite a bit in differnt qubits 
    # def t1_quasiparticle_tunneling(self, i, j, ReYqp=NOISE_PARAMS['Y'],  ReZ=NOISE_PARAMS['R_0'], T=NOISE_PARAMS['T'],
                           # total=False,  esys=None, get_rate=False, **params):
        # """Noise due to a bias flux line.
        # """

        # if 't1_flux_bias_line' not in self.supported_noise_channels():
            # raise RuntimeError("Noise channel 't1_flux_bias_line' is not supported in this system.")

        # def spec_dens(omega, ReZ=ReZ):
            # """
            # Our definitions assume that the noise_op is dH/dflux.
            # """
            # ReZ = ReZ(omega) if callable(ReZ) else ReZ
            # therm_ratio = calc_therm_ratio(omega, T)
            # s = 2 * (2 * np.pi)**2 * M**2 * omega * sp.constants.hbar / ReZ  \
                    # * (1/np.tanh(0.5*therm_ratio)) / (1 + np.exp(-therm_ratio))
            # # We assume that system energies are given in units of frequency
            # # and that the noise operator to be used with this `spec_dens` is dH/dflux.
            # # Hence we have to convert  2 powers of frequency to standard units
            # # (TODO this is ugly; what's a cleaner way to do this? )
            # s *= (units.to_standard_units(1))**2.0
            # return s

        # noise_op = self.sin_phi_operator()

        # return self.t1(i=i, j=j, noise_op=noise_op, spec_dens=spec_dens, esys=esys, get_rate=get_rate, **params)
