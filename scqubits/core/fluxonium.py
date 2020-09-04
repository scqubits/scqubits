# fluxonium.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import cmath
import math
import os

import numpy as np
import scipy as sp
from scipy.special import kn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.harmonic_osc as osc
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.spectrum_utils as spec_utils
import scqubits.utils.plotting as plot


# —Fluxonium qubit ————————————————————————

class Fluxonium(base.QubitBaseClass1d, serializers.Serializable):
    r"""Class for the fluxonium qubit. Hamiltonian
    :math:`H_\text{fl}=-4E_\text{C}\partial_\phi^2-E_\text{J}\cos(\phi-\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2`
    is represented in dense form. The employed basis is the EC-EL harmonic oscillator basis. The cosine term in the
    potential is handled via matrix exponentiation. Initialize with, for example::

        qubit = Fluxonium(EJ=1.0, EC=2.0, EL=0.3, flux=0.2, cutoff=120)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    EL: float
        inductive energy
    flux: float
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    cutoff: int
        number of harm. osc. basis states used in diagonalization
    truncated_dim: int, optional
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    """
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    cutoff = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, EL, dC, flux, cutoff, kbt, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.dC = dC
        self.flux = flux
        self.cutoff = cutoff
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # temperature unit mK
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._default_grid = discretization.Grid1d(-4.5 * np.pi, 4.5 * np.pi, 151)
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/fluxonium.png')

    @staticmethod
    def default_params():
        return {
            'EJ': 8.9,
            'EC': 2.5,
            'EL': 0.5,
            'flux': 0.0,
            'cutoff': 110,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['flux', 'cutoff', 'truncated_dim']

    def phi_osc(self):
        """
        Returns
        -------
        float
            Returns oscillator length for the LC oscillator composed of the fluxonium inductance and capacitance.
        """
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def E_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency.
        """
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation energy

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.hilbertdim()
        return (op.creation(dimension) + op.annihilation(dimension)) * self.phi_osc() / math.sqrt(2)

    def n_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n = - i d/d\\phi` operator in the LC harmonic oscillator basis
        """
        dimension = self.hilbertdim()
        return 1j * (op.creation(dimension) - op.annihilation(dimension)) / (self.phi_osc() * math.sqrt(2))

    def exp_i_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\phi}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self.phi_operator()
        return sp.linalg.expm(exponent)

    def cos_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self.exp_i_phi_operator()
        cos_phi_op += cos_phi_op.conjugate().T
        return cos_phi_op

    def sin_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self.exp_i_phi_operator()
        sin_phi_op += sin_phi_op.conjugate().T
        return sin_phi_op

    def _identity(self):
        """
        Identity operator acting only on the :math:`\varphi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.hilbertdim()
        return np.identity(dimension)

    def hamiltonian(self):  # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harmonic-oscillator basis, following Zhu et al., PRB 87, 024510 (2013)

        Returns
        -------
        ndarray
        """
        dimension = self.hilbertdim()
        diag_elements = [i * self.E_plasma() for i in range(dimension)]
        lc_osc_matrix = np.diag(diag_elements)

        exp_matrix = self.exp_i_phi_operator() * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = 0.5 * (exp_matrix + exp_matrix.conjugate().T)

        hamiltonian_mat = lc_osc_matrix - self.EJ * cos_matrix
        return np.real(hamiltonian_mat)  # use np.real to remove rounding errors from matrix exponential,
        # fluxonium Hamiltonian in harm. osc. basis is real-valued

    def current_noise_operator(self):
        dimension = self.hilbertdim()

        exp_matrix = self.exp_i_phi_operator() * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = 0.5 * (exp_matrix + exp_matrix.conjugate().T)

        hamiltonian_mat = - cos_matrix
        return np.real(hamiltonian_mat)

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension."""
        return self.cutoff

    def potential(self, phi):
        """Fluxonium potential evaluated at `phi`.

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`

        Returns
        -------
        float or ndarray
        """
        return 0.5 * self.EL * phi * phi - self.EJ * np.cos(phi + 2.0 * np.pi * self.flux)

    def wavefunction(self, esys, which=0, phi_grid=None):
        """Returns a fluxonium wave function in `phi` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (default value = 0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid

        Returns
        -------
        WaveFunction object
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys
        dim = self.hilbertdim()

        phi_grid = phi_grid or self._default_grid

        phi_basis_labels = phi_grid.make_linspace()
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros(phi_grid.pt_count, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * osc.harm_osc_wavefunction(n, phi_basis_labels,
                                                                                                    phi_osc)
        return storage.WaveFunction(basis_labels=phi_basis_labels, amplitudes=phi_wavefunc_amplitudes,
                                    energy=evals[which])

    def wavefunction1d_defaults(self, mode, evals, wavefunc_count):
        """Plot defaults for plotting.wavefunction1d.

        Parameters
        ----------
        mode: str
            amplitude modifier, needed to give the correct default y label
        evals: ndarray
            eigenvalues to include in plot
        wavefunc_count: int
            number of wave functions to be plotted
        """
        ylabel = r'$\psi_j(\varphi)$'
        ylabel = constants.MODE_STR_DICT[mode](ylabel)
        options = {
            'xlabel': r'$\varphi$',
            'ylabel': ylabel
        }
        if wavefunc_count > 1:
            ymin = - 1.025 * self.EJ
            ymax = max(1.8 * self.EJ, evals[-1] + 0.1 * (evals[-1] - evals[0]))
            options['ylim'] = (ymin, ymax)
        return options

    def ft_wavefunction(self, esys=None, which=0, mode='abs', **kwargs):
        phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 1501)
        wfnc = self.wavefunction(esys=esys, which=which, phi_grid=phi_grid)
        phi_amplitudes = wfnc.amplitudes
        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_amplitudes = np.fft.ifft(phi_amplitudes) * d_phi * phi_grid.pt_count
        n_phi_amplitudes = np.fft.fftshift(n_phi_amplitudes)
        n_phi_wavefunction = storage.WaveFunction(basis_labels=n_phi_list, amplitudes=n_phi_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_wavefunction.amplitudes = amplitude_modifier(spec_utils.standardize_phases(n_phi_wavefunction.amplitudes))
        return plot.wavefunction1d(n_phi_wavefunction, **kwargs)

    def q_cap(self, energy):
        # Devoret paper
        q_cap_0 = 1 * 1e6
        return q_cap_0 * (6 / energy) ** 0.7

        # Schuster paper
        # return 1 / (8e-6)

        # Vlad paper
        # q_cap_0 = 1 / (3 * 1e-6)
        # return q_cap_0 * (6 / energy) ** 0.15

    def get_t1_capacitive_loss(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele = self.get_matelements_vs_paramvals('n_operator', para_name, para_vals,
                                                   evals_count=2).matrixelem_table[:, 0, 1]
        s_vv = 2 * np.pi * 16 * self.EC / self.q_cap(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_cap = np.abs(matele) ** 2 * s_vv
        return 1 / (gamma1_cap) * 1e-6

    def q_ind(self, energy):
        """Frequency dependent quality factor of inductance"""
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt) * np.sinh(0.5 / 2.0 / self.kbt) / kn(0,
                                                                                          energy / 2.0 / self.kbt) / np.sinh(
            energy / 2.0 / self.kbt)

    def get_t1_inductive_loss(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele = self.get_matelements_vs_paramvals('phi_operator', para_name, para_vals,
                                                   evals_count=2).matrixelem_table[:, 0, 1]
        s_ii = 2 * np.pi * 2 * self.EL / self.q_ind(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_ind = np.abs(matele) ** 2 * s_ii
        return 1 / (gamma1_ind) * 1e-6

    def get_t1(self, para_name, para_vals):
        inductive = self.get_t1_inductive_loss(para_name, para_vals)
        capacitive = self.get_t1_capacitive_loss(para_name, para_vals)
        return 1 / (1 / inductive + 1 / capacitive)

    def get_t2_flux_noise(self, para_name, para_vals):
        orginal_flux = getattr(self, 'flux')
        delta = 1e-6
        pts = 51
        flux_list = np.linspace(orginal_flux - delta, orginal_flux + delta, pts)
        energy = np.zeros((pts, para_vals.size))
        for i in range(pts):
            setattr(self, 'flux', flux_list[i])
            energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                          subtract_ground=True).energy_table[:, 1]
        first_derivative = np.gradient(energy, flux_list, axis=0)[int(np.round(pts / 2)), :]
        second_derivative = np.gradient(np.gradient(energy, flux_list, axis=0), flux_list, axis=0)[int(np.round(pts / 2)), :]
        setattr(self, 'flux', orginal_flux)

        first_order = 3e-6 * first_derivative
        second_order = 9e-12 * second_derivative
        # print(first_order)
        # print(second_order)
        # print(first_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def get_t2_current_noise(self, para_name, para_vals):
        orginal_ej = self.EJ
        delta = 1e-5
        pts = 51
        ej_list = np.linspace(orginal_ej - delta, orginal_ej + delta, pts)
        energy = np.zeros((pts, para_vals.size))
        for i in range(pts):
            self.EJ = ej_list[i]
            energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                          subtract_ground=True).energy_table[:, 1]
        first_derivative = np.gradient(energy, ej_list, axis=0)[int(np.round(pts / 2)), :]
        self.EJ = orginal_ej
        # print(first_derivative)
        return np.abs(1 / (5e-7 * orginal_ej * first_derivative) * 1e-6) / (2 * np.pi)  # unit in ms

    def print_noise(self):
        t1_cap = self.get_t1_capacitive_loss('dC', np.array([0]))
        t1_ind = self.get_t1_inductive_loss('dC', np.array([0]))
        t2_current = self.get_t2_current_noise('dC', np.array([0]))
        t2_flux = self.get_t2_flux_noise('dC', np.array([0]))
        return print(' T2_current =', t2_current, ' ms', '\n T2_flux =', t2_flux,
                     ' ms', '\n Tphi_tot =', 1 / (1 / t2_flux + 1 / t2_current), ' ms', '\n T1_cap =',
                     t1_cap, ' ms', '\n T1_ind =', t1_ind, ' ms', '\n T1 =', 1 / (1 / t1_ind + 1 / t1_cap), ' ms',
                     '\n T2 =',
                     1 / (1 / t2_flux + 1 / t2_current + (1 / t1_ind + 1 / t1_cap) / 2), ' ms')

    def get_noise_analysis_2d(self, func, para_name_1, para_vals_1, para_name_2, para_vals_2):
        noise = np.zeros((para_vals_1.size, para_vals_2.size))
        original_para_val = getattr(self, para_name_1)
        for n in range(para_vals_1.size):
            setattr(self, para_name_1, para_vals_1[n])
            noise[n, :] = func(para_name_2, para_vals_2)
        setattr(self, para_name_1, original_para_val)

        # imshow_minval = np.log10(np.min(noise))
        # imshow_maxval = np.log10(np.max(noise))
        # fig, axes = plt.subplots(figsize=(4, 4))
        # im = axes.imshow(np.log10(noise), extent=[para_vals_2[0], para_vals_2[-1], para_vals_1[0], para_vals_1[-1]],
        #                  cmap=plt.cm.viridis, vmin=imshow_minval, vmax=imshow_maxval, origin='lower', aspect='auto')

        imshow_minval = (np.min(noise))
        imshow_maxval = (np.max(noise))
        fig, axes = plt.subplots(figsize=(4, 4))
        im = axes.imshow((noise), extent=[para_vals_2[0], para_vals_2[-1], para_vals_1[0], para_vals_1[-1]],
                         cmap=plt.cm.bwr, vmin=imshow_minval, vmax=imshow_maxval, origin='lower', aspect='auto')

        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        fig.colorbar(im, cax=cax)
        axes.set_xlabel(para_name_2)
        axes.set_ylabel(para_name_1)
        return fig, axes, noise

    def noise_analysis_2d(self, para_name_1, para_vals_1, para_name_2, para_vals_2):
        # fig, axes = self.get_noise_analysis_2d(self.get_t1_capacitive_loss, para_name_1, para_vals_1, para_name_2,
        #                                        para_vals_2)
        # axes.set_title('T1 capacitive loss (ms)')
        # fig, axes = self.get_noise_analysis_2d(self.get_t1_inductive_loss, para_name_1, para_vals_1, para_name_2,
        #                                        para_vals_2)
        # axes.set_title('T1 inductive loss (ms)')
        fig, axes, noise = self.get_noise_analysis_2d(self.get_t1, para_name_1, para_vals_1, para_name_2,
                                               para_vals_2)
        axes.set_title('T1 (ms)')
        return noise

    def noise_analysis(self, para_name, para_vals):
        t1_cap = self.get_t1_capacitive_loss(para_name, para_vals)
        t1_ind = self.get_t1_inductive_loss(para_name, para_vals)

        plt.figure(figsize=(4, 4))
        plt.plot(para_vals, t1_cap)
        plt.plot(para_vals, t1_ind)
        plt.plot(para_vals, 1 / (1 / t1_ind + 1 / t1_cap), '--')
        plt.legend(['T1_cap', 'T1_ind', 'T1_tot'])
        plt.xlabel(para_name)
        plt.ylabel('T1ms)')
        plt.yscale('log')

    def noise_analysis_t2(self, para_name, para_vals):
        t2_flux = self.get_t2_flux_noise(para_name, para_vals)
        t2_current = self.get_t2_current_noise(para_name, para_vals)

        plt.figure(figsize=(4, 4))
        plt.plot(para_vals, t2_flux)
        plt.plot(para_vals, t2_current)
        plt.plot(para_vals, 1 / (1 / t2_flux + 1 / t2_current), '--')
        plt.legend(['T2_flux', 'T2_current', 'T2_tot'])
        plt.xlabel(para_name)
        plt.ylabel('T2 (ms)')
        plt.yscale('log')
