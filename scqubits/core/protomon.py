# protomon.py
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
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import kn

import scqubits.core.constants as constants
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils
import scqubits.utils.spectrum_utils as matele_utils


# — Inductively-shunted Rhombus circuit ————————————————————————
class Protomon(base.QubitBaseClass, serializers.Serializable):
    r"""inductively-shunted Rhombus qubit, with the harmonic mode in the ground state

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        junction charging energy
    EL: float
        inductive energy
    ELA: float
        additional inductive energy
    flux_c: float
        common part of the external flux, e.g., 1 corresponds to one flux quantum
    flux_d: float
        differential part of the external flux, e.g., 1 corresponds to one flux quantum
    kbt: float
        photon temperature
    """

    def __init__(self, EJ, EC, EL, ELA, flux_c, flux_d, kbt):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.ELA = ELA
        self.flux_c = flux_c
        self.flux_d = flux_d
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # input temperature unit mK
        self.phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.ph = 0  # placeholder
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_

    @staticmethod
    def default_params():
        return {
            'EJ': 15.0,
            'EC': 3.5,
            'EL': 0.32,
            'ELA': 0.32,
            'flux_c': 0.5,
            'flux_d': 0.0
        }

    @staticmethod
    def nonfit_params():
        return ['flux_c', 'flux_d']

    def dim_phi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi' degree of freedom."""
        return self.phi_grid.pt_count

    def dim_theta(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return self.theta_grid.pt_count

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi() * self.dim_theta()

    def _phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\phi` operator in the discretized basis
        """
        return sparse.dia_matrix((self.phi_grid.make_linspace(), [0]), shape=(self.dim_phi(), self.dim_phi())).tocsc()

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in total Hilbert space
        """
        return self._kron2(self._phi_operator(), self._identity_theta())

    def _n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator
        """
        return self.phi_grid.first_derivative_matrix(prefactor=-1j)

    def n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi' operator in total Hilbert space
        """
        return self._kron2(self._n_phi_operator(), self._identity_theta())

    def _cos_phi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/div` operator
        """
        cos_phi_div_vals = np.cos(self.phi_grid.make_linspace() / div)
        return sparse.dia_matrix((cos_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())).tocsc()

    def _sin_phi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/div` operator
        """
        sin_phi_div_vals = np.sin(self.phi_grid.make_linspace() / div)
        return sparse.dia_matrix((sin_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())).tocsc()

    def _theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return sparse.dia_matrix((self.theta_grid.make_linspace(), [0]),
                                 shape=(self.dim_theta(), self.dim_theta())).tocsc()

    def theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return self._kron2(self._identity_phi(), self._theta_operator())

    def _n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator
        """
        return self.theta_grid.first_derivative_matrix(prefactor=-1j)

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_theta` in the total Hilbert space
        """
        return self._kron2(self._identity_phi(), self._n_theta_operator())

    def _cos_theta_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\theta/div` operator
        """
        cos_theta_div_vals = np.cos(self.theta_grid.make_linspace() / div)
        return sparse.dia_matrix((cos_theta_div_vals, [0]), shape=(self.dim_theta(), self.dim_theta())).tocsc()

    def _sin_theta_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\theta/div` operator
        """
        sin_theta_div_vals = np.sin(self.theta_grid.make_linspace() / div)
        return sparse.dia_matrix((sin_theta_div_vals, [0]), shape=(self.dim_theta(), self.dim_theta())).tocsc()

    def _kron2(self, mat1, mat2):
        """
        Returns
        -------
        ndarray
            Returns the kronecker product of two operators
        """
        return sparse.kron(mat1, mat2, format='csc')

    def _identity_phi(self):
        """
        Identity operator acting only on the :math:`\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_phi(), format='csc', dtype=np.complex_)

    def _identity_theta(self):
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_theta(), format='csc', dtype=np.complex_)

    def total_identity(self):
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron2(self._identity_phi(), self._identity_theta())

    def hamiltonian(self):
        r"""Construct Hamiltonian matrix in discretized basis
        :math:`H = 2 E_\text{C} n_\theta^2 + E_\text{L}(1+\dfrac{2E_\text{L}}{E_\text{LA}})^{-1} (\theta - \upphi_\text{d})^2+2 E_\text{C} n_\phi^2 + E_\text{L} (\phi - \upphi_\text{c})^2 - 2 E_\text{J}\cos\phi\cos\theta`
        """
        phi_kinetic = self.phi_grid.second_derivative_matrix(prefactor=- 2.0 * self.EC)
        theta_kinetic = self.theta_grid.second_derivative_matrix(prefactor=- 2.0 * self.EC)
        tot_kinetic = self._kron2(phi_kinetic, self._identity_theta()) + self._kron2(self._identity_phi(),
                                                                                     theta_kinetic)

        phi_ind = self.EL * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux_c) ** 2
        theta_ind = self.EL / (1 + 2 * self.EL / self.ELA) * (
                self.theta_operator() - self.total_identity() * 2 * np.pi * self.flux_d) ** 2

        # note the 2EJ constant term is added to be consistent with the 'LM' option in _evals_calc and _esys_calc
        phi_theta_junction = - 2 * self.EJ * self._kron2(self._cos_phi_div_operator(1.0),
                                                         self._cos_theta_div_operator(
                                                             1.0)) + 2 * self.EJ * self.total_identity()

        return tot_kinetic + phi_ind + theta_ind + phi_theta_junction

    def potential(self, phi, theta):
        """
        Potential evaluated at `phi, theta`

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`
        theta: float or ndarray
            float value of the phase variable `theta`

        Returns
        -------
        float or ndarray
        """
        return self.EL * (phi - 2 * np.pi * self.flux_c) ** 2 + self.EL / (1 + 2 * self.EL / self.ELA) * (
                theta - 2 * np.pi * self.flux_d) ** 2 - 2 * self.EJ * np.cos(phi) * np.cos(theta)

    def plot_potential(self, phi_grid=None, theta_grid=None, contour_vals=None, **kwargs):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use self._default_theta_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid

        x_vals = theta_grid.make_linspace()
        y_vals = phi_grid.make_linspace()
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (4, 4)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, sigma=0.0, which='LM')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, sigma=0.0, which='LM')
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None):
        """Returns a wave function in `phi`, `theta` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (default value = 0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi;
        theta_grid: Grid1d, optional
            used for setting a custom grid for theta;

        Returns
        -------
        WaveFunction object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid

        state_amplitudes = evecs[:, which].reshape(self.dim_phi(), self.dim_theta())
        wavefunc_amplitudes = spec_utils.standardize_phases(state_amplitudes)

        grid2d = discretization.GridSpec(
            np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                        [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_phi_theta_wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None, mode='abs',
                                    zero_calibrate=True,
                                    **kwargs):
        """
        Plots 2D wave function in `phi`, `theta` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use self._default_theta_grid
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

        Returns
        -------
        Figure, Axes
        """
        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
             [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_grid.pt_count, theta_grid.pt_count)))

        fig, axes = plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
        axes.set_xlim([-2 * np.pi, 2 * np.pi])
        axes.set_ylim([-1 * np.pi, 3 * np.pi])
        axes.set_ylabel(r'$\phi$')
        axes.set_xlabel(r'$\theta$')
        axes.set_xticks([-np.pi, 0, np.pi])
        axes.set_xticklabels(['-$\pi$', '$0$', '$\pi$'])
        axes.set_yticks([0, np.pi, 2 * np.pi])
        axes.set_yticklabels(['0', '$\pi$', '$2\pi$'])
        return fig, axes

    def plot_n_phi_n_theta_wavefunction(self, esys=None, mode='real', which=0, zero_calibrate=True, **kwargs):
        """
        Plots 2D wave function in `n_phi`, `n_theta` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT`
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

        Returns
        -------
        Figure, Axes
        """
        phi_grid = self.phi_grid
        theta_grid = self.theta_grid

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid,
                                     which=which)

        amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, theta_grid.pt_count))

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_list[0], n_phi_list[-1], n_phi_list.size)

        d_theta = theta_grid.make_linspace()[1] - theta_grid.make_linspace()[0]
        n_theta_list = np.sort(np.fft.fftfreq(theta_grid.pt_count, d_theta)) * 2 * np.pi
        n_theta_grid = discretization.Grid1d(n_theta_list[0], n_theta_list[-1], n_theta_list.size)

        n_phi_n_theta_amplitudes = np.fft.ifft2(
            amplitudes) * d_phi * phi_grid.pt_count * d_theta * theta_grid.pt_count
        n_phi_n_theta_amplitudes = np.fft.fftshift(n_phi_n_theta_amplitudes)

        grid2d = discretization.GridSpec(np.asarray([
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count],
            [n_theta_grid.min_val, n_theta_grid.max_val, n_theta_grid.pt_count]]))

        n_phi_n_theta_wavefunction = storage.WaveFunctionOnGrid(grid2d, n_phi_n_theta_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_n_theta_wavefunction.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(n_phi_n_theta_wavefunction.amplitudes))

        fig, axes = plot.wavefunction2d(n_phi_n_theta_wavefunction, zero_calibrate=zero_calibrate, **kwargs)
        return fig, axes

    def effective_zeta_operator(self):
        return self.EL / (self.EL + self.ELA * 0.5) * self.theta_operator()

    def phase_ind_1_operator(self):
        """
        phase drop on inductor 1, used in inductive loss calculation
        """
        return -self.phi_operator() - self.theta_operator() + self.effective_zeta_operator()

    def phase_ind_2_operator(self):
        """
        phase drop on inductor 2, used in inductive loss calculation
        """
        return -self.phi_operator() + self.theta_operator() - self.effective_zeta_operator()

    def phase_ind_a_operator(self):
        """
        phase drop on additional inductor, used in inductive loss calculation
        """
        return self.effective_zeta_operator()

    def q_ind(self, energy):
        """
        Frequency dependent quality factor for inductive loss
        """
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt) * np.sinh(0.5 / 2.0 / self.kbt) / kn(0,
                                                                                          energy / 2.0 / self.kbt) / np.sinh(
            energy / 2.0 / self.kbt)

    def charge_jj_1_operator(self):
        """
        charge across junction 1, used in capacitive loss calculation
        """
        return (self.n_phi_operator() + self.n_theta_operator()) / 2.0

    def charge_jj_2_operator(self):
        """
        charge across junction 2, used in capacitive loss calculation
        """
        return (self.n_phi_operator() - self.n_theta_operator()) / 2.0

    def sin_phase_jj_1_2_operator(self):
        """
        sin(phase_jj_1/2) operator, used in quasiparticle loss calculation
        """
        cos_phi_2 = self._kron2(self._cos_phi_div_operator(2.0), self._identity_theta())
        sin_phi_2 = self._kron2(self._sin_phi_div_operator(2.0), self._identity_theta())
        cos_theta_2 = self._kron2(self._identity_phi(), self._cos_theta_div_operator(2.0))
        sin_theta_2 = self._kron2(self._identity_phi(), self._sin_theta_div_operator(2.0))

        return sin_phi_2 * cos_theta_2 + cos_phi_2 * sin_theta_2

    def sin_phase_jj_2_2_operator(self):
        """
        sin(phase_jj_2/2) operator, used in quasiparticle loss calculation
        """
        cos_phi_2 = self._kron2(self._cos_phi_div_operator(2.0), self._identity_theta())
        sin_phi_2 = self._kron2(self._sin_phi_div_operator(2.0), self._identity_theta())
        cos_theta_2 = self._kron2(self._identity_phi(), self._cos_theta_div_operator(2.0))
        sin_theta_2 = self._kron2(self._identity_phi(), self._sin_theta_div_operator(2.0))

        return sin_phi_2 * cos_theta_2 - cos_phi_2 * sin_theta_2

    def y_qp(self, energy):
        """
        frequency dependent addimitance for quasiparticle loss
        """
        gap = 80.0
        xqp = 1e-8
        return 16 * np.pi * np.sqrt(2 / np.pi) / gap * energy * (2 * gap / energy) ** 1.5 * xqp * np.sqrt(
            energy / 2 / self.kbt) * kn(0, energy / 2 / self.kbt) * np.sinh(energy / 2 / self.kbt)

    def q_cap(self, energy):
        """
        Frequency dependent quality factor of capacitance loss
        """

        # parameters from the Devoret paper
        q_cap_0 = 1 * 1e6
        return q_cap_0 * (6 / energy) ** 0.7

        # parameters from the Schuster paper
        # return 1 / (8e-6)

        # parameters from the Vlad paper
        # q_cap_0 = 1 / (3 * 1e-6)
        # return q_cap_0 * (6 / energy) ** 0.15

    def thermal_factor(self, energy):
        """
        thermal factor for upward and downward transition
        """
        return np.where(energy > 0, 0.5 * (1 / (np.tanh(energy / 2.0 / self.kbt)) + 1),
                        0.5 * (1 / (np.tanh(- energy / 2.0 / self.kbt)) - 1))

    def get_t1_capacitive_loss(self, init_state):
        """
        T1 capacitive loss of one particular state
        """
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('charge_jj_1_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[0,
                    init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('charge_jj_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[0,
                    init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)

        s_vv_1 = 2 * np.pi * 16 * self.EC / self.q_cap(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_vv_2 = 2 * np.pi * 16 * self.EC / self.q_cap(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)

        gamma1_cap_1 = np.abs(matelem_1) ** 2 * s_vv_1
        gamma1_cap_2 = np.abs(matelem_2) ** 2 * s_vv_2

        gamma1_cap_tot = np.sum(gamma1_cap_1) + np.sum(gamma1_cap_2)
        return 1 / (gamma1_cap_tot) * 1e-6

    def get_t1_inductive_loss(self, init_state):
        """
        T1 inductive loss of one particular state
        """
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('phase_ind_1_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('phase_ind_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)
        matelem_a = self.get_matelements_vs_paramvals('phase_ind_a_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_a = np.delete(matelem_a, init_state)

        s_ii_1 = 2 * np.pi * 2 * self.EL / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_ii_2 = 2 * np.pi * 2 * self.EL / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_ii_a = 2 * np.pi * 2 * self.ELA / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_ind_1 = np.abs(matelem_1) ** 2 * s_ii_1
        gamma1_ind_2 = np.abs(matelem_2) ** 2 * s_ii_2
        gamma1_ind_a = np.abs(matelem_a) ** 2 * s_ii_a

        gamma1_ind_tot = np.sum(gamma1_ind_1) + np.sum(gamma1_ind_2) + np.sum(gamma1_ind_a)
        return 1 / (gamma1_ind_tot) * 1e-6

    def get_t1_qp_loss(self, init_state):
        """
        T1 quasiparticle loss of one particular state
        """
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('sin_phase_jj_1_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('sin_phase_jj_2_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)

        s_qp_1 = self.EJ * self.y_qp(np.abs(energy_diff)) * self.thermal_factor(energy_diff)
        s_qp_2 = self.EJ * self.y_qp(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_qp_1 = np.abs(matelem_1) ** 2 * s_qp_1
        gamma1_qp_2 = np.abs(matelem_2) ** 2 * s_qp_2

        gamma1_qp_tot = np.sum(gamma1_qp_1) + np.sum(gamma1_qp_2)
        return 1 / (gamma1_qp_tot) * 1e-6

    def get_t2_flux_c_noise(self, init_state):
        """
        common flux noise
        """
        delta = 1e-6
        pts = 11
        flux_c_list = np.linspace(self.flux_c - delta, self.flux_c + delta, pts)
        energy = self.get_spectrum_vs_paramvals('flux_c', flux_c_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, flux_c_list)[int(np.round(pts / 2))]
        second_derivative = np.gradient(np.gradient(energy, flux_c_list), flux_c_list)[int(np.round(pts / 2))]

        first_order = 3e-6 * np.abs(first_derivative)
        second_order = 9e-12 * np.abs(second_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def get_t2_flux_d_noise(self, init_state):
        """
        differential flux noise
        """
        delta = 1e-6
        pts = 11
        flux_d_list = np.linspace(self.flux_d - delta, self.flux_d + delta, pts)
        energy = self.get_spectrum_vs_paramvals('flux_d', flux_d_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, flux_d_list)[int(np.round(pts / 2))]
        second_derivative = np.gradient(np.gradient(energy, flux_d_list), flux_d_list)[int(np.round(pts / 2))]

        first_order = 3e-6 * np.abs(first_derivative)
        second_order = 9e-12 * np.abs(second_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def current_noise_operator(self):
        return - 2 * self._kron2(self._cos_phi_div_operator(1.0), self._cos_theta_div_operator(1.0))

    def get_t2_current_noise(self, g_state, e_state):
        """Calculate the current noise using operator method up to first order"""
        cutoff = e_state + 2
        matele = self.get_matelements_vs_paramvals('current_noise_operator', 'ph', [0],
                                                   evals_count=cutoff).matrixelem_table
        first_derivative = np.abs(matele[0, e_state, e_state] - matele[0, g_state, g_state])

        return np.abs(1 / (5e-7 * self.EJ * np.abs(first_derivative)) * 1e-6) / (2 * np.pi)  # unit in ms

    # def get_t2_current_noise(self, init_state):
    #     """
    #     T2 critical current noise
    #     """
    #     delta = 1e-7
    #     pts = 11
    #     ej_list = np.linspace(self.EJ - delta, self.EJ + delta, pts)
    #     energy = self.get_spectrum_vs_paramvals('EJ', ej_list, evals_count=init_state + 2,
    #                                             subtract_ground=True).energy_table[:, init_state]
    #     first_derivative = np.gradient(energy, ej_list)[int(np.round(pts / 2))]
    #     return np.abs(1 / (5e-7 * self.EJ * np.abs(first_derivative)) * 1e-6) / (2 * np.pi)  # unit in ms

    def print_noise(self, g_state, e_state, table=True):
        """
        print summary of all noise channels
        :param g_state: the logical 0 state of the qubit
        :param e_state: the logical 1 state of the qubit
        :return: t2_current, t2_flux, t2_fluxa, t1_cap, t1_ind, t1_qp, t1_tot, t2_tot
        """
        t2_current = self.get_t2_current_noise(g_state, e_state)
        t2_flux_c = self.get_t2_flux_c_noise(e_state)
        t2_flux_d = self.get_t2_flux_d_noise(e_state)
        t1_cap = 1 / (1 / self.get_t1_capacitive_loss(g_state) + 1 / self.get_t1_capacitive_loss(e_state))
        t1_ind = 1 / (1 / self.get_t1_inductive_loss(g_state) + 1 / self.get_t1_inductive_loss(e_state))
        t1_qp = 1 / (1 / self.get_t1_qp_loss(g_state) + 1 / self.get_t1_qp_loss(e_state))
        t1_tot = 1 / (1 / t1_cap + 1 / t1_ind + 1 / t1_qp)
        t2_tot = 1 / (1 / t2_current + 1 / t2_flux_c + 1 / t2_flux_d + 1 / t1_tot / 2)

        if table is True:
            print(' T2_current =', t2_current, ' ms', '\n T2_flux_c =', t2_flux_c,
                  ' ms', '\n T2_flux_d =', t2_flux_d,
                  ' ms', '\n T1_cap =',
                  t1_cap, ' ms', '\n T1_ind =', t1_ind, ' ms', '\n T1_qp =', t1_qp, ' ms', '\n T1 =', t1_tot,
                  ' ms', '\n T2 =', t2_tot,
                  ' ms')

        return np.array([t2_current, t2_flux_c, t2_flux_d, t1_cap, t1_ind, t1_qp, t1_tot, t2_tot])

    def dispersive_shift(self, w_readout, beta_phi, beta_theta, cutoff):
        """
        Calculate the dispersive shift
        :param w_readout: frequency of the readout
        :param beta_phi: capacitive coupling ratio for phi modes
        :param beta_theta: capacitive coupling ratio for theta modes
        :param cutoff: states involved in the calculation
        :return: table containing dispersive shift for each state
        """
        eigsys = self.eigensys(evals_count=cutoff)
        energy = eigsys[0]
        states = eigsys[1]

        energy_diff_ij = np.zeros((cutoff, cutoff - 1))
        energy_diff_ji = np.zeros((cutoff, cutoff - 1))
        for i in range(cutoff):
            energy_diff_ij[i, :] = np.delete(energy[i] - energy, i) - w_readout
            energy_diff_ji[i, :] = np.delete(energy - energy[i], i) - w_readout

        ds_table = np.zeros((cutoff, 1))
        if beta_theta == 0:
            matelem_phi = self.get_matelements_vs_paramvals('n_phi_operator', 'ph', [0],
                                                            evals_count=cutoff)
            for i in range(cutoff):
                ds_table[i] = np.sum(np.abs(np.delete(matelem_phi.matrixelem_table[
                                                      0, i, :], i) * beta_phi) ** 2 * (
                                             1 / energy_diff_ij[i] - 1 / energy_diff_ji[i]))

        if beta_phi == 0:
            matelem_theta = self.get_matelements_vs_paramvals('n_theta_operator', 'ph', [0],
                                                              evals_count=cutoff)
            for i in range(cutoff):
                ds_table[i] = np.sum(np.abs(np.delete(matelem_theta.matrixelem_table[
                                                      0, i, :], i) * beta_theta) ** 2 * (
                                             1 / energy_diff_ij[i] - 1 / energy_diff_ji[i]))

        # take care of the random phase
        if beta_phi != 0 and beta_theta != 0:
            for i in range(cutoff):
                for j in range(cutoff):
                    if i != j:
                        ds_temp = matele_utils.matrix_element(states[:, i], self.n_phi_operator(),
                                                              states[:, j]) * beta_phi + matele_utils.matrix_element(
                            states[:, i], self.n_theta_operator(), states[:, j]) * beta_theta
                        ds_table[i] += np.abs(ds_temp) ** 2 * (1 / (eigsys[0][i] - eigsys[0][j] - w_readout) - 1 / (
                                eigsys[0][j] - eigsys[0][i] - w_readout))

        return ds_table
