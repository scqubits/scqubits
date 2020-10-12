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
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils


# — Inductively shunted cos2phi qubit ————————————————————————
class Protomon(base.QubitBaseClass, serializers.Serializable):
    r"""inductively shunted cos2phi qubit decoupled from the harmonic mode

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
    x: float
        ratio of the junction capacitance to the shunt capacitance x = C_J / C_shunt
    flux: float
        external magnetic flux, e.g., 1 corresponds to one flux quantum
    fluxa: float
        additional external magnetic flux, e.g., 1 corresponds to one flux quantum
    limit:
        True -> calculating the full symmetric case
    """

    def __init__(self, EJ, EC, EL, ELA, x, flux, fluxa, kbt, limit=False):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.ELA = ELA
        self.x = x
        self.flux = flux
        self.fluxa = fluxa
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # input temperature unit mK
        self.phi_grid = discretization.Grid1d(-8 * np.pi, 8 * np.pi, 100)
        self.varphi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.ph = 0  # placeholder
        if limit is False:
            self.limit = 1
        else:
            self.limit = 0
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_

    @staticmethod
    def default_params():
        return {
            'EJ': 4.5,
            'EC': 1.05,
            'EL': 0.1,
            'ELA': 0.1,
            'x': 10,
            'flux': 0,
            'fluxa': -0.5
        }

    @staticmethod
    def nonfit_params():
        return ['flux', 'fluxa']

    def set_by_flux_cd(self, flux_c, flux_d):
        """
        set flux and fluxa using flux_c and flux_d
        """
        self.flux = flux_c * 2
        self.fluxa = - flux_d - flux_c

    def dim_phi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi' degree of freedom."""
        return self.phi_grid.pt_count

    def dim_varphi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`varphi' degree of freedom."""
        return self.varphi_grid.pt_count

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi() * self.dim_varphi()

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
        return self._kron2(self._phi_operator(), self._identity_varphi())

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
        return self._kron2(self._n_phi_operator(), self._identity_varphi())

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

    def _varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`varphi' operator in total Hilbert space
        """
        return sparse.dia_matrix((self.varphi_grid.make_linspace(), [0]),
                                 shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`varphi' operator in total Hilbert space
        """
        return self._kron2(self._identity_phi(), self._varphi_operator())

    def _n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\varphi = - i d/d\\varphi` operator
        """
        return self.varphi_grid.first_derivative_matrix(prefactor=-1j)

    def n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_varphi` in the total Hilbert space
        """
        return self._kron2(self._identity_phi(), self._n_varphi_operator())

    def _cos_varphi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\varphi/div` operator
        """
        cos_varphi_div_vals = np.cos(self.varphi_grid.make_linspace() / div)
        return sparse.dia_matrix((cos_varphi_div_vals, [0]), shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def _sin_varphi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\varphi/div` operator
        """
        sin_varphi_div_vals = np.sin(self.varphi_grid.make_linspace() / div)
        return sparse.dia_matrix((sin_varphi_div_vals, [0]), shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

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

    def _identity_varphi(self):
        """
        Identity operator acting only on the :math:`\varphi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_varphi(), format='csc', dtype=np.complex_)

    def total_identity(self):
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron2(self._identity_phi(), self._identity_varphi())

    def hamiltonian(self):
        """Construct Hamiltonian matrix in discretized basis

          Returns
          -------
          ndarray
          """
        phi_kinetic = self.phi_grid.second_derivative_matrix(prefactor=- 8.0 * self.EC)
        varphi_kinetic = self.varphi_grid.second_derivative_matrix(
            prefactor=- 2.0 * self.EC * (1 - 0.5 * self.limit * 1 / (self.x + 0.5)))
        tot_kinetic = self._kron2(phi_kinetic, self._identity_varphi()) + self._kron2(self._identity_phi(),
                                                                                      varphi_kinetic)

        phi_ind = 0.25 * self.EL * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux) ** 2
        varphi_ind = self.EL / (1 + self.limit * 2 * self.EL / self.ELA) * (
                self.varphi_operator() + self.total_identity() * 2 * np.pi * (self.flux / 2.0 + self.fluxa)) ** 2

        # note the 2EJ constant term is added to be consistent with the 'LM' option in eigensolver
        phi_varphi_junction = - 2 * self.EJ * self._kron2(self._cos_phi_div_operator(2.0),
                                                          self._cos_varphi_div_operator(
                                                              1.0)) + 2 * self.EJ * self.total_identity()

        return tot_kinetic + phi_ind + varphi_ind + phi_varphi_junction

    def potential(self, varphi, phi):
        """
        Potential evaluated at `phi, varphi`

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`
        varphi: float or ndarray
            float value of the phase variable `varphi`

        Returns
        -------
        float or ndarray
        """
        return self.EL * (0.25 * phi * phi) - 2 * self.EJ * np.cos(varphi) * np.cos(
            phi * 0.5 + np.pi * self.flux) + self.EL / (1 + 2 * self.EL / self.ELA) * (
                       2 * np.pi * (self.flux / 2.0 + self.fluxa) + varphi) ** 2 + 2 * self.EJ

    def plot_potential(self, phi_grid=None, varphi_grid=None, contour_vals=None, **kwargs):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        varphi_grid: Grid1d, option
            used for setting a custom grid for varphi; if None use self._default_varphi_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        phi_grid = phi_grid or self.phi_grid
        varphi_grid = varphi_grid or self.varphi_grid

        x_vals = varphi_grid.make_linspace()
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

    def wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None):
        """Returns a wave function in `phi`, `varphi` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (default value = 0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi;
        varphi_grid: Grid1d, optional
            used for setting a custom grid for varphi;

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
        varphi_grid = varphi_grid or self.varphi_grid

        state_amplitudes = evecs[:, which].reshape(self.dim_phi(), self.dim_varphi())
        wavefunc_amplitudes = spec_utils.standardize_phases(state_amplitudes)

        grid2d = discretization.GridSpec(
            np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                        [varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_phi_varphi_wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None, mode='abs',
                                     zero_calibrate=True,
                                     **kwargs):
        """
        Plots 2D wave function in `phi`, `varphi` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        varphi_grid: Grid1d, option
            used for setting a custom grid for varphi; if None use self._default_varphi_grid
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
        varphi_grid = varphi_grid or self.varphi_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, varphi_grid=varphi_grid, which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count],
             [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count)))

        # note here the vertical phi axis is rescaled when plotting, i.e., phi -> 2phi
        fig, axes = plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
        axes.set_xlim([-2 * np.pi, 2 * np.pi])
        axes.set_ylim([-4 * np.pi, 4 * np.pi])
        axes.set_ylabel(r'$\phi$')
        axes.set_xlabel(r'$\varphi$')
        axes.set_xticks([-np.pi, 0, np.pi, 2 * np.pi])
        axes.set_xticklabels(['-$\pi$', '$0$', '$\pi$', '$2\pi$'])
        axes.set_yticks([-2 * np.pi, 0, 2 * np.pi])
        axes.set_yticklabels(['-$\pi$', '0', '$\pi$'])
        return fig, axes

    def plot_n_phi_n_varphi_wavefunction(self, esys=None, mode='real', which=0, zero_calibrate=True, **kwargs):
        """
        Plots 2D wave function in `n_phi`, `n_varphi` basis

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
        varphi_grid = self.varphi_grid

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, varphi_grid=varphi_grid,
                                     which=which)

        amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_list[0], n_phi_list[-1], n_phi_list.size)

        d_varphi = varphi_grid.make_linspace()[1] - varphi_grid.make_linspace()[0]
        n_varphi_list = np.sort(np.fft.fftfreq(varphi_grid.pt_count, d_varphi)) * 2 * np.pi
        n_varphi_grid = discretization.Grid1d(n_varphi_list[0], n_varphi_list[-1], n_varphi_list.size)

        n_phi_n_varphi_amplitudes = np.fft.ifft2(
            amplitudes) * d_phi * phi_grid.pt_count * d_varphi * varphi_grid.pt_count
        n_phi_n_varphi_amplitudes = np.fft.fftshift(n_phi_n_varphi_amplitudes)

        grid2d = discretization.GridSpec(np.asarray([
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count],
            [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count]]))

        n_phi_n_varphi_wavefunction = storage.WaveFunctionOnGrid(grid2d, n_phi_n_varphi_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_n_varphi_wavefunction.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(n_phi_n_varphi_wavefunction.amplitudes))

        fig, axes = plot.wavefunction2d(n_phi_n_varphi_wavefunction, zero_calibrate=zero_calibrate, **kwargs)
        return fig, axes

    def phi_1_operator(self):
        """
        phase drop on inductor 1, used in inductive loss calculation
        """
        return - self.phi_operator() / 2.0

    def phi_2_operator(self):
        """
        phase drop on inductor 2, used in inductive loss calculation
        """
        return - self.phi_operator() / 2.0

    def phi_a_operator(self):
        """
        phase drop on additional inductor, used in inductive loss calculation
        """
        return self.varphi_operator()

    def q_ind(self, energy):
        """
        Frequency dependent quality factor for inductive loss
        """
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt) * np.sinh(0.5 / 2.0 / self.kbt) / kn(0,
                                                                                          energy / 2.0 / self.kbt) / np.sinh(
            energy / 2.0 / self.kbt)

    def N_1_operator(self):
        """
        charge across junction 1, used in capacitive loss calculation
        """
        return self.n_phi_operator() + 0.5 * self.n_varphi_operator()

    def N_2_operator(self):
        """
        charge across junction 2, used in capacitive loss calculation
        """
        return self.n_phi_operator() - 0.5 * self.n_varphi_operator()

    def sin_varphi_1_2_operator(self):
        """
        sin(\varphi_1/2) operator, used in quasiparticle loss calculation
        """
        cos_phi_4 = self._kron2(self._cos_phi_div_operator(4.0), self._identity_varphi())
        sin_phi_4 = self._kron2(self._sin_phi_div_operator(4.0), self._identity_varphi())
        cos_varphi_2 = self._kron2(self._identity_phi(), self._cos_varphi_div_operator(2.0))
        sin_varphi_2 = self._kron2(self._identity_phi(), self._sin_varphi_div_operator(2.0))

        return sin_phi_4 * cos_varphi_2 + cos_phi_4 * sin_varphi_2

    def sin_varphi_2_2_operator(self):
        """
        sin(\varphi_2/2) operator, used in quasiparticle loss calculation
        """
        cos_phi_4 = self._kron2(self._cos_phi_div_operator(4.0), self._identity_varphi())
        sin_phi_4 = self._kron2(self._sin_phi_div_operator(4.0), self._identity_varphi())
        cos_varphi_2 = self._kron2(self._identity_phi(), self._cos_varphi_div_operator(2.0))
        sin_varphi_2 = self._kron2(self._identity_phi(), self._sin_varphi_div_operator(2.0))

        return sin_phi_4 * cos_varphi_2 - cos_phi_4 * sin_varphi_2

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

        matelem_1 = self.get_matelements_vs_paramvals('N_1_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[0,
                    init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('N_2_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[0,
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

        matelem_1 = self.get_matelements_vs_paramvals('phi_1_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('phi_2_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)
        matelem_a = self.get_matelements_vs_paramvals('phi_a_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
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

        matelem_1 = self.get_matelements_vs_paramvals('sin_varphi_1_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('sin_varphi_2_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)

        s_qp_1 = self.EJ * self.y_qp(np.abs(energy_diff)) * self.thermal_factor(energy_diff)
        s_qp_2 = self.EJ * self.y_qp(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_qp_1 = np.abs(matelem_1) ** 2 * s_qp_1
        gamma1_qp_2 = np.abs(matelem_2) ** 2 * s_qp_2

        gamma1_qp_tot = np.sum(gamma1_qp_1) + np.sum(gamma1_qp_2)
        return 1 / (gamma1_qp_tot) * 1e-6

    def get_t2_flux_noise(self, init_state):
        """
        T2 flux noise
        """
        delta = 1e-6
        pts = 11
        flux_list = np.linspace(self.flux - delta, self.flux + delta, pts)
        energy = self.get_spectrum_vs_paramvals('flux', flux_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, flux_list)[int(np.round(pts / 2))]
        second_derivative = np.gradient(np.gradient(energy, flux_list), flux_list)[int(np.round(pts / 2))]

        first_order = 3e-6 * np.abs(first_derivative)
        second_order = 9e-12 * np.abs(second_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def get_t2_fluxa_noise(self, init_state):
        """
        T2 flux noise for the addtional loop
        """
        delta = 1e-6
        pts = 11
        flux_list = np.linspace(self.fluxa - delta, self.fluxa + delta, pts)
        energy = self.get_spectrum_vs_paramvals('fluxa', flux_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, flux_list)[int(np.round(pts / 2))]
        second_derivative = np.gradient(np.gradient(energy, flux_list), flux_list)[int(np.round(pts / 2))]

        first_order = 3e-6 * np.abs(first_derivative)
        second_order = 9e-12 * np.abs(second_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def get_t2_current_noise(self, init_state):
        """
        T2 critical current noise
        """
        delta = 1e-7
        pts = 11
        ej_list = np.linspace(self.EJ - delta, self.EJ + delta, pts)
        energy = self.get_spectrum_vs_paramvals('EJ', ej_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, ej_list)[int(np.round(pts / 2))]
        return np.abs(1 / (5e-7 * self.EJ * np.abs(first_derivative)) * 1e-6) / (2 * np.pi)  # unit in ms

    def print_noise(self, g_state, e_state, table=True):
        """
        print summary of all noise channels
        :param g_state: the logical 0 state of the qubit
        :param e_state: the logical 1 state of the qubit
        :return: t2_current, t2_flux, t2_fluxa, t1_cap, t1_ind, t1_qp, t1_tot, t2_tot
        """
        t2_current = self.get_t2_current_noise(e_state)
        t2_flux = self.get_t2_flux_noise(e_state)
        t2_fluxa = self.get_t2_fluxa_noise(e_state)
        t1_cap = 1 / (1 / self.get_t1_capacitive_loss(g_state) + 1 / self.get_t1_capacitive_loss(e_state))
        t1_ind = 1 / (1 / self.get_t1_inductive_loss(g_state) + 1 / self.get_t1_inductive_loss(e_state))
        t1_qp = 1 / (1 / self.get_t1_qp_loss(g_state) + 1 / self.get_t1_qp_loss(e_state))
        t1_tot = 1 / (1 / t1_cap + 1 / t1_ind + 1 / t1_qp)
        t2_tot = 1 / (1 / t2_current + 1 / t2_flux + 1 / t2_fluxa + 1 / t1_tot / 2)

        if table is True:
            print(' T2_current =', t2_current, ' ms', '\n T2_flux =', t2_flux,
                  ' ms', '\n T2_flux_a =', t2_fluxa,
                  ' ms', '\n T1_cap =',
                  t1_cap, ' ms', '\n T1_ind =', t1_ind, ' ms', '\n T1_qp =', t1_qp, ' ms', '\n T1 =', t1_tot,
                  ' ms', '\n T2 =', t2_tot,
                  ' ms')

        return np.array([t2_current, t2_flux, t2_fluxa, t1_cap, t1_ind, t1_qp, t1_tot, t2_tot])
