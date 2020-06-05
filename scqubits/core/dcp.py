# dcp.py
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
from scipy import sparse
from scipy.sparse.linalg import expm, eigsh
from scipy.special import kn
import matplotlib.pyplot as plt

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.harmonic_osc as osc
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils
import scqubits.utils.plot_defaults as defaults


# —Double Cooper pair tunneling qubit ————————————————————————
class Dcp(base.QubitBaseClass, serializers.Serializable):
    r"""Class for the double Cooper pair tunneling qubit. Hamiltonian
    :math:`H_\text{dcp}=4E_\text{C}[2n_\phi^2+\frac{1}{2}(n_\varphi-N_\text{g}-n_\theta)^2+xn_\theta^2]+E_\text{L}(\frac{1}{4}\phi^2+\theta^2)-2E_\text{J}\cos(\varphi)\cos(\frac{\phi}{2}+\frac{\varphi_\text{ext}}{2})`
    is represented in dense form. The employed basis are harmonic basis for :math:`\phi,\theta` and charge basis for :math:`\varphi`. The cosine term in the
    potential is handled via matrix exponentiation. Initialize with, for example::

        qubit = Dcp(EJ=15.0, EC=2.0, EL=1.0, x=0.02, flux=0.5, Ng=0, N0=7, q0=30, p0=7)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    EL: float
        inductive energy
    x: float
        ratio of the junction capacitance to the shunt capacitance
    flux: float
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    Ng: float
        offset charge
    N0: int
        number of charge states used in diagonalization, -N0 <= n_\varphi <= N0
    q0: int
        number of harmonic oscillator basis used in diagonalization of \theta
    p0: int
        number of harmonic oscillator basis used in diagonalization of \phi
    truncated_dim: int, optional
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    """
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    x = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    Ng = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    N0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    q0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    p0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, EL, x, dL, dC, dJ, flux, Ng, N0, q0, p0, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.x = x
        self.dL = dL
        self.dC = dC
        self.dJ = dJ
        self.flux = flux
        self.Ng = Ng
        self.N0 = N0
        self.q0 = q0
        self.p0 = p0
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._default_phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_varphi_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        # TODO change the picture here
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/fluxonium.png')

    @staticmethod
    def default_params():
        return {
            'EJ': 15.0,
            'EC': 2.0,
            'EL': 1.0,
            'x': 0.02,
            'flux': np.pi,
            'Ng': 0.0,
            'N0': 7,
            'q0': 30,
            'p0': 7,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['flux', 'Ng', 'N0', 'q0', 'p0', 'truncated_dim']

    def phi_hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension for phi degree of freedom."""
        return self.p0

    def theta_hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension for phi degree of freedom."""
        return self.q0

    def varphi_hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension for varphi degree of freedom."""
        return 2 * self.N0 + 1

    def hilbertdim(self):
        """Return total Hilbert space dimension."""
        return self.phi_hilbertdim() * self.theta_hilbertdim() * self.varphi_hilbertdim()

    def _dis_el(self):
        return self.EL / (1 - self.dL ** 2)

    def phi_osc(self):
        """Return the oscillator strength of phi degree of freedom"""
        return (32 * self.EC / self._dis_el()) ** 0.25

    def theta_osc(self):
        """Return the oscillator strength of theta degree of freedom"""
        return (4 * self.EC * self.x / self._dis_el()) ** 0.25

    def phi_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the phi degree of freedom.
        """
        return math.sqrt(8.0 * self._dis_el() * self.EC)

    def theta_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the theta degree of freedom.
        """
        return math.sqrt(16.0 * self.x * self._dis_el() * self.EC)

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.phi_hilbertdim()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.phi_osc() / math.sqrt(2)

    def phi_opt(self):
        """phi operator in the total hilbert space"""
        return self._kron3(self.phi_operator(), self.theta_identity(), self.varphi_identity())

    def n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator in the LC harmonic oscillator basis
        """
        dimension = self.phi_hilbertdim()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.phi_osc() * math.sqrt(2))

    def n_phi_opt(self):
        """n_phi operator in the total hilbert space"""
        return self._kron3(self.n_phi_operator(), self.theta_identity(), self.varphi_identity())

    def theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the theta operator in the LC harmonic oscillator basis
        """
        dimension = self.theta_hilbertdim()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.theta_osc() / math.sqrt(2)

    def theta_opt(self):
        """theta operator in the total hilbert space"""
        return self._kron3(self.phi_identity(), self.theta_operator(), self.varphi_identity())

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator in the LC harmonic oscillator basis
        """
        dimension = self.theta_hilbertdim()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.theta_osc() * math.sqrt(2))

    def n_theta_opt(self):
        """n_theta operator in the total hilbert space"""
        return self._kron3(self.phi_identity(), self.n_theta_operator(), self.varphi_identity())

    def exp_i_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\phi/2}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self.phi_operator() * 0.5
        return expm(exponent)

    def cos_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/2` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self.exp_i_phi_2_operator()
        cos_phi_op += cos_phi_op.conjugate().T
        return np.real(cos_phi_op)

    def sin_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/2` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self.exp_i_phi_2_operator()
        sin_phi_op += sin_phi_op.conjugate().T
        return np.real(sin_phi_op)

    def n_varphi_ng_operator(self):
        """Returns charge operator `n_phi - Ng` in the charge basis"""
        diag_elements = np.arange(-self.N0 - self.Ng, self.N0 + 1 - self.Ng)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()

    def n_varphi_operator(self):
        """Returns charge operator `n_phi` in the charge basis"""
        diag_elements = np.arange(-self.N0, self.N0 + 1)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()

    def n_varphi_opt(self):
        """n_varphi operator in the total hilbert space"""
        return self._kron3(self.phi_identity(), self.theta_identity(), self.n_varphi_operator())

    def cos_varphi_operator(self):
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [1]),
                                         shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [-1]),
                                          shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        return cos_op

    def sin_varphi_operator(self):
        """Returns operator :math:`\\sin \\varphi` in the charge basis"""
        sin_op = 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [1]),
                                         shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        sin_op -= 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [-1]),
                                          shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        return sin_op * (-1j)

    def phi_identity(self):
        dimension = self.phi_hilbertdim()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def theta_identity(self):
        dimension = self.theta_hilbertdim()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def varphi_identity(self):
        dimension = self.varphi_hilbertdim()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def _kron3(self, mat1, mat2, mat3):
        return sparse.kron(sparse.kron(mat1, mat2, format='csc'), mat3, format='csc')

    def total_identity(self):
        return self._kron3(self.phi_identity(), self.theta_identity(), self.varphi_identity())

    def hamiltonian(self):
        # follow W.C. Smith, A. Kou, X. Xiao, U. Vool, and M.H. Devoret, Npj Quantum Inf. 6, 8 (2020).
        """Return Hamiltonian

        Returns
        -------
        ndarray
        """
        phi_osc_matrix = self._kron3(op.number_sparse(self.phi_hilbertdim(), self.phi_plasma()), self.theta_identity(),
                                     self.varphi_identity())
        theta_osc_matrix = self._kron3(self.phi_identity(),
                                       op.number_sparse(self.theta_hilbertdim(), self.theta_plasma()),
                                       self.varphi_identity())

        n_varphi_ng_matrix = self._kron3(self.phi_identity(), self.theta_identity(), self.n_varphi_ng_operator())
        n_theta_matrix = self._kron3(self.phi_identity(), self.n_theta_operator(), self.varphi_identity())
        cross_kinetic_matrix = 2 * self.EC * (n_varphi_ng_matrix - n_theta_matrix) * (
                n_varphi_ng_matrix - n_theta_matrix)

        phi_flux_term = self.cos_phi_2_operator() * np.cos(self.flux * np.pi) - self.sin_phi_2_operator() * np.sin(
            self.flux * np.pi)
        junction_matrix = -2 * self.EJ * self._kron3(phi_flux_term, self.theta_identity(), self.cos_varphi_operator())

        hamiltonian_mat = phi_osc_matrix + theta_osc_matrix + cross_kinetic_matrix + junction_matrix
        return hamiltonian_mat

    def potential(self, varphi, phi):
        """Double Cooper pair tunneling qubit potential evaluated at `phi, varphi`, with `theta=0`

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
        return self._dis_el() * (0.25 * phi * phi) - 2 * self.EJ * np.cos(varphi) * np.cos(
            phi * 0.5 + np.pi * self.flux)

    def plot_potential(self, phi_grid=None, varphi_grid=None, contour_vals=None, **kwargs):
        """Draw contour plot of the potential energy.

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
        phi_grid = phi_grid or self._default_phi_grid
        varphi_grid = varphi_grid or self._default_varphi_grid

        x_vals = varphi_grid.make_linspace()
        y_vals = phi_grid.make_linspace()
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (4, 4)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def _tensor_index(self, index_phi, index_theta, index_varphi):
        """Return the index of the coefficient of the wavefunction, corresponding to the indices of phi, theta, and varphi """
        return (index_phi * self.theta_hilbertdim() + index_theta) * self.varphi_hilbertdim() + index_varphi

    def _tensor_index_inv(self, index_evec):
        """Return the indices of phi, theta, and varphi corresponding to the index of the coefficient of the wavefunction"""
        index_varphi = index_evec % self.varphi_hilbertdim()
        index_temp = index_evec // self.varphi_hilbertdim()
        index_theta = index_temp % self.theta_hilbertdim()
        index_phi = index_temp // self.theta_hilbertdim()
        return index_phi, index_theta, index_varphi

    def wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None, varphi_grid=None):
        """
        Return a flux qubit wave function in phi, varphi basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use self._default_phi_grid
        varphi_grid: Grid1d, option
            used for setting a custom grid for varphi; if None use self._default_varphi_grid

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_grid = phi_grid or self._default_phi_grid
        theta_grid = theta_grid or self._default_phi_grid
        varphi_grid = varphi_grid or self._default_varphi_grid

        phi_basis_labels = phi_grid.make_linspace()
        theta_basis_labels = theta_grid.make_linspace()
        varphi_basis_labels = varphi_grid.make_linspace()

        wavefunc_basis_amplitudes = np.reshape(evecs[:, which], self.hilbertdim())
        wavefunc_amplitudes = np.zeros((phi_grid.pt_count, theta_grid.pt_count, varphi_grid.pt_count),
                                       dtype=np.complex_)
        for n in range(self.hilbertdim()):
            n_phi, n_theta, n_varphi = self._tensor_index_inv(n)
            num_varphi = n_varphi - self.N0
            phi_wavefunc_amplitudes = osc.harm_osc_wavefunction(n_phi, phi_basis_labels, self.phi_osc())
            theta_wavefunc_amplitudes = osc.harm_osc_wavefunction(n_theta, theta_basis_labels, self.theta_osc())
            varphi_wavefunc_amplitudes = np.exp(-1j * num_varphi * varphi_basis_labels) / (2 * np.pi) ** 0.5
            wavefunc_amplitudes += wavefunc_basis_amplitudes[n] * np.tensordot(
                np.tensordot(phi_wavefunc_amplitudes, theta_wavefunc_amplitudes, 0), varphi_wavefunc_amplitudes, 0)

        grid3d = discretization.GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                                     [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                                                     [varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid3d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None, mode='abs', zero_calibrate=True,
                          **kwargs):
        """Plots 2d phase-basis wave function for theta = 0

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
        phi_grid = phi_grid or self._default_phi_grid
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = varphi_grid or self._default_varphi_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count],
             [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count)))
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def instanton_path(self, varphi):
        """instanton path phi(varphi)"""
        # TODO this works now in symmetric case, check for disorder
        z = self.EL / self.EJ
        # TODO make sure the minus pi
        return 1.0 / (1.0 + z) * (
                2 * np.abs(varphi - 2 * np.pi * np.round(varphi / (2 * np.pi))) + z * 2 * np.pi * self.flux) - np.pi

    def plot_charge_wavefunction(self, esys=None, mode='real', which=0, n_varphi_list=None, **kwargs):
        """Wavefunction in n_varphi space"""
        phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        d2_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        d1_amplitudes = np.zeros(varphi_grid.pt_count, dtype=np.complex_)
        for n in range(varphi_grid.pt_count):
            phi_instanton = self.instanton_path(varphi_grid_list[n])
            phi_idx = (np.abs(phi_grid.make_linspace() - phi_instanton)).argmin()
            d1_amplitudes[n] = d2_amplitudes[phi_idx, n]

        if n_varphi_list is None:
            n_varphi_list = np.arange(-7, 8)
        n_varphi_val = np.zeros(np.size(n_varphi_list), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            n_varphi_val[n] = 1 / (2 * np.pi) * np.sum(
                d1_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list)) * d_varphi

        n_varphi_wavefunction = storage.WaveFunction(n_varphi_list, n_varphi_val)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_varphi_wavefunction.amplitudes = amplitude_modifier(n_varphi_wavefunction.amplitudes)
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}  # if any duplicates, later ones survive
        return plot.wavefunction1d_discrete(n_varphi_wavefunction, **kwargs)

    def plot_charge_2dwavefunction(self, esys=None, mode='real', which=0, n_varphi_list=None, zero_calibrate=True,
                                   **kwargs):
        """Wavefunction in (phi, n_varphi) space"""
        phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)
        if n_varphi_list is None:
            n_varphi_list = np.arange(-7, 8)
            n_varphi_grid = discretization.Grid1d(-7, 7, 15)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        d2_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        ft_amplitudes = np.zeros((phi_grid.pt_count, n_varphi_list.size), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            ft_amplitudes[:, n] = 1 / (2 * np.pi) * np.sum(
                d2_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list), axis=1) * d_varphi

        # grid2d = discretization.GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
        #                                              [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count]]))
        grid2d = discretization.GridSpec(np.asarray([[n_varphi_grid.min_val, n_varphi_grid.max_val,
                                                      n_varphi_grid.pt_count],
                                                     [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                                     ]))

        wfnc = storage.WaveFunctionOnGrid(grid2d, ft_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wfnc.amplitudes = amplitude_modifier(spec_utils.standardize_phases(wfnc.amplitudes))
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}  # if any duplicates, later ones survive
        return plot.wavefunction2d(wfnc, zero_calibrate=zero_calibrate, **kwargs)

    def plot_n_phi_n_varphi_wavefunction(self, esys=None, mode='real', which=0, zero_calibrate=True, **kwargs):
        """Wavefunction in (n_phi, n_varphi) space"""
        phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 400)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)

        n_varphi_list = np.arange(-7, 8)
        n_varphi_grid = discretization.Grid1d(-7, 7, 15)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        d2_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        ft_amplitudes = np.zeros((phi_grid.pt_count, n_varphi_list.size), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            ft_amplitudes[:, n] = 1 / (2 * np.pi) * np.sum(
                d2_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list), axis=1) * d_varphi

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_linspace = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_linspace[0], n_phi_linspace[-1], n_phi_linspace.size)

        fft_amplitudes = np.zeros((n_phi_linspace.size, n_varphi_list.size), dtype=np.complex_)
        for n in range(n_varphi_list.size):
            fft_amplitudes[:, n] = np.fft.ifft(ft_amplitudes[:, n]) * d_phi * phi_grid.pt_count
            fft_amplitudes[:, n] = np.fft.fftshift(fft_amplitudes[:, n])
        grid2d = discretization.GridSpec(np.asarray([
            [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count],
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count]]))
        wfnc = storage.WaveFunctionOnGrid(grid2d, fft_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wfnc.amplitudes = amplitude_modifier(spec_utils.standardize_phases(wfnc.amplitudes))
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}  # if any duplicates, later ones survive



        fig, axs = plt.subplots(figsize=(4, 2))
        axs.plot(n_phi_grid.make_linspace(),wfnc.amplitudes[:,7],'-o')
        axs.set_xlabel(r'$N_\phi$')
        axs.set_ylabel(r'$|\psi|$')
        axs.set_xlim((-2,2))
        axs.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        axs.set_xticklabels(['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])

        return plot.wavefunction2d(wfnc, zero_calibrate=zero_calibrate, **kwargs)

    def get_n_phi_n_varphi_wavefunction(self, esys=None, which=0):
        phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 400)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)

        n_varphi_list = np.arange(-7, 8)
        n_varphi_grid = discretization.Grid1d(-7, 7, 15)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        d2_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        ft_amplitudes = np.zeros((phi_grid.pt_count, n_varphi_list.size), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            ft_amplitudes[:, n] = 1 / (2 * np.pi) * np.sum(
                d2_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list), axis=1) * d_varphi

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_linspace = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_linspace[0], n_phi_linspace[-1], n_phi_linspace.size)

        fft_amplitudes = np.zeros((n_phi_linspace.size, n_varphi_list.size), dtype=np.complex_)
        for n in range(n_varphi_list.size):
            fft_amplitudes[:, n] = np.fft.ifft(ft_amplitudes[:, n]) * d_phi * phi_grid.pt_count
            fft_amplitudes[:, n] = np.fft.fftshift(fft_amplitudes[:, n])
        grid2d = discretization.GridSpec(np.asarray([
            [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count],
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count]]))
        wfnc = storage.WaveFunctionOnGrid(grid2d, fft_amplitudes)
        wfnc.amplitudes = spec_utils.standardize_phases(wfnc.amplitudes)
        return wfnc

    def wavefunction_overlap(self, esys):
        wfnc1 = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=0)
        wfnc2 = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=1)
        return np.abs(np.sum(wfnc1.amplitudes.conjugate() * wfnc2.amplitudes) / np.sum(wfnc1.amplitudes.conjugate() * wfnc1.amplitudes))**2, np.abs(np.sum(
            wfnc1.amplitudes.conjugate() * wfnc2.amplitudes) / np.sum(wfnc2.amplitudes.conjugate() * wfnc2.amplitudes))**2

    # TODO check higher order disorder on junction and capacitance
    def disorder(self):
        """Return disorder Hamiltonian

        Returns
        -------
        ndarray
        """
        disorder_l = - self._dis_el() * self.dL * self._kron3(self.phi_operator(), self.theta_operator(),
                                                              self.varphi_identity())

        disorder_j = 2 * self.EJ * self.dJ * self._kron3(self.sin_phi_2_operator(), self.theta_identity(),
                                                         self.sin_varphi_operator())

        n_varphi_ng_matrix = self._kron3(self.n_phi_operator(), self.theta_identity(), self.n_varphi_ng_operator())
        n_theta_matrix = self._kron3(self.n_phi_operator(), self.n_theta_operator(), self.varphi_identity())
        disorder_c = - 8 * self.EC * self.dC / (1 - self.dC ** 2) * (n_varphi_ng_matrix - n_theta_matrix)

        return disorder_l + disorder_j + disorder_c

    def phi_1_opt(self):
        """
        Returns
        -------
        ndarray
            Returns the phi_1 operator in the LC harmonic oscillator basis, which is the phase across inductor 1
        """
        return self.theta_opt() - self.phi_opt() / 2.0

    def phi_2_opt(self):
        """
        Returns
        -------
        ndarray
            Returns the phi_1 operator in the LC harmonic oscillator basis, which is the phase across inductor 2
        """
        return - self.theta_opt() - self.phi_opt() / 2.0

    def kbt(self):
        return 10 * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # 10 mK

    def q_ind(self, energy):
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt()) * np.sinh(0.5 / 2.0 / self.kbt()) / kn(0,
                                                                                              energy / 2.0 / self.kbt()) / np.sinh(
            energy / 2.0 / self.kbt())

    def ind_loss(self, dL_list):
        """Return the 1/T1 due to inductive loss"""
        eng_obj = self.get_spectrum_vs_paramvals('dL', dL_list, evals_count=2, subtract_ground=True)
        eng = eng_obj.energy_table[:, 1]

        matele_obj_1 = self.get_matelements_vs_paramvals('phi_1_opt', 'dL', dL_list, evals_count=2)
        matele_1 = matele_obj_1.matrixelem_table[:, 0, 1]
        matele_obj_2 = self.get_matelements_vs_paramvals('phi_2_opt', 'dL', dL_list, evals_count=2)
        matele_2 = matele_obj_2.matrixelem_table[:, 0, 1]

        s_ind_1 = 2 / (1 - dL_list) / self.q_ind(eng) / np.tanh(eng / 2.0 / self.kbt())
        s_ind_2 = 2 / (1 + dL_list) / self.q_ind(eng) / np.tanh(eng / 2.0 / self.kbt())

        t1_ind_1 = np.abs(matele_1) ** 2 * s_ind_1
        t1_ind_2 = np.abs(matele_2) ** 2 * s_ind_2

        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        axs[0, 0].plot(dL_list, s_ind_1)
        axs[0, 0].set_xlabel('dL')
        axs[0, 0].set_ylabel('$S_1(\omega)$')

        axs[0, 1].plot(dL_list, s_ind_2)
        axs[0, 1].set_xlabel('dL')
        axs[0, 1].set_ylabel('$S_2(\omega)$')

        axs[1, 0].plot(dL_list, np.abs(matele_1) ** 2)
        axs[1, 0].set_xlabel('dL')
        axs[1, 0].set_ylabel(r'$|\langle 0 | \phi_1 | 1 \rangle|^2 $')

        axs[1, 1].plot(dL_list, np.abs(matele_2) ** 2)
        axs[1, 1].set_xlabel('dL')
        axs[1, 1].set_ylabel(r'$|\langle 0 | \phi_2 | 1 \rangle|^2 $')

        axs[2, 0].plot(dL_list, eng)
        axs[2, 0].set_xlabel('dL')
        axs[2, 0].set_ylabel('$\Delta E$')

        axs[2, 1].plot(dL_list, 1 / (t1_ind_1 + t1_ind_2))
        axs[2, 1].plot(dL_list, 1 / t1_ind_1)
        axs[2, 1].plot(dL_list, 1 / t1_ind_2)
        axs[2, 1].set_xlabel('dL')
        axs[2, 1].set_ylabel('$T_1$')
        axs[2, 1].legend(['total', '$L_1$', '$L_2$'])
        axs[2, 1].set_yscale('log')
        axs[2, 1].set_ylim((1e3, 1e9))

        return 1 / (t1_ind_1 + t1_ind_2)

    def phiphi_opt(self):
        return self.phi_opt() * self.phi_opt()

    def phi_matele_norm(self, flux_list):
        """Return phi matrix element involving the ground state, normalized by ground state <g|phi**2|?>"""
        matele = self.get_matelements_vs_paramvals('phi_opt', 'flux', flux_list, evals_count=5)
        matele_01 = matele.matrixelem_table[:, 0, 1]
        matele_02 = matele.matrixelem_table[:, 0, 2]
        matele_03 = matele.matrixelem_table[:, 0, 3]
        matele_04 = matele.matrixelem_table[:, 0, 4]

        norm = self.get_matelements_vs_paramvals('phiphi_opt', 'flux', flux_list, evals_count=2)
        norm_val = np.real(norm.matrixelem_table[:, 0, 0])

        norm_matele_01 = np.abs(matele_01) ** 2 / norm_val
        norm_matele_02 = np.abs(matele_02) ** 2 / norm_val
        norm_matele_03 = np.abs(matele_03) ** 2 / norm_val
        norm_matele_04 = np.abs(matele_04) ** 2 / norm_val

        fig = plt.figure(figsize=(4, 4))
        plt.plot(flux_list, norm_matele_01)
        plt.plot(flux_list, norm_matele_02)
        plt.plot(flux_list, norm_matele_03)
        plt.plot(flux_list, norm_matele_04)

        plt.xlabel('flux')
        plt.ylabel('matrix element')
        plt.yscale('log')
        plt.ylim((1e-8, 1))
        plt.xlim((0.4, 0.6))
        plt.legend(['0-1', '0-2', '0-3', '0-4'])

        return norm_matele_01, norm_matele_02, norm_matele_03, norm_matele_04

    def n_phi_sq_opt(self):
        return self.n_phi_opt() * self.n_phi_opt()

    def n_phi_matele_norm(self, flux_list):
        """Return n_phi matrix element involving the ground state, normalized by ground state <g|n_phi**2|?>"""
        matele = self.get_matelements_vs_paramvals('n_phi_opt', 'flux', flux_list, evals_count=5)
        matele_01 = matele.matrixelem_table[:, 0, 1]
        matele_02 = matele.matrixelem_table[:, 0, 2]
        matele_03 = matele.matrixelem_table[:, 0, 3]
        matele_04 = matele.matrixelem_table[:, 0, 4]

        norm = self.get_matelements_vs_paramvals('n_phi_sq_opt', 'flux', flux_list, evals_count=2)
        norm_val = np.real(norm.matrixelem_table[:, 0, 0])

        norm_matele_01 = np.abs(matele_01) ** 2 / norm_val
        norm_matele_02 = np.abs(matele_02) ** 2 / norm_val
        norm_matele_03 = np.abs(matele_03) ** 2 / norm_val
        norm_matele_04 = np.abs(matele_04) ** 2 / norm_val

        fig, axs = plt.subplots(figsize=(4, 4))
        axs.plot(flux_list, norm_matele_01)
        axs.plot(flux_list, norm_matele_02)
        axs.plot(flux_list, norm_matele_03)
        axs.plot(flux_list, norm_matele_04)

        axs.set_xlabel('flux')
        axs.set_ylabel(r'$|\langle 0 | n_\phi | 1 \rangle|^2 $')
        axs.set_yscale('log')
        axs.set_ylim((1e-8, 1))
        axs.set_xlim((0.4, 0.6))
        axs.legend(['0-1', '0-2', '0-3', '0-4'])

        return norm_matele_01, norm_matele_02, norm_matele_03, norm_matele_04

    def n_theta_sq_opt(self):
        return self.n_theta_opt() * self.n_theta_opt()

    def n_theta_matele_norm(self, flux_list):
        """Return n_theta matrix element involving the ground state, normalized by ground state <g|n_theta**2|?>"""
        matele = self.get_matelements_vs_paramvals('n_theta_opt', 'flux', flux_list, evals_count=5)
        matele_01 = matele.matrixelem_table[:, 0, 1]
        matele_02 = matele.matrixelem_table[:, 0, 2]
        matele_03 = matele.matrixelem_table[:, 0, 3]
        matele_04 = matele.matrixelem_table[:, 0, 4]

        norm = self.get_matelements_vs_paramvals('n_theta_sq_opt', 'flux', flux_list, evals_count=2)
        norm_val = np.real(norm.matrixelem_table[:, 0, 0])

        norm_matele_01 = np.abs(matele_01) ** 2 / norm_val
        norm_matele_02 = np.abs(matele_02) ** 2 / norm_val
        norm_matele_03 = np.abs(matele_03) ** 2 / norm_val
        norm_matele_04 = np.abs(matele_04) ** 2 / norm_val

        fig, axs = plt.subplots(figsize=(4, 4))
        axs.plot(flux_list, norm_matele_01)
        axs.plot(flux_list, norm_matele_02)
        axs.plot(flux_list, norm_matele_03)
        axs.plot(flux_list, norm_matele_04)

        axs.set_xlabel('flux')
        axs.set_ylabel(r'$|\langle 0 | n_\theta | 1 \rangle|^2 $')
        axs.set_yscale('log')
        axs.set_ylim((1e-8, 1))
        axs.set_xlim((0.4, 0.6))
        axs.legend(['0-1', '0-2', '0-3', '0-4'])

        return norm_matele_01, norm_matele_02, norm_matele_03, norm_matele_04

    def n_varphi_sq_opt(self):
        return self.n_varphi_opt() * self.n_varphi_opt()

    def n_varphi_matele_norm(self, flux_list):
        """Return n_varphi matrix element involving the ground state, normalized by ground state <g|n_varphi**2|?>"""
        matele = self.get_matelements_vs_paramvals('n_varphi_opt', 'flux', flux_list, evals_count=5)
        matele_01 = matele.matrixelem_table[:, 0, 1]
        matele_02 = matele.matrixelem_table[:, 0, 2]
        matele_03 = matele.matrixelem_table[:, 0, 3]
        matele_04 = matele.matrixelem_table[:, 0, 4]

        norm = self.get_matelements_vs_paramvals('n_varphi_sq_opt', 'flux', flux_list, evals_count=2)
        norm_val = np.real(norm.matrixelem_table[:, 0, 0])

        norm_matele_01 = np.abs(matele_01) ** 2 / norm_val
        norm_matele_02 = np.abs(matele_02) ** 2 / norm_val
        norm_matele_03 = np.abs(matele_03) ** 2 / norm_val
        norm_matele_04 = np.abs(matele_04) ** 2 / norm_val

        fig, axs = plt.subplots(figsize=(4, 4))
        axs.plot(flux_list, norm_matele_01)
        axs.plot(flux_list, norm_matele_02)
        axs.plot(flux_list, norm_matele_03)
        axs.plot(flux_list, norm_matele_04)

        axs.set_xlabel('flux')
        axs.set_ylabel(r'$|\langle 0 | n_\varphi | 1 \rangle|^2 $')
        axs.set_yscale('log')
        axs.set_ylim((1e-8, 1))
        axs.set_xlim((0.4, 0.6))
        axs.legend(['0-1', '0-2', '0-3', '0-4'])

        return norm_matele_01, norm_matele_02, norm_matele_03, norm_matele_04
