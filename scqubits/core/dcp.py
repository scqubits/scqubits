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

    def __init__(self, EJ, EC, EL, x, flux, Ng, N0, q0, p0, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.x = x
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

    def phi_zpf(self):
        """
        Returns
        -------
        float
            Returns zero point fluctuation for the phi degree of freedom.
        """
        return (8.0 * self.EC / self.EL) ** 0.25

    def n_theta_zpf(self):
        """
        Returns
        -------
        float
            Returns zero point fluctuation for the n_theta degree of freedom.
        """
        return 0.5 * (self.EL / self.EC / self.x) ** 0.25

    def phi_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the phi degree of freedom.
        """
        return math.sqrt(8.0 * self.EL * self.EC)

    def theta_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the theta degree of freedom.
        """
        return math.sqrt(16.0 * self.x * self.EL * self.EC)

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.phi_hilbertdim()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.phi_zpf()

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator in the LC harmonic oscillator basis
        """
        dimension = self.theta_hilbertdim()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) * self.n_theta_zpf()

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
        return sparse.dia_matrix((diag_elements, [0]), shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()

    def cos_varphi_operator(self):
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [1]), shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [-1]), shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        return cos_op

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

    def hamiltonian(self):  # follow W.C. Smith, A. Kou, X. Xiao, U. Vool, and M.H. Devoret, Npj Quantum Inf. 6, 8 (2020).
        """Return Hamiltonian

        Returns
        -------
        ndarray
        """
        phi_osc_matrix = self._kron3(op.number_sparse(self.phi_hilbertdim(), self.phi_plasma()), self.theta_identity(), self.varphi_identity())
        theta_osc_matrix = self._kron3(self.phi_identity(), op.number_sparse(self.theta_hilbertdim(), self.theta_plasma()), self.varphi_identity())

        n_varphi_ng_matrix = self._kron3(self.phi_identity(), self.theta_identity(), self.n_varphi_ng_operator())
        n_theta_matrix = self._kron3(self.phi_identity(), self.n_theta_operator(), self.varphi_identity())
        cross_kinetic_matrix = 2 * self.EC * (n_varphi_ng_matrix - n_theta_matrix) * (n_varphi_ng_matrix - n_theta_matrix)

        phi_flux_term = self.cos_phi_2_operator() * np.cos(self.flux * np.pi) - self.sin_phi_2_operator() * np.sin(self.flux * np.pi)
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
        return self.EL * (0.25 * phi * phi) - 2 * self.EJ * np.cos(varphi) * np.cos(phi * 0.5 + np.pi * self.flux)

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
            kwargs['figsize'] = (5, 5)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self):
        """Return total Hilbert space dimension."""
        return self.phi_hilbertdim() * self.theta_hilbertdim() * self.varphi_hilbertdim()

    def phi_osc(self):
        """Return the oscillator strength of phi degree of freedom"""
        return (32 * self.EC / self.EL) ** 0.25

    def theta_osc(self):
        """Return the oscillator strength of theta degree of freedom"""
        return (4 * self.EC * self.x / self.EL) ** 0.25

    def tensor_index(self, index_phi, index_theta, index_varphi):
        """Return the index of the coefficient of the wavefunction, corresponding to the indices of phi, theta, and varphi """
        return (index_phi * self.theta_hilbertdim() + index_theta) * self.varphi_hilbertdim() + index_varphi

    def tensor_index_inv(self, index_evec):
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
        wavefunc_amplitudes = np.zeros((phi_grid.pt_count, theta_grid.pt_count, varphi_grid.pt_count), dtype=np.complex_)
        for n in range(self.hilbertdim()):
            n_phi, n_theta, n_varphi = self.tensor_index_inv(n)
            num_varphi = n_varphi - self.N0
            phi_wavefunc_amplitudes = osc.harm_osc_wavefunction(n_phi, phi_basis_labels, self.phi_osc())
            theta_wavefunc_amplitudes = osc.harm_osc_wavefunction(n_theta, theta_basis_labels, self.theta_osc())
            varphi_wavefunc_amplitudes = np.exp(-1j * num_varphi * varphi_basis_labels) / (2 * np.pi) ** 0.5
            wavefunc_amplitudes += wavefunc_basis_amplitudes[n] * np.tensordot(np.tensordot(phi_wavefunc_amplitudes, theta_wavefunc_amplitudes, 0), varphi_wavefunc_amplitudes, 0)

        grid3d = discretization.GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count], [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count], [varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid3d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None, mode='abs', zero_calibrate=True, **kwargs):
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
        theta_grid = discretization.Grid1d(0,0,1)
        varphi_grid = varphi_grid or self._default_varphi_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid, which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray([[varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count], [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count)))
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
