# rhombus.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import math
import os

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils


# —Rhombus qubit ————————————————————————
class Rhombus(base.QubitBaseClass, serializers.Serializable):
    # TODO: change the following docs
    r"""double Cooper pair tunneling qubit

    | [1] Smith et al., NPJ Quantum Inf. 6, 8 (2020) http://www.nature.com/articles/s41534-019-0231-2

    .. math::

        H H_\text{dcp} = 4E_\text{C}[2n_\phi^2+\frac{1}{2}(n_\varphi-N_\text{g}-n_\theta)^2+xn_\theta^2]
                           +E_\text{L}(\frac{1}{4}\phi^2+\theta^2)
                           -2E_\text{J}\cos(\varphi)\cos(\frac{\phi}{2}+\frac{\varphi_\text{ext}}{2})

    The employed basis are harmonic basis for :math:`\phi,\theta` and charge basis for :math:`\varphi`. The cosine term in the
    potential is handled via matrix exponentiation. Initialize with, for example::

        qubit = Dcp(EJ=15.0, EC=2.0, EL=1.0, x=0.02, dC=0, dL=0, dJ=0, flux=0.5, Ng=0, N0=7, q0=30, p0=7)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    EL: float
        inductive energy
    x: float
        ratio of the junction capacitance to the shunt capacitance x = C_J / C_shunt
    dC: float
        disorder in capacitance, i.e., EC / (1 \pm dC)
    dL: float
        disorder in inductance, i.e., EL / (1 \pm dL)
    dJ: float
        disorder in junction energy, i.e., EJ * (1 \pm dJ)
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
    ECS = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJ1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJ2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ngs = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    NS = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    N1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    N2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, ECS, EC1, EC2, EJ1, EJ2, flux, ngs, ng1, ng2, kbt, NS, N1, N2, truncated_dim=None):
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.ECS = ECS
        self.EC1 = EC1
        self.EC2 = EC2
        self.flux = flux
        self.ngs = ngs
        self.ng1 = ng1
        self.ng2 = ng2
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # temperature unit mK
        self.NS = NS
        self.N1 = N1
        self.N2 = N2
        self._default_phi_1_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        self._default_phi_2_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        self._default_theta_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_

    @staticmethod
    def default_params():
        return {
            'EJ1': 15.0,
            'EJ2': 15.0,
            'ECS': 2.0,
            'EC1': 2.0,
            'EC2': 2.0,
            'flux': np.pi,
            'ngs': 0.0,
            'ng1': 0.0,
            'ng2': 0.0,
            'NS': 7,
            'N1': 7,
            'N2': 7,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['flux', 'ngs', 'ng1', 'ng2', 'NS', 'N1', 'N2', 'truncated_dim']

    def dim_phi_1(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi_1' degree of freedom."""
        return 2 * self.N1 + 1

    def dim_phi_2(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi_2' degree of freedom."""
        return 2 * self.N2 + 1

    def dim_theta(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return 2 * self.NS + 1

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi_1() * self.dim_phi_2() * self.dim_theta()

    def _n_phi_1_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_phi_1` in the charge basis
        """
        diag_elements = np.arange(-self.N1, self.N1 + 1)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.dim_phi_1(), self.dim_phi_1())).tocsc()

    def n_phi_1_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_phi_1` in the total Hilbert space
        """
        return self._kron3(self._n_phi_1_operator(), self._identity_phi_2(), self._identity_theta())

    def _cos_phi_1_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\cos \\phi_1` in the charge basis
        """
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_1()), [1]),
                                         shape=(self.dim_phi_1(), self.dim_phi_1())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_1()), [-1]),
                                          shape=(self.dim_phi_1(), self.dim_phi_1())).tocsc()
        return cos_op

    def _sin_phi_1_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\sin \\phi_1` in the charge basis
        """
        sin_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_1()), [1]),
                                         shape=(self.dim_phi_1(), self.dim_phi_1())).tocsc()
        sin_op -= 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_1()), [-1]),
                                          shape=(self.dim_phi_1(), self.dim_phi_1())).tocsc()
        return sin_op * (-1j)

    def _n_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_phi_2` in the charge basis
        """
        diag_elements = np.arange(-self.N2, self.N2 + 1)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.dim_phi_2(), self.dim_phi_2())).tocsc()

    def n_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_phi_2` in the total Hilbert space
        """
        return self._kron3(self._identity_phi_1(), self._n_phi_2_operator(), self._identity_theta())

    def _cos_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\cos \\phi_2` in the charge basis
        """
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_2()), [1]),
                                         shape=(self.dim_phi_2(), self.dim_phi_2())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_2()), [-1]),
                                          shape=(self.dim_phi_2(), self.dim_phi_2())).tocsc()
        return cos_op

    def _sin_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\sin \\phi_2` in the charge basis
        """
        sin_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_2()), [1]),
                                         shape=(self.dim_phi_2(), self.dim_phi_2())).tocsc()
        sin_op -= 0.5 * sparse.dia_matrix((np.ones(self.dim_phi_2()), [-1]),
                                          shape=(self.dim_phi_2(), self.dim_phi_2())).tocsc()
        return sin_op * (-1j)

    def _n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_theta` in the charge basis
        """
        diag_elements = np.arange(-self.NS, self.NS + 1)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.dim_theta(), self.dim_theta())).tocsc()

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_theta` in the total Hilbert space
        """
        return self._kron3(self._identity_phi_1(), self._identity_phi_2(), self._n_theta_operator())

    def _cos_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\cos \\theta` in the charge basis
        """
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_theta()), [1]),
                                         shape=(self.dim_theta(), self.dim_theta())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.dim_theta()), [-1]),
                                          shape=(self.dim_theta(), self.dim_theta())).tocsc()
        return cos_op

    def _sin_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\sin \\theta` in the charge basis
        """
        sin_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_theta()), [1]),
                                         shape=(self.dim_theta(), self.dim_theta())).tocsc()
        sin_op -= 0.5 * sparse.dia_matrix((np.ones(self.dim_theta()), [-1]),
                                          shape=(self.dim_theta(), self.dim_theta())).tocsc()
        return sin_op * (-1j)

    def _kron3(self, mat1, mat2, mat3):
        """
        Kronecker product of three matrices

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.kron(sparse.kron(mat1, mat2, format='csc'), mat3, format='csc')

    def _identity_phi_1(self):
        """
        Identity operator acting only on the :math:`\phi_1` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_phi_1()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def _identity_phi_2(self):
        """
        Identity operator acting only on the :math:`\phi_2` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_phi_2()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def _identity_theta(self):
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_theta()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def total_identity(self):
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron3(self._identity_phi_1(), self._identity_phi_2(), self._identity_theta())

    def hamiltonian(self):
        """
        Rhombus qubit Hamiltonian

        Returns
        -------
        ndarray
        """

        phi_1_ng_mat = 4 * self.EC1 * (self.n_phi_1_operator() - self.ng1 * self.total_identity()) ** 2
        phi_2_ng_mat = 4 * self.EC2 * (self.n_phi_2_operator() - self.ng2 * self.total_identity()) ** 2
        theta_ng_mat = 4 * self.ECS * (self.n_theta_operator() - self.ngs * self.total_identity()) ** 2

        theta_flux_term = 2 * (
                self._cos_theta_operator() * np.cos(self.flux * 2 * np.pi) + self._sin_theta_operator() * np.sin(
            self.flux * 2 * np.pi))
        potential_mat = - self._kron3(self._identity_phi_1(), self._identity_phi_2(), theta_flux_term) * (
                self._kron3(self.EJ1 * self._cos_phi_1_operator(), self._identity_phi_2(), self._identity_theta()) +
                self._kron3(self._identity_phi_1(), self.EJ2 * self._cos_phi_2_operator(), self._identity_theta()))

        return phi_1_ng_mat + phi_2_ng_mat + theta_ng_mat + potential_mat

    def potential(self, phi_1, phi_2):
        """
        Rhombus qubit potential evaluated at `phi_1, phi_2`, with `theta=0`

        Parameters
        ----------
        phi_1: float or ndarray
            float value of the phase variable `phi_1`
        phi_2: float or ndarray
            float value of the phase variable `phi_2`

        Returns
        -------
        float or ndarray
        """
        return - 2 * np.cos(-self.flux * 2 * np.pi) * (self.EJ1 * np.cos(phi_1) + self.EJ2 * np.cos(phi_2))

    def plot_potential(self, phi_1_grid=None, phi_2_grid=None, contour_vals=None, **kwargs):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_1_grid: Grid1d, option
            used for setting a custom grid for phi_1; if None use self._default_phi_1_grid
        phi_2_grid: Grid1d, option
            used for setting a custom grid for phi_2; if None use self._default_phi_2_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        phi_1_grid = phi_1_grid or self._default_phi_1_grid
        phi_2_grid = phi_2_grid or self._default_phi_2_grid

        x_vals = phi_1_grid.make_linspace()
        y_vals = phi_2_grid.make_linspace()
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

    def _tensor_index(self, index_phi_1, index_phi_2, index_theta):
        """
        Return the index of the coefficient of the wavefunction, corresponding to the indices of phi_1, phi_2, theta
        """
        return (index_phi_1 * self.dim_phi_2() + index_phi_2) * self.dim_theta() + index_theta

    def _tensor_index_inv(self, index_evec):
        """
        Return the indices of phi_1, phi_2, and theta, corresponding to the index of the coefficient of the wavefunction
        """
        index_theta = index_evec % self.dim_theta()
        index_temp = index_evec // self.dim_theta()
        index_phi_2 = index_temp % self.dim_phi_2()
        index_phi_1 = index_temp // self.dim_phi_2()
        return index_phi_1, index_phi_2, index_theta

    def wavefunction(self, esys=None, which=0, phi_1_grid=None, phi_2_grid=None, theta_grid=None):
        """
        Return a 3D wave function in phi_1, phi_2, theta basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)
        phi_1_grid: Grid1d, option
            used for setting a custom grid for phi_1; if None use self._default_phi_1_grid
        phi_2_grid: Grid1d, option
            used for setting a custom grid for phi_2; if None use self._default_phi_2_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use self._default_theta_grid

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_1_grid = phi_1_grid or self._default_phi_1_grid
        phi_2_grid = phi_2_grid or self._default_phi_2_grid
        theta_grid = theta_grid or self._default_theta_grid

        phi_1_basis_labels = phi_1_grid.make_linspace()
        phi_2_basis_labels = phi_2_grid.make_linspace()
        theta_basis_labels = theta_grid.make_linspace()

        wavefunc_basis_amplitudes = np.reshape(evecs[:, which], self.hilbertdim())
        wavefunc_amplitudes = np.zeros((phi_1_grid.pt_count, phi_2_grid.pt_count, theta_grid.pt_count),
                                       dtype=np.complex_)
        for n in range(self.hilbertdim()):
            n_phi_1, n_phi_2, n_theta, = self._tensor_index_inv(n)
            n_phi_1 += - self.N1
            n_phi_2 += - self.N2
            n_theta += - self.NS
            phi_1_wavefunc_amplitudes = np.exp(-1j * n_phi_1 * phi_1_basis_labels) / (2 * np.pi) ** 0.5
            phi_2_wavefunc_amplitudes = np.exp(-1j * n_phi_2 * phi_2_basis_labels) / (2 * np.pi) ** 0.5
            theta_wavefunc_amplitudes = np.exp(-1j * n_theta * theta_basis_labels) / (2 * np.pi) ** 0.5
            wavefunc_amplitudes += wavefunc_basis_amplitudes[n] * np.tensordot(
                np.tensordot(phi_1_wavefunc_amplitudes, phi_2_wavefunc_amplitudes, 0), theta_wavefunc_amplitudes, 0)

        grid3d = discretization.GridSpec(np.asarray([[phi_1_grid.min_val, phi_1_grid.max_val, phi_1_grid.pt_count],
                                                     [phi_2_grid.min_val, phi_2_grid.max_val, phi_2_grid.pt_count],
                                                     [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid3d, wavefunc_amplitudes)

    def plot_phi_1_theta_wavefunction(self, esys=None, which=0, phi_1_grid=None, theta_grid=None, mode='abs',
                                      zero_calibrate=True,
                                      **kwargs):
        """
        Plots 2D phase-basis wave function at phi_2 = 0
        """
        phi_1_grid = phi_1_grid or self._default_phi_1_grid
        phi_2_grid = discretization.Grid1d(0, 0, 1)
        theta_grid = theta_grid or self._default_theta_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_1_grid=phi_1_grid, phi_2_grid=phi_2_grid, theta_grid=theta_grid,
                                     which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
             [phi_1_grid.min_val, phi_1_grid.max_val, phi_1_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_1_grid.pt_count, theta_grid.pt_count)))
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def plot_phi_2_theta_wavefunction(self, esys=None, which=0, phi_2_grid=None, theta_grid=None, mode='abs',
                                      zero_calibrate=True,
                                      **kwargs):
        """
        Plots 2D phase-basis wave function at phi_1 = 0
        """
        phi_1_grid = discretization.Grid1d(0, 0, 1)
        phi_2_grid = phi_2_grid or self._default_phi_2_grid
        theta_grid = theta_grid or self._default_theta_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_1_grid=phi_1_grid, phi_2_grid=phi_2_grid, theta_grid=theta_grid,
                                     which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
             [phi_2_grid.min_val, phi_2_grid.max_val, phi_2_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_2_grid.pt_count, theta_grid.pt_count)))
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def plot_phi_1_phi_2_wavefunction(self, esys=None, which=0, phi_1_grid=None, phi_2_grid=None, mode='abs',
                                      zero_calibrate=True,
                                      **kwargs):
        """
        Plots 2D phase-basis wave function at theta = 0
        """
        phi_1_grid = phi_1_grid or self._default_phi_1_grid
        phi_2_grid = phi_2_grid or self._default_phi_2_grid
        theta_grid = discretization.Grid1d(0, 0, 1)

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_1_grid=phi_1_grid, phi_2_grid=phi_2_grid, theta_grid=theta_grid,
                                     which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[phi_2_grid.min_val, phi_2_grid.max_val, phi_2_grid.pt_count],
             [phi_1_grid.min_val, phi_1_grid.max_val, phi_1_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_1_grid.pt_count, phi_2_grid.pt_count)))
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def ft_wavefunction(self, esys=None, which=0, n_phi_1_list=None, n_phi_1_grid=None, n_phi_2_list=None,
                        n_phi_2_grid=None, n_theta_list=None, n_theta_grid=None):
        phi_1_grid = discretization.Grid1d(0, 2 * np.pi, 100)
        phi_2_grid = discretization.Grid1d(0, 2 * np.pi, 100)
        theta_grid = discretization.Grid1d(0, 2 * np.pi, 100)

        n_phi_1_list = n_phi_1_list or np.arange(-10, 11)
        n_phi_1_grid = n_phi_1_grid or discretization.Grid1d(-10, 10, 21)
        n_phi_2_list = n_phi_2_list or np.arange(-10, 11)
        n_phi_2_grid = n_phi_2_grid or discretization.Grid1d(-10, 10, 21)
        n_theta_list = n_theta_list or np.arange(-10, 11)
        n_theta_grid = n_theta_grid or discretization.Grid1d(-10, 10, 21)

        wavefunc = self.wavefunction(esys, phi_1_grid=phi_1_grid, phi_2_grid=phi_2_grid, theta_grid=theta_grid,
                                     which=which)

        theta_grid_list = theta_grid.make_linspace()
        phi_1_phi_2_theta_amplitudes = spec_utils.standardize_phases(wavefunc.amplitudes)
        phi_1_phi_2_n_theta_amplitudes = np.zeros((phi_1_grid.pt_count, phi_2_grid.pt_count, n_theta_list.size),
                                                  dtype=np.complex_)
        d_theta = theta_grid_list[1] - theta_grid_list[0]
        for n in range(n_theta_list.size):
            phi_1_phi_2_n_theta_amplitudes[:, :, n] = 1 / (2 * np.pi) * np.tensordot(phi_1_phi_2_theta_amplitudes,
                                                                                     np.exp(1j * n_theta_list[
                                                                                         n] * theta_grid_list),
                                                                                     axes=[2, 0]) * d_theta

        phi_2_grid_list = phi_2_grid.make_linspace()
        phi_1_n_phi_2_n_theta_amplitudes = np.zeros((phi_1_grid.pt_count, n_phi_2_list.size, n_theta_list.size),
                                                    dtype=np.complex_)
        d_phi_2 = phi_2_grid_list[1] - phi_2_grid_list[0]
        for n in range(n_phi_2_list.size):
            phi_1_n_phi_2_n_theta_amplitudes[:, n, :] = 1 / (2 * np.pi) * np.tensordot(phi_1_phi_2_n_theta_amplitudes,
                                                                                       np.exp(1j * n_phi_2_list[
                                                                                           n] * phi_2_grid_list),
                                                                                       axes=[1, 0]) * d_phi_2

        phi_1_grid_list = phi_1_grid.make_linspace()
        n_phi_1_n_phi_2_n_theta_amplitudes = np.zeros((n_phi_1_list.size, n_phi_2_list.size, n_theta_list.size),
                                                      dtype=np.complex_)
        d_phi_1 = phi_1_grid_list[1] - phi_1_grid_list[0]
        for n in range(n_phi_1_list.size):
            n_phi_1_n_phi_2_n_theta_amplitudes[n, :, :] = 1 / (2 * np.pi) * np.tensordot(
                phi_1_n_phi_2_n_theta_amplitudes, np.exp(1j * n_phi_1_list[n] * phi_1_grid_list), axes=[0, 0]) * d_phi_1

        grid3d = discretization.GridSpec(np.asarray([
            [n_phi_1_grid.min_val, n_phi_1_grid.max_val, n_phi_1_grid.pt_count],
            [n_phi_2_grid.min_val, n_phi_2_grid.max_val, n_phi_2_grid.pt_count],
            [n_theta_grid.min_val, n_theta_grid.max_val, n_theta_grid.pt_count]]))

        n_phi_1_n_phi_2_n_theta_wavefunction = storage.WaveFunctionOnGrid(grid3d, n_phi_1_n_phi_2_n_theta_amplitudes)
        n_phi_1_n_phi_2_n_theta_wavefunction.amplitudes = spec_utils.standardize_phases(
            n_phi_1_n_phi_2_n_theta_wavefunction.amplitudes)
        return n_phi_1_n_phi_2_n_theta_wavefunction

    def plot_n_phi_1_n_phi_2_wavefunction(self, ft_wfnc, mode='abs', zero_calibrate=True, **kwargs):
        """
        Plots 2D phase-basis wave function at n_theta = 0
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_1_grid = discretization.Grid1d(-10, 10, 21)
        n_phi_2_grid = discretization.Grid1d(-10, 10, 21)

        grid2d = discretization.GridSpec(np.asarray(
            [[n_phi_2_grid.min_val, n_phi_2_grid.max_val, n_phi_2_grid.pt_count],
             [n_phi_1_grid.min_val, n_phi_1_grid.max_val, n_phi_1_grid.pt_count]]))
        amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(ft_wfnc.amplitudes[:, :, 10]))
        wavefunc = storage.WaveFunctionOnGrid(grid2d, amplitudes)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def plot_n_phi_1_n_theta_wavefunction(self, ft_wfnc, mode='abs', zero_calibrate=True, **kwargs):
        """
        Plots 2D phase-basis wave function at n_phi_2 = 0
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_1_grid = discretization.Grid1d(-10, 10, 21)
        n_theta_grid = discretization.Grid1d(-10, 10, 21)

        grid2d = discretization.GridSpec(np.asarray(
            [[n_theta_grid.min_val, n_theta_grid.max_val, n_theta_grid.pt_count],
             [n_phi_1_grid.min_val, n_phi_1_grid.max_val, n_phi_1_grid.pt_count]]))
        amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(ft_wfnc.amplitudes[:, 10, :]))
        wavefunc = storage.WaveFunctionOnGrid(grid2d, amplitudes)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def plot_n_theta_n_phi_2_wavefunction(self, ft_wfnc, mode='abs', zero_calibrate=True, **kwargs):
        """
        Plots 2D phase-basis wave function at n_phi_1 = 0
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_theta_grid = discretization.Grid1d(-10, 10, 21)
        n_phi_2_grid = discretization.Grid1d(-10, 10, 21)

        grid2d = discretization.GridSpec(np.asarray(
            [[n_phi_2_grid.min_val, n_phi_2_grid.max_val, n_phi_2_grid.pt_count],
             [n_theta_grid.min_val, n_theta_grid.max_val, n_theta_grid.pt_count]]))
        amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(ft_wfnc.amplitudes[:, :, 10]))
        wavefunc = storage.WaveFunctionOnGrid(grid2d, amplitudes)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def get_t2_ng1_noise(self, para_name, para_vals):
        original_ng1 = self.ng1
        self.ng1 = 0.0
        energy_ng1_0 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                      subtract_ground=True).energy_table[:, 1]
        self.ng1 = 0.5
        energy_ng1_5 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                      subtract_ground=True).energy_table[:, 1]
        self.ng1 = original_ng1
        epsilon = np.abs(energy_ng1_5 - energy_ng1_0)
        return 1.49734 / epsilon * 1e-6  # unit in ms

    def get_t2_ng2_noise(self, para_name, para_vals):
        original_ng2 = self.ng2
        self.ng2 = 0.0
        energy_ng2_0 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                      subtract_ground=True).energy_table[:, 1]
        self.ng2 = 0.5
        energy_ng2_5 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                      subtract_ground=True).energy_table[:, 1]
        self.ng2 = original_ng2
        epsilon = np.abs(energy_ng2_5 - energy_ng2_0)
        return 1.49734 / epsilon * 1e-6  # unit in ms

    def get_t2_ngs_noise(self, para_name, para_vals):
        original_ngs = self.ngs
        self.ngs = 0.0
        energy_ngs_0 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                      subtract_ground=True).energy_table[:, 1]
        self.ngs = 0.5
        energy_ngs_5 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                      subtract_ground=True).energy_table[:, 1]
        self.ngs = original_ngs
        epsilon = np.abs(energy_ngs_5 - energy_ngs_0)
        return 1.49734 / epsilon * 1e-6  # unit in ms

    def get_t2_ej1_noise(self, para_name, para_vals):
        orginal_ej1 = self.EJ1
        delta = 1e-7
        pts = 11
        ej1_list = np.linspace(orginal_ej1 - delta, orginal_ej1 + delta, pts)
        energy = np.zeros((pts, para_vals.size))
        for i in range(pts):
            self.EJ1 = ej1_list[i]
            energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                          subtract_ground=True).energy_table[:, 1]
        first_derivative = np.gradient(energy, ej1_list, axis=0)[int(np.round(pts / 2)), :]
        self.EJ1 = orginal_ej1
        return np.abs(1 / (5e-7 * orginal_ej1 * first_derivative) * 1e-6)  # unit in ms

    def get_t2_ej2_noise(self, para_name, para_vals):
        orginal_ej2 = self.EJ2
        delta = 1e-7
        pts = 11
        ej2_list = np.linspace(orginal_ej2 - delta, orginal_ej2 + delta, pts)
        energy = np.zeros((pts, para_vals.size))
        for i in range(pts):
            self.EJ2 = ej2_list[i]
            energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                          subtract_ground=True).energy_table[:, 1]
        first_derivative = np.gradient(energy, ej2_list, axis=0)[int(np.round(pts / 2)), :]
        self.EJ2 = orginal_ej2
        return np.abs(1 / (5e-7 * orginal_ej2 * first_derivative) * 1e-6)  # unit in ms

    def noise_analysis(self, para_name, para_vals):
        t2_ng1 = self.get_t2_ng1_noise(para_name, para_vals)
        t2_ng2 = self.get_t2_ng2_noise(para_name, para_vals)
        t2_ngs = self.get_t2_ngs_noise(para_name, para_vals)
        t2_ej1 = self.get_t2_ej1_noise(para_name, para_vals)
        t2_ej2 = self.get_t2_ej2_noise(para_name, para_vals)

        plt.figure(figsize=(4, 4))
        plt.plot(para_vals, t2_ng1, '--')
        plt.plot(para_vals, t2_ng2, '--')
        plt.plot(para_vals, t2_ngs, '--')
        plt.plot(para_vals, t2_ej1, '--')
        plt.plot(para_vals, t2_ej2, '--')
        plt.legend(['T2_ng1', 'T2_ng2', 'T2_ngs', 'T1_ej1', 'T1_ej2'])
        plt.xlabel(para_name)
        plt.ylabel('T1, T2 (ms)')
        plt.yscale('log')

    def print_noise(self):
        t2_ng1 = self.get_t2_ng1_noise('kbt', np.array([10]))
        t2_ng2 = self.get_t2_ng2_noise('kbt', np.array([10]))
        t2_ngs = self.get_t2_ngs_noise('kbt', np.array([10]))
        t2_ej1 = self.get_t2_ej1_noise('kbt', np.array([10]))
        t2_ej2 = self.get_t2_ej2_noise('kbt', np.array([10]))
        return print(' T2_ng1=', t2_ng1, '\n T2_ng2=', t2_ng2, '\n T2_ngs=', t2_ngs, '\n T2_ej1=',
                     t2_ej1, '\n T2_ej2=', t2_ej2)
