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

import math
import os

import numpy as np
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
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    x = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    Ng = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    N0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    q0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    p0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, EL, x, dL, dC, dJ, flux, Ng, kbt, N0, q0, p0, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.x = x
        self.dL = dL
        self.dC = dC
        self.dJ = dJ
        self.flux = flux
        self.Ng = Ng
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # temperature unit mK
        self.N0 = N0
        self.q0 = q0
        self.p0 = p0
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._default_phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_varphi_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/double_cooper_pair_tunneling_qubit.png')

    @staticmethod
    def default_params():
        return {
            'EJ': 15.0,
            'EC': 2.0,
            'EL': 1.0,
            'x': 0.02,
            'dC': 0.0,
            'dL': 0.0,
            'dJ': 0.0,
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

    def dim_phi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi' degree of freedom."""
        return self.p0

    def dim_theta(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return self.q0

    def dim_varphi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`varphi' degree of freedom."""
        return 2 * self.N0 + 1

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi() * self.dim_theta() * self.dim_varphi()

    def _dis_el(self):
        """
        Returns
        -------
        float
            Returns the inductive energy renormalized by with disorder."""
        return self.EL / (1 - self.dL ** 2)

    def _dis_ec(self):
        """
        Returns
        -------
        float
            Returns the capacitance energy renormalized by with disorder."""
        return self.EC / (1 - self.dC ** 2)

    def phi_osc(self):
        """
        Returns
        -------
        float
            Returns the oscillator strength of :math:`phi' degree of freedom."""
        return (32 * self._dis_ec() / self._dis_el()) ** 0.25

    def theta_osc(self):
        """
        Returns
        -------
        float
            Returns the oscillator strength of :math:`theta' degree of freedom."""
        return (4 * self._dis_ec() * self.x / self._dis_el()) ** 0.25

    def phi_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency of :math:`phi' degree of freedom.
        """
        return math.sqrt(8.0 * self._dis_el() * self._dis_ec())

    def theta_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency of :math:`theta' degree of freedom.
        """
        return math.sqrt(16.0 * self.x * self._dis_el() * self._dis_ec())

    def _phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_phi()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.phi_osc() / math.sqrt(2)

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in total Hilbert space
        """
        return self._kron3(self._phi_operator(), self._identity_theta(), self._identity_varphi())

    def _n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_phi()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.phi_osc() * math.sqrt(2))

    def n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi' operator in total Hilbert space
        """
        return self._kron3(self._n_phi_operator(), self._identity_theta(), self._identity_varphi())

    def _theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_theta()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.theta_osc() / math.sqrt(2)

    def theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._theta_operator(), self._identity_varphi())

    def _n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_theta()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.theta_osc() * math.sqrt(2))

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_theta' operator in total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._n_theta_operator(), self._identity_varphi())

    def _exp_i_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\phi/2}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self._phi_operator() * 0.5
        return expm(exponent)

    def _cos_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/2` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self._exp_i_phi_2_operator()
        cos_phi_op += cos_phi_op.conj().T
        return np.real(cos_phi_op)

    def _sin_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/2` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self._exp_i_phi_2_operator()
        sin_phi_op += sin_phi_op.conj().T
        return np.real(sin_phi_op)

    def _exp_i_phi_4_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\phi/4}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self._phi_operator() * 0.25
        return expm(exponent)

    def _cos_phi_4_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/4` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self._exp_i_phi_4_operator()
        cos_phi_op += cos_phi_op.conj().T
        return np.real(cos_phi_op)

    def _sin_phi_4_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/2` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self._exp_i_phi_4_operator()
        sin_phi_op += sin_phi_op.conj().T
        return np.real(sin_phi_op)

    def _exp_i_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\theta}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self._theta_operator()
        return expm(exponent)

    def _cos_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\theta` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self._exp_i_theta_operator()
        cos_phi_op += cos_phi_op.conj().T
        return np.real(cos_phi_op)

    def _sin_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\theta` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self._exp_i_theta_operator()
        sin_phi_op += sin_phi_op.conj().T
        return np.real(sin_phi_op)

    def _n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_varphi` in the charge basis
        """
        diag_elements = np.arange(-self.N0, self.N0 + 1)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_varphi` in the total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._identity_theta(), self._n_varphi_operator())

    def _cos_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\cos \\varphi` in the charge basis
        """
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [1]),
                                         shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [-1]),
                                          shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        return cos_op

    def _sin_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\sin \\varphi` in the charge basis
        """
        sin_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [1]),
                                         shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        sin_op -= 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [-1]),
                                          shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        return sin_op * (-1j)

    def _kron3(self, mat1, mat2, mat3):
        """
        Kronecker product of three matrices

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.kron(sparse.kron(mat1, mat2, format='csc'), mat3, format='csc')

    def _identity_phi(self):
        """
        Identity operator acting only on the :math:`\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_phi()
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

    def _identity_varphi(self):
        """
        Identity operator acting only on the :math:`\varphi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_varphi()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def total_identity(self):
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron3(self._identity_phi(), self._identity_theta(), self._identity_varphi())

    def hamiltonian(self):
        """
        Double Cooper pair tunneling qubit Hamiltonian

        Returns
        -------
        ndarray
        """
        phi_osc_mat = self._kron3(op.number_sparse(self.dim_phi(), self.phi_plasma()), self._identity_theta(),
                                  self._identity_varphi())
        theta_osc_mat = self._kron3(self._identity_phi(),
                                    op.number_sparse(self.dim_theta(), self.theta_plasma()),
                                    self._identity_varphi())

        n_varphi_ng_mat = self.n_varphi_operator() - self.total_identity() * self.Ng
        n_theta_mat = self._kron3(self._identity_phi(), self._n_theta_operator(), self._identity_varphi())
        cross_kinetic_mat = 2 * self._dis_ec() * (n_varphi_ng_mat - n_theta_mat) ** 2

        phi_flux_term = self._cos_phi_2_operator() * np.cos(self.flux * np.pi) - self._sin_phi_2_operator() * np.sin(
            self.flux * np.pi)
        junction_mat = -2 * self.EJ * self._kron3(phi_flux_term, self._identity_theta(),
                                                  self._cos_varphi_operator()) + 2 * self.EJ * self.total_identity()

        hamiltonian_mat = phi_osc_mat + theta_osc_mat + cross_kinetic_mat + junction_mat
        return hamiltonian_mat

    def potential(self, varphi, phi):
        """
        Double Cooper pair tunneling qubit potential evaluated at `phi, varphi`, with `theta=0`

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
            phi * 0.5 + np.pi * self.flux) + 2 * self.dJ * self.EJ * np.sin(phi * 0.5 + np.pi * self.flux) * np.sin(
            varphi)

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
        phi_grid = phi_grid or self._default_phi_grid
        varphi_grid = varphi_grid or self._default_varphi_grid

        x_vals = varphi_grid.make_linspace()
        y_vals = phi_grid.make_linspace()
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (4, 4)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, sigma=0.0, which='LM')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, sigma=0.0, which='LM')
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def _tensor_index(self, index_phi, index_theta, index_varphi):
        """
        Return the index of the coefficient of the wavefunction, corresponding to the indices of phi, theta, and varphi
        """
        return (index_phi * self.dim_theta() + index_theta) * self.dim_varphi() + index_varphi

    def _tensor_index_inv(self, index_evec):
        """
        Return the indices of phi, theta, and varphi corresponding to the index of the coefficient of the wavefunction
        """
        index_varphi = index_evec % self.dim_varphi()
        index_temp = index_evec // self.dim_varphi()
        index_theta = index_temp % self.dim_theta()
        index_phi = index_temp // self.dim_theta()
        return index_phi, index_theta, index_varphi

    def wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None, varphi_grid=None):
        """
        Return a 3D wave function in phi, theta, varphi basis

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
        theta_grid = theta_grid or self._default_theta_grid
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

    def plot_phi_varphi_wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None, mode='abs',
                                     zero_calibrate=True,
                                     **kwargs):
        """
        Plots 2D phase-basis wave function at theta = 0

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
        """
        instanton path phi(varphi), only works in the case of zero disorder
        """
        z = self.EL / self.EJ
        # TODO make sure the minus pi
        return 1.0 / (1.0 + z) * (
                2 * np.abs(varphi - 2 * np.pi * np.round(varphi / (2 * np.pi))) + z * 2 * np.pi * self.flux) - np.pi

    def plot_n_varphi_wavefunction(self, esys=None, mode='real', which=0, n_varphi_list=None, **kwargs):
        """
        Plots 1D charge-basis wave function for n_varphi at theta = 0, and projectes onto instanton path

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        n_varphi_list: ndarray, option
            used for setting a custom grid for varphi; if None use np.arange(-7, 8)
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        **kwargs:
            plot options

        Returns
        -------
        Figure, Axes
        """
        phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        phi_varphi_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        varphi_amplitudes = np.zeros(varphi_grid.pt_count, dtype=np.complex_)
        for n in range(varphi_grid.pt_count):
            phi_instanton = self.instanton_path(varphi_grid_list[n])
            phi_idx = (np.abs(phi_grid.make_linspace() - phi_instanton)).argmin()
            varphi_amplitudes[n] = phi_varphi_amplitudes[phi_idx, n]

        if n_varphi_list is None:
            n_varphi_list = np.arange(-7, 8)
        n_varphi_amplitudes = np.zeros(np.size(n_varphi_list), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            n_varphi_amplitudes[n] = 1 / (2 * np.pi) * np.sum(
                varphi_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list)) * d_varphi

        n_varphi_wavefunction = storage.WaveFunction(n_varphi_list, n_varphi_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_varphi_wavefunction.amplitudes = amplitude_modifier(n_varphi_wavefunction.amplitudes)
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}  # if any duplicates, later ones survive
        return plot.wavefunction1d_discrete(n_varphi_wavefunction, **kwargs)

    def plot_phi_n_varphi_wavefunction(self, esys=None, mode='real', which=0, zero_calibrate=True,
                                       **kwargs):
        """
        Plots 2D wave function for phi and n_varphi at theta = 0

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
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
        phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)
        n_varphi_list = np.arange(-7, 8)
        n_varphi_grid = discretization.Grid1d(-7, 7, 15)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        phi_varphi_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        phi_n_varphi_amplitudes = np.zeros((phi_grid.pt_count, n_varphi_list.size), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            phi_n_varphi_amplitudes[:, n] = 1 / (2 * np.pi) * np.sum(
                phi_varphi_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list), axis=1) * d_varphi

        grid2d = discretization.GridSpec(np.asarray([[n_varphi_grid.min_val, n_varphi_grid.max_val,
                                                      n_varphi_grid.pt_count],
                                                     [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                                     ]))

        phi_n_varphi_wavefunction = storage.WaveFunctionOnGrid(grid2d, phi_n_varphi_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        phi_n_varphi_wavefunction.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(phi_n_varphi_wavefunction.amplitudes))
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}  # if any duplicates, later ones survive
        return plot.wavefunction2d(phi_n_varphi_wavefunction, zero_calibrate=zero_calibrate, **kwargs)

    # TODO: test purpose only
    def ft_wavefunction(self, esys=None, which=0):
        phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 100)
        theta_grid = self._default_theta_grid
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 100)

        n_varphi_list = np.arange(-7, 8)
        n_varphi_grid = discretization.Grid1d(-7, 7, 15)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        phi_theta_varphi_amplitudes = spec_utils.standardize_phases(wavefunc.amplitudes)
        phi_theta_n_varphi_amplitudes = np.zeros((phi_grid.pt_count, theta_grid.pt_count, n_varphi_list.size),
                                                 dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            phi_theta_n_varphi_amplitudes[:, :, n] = 1 / (2 * np.pi) * np.sum(
                phi_theta_varphi_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list), axis=2) * d_varphi

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_list[0], n_phi_list[-1], n_phi_list.size)

        d_theta = theta_grid.make_linspace()[1] - theta_grid.make_linspace()[0]
        n_theta_list = np.sort(np.fft.fftfreq(theta_grid.pt_count, d_theta)) * 2 * np.pi
        n_theta_grid = discretization.Grid1d(n_theta_list[0], n_theta_list[-1], n_theta_list.size)

        n_phi_n_theta_n_varphi_amplitudes = np.zeros((n_phi_list.size, n_theta_list.size, n_varphi_list.size),
                                                     dtype=np.complex_)
        for n in range(n_varphi_list.size):
            n_phi_n_theta_n_varphi_amplitudes[:, :, n] = np.fft.ifft2(
                phi_theta_n_varphi_amplitudes[:, :, n]) * d_phi * phi_grid.pt_count * d_theta * theta_grid.pt_count
            n_phi_n_theta_n_varphi_amplitudes[:, :, n] = np.fft.fftshift(n_phi_n_theta_n_varphi_amplitudes[:, :, n])
        grid3d = discretization.GridSpec(np.asarray([
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count],
            [n_theta_grid.min_val, n_theta_grid.max_val, n_theta_grid.pt_count],
            [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count]]))
        n_phi_n_theta_n_varphi_wavefunction = storage.WaveFunctionOnGrid(grid3d, n_phi_n_theta_n_varphi_amplitudes)
        n_phi_n_theta_n_varphi_wavefunction.amplitudes = spec_utils.standardize_phases(
            n_phi_n_theta_n_varphi_wavefunction.amplitudes)
        return n_phi_n_theta_n_varphi_wavefunction

    def get_n_phi_n_varphi_wavefunction(self, esys=None, which=0):
        """
        Calculate 2D wave function for n_phi and n_varphi at theta = 0

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

        Returns
        -------
        WaveFunctionOnGrid object
        """
        phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 400)
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = discretization.Grid1d(0, 2 * np.pi, 500)

        n_varphi_list = np.arange(-7, 8)
        n_varphi_grid = discretization.Grid1d(-7, 7, 15)

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        varphi_grid_list = varphi_grid.make_linspace()
        phi_varphi_amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))
        phi_n_varphi_amplitudes = np.zeros((phi_grid.pt_count, n_varphi_list.size), dtype=np.complex_)
        d_varphi = varphi_grid_list[1] - varphi_grid_list[0]
        for n in range(n_varphi_list.size):
            phi_n_varphi_amplitudes[:, n] = 1 / (2 * np.pi) * np.sum(
                phi_varphi_amplitudes * np.exp(1j * n_varphi_list[n] * varphi_grid_list), axis=1) * d_varphi

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_list[0], n_phi_list[-1], n_phi_list.size)

        n_phi_n_varphi_amplitudes = np.zeros((n_phi_list.size, n_varphi_list.size), dtype=np.complex_)
        for n in range(n_varphi_list.size):
            n_phi_n_varphi_amplitudes[:, n] = np.fft.ifft(phi_n_varphi_amplitudes[:, n]) * d_phi * phi_grid.pt_count
            n_phi_n_varphi_amplitudes[:, n] = np.fft.fftshift(n_phi_n_varphi_amplitudes[:, n])
        grid2d = discretization.GridSpec(np.asarray([
            [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count],
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count]]))
        n_phi_n_varphi_wavefunction = storage.WaveFunctionOnGrid(grid2d, n_phi_n_varphi_amplitudes)
        n_phi_n_varphi_wavefunction.amplitudes = spec_utils.standardize_phases(n_phi_n_varphi_wavefunction.amplitudes)
        return n_phi_n_varphi_wavefunction

    def plot_n_phi_n_varphi_wavefunction(self, esys=None, mode='real', which=0, zero_calibrate=True, **kwargs):
        """
        Plots 2D wave function for n_phi and n_varphi at theta = 0

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
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
        n_phi_n_varphi_wavefunction = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=which)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_n_varphi_wavefunction.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(n_phi_n_varphi_wavefunction.amplitudes))
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}  # if any duplicates, later ones survive

        # plot at slice n_varphi = 0
        phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 400)
        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_list[0], n_phi_list[-1], n_phi_list.size)

        fig, axs = plt.subplots(figsize=(4, 2))
        axs.plot(n_phi_grid.make_linspace(), n_phi_n_varphi_wavefunction.amplitudes[:, 7], '-o')
        axs.set_xlabel(r'$N_\phi$')
        axs.set_ylabel(r'$|\psi|$')
        axs.set_xlim((-2, 2))
        axs.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        axs.set_xticklabels(['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])  # plot using zeropi notation

        # plot at slice n_phi = 0
        n_varphi_grid = discretization.Grid1d(-7, 7, 15)
        n_varphi_list = n_varphi_grid.make_linspace()
        n_varphi_wavefunction = storage.WaveFunction(n_varphi_list, n_phi_n_varphi_wavefunction.amplitudes[200, :])
        fig, axs = plot.wavefunction1d_discrete(n_varphi_wavefunction, figsize=(4, 2))
        axs.set_xlabel(r'$N_\theta$')
        axs.set_ylabel(r'$|\psi|$')
        axs.set_xlim((-4, 4))

        return plot.wavefunction2d(n_phi_n_varphi_wavefunction, zero_calibrate=zero_calibrate, **kwargs)

    # TODO: for test purpose
    def _n_phi_n_varphi_wavefunction_overlap(self, esys):
        """overlap of wavefunction in n_phi_n_varphi plane"""
        wfnc1 = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=0)
        wfnc2 = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=1)
        return np.abs(np.sum(wfnc1.amplitudes.conj() * wfnc2.amplitudes) / np.sum(
            wfnc1.amplitudes.conj() * wfnc1.amplitudes)) ** 2, np.abs(np.sum(
            wfnc1.amplitudes.conj() * wfnc2.amplitudes) / np.sum(
            wfnc2.amplitudes.conj() * wfnc2.amplitudes)) ** 2

    def _n_phi_wavefunction_overlap(self, esys):
        """overlap of wavefunction in n_phi_n_varphi plane"""
        wfnc1 = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=0)
        wfnc2 = self.get_n_phi_n_varphi_wavefunction(esys=esys, which=1)
        return np.abs(np.sum(wfnc1.amplitudes[:, 7].conj() * wfnc2.amplitudes[:, 7]) / np.sum(
            wfnc1.amplitudes[:, 7].conj() * wfnc1.amplitudes[:, 7])) ** 2, np.abs(np.sum(
            wfnc1.amplitudes[:, 7].conj() * wfnc2.amplitudes[:, 7]) / np.sum(
            wfnc2.amplitudes[:, 7].conj() * wfnc2.amplitudes[:, 7])) ** 2

    def disorder(self):
        """
        Return disorder Hamiltonian due to dL

        Returns
        -------
        ndarray
        """
        disorder_l = - self._dis_el() * self.dL * self._kron3(self._phi_operator(), self._theta_operator(),
                                                              self._identity_varphi())

        phi_flux_term = self._sin_phi_2_operator() * np.cos(self.flux * np.pi) + self._cos_phi_2_operator() * np.sin(
            self.flux * np.pi)
        disorder_j = 2 * self.EJ * self.dJ * self._kron3(phi_flux_term, self._identity_theta(),
                                                         self._sin_varphi_operator())

        n_varphi_ng_mat = self.n_varphi_operator() - self.total_identity() * self.Ng
        n_theta_mat = self._kron3(self._n_phi_operator(), self._n_theta_operator(), self._identity_varphi())
        disorder_c = - 8 * self._dis_ec() * self.dC * (n_varphi_ng_mat - n_theta_mat)

        return disorder_l + disorder_j + disorder_c

    def phi_1_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:'phi_1' operator in the LC harmonic oscillator basis, which is the phase across inductor 1
        """
        return self.theta_operator() - self.phi_operator() / 2.0

    def phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:'phi_2' operator in the LC harmonic oscillator basis, which is the phase across inductor 2
        """
        return - self.theta_operator() - self.phi_operator() / 2.0

    def q_ind(self, energy):
        """Frequency dependent quality factor of inductance"""
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt) * np.sinh(0.5 / 2.0 / self.kbt) / kn(0,
                                                                                          energy / 2.0 / self.kbt) / np.sinh(
            energy / 2.0 / self.kbt)

    # TODO: test purpose
    def t1_ind_loss(self, dl_list):
        """Return the 1/T1 due to inductive loss"""
        """
        calculate 1/T1 due to inductive loss

        Parameters
        ----------
        dL_list: ndarray
            list of inductive disorder

        Returns
        -------
        ndarray
        """
        eng_obj = self.get_spectrum_vs_paramvals('dL', dl_list, evals_count=2, subtract_ground=True)
        eng = eng_obj.energy_table[:, 1]

        matele_obj_1 = self.get_matelements_vs_paramvals('phi_1_operator', 'dL', dl_list, evals_count=2)
        matele_1 = matele_obj_1.matrixelem_table[:, 0, 1]
        matele_obj_2 = self.get_matelements_vs_paramvals('phi_2_operator', 'dL', dl_list, evals_count=2)
        matele_2 = matele_obj_2.matrixelem_table[:, 0, 1]

        s_ind_1 = self.EL * 2 / (1 - dl_list) / self.q_ind(eng) / np.tanh(eng / 2.0 / self.kbt)
        s_ind_2 = self.EL * 2 / (1 + dl_list) / self.q_ind(eng) / np.tanh(eng / 2.0 / self.kbt)

        t1_ind_1 = np.abs(matele_1) ** 2 * s_ind_1
        t1_ind_2 = np.abs(matele_2) ** 2 * s_ind_2

        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        axs[0, 0].plot(dl_list, s_ind_1)
        axs[0, 0].set_xlabel('dL')
        axs[0, 0].set_ylabel('$S_1(\omega)$')

        axs[0, 1].plot(dl_list, s_ind_2)
        axs[0, 1].set_xlabel('dL')
        axs[0, 1].set_ylabel('$S_2(\omega)$')

        axs[1, 0].plot(dl_list, np.abs(matele_1) ** 2)
        axs[1, 0].set_xlabel('dL')
        axs[1, 0].set_ylabel(r'$|\langle 0 | \phi_1 | 1 \rangle|^2 $')

        axs[1, 1].plot(dl_list, np.abs(matele_2) ** 2)
        axs[1, 1].set_xlabel('dL')
        axs[1, 1].set_ylabel(r'$|\langle 0 | \phi_2 | 1 \rangle|^2 $')

        axs[2, 0].plot(dl_list, eng)
        axs[2, 0].set_xlabel('dL')
        axs[2, 0].set_ylabel('$\Delta E$')

        axs[2, 1].plot(dl_list, 1 / (t1_ind_1 + t1_ind_2))
        axs[2, 1].plot(dl_list, 1 / t1_ind_1)
        axs[2, 1].plot(dl_list, 1 / t1_ind_2)
        axs[2, 1].set_xlabel('dL')
        axs[2, 1].set_ylabel('$T_1$')
        axs[2, 1].legend(['total', '$L_1$', '$L_2$'])
        axs[2, 1].set_yscale('log')
        axs[2, 1].set_ylim((1e3, 1e9))

        return 1 / (t1_ind_1 + t1_ind_2)

    def phi_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi^2' operator in total Hilbert space
        """
        return self.phi_operator() ** 2

    def n_phi_n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi^2' operator in total Hilbert space
        """
        return self.n_phi_operator() ** 2

    def n_theta_n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_theta^2' operator in total Hilbert space
        """
        return self.n_theta_operator() ** 2

    def n_varphi_n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_varphi^2' operator in total Hilbert space
        """
        return self._n_varphi_operator() ** 2

    def plot_matele_norm(self, operator, operator_operator, para_name, para_vals):
        """
        Plots matrix element involving the ground state, normalized by ground state <g|operator^2|g> as a function parameter

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix
        operator_operator: str
            name of class method in string form, returning operator matrix
        para_name: str
            name of parameter to be varied
        para_vals: ndarray
            parameter values to be plugged in

        Returns
        -------
        ndarray
        """
        matele = self.get_matelements_vs_paramvals(operator, para_name, para_vals, evals_count=5)
        matele_01 = matele.matrixelem_table[:, 0, 1]
        matele_02 = matele.matrixelem_table[:, 0, 2]
        matele_03 = matele.matrixelem_table[:, 0, 3]
        matele_04 = matele.matrixelem_table[:, 0, 4]

        norm = self.get_matelements_vs_paramvals(operator_operator, para_name, para_vals, evals_count=2)
        norm_val = np.real(norm.matrixelem_table[:, 0, 0])

        norm_matele_01 = np.abs(matele_01) ** 2 / norm_val
        norm_matele_02 = np.abs(matele_02) ** 2 / norm_val
        norm_matele_03 = np.abs(matele_03) ** 2 / norm_val
        norm_matele_04 = np.abs(matele_04) ** 2 / norm_val

        plt.figure(figsize=(4, 4))
        plt.plot(para_vals, norm_matele_01)
        plt.plot(para_vals, norm_matele_02)
        plt.plot(para_vals, norm_matele_03)
        plt.plot(para_vals, norm_matele_04)

        plt.xlabel(para_name)
        plt.ylabel('matrix element')
        plt.yscale('log')
        plt.ylim((1e-8, 1))
        plt.xlim((0.4, 0.6))
        plt.legend(['0-1', '0-2', '0-3', '0-4'])

        return norm_matele_01, norm_matele_02, norm_matele_03, norm_matele_04

    def N_1_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:'N_1' operator in the LC harmonic oscillator basis, which is the charge on junction 1
        """
        return self.n_phi_operator() + 0.5 * (self.n_varphi_operator() - self.n_theta_operator())

    def N_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:'N_2' operator in the LC harmonic oscillator basis, which is the charge on junction 2
        """
        return self.n_phi_operator() - 0.5 * (self.n_varphi_operator() - self.n_theta_operator())

    def q_cap(self, energy):
        """Frequency dependent quality factor of capacitance"""
        q_cap_0 = 1 * 1e6
        return q_cap_0 * (2 * np.pi * 6 / energy) ** 0.7

    # TODO: test purpose
    def t1_cap_loss(self, dl_list):
        """Return the 1/T1 due to capcitive loss"""
        """
        calculate 1/T1 due to inductive loss

        Parameters
        ----------
        dL_list: ndarray
            list of inductive disorder

        Returns
        -------
        ndarray
        """
        eng_obj = self.get_spectrum_vs_paramvals('dL', dl_list, evals_count=2, subtract_ground=True)
        eng = eng_obj.energy_table[:, 1]

        matele_obj_1 = self.get_matelements_vs_paramvals('N_1_operator', 'dL', dl_list, evals_count=2)
        matele_1 = matele_obj_1.matrixelem_table[:, 0, 1]
        matele_obj_2 = self.get_matelements_vs_paramvals('N_2_operator', 'dL', dl_list, evals_count=2)
        matele_2 = matele_obj_2.matrixelem_table[:, 0, 1]

        s_vv_1 = 2 * 8 * self.EC / (1 - self.dC) / self.q_cap(eng) / np.tanh(eng / 2.0 / self.kbt)
        s_vv_2 = 2 * 8 * self.EC / (1 + self.dC) / self.q_cap(eng) / np.tanh(eng / 2.0 / self.kbt)

        t1_cap_1 = np.abs(matele_1) ** 2 * s_vv_1
        t1_cap_2 = np.abs(matele_2) ** 2 * s_vv_2

        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        axs[0, 0].plot(dl_list, s_vv_1)
        axs[0, 0].set_xlabel('dL')
        axs[0, 0].set_ylabel('$S_1(\omega)$')

        axs[0, 1].plot(dl_list, s_vv_2)
        axs[0, 1].set_xlabel('dL')
        axs[0, 1].set_ylabel('$S_2(\omega)$')

        axs[1, 0].plot(dl_list, np.abs(matele_1) ** 2)
        axs[1, 0].set_xlabel('dL')
        axs[1, 0].set_ylabel(r'$|\langle 0 | \phi_1 | 1 \rangle|^2 $')

        axs[1, 1].plot(dl_list, np.abs(matele_2) ** 2)
        axs[1, 1].set_xlabel('dL')
        axs[1, 1].set_ylabel(r'$|\langle 0 | \phi_2 | 1 \rangle|^2 $')

        axs[2, 0].plot(dl_list, eng)
        axs[2, 0].set_xlabel('dL')
        axs[2, 0].set_ylabel('$\Delta E$')

        axs[2, 1].plot(dl_list, 1 / (t1_cap_1 + t1_cap_2))
        axs[2, 1].plot(dl_list, 1 / t1_cap_1)
        axs[2, 1].plot(dl_list, 1 / t1_cap_2)
        axs[2, 1].set_xlabel('dL')
        axs[2, 1].set_ylabel('$T_1$')
        axs[2, 1].legend(['total', '$J_1$', '$J_2$'])
        axs[2, 1].set_yscale('log')
        axs[2, 1].set_ylim((1e3, 1e12))

        return 1 / (t1_cap_1 + t1_cap_2)

    def get_t1_capacitive_loss(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele_1 = self.get_matelements_vs_paramvals('N_1_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        matele_2 = self.get_matelements_vs_paramvals('N_2_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        s_vv_1 = 16 * self.EC / (1 - self.dC) / self.q_cap(energy) / np.tanh(energy / 2.0 / self.kbt)
        s_vv_2 = 16 * self.EC / (1 + self.dC) / self.q_cap(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_cap_1 = np.abs(matele_1) ** 2 * s_vv_1
        gamma1_cap_2 = np.abs(matele_2) ** 2 * s_vv_2
        return 1 / (gamma1_cap_1 + gamma1_cap_2) * 1e-6, np.abs(matele_1) ** 2, np.abs(matele_2) ** 2, s_vv_1, s_vv_2

    def get_t1_inductive_loss(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele_1 = self.get_matelements_vs_paramvals('phi_1_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        matele_2 = self.get_matelements_vs_paramvals('phi_2_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        s_ii_1 = 2 * self.EL / (1 - self.dL) / self.q_ind(energy) / np.tanh(energy / 2.0 / self.kbt)
        s_ii_2 = 2 * self.EL / (1 + self.dL) / self.q_ind(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_ind_1 = np.abs(matele_1) ** 2 * s_ii_1
        gamma1_ind_2 = np.abs(matele_2) ** 2 * s_ii_2
        return 1 / (gamma1_ind_1 + gamma1_ind_2) * 1e-6, np.abs(matele_1) ** 2, np.abs(matele_2) ** 2, s_ii_1, s_ii_2

    def get_t2_charge_noise(self, para_name, para_vals):
        original_ng = self.Ng
        self.Ng = 0.0
        energy_ng_0 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                     subtract_ground=True).energy_table[:, 1]
        self.Ng = 0.5
        energy_ng_5 = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                     subtract_ground=True).energy_table[:, 1]
        self.Ng = original_ng
        epsilon = np.abs(energy_ng_5 - energy_ng_0)
        return 1.49734 / epsilon * 1e-6  # unit in ms

    def get_t2_flux_noise(self, para_name, para_vals):
        orginal_flux = self.flux
        delta = 1e-7
        pts = 11
        flux_list = np.linspace(0.5 - delta, 0.5 + delta, pts)
        energy = np.zeros((pts, para_vals.size))
        for i in range(pts):
            self.flux = flux_list[i]
            energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                          subtract_ground=True).energy_table[:, 1]
        second_derivative = np.gradient(np.gradient(energy, flux_list, axis=0), flux_list, axis=0)[
                            int(np.round(pts / 2)), :]
        self.flux = orginal_flux
        return np.abs(1 / (9e-12 * second_derivative) * 1e-6)  # unit in ms

    def get_t2_current_noise(self, para_name, para_vals):
        orginal_ej = self.EJ
        delta = 1e-7
        pts = 11
        ej_list = np.linspace(orginal_ej - delta, orginal_ej + delta, pts)
        energy = np.zeros((pts, para_vals.size))
        for i in range(pts):
            self.EJ = ej_list[i]
            energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
                                                          subtract_ground=True).energy_table[:, 1]
        first_derivative = np.gradient(energy, ej_list, axis=0)[int(np.round(pts / 2)), :]
        self.EJ = orginal_ej
        return np.abs(1 / (5e-7 * orginal_ej * first_derivative) * 1e-6)  # unit in ms

    def noise_analysis(self, para_name, para_vals):
        t2_charge = self.get_t2_charge_noise(para_name, para_vals)
        t2_flux = self.get_t2_flux_noise(para_name, para_vals)
        t2_current = self.get_t2_current_noise(para_name, para_vals)
        t1_cap = self.get_t1_capacitive_loss(para_name, para_vals)
        t1_ind = self.get_t1_inductive_loss(para_name, para_vals)

        plt.figure(figsize=(4, 4))
        plt.plot(para_vals, t2_charge, '--')
        plt.plot(para_vals, t2_current, '--')
        plt.plot(para_vals, t2_flux, '--')
        plt.plot(para_vals, t1_cap[0])
        plt.plot(para_vals, t1_ind[0])
        plt.legend(['T2_charge', 'T2_current', 'T2_flux', 'T1_cap', 'T1_ind'])
        plt.xlabel(para_name)
        plt.ylabel('T1, T2 (ms)')
        plt.yscale('log')

    def print_noise(self):
        t2_charge = self.get_t2_charge_noise('dC', np.array([0]))
        t2_current = self.get_t2_current_noise('dC', np.array([0]))
        t2_flux = self.get_t2_flux_noise('dC', np.array([0]))
        t1_cap = self.get_t1_capacitive_loss('dC', np.array([0]))
        t1_ind = self.get_t1_inductive_loss('dC', np.array([0]))
        return print(' T2_charge=', t2_charge, '\n T2_current=', t2_current, '\n T2_flux=', t2_flux, '\n T1_cap=', t1_cap[0], '\n T1_ind=', t1_ind[0])
