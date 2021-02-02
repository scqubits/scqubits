# cosine_two_phi_qubit.py
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
from typing import Any, Dict, List, Optional, Tuple, Union

from scqubits.core.noise import NoisySystem
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scqubits.core.storage import WaveFunctionOnGrid


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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scqubits.utils.spectrum_utils import matrix_element

# -Cosine two phi qubit noise class

class NoisyCosineTwoPhiQubit(NoisySystem):
    pass

# -Cosine two phi qubit ----------------------------------------------------------------------------------
class CosineTwoPhiQubit(base.QubitBaseClass, serializers.Serializable, NoisyCosineTwoPhiQubit):
    r"""Cosine Two Phi Qubit

    | [1] Smith et al., NPJ Quantum Inf. 6, 8 (2020) http://www.nature.com/articles/s41534-019-0231-2

    .. math::

        H = 4E_\text{C}[2n_\phi^2+\frac{1}{2}(n_\varphi-N_\text{g}-n_\theta)^2+xn_\theta^2]
                           +E_\text{L}(\frac{1}{4}\phi^2+\theta^2)
                           -2E_\text{J}\cos(\varphi)\cos(\frac{\phi}{2}+\frac{\varphi_\text{ext}}{2})

    The Hamiltonian is formed with harmonic basis for :math:`\phi,\theta` variables and charge basis for :math:`\varphi`
    variable.

    Parameters
    ----------
    EJ: float
        Josephson energy of the two junctions
    EC: float
        charging energy of the two junctions
    EL: float
        inductive energy of the two inductors
    x: float
        ratio of the junction capacitance to the shunt capacitance x = C_J / C_shunt
    dC: float
        disorder in charging energy, i.e., `EC / (1 \pm dC)`
    dL: float
        disorder in inductive energy, i.e., `EL / (1 \pm dL)`
    dJ: float
        disorder in junction energy, i.e., `EJ * (1 \pm dJ)`
    flux: float
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    Ng: float
        offset charge
    N0: int
        number of charge states, `-N0 <= n_\varphi <= N0`
    q0: int
        number of harmonic oscillator basis for `\theta` variable
    p0: int
        number of harmonic oscillator basis for `\phi` variable
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

    def __init__(self, EJ: float, EC: float, EL: float, x: float, dL: float, dC: float, dJ: float, flux: float,
                 Ng: float, N0: int, q0: int, p0: int) -> None:
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
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._default_phi_grid = discretization.Grid1d(-10 * np.pi, 10 * np.pi, 400)
        self._default_theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_varphi_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/cosine_two_phi_qubit.png')

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            'EJ': 15.0,
            'EC': 2.0,
            'EL': 1.0,
            'x': 0.02,
            'dC': 0.0,
            'dL': 0.3,
            'dJ': 0.0,
            'flux': 0.5,
            'Ng': 0.0,
            'N0': 7,
            'q0': 30,
            'p0': 7
        }

    @classmethod
    def create(cls) -> 'CosineTwoPhiQubit':
        init_params = cls.default_params()
        cosinetwophiqubit = cls(**init_params)
        cosinetwophiqubit.widget()
        return cosinetwophiqubit

    def supported_noise_channels(self) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            'tphi_1_over_f_cc',
            'tphi_1_over_f_flux',
            'tphi_1_over_f_charge',
            't1_capacitive',
            't1_inductive'
        ]

    def dim_phi(self) -> int:
        """
        Returns
        -------
        int
            Hilbert space dimension of `phi` degree of freedom"""
        return self.p0

    def dim_theta(self) -> int:
        """
        Returns
        -------
        int
            Hilbert space dimension of `theta` degree of freedom"""
        return self.q0

    def dim_varphi(self) -> int:
        """
        Returns
        -------
        int
            Hilbert space dimension of `varphi` degree of freedom"""
        return 2 * self.N0 + 1

    def hilbertdim(self) -> int:
        """
        Returns
        -------
        int
            total Hilbert space dimension"""
        return self.dim_phi() * self.dim_theta() * self.dim_varphi()

    def _dis_el(self) -> float:
        """
        Returns
        -------
        float
            inductive energy renormalized by with disorder"""
        return self.EL / (1 - self.dL ** 2)

    def _dis_ec(self) -> float:
        """
        Returns
        -------
        float
            capacitance energy renormalized by with disorder"""
        return self.EC / (1 - self.dC ** 2)

    def phi_osc(self) -> float:
        """
        Returns
        -------
        float
            oscillator strength of `phi` degree of freedom"""
        return (32 * self._dis_ec() / self._dis_el()) ** 0.25

    def theta_osc(self) -> float:
        """
        Returns
        -------
        float
            oscillator strength of `theta` degree of freedom"""
        return (4 * self._dis_ec() * self.x / self._dis_el()) ** 0.25

    def phi_plasma(self) -> float:
        """
        Returns
        -------
        float
            plasma oscillation frequency of `phi` degree of freedom"""
        return math.sqrt(8.0 * self._dis_el() * self._dis_ec())

    def theta_plasma(self) -> float:
        """
        Returns
        -------
        float
            plasma oscillation frequency of `theta` degree of freedom"""
        return math.sqrt(16.0 * self.x * self._dis_el() * self._dis_ec())

    def _phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `phi` operator in the harmonic oscillator basis"""
        dimension = self.dim_phi()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.phi_osc() / math.sqrt(2)

    def phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `phi` operator in total Hilbert space"""
        return self._kron3(self._phi_operator(), self._identity_theta(), self._identity_varphi())

    def _n_phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `n_\phi` operator in the harmonic oscillator basis"""
        dimension = self.dim_phi()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.phi_osc() * math.sqrt(2))

    def n_phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `n_phi` operator in total Hilbert space"""
        return self._kron3(self._n_phi_operator(), self._identity_theta(), self._identity_varphi())

    def _theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `theta` operator in the harmonic oscillator basis"""
        dimension = self.dim_theta()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.theta_osc() / math.sqrt(2)

    def theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `theta` operator in total Hilbert space"""
        return self._kron3(self._identity_phi(), self._theta_operator(), self._identity_varphi())

    def _n_theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `n_\theta` operator in the harmonic oscillator basis"""
        dimension = self.dim_theta()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.theta_osc() * math.sqrt(2))

    def n_theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `n_theta` operator in total Hilbert space"""
        return self._kron3(self._identity_phi(), self._n_theta_operator(), self._identity_varphi())

    def _exp_i_phi_2_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `e^{i*phi/2}` operator in the  harmonic oscillator basis"""
        exponent = 1j * self._phi_operator() * 0.5
        return expm(exponent)

    def _cos_phi_2_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `cos phi/2` operator in the harmonic oscillator basis"""
        cos_phi_op = 0.5 * self._exp_i_phi_2_operator()
        cos_phi_op += cos_phi_op.conj().T
        return cos_phi_op

    def _sin_phi_2_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `sin phi/2` operator in the LC harmonic oscillator basis"""
        sin_phi_op = -1j * 0.5 * self._exp_i_phi_2_operator()
        sin_phi_op += sin_phi_op.conj().T
        return sin_phi_op

    def _exp_i_phi_4_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `e^{i*phi/4}` operator in the harmonic oscillator basis"""
        exponent = 1j * self._phi_operator() * 0.25
        return expm(exponent)

    def _cos_phi_4_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `cos phi/4` operator in the harmonic oscillator basis"""
        cos_phi_op = 0.5 * self._exp_i_phi_4_operator()
        cos_phi_op += cos_phi_op.conj().T
        return cos_phi_op

    def _sin_phi_4_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `sin phi/2` operator in the harmonic oscillator basis"""
        sin_phi_op = -1j * 0.5 * self._exp_i_phi_4_operator()
        sin_phi_op += sin_phi_op.conj().T
        return sin_phi_op

    def _exp_i_theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
             `e^{i theta}` operator in the harmonic oscillator basis"""
        exponent = 1j * self._theta_operator()
        return expm(exponent)

    def _cos_theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `cos theta` operator in the harmonic oscillator basis"""
        cos_phi_op = 0.5 * self._exp_i_theta_operator()
        cos_phi_op += cos_phi_op.conj().T
        return cos_phi_op

    def _sin_theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `sin theta` operator in the harmonic oscillator basis"""
        sin_phi_op = -1j * 0.5 * self._exp_i_theta_operator()
        sin_phi_op += sin_phi_op.conj().T
        return sin_phi_op

    def _n_varphi_operator(self) -> dia_matrix:
        """
        Returns
        -------
        ndarray
            `n_varphi` operator in the charge basis"""
        diag_elements = np.arange(-self.N0, self.N0 + 1)
        return sparse.dia_matrix((diag_elements, [0]),
                                 shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def n_varphi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            `n_varphi` in the total Hilbert space"""
        return self._kron3(self._identity_phi(), self._identity_theta(), self._n_varphi_operator())

    def _cos_varphi_operator(self) -> dia_matrix:
        """
        Returns
        -------
        ndarray
            `cos varphi` in the charge basis"""
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [1]),
                                         shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [-1]),
                                          shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        return cos_op

    def _sin_varphi_operator(self) -> dia_matrix:
        """
        Returns
        -------
        ndarray
            `sin varphi` in the charge basis"""
        sin_op = 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [1]),
                                         shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        sin_op -= 0.5 * sparse.dia_matrix((np.ones(self.dim_varphi()), [-1]),
                                          shape=(self.dim_varphi(), self.dim_varphi())).tocsc()
        return sin_op * (-1j)

    def _kron3(self, mat1, mat2, mat3) -> csc_matrix:
        """
        Kronecker product of three matrices

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.kron(sparse.kron(mat1, mat2, format='csc'), mat3, format='csc')

    def _identity_phi(self) -> csc_matrix:
        """
        Identity operator acting only on the :math:`\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_phi()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def _identity_theta(self) -> csc_matrix:
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_theta()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def _identity_varphi(self) -> csc_matrix:
        """
        Identity operator acting only on the :math:`\varphi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        dimension = self.dim_varphi()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def total_identity(self) -> csc_matrix:
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron3(self._identity_phi(), self._identity_theta(), self._identity_varphi())

    def hamiltonian(self) -> csc_matrix:
        """
        Cosine two phi qubit Hamiltonian

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

    def potential(self, varphi, phi) -> float:
        """
        potential evaluated at `phi, varphi`, with `theta=0`

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

    def plot_potential(self, phi_grid=None, varphi_grid=None, contour_vals=None, **kwargs) -> Tuple[Figure, Axes]:
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

    def _evals_calc(self, evals_count) -> ndarray:
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, sigma=0.0, which='LM')
        return np.sort(evals)

    def _esys_calc(self, evals_count) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, sigma=0.0, which='LM')
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def _tensor_index(self, index_phi, index_theta, index_varphi) -> int:
        """
        Return the index of the coefficient of the wavefunction, corresponding to the indices of phi, theta, and varphi
        """
        return (index_phi * self.dim_theta() + index_theta) * self.dim_varphi() + index_varphi

    def _tensor_index_inv(self, index_evec) -> Tuple[int, int, int]:
        """
        Return the indices of phi, theta, and varphi corresponding to the index of the coefficient of the wavefunction
        """
        index_varphi = index_evec % self.dim_varphi()
        index_temp = index_evec // self.dim_varphi()
        index_theta = index_temp % self.dim_theta()
        index_phi = index_temp // self.dim_theta()
        return index_phi, index_theta, index_varphi

    def wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None, varphi_grid=None) -> WaveFunctionOnGrid:
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

    def plot_wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None, mode='abs',
                          zero_calibrate=True,
                          **kwargs) -> Tuple[Figure, Axes]:
        """
        Plots 2D wave function in `varphi` and `phi` basis, at `theta` = 0

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
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, xlabel=r'$\varphi$', ylabel=r'$\phi$', **kwargs)

    def disorder(self) -> csc_matrix:
        """
        Return disordered part of Hamiltonian

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

    def phi_1_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            operator represents phase across inductor 1
        """
        return self.theta_operator() - self.phi_operator() / 2.0

    def phi_2_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            operator represents phase across inductor 2
        """
        return - self.theta_operator() - self.phi_operator() / 2.0

    def q_ind(self, energy) -> float:
        """Frequency dependent quality factor of inductance"""
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt) * np.sinh(0.5 / 2.0 / self.kbt) / kn(0,
                                                                                          energy / 2.0 / self.kbt) / np.sinh(
            energy / 2.0 / self.kbt)

    def N_1_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            operator represents charge on junction 1
        """
        return self.n_phi_operator() + 0.5 * (self.n_varphi_operator() - self.n_theta_operator())

    def N_2_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            operator represents charge on junction 2
        """
        return self.n_phi_operator() - 0.5 * (self.n_varphi_operator() - self.n_theta_operator())

    def q_cap(self, energy) -> float:
        """Frequency dependent quality factor of capacitance"""
        # Devoret paper
        q_cap_0 = 1 * 1e6
        return q_cap_0 * (6 / energy) ** 0.7

        # Schuster paper
        # return 1 / (8e-6)

        # Vlad paper
        # q_cap_0 = 1/ (3 * 1e-6 )
        # return q_cap_0 * (6 / energy) ** 0.15

    def get_t1_capacitive_loss(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele_1 = self.get_matelements_vs_paramvals('N_1_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        matele_2 = self.get_matelements_vs_paramvals('N_2_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        s_vv_1 = 2 * np.pi * 16 * self.EC / (1 - self.dC) / self.q_cap(energy) / np.tanh(energy / 2.0 / self.kbt)
        s_vv_2 = 2 * np.pi * 16 * self.EC / (1 + self.dC) / self.q_cap(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_cap_1 = np.abs(matele_1) ** 2 * s_vv_1
        gamma1_cap_2 = np.abs(matele_2) ** 2 * s_vv_2
        return 1 / (gamma1_cap_1 + gamma1_cap_2) * 1e-6

    def get_t1_purcell(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele = self.get_matelements_vs_paramvals('n_theta_operator', para_name, para_vals,
                                                   evals_count=2).matrixelem_table[:, 0, 1]
        # note here only matters the shunt capacitance, so EC*x
        s_vv = 2 * np.pi * 16 * self.EC * self.x / self.q_cap(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_purcell = np.abs(matele) ** 2 * s_vv
        return 1 / gamma1_purcell * 1e-6

    def get_t1_inductive_loss(self, para_name, para_vals):
        energy = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2, subtract_ground=True).energy_table[
                 :, 1]
        matele_1 = self.get_matelements_vs_paramvals('phi_1_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        matele_2 = self.get_matelements_vs_paramvals('phi_2_operator', para_name, para_vals,
                                                     evals_count=2).matrixelem_table[:, 0, 1]
        s_ii_1 = 2 * np.pi * 2 * self.EL / (1 - self.dL) / self.q_ind(energy) / np.tanh(energy / 2.0 / self.kbt)
        s_ii_2 = 2 * np.pi * 2 * self.EL / (1 + self.dL) / self.q_ind(energy) / np.tanh(energy / 2.0 / self.kbt)
        gamma1_ind_1 = np.abs(matele_1) ** 2 * s_ii_1
        gamma1_ind_2 = np.abs(matele_2) ** 2 * s_ii_2
        return 1 / (gamma1_ind_1 + gamma1_ind_2) * 1e-6

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

    # TODO: noise calculated by derivative of energy
    def get_t2_flux_noise(self, para_name, para_vals):
        # orginal_flux = getattr(self, 'flux')
        # delta = 1e-7
        # pts = 11
        # flux_list = np.linspace(0.5 - delta, 0.5 + delta, pts)
        # energy = np.zeros((pts, para_vals.size))
        # for i in range(pts):
        #     setattr(self, 'flux', flux_list[i])
        #     energy[i, :] = self.get_spectrum_vs_paramvals(para_name, para_vals, evals_count=2,
        #                                                   subtract_ground=True).energy_table[:, 1]
        # second_derivative = np.gradient(np.gradient(energy, flux_list, axis=0), flux_list, axis=0)[
        #                     int(np.round(pts / 2)), :]
        # setattr(self, 'flux', orginal_flux)
        # return np.abs(1 / (9e-12 * second_derivative) * 1e-6) / (2 * np.pi)  # unit in ms

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
        second_derivative = np.gradient(np.gradient(energy, flux_list, axis=0), flux_list, axis=0)[
                            int(np.round(pts / 2)), :]
        setattr(self, 'flux', orginal_flux)

        first_order = 3e-6 * first_derivative
        second_order = 9e-12 * second_derivative
        # print(first_order)
        # print(second_order)
        # print(first_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

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
        return np.abs(1 / (5e-7 * orginal_ej * first_derivative) * 1e-6) / (2 * np.pi)  # unit in ms

    # #TODO: noise calculated by derivative of Hamiltonian
    # def _flux_noise_opertor(self):
    #     phi_flux_term = self._cos_phi_2_operator() * np.cos(self.flux * np.pi) - self._sin_phi_2_operator() * np.sin(
    #         self.flux * np.pi)
    #     junction_mat = 1 / 2 * self.EJ * self._kron3(phi_flux_term, self._identity_theta(),
    #                                               self._cos_varphi_operator()) + 2 * self.EJ * self.total_identity()
    #     return junction_mat
    #
    # def _current_noise_opertor(self):
    #     phi_flux_term = self._cos_phi_2_operator() * np.cos(self.flux * np.pi) - self._sin_phi_2_operator() * np.sin(
    #         self.flux * np.pi)
    #     junction_mat = - 2 * self._kron3(phi_flux_term, self._identity_theta(),
    #                                               self._cos_varphi_operator()) + 2 * self.EJ * self.total_identity()
    #     return junction_mat
    #
    # def get_t2_flux_noise(self, para_name, para_vals):
    #     matele_0 = self.get_matelements_vs_paramvals('_flux_noise_opertor', para_name, para_vals,
    #                                                  evals_count=2).matrixelem_table[:, 0, 0]
    #     matele_1 = self.get_matelements_vs_paramvals('_flux_noise_opertor', para_name, para_vals,
    #                                                  evals_count=2).matrixelem_table[:, 1, 1]
    #     matele = np.abs(matele_0 - matele_1)
    #     return np.abs(1 / (9e-12 * matele) * 1e-6) / (2 * np.pi)  # unit in ms
    #
    # def get_t2_current_noise(self, para_name, para_vals):
    #     matele_0 = self.get_matelements_vs_paramvals('_current_noise_opertor', para_name, para_vals,
    #                                                  evals_count=2).matrixelem_table[:, 0, 0]
    #     matele_1 = self.get_matelements_vs_paramvals('_current_noise_opertor', para_name, para_vals,
    #                                                  evals_count=2).matrixelem_table[:, 1, 1]
    #     matele = np.abs(matele_0 - matele_1)
    #     return np.abs(1 / (5e-7 * self.EJ * matele) * 1e-6) / (2 * np.pi)  # unit in ms

    def print_noise(self):
        t2_charge = self.get_t2_charge_noise('dC', np.array([0]))
        t2_current = self.get_t2_current_noise('dC', np.array([0]))
        t2_flux = self.get_t2_flux_noise('dC', np.array([0]))
        t1_cap = self.get_t1_capacitive_loss('dC', np.array([0]))
        t1_purcell = self.get_t1_purcell('dC', np.array([0]))
        t1_ind = self.get_t1_inductive_loss('dC', np.array([0]))
        t1_tot = 1 / (1 / t1_cap + 1 / t1_ind + 1 / t1_purcell)
        t2_tot = 1 / (1 / t2_current + 1 / t2_charge + 1 / t2_flux + 1 / t1_tot / 2)

        return print(' T2_charge =', t2_charge, ' ms', '\n T2_current =', t2_current, ' ms', '\n T2_flux =', t2_flux,
                     ' ms', '\n T1_cap =',
                     t1_cap, ' ms', '\n T1_Purcell =',
                     t1_purcell, ' ms', '\n T1_ind =', t1_ind, ' ms', '\n T1 =', t1_tot, ' ms', '\n T2 =', t2_tot,
                     ' ms')
