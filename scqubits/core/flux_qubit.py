# flux_qubit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import os

import numpy as np
import scipy as sp
import scipy.constants as const

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils
from scqubits.core.noise import NoisySystem, NOISE_PARAMS


# -Flux qubit, both degrees of freedom in charge basis---------------------------------------------------------


class FluxQubitFunctions:
    EJ1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJ2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJ3 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ3 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux):
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ3 = EJ3
        self.ECJ1 = ECJ1
        self.ECJ2 = ECJ2
        self.ECJ3 = ECJ3
        self.ECg1 = ECg1
        self.ECg2 = ECg2
        self.ng1 = ng1
        self.ng2 = ng2
        self.flux = flux
        self.number_degrees_freedom = 2

    def build_capacitance_matrix(self):
        C_matrix = np.zeros((2, 2))
        CJ1 = self.e**2 / (2 * self.ECJ1)
        CJ2 = self.e**2 / (2 * self.ECJ2)
        CJ3 = self.e**2 / (2 * self.ECJ3)
        Cg1 = self.e**2 / (2 * self.ECg1)
        Cg2 = self.e**2 / (2 * self.ECg2)

        C_matrix[0, 0] = CJ1 + CJ3 + Cg1
        C_matrix[1, 1] = CJ2 + CJ3 + Cg2
        C_matrix[0, 1] = -CJ3
        C_matrix[1, 0] = -CJ3

        return C_matrix

    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        C_matrix = self.build_capacitance_matrix()
        return 0.5 * self.e ** 2 * sp.linalg.inv(C_matrix)

    def potential(self, phi_array):
        """Return value of the potential energy at phi1 and phi2."""
        phi1 = phi_array[0]
        phi2 = phi_array[1]
        return (-self.EJ1 * np.cos(phi1) - self.EJ2 * np.cos(phi2)
                - self.EJ3 * np.cos(2.0 * np.pi * self.flux + phi1 - phi2)
                + self.EJ1 + self.EJ2 + self.EJ3)


class FluxQubit(FluxQubitFunctions, base.QubitBaseClass, serializers.Serializable):
    r"""Flux Qubit

    | [1] Orlando et al., Physical Review B, 60, 15398 (1999). https://link.aps.org/doi/10.1103/PhysRevB.60.15398

    The original flux qubit as defined in [1], where the junctions are allowed to have varying junction
    energies and capacitances to allow for junction asymmetry. Typically, one takes :math:`E_{J1}=E_{J2}=E_J`, and
    :math:`E_{J3}=\alpha E_J` where :math:`0\le \alpha \le 1`. The same relations typically hold
    for the junction capacitances. The Hamiltonian is given by

    .. math::

       H_\text{flux}=&(n_{i}-n_{gi})4(E_\text{C})_{ij}(n_{j}-n_{gj}) \\
                    -&E_{J}\cos\phi_{1}-E_{J}\cos\phi_{2}-\alpha E_{J}\cos(2\pi f + \phi_{1} - \phi_{2}),

    where :math:`i,j\in\{1,2\}` is represented in the charge basis for both degrees of freedom.
    Initialize with, for example::

        EJ = 35.0
        alpha = 0.6
        flux_qubit = qubit.FluxQubit(EJ1 = EJ, EJ2 = EJ, EJ3 = alpha*EJ,
                                     ECJ1 = 1.0, ECJ2 = 1.0, ECJ3 = 1.0/alpha,
                                     ECg1 = 50.0, ECg2 = 50.0, ng1 = 0.0, ng2 = 0.0,
                                     flux = 0.5, ncut = 10)

    Parameters
    ----------
    EJ1, EJ2, EJ3: float
        Josephson energy of the ith junction
        `EJ1 = EJ2`, with `EJ3 = alpha * EJ1` and `alpha <= 1`
    ECJ1, ECJ2, ECJ3: float
        charging energy associated with the ith junction
    ECg1, ECg2: float
        charging energy associated with the capacitive coupling to ground for the two islands
    ng1, ng2: float
        offset charge associated with island i
    flux: float
        magnetic flux through the circuit loop, measured in units of the flux quantum
    ncut: int
        charge number cutoff for the charge on both islands `n`,  `n = -ncut, ..., ncut`
    truncated_dim: int, optional
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    """
    ncut = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, ncut, truncated_dim=None):
        FluxQubitFunctions.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux)
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._default_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 100)    # for plotting in phi_j basis
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/fluxqubit.png')

    @staticmethod
    def default_params():
        return {
            'EJ1': 1.0,
            'EJ2': 1.0,
            'EJ3': 0.8,
            'ECJ1': 0.016,
            'ECJ2': 0.016,
            'ECJ3': 0.021,
            'ECg1': 0.83,
            'ECg2': 0.83,
            'ng1': 0.0,
            'ng2': 0.0,
            'flux': 0.4,
            'ncut': 10,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['ng1', 'ng2', 'flux', 'ncut', 'truncated_dim']

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(hamiltonian_mat, eigvals=(0, evals_count - 1), eigvals_only=True)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, eigvals=(0, evals_count - 1), eigvals_only=False)
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return (2 * self.ncut + 1) ** 2

    def kineticmat(self):
        """Return the kinetic energy matrix."""
        ECmat = self.build_EC_matrix()

        kinetic_mat = 4.0 * ECmat[0, 0] * np.kron(np.matmul(self._n_operator() - self.ng1 * self._identity(),
                                                            self._n_operator() - self.ng1 * self._identity()),
                                                  self._identity())
        kinetic_mat += 4.0 * ECmat[1, 1] * np.kron(self._identity(),
                                                   np.matmul(self._n_operator() - self.ng2 * self._identity(),
                                                             self._n_operator() - self.ng2 * self._identity()))
        kinetic_mat += 4.0 * (ECmat[0, 1] + ECmat[1, 0]) * np.kron(self._n_operator() - self.ng1 * self._identity(),
                                                                   self._n_operator() - self.ng2 * self._identity())
        return kinetic_mat

    def potentialmat(self):
        """Return the potential energy matrix for the potential."""
        potential_mat = -0.5 * self.EJ1 * np.kron(self._exp_i_phi_operator() + self._exp_i_phi_operator().T,
                                                  self._identity())
        potential_mat += -0.5 * self.EJ2 * np.kron(self._identity(),
                                                   self._exp_i_phi_operator() + self._exp_i_phi_operator().T)
        potential_mat += -0.5 * self.EJ3 * (np.exp(1j * 2 * np.pi * self.flux)
                                            * np.kron(self._exp_i_phi_operator(), self._exp_i_phi_operator().T))
        potential_mat += -0.5 * self.EJ3 * (np.exp(-1j * 2 * np.pi * self.flux)
                                            * np.kron(self._exp_i_phi_operator().T, self._exp_i_phi_operator()))
        potential_mat += (self.EJ1+self.EJ2+self.EJ3)*np.kron(self._identity(), self._identity())
        
        return potential_mat

    def hamiltonian(self):
        """Return Hamiltonian in basis obtained by employing charge basis for both degrees of freedom"""
        return self.kineticmat() + self.potentialmat()

    def _n_operator(self):
        diag_elements = np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_)
        return np.diag(diag_elements)

    def _exp_i_phi_operator(self):
        dim = 2 * self.ncut + 1
        off_diag_elements = np.ones(dim - 1, dtype=np.complex_)
        e_iphi_matrix = np.diag(off_diag_elements, k=1)
        return e_iphi_matrix

    def _identity(self):
        dim = 2 * self.ncut + 1
        return np.eye(dim)

    def n_1_operator(self):
        r"""Return charge number operator conjugate to :math:`\phi_1`"""
        return np.kron(self._n_operator(), self._identity())

    def n_2_operator(self):
        r"""Return charge number operator conjugate to :math:`\phi_2`"""
        return np.kron(self._identity(), self._n_operator())

    def exp_i_phi_1_operator(self):
        r"""Return operator :math:`e^{i\phi_1}` in the charge basis."""
        return np.kron(self._exp_i_phi_operator(), self._identity())

    def exp_i_phi_2_operator(self):
        r"""Return operator :math:`e^{i\phi_2}` in the charge basis."""
        return np.kron(self._identity(), self._exp_i_phi_operator())

    def cos_phi_1_operator(self):
        """Return operator :math:`\\cos \\phi_1` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_1_operator()
        cos_op += cos_op.T
        return cos_op

    def cos_phi_2_operator(self):
        """Return operator :math:`\\cos \\phi_2` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_2_operator()
        cos_op += cos_op.T
        return cos_op

    def sin_phi_1_operator(self):
        """Return operator :math:`\\sin \\phi_1` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_1_operator()
        sin_op += sin_op.conj().T
        return sin_op

    def sin_phi_2_operator(self):
        """Return operator :math:`\\sin \\phi_2` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_2_operator()
        sin_op += sin_op.conj().T
        return sin_op

    def plot_potential(self, phi_grid=None, contour_vals=None, **kwargs):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
        contour_vals: list of float, optional
            specific contours to draw
        **kwargs:
            plot options
        """
        phi_grid = phi_grid or self._default_grid
        x_vals = y_vals = phi_grid.make_linspace()
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (5, 5)
        return plot.contours(x_vals, y_vals, lambda x, y: self.potential([x, y]), contour_vals=contour_vals, **kwargs)

    def wavefunction(self, esys=None, which=0, phi_grid=None):
        """
        Return a flux qubit wave function in phi1, phi2 basis

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
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        phi_grid = phi_grid or self._default_grid

        dim = 2 * self.ncut + 1
        state_amplitudes = np.reshape(evecs[:, which], (dim, dim))

        n_vec = np.arange(-self.ncut, self.ncut + 1)
        phi_vec = phi_grid.make_linspace()
        a_1_phi = np.exp(1j * np.outer(phi_vec, n_vec)) / (2 * np.pi) ** 0.5
        a_2_phi = a_1_phi.T
        wavefunc_amplitudes = np.matmul(a_1_phi, state_amplitudes)
        wavefunc_amplitudes = np.matmul(wavefunc_amplitudes, a_2_phi)
        wavefunc_amplitudes = spec_utils.standardize_phases(wavefunc_amplitudes)

        grid2d = discretization.GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                                     [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, phi_grid=None, mode='abs', zero_calibrate=True, **kwargs):
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
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
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (5, 5)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
