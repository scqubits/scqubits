# zeropi.py
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
import warnings

import numpy as np
from scipy import sparse

import scqubits.core.central_dispatch as dispatch
import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
from scqubits.core.noise import NoisySystem, NOISE_PARAMS
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.ui.qubit_widget as ui
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils


# - ZeroPi noise class

class NoisyZeroPi(NoisySystem):
    pass


# -Symmetric 0-pi qubit, phi discretized, theta in charge basis---------------------------------------------------------

class ZeroPi(base.QubitBaseClass, serializers.Serializable, NoisyZeroPi):
    r"""Zero-Pi Qubit

    | [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    | [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    | [3] Groszkowski et al., New J. Phys. 20, 043053 (2018). https://doi.org/10.1088/1367-2630/aab7cd

    Zero-Pi qubit without coupling to the `zeta` mode, i.e., no disorder in `EC` and `EL`,
    see Eq. (4) in Groszkowski et al., New J. Phys. 20, 043053 (2018),

    .. math::

        H &= -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
               +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta
               -2E_\text{J}\cos\theta\cos(\phi-\varphi_\text{ext}/2)+E_L\phi^2\\
          &\qquad +2E_\text{J} + E_J dE_J \sin\theta\sin(\phi-\phi_\text{ext}/2).

    Formulation of the Hamiltonian matrix proceeds by discretization of the `phi` variable, and using charge basis for
    the `theta` variable.

    Parameters
    ----------
    EJ: float
        mean Josephson energy of the two junctions
    EL: float
        inductive energy of the two (super-)inductors
    ECJ: float
        charging energy associated with the two junctions
    EC: float or None
        charging energy of the large shunting capacitances; set to `None` if `ECS` is provided instead
    dEJ: float
        relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dCJ: float
        relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    ng: float
        offset charge associated with theta
    flux: float
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid1d object
        specifies the range and spacing of the discretization lattice
    ncut: int
        charge number cutoff for `n_theta`,  `n_theta = -ncut, ..., ncut`
    ECS: float, optional
        total charging energy including large shunting capacitances and junction capacitances; may be provided instead
        of EC
    truncated_dim: int, optional
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
   """
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dEJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dCJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ncut = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EL, ECJ, EC, ng, flux, grid, ncut, dEJ=0, dCJ=0, ECS=None, truncated_dim=None):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ

        if EC is None and ECS is None:
            raise ValueError("Argument missing: must either provide EC or ECS")
        if EC and ECS:
            raise ValueError("Argument error: can only provide either EC or ECS")
        if EC:
            self.EC = EC
        else:
            self.EC = 1 / (1 / ECS - 1 / self.ECJ)
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.ng = ng
        self.flux = flux
        self.grid = grid
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        # for theta, needed for plotting wavefunction
        self._default_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 100)
        self._init_params.remove('ECS')  # used in for file Serializable purposes; remove ECS as init parameter
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_img/zeropi.jpg')
        dispatch.CENTRAL_DISPATCH.register('GRID_UPDATE', self)

    @staticmethod
    def default_params():
        return {
            'EJ': 10.0,
            'EL': 0.04,
            'ECJ': 20.0,
            'EC': 0.04,
            'dEJ': 0.0,
            'dCJ': 0.0,
            'ng': 0.1,
            'flux': 0.23,
            'ncut': 30,
            'truncated_dim': 10
        }

    @classmethod
    def create(cls):
        phi_grid = discretization.Grid1d(-19.0, 19.0, 200)
        init_params = cls.default_params()
        zeropi = cls(**init_params, grid=phi_grid)
        zeropi.widget()
        return zeropi

    def supported_noise_channels(self):
        """Return a list of supported noise channels"""
        return [
                'tphi_1_over_f_cc', 
                'tphi_1_over_f_flux',
                't1_flux_bias_line',
                # 't1_capacitive',
                't1_inductive',
                ]

    def widget(self, params=None):
        init_params = params or self.get_initdata()
        del init_params['grid']
        init_params['grid_max_val'] = self.grid.max_val
        init_params['grid_min_val'] = self.grid.min_val
        init_params['grid_pt_count'] = self.grid.pt_count
        ui.create_widget(self.set_params, init_params, image_filename=self._image_filename)

    def set_params(self, **kwargs):
        phi_grid = discretization.Grid1d(kwargs.pop('grid_min_val'),
                                         kwargs.pop('grid_max_val'),
                                         kwargs.pop('grid_pt_count'))
        self.grid = phi_grid
        for param_name, param_val in kwargs.items():
            setattr(self, param_name, param_val)

    def receive(self, event, sender, **kwargs):
        if sender is self.grid:
            self.broadcast('QUANTUMSYSTEM_UPDATE')

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, sigma=0.0, which='LM', return_eigenvectors=False)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, sigma=0.0, which='LM',
                                           return_eigenvectors=True)
        # TODO consider normalization of zeropi wavefunctions
        # evecs /= np.sqrt(self.grid.grid_spacing())
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def get_ECS(self):
        return 1 / (1 / self.EC + 1 / self.ECJ)

    def set_ECS(self, value):
        warnings.warn("It is not possible to directly set ECS (except in initialization). Instead, set EC or ECJ, "
                      "or use set_EC_via_ECS() to update EC indirectly.", Warning)

    ECS = property(get_ECS, set_ECS)

    def set_EC_via_ECS(self, ECS):
        """Helper function to set `EC` by providing `ECS`, keeping `ECJ` constant."""
        self.EC = 1 / (1 / ECS - 1 / self.ECJ)

    def hilbertdim(self):
        """Returns Hilbert space dimension"""
        return self.grid.pt_count * (2 * self.ncut + 1)

    def potential(self, phi, theta):
        """
        Parameters
        ----------
        phi: float
        theta: float

        Returns
        -------
        float
            value of the potential energy evaluated at phi, theta
        """
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)
                + self.EL * phi ** 2 + 2.0 * self.EJ
                + self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def sparse_kinetic_mat(self):
        """
        Kinetic energy portion of the Hamiltonian.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the kinetic energy operator
        """
        pt_count = self.grid.pt_count
        dim_theta = 2 * self.ncut + 1
        identity_phi = sparse.identity(pt_count, format='csc')
        identity_theta = sparse.identity(dim_theta, format='csc')
        kinetic_matrix_phi = self.grid.second_derivative_matrix(prefactor=-2.0 * self.ECJ)
        diag_elements = 2.0 * self.ECS * np.square(np.arange(-self.ncut + self.ng, self.ncut + 1 + self.ng))
        kinetic_matrix_theta = sparse.dia_matrix((diag_elements, [0]), shape=(dim_theta, dim_theta)).tocsc()
        kinetic_matrix = (sparse.kron(kinetic_matrix_phi, identity_theta, format='csc')
                          + sparse.kron(identity_phi, kinetic_matrix_theta, format='csc'))
        if self.dCJ != 0:
            kinetic_matrix -= 2.0 * self.ECS * self.dCJ * self.i_d_dphi_operator() * self.n_theta_operator()

        return kinetic_matrix

    def sparse_potential_mat(self):
        """
        Potential energy portion of the Hamiltonian.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the potential energy operator
        """
        pt_count = self.grid.pt_count
        grid_linspace = self.grid.make_linspace()
        dim_theta = 2 * self.ncut + 1

        phi_inductive_vals = self.EL * np.square(grid_linspace)
        phi_inductive_potential = sparse.dia_matrix((phi_inductive_vals, [0]), shape=(pt_count, pt_count)).tocsc()
        phi_cos_vals = np.cos(grid_linspace - 2.0 * np.pi * self.flux / 2.0)
        phi_cos_potential = sparse.dia_matrix((phi_cos_vals, [0]), shape=(pt_count, pt_count)).tocsc()
        phi_sin_vals = np.sin(grid_linspace - 2.0 * np.pi * self.flux / 2.0)
        phi_sin_potential = sparse.dia_matrix((phi_sin_vals, [0]), shape=(pt_count, pt_count)).tocsc()

        theta_cos_potential = (-self.EJ
                               * (sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)) +
                                  sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)))).tocsc()
        potential_mat = (sparse.kron(phi_cos_potential, theta_cos_potential, format='csc')
                         + sparse.kron(phi_inductive_potential, self._identity_theta(), format='csc')
                         + 2 * self.EJ * sparse.kron(self._identity_phi(), self._identity_theta(), format='csc'))
        if self.dEJ != 0:
            potential_mat += (self.EJ * self.dEJ * sparse.kron(phi_sin_potential, self._identity_theta(), format='csc')
                              * self.sin_theta_operator())
        return potential_mat

    def hamiltonian(self):
        """Calculates Hamiltonian in basis obtained by discretizing phi and employing charge basis for theta.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the potential energy operator
        """
        return self.sparse_kinetic_mat() + self.sparse_potential_mat()

    def sparse_d_potential_d_flux_mat(self):
        r"""Calculates a of the potential energy w.r.t flux, at the current value of flux,
        as stored in the object.

        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0.
        So if \frac{\partial U}{ \partial \Phi_{\rm ext}}, is needed, the expression returned
        by this function, needs to be multiplied by 1/\Phi_0.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the derivative of the potential energy
        """
        op_1 = sparse.kron(self._sin_phi_operator(x=- 2.0 * np.pi * self.flux / 2.0),
                           self._cos_theta_operator(), format='csc')
        op_2 = sparse.kron(self._cos_phi_operator(x=- 2.0 * np.pi * self.flux / 2.0),
                           self._sin_theta_operator(), format='csc')
        return - 2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2

    def d_hamiltonian_d_flux(self):
        r"""Calculates a derivative of the Hamiltonian w.r.t flux, at the current value of flux,
        as stored in the object.

        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0.
        So if \frac{\partial H}{ \partial \Phi_{\rm ext}}, is needed, the expression returned
        by this function, needs to be multiplied by 1/\Phi_0.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the derivative of the Hamiltonian
        """
        return self.sparse_d_potential_d_flux_mat()

    def sparse_d_potential_d_EJ_mat(self):
        r"""Calculates a of the potential energy w.r.t EJ.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the derivative of the potential energy
        """
        return - 2.0 * sparse.kron(self._cos_phi_operator(x=- 2.0 * np.pi * self.flux / 2.0),
                                   self._cos_theta_operator(), format='csc')

    def d_hamiltonian_d_EJ(self):
        r"""Calculates a derivative of the Hamiltonian w.r.t EJ.
        for calcu 

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the derivative of the Hamiltonian
        """
        return self.sparse_d_potential_d_EJ_mat()


    def d_hamiltonian_d_ng(self):
        r"""Calculates a derivative of the Hamiltonian w.r.t ng.
        as stored in the object.

        Returns
        -------
        scipy.sparse.csc_matrix
            matrix representing the derivative of the Hamiltonian
        """
        return -8 * self.EC * self.n_theta_operator()


    def _identity_phi(self):
        r"""
        Identity operator acting only on the `\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        pt_count = self.grid.pt_count
        return sparse.identity(pt_count, format='csc')

    def _identity_theta(self):
        r"""
        Identity operator acting only on the `\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        dim_theta = 2 * self.ncut + 1
        return sparse.identity(dim_theta, format='csc')

    def i_d_dphi_operator(self):
        r"""
        Operator :math:`i d/d\phi`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        return sparse.kron(self.grid.first_derivative_matrix(prefactor=1j), self._identity_theta(), format='csc')

    def _phi_operator(self):
        r"""
        Operator :math:`\phi`, acting only on the `\phi` Hilbert subspace.


        Returns
        -------
            scipy.sparse.csc_matrix
        """
        pt_count = self.grid.pt_count

        phi_matrix = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = self.grid.make_linspace()
        phi_matrix.setdiag(diag_elements)
        return phi_matrix

    def phi_operator(self):
        r"""
        Operator :math:`\phi`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        return sparse.kron(self._phi_operator(), self._identity_theta(), format='csc')

    def n_theta_operator(self):
        r"""
        Operator :math:`n_\theta`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        dim_theta = 2 * self.ncut + 1
        diag_elements = np.arange(-self.ncut, self.ncut + 1)
        n_theta_matrix = sparse.dia_matrix((diag_elements, [0]), shape=(dim_theta, dim_theta)).tocsc()
        return sparse.kron(self._identity_phi(), n_theta_matrix, format='csc')

    def _sin_phi_operator(self, x=0):
        r"""
        Operator :math:`\sin(\phi + x)`, acting only on the `\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        pt_count = self.grid.pt_count

        vals = np.sin(self.grid.make_linspace() + x)
        sin_phi_matrix = sparse.dia_matrix((vals, [0]), shape=(pt_count, pt_count)).tocsc()
        return sin_phi_matrix

    def _cos_phi_operator(self, x=0):
        r"""
        Operator :math:`\cos(\phi + x)`, acting only on the `\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        pt_count = self.grid.pt_count

        vals = np.cos(self.grid.make_linspace() + x)
        cos_phi_matrix = sparse.dia_matrix((vals, [0]), shape=(pt_count, pt_count)).tocsc()
        return cos_phi_matrix

    def _cos_theta_operator(self):
        r"""
        Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        dim_theta = 2 * self.ncut + 1
        cos_theta_matrix = 0.5 * (sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)) +
                                  sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta))).tocsc()
        return cos_theta_matrix

    def cos_theta_operator(self):
        r"""
        Operator :math:`\cos(\theta)`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        return sparse.kron(self._identity_phi(), self._cos_phi_operator(), format='csc')

    def _sin_theta_operator(self):
        r"""
        Operator :math:`\sin(\theta)`, acting only on the `\theta` Hilbert space.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        dim_theta = 2 * self.ncut + 1
        sin_theta_matrix = (-0.5 * 1j
                            * (sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)) -
                               sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))).tocsc())
        return sin_theta_matrix

    def sin_theta_operator(self):
        r"""
        Operator :math:`\sin(\theta)`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        return sparse.kron(self._identity_phi(), self._sin_theta_operator(), format='csc')

    def plot_potential(self, theta_grid=None, contour_vals=None, **kwargs):
        """Draw contour plot of the potential energy.

        Parameters
        ----------
        theta_grid: Grid1d, optional
            used for setting a custom grid for theta; if None use self._default_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        theta_grid = theta_grid or self._default_grid

        x_vals = self.grid.make_linspace()
        y_vals = theta_grid.make_linspace()
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals,
                             xlabel=r'$\phi$', ylabel=r'$\theta$', **kwargs)

    def wavefunction(self, esys=None, which=0, theta_grid=None):
        """Returns a zero-pi wave function in `phi`, `theta` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (default value = 0)
        theta_grid: Grid1d, optional
            used for setting a custom grid for theta; if None use self._default_grid

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        theta_grid = theta_grid or self._default_grid
        dim_theta = 2 * self.ncut + 1
        state_amplitudes = evecs[:, which].reshape(self.grid.pt_count, dim_theta)

        # Calculate psi_{phi, theta} = sum_n state_amplitudes_{phi, n} A_{n, theta}
        # where a_{n, theta} = 1/sqrt(2 pi) e^{i n theta}
        n_vec = np.arange(-self.ncut, self.ncut + 1)
        theta_vec = theta_grid.make_linspace()
        a_n_theta = np.exp(1j * np.outer(n_vec, theta_vec)) / (2 * np.pi) ** 0.5
        wavefunc_amplitudes = np.matmul(state_amplitudes, a_n_theta).T
        wavefunc_amplitudes = spec_utils.standardize_phases(wavefunc_amplitudes)

        grid2d = discretization.GridSpec(np.asarray([[self.grid.min_val, self.grid.max_val, self.grid.pt_count],
                                                     [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, theta_grid=None, mode='abs', zero_calibrate=True, **kwargs):
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        theta_grid: Grid1d, optional
            used for setting a custom grid for theta; if None use self._default_grid
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
        theta_grid = theta_grid or self._default_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, theta_grid=theta_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate,
                                   xlabel=r'$\phi$', ylabel=r'$\theta$', **kwargs)
