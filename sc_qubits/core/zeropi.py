# zeropi.py
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################

import numpy as np
from scipy import sparse

import sc_qubits.utils.constants as constants
import sc_qubits.utils.plotting as plot

from sc_qubits.core.qubit_base import QubitBaseClass
from sc_qubits.utils.spectrum_utils import extract_phase, order_eigensystem
from sc_qubits.core.discretization import GridSpec
from sc_qubits.core.data_containers import WaveFunctionOnGrid


# -Symmetric 0-pi qubit, phi discretized, theta in charge basis---------------------------------------------------------


class ZeroPi(QubitBaseClass):
    r"""Zero-Pi Qubit

    | [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    | [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    | [3] Groszkowski et al., New J. Phys. 20, 043053 (2018). https://doi.org/10.1088/1367-2630/aab7cd

    Zero-Pi qubit without coupling to the `zeta` mode, i.e., no disorder in `EC` and `EL`,
    see Eq. (4) in Groszkowski et al., New J. Phys. 20, 043053 (2018),
    :math:`H = -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
    +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta
    -2E_\text{J}\cos\theta\cos(\phi-\varphi_\text{ext}/2)+E_L\phi^2+2E_\text{J}
    + E_J dE_J \sin\theta\sin(\phi-\phi_\text{ext}/2)`.

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
        desired dimension of the truncated quantum system
    """

    def __init__(self, EJ, EL, ECJ, EC, ng, flux, grid, ncut, dEJ=0, dCJ=0, ECS=None, truncated_dim=None):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ

        if EC is None and ECS is None:
            raise ValueError("Argument missing: must either provide EC or ECS")
        elif EC and ECS:
            raise ValueError("Argument error: can only provide either EC or ECS")
        elif EC:
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
        self._sys_type = '0-Pi qubit without EL and EC disorder, no coupling to zeta mode'


    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        # evecs /= np.sqrt(self.grid.grid_spacing())

        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def ECS(self):
        """ """
        return 1 / (1 / self.EC + 1 / self.ECJ)

    def set_EC_via_ECS(self, ECS):
        """Helper function to set `EC` by providing `ECS`, keeping `ECJ` constant."""
        self.EC = 1 / (1 / ECS - 1 / self.ECJ)
        return None

    def hilbertdim(self):
        """ """
        pt_count = self.grid.pt_count
        return pt_count * (2 * self.ncut + 1)

    def potential(self, phi, theta):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(
            phi - 2.0 * np.pi * self.flux / 2.0) + self.EL * phi ** 2 + 2.0 * self.EJ +
                self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def sparse_kineticmat(self):
        """ """
        pt_count = self.grid.pt_count
        dim_theta = 2 * self.ncut + 1
        identity_phi = sparse.identity(pt_count, format='csc')
        identity_theta = sparse.identity(dim_theta, format='csc')

        kinetic_matrix_phi = self.grid.second_derivative_matrix(prefactor=-2.0 * self.ECJ)

        diag_elements = 2.0 * self.ECS() * np.square(np.arange(-self.ncut + self.ng, self.ncut + 1 + self.ng))
        kinetic_matrix_theta = sparse.dia_matrix((diag_elements, [0]), shape=(dim_theta, dim_theta)).tocsc()

        kinetic_matrix = sparse.kron(kinetic_matrix_phi, identity_theta, format='csc') \
                         + sparse.kron(identity_phi, kinetic_matrix_theta, format='csc')
        kinetic_matrix -= 2.0 * self.ECS() * self.dCJ * self.i_d_dphi_operator() * self.n_theta_operator()

        return kinetic_matrix


    def sparse_potentialmat(self):
        """Returns the potential energy matrix for the potential in sparse (`csc_matrix`) form."""
        min_val = self.grid.min_val
        max_val = self.grid.max_val
        pt_count = self.grid.pt_count
        dim_theta = 2 * self.ncut + 1

        phi_inductive_vals = self.EL * np.square(np.linspace(min_val, max_val, pt_count))
        phi_inductive_potential = sparse.dia_matrix((phi_inductive_vals, [0]), shape=(pt_count, pt_count)).tocsc()
        phi_cos_vals = np.cos(np.linspace(min_val, max_val, pt_count) - 2.0 * np.pi * self.flux / 2.0)
        phi_cos_potential = sparse.dia_matrix((phi_cos_vals, [0]), shape=(pt_count, pt_count)).tocsc()
        phi_sin_vals = np.sin(np.linspace(min_val, max_val, pt_count) - 2.0 * np.pi * self.flux / 2.0)
        phi_sin_potential = sparse.dia_matrix((phi_sin_vals, [0]), shape=(pt_count, pt_count)).tocsc()


        theta_cos_potential = (-self.EJ * (sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)) +
                                           sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)))).tocsc()
        potential_mat = (sparse.kron(phi_cos_potential, theta_cos_potential, format='csc')
                         + sparse.kron(phi_inductive_potential, self.identity_theta(), format='csc')
                         + 2 * self.EJ * sparse.kron(self.identity_phi(), self.identity_theta(), format='csc'))
        potential_mat += self.EJ * self.dEJ * sparse.kron(phi_sin_potential, self.identity_theta(), format='csc') \
                         * self.sin_theta_operator()

        return potential_mat

    def hamiltonian(self):
        """Returns Hamiltonian in basis obtained by discretizing phi and employing charge basis for theta"""
        return self.sparse_kineticmat() + self.sparse_potentialmat()

    def identity_phi(self):
        pt_count = self.grid.pt_count
        return sparse.identity(pt_count, format='csc')

    def identity_theta(self):
        dim_theta = 2 * self.ncut + 1
        return sparse.identity(dim_theta, format='csc')

    def i_d_dphi_operator(self):
        """ """
        return sparse.kron(self.grid.first_derivative_matrix(prefactor=1j), self.identity_theta(), format='csc')

    def phi_operator(self):
        """ """
        min_val = self.grid.min_val
        max_val = self.grid.max_val
        pt_count = self.grid.pt_count

        phi_matrix = sparse.dia_matrix((pt_count, pt_count), dtype=np.float_)
        diag_elements = np.linspace(min_val, max_val, pt_count)
        phi_matrix.setdiag(diag_elements)

        return sparse.kron(phi_matrix, self.identity_theta(), format='csc')

    def n_theta_operator(self):
        dim_theta = 2 * self.ncut + 1
        diag_elements = np.arange(-self.ncut, self.ncut + 1)
        n_theta_matrix = sparse.dia_matrix((diag_elements, [0]), shape=(dim_theta, dim_theta)).tocsc()

        return sparse.kron(self.identity_phi(), n_theta_matrix, format='csc')

    def cos_theta_operator(self):
        dim_theta = 2 * self.ncut + 1
        cos_theta_matrix = 0.5 * (sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)) +
                                  sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta))).tocsc()

        return sparse.kron(self.identity_phi(), cos_theta_matrix, format='csc')

    def sin_theta_operator(self):
        dim_theta = 2 * self.ncut + 1
        sin_theta_matrix = - 0.5 * 1j * (sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)) -
                                         sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))).tocsc()

        return sparse.kron(self.identity_phi(), sin_theta_matrix, format='csc')

    def plot_potential(self, theta_pts=100, contour_vals=None, aspect_ratio=None, filename=None):
        """Make contour plot of the potential energy.

        Parameters
        ----------
        theta_pts: int, optional
            (Default value = 100)
        contour_vals: list, optional
            (Default value = None)
        aspect_ratio: float, optional
            (Default value = None)
        filename: str, optional
            (Default value = None)
        """
        min_val = self.grid.min_val
        max_val = self.grid.max_val
        pt_count = self.grid.pt_count

        x_vals = np.linspace(min_val, max_val, pt_count)
        y_vals = np.linspace(-np.pi / 2, 3 * np.pi / 2, theta_pts)
        plot.contours(x_vals, y_vals, self.potential, contour_vals, aspect_ratio, filename)
        return None

    def wavefunction(self, esys=None, theta_pts=100, which=0):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        pt_count = self.grid.pt_count
        dim_theta = 2 * self.ncut + 1
        state_amplitudes = evecs[:, which].reshape(pt_count, dim_theta)

        # Calculate psi_{phi, theta} = sum_n state_amplitudes_{phi, n} A_{n, theta}
        # where a_{n, theta} = 1/sqrt(2 pi) e^{i n theta}
        n_vec = np.arange(-self.ncut, self.ncut+1)
        theta_vec = np.linspace(-np.pi / 2, 3 * np.pi / 2, theta_pts)
        a_n_theta = np.exp(1j * np.outer(n_vec, theta_vec)) / (2 * np.pi)**0.5
        wavefunc_amplitudes = np.matmul(state_amplitudes, a_n_theta).T
        phase = extract_phase(wavefunc_amplitudes)
        wavefunc_amplitudes = np.exp(-1j * phase) * wavefunc_amplitudes

        grid2d = GridSpec(np.asarray([[self.grid.min_val, self.grid.max_val, pt_count], [-np.pi / 2, 3 * np.pi / 2, theta_pts]]))

        return WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, theta_pts=100, mode='abs', figsize=(10, 5), aspect_ratio=3,
                          zero_calibrate=False, fig_ax=None):
        """Different modes:
        'abs_sqr': :math:`|\\psi|^2`
        'abs':  :math:`|\\psi|`
        'real': :math:`\\text{Re}(\\psi)`
        'imag': :math:`\\text{Im}(\\psi)`

        Parameters
        ----------
        esys :

        which :
             (Default value = 0)
        mode :
             (Default value = 'abs')
        figsize :
             (Default value = (10)
        5) :

        aspect_ratio :
             (Default value = 3)
        zero_calibrate :
             (Default value = False)
        fig_ax :
             (Default value = None)

        Returns
        -------
        Figure, Axes

        """
        modefunction = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, theta_pts=theta_pts, which=which)
        wavefunc.amplitudes = modefunction(wavefunc.amplitudes)
        return plot.wavefunction2d(wavefunc, figsize=figsize, aspect_ratio=aspect_ratio, zero_calibrate=zero_calibrate,
                                   fig_ax=fig_ax)

