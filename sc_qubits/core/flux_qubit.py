# flux_qubit.py
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import numpy as np
from scipy import sparse
import scipy.integrate as integrate

import sc_qubits.utils.constants as constants
import sc_qubits.utils.plotting as plot

from sc_qubits.core.qubit_base import QubitBaseClass
from sc_qubits.utils.spectrum_utils import extract_phase, order_eigensystem
from sc_qubits.core.discretization import GridSpec
from sc_qubits.core.data_containers import WaveFunctionOnGrid


# -Flux qubit, both degrees of freedom in charge basis---------------------------------------------------------

class FluxQubit(QubitBaseClass):
    r"""Flux Qubit

    | [1] Orlando et al., Physical Review B, 60, 15398 (1999). https://link.aps.org/doi/10.1103/PhysRevB.60.15398

    Flux qubit where the two big junctions are assumed to be identical, and 
    the smaller junction has junction energy and capacitance reduced by 
    the multiplicative constant `alpha`.
    :math:`H = H_\text{flux}=2E_\text{m}(n_m-n_{gm})^2+2E_\text{p}(n_p-n_{gp})^2
                -2E_{J}\cos\phi_{p}\cos\phi_{m}-\alpha E_{J}\cos(2\pi f + 2\phi_{m}),`
                `$E_\text{m}=\frac{e^2}{2(C_{J}+2\alpha C_{J}+ C_{g})}$, 
                 $E_\text{p}=\frac{e^2}{2(C_{J}+C_{g})}$`

    Formulation of the Hamiltonian matrix proceeds by using charge basis for
    both degrees of freedom.

    Parameters
    ----------
    EJ: float
        Josephson energy of the two big junctions
    ECJ: float
        charging energy associated with the two big junctions
    ECg: float
        charging energy associated with the capacitive coupling to ground
    alpha: float
        multiplicative constant reducing EJ and increasing ECJ of the small junction
    ng1: float
        offset charge associated with island 1
    ng2: float
        offset charge associated with island 2
    flux: float
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    ncut: int
        charge number cutoff for the charge on both islands `n`,  `n = -ncut, ..., ncut`
    truncated_dim: int, optional
        desired dimension of the truncated quantum system
    """

    def __init__(self, EJ, ECJ, ECg, alpha, ng1, ng2, flux, ncut, truncated_dim=None):
        self.EJ = EJ
        self.ECJ = ECJ
        self.ECg = ECg
        self.alpha = alpha
        self.ng1 = ng1
        self.ng2 = ng2
        self.flux = flux
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._define_parameters()
        
        self._sys_type = 'flux qubit without disorder in the two large junctions'


    def _define_parameters(self):
        """Defines parameters necessary for defining the Hamiltonian"""
        self.ngm = self.ng1 - self.ng2
        self.ngp = self.ng1 + self.ng2
        self.CJ = 1. / (2*self.ECJ) #capacitances in units where e is set to 1
        self.Cg = 1. / (2*self.ECg)
        self.Cp = self.CJ + self.Cg
        self.Cm = self.CJ + 2.*self.alpha*self.CJ + self.Cg
        self.ECm = 1. / (2.*self.Cm)
        self.ECp = 1. / (2.*self.Cp)
        
    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self):
        """Returns Hilbert space dimension"""
        return (2 * self.ncut + 1)**2

    def potential(self, phim, phip):
        """
        Returns the value of the potential energy at the location
        specified by phim and phip, disregarding constants.
        """
        return (-2.0*self.EJ*np.cos(phim)*np.cos(phip) - self.alpha*self.EJ*np.cos(2.0*np.pi*self.flux + 2.0*phim))

    def sparse_kineticmat(self):
        """Returns the kinetic energy matrix in sparse (`csc_matrix`) form."""
        dimm = dimp = 2*self.ncut + 1
        identity_m = sparse.identity(2*self.ncut+1, format='csc')
        identity_p = sparse.identity(2*self.ncut+1, format='csc')

        diag_elements_m = 2.0*self.ECm*np.square(np.arange(-self.ncut + self.ngm, self.ncut + 1 + self.ngm))
        diag_elements_p = 2.0*self.ECp*np.square(np.arange(-self.ncut + self.ngp, self.ncut + 1 + self.ngp))

        kinetic_matrix_m = sparse.dia_matrix((diag_elements_m, [0]), shape=(dimm, dimm)).tocsc()
        kinetic_matrix_p = sparse.dia_matrix((diag_elements_p, [0]), shape=(dimp, dimp)).tocsc()

        kinetic_matrix = sparse.kron(kinetic_matrix_m, identity_p, format='csc') \
                         + sparse.kron(identity_m, kinetic_matrix_p, format='csc')

        return kinetic_matrix


    def sparse_potentialmat(self):
        """Returns the potential energy matrix for the potential in sparse (`csc_matrix`) form."""
        dimm = dimp = 2*self.ncut + 1
        
        potential_mat = -2.*self.EJ*self.cos_phim_operator()*self.cos_phip_operator()
        
        #define cos(2*pi*f+2*phi_m) operator
        creation_op = sparse.dia_matrix(([1.0] * dimm, [2]), shape=(dimm, dimm))
        creation_op_full = sparse.kron(creation_op, self.identity_phip())
        annihilation_op = sparse.dia_matrix(([1.0] * dimm, [-2]), shape=(dimm, dimm))
        annihilation_op_full = sparse.kron(annihilation_op, self.identity_phip())
        
        potential_mat += -0.5*self.alpha*self.EJ*(np.exp(1j*2*np.pi*self.flux)*creation_op_full 
                                                  + np.exp(-1j*2*np.pi*self.flux)*annihilation_op_full)
        
        return potential_mat

    def hamiltonian(self):
        """Returns Hamiltonian in basis obtained by employing charge basis for both degrees of freedom"""
        return self.sparse_kineticmat() + self.sparse_potentialmat()

    def identity_phim(self):
        dimm = 2 * self.ncut + 1
        return sparse.identity(dimm, format='csc')
    
    def identity_phip(self):
        dimp = 2 * self.ncut + 1
        return sparse.identity(dimp, format='csc')

    def cos_phim_operator(self):
        r"""
        Operator :math:`\cos(phim)`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        dimm = 2 * self.ncut + 1
        cos_phim_matrix = 0.5 * (sparse.dia_matrix(([1.0] * dimm, [-1]), shape=(dimm, dimm)) +
                                  sparse.dia_matrix(([1.0] * dimm, [1]), shape=(dimm, dimm))).tocsc()

        return sparse.kron(cos_phim_matrix, self.identity_phip(), format='csc')
    
    def cos_phip_operator(self):
        r"""
        Operator :math:`\cos(phip)`.

        Returns
        -------
            scipy.sparse.csc_matrix
        """
        dimp = 2 * self.ncut + 1
        cos_phip_matrix = 0.5 * (sparse.dia_matrix(([1.0] * dimp, [-1]), shape=(dimp, dimp)) +
                                  sparse.dia_matrix(([1.0] * dimp, [1]), shape=(dimp, dimp))).tocsc()

        return sparse.kron(self.identity_phim(), cos_phip_matrix, format='csc')

    def plot_potential(self, phi_pts=100, contour_vals=None, aspect_ratio=None, filename=None):
        """Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_pts: int, optional
            (Default value = 100)
        contour_vals: list, optional
            (Default value = None)
        aspect_ratio: float, optional
            (Default value = None)
        filename: str, optional
            (Default value = None)
        """
        x_vals = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        y_vals = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals, aspect_ratio, filename)

    def wavefunction(self, esys=None, which=0, phi_pts=100):
        """Returns a flux qubit wave function in `phim`, `phip` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (Default value = 0)
        phi_pts: int, optional
            number of points to use on grid in each direction

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        dimm = dimp = 2*self.ncut + 1
        state_amplitudes = np.reshape(evecs[:, which],(dimm,dimp))

        nm_vec = np.arange(-self.ncut, self.ncut+1)
        np_vec = np.arange(-self.ncut, self.ncut+1)
        phim_vec = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        phip_vec = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        a_n_phim = np.exp(-1j * np.outer(phim_vec, nm_vec)) / (2 * np.pi)**0.5
        a_n_phip = np.exp(1j * np.outer(np_vec, phip_vec)) / (2 * np.pi)**0.5
        wavefunc_amplitudes = np.matmul(a_n_phim,state_amplitudes)
        wavefunc_amplitudes = np.matmul(wavefunc_amplitudes,a_n_phip).T
        phase = extract_phase(wavefunc_amplitudes)
        wavefunc_amplitudes = np.exp(-1j * phase) * wavefunc_amplitudes

        grid2d = GridSpec(np.asarray([[-np.pi / 2, 3 * np.pi / 2, phi_pts],
                                      [-np.pi / 2, 3 * np.pi / 2, phi_pts]]))

        return WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, phi_pts=100, mode='abs', zero_calibrate=False, figsize=(10, 10),
                          aspect_ratio=1, fig_ax=None):
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (Default value = (0)
        phi_pts: int, optional
            number of points to be used in the 2pi interval in each direction
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (Default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        figsize: (float, float), optional
            figure size specifications for matplotlib
        aspect_ratio: float, optional
            aspect ratio for matplotlib
        fig_ax: Figure, Axes, optional
            existing Figure, Axis if previous objects are to be appended

        Returns
        -------
        Figure, Axes
        """
        modefunction = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_pts=phi_pts, which=which)
        wavefunc.amplitudes = modefunction(wavefunc.amplitudes)
        return plot.wavefunction2d(wavefunc, figsize=figsize, aspect_ratio=aspect_ratio, zero_calibrate=zero_calibrate,
                                   fig_ax=fig_ax)

