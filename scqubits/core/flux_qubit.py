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

import numpy as np
import scipy as sp
import scipy.integrate as integrate

import scqubits.utils.constants as constants
import scqubits.utils.plotting as plot

from scqubits.core.qubit_base import QubitBaseClass
from scqubits.utils.spectrum_utils import extract_phase, order_eigensystem
from scqubits.core.discretization import GridSpec
from scqubits.core.data_containers import WaveFunctionOnGrid


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
    EJ*: float
        Josephson energy of the *th junction in GHz; typically
        EJ1 \approx EJ2, with EJ3 = \alpha EJ1 with \alpha < 1
    ECJ*: float
        charging energy associated with the *th junction in GHz
    ECg*: float
        charging energy associated with the capacitive coupling to ground for the two islands in GHz
    ng*: float
        offset charge associated with island *
    flux: float
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    ncut: int
        charge number cutoff for the charge on both islands `n`,  `n = -ncut, ..., ncut`
    truncated_dim: int, optional
        desired dimension of the truncated quantum system
    """

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, ncut, truncated_dim=None):
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
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._define_parameters()
        self._define_capacitance_matrix()
        self._define_charging_energy_matrix()
        
        self._sys_type = 'flux qubit without disorder in the two large junctions'

    def _define_parameters(self):
        """Defines parameters necessary for defining the Hamiltonian"""
        self.CJ1 = 1. / (2*self.ECJ1) #capacitances in units where e is set to 1
        self.CJ2 = 1. / (2*self.ECJ2)
        self.CJ3 = 1. / (2*self.ECJ3)
        self.Cg1 = 1. / (2*self.ECg1)
        self.Cg2 = 1. / (2*self.ECg2)
    
    def _define_capacitance_matrix(self):
        Cmat = np.zeros((2,2))
        Cmat[0,0] = self.CJ1 + self.CJ3 + self.Cg1
        Cmat[1,1] = self.CJ2 + self.CJ3 + self.Cg2
        Cmat[0,1] = -self.CJ3
        Cmat[1,0] = -self.CJ3
        self.Cmat = Cmat
        
    def _define_charging_energy_matrix(self):
        self.ECmat = np.linalg.inv(self.Cmat) / 2.
    
    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(hamiltonian_mat, eigvals=(0,evals_count-1), eigvals_only=True)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, eigvals=(0,evals_count-1), eigvals_only=False)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def hilbertdim(self):
        """Returns Hilbert space dimension"""
        return (2 * self.ncut + 1)**2

    def potential(self, phi1, phi2):
        """
        Returns the value of the potential energy at the location
        specified by phim and phip, disregarding constants.
        """
        return (-self.EJ1*np.cos(phi1) - self.EJ2*np.cos(phi2) - self.EJ3*np.cos(2.0*np.pi*self.flux + phi1 - phi2))

    def kineticmat(self):
        """Returns the kinetic energy matrix."""
        dim = 2*self.ncut + 1
        ECmat = self.ECmat
        
        T = 0.
        T += 4.0*ECmat[0,0]*np.kron(np.matmul(self.n1_operator(),self.n1_operator()),self.identity())
        T += 4.0*ECmat[1,1]*np.kron(self.identity(),np.matmul(self.n2_operator(),self.n2_operator()))
        T += 4.0*ECmat[0,1]*np.kron(self.n1_operator(),self.n2_operator())
        T += 4.0*ECmat[1,0]*np.kron(self.n1_operator(),self.n2_operator())

        return T


    def potentialmat(self):
        """Returns the potential energy matrix for the potential."""
        dim = 2*self.ncut + 1
        
        U = 0.
        U += -0.5*self.EJ1*np.kron(self.e_plusiphi_operator()+self.e_minusiphi_operator(),self.identity())
        U += -0.5*self.EJ2*np.kron(self.identity(),self.e_plusiphi_operator()+self.e_minusiphi_operator())
        U += -0.5*self.EJ3*(np.exp(1j*2*np.pi*self.flux)*np.kron(self.e_plusiphi_operator(),self.e_minusiphi_operator())
                            + np.exp(-1j*2*np.pi*self.flux)*np.kron(self.e_minusiphi_operator(),self.e_plusiphi_operator()))
        
        return U

    def hamiltonian(self):
        """Returns Hamiltonian in basis obtained by employing charge basis for both degrees of freedom"""
        return self.kineticmat() + self.potentialmat()

    def identity(self):
        dim = 2 * self.ncut + 1
        return np.eye(dim)
    
    def n1_operator(self):
        dim = 2 * self.ncut + 1
        diag_elements_1 = np.arange(-self.ncut + self.ng1, self.ncut + 1 + self.ng1, dtype=np.complex128)
        return np.diag(diag_elements_1)
        
    def n2_operator(self):
        dim = 2 * self.ncut + 1
        diag_elements_2 = np.arange(-self.ncut + self.ng2, self.ncut + 1 + self.ng2, dtype=np.complex128)
        return np.diag(diag_elements_2)
        
    def e_plusiphi_operator(self):
        r"""
        Operator :math:`\e^(iphi)`.

        Returns
        -------
            np.ndarray
        """
        dim = 2 * self.ncut + 1
        off_diag_elements = np.ones(dim-1, dtype=np.complex128)
        e_plusiphi_matrix = np.diag(off_diag_elements, k=1)

        return e_plusiphi_matrix
    
    def e_minusiphi_operator(self):
        r"""
        Operator :math:`\e^(-iphi)`.

        Returns
        -------
            np.ndarray
        """
        dim = 2 * self.ncut + 1
        off_diag_elements = np.ones(dim-1, dtype=np.complex128)
        e_minusiphi_matrix = np.diag(off_diag_elements, k=-1)

        return e_minusiphi_matrix
    
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

        dim = 2*self.ncut + 1
        state_amplitudes = np.reshape(evecs[:, which],(dim,dim))

        n_vec = np.arange(-self.ncut, self.ncut+1)
        phi_vec = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        a_1_phim = np.exp(-1j * np.outer(phi_vec, n_vec)) / (2 * np.pi)**0.5
        a_2_phip = np.exp(1j * np.outer(n_vec, phi_vec)) / (2 * np.pi)**0.5
        wavefunc_amplitudes = np.matmul(a_1_phim,state_amplitudes)
        wavefunc_amplitudes = np.matmul(wavefunc_amplitudes,a_2_phip).T
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

