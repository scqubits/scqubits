# protomon.py
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
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import kn

import scqubits.core.constants as constants
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils
import scqubits.utils.spectrum_utils as matele_utils


# — Inductively-shunted Rhombus circuit ————————————————————————
class BareProtomon(base.QubitBaseClass, serializers.Serializable):
    r"""inductively-shunted Rhombus qubit, with the harmonic mode in the ground state

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        junction charging energy
    EL: float
        inductive energy
    flux_c: float
        common part of the external flux, e.g., 1 corresponds to one flux quantum
    flux_d: float
        differential part of the external flux, e.g., 1 corresponds to one flux quantum
    kbt: float
        photon temperature
    """

    def __init__(self, EJ, EC, EL, flux_c, flux_d):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux_c = flux_c
        self.flux_d = flux_d
        self.phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.truncated_dim = 30
        self.ph = 0  # placeholder
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_

    @staticmethod
    def default_params():
        return {
            "EJ": 15.0,
            "EC": 3.5,
            "EL": 0.32,
            "ELA": 0.32,
            "flux_c": 0.5,
            "flux_d": 0.0,
        }

    @staticmethod
    def nonfit_params():
        return ["flux_c", "flux_d"]

    def dim_phi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi' degree of freedom."""
        return self.phi_grid.pt_count

    def dim_theta(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return self.theta_grid.pt_count

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi() * self.dim_theta()

    def _phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\phi` operator in the discretized basis
        """
        return sparse.dia_matrix(
            (self.phi_grid.make_linspace(), [0]), shape=(self.dim_phi(), self.dim_phi())
        ).tocsc()

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in total Hilbert space
        """
        return self._kron2(self._phi_operator(), self._identity_theta())

    def _n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator
        """
        return self.phi_grid.first_derivative_matrix(prefactor=-1j)

    def n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi' operator in total Hilbert space
        """
        return self._kron2(self._n_phi_operator(), self._identity_theta())

    def _cos_phi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/div` operator
        """
        cos_phi_div_vals = np.cos(self.phi_grid.make_linspace() / div)
        return sparse.dia_matrix(
            (cos_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())
        ).tocsc()

    def _sin_phi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/div` operator
        """
        sin_phi_div_vals = np.sin(self.phi_grid.make_linspace() / div)
        return sparse.dia_matrix(
            (sin_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())
        ).tocsc()

    def _theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return sparse.dia_matrix(
            (self.theta_grid.make_linspace(), [0]),
            shape=(self.dim_theta(), self.dim_theta()),
        ).tocsc()

    def theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return self._kron2(self._identity_phi(), self._theta_operator())

    def _n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator
        """
        return self.theta_grid.first_derivative_matrix(prefactor=-1j)

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_theta` in the total Hilbert space
        """
        return self._kron2(self._identity_phi(), self._n_theta_operator())

    def _cos_theta_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\theta/div` operator
        """
        cos_theta_div_vals = np.cos(self.theta_grid.make_linspace() / div)
        return sparse.dia_matrix(
            (cos_theta_div_vals, [0]), shape=(self.dim_theta(), self.dim_theta())
        ).tocsc()

    def _sin_theta_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\theta/div` operator
        """
        sin_theta_div_vals = np.sin(self.theta_grid.make_linspace() / div)
        return sparse.dia_matrix(
            (sin_theta_div_vals, [0]), shape=(self.dim_theta(), self.dim_theta())
        ).tocsc()

    def _kron2(self, mat1, mat2):
        """
        Returns
        -------
        ndarray
            Returns the kronecker product of two operators
        """
        return sparse.kron(mat1, mat2, format="csc")

    def _identity_phi(self):
        """
        Identity operator acting only on the :math:`\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_phi(), format="csc", dtype=np.complex_)

    def _identity_theta(self):
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_theta(), format="csc", dtype=np.complex_)

    def total_identity(self):
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron2(self._identity_phi(), self._identity_theta())

    def hamiltonian(self):
        r"""Construct Hamiltonian matrix in discretized basis
        :math:`H = 2 E_\text{C} n_\theta^2 + E_\text{L}(1+\dfrac{2E_\text{L}}{E_\text{LA}})^{-1} (\theta - \upphi_\text{d})^2+2 E_\text{C} n_\phi^2 + E_\text{L} (\phi - \upphi_\text{c})^2 - 2 E_\text{J}\cos\phi\cos\theta`
        """
        phi_kinetic = self.phi_grid.second_derivative_matrix(prefactor=-2.0 * self.EC)
        theta_kinetic = self.theta_grid.second_derivative_matrix(
            prefactor=-2.0 * self.EC
        )
        tot_kinetic = self._kron2(phi_kinetic, self._identity_theta()) + self._kron2(
            self._identity_phi(), theta_kinetic
        )

        phi_ind = (
            self.EL
            * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux_c)
            ** 2
        )
        theta_ind = (
            self.EL
            * (self.theta_operator() - self.total_identity() * 2 * np.pi * self.flux_d)
            ** 2
        )

        # note the 2EJ constant term is added to be consistent with the 'LM' option in _evals_calc and _esys_calc
        phi_theta_junction = (
            -2
            * self.EJ
            * self._kron2(
                self._cos_phi_div_operator(1.0), self._cos_theta_div_operator(1.0)
            )
            + 2 * self.EJ * self.total_identity()
        )

        return tot_kinetic + phi_ind + theta_ind + phi_theta_junction

    def potential(self, theta, phi):
        """
        Potential evaluated at `phi, theta`

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`
        theta: float or ndarray
            float value of the phase variable `theta`

        Returns
        -------
        float or ndarray
        """
        return 0

    def plot_potential(
        self, phi_grid=None, theta_grid=None, contour_vals=None, **kwargs
    ):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use self._default_theta_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid

        x_vals = theta_grid.make_linspace()
        y_vals = phi_grid.make_linspace()
        if "figsize" not in kwargs:
            kwargs["figsize"] = (4, 4)
        return plot.contours(
            x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs
        )

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(
            hamiltonian_mat,
            k=evals_count,
            return_eigenvectors=False,
            sigma=0.0,
            which="LM",
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigsh(
            hamiltonian_mat,
            k=evals_count,
            return_eigenvectors=True,
            sigma=0.0,
            which="LM",
        )
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None):
        """Returns a wave function in `phi`, `theta` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (default value = 0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi;
        theta_grid: Grid1d, optional
            used for setting a custom grid for theta;

        Returns
        -------
        WaveFunction object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid

        state_amplitudes = evecs[:, which].reshape(self.dim_phi(), self.dim_theta())
        wavefunc_amplitudes = spec_utils.standardize_phases(state_amplitudes)

        grid2d = discretization.GridSpec(
            np.asarray(
                [
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                    [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                ]
            )
        )
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_phi_theta_wavefunction(
        self,
        esys=None,
        which=0,
        phi_grid=None,
        theta_grid=None,
        mode="abs",
        zero_calibrate=True,
        **kwargs
    ):
        """
        Plots 2D wave function in `phi`, `theta` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use self._default_theta_grid
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
        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(
            esys, phi_grid=phi_grid, theta_grid=theta_grid, which=which
        )

        wavefunc.gridspec = discretization.GridSpec(
            np.asarray(
                [
                    [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                ]
            )
        )
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(
                wavefunc.amplitudes.reshape(phi_grid.pt_count, theta_grid.pt_count)
            )
        )

        fig, axes = plot.wavefunction2d(
            wavefunc, zero_calibrate=zero_calibrate, **kwargs
        )
        axes.set_xlim([-2 * np.pi, 2 * np.pi])
        axes.set_ylim([-1 * np.pi, 3 * np.pi])
        axes.set_ylabel(r"$\phi$")
        axes.set_xlabel(r"$\theta$")
        axes.set_xticks([-np.pi, 0, np.pi])
        axes.set_xticklabels(["-$\pi$", "$0$", "$\pi$"])
        axes.set_yticks([0, np.pi, 2 * np.pi])
        axes.set_yticklabels(["0", "$\pi$", "$2\pi$"])
        return fig, axes
