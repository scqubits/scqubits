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
class DisorderProtomon(base.QubitBaseClass, serializers.Serializable):
    r"""inductively-shunted Rhombus qubit, with the harmonic mode in the ground state

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        junction charging energy
    EL: float
        inductive energy
    ELA: float
        additional inductive energy
    flux_c: float
        common part of the external flux, e.g., 1 corresponds to one flux quantum
    flux_d: float
        differential part of the external flux, e.g., 1 corresponds to one flux quantum
    kbt: float
        photon temperature
    """

    def __init__(self, EJ, EC, EL, ELA, dC, dJ, dL, flux_c, flux_d, kbt):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.ELA = ELA
        self.dC = dC
        self.dJ = dJ
        self.dL = dL
        self.flux_c = flux_c
        self.flux_d = flux_d
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # input temperature unit mK
        self.phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
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
        EC_dis = self.EC / (1 - self.dC ** 2)
        EL_dis = self.EL / (1 - self.dL ** 2)

        phi_kinetic = self.phi_grid.second_derivative_matrix(prefactor=-2.0 * EC_dis)
        theta_kinetic = self.theta_grid.second_derivative_matrix(
            prefactor=-2.0 * EC_dis
        )
        tot_kinetic = self._kron2(phi_kinetic, self._identity_theta()) + self._kron2(
            self._identity_phi(), theta_kinetic
        )

        phi_ind = (
                EL_dis
                * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux_c)
                ** 2
        )
        theta_ind = (
                EL_dis
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

        res = tot_kinetic + phi_ind + theta_ind + phi_theta_junction

        # below is due to disorder and SWT
        flux_c_abbr = self.total_identity() * 2 * np.pi * self.flux_c
        flux_d_abbr = self.total_identity() * 2 * np.pi * self.flux_d

        res += - EL_dis ** 2 / (EL_dis + 0.5 * self.ELA) * (self.theta_operator() -
                                                          flux_d_abbr + self.dL * (
                                                              self.phi_operator() -
                                                              flux_c_abbr)) ** 2

        res += self.dC * 4 * EC_dis * self.n_phi_operator() * self.n_theta_operator()

        res += self.dL * 2 * EL_dis * (self.phi_operator() - flux_c_abbr) * (
            self.theta_operator() -flux_d_abbr)

        res += - self.dJ * 2 * self.EJ * self._kron2(self._sin_phi_div_operator(1.0),
                                                     self._sin_theta_div_operator(
                                                         1.0))  + self.dJ * 2 * \
                                                                 self.EJ * \
               self.total_identity()

        return res

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
