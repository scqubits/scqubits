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

import cmath
import math
import os

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import expm, eigsh

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.harmonic_osc as osc
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers


# —Double Cooper pair tunneling qubit ————————————————————————
class Dcp(base.QubitBaseClass, serializers.Serializable):
    r"""Class for the double Cooper pair tunneling qubit. Hamiltonian
    :math:`H_\text{dcp}=4E_\text{C}[2n_\phi^2+\frac{1}{2}(n_\varphi-N_\text{g}-n_\theta)^2+xn_\theta^2]+E_\text{L}(\frac{1}{4}\phi^2+\theta^2)-2E_\text{J}\cos(\varphi)\cos(\frac{\phi}{2}+\frac{\varphi_\text{ext}}{2})`
    is represented in dense form. The employed basis are harmonic basis for :math:`\phi,\theta` and charge basis for :math:`\varphi`. The cosine term in the
    potential is handled via matrix exponentiation. Initialize with, for example::

        qubit = Dcp(EJ=15.0, EC=2.0, EL=1.0, x=0.02, flux=0.5, Ng=0, N0=7, q0=30, p0=7)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    EL: float
        inductive energy
    x: float
        ratio of the junction capacitance to the shunt capacitance
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
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    Ng = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    N0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    q0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    p0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, EL, x, flux, Ng, N0, q0, p0, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.x = x
        self.flux = flux
        self.Ng = Ng
        self.N0 = N0
        self.q0 = q0
        self.p0 = p0
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        # TODO change the picture here
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/fluxonium.png')

    @staticmethod
    def default_params():
        return {
            'EJ': 15.0,
            'EC': 2.0,
            'EL': 1.0,
            'x': 0.02,
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

    def phi_hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension for phi degree of freedom."""
        return self.p0

    def theta_hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension for phi degree of freedom."""
        return self.q0

    def varphi_hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension for varphi degree of freedom."""
        return 2 * self.N0 + 1

    def phi_zpf(self):
        """
        Returns
        -------
        float
            Returns zero point fluctuation for the phi degree of freedom.
        """
        return (8.0 * self.EC / self.EL) ** 0.25

    def n_theta_zpf(self):
        """
        Returns
        -------
        float
            Returns zero point fluctuation for the n_theta degree of freedom.
        """
        return 0.5 * (self.EL / self.EC / self.x) ** 0.25

    def phi_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the phi degree of freedom.
        """
        return math.sqrt(8.0 * self.EL * self.EC)

    def theta_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the theta degree of freedom.
        """
        return math.sqrt(16.0 * self.x * self.EL * self.EC)

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.phi_hilbertdim()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.phi_zpf()

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator in the LC harmonic oscillator basis
        """
        dimension = self.theta_hilbertdim()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) * self.n_theta_zpf()

    def exp_i_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\phi/2}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self.phi_operator() * 0.5
        return expm(exponent)

    def cos_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/2` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self.exp_i_phi_2_operator()
        cos_phi_op += cos_phi_op.conjugate().T
        return np.real(cos_phi_op)

    def sin_phi_2_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/2` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self.exp_i_phi_2_operator()
        sin_phi_op += sin_phi_op.conjugate().T
        return np.real(sin_phi_op)

    def n_varphi_ng_operator(self):
        """Returns charge operator `n_phi - Ng` in the charge basis"""
        diag_elements = np.arange(-self.N0 - self.Ng, self.N0 + 1 - self.Ng)
        return sparse.dia_matrix((diag_elements, [0]), shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()

    def cos_varphi_operator(self):
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [1]), shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        cos_op += 0.5 * sparse.dia_matrix((np.ones(self.varphi_hilbertdim()), [-1]), shape=(self.varphi_hilbertdim(), self.varphi_hilbertdim())).tocsc()
        return cos_op

    def phi_identity(self):
        dimension = self.phi_hilbertdim()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def theta_identity(self):
        dimension = self.theta_hilbertdim()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def varphi_identity(self):
        dimension = self.varphi_hilbertdim()
        return sparse.identity(dimension, format='csc', dtype=np.complex_)

    def kron3(self, mat1, mat2, mat3):
        return sparse.kron(sparse.kron(mat1, mat2, format='csc'), mat3, format='csc')

    def hamiltonian(self):  # follow W.C. Smith, A. Kou, X. Xiao, U. Vool, and M.H. Devoret, Npj Quantum Inf. 6, 8 (2020).
        """Return Hamiltonian

        Returns
        -------
        ndarray
        """
        phi_osc_matrix = self.kron3(op.number_sparse(self.phi_hilbertdim(), self.phi_plasma()), self.theta_identity(), self.varphi_identity())
        theta_osc_matrix = self.kron3(self.phi_identity(), op.number_sparse(self.theta_hilbertdim(), self.theta_plasma()), self.varphi_identity())

        n_varphi_ng_matrix = self.kron3(self.phi_identity(), self.theta_identity(), self.n_varphi_ng_operator())
        n_theta_matrix = self.kron3(self.phi_identity(), self.n_theta_operator(), self.varphi_identity())
        cross_kinetic_matrix = 2 * self.EC * (n_varphi_ng_matrix - n_theta_matrix) * (n_varphi_ng_matrix - n_theta_matrix)

        phi_flux_term = self.cos_phi_2_operator() * np.cos(self.flux * np.pi) - self.sin_phi_2_operator() * np.sin(self.flux * np.pi)
        junction_matrix = -2 * self.EJ * self.kron3(phi_flux_term, self.theta_identity(), self.cos_varphi_operator())

        hamiltonian_mat = phi_osc_matrix + theta_osc_matrix + cross_kinetic_matrix + junction_matrix
        return hamiltonian_mat

    def potential(self, phi, theta, varphi):
        """Double Cooper pair tunneling qubit potential evaluated at `phi, theta, varphi`.

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`
        theta: float or ndarray
            float value of the phase variable `theta`
        varphi: float or ndarray
            float value of the phase variable `varphi`

        Returns
        -------
        float or ndarray
        """
        return self.EL * (0.25 * phi * phi + theta * theta) - 2 * self.EJ * np.cos(varphi) * np.cos(phi * 0.5 + np.pi * self.flux)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    # TODO not use it
    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return 0
