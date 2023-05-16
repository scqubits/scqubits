# snail.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
import itertools
import math
import os

from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import scipy as sp
from ninatool.internal.elements import J as J_nina
from ninatool.internal.structures import loop as loop_nina

from numpy import ndarray
from scipy.linalg import eigh, inv, cosm

import scqubits.core.descriptors as descriptors
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
from scqubits.core.operators import identity_wrap_array, annihilation
import scqubits.io_utils.fileio_serializers as serializers
from scqubits import Grid1d
from scqubits.core import discretization
from scqubits.core.storage import WaveFunction

import qutip as qt
from scqubits.core.noise import NoisySystem


class SNAIL(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the SNAIL. Hamiltonian :math:`H_\text{sn}=-4E_\text{
    C}\partial_\phi^2-\alpha E_\text{J}\cos(\phi) -n E_\text{J}\cos([\phi - \varphi_\text{ext}] / n)`
    is represented using the basis of exponential functions periodic from :math:`- n\\pi->n\\pi`.
    Initialize with, for example::

        snail = SNAIL(EJ=47.0, EC=0.1, flux=0.2, alpha=0.29, ncut=50, n=3)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    flux: float
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    alpha : float
        multiplicative factor representing ratio of small to large junction energies
    n : int
        number of large junctions
    ncut: int
        number of harm. osc. basis states used in diagonalization
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    alpha = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    n = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
            self,
            EJ: float,
            EC: float,
            flux: float,
            alpha: float,
            n: int,
            ncut: int,
            truncated_dim: int = 6,
            id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.EJ = EJ
        self.EC = EC
        self.flux = flux
        self.alpha = alpha
        self.n = n
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(
            -n * np.pi, n * np.pi, 151 + (n - 1) * 151
        )
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/snail.jpg"
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 8.9,
            "EC": 5.0,
            "flux": 0.41,
            "alpha": 0.29,
            "n": 3,
            "ncut": 110,
            "truncated_dim": 10,
        }

    def n_operator(self) -> ndarray:
        """Returns charge operator `n` in the charge basis"""
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1) / self.n
        return np.diag(diag_elements)

    def exp_i_phi_operator(self) -> ndarray:
        """Returns operator :math:`e^{i\\varphi}` in the charge basis"""
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - self.n)
        exp_op = np.diag(entries, -self.n)
        return exp_op

    def cos_phi_operator(self) -> ndarray:
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_operator()
        cos_op += cos_op.T
        return cos_op

    def sin_phi_operator(self) -> ndarray:
        """Returns operator :math:`\\sin \\varphi` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_operator()
        sin_op += sin_op.conjugate().T
        return sin_op

    def hamiltonian(self):
        dim = self.hilbertdim()
        hamiltonian_mat = np.zeros((dim, dim), dtype=complex)
        hamiltonian_mat += np.diag(
            [4.0 * self.EC * (ind - self.ncut) ** 2 / self.n ** 2 for ind in range(dim)]
        )
        ind_n = np.arange(dim - self.n)
        hamiltonian_mat[ind_n, ind_n + self.n] = -self.alpha * self.EJ / 2.0
        hamiltonian_mat[ind_n + self.n, ind_n] = -self.alpha * self.EJ / 2.0
        ind_1 = np.arange(dim - 1)
        hamiltonian_mat[ind_1, ind_1 + 1] = (
                -self.n * self.EJ * np.exp(1j * 2.0 * np.pi * self.flux / self.n) / 2.0
        )
        hamiltonian_mat[ind_1 + 1, ind_1] = (
                -self.n * self.EJ * np.exp(-1j * 2.0 * np.pi * self.flux / self.n) / 2.0
        )
        return hamiltonian_mat

    def hilbertdim(self) -> int:
        """
        Returns
        -------
            Returns the Hilbert space dimension."""
        return 2 * self.ncut + 1

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """SNAIL potential evaluated at `phi`.

        Parameters
        ----------
            float value of the phase variable `phi`

        Returns
        -------
        float or ndarray
        """
        return -self.alpha * self.EJ * np.cos(phi) - self.n * self.EJ * np.cos(
            (phi - 2.0 * np.pi * self.flux) / self.n
        )

    def numberbasis_wavefunction(
            self, esys: Tuple[ndarray, ndarray] = None, which: int = 0
    ) -> WaveFunction:
        """Return the snail wave function in number basis. The specific index of the
        wave function to be returned is `which`.

        Parameters
        ----------
        esys:
            if `None`, the eigensystem is calculated on the fly; otherwise, the provided
            eigenvalue, eigenvector arrays as obtained from `.eigensystem()`,
            are used (default value = None)
        which:
            eigenfunction index (default value = 0)
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count=evals_count)
        evals, evecs = esys

        n_vals = np.arange(-self.ncut, self.ncut + 1)
        return storage.WaveFunction(n_vals, evecs[:, which], evals[which])

    def wavefunction(
            self,
            esys: Optional[Tuple[ndarray, ndarray]] = None,
            which: int = 0,
            phi_grid: Grid1d = None,
    ) -> WaveFunction:
        """Return the snail wave function in phase basis. The specific index of the
        wavefunction is `which`. `esys` can be provided, but if set to `None` then it is
        calculated on the fly.

        Parameters
        ----------
        esys:
            if None, the eigensystem is calculated on the fly; otherwise, the provided
            eigenvalue, eigenvector arrays as obtained from `.eigensystem()` are used
        which:
            eigenfunction index (default value = 0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count=evals_count)
        else:
            evals, evecs = esys

        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        phi_grid = phi_grid or self._default_grid
        phi_basis_labels = phi_grid.make_linspace()
        phi_wavefunc_amplitudes = np.empty(phi_grid.pt_count, dtype=np.complex_)
        for k in range(phi_grid.pt_count):
            phi_wavefunc_amplitudes[k] = (
                                                 1j ** which / math.sqrt(2 * np.pi * self.n)
                                         ) * np.sum(
                n_wavefunc.amplitudes
                * np.exp(1j * phi_basis_labels[k] * n_wavefunc.basis_labels / self.n)
            )
        return storage.WaveFunction(
            basis_labels=phi_basis_labels,
            amplitudes=phi_wavefunc_amplitudes,
            energy=evals[which],
        )


class SNAILThreeMode(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the three mode SNAIL.

    Parameters
    ----------
    EJ0: float
        Josephson energy of the lone JJ on the left branch
    EJ1: float
        Josephson energy of the top JJ on the right branch
    EJ2: float
        Josephson energy of the middle JJ on the right branch
    EJ3: float
        Josephson energy of the bottom JJ on the right branch
    EC0: float
        Charging energy of the lone JJ on the left branch (includes shunt capacitance)
    EC1: float
        Charging energy of the top JJ on the right branch
    EC2: float
        Charging energy of the middle JJ on the right branch
    EC3: float
        Charging energy of the bottom JJ on the right branch
    flux: float
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    cutoff: int
        number of basis states to keep in the harmonic basis for each normal mode
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """

    def __init__(
            self,
            EJ0: float,
            EJ1: float,
            EJ2: float,
            EJ3: float,
            EC0: float,
            EC1: float,
            EC2: float,
            EC3: float,
            flux: float,
            cutoff: int = 5,
            truncated_dim: int = 6,
            id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.EJ0 = EJ0
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ3 = EJ3
        self.EC0 = EC0
        self.EC1 = EC1
        self.EC2 = EC2
        self.EC3 = EC3
        self.flux = flux
        self.cutoff = cutoff
        self.truncated_dim = truncated_dim

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ0": 1.0,
            "EJ1": 1.0,
            "EJ2": 0.1,
            "EJ3": 1.0,
            "EC0": 0.1,
            "EC1": 0.1,
            "EC2": 1.0,
            "EC3": 1.0,
            "flux": 0.0,
            "cutoff": 5
        }

    def find_minimum(self):
        """ Find the minimum (equilibrium) position of the node variables.
        Sandro's tool NINA does this computation automatically, and returns the
        equilibrium junction phases. We want the equilibirum node-variable phases,
        and the conversion is done below. Additionally, NINA solves for these positions
        for many values of flux in a dense array. We isolate below the flux value of interest. """
        order = 3
        # LEFT BRANCH ELEMENTS
        J0 = J_nina(ic=self.EJ0, order=order, name='J0')
        left_elements = [J0]
        J1 = J_nina(ic=self.EJ1, order=order, name='J1')
        J2 = J_nina(ic=self.EJ2, order=order, name='J2')
        J3 = J_nina(ic=self.EJ3, order=order, name='J3')
        right_elements = [J1, J2, J3]
        snail = loop_nina(left_branch=left_elements,
                          right_branch=right_elements,
                          stray_inductance=False,
                          name='SNAIL')
        snail.interpolate_results(2.0 * np.pi * self.flux)
        node_vars_to_phase_vars = np.array([[1.0, 0.0, 0.0],
                                            [-1.0, 1.0, 0.0],
                                            [0.0, -1.0, 1.0]])
        junc_phase_array = np.array([snail.elements[elem_idx].phi[0] for elem_idx in range(0, 3)])
        node_phase_array = sp.linalg.inv(node_vars_to_phase_vars) @ (junc_phase_array - 2.0 * np.pi * self.flux *
                                                                     np.array([1, 0, 0]))
        return node_phase_array

    def capacitance_matrix(self):
        """Find the capacitance matrix. We use units where e=1 (the e's cancel out anyways, once we invert
        to obtain the charging energy matrix when calculating the Hamiltonian)"""
        C0 = 1 / (2 * self.EC0)
        C1 = 1 / (2 * self.EC1)
        C2 = 1 / (2 * self.EC2)
        C3 = 1 / (2 * self.EC3)
        Cmat = np.array([[C0 + C1, -C1, 0.0],
                         [-C1, C1 + C2, -C2],
                         [0.0, -C2, C2 + C3]])
        return Cmat

    def gamma_matrix(self):
        """Find the inverse inductance matrix. This matrix is necessary for
         solving for the modes of the SNAIL. See D. K. Weiss et al. PRR (2021) for more information"""
        Phi0 = 0.5  # units where e, hbar = 1; Phi0 = hbar / (2 * e)
        nina_min = self.find_minimum()
        g00 = self.EJ0 * np.cos(nina_min[0] + 2.0 * np.pi * self.flux) + self.EJ1 * np.cos(nina_min[1] - nina_min[0])
        g01 = self.EJ1 * np.cos(nina_min[1] - nina_min[0])
        g11 = self.EJ1 * np.cos(nina_min[1] - nina_min[0]) + self.EJ2 * np.cos(nina_min[2] - nina_min[1])
        g12 = self.EJ2 * np.cos(nina_min[2] - nina_min[1])
        g22 = self.EJ2 * np.cos(nina_min[2] - nina_min[1]) + self.EJ3 * np.cos(nina_min[2])
        gamma_mat = np.array([[g00, -g01, 0.0],
                              [-g01, g11, -g12],
                              [0.0, -g12, g22]]) / Phi0 ** 2
        return gamma_mat

    def eigensystem_normal_modes(self) -> (ndarray, ndarray):
        """Returns squared normal mode frequencies, matrix of eigenvectors"""
        omega_squared, normal_mode_eigenvectors = eigh(
            self.gamma_matrix(), b=self.capacitance_matrix()
        )
        return omega_squared, normal_mode_eigenvectors

    def Xi_matrix(self) -> ndarray:
        """Returns Xi matrix of the normal-mode eigenvectors normalized
        according to \Xi^T C \Xi = \Omega^{-1}/Z0, or equivalently \Xi^T
        \Gamma \Xi = \Omega/Z0. The \Xi matrix
        simultaneously diagonalizes the capacitance and effective
        inductance matrices by a congruence transformation.
        """
        omega_squared_array, eigenvectors = self.eigensystem_normal_modes()
        Z0 = 0.25  # units where e and hbar = 1; Z0 = hbar / (2 * e)**2
        Xi_mat = np.array(
            [
                eigenvectors[:, i]
                * omega_squared ** (-1 / 4)
                * np.sqrt(1.0 / Z0)
                for i, omega_squared in enumerate(omega_squared_array)
            ]
        ).T
        assert np.allclose(Xi_mat.T @ self.capacitance_matrix() @ Xi_mat,
                           np.diag(1./np.sqrt(omega_squared_array)) / Z0)
        assert np.allclose(Xi_mat.T @ self.gamma_matrix() @ Xi_mat,
                           np.diag(np.sqrt(omega_squared_array)) / Z0)
        return Xi_mat

    def hamiltonian(self):
        """Find the Hamiltonian"""
        phi0 = self.phi_operator(0)
        phi1 = self.phi_operator(1)
        phi2 = self.phi_operator(2)
        EC_mat = 0.5 * inv(self.capacitance_matrix())
        H = 0.0 * self.identity_operator()
        for j, k in itertools.product(range(3), range(3)):
            H += 4 * EC_mat[j, k] * self.n_operator(j) @ self.n_operator(k)
        H += -self.EJ0 * cosm(phi0 + 2.0 * np.pi * self.flux * self.identity_operator())
        H += -self.EJ1 * cosm(phi1 - phi0)
        H += -self.EJ2 * cosm(phi2 - phi1)
        H += -self.EJ3 * cosm(phi2)
        return H

    def hilbertdim(self) -> int:
        return self.cutoff**3

    def potential(self, phi: Union[float, ndarray]) -> Union[float, ndarray]:
        """Calculate the value of the classical potential"""
        return (-self.EJ0 * np.cos(phi[0] + 2.0 * np.pi * self.flux)
                - self.EJ1 * np.cos(phi[1] - phi[0])
                - self.EJ2 * np.cos(phi[2] - phi[1])
                - self.EJ3 * np.cos(phi[2])
                )

    def decompose_evec_in_bare_basis(self, evec):
        bare_labels = list(itertools.product(range(self.cutoff), repeat=3))
        ovlp_list = np.zeros(self.hilbertdim())
        for idx, (i, j, k) in enumerate(bare_labels):
            bare_evec = qt.tensor(qt.basis(self.cutoff, i), qt.basis(self.cutoff, j), qt.basis(self.cutoff, k))
            ovlp_list[idx] = np.abs(evec @ bare_evec.data.toarray())**2
        sorted_idxs = np.argsort(1./ovlp_list)
        return ovlp_list[sorted_idxs], np.array(bare_labels)[sorted_idxs]

    def _identity_operator(self) -> ndarray:
        return np.identity(self.cutoff, dtype=complex)

    def _identity_operator_list(self) -> List[ndarray]:
        return [self._identity_operator() for _ in range(3)]

    def identity_operator(self) -> ndarray:
        """Returns the identity operator in the full Hilbert space
        """
        return identity_wrap_array([], [], self._identity_operator_list())

    def a_operator(self, dof_index):
        """Returns the annihilation operator for each normal mode"""
        dim = self.cutoff
        a_op_wrapped = identity_wrap_array([annihilation(dim)], [dof_index], self._identity_operator_list())
        return a_op_wrapped

    def phi_operator(self, node_idx):
        """Returns the phi operator for each node variable"""
        Xi_matrix = self.Xi_matrix()
        return (1 / np.sqrt(2)) * np.sum([Xi_matrix[node_idx, dof_idx]
                                          * (self.a_operator(dof_idx) + self.a_operator(dof_idx).T)
                                          for dof_idx in range(3)], axis=0, dtype=complex)

    def n_operator(self, node_idx):
        """Returns the charge number operator for each node variable"""
        Xi_inv_T = inv(self.Xi_matrix()).T
        return (-1j / np.sqrt(2)) * np.sum([Xi_inv_T[node_idx, dof_idx]
                                            * (self.a_operator(dof_idx) - self.a_operator(dof_idx).T)
                                            for dof_idx in range(3)], axis=0, dtype=complex)

    def wavefunction(
        self,
        esys: Optional[Tuple[ndarray, ndarray]],
        which: int = 0,
        phi_grid: Grid1d = None,
    ) -> "WaveFunction":
        raise NotImplementedError("wavefunction method not implemented for the 3D SNAIL")
