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

import cmath
import math
import os

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse.csc import csc_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem



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
            "EJ": 47.0,
            "EC": 0.1,
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


# n in Ec goes over all integers, phi_ext is lumped with the large junctions

    # def hamiltonian(self):
    #     dim = self.hilbertdim()
    #     hamiltonian_mat = np.diag(
    #         [4.0 * self.EC * (ind - self.ncut) ** 2 / self.n ** 2 for ind in range(dim)]
    #     )
    #     ind_n = np.arange(dim - self.n)
    #     hamiltonian_mat[ind_n, ind_n + self.n] = -self.alpha * self.EJ / 2.0
    #     hamiltonian_mat[ind_n + self.n, ind_n] = -self.alpha * self.EJ / 2.0
    #     ind_1 = np.arange(dim - 1)
    #     hamiltonian_mat[ind_1, ind_1 + 1] = (
    #         -self.n * self.EJ * np.exp(1j * 2.0 * np.pi * self.flux/self.n) / 2.0
    #     )
    #     hamiltonian_mat[ind_1 + 1, ind_1] = (
    #         -self.n * self.EJ * np.exp(-1j * 2.0 * np.pi * self.flux/self.n) / 2.0
    #     )
    #
    #     return hamiltonian_mat

    # n in Ec goes over all integers, phi_ext is lumped with the small junction

    def hamiltonian(self):
        dim = self.hilbertdim()
        hamiltonian_mat = np.diag(
            [4.0 * self.EC * (ind - self.ncut) ** 2 / self.n ** 2 for ind in range(dim)]
        )
        ind_n = np.arange(dim - self.n)
        hamiltonian_mat[ind_n, ind_n + self.n] = -self.alpha * self.EJ / 2.0 * np.exp(1j * 2.0 * np.pi * self.flux)
        hamiltonian_mat[ind_n + self.n, ind_n] = -self.alpha * self.EJ / 2.0 * np.exp(-1j * 2.0 * np.pi * self.flux)
        ind_1 = np.arange(dim - 1)
        hamiltonian_mat[ind_1, ind_1 + 1] = (
            -self.n * self.EJ / 2.0
        )
        hamiltonian_mat[ind_1 + 1, ind_1] = (
            -self.n * self.EJ / 2.0
        )
        return hamiltonian_mat

    # n in Ec goes over all multiples of n, phi_ext is lumped with the large junction

    # def hamiltonian(self):
    #     dim = self.hilbertdim()
    #     hamiltonian_mat = np.diag(
    #         [(4.0 * self.EC * (ind - self.ncut) ** 2 / self.n ** 2) * (ind%self.n == 0) for ind in range(dim)]
    #     )
    #     ind_n = np.arange(dim - self.n)
    #     hamiltonian_mat[ind_n, ind_n + self.n] = -self.alpha * self.EJ / 2.0
    #     hamiltonian_mat[ind_n + self.n, ind_n] = -self.alpha * self.EJ / 2.0
    #     ind_1 = np.arange(dim - 1)
    #     hamiltonian_mat[ind_1, ind_1 + 1] = (
    #         -self.n * self.EJ * np.exp(1j * 2.0 * np.pi * self.flux/self.n) / 2.0
    #     )
    #     hamiltonian_mat[ind_1 + 1, ind_1] = (
    #         -self.n * self.EJ * np.exp(-1j * 2.0 * np.pi * self.flux/self.n) / 2.0
    #     )
    #
    #     return hamiltonian_mat

    # n in Ec goes over all multiples of n, phi_ext is lumped with the small junction

    # def hamiltonian(self):
    #     dim = self.hilbertdim()
    #     hamiltonian_mat = np.diag(
    #         [4.0 * self.EC * (ind - self.ncut) ** 2 / self.n ** 2 * (ind%self.n == 0) for ind in range(dim)]
    #     )
    #     ind_n = np.arange(dim - self.n)
    #     hamiltonian_mat[ind_n, ind_n + self.n] = -self.alpha * self.EJ / 2.0 * np.exp(1j * 2.0 * np.pi * self.flux)
    #     hamiltonian_mat[ind_n + self.n, ind_n] = -self.alpha * self.EJ / 2.0 * np.exp(-1j * 2.0 * np.pi * self.flux)
    #     ind_1 = np.arange(dim - 1)
    #     hamiltonian_mat[ind_1, ind_1 + 1] = (
    #         -self.n * self.EJ / 2.0
    #     )
    #     hamiltonian_mat[ind_1 + 1, ind_1] = (
    #         -self.n * self.EJ / 2.0
    #     )
    #     return hamiltonian_mat

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
    ) -> storage.WaveFunction:
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
    ) -> storage.WaveFunction:
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
