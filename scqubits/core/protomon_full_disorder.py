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

import math
import os

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import scipy as sp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy import sparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.oscillator as osc
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.core.units as units
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.noise import NOISE_PARAMS, NoisySystem, calc_therm_ratio
from scqubits.core.storage import WaveFunctionOnGrid


# — Inductively-shunted Rhombus circuit ————————————————————————
class DisorderFullProtomon(base.QubitBaseClass, serializers.Serializable):
    r"""inductively-shunted Rhombus qubit, with the harmonic mode in the ground state

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        junction charging energy
    ECP: float
        parasitic capacitance energy
    EL: float
        inductive energy
    ELA: float
        additional inductive energy
    flux_c: float
        common part of the external flux, e.g., 1 corresponds to one flux quantum
    flux_d: float
        differential part of the external flux, e.g., 1 corresponds to one flux quantum
    """
    EJ = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    ECP = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    ELA = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    dC = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    dL = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    dJ = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    flux_c = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    flux_d = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    zeta_cut = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")

    def __init__(
            self,
            EJ: float,
            EC: float,
            ECP: float,
            EL: float,
            ELA: float,
            dC: float,
            dL: float,
            dJ: float,
            flux_c: float,
            flux_d: float,
    ) -> None:
        self.EJ = EJ
        self.EC = EC
        self.ECP = ECP
        self.EL = EL
        self.ELA = ELA
        self.dC = dC
        self.dL = dL
        self.dJ = dJ
        self.flux_c = flux_c
        self.flux_d = flux_d
        self.phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 90)
        self.theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.zeta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 110)
        self.zeta_cut = 10
        self.truncated_dim = 20
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 1.18248137e+01,
            "EC": 2.40141093e+00,
            "ECP": 4.1594940274629,
            "EL": 3.55481177e-01,
            "ELA": 0.3709,
            "dC": 0.0,
            "dL": 4.5e-3,
            "dJ": 0.0,
            "flux_c": 0.5,
            "flux_d": 0.,
        }

    @classmethod
    def create(cls) -> "DisorderFullProtomon":
        init_params = cls.default_params()
        protomon = cls(**init_params)
        protomon.widget()
        return protomon

    def dim_phi(self) -> int:
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi' degree of freedom."""
        return self.phi_grid.pt_count

    def dim_theta(self) -> int:
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return self.theta_grid.pt_count

    def dim_zeta(self) -> int:
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return self.zeta_cut

    def zeta_plasma(self) -> float:
        return np.sqrt(
            4
            * 4
            * self.ECP
            * (
                0.5 * self.ELA
                + 0.5 * self.EL / (1 - self.dL)
                + 0.5 * self.EL / (1 + self.dL)
            )
        )

    def zeta_osc(self) -> float:
        """
        Returns
        -------
        float
            Returns the oscillator strength of :math:`zeta' degree of freedom."""
        return (
            4
            * self.ECP
            / (
                0.5 * self.ELA
                + 0.5 * self.EL / (1 - self.dL)
                + 0.5 * self.EL / (1 + self.dL)
            )
        ) ** 0.25

    def hilbertdim(self) -> int:
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi() * self.dim_zeta() * self.dim_theta()

    def _zeta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`zeta' operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_zeta()
        return (
            (op.creation_sparse(dimension) + op.annihilation_sparse(dimension))
            * self.zeta_osc()
            / np.sqrt(2)
        )

    def zeta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in total Hilbert space
        """
        return self._kron3(
            self._identity_phi(), self._zeta_operator(), self._identity_theta()
        )

    def _n_zeta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator
        """
        dimension = self.dim_zeta()
        return (
            1j
            * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension))
            / (self.zeta_osc() * np.sqrt(2))
        )

    def n_zeta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi' operator in total Hilbert space
        """
        return self._kron3(
            self._identity_phi(), self._n_zeta_operator(), self._identity_theta()
        )

    def _phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`\phi` operator in the discretized basis
        """
        return sparse.dia_matrix(
            (self.phi_grid.make_linspace(), [0]), shape=(self.dim_phi(), self.dim_phi())
        ).tocsc()

    def phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in total Hilbert space
        """
        return self._kron3(
            self._phi_operator(), self._identity_zeta(), self._identity_theta()
        )

    def _n_phi_operator(self) -> dia_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator
        """
        return self.phi_grid.first_derivative_matrix(prefactor=-1j)

    def n_phi_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi' operator in total Hilbert space
        """
        return self._kron3(
            self._n_phi_operator(), self._identity_zeta(), self._identity_theta()
        )

    def _cos_phi_div_operator(self, div, off=0.0) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos (\\phi+off)/div` operator
        """
        cos_phi_div_vals = np.cos((self.phi_grid.make_linspace() + off) / div)
        return sparse.dia_matrix(
            (cos_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())
        ).tocsc()

    def _sin_phi_div_operator(self, div, off=0.0) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin (\\phi+off)/div` operator
        """
        sin_phi_div_vals = np.sin((self.phi_grid.make_linspace() + off) / div)
        return sparse.dia_matrix(
            (sin_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())
        ).tocsc()

    def _theta_operator(self) -> csc_matrix:
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

    def theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return self._kron3(
            self._identity_phi(), self._identity_zeta(), self._theta_operator()
        )

    def _n_theta_operator(self) -> dia_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator
        """
        return self.theta_grid.first_derivative_matrix(prefactor=-1j)

    def n_theta_operator(self) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_theta` in the total Hilbert space
        """
        return self._kron3(
            self._identity_phi(), self._identity_zeta(), self._n_theta_operator()
        )

    def _cos_theta_div_operator(self, div, off=0.0) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos (\\theta+off)/div` operator
        """
        cos_theta_div_vals = np.cos((self.theta_grid.make_linspace() + off) / div)
        return sparse.dia_matrix(
            (cos_theta_div_vals, [0]), shape=(self.dim_theta(), self.dim_theta())
        ).tocsc()

    def _sin_theta_div_operator(self, div, off=0.0) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin (\\theta+off)/div` operator
        """
        sin_theta_div_vals = np.sin((self.theta_grid.make_linspace() + off) / div)
        return sparse.dia_matrix(
            (sin_theta_div_vals, [0]), shape=(self.dim_theta(), self.dim_theta())
        ).tocsc()

    def _kron3(self, mat1, mat2, mat3) -> csc_matrix:
        """
        Returns
        -------
        ndarray
            Returns the kronecker product of two operators
        """
        return sparse.kron(sparse.kron(mat1, mat2, format="csc"), mat3, format="csc")

    def _identity_phi(self) -> csc_matrix:
        """
        Identity operator acting only on the :math:`\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_phi(), format="csc", dtype=np.complex_)

    def _identity_theta(self) -> csc_matrix:
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_theta(), format="csc", dtype=np.complex_)

    def _identity_zeta(self) -> csc_matrix:
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_zeta(), format="csc", dtype=np.complex_)

    def total_identity(self) -> csc_matrix:
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron3(
            self._identity_phi(), self._identity_zeta(), self._identity_theta()
        )

    def hamiltonian(self):
        zeta_osc = self._kron3(
            self._identity_phi(),
            op.number_sparse(self.dim_zeta(), self.zeta_plasma()),
            self._identity_theta(),
        )

        phi_kinetic = self.phi_grid.second_derivative_matrix(
            prefactor=-2.0 * self.EC / (1 - self.dC ** 2)
        )
        theta_kinetic = self.theta_grid.second_derivative_matrix(
            prefactor=-2.0 * self.EC / (1 - self.dC ** 2)
        )
        cross_kinetic = (
            4
            * self.dC
            * self.EC
            / (1 - self.dC ** 2)
            * self.n_phi_operator()
            * self.n_theta_operator()
        )
        tot_kinetic = (
            self._kron3(phi_kinetic, self._identity_zeta(), self._identity_theta())
            + self._kron3(self._identity_phi(), self._identity_zeta(), theta_kinetic)
            + cross_kinetic
        )

        diag_ind = (
            0.5
            * (self.EL / (1 - self.dL) + self.EL / (1 + self.dL))
            * (self.phi_operator() ** 2 + self.theta_operator() ** 2)
        )
        off_ind = (
            self.EL
            / (1 - self.dL)
            * (
                self.phi_operator() * self.theta_operator()
                - self.theta_operator() * self.zeta_operator()
                - self.phi_operator() * self.zeta_operator()
            )
        )
        off_ind += (
            self.EL
            / (1 + self.dL)
            * (
                -self.phi_operator() * self.theta_operator()
                - self.theta_operator() * self.zeta_operator()
                + self.phi_operator() * self.zeta_operator()
            )
        )
        total_ind = diag_ind + off_ind

        junction = (
            -2
            * self.EJ
            * self._kron3(
                self._cos_phi_div_operator(1.0, 2 * np.pi * self.flux_c),
                self._identity_zeta(),
                self._cos_theta_div_operator(1.0, 2 * np.pi * self.flux_d),
            )
            - self.dJ
            * 2
            * self.EJ
            * self._kron3(
                self._sin_phi_div_operator(1.0, 2 * np.pi * self.flux_c),
                self._identity_zeta(),
                self._sin_theta_div_operator(1.0, 2 * np.pi * self.flux_d),
            )
            + 2 * self.EJ * (1 + np.abs(self.dJ)) * self.total_identity()
        )

        return zeta_osc + tot_kinetic + total_ind + junction

    def _evals_calc(self, evals_count) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(
            hamiltonian_mat,
            k=evals_count,
            return_eigenvectors=False,
            sigma=0.0,
            which="LM",
            v0=settings.RANDOM_ARRAY[: self.hilbertdim()],
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(
            hamiltonian_mat,
            k=evals_count,
            return_eigenvectors=True,
            sigma=0.0,
            which="LM",
            v0=settings.RANDOM_ARRAY[: self.hilbertdim()],
        )
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def wavefunction(
        self, esys=None, which=0, phi_grid=None, zeta_grid=None, theta_grid=None
    ):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_grid = phi_grid or self.phi_grid
        zeta_grid = zeta_grid or self.zeta_grid
        theta_grid = theta_grid or self.theta_grid

        state_amplitudes = evecs[:, which].reshape(
            self.dim_phi(), self.dim_zeta(), self.dim_theta()
        )

        zeta_osc_amplitudes = np.zeros(
            (self.dim_zeta(), zeta_grid.pt_count), dtype=np.complex_
        )
        for i in range(self.dim_zeta()):
            zeta_osc_amplitudes[i, :] = osc.harm_osc_wavefunction(
                i, zeta_grid.make_linspace(), self.zeta_osc()
            )

        wavefunc_amplitudes = np.swapaxes(
            np.tensordot(zeta_osc_amplitudes, state_amplitudes, axes=([0], [1])), 0, 1
        )
        wavefunc_amplitudes = spec_utils.standardize_phases(wavefunc_amplitudes)

        grid3d = discretization.GridSpec(
            np.asarray(
                [
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                    [zeta_grid.min_val, zeta_grid.max_val, zeta_grid.pt_count],
                    [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                ]
            )
        )
        return storage.WaveFunctionOnGrid(grid3d, wavefunc_amplitudes)

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
        Plots 2D phase-basis wave function at zeta = 0

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
        zeta_grid = discretization.Grid1d(0, 0, 1)
        theta_grid = theta_grid or self.theta_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(
            esys,
            phi_grid=phi_grid,
            zeta_grid=zeta_grid,
            theta_grid=theta_grid,
            which=which,
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
        # axes.set_xlim([-2 * np.pi, 2 * np.pi])
        # axes.set_ylim([-1 * np.pi, 3 * np.pi])
        axes.set_ylabel(r"$\phi$")
        axes.set_xlabel(r"$\theta$")
        axes.set_xticks([-np.pi, 0, np.pi])
        axes.set_xticklabels(["-$\pi$", "$0$", "$\pi$"])
        axes.set_yticks([0, np.pi, 2 * np.pi])
        axes.set_yticklabels(["0", "$\pi$", "$2\pi$"])

        return fig, axes
