# flux_qubit.py
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

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy import sparse

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.core.diag as diag
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils


# -Dimon, both degrees of freedom in charge basis---------------------------------


class Dimon(base.QubitBaseClass, serializers.Serializable):
    r"""Dimon

    | [1] Hazra et al., arXiv:2407.10934 (2024).
          https://arxiv.org/abs/2407.10934

    The dimon device as described in [1], where the junctions are taken to be
    symmetric for simplicity.

    Parameters
    ----------
    EJ1, EJ2: float
        Josephson energies of the individual junctions
    ECJ1, ECJ2: float
        charging energies associated with each junction
    ECs: float
        charging energy associated with the shunt capacitance
    ng1, ng2: float, float
        offset charges associated with the two nodes
    ncut: int
        charge number cutoff for the charge on both islands `n`,  `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    esys_method:
        method for esys diagonalization, callable or string representation
    esys_method_options:
        dictionary with esys diagonalization options
    evals_method:
        method for evals diagonalization, callable or string representation
    evals_method_options:
        dictionary with evals diagonalization options
    """

    EJ1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EJ2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECs = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng1 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng2 = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ1: float,
        EJ2: float,
        ECJ1: float,
        ECJ2: float,
        ECs: float,
        ng1: float,
        ng2: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.ECJ1 = ECJ1
        self.ECJ2 = ECJ2
        self.ECs = ECs
        self.ng1 = ng1
        self.ng2 = ng2
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(
            -np.pi / 2, 3 * np.pi / 2, 100
        )  # for plotting in phi_j basis

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ1": 1.0,
            "EJ2": 1.0,
            "ECJ1": 0.016,
            "ECJ2": 0.016,
            "ECs": 0.16,
            "ng1": 0.0,
            "ng2": 0.0,
            "ncut": 10,
            "truncated_dim": 10,
        }

    def EC_matrix(self) -> ndarray:
        """Return the charging energy matrix"""
        Cmat = np.zeros((2, 2))
        CJ1 = 1.0 / (2 * self.ECJ1)  # capacitances in units where e is set to 1
        CJ2 = 1.0 / (2 * self.ECJ2)
        Cs = 1.0 / (2 * self.ECs)

        Cmat[0, 0] = CJ1 + Cs
        Cmat[1, 1] = CJ2 + Cs
        Cmat[0, 1] = -Cs
        Cmat[1, 0] = -Cs

        return np.linalg.inv(Cmat) / 2.0

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.ncut + 1) ** 2

    def potential(self, phi1: ndarray, phi2: ndarray) -> ndarray:
        """Return value of the potential energy at phi1 and phi2, disregarding
        constants."""
        return -self.EJ1 * np.cos(phi1) - self.EJ2 * np.cos(phi2)

    def kineticmat(self) -> ndarray:
        """Return the kinetic energy matrix."""
        ECmat = self.EC_matrix()

        kinetic_mat = (
            4.0
            * ECmat[0, 0]
            * sparse.kron(
                (self._n_operator() - self.ng1 * self._identity())
                @ (self._n_operator() - self.ng1 * self._identity()),
                self._identity(),
            )
        )
        kinetic_mat += (
            4.0
            * ECmat[1, 1]
            * sparse.kron(
                self._identity(),
                (self._n_operator() - self.ng2 * self._identity())
                @ (self._n_operator() - self.ng2 * self._identity()),
            )
        )
        kinetic_mat += (
            4.0
            * (ECmat[0, 1] + ECmat[1, 0])
            * sparse.kron(
                self._n_operator() - self.ng1 * self._identity(),
                self._n_operator() - self.ng2 * self._identity(),
            )
        )
        return kinetic_mat

    def potentialmat(self) -> ndarray:
        """Return the potential energy matrix for the potential."""
        potential_mat = (
            -0.5
            * self.EJ1
            * sparse.kron(
                self._exp_i_phi_operator() + self._exp_i_phi_operator().T,
                self._identity(),
            )
        )
        potential_mat += (
            -0.5
            * self.EJ2
            * sparse.kron(
                self._identity(),
                self._exp_i_phi_operator() + self._exp_i_phi_operator().T,
            )
        )
        return potential_mat

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Return Hamiltonian in the basis obtained by employing charge basis for both
        degrees of freedom or in the eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the basis obtained by employing charge basis for both degrees of freedom.
            If `True`, the energy eigenspectrum is computed, returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns Hamiltonian in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, the Hamiltonian has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, Hamiltonian has dimensions of m x m,
            for m given eigenvectors.
        """
        hamiltonian_mat = self.kineticmat() + self.potentialmat()
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )

    def _n_operator(self) -> ndarray:
        diag_elements = np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_)
        return sparse.diags_array(diag_elements)

    def _exp_i_phi_operator(self, harmonic=1) -> ndarray:
        dim = 2 * self.ncut + 1
        off_diag_elements = np.ones(dim - 1 - (harmonic - 1), dtype=np.complex_)
        e_iphi_matrix = sparse.diags_array(
            off_diag_elements, offsets=-1 - (harmonic - 1)
        )
        return e_iphi_matrix

    def _identity(self) -> ndarray:
        dim = 2 * self.ncut + 1
        return sparse.eye(dim)

    def plot_potential(
        self,
        phi_grid: discretization.Grid1d = None,
        contour_vals: ndarray = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        contour_vals:
            specific contours to draw
        **kwargs:
            plot options
        """
        phi_grid = phi_grid or self._default_grid
        x_vals = y_vals = phi_grid.make_linspace()
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5, 5)
        return plot.contours(
            x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs
        )

    def wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        phi_grid: discretization.Grid1d = None,
    ) -> storage.WaveFunctionOnGrid:
        """
        Return a flux qubit wave function in phi1, phi2 basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        which:
            index of desired wave function (default value = 0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count=evals_count)
        else:
            _, evecs = esys
        phi_grid = phi_grid or self._default_grid

        dim = 2 * self.ncut + 1
        state_amplitudes = np.reshape(evecs[:, which], (dim, dim))

        n_vec = np.arange(-self.ncut, self.ncut + 1)
        phi_vec = phi_grid.make_linspace()
        a_1_phi = np.exp(1j * np.outer(phi_vec, n_vec)) / (2 * np.pi) ** 0.5
        a_2_phi = a_1_phi.T
        wavefunc_amplitudes = np.matmul(a_1_phi, state_amplitudes)
        wavefunc_amplitudes = np.matmul(wavefunc_amplitudes, a_2_phi)
        wavefunc_amplitudes = spec_utils.standardize_phases(wavefunc_amplitudes)

        grid2d = discretization.GridSpec(
            np.asarray(
                [
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                ]
            )
        )
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        phi_grid: discretization.Grid1d = None,
        mode: str = "abs",
        zero_calibrate: bool = True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which:
            index of wave function to be plotted (default value = (0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        mode:
            choices as specified in `constants.MODE_FUNC_DICT`
            (default value = 'abs_sqr')
        zero_calibrate:
            if True, colors are adjusted to use zero wavefunction amplitude as the
            neutral color in the palette
        **kwargs:
            plot options
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5, 5)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        return diag.evals_scipy_sparse(hamiltonian_mat, evals_count)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        return diag.esys_scipy_sparse(hamiltonian_mat, evals_count)


class DimonHigherHarmonics(Dimon):
    EJs_higher = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(self, EJs_higher: ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.EJs_higher = EJs_higher

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return super().default_params() | {
            "EJs_higher": np.array(
                [
                    0.0,
                ]
            )
        }

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns Hamiltonian in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the charge basis.
            If `True`, the energy eigenspectrum is computed; returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors); then return the Hamiltonian in the energy eigenbasis, do not recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as ndarray. For `energy_esys=False`, the Hamiltonian has dimensions of
            `truncated_dim` x `truncated_dim`. For `energy_sys=esys`, the Hamiltonian has dimensions of m x m,
            for m given eigenvectors.
        """
        hamiltonian_mat = super().hamiltonian()
        for ind_idx, EJ_higher in enumerate(self.EJs_higher):
            exp_i_harm_phi_1 = sparse.kron(
                self._exp_i_phi_operator(harmonic=ind_idx + 2), self._identity()
            )
            exp_i_harm_phi_2 = sparse.kron(
                self._identity(), self._exp_i_phi_operator(harmonic=ind_idx + 2)
            )
            hamiltonian_mat += (
                -0.5 * EJ_higher * (exp_i_harm_phi_1 + exp_i_harm_phi_1.T)
            )
            hamiltonian_mat += (
                -0.5 * EJ_higher * (exp_i_harm_phi_2 + exp_i_harm_phi_2.T)
            )
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )
