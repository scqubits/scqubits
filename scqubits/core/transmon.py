# transmon.py
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

import math
import os

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem
from scqubits.core.storage import WaveFunction

LevelsTuple = Tuple[int, ...]
Transition = Tuple[int, int]
TransitionsTuple = Tuple[Transition, ...]

# —Cooper pair box / transmon——————————————————————————————————————————————


class Transmon(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the Cooper-pair-box and transmon qubit. The Hamiltonian is
    represented in dense form in the number basis,
    :math:`H_\text{CPB}=4E_\text{C}(\hat{n}-n_g)^2-\frac{E_\text{J}}{2}(
    |n\rangle\langle n+1|+\text{h.c.})`.
    Initialize with, for example::

        Transmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)

    Parameters
    ----------
    EJ:
       Josephson energy
    EC:
        charging energy
    ng:
        offset charge
    ncut:
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        EC: float,
        ng: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, 151)
        self._default_n_range = (-5, 6)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/fixed-transmon.jpg"
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and
        t2 noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def n_operator(self) -> ndarray:
        """Returns charge operator `n` in the charge basis"""
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1)
        return np.diag(diag_elements)

    def exp_i_phi_operator(self) -> ndarray:
        """Returns operator :math:`e^{i\\varphi}` in the charge basis"""
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - 1)
        exp_op = np.diag(entries, -1)
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

    def hamiltonian(self) -> ndarray:
        """Returns Hamiltonian in charge basis"""
        dimension = self.hilbertdim()
        hamiltonian_mat = np.diag(
            [
                4.0 * self.EC * (ind - self.ncut - self.ng) ** 2
                for ind in range(dimension)
            ]
        )
        ind = np.arange(dimension - 1)
        hamiltonian_mat[ind, ind + 1] = -self.EJ / 2.0
        hamiltonian_mat[ind + 1, ind] = -self.EJ / 2.0
        return hamiltonian_mat

    def d_hamiltonian_d_ng(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect to
        charge offset `ng`."""
        return -8 * self.EC * self.n_operator()

    def d_hamiltonian_d_EJ(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to EJ."""
        return -self.cos_phi_operator()

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return 2 * self.ncut + 1

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi:
            phase variable value
        """
        return -self.EJ * np.cos(phi)

    def plot_n_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        mode: str = "real",
        which: int = 0,
        nrange: Tuple[int, int] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plots transmon wave function in charge basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        mode:
            `'abs_sqr', 'abs', 'real', 'imag'`
        which:
             index or indices of wave functions to plot (default value = 0)
        nrange:
             range of `n` to be included on the x-axis (default value = (-5,6))
        **kwargs:
            plotting parameters
        """
        if nrange is None:
            nrange = self._default_n_range
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_wavefunc.amplitudes = amplitude_modifier(n_wavefunc.amplitudes)
        kwargs = {
            **defaults.wavefunction1d_discrete(mode),
            **kwargs,
        }  # if any duplicates, later ones survive
        return plot.wavefunction1d_discrete(n_wavefunc, xlim=nrange, **kwargs)

    def plot_phi_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        phi_grid: Grid1d = None,
        mode: str = "abs_sqr",
        scaling: float = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Alias for plot_wavefunction"""
        return self.plot_wavefunction(
            esys=esys,
            which=which,
            phi_grid=phi_grid,
            mode=mode,
            scaling=scaling,
            **kwargs
        )

    def numberbasis_wavefunction(
        self, esys: Tuple[ndarray, ndarray] = None, which: int = 0
    ) -> WaveFunction:
        """Return the transmon wave function in number basis. The specific index of the
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
        """Return the transmon wave function in phase basis. The specific index of the
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
            phi_wavefunc_amplitudes[k] = (1j**which / math.sqrt(2 * np.pi)) * np.sum(
                n_wavefunc.amplitudes
                * np.exp(1j * phi_basis_labels[k] * n_wavefunc.basis_labels)
            )
        return storage.WaveFunction(
            basis_labels=phi_basis_labels,
            amplitudes=phi_wavefunc_amplitudes,
            energy=evals[which],
        )

    def _compute_dispersion(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        transitions_tuple: TransitionsTuple = ((0, 1),),
        levels_tuple: Optional[LevelsTuple] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> Tuple[ndarray, ndarray]:
        if dispersion_name != "ng":
            return super()._compute_dispersion(
                dispersion_name,
                param_name,
                param_vals,
                transitions_tuple=transitions_tuple,
                levels_tuple=levels_tuple,
                point_count=point_count,
                num_cpus=num_cpus,
            )

        max_level = (
            np.max(transitions_tuple) if levels_tuple is None else np.max(levels_tuple)
        )
        previous_ng = self.ng
        self.ng = 0.0
        specdata_ng_0 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.ng = 0.5
        specdata_ng_05 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.ng = previous_ng

        if levels_tuple is not None:
            dispersion = np.asarray(
                [
                    [
                        np.abs(
                            specdata_ng_0.energy_table[param_index, j]
                            - specdata_ng_05.energy_table[param_index, j]
                        )
                        for param_index, _ in enumerate(param_vals)
                    ]
                    for j in levels_tuple
                ]
            )
            return specdata_ng_0.energy_table, dispersion

        dispersion_list = []
        for i, j in transitions_tuple:
            list_ij = []
            for param_index, _ in enumerate(param_vals):
                ei_0 = specdata_ng_0.energy_table[param_index, i]
                ei_05 = specdata_ng_05.energy_table[param_index, i]
                ej_0 = specdata_ng_0.energy_table[param_index, j]
                ej_05 = specdata_ng_05.energy_table[param_index, j]
                list_ij.append(
                    np.max([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                    - np.min([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                )
            dispersion_list.append(list_ij)
        return specdata_ng_0.energy_table, np.asarray(dispersion_list)


# — Flux-tunable Cooper pair box / transmon———————————————————————————————————————————


class TunableTransmon(Transmon, serializers.Serializable, NoisySystem):
    r"""Class for the flux-tunable transmon qubit. The Hamiltonian is represented in
    dense form in the number basis, :math:`H_\text{CPB}=4E_\text{C}(\hat{
    n}-n_g)^2-\frac{\mathcal{E}_\text{J}(\Phi)}{2}(|n\rangle\langle n+1|+\text{
    h.c.})`, Here, the effective Josephson energy is flux-tunable: :math:`\mathcal{
    E}_J(\Phi) = E_{J,\text{max}} \sqrt{\cos^2(\pi\Phi/\Phi_0) + d^2 \sin^2(
    \pi\Phi/\Phi_0)}` and :math:`d=(E_{J2}-E_{J1})(E_{J1}+E_{J2})` parametrizes the
    junction asymmetry.

    Initialize with, for example::

        TunableTransmon(EJmax=1.0, d=0.1, EC=2.0, flux=0.3, ng=0.2, ncut=30)

    Parameters
    ----------
    EJmax:
       maximum effective Josephson energy (sum of the Josephson energies of the two
       junctions)
    d:
        junction asymmetry parameter
    EC:
        charging energy
    flux:
        flux threading the SQUID loop, in units of the flux quantum
    ng:
        offset charge
    ncut:
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJmax = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    d = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJmax: float,
        EC: float,
        d: float,
        flux: float,
        ng: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.EJmax = EJmax
        self.EC = EC
        self.d = d
        self.flux = flux
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, 151)
        self._default_n_range = (-5, 6)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/tunable-transmon.jpg"
        )

    @property
    def EJ(self) -> float:  # type: ignore
        """This is the effective, flux dependent Josephson energy, playing the role
        of EJ in the parent class `Transmon`"""
        return self.EJmax * np.sqrt(
            np.cos(np.pi * self.flux) ** 2
            + self.d**2 * np.sin(np.pi * self.flux) ** 2
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJmax": 20.0,
            "EC": 0.3,
            "d": 0.01,
            "flux": 0.0,
            "ng": 0.0,
            "ncut": 30,
            "truncated_dim": 10,
        }

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_flux",
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_flux_bias_line",
            "t1_charge_impedance",
        ]

    def d_hamiltonian_d_flux(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `flux`."""
        return (
            np.pi
            * self.EJmax
            * np.cos(np.pi * self.flux)
            * np.sin(np.pi * self.flux)
            * (self.d**2 - 1)
            / np.sqrt(
                np.cos(np.pi * self.flux) ** 2
                + self.d**2 * np.sin(np.pi * self.flux) ** 2
            )
            * self.cos_phi_operator()
        )

    def _compute_dispersion(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        transitions_tuple: TransitionsTuple = ((0, 1),),
        levels_tuple: Optional[LevelsTuple] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> Tuple[ndarray, ndarray]:
        if dispersion_name != "flux":
            return super()._compute_dispersion(
                dispersion_name,
                param_name,
                param_vals,
                transitions_tuple=transitions_tuple,
                levels_tuple=levels_tuple,
                point_count=point_count,
                num_cpus=num_cpus,
            )

        max_level = (
            np.max(transitions_tuple) if levels_tuple is None else np.max(levels_tuple)
        )
        previous_flux = self.flux
        self.flux = 0.0
        specdata_flux_0 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.flux = 0.5
        specdata_flux_05 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.flux = previous_flux

        if levels_tuple is not None:
            dispersion = np.asarray(
                [
                    [
                        np.abs(
                            specdata_flux_0.energy_table[param_index, j]  # type:ignore
                            - specdata_flux_05.energy_table[
                                param_index, j
                            ]  # type:ignore
                        )
                        for param_index, _ in enumerate(param_vals)
                    ]
                    for j in levels_tuple
                ]
            )
            return specdata_flux_0.energy_table, dispersion  # type:ignore

        dispersion_list = []
        for i, j in transitions_tuple:
            list_ij = []
            for param_index, _ in enumerate(param_vals):
                ei_0 = specdata_flux_0.energy_table[param_index, i]  # type:ignore
                ei_05 = specdata_flux_05.energy_table[param_index, i]  # type:ignore
                ej_0 = specdata_flux_0.energy_table[param_index, j]  # type:ignore
                ej_05 = specdata_flux_05.energy_table[param_index, j]  # type:ignore
                list_ij.append(
                    np.max([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                    - np.min([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                )
            dispersion_list.append(list_ij)
        return specdata_flux_0.energy_table, np.asarray(dispersion_list)  # type:ignore
