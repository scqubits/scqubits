# fluxonium.py
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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp

from numpy import ndarray

import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.oscillator as osc
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.noise import NoisySystem

if TYPE_CHECKING:
    from scqubits.core.discretization import Grid1d


class Fluxonium(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the fluxonium qubit. Hamiltonian :math:`H_\text{fl}=-4E_\text{
    C}\partial_\phi^2-E_\text{J}\cos(\phi+\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2`
    is represented in dense form. The employed basis is the EC-EL harmonic oscillator
    basis. The cosine term in the potential is handled via matrix exponentiation.
    Initialize with, for example::

        qubit = Fluxonium(EJ=1.0, EC=2.0, EL=0.3, flux=0.2, cutoff=120)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    EL: float
        inductive energy
    flux: float
        external magnetic flux in units of one flux quantum
    cutoff: int
        number of harm. osc. basis states used in diagonalization
    truncated_dim: int
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str: str
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

    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    cutoff = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        EC: float,
        EL: float,
        flux: float,
        cutoff: int,
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
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux = flux
        self.cutoff = cutoff
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-4.5 * np.pi, 4.5 * np.pi, 151)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 8.9,
            "EC": 2.5,
            "EL": 0.5,
            "flux": 0.0,
            "cutoff": 110,
            "truncated_dim": 10,
        }

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            "t1_capacitive",
            "t1_charge_impedance",
            "t1_flux_bias_line",
            "t1_inductive",
            "t1_quasiparticle_tunneling",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and t2
        noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def phi_osc(self) -> float:
        """
        Returns
        -------
            Returns oscillator length for the LC oscillator composed of the fluxonium
             inductance and capacitance.
        """
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def plasma_energy(self) -> float:
        """
        Returns
        -------
            Returns the plasma oscillation frequency, sqrt(8*EL*EC).
        """
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation energy

    def phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns the phi operator in the LC harmonic oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns phi operator in the LC harmonic oscillator basis.
            If `True`, the energy eigenspectrum is computed, returns phi operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns phi operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Phi operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, phi operator has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, phi operator has dimensions of m x m,
            for m given eigenvectors.
        """
        dimension = self.hilbertdim()
        native = (
            (op.creation(dimension) + op.annihilation(dimension))
            * self.phi_osc()
            / math.sqrt(2)
        )

        return self.process_op(native_op=native, energy_esys=energy_esys)

    def n_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns the :math:`n = - i d/d\\phi` operator in the LC harmonic oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`n = - i d/d\\phi` operator in the LC harmonic oscillator basis.
            If `True`, the energy eigenspectrum is computed, returns the :math:`n = - i d/d\\phi` operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns the :math:`n = - i d/d\\phi` operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`n = - i d/d\\phi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`n = - i d/d\\phi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`n = - i d/d\\phi` has dimensions of
            m x m, for m given eigenvectors.
        """
        dimension = self.hilbertdim()
        native = (
            1j
            * (op.creation(dimension) - op.annihilation(dimension))
            / (self.phi_osc() * math.sqrt(2))
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def exp_i_phi_operator(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> ndarray:
        """
        Returns the :math:`e^{i (\\alpha \\phi + \\beta) }` operator, with :math:`\\alpha` and :math:`\\beta` being
        numbers, in the LC harmonic oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`e^{i (\\alpha \\phi + \\beta) }` operator in the LC harmonic
            oscillator basis. If `True`, the energy eigenspectrum is computed, returns the
            :math:`e^{i (\\alpha \\phi + \\beta) }` operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns the :math:`e^{i (\\alpha \\phi + \\beta) }` operator in the energy eigenbasis, and does not have to
            recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`e^{i (\\alpha \\phi + \\beta) }` in chosen basis as ndarray. If the eigenenergy basis is
            chosen, unless energy_esys is specified, :math:`e^{i (\\alpha \\phi + \\beta) }` has dimensions of
            `truncated_dim`x `truncated_dim`. Otherwise, if eigenenergy basis is chosen,
            :math:`e^{i (\\alpha \\phi + \\beta) }` has dimensions of m x m, for m given eigenvectors.
        """
        exponent = 1j * (alpha * self.phi_operator())
        native = sp.linalg.expm(exponent) * cmath.exp(1j * beta)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def cos_phi_operator(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> ndarray:
        """
        Returns the :math:`\\cos (\\alpha \\phi + \\beta)` operator with :math:`\\alpha` and :math:`\\beta` being
        numbers, in the LC harmonic oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`\\cos (\\alpha \\phi + \\beta)` operator in the LC harmonic oscillator basis.
            If `True`, the energy eigenspectrum is computed, returns the :math:`\\cos (\\alpha \\phi + \\beta)` operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns the :math:`\\cos (\\alpha \\phi + \\beta)` operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos (\\alpha \\phi + \\beta)` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos (\\alpha \\phi + \\beta)` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos (\\alpha \\phi + \\beta)` has dimensions of m x m, for m given eigenvectors.
        """
        argument = alpha * self.phi_operator() + beta * np.eye(self.hilbertdim())
        native = sp.linalg.cosm(argument)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def sin_phi_operator(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> ndarray:
        """
        Returns the :math:`\\sin (\\alpha \\phi + \\beta)` operator with :math:`\\alpha` and :math:`\\beta` being
        numbers, in the LC harmonic oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`\\sin (\\alpha \\phi + \\beta)` operator in the LC harmonic oscillator basis.
            If `True`, the energy eigenspectrum is computed, returns the :math:`\\sin (\\alpha \\phi + \\beta)` operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns the :math:`\\sin (\\alpha \\phi + \\beta)` operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\sin (\\alpha \\phi + \\beta)` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\sin (\\alpha \\phi + \\beta)` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\sin (\\alpha \\phi + \\beta)` has dimensions of m x m, for m given eigenvectors.
        """
        argument = alpha * self.phi_operator() + beta * np.eye(self.hilbertdim())
        native = sp.linalg.sinm(argument)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:  # follow Zhu et al., PRB 87, 024510 (2013)
        """
        Constructs Hamiltonian matrix in harmonic-oscillator, following Zhu
        et al., PRB 87, 024510 (2013), or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the harmonic-oscillator basis.
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
        dimension = self.hilbertdim()
        diag_elements = [(i + 0.5) * self.plasma_energy() for i in range(dimension)]
        lc_osc_matrix = np.diag(diag_elements)

        cos_matrix = self.cos_phi_operator(beta=2 * np.pi * self.flux)

        hamiltonian_mat = lc_osc_matrix - self.EJ * cos_matrix
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )

    def d_hamiltonian_d_EJ(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJ in the harmonic-oscillator or eigenenergy basis. The flux is grouped as in the Hamiltonian.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -self.cos_phi_operator(1, 2 * np.pi * self.flux)

        return self.process_op(native_op=native, energy_esys=energy_esys)

    # def d_hamiltonian_d_flux(self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False) -> ndarray:
    #     """Returns operator representing a derivative of the Hamiltonian with respect
    #     to flux.
    #
    #     The flux is grouped as in the Hamiltonian."""
    #     return -2 * np.pi * self.EJ * self.sin_phi_operator(1, 2 * np.pi * self.flux, energy_esys=energy_esys)

    def d_hamiltonian_d_flux(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        flux in the harmonic-oscillator or eigenenergy basis. The flux is grouped as in the Hamiltonian.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -2 * np.pi * self.EJ * self.sin_phi_operator(1, 2 * np.pi * self.flux)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """
        Returns
        -------
            Returns the Hilbert space dimension."""
        return self.cutoff

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Fluxonium potential evaluated at `phi`.

        Parameters
        ----------
            float value of the phase variable `phi`

        Returns
        -------
        float or ndarray
        """
        return 0.5 * self.EL * phi * phi - self.EJ * np.cos(
            phi + 2.0 * np.pi * self.flux
        )

    def wavefunction(
        self,
        esys: Optional[Tuple[ndarray, ndarray]],
        which: int = 0,
        phi_grid: "Grid1d" = None,
    ) -> storage.WaveFunction:
        """Returns a fluxonium wave function in `phi` basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        which:
             index of desired wave function (default value = 0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count=evals_count)
        else:
            evals, evecs = esys
        dim = self.hilbertdim()

        phi_grid = phi_grid or self._default_grid

        phi_basis_labels = phi_grid.make_linspace()
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros(phi_grid.pt_count, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[
                n
            ] * osc.harm_osc_wavefunction(n, phi_basis_labels, phi_osc)
        return storage.WaveFunction(
            basis_labels=phi_basis_labels,
            amplitudes=phi_wavefunc_amplitudes,
            energy=evals[which],
        )
