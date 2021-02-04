# fluxonium.py
#
# This file is part of scqubits.
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
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np
import scipy as sp
from numpy import ndarray

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.harmonic_osc as osc
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.noise import NoisySystem

if TYPE_CHECKING:
    from scqubits.core.discretization import Grid1d


# —Fluxonium qubit ————————————————————————


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
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    cutoff: int
        number of harm. osc. basis states used in diagonalization
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    """
    EJ = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    cutoff = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        EC: float,
        EL: float,
        flux: float,
        cutoff: int,
        truncated_dim: int = 6,
    ) -> None:
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux = flux
        self.cutoff = cutoff
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._default_grid = discretization.Grid1d(-4.5 * np.pi, 4.5 * np.pi, 151)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/fluxonium.jpg"
        )

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

    def supported_noise_channels(self) -> List[str]:
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

    def phi_osc(self) -> float:
        """
        Returns
        -------
            Returns oscillator length for the LC oscillator composed of the fluxonium
             inductance and capacitance.
        """
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def E_plasma(self) -> float:
        """
        Returns
        -------
            Returns the plasma oscillation frequency.
        """
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation energy

    def phi_operator(self) -> ndarray:
        """
        Returns
        -------
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.hilbertdim()
        return (
            (op.creation(dimension) + op.annihilation(dimension))
            * self.phi_osc()
            / math.sqrt(2)
        )

    def n_operator(self) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`n = - i d/d\\phi` operator in the LC harmonic
            oscillator basis
        """
        dimension = self.hilbertdim()
        return (
            1j
            * (op.creation(dimension) - op.annihilation(dimension))
            / (self.phi_osc() * math.sqrt(2))
        )

    def exp_i_phi_operator(self, alpha: float = 1.0, beta: float = 0.0) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`e^{i (\\alpha \\phi + \beta) }` operator in the
            LC harmonic oscillator basis,
            with :math:`\\alpha` and :math:`\\beta` being numbers
        """
        exponent = 1j * (alpha * self.phi_operator())
        return sp.linalg.expm(exponent) * cmath.exp(1j * beta)

    def cos_phi_operator(self, alpha: float = 1.0, beta: float = 0.0) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`\\cos (\\alpha \\phi + \\beta)` operator in the LC
            harmonic oscillator basis,
            with :math:`\\alpha` and :math:`\\beta` being numbers
        """
        exp_matrix = self.exp_i_phi_operator(alpha, beta)
        return 0.5 * (exp_matrix + exp_matrix.conjugate().T)

    def sin_phi_operator(self, alpha: float = 1.0, beta: float = 0.0) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`\\sin (\\alpha \\phi + \\beta)` operator in the
            LC harmonic oscillator basis
            with :math:`\\alpha` and :math:`\\beta` being numbers
        """
        exp_matrix = self.exp_i_phi_operator(alpha, beta)
        return -1j * 0.5 * (exp_matrix - exp_matrix.conjugate().T)

    def hamiltonian(self) -> ndarray:  # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harmonic-oscillator basis, following Zhu
        et al., PRB 87, 024510 (2013)"""
        dimension = self.hilbertdim()
        diag_elements = [i * self.E_plasma() for i in range(dimension)]
        lc_osc_matrix = np.diag(diag_elements)

        exp_matrix = self.exp_i_phi_operator() * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = 0.5 * (exp_matrix + exp_matrix.conjugate().T)

        hamiltonian_mat = lc_osc_matrix - self.EJ * cos_matrix
        return np.real(
            hamiltonian_mat
        )  # use np.real to remove rounding errors from matrix exponential,
        # fluxonium Hamiltonian in harm. osc. basis is real-valued

    def d_hamiltonian_d_EJ(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `EJ`.

        The flux is grouped as in the Hamiltonian.
        """
        return -self.cos_phi_operator(1, 2 * np.pi * self.flux)

    def d_hamiltonian_d_flux(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `flux`.

        Flux is grouped as in the Hamiltonian.
        """
        return -2 * np.pi * self.EJ * self.sin_phi_operator(1, 2 * np.pi * self.flux)

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
        self, esys: Tuple[ndarray, ndarray], which: int = 0, phi_grid: "Grid1d" = None
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
            evals, evecs = self.eigensys(evals_count)
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


class FluxoniumFluxWithHarmonic(Fluxonium):
    def __init__(
        self,
        EJ: float,
        EC: float,
        EL: float,
        flux: float,
        cutoff: int,
        truncated_dim: int = 6,
    ):
        Fluxonium.__init__(self, EJ, EC, EL, flux, cutoff, truncated_dim)

    def hamiltonian(self) -> ndarray:
        """Construct Hamiltonian matrix in harmonic-oscillator basis"""
        dimension = self.hilbertdim()
        lc_osc_matrix = np.diag([i * self.E_plasma() for i in range(dimension)])
        lc_osc_matrix += -self.EL * 2.0 * np.pi * self.flux * self.phi_operator()

        exp_matrix = self.exp_i_phi_operator()
        hamiltonian_mat = lc_osc_matrix - self.EJ * 0.5 * (
            exp_matrix + exp_matrix.conjugate().T
        )
        return np.real(hamiltonian_mat)

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Fluxonium potential evaluated at `phi`.

        Parameters
        ----------
            float value of the phase variable `phi`

        Returns
        -------
        float or ndarray
        """
        return 0.5 * self.EL * (phi - 2.0 * np.pi * self.flux) ** 2 - self.EJ * np.cos(
            phi
        )
