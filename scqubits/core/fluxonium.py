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

from __future__ import annotations

import cmath
import copy
import math

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy as sp

from numpy import ndarray
from scipy.sparse import csc_matrix

import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.oscillator as osc
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.convergence import ConvergenceCheckable
from scqubits.core.convergence_report import TruncationChannel
from scqubits.core.noise import NoisySystem
from scqubits.utils.convergence_utils import ho_window_resolvent_estimate

if TYPE_CHECKING:
    from scqubits.core.discretization import Grid1d


class Fluxonium(
    base.QubitBaseClass1d, serializers.Serializable, NoisySystem, ConvergenceCheckable
):
    r"""Class for the fluxonium qubit.

    Hamiltonian :math:`H_\text{fl}=-4E_\text{C}\partial_\phi^2-E_\text{J}\cos(
    \phi+\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2` is represented in dense form.
    The employed basis is the EC-EL harmonic oscillator basis. The cosine term in
    the potential is handled via matrix exponentiation. Initialize with, for
    example::

        qubit = Fluxonium(EJ=1.0, EC=2.0, EL=0.3, flux=0.2, cutoff=120)

    Parameters
    ----------
    EJ:
        Josephson energy
    EC:
        charging energy
    EL:
        inductive energy
    flux:
        external magnetic flux in units of one flux quantum
    cutoff:
        number of harm. osc. basis states used in diagonalization
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in :class:`HilbertSpace`
        and :class:`ParameterSweep`. If not provided, an id is auto-generated.
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

    _convergence_axes: tuple[str, ...] = ("cutoff",)
    _convergence_basis: str = "harmonic_osc"

    def __init__(
        self,
        EJ: float,
        EC: float,
        EL: float,
        flux: float,
        cutoff: int,
        truncated_dim: int = 6,
        id_str: str | None = None,
        evals_method: Callable[..., Any] | str | None = None,
        evals_method_options: dict[str, Any] | None = None,
        esys_method: Callable[..., Any] | str | None = None,
        esys_method_options: dict[str, Any] | None = None,
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
    def default_params() -> dict[str, Any]:
        """Return a default-parameter dict suitable for instantiating the class."""
        return {
            "EJ": 8.9,
            "EC": 2.5,
            "EL": 0.5,
            "flux": 0.0,
            "cutoff": 110,
            "truncated_dim": 10,
        }

    @classmethod
    def supported_noise_channels(cls) -> list[str]:
        """Return a list of supported noise channels."""
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
    def effective_noise_channels(cls) -> list[str]:
        """Return the default channels used for effective t1 and t2 noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    # ----- Convergence-diagnostics hooks ----------------------------------------------

    def _convergence_truncation_channel(self, axis: str) -> TruncationChannel:
        """Report the harmonic-oscillator (Fock) tail channel for ``cutoff``."""
        return "HO_tail"

    def _convergence_boundary_diagnostic(
        self,
        esys: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        axis: str,
    ) -> npt.NDArray[np.float64] | None:
        """Per-level boundary-amplitude cheap diagnostic for ``cutoff``.

        Returns the squared amplitude in the highest harmonic-oscillator basis
        state, ``|c_{cutoff-1, k}|^2``, for each kept level ``k``. Large values
        signal appreciable support at the top of the kept Fock space, a dismissal
        signal; cheap mode raises a warning only when this exceeds a small
        threshold. Returns ``None`` if ``axis`` is not ``"cutoff"``.
        """
        if axis != "cutoff":
            return None
        _, evecs = esys
        return (np.abs(evecs[-1, :]) ** 2).astype(np.float64)

    def _convergence_tail_estimate(
        self,
        esys: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        axis: str,
    ) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]
        | None
    ):
        """Finite-window block-resolvent perturbative estimate for ``cutoff``.

        The cosine is not banded in the oscillator basis, so the dropped-space
        residual is built from the full kept vector (design spec): the
        kept-to-dropped coupling ``<m|H|j> = -EJ <m|cos(phi+phi_ext)|j>`` (the LC
        term is diagonal and does not couple kept to dropped) is obtained from the
        cosine on an extended Fock basis, and the second-order window estimate is
        formed by :func:`ho_window_resolvent_estimate`. This replaces a bare
        top-Fock boundary band, which the spec cautions against. The result is a
        perturbative estimate with an omitted-tail diagnostic, not a bound: a
        finite dropped window omits both the far-dropped residual and the Schur
        self-energy of the remaining dropped space, so it is not sign-definite. If
        the omitted-window residual is not small the level is reported unverified
        in cheap mode and should be checked by refinement. Returns ``None`` for an
        unrecognized axis.
        """
        if axis != "cutoff":
            return None
        evals, evecs = esys
        n_kept = evecs.shape[0]
        n_levels = evecs.shape[1]
        window, tail = 8, 8

        clone = copy.deepcopy(self)
        clone.cutoff = n_kept + window + tail
        cos_ext = np.asarray(clone.cos_phi_operator(beta=2.0 * np.pi * self.flux))
        omega_lc = self.plasma_energy()

        win_lo, win_hi = n_kept, n_kept + window
        tail_lo, tail_hi = n_kept + window, n_kept + window + tail
        coupling_window = -self.EJ * cos_ext[win_lo:win_hi, :n_kept]
        coupling_tail = -self.EJ * cos_ext[tail_lo:tail_hi, :n_kept]
        window_fock = np.arange(win_lo, win_hi)
        h_window = (
            np.diag(omega_lc * (window_fock + 0.5))
            - self.EJ * cos_ext[win_lo:win_hi, win_lo:win_hi]
        )

        estimate, perturbative_ok = ho_window_resolvent_estimate(
            coupling_window,
            coupling_tail,
            h_window,
            evecs,
            evals,
            n_levels,
        )
        # Boundary probability: occupation of the top few kept Fock states (the
        # oscillator analog of charge-edge support).
        band = min(4, n_kept)
        boundary_prob = np.sum(
            np.abs(evecs[n_kept - band :, :n_levels]) ** 2, axis=0
        ).astype(np.float64)
        return estimate, perturbative_ok, boundary_prob

    def _convergence_pad_eigenvectors(
        self,
        evecs: npt.NDArray[np.float64],
        value_from: int,
        value_to: int,
    ) -> npt.NDArray[np.float64]:
        """Zero-pad harmonic-oscillator eigenvectors to a larger ``cutoff``.

        The Fock basis is ordered ``0 .. cutoff-1``; embedding into a larger
        cutoff appends zero rows for the added high-Fock states.
        """
        if value_to < value_from:
            raise ValueError(
                f"value_to ({value_to}) must be >= value_from ({value_from})"
            )
        pad = value_to - value_from
        if pad == 0:
            return evecs
        return np.pad(evecs, ((0, pad), (0, 0)))

    def phi_osc(self) -> float:
        """Return the oscillator length for the fluxonium LC oscillator."""
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def plasma_energy(self) -> float:
        r"""Return the plasma oscillation frequency :math:`\sqrt{8 E_L E_C}`."""
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation energy

    def phi_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`\phi` operator in the harmonic-oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the :math:`\phi` operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The :math:`\phi` operator in the chosen basis as an ndarray. For
        ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        dimension = self.hilbertdim()
        native = (
            (op.creation(dimension) + op.annihilation(dimension))
            * self.phi_osc()
            / math.sqrt(2)
        )

        return self.process_op(native_op=native, energy_esys=energy_esys)

    def n_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return :math:`n = -i\,d/d\phi` in the harmonic-oscillator or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The operator :math:`n = -i\,d/d\phi` in the chosen basis as an ndarray.
        For ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
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
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return :math:`e^{i(\alpha\phi+\beta)}` in the harmonic-oscillator or eigenenergy basis.

        Parameters
        ----------
        alpha:
            prefactor multiplying :math:`\phi` in the exponent.
        beta:
            additive phase in the exponent.
        energy_esys:
            If ``False`` (default), returns the operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The operator :math:`e^{i(\alpha\phi+\beta)}` in the chosen basis as an
        ndarray. For ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        exponent = 1j * (alpha * np.asarray(self.phi_operator()))
        native = sp.linalg.expm(exponent) * cmath.exp(1j * beta)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def cos_phi_operator(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return :math:`\cos(\alpha\phi+\beta)` in the harmonic-oscillator or eigenenergy basis.

        Parameters
        ----------
        alpha:
            prefactor multiplying :math:`\phi` inside the cosine.
        beta:
            additive phase inside the cosine.
        energy_esys:
            If ``False`` (default), returns the operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The operator :math:`\cos(\alpha\phi+\beta)` in the chosen basis as an
        ndarray. For ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        argument = alpha * np.asarray(self.phi_operator()) + beta * np.eye(
            self.hilbertdim()
        )
        native = sp.linalg.cosm(argument)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def sin_phi_operator(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return :math:`\sin(\alpha\phi+\beta)` in the harmonic-oscillator or eigenenergy basis.

        Parameters
        ----------
        alpha:
            prefactor multiplying :math:`\phi` inside the sine.
        beta:
            additive phase inside the sine.
        energy_esys:
            If ``False`` (default), returns the operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The operator :math:`\sin(\alpha\phi+\beta)` in the chosen basis as an
        ndarray. For ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        argument = alpha * np.asarray(self.phi_operator()) + beta * np.eye(
            self.hilbertdim()
        )
        native = sp.linalg.sinm(argument)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hamiltonian(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:  # follow Zhu et al., PRB 87, 024510 (2013)
        """Return the Hamiltonian in the harmonic-oscillator or eigenenergy basis.

        Follows Zhu et al., PRB 87, 024510 (2013).

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the Hamiltonian in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the Hamiltonian is returned in the energy eigenbasis.
            If ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the Hamiltonian is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The Hamiltonian in the chosen basis as an ndarray. For
        ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        dimension = self.hilbertdim()
        diag_elements = [(i + 0.5) * self.plasma_energy() for i in range(dimension)]
        lc_osc_matrix = np.diag(diag_elements)

        cos_matrix = np.asarray(self.cos_phi_operator(beta=2 * np.pi * self.flux))

        hamiltonian_mat = lc_osc_matrix - self.EJ * cos_matrix
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )

    def d_hamiltonian_d_EJ(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        """Return the derivative of the Hamiltonian with respect to ``EJ``.

        Returned in the harmonic-oscillator or eigenenergy basis. The flux is
        grouped as in the Hamiltonian.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The operator in the chosen basis as an ndarray. For
        ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        native = -self.cos_phi_operator(1, 2 * np.pi * self.flux)

        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_flux(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        """Return the derivative of the Hamiltonian with respect to :attr:`flux`.

        Returned in the harmonic-oscillator or eigenenergy basis. The flux is
        grouped as in the Hamiltonian.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the
            harmonic-oscillator basis. If ``True``, the energy eigenspectrum is
            computed and the operator is returned in the energy eigenbasis. If
            ``energy_esys=esys``, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recalculating the eigenspectrum.

        Returns
        -------
        The operator in the chosen basis as an ndarray. For
        ``energy_esys=True``, it has dimensions
        :attr:`truncated_dim` x :attr:`truncated_dim`; for an explicitly
        supplied ``esys``, it has dimensions m x m, where m is the number of
        given eigenvectors.
        """
        native = -2 * np.pi * self.EJ * self.sin_phi_operator(1, 2 * np.pi * self.flux)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """Return the Hilbert space dimension."""
        return self.cutoff

    def potential(self, phi: float | ndarray) -> ndarray:
        r"""Return the fluxonium potential evaluated at :math:`\phi`.

        Parameters
        ----------
        phi:
            phase variable value(s) at which the potential is evaluated.
        """
        return 0.5 * self.EL * phi * phi - self.EJ * np.cos(
            phi + 2.0 * np.pi * self.flux
        )

    def wavefunction(
        self,
        esys: tuple[ndarray, ndarray] | None = None,
        which: int = 0,
        phi_grid: Grid1d | None = None,
    ) -> storage.WaveFunction:
        r"""Return a fluxonium wave function in the :math:`\phi` basis.

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        which:
            index of desired wave function (default: 0)
        phi_grid:
            custom grid for :math:`\phi`; if ``None``, ``self._default_grid``
            is used.
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
        phi_wavefunc_amplitudes = np.zeros(phi_grid.pt_count, dtype=np.complex128)
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
