# cos2phi_qubit.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from __future__ import annotations

from collections.abc import Callable
import math

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy as sp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy import sparse
from scipy.sparse import coo_matrix, csc_matrix, dia_matrix

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.oscillator as osc
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.core.units as units
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as utils

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NOISE_PARAMS, NoisySystem, calc_therm_ratio
from scqubits.core.storage import WaveFunctionOnGrid


# - Cosine-2-phi qubit noise class ------------------------------------------
class NoisyCos2PhiQubit(NoisySystem, ABC):
    """Mixin providing noise channels specific to the cosine-two-phi qubit."""

    @abstractmethod
    def phi_1_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        """Return the phase operator across inductor 1.

        Parameters
        ----------
        energy_esys:
            See :meth:`Cos2PhiQubit.phi_1_operator`.
        """
        pass

    @abstractmethod
    def phi_2_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        """Return the phase operator across inductor 2.

        Parameters
        ----------
        energy_esys:
            See :meth:`Cos2PhiQubit.phi_2_operator`.
        """
        pass

    @abstractmethod
    def n_1_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        """Return the charge operator for junction 1.

        Parameters
        ----------
        energy_esys:
            See :meth:`Cos2PhiQubit.n_1_operator`.
        """
        pass

    @abstractmethod
    def n_2_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        """Return the charge operator for junction 2.

        Parameters
        ----------
        energy_esys:
            See :meth:`Cos2PhiQubit.n_2_operator`.
        """
        pass

    @abstractmethod
    def n_zeta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`n_\zeta` operator for the shunt-capacitor mode.

        Parameters
        ----------
        energy_esys:
            See :meth:`Cos2PhiQubit.n_zeta_operator`.
        """
        pass

    def t1_inductive(  # type: ignore[override]
        self,
        i: int = 1,
        j: int = 0,
        Q_ind: float | Callable | None = None,
        T: float = NOISE_PARAMS["T"],
        total: bool = True,
        esys: tuple[ndarray, ndarray] | None = None,
        get_rate: bool = False,
    ) -> float:
        r""":math:`T_1` due to inductive dissipation in superinductors.

        References: nguyen et al (2019), Smith et al (2020)

        Parameters
        ----------
        i:
            state index that along with j defines a transition (i->j)
        j:
            state index that along with i defines a transition (i->j)
        Q_ind:
            inductive quality factor; a fixed value or function of `omega`
        T:
            temperature in Kelvin
        total:
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
        time or rate
            decoherence time in units of :math:`2\pi` (system units), or rate in
            inverse units.
        """
        if "t1_inductive" not in self.supported_noise_channels():
            raise RuntimeError(
                "Noise channel 't1_inductive' is not supported in this system."
            )

        if Q_ind is None:
            # See Smith et al (2020)
            def q_ind_fun(omega):
                therm_ratio = abs(calc_therm_ratio(omega, T))
                therm_ratio_500MHz = calc_therm_ratio(
                    2 * np.pi * 500e6, T, omega_in_standard_units=True
                )
                return (
                    500e6
                    * (
                        sp.special.kv(0, 1 / 2 * therm_ratio_500MHz)
                        * np.sinh(1 / 2 * therm_ratio_500MHz)
                    )
                    / (
                        sp.special.kv(0, 1 / 2 * therm_ratio)
                        * np.sinh(1 / 2 * therm_ratio)
                    )
                )

        elif callable(Q_ind):  # Q_ind is a function of omega
            q_ind_fun = Q_ind
        else:  # Q_ind is given as a number

            def q_ind_fun(omega):
                return Q_ind

        def spectral_density1(omega, T):
            r"""Calculates the first spectral density from the angular frequency and
            temperature.

            Parameters
            ----------
            omega:
                angular frequency in system units (units of frequency)
            T:
                temperature in Kelvin

            Returns
            -------
            Spectral density
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s = (
                2
                * self.EL
                / (1 - self.dL)
                / q_ind_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s

        noise_op1 = self.phi_1_operator()

        def spectral_density2(omega, T):
            r"""Calculates the second spectral density from the angular frequency and
            temperature.

            Parameters
            ----------
            omega:
                angular frequency in system units (units of frequency)
            T:
                temperature in Kelvin

            Returns
            -------
            Spectral Density
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s = (
                2
                * self.EL
                / (1 + self.dL)
                / q_ind_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s

        noise_op2 = self.phi_2_operator()

        rate_1 = self.t1(
            i=i,
            j=j,
            noise_op=noise_op1,
            spectral_density=spectral_density1,
            total=total,
            esys=esys,
            get_rate=True,
        )
        rate_2 = self.t1(
            i=i,
            j=j,
            noise_op=noise_op2,
            spectral_density=spectral_density2,
            total=total,
            esys=esys,
            get_rate=True,
        )

        if get_rate:
            return rate_1 + rate_2
        else:
            return 1 / (rate_1 + rate_2)

    def t1_capacitive(  # type: ignore[override]
        self,
        i: int = 1,
        j: int = 0,
        Q_cap: float | Callable | None = None,
        T: float = NOISE_PARAMS["T"],
        total: bool = True,
        esys: tuple[ndarray, ndarray] | None = None,
        get_rate: bool = False,
    ) -> float:
        r""":math:`T_1` due to dielectric dissipation in the Josephson-junction capacitances.

        References:  Nguyen et al (2019), Smith et al (2020).

        Parameters
        ----------
        i:
            state index that along with j defines a transition (i->j)
        j:
            state index that along with i defines a transition (i->j)
        Q_cap:
            capacitive quality factor; a fixed value or function of `omega`
        T:
            temperature in Kelvin
        total:
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
        time or rate: float
            decoherence time in units of :math:`2\pi` (system units), or rate
             in inverse units.
        """
        if "t1_capacitive" not in self.supported_noise_channels():
            raise RuntimeError(
                "Noise channel 't1_capacitive' is not supported in this system."
            )

        if Q_cap is None:
            # See Smith et al (2020)
            def q_cap_fun(omega):
                return (
                    1e6
                    * (2 * np.pi * 6e9 / np.abs(units.to_standard_units(omega))) ** 0.7
                )

        elif callable(Q_cap):  # Q_cap is a function of omega
            q_cap_fun = Q_cap
        else:  # Q_cap is given as a number

            def q_cap_fun(omega):
                return Q_cap

        def spectral_density1(omega, T):
            r"""Calculates the first spectral density from the angular frequency and
            temperature.

            Parameters
            ----------
            omega:
                angular frequency in system units (units of frequency)
            T:
                temperature in Kelvin

            Returns
            -------
            Spectral density
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s1 = (
                2
                * 8
                * self.ECJ
                / (1 - self.dCJ)
                / q_cap_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s1 *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s1

        def spectral_density2(omega, T):
            r"""Calculates the second spectral density from the angular frequency and
            temperature.

            Parameters
            ----------
            omega:
                angular frequency in system units (units of frequency)
            T:
                temperature in Kelvin

            Returns
            -------
            Spectral density
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s2 = (
                2
                * 8
                * self.ECJ
                / (1 + self.dCJ)
                / q_cap_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s2 *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s2

        noise_op1 = self.n_1_operator()
        noise_op2 = self.n_2_operator()

        rate_1 = self.t1(
            i=i,
            j=j,
            noise_op=noise_op1,
            spectral_density=spectral_density1,
            total=total,
            esys=esys,
            get_rate=True,
        )
        rate_2 = self.t1(
            i=i,
            j=j,
            noise_op=noise_op2,
            spectral_density=spectral_density2,
            total=total,
            esys=esys,
            get_rate=True,
        )

        if get_rate:
            return rate_1 + rate_2
        else:
            return 1 / (rate_1 + rate_2)

    def t1_purcell(
        self,
        i: int = 1,
        j: int = 0,
        Q_cap: float | Callable | None = None,
        T: float = NOISE_PARAMS["T"],
        total: bool = True,
        esys: tuple[ndarray, ndarray] | None = None,
        get_rate: bool = False,
    ) -> float:
        r""":math:`T_1` due to dielectric dissipation in the shunt capacitor.

        References:  Nguyen et al (2019), Smith et al (2020).

        Parameters
        ----------
        i:
            state index that along with j defines a transition (i->j)
        j:
            state index that along with i defines a transition (i->j)
        Q_cap:
            capacitive quality factor; a fixed value or function of `omega`
        T:
            temperature in Kelvin
        total:
            if False return a time/rate associated with a transition from state i to state j.
            if True return a time/rate associated with both i to j and j to i transitions
        esys:
            evals, evecs tuple
        get_rate:
            get rate or time

        Returns
        -------
        time or rate
            decoherence time in units of :math:`2\pi` (system units), or rate
            in inverse units.
        """
        if "t1_purcell" not in self.supported_noise_channels():
            raise RuntimeError(
                "Noise channel 't1_purcell' is not supported in this system."
            )

        if Q_cap is None:
            # See Smith et al (2020)
            def q_cap_fun(omega):
                return (
                    1e6
                    * (2 * np.pi * 6e9 / np.abs(units.to_standard_units(omega))) ** 0.7
                )

        elif callable(Q_cap):  # Q_cap is a function of omega
            q_cap_fun = Q_cap
        else:  # Q_cap is given as a number

            def q_cap_fun(omega):
                return Q_cap

        def spectral_density(omega, T):
            r"""Calculates the spectral density from the angular frequency and
            temperature.

            Parameters
            ----------
            omega:
                angular frequency in system units (units of frequency)
            T:
                temperature in Kelvin

            Returns
            -------
            Spectral density
            """
            therm_ratio = calc_therm_ratio(omega, T)
            s = (
                2
                * 8
                * self.EC
                / q_cap_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s

        noise_op = self.n_zeta_operator()

        return self.t1(
            i=i,
            j=j,
            noise_op=noise_op,
            spectral_density=spectral_density,
            total=total,
            esys=esys,
            get_rate=get_rate,
        )


# -Cosine two phi qubit ----------------------------------------------------------------
class Cos2PhiQubit(base.QubitBaseClass, serializers.Serializable, NoisyCos2PhiQubit):
    r"""Cosine Two Phi Qubit.

    | [1] Smith et al., NPJ Quantum Inf. 6, 8 (2020)
    http://www.nature.com/articles/s41534-019-0231-2

    .. math::

        H = & \,2 E_\text{CJ}'n_\phi^2
              + 2 E_\text{CJ}' (n_\theta - n_\text{g} - n_\zeta)^2
              + 4 E_\text{C} n_\zeta^2\\
            & + E_\text{L}'(\phi - \pi\Phi_\text{ext}/\Phi_0)^2
              + E_\text{L}' \zeta^2
              - 2 E_\text{J}\cos{\theta}\cos{\phi} \\
            & + 2 dE_\text{J} E_\text{J}\sin{\theta}\sin{\phi} \\
            & - 4 dC_\text{J} E_\text{CJ}' n_\phi (n_\theta - n_\text{g}-n_\zeta) \\
            & + dL E_\text{L}'(2\phi - \varphi_\text{ext})\zeta ,

    where :math:`E_\text{CJ}' = E_\text{CJ} / (1 - dC_\text{J})^2` and
    :math:`E_\text{L}' = E_\text{L} / (1 - dL)^2`.

    Parameters
    ----------
    EJ:
        Josephson energy of the two junctions
    ECJ:
        charging energy of the two junctions
    EL:
        inductive energy of the two inductors
    EC:
        charging energy of the shunt capacitor
    dCJ:
        disorder in junction charging energy
    dL:
        disorder in inductive energy
    dEJ:
        disorder in junction energy
    flux:
        external magnetic flux in units of one flux quantum
    ng:
        offset charge
    ncut:
        cutoff in charge basis, -ncut <= :math:`n_\theta` <= ncut
    zeta_cut:
        number of harmonic oscillator basis states for :math:`\zeta` variable
    phi_cut:
        number of harmonic oscillator basis states for :math:`\phi` variable
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
    ECJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dCJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dEJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    zeta_cut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    phi_cut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        ECJ: float,
        EL: float,
        EC: float,
        dL: float,
        dCJ: float,
        dEJ: float,
        flux: float,
        ng: float,
        ncut: int,
        zeta_cut: int,
        phi_cut: int,
        truncated_dim: int = 6,
        id_str: str | None = None,
        evals_method: Callable | str | None = None,
        evals_method_options: dict | None = None,
        esys_method: Callable | str | None = None,
        esys_method_options: dict | None = None,
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
        self.ECJ = ECJ
        self.EL = EL
        self.EC = EC
        self.dL = dL
        self.dCJ = dCJ
        self.dEJ = dEJ
        self.flux = flux
        self.ng = ng
        self.ncut = ncut
        self.zeta_cut = zeta_cut
        self.phi_cut = phi_cut
        self.truncated_dim = truncated_dim
        self._default_phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_zeta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_theta_grid = discretization.Grid1d(-0.5 * np.pi, 1.5 * np.pi, 100)

    @staticmethod
    def default_params() -> dict[str, Any]:
        r"""Return a dictionary of default parameter values.

        Returns
        -------
        Dictionary mapping each parameter name (``EJ``, ``ECJ``, ``EL``, ``EC``,
        ``dCJ``, ``dL``, ``dEJ``, ``flux``, ``ng``, ``ncut``, ``zeta_cut``,
        ``phi_cut``) to its default numerical value.
        """
        return {
            "EJ": 15.0,
            "ECJ": 2.0,
            "EL": 1.0,
            "EC": 0.04,
            "dCJ": 0.0,
            "dL": 0.6,
            "dEJ": 0.0,
            "flux": 0.5,
            "ng": 0.0,
            "ncut": 7,
            "zeta_cut": 30,
            "phi_cut": 7,
        }

    @classmethod
    def create(cls) -> "Cos2PhiQubit":
        r"""Use an interactive widget to create a :class:`Cos2PhiQubit` instance.

        Returns
        -------
        New :class:`Cos2PhiQubit` instance initialized from default parameters.
        """
        init_params = cls.default_params()
        cosinetwophiqubit = cls(**init_params)
        cosinetwophiqubit.widget()
        return cosinetwophiqubit

    @classmethod
    def supported_noise_channels(cls) -> list[str]:
        r"""Return a list of supported noise channels.

        Returns
        -------
        Names of noise channels supported by :class:`Cos2PhiQubit`.
        """
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_inductive",
            "t1_purcell",
        ]

    def _dim_phi(self) -> int:
        r"""Return the Hilbert space dimension of the :math:`\phi` mode.

        Returns
        -------
        Number of harmonic oscillator basis states retained for :math:`\phi`.
        """
        return self.phi_cut

    def _dim_zeta(self) -> int:
        r"""Return the Hilbert space dimension of the :math:`\zeta` mode.

        Returns
        -------
        Number of harmonic oscillator basis states retained for :math:`\zeta`.
        """
        return self.zeta_cut

    def _dim_theta(self) -> int:
        r"""Return the Hilbert space dimension of the :math:`\theta` mode.

        Returns
        -------
        Number of charge basis states ``2 * ncut + 1``.
        """
        return 2 * self.ncut + 1

    def hilbertdim(self) -> int:
        r"""Return the total Hilbert space dimension.

        Returns
        -------
        Product of the dimensions of the :math:`\phi`, :math:`\zeta`, and
        :math:`\theta` subspaces.
        """
        return self._dim_phi() * self._dim_zeta() * self._dim_theta()

    def _disordered_el(self) -> float:
        r"""Return the inductive energy renormalized by inductor disorder.

        Returns
        -------
        :math:`E_\text{L} / (1 - dL^2)`.
        """
        return self.EL / (1 - self.dL**2)

    def _disordered_ecj(self) -> float:
        r"""Return the junction charging energy renormalized by junction disorder.

        Returns
        -------
        :math:`E_\text{CJ} / (1 - dC_\text{J}^2)`.
        """
        return self.ECJ / (1 - self.dCJ**2)

    def phi_osc(self) -> float:
        r"""Return the oscillator length of the :math:`\phi` degree of freedom.

        Returns
        -------
        Harmonic-oscillator length scale for :math:`\phi`.
        """
        return (2 * self._disordered_ecj() / self._disordered_el()) ** 0.25

    def zeta_osc(self) -> float:
        r"""Return the oscillator length of the :math:`\zeta` degree of freedom.

        Returns
        -------
        Harmonic-oscillator length scale for :math:`\zeta`.
        """
        return (4 * self.EC / self._disordered_el()) ** 0.25

    def phi_plasma(self) -> float:
        r"""Return the plasma oscillation frequency of the :math:`\phi` mode.

        Returns
        -------
        Plasma frequency :math:`\sqrt{8 E_\text{L}' E_\text{CJ}'}`.
        """
        return math.sqrt(8.0 * self._disordered_el() * self._disordered_ecj())

    def zeta_plasma(self) -> float:
        r"""Return the plasma oscillation frequency of the :math:`\zeta` mode.

        Returns
        -------
        Plasma frequency :math:`\sqrt{16 E_\text{C} E_\text{L}'}`.
        """
        return math.sqrt(16.0 * self.EC * self._disordered_el())

    def _phi_operator(self) -> csc_matrix:
        r"""Return the :math:`\phi` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`\phi` operator as :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_phi()
        return (
            (op.creation_sparse(dimension) + op.annihilation_sparse(dimension))
            * self.phi_osc()
            / math.sqrt(2)
        )

    def phi_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`\phi` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`\phi` operator in the
            native basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        :math:`\phi` operator in the chosen basis. In the native basis the
        operator is returned as a :class:`scipy.sparse.csc_matrix`. In the
        eigenenergy basis it is returned as an ndarray of shape
        :attr:`truncated_dim` x :attr:`truncated_dim` (or m x m if a custom
        ``esys`` with m eigenvectors is supplied).
        """
        native = self._kron3(
            self._phi_operator(), self._identity_zeta(), self._identity_theta()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _n_phi_operator(self) -> csc_matrix:
        r"""Return the :math:`n_\phi` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`n_\phi` operator as :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_phi()
        return (
            1j
            * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension))
            / (self.phi_osc() * math.sqrt(2))
        )

    def n_phi_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`n_\phi` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`n_\phi` operator in the
            native basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        :math:`n_\phi` operator in the chosen basis. In the native basis the
        operator is returned as a :class:`scipy.sparse.csc_matrix`. In the
        eigenenergy basis it is returned as an ndarray of shape
        :attr:`truncated_dim` x :attr:`truncated_dim` (or m x m if a custom
        ``esys`` with m eigenvectors is supplied).
        """
        native = self._kron3(
            self._n_phi_operator(), self._identity_zeta(), self._identity_theta()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _zeta_operator(self) -> csc_matrix:
        r"""Return the :math:`\zeta` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`\zeta` operator as :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_zeta()
        return (
            (op.creation_sparse(dimension) + op.annihilation_sparse(dimension))
            * self.zeta_osc()
            / math.sqrt(2)
        )

    def zeta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`\zeta` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`\zeta` operator in the
            native basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        :math:`\zeta` operator in the chosen basis. In the native basis the
        operator is returned as a :class:`scipy.sparse.csc_matrix`. In the
        eigenenergy basis it is returned as an ndarray of shape
        :attr:`truncated_dim` x :attr:`truncated_dim` (or m x m if a custom
        ``esys`` with m eigenvectors is supplied).
        """
        native = self._kron3(
            self._identity_phi(), self._zeta_operator(), self._identity_theta()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _n_zeta_operator(self) -> csc_matrix:
        r"""Return the :math:`n_\zeta` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`n_\zeta` operator as :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_zeta()
        return (
            1j
            * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension))
            / (self.zeta_osc() * math.sqrt(2))
        )

    def n_zeta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`n_\zeta` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`n_\zeta` operator in the
            native basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        :math:`n_\zeta` operator in the chosen basis. In the native basis the
        operator is returned as a :class:`scipy.sparse.csc_matrix`. In the
        eigenenergy basis it is returned as an ndarray of shape
        :attr:`truncated_dim` x :attr:`truncated_dim` (or m x m if a custom
        ``esys`` with m eigenvectors is supplied).
        """
        native = self._kron3(
            self._identity_phi(), self._n_zeta_operator(), self._identity_theta()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _exp_i_phi_operator(self) -> csc_matrix:
        r"""Return the :math:`e^{i\phi}` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`e^{i\phi}` operator as :class:`scipy.sparse.csc_matrix`.
        """
        exponent = 1j * self._phi_operator()
        return sp.sparse.linalg.expm(exponent)

    def _cos_phi_operator(self) -> csc_matrix:
        r"""Return the :math:`\cos\phi` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`\cos\phi` operator as :class:`scipy.sparse.csc_matrix`.
        """
        cos_phi_op = 0.5 * self._exp_i_phi_operator()
        cos_phi_op += cos_phi_op.conj().T  # type: ignore[arg-type]
        return cos_phi_op

    def _sin_phi_operator(self) -> csc_matrix:
        r"""Return the :math:`\sin\phi` operator in the harmonic-oscillator basis.

        Returns
        -------
        Sparse :math:`\sin\phi` operator as :class:`scipy.sparse.csc_matrix`.
        """
        sin_phi_op = -1j * 0.5 * self._exp_i_phi_operator()
        sin_phi_op += sin_phi_op.conj().T  # type: ignore[arg-type]
        return sin_phi_op

    def _n_theta_operator(self) -> csc_matrix:
        r"""Return the :math:`n_\theta` operator in the charge basis.

        Returns
        -------
        Sparse :math:`n_\theta` operator as :class:`scipy.sparse.csc_matrix`.
        """
        diag_elements = np.arange(-self.ncut, self.ncut + 1)
        return dia_matrix(
            (diag_elements, [0]), shape=(self._dim_theta(), self._dim_theta())
        ).tocsc()  # type: ignore[type-var,misc,return-value]

    def n_theta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`n_\theta` operator in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the :math:`n_\theta` operator in the
            charge basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        :math:`n_\theta` operator in the chosen basis. In the charge basis the
        operator is returned as a :class:`scipy.sparse.csc_matrix`. In the
        eigenenergy basis it is returned as an ndarray of shape
        :attr:`truncated_dim` x :attr:`truncated_dim` (or m x m if a custom
        ``esys`` with m eigenvectors is supplied).
        """
        native = self._kron3(
            self._identity_phi(), self._identity_zeta(), self._n_theta_operator()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _cos_theta_operator(self) -> csc_matrix:
        r"""Return the :math:`\cos\theta` operator in the charge basis.

        Returns
        -------
        Sparse :math:`\cos\theta` operator as :class:`scipy.sparse.csc_matrix`.
        """
        cos_op = (
            0.5
            * sparse.dia_matrix(
                (np.ones(self._dim_theta()), [1]),
                shape=(self._dim_theta(), self._dim_theta()),
            ).tocsc()  # type: ignore[type-var,misc,operator]
        )
        cos_op += (
            0.5
            * sparse.dia_matrix(
                (np.ones(self._dim_theta()), [-1]),
                shape=(self._dim_theta(), self._dim_theta()),
            ).tocsc()  # type: ignore[type-var,misc,operator]
        )
        return cos_op

    def _sin_theta_operator(self) -> csc_matrix:
        r"""Return the :math:`\sin\theta` operator in the charge basis.

        Returns
        -------
        Sparse :math:`\sin\theta` operator as :class:`scipy.sparse.csc_matrix`.
        """
        sin_op = (
            0.5
            * sparse.dia_matrix(
                (np.ones(self._dim_theta()), [-1]),
                shape=(self._dim_theta(), self._dim_theta()),
            ).tocsc()  # type: ignore[type-var,misc,operator]
        )
        sin_op -= (
            0.5
            * sparse.dia_matrix(
                (np.ones(self._dim_theta()), [1]),
                shape=(self._dim_theta(), self._dim_theta()),
            ).tocsc()  # type: ignore[type-var,misc,operator]
        )
        return sin_op * (-1j)

    def _kron3(
        self,
        mat1: csc_matrix | dia_matrix | coo_matrix | ndarray,
        mat2: csc_matrix | dia_matrix | coo_matrix | ndarray,
        mat3: csc_matrix | dia_matrix | coo_matrix | ndarray,
    ) -> csc_matrix:
        r"""Return the Kronecker product of three matrices.

        Parameters
        ----------
        mat1:
            leftmost factor in the Kronecker product
        mat2:
            middle factor in the Kronecker product
        mat3:
            rightmost factor in the Kronecker product

        Returns
        -------
        Sparse Kronecker product ``mat1 \\otimes mat2 \\otimes mat3``.
        """
        return sparse.kron(sparse.kron(mat1, mat2), mat3)

    def _identity_phi(self) -> csc_matrix:
        r"""Return the identity operator acting on the :math:`\phi` subspace.

        Returns
        -------
        Sparse identity on the :math:`\phi` subspace as
        :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_phi()
        return sparse.eye(dimension)  # type: ignore[return-value]

    def _identity_zeta(self) -> csc_matrix:
        r"""Return the identity operator acting on the :math:`\zeta` subspace.

        Returns
        -------
        Sparse identity on the :math:`\zeta` subspace as
        :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_zeta()
        return sparse.eye(dimension)  # type: ignore[return-value]

    def _identity_theta(self) -> csc_matrix:
        r"""Return the identity operator acting on the :math:`\theta` subspace.

        Returns
        -------
        Sparse identity on the :math:`\theta` subspace as
        :class:`scipy.sparse.csc_matrix`.
        """
        dimension = self._dim_theta()
        return sparse.eye(dimension)  # type: ignore[return-value]

    def total_identity(self) -> csc_matrix:
        r"""Return the identity operator on the full Hilbert space.

        Returns
        -------
        Sparse identity on the full Hilbert space as
        :class:`scipy.sparse.csc_matrix`.
        """
        return self._kron3(
            self._identity_phi(), self._identity_zeta(), self._identity_theta()
        )

    def hamiltonian(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the Hamiltonian in the native or eigenenergy basis.

        The native basis uses harmonic-oscillator states for :math:`\phi` and
        :math:`\zeta` and charge states for :math:`\theta`.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the Hamiltonian in the native basis
            (harmonic basis for :math:`\phi, \zeta` and charge basis for
            :math:`\theta`).
            If `True`, the energy eigenspectrum is computed and the Hamiltonian
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the Hamiltonian is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Hamiltonian in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        phi_osc_mat = self._kron3(
            op.number_sparse(self._dim_phi(), self.phi_plasma()),
            self._identity_zeta(),
            self._identity_theta(),
        )

        zeta_osc_mat = self._kron3(
            self._identity_phi(),
            op.number_sparse(self._dim_zeta(), self.zeta_plasma()),
            self._identity_theta(),
        )

        cross_kinetic_mat = (
            2
            * self._disordered_ecj()
            * (
                self.n_theta_operator()
                - self.total_identity() * self.ng
                - self.n_zeta_operator()
            )
            ** 2
        )

        phi_flux_term = self._cos_phi_operator() * np.cos(
            self.flux * np.pi
        ) - self._sin_phi_operator() * np.sin(self.flux * np.pi)
        junction_mat = (
            -2
            * self.EJ
            * self._kron3(
                phi_flux_term, self._identity_zeta(), self._cos_theta_operator()
            )
            + 2 * self.EJ * self.total_identity()
        )

        disorder_l = (
            -2
            * self._disordered_el()
            * self.dL
            * self._kron3(
                self._phi_operator(), self._zeta_operator(), self._identity_theta()
            )
        )

        dis_phi_flux_term = self._sin_phi_operator() * np.cos(
            self.flux * np.pi
        ) + self._cos_phi_operator() * np.sin(self.flux * np.pi)
        disorder_j = (
            2
            * self.EJ
            * self.dEJ
            * self._kron3(
                dis_phi_flux_term, self._identity_zeta(), self._sin_theta_operator()
            )
        )

        dis_c_opt = (
            self._kron3(
                self._n_phi_operator(), self._identity_zeta(), self._n_theta_operator()
            )
            - self.n_phi_operator() * self.ng
            - self._kron3(
                self._n_phi_operator(), self._n_zeta_operator(), self._identity_theta()
            )
        )
        disorder_c = -4 * self._disordered_ecj() * self.dCJ * dis_c_opt

        hamiltonian_mat = (
            phi_osc_mat
            + zeta_osc_mat
            + cross_kinetic_mat
            + junction_mat
            + disorder_l
            + disorder_j
            + disorder_c
        )
        native = hamiltonian_mat.tocsc()
        return self.process_hamiltonian(
            native_hamiltonian=native, energy_esys=energy_esys
        )

    def _evals_calc(self, evals_count: int) -> ndarray:
        r"""Diagonalize the Hamiltonian and return the requested eigenvalues.

        Eigenvalues are obtained from a sparse shift-invert
        ``eigsh_safe`` call and post-sorted via :func:`numpy.sort`, since
        Lanczos solvers do not guarantee ordered output.

        Parameters
        ----------
        evals_count:
            number of eigenvalues to return

        Returns
        -------
        Sorted ndarray of the lowest ``evals_count`` eigenvalues.
        """
        hamiltonian_mat = self.hamiltonian()
        evals = utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            return_eigenvectors=False,
            sigma=0.0,
            which="LM",
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> tuple[ndarray, ndarray]:
        r"""Diagonalize the Hamiltonian and return eigenvalues and eigenvectors.

        Parameters
        ----------
        evals_count:
            number of eigenpairs to return

        Returns
        -------
        Tuple ``(evals, evecs)`` containing the sorted eigenvalues and the
        associated eigenvectors as columns of an ndarray.
        """
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            return_eigenvectors=True,
            sigma=0.0,
            which="LM",
        )
        evals, evecs = utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def potential(
        self,
        phi: float | ndarray,
        zeta: float | ndarray,
        theta: float | ndarray,
    ) -> float:
        r"""Return the potential evaluated at :math:`\phi, \zeta, \theta`.

        Parameters
        ----------
        phi:
            value(s) of the phase variable :math:`\phi`
        zeta:
            value(s) of the phase variable :math:`\zeta`
        theta:
            value(s) of the phase variable :math:`\theta`

        Returns
        -------
        Potential energy at the supplied coordinates. Broadcasts when array
        inputs are supplied.
        """
        return (
            self._disordered_el() * (phi * phi)
            + self._disordered_el() * (zeta * zeta)
            - 2 * self.EJ * np.cos(theta) * np.cos(phi + np.pi * self.flux)
            + 2 * self.dEJ * self.EJ * np.sin(phi + np.pi * self.flux) * np.sin(theta)
        )

    def reduced_potential(self, phi: float | ndarray, theta: float | ndarray) -> float:
        r"""Return the reduced potential at :math:`\zeta = 0`.

        Parameters
        ----------
        phi:
            value(s) of the phase variable :math:`\phi`
        theta:
            value(s) of the phase variable :math:`\theta`

        Returns
        -------
        Potential energy at :math:`\zeta = 0`. Broadcasts when array inputs are
        supplied.
        """
        return self.potential(phi, 0, theta)

    def plot_potential(
        self,
        phi_grid: Grid1d | None = None,
        theta_grid: Grid1d | None = None,
        contour_vals: list[float] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        r"""Draw a contour plot of the potential energy at :math:`\zeta = 0`.

        Parameters
        ----------
        phi_grid:
            custom grid for :math:`\phi`; if `None`, uses the default phi grid
        theta_grid:
            custom grid for :math:`\theta`; if `None`, uses the default theta
            grid
        contour_vals:
            optional list of contour level values
        **kwargs:
            additional plotting parameters

        Returns
        -------
        Matplotlib :class:`matplotlib.figure.Figure` and
        :class:`matplotlib.axes.Axes` for the contour plot.
        """
        phi_grid = phi_grid or self._default_phi_grid
        theta_grid = theta_grid or self._default_theta_grid

        y_vals = theta_grid.make_linspace()
        x_vals = phi_grid.make_linspace()
        return plot.contours(
            x_vals,
            y_vals,
            self.reduced_potential,
            contour_vals=contour_vals,
            ylabel=r"$\theta$",
            xlabel=r"$\phi$",
            **kwargs,
        )

    def wavefunction(
        self,
        esys: tuple[ndarray, ndarray] | None = None,
        which: int = 0,
        phi_grid: Grid1d | None = None,
        zeta_grid: Grid1d | None = None,
        theta_grid: Grid1d | None = None,
    ) -> WaveFunctionOnGrid:
        r"""Return a 3D wave function in the :math:`\phi, \zeta, \theta` basis.

        Parameters
        ----------
        esys:
            eigenvalues and eigenvectors as a tuple of ndarrays; if `None`,
            the eigensystem is recomputed
        which:
            index of the desired wave function (default: 0)
        phi_grid:
            custom grid for :math:`\phi`; if `None`, uses the default phi grid
        zeta_grid:
            custom grid for :math:`\zeta`; if `None`, uses the default zeta grid
        theta_grid:
            custom grid for :math:`\theta`; if `None`, uses the default theta
            grid

        Returns
        -------
        :class:`WaveFunctionOnGrid` instance holding the wave-function
        amplitudes on the three-dimensional grid.
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_grid = phi_grid or self._default_phi_grid
        zeta_grid = zeta_grid or self._default_zeta_grid
        theta_grid = theta_grid or self._default_theta_grid

        phi_basis_labels = phi_grid.make_linspace()
        zeta_basis_labels = zeta_grid.make_linspace()
        theta_basis_labels = theta_grid.make_linspace()

        wavefunc_basis_amplitudes = evecs[:, which].reshape(
            self._dim_phi(), self._dim_zeta(), self._dim_theta()
        )
        wavefunc_amplitudes = np.zeros(
            (phi_grid.pt_count, zeta_grid.pt_count, theta_grid.pt_count),
            dtype=np.complex128,
        )
        for i in range(self._dim_phi()):
            for j in range(self._dim_zeta()):
                for k in range(self._dim_theta()):
                    n_phi, n_zeta, n_theta = i, j, k - self.ncut
                    phi_wavefunc_amplitudes = osc.harm_osc_wavefunction(
                        n_phi, phi_basis_labels, self.phi_osc()
                    )
                    zeta_wavefunc_amplitudes = osc.harm_osc_wavefunction(
                        n_zeta, zeta_basis_labels, self.zeta_osc()
                    )
                    theta_wavefunc_amplitudes = (
                        np.exp(-1j * n_theta * theta_basis_labels) / (2 * np.pi) ** 0.5
                    )
                    wavefunc_amplitudes += wavefunc_basis_amplitudes[
                        i, j, k
                    ] * np.tensordot(
                        np.tensordot(
                            phi_wavefunc_amplitudes, zeta_wavefunc_amplitudes, 0
                        ),
                        theta_wavefunc_amplitudes,
                        0,
                    )

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

    def plot_wavefunction(
        self,
        esys: tuple[ndarray, ndarray] | None = None,
        which: int = 0,
        phi_grid: Grid1d | None = None,
        theta_grid: Grid1d | None = None,
        mode: str = "abs",
        zero_calibrate: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        r"""Plot a 2D wave function in the :math:`\theta, \phi` basis at :math:`\zeta = 0`.

        Parameters
        ----------
        esys:
            eigenvalues and eigenvectors as obtained from
            :meth:`eigensys`; if `None`, the eigensystem is recomputed
        which:
            index of the wave function to be plotted (default: 0)
        phi_grid:
            custom grid for :math:`\phi`; if `None`, uses the default phi grid
        theta_grid:
            custom grid for :math:`\theta`; if `None`, uses the default theta
            grid
        mode:
            amplitude-modifier choice from
            :data:`scqubits.core.constants.MODE_FUNC_DICT` (default: ``'abs'``)
        zero_calibrate:
            if `True`, colors are calibrated so that zero amplitude maps to
            the palette's neutral color
        **kwargs:
            additional plotting parameters

        Returns
        -------
        Matplotlib :class:`matplotlib.figure.Figure` and
        :class:`matplotlib.axes.Axes` for the wavefunction plot.
        """
        phi_grid = phi_grid or self._default_phi_grid
        zeta_grid = discretization.Grid1d(0, 0, 1)
        theta_grid = theta_grid or self._default_theta_grid

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
                    [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                    [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                ]
            )
        )
        wavefunc.amplitudes = np.transpose(
            amplitude_modifier(  # type: ignore[operator]
                utils.standardize_phases(
                    wavefunc.amplitudes.reshape(phi_grid.pt_count, theta_grid.pt_count)
                )
            )
        )
        return plot.wavefunction2d(
            wavefunc,
            zero_calibrate=zero_calibrate,
            ylabel=r"$\theta$",
            xlabel=r"$\phi$",
            **kwargs,
        )

    def phi_1_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the phase operator across inductor 1 in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        native = self.zeta_operator() - self.phi_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)  # type: ignore[arg-type]

    def phi_2_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the phase operator across inductor 2 in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        native = -self.zeta_operator() - self.phi_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)  # type: ignore[arg-type]

    def n_1_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the charge difference across junction 1 in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        native = 0.5 * self.n_phi_operator() + 0.5 * (
            self.n_theta_operator() - self.n_zeta_operator()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)  # type: ignore[arg-type]

    def n_2_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the charge difference across junction 2 in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        native = 0.5 * self.n_phi_operator() - 0.5 * (
            self.n_theta_operator() - self.n_zeta_operator()
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)  # type: ignore[arg-type]

    def d_hamiltonian_d_flux(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the derivative of the Hamiltonian with respect to ``flux``.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        phi_flux_term = self._sin_phi_operator() * np.cos(
            self.flux * np.pi
        ) + self._cos_phi_operator() * np.sin(self.flux * np.pi)
        junction_mat = (
            2
            * self.EJ
            * self._kron3(
                phi_flux_term, self._identity_zeta(), self._cos_theta_operator()
            )
            * np.pi
        )

        dis_phi_flux_term = self._cos_phi_operator() * np.cos(
            self.flux * np.pi
        ) - self._sin_phi_operator() * np.sin(self.flux * np.pi)
        dis_junction_mat = (
            2
            * self.dEJ
            * self.EJ
            * self._kron3(
                dis_phi_flux_term, self._identity_zeta(), self._sin_theta_operator()
            )
            * np.pi
        )
        native = junction_mat + dis_junction_mat
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the derivative of the Hamiltonian with respect to ``EJ``.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        phi_flux_term = self._cos_phi_operator() * np.cos(
            self.flux * np.pi
        ) - self._sin_phi_operator() * np.sin(self.flux * np.pi)
        junction_mat = -2 * self._kron3(
            phi_flux_term, self._identity_zeta(), self._cos_theta_operator()
        )

        dis_phi_flux_term = self._sin_phi_operator() * np.cos(
            self.flux * np.pi
        ) + self._cos_phi_operator() * np.sin(self.flux * np.pi)
        dis_junction_mat = (
            2
            * self.dEJ
            * self._kron3(
                dis_phi_flux_term, self._identity_zeta(), self._sin_theta_operator()
            )
        )
        native = junction_mat + dis_junction_mat
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_ng(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the derivative of the Hamiltonian with respect to ``ng``.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns the operator in the native
            (harmonic-oscillator) basis.
            If `True`, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis.
            If `energy_esys = esys`, where ``esys`` is a tuple of two ndarrays
            (eigenvalues and eigenvectors), the operator is returned in the
            energy eigenbasis without recomputing the eigenspectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is an
        ndarray of shape :attr:`truncated_dim` x :attr:`truncated_dim` (or
        m x m if a custom ``esys`` with m eigenvectors is supplied).
        """
        native = (
            4 * self.dCJ * self._disordered_ecj() * self.n_phi_operator()
            - 4
            * self._disordered_ecj()
            * (self.n_theta_operator() - self.ng - self.n_zeta_operator())
        )  # type: ignore[operator]
        return self.process_op(native_op=native, energy_esys=energy_esys)
