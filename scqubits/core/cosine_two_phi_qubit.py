# cosine_two_phi_qubit.py
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
import scqubits.core.harmonic_osc as osc
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.core.units as units
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.noise import NOISE_PARAMS, NoisySystem, calc_therm_ratio
from scqubits.core.storage import WaveFunctionOnGrid


# -Cosine two phi qubit noise class


class NoisyCosineTwoPhiQubit(NoisySystem, ABC):
    @abstractmethod
    def phi_1_operator(self) -> ndarray:
        pass

    @abstractmethod
    def phi_2_operator(self) -> ndarray:
        pass

    @abstractmethod
    def N_1_operator(self) -> ndarray:
        pass

    @abstractmethod
    def N_2_operator(self) -> ndarray:
        pass

    @abstractmethod
    def n_zeta_operator(self) -> ndarray:
        pass

    def t1_inductive(
        self,
        i: int = 1,
        j: int = 0,
        Q_ind: Union[float, Callable] = None,
        T: float = NOISE_PARAMS["T"],
        total: bool = True,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        :math:`T_1` due to inductive dissipation in a superinductor.

        References: nguyen et al (2019), Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
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
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.
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

        def spectral_density1(omega):
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

        noise_op1 = self.phi_1_operator()  # type: ignore

        def spectral_density2(omega):
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

        noise_op2 = self.phi_2_operator()  # type: ignore

        if get_rate:
            return self.t1(
                i=i,
                j=j,
                noise_op=noise_op1,
                spectral_density=spectral_density1,
                total=total,
                esys=esys,
                get_rate=get_rate,
                **kwargs
            ) + self.t1(
                i=i,
                j=j,
                noise_op=noise_op2,
                spectral_density=spectral_density2,
                total=total,
                esys=esys,
                get_rate=get_rate,
                **kwargs
            )
        else:
            return 1 / (
                1
                / self.t1(
                    i=i,
                    j=j,
                    noise_op=noise_op1,
                    spectral_density=spectral_density1,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    **kwargs
                )
                + 1
                / self.t1(
                    i=i,
                    j=j,
                    noise_op=noise_op2,
                    spectral_density=spectral_density2,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    **kwargs
                )
            )

    def t1_capacitive(
        self,
        i: int = 1,
        j: int = 0,
        Q_cap: Union[float, Callable] = None,
        T: float = NOISE_PARAMS["T"],
        total: bool = True,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        :math:`T_1` due to dielectric dissipation in the Jesephson junction capacitances.

        References:  nguyen et al (2019), Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Q_cap
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
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

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

        def spectral_density1(omega):
            therm_ratio = calc_therm_ratio(omega, T)
            s1 = (
                2
                * 8
                * self.ECJ
                / (1 - self.dC)
                / q_cap_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s1 *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s1

        def spectral_density2(omega):
            therm_ratio = calc_therm_ratio(omega, T)
            s2 = (
                2
                * 8
                * self.ECJ
                / (1 + self.dC)
                / q_cap_fun(omega)
                * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
                / (1 + np.exp(-therm_ratio))
            )
            s2 *= (
                2 * np.pi
            )  # We assume that system energies are given in units of frequency
            return s2

        noise_op1 = self.N_1_operator()  # type: ignore
        noise_op2 = self.N_2_operator()  # type: ignore

        if get_rate:
            return self.t1(
                i=i,
                j=j,
                noise_op=noise_op1,
                spectral_density=spectral_density1,
                total=total,
                esys=esys,
                get_rate=get_rate,
                **kwargs
            ) + self.t1(
                i=i,
                j=j,
                noise_op=noise_op2,
                spectral_density=spectral_density2,
                total=total,
                esys=esys,
                get_rate=get_rate,
                **kwargs
            )
        else:
            return 1 / (
                1
                / self.t1(
                    i=i,
                    j=j,
                    noise_op=noise_op1,
                    spectral_density=spectral_density1,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    **kwargs
                )
                + 1
                / self.t1(
                    i=i,
                    j=j,
                    noise_op=noise_op2,
                    spectral_density=spectral_density2,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    **kwargs
                )
            )

    def t1_purcell(
        self,
        i: int = 1,
        j: int = 0,
        Q_cap: Union[float, Callable] = None,
        T: float = NOISE_PARAMS["T"],
        total: bool = True,
        esys: Tuple[ndarray, ndarray] = None,
        get_rate: bool = False,
        **kwargs
    ) -> float:
        r"""
        :math:`T_1` due to dielectric dissipation in the shunt capacitances.

        References:  Nguyen et al (2019), Smith et al (2020)

        Parameters
        ----------
        i: int >=0
            state index that along with j defines a transition (i->j)
        j: int >=0
            state index that along with i defines a transition (i->j)
        Q_cap
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
            decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.

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

        def spectral_density(omega):
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

        noise_op = self.n_zeta_operator()  # type: ignore

        return self.t1(
            i=i,
            j=j,
            noise_op=noise_op,
            spectral_density=spectral_density,
            total=total,
            esys=esys,
            get_rate=get_rate,
            **kwargs
        )


# -Cosine two phi qubit ----------------------------------------------------------------------------------
class CosineTwoPhiQubit(
    base.QubitBaseClass, serializers.Serializable, NoisyCosineTwoPhiQubit
):
    r"""Cosine Two Phi Qubit

    | [1] Smith et al., NPJ Quantum Inf. 6, 8 (2020) http://www.nature.com/articles/s41534-019-0231-2

    .. math::
    # TODO:
        H = 4E_\text{C}[2n_\phi^2+\frac{1}{2}(n_\varphi-N_\text{g}-n_\theta)^2+xn_\theta^2]
                           +E_\text{L}(\frac{1}{4}\phi^2+\theta^2)
                           -2E_\text{J}\cos(\varphi)\cos(\frac{\phi}{2}+\frac{\varphi_\text{ext}}{2})

    The Hamiltonian is formed with harmonic basis for :math:`\phi,\theta` variables and charge basis for :math:`\varphi`
    variable.

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
    dC:
        disorder in junction charging energy, i.e., `ECJ / (1 \pm dC)`
    dL:
        disorder in inductive energy, i.e., `EL / (1 \pm dL)`
    dJ:
        disorder in junction energy, i.e., `EJ * (1 \pm dJ)`
    flux:
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    ng:
        offset charge
    n_cut:
        number of charge states, `-n_cut <= n_\varphi <= n_cut`
    zeta_cut:                                                                                              
        number of harmonic oscillator basis for `\zeta` variable
    phi_cut:
        number of harmonic oscillator basis for `\phi` variable
    """
    EJ = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    ECJ = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    dC = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    dL = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    dJ = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    n_cut = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    zeta_cut = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")
    phi_cut = descriptors.WatchedProperty("QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        ECJ: float,
        EL: float,
        EC: float,
        dL: float,
        dC: float,
        dJ: float,
        flux: float,
        ng: float,
        n_cut: int,
        zeta_cut: int,
        phi_cut: int,
    ) -> None:
        self.EJ = EJ
        self.ECJ = ECJ
        self.EL = EL
        self.EC = EC
        self.dL = dL
        self.dC = dC
        self.dJ = dJ
        self.flux = flux
        self.ng = ng
        self.n_cut = n_cut
        self.zeta_cut = zeta_cut
        self.phi_cut = phi_cut
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._default_phi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 400)
        self._default_zeta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self._default_theta_grid = discretization.Grid1d(-2 * np.pi, 3 * np.pi, 100)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "qubit_img/cosine_two_phi_qubit.jpg",  ##                                        ASK ABOUT
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 15.0,
            "ECJ": 2.0,
            "EL": 1.0,
            "EC": 0.04,
            "dC": 0.0,
            "dL": 0.6,
            "dJ": 0.0,
            "flux": 0.5,
            "ng": 0.0,
            "n_cut": 2,
            "zeta_cut": 5,
            "phi_cut": 3,
        }

    @classmethod
    def create(cls) -> "CosineTwoPhiQubit":
        init_params = cls.default_params()
        cosinetwophiqubit = cls(**init_params)
        cosinetwophiqubit.widget()
        return cosinetwophiqubit

    def supported_noise_channels(self) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_inductive",
            "t1_purcell",
        ]

    def dim_phi(self) -> int:
        """
        Returns
        -------
            Hilbert space dimension of `phi` degree of freedom"""
        return self.phi_cut

    def dim_zeta(self) -> int:
        """
        Returns
        -------
            Hilbert space dimension of `zeta` degree of freedom"""
        return self.zeta_cut

    def dim_theta(self) -> int:
        """
        Returns
        -------
            Hilbert space dimension of `theta` degree of freedom"""
        return 2 * self.n_cut + 1

    def hilbertdim(self) -> int:
        """
        Returns
        -------
            total Hilbert space dimension"""
        return self.dim_phi() * self.dim_zeta() * self.dim_theta()

    def _disordered_el(self) -> float:
        """
        Returns
        -------
            inductive energy renormalized by with disorder"""
        return self.EL / (1 - self.dL ** 2)

    def _disordered_ecj(self) -> float:
        """
        Returns
        -------
            junction capacitance energy renormalized by with disorder"""
        return self.ECJ / (1 - self.dC ** 2)

    def phi_osc(self) -> float:
        """
        Returns
        -------
            oscillator strength of `phi` degree of freedom"""
        return (2 * self._disordered_ecj() / self._disordered_el()) ** 0.25

    def zeta_osc(self) -> float:
        """
        Returns
        -------
            oscillator strength of `zeta` degree of freedom"""
        return (4 * self.EC / self._disordered_el()) ** 0.25

    def phi_plasma(self) -> float:
        """
        Returns
        -------
            plasma oscillation frequency of `phi` degree of freedom"""
        return math.sqrt(8.0 * self._disordered_el() * self._disordered_ecj())

    def zeta_plasma(self) -> float:
        """
        Returns
        -------
            plasma oscillation frequency of `zeta` degree of freedom"""
        return math.sqrt(16.0 * self.EC * self._disordered_el())

    def _phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `phi` operator in the harmonic oscillator basis"""
        dimension = self.dim_phi()
        return (
            (op.creation(dimension) + op.annihilation(dimension))
            * self.phi_osc()
            / math.sqrt(2)
        )

    def phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `phi` operator in total Hilbert space"""
        return self._kron3(
            self._phi_operator(), self._identity_zeta(), self._identity_theta()
        )

    def _n_phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `n_\phi` operator in the harmonic oscillator basis"""
        dimension = self.dim_phi()
        return (
            1j
            * (op.creation(dimension) - op.annihilation(dimension))
            / (self.phi_osc() * math.sqrt(2))
        )

    def n_phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `n_phi` operator in total Hilbert space"""
        return self._kron3(
            self._n_phi_operator(), self._identity_zeta(), self._identity_theta()
        )

    def _zeta_operator(self) -> ndarray:
        """
        Returns
        -------
            `zeta` operator in the harmonic oscillator basis"""
        dimension = self.dim_zeta()
        return (
            (op.creation(dimension) + op.annihilation(dimension))
            * self.zeta_osc()
            / math.sqrt(2)
        )

    def zeta_operator(self) -> ndarray:
        """
        Returns
        -------
            `zeta` operator in total Hilbert space"""
        return self._kron3(
            self._identity_phi(), self._zeta_operator(), self._identity_theta()
        )

    def _n_zeta_operator(self) -> ndarray:
        """
        Returns
        -------
            `n_\zeta` operator in the harmonic oscillator basis"""
        dimension = self.dim_zeta()
        return (
            1j
            * (op.creation(dimension) - op.annihilation(dimension))
            / (self.zeta_osc() * math.sqrt(2))
        )

    def n_zeta_operator(self) -> ndarray:
        """
        Returns
        -------
            `n_zeta` operator in total Hilbert space"""
        return self._kron3(
            self._identity_phi(), self._n_zeta_operator(), self._identity_theta()
        )

    def _exp_i_phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `e^{i*phi}` operator in the  harmonic oscillator basis"""
        exponent = 1j * self._phi_operator()
        return sp.linalg.expm(exponent)

    def _cos_phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `cos phi` operator in the harmonic oscillator basis"""
        cos_phi_op = 0.5 * self._exp_i_phi_operator()
        cos_phi_op += cos_phi_op.conj().T
        return cos_phi_op

    def _sin_phi_operator(self) -> ndarray:
        """
        Returns
        -------
            `sin phi/2` operator in the LC harmonic oscillator basis"""
        sin_phi_op = -1j * 0.5 * self._exp_i_phi_operator()
        sin_phi_op += sin_phi_op.conj().T
        return sin_phi_op

    def _n_theta_operator(self) -> ndarray:
        """
        Returns
        -------
            `n_theta` operator in the charge basis"""
        diag_elements = np.arange(-self.n_cut, self.n_cut + 1)
        return np.diag(diag_elements)

    def n_theta_operator(self) -> ndarray:
        """
        Returns
        -------
            `n_theta` in the total Hilbert space"""
        return self._kron3(
            self._identity_phi(), self._identity_zeta(), self._n_theta_operator()
        )

    def _exp_i_theta_operator(self) -> ndarray:
        """Returns operator :math:`e^{i\\theta}` in the charge basis"""
        dimension = self.dim_theta()
        entries = np.repeat(1.0, dimension - 1)
        exp_op = np.diag(entries, -1)
        return exp_op

    def _cos_theta_operator(self) -> ndarray:
        """Returns operator :math:`\\cos \\theta` in the charge basis"""
        cos_op = 0.5 * self._exp_i_theta_operator()
        cos_op += cos_op.T
        return cos_op

    def _sin_theta_operator(self) -> ndarray:
        """Returns operator :math:`\\sin \\varphi` in the charge basis"""
        sin_op = -1j * 0.5 * self._exp_i_theta_operator()
        sin_op += sin_op.conjugate().T
        return sin_op

    def _kron3(self, mat1, mat2, mat3) -> ndarray:
        """
        Returns Kronecker product of three matrices
        """
        return np.kron(np.kron(mat1, mat2), mat3)

    def _identity_phi(self) -> ndarray:
        """
        Returns Identity operator acting only on the :math:`\phi` Hilbert subspace.
        """
        dimension = self.dim_phi()
        return np.eye(dimension)

    def _identity_zeta(self) -> ndarray:
        """
        Returns Identity operator acting only on the :math:`\zeta` Hilbert subspace.
        """
        dimension = self.dim_zeta()
        return np.eye(dimension)

    def _identity_theta(self) -> ndarray:
        """
        Returns Identity operator acting only on the :math:`\theta` Hilbert subspace.
        """
        dimension = self.dim_theta()
        return np.eye(dimension)

    def total_identity(self) -> ndarray:
        """
        Returns Identity operator acting only on the total Hilbert space.
        """
        return self._kron3(
            self._identity_phi(), self._identity_zeta(), self._identity_theta()
        )

    def hamiltonian(self) -> ndarray:
        """
        Returns Cosine two phi qubit Hamiltonian
        """
        phi_osc_mat = self._kron3(
            op.number(self.dim_phi(), self.phi_plasma()),
            self._identity_zeta(),
            self._identity_theta(),
        )

        zeta_osc_mat = self._kron3(
            self._identity_phi(),
            op.number(self.dim_zeta(), self.zeta_plasma()),
            self._identity_theta(),
        )

        n_theta_ng_mat = self.n_theta_operator() - self.total_identity() * self.ng
        cross_kinetic_mat = (
            2
            * self._disordered_ecj()
            * np.matmul(
                n_theta_ng_mat - self.n_zeta_operator(),
                n_theta_ng_mat - self.n_zeta_operator(),
            )
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

        hamiltonian_mat = phi_osc_mat + zeta_osc_mat + cross_kinetic_mat + junction_mat
        return hamiltonian_mat

    def disorder(self) -> ndarray:
        """
        Return disordered part of Hamiltonian
        """
        disorder_l = (
            -2
            * self._disordered_el()
            * self.dL
            * self._kron3(
                self._phi_operator(), self._zeta_operator(), self._identity_theta()
            )
        )

        phi_flux_term = self._sin_phi_operator() * np.cos(
            self.flux * np.pi
        ) + self._cos_phi_operator() * np.sin(self.flux * np.pi)
        disorder_j = (
            2
            * self.EJ
            * self.dJ
            * self._kron3(
                phi_flux_term, self._identity_zeta(), self._sin_theta_operator()
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
        disorder_c = -4 * self._disordered_ecj() * self.dC * dis_c_opt
        return disorder_l + disorder_j + disorder_c

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals = sp.linalg.eigh(
            hamiltonian_mat, eigvals=(0, evals_count - 1), eigvals_only=True
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals, evecs = sp.linalg.eigh(
            hamiltonian_mat, eigvals=(0, evals_count - 1), eigvals_only=False
        )
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def potential(self, theta, phi) -> float:
        """
        potential evaluated at `phi, theta`, with `zeta=0`

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`
        theta: float or ndarray
            float value of the phase variable `theta`
        """
        return (
            self._disordered_el() * (phi * phi)
            - 2 * self.EJ * np.cos(theta) * np.cos(phi + np.pi * self.flux)
            + 2 * self.dJ * self.EJ * np.sin(phi + np.pi * self.flux) * np.sin(theta)
        )

    def plot_potential(
        self, phi_grid=None, theta_grid=None, contour_vals=None, **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use
            self._default_theta_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        phi_grid = phi_grid or self._default_phi_grid
        theta_grid = theta_grid or self._default_theta_grid

        x_vals = theta_grid.make_linspace()
        y_vals = phi_grid.make_linspace()
        if "figsize" not in kwargs:
            kwargs["figsize"] = (4, 4)
        return plot.contours(
            x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs
        )

    def _tensor_index(self, index_phi, index_zeta, index_theta) -> int:
        """
        Return the index of the coefficient of the wavefunction, corresponding to the indices of phi, zeta, and theta
        """
        return (
            index_phi * self.dim_zeta() + index_zeta
        ) * self.dim_theta() + index_theta

    def _tensor_index_inv(self, index_evec) -> Tuple[int, int, int]:
        """
        Return the indices of phi, zeta, and theta corresponding to the index of the
        coefficient of the wavefunction
        """
        index_theta = index_evec % self.dim_theta()
        index_temp = index_evec // self.dim_theta()
        index_zeta = index_temp % self.dim_zeta()
        index_phi = index_temp // self.dim_zeta()
        return index_phi, index_zeta, index_theta

    def wavefunction(
        self, esys=None, which=0, phi_grid=None, zeta_grid=None, theta_grid=None
    ) -> WaveFunctionOnGrid:
        """
        Return a 3D wave function in phi, zeta, theta basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        zeta_grid: Grid1d, option
            used for setting a custom grid for zeta; if None use
            self._default_zeta_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use
            self._default_theta_grid
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

        wavefunc_basis_amplitudes = np.reshape(evecs[:, which], self.hilbertdim())
        wavefunc_amplitudes = np.zeros(
            (phi_grid.pt_count, zeta_grid.pt_count, theta_grid.pt_count),
            dtype=np.complex_,
        )
        for n in range(self.hilbertdim()):
            n_phi, n_zeta, n_theta = self._tensor_index_inv(n)
            num_theta = n_theta - self.n_cut
            phi_wavefunc_amplitudes = osc.harm_osc_wavefunction(
                n_phi, phi_basis_labels, self.phi_osc()
            )
            zeta_wavefunc_amplitudes = osc.harm_osc_wavefunction(
                n_zeta, zeta_basis_labels, self.zeta_osc()
            )
            theta_wavefunc_amplitudes = (
                np.exp(-1j * num_theta * theta_basis_labels) / (2 * np.pi) ** 0.5
            )
            wavefunc_amplitudes += wavefunc_basis_amplitudes[n] * np.tensordot(
                np.tensordot(phi_wavefunc_amplitudes, zeta_wavefunc_amplitudes, 0),
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
        esys=None,
        which=0,
        phi_grid=None,
        theta_grid=None,
        mode="abs",
        zero_calibrate=True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plots 2D wave function in `theta` and `phi` basis, at `zeta` = 0

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        theta_grid: Grid1d, option
            used for setting a custom grid for theta; if None use
            self._default_theta_grid
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

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
        return plot.wavefunction2d(
            wavefunc,
            zero_calibrate=zero_calibrate,
            xlabel=r"$\theta$",
            ylabel=r"$\phi$",
            **kwargs
        )

    def phi_1_operator(self) -> ndarray:
        """
        Returns
        -------
            operator represents phase across inductor 1
        """
        return self.zeta_operator() - self.phi_operator()

    def phi_2_operator(self) -> ndarray:
        """
        Returns
        -------
            operator represents phase across inductor 2
        """
        return -self.zeta_operator() - self.phi_operator()

    def N_1_operator(self) -> ndarray:
        """
        Returns
        -------
            operator represents charge difference across junction 1
        """
        return 0.5 * self.n_phi_operator() + 0.5 * (
            self.n_theta_operator() - self.n_zeta_operator()
        )

    def N_2_operator(self) -> ndarray:
        """
        Returns
        -------
            operator represents charge difference across junction 2
        """
        return 0.5 * self.n_phi_operator() - 0.5 * (
            self.n_theta_operator() - self.n_zeta_operator()
        )

    def d_hamiltonian_d_flux(self) -> ndarray:
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
            * self.dJ
            * self.EJ
            * self._kron3(
                dis_phi_flux_term, self._identity_zeta(), self._sin_theta_operator()
            )
            * np.pi
        )
        return junction_mat + dis_junction_mat

    def d_hamiltonian_d_EJ(self) -> ndarray:
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
            * self.dJ
            * self._kron3(
                dis_phi_flux_term, self._identity_zeta(), self._sin_theta_operator()
            )
        )
        return junction_mat + dis_junction_mat

    def d_hamiltonian_d_ng(self) -> ndarray:
        return 4 * self.dC * self._disordered_ecj() * self.n_phi_operator() - 4 * self._disordered_ecj() * (
            self.n_theta_operator() - self.ng - self.n_zeta_operator()
        )
