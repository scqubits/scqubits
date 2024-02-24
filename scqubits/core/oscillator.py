# oscillator.py
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

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy as sp

from numpy import ndarray
from scipy.special import factorial, pbdv

import scqubits.core.descriptors as descriptors
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers

_default_evals_count = 6


def harm_osc_wavefunction(
    n: int, x: Union[float, ndarray], l_osc: float
) -> Union[float, ndarray]:
    r"""For given quantum number n=0,1,2,... return the value of the harmonic
    oscillator wave function :math:`\psi_n(x) = N H_n(x/l_{osc}) \exp(-x^2/2l_\text{
    osc})`, N being the proper normalization factor.

    Directly uses `scipy.special.pbdv` (implementation of the parabolic cylinder
    function) to mitigate numerical stability issues with the more commonly used
    expression in terms of a Gaussian and a Hermite polynomial factor.

    Parameters
    ----------
    n:
        index of wave function, n=0 is ground state
    x:
        coordinate(s) where wave function is evaluated
    l_osc:
        oscillator length, defined via <0|x^2|0> = l_osc^2/2

    Returns
    -------
        value of harmonic oscillator wave function
    """
    result = pbdv(n, np.sqrt(2.0) * x / l_osc) / np.sqrt(
        l_osc * np.sqrt(np.pi) * factorial(n)
    )
    return result[0]


def convert_to_E_osc(E_kin: float, E_pot: float) -> float:
    r"""Returns the oscillator energy given a harmonic Hamiltonian of the form
    :math:`H=\frac{1}{2}E_{\text{kin}}p^2 + \frac{1}{2}E_{\text{pot}}x^2`"""
    return np.sqrt(E_kin * E_pot)


def convert_to_l_osc(E_kin: float, E_pot: float) -> float:
    r"""Returns the oscillator length given a harmonic Hamiltonian of the form
    :math:`H=\frac{1}{2}E_{\text{kin}}p^2 + \frac{1}{2}E_{\text{pot}}x^2`"""
    return (E_kin / E_pot) ** (1 / 4)


# -Oscillator class-------------------------------------------------------------------


class Oscillator(base.QuantumSystem, serializers.Serializable):
    r"""Class representing a harmonic oscillator/resonator governed by a Hamiltonian
    :math:`H=E_\text{osc} a^{\dagger} a`, with :math:`a` being the annihilation
    operator.

    Parameters
    ----------
    E_osc:
        energy of the oscillator
    l_osc:
        oscillator length (required to define phi_operator and n_operator)
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """

    E_osc = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    l_osc = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        E_osc: float,
        l_osc: Union[float, None] = None,
        truncated_dim: int = _default_evals_count,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.truncated_dim: int = truncated_dim  # type:ignore
        self.l_osc: Union[None, float] = l_osc  # type:ignore
        self.E_osc = E_osc
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/oscillator.jpg"
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {"E_osc": 5.0, "l_osc": 1, "truncated_dim": _default_evals_count}

    def eigenvals(self, evals_count: int = _default_evals_count) -> ndarray:
        """Returns array of eigenvalues.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues (default value = 6)
        """
        evals = [self.E_osc * n for n in range(evals_count)]
        return np.asarray(evals)

    def eigensys(
        self, evals_count: int = _default_evals_count
    ) -> Tuple[ndarray, ndarray]:
        """Returns array of eigenvalues and eigenvectors

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues (default value = 6)
        """
        evals_count = evals_count or _default_evals_count
        evecs = np.zeros(shape=(self.truncated_dim, evals_count), dtype=np.float_)
        np.fill_diagonal(evecs, 1.0)

        return self.eigenvals(evals_count=evals_count), evecs

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return self.truncated_dim

    def creation_operator(self) -> ndarray:
        """Returns the creation operator"""
        return op.creation(self.truncated_dim)

    def annihilation_operator(self) -> ndarray:
        """Returns the annihilation operator"""
        return op.annihilation(self.truncated_dim)

    def matrixelement_table(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError(
            "The Oscillator class does not implement the matrixelement_table method."
        )

    def phi_operator(self) -> ndarray:
        r"""Returns the phase operator defined as
        :math:`l_\text{osc} (a + a^{\dagger})/\sqrt{2}`, with :math:`a` representing
        an annihilation operator, and :math:`l_\text{osc}` the oscillator length.
        """
        if self.l_osc is None:
            raise ValueError(
                "Variable l_osc has to be set to something other than None\n"
                + "in order to use the phi_operator() method. This can be done by either\n"
                + "passing it to the class constructor, or by setting it afterwards."
            )
        a = op.annihilation(self.truncated_dim)
        return self.l_osc / np.sqrt(2) * (a + a.T)

    def n_operator(self) -> ndarray:
        r"""Returns the charge-number n operator defined as
        :math:`i (a^{\dagger} - a)/ ( \sqrt{2} l_\text{osc})`, with :math:`a` representing
        an annihilation operator, and :math:`l_\text{osc}` the oscillator length.
        """

        if self.l_osc is None:
            raise ValueError(
                "Variable l_osc has to be set to something other than None\n"
                + "in order to use the n_operator() method. This can be done by either\n"
                + "passing it to the class constructor, or by setting it afterwards."
            )
        a = op.annihilation(self.truncated_dim)
        return 1.0j / (self.l_osc * np.sqrt(2)) * (a.T - a)


# -KerrOscillator class-------------------------------------------------------------------


class KerrOscillator(Oscillator, serializers.Serializable):
    r"""Class representing a nonlinear Kerr oscillator/resonator governed by a Hamiltonian
    :math:`H_\text{Kerr}=E_\text{osc} a^{\dagger} a - K a^{\dagger} a^{\dagger} a a`, with :math:`a`
    being the annihilation operator.

    Parameters
    ----------
    E_osc:
        energy of harmonic term
    K:
        energy of the Kerr term
    l_osc:
        oscillator length (used to define phi_operator and n_operator)
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """

    K = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        E_osc: float,
        K: float,
        l_osc: Union[float, None] = None,
        truncated_dim: int = _default_evals_count,
        id_str: Optional[str] = None,
    ) -> None:
        self.K: float = K  # type:ignore

        super().__init__(
            E_osc=E_osc,
            l_osc=l_osc,
            truncated_dim=truncated_dim,
            id_str=id_str,
        )

        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/KerrOscillator.jpg"
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "E_osc": 5.0,
            "K": 0.05,
            "l_osc": 1,
            "truncated_dim": _default_evals_count,
        }

    def eigenvals(self, evals_count: int = _default_evals_count) -> ndarray:
        """Returns array of eigenvalues.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues (default value = 6)
        """
        evals = [(self.E_osc + self.K) * n - self.K * n**2 for n in range(evals_count)]
        return np.asarray(evals)
