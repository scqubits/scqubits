# fluxonium.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import cmath
import math

import numpy as np
import scipy as sp

import scqubits.core.operators as op
from scqubits.core.data_containers import WaveFunction
from scqubits.core.harmonic_osc import harm_osc_wavefunction
from scqubits.core.qubit_base import QubitBaseClass1d


# —Fluxonium qubit ————————————————————————

class Fluxonium(QubitBaseClass1d):
    r"""Class for the fluxonium qubit. Hamiltonian
    :math:`H_\text{fl}=-4E_\text{C}\partial_\phi^2-E_\text{J}\cos(\phi-\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2`
    is represented in dense form. The employed basis is the EC-EL harmonic oscillator basis. The cosine term in the
    potential is handled via matrix exponentiation. Initialize with, for example::

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
    truncated_dim: int, optional
        desired dimension of the truncated quantum system
    """

    def __init__(self, EJ, EC, EL, flux, cutoff, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux = flux
        self.cutoff = cutoff
        self.truncated_dim = truncated_dim
        self._sys_type = 'fluxonium'
        self._evec_dtype = np.float_
        self._default_var_range = (-4.5*np.pi, 4.5*np.pi)
        self._default_var_count = 151

    def phi_osc(self):
        """
        Returns
        -------
        float
            Returns oscillator length for the LC oscillator composed of the fluxonium inductance and capacitance.
        """
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def omega_p(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency.
        """
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation frequency

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.hilbertdim()
        return (op.creation(dimension) + op.annihilation(dimension)) * self.phi_osc() / math.sqrt(2)

    def n_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n = - i d/d\\phi` operator in the LC harmonic oscillator basis
        """
        dimension = self.hilbertdim()
        return 1j * (op.creation(dimension) - op.annihilation(dimension)) / (self.phi_osc() * math.sqrt(2))

    def exp_i_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\phi}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self.phi_operator()
        return sp.linalg.expm(exponent)

    def cos_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi` operator in the LC harmonic oscillator basis
        """
        cos_phi_op = 0.5 * self.exp_i_phi_operator()
        cos_phi_op += cos_phi_op.conjugate().T
        return cos_phi_op

    def sin_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi` operator in the LC harmonic oscillator basis
        """
        sin_phi_op = -1j * 0.5 * self.exp_i_phi_operator()
        sin_phi_op += sin_phi_op.conjugate().T
        return sin_phi_op

    def hamiltonian(self):  # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harmonic-oscillator basis, following Zhu et al., PRB 87, 024510 (2013)

        Returns
        -------
        ndarray
        """
        dimension = self.hilbertdim()
        diag_elements = [i * self.omega_p() for i in range(dimension)]
        lc_osc_matrix = np.diag(diag_elements)

        exp_matrix = self.exp_i_phi_operator() * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = 0.5 * (exp_matrix + exp_matrix.conjugate().T)

        hamiltonian_mat = lc_osc_matrix - self.EJ * cos_matrix
        return np.real(hamiltonian_mat)  # use np.real to remove rounding errors from matrix exponential,
        # fluxonium Hamiltonian in harm. osc. basis is real-valued

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension."""
        return self.cutoff

    def potential(self, phi):
        """Fluxonium potential evaluated at `phi`.

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`

        Returns
        -------
        float or ndarray
        """
        return 0.5 * self.EL * phi * phi - self.EJ * np.cos(phi + 2.0 * np.pi * self.flux)

    def wavefunction(self, esys, which=0, phi_range=None, phi_count=None):
        """Returns a fluxonium wave function in `phi` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (Default value = 0)
        phi_range: None or tuple(float,float), optional
             custom boundaries of `phi` values range
        phi_count: int
             number of points in the specified `phi` interval (Default value = 251)

        Returns
        -------
        WaveFunction object
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys
        dim = self.hilbertdim()

        phi_range, phi_count = self.try_defaults(phi_range, phi_count)

        phi_basis_labels = np.linspace(phi_range[0], phi_range[1], phi_count)
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros(phi_count, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * harm_osc_wavefunction(n, phi_basis_labels,
                                                                                                phi_osc)
        return WaveFunction(basis_labels=phi_basis_labels, amplitudes=phi_wavefunc_amplitudes, energy=evals[which])
