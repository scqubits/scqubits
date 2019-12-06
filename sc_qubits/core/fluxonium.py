# fluxonium.py
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################

import cmath
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import sparse

import sc_qubits.core.operators as op
import sc_qubits.utils.constants as constants
import sc_qubits.utils.plotting as plot

from sc_qubits.core.qubit_base import QubitBaseClass
from sc_qubits.core.data_containers import WaveFunction
from sc_qubits.core.harmonic_osc import harm_osc_wavefunction
from sc_qubits.utils.spectrum_utils import extract_phase

# —Fluxonium qubit ————————————————————————

class Fluxonium(QubitBaseClass):
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
        self._sys_type = 'Fluxonium qubit'

    def phi_osc(self):
        """Returns oscillator length for the LC oscillator by fluxonium inductance and capacitance."""
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def omega_p(self):
        """Returns the plasma oscillation frequency."""
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation frequency

    def phi_operator(self):
        """Returns the phi operator in the LC harmonic oscillator basis"""
        dimension = self.hilbertdim()
        return (op.creation(dimension) + op.annihilation(dimension)) * self.phi_osc() / math.sqrt(2)

    def n_operator(self):
        """Returns the :math:`n = - i d/d\\phi` operator in the LC harmonic oscillator basis"""
        dimension = self.hilbertdim()
        return 1j * (op.creation(dimension) - op.annihilation(dimension)) / (self.phi_osc() * math.sqrt(2))

    def hamiltonian(self):  # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harmonic-oscillator basis, following Zhu et al., PRB 87, 024510 (2013)

        Returns
        -------
        sparse.dia_matrix
        """
        dimension = self.hilbertdim()
        diag_elements = [i * self.omega_p() for i in range(dimension)]
        lc_osc_matrix = np.diagflat(diag_elements)

        exponent = 1j * self.phi_operator()
        exp_matrix = 0.5 * sp.linalg.expm(exponent) * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = exp_matrix + np.conj(exp_matrix.T)

        hamiltonian_mat = lc_osc_matrix - self.EJ * cos_matrix
        return np.real(hamiltonian_mat)  # use np.real to remove rounding errors from matrix exponential

    def hilbertdim(self):
        """Returns the Hilbert space dimension."""
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

    def wavefunction(self, esys, which=0, phi_range=(-6 * np.pi, 6 * np.pi), phi_points=251):
        """Returns a fluxonium wave function in `phi` basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
             index of desired wave function (Default value = 0)
        phi_range: tuple of float, optional
             boundaries of `phi` values range  (Default value = `(-6 * np.pi, 6 * np.pi)`)
        phi_points: int
             number of points in the specified `phi` interval (Default value = 251)

        Returns
        -------
        WaveFunction object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys

        dim = self.hilbertdim()
        phi_basis_labels = np.linspace(phi_range[0], phi_range[1], phi_points)
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros(phi_points, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * harm_osc_wavefunction(n, phi_basis_labels,
                                                                                                phi_osc)
        return WaveFunction(phi_basis_labels, phi_wavefunc_amplitudes, energy=evals[which])

    def plot_wavefunction(self, esys, which=(0,), phi_range=(-6 * np.pi, 6 * np.pi), yrange=None, mode='abs_sqr',
                          scaling= None, phi_points=251, filename=None):
        """Plot phase-basis wave function(s).

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int or tuple(i1, i2, ...), optional
            single index or tuple of integers indexing the wave function(s) to be plotted (Default value = (0)
        phi_range: tuple(float, float), optional
            phi range to be plotted (Default value = (-6 * np.pi, 6 * np.pi))
        yrange: None or tuple(float, float), optional
            yrange to be plotted (Default value = None)
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (Default value = 'abs_sqr')
        scaling: float, optional
            custom choice for scaling of wave functions
        phi_points: int, optional
            number of points on the x-axis (resolution) (Default value = 251)

        Returns
        -------
        Figure, Axes
        """
        modefunction = constants.MODE_FUNC_DICT[mode]

        if isinstance(which, int):
            index_tuple = (which,)
        else:
            index_tuple = which

        if scaling is None:
            scale = 5 * self.EJ
        else:
            scale = scaling

        fig = plt.figure()
        ax = fig.add_subplot(111)


        for wavefunc_index in index_tuple:
            phi_wavefunc = self.wavefunction(esys, wavefunc_index, phi_range, phi_points)
            phase = extract_phase(phi_wavefunc.amplitudes)
            phi_wavefunc.amplitudes *= cmath.exp(-1j * phase)
            if np.sum(phi_wavefunc.amplitudes) < 0:
                phi_wavefunc.amplitudes *= -1.0

            phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)
            plot.wavefunction1d(phi_wavefunc, potential_vals=self.potential(phi_wavefunc.basis_labels),
                                offset=phi_wavefunc.energy, scaling=scale, xlabel='phi', yrange=yrange,
                                fig_ax=(fig, ax), filename=filename)
        return fig, ax