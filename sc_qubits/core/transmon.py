# transmon.py
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

import sc_qubits.utils.constants as constants
import sc_qubits.utils.plotting as plot

from sc_qubits.core.qubit_base import QubitBaseClass
from sc_qubits.core.data_containers import WaveFunction
from sc_qubits.utils.spectrum_utils import extract_phase


# —Cooper pair box / transmon———————————————————————————————————————————————————————————————————————————————————————————

class Transmon(QubitBaseClass):
    r"""Class for the Cooper-pair-box and transmon qubit. The Hamiltonian is represented in dense form in the number
    basis, :math:`H_\text{CPB}=4E_\text{C}(\hat{n}-n_g)^2+\frac{E_\text{J}}{2}(|n\rangle\langle n+1|+\text{h.c.})`.
    Initialize with, for example::

        Transmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)

    Parameters
    ----------
    EJ: float
       Josephson energy
    EC: float
        charging energy
    ng: float
        offset charge
    ncut: int
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim: int, optional
        desired dimension of the truncated quantum system
    """

    def __init__(self, EJ, EC, ng, ncut, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = 'Transmon qubit'

    def n_operator(self):
        """Returns charge operator `n` in charge basis"""
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1)
        return np.diagflat(diag_elements)

    # TODO: Provide cos(phi), sin(phi), e^(iphi) operators

    def hamiltonian(self):
        """Returns Hamiltonian in charge basis"""
        dimension = self.hilbertdim()
        hamiltonian_mat = np.zeros((dimension, dimension), dtype=np.float_)
        for i in range(dimension):
            hamiltonian_mat[i][i] = 4.0 * self.EC * (i - self.ncut - self.ng) ** 2
        for i in range(dimension - 1):
            hamiltonian_mat[i][i + 1] = -self.EJ / 2.0
            hamiltonian_mat[i + 1][i] = -self.EJ / 2.0
        return hamiltonian_mat

    def hilbertdim(self):
        """Returns Hilbert space dimension"""
        return 2 * self.ncut + 1

    def potential(self, phi):
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi: float
            phase variable value

        Returns
        -------
        float
        """
        return -self.EJ * np.cos(phi)

    def plot_n_wavefunction(self, esys, mode, which=0, nrange=(-5, 6), filename=None):
        """Plots transmon wave function in charge basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        mode: str from MODE_FUNC_DICT
            `'abs_sqr', 'abs', 'real', 'imag'`
        which: int or tuple of ints, optional
             index or indices of wave functions to plot (Default value = 0)
        nrange: tuple of two ints
             range of `n` to be included on the x-axis (Default value = (-5,6))

        Returns
        -------
        Figure, Axes
        """
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        modefunction = constants.MODE_FUNC_DICT[mode]
        n_wavefunc.amplitudes = modefunction(n_wavefunc.amplitudes)
        plot.wavefunction1d_discrete(n_wavefunc, nrange, filename=filename)
        return None

    def plot_phi_wavefunction(self, esys, which=(0,), phi_points=251, mode='abs_sqr', filename=None, scaling=None):
        """Plot phase-basis wave function(s).

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int or tuple of ints, optional
            single index or tuple of integers indexing the wave function(s) to be plotted (Default value = (0)
        phi_points: int, optional
            number of points on the x-axis (resolution) (Default value = 251)
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (Default value = 'abs_sqr')
        scaling: float or None, optional
            custom scaling of wave function amplitude/modulus

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
            scale = 0.3 * self.EJ
        else:
            scale = scaling

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for wavefunc_index in index_tuple:
            phi_wavefunc = self.phasebasis_wavefunction(esys, which=wavefunc_index, phi_points=phi_points)
            phase = extract_phase(phi_wavefunc.amplitudes)
            phi_wavefunc.amplitudes *= cmath.exp(-1j * phase)
            if np.sum(phi_wavefunc.amplitudes) < 0:
                phi_wavefunc.amplitudes *= -1.0

            phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)
            potential_vals = -self.EJ * np.cos(phi_wavefunc.basis_labels)
            plot.wavefunction1d(phi_wavefunc, potential_vals=potential_vals, offset=phi_wavefunc.energy,
                                scaling=scale, xlabel='phi', fig_ax=(fig, ax), filename=filename)
        return fig, ax

    def numberbasis_wavefunction(self, esys=None, which=0):
        """Return the transmon wave function in number basis. The specific index of the wave function to be returned is
        `which`.

        Parameters
        ----------
        esys: ndarray, ndarray, optional
            if `None`, the eigensystem is calculated on the fly; otherwise, the provided eigenvalue, eigenvector arrays
            as obtained from `.eigensystem()`, are used (Default value = None)
        which: int, optional
            eigenfunction index (Default value = 0)

        Returns
        -------
        WaveFunction object
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, evecs = esys

        n_vals = np.arange(-self.ncut, self.ncut + 1)
        return WaveFunction(n_vals, evecs[:, which], evals[which])

    def phasebasis_wavefunction(self, esys, which=0, phi_points=251):
        """Return the transmon wave function in phase basis. The specific index of the wavefunction is `which`.
        `esys` can be provided, but if set to `None` then it is calculated on the fly.

        Parameters
        ----------
        esys: `None` or tuple (ndarray, ndarray)
            if None, the eigensystem is calculated on the fly; otherwise, the provided eigenvalue, eigenvector arrays
            as obtained from `.eigensystem()` are used
        which: int, optional
            eigenfunction index (Default value = 0)
        phi_points: int, optional
            number of phi values at which the wave function is evaluated (Default value = 251)

        Returns
        -------
        WaveFunction object
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, evecs = esys
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        phi_basis_labels = np.linspace(-np.pi, np.pi, phi_points)
        phi_wavefunc_amplitudes = np.empty(phi_points, dtype=np.complex_)
        for k in range(phi_points):
            phi_wavefunc_amplitudes[k] = ((1.0 / math.sqrt(2 * np.pi)) *
                                          np.sum(n_wavefunc.amplitudes * np.exp(1j * phi_basis_labels[k] *
                                                                                n_wavefunc.basis_labels)))
        return WaveFunction(phi_basis_labels, phi_wavefunc_amplitudes, energy=evals[which])