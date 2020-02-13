# transmon.py
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

import numpy as np

import scqubits.core.constants as constants
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot
from scqubits.core.constants import MODE_STR_DICT
from scqubits.core.descriptors import WatchedProperty
from scqubits.core.discretization import Grid1d
from scqubits.core.qubit_base import QubitBaseClass1d
from scqubits.core.storage import WaveFunction


# —Cooper pair box / transmon———————————————————————————————————————————————————————————————————————————————————————————

class Transmon(QubitBaseClass1d):
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

    EJ = WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng = WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ncut = WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, ng, ncut, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = 'transmon'
        self._evec_dtype = np.float_
        self._default_grid = Grid1d(-np.pi, np.pi, 151)
        self._default_n_range = (-5, 6)

    def n_operator(self):
        """Returns charge operator `n` in the charge basis"""
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1)
        return np.diag(diag_elements)

    def exp_i_phi_operator(self):
        """Returns operator :math:`e^{i\\varphi}` in the charge basis"""
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - 1)
        exp_op = np.diag(entries, -1)
        return exp_op

    def cos_phi_operator(self):
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * self.exp_i_phi_operator()
        cos_op += cos_op.T
        return cos_op

    def sin_phi_operator(self):
        """Returns operator :math:`\\sin \\varphi` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_phi_operator()
        sin_op += sin_op.conjugate().T
        return sin_op

    def hamiltonian(self):
        """Returns Hamiltonian in charge basis"""
        dimension = self.hilbertdim()
        hamiltonian_mat = np.diag([4.0 * self.EC * (ind - self.ncut - self.ng) ** 2 for ind in range(dimension)])
        ind = np.arange(dimension - 1)
        hamiltonian_mat[ind, ind+1] = -self.EJ / 2.0
        hamiltonian_mat[ind+1, ind] = -self.EJ / 2.0
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

    def plot_n_wavefunction(self, esys=None, mode='real', which=0, nrange=None, **kwargs):
        """Plots transmon wave function in charge basis

        Parameters
        ----------
        esys: tuple(ndarray, ndarray), optional
            eigenvalues, eigenvectors
        mode: str from MODE_FUNC_DICT, optional
            `'abs_sqr', 'abs', 'real', 'imag'`
        which: int or tuple of ints, optional
             index or indices of wave functions to plot (default value = 0)
        nrange: tuple of two ints, optional
             range of `n` to be included on the x-axis (default value = (-5,6))
        **kwargs:
            plotting parameters

        Returns
        -------
        Figure, Axes
        """
        if nrange is None:
            nrange = self._default_n_range
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_wavefunc.amplitudes = amplitude_modifier(n_wavefunc.amplitudes)
        kwargs = {**defaults.wavefunction1d_discrete(mode), **kwargs}    # if any duplicates, later ones survive
        return plot.wavefunction1d_discrete(n_wavefunc, xlim=nrange, **kwargs)

    def wavefunction1d_defaults(self, mode, evals, wavefunc_count):
        """Plot defaults for plotting.wavefunction1d.

        Parameters
        ----------
        mode: str
            amplitude modifier, needed to give the correct default y label
        evals: ndarray
            eigenvalues to include in plot
        wavefunc_count: int
        """
        ylabel = r'$\psi_j(\varphi)$'
        ylabel = MODE_STR_DICT[mode](ylabel)
        options = {
            'xlabel': r'$\varphi$',
            'ylabel': ylabel
        }
        if wavefunc_count > 1:
            ymin = -1.05 * self.EJ
            ymax = max(1.1 * self.EJ, evals[-1] + 0.05 * (evals[-1] - evals[0]))
            options['ylim'] = (ymin, ymax)
        return options

    def plot_phi_wavefunction(self, esys=None, which=0, phi_grid=None, mode='abs_sqr', scaling=None, **kwargs):
        """Alias for plot_wavefunction"""
        return self.plot_wavefunction(esys=esys, which=which, phi_grid=phi_grid, mode=mode, scaling=scaling, **kwargs)

    def numberbasis_wavefunction(self, esys=None, which=0):
        """Return the transmon wave function in number basis. The specific index of the wave function to be returned is
        `which`.

        Parameters
        ----------
        esys: ndarray, ndarray, optional
            if `None`, the eigensystem is calculated on the fly; otherwise, the provided eigenvalue, eigenvector arrays
            as obtained from `.eigensystem()`, are used (default value = None)
        which: int, optional
            eigenfunction index (default value = 0)

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

    def wavefunction(self, esys=None, which=0, phi_grid=None):
        """Return the transmon wave function in phase basis. The specific index of the wavefunction is `which`.
        `esys` can be provided, but if set to `None` then it is calculated on the fly.

        Parameters
        ----------
        esys: tuple(ndarray, ndarray), optional
            if None, the eigensystem is calculated on the fly; otherwise, the provided eigenvalue, eigenvector arrays
            as obtained from `.eigensystem()` are used
        which: int, optional
            eigenfunction index (default value = 0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid

        Returns
        -------
        WaveFunction object
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, _ = esys
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        phi_grid = phi_grid or self._default_grid

        phi_basis_labels = phi_grid.make_linspace()
        phi_wavefunc_amplitudes = np.empty(phi_grid.pt_count, dtype=np.complex_)
        for k in range(phi_grid.pt_count):
            phi_wavefunc_amplitudes[k] = ((1j**which / math.sqrt(2 * np.pi)) *
                                          np.sum(n_wavefunc.amplitudes *
                                                 np.exp(1j * phi_basis_labels[k] * n_wavefunc.basis_labels)))
        return WaveFunction(basis_labels=phi_basis_labels, amplitudes=phi_wavefunc_amplitudes, energy=evals[which])
