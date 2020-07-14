# harmonic_osc.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import os
import warnings

import numpy as np
import scipy as sp

import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


def harm_osc_wavefunction(n, x, losc):
    """For given quantum number n=0,1,2,... return the value of the harmonic oscillator wave function
    :math:`\\psi_n(x) = N H_n(x/l_{osc}) \\exp(-x^2/2l_{osc})`, N being the proper normalization factor.

    Parameters
    ----------
    n: int
        index of wave function, n=0 is ground state
    x: float or ndarray
        coordinate(s) where wave function is evaluated
    losc: float
        oscillator length, defined via <0|x^2|0> = losc^2/2

    Returns
    -------
    float
        value of harmonic oscillator wave function
    """
    return ((2.0 ** n * sp.special.gamma(n + 1.0) * losc) ** (-0.5) * np.pi ** (-0.25) *
            sp.special.eval_hermite(n, x / losc) * np.exp(-(x * x) / (2 * losc * losc)))


# —Oscillator class—————————————————————————————————————————————————————————————————————————————————————————————————————

class Oscillator(base.QuantumSystem, serializers.Serializable):
    """General class for mode of an oscillator/resonator."""
    def __init__(self, E_osc=None, omega=None, truncated_dim=None):
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self.truncated_dim = truncated_dim

        # Support for omega will be rolled back eventually. For now allow with deprecation warnings.
        if E_osc is None and omega is None:
            raise ValueError('E_osc is a mandatory argument.')
        if omega:
            warnings.warn('To avoid confusion about 2pi factors, use of omega is deprecated. Use E_osc instead.',
                          FutureWarning)
            self.E_osc = omega
        # end of code supporting deprecated omega
        else:
            self.E_osc = E_osc

        self._init_params.remove('omega')
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_img/oscillator.png')

    @staticmethod
    def default_params():
        return {'E_osc': 5.0,
                'truncated_dim': 10}

    def get_omega(self):
        # Support for omega will be rolled back eventually. For now allow with deprecation warnings.
        warnings.warn('To avoid confusion about 2pi factors, use of omega is deprecated. Use E_osc instead.',
                      FutureWarning)
        return self.E_osc

    def set_omega(self, value):
        warnings.warn('To avoid confusion about 2pi factors, use of omega is deprecated. Use E_osc instead.',
                      FutureWarning)
        self.E_osc = value

    omega = property(get_omega, set_omega)
    # end of code for deprecated omega

    def eigenvals(self, evals_count=6):
        """Returns array of eigenvalues.

        Parameters
        ----------
        evals_count: int, optional
            number of desired eigenvalues (default value = 6)

        Returns
        -------
        ndarray
        """
        evals = [self.E_osc * n for n in range(evals_count)]
        return np.asarray(evals)

    def eigensys(self, evals_count=6):
        """Returns array of eigenvalues and eigenvectors

        Parameters
        ----------
        evals_count: int, optional
            number of desired eigenvalues (default value = 6)

        Returns
        -------
        ndarray, ndarray
        """
        evecs = np.zeros(shape=(self.truncated_dim, evals_count), dtype=np.float_)
        np.fill_diagonal(evecs, 1.0)

        return self.eigenvals(evals_count=evals_count), evecs

    def hilbertdim(self):
        """Returns Hilbert space dimension

        Returns
        -------
        int
        """
        return self.truncated_dim

    def creation_operator(self):
        """Returns the creation operator"""
        return op.creation(self.truncated_dim)

    def annihilation_operator(self):
        """Returns the creation operator"""
        return op.annihilation(self.truncated_dim)
