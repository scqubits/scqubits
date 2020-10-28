# discretization.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import numpy as np
from scipy import sparse

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.misc as utils
from scqubits.settings import STENCIL

FIRST_STENCIL_COEFFS = {3: [-1/2, 0, 1/2],
                        5: [1/12, -2/3, 0, 2/3, -1/12],
                        7: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60],
                        9: [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
                        }
SECOND_STENCIL_COEFFS = {3: [1, -2, 1],
                         5: [-1/12, 4/3, -5/2, 4/3, -1/12],
                         7: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                         9: [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]}


def band_matrix(band_coeffs, band_offsets, dim, dtype=None, has_corners=False):
    """
    Returns a dim x dim sparse matrix with constant diagonals of values `band_coeffs[0]`, `band_coeffs[1]`, ...
    along the (off-)diagonals specified by the offsets `band_offsets[0]`, `band_offsets[1]`, ... The `has_corners`
    option allows generation of band matrices with corner elements, in which lower off-diagonals wrap into the top
    right corner and upper off-diagonals wrap into the bottom left corner.
    Parameters
    ----------
    band_coeffs: list or ndarray
        each element of band_coeffs is a number to be assigned as a constant to the (off-)diagonals
    band_offsets: list or ndarray of int
        offsets specifying the positions of the (off-)diagonals
    dim: int
        dimension of the matrix
    dtype: dtype, optional
        if not specified, dtype is inferred from the dtype of `band_vecs`
    has_corners: bool, optional (default = False)
        if set to True, the off diagonals are wrapped into the opposing corners of the matrix
    Returns
    -------
    scipy.sparse.dia_matrix
    """
    ones_vector = np.ones(dim)
    vectors = [ones_vector * number for number in band_coeffs]
    matrix = sparse.dia_matrix((vectors, band_offsets), shape=(dim, dim), dtype=dtype)
    if not has_corners:
        return matrix
    for index, offset in enumerate(band_offsets):
        if offset < 0:
            corner_offset = dim + offset
            corner_band = vectors[index]
            corner_band = corner_band[offset:]
        elif offset > 0:
            corner_offset = -dim + offset
            corner_band = vectors[index][:-offset]
            corner_band = corner_band[-offset:]
        else:  # when offset == 0
            continue
        matrix.setdiag(corner_band, k=corner_offset)
    return matrix


class Grid1d(dispatch.DispatchClient, serializers.Serializable):
    """Data structure and methods for setting up discretized 1d coordinate grid, generating corresponding derivative
    matrices.

    Parameters
    ----------
    min_val: float
        minimum value of the discretized variable
    max_val: float
        maximum value of the discretized variable
    pt_count: int
        number of grid points
    """
    min_val = descriptors.WatchedProperty('GRID_UPDATE')
    max_val = descriptors.WatchedProperty('GRID_UPDATE')
    pt_count = descriptors.WatchedProperty('GRID_UPDATE')

    def __init__(self, min_val, max_val, pt_count):
        self.min_val = min_val
        self.max_val = max_val
        self.pt_count = pt_count

    def __repr__(self):
        init_dict = self.get_initdata()
        return type(self).__name__ + f'({init_dict!r})'

    def __str__(self):
        output = '    Grid1d ......'
        for param_name, param_val in sorted(utils.drop_private_keys(self.__dict__).items()):
            output += '\n' + str(param_name) + '\t: ' + str(param_val)
        return output

    def get_initdata(self):
        """Returns dict appropriate for creating/initializing a new Grid1d object.

        Returns
        -------
        dict
        """
        return self.__dict__

    def grid_spacing(self):
        """
        Returns
        -------
        float
            spacing between neighboring grid points
        """
        return (self.max_val - self.min_val) / (self.pt_count - 1)

    def make_linspace(self):
        """Returns a numpy array of the grid points

        Returns
        -------
        ndarray
        """
        return np.linspace(self.min_val, self.max_val, self.pt_count)

    def first_derivative_matrix_five_pt_stencil(self, prefactor=1.0, periodic=False):
        """Generate sparse matrix for first derivative of the form :math:`\\partial_{x_i}`.
        Uses a five point stencil :math:`f'(x) \\approx [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)]/12h`.

        Parameters
        ----------
        prefactor: float or complex, optional
            prefactor of the derivative matrix (default value: 1.0)
        periodic: bool, optional
            set to True if variable is a periodic variable

        Returns
        -------
        sparse matrix in `dia` format
        """
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_
        delta_x = (self.max_val - self.min_val) / (self.pt_count - 1)
        first_off_diag = prefactor * 8 / (12.0 * delta_x)
        second_off_diag = prefactor / (12.0 * delta_x)

        derivative_matrix = sparse.dia_matrix((self.pt_count, self.pt_count), dtype=dtp)
        derivative_matrix.setdiag(-second_off_diag, k=2)
        derivative_matrix.setdiag(first_off_diag, k=1)
        derivative_matrix.setdiag(-first_off_diag, k=-1)
        derivative_matrix.setdiag(second_off_diag, k=-2)

        if periodic:
            derivative_matrix.setdiag(first_off_diag, k=-self.pt_count + 1)
            derivative_matrix.setdiag(-second_off_diag, k=-self.pt_count + 2)
            derivative_matrix.setdiag(-first_off_diag, k=self.pt_count - 1)
            derivative_matrix.setdiag(second_off_diag, k=self.pt_count - 2)

        return derivative_matrix

    def second_derivative_matrix_five_pt_stencil(self, prefactor=1.0, periodic=False):
        """Generate sparse matrix for second derivative of the form :math:`\\partial^2_{x_i}`.
        Uses a five point stencil :math:`f'(x) \\approx [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)]/12h^2`.

        Parameters
        ----------
        prefactor: float or complex, optional
            prefactor of the derivative matrix (default value: 1.0)
        periodic: bool, optional
            set to True if variable is a periodic variable

        Returns
        -------
        sparse matrix in `dia` format
        """
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_
        delta_x = (self.max_val - self.min_val) / (self.pt_count - 1)
        diag = prefactor * 30.0 / (12.0 * delta_x**2)
        first_off_diag = prefactor * 16 / (12.0 * delta_x**2)
        second_off_diag = prefactor / (12.0 * delta_x**2)

        derivative_matrix = sparse.dia_matrix((self.pt_count, self.pt_count), dtype=dtp)
        derivative_matrix.setdiag(-diag, k=0)
        derivative_matrix.setdiag(-second_off_diag, k=2)
        derivative_matrix.setdiag(first_off_diag, k=1)
        derivative_matrix.setdiag(first_off_diag, k=-1)
        derivative_matrix.setdiag(-second_off_diag, k=-2)

        if periodic:
            derivative_matrix.setdiag(first_off_diag, k=-self.pt_count + 1)
            derivative_matrix.setdiag(-second_off_diag, k=-self.pt_count + 2)
            derivative_matrix.setdiag(first_off_diag, k=self.pt_count - 1)
            derivative_matrix.setdiag(-second_off_diag, k=self.pt_count - 2)

        return derivative_matrix

    def first_derivative_matrix(self, prefactor=1.0, periodic=False):
        """Generate sparse matrix for first derivative of the form :math:`\\partial_{x_i}`.
        Uses :math:`f'(x) \\approx [f(x+h) - f(x-h)]/2h`.

        Parameters
        ----------
        prefactor: float or complex, optional
            prefactor of the derivative matrix (default value: 1.0)
        periodic: bool, optional
            set to True if variable is a periodic variable

        Returns
        -------
        sparse matrix in `dia` format
        """
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_

        delta_x = self.grid_spacing()
        matrix_diagonals = [coefficient * prefactor / delta_x for coefficient in FIRST_STENCIL_COEFFS[STENCIL]]
        offset = [i - (STENCIL - 1) // 2 for i in range(STENCIL)]
        derivative_matrix = band_matrix(matrix_diagonals, offset, self.pt_count, dtype=dtp, has_corners=periodic)
        return derivative_matrix

    def second_derivative_matrix(self, prefactor=1.0, periodic=False):
        """Generate sparse matrix for second derivative of the form :math:`\\partial^2_{x_i}`.
        Uses :math:`f''(x) \\approx [f(x+h) - 2f(x) + f(x-h)]/h^2`.

        Parameters
        ----------
        prefactor: float, optional
            optional prefactor of the derivative matrix (default value = 1.0)
        periodic: bool, optional
            set to True if variable is a periodic variable (default value = False)

        Returns
        -------
        sparse matrix in `dia` format
        """
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_

        delta_x = self.grid_spacing()
        matrix_diagonals = [coefficient * prefactor / delta_x ** 2 for coefficient in SECOND_STENCIL_COEFFS[STENCIL]]
        offset = [i - (STENCIL - 1) // 2 for i in range(STENCIL)]
        derivative_matrix = band_matrix(matrix_diagonals, offset, self.pt_count, dtype=dtp, has_corners=periodic)
        return derivative_matrix


class GridSpec(dispatch.DispatchClient, serializers.Serializable):
    """Class for specifying a general discretized coordinate grid (arbitrary dimensions).

    Parameters
    ----------
    minmaxpts_array: ndarray
        array of with entries [minvalue, maxvalue, number of points]
    """
    min_vals = descriptors.WatchedProperty('GRID_UPDATE')
    max_vals = descriptors.WatchedProperty('GRID_UPDATE')
    var_count = descriptors.WatchedProperty('GRID_UPDATE')
    pt_counts = descriptors.WatchedProperty('GRID_UPDATE')

    def __init__(self, minmaxpts_array):
        self.min_vals = minmaxpts_array[:, 0]
        self.max_vals = minmaxpts_array[:, 1]
        self.var_count = len(self.min_vals)
        self.pt_counts = minmaxpts_array[:, 2].astype(np.int)  # these are used as indices; need to be whole numbers.

    def __str__(self):
        output = '    GridSpec ......'
        for param_name, param_val in sorted(self.__dict__.items()):
            output += '\n' + str(param_name) + '\t: ' + str(param_val)
        return output

    def unwrap(self):
        """Auxiliary routine that yields a tuple of the parameters specifying the grid."""
        return self.min_vals, self.max_vals, self.pt_counts, self.var_count
