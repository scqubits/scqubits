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


class Grid1d:
    """Data structure and methods for setting up discretized 1d coordinate grid, generating corresponding derivative
    matrices.

    Parameters
    ----------
    min_val, max_val: float
        minimum and maximum values of the discretized variable
    pt_count: int
        number of grid points
    """

    def __init__(self, min_val, max_val, pt_count):
        self.min_val = min_val
        self.max_val = max_val
        self.pt_count = pt_count

    def __str__(self):
        output = '    Grid (1d) ......'
        for param_name, param_val in sorted(self.__dict__.items()):
            output += '\n' + str(param_name) + '\t: ' + str(param_val)
        return output

    def grid_spacing(self):
        """
        Returns
        -------
        float: spacing between neighboring grid points
        """
        return (self.max_val - self.min_val) / self.pt_count

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

        delta_x = (self.max_val - self.min_val) / self.pt_count
        offdiag_element = prefactor / (2 * delta_x)

        derivative_matrix = sparse.dia_matrix((self.pt_count, self.pt_count), dtype=dtp)
        derivative_matrix.setdiag(offdiag_element, k=1)    # occupy first off-diagonal to the right
        derivative_matrix.setdiag(-offdiag_element, k=-1)  # and left

        if periodic:
            derivative_matrix.setdiag(-offdiag_element, k=self.pt_count - 1)
            derivative_matrix.setdiag(offdiag_element, k=-self.pt_count + 1)

        return derivative_matrix

    def second_derivative_matrix(self, prefactor=1.0, periodic=False):
        """Generate sparse matrix for second derivative of the form :math:`\\partial^2_{x_i}`.
        Uses :math:`f''(x) \\approx [f(x+h) - 2f(x) + f(x-h)]/h^2`.

        Parameters
        ----------
        prefactor: float, optional
            optional prefactor of the derivative matrix (Default value = 1.0)
        periodic: bool, periodic
            set to True if variable is a periodic variable (Default value = False)


        Returns
        -------
        sparse matrix in `dia` format
        """
        delta_x = (self.max_val - self.min_val) / self.pt_count
        offdiag_element = prefactor / delta_x**2

        derivative_matrix = sparse.dia_matrix((self.pt_count, self.pt_count), dtype=np.float_)
        derivative_matrix.setdiag(-2.0 * offdiag_element, k=0)
        derivative_matrix.setdiag(offdiag_element, k=1)
        derivative_matrix.setdiag(offdiag_element, k=-1)

        if periodic:
            derivative_matrix.setdiag(offdiag_element, k=self.pt_count - 1)
            derivative_matrix.setdiag(offdiag_element, k=-self.pt_count + 1)

        return derivative_matrix

    def filewrite_params_h5(self, h5file_root):
        """Write grid parameters to h5 file

        Parameters
        ----------
        h5file_root: h5py.Group
            handle to root group of open h5 file
        """
        h5dataset = h5file_root.create_dataset("grid", data=[])
        h5dataset.attrs['min_val'] = self.min_val
        h5dataset.attrs['max_val'] = self.max_val
        h5dataset.attrs['pt_count'] = self.pt_count

    def set_params_from_h5(self, h5file_grid):
        """Read and store grid parameters from h5 file

        Parameters
        ----------
        h5file_grid: h5py.Group
            handle to grid group in open h5 file
        """
        self.min_val = h5file_grid.attrs['min_val']
        self.max_val = h5file_grid.attrs['max_val']
        self.pt_count = h5file_grid.attrs['pt_count']


class GridSpec:
    """Class for specifying a general discretized coordinate grid (arbitrary dimensions).

    Parameters
    ----------
    minmaxpts_array: ndarray
        array of with entries [minvalue, maxvalue, number of points]
    """
    def __init__(self, minmaxpts_array):
        self.min_vals = minmaxpts_array[:, 0]
        self.max_vals = minmaxpts_array[:, 1]
        self.var_count = len(self.min_vals)
        self.pt_counts = minmaxpts_array[:, 2].astype(np.int)  # these are used as indices; need to be whole numbers.

    def __str__(self):
        output = '    Grid (1d) ......'
        for param_name, param_val in sorted(self.__dict__.items()):
            output += '\n' + str(param_name) + '\t: ' + str(param_val)
        return output

    def unwrap(self):
        """Auxiliary routine that yields a tuple of the parameters specifying the grid."""
        return self.min_vals, self.max_vals, self.pt_counts, self.var_count

    def filewrite_params_h5(self, h5file_root):
        """Write grid parameters to h5 file

        Parameters
        ----------
        h5file_root: h5py.Group
            handle to root group of open h5 file
        """
        h5dataset = h5file_root.create_dataset("grid", data=[])
        h5dataset.attrs['min_vals'] = self.min_vals
        h5dataset.attrs['max_vals'] = self.max_vals
        h5dataset.attrs['var_count'] = self.var_count
        h5dataset.attrs['pt_counts'] = self.pt_counts

    def set_params_from_h5(self, h5file_grid):
        """Read and store grid parameters from h5 file

        Parameters
        ----------
        h5file_grid: h5py.Group
            handle to grid group in open h5 file
        """
        self.min_vals = h5file_grid.attrs['min_vals']
        self.max_vals = h5file_grid.attrs['max_vals']
        self.var_count = h5file_grid.attrs['var_count']
        self.pt_counts = h5file_grid.attrs['pt_count']
