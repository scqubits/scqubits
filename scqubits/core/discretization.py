# discretization.py
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

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.misc as utils


FIRST_STENCIL_COEFFS: Dict[int, List[float]] = {
    3: [-1 / 2, 0.0, 1 / 2],
    5: [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12],
    7: [-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60],
    9: [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280],
}

SECOND_STENCIL_COEFFS: Dict[int, List[float]] = {
    3: [1, -2, 1],
    5: [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
    7: [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    9: [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
}


def band_matrix(
    band_coeffs: Union[List[float], List[complex], ndarray],
    band_offsets: Union[List[int], ndarray],
    dim: int,
    dtype: Any = None,
    has_corners: bool = False,
) -> csc_matrix:
    """
    Returns a dim x dim sparse matrix with constant diagonals of values `band_coeffs[
    0]`, `band_coeffs[1]`, ... along the (off-)diagonals specified by the offsets
    `band_offsets[0]`, `band_offsets[1]`, ... The `has_corners` option allows
    generation of band matrices with corner elements, in which lower off-diagonals
    wrap into the top right corner and upper off-diagonals wrap into the bottom left
    corner.

    Parameters
    ----------
    band_coeffs:
        each element of band_coeffs is a number to be assigned as a constant to the
        (off-)diagonals
    band_offsets:
        offsets specifying the positions of the (off-)diagonals dim: dimension of
        the matrix
    dim:
        (linear) dimension of the matrix
    dtype:
        if not specified, dtype is inferred from the dtype of `band_vecs`
    has_corners:
        if set to True, the off diagonals are wrapped into the opposing corners of
        the matrix
    """
    ones_vector = np.ones(dim)
    vectors = [ones_vector * number for number in band_coeffs]
    matrix = sparse.dia_matrix((vectors, band_offsets), shape=(dim, dim), dtype=dtype)
    if not has_corners:
        return matrix.tocsc()
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
    return matrix.tocsc()


class Grid1d(dispatch.DispatchClient, serializers.Serializable):
    """Data structure and methods for setting up discretized 1d coordinate grid,
    generating corresponding derivative matrices.

    Parameters
    ----------
    min_val:
        minimum value of the discretized variable
    max_val:
        maximum value of the discretized variable
    pt_count:
        number of grid points
    """

    min_val = descriptors.WatchedProperty(float, "GRID_UPDATE")
    max_val = descriptors.WatchedProperty(float, "GRID_UPDATE")
    pt_count = descriptors.WatchedProperty(int, "GRID_UPDATE")

    def __init__(self, min_val: float, max_val: float, pt_count: int) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.pt_count = pt_count

    def __repr__(self) -> str:
        init_dict = self.get_initdata()
        return type(self).__name__ + f"({init_dict!r})"

    def __str__(self) -> str:
        output = "Grid1d -----[ "
        for param_name, param_val in sorted(
            utils.drop_private_keys(self.__dict__).items()
        ):
            output += f"{param_name}: {param_val},  "
        output = output[:-3] + " ]"
        return output

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return super().__hash__()

    def get_initdata(self) -> Dict[str, Any]:
        """Returns dict appropriate for creating/initializing a new Grid1d object.
        Returns
        -------
        dict
        """
        return self.__dict__

    def grid_spacing(self) -> float:
        """
        Returns
        -------
            spacing between neighboring grid points
        """
        return (self.max_val - self.min_val) / (self.pt_count - 1)

    def make_linspace(self) -> ndarray:
        """Returns a numpy array of the grid points
        Returns
        -------
        ndarray
        """
        return np.linspace(self.min_val, self.max_val, self.pt_count)

    def first_derivative_matrix(
        self, prefactor: Union[float, complex] = 1.0, periodic: bool = False
    ) -> csc_matrix:
        """Generate sparse matrix for first derivative of the form
        :math:`\\partial_{x_i}`. Uses STENCIL setting to construct the matrix with a
        multi-point stencil.

        Parameters
        ----------
        prefactor:
            prefactor of the derivative matrix (default value: 1.0)
        periodic:
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
        matrix_diagonals = [
            coefficient * prefactor / delta_x
            for coefficient in FIRST_STENCIL_COEFFS[settings.STENCIL]
        ]
        offset = [i - (settings.STENCIL - 1) // 2 for i in range(settings.STENCIL)]
        derivative_matrix = band_matrix(
            matrix_diagonals, offset, self.pt_count, dtype=dtp, has_corners=periodic
        )
        return derivative_matrix.tocsc()

    def second_derivative_matrix(
        self, prefactor: Union[float, complex] = 1.0, periodic: bool = False
    ) -> csc_matrix:
        """Generate sparse matrix for second derivative of the form
        :math:`\\partial^2_{x_i}`. Uses STENCIL setting to construct the matrix with
        a multi-point stencil.

        Parameters
        ----------
        prefactor:
            optional prefactor of the derivative matrix (default value = 1.0)
        periodic:
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
        matrix_diagonals = [
            coefficient * prefactor / delta_x**2
            for coefficient in SECOND_STENCIL_COEFFS[settings.STENCIL]
        ]
        offset = [i - (settings.STENCIL - 1) // 2 for i in range(settings.STENCIL)]
        derivative_matrix = band_matrix(
            matrix_diagonals, offset, self.pt_count, dtype=dtp, has_corners=periodic
        )
        return derivative_matrix.tocsc()


class GridSpec(dispatch.DispatchClient, serializers.Serializable):
    """Class for specifying a general discretized coordinate grid
    (arbitrary dimensions).

    Parameters
    ----------
    minmaxpts_array:
        array of with entries [minvalue, maxvalue, number of points]
    """

    min_vals = descriptors.WatchedProperty(ndarray, "GRID_UPDATE")
    max_vals = descriptors.WatchedProperty(ndarray, "GRID_UPDATE")
    var_count = descriptors.WatchedProperty(int, "GRID_UPDATE")
    pt_counts = descriptors.WatchedProperty(ndarray, "GRID_UPDATE")

    def __init__(self, minmaxpts_array: ndarray) -> None:
        self.min_vals = minmaxpts_array[:, 0]
        self.max_vals = minmaxpts_array[:, 1]
        self.var_count = len(self.min_vals)
        self.pt_counts = minmaxpts_array[:, 2].astype(int)  # used as int indices

    def __str__(self) -> str:
        output = "    GridSpec ......"
        for param_name, param_val in sorted(self.__dict__.items()):
            output += f"\n{param_name}\t: {param_val}"
        return output

    def unwrap(self) -> Tuple[ndarray, ndarray, Union[List[int], ndarray], int]:
        """Auxiliary routine that yields a tuple of the parameters specifying the
        grid."""
        return self.min_vals, self.max_vals, self.pt_counts, self.var_count
