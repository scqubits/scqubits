# operators.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import Union

import numpy as np
import scipy as sp
from numpy import ndarray
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix


def annihilation(dimension: int) -> ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the annihilation operator in number basis.
    """
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)


def creation(dimension: int) -> ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the creation operator in number basis.
    """
    return annihilation(dimension).T


def number(dimension: int, prefactor: Union[float, complex] = None) -> ndarray:
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor
    can be directly included in the generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    prefactor:
        prefactor multiplying the number operator matrix


    Returns
    -------
        number operator matrix, size dimension x dimension
    """
    diag_elements = np.arange(dimension)
    if prefactor:
        diag_elements *= prefactor
    return np.diagflat(diag_elements)


def annihilation_sparse(dimension: int) -> csc_matrix:
    """Returns a matrix of size dimension x dimension representing the annihilation operator
    in the format of a scipy sparse.csc_matrix.
    """
    offdiag_elements = np.sqrt(range(dimension))
    return sp.sparse.dia_matrix((offdiag_elements, [1]), shape=(dimension, dimension)).tocsc()


def creation_sparse(dimension: int) -> csc_matrix:
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc_matrix
    """
    return annihilation_sparse(dimension).transpose().tocsc()


def number_sparse(dimension: int, prefactor: Union[float, complex] = None) -> dia_matrix:
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor
    can be directly included in the generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    prefactor:
        prefactor multiplying the number operator matrix

    Returns
    -------
        sparse number operator matrix, size dimension x dimension
    """
    diag_elements = np.arange(dimension, dtype=np.float_)
    if prefactor:
        diag_elements *= prefactor
    return sp.sparse.dia_matrix((diag_elements, [0]), shape=(dimension, dimension), dtype=np.float_)


def hubbard_sparse(j1: int, j2: int, dimension: int) -> csc_matrix:
    """The Hubbard operator :math:`|j1\\rangle>\\langle j2|` is returned as a matrix of linear size dimension.

    Parameters
    ----------
    dimension:
    j1, j2:
        indices of the two states labeling the Hubbard operator

    Returns
    -------
        sparse number operator matrix, size dimension x dimension
    """
    hubbardmat = sp.sparse.dok_matrix((dimension, dimension), dtype=np.float_)
    hubbardmat[j1, j2] = 1.0
    return hubbardmat.asformat('csc')
