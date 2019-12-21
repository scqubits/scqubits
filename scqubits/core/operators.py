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

import numpy as np
import scipy as sp


def annihilation(dimension):
    """
    Returns a dense matrix of size dimension x dimension representing the annihilation operator in number basis.

    Parameters
    ----------
    dimension: int

    Returns
    -------
    ndarray
        annihilation operator matrix, size dimension x dimension
    """
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)


def creation(dimension):
    """
    Returns a dense matrix of size dimension x dimension representing the creation operator in number basis.

    Parameters
    ----------
    dimension: int

    Returns
    -------
    ndarray
        creation operator matrix, size dimension x dimension

    """
    return annihilation(dimension).T


def number(dimension, prefactor=None):
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor
    can be directly included in the generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    dimension: int
    prefactor: float or complex, optional
        prefactor multiplying the number operator matrix


    Returns
    -------
    ndarray
        number operator matrix, size dimension x dimension
    """
    diag_elements = np.arange(dimension)
    if prefactor:
        diag_elements *= prefactor
    return np.diagflat(diag_elements)


def annihilation_sparse(dimension):
    """Returns a matrix of size dimension x dimension representing the annihilation operator
    in the format of a scipy sparse.csc_matrix.

    Parameters
    ----------
    dimension: int

    Returns
    -------
    sparse.csc_matrix
        sparse annihilation operator matrix, size dimension x dimension
    """
    offdiag_elements = np.sqrt(range(dimension))
    return sp.sparse.dia_matrix((offdiag_elements, [1]), shape=(dimension, dimension)).tocsc()


def creation_sparse(dimension):
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc_matrix

    Parameters
    ----------
    dimension: int

    Returns
    -------
    sparse.csc_matrix
        sparse annihilation operator matrix, size dimension x dimension
    """
    return annihilation_sparse(dimension).transpose().tocsc()


def number_sparse(dimension, prefactor=None):
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor
    can be directly included in the generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    dimension: int
    prefactor: float or complex, optional
        prefactor multiplying the number operator matrix

    Returns
    -------
    sparse.csc_matrix
        sparse number operator matrix, size dimension x dimension
    """
    diag_elements = np.arange(dimension, dtype=np.float_)
    if prefactor:
        diag_elements *= prefactor
    return sp.sparse.dia_matrix((diag_elements, [0]), shape=(dimension, dimension), dtype=np.float_)


def hubbard_sparse(j1, j2, dimension):
    """The Hubbard operator :math:`|j1\\rangle>\\langle j2|` is returned as a matrix of linear size dimension.

    Parameters
    ----------
    dimension: int
    j1, j2: int
        indices of the two states labeling the Hubbard operator

    Returns
    -------
    sparse.csc_matrix
        sparse number operator matrix, size dimension x dimension
    """
    hubbardmat = sp.sparse.dok_matrix((dimension, dimension), dtype=np.float_)
    hubbardmat[j1, j2] = 1.0
    return hubbardmat.asformat('csc')
