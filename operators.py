# operators.py

from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
from scipy import sparse, linalg
from scipy.sparse import linalg
import math


def annihilation(dimension):
    """Returns a dense matrix of size dimension x dimension representing the annihilation operator."""
    offdiag_elements = [math.sqrt(i) for i in range(1, dimension)]
    return np.diagflat(offdiag_elements, 1)


def creation(dimension):
    """Returns a dense matrix of size dimension x dimension representing the creation operator."""
    return annihilation(dimension).T


def number(dimension, prefactor=None):
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor can be directly
    included in the generation of the matrix by supplying 'prefactor'.
    """
    diag_elements = np.arange(dimension)
    if prefactor:
        diag_elements *= prefactor
    return np.diagflat(diag_elements)


def annihilation_sparse(dimension):
    """Returns a matrix of size dimension x dimension representing the annihilation operator
    in the format of a scipy sparse.csc.
    """
    offdiag_elements = [math.sqrt(i) for i in range(dimension)]
    return sp.sparse.dia_matrix((offdiag_elements, [1]), shape=(dimension, dimension)).tocsc()


def creation_sparse(dimension):
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc.
    """
    return annihilation_sparse(dimension).transpose().tocsc()


def number_sparse(dimension, prefactor=None):
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor can be directly
    included in the generation of the matrix by supplying 'prefactor'.
    """
    diag_elements = np.arange(dimension, dtype=np.float_)
    if prefactor:
        diag_elements *= prefactor
    return sp.sparse.dia_matrix((diag_elements, [0]), shape=(dimension, dimension), dtype=np.float_)


def hubbard_sparse(j1, j2, dimension):
    """The Hubbard operator |j1><j2| is returned as a matrix of linear size 'dimension'."""
    hubbardmat = sp.sparse.dok_matrix((dimension, dimension), dtype=np.float_)
    hubbardmat[j1, j2] = 1.0
    return hubbardmat.asformat('dia')
