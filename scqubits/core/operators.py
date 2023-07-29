# operators.py
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

from typing import Optional, Union

import numpy as np
import scipy as sp

from numpy import ndarray
from scipy.sparse import csc_matrix


def annihilation(dimension: int) -> ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the annihilation
    operator in number basis.
    """
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)


def annihilation_sparse(dimension: int) -> csc_matrix:
    """Returns a matrix of size dimension x dimension representing the annihilation
    operator in the format of a scipy sparse.csc_matrix.
    """
    offdiag_elements = np.sqrt(range(dimension))
    return sp.sparse.dia_matrix(
        (offdiag_elements, [1]), shape=(dimension, dimension)
    ).tocsc()


def creation(dimension: int) -> ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the creation
    operator in number basis.
    """
    return annihilation(dimension).T


def creation_sparse(dimension: int) -> csc_matrix:
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc_matrix
    """
    return annihilation_sparse(dimension).transpose().tocsc()


def hubbard_sparse(j1: int, j2: int, dimension: int) -> csc_matrix:
    """The Hubbard operator :math:`|j1\\rangle>\\langle j2|` is returned as a matrix of
    linear size dimension.

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
    return hubbardmat.asformat("csc")


def number(
    dimension: int, prefactor: Optional[Union[float, complex]] = None
) -> ndarray:
    """Number operator matrix of size dimension x dimension in sparse matrix
    representation. An additional prefactor can be directly included in the
    generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    dimension:
        matrix dimension
    prefactor:
        prefactor multiplying the number operator matrix


    Returns
    -------
        number operator matrix, size dimension x dimension
    """
    diag_elements = np.arange(dimension, dtype=np.float_)
    if prefactor:
        diag_elements *= prefactor
    return np.diagflat(diag_elements)


def number_sparse(
    dimension: int, prefactor: Optional[Union[float, complex]] = None
) -> csc_matrix:
    """Number operator matrix of size dimension x dimension in sparse matrix
    representation. An additional prefactor can be directly included in the
    generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix

    Returns
    -------
        sparse number operator matrix, size dimension x dimension
    """
    diag_elements = np.arange(dimension, dtype=np.float_)
    if prefactor:
        diag_elements *= prefactor
    return sp.sparse.dia_matrix(
        (diag_elements, [0]), shape=(dimension, dimension), dtype=np.float_
    ).tocsc()


def a_plus_adag_sparse(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> csc_matrix:
    """Operator matrix for prefactor(a+a^dag) of size dimension x dimension in
    sparse matrix representation.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor * (a + a^dag) as sparse operator matrix, size dimension x dimension
    """
    prefactor = prefactor if prefactor is not None else 1.0
    return prefactor * (annihilation_sparse(dimension) + creation_sparse(dimension))


def a_plus_adag(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> ndarray:
    """Operator matrix for prefactor(a+a^dag) of size dimension x dimension in
    sparse matrix representation.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor (a + a^dag) as ndarray, size dimension x dimension
    """
    return a_plus_adag_sparse(dimension, prefactor=prefactor).toarray()


def ia_minus_iadag_sparse(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> csc_matrix:
    """Operator matrix for prefactor(ia-ia^dag) of size dimension x dimension as
    ndarray

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor  (ia - ia^dag) as sparse operator matrix, size dimension x dimension
    """
    prefactor = prefactor if prefactor is not None else 1.0
    return prefactor * (
        1j * annihilation_sparse(dimension) - 1j * creation_sparse(dimension)
    )


def ia_minus_iadag(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> ndarray:
    """Operator matrix for prefactor(ia-ia^dag) of size dimension x dimension as
    ndarray

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor  (ia - ia^dag) as ndarray, size dimension x dimension
    """
    return ia_minus_iadag_sparse(dimension, prefactor=prefactor).toarray()


def sigma_minus() -> np.ndarray:
    return sigma_plus().T


def sigma_plus() -> np.ndarray:
    return np.asarray([[0.0, 1.0], [0.0, 0.0]])


def sigma_x() -> np.ndarray:
    return np.asarray([[0.0, 1.0], [1.0, 0.0]])


def sigma_y() -> np.ndarray:
    return np.asarray([[0.0, -1j], [1j, 0.0]])


def sigma_z() -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, -1.0]])
