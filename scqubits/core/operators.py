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

from __future__ import annotations

import numpy as np
import scipy as sp

from numpy import ndarray
from scipy.sparse import csc_matrix


def annihilation(dimension: int) -> ndarray:
    """Return the annihilation operator in the number basis as a dense matrix.

    Parameters
    ----------
    dimension:
        matrix size.

    Returns
    -------
    Annihilation operator matrix, size ``dimension x dimension``.
    """
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)


def annihilation_sparse(dimension: int) -> csc_matrix:
    """Return the annihilation operator in the number basis as a sparse matrix.

    Parameters
    ----------
    dimension:
        matrix size.

    Returns
    -------
    Annihilation operator as :class:`scipy.sparse.csc_matrix`,
    size ``dimension x dimension``.
    """
    offdiag_elements = np.sqrt(range(dimension))
    return sp.sparse.dia_matrix(  # type: ignore[type-var, misc, return-value]
        (offdiag_elements, [1]), shape=(dimension, dimension)
    ).tocsc()


def creation(dimension: int) -> ndarray:
    """Return the creation operator in the number basis as a dense matrix.

    Parameters
    ----------
    dimension:
        matrix size.

    Returns
    -------
    Creation operator matrix, size ``dimension x dimension``.
    """
    return annihilation(dimension).T


def creation_sparse(dimension: int) -> csc_matrix:
    """Return the creation operator in the number basis as a sparse matrix.

    Parameters
    ----------
    dimension:
        matrix size.

    Returns
    -------
    Creation operator as :class:`scipy.sparse.csc_matrix`,
    size ``dimension x dimension``.
    """
    return annihilation_sparse(dimension).transpose().tocsc()


def hubbard_sparse(j1: int, j2: int, dimension: int) -> csc_matrix:
    r"""Return the Hubbard operator :math:`|j_1\rangle\langle j_2|` as a sparse matrix.

    Parameters
    ----------
    j1:
        row index of the Hubbard operator.
    j2:
        column index of the Hubbard operator.
    dimension:
        matrix size.

    Returns
    -------
    Sparse Hubbard operator matrix, size ``dimension x dimension``.
    """
    hubbardmat = sp.sparse.dok_matrix(
        (dimension, dimension), dtype=np.float64  # type: ignore[arg-type]
    )
    hubbardmat[j1, j2] = 1.0
    return hubbardmat.asformat("csc")


def number(dimension: int, prefactor: float | complex | None = None) -> ndarray:
    """Return the number operator as a dense matrix.

    An optional ``prefactor`` is folded directly into the matrix entries.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        prefactor multiplying the number operator matrix.

    Returns
    -------
    Number operator matrix, size ``dimension x dimension``.
    """
    diag_elements = np.arange(dimension, dtype=np.float64)
    if prefactor:
        diag_elements *= prefactor  # type: ignore[arg-type]
    return np.diagflat(diag_elements)


def number_sparse(
    dimension: int, prefactor: float | complex | None = None
) -> csc_matrix:
    """Return the number operator as a sparse matrix.

    An optional ``prefactor`` is folded directly into the matrix entries.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        prefactor multiplying the number operator matrix.

    Returns
    -------
    Sparse number operator matrix, size ``dimension x dimension``.
    """
    diag_elements = np.arange(dimension, dtype=np.float64)
    if prefactor:
        diag_elements *= prefactor  # type: ignore[arg-type]
    return sp.sparse.dia_matrix(
        (diag_elements, [0]), shape=(dimension, dimension), dtype=np.float64
    ).tocsc()


def a_plus_adag_sparse(
    dimension: int, prefactor: float | complex | None = None
) -> csc_matrix:
    r"""Return ``prefactor`` :math:`(a+a^\dagger)` as a sparse matrix.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        overall prefactor (defaults to 1 if not given).

    Returns
    -------
    ``prefactor`` :math:`(a+a^\dagger)` as a sparse operator matrix,
    size ``dimension x dimension``.
    """
    prefactor = prefactor if prefactor is not None else 1.0
    return prefactor * (annihilation_sparse(dimension) + creation_sparse(dimension))


def a_plus_adag(dimension: int, prefactor: float | complex | None = None) -> ndarray:
    r"""Return ``prefactor`` :math:`(a+a^\dagger)` as a dense matrix.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        overall prefactor (defaults to 1 if not given).

    Returns
    -------
    ``prefactor`` :math:`(a+a^\dagger)` as :class:`~numpy.ndarray`,
    size ``dimension x dimension``.
    """
    return a_plus_adag_sparse(dimension, prefactor=prefactor).toarray()


def cos_theta_harmonic(
    dimension: int, prefactor: float | complex | None = None
) -> ndarray:
    r"""Return :math:`\cos(\text{prefactor}\,(a+a^\dagger))` as a dense matrix.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        prefactor inside the cosine argument (defaults to 1 if not given).

    Returns
    -------
    :math:`\cos(\text{prefactor}\,(a+a^\dagger))` as :class:`~numpy.ndarray`,
    size ``dimension x dimension``.
    """
    return sp.linalg.cosm(a_plus_adag_sparse(dimension, prefactor=prefactor).toarray())


def sin_theta_harmonic(
    dimension: int, prefactor: float | complex | None = None
) -> ndarray:
    r"""Return :math:`\sin(\text{prefactor}\,(a+a^\dagger))` as a dense matrix.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        prefactor inside the sine argument (defaults to 1 if not given).

    Returns
    -------
    :math:`\sin(\text{prefactor}\,(a+a^\dagger))` as :class:`~numpy.ndarray`,
    size ``dimension x dimension``.
    """
    return sp.linalg.sinm(a_plus_adag_sparse(dimension, prefactor=prefactor).toarray())


def iadag_minus_ia_sparse(
    dimension: int, prefactor: float | complex | None = None
) -> csc_matrix:
    r"""Return ``prefactor`` :math:`(ia^\dagger-ia)` as a sparse matrix.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        overall prefactor (defaults to 1 if not given).

    Returns
    -------
    ``prefactor`` :math:`(ia^\dagger-ia)` as a sparse operator matrix,
    size ``dimension x dimension``.
    """
    prefactor = prefactor if prefactor is not None else 1.0
    return prefactor * (
        1j * creation_sparse(dimension) - 1j * annihilation_sparse(dimension)
    )


def iadag_minus_ia(dimension: int, prefactor: float | complex | None = None) -> ndarray:
    r"""Return ``prefactor`` :math:`(ia^\dagger-ia)` as a dense matrix.

    Parameters
    ----------
    dimension:
        matrix size.
    prefactor:
        overall prefactor (defaults to 1 if not given).

    Returns
    -------
    ``prefactor`` :math:`(ia^\dagger-ia)` as :class:`~numpy.ndarray`,
    size ``dimension x dimension``.
    """
    return iadag_minus_ia_sparse(dimension, prefactor=prefactor).toarray()


def sigma_minus() -> np.ndarray:
    r"""Return the Pauli lowering operator :math:`\sigma_-` as a 2x2 dense matrix."""
    return sigma_plus().T


def sigma_plus() -> np.ndarray:
    r"""Return the Pauli raising operator :math:`\sigma_+` as a 2x2 dense matrix."""
    return np.asarray([[0.0, 1.0], [0.0, 0.0]])


def sigma_x() -> np.ndarray:
    r"""Return the Pauli :math:`\sigma_x` operator as a 2x2 dense matrix."""
    return np.asarray([[0.0, 1.0], [1.0, 0.0]])


def sigma_y() -> np.ndarray:
    r"""Return the Pauli :math:`\sigma_y` operator as a 2x2 dense matrix."""
    return np.asarray([[0.0, -1j], [1j, 0.0]])


def sigma_z() -> np.ndarray:
    r"""Return the Pauli :math:`\sigma_z` operator as a 2x2 dense matrix."""
    return np.asarray([[1.0, 0.0], [0.0, -1.0]])
