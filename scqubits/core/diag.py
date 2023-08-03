# diag.py
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

from numpy import ndarray
from typing import Any, Dict, List, Optional, Tuple, Union
from qutip import Qobj
from scipy.sparse import csc_matrix
from scqubits.io_utils.fileio_qutip import QutipEigenstates
from scqubits.utils.spectrum_utils import order_eigensystem, has_degeneracy

import copy
import numpy as np
import scipy as sp
import scqubits.settings as settings
import warnings


def _dict_merge(
    d: Dict[str, Any],
    d_other: Dict[str, Any],
    exclude: Union[List[str], None] = None,
    overwrite=False,
) -> Dict[str, Any]:
    """
    Selective dictionary merge. This function makes a copy of the given
    dictionary `d` and selectively updates/adds entries from `d_other`,
    as long as the keys are not given in `exclude`.
    Whether entries in `d` are overwritten by entries in `d_other` is
    determined by the value of the `overwrite` parameter

    Parameters
    ----------
    d: dict
        dictionary
    d_other:
        second dictionary to be merged with the first
    exclude: dict
        list of potential keys in d_other to be excluded from being added to resulting merge
    overwrite: bool
        determines if keys already in d should be overwritten by those in d_other

    Returns
    ----------
        merged dictionary

    """
    exclude = [] if exclude is None else exclude

    d_new = copy.deepcopy(d)
    for key in d_other:
        if key not in exclude and (overwrite or key not in d):
            d_new[key] = d_other[key]

    return d_new


def _cast_matrix(
    matrix: Union[ndarray, csc_matrix, Qobj], cast_to: str, force_cast: bool = True
) -> Union[ndarray, csc_matrix, Qobj]:
    """
    Casts a given matrix into a required form ('sparse' or 'dense')
    as defined by `cast_to` parameter.
    Note that in some cases casting may not be explicitly needed,
    for example: the sparse matrix routines can can often accept
    dense matrices. The parameter `force_cast` determines if the
    casing should be always done, or only where it is necessary.

    Parameters
    ----------
    matrix: Qobj, ndarray or csc_matrx
        matrix given as an ndarray, Qobj, or scipy's sparse matrix format
    cast_to: str
        string representing the format that matrix should be cast into: 'sparse' or 'dense'
    force_cast: bool
        determines of casting should be always performed or only where necessary

    Returns
    ----------
        matrix in the right sparse or dense form

    """
    m = matrix

    if cast_to == "sparse":
        if isinstance(matrix, Qobj):
            m = csc_matrix(matrix.data)
        elif force_cast and isinstance(matrix, ndarray):
            m = csc_matrix(matrix)

    elif cast_to == "dense":
        if isinstance(matrix, Qobj):
            m = matrix.full()
        elif force_cast and sp.sparse.issparse(matrix):
            m = matrix.toarray()
    else:
        raise ValueError("Can only matrix to 'sparse' or 'dense' forms.")

    return m


def _convert_evecs_to_qobjs(evecs: ndarray, matrix_qobj, wrap: bool = False) -> ndarray:
    """
    Converts an `ndarray` containing eigenvectors (that would be typically
    returned from a diagonalization routine, such as `eighs` or `eigh`),
    to a numpy array of qutip's Qobjs.
    Potentially also wraps those into `scqubits.io_utils.fileio_qutip.QutipEigenstates`.

    Parameters
    ----------
    evecs:
        ndarray of eigenvectors (as columns)
    matrix_qobj:
        matrix in the qutipQbj form; if given, used to extract the tensor product structure
    wrap:
        determines if we wrap results in QutipEigenstates

    Returns
    ----------
        eigenvectors represented in terms of Qobjs

    """
    evecs_count = evecs.shape[1]
    evec_dims = [matrix_qobj.dims[0], [1] * len(matrix_qobj.dims[0])]
    evecs_qobj = np.empty((evecs_count,), dtype=object)

    for i in range(evecs_count):
        v = Qobj(evecs[:, i], dims=evec_dims, type="ket")
        evecs_qobj[i] = v / v.norm()

    # Optionally, we wrap the resulting array in QutipEigenstates as is done in HilbertSpace.
    if wrap:
        evecs_qobj = evecs_qobj.view(QutipEigenstates)

    return evecs_qobj


### scipy based routines ####


def evals_scipy_dense(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """
    Diagonalization based on scipy's (dense) `eigh` function.
    Only evals are returned.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of matrix

    """
    m = _cast_matrix(matrix, "dense")

    evals = sp.linalg.eigh(
        m, subset_by_index=(0, evals_count - 1), eigvals_only=True, **kwargs
    )
    return evals


def esys_scipy_dense(
    matrix, evals_count, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """
    Diagonalization based on scipy's (dense) eigh function.
    Both evals and evecs are returned.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if matrix is a Qobj instance

    """
    m = _cast_matrix(matrix, "dense")

    evals, evecs = sp.linalg.eigh(m, subset_by_index=(0, evals_count - 1), **kwargs)

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


def evals_scipy_sparse(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """
    Diagonalization based on scipy's (sparse) `eigsh` function.
    Only evals are returned.

    Note the convoluted convention when it comes to ordering and how it is related
    to the presence of `return_eigenvectors` parameter. See here for details:
    https://github.com/scipy/scipy/issues/9082

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of matrix
    """
    m = _cast_matrix(matrix, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            v0=settings.RANDOM_ARRAY[: matrix.shape[0]],
            return_eigenvectors=False,
        ),
        kwargs,
        overwrite=True,
    )
    evals = sp.sparse.linalg.eigsh(m, k=evals_count, **options)

    # have to reverse order if return_eigenvectors=False and which="SA"
    return evals[::-1]


def esys_scipy_sparse(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """
    Diagonalization based on scipy's (sparse) `eigsh` function.
    Both evals and evecs are returned.

    Note the convoluted convention when it comes to ordering and how it is related
    to the presence of `return_eigenvectors` parameter. See here for details:
    https://github.com/scipy/scipy/issues/9082

    This function ensures that:
    1. We always use the same "random" starting vector v0. Otherwise results show
    random behavior (small deviations between different runs, problem for pytests)
    2. We test for degenerate eigenvalues. If there are any, we orthogonalize the
    eigenvectors properly.

    TODO:
        Right now, this is essentially a copy/paste of spectrum_utils.eigsh_safe().
        When the dust settles, should combine both into one.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if matrix is a Qobj instance

    """
    m = _cast_matrix(matrix, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            v0=settings.RANDOM_ARRAY[: matrix.shape[0]],
            return_eigenvectors=True,
        ),
        kwargs,
        overwrite=True,
    )
    evals, evecs = sp.sparse.linalg.eigsh(m, k=evals_count, **options)

    if has_degeneracy(evals):
        evecs, _ = sp.linalg.qr(evecs, mode="economic")

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


### primme based routines ####


def evals_primme_sparse(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """
    Diagonalization based on primme's (sparse) `eigsh` function.
    Only evals are returned.

    Requires that the primme library is installed.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of matrix

    """
    try:
        import primme
    except:
        raise ImportError("Module primme is not installed.")

    m = _cast_matrix(matrix, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=False,
        ),
        kwargs,
        overwrite=True,
    )

    evals = primme.eigsh(m, k=evals_count, **options)

    return evals


def esys_primme_sparse(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """
    Diagonalization based on primme's (sparse) `eigsh` function.
    Both evals and evecs are returned.

    Requires that the primme library is installed.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if matrix is a Qobj instance
    """
    try:
        import primme
    except:
        raise ImportError("Module primme is not installed.")

    m = _cast_matrix(matrix, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=True,
        ),
        kwargs,
        overwrite=True,
    )

    evals, evecs = primme.eigsh(m, k=evals_count, **options)

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


### cupy based routines ####


def evals_cupy_dense(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """
    Diagonalization based on cupy's (dense) `eighvalsh` function
    Only evals are returned.

    Requires that the cupy library is installed.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of matrix

    """
    try:
        import cupy as cp
    except:
        raise ImportError("Module cupy is not installed.")

    m = _cast_matrix(matrix, "dense")

    evals_gpu = cp.linalg.eigvalsh(cp.asarray(m), **kwargs)
    cp.cuda.Stream.null.synchronize()  # wait for GPU to finish

    return evals_gpu[:evals_count].get()


def esys_cupy_dense(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """
    Diagonalization based on cupy's (dense) `eigh` function.
    Both evals and evecs are returned.

    Requires that the cupy library is installed.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if matrix is a Qobj instance

    """
    try:
        import cupy as cp
    except:
        raise ImportError("Module cupy is not installed.")

    m = _cast_matrix(matrix, "dense")

    evals_gpu, evecs_gpu = cp.linalg.eigh(cp.asarray(m), **kwargs)
    cp.cuda.Stream.null.synchronize()  # wait for GPU to finish

    evals, evecs = evals_gpu[:evals_count].get(), evecs_gpu[:, :evals_count].get()

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


def evals_cupy_sparse(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """
    Diagonalization based on cupy's (sparse) `eigsh` function.
    Only evals are returned.

    Requires that the cupy (and cupyx) library is installed.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of matrix
    """
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
        from cupyx.scipy.sparse.linalg import eigsh
    except:
        raise ImportError("Module cupyx (part of cupy) is not installed.")

    m = cp_csc_matrix(_cast_matrix(matrix, "sparse"))

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=False,
        ),
        kwargs,
        overwrite=True,
    )
    evals_gpu = eigsh(m, k=evals_count, **options)

    # return evals_gpu.get()[::-1]
    return evals_gpu.get()


def esys_cupy_sparse(
    matrix: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """
    Diagonalization based on cupy's (sparse) eigsh function.
    Both evals and evecs are returned.

    Requires that the cupy library is installed.

    Parameters
    ----------
    matrix:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if matrix is a Qobj instance
    """
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
        from cupyx.scipy.sparse.linalg import eigsh
    except:
        raise ImportError("Module cupyx (part of cupy) is not installed.")

    m = cp_csc_matrix(_cast_matrix(matrix, "sparse"))

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=True,
        ),
        kwargs,
        overwrite=True,
    )
    evals_gpu, evecs_gpu = eigsh(m, k=evals_count, **options)

    evals, evecs = evals_gpu.get(), evecs_gpu.get()

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


# Default values of various noise constants and parameters.
DIAG_METHODS = {
    # scipy dense
    "evals_scipy_dense": evals_scipy_dense,
    "esys_scipy_dense": esys_scipy_dense,
    # scipy sparse
    "evals_scipy_sparse": evals_scipy_sparse,
    "esys_scipy_sparse": esys_scipy_sparse,
    "evals_scipy_sparse_SM": lambda matrix, evals_count, **kwargs: evals_scipy_sparse(
        matrix, evals_count, **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "esys_scipy_sparse_SM": lambda matrix, evals_count, **kwargs: esys_scipy_sparse(
        matrix, evals_count, **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "evals_scipy_sparse_LA_shift-inverse": lambda matrix, evals_count, **kwargs: evals_scipy_sparse(
        matrix,
        evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True)
    ),
    "esys_scipy_sparse_LA_shift-inverse": lambda matrix, evals_count, **kwargs: esys_scipy_sparse(
        matrix,
        evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True)
    ),
    "evals_scipy_sparse_LM_shift-inverse": lambda matrix, evals_count, **kwargs: evals_scipy_sparse(
        matrix,
        evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True)
    ),
    "esys_scipy_sparse_LM_shift-inverse": lambda matrix, evals_count, **kwargs: esys_scipy_sparse(
        matrix,
        evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True)
    ),
    # primme sparse
    "evals_primme_sparse": evals_primme_sparse,
    "esys_primme_sparse": esys_primme_sparse,
    "evals_primme_sparse_SM": lambda matrix, evals_count, **kwargs: evals_primme_sparse(
        matrix=matrix,
        evals_count=evals_count,
        **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "esys_primme_sparse_SM": lambda matrix, evals_count, **kwargs: esys_primme_sparse(
        matrix, evals_count, **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "evals_primme_sparse_LA_shift-inverse": lambda matrix, evals_count, **kwargs: evals_primme_sparse(
        matrix=matrix,
        evals_count=evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True)
    ),
    "esys_primme_sparse_LA_shift-inverse": lambda matrix, evals_count, **kwargs: esys_primme_sparse(
        matrix=matrix,
        evals_count=evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True)
    ),
    "evals_primme_sparse_LM_shift-inverse": lambda matrix, evals_count, **kwargs: evals_primme_sparse(
        matrix=matrix,
        evals_count=evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True)
    ),
    "esys_primme_sparse_LM_shift-inverse": lambda matrix, evals_count, **kwargs: esys_primme_sparse(
        matrix=matrix,
        evals_count=evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True)
    ),
    # cupy dense
    "evals_cupy_dense": evals_cupy_dense,
    "esys_cupy_dense": esys_cupy_dense,
    # cupy sparse
    "evals_cupy_sparse": evals_cupy_sparse,
    "esys_cupy_sparse": esys_cupy_sparse,
}
