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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
from qutip import Qobj
import scqubits.settings as settings

from scqubits.utils.spectrum_utils import order_eigensystem, has_degeneracy


def _setup_default_options(
    options: Dict[str, Any],
    defaults: Dict[str, Any],
    exclude: Union[List[str], None] = None,
) -> Dict[str, Any]:
    """
    Makes a copy of the options dictionary and updates keys from defaults
    if not already present in options.

    Parameters
    ----------
    options:
        dictionary of options
    defaults:
        dictionary of key/value pairs that are added if not already given in options
    exclude:
        list of potential keys in options to be excluded from being added to final dictionary

    Returns
    ----------
        dictionary of options that includes desired defaults if not present beforehand

    """
    exclude = {} if exclude is None else exclude
    opts = dict(options)
    for key in defaults:
        if key not in opts and key not in exclude:
            opts[key] = defaults[key]
    return opts


def _convert_evecs_to_qobjs(evecs, qobj):

    evecs_count = evecs.shape[1]
    ekets = np.empty((evecs_count,), dtype=object)
    ekets[:] = [
        Qobj(evecs[:, i], dims=[qobj.dims[0], [1] * len(qobj.dims[0])], type="ket")
        for i in range(evecs_count)
    ]
    norms = np.array([ket.norm() for ket in ekets])
    return ekets / norms


def scipy_dense_evals(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on scipy's (dense) eigh function.
    Only evals are returned.
    """
    m = matrix.full() if isinstance(matrix, Qobj) else matrix

    evals = sp.linalg.eigh(
        m, subset_by_index=(0, evals_count - 1), eigvals_only=True, **kwargs
    )
    return evals


def scipy_dense_esys(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on scipy's (dense) eigh function.
    Both evals and evecs are returned.
    """
    m = matrix.full() if isinstance(matrix, Qobj) else matrix

    evals, evecs = sp.linalg.eigh(m, subset_by_index=(0, evals_count - 1), **kwargs)

    # TODO: do we need this?
    # evals, evecs = order_eigensystem(evals, evecs)

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


def scipy_sparse_evals(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on scipy's (sparse) eigsh function.
    Only evals are returned.

    Note the convoluted convention when it comes to ordering and how it is related
    to the presence of `return_eigenvectors` parameter. See here for details:
    https://github.com/scipy/scipy/issues/9082

    This function ensures that:
    1. We always use the same "random" starting vector v0. Otherwise results show
    random behavior (small deviations between different runs, problem for pytests)
    2. We test for degenerate eigenvalues. If there are any, we orthogonalize the
    eigenvectors properly.
    """
    m = matrix.data if isinstance(matrix, Qobj) else matrix

    options = _setup_default_options(
        kwargs,
        dict(
            # sigma=0.0,
            # which="LA",
            sigma=None,
            which="SA",
            v0=settings.RANDOM_ARRAY[: matrix.shape[0]],
            return_eigenvectors=False,
        ),
    )
    evals = sp.sparse.linalg.eigsh(m, k=evals_count, **options)

    # have to reverse order if return_eigenvectors=False and which="SA"
    return evals[::-1]


def scipy_sparse_esys(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on scipy's (sparse) eigsh function.
    Both evals and evecs are returned.

    Note the convoluted convention when it comes to ordering and how it is related
    to the presence of `return_eigenvectors` parameter. See here for details:
    https://github.com/scipy/scipy/issues/9082

    This function ensures that:
    1. We always use the same "random" starting vector v0. Otherwise results show
    random behavior (small deviations between different runs, problem for pytests)
    2. We test for degenerate eigenvalues. If there are any, we orthogonalize the
    eigenvectors properly.

    TODO: right now, this is essentially a copy/paste of spectrum_utils.eigsh_safe()
        When the dust settles, should combine both into one.

    """
    m = matrix.data if isinstance(matrix, Qobj) else matrix

    options = _setup_default_options(
        kwargs,
        dict(
            # sigma=0.0,
            # which="LA",
            sigma=None,
            which="SA",
            v0=settings.RANDOM_ARRAY[: matrix.shape[0]],
            return_eigenvectors=True,
        ),
    )
    evals, evecs = sp.sparse.linalg.eigsh(m, k=evals_count, **options)
    if has_degeneracy(evals):
        evecs, _ = sp.linalg.qr(evecs, mode="economic")

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


def cupy_dense_evals(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on cuda's (dense) eigvalsh function.
    Only evals are returned.
    """
    try:
        import cupy as cp
    except:
        raise ImportError("Module cupy is not installed.")

    m = matrix.full() if isinstance(matrix, Qobj) else matrix

    evals_gpu = cp.linalg.eigvalsh(cp.asarray(m), **kwargs)
    cp.cuda.Stream.null.synchronize()  # wait for GPU to finish

    return evals_gpu[:evals_count].get()


def cupy_dense_esys(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on cupy's (dense) eigh function.
    Both evals and evecs are returned.
    """
    try:
        import cupy as cp
    except:
        raise ImportError("Module cupy is not installed.")

    m = matrix.full() if isinstance(matrix, Qobj) else matrix

    evals_gpu, evecs_gpu = cp.linalg.eigh(cp.asarray(m), **kwargs)
    cp.cuda.Stream.null.synchronize()  # wait for GPU to finish

    evals, evecs = evals_gpu[:evals_count].get(), evecs_gpu[:, :evals_count].get()

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


def cupy_sparse_evals(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on cupy's (sparse) eighs function.
    Only evals are returned.

    BROKEN: right now this is broken. eighs does not support which="SA",
    and given this problem: https://github.com/cupy/cupy/issues/6863
    can't find algebraically smallest evals.
    Could try to manually implement a shifting paradigm like in scipy's version of eigsh.
    """
    try:
        # from cupyx.scipy.sparse import csc_matrix
        from cupyx.scipy.sparse.linalg import eigsh
        import cupy as cp
    except:
        raise ImportError("Module cupyx (part of cupy) is not installed.")

    m = matrix.data if isinstance(matrix, Qobj) else matrix

    options = _setup_default_options(
        kwargs,
        dict(
            which="LA",
            return_eigenvectors=False,
        ),
    )
    # evals_gpu = eigsh(cp.asarray(matrix), k=evals_count, **options)
    evals_gpu = eigsh(cp.asarray(m), k=matrix.shape[0] - 3, **options)

    return evals_gpu.get()[::-1]


def cupy_sparse_esys(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on cupy's (sparse) eighs function.
    Both evals and evecs are returned.

    TODO: check and potentially update if this properly handles degenerate evals

    BROKEN: right now this is broken. eighs does not support which="SA",
    and given this problem: https://github.com/cupy/cupy/issues/6863
    can't find algebraically smallest evals.
    Could try to manually implement a shifting paradigm like in scipy's version of eigsh.
    """
    try:
        # from cupyx.scipy.sparse import csc_matrix
        from cupyx.scipy.sparse.linalg import eigsh
        import cupy as cp
    except:
        raise ImportError("Module cupyx (part of cupy) is not installed.")

    m = matrix.data if isinstance(matrix, Qobj) else matrix

    options = _setup_default_options(
        kwargs,
        dict(
            return_eigenvectors=True,
        ),
    )
    evals_gpu, evecs_gpu = eigsh(cp.asarray(m), k=evals_count, **options)

    evals, evecs = evals_gpu.get(), evecs_gpu.get()

    evecs = (
        _convert_evecs_to_qobjs(evecs, matrix) if isinstance(matrix, Qobj) else evecs
    )

    return evals, evecs


# Default values of various noise constants and parameters.
DIAG_METHODS = {
    "scipy_dense_evals": scipy_dense_evals,
    "scipy_dense_esys": scipy_dense_esys,
    "scipy_sparse_evals": scipy_sparse_evals,
    "scipy_sparse_esys": scipy_sparse_esys,
    "cupy_dense_evals": cupy_dense_evals,
    "cupy_dense_esys": cupy_dense_esys,
    "cupy_sparse_evals": cupy_sparse_evals,
    "cupy_sparse_esys": cupy_sparse_esys,
}


###################

# temporary; not used beyond this point


def scipy_eigh(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on scipy's (dense) eigh function.
    """
    # Note that eigh (dense) options formatting is different from eigsh (sparse).
    # Here we will accept ether `eigvals_only` or `return_eigenvectors` flag
    # in order to determine whether eigenvectors are to be calculated or not.
    # If both options given at the same time, we raise an error, warning the user.
    # If option not give, we assume evecs are to be calculated
    if kwargs.has_key("eigvals_only") and kwargs.has_key("return_eigenvectors"):
        raise RuntimeError(
            "Only one of 'eigenvals_only' or 'return_eigenvectors' can be given."
        )

    eigvals_only = kwargs.get("eigvals_only", None)
    if eigvals_only is None:
        eigvals_only = not kwargs.get("return_eigenvectors", True)

    options = _setup_default_options(
        kwargs,
        dict(
            eigvals_only=eigvals_only,
        ),
        exclude=["eigvals_only", "return_eigenvectors"],
    )

    if eigvals_only:
        evals = sp.linalg.eigh(matrix, subset_by_index=(0, evals_count - 1), **options)
        return np.sort(evals)

    else:
        evals, evecs = sp.linalg.eigh(
            matrix, subset_by_index=(0, evals_count - 1), **options
        )
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs


def scipy_eigsh(matrix, evals_count=6, **kwargs):
    """
    Diagonalization based on scipy's (sparse) eigsh function.

    This function ensures that:
    1. We always use the same "random" starting vector v0. Otherwise results show
    random behavior (small deviations between different runs, problem for pytests)
    2. We test for degenerate eigenvalues. If there are any, we orthogonalize the
    eigenvectors properly.

    TODO: right now, this is essentially a copy/paste of spectrum_utils.eigsh_safe()
        When the dust settles, should combine both into one.

    """
    options = _setup_default_options(
        kwargs,
        dict(
            sigma=0.0,
            which="LA",
            v0=settings.RANDOM_ARRAY[: matrix.shape[0]],
            return_eigenvectors=True,
        ),
    )
    if options["return_eigenvectors"]:
        evals, evecs = sp.sparse.linalg.eigsh(matrix, k=evals_count, **options)
        if has_degeneracy(evals):
            evecs, _ = sp.linalg.qr(evecs, mode="economic")
            return evals, evecs
    # return np.sort(sp.sparse.linalg.eigsh(matrix, k=evals_count, **options))

    return sp.sparse.linalg.eigsh(matrix, k=evals_count, **options)


