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
from scqubits.utils.cuquantum_utils import get_cuquantum_workstream
from scqubits.io_utils.fileio_qutip import QutipEigenstates
from scqubits.utils.spectrum_utils import order_eigensystem, has_degeneracy

import copy
import numpy as np
import qutip as qt
import scipy as sp
import scqubits.settings as settings
import warnings


def _dict_merge(
    d: Dict[str, Any],
    d_other: Dict[str, Any],
    exclude: Union[List[str], None] = None,
    overwrite=False,
) -> Dict[str, Any]:
    """Selective dictionary merge. This function makes a copy of the given dictionary
    `d` and selectively updates/adds entries from `d_other`, as long as the keys are not
    given in `exclude`. Whether entries in `d` are overwritten by entries in `d_other`
    is determined by the value of the `overwrite` parameter.

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
) -> Union[ndarray, csc_matrix]:
    """Casts a given operator (possibly given as a `Qobj`) into a required form
    ('sparse' or 'dense' numpy array or scipy martrix) as defined by `cast_to`
    parameter.

    Operators of the type `Qobj` are first converted to a `ndarray` or a
    scipy sparse matrix form (depending on the given object's underlying
    `dtype`).

    Later those are converted (or not) to a dense or spare forms depending
    on whether `force_cast` is set.

    NOTE: Currently we only ever cast to the csc format for sparse
    matrices. Internally `Qobj` uses the csr or dia formats instead, however.
    It could be worthwhile to also use those representations directly
    whenever it makes sense, and avoid unnecessary conversions.

    Parameters
    ----------
    matrix: `Qobj`, `ndarray` or scipy's sparse matrix format
        matrix given as an ndarray, Qobj, or scipy's sparse matrix format
    cast_to: str
        string representing the format that matrix should be cast into: 'sparse' or 'dense'
    force_cast: bool
        determines if explicit casting to dense or sparse format should be always
        performed

    Returns
    ----------
        matrix in the sparse or dense form
    """
    if cast_to not in ["sparse", "dense"]:
        raise ValueError("Can only cast matrix to 'sparse' or 'dense' forms.")

    # First, if we are dealing with a Qobj, we convert it to either
    # an ndarray or a scipy sparse matrix.
    if isinstance(matrix, Qobj):
        if qt.__version__ >= "5.0.0":
            if matrix.dtype == qt.core.data.dense.Dense:
                converted_mat = matrix.full()
            else:
                # This could be costly if the data is in a "Dia"
                # form. In the future we may want to support other
                # formats as well.
                converted_mat = matrix.to("CSR").data_as()
        else:
            # In previous versions of qutip data was always in the csr form
            converted_mat = matrix.data
    else:
        converted_mat = matrix

    # Next, we do casting dense or sparse (CSC) representation
    # if force_cast is True
    if force_cast:
        if cast_to == "dense" and not isinstance(converted_mat, ndarray):
            converted_mat = converted_mat.toarray()
        if cast_to == "sparse":
            converted_mat = csc_matrix(converted_mat)

    return converted_mat


def _convert_evecs_to_qobjs(evecs: ndarray, hamiltonian_qobj, wrap: bool = False) -> ndarray:
    """Converts an `ndarray` containing eigenvectors (that would be typically returned
    from a diagonalization routine, such as `eighs` or `eigh`), to a numpy array of
    qutip's Qobjs. Potentially also wraps those into
    `scqubits.io_utils.fileio_qutip.QutipEigenstates`.

    Parameters
    ----------
    evecs:
        ndarray of eigenvectors (as columns)
    hamiltonian_qobj:
        Hamiltonian as ``Qobj``; used to extract the tensor product structure
    wrap:
        determines if we wrap results in QutipEigenstates

    Returns
    ----------
        eigenvectors represented in terms of Qobjs
    """
    evecs_count = evecs.shape[1]
    evec_dims = [hamiltonian_qobj.dims[0], [1] * len(hamiltonian_qobj.dims[0])]
    evecs_qobj = np.empty((evecs_count,), dtype=object)

    for i in range(evecs_count):
        v = Qobj(evecs[:, i], dims=evec_dims)
        evecs_qobj[i] = v / v.norm()

    # Optionally, we wrap the resulting array in QutipEigenstates as is done in HilbertSpace.
    if wrap:
        evecs_qobj = evecs_qobj.view(QutipEigenstates)

    return evecs_qobj


### scipy based routines ####


def evals_scipy_dense(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """Diagonalization based on scipy's (dense) `eigh` function. Only evals are
    returned.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of the Hamiltonian
    """
    dense_matrix = _cast_matrix(hamiltonian, "dense")

    evals = sp.linalg.eigh(
        dense_matrix,
        subset_by_index=(0, evals_count - 1),
        eigvals_only=True,
        **kwargs,
    )
    return evals


def esys_scipy_dense(
    hamiltonian, evals_count, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on scipy's (dense) eigh function. Both evals and evecs are
    returned.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if hamiltonian is a Qobj instance
    """
    dense_matrix = _cast_matrix(hamiltonian, "dense")

    evals, evecs = sp.linalg.eigh(
        dense_matrix, subset_by_index=(0, evals_count - 1), **kwargs
    )

    evecs = (
        _convert_evecs_to_qobjs(evecs, hamiltonian) if isinstance(hamiltonian, Qobj) else evecs
    )

    return evals, evecs


def evals_scipy_sparse(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """Diagonalization based on scipy's (sparse) `eigsh` function. Only evals are
    returned.

    Note the convoluted convention when it comes to ordering and how it is related
    to the presence of `return_eigenvectors` parameter. See here for details:
    https://github.com/scipy/scipy/issues/9082

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of the Hamiltonian
    """
    csc_matrix = _cast_matrix(hamiltonian, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            v0=settings.RANDOM_ARRAY[: hamiltonian.shape[0]],
            return_eigenvectors=False,
        ),
        kwargs,
        overwrite=True,
    )
    evals = sp.sparse.linalg.eigsh(csc_matrix, k=evals_count, **options)

    # have to reverse order if return_eigenvectors=False and which="SA"
    return evals[::-1]


def esys_scipy_sparse(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on scipy's (sparse) `eigsh` function. Both evals and evecs
    are returned.

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
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if hamiltonian is a Qobj instance
    """
    csc_matrix = _cast_matrix(hamiltonian, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            v0=settings.RANDOM_ARRAY[: hamiltonian.shape[0]],
            return_eigenvectors=True,
        ),
        kwargs,
        overwrite=True,
    )
    evals, evecs = sp.sparse.linalg.eigsh(csc_matrix, k=evals_count, **options)

    if has_degeneracy(evals):
        evecs, _ = sp.linalg.qr(evecs, mode="economic")

    evecs = (
        _convert_evecs_to_qobjs(evecs, hamiltonian) if isinstance(hamiltonian, Qobj) else evecs
    )

    return evals, evecs


### primme based routines ####


def evals_primme_sparse(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """Diagonalization based on primme's (sparse) `eigsh` function. Only evals are
    returned.

    Requires that the primme library is installed.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of the Hamiltonian
    """
    try:
        import primme
    except:
        raise ImportError("Package primme is not installed.")

    csc_matrix = _cast_matrix(hamiltonian, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=False,
        ),
        kwargs,
        overwrite=True,
    )

    evals = primme.eigsh(csc_matrix, k=evals_count, **options)

    return evals


def esys_primme_sparse(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on primme's (sparse) `eigsh` function. Both evals and evecs
    are returned.

    Requires that the primme library is installed.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if hamiltonian is a Qobj instance
    """
    try:
        import primme
    except:
        raise ImportError("Package primme is not installed.")

    csc_matrix = _cast_matrix(hamiltonian, "sparse")

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=True,
        ),
        kwargs,
        overwrite=True,
    )

    evals, evecs = primme.eigsh(csc_matrix, k=evals_count, **options)

    evecs = (
        _convert_evecs_to_qobjs(evecs, hamiltonian) if isinstance(hamiltonian, Qobj) else evecs
    )

    return evals, evecs


### cupy based routines ####


def evals_cupy_dense(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """Diagonalization based on cupy's (dense) `eighvalsh` function Only evals are
    returned.

    Requires that the cupy library is installed.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of the Hamiltonian
    """
    try:
        import cupy as cp
    except:
        raise ImportError("Package cupy is not installed.")

    dense_matrix = _cast_matrix(hamiltonian, "dense")

    evals_gpu = cp.linalg.eigvalsh(cp.asarray(dense_matrix), **kwargs)
    cp.cuda.Stream.null.synchronize()  # wait for GPU to finish

    return evals_gpu[:evals_count].get()


def esys_cupy_dense(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on cupy's (dense) `eigh` function. Both evals and evecs are
    returned.

    Requires that the cupy library is installed.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if hamiltonian is a Qobj instance
    """
    try:
        import cupy as cp
    except:
        raise ImportError("Package cupy is not installed.")

    dense_matrix = _cast_matrix(hamiltonian, "dense")

    evals_gpu, evecs_gpu = cp.linalg.eigh(cp.asarray(dense_matrix), **kwargs)
    cp.cuda.Stream.null.synchronize()  # wait for GPU to finish

    evals, evecs = evals_gpu[:evals_count].get(), evecs_gpu[:, :evals_count].get()

    evecs = (
        _convert_evecs_to_qobjs(evecs, hamiltonian) if isinstance(hamiltonian, Qobj) else evecs
    )

    return evals, evecs


def evals_cupy_sparse(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> ndarray:
    """Diagonalization based on cupy's (sparse) `eigsh` function. Only evals are
    returned.

    Requires that the cupy (and cupyx) library is installed.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of the Hamiltonian
    """
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
        from cupyx.scipy.sparse.linalg import eigsh
    except:
        raise ImportError("Package cupyx (part of cupy) is not installed.")

    csc_matrix = cp_csc_matrix(_cast_matrix(hamiltonian, "sparse"))

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=False,
        ),
        kwargs,
        overwrite=True,
    )
    evals_gpu = eigsh(csc_matrix, k=evals_count, **options)

    # return evals_gpu.get()[::-1]
    return evals_gpu.get()


def esys_cupy_sparse(
    hamiltonian: Union[ndarray, csc_matrix, Qobj], evals_count: int, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on cupy's (sparse) eigsh function. Both evals and evecs are
    returned.

    Requires that the cupy library is installed.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if hamiltonian is a Qobj instance
    """
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
        from cupyx.scipy.sparse.linalg import eigsh
    except:
        raise ImportError("Package cupyx (part of cupy) is not installed.")

    csc_matrix = cp_csc_matrix(_cast_matrix(hamiltonian, "sparse"))

    options = _dict_merge(
        dict(
            which="SA",
            return_eigenvectors=True,
        ),
        kwargs,
        overwrite=True,
    )
    evals_gpu, evecs_gpu = eigsh(csc_matrix, k=evals_count, **options)

    evals, evecs = evals_gpu.get(), evecs_gpu.get()

    evecs = (
        _convert_evecs_to_qobjs(evecs, hamiltonian) if isinstance(hamiltonian, Qobj) else evecs
    )

    return evals, evecs


### jax based routines ####


def evals_jax_dense(
    hamiltonian, evals_count, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on jax's (dense) jax.scipy.linalg.eigh function. Only
    eigenvalues are returned.

    If available, different backends/devics (e.g., particular GPUs) can be set
    though jax's interface, see https://jax.readthedocs.io/en/latest/user_guides.html

    Note, that jax's documentation is inconsistent, and `eigvals` and/or
    `subset_by_index` seems not to be implemented. Hence, here we calculate all the
    eigenvalues, but then only return the requested subset.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        eigenvalues of the Hamiltonian
    """
    try:
        import jax

        # jax defaults to single precision, but we need to default to double precision
        jax.config.update("jax_enable_x64", True)
    except:
        raise ImportError("Package jax is not installed.")

    dense_matrix = _cast_matrix(hamiltonian, "dense")

    # We explicitly cast to a numpy array
    evals = np.asarray(
        jax.scipy.linalg.eigh(dense_matrix, eigvals_only=True, **kwargs)
    )

    # In eigh, the eigvals options is not currently implemented, although listed
    # in the jax docs, hence we have to "manually" only return the number of
    # evals that the user requested. We also "cast" to a numpy array via np.asarray.
    return np.asarray(evals[:evals_count])


def esys_jax_dense(
    hamiltonian, evals_count, **kwargs
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, QutipEigenstates]]:
    """Diagonalization based on jax's (dense) jax.scipy.linalg.eigh function. Both evals
    and evecs are returned.

    If available, different backends/devics (e.g., particular GPUs) can be set
    though jax's interface, see https://jax.readthedocs.io/en/latest/user_guides.html

    Note, that jax's documentation is inconsistent, and `eigvals` and/or
    `subset_by_index` seems not to be implemented. Hence, here we calculate all the
    eigenvalues and eigenvectors, but then only return the requested subset.

    Parameters
    ----------
    hamiltonian:
        ndarray or qutip.Qobj to be diagonalized
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    ----------
        a tuple of eigenvalues and eigenvectors. Eigenvectors are Qobjs if hamiltonian is a Qobj instance
    """
    try:
        import jax

        # jax defaults to single precision, but we need to default to double precision
        jax.config.update("jax_enable_x64", True)
    except:
        raise ImportError("Package jax is not installed.")

    dense_matrix = _cast_matrix(hamiltonian, "dense")

    evals, evecs = jax.scipy.linalg.eigh(
        dense_matrix, eigvals_only=False, **kwargs
    )

    # In eigh, the eigvals options is not currently implemented, although listed
    # in the jax docs, hence we only "manually" select the number of evals/evecs
    # that the user requested. We also "cast" to a numpy array via np.asarray.
    evals, evecs = np.asarray(evals[:evals_count]), np.asarray(evecs[:, :evals_count])

    evecs = (
        _convert_evecs_to_qobjs(evecs, hamiltonian) if isinstance(hamiltonian, Qobj) else evecs
    )
    return evals, evecs


def esys_cuquantum(
    hamiltonian: Qobj, evals_count: int, **kwargs
) -> Tuple[ndarray, QutipEigenstates]:
    """Diagonalization via cuQuantum density-matrix OperatorSpectrumSolver (Krylov).

    Requires qutip-cuquantum, cuQuantum, and cupy. Uses the shared workstream from
    :func:`~scqubits.utils.cuquantum_utils.get_cuquantum_workstream` and
    CUQUANTUM_* Krylov parameters from ``settings``.

    Parameters
    ----------
    hamiltonian:
        qutip.Qobj Hamiltonian to be diagonalized.
    evals_count:
        how many eigenvalues/vectors should be returned
    kwargs:
        optional settings that are passed onto the diagonalization routine

    Returns
    -------
    tuple of ndarray (eigenvalues) and QutipEigenstates (eigenvectors as Qobj)
    """
    try:
        import cupy
        import qutip_cuquantum as qcu
        import cuquantum.densitymat as cudm
    except ImportError as e:
        raise ImportError(str(e)) from e

    ctx = get_cuquantum_workstream()

    subsys_dims = hamiltonian.dims[0]
    hspace_dim = hamiltonian.shape[0]

    min_krylov_block_size = settings.CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE
    max_buffer_ratio = settings.CUQUANTUM_MAX_BUFFER_RATIO
    max_restarts = settings.CUQUANTUM_MAX_RESTARTS

    config = cudm.OperatorSpectrumConfig(
        min_krylov_block_size=min_krylov_block_size,
        max_buffer_ratio=max_buffer_ratio,
        max_restarts=max_restarts,
    )

    if (
        min_krylov_block_size * max_buffer_ratio * evals_count
        > hspace_dim / 2 # ask nvidia if they can provide this on their end and provide in docs
    ):
        allowed_num_eigvals = int(
            np.ceil(
                hspace_dim / (2 *min_krylov_block_size * max_buffer_ratio)
            )
            - 1
        )
        raise ValueError(
            f"Too many eigenvalues requested. Maximum number of eigenvalues "
            f"allowed is {allowed_num_eigvals}. Reduce min_krylov_block_size, "
            f"max_buffer_ratio, or increase the Hilbert space dimension."
        )

    batch_size = 1 # densepurestate entry is not well explained. we don't know what it is. ## I think one can create a batch of initial states and solve them in parallel.

    init_states = [] # rename to seed_states
    for _ in range(evals_count):
        init_state = cudm.DensePureState(
            ctx, subsys_dims, batch_size, "complex128"
        )
        init_state.allocate_storage()
        init_state.storage[:] = cupy.random.randn(hspace_dim * batch_size)
        norm = init_state.norm()
        init_state.inplace_scale(1.0 / cupy.sqrt(norm))
        init_states.append(init_state)

    cudm_operator = qcu.CuQobjEvo(qt.QobjEvo(hamiltonian)).operator #This is a temporary solution. We need to wait for the formal convertion function.
    spectrum = cudm.OperatorSpectrumSolver(cudm_operator, "SA", True, config)
    spectrum.prepare(ctx, init_states[0], max_num_eigvals=evals_count)
    result = spectrum.compute(0.0, None, init_states, 1e-10)

    evals = result.evals[:, 0].get()
    evecs = np.empty((evals_count,), dtype=object)

    with qcu.CuQuantumBackend(ctx):
        for i, evec in enumerate(result.evecs): #consider converting list on densepurestates to numpy array.
            evecs[i] = Qobj(
                qcu.state.CuState(evec).to_array(),
                dims=[subsys_dims, [1]],
            )

    return evals, evecs.view(QutipEigenstates)


def evals_cuquantum(
    hamiltonian: Union[Qobj], evals_count: int, **kwargs
) -> ndarray:
    return esys_cuquantum(hamiltonian, evals_count, **kwargs)[0]

# Default values of various noise constants and parameters.
DIAG_METHODS = {
    # scipy dense
    "evals_scipy_dense": evals_scipy_dense,
    "esys_scipy_dense": esys_scipy_dense,
    # scipy sparse
    "evals_scipy_sparse": evals_scipy_sparse,
    "esys_scipy_sparse": esys_scipy_sparse,
    "evals_scipy_sparse_SM": lambda hamiltonian, evals_count, **kwargs: evals_scipy_sparse(
        hamiltonian, evals_count, **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "esys_scipy_sparse_SM": lambda hamiltonian, evals_count, **kwargs: esys_scipy_sparse(
        hamiltonian, evals_count, **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "evals_scipy_sparse_LA_shift-inverse": lambda hamiltonian, evals_count, **kwargs: evals_scipy_sparse(
        hamiltonian,
        evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True),
    ),
    "esys_scipy_sparse_LA_shift-inverse": lambda hamiltonian, evals_count, **kwargs: esys_scipy_sparse(
        hamiltonian,
        evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True),
    ),
    "evals_scipy_sparse_LM_shift-inverse": lambda hamiltonian, evals_count, **kwargs: evals_scipy_sparse(
        hamiltonian,
        evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True),
    ),
    "esys_scipy_sparse_LM_shift-inverse": lambda hamiltonian, evals_count, **kwargs: esys_scipy_sparse(
        hamiltonian,
        evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True),
    ),
    # primme sparse
    "evals_primme_sparse": evals_primme_sparse,
    "esys_primme_sparse": esys_primme_sparse,
    "evals_primme_sparse_SM": lambda hamiltonian, evals_count, **kwargs: evals_primme_sparse(
        hamiltonian=hamiltonian,
        evals_count=evals_count,
        **_dict_merge(dict(which="SM"), kwargs, overwrite=True),
    ),
    "esys_primme_sparse_SM": lambda hamiltonian, evals_count, **kwargs: esys_primme_sparse(
        hamiltonian, evals_count, **_dict_merge(dict(which="SM"), kwargs, overwrite=True)
    ),
    "evals_primme_sparse_LA_shift-inverse": lambda hamiltonian, evals_count, **kwargs: evals_primme_sparse(
        hamiltonian=hamiltonian,
        evals_count=evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True),
    ),
    "esys_primme_sparse_LA_shift-inverse": lambda hamiltonian, evals_count, **kwargs: esys_primme_sparse(
        hamiltonian=hamiltonian,
        evals_count=evals_count,
        **_dict_merge(dict(which="LA", sigma=0), kwargs, overwrite=True),
    ),
    "evals_primme_sparse_LM_shift-inverse": lambda hamiltonian, evals_count, **kwargs: evals_primme_sparse(
        hamiltonian=hamiltonian,
        evals_count=evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True),
    ),
    "esys_primme_sparse_LM_shift-inverse": lambda hamiltonian, evals_count, **kwargs: esys_primme_sparse(
        hamiltonian=hamiltonian,
        evals_count=evals_count,
        **_dict_merge(dict(which="LM", sigma=0), kwargs, overwrite=True),
    ),
    # cupy dense
    "evals_cupy_dense": evals_cupy_dense,
    "esys_cupy_dense": esys_cupy_dense,
    # cupy sparse
    "evals_cupy_sparse": evals_cupy_sparse,
    "esys_cupy_sparse": esys_cupy_sparse,
    # jax dense
    "evals_jax_dense": evals_jax_dense,
    "esys_jax_dense": esys_jax_dense,
    # cuquantum
    "esys_cuquantum": esys_cuquantum,
    "evals_cuquantum": evals_cuquantum,
}
