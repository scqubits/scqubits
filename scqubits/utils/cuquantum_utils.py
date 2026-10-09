# cuquantum_utils.py
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

"""Process-wide cuQuantum density-matrix runtime resources (workstream, etc.)."""

import scqubits.settings as settings

_CUQUANTUM_BACKEND_ERROR = (
    "Running scqubits code inside qutip-cuquantum's CuQuantumBackend is not "
    "supported. Exit the backend context before calling scqubits."
)

try:
    from cuquantum.densitymat import WorkStream

    _HAS_CUQUANTUM = True
except ImportError:
    _HAS_CUQUANTUM = False


def max_eigvals(dimension: int) -> int:
    """Return the largest eigenvalue count the cuQuantum Krylov solver accepts.

    The limit is ``floor((D - b) / (2 * b * r))``, where ``D`` is ``dimension``,
    ``b`` is ``settings.CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE``, and ``r`` is
    ``settings.CUQUANTUM_MAX_BUFFER_RATIO``.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension of the operator being diagonalized.

    Returns
    -------
    int
        Maximum number of eigenvalues the solver can return for this dimension
        and the current Krylov settings.

    Raises
    ------
    ValueError
        If the dimension is too small for the current Krylov settings, so the
        limit would be below 1.
    """
    block = settings.CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE
    ratio = settings.CUQUANTUM_MAX_BUFFER_RATIO
    allowed = (dimension - block) // (2 * block * ratio)
    if allowed < 1:
        raise ValueError(
            "Hilbert space dimension is too small for the cuQuantum Krylov "
            "settings scqubits.settings.CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE and "
            "scqubits.settings.CUQUANTUM_MAX_BUFFER_RATIO."
        )
    return allowed


# Lazy singleton; not on ``settings`` so user code cannot replace or clear it.
_cuquantum_workstream = None


def set_cuquantum_workstream(workstream: "WorkStream") -> None:
    """Set the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    Must be called before the first :func:`get_cuquantum_workstream`; afterwards
    the workstream is fixed for the process.

    Parameters
    ----------
    workstream
        A ``cuquantum.densitymat.WorkStream`` instance containing the library
        handle, CUDA stream, workspace, and configuration parameters. Handles
        GPU memory allocation and synchronization.

    Raises
    ------
    RuntimeError
        If the cuQuantum workstream is already set.
    """
    global _cuquantum_workstream
    if _cuquantum_workstream is not None:
        raise RuntimeError(
            "cuQuantum workstream already set. Changing the workstream is not "
            "supported. Use get_cuquantum_workstream() to retrieve the existing "
            "workstream."
        )
    _cuquantum_workstream = workstream


def get_cuquantum_workstream() -> "WorkStream":
    """Return the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    The stream is created on first call and cached for the process. Prefer this
    accessor over reading module attributes directly.

    Returns
    -------
    WorkStream
        A ``cuquantum.densitymat.WorkStream`` instance containing the library
        handle, CUDA stream, workspace, and configuration parameters. Handles GPU
        memory allocation and synchronization.

    Raises
    ------
    ImportError
        If ``cuquantum.densitymat`` cannot be imported.
    """
    global _cuquantum_workstream
    if _cuquantum_workstream is not None:
        return _cuquantum_workstream
    if not _HAS_CUQUANTUM:
        raise ImportError(
            "cuDensityMat could not be imported; install the cuquantum "
            "package with CUDA support and qutip-cuquantum."
        )
    _cuquantum_workstream = WorkStream()
    return _cuquantum_workstream
