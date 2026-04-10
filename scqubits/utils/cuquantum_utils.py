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

from typing import Any, Optional

# Lazy singleton; not on ``settings`` so user code cannot replace or clear it.
_cuquantum_workstream: Optional[Any] = None

try:
    from cuquantum.densitymat import WorkStream
except:
    WorkStream = None

def set_cuquantum_workstream(workstream: "WorkStream"):
    """Set the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    Parameters
    ----------
    workstream: WorkStream
        ``cuquantum.densitymat.WorkStream`` instance

    Raises
    ------
    RuntimeError
        If the cuQuantum workstream is already set.
    """
    global _cuquantum_workstream
    if _cuquantum_workstream is not None:
        raise RuntimeError("cuQuantum workstream already set. Use get_cuquantum_workstream() to retrieve it.")
    _cuquantum_workstream = workstream

def get_cuquantum_workstream() -> "WorkStream":
    """Return the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    The stream is created on first successful call and cached for the process.
    Read-only access for introspection or advanced use with qutip-cuquantum;
    do not assign to module attributes to replace it.

    Returns
    -------
    WorkStream
        ``cuquantum.densitymat.WorkStream`` instance

    Raises
    ------
    ImportError
        If ``cuquantum.densitymat`` cannot be imported.
    """
    global _cuquantum_workstream
    if _cuquantum_workstream is not None:
        return _cuquantum_workstream
    if WorkStream is None:
        raise ImportError(
            "cuDensityMat could not be imported; install the cuquantum "
            "package with CUDA support and qutip-cuquantum."
        )
    _cuquantum_workstream = WorkStream()
    return _cuquantum_workstream
