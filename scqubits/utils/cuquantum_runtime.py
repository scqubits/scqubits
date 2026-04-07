# cuquantum_runtime.py
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


def get_cuquantum_workstream() -> Any:
    """Return the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    The stream is created on first successful call and cached for the process.
    Read-only access for introspection or advanced use with qutip-cuquantum;
    do not assign to module attributes to replace it.

    Returns
    -------
    object
        ``cuquantum.densitymat.WorkStream`` instance

    Raises
    ------
    ImportError
        If ``cuquantum.densitymat`` cannot be imported.
    """
    global _cuquantum_workstream
    if _cuquantum_workstream is not None:
        return _cuquantum_workstream
    try:
        import cuquantum.densitymat as cuDM
    except ImportError as exc:
        raise ImportError(
            "cuQuantum density-matrix could not be imported; install the cuquantum "
            "package with CUDA support."
        ) from exc
    _cuquantum_workstream = cuDM.WorkStream()
    return _cuquantum_workstream
