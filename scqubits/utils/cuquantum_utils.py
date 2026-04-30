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

try:
    from cuquantum.densitymat import WorkStream
    _cuquantum_workstream = WorkStream()
except ImportError:
    _cuquantum_workstream = None


def set_cuquantum_workstream(workstream: "WorkStream"):
    """Set the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    Parameters
    ----------
    ################Describe the return object.

    Raises
    ------
    RuntimeError
        If the cuQuantum workstream is already set.
    """
    global _cuquantum_workstream
    if _cuquantum_workstream:
        raise RuntimeError(
            "cuQuantum workstream already set. Changing the workstream is not supported. "
            "Use get_cuquantum_workstream() to retrieve the existing workstream."
        )
    _cuquantum_workstream = workstream


def get_cuquantum_workstream() -> "WorkStream":
    """Return the cuQuantum density-matrix ``WorkStream`` used by scqubits.

    On successful import of ``cuquantum.densitymat``, a ``WorkStream`` is created when
    this module loads and reused for the process. Prefer
    :func:`get_cuquantum_workstream` over reading module attributes directly.

    Returns
    -------
    ################Describe the return object.

    Raises
    ------
    ImportError
        If ``cuquantum.densitymat`` cannot be imported.
    """
    if _cuquantum_workstream is None:
        raise ImportError(
            "cuDensityMat could not be imported; install the cuquantum "
            "package with CUDA support and qutip-cuquantum."
        )
    return _cuquantum_workstream
