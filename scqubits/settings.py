"""
scqubits: superconducting qubits in Python
===========================================

[J. Koch](https://github.com/jkochNU), [P. Groszkowski](https://github.com/petergthatsme)

scqubits is an open-source Python library for simulating superconducting qubits.
It is meant to give the user a convenient way to obtain energy spectra of common
superconducting qubits, plot energy levels as a function of external parameters,
calculate matrix elements etc. The library further provides an interface to QuTiP,
making it easy to work with composite Hilbert spaces consisting of coupled
superconducting qubits and harmonic modes. Internally, numerics within scqubits is
carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.
"""

# settings.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
#######################################################################################

from __future__ import annotations

import warnings

from typing import Any, Optional, Type

import matplotlib.font_manager as mpl_font
import numpy as np

from cycler import cycler


# Set format for output of warnings
def warning_on_one_line(
    message: Warning | str,
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    return "{}: {}\n {}: {}".format(category.__name__, message, filename, lineno)


warnings.formatwarning = warning_on_one_line


# Function checking whether code is run from a jupyter notebook or inside ipython
def executed_in_ipython():
    try:  # inside ipython, the function get_ipython is always in globals()
        shell = get_ipython().__class__.__name__
        if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
            return True  # Jupyter notebook or qtconsole of IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# a switch for displaying of progress bar; default: show only in ipython
if executed_in_ipython():
    PROGRESSBAR_DISABLED = False
    IN_IPYTHON = True
else:
    PROGRESSBAR_DISABLED = True
    IN_IPYTHON = False

# use vector graphics display in jupyter
if executed_in_ipython():
    import matplotlib_inline.backend_inline

    matplotlib_inline.backend_inline.set_matplotlib_formats("pdf", "svg")


# run ParameterSweep directly upon initialization
AUTORUN_SWEEP = True

# enable/disable the CENTRAL_DISPATCH system
DISPATCH_ENABLED = True

# For parallel processing --------------------------------------------------------------
# store processing pool once generated
POOL: Any = None
# number of cores to be used by default in methods that enable parallel processing
NUM_CPUS = 1

# Select multiprocessing library
# Options:  'multiprocessing'
#           'pathos'
MULTIPROC = "pathos"

# Cap BLAS/OpenMP threads *per worker process* during parallel sweeps
# (num_cpus > 1). Must be a positive int or None; with the default (None) the
# environment is left untouched. When many small diagonalizations are swept, the
# worker processes and the BLAS thread pool otherwise oversubscribe the cores
# (num_cpus x BLAS-threads >> physical cores); setting this to a small value
# (e.g. 1, or cores // num_cpus) avoids that.
#
# The cap reaches workers via the thread-count environment variables, which
# spawn-based workers (macOS, Windows) re-read when they re-import numpy/scipy.
# On Linux, where workers are fork-based, the cap relies on 'threadpoolctl'
# retuning the already-loaded BLAS in the parent before the fork; without it,
# export OPENBLAS_NUM_THREADS (etc.) in the shell *before* importing scqubits.
# It has no effect on a BLAS that exposes no thread control (e.g. numpy on Apple
# Accelerate), but scqubits' eigensolvers use scipy's OpenBLAS, which is
# controllable. The cap is applied only while the worker pool is created; the
# parent environment is restored afterwards.
MULTIPROC_BLAS_THREADS: Optional[int] = None

# Auto-parallelization (opt-in) --------------------------------------------------------
# When True, sweep/spectrum methods called WITHOUT an explicit num_cpus behave as if
# num_cpus="auto": the parallel-tuning heuristic (scqubits.recommend_parallelization)
# picks num_cpus plus a per-worker BLAS-thread cap from the workload and applies them
# live, with no kernel restart. Default False keeps current behavior and avoids
# surprising plain scripts, where spawning workers needs an `if __name__ == "__main__":`
# guard. Per-call opt-in via num_cpus="auto" works regardless of this flag.
AUTO_PARALLEL = False

# Where the one-time machine calibration produced by
# scqubits.calibrate_parallelization() is stored and read from. None uses the
# default ~/.scqubits/parallel_calibration.json. When a calibration is present the
# parallelization heuristic replaces its default break-even with a measured one.
PARALLEL_CALIBRATION_PATH: Optional[str] = None

# Automatic sparse diagonalization (prototype) -----------------------------------------
# When evals_method / esys_method are left at their default (None), use sparse
# scipy `eigsh` instead of dense QuTiP diagonalization when only a small fraction of a
# large spectrum is requested -- the regime of dressed spectra of composite
# HilbertSpaces, where sparse `eigsh` is dramatically faster than dense (and dense may
# not even fit in memory). Falls back to the dense QuTiP path if the sparse solver
# raises. Set AUTO_SPARSE_DIAG = False to restore the always-dense default.
AUTO_SPARSE_DIAG = True
# Minimum Hilbert-space dimension at which auto sparse diagonalization is considered.
SPARSE_DIAG_MIN_DIM = 1000
# Auto sparse is used only when evals_count <= SPARSE_DIAG_MAX_EVALS_FRAC * dim
# (sparse `eigsh` only pays off when computing few of many eigenstates).
SPARSE_DIAG_MAX_EVALS_FRAC = 0.1

# Matplotlib options -------------------------------------------------------------------
# select fonts
FONT_SELECTED = None
try:
    font_names = mpl_font.get_font_names()
    for font in ["IBM Plex Sans", "Roboto", "Arial", "Helvetica"]:
        if font in font_names:
            FONT_SELECTED = font
            break
    if not FONT_SELECTED:
        FONT_SELECTED = "sans-serif"
except AttributeError:
    FONT_SELECTED = "sans-serif"

# set matplotlib defaults for use in @mpl.rc_context
OFF_BLACK = "0.2"
matplotlib_settings = {
    "axes.prop_cycle": cycler(
        color=[
            "#016E82",
            "#333795",
            "#2E5EAC",
            "#4498D3",
            "#CD85B9",
            "#45C3D1",
            "#AA1D3F",
            "#F47752",
            "#19B35A",
            "#EDE83B",
            "#ABD379",
            "#F9E6BE",
        ]
    ),
    "font.family": FONT_SELECTED,
    "font.size": 11,
    "font.weight": 500,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": 500,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.labelcolor": OFF_BLACK,
    "ytick.labelcolor": OFF_BLACK,
    "xtick.color": OFF_BLACK,
    "ytick.color": OFF_BLACK,
    "axes.labelcolor": OFF_BLACK,
    "axes.edgecolor": OFF_BLACK,
    "axes.titlecolor": OFF_BLACK,
}


# toggle top and right axes on and off
DESPINE = True

# This is a setting for number of points in stencil to approximate derivatives
STENCIL = 7

# global random number generator for consistent initial state vector v0 in ARPACK
_SEED = 63142
_RNG = np.random.default_rng(seed=_SEED)
RANDOM_ARRAY = _RNG.random(size=10000000)

# toggle fuzzy value-based slicing and warnings about it on and off
FUZZY_SLICING = False
FUZZY_WARNING = True

# Enable/disable warning about default used in t1 coherence calculations
T1_DEFAULT_WARNING = True

# Overlap threshold in establishing a map between dressed states and bare product states
# (lookups need to be manually regenerated for a change by the user to take effect
OVERLAP_THRESHOLD = 0.5

# Settings for Circuit and SymbolicCircuit class.
# The following determines the threshold for the number of nodes above which the
# symbolic inversion of the capacitance matrix is skipped.
SYM_INVERSION_MAX_NODES = 3
