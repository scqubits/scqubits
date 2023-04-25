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

import warnings

from typing import Any, Type, Union

import matplotlib.font_manager as mpl_font
import numpy as np

from cycler import cycler


# Set format for output of warnings
def warning_on_one_line(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: str = None,
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

# Matplotlib options -------------------------------------------------------------------
# select fonts
font_selected = None
try:
    font_names = mpl_font.get_font_names()
    for font in ["IBM Plex Sans", "Roboto", "Arial", "Helvetica"]:
        if font in font_names:
            font_selected = font
            break
    if not font_selected:
        font_selected = "sans-serif"

except AttributeError:
    font_selected = "sans-serif"

# set matplotlib defaults for use in @mpl.rc_context
off_black = "0.2"
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
    "font.family": font_selected,
    "font.size": 11,
    "font.weight": 500,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.labelcolor": off_black,
    "ytick.labelcolor": off_black,
    "xtick.color": off_black,
    "ytick.color": off_black,
    "axes.labelcolor": off_black,
    "axes.edgecolor": off_black,
    "axes.titlecolor": off_black,
}


# toggle top and right axes on and off
DESPINE = True

# This is a setting for number of points in stencil to approximate derivatives
STENCIL = 7

# global random number generator for consistent initial state vector v0 in ARPACK
SEED = 63142
RNG = np.random.default_rng(seed=SEED)
RANDOM_ARRAY = RNG.random(size=10000000)

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
