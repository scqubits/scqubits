"""
scqubits: superconducting qubits in Python
===========================================

[J. Koch](https://github.com/jkochNU), [P. Groszkowski](https://github.com/petergthatsme)

scqubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within scqubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.
"""
# settings.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
#######################################################################################################################

import warnings
import matplotlib as mpl
from cycler import cycler


# Set format for output of warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "{}: {}\n {}: {}".format(category.__name__, message, filename, lineno)


warnings.formatwarning = warning_on_one_line


# a switch for displaying of progress bar; default: show only in ipython
PROGRESSBAR_DISABLED = False
try:
    if __IPYTHON__:
        IN_IPYTHON = True
except NameError:
    PROGRESSBAR_DISABLED = True
    IN_IPYTHON = False

# run ParameterSweep directly upon initialization
AUTORUN_SWEEP = True

# enable/disable the CENTRAL_DISPATCH system
DISPATCH_ENABLED = True

# For parallel processing ----------------------------------------------------------------------------------------------
# store processing pool once generated
POOL = None
# number of cores to be used by default in methods that enable parallel processing
NUM_CPUS = 1

# Select multiprocessing library
# Options:  'multiprocessing'
#           'pathos'
MULTIPROC = 'pathos'

# Matplotlib options ---------------------------------------------------------------------------------------------------
# set custom matplotlib color cycle
mpl.rcParams['axes.prop_cycle'] = cycler(color=["#016E82",
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
                                                "#F9E6BE"])

# set matplotlib defaults
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial, Helvetica, DejaVu Sans"
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# toggle top and right axes on and off
DESPINE = True
