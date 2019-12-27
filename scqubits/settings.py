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

from cycler import cycler
import matplotlib as mpl
from scqubits.utils.constants import FileType

file_format = FileType.h5   # choose FileType.csv instead for generation of comma-separated values files

# a switch for displaying of progress bar; default: show only in ipython
progressbar_disabled = False
try:
    if __IPYTHON__:
        in_ipython = True
except NameError:
    progressbar_disabled = True
    in_ipython = False

# default energy units
DEFAULT_ENERGY_UNITS = ' [GHz]'


# define settings for tqdm progressbar
TQDM_KWARGS = {'disable': progressbar_disabled,
               'leave': False}

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
