# scqubits: superconducting qubits in Python
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#     Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#     All rights reserved.
#
#     This source code is licensed under the BSD-style license found in the
#     LICENSE file in the root directory of this source tree.
"""scqubits is an open-source Python library for simulating superconducting qubits.
It is meant to give the user a convenient way to obtain energy spectra of common
superconducting qubits, plot energy levels as a function of external parameters,
calculate matrix elements etc. The library further provides an interface to QuTiP,
making it easy to work with composite Hilbert spaces consisting of coupled
superconducting qubits and harmonic modes. Internally, numerics within scqubits is
carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib."""
#######################################################################################


import warnings

# core

# file IO

# GUI
try:
    from scqubits.ui.gui import GUI
except NameError:
    warnings.warn(
        "scqubits: could not import GUI - consider installing ipywidgets "
        "(optional dependency)?",
        ImportWarning,
    )

# for showing scqubits info

# spectrum utils

# version
try:
    from scqubits.version import version as __version__
except ImportError:
    __version__ = "???"
    warnings.warn(
        "scqubits: missing version information - did scqubits install correctly?",
        ImportWarning,
    )
