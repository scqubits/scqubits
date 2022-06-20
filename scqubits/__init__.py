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

from scqubits import settings

# core
from scqubits.core.central_dispatch import CentralDispatch
from scqubits.core.cos2phi_qubit import Cos2PhiQubit
from scqubits.core.discretization import Grid1d
from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.generic_qubit import GenericQubit
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.core.noise import calc_therm_ratio
from scqubits.core.oscillator import KerrOscillator, Oscillator
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.storage import DataStore, SpectrumData
from scqubits.core.transmon import Transmon, TunableTransmon
from scqubits.core.units import (
    from_standard_units,
    get_units,
    get_units_time_label,
    set_units,
    show_supported_units,
    to_standard_units,
)
from scqubits.core.zeropi import ZeroPi
from scqubits.core.zeropi_full import FullZeroPi

from scqubits.core.symbolic_circuit import SymbolicCircuit
from scqubits.core.circuit import Circuit, truncation_template

# file IO
from scqubits.io_utils.fileio import read, write

# GUI
try:
    from scqubits.ui.gui import GUI
    from scqubits.ui.explorer_widget import Explorer
except NameError:
    warnings.warn(
        "scqubits: could not import GUI/Explorer - consider installing ipywidgets "
        "(optional dependency)?",
        ImportWarning,
    )

# for showing scqubits info
from scqubits.utils.misc import about, cite

# spectrum utils
from scqubits.utils.spectrum_utils import identity_wrap

# version
try:
    from scqubits.version import version as __version__
except ImportError:
    __version__ = "???"
    warnings.warn(
        "scqubits: missing version information - did scqubits install correctly?",
        ImportWarning,
    )
