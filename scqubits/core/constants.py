# constants.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# arXiv:2107.08552 (2021). https://arxiv.org/abs/2107.08552
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import numpy as np

from typing_extensions import Final

# supported file types
FILE_TYPES: Final = [".h5 | .hdf5", ".csv"]

# helper functions for plotting wave functions
MODE_FUNC_DICT: Final = {
    "abs_sqr": (lambda x: np.abs(x) ** 2),
    "abs": np.abs,
    "real": np.real,
    "imag": np.imag,
}

# the following string manipulations are used in automatic generation of default
# ylabels of wavefunction plots
MODE_STR_DICT: Final = {
    "abs_sqr": (lambda x: "$|$" + x + "$|^2$"),
    "abs": (lambda x: "$|$" + x + "$|$"),
    "real": (lambda x: "Re " + x),
    "imag": (lambda x: "Im " + x),
}

# enumerate variables for zero-pi qubit
PHI_INDEX: Final = 0
THETA_INDEX: Final = 1
ZETA_INDEX: Final = 2
