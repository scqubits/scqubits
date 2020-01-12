# constants.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import numpy as np

# file name suffices
PARAMETER_FILESUFFIX = '.prm'

# helper functions for plotting wave functions
MODE_FUNC_DICT = {'abs_sqr': (lambda x: np.abs(x)**2),
                  'abs': np.abs,
                  'real': np.real,
                  'imag': np.imag}

# enumerate variables for zero-pi qubit
PHI_INDEX = 0
THETA_INDEX = 1
ZETA_INDEX = 2

# supported keyword arguments for plotting and sweep_plotting functions, default values
SPECIAL_PLOT_OPTIONS = {
    'x_range',
    'y_range',
    'ymax',
    'figsize',
    'fig_ax',
    'filename'
}
