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
import scipy as sp


# supported file types
FILE_TYPES = ['.h5 | .hdf5',
              '.csv']

# helper functions for plotting wave functions
MODE_FUNC_DICT = {'abs_sqr': (lambda x: np.abs(x)**2),
                  'abs': np.abs,
                  'real': np.real,
                  'imag': np.imag}

# the following string manipulations are used in automatic generation of default ylabels of wavefunction plots
MODE_STR_DICT = {'abs_sqr': (lambda x: '$|$' + x + '$|^2$'),
                 'abs': (lambda x: '$|$' + x + '$|$'),
                 'real': (lambda x: 'Re ' + x),
                 'imag': (lambda x: 'Im ' + x)}

# enumerate variables for zero-pi qubit
PHI_INDEX = 0
THETA_INDEX = 1
ZETA_INDEX = 2

