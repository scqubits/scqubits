# constants.py

import numpy as np
from enum import Enum, unique

# file name suffices
EVECS_FILESUFFIX = '_evecs'
EVALS_FILESUFFIX = '_evals'
PARAMETER_FILESUFFIX = '.prm'


# file types
@unique
class FileType(Enum):
    csv = 0
    h5 = 1

# helper functions for plotting wave functions
MODE_FUNC_DICT = {'abs_sqr': (lambda x: np.abs(x)**2),
                  'abs': (lambda x: np.abs(x)),
                  'real': (lambda x: np.real(x)),
                  'imag': (lambda x: np.imag(x))}

# enumerate variables for zero-pi qubit
PHI_INDEX = 0
THETA_INDEX = 1
ZETA_INDEX = 2
