# config.py

import numpy as np

EVECS_FILESUFFIX = '_evecs'
EVALS_FILESUFFIX = '_evals'
PARAMETER_FILESUFFIX = '.prm'

FILE_FORMAT = 'h5'   # choose 'csv' instead for generation of comma-separated values files

MODE_FUNC_DICT = {'abs_sqr': (lambda x: np.abs(x)**2),
                  'abs': (lambda x: np.abs(x)),
                  'real': (lambda x: np.real(x)),
                  'imag': (lambda x: np.imag(x))}

PHI_INDEX = 0
THETA_INDEX = 1
ZETA_INDEX = 2
