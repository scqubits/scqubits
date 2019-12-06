# settings.py
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################

from sc_qubits.utils.constants import FileType

file_format = FileType.h5   # choose FileType.csv instead for generation of comma-separated values files

# switch for display of progress bar; default: show only in ipython
try:
    __IPYTHON__
    progressbar_enabled = True
except:
    progressbar_enabled = False

