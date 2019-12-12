# testing.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import nose
import os
import shutil

def run():
    """
    Run the nose test scripts for scqubits.
    """
    # runs tests in scqubits.tests directory
    if not os.path.exists('./_data'):
        os.makedirs('./_data')
    else:
        raise RuntimeError('Temporary data directory ./_data already exists. Terminating to avoid unwanted overwriting.')
    nose.run(defaultTest="scqubits.tests", argv=['nosetests', '-v', '--nologcapture'])
    shutil.rmtree('./_data')
