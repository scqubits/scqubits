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

import pytest

from scqubits.tests.conftest import TESTDIR


def run():
    """
    Run the pytest scripts for scqubits.
    """
    # runs tests in scqubits.tests directory
    pytest.main(['-v', TESTDIR])
