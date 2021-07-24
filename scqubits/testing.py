# testing.py
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

import pytest

from scqubits.tests.conftest import TESTDIR


def run():
    """
    Run the pytest scripts for scqubits.
    """
    # runs tests in scqubits.tests directory
    pytest.main(["-v", TESTDIR])
