# test_cos2phi_qubit.py
# meant to be run with 'pytest'
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np
import pytest

from scqubits import Cos2PhiQubit
from scqubits.tests.conftest import StandardTests


@pytest.mark.usefixtures("io_type")
class TestCos2PhiQubit(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = Cos2PhiQubit
        cls.file_str = "cos2phiqubit"
        cls.op1_str = "n_theta_operator"
        cls.op2_str = "zeta_operator"
        cls.param_name = "flux"
        cls.param_list = np.linspace(0, 0.5, 5)
