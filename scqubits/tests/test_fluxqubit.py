# test_fluxqubit.py
# meant to be run with 'pytest'
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

from scqubits import FluxQubit
from scqubits.tests.conftest import StandardTests


class TestFluxQubit(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubit
        cls.file_str = 'fluxqubit'
        cls.op1_str = 'n_1_operator'
        cls.op2_str = 'n_2_operator'
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.45, 0.55, 50)
