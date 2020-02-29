# test_fluxonium.py
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

from scqubits import Fluxonium
from scqubits.tests.conftest import StandardTests


class TestFluxonium(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = Fluxonium
        cls.file_str = 'fluxonium'
        cls.op1_str = 'n_operator'
        cls.op2_str = 'phi_operator'
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.45, 0.55, 50)
