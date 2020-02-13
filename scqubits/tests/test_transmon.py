# test_transmon.py
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
import pytest

from scqubits import Transmon
from scqubits.tests.conftest import StandardTests


@pytest.mark.usefixtures("num_cpus", "io_type")
class TestTransmon(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = Transmon
        cls.file_str = 'transmon'
        cls.op1_str = 'n_operator'
        cls.op2_str = 'n_operator'
        cls.param_name = 'ng'
        cls.param_list = np.linspace(-1, 1, 100)

    def test_plot_n_wavefunction(self):
        self.qbt = Transmon(EJ=1.0, EC=1.0, ng=0.0, ncut=10)
        self.qbt.plot_n_wavefunction(esys=None, which=1, mode='real')
