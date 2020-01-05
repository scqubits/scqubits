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

import scqubits.settings
from scqubits import Transmon, FileType
from scqubits.tests.conftest import StandardTests

scqubits.settings.FILE_FORMAT = FileType.h5


class TestTransmon(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = Transmon(EJ=None, EC=None, ng=None, ncut=None)  # dummy values, will read in actual values from h5
        cls.qubit_str = 'transmon'
        cls.op1_str = 'n_operator'
        cls.op2_str = 'n_operator'
        cls.param_name = 'ng'
        cls.param_list = np.linspace(-1, 1, 100)

    def test_plot_evals_vs_paramvals_EJ(self):
        # testname = 'transmon_6'
        ej_vals = self.qbt.EJ * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ', ej_vals)

    def test_plot_n_wavefunction(self):
        # testname = 'transmon_7'
        self.qbt.plot_n_wavefunction(esys=None, which=1, mode='real')
