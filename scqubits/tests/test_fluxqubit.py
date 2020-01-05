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

import scqubits.settings
from scqubits import FluxQubit, FileType
from scqubits.tests.conftest import StandardTests

scqubits.settings.FILE_FORMAT = FileType.h5


class TestFluxQubit(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = FluxQubit(EJ1=None, EJ2=None, EJ3=None, ECJ1=None, ECJ2=None, ECJ3=None, ECg1=None, ECg2=None,
                            ng1=None, ng2=None, flux=None, ncut=None)
        cls.qubit_str = 'fluxqubit'
        cls.op1_str = 'n_1_operator'
        cls.op2_str = 'n_2_operator'
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.45, 0.55, 50)

    def test_plot_wavefunction(self):
        self.qbt.plot_wavefunction(esys=None, which=5, mode='real')

    def test_plot_evals_vs_paramvals_EJ(self):
        ej_vals = self.qbt.EJ1 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ1', ej_vals)
