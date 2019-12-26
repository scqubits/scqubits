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

from scqubits import Transmon
from scqubits.tests.conftest import BaseTest, DATADIR
from scqubits.utils.file_io import read_h5


class TestTransmon(BaseTest):
    qbt = Transmon(EJ=None, EC=None, ng=None, ncut=None)  # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        TESTNAME = 'transmon_1'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        TESTNAME = 'transmon_2'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evecs_reference = datalist[1]
        return self.eigenvecs(evecs_reference)

    def test_plot_evals_vs_paramvals_ng(self):
        TESTNAME = 'transmon_3'
        ng_list = np.linspace(-1, 1, 100)
        return self.plot_evals_vs_paramvals('ng', ng_list)

    def test_get_spectrum_vs_paramvals(self):
        TESTNAME = 'transmon_4'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        ng_list = datalist[0]
        evals_reference = datalist[1]
        evecs_reference = datalist[2]
        return self.get_spectrum_vs_paramvals('ng', ng_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        TESTNAME = 'transmon_5'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        matelem_reference = datalist[0]
        return self.matrixelement_table('n_operator', matelem_reference)

    def test_plot_evals_vs_paramvals_EJ(self):
        TESTNAME = 'transmon_6'
        ej_vals = self.qbt.EJ * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ', ej_vals)

    def test_plot_n_wavefunction(self):
        TESTNAME = 'transmon_7'
        self.qbt.plot_n_wavefunction(esys=None, which=1, mode='real')

    def test_plot_phi_wavefunction(self):
        TESTNAME = 'transmon_8'
        self.qbt.plot_phi_wavefunction(esys=None, which=6, mode='real')
        self.qbt.plot_phi_wavefunction(esys=None, which=(0, 3, 9), mode='abs_sqr')

    def test_plot_matrixelements(self):
        TESTNAME = 'transmon_9'
        self.plot_matrixelements('n_operator', evals_count=10)

    def test_print_matrixelements(self):
        TESTNAME = 'transmon_10'
        self.print_matrixelements('n_operator')

    def test_plot_matelem_vs_paramvals(self):
        TESTNAME = 'transmon_11'
        ng_list = np.linspace(-2, 2, 220)
        self.plot_matelem_vs_paramvals('n_operator', 'ng', ng_list, select_elems=[(0, 0), (1, 4), (1, 0)])
