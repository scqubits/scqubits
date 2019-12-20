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
from scqubits import FluxQubit, FileType
from scqubits.tests.conftest import BaseTest, DATADIR

scqubits.settings.file_format = FileType.h5

class TestFluxQubit(BaseTest):

    qbt = FluxQubit(EJ1 = 1.0, EJ2 = 1.0, EJ3 = 1.0, ECJ1 = 1.0, ECJ2 = 1.0, ECJ3 = 1.0, ECg1 = 1.0, ECg2 = 1.0,
                    ng1 = 1.0, ng2 = 1.0, flux = 1.0, ncut = 1)
    # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        TESTNAME = 'fluxqubit_1'
        h5params, datalist = self.read_h5py(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        TESTNAME = 'fluxqubit_2'
        h5params, datalist = self.read_h5py(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        evecs_reference = datalist[1]
        return self.eigenvecs(evecs_reference)

    def test_plot_evals_vs_paramvals(self):
        TESTNAME = 'fluxqubit_3'
        flux_list = np.linspace(0.45, 0.55, 50)
        return self.plot_evals_vs_paramvals('flux', flux_list)

    def test_get_spectrum_vs_paramvals(self):
        TESTNAME = 'fluxqubit_4'
        h5params, datalist = self.read_h5py(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        flux_list = datalist[0]
        evals_reference = datalist[1]
        evecs_reference = datalist[2]
        return self.get_spectrum_vs_paramvals('flux', flux_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        TESTNAME = 'fluxqubit_5'
        h5params, datalist = self.read_h5py(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        matelem_reference = datalist[0]
        return self.matrixelement_table('n_1_operator', matelem_reference)

    def test_plot_evals_vs_paramvals_EJ(self):
        TESTNAME = 'fluxqubit_6'
        ej_vals = self.qbt.EJ1 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ1', ej_vals)

    # TESTNAME = 'fluxqubit_7'

    def test_plot_wavefunction(self):
        TESTNAME = 'fluxqubit_8'
        self.qbt.plot_wavefunction(esys=None, which=5, mode='real')

    def test_plot_matrixelements(self):
        TESTNAME = 'fluxqubit_9'
        self.plot_matrixelements('n_2_operator', evals_count=10)

    def test_print_matrixelements(self):
        TESTNAME = 'fluxqubit_10'
        self.print_matrixelements('n_2_operator')

    def test_plot_matelem_vs_paramvals(self):
        TESTNAME = 'fluxqubit_11'
        flux_list = np.linspace(0.45, 0.55, 50)
        self.plot_matelem_vs_paramvals('n_1_operator', 'flux', flux_list, select_elems=[(0, 0), (1, 4), (1, 0)])
