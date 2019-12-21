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

from scqubits import Fluxonium
from scqubits.tests.conftest import BaseTest, DATADIR
from scqubits.utils.file_io import read_h5


class TestFluxonium(BaseTest):
    qbt = Fluxonium(EJ=None, EC=None, EL=None, flux=None, cutoff=None)
    # dummy values, will read  actual values from external user h5 files

    def test_eigenvals(self):
        TESTNAME = 'fluxonium_1'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        TESTNAME = 'fluxonium_2'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evecs_reference = datalist[1]
        return self.eigenvecs(evecs_reference)

    def test_plot_evals_vs_paramvals(self):
        # TESTNAME = 'fluxonium_3'
        flux_list = np.linspace(-0.5, 0.5, 50)
        return self.plot_evals_vs_paramvals('flux', flux_list)

    def test_get_spectrum_vs_paramvals(self):
        TESTNAME = 'fluxonium_4'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        flux_list = datalist[0]
        evals_reference = datalist[1]
        evecs_reference = datalist[2]
        return self.get_spectrum_vs_paramvals('flux', flux_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        TESTNAME = 'fluxonium_5'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        matelem_reference = datalist[0]
        return self.matrixelement_table('n_operator', matelem_reference)

    def test_plot_evals_vs_paramvals_EJ(self):
        # TESTNAME = 'fluxonium_6'
        ej_vals = self.qbt.EJ * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ', ej_vals)

    # TESTNAME = 'fluxonium_7'

    def test_plot_wavefunction(self):
        # TESTNAME = 'fluxonium_8'
        self.qbt.plot_wavefunction(esys=None, which=0, mode='real')
        self.qbt.plot_wavefunction(esys=None, which=1, mode='real')

    def test_plot_matrixelements(self):
        # TESTNAME = 'fluxonium_9'
        self.plot_matrixelements('n_operator', evals_count=10)

    def test_print_matrixelements(self):
        # TESTNAME = 'fluxonium_10'
        self.print_matrixelements('phi_operator')

    def test_plot_matelem_vs_paramvals(self):
        # TESTNAME = 'fluxonium_11'
        flux_list = np.linspace(-0.5, 0.5, 50)
        self.plot_matelem_vs_paramvals('n_operator', 'flux', flux_list, select_elems=[(0, 0), (1, 4), (1, 0)])
