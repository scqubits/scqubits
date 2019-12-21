# test_zeropi.py
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

import scqubits as qubit
from scqubits import ZeroPi
from scqubits.tests.conftest import BaseTest, DATADIR
from scqubits.utils.file_io import read_h5


class TestZeroPi(BaseTest):

    phi_grid = qubit.Grid1d(1, 2, 3)
    qbt = ZeroPi(grid=phi_grid, EJ=None, EL=None, ECJ=1, EC=None, ECS=2, ng=None, flux=None, ncut=None)
    # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        TESTNAME = 'zeropi_1'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        TESTNAME = 'zeropi_2'
        h5params, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        evecs_reference = datalist[1]
        return self.eigenvecs(evecs_reference)

    def test_plot_evals_vs_paramvals(self):
        TESTNAME = 'zeropi_3'
        flux_list = np.linspace(0, 0.5, 15)
        return self.plot_evals_vs_paramvals('flux', flux_list)

    def test_get_spectrum_vs_paramvals(self):
        TESTNAME = 'zeropi_4'
        hfile_root, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(hfile_root)
        flux_list = datalist[0]
        evals_reference = datalist[1]
        evecs_reference = datalist[2]
        return self.get_spectrum_vs_paramvals('flux', flux_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        TESTNAME = 'zeropi_5'
        h5file_root, datalist = read_h5(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5file_root)
        matelem_reference = datalist[0]
        return self.matrixelement_table('n_theta_operator', matelem_reference)

    def test_plot_evals_vs_paramvals_EJ(self):
        TESTNAME = 'zeropi_6'
        ej_vals = self.qbt.EJ * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ', ej_vals)

    #    TESTNAME = 'zeropi_7'

    def test_plot_wavefunction(self):
        TESTNAME = 'zeropi_8'
        self.qbt.plot_wavefunction(esys=None, which=4, mode='real', zero_calibrate=True)
        self.qbt.plot_potential(contour_vals=np.linspace(0, 3, 25), aspect_ratio=0.12)

    def test_plot_matrixelements(self):
        TESTNAME = 'zeropi_9'
        self.plot_matrixelements('n_theta_operator', evals_count=10)

    def test_print_matrixelements(self):
        TESTNAME = 'zeropi_10'
        self.print_matrixelements('i_d_dphi_operator')

    def test_plot_matelem_vs_paramvals(self):
        TESTNAME = 'zeropi_11'
        flux_list = np.linspace(0, 0.5, 15)
        self.plot_matelem_vs_paramvals('n_theta_operator', 'flux', flux_list, select_elems=[(0, 0), (1, 4), (1, 0)])
