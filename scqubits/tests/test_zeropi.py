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
import scqubits.settings
from scqubits import ZeroPi, FileType
from scqubits.core.data_containers import SpectrumData
from scqubits.tests.conftest import BaseTest, DATADIR

scqubits.settings.file_format = FileType.h5


class TestZeroPi(BaseTest):
    grid = qubit.Grid1d(1, 2, 3)
    qbt = ZeroPi(grid=grid, EJ=None, EL=None, ECJ=1, EC=None, ECS=2, ng=None, flux=None, ncut=None)

    # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        testname = 'zeropi_1'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        evals_reference = specdata.energy_table
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        testname = 'zeropi_2'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        evecs_reference = specdata.state_table
        return self.eigenvecs(evecs_reference)

    def test_plot_evals_vs_paramvals(self):
        # testname = 'zeropi_3'
        flux_list = np.linspace(0, 0.5, 15)
        return self.plot_evals_vs_paramvals('flux', flux_list)

    def test_get_spectrum_vs_paramvals(self):
        testname = 'zeropi_4'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        flux_list = specdata.param_vals
        evecs_reference = specdata.state_table
        evals_reference = specdata.energy_table
        return self.get_spectrum_vs_paramvals('flux', flux_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        testname = 'zeropi_5'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        matelem_reference = specdata.matrixelem_table
        return self.matrixelement_table('n_theta_operator', matelem_reference)

    #   testname = 'zeropi_6'

    #    testname = 'zeropi_7'

    def test_plot_wavefunction(self):
        # testname = 'zeropi_8'
        self.qbt.plot_wavefunction(esys=None, which=4, mode='real', zero_calibrate=True)
        self.qbt.plot_potential(contour_vals=np.linspace(0, 3, 25))

    def test_plot_matrixelements(self):
        # testname = 'zeropi_9'
        self.plot_matrixelements('n_theta_operator', evals_count=10)

    def test_print_matrixelements(self):
        # testname = 'zeropi_10'
        self.print_matrixelements('i_d_dphi_operator')

    def test_plot_matelem_vs_paramvals(self):
        # testname = 'zeropi_11'
        flux_list = np.linspace(0, 0.5, 15)
        self.plot_matelem_vs_paramvals('n_theta_operator', 'flux', flux_list, select_elems=[(0, 0), (1, 4), (1, 0)])
