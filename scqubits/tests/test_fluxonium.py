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

from scqubits import Fluxonium, FileType
import scqubits.settings
from scqubits.core.data_containers import SpectrumData
from scqubits.tests.conftest import BaseTest, DATADIR

scqubits.settings.file_format = FileType.h5


class TestFluxonium(BaseTest):
    qbt = Fluxonium(EJ=None, EC=None, EL=None, flux=None, cutoff=None)
    # dummy values, will read  actual values from external user h5 files

    def test_eigenvals(self):
        testname = 'fluxonium_1'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        evals_reference = specdata.energy_table
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        testname = 'fluxonium_2'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        evecs_reference = specdata.state_table
        return self.eigenvecs(evecs_reference)

    def test_plot_evals_vs_paramvals(self):
        # testname = 'fluxonium_3'
        flux_list = np.linspace(0.45, 0.55, 50)
        return self.plot_evals_vs_paramvals('flux', flux_list)

    def test_get_spectrum_vs_paramvals(self):
        testname = 'fluxonium_4'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        flux_list = specdata.param_vals
        evecs_reference = specdata.state_table
        evals_reference = specdata.energy_table
        return self.get_spectrum_vs_paramvals('flux', flux_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        testname = 'fluxonium_5'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        matelem_reference = specdata.matrixelem_table
        return self.matrixelement_table('n_operator', matelem_reference)

    def test_plot_evals_vs_paramvals_EJ(self):
        # testname = 'fluxonium_6'
        ej_vals = self.qbt.EJ * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ', ej_vals)

    # testname = 'fluxonium_7'

    def test_plot_wavefunction(self):
        # testname = 'fluxonium_8'
        self.qbt.plot_wavefunction(esys=None, which=5, mode='real')

    def test_plot_matrixelements(self):
        # testname = 'fluxonium_9'
        self.plot_matrixelements('n_operator', evals_count=10)

    def test_print_matrixelements(self):
        # testname = 'fluxonium_10'
        self.print_matrixelements('phi_operator')

    def test_plot_matelem_vs_paramvals(self):
        # testname = 'fluxonium_11'
        flux_list = np.linspace(0.45, 0.55, 50)
        self.plot_matelem_vs_paramvals('n_operator', 'flux', flux_list, select_elems=[(0, 0), (1, 4), (1, 0)])
