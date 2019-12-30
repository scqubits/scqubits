# test_zeropifull.py
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

import scqubits as qubit
from scqubits import FullZeroPi
from scqubits.core.data_containers import SpectrumData
from scqubits.tests.conftest import BaseTest, DATADIR


class TestFullZeroPi(BaseTest):

    phi_grid = qubit.Grid1d(1, 2, 3)
    qbt = FullZeroPi(zeropi_cutoff=None, zeta_cutoff=None, grid=phi_grid, ncut=None, EJ=None, dEJ=None,
                     EL=None, dEL=None, ECJ=1, dCJ=None, EC=0.001, ECS=None, dC=None, ng=None, flux=None)
    # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        testname = 'fullzeropi_1'
        specdata = SpectrumData(param_name=None, param_vals=None, energy_table=None, system_params=None)
        specdata.fileread(DATADIR + testname)
        self.qbt.set_params_from_dict(specdata._get_metadata_dict())
        evals_reference = specdata.energy_table
        return self.eigenvals(evals_reference)
