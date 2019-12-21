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
from scqubits.tests.conftest import BaseTest, DATADIR


class TestFullZeroPi(BaseTest):

    phi_grid = qubit.Grid1d(1, 2, 3)
    qbt = FullZeroPi(zeropi_cutoff=None, zeta_cutoff=None, grid=phi_grid, ncut=None, EJ=None, dEJ=None,
                     EL=None, dEL=None, ECJ=None, dCJ=None, EC=0.001, ECS=None, dC=None, ng=None, flux=None)
    # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        TESTNAME = 'fullzeropi_1'
        h5params, datalist = self.read_h5py(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        return self.eigenvals(evals_reference)
