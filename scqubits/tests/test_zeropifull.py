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
    qbt = FullZeroPi(zeropi_cutoff=1,
                     zeta_cutoff=1,
                     grid=phi_grid,
                     ncut=1,
                     EJ=2,
                     dEJ=0.05,
                     EL=0.1,
                     dEL=0.01,
                     ECJ=1,
                     dCJ=0.05,
                     EC=0.001,
                     ECS=None,
                     dC=0.08,
                     ng=0.3,
                     flux=0.2)
    # dummy values, will read in actual values from h5 files

    def test_eigenvals(self):
        TESTNAME = 'fullzeropi_1'
        h5params, datalist = self.read_h5py(DATADIR + TESTNAME + '.hdf5')
        self.qbt.set_params_from_h5(h5params)
        evals_reference = datalist[0]
        print(evals_reference[1:-1])
        return self.eigenvals(evals_reference)
