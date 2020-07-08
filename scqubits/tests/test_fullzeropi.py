# test_fullzeropi.py
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


import pytest
import numpy as np

import scqubits as qubit
from scqubits.core.storage import SpectrumData
from scqubits.tests.conftest import BaseTest, DATADIR


@pytest.mark.usefixtures("io_type")
class TestFullZeroPi(BaseTest):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = qubit.FullZeroPi
        cls.file_str = 'fullzeropi'

    def test_hamiltonian_is_hermitean(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        hamiltonian = self.qbt.hamiltonian()
        assert np.isclose(np.max(np.abs(hamiltonian - hamiltonian.conj().T)), 0.0)

    def test_eigenvals(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        evals_reference = specdata.energy_table
        return self.eigenvals(io_type, evals_reference)
