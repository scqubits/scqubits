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
from scqubits.core.spectrum import SpectrumData
from scqubits.tests.conftest import BaseTest, DATADIR


class TestFullZeroPi(BaseTest):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = qubit.FullZeroPi
        cls.file_str = 'fullzeropi'

    def test_eigenvals(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        evals_reference = specdata.energy_table
        return self.eigenvals(evals_reference)
