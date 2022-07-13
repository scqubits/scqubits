# test_fullzeropi.py
# meant to be run with 'pytest'
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np
import pytest

from scqubits import FullZeroPi
from scqubits.core.storage import SpectrumData
from scqubits.tests.conftest import DATADIR, BaseTest


@pytest.mark.usefixtures("io_type")
class TestFullZeroPi(BaseTest):
    @classmethod
    def setup_class(cls):
        cls.qbt = FullZeroPi.create()
        cls.qbt_type = FullZeroPi
        cls.file_str = "fullzeropi"

    def test_hamiltonian_is_hermitian(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        hamiltonian = self.qbt.hamiltonian()
        assert np.isclose(np.max(np.abs(hamiltonian - hamiltonian.conj().T)), 0.0)

    def test_eigenvals(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        evals_reference = specdata.energy_table
        return self.eigenvals(io_type, evals_reference)

    def test_eigenvecs(self, io_type):
        testname = self.file_str + "_2." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        evecs_reference = specdata.state_table
        self.qbt = self.qbt_type(**specdata.system_params)
        evals_count = evecs_reference.shape[1]
        _, evecs_tst = self.qbt.eigensys(
            evals_count=evals_count, filename=self.tmpdir + "test." + io_type
        )
        assert np.allclose(np.abs(evecs_reference), np.abs(evecs_tst))
