# conftest.py  ---  for use with pytest
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
#######################################################################################################################


import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import scqubits
import scqubits.settings
import scqubits.utils.plotting as plot
from scqubits.core.constants import FileType
from scqubits.core.spectrum import SpectrumData
from scqubits.settings import IN_IPYTHON

if not IN_IPYTHON:
    matplotlib.use('Agg')

scqubits.settings.FILE_FORMAT = FileType.h5

TESTDIR, _ = os.path.split(scqubits.tests.__file__)
DATADIR = os.path.join(TESTDIR, 'data', '')


class BaseTest:
    """Used as base class for pytests of qubit classes"""
    qbt = None

    @pytest.fixture(autouse=True)
    def set_tmpdir(self, request):
        """Pytest fixture that provides a temporary directory for writing test files"""
        setattr(self, 'tmpdir', request.getfixturevalue('tmpdir'))

    @classmethod
    def teardown_class(cls):
        plt.close('all')

    def set_params(self, h5file_root):
        """Read and store parameters from open h5 file

         Parameters
         ----------
         h5file_root: h5py.Group
             handle to root group in open h5 file
         """
        h5params = h5file_root.attrs
        for paramname in h5params.keys():
            paramvalue = h5params[paramname]
            if isinstance(paramvalue, (int, float, np.number)):
                setattr(self.qbt, paramname, h5params[paramname])

    def eigenvals(self, evals_reference):
        evals_count = len(evals_reference)
        evals_tst = self.qbt.eigenvals(evals_count=evals_count, filename=self.tmpdir + 'test')
        assert np.allclose(evals_reference, evals_tst)

    def eigenvecs(self, evecs_reference):
        evals_count = evecs_reference.shape[1]
        _, evecs_tst = self.qbt.eigensys(evals_count=evals_count, filename=self.tmpdir + 'test')
        assert np.allclose(np.abs(evecs_reference), np.abs(evecs_tst))

    def plot_evals_vs_paramvals(self, param_name, param_list):
        self.qbt.plot_evals_vs_paramvals(param_name, param_list, evals_count=5, subtract_ground=True,
                                         filename=self.tmpdir + 'test')

    def get_spectrum_vs_paramvals(self, param_name, param_list, evals_reference, evecs_reference):
        evals_count = len(evals_reference[0])
        calculated_spectrum = self.qbt.get_spectrum_vs_paramvals(param_name, param_list, evals_count=evals_count,
                                                                 subtract_ground=False, get_eigenstates=True,
                                                                 filename=self.tmpdir + 'test')

        assert np.allclose(evals_reference, calculated_spectrum.energy_table)
        assert np.allclose(np.abs(evecs_reference), np.abs(calculated_spectrum.state_table), atol=1e-07)

    def matrixelement_table(self, op, matelem_reference):
        evals_count = len(matelem_reference)
        calculated_matrix = self.qbt.matrixelement_table(op, evecs=None, evals_count=evals_count,
                                                         filename=self.tmpdir + 'test')
        assert np.allclose(np.abs(matelem_reference), np.abs(calculated_matrix))

    def plot_matrixelements(self, op, evals_count=7):
        self.qbt.plot_matrixelements(op, evecs=None, evals_count=evals_count)

    def print_matrixelements(self, op):
        mat_data = self.qbt.matrixelement_table(op)
        plot.print_matrix(abs(mat_data))

    def plot_matelem_vs_paramvals(self, op, param_name, param_list, select_elems):
        self.qbt.plot_matelem_vs_paramvals(op, param_name, param_list, select_elems=select_elems,
                                           filename=self.tmpdir + 'test')


class StandardTests(BaseTest):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = None
        cls.file_str = ''
        cls.op1_str = ''
        cls.op2_str = ''
        cls.param_name = ''
        cls.param_list = None

    def test_eigenvals(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        evals_reference = specdata.energy_table
        return self.eigenvals(evals_reference)

    def test_eigenvecs(self):
        testname = self.file_str + '_2'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        evecs_reference = specdata.state_table
        return self.eigenvecs(evecs_reference)

    def test_plot_wavefunction(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        self.qbt.plot_wavefunction(esys=None, which=5, mode='real')
        self.qbt.plot_wavefunction(esys=None, which=9, mode='abs_sqr')

    def test_plot_evals_vs_paramvals(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        return self.plot_evals_vs_paramvals(self.param_name, self.param_list)

    def test_get_spectrum_vs_paramvals(self):
        testname = self.file_str + '_4'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        self.param_list = specdata.param_vals
        evecs_reference = specdata.state_table
        evals_reference = specdata.energy_table
        return self.get_spectrum_vs_paramvals(self.param_name, self.param_list, evals_reference, evecs_reference)

    def test_matrixelement_table(self):
        testname = self.file_str + '_5'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        matelem_reference = specdata.matrixelem_table
        return self.matrixelement_table(self.op1_str, matelem_reference)

    def test_plot_matrixelements(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        self.plot_matrixelements(self.op1_str, evals_count=10)

    def test_print_matrixelements(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        self.print_matrixelements(self.op2_str)

    def test_plot_matelem_vs_paramvals(self):
        testname = self.file_str + '_1'
        specdata = SpectrumData.create_from_fileread(DATADIR + testname)
        self.qbt = self.qbt_type.create_from_dict(specdata._get_metadata_dict())
        self.plot_matelem_vs_paramvals(self.op1_str, self.param_name, self.param_list,
                                       select_elems=[(0, 0), (1, 4), (1, 0)])
