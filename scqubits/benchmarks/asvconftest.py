# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import scqubits as scq
import scqubits.settings
import scqubits.utils.plotting as plot

from scqubits.core.storage import SpectrumData
from scqubits.settings import IN_IPYTHON

TESTDIR, _ = os.path.split(scqubits.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")  # local scqubits directory holding tests
DATADIR = os.path.join(TESTDIR, "data", "")  # local data collection within scqubits


# adding comment for difference

class BaseTest:
    """Used as base class for the pytests of qubit classes"""

    qbt = None  # class instance of qubit to be tested
    qbt_type = None

    def set_tmpdir(self, request):
        """Pytest fixture that provides a temporary directory for writing test files"""
        setattr(self, "tmpdir", request.getfixturevalue("tmpdir"))

    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def plot_evals_vs_paramvals(self, num_cpus, param_name, param_list):
        self.qbt.plot_evals_vs_paramvals(
            param_name,
            param_list,
            evals_count=5,
            subtract_ground=True,
            filename=self.tmpdir + "test",
            num_cpus=num_cpus,
        )

    def get_spectrum_vs_paramvals(
        self,
        num_cpus,
        io_type,
        param_name,
        param_list,
        evals_reference,
        evecs_reference,
    ):
        evals_count = len(evals_reference[0])
        calculated_spectrum = self.qbt.get_spectrum_vs_paramvals(
            param_name,
            param_list,
            evals_count=evals_count,
            subtract_ground=False,
            get_eigenstates=True,
            num_cpus=num_cpus,
        )
        calculated_spectrum.filewrite(filename=self.tmpdir + "test." + io_type)

        assert np.allclose(evals_reference, calculated_spectrum.energy_table)
        assert np.allclose(
            np.abs(evecs_reference), np.abs(calculated_spectrum.state_table), atol=1e-07
        )

    def matrixelement_table(self, io_type, op, matelem_reference):
        evals_count = len(matelem_reference)
        calculated_matrix = self.qbt.matrixelement_table(
            op,
            evecs=None,
            evals_count=evals_count,
            filename=self.tmpdir + "test." + io_type,
        )
        assert np.allclose(np.abs(matelem_reference), np.abs(calculated_matrix))

    def plot_matrixelements(self, op, evals_count=7):
        self.qbt.plot_matrixelements(
            op, evecs=None, evals_count=evals_count, show_numbers=True
        )

    def print_matrixelements(self, op):
        mat_data = self.qbt.matrixelement_table(op)
        plot.matrix2d(abs(mat_data))

    def plot_matelem_vs_paramvals(
        self, num_cpus, op, param_name, param_list, select_elems
    ):
        self.qbt.plot_matelem_vs_paramvals(
            op,
            param_name,
            param_list,
            select_elems=select_elems,
            filename=self.tmpdir + "test",
            num_cpus=num_cpus,
        )

    def time_file_io(self):
        self.qbt = self.qbt_type.create()
        # self.qbt.filewrite(self.tmpdir + "test.h5")
        # qbt_copy = scq.read(self.tmpdir + "test.h5")
        # assert self.qbt == qbt_copy


class StandardTests(BaseTest):
    def time_eigenvals(self):
        self.qbt = self.qbt_type.create()
        return self.qbt.eigenvals()

    def time_eigenvecs(self, io_type):
        self.qbt = self.qbt_type.create()
        return self.qbt.eigenvecs()

    """
    def time_plot_wavefunction(self, io_type):
        if "plot_wavefunction" not in dir(self.qbt_type):
            pytest.skip("This is expected, no reason for concern.")
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.qbt.plot_wavefunction(esys=None, which=5, mode="real")
        self.qbt.plot_wavefunction(esys=None, which=9, mode="abs_sqr")
    
    def time_plot_evals_vs_paramvals(self, num_cpus, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        return self.plot_evals_vs_paramvals(num_cpus, self.param_name, self.param_list)

    def time_matrixelement_table(self, io_type):
        testname = self.file_str + "_5." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        matelem_reference = specdata.matrixelem_table
        return self.matrixelement_table(io_type, self.op1_str, matelem_reference)

    def time_plot_matrixelements(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.plot_matrixelements(self.op1_str, evals_count=10)

    def time_print_matrixelements(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.print_matrixelements(self.op2_str)

    def time_plot_matelem_vs_paramvals(self, num_cpus, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.plot_matelem_vs_paramvals(
            num_cpus,
            self.op1_str,
            self.param_name,
            self.param_list,
            select_elems=[(0, 0), (1, 4), (1, 0)],
        )
    
    def time_plot_potential(self, io_type):
        if "plot_potential" not in dir(self.qbt_type):
            pytest.skip("This is expected, no reason for concern.")
        testname = self.file_str + "_1.hdf5"
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.qbt.plot_potential()
"""
