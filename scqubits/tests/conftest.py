# conftest.py  ---  for use with pytest
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
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

import scqubits.settings
import scqubits.utils.plotting as plot
from scqubits.core.storage import SpectrumData
from scqubits.settings import IN_IPYTHON

if not IN_IPYTHON:
    matplotlib.use('Agg')


TESTDIR, _ = os.path.split(scqubits.__file__)
TESTDIR = os.path.join(TESTDIR, 'tests', '')
DATADIR = os.path.join(TESTDIR, 'data', '')


def pytest_addoption(parser):
    parser.addoption("--num_cpus", action="store", default=1, help="number of cores to be used")
    parser.addoption("--io_type", action="store", default='hdf5', help="Serializable file type to be used")


@pytest.fixture(scope='session')
def num_cpus(pytestconfig):
    return int(pytestconfig.getoption("num_cpus"))


@pytest.fixture(scope='session')
def io_type(pytestconfig):
    return pytestconfig.getoption("io_type")


@pytest.mark.usefixtures("num_cpus", "io_type")
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

    def eigenvals(self, io_type, evals_reference):
        evals_count = len(evals_reference)
        evals_tst = self.qbt.eigenvals(evals_count=evals_count, filename=self.tmpdir + 'test.' + io_type)
        assert np.allclose(evals_reference, evals_tst)

    def eigenvecs(self, io_type, evecs_reference):
        evals_count = evecs_reference.shape[1]
        _, evecs_tst = self.qbt.eigensys(evals_count=evals_count, filename=self.tmpdir + 'test.' + io_type)
        assert np.allclose(np.abs(evecs_reference), np.abs(evecs_tst), atol=1e-06)

    def plot_evals_vs_paramvals(self, num_cpus, param_name, param_list):
        self.qbt.plot_evals_vs_paramvals(param_name, param_list, evals_count=5, subtract_ground=True,
                                         filename=self.tmpdir + 'test', num_cpus=num_cpus)

    def get_spectrum_vs_paramvals(self, num_cpus, io_type, param_name, param_list, evals_reference, evecs_reference):
        evals_count = len(evals_reference[0])
        calculated_spectrum = self.qbt.get_spectrum_vs_paramvals(param_name, param_list, evals_count=evals_count,
                                                                 subtract_ground=False, get_eigenstates=True,
                                                                 num_cpus=num_cpus)
        calculated_spectrum.filewrite(filename=self.tmpdir + 'test.' + io_type)
        assert np.allclose(evals_reference, calculated_spectrum.energy_table)
        assert np.allclose(np.abs(evecs_reference), np.abs(calculated_spectrum.state_table), atol=1e-06)

    def matrixelement_table(self, io_type, op, matelem_reference):
        evals_count = len(matelem_reference)
        calculated_matrix = self.qbt.matrixelement_table(op, evecs=None, evals_count=evals_count,
                                                         filename=self.tmpdir + 'test.' + io_type)
        assert np.allclose(np.abs(matelem_reference), np.abs(calculated_matrix))

    def plot_matrixelements(self, op, evals_count=7):
        self.qbt.plot_matrixelements(op, evecs=None, evals_count=evals_count)

    def print_matrixelements(self, op):
        mat_data = self.qbt.matrixelement_table(op)
        plot.print_matrix(abs(mat_data))

    def plot_matelem_vs_paramvals(self, num_cpus, op, param_name, param_list, select_elems):
        self.qbt.plot_matelem_vs_paramvals(op, param_name, param_list, select_elems=select_elems,
                                           filename=self.tmpdir + 'test', num_cpus=num_cpus)


@pytest.mark.usefixtures("num_cpus", "io_type")
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
        cls.compare_qbt_type = None
        cls.compare_file_str = ''

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

    def test_eigenvecs(self, io_type):
        testname = self.file_str + '_2.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        evecs_reference = specdata.state_table
        return self.eigenvecs(io_type, evecs_reference)

    def test_plot_wavefunction(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.qbt.plot_wavefunction(esys=None, which=5, mode='real')
        self.qbt.plot_wavefunction(esys=None, which=9, mode='abs_sqr')

    def test_plot_evals_vs_paramvals(self, num_cpus, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        return self.plot_evals_vs_paramvals(num_cpus, self.param_name, self.param_list)

    def test_get_spectrum_vs_paramvals(self, num_cpus, io_type):
        testname = self.file_str + '_4.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.param_list = specdata.param_vals
        evecs_reference = specdata.state_table
        evals_reference = specdata.energy_table
        return self.get_spectrum_vs_paramvals(num_cpus, io_type, self.param_name, self.param_list, evals_reference,
                                              evecs_reference)

    def test_matrixelement_table(self, io_type):
        testname = self.file_str + '_5.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        matelem_reference = specdata.matrixelem_table
        return self.matrixelement_table(io_type, self.op1_str, matelem_reference)

    def test_plot_matrixelements(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.plot_matrixelements(self.op1_str, evals_count=10)

    def test_print_matrixelements(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.print_matrixelements(self.op2_str)

    def test_plot_matelem_vs_paramvals(self, num_cpus, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.plot_matelem_vs_paramvals(num_cpus, self.op1_str, self.param_name, self.param_list,
                                       select_elems=[(0, 0), (1, 4), (1, 0)])

    def test_plot_potential(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        if 'plot_potential' not in dir(self.qbt):
            pytest.skip('This is expected, no reason for concern.')
        self.qbt.plot_potential()


class VTBTestFunctions(StandardTests):
    def test_gamma_matrix(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        gamma_matrix = self.qbt.build_gamma_matrix()
        reference_gamma_matrix = specdata.gamma_matrix
        assert np.allclose(reference_gamma_matrix, gamma_matrix)

    def test_eigensystem_normal_modes(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        omega_squared, normal_mode_eigenvectors = self.qbt.eigensystem_normal_modes()
        ref_omega_squared, ref_normal_mode_eigenvectors = specdata.eigensystem_normal_modes
        assert np.allclose(ref_omega_squared, omega_squared)
        assert np.allclose(np.abs(ref_normal_mode_eigenvectors), np.abs(normal_mode_eigenvectors))

    def test_Xi_matrix(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        Xi_matrix = self.qbt.Xi_matrix()
        reference_Xi_matrix = specdata.Xi_matrix
        assert np.allclose(reference_Xi_matrix, Xi_matrix)

    def test_relevant_periodic_continuation_vectors(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.qbt.find_relevant_periodic_continuation_vectors()
        reference_nearest_neighbors = specdata.nearest_neighbors
        for key in reference_nearest_neighbors:
            assert np.allclose(reference_nearest_neighbors[key], self.qbt.nearest_neighbors[key])

    def test_kinetic_matrix(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        kinetic_matrix = self.qbt.kinetic_matrix()
        reference_kinetic_matrix = specdata.kinetic_matrix
        assert np.allclose(np.abs(reference_kinetic_matrix), np.abs(kinetic_matrix), atol=1e-6)

    def test_potential_matrix(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        potential_matrix = self.qbt.potential_matrix()
        reference_potential_matrix = specdata.potential_matrix
        assert np.allclose(np.abs(reference_potential_matrix), np.abs(potential_matrix), atol=1e-6)

    def test_inner_product_matrix(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        inner_product_matrix = self.qbt.inner_product_matrix()
        reference_inner_product_matrix = specdata.inner_product_matrix
        assert np.allclose(np.abs(reference_inner_product_matrix), np.abs(inner_product_matrix), atol=1e-6)

    def test_transfer_matrix_comparison(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        transfer_matrix = self.qbt.transfer_matrix()
        reference_transfer_matrix = specdata.transfer_matrix
        assert np.allclose(np.abs(reference_transfer_matrix), np.abs(transfer_matrix), atol=1e-6)

    def test_sorted_minima(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        sorted_minima = self.qbt.sorted_minima()
        reference_sorted_minima = specdata.sorted_minima
        assert np.allclose(reference_sorted_minima, sorted_minima)

    def test_compare_eigenvals_with_Qubit(self, io_type):
        num_compare = 3
        compare_name = self.compare_file_str + '_1.' + io_type
        exact_specdata = SpectrumData.create_from_file(DATADIR + compare_name)
        evals_reference = exact_specdata.energy_table[0:num_compare]
        system_params = exact_specdata.system_params
        system_params.pop('ncut')
        self.qbt = self.initialize_vtb_qbt(system_params)
        evals_count = len(evals_reference)
        evals_tst = self.qbt.eigenvals(evals_count=evals_count, filename=self.tmpdir + 'test.' + io_type)
        assert np.allclose(evals_reference, evals_tst, rtol=1e-2)

    def test_compare_spectrum_vs_paramvals_with_Qubit(self, io_type):
        num_compare = 3
        compare_name = self.compare_file_str + '_4.' + io_type
        exact_specdata = SpectrumData.create_from_file(DATADIR + compare_name)
        evals_reference = exact_specdata.energy_table[:, 0:num_compare]
        system_params = exact_specdata.system_params
        system_params.pop('ncut')
        self.qbt = self.initialize_vtb_qbt(system_params)
        self.compare_spectrum_vs_paramvals(io_type=io_type, param_name=exact_specdata.param_name,
                                           param_list=exact_specdata.param_vals, evals_reference=evals_reference)

    def compare_spectrum_vs_paramvals(self, io_type, param_name, param_list, evals_reference):
        evals_count = len(evals_reference[0])
        calculated_spectrum = self.qbt.get_spectrum_vs_paramvals(param_name, param_list, evals_count=evals_count,
                                                                 subtract_ground=False, get_eigenstates=False)
        calculated_spectrum.filewrite(filename=self.tmpdir + 'test.' + io_type)

        assert np.allclose(evals_reference, calculated_spectrum.energy_table, rtol=1e-2)

    def test_hamiltonian_is_hermitean(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        transfer_matrix = self.qbt.transfer_matrix()
        assert np.isclose(np.max(np.abs(transfer_matrix - transfer_matrix.conj().T)), 0.0)

    def initialize_vtb_qbt(self, system_params):
        return self.qbt_type(**system_params, maximum_periodic_vector_length=8, num_exc=4)

    def test_matrixelement_table(self, io_type):
        pytest.skip('not implemented yet for vtb')

    def test_plot_matrixelements(self, io_type):
        pytest.skip('not implemented yet for vtb')

    def test_print_matrixelements(self, io_type):
        pytest.skip('not implemented yet for vtb')

    def test_plot_matelem_vs_paramvals(self, num_cpus, io_type):
        pytest.skip('not implemented yet for vtb')
