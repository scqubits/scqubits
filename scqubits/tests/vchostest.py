import numpy as np
import pytest

from scqubits.tests.conftest import StandardTests, DATADIR
from scqubits.core.storage import SpectrumData


class VCHOSTestFunctions(StandardTests):
    def test_transfer_matrix_comparison(self, io_type):
        testname = self.file_str + '_1.' + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        transfer_matrix = self.qbt.transfer_matrix()
        reference_transfer_matrix = specdata.transfer
        assert np.allclose(np.abs(reference_transfer_matrix), np.abs(transfer_matrix))

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
        self.qbt = self.initialize_vchos_qbt(system_params)
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
        self.qbt = self.initialize_vchos_qbt(system_params)
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

    def initialize_vchos_qbt(self, system_params):
        return self.qbt_type(**system_params, maximum_periodic_vector_length=8, num_exc=4)

    def test_matrixelement_table(self, io_type):
        pytest.skip('not implemented yet for vchos')

    def test_plot_matrixelements(self, io_type):
        pytest.skip('not implemented yet for vchos')

    def test_print_matrixelements(self, io_type):
        pytest.skip('not implemented yet for vchos')

    def test_plot_matelem_vs_paramvals(self, num_cpus, io_type):
        pytest.skip('not implemented yet for vchos')
