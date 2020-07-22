import os

import numpy as np
import pytest

import scqubits.settings
from scqubits.core.flux_qubit_vchos import FluxQubitVCHOS
from scqubits.tests.conftest import StandardTests
from scqubits.core.storage import SpectrumData

TESTDIR, _ = os.path.split(scqubits.__file__)
TESTDIR = os.path.join(TESTDIR, 'tests', '')
DATADIR = os.path.join(TESTDIR, 'data', '')


class TestFluxQubitVCHOS(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVCHOS
        cls.file_str = 'fluxqubitvchos'
        cls.op1_str = ''
        cls.op2_str = ''
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.45, 0.55, 50)

    def test_compare_eigenvals_with_FluxQubit(self, num_compare=3):
        FQSpecdata = SpectrumData.create_from_file(DATADIR + 'fluxqubit_1.hdf5')
        evals_reference = FQSpecdata.energy_table[0:num_compare]
        system_params = FQSpecdata.system_params
        system_params.pop('ncut')
        self.qbt = FluxQubitVCHOS(**system_params, kmax=1, num_exc=5)
        self.eigenvals(io_type='hdf5', evals_reference=evals_reference)

    def test_compare_spectrum_vs_paramvals_with_FluxQubit(self, num_compare=3):
        FQSpecdata = SpectrumData.create_from_file(DATADIR + 'fluxqubit_4.hdf5')
        evals_reference = FQSpecdata.energy_table[:, 0:num_compare]
        system_params = FQSpecdata.system_params
        system_params.pop('ncut')
        self.qbt = FluxQubitVCHOS(**system_params, kmax=1, num_exc=5)
        self.compare_spectrum_vs_paramvals(num_cpus=1, io_type='hdf5', param_name=FQSpecdata.param_name,
                                       param_list=FQSpecdata.param_vals, evals_reference=evals_reference,
                                       evecs_reference=None)

    def compare_spectrum_vs_paramvals(self, num_cpus, io_type, param_name, param_list, evals_reference, evecs_reference):
        evals_count = len(evals_reference[0])
        calculated_spectrum = self.qbt.get_spectrum_vs_paramvals(param_name, param_list, evals_count=evals_count,
                                                                 subtract_ground=False, get_eigenstates=True,
                                                                 num_cpus=num_cpus)
        calculated_spectrum.filewrite(filename=self.tmpdir + 'test.' + io_type)

        assert np.allclose(evals_reference, calculated_spectrum.energy_table, atol=1e-3)

    def test_matrixelement_table(self, io_type):
        pytest.skip('not implemented yet for vchos')

    def test_plot_matrixelements(self, io_type):
        pytest.skip('not implemented yet for vchos')

    def test_print_matrixelements(self, io_type):
        pytest.skip('not implemented yet for vchos')

    def test_plot_matelem_vs_paramvals(self, num_cpus, io_type):
        pytest.skip('not implemented yet for vchos')
