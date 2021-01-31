import numpy as np
import pytest

from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.flux_qubit_vtb import FluxQubitVTB, FluxQubitVTBSqueezing
from scqubits.tests.conftest import VTBTestFunctions


class TestFluxQubitVTB(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVTB
        cls.file_str = 'fluxqubitvtb'
        cls.op1_str = 'n_operator'
        cls.op1_arg = {'j': 0}
        cls.op2_str = 'exp_i_phi_operator'
        cls.op2_arg = {'j': 0}
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = FluxQubit
        cls.compare_file_str = 'fluxqubit'


class TestFluxQubitVTBSqueezing(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVTBSqueezing
        cls.file_str = 'fluxqubitvtbsqueezing'
#        cls.op1_str = 'n_operator'
#        cls.op1_arg = {'j': 0}
#        cls.op2_str = 'exp_i_phi_operator'
#        cls.op2_arg = {'j': 0}
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = FluxQubit
        cls.compare_file_str = 'fluxqubit'

    def test_harmonic_length_optimization_gradient(self, io_type):
        pytest.skip('not implemented for squeezing')

    def test_print_matrixelements(self, io_type):
        pytest.skip('not implemented for squeezing')

    def test_plot_matrixelements(self, io_type):
        pytest.skip('not implemented for squeezing')

    def test_matrixelement_table(self, io_type):
        pytest.skip('not implemented for squeezing')
