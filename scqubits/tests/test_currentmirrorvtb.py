import numpy as np
import pytest

from scqubits.core.current_mirror import CurrentMirror
from scqubits.core.current_mirror_vtb import CurrentMirrorVTB, CurrentMirrorVTBSqueezing
from scqubits.tests.conftest import VTBTestFunctions


class TestCurrentMirrorVTB(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = CurrentMirrorVTB
        cls.file_str = 'currentmirrorvtb'
        cls.op1_str = 'n_operator'
        cls.op1_arg = {'j': 0}
        cls.op2_str = 'exp_i_phi_operator'
        cls.op2_arg = {'j': 0}
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = CurrentMirror
        cls.compare_file_str = 'currentmirror'

    def test_plot_wavefunction(self, io_type):
        pytest.skip('not relevant for current mirror')


class TestCurrentMirrorVTBSqueezing(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = CurrentMirrorVTBSqueezing
        cls.file_str = 'currentmirrorvtbsqueezing'
#        cls.op1_str = 'n_operator'
#        cls.op1_arg = {'j': 0}
#        cls.op2_str = 'exp_i_phi_operator'
#        cls.op2_arg = {'j': 0}
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = CurrentMirror
        cls.compare_file_str = 'currentmirror'

    def test_plot_wavefunction(self, io_type):
        pytest.skip('not relevant for current mirror')

    def test_harmonic_length_optimization_gradient(self, io_type):
        pytest.skip('not implemented for squeezing')

    def test_print_matrixelements(self, io_type):
        pytest.skip('not implemented for squeezing')

    def test_plot_matrixelements(self, io_type):
        pytest.skip('not implemented for squeezing')

    def test_matrixelement_table(self, io_type):
        pytest.skip('not implemented for squeezing')
