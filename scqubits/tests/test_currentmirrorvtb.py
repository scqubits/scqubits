import numpy as np
import pytest

from scqubits.core.current_mirror import CurrentMirror
from scqubits.core.current_mirror_vtb import CurrentMirrorVTB
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
