import numpy as np
import pytest

from scqubits.core.current_mirror import CurrentMirror
from scqubits.core.current_mirror_vchos import CurrentMirrorVCHOS, CurrentMirrorVCHOSGlobal
from scqubits.tests.vchostest import VCHOSTestFunctions


class TestCurrentMirrorVCHOS(VCHOSTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = CurrentMirrorVCHOS
        cls.file_str = 'currentmirrorvchos'
        cls.op1_str = ''
        cls.op2_str = ''
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.4, 0.6, 21)
        cls.compare_qbt_type = CurrentMirror
        cls.compare_file_str = 'currentmirror'

    def test_plot_wavefunction(self, io_type):
        pytest.skip('not relevant for current mirror')


class TestCurrentMirrorVCHOSGlobal(VCHOSTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = CurrentMirrorVCHOSGlobal
        cls.file_str = 'currentmirrorvchosglobal'
        cls.op1_str = ''
        cls.op2_str = ''
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.4, 0.6, 21)
        cls.compare_qbt_type = CurrentMirror
        cls.compare_file_str = 'currentmirror'

    def initialize_vchos_qbt(self, system_params):
        return self.qbt_type(**system_params, kmax=1, global_exc=5)
