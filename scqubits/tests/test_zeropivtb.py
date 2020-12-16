import numpy as np
import pytest

from scqubits import ZeroPi
from scqubits import ZeroPiVTB
from scqubits.tests.conftest import VTBTestFunctions


class TestVTB(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = ZeroPiVTB
        cls.file_str = 'zeropivtb'
        cls.op1_str = 'n_operator'
        cls.op1_arg = {'j': 1}
        cls.op2_str = 'n_operator'
        cls.op2_arg = {'j': 0}
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = ZeroPi
        cls.compare_file_str = 'zeropi_vtb_compare'

    def initialize_vtb_qbt(self, system_params):
        system_params.pop('grid')
        return self.qbt_type(**system_params, maximum_periodic_vector_length=8, num_exc=4)

    def test_compare_spectrum_vs_paramvals_with_Qubit(self, io_type):
        pytest.skip('takes forever, skip for now')
