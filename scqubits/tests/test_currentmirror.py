import numpy as np
import pytest

from scqubits import CurrentMirror
from scqubits.tests.conftest import StandardTests


class TestCurrentMirror(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = CurrentMirror
        cls.file_str = 'currentmirror'
        cls.op1_str = 'charge_number_operator'
        cls.op2_str = 'exp_i_phi_j_operator'
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.4, 0.6, 21)

    def test_plot_wavefunction(self, io_type):
        pytest.skip('not relevant for current mirror')
