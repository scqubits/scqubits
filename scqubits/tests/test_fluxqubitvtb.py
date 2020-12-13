import numpy as np

from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.flux_qubit_vtb import FluxQubitVTB
from scqubits.tests.conftest import VTBTestFunctions


class TestFluxQubitVTB(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVTB
        cls.file_str = 'fluxqubitvtb'
        cls.op1_str = ''
        cls.op2_str = ''
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = FluxQubit
        cls.compare_file_str = 'fluxqubit'
