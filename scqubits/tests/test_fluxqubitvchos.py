import numpy as np

from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.flux_qubit_vchos import FluxQubitVCHOS, FluxQubitVCHOSGlobal
from scqubits.tests.vchostest import VCHOSTestFunctions


class TestFluxQubitVCHOS(VCHOSTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVCHOS
        cls.file_str = 'fluxqubitvchos'
        cls.op1_str = ''
        cls.op2_str = ''
        cls.param_name = 'flux'
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = FluxQubit
        cls.compare_file_str = 'fluxqubit'


# class TestFluxQubitVCHOSGlobal(VCHOSTestFunctions):
#     @classmethod
#     def setup_class(cls):
#         cls.qbt = None
#         cls.qbt_type = FluxQubitVCHOSGlobal
#         cls.file_str = 'fluxqubitvchosglobal'
#         cls.op1_str = ''
#         cls.op2_str = ''
#         cls.param_name = 'flux'
#         cls.param_list = np.linspace(0.46, 0.54, 21)
#         cls.compare_qbt_type = FluxQubit
#         cls.compare_file_str = 'fluxqubit'
#
#     def initialize_vchos_qbt(self, system_params):
#         return self.qbt_type(**system_params, maximum_periodic_vector_length=8, global_exc=4)
