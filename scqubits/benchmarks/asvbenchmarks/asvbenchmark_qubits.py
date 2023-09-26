import numpy as np
import scqubits as scq
from scqubits import Transmon
from scqubits import TunableTransmon
from scqubits import Fluxonium
from scqubits import FluxQubit
from scqubits import Cos2PhiQubit
from scqubits import ZeroPi
from scqubits import FullZeroPi

# from scqubits.benchmarks.asvconftest import StandardTests
# import scqubits.benchmarks.asvconftest as asvtest


class TestQubit:

    def time_transmon_get_spectrum_vs_paramvals(self):
        init_params = Transmon.default_params()
        CPB = Transmon(**init_params)
        ng_list = np.linspace(-2, 2, 100)
        return CPB.get_spectrum_vs_paramvals('ng', ng_list, evals_count=4, subtract_ground=False, get_eigenstates=True)
    
    def time_tunabletransmon_get_spectrum_vs_paramvals(self):
        init_params = TunableTransmon.default_params()
        qbt = TunableTransmon(**init_params)
        flux = np.linspace(0, 1, 100)
        return qbt.get_spectrum_vs_paramvals('flux', flux, evals_count=4, subtract_ground=False, get_eigenstates=True)
    
    def time_fluxonium_get_spectrum_vs_paramvals(self):
        init_params = Fluxonium.default_params()
        qbt = Fluxonium(**init_params)
        flux_list = np.linspace(-0.5, 0.5, 51)
        return qbt.get_spectrum_vs_paramvals('flux', flux_list, evals_count=4, subtract_ground=False, get_eigenstates=True)
    
    def time_fluxqubit_get_spectrum_vs_paramvals(self):
        init_params = FluxQubit.default_params()
        qbt = FluxQubit(**init_params)
        flux_list = np.linspace(0.45, 0.55, 50)
        qbt.get_spectrum_vs_paramvals('flux', flux_list, evals_count=4, subtract_ground=False, get_eigenstates=True)
    
    def time_cos2phi_get_spectrum_vs_paramvals(self):
        init_params = Cos2PhiQubit.default_params()
        qbt = Cos2PhiQubit(**init_params)
        flux_list = np.linspace(0.0, 0.5, 3)
        tst = qbt.get_spectrum_vs_paramvals('flux', flux_list, evals_count=3, subtract_ground=False, get_eigenstates=True)

    def time_ZeroPi_get_spectrum_vs_paramvals(self):
        phi_grid = scq.Grid1d(-19.0, 19.0, 200)
        init_params = ZeroPi.default_params()
        init_params["grid"] = phi_grid
        qbt = ZeroPi(**init_params)
        flux_list = np.linspace(0.0, 0.5, 3)
        tst = qbt.get_spectrum_vs_paramvals('flux', flux_list, evals_count=3, subtract_ground=False, get_eigenstates=True)
    
    def time_FullZeroPi_get_spectrum_vs_paramvals(self):
        phi_grid = scq.Grid1d(-19.0, 19.0, 200)
        init_params = FullZeroPi.default_params()
        init_params["grid"] = phi_grid
        qbt = FullZeroPi(**init_params)
        flux_list = np.linspace(0.0, 0.5, 3)
        tst = qbt.get_spectrum_vs_paramvals('flux', flux_list, evals_count=3, subtract_ground=False, get_eigenstates=True)
    