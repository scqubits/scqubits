import numpy as np
from scqubits import Transmon
from scqubits import Fluxonium

# from scqubits.benchmarks.asvconftest import StandardTests
# import scqubits.benchmarks.asvconftest as asvtest


class TestQubit:

    def time_transmon_get_spectrum_vs_paramvals(self):
        CPB = Fluxonium.create()
        ng_list = np.linspace(-2, 2, 100)
        return CPB.get_spectrum_vs_paramvals('ng', ng_list, evals_count=4, subtract_ground=False, get_eigenstates=True)
    
    def time_fluxonium_get_spectrum_vs_paramvals(self):
        qbt = Fluxonium.create()
        # flux_list = np.linspace(-0.5, 0.5, 51)
        # return qbt.get_spectrum_vs_paramvals('flux', flux_list, evals_count=4, subtract_ground=False, get_eigenstates=True)