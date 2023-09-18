import numpy as np
from scqubits import Transmon

# from scqubits.benchmarks.asvconftest import StandardTests
# import scqubits.benchmarks.asvconftest as asvtest


class TestTransmon:

    def time_get_spectrum_vs_paramvals(self):
        CPB = Transmon.create()
        ng_list = np.linspace(-2, 2, 50)
        return CPB.get_spectrum_vs_paramvals('ng', ng_list, evals_count=4, subtract_ground=False, get_eigenstates=True)