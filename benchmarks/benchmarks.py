import numpy as np

import scqubits as scq


class TransmonBenchmark:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.qubit = scq.Transmon(EJ=30.1, EC=0.5, ng=0.3, ncut=30)
        self.param = "ng"
        self.pts = 20
        self.param_vals = np.linspace(-0.5, 0.5, self.pts)

    def time_Transmon(self):
        self.qubit.get_spectrum_vs_paramvals(
            param_name=self.param, param_vals=self.param_vals
        )


class FluxoniumBenchmark:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.qubit = scq.Fluxonium(EJ=9.0, EC=2.5, EL=0.3, cutoff=80, flux=0.3)
        self.param = "flux"
        self.pts = 20
        self.param_vals = np.linspace(-0.5, 0.5, self.pts)

    def time_Fluxonium(self):
        self.qubit.get_spectrum_vs_paramvals(
            param_name=self.param, param_vals=self.param_vals
        )


#
#
# class MemSuite:
#     def mem_list(self):
#         return [0] * 256
