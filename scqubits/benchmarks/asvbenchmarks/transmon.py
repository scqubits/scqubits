import numpy as np
from scqubits import Transmon

import os
import sys

# from scqubits.benchmarks.asvconftest import StandardTests
import scqubits.benchmarks.asvconftest as asvtest


class TestTransmon(asvtest.StandardTests):
    def setup(self):
        self.qbt = None
        self.qbt_type = Transmon
        self.file_str = "transmon"
        self.op1_str = "n_operator"
        self.op2_str = "n_operator"
        self.param_name = "ng"
        self.param_list = np.linspace(-1, 1, 100)

    def time_plot_n_wavefunction(self):
        self.qbt = Transmon(EJ=1.0, EC=1.0, ng=0.0, ncut=10)
        self.qbt.plot_n_wavefunction(esys=None, which=1, mode="real")
