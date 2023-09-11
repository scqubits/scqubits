import numpy as np
from scqubits import Transmon

import os
import sys

from scqubits.benchmarks.asvconftest import StandardTests


class TestTransmon(StandardTests):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = Transmon
        cls.file_str = "transmon"
        cls.op1_str = "n_operator"
        cls.op2_str = "n_operator"
        cls.param_name = "ng"
        cls.param_list = np.linspace(-1, 1, 100)

    def time_plot_n_wavefunction(self):
        self.qbt = Transmon(EJ=1.0, EC=1.0, ng=0.0, ncut=10)
        self.qbt.plot_n_wavefunction(esys=None, which=1, mode="real")
