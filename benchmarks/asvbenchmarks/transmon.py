import numpy as np
from scqubits import Transmon
import time

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
print("Current: ", current)
parent = os.path.dirname(current)
print("Parent: ", parent)
sys.path.append(parent)

import asvconftest
from asvconftest import StandardTests


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

    def test_plot_n_wavefunction(self):
        self.qbt = Transmon(EJ=1.0, EC=1.0, ng=0.0, ncut=10)
        self.qbt.plot_n_wavefunction(esys=None, which=1, mode="real")
