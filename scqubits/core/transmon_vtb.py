import os

import numpy as np
import scipy as sp
import itertools

import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.core.qubit_base as base
from scqubits import VariationalTightBinding


# -Transmon using VariationalTightBinding

class TransmonVTB(VariationalTightBinding, base.QubitBaseClass, serializers.Serializable):
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE', attr_name='EJlist', attr_location=1)
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE', attr_name='nglist', attr_location=1)
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, ng, truncated_dim=None, **kwargs):
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.truncated_dim = truncated_dim
        VariationalTightBinding.__init__(self, np.array([EJ]), np.array([ng]), 0.0,
                                         number_degrees_freedom=1, number_periodic_degrees_freedom=1, **kwargs)

    @staticmethod
    def default_params():
        return {
            'EJ': 15.0,
            'EC': 0.3,
            'ng': 0.0,
            'ncut': 30,
            'truncated_dim': 10
        }

    def potential(self, phi):
        return -self.EJ * np.cos(phi[0])

    def build_EC_matrix(self):
        return self.EC

    def build_capacitance_matrix(self):
        return self.e**2 / (2 * self.EC)

    def find_minima(self):
        return np.array([0.0])