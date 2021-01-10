from typing import Any, Dict, Tuple

import numpy as np
from numpy import ndarray

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits import Transmon, VTBBaseMethods


# -Transmon using VariationalTightBinding

class TransmonVTB(VTBBaseMethods, Transmon, base.QubitBaseClass, serializers.Serializable):
    def __init__(self,
                 EJ: float,
                 EC: float,
                 ng: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs
                 ) -> None:
        VTBBaseMethods.__init__(self, num_exc, maximum_periodic_vector_length,
                                number_degrees_freedom=1, number_periodic_degrees_freedom=1,
                                number_junctions=1, **kwargs)
        Transmon.__init__(self, EJ, EC, ng, 0, truncated_dim)
        delattr(self, 'ncut')

    def set_EJlist(self, EJlist) -> None:
        self.EJ = EJlist[0]
        self.__dict__['EJlist'] = EJlist

    def get_EJlist(self) -> ndarray:
        return np.array([self.EJ])

    EJlist = property(get_EJlist, set_EJlist)

    def set_nglist(self, nglist) -> None:
        self.ng = nglist[0]
        self.__dict__['nglist'] = nglist

    def get_nglist(self) -> ndarray:
        return np.array([self.ng])

    nglist = property(get_nglist, set_nglist)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            'EJ': 15.0,
            'EC': 0.3,
            'ng': 0.0,
            'truncated_dim': 10
        }

    def gamma_matrix(self, minimum: int = 0) -> ndarray:
        gamma_matrix = np.array([[0.0]])
        min_loc = self.sorted_minima()[minimum]
        e_charge = 1.0
        Phi0 = 1. / (2 * e_charge)
        gamma_list = self.EJlist / Phi0 ** 2
        gamma_matrix[0, 0] = gamma_list[0] * np.cos(min_loc[0])
        return gamma_matrix

    def _local_potential(self, exp_i_phi_j: ndarray, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                         Xi: ndarray, phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_j = exp_i_phi_j * np.exp(1j * phi_bar[0])
        potential_matrix = -0.5 * self.EJlist[0] * (exp_i_phi_j + exp_i_phi_j.conjugate())
        potential_matrix += self.EJlist[0] * self.identity()
        return potential_matrix

    def EC_matrix(self) -> ndarray:
        return np.array([[self.EC]])

    def capacitance_matrix(self) -> ndarray:
        return np.array([[1.0 / (2 * self.EC)]])

    def sorted_minima(self) -> ndarray:
        return np.array([[0.0]])
