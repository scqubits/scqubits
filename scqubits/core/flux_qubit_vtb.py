from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.hashing import Hashing
from scqubits.core.noise import NOISE_PARAMS
from scqubits.core.vtbbasemethods import VTBBaseMethods
from scqubits.core.vtbsqueezingbasemethods import VTBBaseMethodsSqueezing


class FluxQubitVTB(VTBBaseMethods, FluxQubit, base.QubitBaseClass, serializers.Serializable):
    r""" Flux Qubit using VTB

    See class FluxQubit for documentation on the qubit itself.

    Initialize in the same way as for FluxQubit, however now `num_exc` and `maximum_periodic_vector_length`
    must be set. See VTB for explanation of other kwargs.
    """
    _check_if_new_minima: Callable
    _normalize_minimum_inside_pi_range: Callable

    def __init__(self,
                 EJ1: float,
                 EJ2: float,
                 EJ3: float,
                 ECJ1: float,
                 ECJ2: float,
                 ECJ3: float,
                 ECg1: float,
                 ECg2: float,
                 ng1: float,
                 ng2: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs
                 ) -> None:
        VTBBaseMethods.__init__(self, num_exc, maximum_periodic_vector_length,
                                number_degrees_freedom=2, number_periodic_degrees_freedom=2,
                                number_junctions=3, **kwargs)
        FluxQubit.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, 0, truncated_dim)
        self._stitching_coefficients = np.array([+1, -1])
        delattr(self, 'ncut')

    def EC_matrix(self):
        return super(VTBBaseMethods, self).EC_matrix()

    def capacitance_matrix(self):
        return super(VTBBaseMethods, self).capacitance_matrix()

    def _ramp(self, k: int, minima_holder: List) -> Tuple[List, bool]:
        """Helper function for find_minima"""
        guess = np.array([1.15 * 2.0 * np.pi * k / 3.0, 2.0 * np.pi * k / 3.0])
        result = minimize(self.vtb_potential, guess)
        new_minima = self._check_if_new_minima(result.x, minima_holder)
        if new_minima:
            minima_holder.append(self._normalize_minimum_inside_pi_range(result.x))
        return minima_holder, new_minima

    def find_minima(self) -> ndarray:
        """
        Index all minima in the variable space of phi1 and phi2
        """
        minima_holder = []
        if self.flux == 0.5:
            guess = np.array([0.15, 0.1])
        else:
            guess = np.array([0.0, 0.0])
        result = minimize(self.vtb_potential, guess)
        minima_holder.append(self._normalize_minimum_inside_pi_range(result.x))
        for k in range(1, 4):
            (minima_holder, new_minima_positive) = self._ramp(k, minima_holder)
            (minima_holder, new_minima_negative) = self._ramp(-k, minima_holder)
            if not (new_minima_positive and new_minima_negative):
                break
        return np.array(minima_holder)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            'EJ1': 1.0,
            'EJ2': 1.0,
            'EJ3': 0.8,
            'ECJ1': 1.0 / 10.0,
            'ECJ2': 1.0 / 10.0,
            'ECJ3': 1.0 / (10.0 * 0.8),
            'ECg1': 5.0,
            'ECg2': 5.0,
            'ng1': 0.0,
            'ng2': 0.0,
            'flux': 0.46,
            'num_exc': 3,
            'maximum_periodic_vector_length': 8,
            'truncated_dim': 6
        }

    def set_EJlist(self, EJlist) -> None:
        self.EJ1 = EJlist[0]
        self.EJ2 = EJlist[1]
        self.EJ3 = EJlist[2]
        self.__dict__['EJlist'] = EJlist

    def get_EJlist(self) -> ndarray:
        return np.array([self.EJ1, self.EJ2, self.EJ3])

    EJlist = property(get_EJlist, set_EJlist)

    def set_nglist(self, nglist) -> None:
        self.ng1 = nglist[0]
        self.ng2 = nglist[1]
        self.__dict__['nglist'] = nglist

    def get_nglist(self) -> ndarray:
        return np.array([self.ng1, self.ng2])

    nglist = property(get_nglist, set_nglist)

    def vtb_potential(self, phi_array):
        """Helper method for converting potential method arguments"""
        phi_1 = phi_array[0]
        phi_2 = phi_array[1]
        return self.potential(phi_1, phi_2)

    def n_1_operator(self) -> ndarray:
        return self.n_operator(0)

    def n_2_operator(self) -> ndarray:
        return self.n_operator(1)

    def exp_i_phi_1_operator(self) -> ndarray:
        return self.exp_i_phi_operator(0)

    def exp_i_phi_2_operator(self) -> ndarray:
        return self.exp_i_phi_operator(1)

    def d_hamiltonian_d_EJ1(self) -> ndarray:
        raise NotImplementedError("Not implemented yet for tight binding")

    def d_hamiltonian_d_EJ2(self) -> ndarray:
        raise NotImplementedError("Not implemented yet for tight binding")

    def d_hamiltonian_d_EJ3(self) -> ndarray:
        raise NotImplementedError("Not implemented yet for tight binding")

    def tphi_1_over_f_cc(self,
                         A_noise: float = NOISE_PARAMS['A_cc'],
                         i: int = 0,
                         j: int = 1,
                         esys: Tuple[ndarray, ndarray] = None,
                         get_rate: bool = False,
                         **kwargs
                         ) -> float:
        raise NotImplementedError("Not implemented yet for tight binding")

    def tphi_1_over_f_cc1(self,
                          A_noise: float = NOISE_PARAMS['A_cc'],
                          i: int = 0,
                          j: int = 1,
                          esys: Tuple[ndarray, ndarray] = None,
                          get_rate: bool = False,
                          **kwargs
                          ) -> float:
        raise NotImplementedError("Not implemented yet for tight binding")

    def tphi_1_over_f_cc2(self,
                          A_noise: float = NOISE_PARAMS['A_cc'],
                          i: int = 0,
                          j: int = 1,
                          esys: Tuple[ndarray, ndarray] = None,
                          get_rate: bool = False,
                          **kwargs
                          ) -> float:
        raise NotImplementedError("Not implemented yet for tight binding")

    def tphi_1_over_f_cc3(self,
                          A_noise: float = NOISE_PARAMS['A_cc'],
                          i: int = 0,
                          j: int = 1,
                          esys: Tuple[ndarray, ndarray] = None,
                          get_rate: bool = False,
                          **kwargs
                          ) -> float:
        raise NotImplementedError("Not implemented yet for tight binding")


class FluxQubitVTBSqueezing(VTBBaseMethodsSqueezing, FluxQubitVTB):
    def __init__(self,
                 EJ1: float,
                 EJ2: float,
                 EJ3: float,
                 ECJ1: float,
                 ECJ2: float,
                 ECJ3: float,
                 ECg1: float,
                 ECg2: float,
                 ng1: float,
                 ng2: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs
                 ) -> None:
        FluxQubitVTB.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, num_exc,
                              maximum_periodic_vector_length, truncated_dim, **kwargs)


class FluxQubitVTBGlobal(Hashing, FluxQubitVTB):
    def __init__(self,
                 EJ1: float,
                 EJ2: float,
                 EJ3: float,
                 ECJ1: float,
                 ECJ2: float,
                 ECJ3: float,
                 ECg1: float,
                 ECg2: float,
                 ng1: float,
                 ng2: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs
                 ) -> None:
        Hashing.__init__(self)
        FluxQubitVTB.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, num_exc,
                              maximum_periodic_vector_length, truncated_dim, **kwargs)


class FluxQubitVTBGlobalSqueezing(Hashing, FluxQubitVTBSqueezing):
    def __init__(self,
                 EJ1: float,
                 EJ2: float,
                 EJ3: float,
                 ECJ1: float,
                 ECJ2: float,
                 ECJ3: float,
                 ECg1: float,
                 ECg2: float,
                 ng1: float,
                 ng2: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs
                 ) -> None:
        Hashing.__init__(self)
        FluxQubitVTBSqueezing.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, num_exc,
                                       maximum_periodic_vector_length, truncated_dim, **kwargs)
