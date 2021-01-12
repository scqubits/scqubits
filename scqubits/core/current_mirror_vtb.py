import math
from typing import Callable, Any, Dict

import numpy as np
from numpy import ndarray
from scipy.linalg import inv
from scipy.optimize import minimize

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits import CurrentMirror, VTBBaseMethods, VTBBaseMethodsSqueezing
from scqubits.core.hashing import Hashing


class CurrentMirrorVTB(VTBBaseMethods, CurrentMirror, base.QubitBaseClass, serializers.Serializable):
    r""" Current Mirror using VTB

    See class CurrentMirror for documentation on the qubit itself.

    Initialize in the same way as for CurrentMirror, however now `num_exc` and `maximum_periodic_vector_length`
    must be set. See VTB for explanation of other kwargs.
    """
    _check_if_new_minima: Callable
    _normalize_minimum_inside_pi_range: Callable

    def __init__(self,
                 N: int,
                 ECB: float,
                 ECJ: float,
                 ECg: float,
                 EJlist: ndarray,
                 nglist: ndarray,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs
                 ) -> None:
        VTBBaseMethods.__init__(self, num_exc, maximum_periodic_vector_length,
                                number_degrees_freedom=2 * N - 1,
                                number_periodic_degrees_freedom=2 * N - 1,
                                number_junctions=2 * N, **kwargs)
        CurrentMirror.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, 0, truncated_dim)
        self._EJlist = EJlist
        self._nglist = nglist
        self.stitching_coefficients = np.ones(2 * N - 1)
        delattr(self, 'ncut')

    def set_EJlist(self, EJlist) -> None:
        self.__dict__['EJlist'] = EJlist

    def get_EJlist(self) -> ndarray:
        return self._EJlist

    EJlist = property(get_EJlist, set_EJlist)

    def set_nglist(self, nglist) -> None:
        self.__dict__['nglist'] = nglist

    def get_nglist(self) -> ndarray:
        return self._nglist

    nglist = property(get_nglist, set_nglist)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            'N': 2,
            'ECB': 0.2,
            'ECJ': 20.0 / 2.7,
            'ECg': 20.0,
            'EJlist': np.array(4 * [18.95]),
            'nglist': np.array(3 * [0.0]),
            'flux': 0.0,
            'num_exc': 2,
            'maximum_periodic_vector_length': 8,
            'truncated_dim': 6
        }

    def convert_node_ng_to_junction_ng(self, node_nglist: ndarray) -> ndarray:
        """Convert offset charge from node variables to junction variables."""
        return inv(self._build_V_m()).T @ node_nglist.T

    def convert_junction_ng_to_node_ng(self, junction_nglist: ndarray) -> ndarray:
        """Convert offset charge from junction variables to node variables."""
        return self._build_V_m().T @ junction_nglist.T

    def find_minima(self) -> ndarray:
        """Find all minima in the potential energy landscape of the current mirror.

        Returns
        -------
        ndarray
            Location of all minima, unsorted
        """
        minima_holder = []
        N = self.N
        for m in range(int(math.ceil(N / 2)) + 1):
            guess_pos = np.array([np.pi * (m - self.flux) / N for _ in range(self.number_degrees_freedom)])
            guess_neg = np.array([np.pi * (-m - self.flux) / N for _ in range(self.number_degrees_freedom)])
            result_pos = minimize(self.vtb_potential, guess_pos, options={'disp': False})
            result_neg = minimize(self.vtb_potential, guess_neg, options={'disp': False})
            new_minimum_pos = (self._check_if_new_minima(result_pos.x, minima_holder)
                               and self._check_if_second_derivative_potential_positive(result_pos.x))
            if new_minimum_pos and result_pos.success:
                minima_holder.append(self._normalize_minimum_inside_pi_range(result_pos.x))
            new_minimum_neg = (self._check_if_new_minima(result_neg.x, minima_holder)
                               and self._check_if_second_derivative_potential_positive(result_neg.x))
            if new_minimum_neg and result_neg.success:
                minima_holder.append(self._normalize_minimum_inside_pi_range(result_neg.x))
        return np.array(minima_holder)

    def _check_if_second_derivative_potential_positive(self, phi_array: ndarray) -> bool:
        """Helper method for determining whether the location specified by `phi_array` is a minimum."""
        second_derivative = np.round(-(self.vtb_potential(phi_array) - np.sum(self.EJlist)), decimals=3)
        return second_derivative > 0.0

    def vtb_potential(self, phi_array: ndarray) -> float:
        """Helper method for converting potential method arguments"""
        return self.potential(phi_array)


class CurrentMirrorVTBSqueezing(VTBBaseMethodsSqueezing, CurrentMirrorVTB):
    def __init__(self,
                 N: int,
                 ECB: float,
                 ECJ: float,
                 ECg: float,
                 EJlist: ndarray,
                 nglist: ndarray,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs) -> None:
        CurrentMirrorVTB.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, num_exc,
                                  maximum_periodic_vector_length, truncated_dim, **kwargs)


class CurrentMirrorVTBGlobal(Hashing, CurrentMirrorVTB):
    def __init__(self,
                 N: int,
                 ECB: float,
                 ECJ: float,
                 ECg: float,
                 EJlist: ndarray,
                 nglist: ndarray,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs):
        Hashing.__init__(self)
        CurrentMirrorVTB.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, num_exc,
                                  maximum_periodic_vector_length, truncated_dim, **kwargs)


class CurrentMirrorVTBGlobalSqueezing(Hashing, CurrentMirrorVTBSqueezing):
    def __init__(self,
                 N: int,
                 ECB: float,
                 ECJ: float,
                 ECg: float,
                 EJlist: ndarray,
                 nglist: ndarray,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 truncated_dim: int = None,
                 **kwargs) -> None:
        Hashing.__init__(self)
        CurrentMirrorVTBSqueezing.__init__(self,  N, ECB, ECJ, ECg, EJlist, nglist, flux, num_exc,
                                           maximum_periodic_vector_length, truncated_dim, **kwargs)
