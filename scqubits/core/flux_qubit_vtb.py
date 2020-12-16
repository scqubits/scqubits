import os

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from typing import Callable, List, Dict, Any, Tuple

from scqubits.core.flux_qubit import FluxQubitFunctions
from scqubits.core.hashing import Hashing
from scqubits.core.variationaltightbinding import VariationalTightBinding
from scqubits.core.variationaltightbindingsqueezing import VariationalTightBindingSqueezing
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


class FluxQubitVTBFunctions(FluxQubitFunctions):
    """Helper class for defining functions for VTB relevant to the Flux Qubit"""
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
                 flux: float
                 ) -> None:
        FluxQubitFunctions.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux)
        # final term in potential is cos[(+1)\phi_1+(-1)\phi_2+2pi f]
        self.boundary_coefficients = np.array([+1, -1])

    def _ramp(self, k: int, minima_holder: List) -> Tuple[List, bool]:
        """Helper function for find_minima"""
        guess = np.array([1.15 * 2.0 * np.pi * k / 3.0, 2.0 * np.pi * k / 3.0])
        result = minimize(self.potential, guess)
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
        result = minimize(self.potential, guess)
        minima_holder.append(self._normalize_minimum_inside_pi_range(result.x))
        for k in range(1, 4):
            (minima_holder, new_minima_positive) = self._ramp(k, minima_holder)
            (minima_holder, new_minima_negative) = self._ramp(-k, minima_holder)
            if not (new_minima_positive and new_minima_negative):
                break
        return np.array(minima_holder)


class FluxQubitVTB(FluxQubitVTBFunctions, VariationalTightBinding, base.QubitBaseClass, serializers.Serializable):
    r""" Flux Qubit using VTB

    See class FluxQubit for documentation on the qubit itself.

    Initialize in the same way as for FluxQubit, however now `num_exc` and `maximum_periodic_vector_length`
    must be set. See VTB for explanation of other kwargs.
    """

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
        VariationalTightBinding.__init__(self, num_exc, maximum_periodic_vector_length,
                                         number_degrees_freedom=2, number_periodic_degrees_freedom=2, **kwargs)
        FluxQubitVTBFunctions.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux)
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/'+str(type(self).__name__)+'.png')

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

    @property
    def EJlist(self) -> ndarray:
        return np.array([self.EJ1, self.EJ2, self.EJ3])

    @property
    def nglist(self) -> ndarray:
        return np.array([self.ng1, self.ng2])


class FluxQubitVTBSqueezing(VariationalTightBindingSqueezing, FluxQubitVTB):
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
