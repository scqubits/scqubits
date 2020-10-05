import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv
from typing import Callable

import scqubits.core.descriptors as descriptors
from scqubits.core.current_mirror import CurrentMirrorFunctions
from scqubits.core.hashing import Hashing
from scqubits.core.vchos import VCHOS
from scqubits.core.vchos_squeezing import VCHOSSqueezing
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


class CurrentMirrorVCHOSFunctions(CurrentMirrorFunctions):
    """Helper class for defining functions for VCHOS relevant to the Current Mirror"""
    _check_if_new_minima: Callable
    _normalize_minimum_inside_pi_range: Callable

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux):
        CurrentMirrorFunctions.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux)
        self.boundary_coefficients = np.ones(2 * N - 1)

    def convert_node_ng_to_junction_ng(self, node_nglist):
        """Convert offset charge from node variables to junction variables."""
        return inv(self._build_V_m()).T @ node_nglist.T

    def convert_junction_ng_to_node_ng(self, junction_nglist):
        """Convert offset charge from junction variables to node variables."""
        return self._build_V_m().T @ junction_nglist.T

    def find_minima(self):
        """Find all minima in the potential energy landscape of the current mirror.

        Returns
        -------
        ndarray
            Location of all minima, unsorted
        """
        minima_holder = []
        N = self.N
        for m in range(int(math.ceil(N / 2 - np.abs(self.flux))) + 1):
            guess_pos = np.array([np.pi * (m + self.flux) / N for _ in range(self.number_degrees_freedom)])
            guess_neg = np.array([np.pi * (-m + self.flux) / N for _ in range(self.number_degrees_freedom)])
            result_pos = minimize(self.potential, guess_pos)
            result_neg = minimize(self.potential, guess_neg)
            new_minimum_pos = (self._check_if_new_minima(result_pos.x, minima_holder)
                               and self._check_if_second_derivative_potential_positive(result_pos.x))
            if new_minimum_pos and result_pos.success:
                minima_holder.append(self._normalize_minimum_inside_pi_range(result_pos.x))
            new_minimum_neg = (self._check_if_new_minima(result_neg.x, minima_holder)
                               and self._check_if_second_derivative_potential_positive(result_neg.x))
            if new_minimum_neg and result_neg.success:
                minima_holder.append(self._normalize_minimum_inside_pi_range(result_neg.x))
        return minima_holder

    def _check_if_second_derivative_potential_positive(self, phi_array):
        """Helper method for determining whether the location specified by `phi_array` is a minimum."""
        second_derivative = np.round(-(self.potential(phi_array) - np.sum(self.EJlist)), decimals=3)
        return second_derivative > 0.0

    def potential(self, phi_array):
        """Potential evaluated at the location specified by phi_array.

        Parameters
        ----------
        phi_array: ndarray
            float value of the phase variable `phi`

        Returns
        -------
        float
        """
        dim = self.number_degrees_freedom
        pot_sum = np.sum([- self.EJlist[j] * np.cos(phi_array[j]) for j in range(dim)])
        pot_sum += (-self.EJlist[-1] * np.cos(np.sum([self.boundary_coefficients[i]*phi_array[i]
                                                      for i in range(dim)]) + 2*np.pi*self.flux))
        pot_sum += np.sum(self.EJlist)
        return pot_sum


class CurrentMirrorVCHOS(CurrentMirrorVCHOSFunctions, VCHOS, base.QubitBaseClass, serializers.Serializable):
    r""" Current Mirror using VCHOS

    See class CurrentMirror for documentation on the qubit itself.

    Initialize in the same way as for CurrentMirror, however now `num_exc` and `maximum_periodic_vector_length`
    must be set. See VCHOS for explanation.
    """
    maximum_periodic_vector_length = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, truncated_dim=None, **kwargs):
        VCHOS.__init__(self, EJlist, nglist, flux, number_degrees_freedom=2 * N - 1,
                       number_periodic_degrees_freedom=2 * N - 1, **kwargs)
        CurrentMirrorVCHOSFunctions.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux)
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self.truncated_dim = truncated_dim
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/' + str(type(self).__name__) + '.png')

    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0 / 2.7,
            'ECg': 20.0,
            'EJlist': np.array(6 * [18.95]),
            'nglist': np.array(5 * [0.0]),
            'flux': 0.0,
            'maximum_periodic_vector_length': 1,
            'num_exc': 2,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'maximum_periodic_vector_length', 'num_exc', 'truncated_dim']


class CurrentMirrorVCHOSSqueezing(VCHOSSqueezing, CurrentMirrorVCHOS):
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, truncated_dim, **kwargs):
        VCHOSSqueezing.__init__(self, EJlist=EJlist, nglist=nglist, flux=flux, number_degrees_freedom=2*N - 1,
                                number_periodic_degrees_freedom=2*N - 1, **kwargs)
        CurrentMirrorVCHOS.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, truncated_dim, **kwargs)


class CurrentMirrorVCHOSGlobal(Hashing, CurrentMirrorVCHOS):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, global_exc, **kwargs):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2*N - 1)
        CurrentMirrorVCHOS.__init__(self, N, **kwargs)


class CurrentMirrorVCHOSGlobalSqueezing(Hashing, CurrentMirrorVCHOSSqueezing):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, global_exc, **kwargs):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2*N - 1)
        CurrentMirrorVCHOSSqueezing.__init__(self, N, **kwargs)
