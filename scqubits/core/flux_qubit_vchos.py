import os

import numpy as np
from scipy.optimize import minimize
from typing import Callable

from scqubits.core import descriptors
from scqubits.core.flux_qubit import FluxQubitFunctions
from scqubits.core.hashing import Hashing
from scqubits.core.vchos import VCHOS
from scqubits.core.vchos_squeezing import VCHOSSqueezing
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


class FluxQubitVCHOSFunctions(FluxQubitFunctions):
    """Helper class for defining functions for VCHOS relevant to the Flux Qubit"""
    _check_if_new_minima: Callable
    _normalize_minimum_inside_pi_range: Callable

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux):
        FluxQubitFunctions.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux)
        # final term in potential is cos[(+1)\phi_1+(-1)\phi_2+2pi f]
        self.boundary_coefficients = np.array([+1, -1])

    def _ramp(self, k, minima_holder):
        """Helper function for find_minima"""
        guess = np.array([1.15 * 2.0 * np.pi * k / 3.0, 2.0 * np.pi * k / 3.0])
        result = minimize(self.potential, guess)
        new_minima = self._check_if_new_minima(result.x, minima_holder)
        if new_minima:
            minima_holder.append(self._normalize_minimum_inside_pi_range(result.x))
        return minima_holder, new_minima

    def find_minima(self):
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


class FluxQubitVCHOS(FluxQubitVCHOSFunctions, VCHOS, base.QubitBaseClass, serializers.Serializable):
    r""" Flux Qubit using VCHOS

    See class FluxQubit for documentation on the qubit itself.

    Initialize in the same way as for FluxQubit, however now `num_exc` and `maximum_periodic_vector_length`
    must be set. See VCHOS for explanation of other kwargs.
    """
    maximum_periodic_vector_length = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, truncated_dim=None, **kwargs):
        EJlist = np.array([EJ1, EJ2, EJ3])
        nglist = np.array([ng1, ng2])
        VCHOS.__init__(self, EJlist, nglist, flux, number_degrees_freedom=2,
                       number_periodic_degrees_freedom=2, **kwargs)
        FluxQubitVCHOSFunctions.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux)
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/'+str(type(self).__name__)+'.png')

    @staticmethod
    def default_params():
        return {
            'ECJ': 1.0 / 10.0,
            'ECg': 5.0,
            'EJlist': np.array([1.0, 1.0, 0.8]),
            'alpha': 0.8,
            'nglist': np.array(2 * [0.0]),
            'flux': 0.46,
            'maximum_periodic_vector_length': 1,
            'num_exc': 4,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['alpha', 'nglist', 'maximum_periodic_vector_length', 'num_exc', 'squeezing', 'truncated_dim']


class FluxQubitVCHOSSqueezing(VCHOSSqueezing, FluxQubitVCHOS):
    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, truncated_dim, **kwargs):
        EJlist = np.array([EJ1, EJ2, EJ3])
        nglist = np.array([ng1, ng2])
        VCHOSSqueezing.__init__(self, EJlist=EJlist, nglist=nglist, flux=flux, number_degrees_freedom=2,
                                number_periodic_degrees_freedom=2, **kwargs)
        FluxQubitVCHOS.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux,
                                truncated_dim, **kwargs)


class FluxQubitVCHOSGlobal(Hashing, FluxQubitVCHOS):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, global_exc=0, **kwargs):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2)
        FluxQubitVCHOS.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, **kwargs)


class FluxQubitVCHOSGlobalSqueezing(Hashing, FluxQubitVCHOSSqueezing):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, global_exc=0, **kwargs):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2)
        FluxQubitVCHOSSqueezing.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2, flux, **kwargs)
