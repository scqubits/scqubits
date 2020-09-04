import os

import numpy as np
from scipy.optimize import minimize

from scqubits.core import descriptors
from scqubits.core.flux_qubit import FluxQubitFunctions
from scqubits.core.hashing import Hashing
from scqubits.core.vchos import VCHOS
from scqubits.core.vchos_squeezing import VCHOSSqueezing
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


# Flux Qubit using VCHOS

class FluxQubitVCHOS(FluxQubitFunctions, VCHOS, base.QubitBaseClass, serializers.Serializable):
    EJ1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJ2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJ3 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ3 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng1 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng2 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    maximum_periodic_vector_length = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2,
                 flux, maximum_periodic_vector_length, num_exc, truncated_dim=None):
        EJlist = np.array([EJ1, EJ2, EJ3])
        nglist = np.array([ng1, ng2])
        VCHOS.__init__(self, EJlist, nglist, flux, maximum_periodic_vector_length, number_degrees_freedom=2,
                       number_periodic_degrees_freedom=2, num_exc=num_exc)
        FluxQubitFunctions.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2,
                                    ECJ3, ECg1, ECg2, ng1, ng2, flux)
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/fluxqubitvchos.png')
        # final term in potential is cos[(+1)\phi_1+(-1)\phi_2-2pi f]
        self.boundary_coefficients = np.array([+1, -1])

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

    def _ramp(self, k, minima_holder):
        """
        Helper function for find_minima, performing the ramp that
        is described in Sec. III E of [0]

        [0] PRB ...
        """
        guess = np.array([1.15 * 2.0 * np.pi * k / 3.0, 2.0 * np.pi * k / 3.0])
        result = minimize(self.potential, guess)
        new_minima = self._check_if_new_minima(result.x, minima_holder)
        if new_minima:
            minima_holder.append(self.normalize_minimum_inside_pi_range(result.x))
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
        minima_holder.append(self.normalize_minimum_inside_pi_range(result.x))
        for k in range(1, 4):
            (minima_holder, new_minima_positive) = self._ramp(k, minima_holder)
            (minima_holder, new_minima_negative) = self._ramp(-k, minima_holder)
            if not (new_minima_positive and new_minima_negative):
                break
        return minima_holder

    def villain_potential(self, m_list, phi_array):
        """Harmonic approximation of the potential with Villain shifts"""
        phi1 = phi_array[0]
        phi2 = phi_array[1]
        return (0.5*self.EJ1*(phi1-2*np.pi*m_list[0])**2 + 0.5*self.EJ2*(phi2-2*np.pi*m_list[1])**2
                + 0.5*self.EJ3*(2.0*np.pi*self.flux + phi1 - phi2 - 2.0*np.pi*m_list[2])**2
                + self.EJ1 + self.EJ2 + self.EJ3)


class FluxQubitVCHOSSqueezing(VCHOSSqueezing, FluxQubitVCHOS, base.QubitBaseClass, serializers.Serializable):
    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2,
                 ECJ3, ECg1, ECg2, ng1, ng2, flux, maximum_periodic_vector_length,
                 num_exc, truncated_dim=None):
        EJlist = np.array([EJ1, EJ2, EJ3])
        nglist = np.array([ng1, ng2])
        VCHOSSqueezing.__init__(self, EJlist, nglist, flux, maximum_periodic_vector_length, number_degrees_freedom=2,
                                number_periodic_degrees_freedom=2, num_exc=num_exc)
        FluxQubitVCHOS.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2,
                                flux, maximum_periodic_vector_length, num_exc=num_exc, truncated_dim=truncated_dim)
        self._sys_type = type(self).__name__
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/fluxqubitvchossqueezing.png')


class FluxQubitVCHOSGlobal(Hashing, FluxQubitVCHOS, base.QubitBaseClass, serializers.Serializable):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2,
                 ECJ3, ECg1, ECg2, ng1, ng2, flux, maximum_periodic_vector_length,
                 global_exc, truncated_dim=None):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2)
        FluxQubitVCHOS.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2,
                                flux, maximum_periodic_vector_length, num_exc=None, truncated_dim=truncated_dim)
        self._sys_type = type(self).__name__
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/fluxqubitvchosglobal.png')

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
            'global_exc': 4,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['alpha', 'nglist', 'maximum_periodic_vector_length', 'global_exc', 'squeezing', 'truncated_dim']


class FluxQubitVCHOSGlobalSqueezing(VCHOSSqueezing, FluxQubitVCHOSGlobal):
    def __init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2,
                 ECJ3, ECg1, ECg2, ng1, ng2, flux, maximum_periodic_vector_length,
                 global_exc, truncated_dim=None):
        EJlist = np.array([EJ1, EJ2, EJ3])
        nglist = np.array([ng1, ng2])
        VCHOSSqueezing.__init__(self, EJlist, nglist, flux, maximum_periodic_vector_length, number_degrees_freedom=2,
                                number_periodic_degrees_freedom=2, num_exc=None)
        FluxQubitVCHOSGlobal.__init__(self, EJ1, EJ2, EJ3, ECJ1, ECJ2, ECJ3, ECg1, ECg2, ng1, ng2,
                                      flux, maximum_periodic_vector_length, global_exc=global_exc,
                                      truncated_dim=truncated_dim)
        self._sys_type = type(self).__name__
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/fluxqubitvchosglobalsqueezing.png')
