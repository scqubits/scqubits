import math
import os

import numpy as np
from scipy.optimize import minimize

import scqubits.core.descriptors as descriptors
from scqubits.core.current_mirror import CurrentMirrorFunctions
from scqubits.core.hashing import Hashing
from scqubits.core.vchos import VCHOS
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


# Current Mirror using VCHOS. Truncation scheme used is defining a cutoff num_exc
# of the number of excitations kept for each mode. The dimension of the hilbert space
# is then m*(num_exc+1)**(2*N - 1), where m is the number of inequivalent minima in 
# the first unit cell and N is the number of big capacitors.


class CurrentMirrorVCHOS(CurrentMirrorFunctions, VCHOS, base.QubitBaseClass, serializers.Serializable):
    N = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECB = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJlist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    nglist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, kmax, num_exc, truncated_dim=None):
        CurrentMirrorFunctions.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux)
        VCHOS.__init__(self, EJlist, nglist, flux, kmax, number_degrees_freedom=2*N - 1,
                       number_periodic_degrees_freedom=2*N - 1, num_exc=num_exc)
        self.boundary_coeffs = np.ones(2 * N - 1)
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self.truncated_dim = truncated_dim
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/currentmirrorvchos.png')

    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0 / 2.7,
            'ECg': 20.0,
            'EJlist': np.array(5 * [18.95]),
            'nglist': np.array(5 * [0.0]),
            'flux': 0.0,
            'kmax': 1,
            'num_exc': 2,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'kmax', 'num_exc', 'truncated_dim']

    def find_minima(self):
        """
        Index all minima
        """
        minima_holder = []
        N = self.N
        for m in range(int(math.ceil(N / 2 - np.abs(self.flux))) + 1):
            guess_pos = np.array([np.pi * (m + self.flux) / N for _ in range(self.number_degrees_freedom)])
            guess_neg = np.array([np.pi * (-m + self.flux) / N for _ in range(self.number_degrees_freedom)])
            result_pos = minimize(self.potential, guess_pos)
            result_neg = minimize(self.potential, guess_neg)
            new_minimum_pos = self._check_if_new_minima(result_pos.x, minima_holder)
            if new_minimum_pos and result_pos.success:
                minima_holder.append(np.array([np.mod(elem, 2 * np.pi) for elem in result_pos.x]))
            new_minimum_neg = self._check_if_new_minima(result_neg.x, minima_holder)
            if new_minimum_neg and result_neg.success:
                minima_holder.append(np.array([np.mod(elem, 2 * np.pi) for elem in result_neg.x]))
        return minima_holder

    def potential(self, phi_array):
        """
        Potential evaluated at the location specified by phi_array

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
        pot_sum += (- self.EJlist[-1] * np.cos(np.sum([self.boundary_coeffs[i] * phi_array[i]
                                                       for i in range(dim)]) + 2 * np.pi * self.flux))
        return pot_sum


class CurrentMirrorVCHOSGlobal(Hashing, CurrentMirrorVCHOS, base.QubitBaseClass, serializers.Serializable):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, kmax, global_exc, truncated_dim=None):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2*N - 1)
        CurrentMirrorVCHOS.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux,
                                    kmax, num_exc=None, truncated_dim=truncated_dim)
        self._sys_type = type(self).__name__
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/currentmirrorvchosglobal.png')

    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0 / 2.7,
            'ECg': 20.0,
            'EJlist': np.array(5 * [18.95]),
            'nglist': np.array(5 * [0.0]),
            'flux': 0.0,
            'kmax': 1,
            'global_exc': 2,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'kmax', 'global_exc', 'truncated_dim']
