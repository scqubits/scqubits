import numpy as np
from scipy.special import comb

from scqubits.core.current_mirror_vchos import CurrentMirrorVCHOS
import scqubits.core.descriptors as descriptors
from scqubits.core.hashing import Hashing


# Current Mirror using VCHOS with a global excitation number cutoff scheme.
# The dimension of the hilbert space is then m*\frac{(global_exc+2N-2)!}{global_exc!(2N-2)!},
# where m is the number of inequivalent minima in 
# the first unit cell, N is the number of big capacitors and global_exc is the cutoff for 
# total number of excitations that we allow to be kept.

class CurrentMirrorVCHOSGlobal(CurrentMirrorVCHOS, Hashing):
    global_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux,
                 kmax, global_exc, squeezing=False, truncated_dim=None):
        CurrentMirrorVCHOS.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux,
                                    kmax, num_exc=None, truncated_dim=truncated_dim)
        Hashing.__init__(self)
        self._sys_type = type(self).__name__
        self.global_exc = global_exc

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
            'squeezing': False,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'kmax', 'global_exc', 'truncated_dim']

    def a_operator(self, i):
        """
        This method for defining the a_operator is based on
        J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
        We ask the question, for each basis vector, what is the action of a_i
        on it? In this way, we can define a_i using a single for loop.
        """
        basis_vecs = self._gen_basis_vecs()
        tags, index_array = self._gen_tags(basis_vecs)
        dim = basis_vecs.shape[0]
        a = np.zeros((dim, dim))
        for w, vec in enumerate(basis_vecs):
            if vec[i] >= 1:
                temp_vec = np.copy(vec)
                temp_vec[i] = vec[i] - 1
                temp_coeff = np.sqrt(vec[i])
                temp_vec_tag = self._hash(temp_vec)
                index = np.searchsorted(tags, temp_vec_tag)
                basis_index = index_array[index]
                a[basis_index, w] = temp_coeff
        return a

    def number_states_per_minimum(self):
        """
        Using the global excitation scheme the total number of states
        per minimum is given by the hockey-stick identity
        """
        return int(comb(self.global_exc + self.number_degrees_freedom(), self.number_degrees_freedom()))
