import numpy as np
import scipy as sp
import itertools
from scipy.optimize import minimize
import scipy.constants as const
from scipy.special import hermite
from scipy.linalg import LinAlgError

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core.discretization import GridSpec, Grid1d
from scqubits.core.current_mirror_vchos import CurrentMirrorVCHOS
from scqubits.core.hashing import Hashing
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem

# Current Mirror using VCHOS with a global excitation number cutoff scheme. 
# The dimension of the hilbert space is then m*\frac{(global_exc+2N-2)!}{global_exc!(2N-2)!},
# where m is the number of inequivalent minima in 
# the first unit cell, N is the number of big capacitors and global_exc is the cutoff for 
# total number of excitations that we allow to be kept.

class CurrentMirrorVCHOSGlobal(CurrentMirrorVCHOS, Hashing):
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, 
                 kmax, global_exc, squeezing=False, truncated_dim=None):
        CurrentMirrorVCHOS.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, 
                                    kmax, num_exc=None, squeezing=squeezing, truncated_dim=truncated_dim)
        Hashing.__init__(self, num_deg_freedom=2*N-1, global_exc=global_exc)
        
    def a_operator(self, i):
        """
        This method for defining the a_operator is based on
        J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
        We ask the question, for each basis vector, what is the action of a_i
        on it? In this way, we can define a_i using a single for loop.
        """
        basis_vecs = self._gen_basis_vecs()
        tags, index_array = self._gen_tags()
        dim = basis_vecs.shape[0]
        a = np.zeros((dim, dim))
        for w, vec in enumerate(basis_vecs):
            temp_vec = np.copy(vec)
            if vec[i] >= 1:
                temp_vec[i] = vec[i] - 1
                temp_coeff = np.sqrt(vec[i])
                temp_vec_tag = self._hash(temp_vec)
                index = np.searchsorted(self.tag_list, temp_vec_tag)
                basis_index = self.index_array[index]
                a[basis_index, w] = temp_coeff
        return a
        
    def hilbertdim(self):
        return len(self.sorted_minima())*len(self.tag_list)
    