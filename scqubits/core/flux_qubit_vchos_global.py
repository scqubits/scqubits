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
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.flux_qubit_vchos import FluxQubitVCHOS
from scqubits.core.hashing import Hashing
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem


#-Flux Qubit using VCHOS and a global cutoff

class FluxQubitVCHOSGlobal(FluxQubitVCHOS, Hashing):
    def __init__(self, ECJ, ECg, EJlist, alpha, nglist, flux, kmax,
                 global_exc, squeezing=False, truncated_dim=None):
        FluxQubitVCHOS.__init__(self, ECJ, ECg, EJlist, alpha, nglist, flux, 
                                kmax, num_exc=None, squeezing=squeezing, truncated_dim=truncated_dim)
        Hashing.__init__(self, num_deg_freedom=2, global_exc=global_exc)
        
    def a_operator(self, i):
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
    