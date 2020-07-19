import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
import itertools
import scipy.constants as const

import scqubits.core.qubit_base as base
import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.spectrum_utils import order_eigensystem
from scqubits.utils.misc import full_o


class CurrentMirrorFunctions:
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux):
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.N = N
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist
        V_m = self._build_V_m()
        self.nglist = np.dot(sp.linalg.inv(V_m).T, nglist)[0:-1]
        self.flux = flux

    def build_capacitance_matrix(self):
        N = self.N
        CB = self.e ** 2 / (2. * self.ECB)
        CJ = self.e ** 2 / (2. * self.ECJ)
        Cg = self.e ** 2 / (2. * self.ECg)

        C_matrix = np.diagflat(
            [Cg + 2 * CJ + CB for _ in range(2 * N)], 0)
        C_matrix += np.diagflat([- CJ for _ in range(2 * N - 1)], +1)
        C_matrix += np.diagflat([- CJ for _ in range(2 * N - 1)], -1)
        C_matrix += np.diagflat([- CB for _ in range(N)], +N)
        C_matrix += np.diagflat([- CB for _ in range(N)], -N)
        C_matrix[0, -1] = C_matrix[-1, 0] = - CJ

        V_m_inv = sp.linalg.inv(self._build_V_m())
        C_matrix = np.matmul(V_m_inv.T, np.matmul(C_matrix, V_m_inv))

        return C_matrix[0:-1, 0:-1]

    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        C_matrix = self.build_capacitance_matrix()
        return 0.5 * self.e**2 * sp.linalg.inv(C_matrix)

    def _build_V_m(self):
        N = self.N
        V_m = np.diagflat([-1 for _ in range(2 * N)], 0)
        V_m += np.diagflat([1 for _ in range(2 * N - 1)], 1)
        V_m[-1] = np.array([1 for _ in range(2 * N)])
        return V_m

    def number_degrees_freedom(self):
        return 2*self.N - 1


class CurrentMirror(base.QubitBaseClass, serializers.Serializable, CurrentMirrorFunctions):
    N = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECB = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJlist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    nglist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ncut = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, 
                 flux, ncut, truncated_dim=None):
        CurrentMirrorFunctions.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux)
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._evec_dtype = np.complex_
        
    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0/2.7,
            'ECg': 20.0,
            'EJlist': np.array(5*[18.95]),
            'nglist': np.array(5*[0.0]),
            'flux': 0.0,
            'ncut': 10,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'ncut', 'truncated_dim']

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=False)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=True)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs
    
    def hilbertdim(self):
        return (2*self.ncut+1)**self.number_degrees_freedom()

    def hamiltonian(self):
        no_node = self.number_degrees_freedom()
        ECmat, E_j_npl, n_gd_npl = self.build_EC_matrix(), self.EJlist, self.nglist
        phi = 2*np.pi*self.flux
        n_o, g_o, g_o_dg, i_o, i_o_list = self._basic_operators(np.complex_)
        
        H = 0.*full_o([], [], i_o_list)
        for j, k in itertools.product(range(no_node), range(no_node)):
            if j != k:
                H += 4 * ECmat[j, k] * full_o([n_o - n_gd_npl[j] * i_o,
                                               n_o - n_gd_npl[k] * i_o], [j, k], i_o_list)
            else:
                H += 4 * ECmat[j, j] * full_o([(n_o - n_gd_npl[j] * i_o)
                                              .dot(n_o - n_gd_npl[j] * i_o)], [j], i_o_list)
        
        for j in range(no_node):
            H += ((-E_j_npl[j] / 2.) * full_o([g_o], [j], i_o_list))
            H += ((-E_j_npl[j] / 2.) * full_o([g_o_dg], [j], i_o_list))
            H += E_j_npl[j]*full_o([], [], i_o_list)
        H += ((-E_j_npl[-1] / 2.) * np.exp(phi * 1j)
              * full_o([g_o for _ in range(no_node)], [j for j in range(no_node)], i_o_list))
        H += ((-E_j_npl[-1] / 2.) * np.exp(-phi * 1j)
              * full_o([g_o_dg for _ in range(no_node)], [j for j in range(no_node)], i_o_list))
        H += E_j_npl[-1]*full_o([], [], i_o_list)
        
        return H
    
    def _basic_operators(self, operator_dtype):
        no_node = self.number_degrees_freedom()
        cutoff = self.ncut
        nstate_s = 2 * cutoff + 1
        cutoff_range = [j for j in range(-cutoff, cutoff + 1, 1)]
        n_o = sps.diags(
            [j for j in cutoff_range], offsets=0,
            format="csr", dtype=operator_dtype)
        g_o = sps.eye(nstate_s, k=1, format="csr", dtype=operator_dtype)
        g_o_dg = sps.eye(nstate_s, k=-1, format="csr", dtype=operator_dtype)
        i_o = sps.eye(nstate_s, k=0, format="csr", dtype=operator_dtype)
        i_o_list = [i_o for _ in range(no_node)]
        return n_o, g_o, g_o_dg, i_o, i_o_list
