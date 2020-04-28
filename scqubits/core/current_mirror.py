import numpy as np
import scipy as sp
import scipy.sparse as sps
import itertools
import scipy.constants as const

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core.discretization import GridSpec, Grid1d
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem

class CurrentMirror(QubitBaseClass):
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, 
                 flux, ncut, truncated_dim=None):
        self.num_big_cap = N
        self.num_deg_freedom = 2*N - 1
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist
        
        V_m = self._build_V_m()
        self.nglist = np.dot(sp.linalg.inv(V_m).T, nglist)[0:-1]
        
        self.flux = flux
        self.ncut = ncut
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        
        self._basic_operators(self.ncut, self.num_deg_freedom, np.complex_)
        self._evec_dtype = np.complex_

    def build_capacitance_matrix(self):
        N = self.num_big_cap
        CB = self.e**2 / (2.*self.ECB)
        CJ = self.e**2 / (2.*self.ECJ)
        Cg = self.e**2 / (2.*self.ECg)
        
        Cmat = np.diagflat(
            [Cg + 2 * CJ + CB for j in range(2 * N)], 0)
        Cmat += np.diagflat([- CJ for j in range(2 * N - 1)], +1)
        Cmat += np.diagflat([- CJ for j in range(2 * N - 1)], -1)
        Cmat += np.diagflat([- CB for j in range(N)], +N)
        Cmat += np.diagflat([- CB for j in range(N)], -N)
        Cmat[0, -1] = Cmat[-1, 0] = - CJ
        
        V_m_inv = sp.linalg.inv(self._build_V_m())
        Cmat = np.matmul(V_m_inv.T, np.matmul(Cmat, V_m_inv))
        
        return Cmat[0:-1, 0:-1]
    
    def _build_V_m(self):
        N = self.num_big_cap
        V_m = np.diagflat([-1 for j in range(2*N)], 0)
        V_m += np.diagflat([1 for j in range(2*N - 1)], 1)
        V_m[-1] = np.array([1 for j in range(2*N)])
        
        return V_m
    
    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        Cmat = self.build_capacitance_matrix()
        return 0.5 * self.e**2 * sp.linalg.inv(Cmat)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sps.linalg.eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=False)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sps.linalg.eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=True)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs
    
    def hilbertdim(self):
        return((2*self.ncut+1)**(self.num_deg_freedom))

    def hamiltonian(self):
        no_node = self.num_deg_freedom
        ECmat, E_j_npl, n_gd_npl = self.build_EC_matrix(), self.EJlist, self.nglist
        phi = 2*np.pi*self.flux
        n_o, g_o, g_o_dg, i_o = self._n_o, self._g_o, self._g_o_dg, self._i_o
        full_o = self.full_o
        
        H = 0.
        for j, k in itertools.product(range(no_node), range(no_node)):
            if j != k:
                H += 4 * ECmat[j, k] * full_o([n_o - n_gd_npl[j] * i_o,
                                               n_o - n_gd_npl[k] * i_o], [j, k])
            else:
                H += 4 * ECmat[j, j] * full_o([(n_o - n_gd_npl[j] * i_o)
                                               .dot(n_o - n_gd_npl[j] * i_o)],[j])
        
        for j in range(no_node):
            H += ((-E_j_npl[j] / 2.)* full_o([g_o], [j]))
            H += ((-E_j_npl[j] / 2.) * full_o([g_o_dg], [j]))
            H += E_j_npl[j]*full_o([],[])
        H += ((-E_j_npl[-1] / 2.) * np.exp(phi * 1j)
              * full_o([g_o for j in range(no_node)], [j for j in range(no_node)]))
        H += ((-E_j_npl[-1] / 2.) * np.exp(-phi * 1j)
              * full_o([g_o_dg for j in range(no_node)],
                       [j for j in range(no_node)]))
        H += E_j_npl[-1]*full_o([],[])
        
        return H
    
    def _basic_operators(self, cutoff, no_node, operator_dtype):
        nstate_s = 2 * cutoff + 1
        cutoff_range = [j for j in range(-cutoff, cutoff + 1, 1)]
        n_o = sps.diags(
            [j for j in cutoff_range], offsets=0,
            format="csr", dtype=operator_dtype)
        g_o = sps.eye(nstate_s, k=1, format="csr", dtype=operator_dtype)
        g_o_dg = sps.eye(nstate_s, k=-1, format="csr", dtype=operator_dtype)
        i_o = sps.eye(nstate_s, k=0, format="csr", dtype=operator_dtype)
        i_o_list = [i_o for j in range(no_node)]
        self._n_o = n_o
        self._g_o = g_o
        self._g_o_dg = g_o_dg
        self._i_o = i_o
        self._i_o_list = i_o_list
        
    def full_o(self, operators, indices, i_o_list=None):
        if i_o_list is None:
            i_o_list = self._i_o_list
        product_list = i_o_list[:]
        oi_list = zip(operators, indices)
        for oi in oi_list:
            product_list[oi[1]] = oi[0]
        full_op = self.kron_sparse_matrix_list(product_list)
        return(full_op)
    
    def kron_sparse_matrix_list(self, sparse_list):
        output = sparse_list[0]
        for matrix in sparse_list[1:]:
            output = sps.kron(output, matrix, format="csr")
        return(output)
    