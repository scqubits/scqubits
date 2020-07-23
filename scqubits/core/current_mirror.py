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
from scqubits.core.operators import operator_in_full_Hilbert_space


class CurrentMirrorFunctions:
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux):
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.N = N
        self.number_degrees_freedom = 2*N - 1
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist
        self.nglist = nglist
        self.flux = flux

    def build_capacitance_matrix(self):
        N = self.N
        CB = self.e**2 / (2.*self.ECB)
        CJ = self.e**2 / (2.*self.ECJ)
        Cg = self.e**2 / (2.*self.ECg)

        C_matrix = np.diagflat([Cg + 2*CJ + CB for _ in range(2*N)], 0)
        C_matrix += np.diagflat([-CJ for _ in range(2*N - 1)], +1)
        C_matrix += np.diagflat([-CJ for _ in range(2*N - 1)], -1)
        C_matrix += np.diagflat([-CB for _ in range(N)], +N)
        C_matrix += np.diagflat([-CB for _ in range(N)], -N)
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
        V_m = np.diagflat([-1 for _ in range(2*N)], 0)
        V_m += np.diagflat([1 for _ in range(2*N - 1)], 1)
        V_m[-1] = np.array([1 for _ in range(2*N)])
        return V_m


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
        return (2*self.ncut+1)**self.number_degrees_freedom

    def hamiltonian(self):
        dim = self.number_degrees_freedom
        EC_matrix = self.build_EC_matrix()
        number_op = self._charge_number_operator()
        identity_op = self._identity_operator()
        identity_operator_list = self._identity_operator_list()
        
        H = 0.*self.identity_operator()
        for j, k in itertools.product(range(dim), range(dim)):
            if j != k:
                H += 4*EC_matrix[j, k]*operator_in_full_Hilbert_space([number_op - self.nglist[j]*identity_op,
                                                                       number_op - self.nglist[k]*identity_op],
                                                                      [j, k], identity_operator_list, sparse=True)
            else:
                n_squared = (number_op - self.nglist[j]*identity_op).dot(number_op - self.nglist[j]*identity_op)
                H += 4 * EC_matrix[j, j] * operator_in_full_Hilbert_space([n_squared], [j],
                                                                          identity_operator_list, sparse=True)
        for j in range(dim):
            H += (-self.EJlist[j]/2.)*(self.exp_i_phi_j_operator(j) + self.exp_i_phi_j_operator(j).T)
            H += self.EJlist[j]*self.identity_operator()
        H += (-self.EJlist[-1] / 2.) * np.exp(1j*2*np.pi*self.flux) * self.exp_i_phi_boundary_term().T
        H += (-self.EJlist[-1] / 2.) * np.exp(-1j*2*np.pi*self.flux) * self.exp_i_phi_boundary_term()
        H += self.EJlist[-1]*self.identity_operator()
        
        return H

    def _identity_operator(self):
        return sps.eye(2 * self.ncut + 1, k=0, format="csr", dtype=np.complex_)

    def _identity_operator_list(self):
        return [self._identity_operator() for _ in range(self.number_degrees_freedom)]

    def identity_operator(self):
        return operator_in_full_Hilbert_space([], [], self._identity_operator_list(), sparse=True)

    def _charge_number_operator(self):
        return sps.diags([i for i in range(-self.ncut, self.ncut + 1, 1)], offsets=0, format="csr", dtype=np.complex_)

    def charge_number_operator(self, j=0):
        number_operator = self._charge_number_operator()
        return operator_in_full_Hilbert_space([number_operator], [j], self._identity_operator_list(), sparse=True)

    def _exp_i_phi_j_operator(self):
        return sps.eye(2*self.ncut + 1, k=-1, format="csr", dtype=np.complex_)

    def exp_i_phi_j_operator(self, j=0):
        exp_i_phi_j = self._exp_i_phi_j_operator()
        return operator_in_full_Hilbert_space([exp_i_phi_j], [j], self._identity_operator_list(), sparse=True)

    def exp_i_phi_boundary_term(self):
        dim = self.number_degrees_freedom
        exp_i_phi_op = self._exp_i_phi_j_operator()
        identity_operator_list = self._identity_operator_list()
        return operator_in_full_Hilbert_space([exp_i_phi_op for _ in range(dim)],
                                              [j for j in range(dim)], identity_operator_list, sparse=True)
