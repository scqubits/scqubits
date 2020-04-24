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
from scqubits.core.vchos import VCHOS
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem


#-Flux Qubit using VCHOS 

class CurrentMirrorVCHOS(VCHOS):
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, kmax, num_exc, squeezing=False):
        self.num_big_cap = N
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist
        
        V_m = self._build_V_m()
        self.nglist = np.dot(sp.linalg.inv(V_m).T, nglist)[0:-1]
        
        self.flux = flux
        self.kmax = kmax
        self.num_exc = num_exc
        self.squeezing = squeezing
        self.hGHz = const.h * 10**9
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        self.Z0 = 1. / (2*self.e)**2
        self.Phi0 = 1. / (2*self.e)
        self.boundary_coeffs = np.ones(2*N-1)
        self.num_deg_freedom = 2*N - 1
        
        self._evec_dtype = np.complex_
        self._default_grid = Grid1d(-6.5*np.pi, 6.5*np.pi, 651)
    
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
    
    def a_operator(self, mu):
        """Return the lowering operator associated with the xth d.o.f. in the full Hilbert space"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a,k=1)
        return self._full_o([a_mat], [mu])
        
    def _identity(self):
        return(np.identity(self.hilbertdim(), dtype=np.complex_))
    
    def matrixdim(self):
        """Return N if the size of the Hamiltonian matrix is NxN"""
        return len(self.sorted_minima())*(self.num_exc+1)**(2*self.num_big_cap - 1)
    
    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return (self.num_exc+1)**(2*self.num_big_cap - 1)
    
    def _check_if_new_minima(self, new_minima, minima_holder):
        """
        Helper function for find_minima, checking if minima is
        already represented in minima_holder. If so, 
        _check_if_new_minima returns False.
        """
        new_minima_bool = True
        for minima in minima_holder:
            diff_array = minima - new_minima
            diff_array_reduced = np.array([np.mod(x,2*np.pi) for x in diff_array])
            elem_bool = True
            for elem in diff_array_reduced:
                # if every element is zero or 2pi, then we have a repeated minima
                elem_bool = elem_bool and (np.allclose(elem,0.0,atol=1e-3) 
                                           or np.allclose(elem,2*np.pi,atol=1e-3))
            if elem_bool:
                new_minima_bool = False
                break
        return new_minima_bool
    
    def find_minima(self):
        """
        Index all minima
        """
        minima_holder = []
        N = self.num_big_cap
        for l in range(int(N/2 + self.flux)):
            guess_pos = np.array([np.pi*l/N for j in range(self.num_deg_freedom)])
            guess_neg = np.array([-np.pi*l/N for j in range(self.num_deg_freedom)])
            result_pos = minimize(self.potential, guess_pos)
            result_neg = minimize(self.potential, guess_neg)
            new_minimum_pos = self._check_if_new_minima(result_pos.x, minima_holder)
            if new_minimum_pos:
                minima_holder.append(np.array([np.mod(elem,2*np.pi) 
                                               for elem in result_pos.x]))
            new_minimum_neg = self._check_if_new_minima(result_neg.x, minima_holder)
            if new_minimum_neg:
                minima_holder.append(np.array([np.mod(elem,2*np.pi) 
                                               for elem in result_neg.x]))
        return(minima_holder)
    
    def sorted_minima(self):
        """Sort the minima based on the value of the potential at the minima """
        minima_holder = self.find_minima()
        value_of_potential = np.array([self.potential(minima_holder[x]) 
                                       for x in range(len(minima_holder))])
        sorted_minima_holder = np.array([x for _, x in 
                                         sorted(zip(value_of_potential, minima_holder))])
        return sorted_minima_holder
    
    def _full_o(self, operators, indices):
        i_o = np.eye(self.num_exc + 1)
        i_o_list = [i_o for k in range(self.num_deg_freedom)]
        product_list = i_o_list[:]
        oi_list = zip(operators, indices)
        for oi in oi_list:
            product_list[oi[1]] = oi[0]
        full_op = self._kron_matrix_list(product_list)
        return(full_op)
    
    def _kron_matrix_list(self, matrix_list):
        output = matrix_list[0]
        for matrix in matrix_list[1:]:
            output = np.kron(output, matrix)
        return(output)
