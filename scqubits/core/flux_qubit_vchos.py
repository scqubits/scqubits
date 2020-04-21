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
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem


#-Flux Qubit using VCHOS 

class FluxQubitVCHOS(QubitBaseClass):
    def __init__(self, ECJ, ECg, EJ, ng1, ng2, alpha, flux, kmax, num_exc):
        self.ECJ = ECJ
        self.EJ = EJ
        self.ECg = ECg
        self.ng1 = ng1
        self.ng2 = ng2
        self.alpha = alpha
        self.flux = flux
        self.kmax = kmax
        self.num_exc = num_exc
        self.hGHz = const.h * 10**9
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        self.Z0 = 1. / (2*self.e)**2
        self.Phi0 = 1. / (2*self.e)
        
        self._evec_dtype = np.float_
        self._default_grid = Grid1d(-6.5*np.pi, 6.5*np.pi, 651)
    
    def _normal_ordered_adag_a_exponential(self, x):
        """Expectation is that exp(a_{i}^{\dagger}x_{ij}a_{j}) needs to be normal ordered"""
        expx = sp.linalg.expm(x)
        result = self._identity()
        dim = result.shape[0]
        additionalterm = np.eye(dim)
        k = 1
        while not np.allclose(additionalterm, np.zeros((dim, dim))):
            additionalterm = np.sum([((expx-np.eye(2))[i, j])**(k)
                                     *(sp.special.factorial(k))**(-1)
                                     *np.matmul(np.linalg.matrix_power(self.a_operator(i).T, k), 
                                                np.linalg.matrix_power(self.a_operator(j), k))
                                    for i in range(2) for j in range(2)], axis=0)
            result += additionalterm
            k += 1
        return result
        
    def potential(self, phiarray):
        """
        Flux qubit potential evaluated at `phi1` and `phi2` 
        """
        phi1 = phiarray[0]
        phi2 = phiarray[1]
        return (-self.EJ*np.cos(phi1) -self.EJ*np.cos(phi2)
                -self.EJ*self.alpha*np.cos(phi1-phi2+2.0*np.pi*self.flux))
    
    def build_capacitance_matrix(self):
        """Return the capacitance matrix"""
        Cmat = np.zeros((2, 2))
                
        CJ = self.e**2 / (2.*self.ECJ)
        Cg = self.e**2 / (2.*self.ECg)
        
        Cmat[0, 0] = CJ + self.alpha*CJ + Cg
        Cmat[1, 1] = CJ + self.alpha*CJ + Cg
        Cmat[0, 1] = -self.alpha*CJ
        Cmat[1, 0] = -self.alpha*CJ
        
        return Cmat
    
    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        Cmat = self.build_capacitance_matrix()
        return  0.5 * self.e**2 * sp.linalg.inv(Cmat)
    
    def build_gamma_matrix(self, i):
        """
        Return linearized potential matrix
        
        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.
        
        """
        gmat = np.zeros((2,2))
        
        local_or_global_min = self.sorted_minima()[i]
        phi1_min = local_or_global_min[0]
        phi2_min = local_or_global_min[1]
        
        gamma = self.EJ / self.Phi0**2
        
        gmat[0, 0] = gamma*np.cos(phi1_min) + self.alpha*gamma*np.cos(2*np.pi*self.flux 
                                                                      + phi1_min - phi2_min)
        gmat[1, 1] = gamma*np.cos(phi2_min) + self.alpha*gamma*np.cos(2*np.pi*self.flux 
                                                                      + phi1_min - phi2_min)
        gmat[0, 1] = gmat[1, 0] = -self.alpha*gamma*np.cos(2*np.pi*self.flux + phi1_min - phi2_min)
        
        return gmat
    
    def omegamat(self):
        """Return a diagonal matrix of the normal mode frequencies of the global min """
        Cmat = self.build_capacitance_matrix()
        gmat = self.build_gamma_matrix(0)
        
        omegasq, eigvec = sp.linalg.eigh(gmat, b=Cmat)
        return np.diag(np.sqrt(omegasq))
            
    def Xi_matrix(self):
        """Construct the Xi matrix, encoding the oscillator lengths of each dimension"""
        Cmat = self.build_capacitance_matrix()
        gmat = self.build_gamma_matrix(0)
        
        omegasq, eigvec = sp.linalg.eigh(gmat, b=Cmat)
        
        Ximat = np.array([eigvec[:,i]*np.sqrt(np.sqrt(1./omegasq[i]))
                          * np.sqrt(1./self.Z0) for i in range(Cmat.shape[0])])
        
        # Note that the actual Xi matrix is the transpose of above, 
        # due to list comprehension syntax reasons. Here we are 
        # asserting that \Xi^T C \Xi = \Omega^{-1}/Z0
        assert(np.allclose(np.matmul(Ximat, np.matmul(Cmat, np.transpose(Ximat))),
                              sp.linalg.inv(np.diag(np.sqrt(omegasq)))/self.Z0))

        return np.transpose(Ximat)
    
    def a_operator(self, mu):
        """Return the lowering operator associated with the xth d.o.f. in the full Hilbert space"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a,k=1)
        return self._full_o([a_mat], [mu])
                    
    def normal_ordered_exp_i_phi_operator(self, x):
        """Return the normal ordered e^{i\phi_x} operator, expressed using ladder ops"""
        Xi_mat = self.Xi_matrix()
        return(np.exp(-.25*np.dot(Xi_mat[x, :], np.transpose(Xi_mat)[:, x]))
               *np.matmul(sp.linalg.expm(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu).T 
                                            for mu in range(2)], axis=0)/np.sqrt(2)), 
                          sp.linalg.expm(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu) 
                                            for mu in range(2)], axis=0)/np.sqrt(2))))
    
    def normal_ordered_exp_i_phix_mi_phiy(self, x, y):
        """Return the normal ordered e^{i\phi_x-i\phi_y} operator, expressed using ladder ops"""
        Xi_mat = self.Xi_matrix()
        a_dag_prod = np.matmul(sp.linalg.expm(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu).T 
                                                          for mu in range(2)], axis=0)/np.sqrt(2)),
                               sp.linalg.expm(-1j*np.sum([Xi_mat[y,mu]*self.a_operator(mu).T 
                                                           for mu in range(2)], axis=0)/np.sqrt(2)))
        a_prod = np.matmul(sp.linalg.expm(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu)
                                                      for mu in range(2)], axis=0)/np.sqrt(2)),
                           sp.linalg.expm(-1j*np.sum([Xi_mat[y,mu]*self.a_operator(mu)
                                                       for mu in range(2)], axis=0)/np.sqrt(2)))
        return(np.matmul(a_dag_prod, a_prod)
               *np.exp(-.25*np.dot(Xi_mat[x, :], np.transpose(Xi_mat)[:, x]))
               *np.exp(-.25*np.dot(Xi_mat[y, :], np.transpose(Xi_mat)[:, y]))
               *np.exp(0.5*np.dot(Xi_mat[y, :], np.transpose(Xi_mat)[:, x])))
        
    def _identity(self):
        return(np.identity((self.num_exc+1)**2))
        
    def delta_inv_matrix(self):
        """"Construct the delta inverse matrix, as described in David's notes """
        Xi_T_inv = np.transpose(sp.linalg.inv(self.Xi_matrix()))
        Xi_inv = sp.linalg.inv(self.Xi_matrix())
        return np.matmul(Xi_T_inv,Xi_inv)
    
    def _exp_a_operators(self):
        """Return the exponential of the a operators with appropriate coefficients for efficiency purposes """
        Xi = self.Xi_matrix()
        Xi_inv_T = sp.linalg.inv(Xi).T
        exp_a_0 = sp.linalg.expm(np.sum([2.0*np.pi*Xi_inv_T[0, mu]*self.a_operator(mu)/np.sqrt(2.0)
                                 for mu in range(2)], axis=0))
        exp_a_1 = sp.linalg.expm(np.sum([2.0*np.pi*Xi_inv_T[1, mu]*self.a_operator(mu)/np.sqrt(2.0)
                                 for mu in range(2)], axis=0))
        return(exp_a_0, exp_a_1)
    
    def _exp_a_operators_minima_diff(self, minima_diff):
        Xi_inv_T = sp.linalg.inv(self.Xi_matrix()).T
        exp_min_diff = sp.linalg.expm(np.sum([minima_diff[x]*Xi_inv_T[x, mu]*self.a_operator(mu)
                                               for x in range(2) for mu in range(2)], axis=0)/np.sqrt(2.0))
        return(exp_min_diff)
    
    def _V_operator_helper_using_exp_a_operators(self, phi, exp_a_list):
        """Return the periodic continuation part of the V operator without 
        additional calls to matrix_exp and without the prefactor """
        jkvals = phi/(2.0*np.pi)
                
        V0_op = np.linalg.matrix_power(exp_a_list[0], int(jkvals[0]))
        V1_op = np.linalg.matrix_power(exp_a_list[1], int(jkvals[1]))
        
        return(np.matmul(V0_op, V1_op))
            
    def V_operator(self, phi):
        """Return the V operator """
        phi_delta_phi = np.matmul(phi,np.matmul(self.delta_inv_matrix(),phi))
        prefactor = np.exp(-.125 * phi_delta_phi)
        phi_Xi_inv = np.matmul(phi,np.transpose(sp.linalg.inv(self.Xi_matrix())))
        phi_Xi_inv_a = np.sum([phi_Xi_inv[mu]*self.a_operator(mu) for mu in range(2)], axis=0)
        op = sp.linalg.expm((1./np.sqrt(2.))*phi_Xi_inv_a)
        return prefactor * op
    
    def V_operator_full(self, minima_diff, phik, exp_min_diff, exp_a_list, delta_inv):
        """Return the V operator using the more efficient methods """
        delta_phi_kpm = phik+minima_diff
        phi_delta_phi = np.matmul(delta_phi_kpm,np.matmul(delta_inv,delta_phi_kpm))
        prefactor = np.exp(-.125 * phi_delta_phi)
        V_op_phik = self._V_operator_helper_using_exp_a_operators(phik, exp_a_list)
        V_op = prefactor * np.matmul(exp_min_diff, V_op_phik)
        return V_op
        
    def _unordered_kineticmat(self):
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        dim = self.matrixdim()
        minima_list = self.sorted_minima()
        kinetic_mat = np.zeros((dim,dim), dtype=np.complex128)
        nglist = np.array([self.ng1, self.ng2])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = phik-(minima_m-minima_p)
                    minima_diff = minima_p-minima_m
                    
                    right_op = sp.linalg.expm(np.sum([-(1/np.sqrt(2.))*(phik+minima_p)[x]*np.transpose(Xi_inv)[x, mu]
                                                     *(self.a_operator(mu)-self.a_operator(mu).T)
                                                     for x in range(2) for mu in range(2)], axis=0))
                    left_op = sp.linalg.expm(np.sum([(1/np.sqrt(2.))*(minima_m)[x]*np.transpose(Xi_inv)[x, mu]
                                                     *(self.a_operator(mu)-self.a_operator(mu).T)
                                                     for x in range(2) for mu in range(2)], axis=0))
                    kinetic_temp = 0.0
                    for mu in range(2):
                        for nu in range(2):
                            kinetic_temp += -2.0*EC_mat_t[mu, nu]*(np.matmul(self.a_operator(mu)-self.a_operator(mu).T,
                                                                             self.a_operator(nu)-self.a_operator(nu).T))
                    kinetic_temp = (np.exp(-1j*np.dot(nglist, delta_phi_kpm))*
                                    np.matmul(left_op, np.matmul(kinetic_temp, right_op )))
                    num_exc_tot = self.hilbertdim()
                    kinetic_mat[m*num_exc_tot : m*num_exc_tot + num_exc_tot, 
                                p*num_exc_tot : p*num_exc_tot + num_exc_tot] += kinetic_temp
                    jkvals = next(klist,-1)
                    
        return kinetic_mat
        
    def kineticmat(self):
        """Return the kinetic part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        delta_inv = np.matmul(np.transpose(Xi_inv), Xi_inv)
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        dim = self.matrixdim()
        minima_list = self.sorted_minima()
        kinetic_mat = np.zeros((dim,dim), dtype=np.complex128)
        exp_a_list = self._exp_a_operators()
        nglist = np.array([self.ng1, self.ng2])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                exp_min_diff = self._exp_a_operators_minima_diff(minima_p-minima_m)
                exp_min_diff_inv = sp.linalg.inv(exp_min_diff)
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = phik-(minima_m-minima_p) #XXXXXXXXXX
                    minima_diff = minima_p-minima_m
                       
                    V_op = self.V_operator_full(-minima_diff, -phik, exp_min_diff_inv, exp_a_list, delta_inv)
                    V_op_dag = self.V_operator_full(minima_diff, phik, exp_min_diff, exp_a_list, delta_inv).T
                    
                    a_0 = self.a_operator(0)
                    a_1 = self.a_operator(1)
                    
                    left_mult = np.matmul(np.transpose(Xi_inv), EC_mat_t)
                    delta_left = np.matmul(delta_phi_kpm, left_mult)
                    
                    right_mult = np.matmul(EC_mat_t, Xi_inv)
                    delta_right = np.matmul(right_mult, delta_phi_kpm)
                    
                    delta_left_right = np.matmul(np.matmul(delta_left, Xi_inv), delta_phi_kpm)
                    
                    kinetic_temp = -0.5*4*EC_mat_t[0, 0]*(np.matmul(a_0, a_0) - 2.0*np.matmul(a_0.T, a_0)
                                                           + np.matmul(a_0.T, a_0.T) - self._identity())
                    kinetic_temp += -(1./(2*np.sqrt(2)))*4*(a_0 - a_0.T)*(delta_right[0] + delta_left[0])
                                                         
                    kinetic_temp += -0.5*4*EC_mat_t[1, 1]*(np.matmul(a_1, a_1) - 2.0*np.matmul(a_1.T, a_1)
                                                           + np.matmul(a_1.T, a_1.T) - self._identity())
                    kinetic_temp += -(1./(2*np.sqrt(2)))*4*(a_1 - a_1.T)*(delta_right[1] + delta_left[1])
                                     
                    kinetic_temp += -self._identity()*delta_left_right
                    
#                    if m == 0 and p == 0: print(kinetic_temp)
                                             
                    kinetic_temp = (np.exp(-1j*np.dot(nglist, delta_phi_kpm))
                                    *np.matmul(V_op_dag, kinetic_temp))
                    kinetic_temp = np.matmul(kinetic_temp, V_op)
#                    if m == 0 and p == 0: print(kinetic_temp)
                    
                    num_exc_tot = self.hilbertdim()
                    kinetic_mat[m*num_exc_tot : m*num_exc_tot + num_exc_tot, 
                                p*num_exc_tot : p*num_exc_tot + num_exc_tot] += kinetic_temp
                    
                    jkvals = next(klist,-1)
                                           
        return kinetic_mat
        
    def potentialmat(self):
        """Return the potential part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        delta_inv = np.matmul(np.transpose(Xi_inv), Xi_inv)
        dim = self.matrixdim()
        potential_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        exp_i_phi_0 = self.normal_ordered_exp_i_phi_operator(0)
        exp_i_phi_1 = self.normal_ordered_exp_i_phi_operator(1)
        exp_i_phi_0_m1 = self.normal_ordered_exp_i_phix_mi_phiy(0, 1)
        exp_a_list = self._exp_a_operators()
        nglist = np.array([self.ng1, self.ng2])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                exp_min_diff = self._exp_a_operators_minima_diff(minima_p-minima_m)
                exp_min_diff_inv = sp.linalg.inv(exp_min_diff)
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = phik-(minima_m-minima_p) 
                    phibar_kpm = 0.5*(phik+(minima_m+minima_p)) 
                    minima_diff = minima_p-minima_m
                        
                    exp_i_phi_0_op = np.exp(1j*phibar_kpm[0])*exp_i_phi_0
                    exp_i_phi_1_op = np.exp(1j*phibar_kpm[1])*exp_i_phi_1
                        
                    V_op = self.V_operator_full(-minima_diff, -phik, exp_min_diff_inv, exp_a_list, delta_inv)
                    V_op_dag = self.V_operator_full(minima_diff, phik, exp_min_diff, exp_a_list, delta_inv).T
                                
                    potential_temp = -0.5*self.EJ*(exp_i_phi_0_op+exp_i_phi_0_op.conjugate().T)
                    potential_temp += -0.5*self.EJ*(exp_i_phi_1_op+exp_i_phi_1_op.conjugate().T)
                    potential_temp += -0.5*self.alpha*self.EJ*((exp_i_phi_0_m1
                                                                *np.exp(1j*2.0*np.pi*self.flux)
                                                                *np.exp(1j*phibar_kpm[0])
                                                                *np.exp(-1j*phibar_kpm[1]))
                                                               +(exp_i_phi_0_m1.conjugate().T
                                                                 *np.exp(-1j*2.0*np.pi*self.flux)
                                                                 *np.exp(-1j*phibar_kpm[0])
                                                                 *np.exp(1j*phibar_kpm[1])))
                    potential_temp += self.EJ*(self.alpha+2.0)*self._identity()

                    potential_temp = (np.exp(-1j*np.dot(nglist, delta_phi_kpm))
                                     *np.matmul(V_op_dag, potential_temp))
                    potential_temp = np.matmul(potential_temp, V_op)
                    
                    num_exc_tot = self.hilbertdim()
                    potential_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                  p*num_exc_tot:p*num_exc_tot+num_exc_tot] += potential_temp
                    jkvals = next(klist,-1)
                        
        return potential_mat                                                                          
                                                                          
    def hamiltonian(self):
        """Construct the Hamiltonian"""
        return (self.kineticmat() + self.potentialmat())
        
    def inner_product(self):
        """Return the inner product matrix, which is nontrivial with VCHOS states"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        delta_inv = np.matmul(np.transpose(Xi_inv), Xi_inv)
        dim = self.matrixdim()
        inner_product_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        exp_a_list = self._exp_a_operators()
        nglist = np.array([self.ng1, self.ng2])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                exp_min_diff = self._exp_a_operators_minima_diff(minima_p-minima_m)
                exp_min_diff_inv = sp.linalg.inv(exp_min_diff)
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = phik-(minima_m-minima_p)
                    minima_diff = minima_p-minima_m
                    
                    V_op = self.V_operator_full(-minima_diff, -phik, exp_min_diff_inv, exp_a_list, delta_inv)
                    V_op_dag = self.V_operator_full(minima_diff, phik, exp_min_diff, exp_a_list, delta_inv).T
                    
                    inner_temp = (np.exp(-1j*np.dot(nglist, delta_phi_kpm))
                                  *np.matmul(V_op_dag, V_op))
                    
                    num_exc_tot = self.hilbertdim()
                    inner_product_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                      p*num_exc_tot:p*num_exc_tot+num_exc_tot] += inner_temp
                    jkvals = next(klist,-1)
                        
        return inner_product_mat
    
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
    
    def _ramp(self, k, minima_holder):
        """
        Helper function for find_minima, performing the ramp that
        is described in Sec. III E of [0]
        
        [0] PRB ...
        """
        guess = np.array([1.15*2.0*np.pi*k/3.0,2.0*np.pi*k/3.0])
        result = minimize(self.potential, guess)
        new_minima = self._check_if_new_minima(result.x, minima_holder)
        if new_minima:
            minima_holder.append(np.array([np.mod(elem,2*np.pi) for elem in result.x]))
        return (minima_holder, new_minima)
    
    def find_minima(self):
        """
        Index all minima in the variable space of phi1 and phi2
        """
        minima_holder = []
        if self.flux == 0.5:
            guess = np.array([0.15,0.1])
        else:
            guess = np.array([0.0,0.0])
        result = minimize(self.potential,guess)
        minima_holder.append(np.array([np.mod(elem,2*np.pi) for elem in result.x]))
        k = 0
        for k in range(1,4):
            (minima_holder, new_minima_positive) = self._ramp(k, minima_holder)
            (minima_holder, new_minima_negative) = self._ramp(-k, minima_holder)
            if not (new_minima_positive and new_minima_negative):
                break
        return(minima_holder)
    
    def sorted_minima(self):
        """Sort the minima based on the value of the potential at the minima """
        minima_holder = self.find_minima()
        value_of_potential = np.array([self.potential(minima_holder[x]) 
                                       for x in range(len(minima_holder))])
        sorted_minima_holder = np.array([x for _, x in 
                                         sorted(zip(value_of_potential, minima_holder))])
        return sorted_minima_holder
    
    def matrixdim(self):
        """Return N if the size of the Hamiltonian matrix is NxN"""
        return len(self.sorted_minima())*(self.num_exc+1)**2
    
    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return (self.num_exc+1)**2
    
    def wavefunction(self, esys=None, which=0, phi_grid=None):
        """
        Return a flux qubit wave function in phi1, phi2 basis

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)
        phi_range: tuple(float, float), optional
            used for setting a custom plot range for phi
        phi_count: int, optional
            number of points to use on grid in each direction

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        phi_grid = self._try_defaults(phi_grid)
        phi_vec = phi_grid.make_linspace()
        zeta_vec = phi_grid.make_linspace()
#        phi_vec = np.linspace(phi_grid.min_val, phi_grid.max_val, 10)
        
        minima_list = self.sorted_minima()
        num_minima = len(minima_list)
        dim = self.hilbertdim()
        num_deg_freedom = (self.num_exc+1)**2
        
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        norm = np.sqrt(np.abs(np.linalg.det(Xi)))**(-1)
        
        state_amplitudes_list = []
        
        phi1_phi2_outer = np.outer(phi_vec, phi_vec)
        wavefunc_amplitudes = np.zeros_like(phi1_phi2_outer)
        
        for i, minimum in enumerate(minima_list):
            klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
            jkvals = next(klist,-1)
            while jkvals != -1:
                phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                phi1_s1_arg = (Xi_inv[0,0]*phik[0] - Xi_inv[0,0]*minimum[0])
                phi2_s1_arg = (Xi_inv[0,1]*phik[1] - Xi_inv[0,1]*minimum[1])
                phi1_s2_arg = (Xi_inv[1,0]*phik[0] - Xi_inv[1,0]*minimum[0])
                phi2_s2_arg = (Xi_inv[1,1]*phik[1] - Xi_inv[1,1]*minimum[1])
                state_amplitudes = np.real(np.reshape(evecs[i*num_deg_freedom : 
                                                            (i+1)*num_deg_freedom, which],
                                                      (self.num_exc+1, self.num_exc+1)))
#                state_amplitudes = np.zeros_like(state_amplitudes)
#                state_amplitudes[2,0] = 1.0
                wavefunc_amplitudes += np.sum([state_amplitudes[s1, s2] * norm
                * np.multiply(self.harm_osc_wavefunction(s1, np.add.outer(Xi_inv[0,0]*phi_vec+phi1_s1_arg, 
                                                                          Xi_inv[0,1]*phi_vec+phi2_s1_arg)), 
                              self.harm_osc_wavefunction(s2, np.add.outer(Xi_inv[1,0]*phi_vec+phi1_s2_arg,
                                                                          Xi_inv[1,1]*phi_vec+phi2_s2_arg)))
                                               for s2 in range(self.num_exc+1) 
                                               for s1 in range(self.num_exc+1)], axis=0).T #FIX .T NOT CORRECT
                jkvals = next(klist,-1)
        
        grid2d = GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                      [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
    
        wavefunc_amplitudes = standardize_phases(wavefunc_amplitudes)

        return WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)
    
   
    def plot_wavefunction(self, esys=None, which=0, phi_grid=None, mode='abs', zero_calibrate=True, **kwargs):
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

        Returns
        -------
        Figure, Axes
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (5, 5)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
    
    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        inner_product_mat = self.inner_product()
        try:
            evals = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat, 
                                   eigvals_only=True, eigvals=(0, evals_count - 1))
        except LinAlgError:
            print("exception")
#            global_min = self.sorted_minima()[0]
#            global_min_value = self.potential(global_min)
#            hamiltonian_mat += -global_min_value*inner_product_mat
            evals = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, M=inner_product_mat, 
                                           sigma=0.00001, return_eigenvectors=False)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        inner_product_mat = self.inner_product()
        try:
            evals, evecs = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat,
                                          eigvals_only=False, eigvals=(0, evals_count - 1))
            evals, evecs = order_eigensystem(evals, evecs)
        except LinAlgError:
            print("exception")
#            global_min = self.sorted_minima()[0]
#            global_min_value = self.potential(global_min)
#            hamiltonian_mat += -global_min_value*inner_product_mat
            evals, evecs = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, M=inner_product_mat, 
                                                  sigma=0.00001, return_eigenvectors=True)
            evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs
    
    def plot_potential(self, phi_grid=None, contour_vals=None, **kwargs):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
        contour_vals: list of float, optional
            specific contours to draw
        **kwargs:
            plot options
        """
        phi_grid = self._try_defaults(phi_grid)
        x_vals = y_vals = phi_grid.make_linspace()
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (5, 5)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)
    
    def _full_o(self, operators, indices):
        i_o = np.eye(self.num_exc + 1)
        i_o_list = [i_o for k in range(2)]
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
    
    def harm_osc_wavefunction(self, n, x):
        """For given quantum number n=0,1,2,... return the value of the harmonic oscillator wave function
        :math:`\\psi_n(x) = N H_n(x) \\exp(-x^2/2)`, N being the proper normalization factor. It is assumed
        that the harmonic length has already been accounted for. Therefore that portion of the normalization
        factor must be accounted for outside the function.

        Parameters
        ----------
        n: int
            index of wave function, n=0 is ground state
        x: float or ndarray
            coordinate(s) where wave function is evaluated

        Returns
        -------
        float or ndarray
            value(s) of harmonic oscillator wave function
        """
        return ((2.0 ** n * sp.special.gamma(n + 1.0)) ** (-0.5) * np.pi ** (-0.25) 
                * sp.special.eval_hermite(n, x) 
                * np.exp(-x**2/2.))
 