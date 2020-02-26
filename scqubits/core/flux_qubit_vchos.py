import numpy as np
import scipy as sp
import itertools
from scipy.optimize import minimize
import scipy.constants as const
from scipy.special import hermite
from scipy.special import factorial
from scipy.special import comb
import scipy.integrate as integrate
import math

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core.discretization import GridSpec, Grid1d
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem


#-Flux Qubit using VCHOS 

def heat(x, y, n):
    heatsum = 0.0
    for k in range(math.floor(float(n)/2.0)+1):
        heatsum += x**(n-2*k)*y**(k)/(factorial(n-2*k)*factorial(k))
    return(heatsum*factorial(n))

def Hmn(m, n, x, y, w, z, tau):
    Hmnsum = 0.0
    for i in range(min(m,n)+1):
        Hmnsum += (factorial(m)*factorial(n)/(factorial(m-i)*factorial(n-i)*factorial(i))
                   *heat(x,y,m-i)*heat(w,z,n-i)*tau**i)
    return(Hmnsum)

def Imn(m,n,y,z,a,b,c,d,f,alpha):
    xbar = b+(a*alpha/(2.0*f))
    ybar = y+(a**2)/(4.0*f)
    wbar = d+(c*alpha)/(2.0*f)
    zbar = z+(c**2)/(4.0*f)
    tau  = a*c/(2.0*f)
    return(np.sqrt(np.pi/f)*np.exp(alpha**2/(4.0*f))*Hmn(m,n,xbar,ybar,wbar,zbar,tau)/
           (np.sqrt(np.sqrt(np.pi)*2**n*factorial(n))*np.sqrt(np.sqrt(np.pi)*2**m*factorial(m))))

def Rmnk(m,n,y,z,a,b,c,d,f,alpha,k):
    xbar = b+(a*alpha/(2.0*f))
    ybar = y+(a**2)/(4.0*f)
    wbar = d+(c*alpha)/(2.0*f)
    zbar = z+(c**2)/(4.0*f)
    tau  = a*c/(2.0*f)
    Rmnksum = 0.0
    for l in range(k+1):
        if(m-l>=0 and n-k+l>=0):
            Rmnksum+=(comb(k,l)*((a/c)**l)*factorial(m)*factorial(n)/
                      (factorial(m-l)*factorial(n-k+l)))*Hmn(m-l,n-k+l,xbar,ybar,wbar,zbar,tau)
    return(Rmnksum*(c/(2.0*f))**k)

def pImn(p,m,n,y,z,a,b,c,d,f,alpha):
    pImnsum=0.0
    if (m<0 or n<0):
        return (0.0)                                       
    for k in range(p+1):
        pImnsum+=comb(p,k)*heat(alpha/(2.0*f),1.0/(4.0*f),p-k)*Rmnk(m,n,y,z,a,b,c,d,f,alpha,k)
    return(np.sqrt(np.pi/f)*np.exp(alpha**2/(4.0*f))*pImnsum/
           (np.sqrt(np.sqrt(np.pi)*(2**n)*factorial(n))
            *np.sqrt(np.sqrt(np.pi)*(2**m)*factorial(m))))

class FluxQubitVCHOS(QubitBaseClass):
    def __init__(self, ECJ, ECg, EJ, ng1, ng2, alpha, flux, kmax, num_exc):
        self.ECJ = ECJ
        self.EJ = EJ
        self.ECg = ECg
        self.ng1 = ng1
        self.ng2 = ng2
        self.nglist = np.array([ng1, ng2])
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
        
    def potential(self, phiarray):
        """
        Flux qubit potential evaluated at `phi1` and `phi2` 
        """
        phi1 = phiarray[0]
        phi2 = phiarray[1]
        return (-self.EJ*np.cos(phi1) -self.EJ*np.cos(phi2)
                -self.EJ*self.alpha*np.cos(phi2-phi1-2.0*np.pi*self.flux))
    
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
    
    def build_gamma_matrix(self):
        """
        Return linearized potential matrix
        
        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.
        
        """
        gmat = np.zeros((2,2))
        
        global_min = self.sorted_minima()[0]
        phi1_min = global_min[0]
        phi2_min = global_min[1]
        
        gamma = self.EJ / self.Phi0**2
        
        gmat[0, 0] = gamma*np.cos(phi1_min) + self.alpha*gamma*np.cos(2*np.pi*self.flux 
                                                                      + phi1_min - phi2_min)
        gmat[1, 1] = gamma*np.cos(phi2_min) + self.alpha*gamma*np.cos(2*np.pi*self.flux 
                                                                      + phi1_min - phi2_min)
        gmat[0, 1] = gmat[1, 0] = -self.alpha*gamma*np.cos(2*np.pi*self.flux + phi1_min - phi2_min)
        
        return gmat
        
    def Xi_matrix(self):
        """Construct the Xi matrix, encoding the oscillator lengths of each dimension"""
        Cmat = self.build_capacitance_matrix()
        gmat = self.build_gamma_matrix()
        
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
               *np.matmul(self.matrix_exp(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu).T 
                                            for mu in range(2)], axis=0)/np.sqrt(2)), 
                          self.matrix_exp(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu) 
                                            for mu in range(2)], axis=0)/np.sqrt(2))))
    
    def normal_ordered_exp_i_phix_mi_phiy(self, x, y):
        """Return the normal ordered e^{i\phi_x-i\phi_y} operator, expressed using ladder ops"""
        Xi_mat = self.Xi_matrix()
        a_dag_prod = np.matmul(self.matrix_exp(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu).T 
                                                          for mu in range(2)], axis=0)/np.sqrt(2)),
                               self.matrix_exp(-1j*np.sum([Xi_mat[y,mu]*self.a_operator(mu).T 
                                                           for mu in range(2)], axis=0)/np.sqrt(2)))
        a_prod = np.matmul(self.matrix_exp(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu)
                                                      for mu in range(2)], axis=0)/np.sqrt(2)),
                           self.matrix_exp(-1j*np.sum([Xi_mat[y,mu]*self.a_operator(mu)
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
    
    def matrix_exp_old(self, matrix):
        """Perform the matrix exponentiation"""
        dim = matrix.shape[0]
        expm = np.identity(dim,dtype=np.complex128)
        prod = matrix
        num = 1
        while not np.array_equal(prod, np.zeros((dim, dim))):
            prod = matrix
            prefact = sp.special.factorial(num)**(-1)
            for m in range(1,num):
                prod = np.matmul(prod,matrix)
            expm += prefact * prod
            num += 1
        return(expm)
    
    def matrix_exp(self, matrix):
        return (sp.linalg.expm(matrix))
    
    def V_operator(self, phi):
        """Return the V operator """
        phi_delta_phi = np.matmul(phi,np.matmul(self.delta_inv_matrix(),phi))
        prefactor = np.exp(-.125 * phi_delta_phi)
        phi_Xi_inv = np.matmul(phi,np.transpose(sp.linalg.inv(self.Xi_matrix())))
        phi_Xi_inv_a = np.sum([phi_Xi_inv[mu]*self.a_operator(mu) for mu in range(2)], axis=0)
        op = self.matrix_exp((1./np.sqrt(2.))*phi_Xi_inv_a)
        return prefactor * op
    
    def kineticmat(self):
        """Return the kinetic part of the hamiltonian"""
        Xi_inv = sp.linalg.inv(self.Xi_matrix())
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        dim = self.hilbertdim()
        minima_list = self.sorted_minima()
        kinetic_mat = np.zeros((dim,dim), dtype=np.complex128)
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = -phik-(minima_m-minima_p) #XXXXXXXXXX
                       
                    V_op = self.V_operator(-delta_phi_kpm)
                    V_op_dag = self.V_operator(delta_phi_kpm).T
                    
                    num_exc_tot = (self.num_exc+1)**2
                    
                    kinetic_temp = 0.
                    
                    for mu in range(2):
                        for nu in range(2):
                            a_mu = self.a_operator(mu)
                            a_nu = self.a_operator(nu)
                            kinetic_temp += (- 0.5*4*EC_mat_t[mu, nu]*np.matmul(a_mu, a_nu)
                                             + 0.5*4*EC_mat_t[mu, nu]*np.matmul(a_mu.T, a_nu)
                                             + 0.5*4*EC_mat_t[mu, nu]*np.matmul(a_nu.T, a_mu)
                                             - 0.5*4*EC_mat_t[mu, nu]*np.matmul(a_mu.T, a_nu.T)
                                             +((2.*np.sqrt(2))**(-1) * (a_mu - a_mu.T)
                                               *4*EC_mat_t[mu, nu] * np.dot(Xi_inv[nu,:], delta_phi_kpm))
                                             +((2.*np.sqrt(2))**(-1) * np.dot(delta_phi_kpm, np.transpose(Xi_inv)[:,mu])
                                               *4*EC_mat_t[mu, nu]*(a_nu - a_nu.T))
                                             -(0.25*np.dot(delta_phi_kpm, np.transpose(Xi_inv)[:,mu])
                                               *EC_mat_t[mu, nu]*np.dot(Xi_inv[nu,:], delta_phi_kpm)*self._identity()))
                            if (mu == nu):
                                kinetic_temp += 0.5*4*EC_mat_t[mu, nu]*self._identity()
                                                
                    kinetic_temp = (np.exp(1j*np.dot(self.nglist, delta_phi_kpm))
                                    *np.matmul(V_op_dag, kinetic_temp))
                    kinetic_temp = np.matmul(kinetic_temp, V_op)
#                    if (m != p):
#                        print(kinetic_temp[-1, 0], jkvals)
                    
                    kinetic_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                p*num_exc_tot:p*num_exc_tot+num_exc_tot] += kinetic_temp
                    
                    jkvals = next(klist,-1)
                                           
        return kinetic_mat
        
    def potentialmat(self):
        """Return the potential part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        dim = self.hilbertdim()
        potential_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = phik-(minima_m-minima_p) #XXXXXXXXXXXXX
                    phibar_kpm = 0.5*(phik+(minima_m+minima_p)) #XXXXXXXXX
                        
                    exp_i_phi_0_op = np.exp(1j*phibar_kpm[0])*self.normal_ordered_exp_i_phi_operator(0)
                    exp_i_phi_1_op = np.exp(1j*phibar_kpm[1])*self.normal_ordered_exp_i_phi_operator(1)
                        
                    V_op = self.V_operator(-delta_phi_kpm)
                    V_op_dag = self.V_operator(delta_phi_kpm).T
                        
                    potential_temp = -0.5*self.EJ*(exp_i_phi_0_op+exp_i_phi_0_op.conjugate().T)
                    potential_temp += -0.5*self.EJ*(exp_i_phi_1_op+exp_i_phi_1_op.conjugate().T)
                    potential_temp += -0.5*self.alpha*self.EJ*((self.normal_ordered_exp_i_phix_mi_phiy(0, 1)
                                                                *np.exp(-1j*2.0*np.pi*self.flux)
                                                                *np.exp(1j*phibar_kpm[0])
                                                                *np.exp(-1j*phibar_kpm[1]))
                                                               +(self.normal_ordered_exp_i_phix_mi_phiy(0, 1).conjugate().T
                                                                 *np.exp(1j*2.0*np.pi*self.flux)
                                                                 *np.exp(-1j*phibar_kpm[0])
                                                                 *np.exp(1j*phibar_kpm[1])))

                    potential_temp = (np.exp(1j*np.dot(self.nglist, delta_phi_kpm))
                                      *np.matmul(V_op_dag, potential_temp))
                    potential_temp = np.matmul(potential_temp, V_op)
                    
                    num_exc_tot = (self.num_exc+1)**2
                    potential_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                  p*num_exc_tot:p*num_exc_tot+num_exc_tot] += potential_temp
                    jkvals = next(klist,-1)
                        
        return potential_mat                                                                          
                                                                          
    def hamiltonian(self):
        """Construct the Hamiltonian"""
        return (self.kineticmat() + self.potentialmat())
    
    def inner_product(self):
        """Return the inner product matrix, which is nontrivial with VCHOS states"""
        dim = self.hilbertdim()
        inner_product_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = -phik-(minima_m-minima_p) #XXXXXXXXX
                        
                    V_op = self.V_operator(-delta_phi_kpm)
                    V_op_dag = self.V_operator(delta_phi_kpm).T
                    
                    inner_temp = (np.exp(1j*np.dot(self.nglist, delta_phi_kpm))
                                  *np.matmul(V_op_dag, V_op))
#                    print(m, p, jkvals, inner_temp)
                        
                    num_exc_tot = (self.num_exc+1)**2
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
    
    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return len(self.sorted_minima())*(self.num_exc+1)**2
    
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
                wavefunc_amplitudes += np.sum([state_amplitudes[s1, s2] * np.sqrt(np.abs(np.linalg.det(Xi)))**(-1)
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
        evals = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat, 
                               eigvals_only=True, eigvals=(0, evals_count - 1))
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        inner_product_mat = self.inner_product()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat,
                                      eigvals_only=False, eigvals=(0, evals_count - 1))
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs
    
    def plot_potential(self, phi_pts=100, contour_vals=None, aspect_ratio=None, filename=None):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_pts: int, optional
            (Default value = 100)
        contour_vals: list, optional
            (Default value = None)
        aspect_ratio: float, optional
            (Default value = None)
        filename: str, optional
            (Default value = None)
        """
        x_vals = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        y_vals = np.linspace(-np.pi / 2, 3 * np.pi / 2, phi_pts)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals, aspect_ratio, filename)
    
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
        
    
    def _kinetic_mat_test_integral_helper(self, i, j, sone, stwo, soneprime, stwoprime, minima_m, minima_p, phik):
        lim = 20.0
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        detXi_inv = sp.linalg.det(Xi_inv)
        iindex = i-1
        jindex = j-1
        Hsone = hermite(sone)/np.sqrt(np.sqrt(np.pi)*factorial(sone)*2**sone)
        Hstwo = hermite(stwo)/np.sqrt(np.sqrt(np.pi)*factorial(stwo)*2**stwo)
        Hsoneprime = hermite(soneprime)/np.sqrt(np.sqrt(np.pi)*factorial(soneprime)*2**soneprime)
        Hstwoprime = hermite(stwoprime)/np.sqrt(np.sqrt(np.pi)*factorial(stwoprime)*2**stwoprime)
        Hsonem1prime = lambda x : 0.0
        Hstwom1prime = lambda x : 0.0
        Hsonem2prime = lambda x : 0.0
        Hstwom2prime = lambda x : 0.0
        zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
        zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
        zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                              + Xi_inv[0,1]*(phik[1]+minima_p[1]))
        zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                              + Xi_inv[1,1]*(phik[1]+minima_p[1]))
        if soneprime >= 1:
            Hsonem1prime = (hermite(soneprime-1)/np.sqrt(np.sqrt(np.pi)*factorial(soneprime-1)
                                                         *2**(soneprime-1)))
            if soneprime >= 2:
                Hsonem2prime = (hermite(soneprime-2)/np.sqrt(np.sqrt(np.pi)*factorial(soneprime-2)
                                                             *2**(soneprime-2)))
        if stwoprime >= 1:
            Hstwom1prime = (hermite(stwoprime-1)/np.sqrt(np.sqrt(np.pi)*factorial(stwoprime-1)
                                                         *2**(stwoprime-1)))
            if stwoprime >= 2:
                Hstwom2prime = (hermite(stwoprime-2)/np.sqrt(np.sqrt(np.pi)*factorial(stwoprime-2)
                                                             *2**(stwoprime-2)))
        result = integrate.dblquad(lambda phi1, phi2: 
                                    ((4.0 * soneprime * (soneprime - 1) 
                                     * Xi_inv[0, iindex] * Xi_inv[0, jindex] 
                                     * Hsonem2prime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                     * Hstwoprime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset))
                                     + (8.0 * soneprime * stwoprime * Xi_inv[0, iindex] * Xi_inv[1, jindex]
                                        * Hsonem1prime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                        * Hstwom1prime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset))
                                     + (4.0 * stwoprime * (stwoprime - 1) * Xi_inv[1, iindex] * Xi_inv[1, jindex] 
                                        * Hsoneprime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                        * Hstwom2prime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)))
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)**2)
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)**2)
                                    * Hsone(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset) 
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset)**2)
                                    * Hstwo(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset) 
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset)**2)
                                    
                                    +
                                    
                                    (2.0 * soneprime * Xi_inv[0, iindex]
                                     * Hsonem1prime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                     * Hstwoprime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)
                                     + 2.0 * stwoprime * Xi_inv[1, iindex]
                                     * Hsoneprime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                     * Hstwom1prime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset))
                                    * ((- Xi_inv[0, 0]*phi1 - Xi_inv[0, 1]*phi2 + zetaoneprimeoffset)*Xi_inv[0, jindex]
                                       + (- Xi_inv[1, 0]*phi1 - Xi_inv[1, 1]*phi2 + zetatwoprimeoffset)*Xi_inv[1, jindex])
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)**2)
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)**2)
                                    * Hsone(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset) 
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset)**2)
                                    * Hstwo(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset) 
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset)**2)
                                    
                                    +
                                    
                                    (2.0 * soneprime * Xi_inv[0, jindex]
                                     * Hsonem1prime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                     * Hstwoprime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)
                                     + 2.0 * stwoprime * Xi_inv[1, jindex]
                                     * Hsoneprime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                     * Hstwom1prime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset))
                                    * ((- Xi_inv[0, 0]*phi1 - Xi_inv[0, 1]*phi2 + zetaoneprimeoffset)*Xi_inv[0, iindex]
                                       + (- Xi_inv[1, 0]*phi1 - Xi_inv[1, 1]*phi2 + zetatwoprimeoffset)*Xi_inv[1, iindex])
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)**2)
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)**2)
                                    * Hsone(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset) 
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset)**2)
                                    * Hstwo(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset) 
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset)**2)
                                    
                                    +
                                    
                                    Hsoneprime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)
                                    * Hstwoprime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)
                                    * (- Xi_inv[0, iindex]*Xi_inv[0, jindex] - Xi_inv[1, iindex]*Xi_inv[1, jindex] 
                                       + ((- Xi_inv[0, 0]*phi1 - Xi_inv[0, 1]*phi2 + zetaoneprimeoffset)*Xi_inv[0, iindex]
                                          + (- Xi_inv[1, 0]*phi1 - Xi_inv[1, 1]*phi2 + zetatwoprimeoffset)*Xi_inv[1, iindex])
                                       * ((- Xi_inv[0, 0]*phi1 - Xi_inv[0, 1]*phi2 + zetaoneprimeoffset)*Xi_inv[0, jindex]
                                          + (- Xi_inv[1, 0]*phi1 - Xi_inv[1, 1]*phi2 + zetatwoprimeoffset)*Xi_inv[1, jindex]))
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneprimeoffset)**2)
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwoprimeoffset)**2)
                                    * Hsone(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset) 
                                    * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 - zetaoneoffset)**2)
                                    * Hstwo(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset) 
                                    * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 - zetatwooffset)**2),
                                    -lim, lim, lambda phi1 : -lim, lambda phi2 : lim)
        return (result[0]/detXi_inv)
    
    
    
    def _kinetic_mat_test(self):
        dim = self.hilbertdim()
        kinetic_test_mat = np.zeros((dim,dim))
        minima_list = self.sorted_minima()
        EC_mat = self.build_EC_matrix()
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    kinetic_temp = ((- 4.0*EC_mat[0, 1] - 4.0*EC_mat[1, 0])
                                                     * self._kinetic_mat_test_integral_helper(1, 2, sone, stwo, 
                                                                                              soneprime, stwoprime, 
                                                                                              minima_m, minima_p,
                                                                                              phik)
                                                    - (4.0*EC_mat[0, 0] 
                                                       * self._kinetic_mat_test_integral_helper(1, 1, sone, stwo, 
                                                                                              soneprime, stwoprime, 
                                                                                              minima_m, minima_p,
                                                                                              phik))
                                                    - (4.0*EC_mat[1, 1] 
                                                       * self._kinetic_mat_test_integral_helper(2, 2, sone, stwo, 
                                                                                              soneprime, stwoprime, 
                                                                                              minima_m, minima_p,
                                                                                              phik)))
                                    print(sone, stwo, soneprime, stwoprime, kinetic_temp, jkvals)
                                    i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                    j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                    kinetic_test_mat[i, j] += kinetic_temp
                                    jkvals = next(klist,-1)
        return(kinetic_test_mat)       
    
        
    def _identity_mat_test_babusci(self):
        dim = self.hilbertdim()
        identity_test_mat = np.zeros((dim,dim))
        minima_list = self.sorted_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                matelem = 0.
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
                                    zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
                                    zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[0,1]*(phik[1]+minima_p[1]))
                                    zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[1,1]*(phik[1]+minima_p[1]))
                                    matelem += (np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, b=-2*zetatwooffset,
                                                       c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                       alpha=zetatwooffset+zetatwoprimeoffset)
                                                * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, 
                                                      b=-2*zetaoneoffset, c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                      alpha=zetaoneoffset+zetaoneprimeoffset)
                                                * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    jkvals = next(klist, -1)
                                i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                identity_test_mat[i, j] += matelem
        return(identity_test_mat)
    
    def _potential_mat_test_babusci(self):
        dim = self.hilbertdim()
        potential_test_mat = np.zeros((dim,dim), dtype=np.complex_)
        minima_list = self.sorted_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                matelem = 0.0
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
                                    zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
                                    zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[0,1]*(phik[1]+minima_p[1]))
                                    zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[1,1]*(phik[1]+minima_p[1]))
                                    
                                    potential1pos = -0.5*self.EJ*(np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                                  * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetatwooffset,c=2, 
                                                                         d=-2*zetatwoprimeoffset, f=1, 
                                                                         alpha=(zetatwooffset+zetatwoprimeoffset
                                                                                + 1j*Xi[0, 1]))
                                                                  * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetaoneoffset, c=2, 
                                                                         d=-2*zetaoneprimeoffset, f=1, 
                                                                         alpha=(zetaoneoffset+zetaoneprimeoffset
                                                                               + 1j*Xi[0, 0]))
                                                                  * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    potential1neg = -0.5*self.EJ*(np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                                  * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetatwooffset,c=2, 
                                                                         d=-2*zetatwoprimeoffset, f=1, 
                                                                         alpha=(zetatwooffset+zetatwoprimeoffset
                                                                               - 1j*Xi[0, 1]))
                                                                  * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetaoneoffset, c=2, 
                                                                         d=-2*zetaoneprimeoffset, f=1, 
                                                                         alpha=(zetaoneoffset+zetaoneprimeoffset
                                                                               - 1j*Xi[0, 0]))
                                                                  * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    potential2pos = -0.5*self.EJ*(np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                                  * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetatwooffset,c=2, 
                                                                         d=-2*zetatwoprimeoffset, f=1, 
                                                                         alpha=(zetatwooffset+zetatwoprimeoffset
                                                                               + 1j*Xi[1, 1]))
                                                                  * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetaoneoffset, c=2, 
                                                                         d=-2*zetaoneprimeoffset, f=1, 
                                                                         alpha=(zetaoneoffset+zetaoneprimeoffset
                                                                               + 1j*Xi[1, 0]))
                                                                  * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    potential2neg = -0.5*self.EJ*(np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                                  * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetatwooffset,c=2, 
                                                                         d=-2*zetatwoprimeoffset, f=1, 
                                                                         alpha=(zetatwooffset+zetatwoprimeoffset
                                                                               - 1j*Xi[1, 1]))
                                                                  * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, 
                                                                         a=2, b=-2*zetaoneoffset, c=2, 
                                                                         d=-2*zetaoneprimeoffset, f=1, 
                                                                         alpha=(zetaoneoffset+zetaoneprimeoffset
                                                                               - 1j*Xi[1, 0]))
                                                                  * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    potential3pos = -(0.5*self.alpha*self.EJ*np.exp(1j*2.0*np.pi*self.flux)
                                                      * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                      * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, 
                                                             a=2, b=-2*zetatwooffset,c=2, 
                                                             d=-2*zetatwoprimeoffset, f=1, 
                                                             alpha=(zetatwooffset+zetatwoprimeoffset
                                                                    + 1j*(Xi[1, 1] - Xi[0, 1])))
                                                      * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, 
                                                             a=2, b=-2*zetaoneoffset, c=2, 
                                                             d=-2*zetaoneprimeoffset, f=1, 
                                                             alpha=(zetaoneoffset+zetaoneprimeoffset
                                                                    + 1j*(Xi[1, 0] - Xi[0, 0])))
                                                      * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    potential3neg = -(0.5*self.alpha*self.EJ*np.exp(-1j*2.0*np.pi*self.flux)
                                                      * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                      * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, 
                                                             a=2, b=-2*zetatwooffset,c=2, 
                                                             d=-2*zetatwoprimeoffset, f=1, 
                                                             alpha=(zetatwooffset+zetatwoprimeoffset
                                                                    - 1j*(Xi[1, 1] - Xi[0, 1])))
                                                      * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, 
                                                             a=2, b=-2*zetaoneoffset, c=2, 
                                                             d=-2*zetaoneprimeoffset, f=1, 
                                                             alpha=(zetaoneoffset+zetaoneprimeoffset
                                                                    - 1j*(Xi[1, 0] - Xi[0, 0])))
                                                      * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    matelem += (potential1pos + potential1neg + potential2pos
                                               + potential2neg + potential3pos + potential3neg)
                                    
                                    jkvals = next(klist, -1)
                                i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                potential_test_mat[i, j] += matelem
        return(potential_test_mat)
        
    def _kinetic_mat_test_babusci(self):
        dim = self.hilbertdim()
        kinetic_test_mat = np.zeros((dim,dim))
        minima_list = self.sorted_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                matelem = 0.0
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
                                    zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
                                    zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[0,1]*(phik[1]+minima_p[1]))
                                    zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[1,1]*(phik[1]+minima_p[1]))
                                    
                                    elem11 = (4.0*EC_mat_t[0, 0]
                                                * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, b=-2*zetatwooffset,
                                                       c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                       alpha=zetatwooffset+zetatwoprimeoffset)
                                                * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, 
                                                      b=-2*zetaoneoffset, c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                      alpha=zetaoneoffset+zetaoneprimeoffset)
                                                * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                                                        
                                    elem12 = -(4.0*EC_mat_t[0, 0]
                                                 * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                 * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, b=-2*zetatwooffset,
                                                        c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                        alpha=zetatwooffset+zetatwoprimeoffset)
                                                 * pImn(p=2, m=sone, n=soneprime, y=-1, z=-1, a=2, 
                                                        b=-2*(zetaoneoffset-zetaoneprimeoffset), c=2, 
                                                        d=0, f=1, alpha=zetaoneoffset-zetaoneprimeoffset)
                                                 * np.exp(-0.5*(zetaoneprimeoffset-zetaoneoffset)**2))
                                                                        
                                    elem13 = elem14 = 0.
                                    if soneprime >= 1:
                                        elem13 += -((4.0*EC_mat_t[0, 0]/(np.sqrt(soneprime*2)))
                                                     * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                     * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, b=-2*zetatwooffset,
                                                           c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                           alpha=zetatwooffset+zetatwoprimeoffset)
                                                     * 4*zetaoneprimeoffset*soneprime
                                                     * pImn(p=0, m=sone, n=soneprime-1, y=-1, z=-1, a=2, 
                                                           b=-2*zetaoneoffset, c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                           alpha=zetaoneoffset+zetaoneprimeoffset)
                                                     * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                        
                                        elem14 += ((4.0*EC_mat_t[0, 0]/(np.sqrt(soneprime*2)))
                                                    * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                    * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, b=-2*zetatwooffset,
                                                           c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                           alpha=zetatwooffset+zetatwoprimeoffset)
                                                    * 4*soneprime
                                                    * pImn(p=1, m=sone, n=soneprime-1, y=-1, z=-1, a=2, 
                                                          b=-2*zetaoneoffset, c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                          alpha=zetaoneoffset+zetaoneprimeoffset)
                                                    * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                    
                                    elem15 = 0.
                                    if soneprime >= 2:
                                        elem15 += (-(4.0*EC_mat_t[0, 0]/(np.sqrt(soneprime*(soneprime-1)*2*2)))
                                                   * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2))
                                                   * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, b=-2*zetatwooffset,
                                                       c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                        alpha=zetatwooffset+zetatwoprimeoffset)
                                                   * 4*soneprime*(soneprime - 1)
                                                   * pImn(p=0, m=sone, n=soneprime-2, y=-1, z=-1, a=2, 
                                                          b=-2*zetaoneoffset, c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                          alpha=zetaoneoffset+zetaoneprimeoffset)
                                                   * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2)))
                                        
                                    #########
                                    
                                    elem21 = (4.0*EC_mat_t[1, 1]
                                                * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2))
                                                * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, b=-2*zetaoneoffset,
                                                       c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                       alpha=zetaoneoffset+zetaoneprimeoffset)
                                                * pImn(p=0, m=stwo, n=stwoprime, y=-1, z=-1, a=2, 
                                                      b=-2*zetatwooffset, c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                      alpha=zetatwooffset+zetatwoprimeoffset)
                                                * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2)))
                                    
                                    elem22 = -(4.0*EC_mat_t[1, 1]
                                                 * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2))
                                                 * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, b=-2*zetaoneoffset,
                                                        c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                        alpha=zetaoneoffset+zetaoneprimeoffset)
                                                 * pImn(p=2, m=stwo, n=stwoprime, y=-1, z=-1, a=2, 
                                                        b=-2*(zetatwooffset-zetatwoprimeoffset), c=2, 
                                                        d=0, f=1, alpha=zetatwooffset-zetatwoprimeoffset)
                                                 * np.exp(-0.5*(zetatwoprimeoffset-zetatwooffset)**2))
                                    
                                    elem23 = elem24 = 0.
                                    if stwoprime >= 1:
                                        elem23 += -((4.0*EC_mat_t[1, 1]/(np.sqrt(stwoprime*2)))
                                                     * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2))
                                                     * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, b=-2*zetaoneoffset,
                                                            c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                            alpha=zetaoneoffset+zetaoneprimeoffset)
                                                     * 4*zetatwoprimeoffset*stwoprime
                                                     * pImn(p=0, m=stwo, n=stwoprime-1, y=-1, z=-1, a=2, 
                                                           b=-2*zetatwooffset, c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                           alpha=zetatwooffset+zetatwoprimeoffset)
                                                     * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2)))
                                        
                                        elem24 += ((4.0*EC_mat_t[1, 1]/(np.sqrt(stwoprime*2)))
                                                    * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2))
                                                    * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, b=-2*zetaoneoffset,
                                                           c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                           alpha=zetaoneoffset+zetaoneprimeoffset)
                                                    * 4*stwoprime
                                                    * pImn(p=1, m=stwo, n=stwoprime-1, y=-1, z=-1, a=2, 
                                                          b=-2*zetatwooffset, c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                          alpha=zetatwooffset+zetatwoprimeoffset)
                                                    * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2)))
                                    
                                    elem25 = 0.
                                    if stwoprime >= 2:
                                        elem25 += (-(4.0*EC_mat_t[1, 1]/(np.sqrt(stwoprime*(stwoprime-1)*2*2)))
                                                   * np.exp(-0.5*(zetaoneoffset**2 + zetaoneprimeoffset**2))
                                                   * pImn(p=0, m=sone, n=soneprime, y=-1, z=-1, a=2, b=-2*zetaoneoffset,
                                                          c=2, d=-2*zetaoneprimeoffset, f=1, 
                                                          alpha=zetaoneoffset+zetaoneprimeoffset)
                                                   * 4*stwoprime*(stwoprime - 1)
                                                   * pImn(p=0, m=stwo, n=stwoprime-2, y=-1, z=-1, a=2, 
                                                          b=-2*zetatwooffset, c=2, d=-2*zetatwoprimeoffset, f=1, 
                                                          alpha=zetatwooffset+zetatwoprimeoffset)
                                                   * np.exp(-0.5*(zetatwooffset**2 + zetatwoprimeoffset**2)))
                                        
                                    matelem += (elem11 + elem12 + elem13 + elem14 + elem15
                                                + elem21 + elem22 + elem23 + elem24 + elem25)
                                    i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                    j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                    if ((i==17) and (j==0)):
                                        print(elem11, elem12, elem13, elem14, elem15,
                                              elem21, elem22, elem23, elem24, elem25)
                                        print(matelem, "jkvals = ", jkvals)
                                    jkvals = next(klist, -1)
                                i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                kinetic_test_mat[i, j] += matelem
        return(kinetic_test_mat)
                                
        
    def _kinetic_mat_test_zeta_coords(self):
        dim = self.hilbertdim()
        kinetic_test_mat = np.zeros((dim,dim))
        minima_list = self.sorted_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    Hsone = hermite(sone)/np.sqrt(np.sqrt(np.pi)
                                                                  *factorial(sone)*2**sone)
                                    Hstwo = hermite(stwo)/np.sqrt(np.sqrt(np.pi)
                                                                  *factorial(stwo)*2**stwo)
                                    Hsoneprime = hermite(soneprime)/np.sqrt(np.sqrt(np.pi)
                                                                            *factorial(soneprime)*2**soneprime)
                                    Hstwoprime = hermite(stwoprime)/np.sqrt(np.sqrt(np.pi)
                                                                            *factorial(stwoprime)*2**stwoprime)
                                    Hsonem1prime = lambda x : 0.0
                                    Hstwom1prime = lambda x : 0.0
                                    Hsonem2prime = lambda x : 0.0
                                    Hstwom2prime = lambda x : 0.0
                                    zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
                                    zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
                                    zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[0,1]*(phik[1]+minima_p[1]))
                                    zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[1,1]*(phik[1]+minima_p[1]))
                                    if soneprime >= 1:
                                        Hsonem1prime = (hermite(soneprime-1)/np.sqrt(np.sqrt(np.pi)*factorial(soneprime-1)
                                                                                     *2**(soneprime-1)))
                                        if soneprime >= 2:
                                            Hsonem2prime = (hermite(soneprime-2)/np.sqrt(np.sqrt(np.pi)*factorial(soneprime-2)
                                                                                         *2**(soneprime-2)))
                                    if stwoprime >= 1:
                                        Hstwom1prime = (hermite(stwoprime-1)/np.sqrt(np.sqrt(np.pi)*factorial(stwoprime-1)
                                                                                     *2**(stwoprime-1)))
                                        if stwoprime >= 2:
                                            Hstwom2prime = (hermite(stwoprime-2)/np.sqrt(np.sqrt(np.pi)*factorial(stwoprime-2)
                                                                                         *2**(stwoprime-2)))
                                    kinetic_temp = integrate.dblquad(lambda zetaone, zetatwo: 
                                                                (Hsone(zetaone-zetaoneoffset) 
                                                                 * np.exp(-0.5*(zetaone - zetaoneoffset)**2)
                                                                 * Hstwo(zetatwo-zetatwooffset) 
                                                                 * np.exp(-0.5*(zetatwo - zetatwooffset)**2)
                                                                 * (-4.0 * EC_mat_t[0, 0]
                                                                   * Hstwoprime(zetatwo - zetatwoprimeoffset)
                                                                   * np.exp(-0.5*(zetatwo - zetatwoprimeoffset)**2)
                                                                   * np.exp(-0.5*(zetaone - zetaoneprimeoffset)**2))
                                                                * ((4.0 * soneprime * (soneprime - 1)
                                                                    * Hsonem2prime(zetaone - zetaoneprimeoffset))
                                                                   - (4.0 * soneprime 
                                                                      * Hsonem1prime(zetaone - zetaoneprimeoffset)
                                                                      * (zetaone - zetaoneprimeoffset))
                                                                   + ((zetaone - zetaoneprimeoffset)**2
                                                                      * Hsoneprime(zetaone - zetaoneprimeoffset))
                                                                   - Hsoneprime(zetaone - zetaoneprimeoffset)))
                                                                
                                                                +
                                                                
                                                                (Hsone(zetaone-zetaoneoffset) 
                                                                 * np.exp(-0.5*(zetaone - zetaoneoffset)**2)
                                                                 * Hstwo(zetatwo-zetatwooffset) 
                                                                 * np.exp(-0.5*(zetatwo - zetatwooffset)**2)
                                                                 * (-4.0 * EC_mat_t[1, 1]
                                                                   * Hsoneprime(zetaone - zetaoneprimeoffset)
                                                                   * np.exp(-0.5*(zetaone - zetaoneprimeoffset)**2)
                                                                   * np.exp(-0.5*(zetatwo - zetatwoprimeoffset)**2))
                                                                * ((4.0 * stwoprime * (stwoprime - 1)
                                                                    * Hstwom2prime(zetatwo - zetatwoprimeoffset))
                                                                   - (4.0 * stwoprime 
                                                                      * Hstwom1prime(zetatwo - zetatwoprimeoffset)
                                                                      * (zetatwo - zetatwoprimeoffset))
                                                                   + ((zetatwo - zetatwoprimeoffset)**2
                                                                      * Hstwoprime(zetatwo - zetatwoprimeoffset))
                                                                   - Hstwoprime(zetatwo - zetatwoprimeoffset))), 
                                                                -np.inf, np.inf, lambda zetaone : -np.inf, 
                                                                lambda zetaone : np.inf)
                                    print(sone, stwo, soneprime, stwoprime, kinetic_temp[0], jkvals)
                                    i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                    j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                    kinetic_test_mat[i, j] += kinetic_temp[0]
                                    jkvals = next(klist,-1)
        return(kinetic_test_mat)
    
    def _potential_mat_test_phi_coords(self):
        dim = self.hilbertdim()
        potential_test_mat = np.zeros((dim,dim))
        minima_list = self.sorted_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        EC_mat_t = np.matmul(Xi_inv,np.matmul(EC_mat,np.transpose(Xi_inv)))
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    Hsone = hermite(sone)/np.sqrt(np.sqrt(np.pi)
                                                                  *factorial(sone)*2**sone)
                                    Hstwo = hermite(stwo)/np.sqrt(np.sqrt(np.pi)
                                                                  *factorial(stwo)*2**stwo)
                                    Hsoneprime = hermite(soneprime)/np.sqrt(np.sqrt(np.pi)
                                                                            *factorial(soneprime)*2**soneprime)
                                    Hstwoprime = hermite(stwoprime)/np.sqrt(np.sqrt(np.pi)
                                                                            *factorial(stwoprime)*2**stwoprime)
                                    zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
                                    zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
                                    zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[0,1]*(phik[1]+minima_p[1]))
                                    zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[1,1]*(phik[1]+minima_p[1]))
                                    
                                    matelem = integrate.dblquad(lambda phi1, phi2 :
                                                                Hsone(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 
                                                                      - zetaoneoffset) 
                                                                * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 
                                                                               - zetaoneoffset)**2)
                                                                * Hstwo(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 
                                                                        - zetatwooffset) 
                                                                * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 
                                                                               - zetatwooffset)**2)
                                                                * (-self.EJ*np.cos(phi1) -self.EJ*np.cos(phi2)
                                                                   -self.EJ*self.alpha*np.cos(phi2-phi1-2.0*np.pi*self.flux))
                                                                * Hsoneprime(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 
                                                                             - zetaoneprimeoffset) 
                                                                * np.exp(-0.5*(Xi_inv[0, 0]*phi1 + Xi_inv[0, 1]*phi2 
                                                                               - zetaoneprimeoffset)**2)
                                                                * Hstwoprime(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 
                                                                        - zetatwoprimeoffset) 
                                                                * np.exp(-0.5*(Xi_inv[1, 0]*phi1 + Xi_inv[1, 1]*phi2 
                                                                               - zetatwoprimeoffset)**2)
                                                                * np.abs(1./sp.linalg.det(Xi)), -np.inf, np.inf, 
                                                                lambda zetaone : -np.inf, 
                                                                lambda zetaone : np.inf
                                                               )
                                    print(sone, stwo, soneprime, stwoprime, matelem[0], jkvals)
                                    i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                    j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                    potential_test_mat[i, j] += matelem[0]
                                    
                                    jkvals = next(klist,-1)
        return(potential_test_mat)
    
    def _inner_product_test(self):
        dim = self.hilbertdim()
        inner_product_test_mat = np.zeros((dim,dim))
        minima_list = self.sorted_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                for sone in range(self.num_exc+1):
                    for stwo in range(self.num_exc+1):
                        for soneprime in range(self.num_exc+1):
                            for stwoprime in range(self.num_exc+1):
                                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                                jkvals = next(klist,-1)
                                while jkvals != -1:
                                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                                    Hsone = hermite(sone)/np.sqrt(np.sqrt(np.pi)*factorial(sone)*2**sone)
                                    Hstwo = hermite(stwo)/np.sqrt(np.sqrt(np.pi)*factorial(stwo)*2**stwo)
                                    Hsoneprime = hermite(soneprime)/np.sqrt(np.sqrt(np.pi)*factorial(soneprime)*2**soneprime)
                                    Hstwoprime = hermite(stwoprime)/np.sqrt(np.sqrt(np.pi)*factorial(stwoprime)*2**stwoprime)
                                    zetaoneoffset = Xi_inv[0,0]*minima_m[0]+Xi_inv[0,1]*minima_m[1]
                                    zetatwooffset = Xi_inv[1,0]*minima_m[0]+Xi_inv[1,1]*minima_m[1]
                                    zetaoneprimeoffset = (Xi_inv[0,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[0,1]*(phik[1]+minima_p[1]))
                                    zetatwoprimeoffset = (Xi_inv[1,0]*(phik[0]+minima_p[0])
                                                          + Xi_inv[1,1]*(phik[1]+minima_p[1]))
                                    
                                    matelem = integrate.dblquad(lambda zetaone, zetatwo: Hsone(zetaone-zetaoneoffset) 
                                                                * np.exp(-0.5*(zetaone-zetaoneoffset)**2)
                                                                * Hsoneprime(zetaone-zetaoneprimeoffset) 
                                                                * np.exp(-0.5*(zetaone-zetaoneprimeoffset)**2)
                                                                * Hstwo(zetatwo-zetatwooffset) 
                                                                * np.exp(-0.5*(zetatwo-zetatwooffset)**2)
                                                                * Hstwoprime(zetatwo-zetatwoprimeoffset)
                                                                * np.exp(-0.5*(zetatwo-zetatwoprimeoffset)**2), 
                                                                -np.inf, np.inf, 
                                                                lambda zetaone : -np.inf, 
                                                                lambda zetaone : np.inf)
                                    print(sone, stwo, soneprime, stwoprime, matelem[0], jkvals)
                                    i = (self.num_exc+1)*(sone)+stwo+m*(self.num_exc+1)**2
                                    j = (self.num_exc+1)*(soneprime)+stwoprime+p*(self.num_exc+1)**2
                                    inner_product_test_mat[i, j] += matelem[0]
                                    jkvals = next(klist,-1)
        return(inner_product_test_mat)
                            
        
        
        
        
        
        

 