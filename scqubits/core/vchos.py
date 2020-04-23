import numpy as np
import scipy as sp
import itertools
from scipy.optimize import minimize
import scipy.constants as const
from scipy.special import hermite

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core.discretization import GridSpec, Grid1d
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem


#-Flux Qubit using VCHOS 

class VCHOS(QubitBaseClass):
    def __init__(self):
        pass
    
    def potential(self, phiarray):
        """
        Potential evaluated at the location specified by phiarray
        """
        pot_sum = np.sum([-self.EJlist[j]*np.cos(phiarray[j])
                          for j in range(self.num_deg_freedom)])
        pot_sum += (-self.EJlist[-1]
                    *np.cos(np.sum([self.boundary_coeffs[i]*phiarray[i]
                                    for i in range(self.num_deg_freedom)])
                            +2*np.pi*self.flux))
        return pot_sum
    
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
    
    def build_capacitance_matrix(self):
        """The capacitance matrix will be model specific """
        pass
    
    def build_gamma_matrix(self, i):
        """
        Return linearized potential matrix
        
        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.
        
        We are imagining an arbitrary loop of JJs where we have 
        changed variables to the difference variables, so that
        each junction is a function of just one variable, except for
        the last junction, which is a function of all of the variables
        
        """
        gmat = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
        
        min_loc = self.sorted_minima()[i]
        gamma_list = self.EJlist / self.Phi0**2
        
        gamma_diag = np.diag([gamma_list[j]*np.cos(min_loc[j]) 
                              for j in range(self.num_deg_freedom)])
        gmat += gamma_diag
        
        min_loc_bound_sum = np.sum([self.boundary_coeffs[i]*min_loc[i]
                                    for i in range(self.num_deg_freedom)])
        for j in range(self.num_deg_freedom):
            for k in range(self.num_deg_freedom):
                gmat[j, k] += (gamma_list[-1]*self.boundary_coeffs[j]*self.boundary_coeffs[k]
                               *np.cos(min_loc_bound_sum+2*np.pi*self.flux))
        
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
        
    def _build_U_squeezing_operator(self, i, Xi):
        freq, uvmat = self._squeezing_M_builder(i, Xi)
        uvmat = uvmat.T
        dim = uvmat.shape[0]
        u = uvmat[0 : int(dim/2), 0 : int(dim/2)]
        v = uvmat[int(dim/2) : dim, 0 : int(dim/2)]
        u_inv = sp.linalg.inv(u)
        rho = np.matmul(u_inv, v)
        sigma = sp.linalg.logm(u)
        tau = np.matmul(v, u_inv)
        return rho, sigma, tau

    def _define_squeezing_variables(self, rho, rhoprime, Xi):
        Xi_inv = sp.linalg.inv(Xi)
        deltarhoprime = np.matmul(sp.linalg.inv(np.eye(self.num_deg_freedom)
                                                -np.matmul(rhoprime, rho)), rhoprime)
        deltarho = np.matmul(sp.linalg.inv(np.eye(self.num_deg_freedom)
                                           -np.matmul(rho, rhoprime)), rho)
        deltarhobar = sp.linalg.logm(sp.linalg.inv(np.eye(self.num_deg_freedom)-np.matmul(rhoprime, rho)))
        z = 1j*np.transpose(Xi_inv)/np.sqrt(2.)
        zp = (z+0.5*np.matmul(np.matmul(z, rhoprime), deltarho+deltarho.T)
              +0.5*np.matmul(z, deltarho+deltarho.T))
        zpp = np.matmul(z, rhoprime) + z
        return deltarho, deltarhoprime, deltarhobar, zp, zpp
    
    def test_squeezing(self, i):
        freq, uvmat = self.squeezing_M_builder(i)
        freq = np.array([2*freq[i] for i in range(len(freq)) if freq[i] > 0])
        dim = uvmat.shape[0]
        u = uvmat[0 : int(dim/2), 0 : int(dim/2)]
        v = uvmat[0 : int(dim/2), int(dim/2) : dim]
        c_0 = np.sum([u[0, i]*self.a_operator(i)+v[0, i]*self.a_operator(i).T
                      for i in range(self.num_deg_freedom)], axis=0)
        c_1 = np.sum([u[1, i]*self.a_operator(i)+v[1, i]*self.a_operator(i).T
                      for i in range(self.num_deg_freedom)], axis=0)
        H_new = np.real((freq[0]*(np.matmul(c_0.T, c_0)+0.5*self._identity())
                + freq[1]*(np.matmul(c_1.T, c_1)+0.5*self._identity())))
        
        Xi = self.Xi_matrix()
        gamma = self.build_gamma_matrix(i)
        gamma_prime = np.matmul(np.transpose(Xi), np.matmul(gamma, Xi))
        zeta = 0.25*(self.Phi0**2 * gamma_prime + self.omegamat())
        eta = 0.25*(self.Phi0**2 * gamma_prime - self.omegamat())
        H_old = np.sum([2*zeta[mu, nu]*np.matmul(self.a_operator(mu).T, self.a_operator(nu))
                        +eta[mu, nu]*np.matmul(self.a_operator(mu).T, self.a_operator(nu).T)
                        +eta[mu, nu]*np.matmul(self.a_operator(mu), self.a_operator(nu))
                        for mu in range(self.num_deg_freedom) 
                        for nu in range(self.num_deg_freedom)], axis=0)
        H_old += np.sum([zeta[mu, mu]*np.eye(self.hilbertdim()) 
                         for mu in range(self.num_deg_freedom)], axis=0)
        print(H_new[0:5, 0:5])
        print(H_old[0:5, 0:5])
        return 0
    
    def _squeezing_M_builder(self, i, Xi):
        dim = Xi.shape[0]
        gamma = self.build_gamma_matrix(i)
        gamma_prime = np.matmul(Xi.T, np.matmul(gamma, Xi))
        omegamat = self.omegamat()
        zeta = 0.25*(self.Phi0**2 * gamma_prime + omegamat)
        eta = 0.25*(self.Phi0**2 * gamma_prime - omegamat)
        hmat = np.block([[zeta, -eta],
                         [eta, -zeta]])
        K = np.block([[np.eye(dim), np.zeros((dim, dim))], 
                      [np.zeros((dim, dim)), -np.eye(dim)]])
        eigvals, eigvec = sp.linalg.eig(hmat)
        eigvals, eigvec = self._order_eigensystem_squeezing(np.real(eigvals), eigvec)
        eigvec = eigvec.T #since eigvec represents M.T
        dim = eigvec.shape[0]
        u = eigvec[0 : int(dim/2), 0 : int(dim/2)]
        v = eigvec[int(dim/2) : dim, 0 : int(dim/2)]
        eigvals, eigvec = self._normalize_symplectic_eigensystem_squeezing(eigvals, eigvec)
        assert(np.allclose(np.matmul(eigvec.T, np.matmul(K, eigvec)), K))
        return (eigvals, eigvec)
    
    def _order_eigensystem_squeezing(self, eigvals, eigvec):
        """Order eigensystem to have positive eigenvalues followed by negative, in same order"""
        eigval_holder = []
        eigvec_holder = []
        for k, eigval in enumerate(eigvals):
            if eigval > 0:
                eigval_holder.append(eigval)
                eigvec_holder.append(eigvec[:, k])
        eigval_result = np.copy(eigval_holder).tolist()
        eigvec_result = np.copy(eigvec_holder).tolist()
        for k, eigval in enumerate(eigval_holder):
            index = np.argwhere(np.isclose(eigvals, -1.0*eigval))[0, 0]
            eigval_result.append(eigvals[index])
            eigvec_result.append((eigvec[:, index]).tolist())
        return(eigval_result, np.array(eigvec_result))
    
    def _normalize_symplectic_eigensystem_squeezing(self, eigvals, eigvec):
        dim = eigvec.shape[0]
        dim2 = int(dim/2)
        for col in range(dim2):
            a = np.sum([eigvec[row, col] for row in range(dim)])
            if a < 0.0:
                eigvec[:, col] *= -1
        A = eigvec[0 : dim2, 0 : dim2]
        B = eigvec[dim2 : dim, 0 : dim2]
        for vec in range(dim2):
            a = 1./np.sqrt(np.sum([A[num, vec]*A[num, vec] - B[num, vec]*B[num, vec] 
                                   for num in range(dim2)]))
            eigvec[:, vec] *= a
        A = eigvec[0 : dim2, 0 : dim2]
        B = eigvec[dim2 : dim, 0 : dim2]
        eigvec[dim2 : dim, dim2 : dim] = A
        eigvec[0 : dim2, dim2 : dim] = B
        return (eigvals, eigvec)
    
    def _normal_ordered_adag_a_exponential(self, x):
        """Expectation is that exp(a_{i}^{\dagger}x_{ij}a_{j}) needs to be normal ordered"""
        expx = sp.linalg.expm(x)
        result = self._identity()
        dim = result.shape[0]
        additionalterm = np.eye(dim)
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        k = 1
        while not np.allclose(additionalterm, np.zeros((dim, dim))):
            additionalterm = np.sum([((expx-np.eye(self.num_deg_freedom))[i, j])**(k)
                                     *(sp.special.factorial(k))**(-1)
                                     *np.matmul(np.linalg.matrix_power(a_op_list[i].T, k), 
                                                np.linalg.matrix_power(a_op_list[j], k))
                                     for i in range(self.num_deg_freedom) 
                                     for j in range(self.num_deg_freedom)], axis=0)
            result += additionalterm
            k += 1
        return result
    
    def _build_squeezing_ops(self, m, p, minima_diff, Xi, a_op_list):
        if self.squeezing:
            if m == 0: #At the global minimum, no squeezing required
                rho = np.zeros((self.num_deg_freedom, self.num_deg_freedom)) 
                sigma = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
                tau = np.zeros((self.num_deg_freedom, self.num_deg_freedom)) 
            else:
                rho, sigma, tau = self._build_U_squeezing_operator(m, Xi)
            if p == 0:
                rhoprime = np.zeros((self.num_deg_freedom, self.num_deg_freedom)) # 2 d.o.f.
                sigmaprime = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
                tauprime = np.zeros((self.num_deg_freedom, self.num_deg_freedom)) 
            elif p == m:
                rhoprime = np.copy(rho)
                sigmaprime = np.copy(sigma)
                tauprime = np.copy(tau)
            else:
                rhoprime, sigmaprime, tauprime = self._build_U_squeezing_operator(p, Xi)
                
            deltarho, deltarhoprime, deltarhobar, zp, zpp = self._define_squeezing_variables(rho, rhoprime, Xi)
        
            expsigma = sp.linalg.expm(-sigma)
            expsigmaprime = sp.linalg.expm(-sigmaprime)
            expdeltarhobar = sp.linalg.expm(deltarhobar)
            expdrbs = np.matmul(expdeltarhobar.T, expsigma)
                    
            prefactor_adag_adag = 0.5*(tau.T-np.matmul(expsigma.T, np.matmul(deltarhoprime, expsigma)))
            prefactor_a_a = 0.5*(tauprime-np.matmul(expsigmaprime.T, np.matmul(deltarho, expsigmaprime)))
            prefactor_adag_a = sp.linalg.logm(np.matmul(expsigma.T, np.matmul(expdeltarhobar, expsigmaprime)))
                    
            exp_adag_adag = sp.linalg.expm(np.sum([prefactor_adag_adag[i, j]
                                                   *np.matmul(a_op_list[i].T, 
                                                              a_op_list[j].T)
                                                   for i in range(self.num_deg_freedom) 
                                                   for j in range(self.num_deg_freedom)], axis=0))
            exp_a_a = sp.linalg.expm(np.sum([prefactor_a_a[i, j]
                                             *np.matmul(a_op_list[i], 
                                                        a_op_list[j])
                                             for i in range(self.num_deg_freedom) 
                                             for j in range(self.num_deg_freedom)], axis=0))
            exp_adag_a = self._normal_ordered_adag_a_exponential(prefactor_adag_a)
        else:
            N = self.num_deg_freedom
            rho, sigma, tau = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            rhoprime, sigmaprime, tauprime = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            deltarho, deltarhoprime, deltarhobar, zp, zpp = self._define_squeezing_variables(rho, rhoprime, Xi)
            expsigma, expsigmaprime = np.eye(N), np.eye(N)
            expdeltarhobar, expdrbs = np.eye(N), np.eye(N)
            
            exp_adag_adag, exp_a_a, exp_adag_a = self._identity(), self._identity(), self._identity()
            
        Xi_inv = sp.linalg.inv(Xi)

        prefactor_adag = np.matmul(np.eye(self.num_deg_freedom) + rhoprime, expdrbs)
        a_temp_coeff = 0.5*np.matmul(np.eye(self.num_deg_freedom) + rhoprime, deltarho + deltarho.T)
        prefactor_a = np.matmul(np.eye(self.num_deg_freedom) + a_temp_coeff, expsigmaprime)
        
        exp_adag_list = []
        for j in range(self.num_deg_freedom):
            exp_adag_j = sp.linalg.expm(np.sum([2.0*np.pi*(np.matmul(Xi_inv.T, prefactor_adag)[j, i]/np.sqrt(2.0))
                                            *a_op_list[i].T for i in range(self.num_deg_freedom)], axis=0))
            exp_adag_list.append(exp_adag_j)
            
        exp_adag_mindiff = sp.linalg.expm(np.sum([minima_diff[x]*np.matmul(Xi_inv.T, prefactor_adag)[x, i]
                                                  *a_op_list[i].T for x in range(self.num_deg_freedom) 
                                                  for i in range(self.num_deg_freedom)], axis=0)/np.sqrt(2.0))
        exp_a_list = []
        for j in range(self.num_deg_freedom):
            exp_a_j = sp.linalg.expm(np.sum([2.0*np.pi*(np.matmul(Xi_inv.T, prefactor_a)[j, i]/np.sqrt(2.0))
                                            *a_op_list[i] for i in range(self.num_deg_freedom)], axis=0))
            exp_a_list.append(exp_a_j)
        
        exp_a_mindiff = sp.linalg.expm(np.sum([-minima_diff[x]*np.matmul(Xi_inv.T, prefactor_a)[x, i]
                                               *a_op_list[i] for x in range(self.num_deg_freedom) 
                                               for i in range(self.num_deg_freedom)], axis=0)/np.sqrt(2.0))
        
        prefactor_adag = np.matmul(np.eye(self.num_deg_freedom) - rhoprime, expdrbs)
        a_temp_coeff = 0.5*np.matmul(np.eye(self.num_deg_freedom) - rhoprime, deltarho + deltarho.T)
        prefactor_a = np.matmul(np.eye(self.num_deg_freedom) - a_temp_coeff, expsigmaprime)
        
        exp_i_list = []
        for j in range(self.num_deg_freedom):
            exp_i_j_adag_part = sp.linalg.expm(np.sum([1j*(np.matmul(Xi, prefactor_adag)[j, i]/np.sqrt(2.0))
                                                   *a_op_list[i].T for i in range(self.num_deg_freedom)], axis=0))
            exp_i_j_a_part = sp.linalg.expm(np.sum([1j*(np.matmul(Xi, prefactor_a)[j, i]/np.sqrt(2.0))
                                                   *a_op_list[i] for i in range(self.num_deg_freedom)], axis=0))
            exp_i_j = np.matmul(exp_i_j_adag_part, np.matmul(exp_adag_a, exp_i_j_a_part))
            exp_i_list.append(exp_i_j)
            
        exp_i_sum_adag_part = sp.linalg.expm(np.sum([1j*self.boundary_coeffs[j]*
                                                     np.matmul(Xi[j, :], prefactor_adag)[i]
                                                     *a_op_list[i].T 
                                                     for i in range(self.num_deg_freedom)
                                                     for j in range(self.num_deg_freedom)], 
                                                     axis=0)/np.sqrt(2.0))
        exp_i_sum_a_part = sp.linalg.expm(np.sum([1j*self.boundary_coeffs[j]*
                                                  np.matmul(Xi[j, :], prefactor_a)[i]
                                                  *a_op_list[i]
                                                  for i in range(self.num_deg_freedom)
                                                  for j in range(self.num_deg_freedom)], 
                                                  axis=0)/np.sqrt(2.0))
        exp_i_sum = np.matmul(exp_i_sum_adag_part, np.matmul(exp_adag_a, exp_i_sum_a_part))
        
        exp_list = [exp_adag_adag, exp_a_a, exp_adag_a, 
                    exp_adag_list, exp_adag_mindiff, 
                    exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum]
        
        return (exp_list, rho, rhoprime, sigma, sigmaprime, deltarho, deltarhobar, zp, zpp)
    
    def kineticmat(self):
        """Return the kinetic part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        delta_inv = np.matmul(np.transpose(Xi_inv), Xi_inv)
        num_exc_tot = self.hilbertdim()
        EC_mat = self.build_EC_matrix()
        dim = self.matrixdim()
        minima_list = self.sorted_minima()
        kinetic_mat = np.zeros((dim,dim), dtype=np.complex128)
        nglist = self.nglist
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                minima_diff = minima_p-minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime, 
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi, a_op_list)
                (exp_adag_adag, exp_a_a, exp_adag_a, 
                 exp_adag_list, exp_adag_mindiff, 
                 exp_a_list, exp_a_mindiff, _, _) = exp_list
                expsdrb = np.matmul(sp.linalg.expm(-sigma).T, sp.linalg.expm(deltarhobar))
                expsigma = sp.linalg.expm(-sigma)
                expsigmaprime = sp.linalg.expm(-sigmaprime)
                scale = 1./np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom)-np.matmul(rho, rhoprime)))
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.num_deg_freedom)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[i] for i in range(self.num_deg_freedom)])
                    delta_phi_kpm = phik-(minima_m-minima_p)
                    
                    x = np.matmul(delta_phi_kpm, Xi_inv.T)/np.sqrt(2.)
                    y = -x
                    z = 1j*Xi_inv.T/np.sqrt(2.)
                    
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    yrhop = np.matmul(y, rhoprime)
                    deltarhopp = 0.5*np.matmul(x-yrhop, deltarho+deltarho.T)
                    
                    epsilon = (-np.matmul(z, np.matmul(rhoprime, deltarhopp) - yrhop + deltarhopp)
                               - (1j/2.)*np.matmul(Xi_inv.T, np.matmul(Xi_inv, delta_phi_kpm)))
                    
                    V_op_dag = self._identity()
                    for j in range(self.num_deg_freedom):
                        V_op_dag_temp = np.linalg.matrix_power(exp_adag_list[j], jkvals[j])
                        V_op_dag = np.matmul(V_op_dag, V_op_dag_temp)
                    exp_adag = np.matmul(V_op_dag, exp_adag_mindiff)
                    
                    V_op = self._identity()
                    for j in range(self.num_deg_freedom):
                        V_op_temp = np.linalg.matrix_power(exp_a_list[j], -jkvals[j])
                        V_op = np.matmul(V_op, V_op_temp)
                    exp_a = np.matmul(V_op, exp_a_mindiff)

                    kinetic_temp = np.sum([+4*np.matmul(exp_adag_a, np.matmul(a_op_list[mu],
                                                                              a_op_list[nu]))
                                           *np.matmul(np.matmul(expsigmaprime.T, zp.T), 
                                                      np.matmul(EC_mat, np.matmul(zp, expsigmaprime)))[mu, nu]
                                           -8*np.matmul(a_op_list[mu].T, np.matmul(exp_adag_a, a_op_list[nu]))
                                           *np.matmul(np.matmul(expsdrb, zpp.T), 
                                                      np.matmul(EC_mat, np.matmul(zp, expsigmaprime)))[mu, nu]
                                           +4*np.matmul(a_op_list[mu].T, np.matmul(a_op_list[nu].T, exp_adag_a))
                                           *np.matmul(np.matmul(expsdrb, zpp.T), 
                                                      np.matmul(EC_mat , np.matmul(zpp, expsdrb.T)))[mu, nu]
                                           -4*exp_adag_a*np.matmul(zpp.T, np.matmul(EC_mat, zp))[mu, nu]
                                           -8*np.matmul(exp_adag_a, a_op_list[mu])
                                           *epsilon[nu]*np.matmul(np.matmul(EC_mat[nu, :], zp), expsigmaprime[:, mu])
                                           +8*np.matmul(a_op_list[nu].T, exp_adag_a)
                                           *epsilon[mu]*np.matmul(np.matmul(EC_mat[mu, :], zpp), 
                                                                  np.transpose(expsdrb)[:, nu])
                                           for mu in range(self.num_deg_freedom) 
                                           for nu in range(self.num_deg_freedom)], axis = 0)
                    
                    kinetic_temp += 4*exp_adag_a*np.matmul(epsilon, np.matmul(EC_mat, epsilon))
                                        
                    kinetic_temp = (alpha * np.exp(-1j*np.dot(nglist, delta_phi_kpm)) 
                                    * np.exp(-0.5*np.trace(sigma)-0.5*np.trace(sigmaprime)) #from U, U'
                                    * np.exp(-0.25*np.matmul(np.matmul(delta_phi_kpm, Xi_inv.T), 
                                                             np.matmul(Xi_inv, delta_phi_kpm))) #from V ops
                                    * np.matmul(np.matmul(exp_adag_adag, exp_adag), 
                                                np.matmul(kinetic_temp, np.matmul(exp_a, exp_a_a))))
                    
                    kinetic_mat[m*num_exc_tot : m*num_exc_tot + num_exc_tot, 
                                p*num_exc_tot : p*num_exc_tot + num_exc_tot] += kinetic_temp
                    
                    jkvals = next(klist,-1)
                                           
        return kinetic_mat
    
    def _alpha_helper(self, x, y, rhoprime, deltarho):
        yrhop = np.matmul(y, rhoprime)
        alpha = np.exp(-0.5*np.matmul(y, yrhop)-0.5*np.matmul(x-yrhop, np.matmul(deltarho, x-yrhop)))
        return alpha
    
    def potentialmat(self):
        """Return the potential part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        delta_inv = np.matmul(np.transpose(Xi_inv), Xi_inv)
        dim = self.matrixdim()
        num_exc_tot = self.hilbertdim()
        potential_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        nglist = self.nglist
        EJlist = self.EJlist
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                minima_diff = minima_p-minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime, 
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi, a_op_list)
                (exp_adag_adag, exp_a_a, exp_adag_a, 
                 exp_adag_list, exp_adag_mindiff, 
                 exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum) = exp_list
                expsdrb = np.matmul(sp.linalg.expm(-sigma).T, sp.linalg.expm(deltarhobar))
                expsigma = sp.linalg.expm(-sigma)
                expsigmaprime = sp.linalg.expm(-sigmaprime)
                scale = 1./np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom)-np.matmul(rho, rhoprime)))
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.num_deg_freedom)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[i] for i in range(self.num_deg_freedom)])
                    delta_phi_kpm = phik-(minima_m-minima_p) 
                    phibar_kpm = 0.5*(phik+(minima_m+minima_p))  
                    
                    V_op_dag = self._identity()
                    for j in range(self.num_deg_freedom):
                        V_op_dag_temp = np.linalg.matrix_power(exp_adag_list[j], jkvals[j])
                        V_op_dag = np.matmul(V_op_dag, V_op_dag_temp)
                    exp_adag = np.matmul(V_op_dag, exp_adag_mindiff)
                    
                    V_op = self._identity()
                    for j in range(self.num_deg_freedom):
                        V_op_temp = np.linalg.matrix_power(exp_a_list[j], -jkvals[j])
                        V_op = np.matmul(V_op, V_op_temp)
                    exp_a = np.matmul(V_op, exp_a_mindiff)
                    
                    exp_i_phi_list = np.array([exp_i_list[i]*np.exp(1j*phibar_kpm[i])
                                               for i in range(self.num_deg_freedom)])
                    exp_i_phi_sum_op = (exp_i_sum*np.exp(1j*2.0*np.pi*self.flux)
                                        *np.prod([np.exp(1j*self.boundary_coeffs[i]*phibar_kpm[i])
                                                 for i in range(self.num_deg_freedom)]))
                    
                    for num in range(self.num_deg_freedom): #summing over potential terms cos(\phi_x)
                        x = (np.matmul(delta_phi_kpm, Xi_inv.T) + 1j*Xi[num, :])/np.sqrt(2.)
                        y = (-np.matmul(delta_phi_kpm, Xi_inv.T) + 1j*Xi[num, :])/np.sqrt(2.)
                        
                        alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                        alpha_con = scale * self._alpha_helper(x.conjugate(), y.conjugate(), 
                                                               rhoprime, deltarho)
                        
                        potential_temp = -0.5*EJlist[num]*alpha*exp_i_phi_list[num]
                        potential_temp += -0.5*EJlist[num]*alpha_con*exp_i_phi_list[num].conjugate()
                        potential_temp = (np.matmul(exp_adag, np.matmul(potential_temp, exp_a))
                                          *np.exp(-.25*np.dot(Xi[num, :], np.transpose(Xi)[:, num])))
                        
                        potential_temp = (np.exp(-1j*np.dot(nglist, delta_phi_kpm)) 
                                          * np.exp(-0.5*np.trace(sigma)-0.5*np.trace(sigmaprime))
                                          * np.exp(-0.25*np.matmul(np.matmul(delta_phi_kpm, Xi_inv.T), 
                                                   np.matmul(Xi_inv, delta_phi_kpm)))
                                          * np.matmul(np.matmul(exp_adag_adag, potential_temp), exp_a_a))
                        
                        potential_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                      p*num_exc_tot:p*num_exc_tot+num_exc_tot] += potential_temp
                            
                    #cos(\phi_0-\phi1-2\pi f)
                    x = (np.matmul(delta_phi_kpm, Xi_inv.T) 
                         + np.sum([1j*Xi[i, :]*self.boundary_coeffs[i]
                                   for i in range(self.num_deg_freedom)], axis=0))/np.sqrt(2.)
                    y = (- np.matmul(delta_phi_kpm, Xi_inv.T) 
                         + np.sum([1j*Xi[i, :]*self.boundary_coeffs[i]
                                   for i in range(self.num_deg_freedom)], axis=0))/np.sqrt(2.)
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    alpha_con = scale * self._alpha_helper(x.conjugate(), y.conjugate(), 
                                                           rhoprime, deltarho)
                    
                    potential_temp = -0.5*EJlist[-1]*alpha*exp_i_phi_sum_op
                    potential_temp += -0.5*EJlist[-1]*alpha_con*exp_i_phi_sum_op.conjugate()
                    
                    potential_temp = (np.matmul(exp_adag, np.matmul(potential_temp, exp_a))
                                      *np.exp(-0.25*np.sum([self.boundary_coeffs[j]
                                                            *self.boundary_coeffs[k]
                                                            *np.dot(Xi[j,:], np.transpose(Xi)[:,k])
                                                            for j in range(self.num_deg_freedom)
                                                            for k in range(self.num_deg_freedom)])))
                    
                    x = np.matmul(delta_phi_kpm, Xi_inv.T)/np.sqrt(2.)
                    y = -x
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)

                    potential_temp += (alpha*np.sum(EJlist)
                                       *np.matmul(exp_adag, np.matmul(exp_adag_a, exp_a)))
                    
                    potential_temp = (np.exp(-1j*np.dot(nglist, delta_phi_kpm)) 
                                      * np.exp(-0.5*np.trace(sigma)-0.5*np.trace(sigmaprime))
                                      * np.exp(-0.25*np.matmul(np.matmul(delta_phi_kpm, Xi_inv.T), 
                                                               np.matmul(Xi_inv, delta_phi_kpm)))
                                      * np.matmul(np.matmul(exp_adag_adag, potential_temp), exp_a_a))
                    
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
        num_exc_tot = self.hilbertdim()
        inner_product_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        nglist = self.nglist
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                minima_diff = minima_p-minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime, 
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi, a_op_list)
                (exp_adag_adag, exp_a_a, exp_adag_a, 
                 exp_adag_list, exp_adag_mindiff, 
                 exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum) = exp_list
                scale = 1./np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom)-np.matmul(rho, rhoprime)))
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.num_deg_freedom)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[i] for i in range(self.num_deg_freedom)])
                    delta_phi_kpm = phik-(minima_m-minima_p) 
                            
                    x = np.matmul(delta_phi_kpm, Xi_inv.T)/np.sqrt(2.)
                    y = -x
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    
                    V_op_dag = self._identity()
                    for j in range(self.num_deg_freedom):
                        V_op_dag_temp = np.linalg.matrix_power(exp_adag_list[j], jkvals[j])
                        V_op_dag = np.matmul(V_op_dag, V_op_dag_temp)
                    exp_adag = np.matmul(V_op_dag, exp_adag_mindiff)
                    
                    V_op = self._identity()
                    for j in range(self.num_deg_freedom):
                        V_op_temp = np.linalg.matrix_power(exp_a_list[j], -jkvals[j])
                        V_op = np.matmul(V_op, V_op_temp)
                    exp_a = np.matmul(V_op, exp_a_mindiff)
                    
                    inner_temp = (alpha * np.exp(-1j*np.dot(nglist, delta_phi_kpm)) 
                                  * np.exp(-0.5*np.trace(sigma)-0.5*np.trace(sigmaprime))
                                  * np.exp(-0.25*np.matmul(np.matmul(delta_phi_kpm, Xi_inv.T), 
                                                           np.matmul(Xi_inv, delta_phi_kpm)))
                                  * np.matmul(np.matmul(exp_adag_adag, np.matmul(exp_adag, exp_adag_a)), 
                                              np.matmul(exp_a, exp_a_a)))
                    
                    inner_product_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                      p*num_exc_tot:p*num_exc_tot+num_exc_tot] += inner_temp
                    jkvals = next(klist,-1)
                        
        return inner_product_mat
    
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
    
   