import numpy as np
import scipy as sp
import itertools
from scipy.optimize import minimize
import scipy.constants as const

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
        self.nglist = np.array([ng1, ng2])
        self.alpha = alpha
        self.flux = flux
        self.kmax = kmax
        self.num_exc = num_exc
        self.hGHz = const.h * 10**9
        self.Z0 = 1. / (32*np.pi**2*const.alpha)
        self.Phi0 = 1. / (2*np.sqrt(4*np.pi*const.alpha))
        self._evec_dtype = np.float_
        self._default_grid = Grid1d(-6.5*np.pi, 6.5*np.pi, 351)
        
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
                
        CJ = 2*np.pi*const.alpha / self.ECJ
        Cg = 2*np.pi*const.alpha / self.ECg
        
        Cmat[0, 0] = CJ + self.alpha*CJ + Cg
        Cmat[1, 1] = CJ + self.alpha*CJ + Cg
        Cmat[0, 1] = -self.alpha*CJ
        Cmat[1, 0] = -self.alpha*CJ
        
        return Cmat
    
    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        Cmat = self.build_capacitance_matrix()
        return 2 * np.pi * const.alpha * np.linalg.inv(Cmat)
    
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

        return np.transpose(Ximat)
    
    def a_operator(self, mu):
        """Return the lowering operator associated with the xth d.o.f. in the full Hilbert space"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a,k=1)
        return self._full_o([a_mat], [mu])
    
    def n_operator(self, x):
        """Return the number operator associated with the x d.o.f., expressed using ladder ops"""
        #Note that this is not used below
        Xi_inv_mat = np.linalg.inv(self.Xi_matrix())
        n_op = -(1./np.sqrt(2.))*1j*np.sum([Xi_inv_mat[x,mu]*(self.a_operator(mu)-self.a_operator(mu).T) 
                                            for mu in range(2)], axis=0)
        
    def normal_ordered_exp_i_phi_operator(self, x):
        """Return the normal ordered e^{i\phi_x} operator, expressed using ladder ops"""
        Xi_mat = self.Xi_matrix()
        return(np.exp(-.25*np.dot(Xi_mat[x, :], Xi_mat[:, x]))
               *np.matmul(self.matrix_exp(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu).T 
                                            for mu in range(2)], axis=0)/np.sqrt(2)), 
                          self.matrix_exp(1j*np.sum([Xi_mat[x,mu]*self.a_operator(mu) 
                                            for mu in range(2)], axis=0)/np.sqrt(2))))
    
    def phi_operator(self, x):
        """Return the phi operator associated with the x d.o.f., expressed using ladder ops"""
        #Note that this is not used below
        Xi_mat = self.Xi_matrix()
        return (1./np.sqrt(2.))*np.sum([Xi_mat[x,mu]*(self.a_operator(mu)+self.a_operator(mu).T) 
                                        for mu in range(2)], axis=0)
    
    def _identity(self):
        return(np.identity((self.num_exc+1)**2))
    
    def delta_matrix(self):
        """"Construct the delta matrix, as described in David's notes """
        #Note that this is not used below
        return np.matmul(self.Xi_matrix(),np.transpose(self.Xi_matrix()))
    
    def delta_inv_matrix(self):
        """"Construct the delta inverse matrix, as described in David's notes """
        Xi_T_inv = np.transpose(np.linalg.inv(self.Xi_matrix()))
        Xi_inv = np.linalg.inv(self.Xi_matrix())
        return np.matmul(Xi_T_inv,Xi_inv)
    
    def matrix_exp(self, matrix):
        """Perform the matrix exponentiation"""
        expm = np.identity((self.num_exc+1)**2,dtype=np.complex128)
        for num in range(1,self.num_exc+1):
            prefact = sp.special.factorial(num)**(-1)
            prod = matrix
            for m in range(1,num):
                prod = np.matmul(prod,matrix)
            expm += prefact * prod
        return(expm)
    
    def V_operator(self, phi):
        """Return the V operator """
        phi_delta_phi = np.matmul(phi,np.matmul(self.delta_inv_matrix(),phi))
        prefactor = np.exp(-.125 * phi_delta_phi)
        phi_Xi_inv = np.matmul(phi,np.transpose(np.linalg.inv(self.Xi_matrix())))
        phi_Xi_inv_a = np.sum([phi_Xi_inv[mu]*self.a_operator(mu) for mu in range(2)], axis=0)
        op = self.matrix_exp((1./np.sqrt(2.))*phi_Xi_inv_a)
        return prefactor * op
    
    def kineticmat(self):
        """Return the kinetic part of the hamiltonian"""
        Xi_inv = np.linalg.inv(self.Xi_matrix())
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
                    delta_phi_kpm = -(phik+(minima_p-minima_m)) #XXXXXXXXXX
                        
                    V_op = self.V_operator(delta_phi_kpm)
                    V_op_dag = self.V_operator(-delta_phi_kpm).T
                    
                    kinetic_temp = 0.
                    
                    for mu in range(2):
                        for nu in range(2):
                            a_mu = self.a_operator(mu)
                            a_nu = self.a_operator(nu)
                            kinetic_temp += (- 0.5*4*EC_mat_t[mu, nu]*a_mu*a_nu
                                            + 0.5*4*EC_mat_t[mu, nu]*a_mu.T*a_nu
                                            - 0.5*4*EC_mat_t[mu, nu]*a_mu.T*a_nu.T
                                            +((2.*np.sqrt(2))**(-1) * (a_mu - a_mu.T)
                                              *4*EC_mat_t[mu, nu] * np.dot(Xi_inv[nu,:], delta_phi_kpm))
                                            +((2.*np.sqrt(2))**(-1) * np.dot(delta_phi_kpm, np.transpose(Xi_inv[:,mu]))
                                              *4*EC_mat_t[mu, nu]*(a_nu - a_nu.T))
                                            -(0.25*np.dot(delta_phi_kpm, np.transpose(Xi_inv[:,mu]))
                                              *EC_mat_t[mu, nu]*np.dot(Xi_inv[nu,:], delta_phi_kpm)*self._identity()))
                            if (mu == nu):
                                kinetic_temp += 0.5*4*EC_mat_t[mu, nu]*(a_nu.T*a_mu+self._identity())
                            else:
                                kinetic_temp += 0.5*4*EC_mat_t[mu, nu]*a_nu.T*a_mu
                                                
                    kinetic_temp = (np.exp(1j*np.dot(self.nglist, delta_phi_kpm))
                                    *np.matmul(V_op_dag, kinetic_temp))
                    kinetic_temp = np.matmul(kinetic_temp, V_op)
                    
                    num_exc_tot = (self.num_exc+1)**2
#                    print(kinetic_temp)
                    kinetic_mat[m*num_exc_tot:m*num_exc_tot+num_exc_tot, 
                                p*num_exc_tot:p*num_exc_tot+num_exc_tot] += kinetic_temp
                    jkvals = next(klist,-1)
                                           
        return kinetic_mat
        
    def potentialmat(self):
        """Return the potential part of the hamiltonian"""
        dim = self.hilbertdim()
        potential_mat = np.zeros((dim,dim), dtype=np.complex128)
        minima_list = self.sorted_minima()
        for m, minima_m in enumerate(minima_list):
            for p, minima_p in enumerate(minima_list):
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
                jkvals = next(klist,-1)
                while jkvals != -1:
                    phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                    delta_phi_kpm = -(phik+(minima_p-minima_m)) #XXXXXXXXXXXXX
                    phibar_kpm = -0.5*(phik+(minima_m+minima_p)) #XXXXXXXXX
                        
                    exp_i_phi_0_op = np.exp(-1j*phibar_kpm[0])*self.normal_ordered_exp_i_phi_operator(0)
                    exp_i_phi_1_op = np.exp(-1j*phibar_kpm[1])*self.normal_ordered_exp_i_phi_operator(1)
                        
                    V_op = self.V_operator(delta_phi_kpm)
                    V_op_dag = self.V_operator(-delta_phi_kpm).T
                        
                    potential_temp = -0.5*self.EJ*(exp_i_phi_0_op+exp_i_phi_0_op.conjugate().T)
                    potential_temp += -0.5*self.EJ*(exp_i_phi_1_op+exp_i_phi_1_op.conjugate().T)
                    potential_temp += (-0.5*self.alpha*self.EJ
                                       *(np.matmul(exp_i_phi_0_op, exp_i_phi_1_op.conjugate().T)
                                         *np.exp(1j*2.0*np.pi*self.flux)
                                         +np.matmul(exp_i_phi_0_op.conjugate().T, exp_i_phi_1_op)
                                         *np.exp(-1j*2.0*np.pi*self.flux)))
                    #normalization to compare with the literature
                    potential_temp += (2.0*self.EJ+self.alpha*self.EJ)*np.identity((self.num_exc+1)**2)
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
        return self.kineticmat() + self.potentialmat()
    
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
                    delta_phi_kpm = -(phik+(minima_p-minima_m)) #XXXXXXXXX
                        
                    V_op = self.V_operator(delta_phi_kpm)
                    V_op_dag = self.V_operator(-delta_phi_kpm).T
                    
                    inner_temp = (np.exp(1j*np.dot(self.nglist, delta_phi_kpm))
                                  *np.matmul(V_op_dag, V_op))
                        
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
                wavefunc_amplitudes += np.sum([state_amplitudes[s1, s2] 
                * np.multiply(np.sqrt(sp.linalg.norm(Xi_inv[0,:]))
                              *self.harm_osc_wavefunction(s1, np.add.outer(Xi_inv[0,0]*phi_vec+phi1_s1_arg, 
                                                                          Xi_inv[0,1]*phi_vec+phi2_s1_arg)), 
                              np.sqrt(sp.linalg.norm(Xi_inv[1,:]))
                              *self.harm_osc_wavefunction(s2, np.add.outer(Xi_inv[1,0]*phi_vec+phi1_s2_arg,
                                                                          Xi_inv[1,1]*phi_vec+phi2_s2_arg)))
                                               for s2 in range(self.num_exc+1)
                                               for s1 in range(self.num_exc+1)], axis=0).T #FIX .T NOT CORRECT
                jkvals = next(klist,-1)
        
        grid2d = GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                      [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
    
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
    
 