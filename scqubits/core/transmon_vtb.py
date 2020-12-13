import os

import numpy as np
import scipy as sp
import itertools

import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.core.qubit_base as base
from scqubits.core.discretization import Grid1d
from scqubits.utils.spectrum_utils import order_eigensystem


#-Transmon using VariationalTightBinding

class TransmonVTB(base.QubitBaseClass1d, serializers.Serializable):
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ng = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    def __init__(self, EJ, EC, ng, kmax, num_exc, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.kmax = kmax
        self.num_exc = num_exc
        self.truncated_dim = truncated_dim
        
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._default_grid = Grid1d(-6.5*np.pi, 6.5*np.pi, 651)
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            'qubit_pngs/transmonvtb.png')
        
    @staticmethod
    def default_params():
        return {
            'EJ': 30.0,
            'EC': 1.2,
            'ng': 0.0,
            'kmax': 1,
            'num_exc' : 3,
            'truncated_dim': 3
        }

    @staticmethod
    def nonfit_params():
        return ['ng', 'kmax', 'num_exc', 'truncated_dim']
                
    def potential(self, phi):
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi: float
            phase variable value

        Returns
        -------
        float
        """
        return -self.EJ * np.cos(phi)
    
    def kineticmat(self):
        Xi = (8.0*self.EC/self.EJ)**(1/4)
        a = self.a_operator()
        klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=1)
        jkvals = next(klist,-1)
        kinetic_mat = np.zeros((self.num_exc+1, self.num_exc+1), dtype=np.complex128)
        while jkvals != -1:
            phik = 2.0*np.pi*jkvals[0]
            
            V_op = self.V_operator(-phik)
            V_op_dag = self.V_operator(phik).T
            
            kin = 2*self.EC*Xi**(-2)*(- np.matmul(a, a) + 2.0*np.matmul(a.T, a)
                                      - np.matmul(a.T, a.T) + self._identity())
            kin += -(4*self.EC/np.sqrt(2.))*Xi**(-3)*phik*(a - a.T)
            kin += -self.EC*Xi**(-4)*phik**2*self._identity()
            
            kin = np.exp(-1j*self.ng*phik)*np.matmul(V_op_dag, np.matmul(kin, V_op))
            kinetic_mat += kin
            
            jkvals = next(klist, -1)
        return kinetic_mat
        
    def potentialmat(self):
        Xi = (8.0*self.EC/self.EJ)**(1/4)
        a = self.a_operator()
        klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=1)
        jkvals = next(klist,-1)
        potential_mat = np.zeros((self.num_exc+1, self.num_exc+1), dtype=np.complex128)
        while jkvals != -1:
            phik = 2.0*np.pi*jkvals[0]
            
            V_op = self.V_operator(-phik)
            V_op_dag = self.V_operator(phik).T
            
            pot_op = (np.exp(-(Xi**2)/4.0)*np.exp(1j*phik/2.)
                      *np.matmul(sp.linalg.expm(1j*(Xi/np.sqrt(2.0))*a.T),
                                 sp.linalg.expm(1j*(Xi/np.sqrt(2.0))*a)))
            pot = -0.5*self.EJ*(pot_op + pot_op.conjugate().T)
            
            pot = np.exp(-1j*self.ng*phik)*np.matmul(V_op_dag, np.matmul(pot, V_op))
            potential_mat += pot
            
            jkvals = next(klist, -1)
        return potential_mat
    
    def inner_product(self):
        inner_mat = np.zeros((self.num_exc+1, self.num_exc+1), dtype=np.complex128)
        klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=1)
        jkvals = next(klist,-1)
        while jkvals != -1:
            phik = 2.0*np.pi*jkvals[0]
            
            V_op = self.V_operator(-phik)
            V_op_dag = self.V_operator(phik).T
            
            inner = np.exp(-1j*self.ng*phik)*np.matmul(V_op_dag, V_op)
            inner_mat += inner
            
            jkvals = next(klist, -1)
        return inner_mat
        
    def a_operator(self):
        """Return the lowering operator"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a,k=1)
        return a_mat
    
    def hilbertdim(self):
        return self.num_exc+1
    
    def _identity(self):
        return np.identity(self.num_exc+1)
                                
    def V_operator(self, phi):
        """Return the V operator """
        Xi = (8.0*self.EC/self.EJ)**(1/4)
        prefactor = np.exp(-.125 * (phi*Xi**(-1))**2)
        op = sp.linalg.expm((phi*Xi**(-1)/np.sqrt(2.))
                            *self.a_operator())
        return prefactor * op                                               
                                                                          
    def hamiltonian(self):
        """Construct the Hamiltonian"""
        return (self.kineticmat() + self.potentialmat())
        
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
    
    def wavefunction1d_defaults(self, mode, evals, wavefunc_count):
        pass
    
    def wavefunction(self, esys=None, which=0, phi_grid=None):
        pass
    
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
 