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
from scqubits.core.storage import WaveFunctionOnGrid
from scqubits.utils.spectrum_utils import standardize_phases, order_eigensystem


#-Flux Qubit using VCHOS 

class FluxQubitVCHOSGlobal(FluxQubitVCHOS):
    def __init__(self, ECJ, ECg, EJ, ng1, ng2, alpha, flux, kmax, global_exc):
        FluxQubitVCHOS.__init__(self, ECJ, ECg, EJ, ng1, ng2, alpha, flux, kmax, num_exc=None)
        self.global_exc = global_exc

        self.prime_list = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 
                                    29, 31, 37, 41, 43, 47, 53, 59,
                                    61, 67, 71, 73, 79, 83, 89, 97, 
                                    101, 103, 107, 109, 113, 127, 
                                    131, 137, 139, 149, 151, 157, 
                                    163, 167, 173, 179, 181, 191, 
                                    193, 197, 199, 211, 223, 227,
                                    229, 233, 239, 241, 251, 257, 
                                    263, 269, 271, 277, 281, 283, 
                                    293, 307, 311, 313, 317, 331, 
                                    337, 347, 349, 353, 359, 367, 
                                    373, 379, 383, 389, 397, 401,
                                    409, 419, 421, 431, 433, 439, 
                                    443, 449, 457, 461, 463, 467, 
                                    479, 487, 491, 499, 503, 509, 
                                    521, 523, 541, 547, 557, 563, 
                                    569, 571, 577, 587, 593, 599, 
                                    601, 607, 613, 617, 619, 631, 
                                    641, 643, 647, 653, 659, 661, 
                                    673, 677, 683, 691, 701, 709, 
                                    719, 727, 733, 739, 743, 751, 
                                    757, 761, 769, 773, 787, 797, 
                                    809, 811, 821, 823, 827, 829, 
                                    839, 853, 857, 859, 863, 877, 
                                    881, 883, 887, 907, 911, 919, 
                                    929, 937, 941, 947, 953, 967, 
                                    971, 977, 983, 991, 997])
        self.basis_vecs = self._gen_basis_vecs()
        self.tag_list, self.index_array = self._gen_tags()
        
    def _hash(self, vec):
        dim = len(vec)
        return np.sum([np.sqrt(self.prime_list[i])*vec[i] for i in range(dim)])
    
    def _gen_tags(self):
        basis_vecs = self.basis_vecs
        dim = basis_vecs.shape[0]
        tag_list = np.array([self._hash(basis_vecs[i,:]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return (tag_list, index_array)
    
    def _gen_basis_vecs(self):
        sites = 2
        vec_list = []
        vec_list.append(np.zeros(sites))
        for total_exc in range(1, self.global_exc+1):
            prev_vec = np.zeros(sites)
            prev_vec[0] = total_exc
            vec_list.append(prev_vec)
            while prev_vec[-1] != total_exc:
                k = self._find_k(prev_vec)
                next_vec = np.zeros(sites)
                next_vec[0:k] = prev_vec[0:k]
                next_vec[k] = prev_vec[k]-1
                next_vec[k+1] = total_exc-np.sum([next_vec[i] for i in range(k+1)])
                vec_list.append(next_vec)
                prev_vec = next_vec
        return np.array(vec_list)
                
    def _find_k(self, vec):
        dim = len(vec)
        for num in range(dim-2, -1, -1):
            if vec[num]!=0:
                return num
    
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
    
            
    def _identity(self):
        return(np.identity(len(self.tag_list)))
            
    def matrixdim(self):
        return len(self.sorted_minima())*len(self.tag_list)
    
    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return len(self.tag_list)
    
    def wavefunction(self, esys=None, which=0, phi_grid=None):
        #TODO fix this for global hashing
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
 