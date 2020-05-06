import numpy as np
import scipy as sp
import itertools

import scqubits.core.discretization as discretization
import scqubits.core.storage as storage
from scqubits.utils.spectrum_utils import standardize_phases
from scqubits.core.flux_qubit_vchos import FluxQubitVCHOS
from scqubits.core.hashing import Hashing


#-Flux Qubit using VCHOS and a global cutoff

class FluxQubitVCHOSGlobal(FluxQubitVCHOS, Hashing):
    def __init__(self, ECJ, ECg, EJlist, alpha, nglist, flux, kmax,
                 global_exc, squeezing=False, truncated_dim=None):
        FluxQubitVCHOS.__init__(self, ECJ, ECg, EJlist, alpha, nglist, flux, 
                                kmax, num_exc=None, squeezing=squeezing, truncated_dim=truncated_dim)
        Hashing.__init__(self, num_deg_freedom=2, global_exc=global_exc)
        
    @staticmethod
    def default_params():
        return {
            'ECJ': 1.0/10.0,
            'ECg': 5.0,
            'EJlist': np.array([1.0, 1.0, 0.8]),
            'alpha' : 0.8,
            'nglist': np.array(2*[0.0]),
            'flux': 0.46,
            'kmax' : 1,
            'global_exc' : 4,
            'squeezing' : False,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['alpha', 'nglist', 'kmax', 'global_exc', 'squeezing', 'truncated_dim']
        
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
        
    def hilbertdim(self):
        return len(self.sorted_minima())*len(self.tag_list)
    
    def wavefunction(self, esys=None, which=0, phi_grid=None):
        """
        Return a flux qubit wave function in phi1, phi2 basis. This implementation
        is for the global excitation cutoff scheme, and similarly to FQV 
        does not include the effects of squeezing.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)
        phi_grid: Grid1D object, optional
            used for setting a custom plot range for phi

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        phi_grid = phi_grid or self._default_grid
        phi_vec = phi_grid.make_linspace()
        
        minima_list = self.sorted_minima()
        num_minima = len(minima_list)
        total_num_states = int(self.hilbertdim()/num_minima)
        basis_vecs = self._gen_basis_vecs()
        
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        norm = np.sqrt(np.abs(np.linalg.det(Xi)))**(-1)
        
        state_amplitudes_list = []
        
        wavefunc_amplitudes = np.zeros_like(np.outer(phi_vec, phi_vec))
        
        for i, minimum in enumerate(minima_list):
            klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
            jkvals = next(klist,-1)
            while jkvals != -1:
                phik = 2.0*np.pi*np.array([jkvals[0],jkvals[1]])
                phi1_s1_arg = Xi_inv[0,0]*(phik - minimum)[0]
                phi2_s1_arg = Xi_inv[0,1]*(phik - minimum)[1]
                phi1_s2_arg = Xi_inv[1,0]*(phik - minimum)[0]
                phi2_s2_arg = Xi_inv[1,1]*(phik - minimum)[1]
                state_amplitudes = np.real(evecs[i*total_num_states : (i+1)*total_num_states, which])
                for j in range(total_num_states):
                    basis_vec = basis_vecs[j]
                    s1 = int(basis_vec[0])
                    s2 = int(basis_vec[1])
                    ho_2d = np.multiply(self.harm_osc_wavefunction(s1, np.add.outer(Xi_inv[0,0]*phi_vec+phi1_s1_arg, 
                                                                                    Xi_inv[0,1]*phi_vec+phi2_s1_arg)), 
                                        self.harm_osc_wavefunction(s2, np.add.outer(Xi_inv[1,0]*phi_vec+phi1_s2_arg,
                                                                                    Xi_inv[1,1]*phi_vec+phi2_s2_arg)))
                    wavefunc_amplitudes += norm * state_amplitudes[j] * ho_2d.T
                jkvals = next(klist,-1)
        
        grid2d = discretization.GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                                     [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
    
        wavefunc_amplitudes = standardize_phases(wavefunc_amplitudes)

        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)
    
    