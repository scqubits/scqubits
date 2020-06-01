import numpy as np
import math
# Helper class for efficiently constructing raising and lowering operators
# using a global excitation cutoff scheme, as opposed to the more commonly used
# number of excitations per mode cutoff, which can be easily constructed 
# using kronecker product. The ideas herein are based on the excellent 
# paper 
# [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).


class Hashing:
    def __init__(self):
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
        
    def _hash(self, vec):
        dim = len(vec)
        return np.sum([np.sqrt(self.prime_list[i])*vec[i] for i in range(dim)])
    
    def _gen_tags(self, basis_vecs):
        dim = basis_vecs.shape[0]
        tag_list = np.array([self._hash(basis_vecs[i,:]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return (tag_list, index_array)
    
    def _gen_basis_vecs(self):
        sites = self.num_deg_freedom()
        vec_list = []
        vec_list.append(np.zeros(sites))
        for total_exc in range(1, self.global_exc+1):  # No excitation number conservation as in [1]
            prev_vec = np.zeros(sites)
            prev_vec[0] = total_exc
            vec_list.append(prev_vec)
            while prev_vec[-1] != total_exc:  # step through until the last entry is total_exc
                k = self._find_k(prev_vec)
                next_vec = np.zeros(sites)
                next_vec[0:k] = prev_vec[0:k]
                next_vec[k] = prev_vec[k]-1
                next_vec[k+1] = total_exc-np.sum([next_vec[i] for i in range(k+1)])
                vec_list.append(next_vec)
                prev_vec = next_vec
        return np.array(vec_list)
    
    def eigvec_population(self, eigvec):
        basis_vecs = self._gen_basis_vecs()
        dim = len(basis_vecs)
        pop_list = []
        min_list = []
        vec_list = []
        for k, elem in enumerate(eigvec):
            if not np.allclose(elem, 0.0, atol=1e-4):
                minimum = math.floor(k/dim)
                pop_list.append(elem)
                min_list.append(minimum)
                vec_list.append(basis_vecs[np.mod(k, dim)])
        pop_list = np.abs(pop_list)**2
        index_array = np.argsort(np.abs(pop_list))
        pop_list = (pop_list[index_array])[::-1]
        min_list = (np.array(min_list)[index_array])[::-1]
        vec_list = (np.array(vec_list)[index_array])[::-1]
        return pop_list, zip(min_list, vec_list)
                
    def _find_k(self, vec):
        dim = len(vec)
        for num in range(dim-2, -1, -1):
            if vec[num] != 0:
                return num
