import numpy as np
from scipy.special import comb
import math
from typing import Callable

import scqubits.utils.plotting as plot
# Helper class for efficiently constructing raising and lowering operators
# using a global excitation cutoff scheme, as opposed to the more commonly used
# number of excitations per mode cutoff, which can be easily constructed 
# using kronecker product. The ideas herein are based on the excellent 
# paper 
# [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).


class Hashing:
    _generate_vectors_up_to_maximum_length: Callable

    def __init__(self, global_exc, number_degrees_freedom):
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
        self.global_exc = global_exc
        self.number_degrees_freedom = number_degrees_freedom

    def a_operator(self, i):
        """
        This method for defining the a_operator is based on
        J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
        We ask the question, for each basis vector, what is the action of a_i
        on it? In this way, we can define a_i using a single for loop.
        """
        basis_vecs = self._generate_vectors_up_to_maximum_length(self.global_exc, self.number_degrees_freedom)
        tags, index_array = self._gen_tags(basis_vecs)
        dim = self.number_states_per_minimum()
        a = np.zeros((dim, dim), dtype=np.complex_)
        for w, vec in enumerate(basis_vecs):
            if vec[i] >= 1:
                temp_vec = np.copy(vec)
                temp_vec[i] = vec[i] - 1
                temp_coeff = np.sqrt(vec[i])
                temp_vec_tag = self._hash(temp_vec)
                index = np.searchsorted(tags, temp_vec_tag)
                basis_index = index_array[index]
                a[basis_index, w] = temp_coeff
        return a

    def number_states_per_minimum(self):
        """
        Using the global excitation scheme the total number of states
        per minimum is given by the hockey-stick identity
        """
        return int(comb(self.global_exc + self.number_degrees_freedom, self.number_degrees_freedom))

    def _hash(self, vec):
        dim = len(vec)
        return np.sum([np.sqrt(self.prime_list[i])*vec[i] for i in range(dim)])
    
    def _gen_tags(self, basis_vecs):
        dim = basis_vecs.shape[0]
        tag_list = np.array([self._hash(basis_vecs[i, :]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return tag_list, index_array

    @staticmethod
    def _append_relevant_vectors(next_vec, vec_list):
        vec_list.append(next_vec)
    
    def eigvec_population(self, eigvec):
        basis_vecs = self._generate_vectors_up_to_maximum_length(self.global_exc, self.number_degrees_freedom)
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

    def state_amplitudes_function(self, i, evecs, which):
        total_num_states = self.number_states_per_minimum()
        return np.real(evecs[i * total_num_states: (i + 1) * total_num_states, which])

    def wavefunc_amplitudes_function(self, state_amplitudes, normal_mode_1, normal_mode_2):
        total_num_states = self.number_states_per_minimum()
        basis_vecs = self._generate_vectors_up_to_maximum_length(self.global_exc, self.number_degrees_freedom)
        wavefunc_amplitudes = np.zeros_like(normal_mode_1).T
        for j in range(total_num_states):
            basis_vec = basis_vecs[j]
            s1 = int(basis_vec[0])
            s2 = int(basis_vec[1])
            ho_2d = plot.multiply_two_harm_osc_functions(s1, s2, normal_mode_1, normal_mode_2)
            wavefunc_amplitudes += state_amplitudes[j] * ho_2d.T
        return wavefunc_amplitudes
