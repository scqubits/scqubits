import itertools
from typing import Callable, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.special import comb

import scqubits.utils.plotting as plot


def generate_next_vector(prev_vec: ndarray, radius: int) -> ndarray:
    """Algorithm for generating all vectors with positive entries of a given Manhattan length, specified in
    [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010)"""
    k = 0
    for num in range(len(prev_vec) - 2, -1, -1):
        if prev_vec[num] != 0:
            k = num
            break
    next_vec = np.zeros_like(prev_vec)
    next_vec[0:k] = prev_vec[0:k]
    next_vec[k] = prev_vec[k] - 1
    next_vec[k + 1] = radius - np.sum([next_vec[i] for i in range(k + 1)])
    return next_vec


def reflect_vectors(vec: ndarray) -> ndarray:
    """Helper function for generating all possible reflections of a given vector"""
    reflected_vec_list = []
    nonzero_indices = np.nonzero(vec)
    nonzero_vec = vec[nonzero_indices]
    multiplicative_factors = itertools.product(np.array([1, -1]), repeat=len(nonzero_vec))
    for factor in multiplicative_factors:
        reflected_vec = np.copy(vec)
        np.put(reflected_vec, nonzero_indices, np.multiply(nonzero_vec, factor))
        reflected_vec_list.append(reflected_vec)
    return np.array(reflected_vec_list)


class Hashing:
    """Helper class for efficiently constructing raising and lowering operators
    using a global excitation cutoff scheme, as opposed to the more commonly used
    number of excitations per mode cutoff, which can be easily constructed
    using kronecker product. The ideas herein are based on the excellent
    paper
    [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
    """
    num_exc: int  # up to and including the number of global excitations to keep
    number_degrees_freedom: int  # number of degrees of freedom of the system

    def __init__(self) -> None:
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

    def gen_basis_vectors(self) -> ndarray:
        """Generate all basis vectors"""
        return self._gen_basis_vectors(lambda x: [x])

    def _gen_basis_vectors(self, func: Callable) -> ndarray:
        """Generate basis vectors using Zhang algorithm. `func` allows for inclusion of other vectors,
        such as those with negative entries (see CurrentMirrorGlobal)"""
        sites = self.number_degrees_freedom
        vector_list = [np.zeros(sites)]
        for total_exc in range(1, self.num_exc + 1):  # No excitation number conservation as in [1]
            previous_vector = np.zeros(sites)
            previous_vector[0] = total_exc
            vector_list = self._append_similar_vectors(vector_list, previous_vector, func)
            while previous_vector[-1] != total_exc:  # step through until the last entry is total_exc
                next_vector = generate_next_vector(previous_vector, total_exc)
                vector_list = self._append_similar_vectors(vector_list, next_vector, func)
                previous_vector = next_vector
        return np.array(vector_list)

    @staticmethod
    def _append_similar_vectors(vector_list: List, vec: ndarray, func: Callable) -> List:
        similar_vectors = func(vec)
        for vec in similar_vectors:
            vector_list.append(vec)
        return vector_list

    def a_operator(self, i: int) -> ndarray:
        """ Construct the lowering operator for mode `i`.

        Parameters
        ----------
        i: int
            integer specifying the mode whose annihilation operator we would like to construct

        Returns
        -------
        ndarray
        """
        basis_vectors = self.gen_basis_vectors()
        tags, index_array = self._gen_tags(basis_vectors)
        dim = self.number_states_per_minimum()
        a = np.zeros((dim, dim), dtype=np.complex_)
        for w, vec in enumerate(basis_vectors):
            if vec[i] >= 1:
                temp_coefficient = np.sqrt(vec[i])
                basis_index = self._find_lowered_vector(vec, i, tags, index_array)
                a[basis_index, w] = temp_coefficient
        return a

    def _find_lowered_vector(self, vector: ndarray, i: int, tags: ndarray, index_array: ndarray) -> int:
        temp_vector = np.copy(vector)
        temp_vector[i] = vector[i] - 1
        temp_vector_tag = self._hash(temp_vector)
        index = np.searchsorted(tags, temp_vector_tag)
        basis_index = index_array[index]
        return basis_index

    def number_states_per_minimum(self) -> int:
        """Using the global excitation scheme the total number of states
        per minimum is given by the hockey-stick identity"""
        return int(comb(self.num_exc + self.number_degrees_freedom, self.number_degrees_freedom))

    def _hash(self, vector: ndarray) -> ndarray:
        """Generate the (unique) identifier for a given vector `vector`"""
        dim = len(vector)
        return np.sum([np.sqrt(self.prime_list[i]) * vector[i] for i in range(dim)])
    
    def _gen_tags(self, basis_vectors: ndarray) -> Tuple[ndarray, ndarray]:
        """Generate the identifiers for all basis vectors `basis_vectors`"""
        dim = basis_vectors.shape[0]
        tag_list = np.array([self._hash(basis_vectors[i, :]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return tag_list, index_array

    def state_amplitudes_function(self, i: int, evecs: ndarray, which: int) -> ndarray:
        """Overrides method in VCHOS, appropriate for the global excitation cutoff scheme."""
        total_num_states = self.number_states_per_minimum()
        return np.real(evecs[i * total_num_states: (i + 1) * total_num_states, which])

    def wavefunction_amplitudes_function(self, state_amplitudes: ndarray,
                                         normal_mode_1: ndarray,
                                         normal_mode_2: ndarray) -> ndarray:
        """Overrides method in VCHOS, appropriate for the global excitation cutoff scheme."""
        total_num_states = self.number_states_per_minimum()
        basis_vectors = self.gen_basis_vectors()
        wavefunction_amplitudes = np.zeros_like(normal_mode_1).T
        for j in range(total_num_states):
            basis_vector = basis_vectors[j]
            s1 = int(basis_vector[0])
            s2 = int(basis_vector[1])
            ho_2d = plot.multiply_two_harm_osc_functions(s1, s2, normal_mode_1, normal_mode_2)
            wavefunction_amplitudes += state_amplitudes[j] * ho_2d.T
        return wavefunction_amplitudes
