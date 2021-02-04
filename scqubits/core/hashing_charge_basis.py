import itertools
from functools import reduce

import numpy as np
from numpy import ndarray
from scipy.sparse import eye
from scipy.sparse.coo import coo_matrix

from scqubits.core.hashing import Hashing
from scqubits.core.vtbbasemethods import reflect_vectors


class HashingChargeBasis(Hashing):
    """
    Allow for a global charge number cutoff in the charge basis
    """
    def __init__(self) -> None:
        Hashing.__init__(self)

    def gen_basis_vectors(self):
        return self._gen_basis_vectors(reflect_vectors)

    def n_operator(self, j: int = 0) -> ndarray:
        basis_vectors = self.gen_basis_vectors()
        tags, index_array = self._gen_tags(basis_vectors)
        num_states = len(tags)
        row, data = [], []
        for w, vector in enumerate(basis_vectors):
            if vector[j] != 0:
                row.append(w)
                data.append(vector[j])
        return coo_matrix((data, (row, row)), shape=(num_states, num_states), dtype=np.complex_).tocsr()

    def exp_i_phi_j_operator(self, j: int = 0) -> ndarray:
        basis_vectors = self.gen_basis_vectors()
        tags, index_array = self._gen_tags(basis_vectors)
        num_states = len(tags)
        row, col, data = [], [], []
        for w, vector in enumerate(basis_vectors):
            basis_index = self._find_lowered_vector(vector, j, tags, index_array)
            if basis_index is not None:
                row.append(basis_index)
                col.append(w)
                data.append(1.0)
        return coo_matrix((data, (row, col)), shape=(num_states, num_states), dtype=np.complex_).tocsr()

    def identity_operator(self) -> ndarray:
        basis_vectors = self.gen_basis_vectors()
        num_states = len(basis_vectors)
        return eye(num_states, k=0, format="csr", dtype=np.complex_)

    def exp_i_phi_stitching_term(self) -> ndarray:
        dim = self.number_degrees_freedom
        return reduce((lambda x, y: x @ y), np.array([self.exp_i_phi_j_operator(j) for j in range(dim)]))


class ChargeBasisLinearOperator(Hashing):
    """
    Use the regular charge basis, but use LinearOperator to not
    implement the entire Hamiltonian
    """

    def __init__(self, ncut, number_degrees_freedom) -> None:
        self.ncut = ncut
        self._number_degrees_freedom = number_degrees_freedom
        Hashing.__init__(self)
        self.basis_vecs = self.gen_basis_vectors()
        self.tags, self.index_array = self._gen_tags(self.basis_vecs)

    def gen_basis_vectors(self):
        basis_vecs = itertools.product(np.arange(-self.ncut, self.ncut + 1, 1), repeat=self.number_degrees_freedom)
        np_basis_vecs = np.array(list(map(np.array, basis_vecs)))
        return np_basis_vecs

    def n_operator(self, j: int, vec: ndarray) -> ndarray:
        basis_vectors = self.basis_vecs
        new_vec = np.zeros_like(vec)
        for k, vec_val in enumerate(vec):
            corresponding_basis_vec = basis_vectors[k]
            new_vec[k] += vec_val * corresponding_basis_vec[j]
        return new_vec

    def exp_i_phi_j_operator(self, j: int, vec: ndarray) -> ndarray:
        basis_vectors = self.basis_vecs
        tags, index_array = self.tags, self.index_array
        new_vec = np.zeros_like(vec)
        for k, vec_val in enumerate(vec):
            corresponding_basis_vec = basis_vectors[k]
            if corresponding_basis_vec[j] != -self.ncut:
                basis_index = self._find_lowered_vector(corresponding_basis_vec, j, tags, index_array)
                if basis_index is not None:
                    new_vec[basis_index] += vec_val
        return new_vec

    def exp_m_i_phi_j_operator(self, j: int, vec: ndarray) -> ndarray:
        basis_vectors = self.basis_vecs
        tags, index_array = self.tags, self.index_array
        new_vec = np.zeros_like(vec)
        for k, vec_val in enumerate(vec):
            corresponding_basis_vec = basis_vectors[k]
            if corresponding_basis_vec[j] != self.ncut:
                basis_index = self._find_lowered_vector(corresponding_basis_vec, j, tags, index_array,
                                                        raised_or_lowered="raised")
                if basis_index is not None:
                    new_vec[basis_index] += vec_val
        return new_vec

    def exp_i_phi_stitching_term(self, vec: ndarray) -> ndarray:
        for j in range(self.number_degrees_freedom):
            vec = self.exp_i_phi_j_operator(j, vec)
        return vec

    def exp_m_i_phi_stitching_term(self, vec: ndarray) -> ndarray:
        for j in range(self.number_degrees_freedom):
            vec = self.exp_m_i_phi_j_operator(j, vec)
        return vec

    def identity_operator(self, vec: ndarray) -> ndarray:
        return vec