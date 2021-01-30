import itertools
from functools import partial, reduce
from typing import Tuple, Callable

import numpy as np
import scipy as sp
from numpy import ndarray
from numpy.linalg import matrix_power
from scipy.linalg import LinAlgError, inv, expm, logm, det
from scipy.optimize import minimize
from scipy.special import factorial

from scqubits.core.vtbbasemethods import VTBBaseMethods


class VTBBaseMethodsSqueezing(VTBBaseMethods):
    r""" VariationalTightBinding allowing for squeezing

    See class VariationalTightBinding for documentation and explanation of parameters.

    """

    def _U_squeezing_operator(self, minimum: int, Xi: ndarray) -> Tuple:
        """
        Return the rho, sigma, tau matrices that define the overall squeezing operator U

        Parameters
        ----------
        minimum: int
            integer representing the minimum for which to build the squeezing operator U,
            0<i<=total number of minima (no squeezing need be performed for the global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times

        Returns
        -------
        ndarray, ndarray, ndarray
        """
        Xi_prime = self.Xi_matrix(minimum)
        M_matrix = self._squeezing_M(minimum, Xi, Xi_prime)
        dim = self.number_degrees_freedom
        u = M_matrix[0: dim, 0: dim]
        v = M_matrix[dim: 2*dim, 0: dim]
        rho = inv(u) @ v
        sigma = logm(u)
        tau = v @ inv(u)
        return rho, sigma, tau
    
    def _delta_rho_matrices(self, rho: ndarray, rho_prime: ndarray) -> Tuple:
        dim = self.number_degrees_freedom
        delta_rho_prime = inv(np.eye(dim) - rho_prime @ rho) @ rho_prime
        delta_rho = inv(np.eye(dim) - rho @ rho_prime) @ rho
        delta_rho_bar = logm(inv(np.eye(dim) - rho_prime @ rho))
        return delta_rho, delta_rho_prime, delta_rho_bar

    @staticmethod
    def _linear_coefficient_matrices(rho_prime: ndarray, delta_rho: ndarray,
                                     A: ndarray, B: ndarray) -> Tuple:
        """Build variables helpful for constructing the Hamiltonian """
        a_coefficient = A - 0.5 * (B - A @ rho_prime) @ (delta_rho + delta_rho.T)
        a_dagger_coefficient = B - A @ rho_prime
        return a_coefficient, a_dagger_coefficient

    def _squeezing_M(self, minimum: int, Xi: ndarray, Xi_prime: ndarray) -> ndarray:
        """
        Returns the M matrix as defined in G. Qin et. al “General multi-mode-squeezed states,”
        (2001) arXiv: quant-ph/0109020, M=[[u, v],[v, u]] where u and v are the matrices
        that define the Bogoliubov transformation
        Parameters
        ----------
        minimum: int
            integer representing the minimum for which to build the squeezing operator U,
            0<i<=total number of minima (no squeezing need be performed for the global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times
        Xi_prime: ndarray
            Xi matrix for the non-global minimum that requires squeezing

        Returns
        -------
        ndarray
        """
        omega_squared, _ = self.eigensystem_normal_modes(minimum)
        omega_matrix = np.diag(np.sqrt(omega_squared))
        Xi_inv = inv(Xi)
        Xi_prime_inv = inv(Xi_prime)
        kinetic_matrix = Xi_inv @ Xi_prime @ omega_matrix @ Xi_prime.T @ Xi_inv.T
        potential_matrix = Xi.T @ Xi_prime_inv.T @ omega_matrix @ Xi_prime_inv @ Xi
        zeta = 0.25 * (potential_matrix + kinetic_matrix)
        eta = 0.25 * (potential_matrix - kinetic_matrix)
        H_matrix = np.block([[zeta, -eta],
                             [eta, -zeta]])
        eigvals, eigvec = sp.linalg.eig(H_matrix)
        eigvals, eigvec = self._order_eigensystem_squeezing(np.real(eigvals), eigvec)
        eigvec = eigvec.T  # since eigvec represents M.T
        # Normalization ensures that eigvec.T K eigvec = K, K = [[1, 0],[0, -1]] (1, 0 are matrices)
        _, eigvec = self._normalize_symplectic_eigensystem_squeezing(eigvals, eigvec)
        return eigvec

    def _order_eigensystem_squeezing(self, eigvals: ndarray, eigvec: ndarray) -> Tuple:
        """Order eigensystem to have positive eigenvalues followed by negative, in same order"""
        dim = self.number_degrees_freedom
        eigval_holder = np.zeros(dim)
        eigvec_holder = np.zeros_like(eigvec)
        count = 0
        for k, eigval in enumerate(eigvals):
            if eigval > 0:
                eigval_holder[count] = eigval
                eigvec_holder[:, count] = eigvec[:, k]
                count += 1
        index_array = np.argsort(eigval_holder)
        eigval_holder = eigval_holder[index_array]
        eigvec_holder[:, 0: dim] = eigvec_holder[:, index_array]
        # Now attempt to deal with degenerate modes
        for k in range(0, len(eigval_holder) - 1):
            if np.allclose(eigval_holder[k], eigval_holder[k + 1], atol=1e-6):
                evec_1 = eigvec_holder[:, k]
                evec_2 = eigvec_holder[:, k + 1]
                mat = np.array([[evec_1[k], evec_1[k + 1]],  # Assume maximal elements are same as global min
                                [evec_2[k], evec_2[k + 1]]])
                sol = inv(mat)  # Find linear transformation to get (1, 0) and (0, 1) vectors
                new_evec_1 = sol[0, 0] * evec_1 + sol[0, 1] * evec_2
                new_evec_2 = sol[1, 0] * evec_1 + sol[1, 1] * evec_2
                eigvec_holder[:, k] = new_evec_1
                eigvec_holder[:, k + 1] = new_evec_2
        u = eigvec_holder[0: dim, 0: dim]
        v = eigvec_holder[dim: 2*dim, 0: dim]
        eigvec_holder[0: dim, dim: 2*dim] = v
        eigvec_holder[dim: 2*dim, dim: 2*dim] = u
        return eigval_holder, eigvec_holder

    def _normalize_symplectic_eigensystem_squeezing(self, eigvals: ndarray, eigvec: ndarray) -> Tuple:
        """Enforce commutation relations so that Bogoliubov transformation is symplectic """
        dim = self.number_degrees_freedom
        for col in range(dim):
            a = np.sum([eigvec[row, col] for row in range(2*dim)])
            if a < 0.0:
                eigvec[:, col] *= -1
        A = eigvec[0: dim, 0: dim]
        B = eigvec[dim: 2*dim, 0: dim]
        for vec in range(dim):
            a = 1. / np.sqrt(np.sum([A[num, vec] * A[num, vec] - B[num, vec] * B[num, vec]
                                     for num in range(dim)]))
            eigvec[:, vec] *= a
        A = eigvec[0: dim, 0: dim]
        B = eigvec[dim: 2*dim, 0: dim]
        eigvec[dim: 2*dim, dim: 2*dim] = A
        eigvec[0: dim, dim: 2*dim] = B
        return eigvals, eigvec

    def _find_closest_periodic_minimum(self,
                                       minima_index_pair: Tuple
                                       ) -> float:
        """Overrides method in VariationalTightBinding, need to consider states localized in both minima."""
        (m, minima_m), (p, minima_p) = minima_index_pair
        max_for_m = self._max_localization_ratio_for_minima_pair(minima_index_pair, m)
        max_for_p = self._max_localization_ratio_for_minima_pair(minima_index_pair, p)
        return max(max_for_m, max_for_p)

    def _normal_ordered_a_dagger_a_exponential(self, x: ndarray, a_operator_array: ndarray) -> ndarray:
        """Return normal ordered exponential matrix of exp(a_{i}^{\dagger}x_{ij}a_{j})"""
        expm_x = expm(x)
        num_states = self.number_states_per_minimum()
        dim = self.number_degrees_freedom
        result = np.eye(num_states, dtype=np.complex128)
        additional_term = np.eye(num_states, dtype=np.complex128)
        k = 1
        while not np.allclose(additional_term, np.zeros((num_states, num_states))):
            additional_term = np.sum([((expm_x - np.eye(dim))[i, j]) ** k * (factorial(k)) ** (-1)
                                     * matrix_power(a_operator_array[i].T, k) @ matrix_power(a_operator_array[j], k)
                                     for i in range(dim) for j in range(dim)], axis=0)
            result += additional_term
            k += 1
        return result

    def _rho_sigma_tau_matrices(self, m: int, p: int, Xi: ndarray) -> Tuple:
        """Return the `\rho, \sigma, \tau` matrices that define the squeezing operator `U`."""
        dim = self.number_degrees_freedom
        if m == 0:  # At the global minimum, no squeezing required
            rho = np.zeros((dim, dim))
            sigma = np.zeros((dim, dim))
            tau = np.zeros((dim, dim))
        else:
            rho, sigma, tau = self._U_squeezing_operator(m, Xi)
        if p == 0:
            rho_prime = np.zeros((dim, dim))
            sigma_prime = np.zeros((dim, dim))
            tau_prime = np.zeros((dim, dim))
        elif p == m:
            rho_prime = np.copy(rho)
            sigma_prime = np.copy(sigma)
            tau_prime = np.copy(tau)
        else:
            rho_prime, sigma_prime, tau_prime = self._U_squeezing_operator(p, Xi)
        return rho, rho_prime, sigma, sigma_prime, tau, tau_prime

    def _translation_operator_prefactors(self, disentangled_squeezing_matrices: Tuple,
                                         delta_rho_matrices: Tuple):
        """Helper method for building the translation operator argument prefactors"""
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        prefactor_a_dagger = (np.eye(dim) + rho_prime) @ expm(delta_rho_bar).T @ expm(-sigma)
        prefactor_a = (np.eye(dim) + 0.5 * (np.eye(dim) + rho_prime) @ (delta_rho + delta_rho.T)) @ expm(-sigma_prime)
        return prefactor_a, prefactor_a_dagger

    def _general_translation_operators(self, minima_diff: ndarray, Xi: ndarray,
                                         disentangled_squeezing_matrices: Tuple,
                                         delta_rho_matrices: Tuple) -> Tuple:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. The resulting matrices yield a 2pi translation
        in each degree of freedom, so that any translation can be built from these by an appropriate
        call to np.matrix_power"""
        dim = self.number_degrees_freedom
        num_states_per_min = self.number_states_per_minimum()
        prefactor_a, prefactor_a_dagger = self._translation_operator_prefactors(disentangled_squeezing_matrices,
                                                                                delta_rho_matrices)
        a_operator_array = self._a_operator_array()
        Xi_inv = inv(Xi)
        exp_a_list = np.zeros((dim, num_states_per_min, num_states_per_min), dtype=np.complex_)
        exp_a_dagger_list = np.zeros_like(exp_a_list)
        for i in range(dim):
            exp_a_dagger_list[i] = expm(np.sum(2.0 * np.pi * (Xi_inv.T[i] @ prefactor_a_dagger) * a_operator_array.T,
                                               axis=2) / np.sqrt(2.0))
            exp_a_list[i] = expm(np.sum(2.0 * np.pi * (Xi_inv.T[i] @ prefactor_a)
                                     * np.transpose(a_operator_array, axes=(1, 2, 0)), axis=2) / np.sqrt(2.0))
        return exp_a_list, exp_a_dagger_list

    def _minima_dependent_translation_operators(self, minima_diff: ndarray, Xi: ndarray,
                                         disentangled_squeezing_matrices: Tuple,
                                         delta_rho_matrices: Tuple) -> Tuple:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. This part of the translation operator accounts
        for the differing location of minima within a single unit cell."""
        dim = self.number_degrees_freedom
        a_operator_array = self._a_operator_array()
        prefactor_a, prefactor_a_dagger = self._translation_operator_prefactors(disentangled_squeezing_matrices,
                                                                                delta_rho_matrices)
        Xi_inv = inv(Xi)
        exp_a_dagger_minima_difference = expm(np.sum(minima_diff @ Xi_inv.T @ prefactor_a_dagger * a_operator_array.T,
                                                      axis=2) / np.sqrt(2.0))
        exp_a_minima_difference = expm(np.sum(-minima_diff @ Xi_inv.T @ prefactor_a
                                              * np.transpose(a_operator_array, axes=(1, 2, 0)), axis=2) / np.sqrt(2.0))
        return exp_a_minima_difference, exp_a_dagger_minima_difference

    def _potential_operators_squeezing(self, a_operator_array: ndarray, Xi: ndarray, exp_a_dagger_a: ndarray,
                                       disentangled_squeezing_matrices: Tuple,
                                       delta_rho_matrices: Tuple) -> Tuple:
        """Helper method for building the potential operators."""
        exp_i_list = []
        dim = self.number_degrees_freedom
        prefactor_a, prefactor_a_dagger = self._potential_exp_prefactors(disentangled_squeezing_matrices,
                                                                         delta_rho_matrices)
        for j in range(dim):
            exp_i_j_a_dagger_part = expm(np.sum(1j * (Xi[j] @ prefactor_a_dagger) * a_operator_array.T,
                                                axis=2) / np.sqrt(2.0))
            exp_i_j_a_part = expm(np.sum(1j * (Xi[j] @ prefactor_a) * np.transpose(a_operator_array, axes=(1, 2, 0)),
                                         axis=2) / np.sqrt(2.0))
            exp_i_j = exp_i_j_a_dagger_part @ exp_a_dagger_a @ exp_i_j_a_part
            exp_i_list.append(exp_i_j)

        exp_i_sum_a_dagger_part = expm(np.sum(1j * self.stitching_coefficients @ Xi @ prefactor_a_dagger
                                              * a_operator_array.T, axis=2) / np.sqrt(2.0))
        exp_i_sum_a_part = expm(np.sum(1j * self.stitching_coefficients @ Xi @ prefactor_a
                                       * np.transpose(a_operator_array, axes=(1, 2, 0)), axis=2) / np.sqrt(2.0))
        exp_i_sum = exp_i_sum_a_dagger_part @ exp_a_dagger_a @ exp_i_sum_a_part
        return exp_i_list, exp_i_sum

    def _bilinear_squeezing_operators(self, a_operator_array: ndarray,
                                      disentangled_squeezing_matrices: Tuple,
                                      delta_rho_matrices: Tuple) -> Tuple:
        """Helper method for building the bilinear operators necessary for constructing the Hamiltonian
        in the presence of squeezing."""
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        prefactor_a_dagger_a_dagger = 0.5 * (tau.T - expm(-sigma).T @ delta_rho_prime @ expm(-sigma))
        prefactor_a_a = 0.5 * (tau_prime - expm(-sigma_prime).T @ delta_rho @ expm(-sigma_prime))
        prefactor_a_dagger_a = sp.linalg.logm(expm(-sigma).T @ expm(delta_rho_bar) @ expm(-sigma_prime))

        exp_a_dagger_a_dagger = expm(np.sum([prefactor_a_dagger_a_dagger[i, j]
                                             * a_operator_array[i].T @ a_operator_array[j].T
                                     for i in range(dim) for j in range(dim)], axis=0))
        exp_a_a = expm(np.sum([prefactor_a_a[i, j] * a_operator_array[i] @ a_operator_array[j]
                               for i in range(dim) for j in range(dim)], axis=0))
        exp_a_dagger_a = self._normal_ordered_a_dagger_a_exponential(prefactor_a_dagger_a, a_operator_array)
        return exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a

    def _local_translation_operators(self, exp_operators: Tuple, squeezing_operators: Tuple,
                                     neighbor: ndarray) -> Tuple:
        """Build translation operators using matrix_power"""
        dim = self.number_degrees_freedom
        (exp_a_list, exp_a_dagger_list), (exp_a_minima_difference, exp_a_dagger_minima_difference) = exp_operators
        exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
        individual_op_a_dagger = np.array([matrix_power(exp_a_dagger_list[j], int(neighbor[j])) for j in range(dim)])
        individual_op_a = np.array([matrix_power(exp_a_list[j], -int(neighbor[j])) for j in range(dim)])
        translation_op_a_dag = (reduce((lambda x, y: x @ y), individual_op_a_dagger)
                                @ exp_a_dagger_minima_difference @ exp_a_dagger_a_dagger)
        translation_op_a = reduce((lambda x, y: x @ y), individual_op_a) @ exp_a_minima_difference @ exp_a_a
        return translation_op_a_dag, translation_op_a

    def _periodic_continuation(self, minima_pair_func: Callable, local_func: Callable) -> ndarray:
        """See VariationalTightBinding for documentation. This function generalizes
        _periodic_continuation to allow for squeezing"""
        if not self.retained_unit_cell_displacement_vectors:
            self.find_relevant_unit_cell_vectors()
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        sorted_minima_dict = self.sorted_minima
        hilbertdim = self.hilbertdim()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        all_minima_pairs = itertools.combinations_with_replacement(sorted_minima_dict, 2)
        for m, p in all_minima_pairs:
            minima_diff = sorted_minima_dict[p] - sorted_minima_dict[m]
            retained_unit_cell_displacement_vectors = self.retained_unit_cell_displacement_vectors[(m, p)]
            disentangled_squeezing_matrices = self._rho_sigma_tau_matrices(m, p, Xi)
            rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
            delta_rho_matrices = self._delta_rho_matrices(rho, rho_prime)
            squeezing_operators = self._bilinear_squeezing_operators(a_operator_array,
                                                                     disentangled_squeezing_matrices,
                                                                     delta_rho_matrices)
            exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
            exp_operators = (self._general_translation_operators(minima_diff, Xi, disentangled_squeezing_matrices,
                                                                 delta_rho_matrices),
                             self._minima_dependent_translation_operators(minima_diff, Xi,
                                                                          disentangled_squeezing_matrices,
                                                                          delta_rho_matrices))
            minima_pair_results = minima_pair_func(exp_a_dagger_a, disentangled_squeezing_matrices, delta_rho_matrices)
            scale = 1. / np.sqrt(det(np.eye(self.number_degrees_freedom) - np.matmul(rho, rho_prime)))
            matrix_element = self._periodic_continuation_for_minima_pair(sorted_minima_dict[m], sorted_minima_dict[p],
                                                                         retained_unit_cell_displacement_vectors, local_func,
                                                                         squeezing_operators, exp_operators,
                                                                         disentangled_squeezing_matrices,
                                                                         delta_rho_matrices,
                                                                         minima_pair_results, Xi_inv)
            operator_matrix[m * num_states_min: (m + 1) * num_states_min,
                            p * num_states_min: (p + 1) * num_states_min] += matrix_element * scale
        operator_matrix = self._populate_hermitian_matrix(operator_matrix)
        return operator_matrix

    def _periodic_continuation_for_minima_pair(self, minima_m: ndarray, minima_p: ndarray,
                                               retained_unit_cell_displacement_vectors: ndarray,
                                               local_func: Callable, squeezing_operators: Tuple, exp_operators: Tuple,
                                               disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple,
                                               minima_pair_results: Tuple, Xi_inv: ndarray):
        num_states = self.number_states_per_minimum()
        matrix_element = np.zeros((num_states, num_states), dtype=np.complex_)
        if retained_unit_cell_displacement_vectors is not None:
            rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
            exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
            for neighbor in retained_unit_cell_displacement_vectors:
                phi_neighbor = 2.0 * np.pi * np.array(neighbor)
                translation_operators = self._local_translation_operators(exp_operators, squeezing_operators, neighbor)
                translation_a_dagger, translation_a = translation_operators
                exp_prod_coefficient = self._exp_product_coefficient_squeezing(phi_neighbor + minima_p - minima_m,
                                                                               Xi_inv, sigma, sigma_prime)
                matrix_element += (exp_prod_coefficient * translation_a_dagger
                                   @ local_func(phi_neighbor, minima_m, minima_p,
                                                disentangled_squeezing_matrices, delta_rho_matrices,
                                                exp_a_dagger_a, minima_pair_results)
                                   @ translation_a)
        return matrix_element

    def _construct_kinetic_alpha_epsilon_squeezing(self, Xi_inv: ndarray, delta_phi: ndarray,
                                                   rho_prime: ndarray, delta_rho: ndarray) -> Tuple:
        """Construct the `alpha` and `epsilon` variables necessary for the kinetic matrix."""
        arg_exp_a_dag = delta_phi @ Xi_inv.T / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho)
        delta_rho_pp = 0.5 * (arg_exp_a_dag - arg_exp_a @ rho_prime) @ (delta_rho + delta_rho.T)
        epsilon = -(1j / np.sqrt(2.0)) * Xi_inv.T @ (rho_prime @ delta_rho_pp - arg_exp_a @ rho_prime + delta_rho_pp
                                                     + Xi_inv @ delta_phi / np.sqrt(2.0))
        return alpha, epsilon

    def _minima_pair_transfer_squeezing_function(self, EC_mat: ndarray, a_operator_array: ndarray, Xi: ndarray,
                                                 exp_a_dagger_a: ndarray,
                                                 disentangled_squeezing_matrices: Tuple,
                                                 delta_rho_matrices: Tuple) -> Tuple:
        """Minima pair calculations for the kinetic and potential matrices."""
        return (self._minima_pair_kinetic_squeezing_function(EC_mat, a_operator_array, inv(Xi), exp_a_dagger_a,
                                                             disentangled_squeezing_matrices,
                                                             delta_rho_matrices),
                self._minima_pair_potential_squeezing_function(a_operator_array, Xi, exp_a_dagger_a,
                                                               disentangled_squeezing_matrices,
                                                               delta_rho_matrices))

    def _minima_pair_kinetic_squeezing_function(self, EC_mat: ndarray, a_operator_array: ndarray,
                                                Xi_inv: ndarray, exp_a_dagger_a: ndarray,
                                                disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple
                                                ) -> Tuple:
        """Return data necessary for constructing the kinetic matrix that only depends on the minima
        pair, and not on the specific periodic continuation operator."""
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        linear_coefficients_kinetic = self._linear_coefficient_matrices(rho_prime, delta_rho,
                                                                        -1j * Xi_inv.T / np.sqrt(2.0),
                                                                        1j * Xi_inv.T / np.sqrt(2.0))
        a_coefficient, a_dagger_coefficient = linear_coefficients_kinetic
        (xa, xaa, dxa, dx, ddx) = self._premultiplying_exp_a_dagger_a_with_a(exp_a_dagger_a, a_operator_array)
        sigma_delta_rho_bar_a_dagger_coefficient_EC = (expm(-sigma).T @ expm(delta_rho_bar)
                                                       @ a_dagger_coefficient.T @ EC_mat)
        xaa_coefficient = (a_coefficient @ expm(-sigma_prime)).T @ EC_mat @ a_coefficient @ expm(-sigma_prime)
        dxa_coefficient = sigma_delta_rho_bar_a_dagger_coefficient_EC @ a_coefficient @ expm(-sigma_prime)
        ddx_coefficient = (sigma_delta_rho_bar_a_dagger_coefficient_EC @ (expm(-sigma).T @ expm(delta_rho_bar)
                                                                          @ a_dagger_coefficient.T).T)
        x_coefficient = a_dagger_coefficient.T @ EC_mat @ a_coefficient
        xa_coefficient = EC_mat @ a_coefficient @ expm(-sigma_prime)
        dx_coefficient = EC_mat @ a_dagger_coefficient @ (expm(-sigma).T @ expm(delta_rho_bar)).T
        kinetic_matrix = np.sum(+ 4 * np.transpose(xaa, axes=(1, 2, 0)) * np.diag(xaa_coefficient)
                                + 8 * np.transpose(dxa, axes=(1, 2, 0)) * np.diag(dxa_coefficient)
                                + 4 * np.transpose(ddx, axes=(1, 2, 0)) * np.diag(ddx_coefficient), axis=2)
        kinetic_matrix += 4 * exp_a_dagger_a * np.sum(np.diag(x_coefficient))
        return kinetic_matrix, xa, dx, xa_coefficient, dx_coefficient

    def _local_transfer_squeezing_function(self, EC_mat: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                           phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                           disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple,
                                           exp_a_dagger_a: ndarray, minima_pair_results: Tuple) -> ndarray:
        """Local contribution to the transfer matrix in the presence of squeezing."""
        kinetic_minima_pair_results, potential_minima_pair_results = minima_pair_results
        return (self._local_kinetic_squeezing_function(EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                       disentangled_squeezing_matrices, delta_rho_matrices,
                                                       exp_a_dagger_a, kinetic_minima_pair_results)
                + self._local_potential_squeezing_function(Xi, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                           disentangled_squeezing_matrices,
                                                           delta_rho_matrices, exp_a_dagger_a,
                                                           potential_minima_pair_results))

    def _local_kinetic_squeezing_function(self, EC_mat: ndarray, Xi_inv: ndarray,
                                          phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                          disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple,
                                          exp_a_dagger_a: ndarray, minima_pair_results: Tuple) -> ndarray:
        """Local contribution to the kinetic matrix in the presence of squeezing."""
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        kinetic_matrix_minima_pair, xa, dx, xa_coefficient, dx_coefficient = minima_pair_results
        alpha, epsilon = self._construct_kinetic_alpha_epsilon_squeezing(Xi_inv, delta_phi, rho_prime, delta_rho)
        e_xa_coefficient = epsilon @ xa_coefficient
        e_dx_coefficient = epsilon @ dx_coefficient
        return alpha * (np.sum(8 * np.transpose(xa, axes=(1, 2, 0)) * e_xa_coefficient
                               + 8 * np.transpose(dx, axes=(1, 2, 0)) * e_dx_coefficient, axis=2)
                        + kinetic_matrix_minima_pair + 4 * exp_a_dagger_a * (epsilon @ EC_mat @ epsilon))

    @staticmethod
    def _premultiplying_exp_a_dagger_a_with_a(exp_a_dagger_a: ndarray, a_operator_array: ndarray) -> Tuple:
        """
        Helper function for building the kinetic part of the Hamiltonian.
        Naming scheme is  x -> exp(A_{ij}a_{i}^{\dag}a_{j}) (for whatever matrix A is)
                          a -> a_{i}
                          d -> a_{i}^{\dag}
        """
        xa = exp_a_dagger_a @ a_operator_array
        xaa = xa @ a_operator_array
        a_dagger_operator_array = np.transpose(a_operator_array, axes=(0, 2, 1))
        dxa = a_dagger_operator_array @ xa
        dx = a_dagger_operator_array @ exp_a_dagger_a
        ddx = a_dagger_operator_array @ dx
        return xa, xaa, dxa, dx, ddx

    @staticmethod
    def _alpha_helper(arg_exp_a_dag: ndarray, arg_exp_a: ndarray, rho_prime: ndarray, delta_rho: ndarray) -> ndarray:
        """Build the prefactor that arises due to squeezing. With no squeezing, alpha=1 (number, not matrix)"""
        arg_exp_a_rho_prime = np.matmul(arg_exp_a, rho_prime)
        alpha = np.exp(-0.5 * arg_exp_a @ arg_exp_a_rho_prime - 0.5 * (arg_exp_a_dag - arg_exp_a_rho_prime)
                       @ delta_rho @ (arg_exp_a_dag - arg_exp_a_rho_prime))
        return alpha

    def transfer_matrix(self) -> ndarray:
        """Returns the transfer matrix

        Returns
        -------
        ndarray
        """
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        EC_mat = self.EC_matrix()
        minima_pair_transfer_function = partial(self._minima_pair_transfer_squeezing_function, EC_mat,
                                                a_operator_array, Xi)
        local_transfer_function = partial(self._local_transfer_squeezing_function, EC_mat, Xi, Xi_inv)
        return self._periodic_continuation(minima_pair_transfer_function, local_transfer_function)

    def kinetic_matrix(self) -> ndarray:
        """Returns the kinetic energy matrix

        Returns
        -------
        ndarray
        """
        Xi_inv = inv(self.Xi_matrix())
        a_operator_array = self._a_operator_array()
        EC_mat = self.EC_matrix()
        minima_pair_kinetic_function = partial(self._minima_pair_kinetic_squeezing_function, EC_mat,
                                               a_operator_array, Xi_inv)
        local_kinetic_function = partial(self._local_kinetic_squeezing_function, EC_mat, Xi_inv)
        return self._periodic_continuation(minima_pair_kinetic_function, local_kinetic_function)

    def potential_matrix(self) -> ndarray:
        """Returns the potential energy matrix

        Returns
        -------
        ndarray
        """
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        minima_pair_potential_function = partial(self._minima_pair_potential_squeezing_function, a_operator_array, Xi)
        local_potential_function = partial(self._local_potential_squeezing_function, Xi, Xi_inv)
        return self._periodic_continuation(minima_pair_potential_function, local_potential_function)

    def _potential_exp_prefactors(self, disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple
                                  ) -> Tuple:
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        prefactor_a_dagger = (np.eye(dim) - rho_prime) @ expm(delta_rho_bar).T @ expm(-sigma)
        prefactor_a = (np.eye(dim) - 0.5 * (np.eye(dim) - rho_prime) @ (delta_rho + delta_rho.T)) @ expm(-sigma_prime)
        return prefactor_a, prefactor_a_dagger

    def _local_potential_squeezing_function(self, Xi: ndarray, Xi_inv: ndarray, phi_neighbor: ndarray,
                                            minima_m: ndarray, minima_p: ndarray,
                                            disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple,
                                            exp_a_dagger_a: ndarray, minima_pair_results: Tuple) -> ndarray:
        """Local contribution to the potential matrix in the presence of squeezing."""
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_list, exp_i_sum = minima_pair_results
        exp_i_phi_list = np.array([exp_i_list[i] * np.exp(1j * phi_bar[i]) for i in range(dim)])
        exp_i_phi_sum_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                            * np.exp(np.sum(1j * self.stitching_coefficients * phi_bar)))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      delta_rho_matrices,
                                                                                      exp_i_phi_list)
                                   for j in range(dim)], axis=0)
        potential_matrix += self._local_contribution_stitching_squeezing(delta_phi, Xi, Xi_inv,
                                                                         disentangled_squeezing_matrices,
                                                                         delta_rho_matrices,
                                                                         exp_i_phi_sum_op)
        potential_matrix += (self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                         disentangled_squeezing_matrices,
                                                                         delta_rho_matrices, exp_a_dagger_a,
                                                                         minima_pair_results)
                             * np.sum(self.EJlist))
        return potential_matrix

    def _local_contribution_stitching_squeezing(self, delta_phi: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                                disentangled_squeezing_matrices: Tuple,
                                                delta_rho_matrices: Tuple, exp_i_sum: ndarray) -> ndarray:
        """Local contribution to the potential due to the stitching term"""
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + np.sum([1j * Xi[i, :] * self.stitching_coefficients[i]
                                                     for i in range(dim)], axis=0)) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rho_prime, delta_rho)
        potential_matrix = -0.5 * self.EJlist[-1] * (alpha * exp_i_sum + (alpha * exp_i_sum).conj())
        potential_matrix *= self._BCH_factor_for_potential_stitching(Xi)
        return potential_matrix

    def _local_contribution_single_junction_squeezing(self, j: int, delta_phi: ndarray, Xi: ndarray,
                                                      Xi_inv: ndarray, disentangled_squeezing_matrices: Tuple,
                                                      delta_rho_matrices: Tuple, exp_i_phi_list: ndarray
                                                      ) -> ndarray:
        """Local contribution to the potential due to `\cos(\phi_j)`"""
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + 1j * Xi[j, :]) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rho_prime, delta_rho)
        potential_matrix = -0.5 * self.EJlist[j] * (alpha * exp_i_phi_list[j] + (alpha * exp_i_phi_list[j]).conj())
        potential_matrix *= np.exp(-.25 * np.dot(Xi[j, :], Xi.T[:, j]))
        return potential_matrix

    def _minima_pair_potential_squeezing_function(self, a_operator_array: ndarray, Xi: ndarray,
                                                  exp_a_dagger_a: ndarray,
                                                  disentangled_squeezing_matrices: Tuple,
                                                  delta_rho_matrices: Tuple) -> Tuple:
        """Return data necessary for constructing the potential matrix that only depends on the minima
        pair, and not on the specific periodic continuation operator."""
        return self._potential_operators_squeezing(a_operator_array, Xi, exp_a_dagger_a,
                                                   disentangled_squeezing_matrices, delta_rho_matrices)

    def _local_contribution_identity_squeezing(self, Xi_inv: ndarray, phi_neighbor: ndarray,
                                               minima_m: ndarray, minima_p: ndarray,
                                               disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple,
                                               exp_a_dagger_a: ndarray, minima_pair_results: Tuple
                                               ) -> ndarray:
        """Local contribution to the identity matrix in the presence of squeezing."""
        _ = minima_pair_results
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        arg_exp_a_dag = np.matmul(delta_phi, Xi_inv.T) / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho)
        return alpha * exp_a_dagger_a

    def inner_product_matrix(self) -> ndarray:
        """Returns the inner product matrix

        Returns
        -------
        ndarray
        """
        Xi_inv = inv(self.Xi_matrix())
        local_identity_function = partial(self._local_contribution_identity_squeezing, Xi_inv)
        return self._periodic_continuation(lambda x, y, z: None, local_identity_function)

    def _exp_product_coefficient_squeezing(self, delta_phi: ndarray, Xi_inv: ndarray,
                                           sigma: ndarray, sigma_prime: ndarray) -> ndarray:
        """Overall multiplicative factor. Includes offset charge, Gaussian suppression factor in the
        absence of squeezing. With squeezing, also includes exponential of trace over
        sigma and sigma_prime, see Qin et. al"""
        return (np.exp(-0.5 * np.trace(sigma) - 0.5 * np.trace(sigma_prime))
                * self._exp_product_coefficient(delta_phi, Xi_inv))

    def _optimize_harmonic_lengths(self) -> None:
        """Overrides method in VariationalTightBinding. Allows for harmonic length optimization of states localized
        in all minima if the optimize_all_minima flag is set. Optimize the Xi matrix by adjusting
        the harmonic lengths of the ground state to minimize its energy.
        For tight-binding without squeezing, this is only done for the ansatz ground state wavefunction
        localized in the global minimum."""
        sorted_minima_dict = self.sorted_minima
        self.optimized_lengths = np.ones((len(sorted_minima_dict), self.number_degrees_freedom))
        self._optimize_harmonic_lengths_given_minimum(0, sorted_minima_dict[0])
        Xi_global = self.Xi_matrix(minimum_index=0)
        harmonic_lengths_global = np.array([np.linalg.norm(Xi_global[:, i])
                                            for i in range(self.number_degrees_freedom)])
        for m, minimum in sorted_minima_dict.items():
            if self.optimize_all_minima and m != 0:
                self._optimize_harmonic_lengths_given_minimum(m, minimum)
            elif m != 0:
                Xi_local = self.Xi_matrix(minimum_index=m)
                harmonic_lengths_local = np.array([np.linalg.norm(Xi_local[:, i])
                                                   for i in range(self.number_degrees_freedom)])
                self.optimized_lengths[m] = harmonic_lengths_global / harmonic_lengths_local

    def _optimize_harmonic_lengths_given_minimum(self, minimum_index: int = 0, minimum_location: ndarray = None) -> None:
        default_Xi = self.Xi_matrix(minimum_index)
        EC_mat = self.EC_matrix()
        if not self.retained_unit_cell_displacement_vectors:
            self.find_relevant_unit_cell_vectors()
        optimized_lengths_result = minimize(self._evals_calc_variational, self.optimized_lengths[minimum_index],
                                            args=(minimum_location, minimum_index, EC_mat, default_Xi), tol=1e-1)
        assert optimized_lengths_result.success
        optimized_lengths = optimized_lengths_result.x
        if not self.quiet:
            print("completed harmonic length optimization for the m={m} minimum".format(m=minimum_index))
        self.optimized_lengths[minimum_index] = optimized_lengths

    def _one_state_periodic_continuation_squeezing(self, minimum_location: ndarray, minimum: int,
                                                   retained_unit_cell_displacement_vectors: ndarray, local_func: Callable,
                                                   Xi: ndarray, Xi_inv: ndarray) -> complex:
        """Periodic continuation when considering only the ground state."""
        disentangled_squeezing_matrices = self._rho_sigma_tau_matrices(minimum, minimum, Xi)
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho_matrices = self._delta_rho_matrices(rho, rho_prime)
        scale = 1. / np.sqrt(det(np.eye(self.number_degrees_freedom) - np.matmul(rho, rho_prime)))
        ground_state_value = 0.0 + 0.0j
        for neighbor in retained_unit_cell_displacement_vectors:
            phi_neighbor = 2.0 * np.pi * np.array(neighbor)
            exp_prod_coefficient = self._exp_product_coefficient_squeezing(phi_neighbor, Xi_inv, sigma, sigma_prime)
            ground_state_value += (scale * exp_prod_coefficient * local_func(phi_neighbor, minimum_location,
                                                                             minimum_location,
                                                                             disentangled_squeezing_matrices,
                                                                             delta_rho_matrices))
        return ground_state_value

    def _one_state_local_identity_squeezing(self, Xi_inv: ndarray, phi_neighbor: ndarray,
                                            minima_m: ndarray, minima_p: ndarray,
                                            disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple
                                            ) -> ndarray:
        """Local identity contribution when considering only the ground state."""
        return self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                           disentangled_squeezing_matrices, delta_rho_matrices,
                                                           np.array([1.0]), ())

    def _one_state_local_transfer_squeezing(self, EC_mat: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                            phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                            disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple
                                            ) -> ndarray:
        """Local transfer contribution when considering only the ground state."""
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        linear_coefficients_kinetic = self._linear_coefficient_matrices(rho_prime, delta_rho,
                                                                        -1j * Xi_inv.T / np.sqrt(2.0),
                                                                        1j * Xi_inv.T / np.sqrt(2.0))
        return (self._one_state_local_kinetic_squeezing_function(EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                 disentangled_squeezing_matrices,
                                                                 delta_rho_matrices, linear_coefficients_kinetic)
                + self._one_state_local_potential_squeezing_function(Xi, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                     disentangled_squeezing_matrices,
                                                                     delta_rho_matrices))

    def _one_state_local_potential_squeezing_function(self, Xi: ndarray, Xi_inv: ndarray,
                                                      phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                                      disentangled_squeezing_matrices: Tuple,
                                                      delta_rho_matrices: Tuple
                                                      ) -> ndarray:
        """Local potential contribution when considering only the ground state."""
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_list = np.array([np.exp(1j * phi_bar[i]) for i in range(dim)])
        exp_i_phi_sum_op = (np.exp(1j * 2.0 * np.pi * self.flux)
                            * np.prod([np.exp(1j * self.stitching_coefficients[i] * phi_bar[i]) for i in range(dim)]))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      delta_rho_matrices,
                                                                                      exp_i_phi_list)
                                   for j in range(dim)], axis=0)
        potential_matrix += self._local_contribution_stitching_squeezing(delta_phi, Xi, Xi_inv,
                                                                         disentangled_squeezing_matrices,
                                                                         delta_rho_matrices,
                                                                         exp_i_phi_sum_op)
        potential_matrix += (self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                         disentangled_squeezing_matrices,
                                                                         delta_rho_matrices, np.array([1.0]), ())
                             * np.sum(self.EJlist))
        return potential_matrix

    def _one_state_local_kinetic_squeezing_function(self, EC_mat: ndarray, Xi_inv: ndarray,
                                                    phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                                    disentangled_squeezing_matrices: Tuple, delta_rho_matrices: Tuple,
                                                    linear_coefficient_matrices: Tuple) -> ndarray:
        """Local kinetic contribution when considering only the ground state."""
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        a_coefficient, a_dagger_coefficient = linear_coefficient_matrices
        alpha, epsilon = self._construct_kinetic_alpha_epsilon_squeezing(Xi_inv, delta_phi, rho_prime, delta_rho)
        result = 4 * alpha * (epsilon @ EC_mat @ epsilon + np.trace(a_dagger_coefficient.T @ EC_mat @ a_coefficient))
        return result

    def _evals_calc_variational(self, optimized_lengths: ndarray, minimum_location: ndarray, minimum: int,
                                EC_mat: ndarray, default_Xi: ndarray) -> ndarray:
        """Function to be optimized in the minimization procedure, corresponding to the variational estimate of
        the ground state energy."""
        self.optimized_lengths[minimum] = optimized_lengths
        Xi = self._update_Xi(default_Xi, minimum)
        Xi_inv = inv(Xi)
        transfer, inner = self._one_state_construct_transfer_inner_squeezing(Xi, Xi_inv, minimum_location,
                                                                             minimum, EC_mat)
        return np.real([transfer / inner])

    def _one_state_construct_transfer_inner_squeezing(self, Xi: ndarray, Xi_inv: ndarray,
                                                      minimum_location: ndarray, minimum: int,
                                                      EC_mat: ndarray) -> Tuple:
        """Transfer matrix and inner product matrix when considering only the ground state."""
        retained_unit_cell_displacement_vectors = self.retained_unit_cell_displacement_vectors[(minimum, minimum)]
        transfer_function = partial(self._one_state_local_transfer_squeezing, EC_mat, Xi, Xi_inv)
        inner_product_function = partial(self._one_state_local_identity_squeezing, Xi_inv)
        transfer = self._one_state_periodic_continuation_squeezing(minimum_location, minimum, retained_unit_cell_displacement_vectors,
                                                                   transfer_function, Xi, Xi_inv)
        inner_product = self._one_state_periodic_continuation_squeezing(minimum_location, minimum, retained_unit_cell_displacement_vectors,
                                                                        inner_product_function, Xi, Xi_inv)
        return transfer, inner_product
