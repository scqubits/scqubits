import itertools
from functools import partial, reduce
from typing import Callable, Tuple

import numpy as np
import scipy as sp
from numpy import ndarray
from numpy.linalg import matrix_power
from scipy.linalg import LinAlgError, det, expm, inv, logm
from scipy.optimize import minimize
from scipy.special import factorial

from scqubits.core.vtbbasemethods import VTBBaseMethods
from scqubits.utils.cpu_switch import get_map_method


class VTBBaseMethodsSqueezing(VTBBaseMethods):
    r"""VariationalTightBinding allowing for squeezing

    See class VariationalTightBinding for documentation and explanation of parameters.

    """

    def _X_Y_Z_matrices(
        self, minimum_index: int, Xi: ndarray, Xi_prime: ndarray
    ) -> Tuple:
        """
        Return the X, Y, Z matrices that define the overall squeezing operator U

        Parameters
        ----------
        minimum_index: int
            integer representing the minimum for which to build the squeezing operator
             U, 0<i<=total number of minima (no squeezing need be performed for the
             global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times

        Returns
        -------
        ndarray, ndarray, ndarray
        """
        Xi_inv = inv(Xi)
        Xi_prime_inv = inv(Xi_prime)
        u = 0.5 * (Xi_prime_inv @ Xi + Xi_prime.T @ Xi_inv.T)
        v = 0.5 * (Xi_prime_inv @ Xi - Xi_prime.T @ Xi_inv.T)
        X = self._symmetrize_matrix(inv(u) @ v)
        Y = self._symmetrize_matrix(logm(u))
        Z = self._symmetrize_matrix(v @ inv(u))
        return X, Y, Z

    @staticmethod
    def _symmetrize_matrix(mat: ndarray) -> ndarray:
        dim = len(mat)
        for i in range(dim):
            for j in range(i+1, dim):
                element_average = (mat[i, j] + mat[j, i]) / 2.0
                mat[i, j] = mat[j, i] = element_average
        return mat

    def _delta_X_matrices(self, X: ndarray, X_prime: ndarray) -> Tuple:
        dim = self.number_degrees_freedom
        delta_X_prime = inv(np.eye(dim) - X_prime @ X) @ X_prime
        delta_X = inv(np.eye(dim) - X @ X_prime) @ X
        delta_X_bar = logm(inv(np.eye(dim) - X_prime @ X))
        return delta_X, delta_X_prime, delta_X_bar

    @staticmethod
    def _linear_coefficient_matrices(
        X_prime: ndarray, delta_X: ndarray, A: ndarray, B: ndarray
    ) -> Tuple:
        """Build variables helpful for constructing the Hamiltonian """
        a_coefficient = A - 0.5 * (B - A @ X_prime) @ (delta_X + delta_X.T)
        a_dagger_coefficient = B - A @ X_prime
        return a_coefficient, a_dagger_coefficient

    def _find_closest_periodic_minimum(
        self, relevant_unit_cell_vectors: dict, minima_index_pair: Tuple
    ) -> float:
        """Overrides method in VariationalTightBinding, need to consider states
        localized in both minima."""
        (m, minima_m), (p, minima_p) = minima_index_pair

        max_for_m = self._min_localization_ratio_for_minima_pair(
            minima_index_pair, m, relevant_unit_cell_vectors
        )
        max_for_p = self._min_localization_ratio_for_minima_pair(
            minima_index_pair, p, relevant_unit_cell_vectors
        )
        return max(max_for_m, max_for_p)

    def _normal_ordered_a_dagger_a_exponential(
        self, x: ndarray, a_operator_array: ndarray
    ) -> ndarray:
        """Return normal ordered exponential matrix of
        exp(a_{i}^{\dagger}x_{ij}a_{j})"""
        expm_x = expm(x)
        num_states = self.number_states_per_minimum()
        dim = self.number_degrees_freedom
        result = np.eye(num_states, dtype=np.complex128)
        additional_term = np.eye(num_states, dtype=np.complex128)
        k = 1
        while not np.allclose(additional_term, np.zeros((num_states, num_states))):
            additional_term = np.sum(
                [
                    ((expm_x - np.eye(dim))[i, j]) ** k
                    * (factorial(k)) ** (-1)
                    * matrix_power(a_operator_array[i].T, k)
                    @ matrix_power(a_operator_array[j], k)
                    for i in range(dim)
                    for j in range(dim)
                ],
                axis=0,
            )
            result += additional_term
            k += 1
        return result

    def _X_Y_Z_matrices_both_minima(
        self, m: int, p: int, Xi: ndarray, harmonic_lengths: ndarray
    ) -> Tuple:
        """Return the `X, Y, Z` matrices that define the squeezing
        operator `U`."""
        dim = self.number_degrees_freedom
        if m == 0:  # At the global minimum, no squeezing required
            X = np.zeros((dim, dim))
            Y = np.zeros((dim, dim))
            Z = np.zeros((dim, dim))
        else:
            Xi_prime = self.Xi_matrix(m, harmonic_lengths)
            X, Y, Z = self._X_Y_Z_matrices(m, Xi, Xi_prime)
        if p == 0:
            X_prime = np.zeros((dim, dim))
            Y_prime = np.zeros((dim, dim))
            Z_prime = np.zeros((dim, dim))
        elif p == m:
            X_prime = np.copy(X)
            Y_prime = np.copy(Y)
            Z_prime = np.copy(Z)
        else:
            Xi_prime = self.Xi_matrix(p, harmonic_lengths)
            X_prime, Y_prime, Z_prime = self._X_Y_Z_matrices(
                p, Xi, Xi_prime
            )
        return X, X_prime, Y, Y_prime, Z, Z_prime

    def _translation_operator_prefactors(
        self, disentangled_squeezing_matrices: Tuple, delta_X_matrices: Tuple
    ):
        """Helper method for building the translation operator argument prefactors"""
        dim = self.number_degrees_freedom
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        prefactor_a_dagger = (
            (np.eye(dim) + X_prime) @ expm(delta_X_bar).T @ expm(-Y)
        )
        prefactor_a = (
            np.eye(dim) + 0.5 * (np.eye(dim) + X_prime) @ (delta_X + delta_X.T)
        ) @ expm(-Y_prime)
        return prefactor_a, prefactor_a_dagger

    def _general_translation_operators(
        self,
        minima_diff: ndarray,
        Xi: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. The resulting matrices yield a
        2pi translation in each degree of freedom, so that any translation can be
        built from these by an appropriate call to np.matrix_power"""
        dim = self.number_degrees_freedom
        num_states_per_min = self.number_states_per_minimum()
        prefactor_a, prefactor_a_dagger = self._translation_operator_prefactors(
            disentangled_squeezing_matrices, delta_X_matrices
        )
        a_operator_array = self._a_operator_array()
        Xi_inv = inv(Xi)
        exp_a_list = np.zeros(
            (dim, num_states_per_min, num_states_per_min), dtype=np.complex_
        )
        exp_a_dagger_list = np.zeros_like(exp_a_list)
        for i in range(dim):
            exp_a_dagger_list[i] = expm(
                np.sum(
                    2.0
                    * np.pi
                    * (Xi_inv.T[i] @ prefactor_a_dagger)
                    * a_operator_array.T,
                    axis=2,
                )
                / np.sqrt(2.0)
            )
            exp_a_list[i] = expm(
                np.sum(
                    2.0
                    * np.pi
                    * (Xi_inv.T[i] @ prefactor_a)
                    * np.transpose(a_operator_array, axes=(1, 2, 0)),
                    axis=2,
                )
                / np.sqrt(2.0)
            )
        return exp_a_list, exp_a_dagger_list

    def _minima_dependent_translation_operators(
        self,
        minima_diff: ndarray,
        Xi: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. This part of the translation
        operator accounts for the differing location of minima within a single unit
        cell."""
        a_operator_array = self._a_operator_array()
        prefactor_a, prefactor_a_dagger = self._translation_operator_prefactors(
            disentangled_squeezing_matrices, delta_X_matrices
        )
        Xi_inv = inv(Xi)
        exp_a_dagger_minima_difference = expm(
            np.sum(
                minima_diff @ Xi_inv.T @ prefactor_a_dagger * a_operator_array.T, axis=2
            )
            / np.sqrt(2.0)
        )
        exp_a_minima_difference = expm(
            np.sum(
                -minima_diff
                @ Xi_inv.T
                @ prefactor_a
                * np.transpose(a_operator_array, axes=(1, 2, 0)),
                axis=2,
            )
            / np.sqrt(2.0)
        )
        return exp_a_minima_difference, exp_a_dagger_minima_difference

    def _potential_operators_squeezing(
        self,
        precalculated_quantities: Tuple,
        exp_a_dagger_a: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Helper method for building the potential operators."""
        exp_i_list = []
        Xi, _, a_operator_array, _ = precalculated_quantities
        dim = self.number_degrees_freedom
        prefactor_a, prefactor_a_dagger = self._potential_exp_prefactors(
            disentangled_squeezing_matrices, delta_X_matrices
        )
        for j in range(dim):
            exp_i_j_a_dagger_part = expm(
                np.sum(1j * (Xi[j] @ prefactor_a_dagger) * a_operator_array.T, axis=2)
                / np.sqrt(2.0)
            )
            exp_i_j_a_part = expm(
                np.sum(
                    1j
                    * (Xi[j] @ prefactor_a)
                    * np.transpose(a_operator_array, axes=(1, 2, 0)),
                    axis=2,
                )
                / np.sqrt(2.0)
            )
            exp_i_j = exp_i_j_a_dagger_part @ exp_a_dagger_a @ exp_i_j_a_part
            exp_i_list.append(exp_i_j)

        exp_i_sum_a_dagger_part = expm(
            np.sum(
                1j
                * self.stitching_coefficients
                @ Xi
                @ prefactor_a_dagger
                * a_operator_array.T,
                axis=2,
            )
            / np.sqrt(2.0)
        )
        exp_i_sum_a_part = expm(
            np.sum(
                1j
                * self.stitching_coefficients
                @ Xi
                @ prefactor_a
                * np.transpose(a_operator_array, axes=(1, 2, 0)),
                axis=2,
            )
            / np.sqrt(2.0)
        )
        exp_i_sum = exp_i_sum_a_dagger_part @ exp_a_dagger_a @ exp_i_sum_a_part
        return exp_i_list, exp_i_sum

    def _bilinear_squeezing_operators(
        self,
        a_operator_array: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Helper method for building the bilinear operators necessary for constructing
        the Hamiltonian in the presence of squeezing."""
        dim = self.number_degrees_freedom
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        prefactor_a_dagger_a_dagger = 0.5 * (
            Z.T - expm(-Y).T @ delta_X_prime @ expm(-Y)
        )
        prefactor_a_a = 0.5 * (
            Z_prime - expm(-Y_prime).T @ delta_X @ expm(-Y_prime)
        )
        prefactor_a_dagger_a = sp.linalg.logm(
            expm(-Y).T @ expm(delta_X_bar) @ expm(-Y_prime)
        )

        exp_a_dagger_a_dagger = expm(
            np.sum(
                [
                    prefactor_a_dagger_a_dagger[i, j]
                    * a_operator_array[i].T
                    @ a_operator_array[j].T
                    for i in range(dim)
                    for j in range(dim)
                ],
                axis=0,
            )
        )
        exp_a_a = expm(
            np.sum(
                [
                    prefactor_a_a[i, j] * a_operator_array[i] @ a_operator_array[j]
                    for i in range(dim)
                    for j in range(dim)
                ],
                axis=0,
            )
        )
        exp_a_dagger_a = self._normal_ordered_a_dagger_a_exponential(
            prefactor_a_dagger_a, a_operator_array
        )
        return exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a

    def _local_translation_operators(
        self,
        exp_operators: Tuple,
        squeezing_operators: Tuple,
        unit_cell_vector: ndarray,
    ) -> Tuple:
        """Build translation operators using matrix_power"""
        dim = self.number_degrees_freedom
        (exp_a_list, exp_a_dagger_list), (
            exp_a_minima_difference,
            exp_a_dagger_minima_difference,
        ) = exp_operators
        exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
        individual_op_a_dagger = np.array(
            [
                matrix_power(exp_a_dagger_list[j], int(unit_cell_vector[j]))
                for j in range(dim)
            ]
        )
        individual_op_a = np.array(
            [matrix_power(exp_a_list[j], -int(unit_cell_vector[j])) for j in range(dim)]
        )
        translation_op_a_dag = (
            reduce((lambda x, y: x @ y), individual_op_a_dagger)
            @ exp_a_dagger_minima_difference
            @ exp_a_dagger_a_dagger
        )
        translation_op_a = (
            reduce((lambda x, y: x @ y), individual_op_a)
            @ exp_a_minima_difference
            @ exp_a_a
        )
        return translation_op_a_dag, translation_op_a

    def _periodic_continuation(
        self,
        minima_pair_func: Callable,
        local_func: Callable,
        relevant_unit_cell_vectors: dict,
        optimized_harmonic_lengths: ndarray,
        num_cpus: int = 1,
    ) -> ndarray:
        """See VariationalTightBinding for documentation. This function generalizes
        _periodic_continuation to allow for squeezing"""
        Xi = self.Xi_matrix(0, optimized_harmonic_lengths)
        Xi_inv = inv(Xi)
        target_map = get_map_method(num_cpus)
        a_operator_array = self._a_operator_array()
        all_minima_index_pairs = list(
            itertools.combinations_with_replacement(self.sorted_minima_dict.items(), 2)
        )
        wrapper_minima_pair = partial(
            self._wrapper_minima_pair,
            minima_pair_func,
            local_func,
            Xi,
            Xi_inv,
            optimized_harmonic_lengths,
            a_operator_array,
            relevant_unit_cell_vectors,
        )
        matrix_elements = list(target_map(wrapper_minima_pair, all_minima_index_pairs))
        return self._construct_VTB_operator_given_blocks(
            matrix_elements, all_minima_index_pairs
        )

    def _wrapper_minima_pair(
        self,
        minima_pair_func: Callable,
        local_func: Callable,
        Xi: ndarray,
        Xi_inv: ndarray,
        harmonic_lengths: ndarray,
        a_operator_array: ndarray,
        relevant_unit_cell_vectors: dict,
        minima_index_pair: Tuple,
    ):
        ((m, minima_m), (p, minima_p)) = minima_index_pair
        minima_diff = minima_p - minima_m
        disentangled_squeezing_matrices = self._X_Y_Z_matrices_both_minima(
            m, p, Xi, harmonic_lengths
        )
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X_matrices = self._delta_X_matrices(X, X_prime)
        squeezing_operators = self._bilinear_squeezing_operators(
            a_operator_array, disentangled_squeezing_matrices, delta_X_matrices
        )
        exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
        exp_operators = (
            self._general_translation_operators(
                minima_diff, Xi, disentangled_squeezing_matrices, delta_X_matrices
            ),
            self._minima_dependent_translation_operators(
                minima_diff, Xi, disentangled_squeezing_matrices, delta_X_matrices
            ),
        )
        minima_pair_results = minima_pair_func(
            exp_a_dagger_a, disentangled_squeezing_matrices, delta_X_matrices
        )
        scale = 1.0 / np.sqrt(
            det(np.eye(self.number_degrees_freedom) - np.matmul(X, X_prime))
        )
        return (
            self._periodic_continuation_for_minima_pair(
                local_func,
                exp_operators,
                Xi_inv,
                squeezing_operators,
                disentangled_squeezing_matrices,
                delta_X_matrices,
                relevant_unit_cell_vectors,
                minima_pair_results,
                minima_index_pair,
            )
            * scale
        )

    def _periodic_continuation_for_minima_pair(
        self,
        local_func: Callable,
        exp_operators: Tuple,
        Xi_inv: ndarray,
        squeezing_operators: Tuple,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        relevant_unit_cell_vectors: dict,
        minima_pair_results: Tuple,
        minima_index_pair: Tuple,
    ):
        ((m, minima_m), (p, minima_p)) = minima_index_pair
        minima_pair_displacement_vectors = relevant_unit_cell_vectors[(m, p)]
        num_states_per_min = self.number_states_per_minimum()
        if minima_pair_displacement_vectors is not None:
            displacement_vector_contribution = partial(
                self._displacement_vector_contribution,
                local_func,
                minima_m,
                minima_p,
                exp_operators,
                Xi_inv,
                squeezing_operators,
                disentangled_squeezing_matrices,
                delta_X_matrices,
                minima_pair_results,
            )
            relevant_vector_contributions = sum(
                map(displacement_vector_contribution, minima_pair_displacement_vectors)
            )
        else:
            relevant_vector_contributions = np.zeros(
                (num_states_per_min, num_states_per_min), dtype=np.complex_
            )
        return relevant_vector_contributions

    def _displacement_vector_contribution(
        self,
        local_func: Callable,
        minima_m: ndarray,
        minima_p: ndarray,
        exp_operators: Tuple,
        Xi_inv: ndarray,
        squeezing_operators: Tuple,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        minima_pair_results: Tuple,
        unit_cell_vector: ndarray,
    ) -> ndarray:
        displacement_vector = 2.0 * np.pi * np.array(unit_cell_vector)
        translation_operators = self._local_translation_operators(
            exp_operators, squeezing_operators, unit_cell_vector
        )
        (
            _,
            _,
            Y,
            Y_prime,
            _,
            _,
        ) = disentangled_squeezing_matrices
        exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
        translation_a_dagger, translation_a = translation_operators
        exp_prod_coefficient = self._exp_product_coefficient_squeezing(
            displacement_vector + minima_p - minima_m,
            Xi_inv,
            Y,
            Y_prime,
        )
        return (
            exp_prod_coefficient
            * translation_a_dagger
            @ local_func(
                displacement_vector,
                minima_m,
                minima_p,
                disentangled_squeezing_matrices,
                delta_X_matrices,
                exp_a_dagger_a,
                minima_pair_results,
            )
            @ translation_a
        )

    def _kinetic_alpha_epsilon_squeezing(
        self,
        Xi_inv: ndarray,
        delta_phi: ndarray,
        X_prime: ndarray,
        delta_X: ndarray,
    ) -> Tuple:
        """Construct the `alpha` and `epsilon` variables necessary for the kinetic
        matrix."""
        arg_exp_a_dag = delta_phi @ Xi_inv.T / np.sqrt(2.0)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, X_prime, delta_X)
        delta_X_pp = (
            0.5 * (arg_exp_a_dag - arg_exp_a @ X_prime) @ (delta_X + delta_X.T)
        )
        epsilon = (
            -(1j / np.sqrt(2.0))
            * Xi_inv.T
            @ (
                X_prime @ delta_X_pp
                - arg_exp_a @ X_prime
                + delta_X_pp
                + Xi_inv @ delta_phi / np.sqrt(2.0)
            )
        )
        return alpha, epsilon

    def _minima_pair_transfer_squeezing(
        self,
        precalculated_quantities: Tuple,
        exp_a_dagger_a: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Minima pair calculations for the kinetic and potential matrices."""
        return (
            self._minima_pair_kinetic_squeezing(
                precalculated_quantities,
                exp_a_dagger_a,
                disentangled_squeezing_matrices,
                delta_X_matrices,
            ),
            self._minima_pair_potential_squeezing(
                precalculated_quantities,
                exp_a_dagger_a,
                disentangled_squeezing_matrices,
                delta_X_matrices,
            ),
        )

    def _minima_pair_kinetic_squeezing(
        self,
        precalculated_quantities: Tuple,
        exp_a_dagger_a: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Return data necessary for constructing the kinetic matrix that only depends
        on the minima pair, and not on the specific periodic continuation operator."""
        Xi, Xi_inv, a_operator_array, EC_mat = precalculated_quantities
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        linear_coefficients_kinetic = self._linear_coefficient_matrices(
            X_prime,
            delta_X,
            -1j * Xi_inv.T / np.sqrt(2.0),
            1j * Xi_inv.T / np.sqrt(2.0),
        )
        a_coefficient, a_dagger_coefficient = linear_coefficients_kinetic
        (xa, xaa, dxa, dx, ddx) = self._premultiplying_exp_a_dagger_a_with_a(
            exp_a_dagger_a, a_operator_array
        )
        Y_delta_X_bar_a_dagger_coefficient_EC = (
            expm(-Y).T @ expm(delta_X_bar) @ a_dagger_coefficient.T @ EC_mat
        )
        xaa_coefficient = (
            (a_coefficient @ expm(-Y_prime)).T
            @ EC_mat
            @ a_coefficient
            @ expm(-Y_prime)
        )
        dxa_coefficient = (
            Y_delta_X_bar_a_dagger_coefficient_EC
            @ a_coefficient
            @ expm(-Y_prime)
        )
        ddx_coefficient = (
            Y_delta_X_bar_a_dagger_coefficient_EC
            @ (expm(-Y).T @ expm(delta_X_bar) @ a_dagger_coefficient.T).T
        )
        x_coefficient = a_dagger_coefficient.T @ EC_mat @ a_coefficient
        xa_coefficient = EC_mat @ a_coefficient @ expm(-Y_prime)
        dx_coefficient = (
            EC_mat @ a_dagger_coefficient @ (expm(-Y).T @ expm(delta_X_bar)).T
        )
        kinetic_matrix = np.sum(
            +4 * np.transpose(xaa, axes=(1, 2, 0)) * np.diag(xaa_coefficient)
            + 8 * np.transpose(dxa, axes=(1, 2, 0)) * np.diag(dxa_coefficient)
            + 4 * np.transpose(ddx, axes=(1, 2, 0)) * np.diag(ddx_coefficient),
            axis=2,
        )
        kinetic_matrix += 4 * exp_a_dagger_a * np.sum(np.diag(x_coefficient))
        return kinetic_matrix, xa, dx, xa_coefficient, dx_coefficient

    def _local_transfer_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        exp_a_dagger_a: ndarray,
        minima_pair_results: Tuple,
    ) -> ndarray:
        """Local contribution to the transfer matrix in the presence of squeezing."""
        kinetic_minima_pair_results, potential_minima_pair_results = minima_pair_results
        return self._local_kinetic_squeezing(
            precalculated_quantities,
            displacement_vector,
            minima_m,
            minima_p,
            disentangled_squeezing_matrices,
            delta_X_matrices,
            exp_a_dagger_a,
            kinetic_minima_pair_results,
        ) + self._local_potential_squeezing(
            precalculated_quantities,
            displacement_vector,
            minima_m,
            minima_p,
            disentangled_squeezing_matrices,
            delta_X_matrices,
            exp_a_dagger_a,
            potential_minima_pair_results,
        )

    def _local_kinetic_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        exp_a_dagger_a: ndarray,
        minima_pair_results: Tuple,
    ) -> ndarray:
        """Local contribution to the kinetic matrix in the presence of squeezing."""
        Xi, Xi_inv, EC_mat = precalculated_quantities
        delta_phi = displacement_vector + minima_p - minima_m
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        (
            kinetic_matrix_minima_pair,
            xa,
            dx,
            xa_coefficient,
            dx_coefficient,
        ) = minima_pair_results
        alpha, epsilon = self._kinetic_alpha_epsilon_squeezing(
            Xi_inv, delta_phi, X_prime, delta_X
        )
        e_xa_coefficient = epsilon @ xa_coefficient
        e_dx_coefficient = epsilon @ dx_coefficient
        return alpha * (
            np.sum(
                8 * np.transpose(xa, axes=(1, 2, 0)) * e_xa_coefficient
                + 8 * np.transpose(dx, axes=(1, 2, 0)) * e_dx_coefficient,
                axis=2,
            )
            + kinetic_matrix_minima_pair
            + 4 * exp_a_dagger_a * (epsilon @ EC_mat @ epsilon)
        )

    @staticmethod
    def _premultiplying_exp_a_dagger_a_with_a(
        exp_a_dagger_a: ndarray, a_operator_array: ndarray
    ) -> Tuple:
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
    def _alpha_helper(
        arg_exp_a_dag: ndarray,
        arg_exp_a: ndarray,
        X_prime: ndarray,
        delta_X: ndarray,
    ) -> ndarray:
        """Build the prefactor that arises due to squeezing. With no squeezing, alpha=1
        (number, not matrix)"""
        arg_exp_a_X_prime = np.matmul(arg_exp_a, X_prime)
        alpha = np.exp(
            -0.5 * arg_exp_a @ arg_exp_a_X_prime
            - 0.25
            * (arg_exp_a_dag - arg_exp_a_X_prime)
            @ (delta_X + delta_X.T)
            @ (arg_exp_a_dag - arg_exp_a_X_prime)
        )
        return alpha

    def transfer_matrix(self, num_cpus: int = 1) -> ndarray:
        """Returns the transfer matrix

        Returns
        -------
        ndarray
        """
        return self._abstract_VTB_operator(
            self._minima_pair_transfer_squeezing,
            self._local_transfer_squeezing,
            num_cpus,
        )

    def _transfer_matrix(
        self,
        relevant_unit_cell_vectors: dict,
        optimized_harmonic_lengths,
        num_cpus: int = 1,
    ):
        Xi = self.Xi_matrix(
            minimum_index=0, harmonic_lengths=optimized_harmonic_lengths
        )
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        EC_mat = self.EC_matrix()
        partial_minima_pair_func = partial(
            self._minima_pair_transfer_squeezing, (Xi, Xi_inv, a_operator_array, EC_mat)
        )
        partial_local_func = partial(
            self._local_transfer_squeezing, (Xi, Xi_inv, EC_mat)
        )
        return self._periodic_continuation(
            partial_minima_pair_func,
            partial_local_func,
            relevant_unit_cell_vectors,
            optimized_harmonic_lengths,
            num_cpus,
        )

    def _inner_product_matrix(
        self,
        relevant_unit_cell_vectors: dict,
        optimized_harmonic_lengths: ndarray,
        num_cpus: int = 1,
    ):
        Xi_inv = inv(self.Xi_matrix(0, optimized_harmonic_lengths))
        local_identity_func = partial(
            self._local_identity_squeezing, (None, Xi_inv, None)
        )
        return self._periodic_continuation(
            lambda x, y, z: None,
            local_identity_func,
            relevant_unit_cell_vectors,
            optimized_harmonic_lengths,
            num_cpus,
        )

    def kinetic_matrix(self, num_cpus: int = 1) -> ndarray:
        """Returns the kinetic energy matrix

        Returns
        -------
        ndarray
        """
        return self._abstract_VTB_operator(
            self._minima_pair_kinetic_squeezing, self._local_kinetic_squeezing, num_cpus
        )

    def _abstract_VTB_operator(
        self, minima_pair_func: Callable, local_func: Callable, num_cpus: int = 1
    ) -> ndarray:
        relevant_unit_cell_vectors, optimized_harmonic_lengths = self._initialize_VTB(
            num_cpus
        )
        Xi = self.Xi_matrix(
            minimum_index=0, harmonic_lengths=optimized_harmonic_lengths
        )
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        EC_mat = self.EC_matrix()
        partial_minima_pair_func = partial(
            minima_pair_func, (Xi, Xi_inv, a_operator_array, EC_mat)
        )
        partial_local_func = partial(local_func, (Xi, Xi_inv, EC_mat))
        return self._periodic_continuation(
            partial_minima_pair_func,
            partial_local_func,
            relevant_unit_cell_vectors,
            optimized_harmonic_lengths,
            num_cpus,
        )

    def potential_matrix(self, num_cpus: int = 1) -> ndarray:
        """Returns the potential energy matrix

        Returns
        -------
        ndarray
        """
        return self._abstract_VTB_operator(
            self._minima_pair_potential_squeezing,
            self._local_potential_squeezing,
            num_cpus,
        )

    def _potential_exp_prefactors(
        self, disentangled_squeezing_matrices: Tuple, delta_X_matrices: Tuple
    ) -> Tuple:
        dim = self.number_degrees_freedom
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        prefactor_a_dagger = (
            (np.eye(dim) - X_prime) @ expm(delta_X_bar).T @ expm(-Y)
        )
        prefactor_a = (
            np.eye(dim) - 0.5 * (np.eye(dim) - X_prime) @ (delta_X + delta_X.T)
        ) @ expm(-Y_prime)
        return prefactor_a, prefactor_a_dagger

    def _local_potential_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        exp_a_dagger_a: ndarray,
        minima_pair_results: Tuple,
    ) -> ndarray:
        """Local contribution to the potential matrix in the presence of squeezing."""
        dim = self.number_degrees_freedom
        Xi, Xi_inv, _ = precalculated_quantities
        delta_phi = displacement_vector + minima_p - minima_m
        phi_bar = 0.5 * (displacement_vector + (minima_m + minima_p))
        exp_i_list, exp_i_sum = minima_pair_results
        exp_i_phi_list = np.array(
            [exp_i_list[i] * np.exp(1j * phi_bar[i]) for i in range(dim)]
        )
        exp_i_phi_sum_op = (
            exp_i_sum
            * np.exp(1j * 2.0 * np.pi * self.flux)
            * np.exp(np.sum(1j * self.stitching_coefficients * phi_bar))
        )
        potential_matrix = np.sum(
            [
                self._local_single_junction_squeezing(
                    j,
                    delta_phi,
                    Xi,
                    Xi_inv,
                    disentangled_squeezing_matrices,
                    delta_X_matrices,
                    exp_i_phi_list,
                )
                for j in range(dim)
            ],
            axis=0,
        )
        potential_matrix += self._local_stitching_squeezing(
            delta_phi,
            Xi,
            Xi_inv,
            disentangled_squeezing_matrices,
            delta_X_matrices,
            exp_i_phi_sum_op,
        )
        potential_matrix += (
            self._local_identity_squeezing(
                precalculated_quantities,
                displacement_vector,
                minima_m,
                minima_p,
                disentangled_squeezing_matrices,
                delta_X_matrices,
                exp_a_dagger_a,
                minima_pair_results,
            )
            * np.sum(self.EJlist)
        )
        return potential_matrix

    def _local_stitching_squeezing(
        self,
        delta_phi: ndarray,
        Xi: ndarray,
        Xi_inv: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        exp_i_sum: ndarray,
    ) -> ndarray:
        """Local contribution to the potential due to the stitching term"""
        dim = self.number_degrees_freedom
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (
            delta_phi_rotated
            + np.sum(
                [1j * Xi[i, :] * self.stitching_coefficients[i] for i in range(dim)],
                axis=0,
            )
        ) / np.sqrt(2.0)
        alpha = self._alpha_helper(
            arg_exp_a_dag, -arg_exp_a_dag.conjugate(), X_prime, delta_X
        )
        potential_matrix = (
            -0.5 * self.EJlist[-1] * (alpha * exp_i_sum + (alpha * exp_i_sum).conj())
        )
        potential_matrix *= self._BCH_factor_for_potential_stitching(Xi)
        return potential_matrix

    def _local_single_junction_squeezing(
        self,
        j: int,
        delta_phi: ndarray,
        Xi: ndarray,
        Xi_inv: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        exp_i_phi_list: ndarray,
    ) -> ndarray:
        """Local contribution to the potential due to `\cos(\phi_j)`"""
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + 1j * Xi[j, :]) / np.sqrt(2.0)
        alpha = self._alpha_helper(
            arg_exp_a_dag, -arg_exp_a_dag.conjugate(), X_prime, delta_X
        )
        potential_matrix = (
            -0.5
            * self.EJlist[j]
            * (alpha * exp_i_phi_list[j] + (alpha * exp_i_phi_list[j]).conj())
        )
        potential_matrix *= np.exp(-0.25 * np.dot(Xi[j, :], Xi.T[:, j]))
        return potential_matrix

    def _minima_pair_potential_squeezing(
        self,
        precalculated_quantities: Tuple,
        exp_a_dagger_a: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> Tuple:
        """Return data necessary for constructing the potential matrix that only depends
        on the minima pair, and not on the specific periodic continuation operator."""
        return self._potential_operators_squeezing(
            precalculated_quantities,
            exp_a_dagger_a,
            disentangled_squeezing_matrices,
            delta_X_matrices,
        )

    def _local_identity_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        exp_a_dagger_a: ndarray,
        minima_pair_results: Tuple,
    ) -> ndarray:
        """Local contribution to the identity matrix in the presence of squeezing."""
        _ = minima_pair_results
        _, Xi_inv, _ = precalculated_quantities
        delta_phi = displacement_vector + minima_p - minima_m
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        arg_exp_a_dag = np.matmul(delta_phi, Xi_inv.T) / np.sqrt(2.0)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, X_prime, delta_X)
        return alpha * exp_a_dagger_a

    def inner_product_matrix(self, num_cpus: int = 1) -> ndarray:
        """Returns the inner product matrix

        Returns
        -------
        ndarray
        """
        return self._abstract_VTB_operator(
            lambda p, x, y, z: None, self._local_identity_squeezing, num_cpus
        )

    def _exp_product_coefficient_squeezing(
        self,
        displacement_vector: ndarray,
        Xi_inv: ndarray,
        Y: ndarray,
        Y_prime: ndarray,
    ) -> ndarray:
        """Overall multiplicative factor. Includes offset charge, Gaussian suppression
        factor in the absence of squeezing. With squeezing, also includes exponential
        of trace over Y and Y_prime, see Qin et. al"""
        return np.exp(
            -0.5 * np.trace(Y) - 0.5 * np.trace(Y_prime)
        ) * self._exp_product_coefficient(displacement_vector, Xi_inv)

    def _optimize_harmonic_lengths(self, relevant_unit_cell_vectors: dict) -> ndarray:
        """Overrides method in VariationalTightBinding. Allows for harmonic length
        optimization of states localized in all minima if the optimize_all_minima
        flag is set. Optimize the Xi matrix by adjusting the harmonic lengths of the
        ground state to minimize its energy. For tight-binding without squeezing,
        this is only done for the ansatz ground state wavefunction localized in the
        global minimum."""
        sorted_minima_dict = self.sorted_minima_dict
        harmonic_lengths = np.ones(
            (len(sorted_minima_dict), self.number_degrees_freedom)
        )
        # No squeezing for the global minimum, so call parent's method
        optimized_harmonic_lengths = self._optimize_harmonic_lengths_minimum(
            0, sorted_minima_dict[0], relevant_unit_cell_vectors
        )
        harmonic_lengths[0] = optimized_harmonic_lengths
        Xi_global = self.Xi_matrix(minimum_index=0)
        Xi_global_inv = inv(Xi_global)
        harmonic_lengths_global = np.array(
            [
                np.linalg.norm(Xi_global[:, i])
                for i in range(self.number_degrees_freedom)
            ]
        )
        for minimum_index, minimum_location in sorted_minima_dict.items():
            if self.optimize_all_minima and minimum_index != 0:
                harmonic_lengths[
                    minimum_index
                ] = self._optimize_harmonic_lengths_minimum_squeezing(
                    minimum_index,
                    minimum_location,
                    relevant_unit_cell_vectors,
                    Xi_global,
                    Xi_global_inv,
                )
            elif self.use_global_min_harmonic_lengths and minimum_index != 0:
                Xi_local = self.Xi_matrix(minimum_index=minimum_index)
                harmonic_lengths_local = np.array(
                    [
                        np.linalg.norm(Xi_local[:, i])
                        for i in range(self.number_degrees_freedom)
                    ]
                )
                harmonic_lengths[minimum_index] = (
                    harmonic_lengths_global / harmonic_lengths_local
                )
            elif minimum_index != 0:
                harmonic_lengths[minimum_index] = np.ones(self.number_degrees_freedom)
        return harmonic_lengths

    def _optimize_harmonic_lengths_minimum_squeezing(
        self,
        minimum_index: int,
        minimum_location: ndarray,
        relevant_unit_cell_vectors: dict,
        Xi_global: ndarray,
        Xi_global_inv: ndarray,
    ) -> ndarray:
        default_Xi = self.Xi_matrix(minimum_index)
        EC_mat = self.EC_matrix()
        optimized_lengths_result = minimize(
            self._evals_calc_variational_squeezing,
            np.ones(self.number_degrees_freedom),
            args=(
                minimum_location,
                minimum_index,
                EC_mat,
                default_Xi,
                relevant_unit_cell_vectors,
                Xi_global,
                Xi_global_inv,
            ),
            tol=1e-1,
        )
        assert optimized_lengths_result.success
        optimized_lengths = optimized_lengths_result.x
        if not self.quiet:
            print(
                "completed harmonic length optimization for the m={m} minimum".format(
                    m=minimum_index
                )
            )
        return optimized_lengths

    def _one_state_periodic_continuation_squeezing(
        self,
        precalculated_quantities: Tuple,
        minimum_location: ndarray,
        minimum_index: int,
        minima_pair_displacement_vectors: ndarray,
        local_func: Callable,
    ) -> complex:
        """Periodic continuation when considering only the ground state."""
        Xi, _, _, Xi_global, Xi_global_inv = precalculated_quantities
        X, Y, Z = self._X_Y_Z_matrices(minimum_index, Xi_global, Xi)
        X_prime, Y_prime, Z_prime = np.copy(X), np.copy(Y), np.copy(Z)
        disentangled_squeezing_matrices = (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        )
        delta_X_matrices = self._delta_X_matrices(X, X_prime)
        scale = 1.0 / np.sqrt(
            det(np.eye(self.number_degrees_freedom) - np.matmul(X, X_prime))
        )
        ground_state_value = 0.0 + 0.0j
        for displacement_vector in minima_pair_displacement_vectors:
            exp_prod_coefficient = self._exp_product_coefficient_squeezing(
                displacement_vector, Xi_global_inv, Y, Y_prime
            )
            ground_state_value += (
                scale
                * exp_prod_coefficient
                * local_func(
                    displacement_vector,
                    minimum_location,
                    minimum_location,
                    disentangled_squeezing_matrices,
                    delta_X_matrices,
                )
            )
        return ground_state_value

    def _one_state_local_identity_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> ndarray:
        """Local identity contribution when considering only the ground state."""
        _, _, _, _, Xi_global_inv = precalculated_quantities
        return self._local_identity_squeezing(
            (_, Xi_global_inv, _),
            displacement_vector,
            minima_m,
            minima_p,
            disentangled_squeezing_matrices,
            delta_X_matrices,
            np.array([1.0]),
            (),
        )

    def _one_state_local_transfer_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> ndarray:
        """Local transfer contribution when considering only the ground state."""
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        _, _, _, _, Xi_global_inv = precalculated_quantities
        linear_coefficients_kinetic = self._linear_coefficient_matrices(
            X_prime,
            delta_X,
            -1j * Xi_global_inv.T / np.sqrt(2.0),
            1j * Xi_global_inv.T / np.sqrt(2.0),
        )
        return self._one_state_local_kinetic_squeezing(
            precalculated_quantities,
            displacement_vector,
            minima_m,
            minima_p,
            disentangled_squeezing_matrices,
            delta_X_matrices,
            linear_coefficients_kinetic,
        ) + self._one_state_local_potential_squeezing(
            precalculated_quantities,
            displacement_vector,
            minima_m,
            minima_p,
            disentangled_squeezing_matrices,
            delta_X_matrices,
        )

    def _one_state_local_potential_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
    ) -> ndarray:
        """Local potential contribution when considering only the ground state."""
        dim = self.number_degrees_freedom
        _, _, _, Xi_global, Xi_global_inv = precalculated_quantities
        delta_phi = displacement_vector + minima_p - minima_m
        phi_bar = 0.5 * (displacement_vector + (minima_m + minima_p))
        exp_i_phi_list = np.array([np.exp(1j * phi_bar[i]) for i in range(dim)])
        exp_i_phi_sum_op = np.exp(1j * 2.0 * np.pi * self.flux) * np.prod(
            [
                np.exp(1j * self.stitching_coefficients[i] * phi_bar[i])
                for i in range(dim)
            ]
        )
        potential_matrix = np.sum(
            [
                self._local_single_junction_squeezing(
                    j,
                    delta_phi,
                    Xi_global,
                    Xi_global_inv,
                    disentangled_squeezing_matrices,
                    delta_X_matrices,
                    exp_i_phi_list,
                )
                for j in range(dim)
            ],
            axis=0,
        )
        potential_matrix += self._local_stitching_squeezing(
            delta_phi,
            Xi_global,
            Xi_global_inv,
            disentangled_squeezing_matrices,
            delta_X_matrices,
            exp_i_phi_sum_op,
        )
        potential_matrix += (
            self._local_identity_squeezing(
                (None, Xi_global_inv, None),
                displacement_vector,
                minima_m,
                minima_p,
                disentangled_squeezing_matrices,
                delta_X_matrices,
                np.array([1.0]),
                (),
            )
            * np.sum(self.EJlist)
        )
        return potential_matrix

    def _one_state_local_kinetic_squeezing(
        self,
        precalculated_quantities: Tuple,
        displacement_vector: ndarray,
        minima_m: ndarray,
        minima_p: ndarray,
        disentangled_squeezing_matrices: Tuple,
        delta_X_matrices: Tuple,
        linear_coefficient_matrices: Tuple,
    ) -> ndarray:
        """Local kinetic contribution when considering only the ground state."""
        _, _, EC_mat, _, Xi_global_inv = precalculated_quantities
        delta_phi = displacement_vector + minima_p - minima_m
        (
            X,
            X_prime,
            Y,
            Y_prime,
            Z,
            Z_prime,
        ) = disentangled_squeezing_matrices
        delta_X, delta_X_prime, delta_X_bar = delta_X_matrices
        a_coefficient, a_dagger_coefficient = linear_coefficient_matrices
        alpha, epsilon = self._kinetic_alpha_epsilon_squeezing(
            Xi_global_inv, delta_phi, X_prime, delta_X
        )
        result = (
            4
            * alpha
            * (
                epsilon @ EC_mat @ epsilon
                + np.trace(a_dagger_coefficient.T @ EC_mat @ a_coefficient)
            )
        )
        return result

    def _evals_calc_variational_squeezing(
        self,
        harmonic_lengths: ndarray,
        minimum_location: ndarray,
        minimum_index: int,
        EC_mat: ndarray,
        default_Xi: ndarray,
        relevant_unit_cell_vectors: dict,
        Xi_global: ndarray,
        Xi_global_inv: ndarray,
    ) -> ndarray:
        """Function to be optimized in the minimization procedure, corresponding to the
        variational estimate of the ground state energy."""
        Xi = self._update_Xi(default_Xi, harmonic_lengths)
        Xi_inv = inv(Xi)
        precalculated_quantities = (Xi, Xi_inv, EC_mat, Xi_global, Xi_global_inv)
        transfer, inner = self._one_state_transfer_inner_squeezing(
            precalculated_quantities,
            minimum_location,
            minimum_index,
            relevant_unit_cell_vectors,
        )
        return np.real(transfer / inner)

    def _one_state_transfer_inner_squeezing(
        self,
        precalculated_quantities: Tuple,
        minimum_location: ndarray,
        minimum_index: int,
        relevant_unit_cell_vectors: dict,
    ) -> Tuple:
        """Transfer matrix and inner product matrix when considering
        only the ground state."""
        minima_pair_displacement_vectors = (
            2.0 * np.pi * relevant_unit_cell_vectors[(minimum_index, minimum_index)]
        )
        transfer_function = partial(
            self._one_state_local_transfer_squeezing, precalculated_quantities
        )
        inner_product_function = partial(
            self._one_state_local_identity_squeezing, precalculated_quantities
        )
        transfer = self._one_state_periodic_continuation_squeezing(
            precalculated_quantities,
            minimum_location,
            minimum_index,
            minima_pair_displacement_vectors,
            transfer_function,
        )
        inner_product = self._one_state_periodic_continuation_squeezing(
            precalculated_quantities,
            minimum_location,
            minimum_index,
            minima_pair_displacement_vectors,
            inner_product_function,
        )
        return transfer, inner_product
