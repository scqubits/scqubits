import itertools
from functools import partial

import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError, inv, expm, logm, det
from numpy.linalg import matrix_power
from scipy.special import factorial

from scqubits.core.vchos import VCHOS


# The VCHOS method (tight binding) allowing for the diagonalization of systems
# with purely periodic potentials. This module assumes that the potential is
# of the form -EJ[1]*cos(phi_1)-EJ[2]*cos(phi_2)-...-EJ[N]*cos(bc[1]*phi_1+bc[2]*phi_2+...-2\pi f).
# For the flux qubit, the last term looks like -alpha*EJ*cos(phi_1-phi_2-2\pi f), whereas for 
# the current mirror it is -EJ[N]*cos(\sum_i(phi_i)-2\pi f). The user must define a new qubit class
# that inherits VCHOS, with all of the qubit specific information. This includes a method for finding 
# minima, the definition of the capacitance matrix, the number of degrees of freedom, etc.

# Specifically, the user must provide in their parent class the functions 
# build_capacitance_matrix(), build_EC_matrix(), hilbertdim(), sorted_minima(), 
# which define the capacitance matrix, the charging energy matrix, the dimension
# of the hilbert space according to the specific truncation scheme used, and 
# a method to find and sort all inequivalent minima (based on the value of the
# potential at that minimum), respectively.


class VCHOSSqueezing(VCHOS):
    def __init__(self, EJlist, nglist, flux, maximum_periodic_vector_length, number_degrees_freedom=0,
                 number_periodic_degrees_freedom=0, num_exc=None, nearest_neighbors=None,
                 harmonic_length_optimization=0, optimize_all_minima=0):
        VCHOS.__init__(self, EJlist, nglist, flux, maximum_periodic_vector_length,
                       number_degrees_freedom=number_degrees_freedom,
                       number_periodic_degrees_freedom=number_periodic_degrees_freedom, num_exc=num_exc,
                       nearest_neighbors=nearest_neighbors, harmonic_length_optimization=harmonic_length_optimization,
                       optimize_all_minima=optimize_all_minima)

    def _build_U_squeezing_operator(self, minimum, Xi):
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
        M_matrix = self._squeezing_M_builder(minimum, Xi, Xi_prime)
        dim = self.number_degrees_freedom
        u = M_matrix[0: dim, 0: dim]
        v = M_matrix[dim: 2*dim, 0: dim]
        rho = inv(u) @ v
        sigma = logm(u)
        tau = v @ inv(u)
        return rho, sigma, tau

    def _helper_squeezing_matrices(self, rho, rho_prime, Xi):
        """Build variables helpful for constructing the Hamiltonian """
        dim = self.number_degrees_freedom
        Xi_inv = inv(Xi)
        delta_rho_prime = inv(np.eye(dim) - rho_prime @ rho) @ rho_prime
        delta_rho = inv(np.eye(dim) - rho @ rho_prime) @ rho
        delta_rho_bar = logm(inv(np.eye(dim) - rho_prime @ rho))
        z = 1j * Xi_inv.T / np.sqrt(2.)
        zp = z + 0.5 * z @ rho_prime @ (delta_rho + delta_rho.T) + 0.5 * z @ (delta_rho + delta_rho.T)
        zpp = z @ rho_prime + z
        return delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp

    def _squeezing_M_builder(self, minimum, Xi, Xi_prime):
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
        omega_matrix = np.diag(self.omega_matrix(minimum))
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

    def _order_eigensystem_squeezing(self, eigvals, eigvec):
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

    def _normalize_symplectic_eigensystem_squeezing(self, eigvals, eigvec):
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

    def _find_closest_periodic_minimum(self, minima_pair):
        (minima_m, m), (minima_p, p) = minima_pair
        max_for_m = self._find_closest_periodic_minimum_for_given_minima(minima_pair, m)
        max_for_p = self._find_closest_periodic_minimum_for_given_minima(minima_pair, p)
        return max(max_for_m, max_for_p)

    def _normal_ordered_a_dagger_a_exponential(self, x, a_operator_list):
        """Return normal ordered exponential matrix of exp(a_{i}^{\dagger}x_{ij}a_{j})"""
        expm_x = expm(x)
        num_states = self.number_states_per_minimum()
        dim = self.number_degrees_freedom
        result = np.eye(num_states, dtype=np.complex128)
        additional_term = np.eye(num_states, dtype=np.complex128)
        k = 1
        while not np.allclose(additional_term, np.zeros((num_states, num_states))):
            additional_term = np.sum([((expm_x - np.eye(dim))[i, j]) ** k * (factorial(k)) ** (-1)
                                     * matrix_power(a_operator_list[i].T, k) @ matrix_power(a_operator_list[j], k)
                                     for i in range(dim) for j in range(dim)], axis=0)
            result += additional_term
            k += 1
        return result

    def _build_rho_sigma_tau_matrices(self, m, p, Xi):
        dim = self.number_degrees_freedom
        if m == 0:  # At the global minimum, no squeezing required
            rho = np.zeros((dim, dim))
            sigma = np.zeros((dim, dim))
            tau = np.zeros((dim, dim))
        else:
            rho, sigma, tau = self._build_U_squeezing_operator(m, Xi)
        if p == 0:
            rho_prime = np.zeros((dim, dim))
            sigma_prime = np.zeros((dim, dim))
            tau_prime = np.zeros((dim, dim))
        elif p == m:
            rho_prime = np.copy(rho)
            sigma_prime = np.copy(sigma)
            tau_prime = np.copy(tau)
        else:
            rho_prime, sigma_prime, tau_prime = self._build_U_squeezing_operator(p, Xi)
        return rho, rho_prime, sigma, sigma_prime, tau, tau_prime

    def find_relevant_periodic_continuation_vectors(self, num_cpus=1):
        """
        We have found that specifically this part of the code is quite slow, that
        is finding the relevant nearest neighbor, next nearest neighbor, etc. lattice vectors
        that meaningfully contribute. This is a calculation that previously had to be done
        for the kinetic, potential and inner product matrices separately, even though
        the results were the same for all three matrices. This helper function allows us to only
        do it once.
        """
        minima_list = self.sorted_minima()
        number_of_minima = len(minima_list)
        Xi_inv_list = np.array([inv(self.Xi_matrix(minimum=minimum)) for minimum, _ in enumerate(minima_list)])
        nearest_neighbors = {}
        minima_list_with_index = zip(minima_list, [m for m in range(number_of_minima)])
        all_minima_pairs = itertools.combinations_with_replacement(minima_list_with_index, 2)
        for (minima_m, m), (minima_p, p) in all_minima_pairs:
            minima_diff = Xi_inv_list[p] @ minima_p - Xi_inv_list[m] @ minima_m
            nearest_neighbors[str(m)+str(p)] = self._filter_for_minima_pair(minima_diff, Xi_inv_list[p], num_cpus)
            print("completed m={m}, p={p} minima pair computation".format(m=m, p=p))
        self.nearest_neighbors = nearest_neighbors

    def _build_translation_operators(self, minima_diff, Xi, disentangled_squeezing_matrices, helper_squeezing_matrices):
        dim = self.number_degrees_freedom
        a_operator_list = self.a_operator_list()
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        prefactor_a_dagger = (np.eye(dim) + rho_prime) @ expm(delta_rho_bar).T @ expm(-sigma)
        prefactor_a = (np.eye(dim) + 0.5 * (np.eye(dim) + rho_prime) @ (delta_rho + delta_rho.T)) @ expm(-sigma_prime)
        Xi_inv = inv(Xi)
        exp_a_dagger_list = np.array([expm(np.sum([2.0 * np.pi * (Xi_inv.T @ prefactor_a_dagger)[j, i] 
                                                   * a_operator_list[i].T for i in range(dim)], axis=0) 
                                           / np.sqrt(2.0)) for j in range(dim)])
        exp_a_dagger_minima_difference = expm(np.sum([minima_diff[j] * (Xi_inv.T @ prefactor_a_dagger)[j, i] 
                                                      * a_operator_list[i].T for i in range(dim) 
                                                      for j in range(dim)], axis=0) / np.sqrt(2.0))
        exp_a_list = np.array([expm(np.sum([2.0 * np.pi * (Xi_inv.T @ prefactor_a)[j, i] * a_operator_list[i]
                               for i in range(dim)], axis=0) / np.sqrt(2.0)) for j in range(dim)])
        exp_a_minima_difference = expm(np.sum([-minima_diff[j] * (Xi_inv.T @ prefactor_a)[j, i] * a_operator_list[i]
                                       for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        return exp_a_dagger_list, exp_a_dagger_minima_difference, exp_a_list, exp_a_minima_difference

    def _build_potential_operators(self, a_operator_list, Xi, exp_a_dagger_a,
                                   disentangled_squeezing_matrices, helper_squeezing_matrices):
        exp_i_list = []
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        prefactor_a_dagger = (np.eye(dim) - rho_prime) @ expm(delta_rho_bar).T @ expm(-sigma)
        prefactor_a = (np.eye(dim) - 0.5 * (np.eye(dim) - rho_prime) @ (delta_rho + delta_rho.T)) @ expm(-sigma_prime)

        for j in range(dim):
            exp_i_j_a_dagger_part = expm(np.sum([1j * (Xi @ prefactor_a_dagger)[j, i] * a_operator_list[i].T
                                         for i in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j_a_part = expm(np.sum([1j * (Xi @ prefactor_a)[j, i] * a_operator_list[i]
                                          for i in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j = exp_i_j_a_dagger_part @ exp_a_dagger_a @ exp_i_j_a_part
            exp_i_list.append(exp_i_j)

        exp_i_sum_a_dagger_part = expm(np.sum([1j * self.boundary_coefficients[j]
                                       * (Xi @ prefactor_a_dagger)[j, i] * a_operator_list[i].T
                                           for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        exp_i_sum_a_part = expm(np.sum([1j * self.boundary_coefficients[j]
                                        * (Xi @ prefactor_a)[j, i] * a_operator_list[i]
                                        for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        exp_i_sum = exp_i_sum_a_dagger_part @ exp_a_dagger_a @ exp_i_sum_a_part
        return exp_i_list, exp_i_sum

    def _build_squeezing_operators(self, a_operator_list, disentangled_squeezing_matrices, helper_squeezing_matrices):
        """
        Build all operators relevant for building the Hamiltonian. If there is no squeezing,
        this routine then just builds the translation operators necessary for periodic
        continuation, as well as the exp(i\phi_{j}) operators for the potential
        """
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices

        prefactor_a_dagger_a_dagger = 0.5 * (tau.T - expm(-sigma).T @ delta_rho_prime @ expm(-sigma))
        prefactor_a_a = 0.5 * (tau_prime - expm(-sigma_prime).T @ delta_rho @ expm(-sigma_prime))
        prefactor_a_dagger_a = sp.linalg.logm(expm(-sigma).T @ expm(delta_rho_bar) @ expm(-sigma_prime))

        exp_a_dagger_a_dagger = expm(np.sum([prefactor_a_dagger_a_dagger[i, j]
                                             * a_operator_list[i].T @ a_operator_list[j].T
                                     for i in range(dim) for j in range(dim)], axis=0))
        exp_a_a = expm(np.sum([prefactor_a_a[i, j] * a_operator_list[i] @ a_operator_list[j]
                               for i in range(dim) for j in range(dim)], axis=0))
        exp_a_dagger_a = self._normal_ordered_a_dagger_a_exponential(prefactor_a_dagger_a, a_operator_list)
        return exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a

    def _translation_squeezing(self, exp_a_dagger_list, exp_a_dagger_minima_difference,
                               exp_a_list, exp_a_minima_difference, exp_a_dagger_a_dagger,
                               exp_a_a, neighbor):
        """
        Build translation operators using matrix_power rather than the
        more costly expm
        """
        num_exc = self.number_states_per_minimum()
        dim = self.number_degrees_freedom
        translation_op_a_dag = np.eye(num_exc)
        translation_op_a = np.eye(num_exc)
        for j in range(dim):
            translation_op_a_dag_for_direction = matrix_power(exp_a_dagger_list[j], int(neighbor[j]))
            translation_op_a_dag = translation_op_a_dag @ translation_op_a_dag_for_direction
        translation_op_a_dag = translation_op_a_dag @ exp_a_dagger_minima_difference @ exp_a_dagger_a_dagger
        for j in range(dim):
            translation_op_a_for_direction = matrix_power(exp_a_list[j], -int(neighbor[j]))
            translation_op_a = translation_op_a @ translation_op_a_for_direction
        translation_op_a = translation_op_a @ exp_a_minima_difference @ exp_a_a
        return translation_op_a_dag, translation_op_a

    def _periodic_continuation_squeezing(self, minima_pair_func, local_func):
        """See VCHOS for documentation. This function generalizes _periodic_continuation to allow for squeezing"""
        if not self.nearest_neighbors:
            self.find_relevant_periodic_continuation_vectors()
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_list = self.a_operator_list()
        minima_list = self.sorted_minima()
        hilbertdim = self.hilbertdim()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                disentangled_squeezing_matrices = self._build_rho_sigma_tau_matrices(m, p, Xi)
                rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
                helper_squeezing_matrices = self._helper_squeezing_matrices(rho, rho_prime, Xi)
                squeezing_operators = self._build_squeezing_operators(a_operator_list, disentangled_squeezing_matrices,
                                                                      helper_squeezing_matrices)
                exp_a_dagger_a_dagger, exp_a_dagger_a, exp_a_a = squeezing_operators
                exp_operators = self._build_translation_operators(minima_diff, Xi, disentangled_squeezing_matrices,
                                                                  helper_squeezing_matrices)
                exp_a_dagger_list, exp_a_dagger_minima_difference, exp_a_list, exp_a_minima_difference = exp_operators
                minima_pair_results = minima_pair_func(exp_a_dagger_a, disentangled_squeezing_matrices,
                                                       helper_squeezing_matrices)
                scale = 1. / np.sqrt(det(np.eye(self.number_degrees_freedom) - np.matmul(rho, rho_prime)))
                for neighbor in self.nearest_neighbors[str(m)+str(p)]:
                    phi_neighbor = 2.0 * np.pi * np.array(neighbor)
                    translation_operators = self._translation_squeezing(exp_a_dagger_list,
                                                                        exp_a_dagger_minima_difference, exp_a_list,
                                                                        exp_a_minima_difference, exp_a_dagger_a_dagger,
                                                                        exp_a_a, neighbor)
                    translation_a_dagger, translation_a = translation_operators
                    exp_prod_coefficient = self._exp_product_coefficient_squeezing(phi_neighbor + minima_p - minima_m,
                                                                                   Xi_inv, sigma, sigma_prime)
                    matrix_element = (scale * exp_prod_coefficient * translation_a_dagger
                                      @ local_func(phi_neighbor, minima_m, minima_p,
                                                   disentangled_squeezing_matrices, helper_squeezing_matrices,
                                                   exp_a_dagger_a, minima_pair_results)
                                      @ translation_a)
                    operator_matrix[m * num_states_min: (m + 1) * num_states_min,
                                    p * num_states_min: (p + 1) * num_states_min] += matrix_element
        operator_matrix = self._populate_hermitian_matrix(operator_matrix)
        return operator_matrix

    def _construct_kinetic_alpha_epsilon_squeezing(self, Xi_inv, delta_phi, rho_prime, delta_rho):
        arg_exp_a_dag = delta_phi @ Xi_inv.T / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho)
        delta_rho_pp = 0.5 * (arg_exp_a_dag - arg_exp_a @ rho_prime) @ (delta_rho + delta_rho.T)
        epsilon = -(1j / np.sqrt(2.0)) * Xi_inv.T @ (rho_prime @ delta_rho_pp - arg_exp_a @ rho_prime + delta_rho_pp
                                                     + Xi_inv @ delta_phi / np.sqrt(2.0))
        return alpha, epsilon

    def _minima_pair_transfer_squeezing_function(self, EC_mat, a_operator_list, Xi, exp_a_dagger_a,
                                                 disentangled_squeezing_matrices, helper_squeezing_matrices):
        return (self._minima_pair_kinetic_squeezing_function(EC_mat, a_operator_list, exp_a_dagger_a,
                                                             disentangled_squeezing_matrices,
                                                             helper_squeezing_matrices),
                self._minima_pair_potential_squeezing_function(a_operator_list, Xi, exp_a_dagger_a,
                                                               disentangled_squeezing_matrices,
                                                               helper_squeezing_matrices))

    def _minima_pair_kinetic_squeezing_function(self, EC_mat, a_operator_list, exp_a_dagger_a,
                                                disentangled_squeezing_matrices, helper_squeezing_matrices):
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        (xa, xaa, dxa, dx, ddx) = self._premultiplying_exp_a_dagger_a_with_a(exp_a_dagger_a, a_operator_list)
        sigma_delta_rho_bar_zpp_EC = expm(-sigma).T @ expm(delta_rho_bar) @ zpp.T @ EC_mat
        xaa_coefficient = (zp @ expm(-sigma_prime)).T @ EC_mat @ zp @ expm(-sigma_prime)
        dxa_coefficient = sigma_delta_rho_bar_zpp_EC @ zp @ expm(-sigma_prime)
        ddx_coefficient = sigma_delta_rho_bar_zpp_EC @ (expm(-sigma).T @ expm(delta_rho_bar) @ zpp.T).T
        x_coefficient = zpp.T @ EC_mat @ zp
        xa_coefficient = EC_mat @ zp @ expm(-sigma_prime)
        dx_coefficient = EC_mat @ zpp @ (expm(-sigma).T @ expm(delta_rho_bar)).T
        kinetic_matrix = np.sum([+4 * xaa[mu] * xaa_coefficient[mu, mu] - 8 * dxa[mu] * dxa_coefficient[mu, mu]
                                 + 4 * ddx[mu] * ddx_coefficient[mu, mu] - 4 * exp_a_dagger_a * x_coefficient[mu, mu]
                                 for mu in range(dim)], axis=0)
        return kinetic_matrix, xa, dx, xa_coefficient, dx_coefficient

    def _local_transfer_squeezing_function(self, EC_mat, Xi, Xi_inv, exp_product_boundary_coefficient,
                                           phi_neighbor, minima_m, minima_p,
                                           disentangled_squeezing_matrices, helper_squeezing_matrices,
                                           exp_a_dagger_a, minima_pair_results):
        kinetic_minima_pair_results, potential_minima_pair_results = minima_pair_results
        return (self._local_kinetic_squeezing_function(EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                       disentangled_squeezing_matrices, helper_squeezing_matrices,
                                                       exp_a_dagger_a, kinetic_minima_pair_results)
                + self._local_potential_squeezing_function(Xi, Xi_inv, exp_product_boundary_coefficient, phi_neighbor,
                                                           minima_m, minima_p, disentangled_squeezing_matrices,
                                                           helper_squeezing_matrices, exp_a_dagger_a,
                                                           potential_minima_pair_results))

    def _local_kinetic_squeezing_function(self, EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                          disentangled_squeezing_matrices, helper_squeezing_matrices,
                                          exp_a_dagger_a, minima_pair_results):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        kinetic_matrix_minima_pair, xa, dx, xa_coefficient, dx_coefficient = minima_pair_results
        alpha, epsilon = self._construct_kinetic_alpha_epsilon_squeezing(Xi_inv, delta_phi, rho_prime, delta_rho)
        e_xa_coefficient = epsilon @ xa_coefficient
        e_dx_coefficient = epsilon @ dx_coefficient
        kinetic_matrix = np.sum([-8 * xa[mu] * e_xa_coefficient[mu] + 8 * dx[mu] * e_dx_coefficient[mu]
                                 for mu in range(dim)], axis=0)
        kinetic_matrix += kinetic_matrix_minima_pair
        kinetic_matrix += 4 * exp_a_dagger_a * (epsilon @ EC_mat @ epsilon)
        kinetic_matrix *= alpha

        return kinetic_matrix

    def _premultiplying_exp_a_dagger_a_with_a(self, exp_a_dagger_a, a_op_list):
        """
        Helper function for building the kinetic part of the Hamiltonian.
        Naming scheme is  x -> exp(A_{ij}a_{i}^{\dag}a_{j}) (for whatever matrix A is)
                          a -> a_{i}
                          d -> a_{i}^{\dag}
        """
        dim = self.number_degrees_freedom
        xa = np.array([exp_a_dagger_a @ a_op_list[mu] for mu in range(dim)])
        xaa = np.array([xa[mu] @ a_op_list[mu] for mu in range(dim)])
        dxa = np.array([a_op_list[mu].T @ xa[mu] for mu in range(dim)])
        dx = np.array([a_op_list[mu].T @ exp_a_dagger_a for mu in range(dim)])
        ddx = np.array([a_op_list[mu].T @ dx[mu] for mu in range(dim)])
        return xa, xaa, dxa, dx, ddx

    @staticmethod
    def _alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho):
        """Build the prefactor that arises due to squeezing. With no squeezing, alpha=1 (number, not matrix)"""
        arg_exp_a_rho_prime = np.matmul(arg_exp_a, rho_prime)
        alpha = np.exp(-0.5 * arg_exp_a @ arg_exp_a_rho_prime - 0.5 * (arg_exp_a_dag - arg_exp_a_rho_prime)
                       @ delta_rho @ (arg_exp_a_dag - arg_exp_a_rho_prime))
        return alpha

    def transfer_matrix(self):
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_list = self.a_operator_list()
        EC_mat = self.build_EC_matrix()
        exp_product_boundary_coefficient = self._exp_product_boundary_coefficient(Xi)
        minima_pair_transfer_function = partial(self._minima_pair_transfer_squeezing_function, EC_mat,
                                                a_operator_list, Xi)
        local_transfer_function = partial(self._local_transfer_squeezing_function, EC_mat, Xi, Xi_inv,
                                          exp_product_boundary_coefficient)
        return self._periodic_continuation_squeezing(minima_pair_transfer_function, local_transfer_function)

    def kinetic_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        Xi_inv = inv(self.Xi_matrix())
        a_operator_list = self.a_operator_list()
        EC_mat = self.build_EC_matrix()
        minima_pair_kinetic_function = partial(self._minima_pair_kinetic_squeezing_function, EC_mat, a_operator_list)
        local_kinetic_function = partial(self._local_kinetic_squeezing_function, EC_mat, Xi_inv)
        return self._periodic_continuation_squeezing(minima_pair_kinetic_function, local_kinetic_function)

    def potential_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_list = self.a_operator_list()
        exp_product_boundary_coefficient = self._exp_product_boundary_coefficient(Xi)
        minima_pair_potential_function = partial(self._minima_pair_potential_squeezing_function, a_operator_list, Xi)
        local_potential_function = partial(self._local_potential_squeezing_function, Xi, Xi_inv,
                                           exp_product_boundary_coefficient)
        return self._periodic_continuation_squeezing(minima_pair_potential_function, local_potential_function)

    def _local_potential_squeezing_function(self, Xi, Xi_inv, exp_product_boundary_coefficient,
                                            phi_neighbor, minima_m, minima_p,
                                            disentangled_squeezing_matrices, helper_squeezing_matrices,
                                            exp_a_dagger_a, minima_pair_results):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_list, exp_i_sum = minima_pair_results
        exp_i_phi_list = np.array([exp_i_list[i] * np.exp(1j * phi_bar[i]) for i in range(dim)])
        exp_i_phi_sum_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                            * np.prod([np.exp(1j * self.boundary_coefficients[i] * phi_bar[i]) for i in range(dim)]))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      helper_squeezing_matrices,
                                                                                      exp_i_phi_list)
                                   for j in range(dim)], axis=0)
        potential_matrix += self._local_contribution_boundary_squeezing(delta_phi, Xi, Xi_inv,
                                                                        disentangled_squeezing_matrices,
                                                                        helper_squeezing_matrices,
                                                                        exp_i_phi_sum_op,
                                                                        exp_product_boundary_coefficient)
        potential_matrix += (self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                         disentangled_squeezing_matrices,
                                                                         helper_squeezing_matrices, exp_a_dagger_a,
                                                                         minima_pair_results)
                             * np.sum(self.EJlist))
        return potential_matrix

    def _local_contribution_boundary_squeezing(self, delta_phi, Xi, Xi_inv,
                                               disentangled_squeezing_matrices,
                                               helper_squeezing_matrices, exp_i_sum, exp_product_boundary_coefficient):
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + np.sum([1j * Xi[i, :] * self.boundary_coefficients[i]
                                                     for i in range(dim)], axis=0)) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rho_prime, delta_rho)
        alpha_conjugate = self._alpha_helper(arg_exp_a_dag.conjugate(), -arg_exp_a_dag, rho_prime, delta_rho)
        potential_matrix = -0.5 * self.EJlist[-1] * alpha * exp_i_sum
        potential_matrix += -0.5 * self.EJlist[-1] * alpha_conjugate * exp_i_sum.conjugate()
        potential_matrix *= exp_product_boundary_coefficient
        return potential_matrix

    def _local_contribution_single_junction_squeezing(self, j, delta_phi, Xi, Xi_inv, disentangled_squeezing_matrices,
                                                      helper_squeezing_matrices, exp_i_phi_list):
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + 1j * Xi[j, :]) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rho_prime, delta_rho)
        alpha_conjugate = self._alpha_helper(arg_exp_a_dag.conjugate(), -arg_exp_a_dag, rho_prime, delta_rho)
        potential_matrix = -0.5 * self.EJlist[j] * alpha * exp_i_phi_list[j]
        # No need to .T the h.c. term, all that is necessary is conjugation
        potential_matrix += -0.5 * self.EJlist[j] * alpha_conjugate * exp_i_phi_list[j].conjugate()
        potential_matrix *= np.exp(-.25 * np.dot(Xi[j, :], Xi.T[:, j]))
        return potential_matrix

    def _minima_pair_potential_squeezing_function(self, a_operator_list, Xi, exp_a_dagger_a,
                                                  disentangled_squeezing_matrices, helper_squeezing_matrices):
        return self._build_potential_operators(a_operator_list, Xi, exp_a_dagger_a,
                                               disentangled_squeezing_matrices, helper_squeezing_matrices)

    def _local_contribution_identity_squeezing(self, Xi_inv, phi_neighbor, minima_m, minima_p,
                                               disentangled_squeezing_matrices, helper_squeezing_matrices,
                                               exp_a_dagger_a, minima_pair_results):
        _ = minima_pair_results
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        arg_exp_a_dag = np.matmul(delta_phi, Xi_inv.T) / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho)
        return alpha * exp_a_dagger_a

    def inner_product_matrix(self):
        Xi_inv = inv(self.Xi_matrix())
        local_identity_function = partial(self._local_contribution_identity_squeezing, Xi_inv)
        return self._periodic_continuation_squeezing(lambda x, y, z: None, local_identity_function)

    def _exp_product_coefficient_squeezing(self, delta_phi, Xi_inv, sigma, sigma_prime):
        """
        Overall multiplicative factor. Includes offset charge,
        Gaussian suppression factor in the absence of squeezing. With squeezing,
        also includes exponential of trace over sigma and sigma_prime, see Qin et. al
        """
        dpkX = np.matmul(Xi_inv, delta_phi)
        nglist = self.nglist
        return (np.exp(-1j * np.dot(nglist, delta_phi))
                * np.exp(-0.5 * np.trace(sigma) - 0.5 * np.trace(sigma_prime))
                * np.exp(-0.25 * np.dot(dpkX, dpkX)))

    def _exp_product_boundary_coefficient(self, Xi):
        dim = self.number_degrees_freedom
        return np.exp(-0.25 * np.sum([self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                      * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                      for j in range(dim) for k in range(dim)]))

    def optimize_Xi_variational_wrapper(self, num_cpus=1):
        minima_list = self.sorted_minima()
        self.optimized_lengths = np.ones((len(minima_list), self.number_degrees_freedom))
        self.optimize_Xi_variational(0, minima_list[0])
        Xi_global = self.Xi_matrix(minimum=0)
        harmonic_lengths_global = np.array([np.linalg.norm(Xi_global[:, i])
                                            for i in range(self.number_degrees_freedom)])
        for minimum, minimum_location in enumerate(minima_list):
            if self.optimize_all_minima and minimum != 0:
                self.optimize_Xi_variational(minimum, minimum_location)
            elif minimum != 0:
                Xi_local = self.Xi_matrix(minimum=minimum)
                harmonic_lengths_local = np.array([np.linalg.norm(Xi_local[:, i])
                                                   for i in range(self.number_degrees_freedom)])
                self.optimized_lengths[minimum] = harmonic_lengths_global / harmonic_lengths_local

    def _one_state_periodic_continuation_squeezing(self, minimum_location, minimum, nearest_neighbors,
                                                   local_func, Xi, Xi_inv):
        disentangled_squeezing_matrices = self._build_rho_sigma_tau_matrices(minimum, minimum, Xi)
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        helper_squeezing_matrices = self._helper_squeezing_matrices(rho, rho_prime, Xi)
        scale = 1. / np.sqrt(det(np.eye(self.number_degrees_freedom) - np.matmul(rho, rho_prime)))
        ground_state_value = 0.0 + 0.0j
        for neighbor in nearest_neighbors:
            phi_neighbor = 2.0 * np.pi * np.array(neighbor)
            exp_prod_coefficient = self._exp_product_coefficient_squeezing(phi_neighbor, Xi_inv, sigma, sigma_prime)
            ground_state_value += (scale * exp_prod_coefficient * local_func(phi_neighbor, minimum_location,
                                                                             minimum_location,
                                                                             disentangled_squeezing_matrices,
                                                                             helper_squeezing_matrices))
        return ground_state_value

    def _one_state_local_identity_squeezing(self, Xi_inv, phi_neighbor, minima_m, minima_p,
                                            disentangled_squeezing_matrices, helper_squeezing_matrices):
        return self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                           disentangled_squeezing_matrices, helper_squeezing_matrices,
                                                           1.0, None)

    def _one_state_local_transfer_squeezing(self, EC_mat, Xi, Xi_inv, exp_product_boundary_coefficient,
                                            phi_neighbor, minima_m, minima_p,
                                            disentangled_squeezing_matrices, helper_squeezing_matrices):
        return (self._one_state_local_kinetic_squeezing_function(EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                 disentangled_squeezing_matrices,
                                                                 helper_squeezing_matrices)
                + self._one_state_local_potential_squeezing_function(Xi, Xi_inv, exp_product_boundary_coefficient,
                                                                     phi_neighbor, minima_m, minima_p,
                                                                     disentangled_squeezing_matrices,
                                                                     helper_squeezing_matrices))

    def _one_state_local_potential_squeezing_function(self, Xi, Xi_inv, exp_product_boundary_coefficient,
                                                      phi_neighbor, minima_m, minima_p,
                                                      disentangled_squeezing_matrices, helper_squeezing_matrices):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_list = np.array([np.exp(1j * phi_bar[i]) for i in range(dim)])
        exp_i_phi_sum_op = (np.exp(1j * 2.0 * np.pi * self.flux)
                            * np.prod([np.exp(1j * self.boundary_coefficients[i] * phi_bar[i]) for i in range(dim)]))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      helper_squeezing_matrices,
                                                                                      exp_i_phi_list)
                                   for j in range(dim)], axis=0)
        potential_matrix += self._local_contribution_boundary_squeezing(delta_phi, Xi, Xi_inv,
                                                                        disentangled_squeezing_matrices,
                                                                        helper_squeezing_matrices,
                                                                        exp_i_phi_sum_op,
                                                                        exp_product_boundary_coefficient)
        potential_matrix += (self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                         disentangled_squeezing_matrices,
                                                                         helper_squeezing_matrices, 1.0, None)
                             * np.sum(self.EJlist))
        return potential_matrix

    def _one_state_local_kinetic_squeezing_function(self, EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                    disentangled_squeezing_matrices, helper_squeezing_matrices):
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = helper_squeezing_matrices
        alpha, epsilon = self._construct_kinetic_alpha_epsilon_squeezing(Xi_inv, delta_phi, rho_prime, delta_rho)
        result = 4 * alpha * (epsilon @ EC_mat @ epsilon - np.trace(zpp.T @ EC_mat @ zp))
        return result

    def _evals_calc_variational(self, optimized_lengths, minimum_location, minimum, EC_mat, default_Xi):
        self.optimized_lengths[minimum] = optimized_lengths
        Xi = self._update_Xi(default_Xi, minimum)
        Xi_inv = inv(Xi)
        exp_product_boundary_coefficient = self._exp_product_boundary_coefficient(Xi)
        transfer, inner = self._one_state_construct_transfer_inner_squeezing(Xi, Xi_inv, minimum_location, minimum,
                                                                             EC_mat, exp_product_boundary_coefficient)
        return np.real([transfer / inner])

    def _one_state_construct_transfer_inner_squeezing(self, Xi, Xi_inv, minimum_location, minimum, EC_mat,
                                                      exp_product_boundary_coefficient):
        if not self.nearest_neighbors:
            self.find_relevant_periodic_continuation_vectors()
        nearest_neighbors = self.nearest_neighbors[str(minimum) + str(minimum)]
        transfer_function = partial(self._one_state_local_transfer_squeezing, EC_mat, Xi, Xi_inv,
                                    exp_product_boundary_coefficient)
        inner_product_function = partial(self._one_state_local_identity_squeezing, Xi_inv)
        transfer = self._one_state_periodic_continuation_squeezing(minimum_location, minimum, nearest_neighbors,
                                                                   transfer_function, Xi, Xi_inv)
        inner_product = self._one_state_periodic_continuation_squeezing(minimum_location, minimum, nearest_neighbors,
                                                                        inner_product_function, Xi, Xi_inv)
        return transfer, inner_product
