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
                 number_periodic_degrees_freedom=0, num_exc=None, optimized_lengths=None, nearest_neighbors=None):
        VCHOS.__init__(self, EJlist, nglist, flux, maximum_periodic_vector_length,
                       number_degrees_freedom=number_degrees_freedom,
                       number_periodic_degrees_freedom=number_periodic_degrees_freedom, num_exc=num_exc,
                       optimized_lengths=optimized_lengths, nearest_neighbors=nearest_neighbors)
        self.boundary_coefficients = np.array([])

    def _build_U_squeezing_operator(self, i, Xi):
        """
        Return the rho, sigma, tau matrices that define the overall squeezing operator U

        Parameters
        ----------
        i: int
            integer representing the minimum for which to build the squeezing operator U,
            0<i<=total number of minima (no squeezing need be performed for the global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times

        Returns
        -------
        ndarray, ndarray, ndarray
        """
        uvmat = self._squeezing_M_builder(i, Xi)
        dim = self.number_degrees_freedom
        u = uvmat[0: dim, 0: dim]
        v = uvmat[dim: 2*dim, 0: dim]
        rho = inv(u) @ v
        sigma = logm(u)
        tau = v @ inv(u)
        return rho, sigma, tau

    def _helper_squeezing_matrices(self, rho, rhoprime, Xi):
        """Build variables helpful for constructing the Hamiltonian """
        dim = self.number_degrees_freedom
        Xi_inv = inv(Xi)
        deltarhoprime = inv(np.eye(dim) - rhoprime @ rho) @ rhoprime
        deltarho = inv(np.eye(dim) - rho @ rhoprime) @ rho
        deltarhobar = logm(inv(np.eye(dim) - rhoprime @ rho))
        z = 1j * Xi_inv.T / np.sqrt(2.)
        zp = z + 0.5 * z @ rhoprime @ (deltarho + deltarho.T) + 0.5 * z @ (deltarho + deltarho.T)
        zpp = z @ rhoprime + z
        return deltarho, deltarhoprime, deltarhobar, zp, zpp

    def _squeezing_M_builder(self, i, Xi):
        """
        Returns the M matrix as defined in G. Qin et. al “General multimodesqueezed states,”
        (2001) arXiv:quant-ph/0109020, M=[[u, v],[v, u]] where u and v are the matrices
        that define the Bogoliubov transformation
        Parameters
        ----------
        i: int
            integer representing the minimum for which to build the squeezing operator U,
            0<i<=total number of minima (no squeezing need be performed for the global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times

        Returns
        -------
        ndarray
        """
        gamma = self.build_gamma_matrix(i)
        gamma_prime = Xi.T @ gamma @ Xi
        omegamat = self.omega_matrix(i)
        zeta = 0.25 * (self.Phi0 ** 2 * gamma_prime + np.diag(omegamat))
        eta = 0.25 * (self.Phi0 ** 2 * gamma_prime - np.diag(omegamat))
        hmat = np.block([[zeta, -eta],
                         [eta, -zeta]])
        eigvals, eigvec = sp.linalg.eig(hmat)
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

    def _normal_ordered_adag_a_exponential(self, x, a_operator_list):
        """Return normal ordered exponential matrix of exp(a_{i}^{\dagger}x_{ij}a_{j})"""
        expx = expm(x)
        num_states = self.number_states_per_minimum()
        dim = self.number_degrees_freedom
        result = np.eye(num_states, dtype=np.complex128)
        additional_term = np.eye(num_states, dtype=np.complex128)
        k = 1
        while not np.allclose(additional_term, np.zeros((num_states, num_states))):
            additional_term = np.sum([((expx - np.eye(dim))[i, j]) ** k * (factorial(k)) ** (-1)
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
            rhoprime = np.zeros((dim, dim))
            sigmaprime = np.zeros((dim, dim))
            tauprime = np.zeros((dim, dim))
        elif p == m:
            rhoprime = np.copy(rho)
            sigmaprime = np.copy(sigma)
            tauprime = np.copy(tau)
        else:
            rhoprime, sigmaprime, tauprime = self._build_U_squeezing_operator(p, Xi)
        return rho, rhoprime, sigma, sigmaprime, tau, tauprime

    def _build_translation_operators(self, minima_diff, Xi, disentangled_squeezing_matrices, helper_squeezing_matrices):
        dim = self.number_degrees_freedom
        a_operator_list = self.a_operator_list()
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        prefactor_adag = (np.eye(dim) + rhoprime) @ expm(deltarhobar).T @ expm(-sigma)
        a_temp_coeff = 0.5 * (np.eye(dim) + rhoprime) @ (deltarho + deltarho.T)
        prefactor_a = (np.eye(dim) + a_temp_coeff) @ expm(-sigmaprime)
        Xi_inv = inv(Xi)
        exp_adag_list = np.array([expm(np.sum([2.0 * np.pi * (Xi_inv.T @ prefactor_adag)[j, i] * a_operator_list[i].T
                                  for i in range(dim)], axis=0) / np.sqrt(2.0)) for j in range(dim)])
        exp_adag_mindiff = expm(np.sum([minima_diff[j] * (Xi_inv.T @ prefactor_adag)[j, i] * a_operator_list[i].T
                                        for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        exp_a_list = np.array([expm(np.sum([2.0 * np.pi * (Xi_inv.T @ prefactor_a)[j, i] * a_operator_list[i]
                               for i in range(dim)], axis=0) / np.sqrt(2.0)) for j in range(dim)])
        exp_a_mindiff = expm(np.sum([-minima_diff[j] * (Xi_inv.T @ prefactor_a)[j, i] * a_operator_list[i]
                                     for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        return exp_adag_list, exp_adag_mindiff, exp_a_list, exp_a_mindiff

    def _build_potential_operators(self, a_operator_list, Xi, exp_adag_a,
                                   disentangled_squeezing_matrices, helper_squeezing_matrices):
        exp_i_list = []
        dim = self.number_degrees_freedom
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        prefactor_adag = (np.eye(dim) - rhoprime) @ expm(deltarhobar).T @ expm(-sigma)
        a_temp_coeff = 0.5 * (np.eye(dim) - rhoprime) @ (deltarho + deltarho.T)
        prefactor_a = (np.eye(dim) - a_temp_coeff) @ expm(-sigmaprime)

        for j in range(dim):
            exp_i_j_adag_part = expm(np.sum([1j * (Xi @ prefactor_adag)[j, i] * a_operator_list[i].T
                                             for i in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j_a_part = expm(np.sum([1j * (Xi @ prefactor_a)[j, i] * a_operator_list[i]
                                          for i in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j = exp_i_j_adag_part @ exp_adag_a @ exp_i_j_a_part
            exp_i_list.append(exp_i_j)

        exp_i_sum_adag_part = expm(np.sum([1j * self.boundary_coefficients[j]
                                           * (Xi @ prefactor_adag)[j, i] * a_operator_list[i].T
                                           for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        exp_i_sum_a_part = expm(np.sum([1j * self.boundary_coefficients[j]
                                        * (Xi @ prefactor_a)[j, i] * a_operator_list[i]
                                        for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        exp_i_sum = exp_i_sum_adag_part @ exp_adag_a @ exp_i_sum_a_part
        return exp_i_list, exp_i_sum

    def _build_squeezing_operators(self, a_operator_list, disentangled_squeezing_matrices, helper_squeezing_matrices):
        """
        Build all operators relevant for building the Hamiltonian. If there is no squeezing,
        this routine then just builds the translation operators necessary for periodic
        continuation, as well as the exp(i\phi_{j}) operators for the potential
        """
        dim = self.number_degrees_freedom
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices

        prefactor_adag_adag = 0.5 * (tau.T - expm(-sigma).T @ deltarhoprime @ expm(-sigma))
        prefactor_a_a = 0.5 * (tauprime - expm(-sigmaprime).T @ deltarho @ expm(-sigmaprime))
        prefactor_adag_a = sp.linalg.logm(expm(-sigma).T @ expm(deltarhobar) @ expm(-sigmaprime))

        exp_adag_adag = expm(np.sum([prefactor_adag_adag[i, j] * a_operator_list[i].T @ a_operator_list[j].T
                                     for i in range(dim) for j in range(dim)], axis=0))
        exp_a_a = expm(np.sum([prefactor_a_a[i, j] * a_operator_list[i] @ a_operator_list[j]
                               for i in range(dim) for j in range(dim)], axis=0))
        exp_adag_a = self._normal_ordered_adag_a_exponential(prefactor_adag_a, a_operator_list)
        return exp_adag_adag, exp_adag_a, exp_a_a

    def _translation_squeezing(self, exp_adag_list, exp_adag_mindiff,
                               exp_a_list, exp_a_mindiff, exp_adag_adag,
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
            translation_op_a_dag_for_direction = matrix_power(exp_adag_list[j], int(neighbor[j]))
            translation_op_a_dag = translation_op_a_dag @ translation_op_a_dag_for_direction
        translation_op_a_dag = translation_op_a_dag @ exp_adag_mindiff @ exp_adag_adag
        for j in range(dim):
            translation_op_a_for_direction = matrix_power(exp_a_list[j], -int(neighbor[j]))
            translation_op_a = translation_op_a @ translation_op_a_for_direction
        translation_op_a = translation_op_a @ exp_a_mindiff @ exp_a_a
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
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                disentangled_squeezing_matrices = self._build_rho_sigma_tau_matrices(m, p, Xi)
                rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
                helper_squeezing_matrices = self._helper_squeezing_matrices(rho, rhoprime, Xi)
                exp_adag_adag, exp_adag_a, exp_a_a = self._build_squeezing_operators(a_operator_list,
                                                                                     disentangled_squeezing_matrices,
                                                                                     helper_squeezing_matrices)
                exp_operators = self._build_translation_operators(minima_diff, Xi, disentangled_squeezing_matrices,
                                                                  helper_squeezing_matrices)
                exp_adag_list, exp_adag_mindiff, exp_a_list, exp_a_mindiff = exp_operators
                minima_pair_results = minima_pair_func(exp_adag_a, disentangled_squeezing_matrices,
                                                       helper_squeezing_matrices)
                scale = 1. / np.sqrt(det(np.eye(self.number_degrees_freedom) - np.matmul(rho, rhoprime)))
                for neighbor in self.nearest_neighbors[counter]:
                    phi_neighbor = 2.0 * np.pi * np.array(neighbor)
                    translation_a_dag, translation_a = self._translation_squeezing(exp_adag_list, exp_adag_mindiff,
                                                                                   exp_a_list, exp_a_mindiff,
                                                                                   exp_adag_adag, exp_a_a, neighbor)
                    exp_prod_coefficient = self._exp_product_coefficient_squeezing(phi_neighbor + minima_p - minima_m,
                                                                                   Xi_inv, sigma, sigmaprime)
                    matrix_element = (scale * exp_prod_coefficient * translation_a_dag
                                      @ local_func(phi_neighbor, minima_m, minima_p,
                                                   disentangled_squeezing_matrices, helper_squeezing_matrices,
                                                   exp_adag_a, minima_pair_results)
                                      @ translation_a)
                    operator_matrix[m * num_states_min: (m + 1) * num_states_min,
                                    p * num_states_min: (p + 1) * num_states_min] += matrix_element
                counter += 1
        operator_matrix = self._populate_hermitian_matrix(operator_matrix)
        return operator_matrix
    
    def _minima_pair_kinetic_function(self, EC_mat, a_operator_list, exp_adag_a,
                                      disentangled_squeezing_matrices, helper_squeezing_matrices):
        dim = self.number_degrees_freedom
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        (xa, xaa, dxa, dx, ddx) = self._premultiplying_exp_adag_a_with_a(exp_adag_a, a_operator_list)
        xaa_coeff = (zp @ expm(-sigmaprime)).T @ EC_mat @ zp @ expm(-sigmaprime)
        dxa_coeff = expm(-sigma).T @ expm(deltarhobar) @ zpp.T @ EC_mat @ zp @ expm(-sigmaprime)
        ddx_coeff = expm(-sigma).T @ expm(deltarhobar) @ zpp.T @ EC_mat @ (expm(-sigma).T @ expm(deltarhobar) @ zpp.T).T
        x_coeff = zpp.T @ EC_mat @ zp 
        xa_coeff = EC_mat @ zp @ expm(-sigmaprime)
        dx_coeff = EC_mat @ zpp @ (expm(-sigma).T @ expm(deltarhobar)).T
        kinetic_matrix = np.sum([+4 * xaa[mu] * xaa_coeff[mu, mu] - 8 * dxa[mu] * dxa_coeff[mu, mu]
                                 + 4 * ddx[mu] * ddx_coeff[mu, mu] - 4 * exp_adag_a * x_coeff[mu, mu]
                                 for mu in range(dim)], axis=0)
        return kinetic_matrix, xa, dx, xa_coeff, dx_coeff

    def _local_kinetic_squeezing_function(self, EC_mat, Xi_inv, phi_neighbor, minima_m, minima_p,
                                          disentangled_squeezing_matrices, helper_squeezing_matrices,
                                          exp_adag_a, minima_pair_results):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        kinetic_matrix_minima_pair, xa, dx, xa_coeff, dx_coeff = minima_pair_results
        arg_exp_a_dag = delta_phi @ Xi_inv.T / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag

        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rhoprime, deltarho)
        deltarhopp = 0.5 * (arg_exp_a_dag - arg_exp_a @ rhoprime) @ (deltarho + deltarho.T)
        epsilon = -(1j / np.sqrt(2.0)) * Xi_inv.T @ (rhoprime @ deltarhopp - arg_exp_a @ rhoprime + deltarhopp
                                                     + Xi_inv @ delta_phi / np.sqrt(2.0))
        e_xa_coeff = epsilon @ xa_coeff
        e_dx_coeff = epsilon @ dx_coeff
        kinetic_matrix = np.sum([-8 * xa[mu] * e_xa_coeff[mu] + 8 * dx[mu] * e_dx_coeff[mu]
                                 for mu in range(dim)], axis=0)
        kinetic_matrix += kinetic_matrix_minima_pair
        kinetic_matrix += 4 * exp_adag_a * (epsilon @ EC_mat @ epsilon)
        kinetic_matrix *= alpha

        return kinetic_matrix

    def _premultiplying_exp_adag_a_with_a(self, exp_adag_a, a_op_list):
        """
        Helper function for building the kinetic part of the Hamiltonian.
        Naming scheme is  x -> exp(A_{ij}a_{i}^{\dag}a_{j}) (for whatever matrix A is)
                          a -> a_{i}
                          d -> a_{i}^{\dag}
        """
        dim = self.number_degrees_freedom
        xa = np.array([exp_adag_a @ a_op_list[mu] for mu in range(dim)])
        xaa = np.array([xa[mu] @ a_op_list[mu] for mu in range(dim)])
        dxa = np.array([a_op_list[mu].T @ xa[mu] for mu in range(dim)])
        dx = np.array([a_op_list[mu].T @ exp_adag_a for mu in range(dim)])
        ddx = np.array([a_op_list[mu].T @ dx[mu] for mu in range(dim)])
        return xa, xaa, dxa, dx, ddx

    @staticmethod
    def _alpha_helper(arg_exp_a_dag, arg_exp_a, rhoprime, deltarho):
        """Build the prefactor that arises due to squeezing. With no squeezing, alpha=1 (number, not matrix)"""
        yrhop = np.matmul(arg_exp_a, rhoprime)
        alpha = np.exp(-0.5 * arg_exp_a @ yrhop - 0.5 * (arg_exp_a_dag - yrhop) @ deltarho @ (arg_exp_a_dag - yrhop))
        return alpha

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
        minima_pair_kinetic_function = partial(self._minima_pair_kinetic_function, EC_mat, a_operator_list)
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
        minima_pair_potential_function = partial(self._minima_pair_potential_function, a_operator_list, Xi)
        local_potential_function = partial(self._local_potential_squeezing_function, Xi, Xi_inv)
        return self._periodic_continuation_squeezing(minima_pair_potential_function, local_potential_function)

    def transfer_matrix(self):
        return self.kinetic_matrix() + self.potential_matrix()

    def _local_potential_squeezing_function(self, Xi, Xi_inv, phi_neighbor, minima_m, minima_p,
                                            disentangled_squeezing_matrices, helper_squeezing_matrices,
                                            exp_adag_a, minima_pair_results):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phibar_kpm = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_list, exp_i_sum = minima_pair_results
        exp_i_phi_list = np.array([exp_i_list[i] * np.exp(1j * phibar_kpm[i]) for i in range(dim)])
        exp_i_phi_sum_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                            * np.prod([np.exp(1j * self.boundary_coefficients[i] * phibar_kpm[i]) for i in range(dim)]))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      helper_squeezing_matrices,
                                                                                      exp_i_phi_list)
                                   for j in range(dim)], axis=0)
        potential_matrix += self._local_contribution_boundary_squeezing(delta_phi, Xi, Xi_inv,
                                                                        disentangled_squeezing_matrices,
                                                                        helper_squeezing_matrices,
                                                                        exp_i_phi_sum_op)
        potential_matrix += (self._local_contribution_identity(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                               disentangled_squeezing_matrices,
                                                               helper_squeezing_matrices, exp_adag_a,
                                                               minima_pair_results)
                             * np.sum(self.EJlist))
        return potential_matrix

    def _local_contribution_boundary_squeezing(self, delta_phi, Xi, Xi_inv,
                                               disentangled_squeezing_matrices,
                                               helper_squeezing_matrices, exp_i_sum):
        dim = self.number_degrees_freedom
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + np.sum([1j * Xi[i, :] * self.boundary_coefficients[i]
                                                     for i in range(dim)], axis=0)) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rhoprime, deltarho)
        alpha_conjugate = self._alpha_helper(arg_exp_a_dag.conjugate(), -arg_exp_a_dag, rhoprime, deltarho)
        potential_matrix = -0.5 * self.EJlist[-1] * alpha * exp_i_sum
        potential_matrix += -0.5 * self.EJlist[-1] * alpha_conjugate * exp_i_sum.conjugate()
        potential_matrix *= self._exp_prod_boundary_coeff(Xi)
        return potential_matrix

    def _local_contribution_single_junction_squeezing(self, j, delta_phi, Xi, Xi_inv, disentangled_squeezing_matrices,
                                                      helper_squeezing_matrices, exp_i_phi_list):
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        delta_phi_rotated = delta_phi @ Xi_inv.T
        arg_exp_a_dag = (delta_phi_rotated + 1j * Xi[j, :]) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rhoprime, deltarho)
        alpha_conjugate = self._alpha_helper(arg_exp_a_dag.conjugate(), -arg_exp_a_dag, rhoprime, deltarho)
        potential_matrix = -0.5 * self.EJlist[j] * alpha * exp_i_phi_list[j]
        # No need to .T the h.c. term, all that is necessary is conjugation
        potential_matrix += -0.5 * self.EJlist[j] * alpha_conjugate * exp_i_phi_list[j].conjugate()
        potential_matrix *= np.exp(-.25 * np.dot(Xi[j, :], Xi.T[:, j]))
        return potential_matrix

    def _minima_pair_potential_function(self, a_operator_list, Xi, exp_adag_a,
                                        disentangled_squeezing_matrices, helper_squeezing_matrices):
        return self._build_potential_operators(a_operator_list, Xi, exp_adag_a,
                                               disentangled_squeezing_matrices, helper_squeezing_matrices)

    def _local_contribution_identity(self, Xi_inv, phi_neighbor, minima_m, minima_p, disentangled_squeezing_matrices,
                                     helper_squeezing_matrices, exp_adag_a, minima_pair_results):
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rhoprime, sigma, sigmaprime, tau, tauprime = disentangled_squeezing_matrices
        deltarho, deltarhoprime, deltarhobar, zp, zpp = helper_squeezing_matrices
        arg_exp_a_dag = np.matmul(delta_phi, Xi_inv.T) / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rhoprime, deltarho)
        return alpha * exp_adag_a

    def inner_product_matrix(self):
        Xi_inv = inv(self.Xi_matrix())
        local_identity_function = partial(self._local_contribution_identity, Xi_inv)
        return self._periodic_continuation_squeezing(lambda x, y, z: None, local_identity_function)

    def _exp_product_coefficient_squeezing(self, delta_phi, Xi_inv, sigma, sigmaprime):
        """
        Overall multiplicative factor. Includes offset charge,
        Gaussian suppression factor in the absence of squeezing. With squeezing,
        also includes exponential of trace over sigma and sigmaprime, see Qin et. al
        """
        dpkX = np.matmul(Xi_inv, delta_phi)
        nglist = self.nglist
        return (np.exp(-1j * np.dot(nglist, delta_phi))
                * np.exp(-0.5 * np.trace(sigma) - 0.5 * np.trace(sigmaprime))
                * np.exp(-0.25 * np.dot(dpkX, dpkX)))

    def _exp_prod_boundary_coeff(self, Xi):
        dim = self.number_degrees_freedom
        return np.exp(-0.25 * np.sum([self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                      * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                      for j in range(dim) for k in range(dim)]))
