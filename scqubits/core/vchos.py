import itertools
import warnings
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.linalg import LinAlgError, expm, inv, eigh
from scipy.optimize import minimize
import scipy.constants as const
from numpy.linalg import matrix_power

from scqubits.core import discretization, storage
import scqubits.core.constants as constants
from scqubits.core.operators import annihilation, operator_in_full_Hilbert_space
import scqubits.utils.plotting as plot
from scqubits.utils.spectrum_utils import order_eigensystem, solve_generalized_eigenvalue_problem_with_QZ, \
    standardize_phases


# The VCHOS method (tight binding) allowing for the diagonalization of systems
# with purely periodic potentials. This module assumes that the potential is
# of the form -EJ[1]*cos(phi_1)-EJ[2]*cos(phi_2)-...-EJ[N]*cos(bc[1]*phi_1+bc[2]*phi_2+...-2\pi f).
# For the flux qubit, the last term looks like -alpha*EJ*cos(phi_1-phi_2-2\pi f), whereas for 
# the current mirror it is -EJ[N]*cos(\sum_i(phi_i)-2\pi f). The user must define a new qubit class
# that inherits VCHOS, with all of the qubit specific information. This includes a method for finding 
# minima, the definition of the capacitance matrix, the number of degrees of freedom, etc.

# Specifically, the user must provide in their parent class the functions 
# build_capacitance_matrix(), build_EC_matrix(), hilbertdim(), find_minima(),
# which define the capacitance matrix, the charging energy matrix, the dimension
# of the hilbert space according to the specific truncation scheme used, and 
# a method to find and find all inequivalent minima, respectively.


class VCHOS(ABC):
    def __init__(self, EJlist, nglist, flux, kmax, number_degrees_freedom=0,
                 number_periodic_degrees_freedom=0, num_exc=None, optimized_lengths=None, nearest_neighbors=None):
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        self.Z0 = 1. / (2 * self.e)**2
        self.Phi0 = 1. / (2 * self.e)
        self.nearest_neighbor_cutoff = 1e-15
        self.EJlist = EJlist
        self.nglist = nglist
        self.flux = flux
        self.kmax = kmax
        if optimized_lengths is not None:
            self.optimized_lengths = optimized_lengths
        else:
            self.optimized_lengths = np.ones(number_degrees_freedom)
        self.number_degrees_freedom = number_degrees_freedom
        self.number_periodic_degrees_freedom = number_periodic_degrees_freedom
        self.number_extended_degrees_freedom = number_degrees_freedom - number_periodic_degrees_freedom
        self.num_exc = num_exc
        self.nearest_neighbors = nearest_neighbors
        self.periodic_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 100)
        self.extended_grid = discretization.Grid1d(-6 * np.pi, 6 * np.pi, 200)
        # This must be set in the individual qubit class and
        # specifies the structure of the boundary term
        self.boundary_coefficients = np.array([])

    def build_gamma_matrix(self, i):
        """Return linearized potential matrix

        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.

        We are imagining an arbitrary loop of JJs where we have
        changed variables to the difference variables, so that
        each junction is a function of just one variable, except for
        the last junction, which is a function of all of the variables

        Parameters
        ----------
        i: int
            integer specifying which minimum to linearize around, 0<=i<= total number of minima

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom
        gamma_matrix = np.zeros((dim, dim))
        min_loc = self.sorted_minima()[i]
        gamma_list = self.EJlist / self.Phi0 ** 2

        gamma_diag = np.diag(np.array([gamma_list[j] * np.cos(min_loc[j]) for j in range(dim)]))
        gamma_matrix = gamma_matrix + gamma_diag

        min_loc_bound_sum = np.sum(np.array([self.boundary_coefficients[j] * min_loc[j] for j in range(dim)]))
        for j in range(dim):
            for k in range(dim):
                gamma_matrix[j, k] += (gamma_list[-1] * self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                       * np.cos(min_loc_bound_sum + 2*np.pi*self.flux))
        return gamma_matrix

    def eigensystem_normal_modes(self, i):
        """Return squared normal mode frequencies, matrix of eigenvectors

        Parameters
        ----------
        i: int
            integer specifying which minimum to linearize around, 0<=i<= total number of minima

        Returns
        -------
        ndarray, ndarray
        """
        C_matrix = self.build_capacitance_matrix()
        g_matrix = self.build_gamma_matrix(i)

        omega_squared, normal_mode_eigenvectors = eigh(g_matrix, b=C_matrix)
        return omega_squared, normal_mode_eigenvectors

    def omega_matrix(self, i):
        """Return a diagonal matrix of the normal mode frequencies of a given minimum

        Parameters
        ----------
        i: int
            integer specifying which minimum to linearize around, 0<=i<= total number of minima

        Returns
        -------
        ndarray
        """
        omega_squared, _ = self.eigensystem_normal_modes(i)
        return np.sqrt(omega_squared)

    def compare_harmonic_lengths_with_minima_separations(self):
        """
        Returns
        -------
        ndarray
            ratio of harmonic lengths to minima separations
        """
        return self._wrapper_for_functions_comparing_minima(self._find_closest_periodic_minimum)

    def _wrapper_for_functions_comparing_minima(self, function):
        sorted_minima = self.sorted_minima()
        if not self.nearest_neighbors:
            self.find_relevant_periodic_continuation_vectors()
        nearest_neighbors = self.nearest_neighbors
        all_minima_pairs = list(itertools.combinations_with_replacement(sorted_minima, 2))
        return np.array([function(minima_pair, nearest_neighbors[i])
                         for i, minima_pair in enumerate(all_minima_pairs)])

    def _find_closest_periodic_minimum(self, minima_pair, nearest_neighbors):
        Xi_inv = inv(self.Xi_matrix())
        delta_inv = Xi_inv.T @ Xi_inv
        if np.allclose(minima_pair[1], minima_pair[0]):  # Do not include equivalent minima in the same unit cell
            nearest_neighbors = np.array([vec for vec in nearest_neighbors if not np.allclose(vec, np.zeros_like(vec))])
        minima_distances = np.array([np.linalg.norm(2.0*np.pi*vec + (minima_pair[1] - minima_pair[0])) / 2.0
                                     for vec in nearest_neighbors])
        minima_vectors = np.array([2.0 * np.pi * vec + (minima_pair[1] - minima_pair[0])
                                   for i, vec in enumerate(nearest_neighbors)])
        minima_unit_vectors = np.array([minima_vectors[i] / minima_distances[i] for i in range(len(minima_distances))])
        harmonic_lengths = np.array([4.0*(unit_vec @ delta_inv @ unit_vec)**(-1/2) for unit_vec in minima_unit_vectors])
        return np.max(harmonic_lengths / minima_distances)

    def _generate_vectors_for_harmonic_approx(self, trial_value):
        dim = self.number_degrees_freedom
        P_0_vec = np.ones(self.number_degrees_freedom)
        P_i_vecs = trial_value*np.identity(self.number_degrees_freedom) + np.ones((dim, dim))
        P_ij_vecs = np.array([(row_i + P_i_vecs[j]) / 2.0 for i, row_i in enumerate(P_i_vecs)
                              for j in range(i + 1, len(P_i_vecs))])
        P_0i_vecs = np.array([(row_i + P_0_vec) / 2.0 for row_i in P_i_vecs])
        return P_0_vec, P_i_vecs, P_ij_vecs, P_0i_vecs

    def optimize_Xi_variational(self):
        """
        We would like to optimize the harmonic length of each column of the Xi
        matrix such that the ground state energy is minimized. Here we use
        the BFGS minimization algorithm as implemented in scipy which performs
        well, but which generally requires more function evaluations than the harmonic
        approximation algorithm, with similar results.
        """
        self.optimized_lengths = np.ones(self.number_degrees_freedom)
        self.default_Xi = self.Xi_matrix()
        default_lengths = self.optimized_lengths
        global_minimum = self.sorted_minima()[0]
        evals_function = partial(self._evals_calc_variational, global_minimum)
        optimized_lengths = minimize(evals_function, default_lengths, tol=1e-1)
        assert optimized_lengths.success
        self.optimized_lengths = optimized_lengths.x

    def _update_Xi(self):
        return np.array([row * self.optimized_lengths[i] for i, row in enumerate(self.default_Xi.T)]).T

    def _evals_calc_variational(self, global_minimum, optimized_lengths):
        self.optimized_lengths = optimized_lengths
        Xi = self._update_Xi()
        Xi_inv = inv(Xi)
        exp_i_phi_j = self._one_state_exp_i_phi_j_operators(Xi)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        transfer, inner = self._helper_function_for_Xi_optimization(Xi_inv, global_minimum, EC_mat_t, exp_i_phi_j)
        return np.real(transfer / inner)

    def Xi_matrix(self):
        """
        Returns
        -------
        ndarray
            Xi matrix of the normal mode eigenvectors normalized
            to encode the harmonic length
        """
        omega_squared, normal_mode_eigenvectors = self.eigensystem_normal_modes(0)
        # We introduce a normalization such that \Xi^T C \Xi = \Omega^{-1}/Z0
        Xi_matrix = np.array([normal_mode_eigenvectors[:, i] * self.optimized_lengths[i] * omega**(-1/4)
                              * np.sqrt(1./self.Z0) for i, omega in enumerate(omega_squared)]).T
        return Xi_matrix

    def a_operator(self, mu):
        """Return the lowering operator associated with the mu^th d.o.f. in the full Hilbert space

        Parameters
        ----------
        mu: int
            which degree of freedom, 0<=mu<=self.number_degrees_freedom

        Returns
        -------
        ndarray
        """
        identity_operator = np.eye(self.num_exc + 1, dtype=np.complex_)
        identity_operator_list = np.array([identity_operator for _ in range(self.number_degrees_freedom)])
        return operator_in_full_Hilbert_space(np.array([annihilation(self.num_exc + 1, dtype=np.complex_)]),
                                              np.array([mu]), identity_operator_list, sparse=False)

    def a_operator_list(self):
        return np.array([self.a_operator(i) for i in range(self.number_degrees_freedom)])

    def _gen_periodic_continuation_vectors_in_hypersphere(self):
        sites = self.number_periodic_degrees_freedom
        vec_list = [np.zeros(sites, dtype=int)]
        for radius in range(1, self.kmax+1):
            prev_vec = np.zeros(sites, dtype=int)
            prev_vec[0] = radius
            vec_list.append(prev_vec)
            while prev_vec[-1] != radius:
                k = self._find_k(prev_vec)
                next_vec = np.zeros(sites)
                next_vec[0:k] = prev_vec[0:k]
                next_vec[k] = prev_vec[k]-1
                next_vec[k+1] = radius-np.sum([next_vec[i] for i in range(k+1)])
                vec_list.append(next_vec)
                self._append_reflected_vectors(next_vec, vec_list)
                prev_vec = next_vec
        return np.array(vec_list)

    @staticmethod
    def _append_reflected_vectors(vec, vec_list):
        nonzero_indices = np.nonzero(vec)
        nonzero_vec = vec[nonzero_indices]
        multiplicative_factors = itertools.product(np.array([1, -1]), repeat=len(nonzero_vec))
        for mult_factor in multiplicative_factors:
            vec_copy = np.copy(vec)
            if not np.allclose(mult_factor, np.ones_like(mult_factor)):
                np.put(vec_copy, nonzero_indices, np.multiply(nonzero_vec, mult_factor))
                vec_list.append(vec_copy)

    @staticmethod
    def _find_k(vec):
        dim = len(vec)
        for num in range(dim-2, -1, -1):
            if vec[num] != 0:
                return num

    def identity(self):
        """
        Returns
        -------
        ndarray
            Returns the identity matrix whose dimensions are the same as self.a_operator(mu)
        """
        return np.eye(int(self.number_states_per_minimum()))

    def number_states_per_minimum(self):
        """
        Returns
        -------
        int
            Returns the number of states displaced into each local minimum
        """
        return (self.num_exc + 1)**self.number_degrees_freedom

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension.
        """
        return int(len(self.sorted_minima()) * self.number_states_per_minimum())

    def find_relevant_periodic_continuation_vectors(self):
        """
        We have found that specifically this part of the code is quite slow, that
        is finding the relevant nearest neighbor, next nearest neighbor, etc. lattice vectors
        that meaningfully contribute. This is a calculation that previously had to be done
        for the kinetic, potential and inner product matrices separately, even though
        the results were the same for all three matrices. This helper function allows us to only
        do it once.
        """
        Xi_inv = inv(self.Xi_matrix())
        minima_list = self.sorted_minima()
        nearest_neighbors = []
        nearest_neighbors_single_minimum = []
        dim_extended = self.number_extended_degrees_freedom
        all_neighbors = self._gen_periodic_continuation_vectors_in_hypersphere()
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                filtered_neighbors = itertools.filterfalse(lambda e: self._filter_neighbors(e, minima_diff, Xi_inv),
                                                           all_neighbors)
                for neighbor in filtered_neighbors:
                    nearest_neighbors_single_minimum.append(np.concatenate((np.zeros(dim_extended, dtype=int),
                                                                            neighbor)))
                nearest_neighbors.append(nearest_neighbors_single_minimum)
                nearest_neighbors_single_minimum = []
        self.nearest_neighbors = nearest_neighbors

    def _filter_neighbors(self, neighbor, minima_diff, Xi_inv):
        """
        Want to eliminate periodic continuation terms that are irrelevant, i.e.,
        they add nothing to the transfer matrix. These can be identified as each term
        is suppressed by a gaussian exponential factor. If the argument np.dot(dpkX, dpkX)
        of the exponential is greater than 180.0, this results in a suppression of ~10**(-20),
        and so can be safely neglected.

        Assumption is that extended degrees of freedom precede the periodic d.o.f.
        """
        phi_neighbor = 2.0 * np.pi * np.concatenate((np.zeros(self.number_extended_degrees_freedom), neighbor))
        dpkX = Xi_inv @ (phi_neighbor + minima_diff)
        prod = np.dot(dpkX, dpkX)
        return np.exp(-prod) < self.nearest_neighbor_cutoff

    def _build_premultiplied_a_and_a_dagger(self, a_operator_list):
        dim = self.number_degrees_freedom
        a_a = np.array([a_operator_list[i] @ a_operator_list[i] for i in range(dim)])
        a_dagger_a = np.array([a_operator_list[i].T @ a_operator_list[i] for i in range(dim)])
        return a_operator_list, a_a, a_dagger_a

    def _build_single_exp_i_phi_j_operator(self, j, Xi, a_operator_list):
        dim = self.number_degrees_freedom
        if j == dim:
            exp_i_phi_j_a_component = expm(np.sum(np.array([self.boundary_coefficients[i]
                                                  * 1j * Xi[i, k] * a_operator_list[k] / np.sqrt(2.0)
                                                  for i in range(dim) for k in range(dim)]), axis=0))
            BCH_factor = self._BCH_factor_for_potential_boundary(Xi)
        else:
            exp_i_phi_j_a_component = expm(np.sum(np.array([1j * Xi[j, k] * a_operator_list[k] / np.sqrt(2.0)
                                                  for k in range(dim)]), axis=0))
            BCH_factor = np.exp(-0.25*np.dot(Xi[j, :], Xi.T[:, j]))
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a_component.T
        return BCH_factor * exp_i_phi_j_a_dagger_component @ exp_i_phi_j_a_component

    def _one_state_exp_i_phi_j_operators(self, Xi):
        dim = self.number_degrees_freedom
        exp_factors = np.array([np.exp(-0.25*np.dot(Xi[j, :], Xi.T[:, j])) for j in range(dim)])
        return np.append(exp_factors, self._BCH_factor_for_potential_boundary(Xi))

    def _build_all_exp_i_phi_j_operators(self, Xi, a_operator_list):
        return np.array([self._build_single_exp_i_phi_j_operator(j, Xi, a_operator_list)
                         for j in range(self.number_degrees_freedom + 1)])

    def _build_general_translation_operators(self, Xi_inv, a_operator_list):
        dim = self.number_degrees_freedom
        exp_a_list = np.array([expm(np.sum(np.array([2.0 * np.pi * Xi_inv.T[i, j] * a_operator_list[j] / np.sqrt(2.0)
                                           for j in range(dim)]), axis=0)) for i in range(dim)])
        return exp_a_list

    def _build_minima_dependent_translation_operators(self, minima_diff, Xi_inv, a_operator_list):
        """In general this is a costly part of the code (expm is quite slow)"""
        dim = self.number_degrees_freedom
        exp_a_minima_difference = expm(np.sum(np.array([minima_diff[i] * Xi_inv.T[i, j]
                                                        * a_operator_list[j] / np.sqrt(2.0)
                                              for i in range(dim) for j in range(dim)]), axis=0))
        return exp_a_minima_difference

    def _translation_operator_builder(self, exp_a_list, exp_minima_difference, neighbor):
        """Build translation operators using matrix_power rather than the more costly expm"""
        dim = self.number_degrees_freedom
        translation_op_a_dagger = self.identity()
        translation_op_a = self.identity()
        for j in range(dim):
            translation_op_for_direction = matrix_power(exp_a_list[j].T, int(neighbor[j]))
            translation_op_a_dagger = translation_op_a_dagger @ translation_op_for_direction
            translation_op_a = translation_op_a @ inv(translation_op_for_direction.T)
        translation_op_a_dagger = exp_minima_difference.T @ translation_op_a_dagger
        translation_op_a = translation_op_a @ inv(exp_minima_difference)
        return translation_op_a_dagger, translation_op_a

    def _exp_product_coefficient(self, delta_phi, Xi_inv):
        """Overall multiplicative factor, including offset charge, Gaussian suppression BCH factor
        from the periodic continuation operators"""
        delta_phi_rotated = Xi_inv @ delta_phi
        return np.exp(-1j * self.nglist @ delta_phi) * np.exp(-0.25 * delta_phi_rotated @ delta_phi_rotated)

    def _BCH_factor_for_potential_boundary(self, Xi):
        """BCH factor obtained from the last potential operator"""
        dim = self.number_degrees_freedom
        return np.exp(-0.25*np.sum(np.array([self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                   * np.dot(Xi[j, :], Xi.T[:, k]) for j in range(dim) for k in range(dim)])))

    def hamiltonian(self):
        pass

    def kinetic_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        Xi_inv = inv(self.Xi_matrix())
        a_operator_list = self.a_operator_list()
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger(a_operator_list)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        kinetic_function = partial(self._local_kinetic_contribution_to_transfer_matrix,
                                   premultiplied_a_and_a_dagger, EC_mat_t, Xi_inv)
        return self._periodic_continuation(kinetic_function)

    def potential_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        Xi = self.Xi_matrix()
        a_operator_list = self.a_operator_list()
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators(Xi, a_operator_list)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger(a_operator_list)
        potential_function = partial(self._local_potential_contribution_to_transfer_matrix, exp_i_phi_list,
                                     premultiplied_a_and_a_dagger, Xi)
        return self._periodic_continuation(potential_function)

    def transfer_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the transfer matrix
        """
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_list = self.a_operator_list()
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators(Xi, a_operator_list)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger(a_operator_list)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        transfer_matrix_function = partial(self._local_contribution_to_transfer_matrix, exp_i_phi_list,
                                           premultiplied_a_and_a_dagger, EC_mat_t, Xi, Xi_inv)
        return self._periodic_continuation(transfer_matrix_function)

    def inner_product_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the inner product matrix
        """
        return self._periodic_continuation(self._inner_product_function)

    def transfer_matrix_and_inner_product(self):
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_list = self.a_operator_list()
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators(Xi, a_operator_list)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger(a_operator_list)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        transfer_matrix_function = partial(self._local_contribution_to_transfer_matrix, exp_i_phi_list,
                                           premultiplied_a_and_a_dagger, EC_mat_t, Xi, Xi_inv)
        exp_a_list = self._build_general_translation_operators(Xi_inv, a_operator_list)
        transfer_matrix = self._periodic_continuation(transfer_matrix_function, exp_a_list=exp_a_list)
        inner_product_matrix = self._periodic_continuation(self._inner_product_function, exp_a_list=exp_a_list)
        return transfer_matrix, inner_product_matrix

    def one_state_transfer_and_inner(self):
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        exp_i_phi_list = self._one_state_exp_i_phi_j_operators(Xi)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        global_minimum = self.sorted_minima()[0]
        if not self.nearest_neighbors:
            self.find_relevant_periodic_continuation_vectors()
        nearest_neighbors = self.nearest_neighbors[0]
        transfer_function = partial(self._one_state_local_transfer, exp_i_phi_list, EC_mat_t, Xi_inv)
        transfer_matrix = self._one_state_periodic_continuation(global_minimum, nearest_neighbors,
                                                                transfer_function, Xi_inv)
        inner_product_matrix = self._one_state_periodic_continuation(global_minimum, nearest_neighbors,
                                                                     self._inner_product_function, Xi_inv)
        return transfer_matrix, inner_product_matrix

    def _helper_function_for_Xi_optimization(self, Xi_inv, global_minimum, EC_mat_t, exp_i_phi_j):
        if not self.nearest_neighbors:
            self.find_relevant_periodic_continuation_vectors()
        transfer_function = partial(self._one_state_local_transfer, exp_i_phi_j, EC_mat_t, Xi_inv)
        transfer = self._one_state_periodic_continuation(global_minimum, self.nearest_neighbors[0],
                                                         transfer_function, Xi_inv)
        inner_product = self._one_state_periodic_continuation(global_minimum, self.nearest_neighbors[0],
                                                              lambda x, y, z: 1.0+0j, Xi_inv)
        return transfer, inner_product

    def _local_kinetic_contribution_to_transfer_matrix(self, premultiplied_a_and_a_dagger, EC_mat_t, Xi_inv,
                                                       phi_neighbor, minima_m, minima_p):
        """Calculating products of a, a_dagger operators is costly,
        as well as repeatedly calculating Xi (or Xi_inv) which is why they are
        passed to this function in this way rather than calculated below"""
        a, a_a, a_dagger_a = premultiplied_a_and_a_dagger
        minima_diff = minima_p - minima_m
        delta_phi = phi_neighbor + minima_diff
        delta_phi_rotated = Xi_inv @ delta_phi
        kinetic_matrix = np.sum(np.array([EC_mat_t[i, i]*(-0.5*4*a_a[i] - 0.5*4*a_a[i].T + 0.5*8*a_dagger_a[i]
                                                          - 4*(a[i] - a[i].T)*delta_phi_rotated[i]/np.sqrt(2.0))
                                          for i in range(self.number_degrees_freedom)]), axis=0)
        identity_coefficient = 0.5 * 4 * np.trace(EC_mat_t)
        identity_coefficient = identity_coefficient - 0.25*4*delta_phi_rotated @ EC_mat_t @ delta_phi_rotated
        kinetic_matrix = kinetic_matrix + identity_coefficient*self.identity()
        return kinetic_matrix

    @staticmethod
    def _one_state_local_kinetic(EC_mat_t, Xi_inv, phi_neighbor, minima_m, minima_p):
        minima_diff = minima_p - minima_m
        delta_phi = phi_neighbor + minima_diff
        delta_phi_rotated = Xi_inv @ delta_phi
        identity_coefficient = 0.5 * 4 * np.trace(EC_mat_t)
        return identity_coefficient - 0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated

    def _one_state_local_potential(self, exp_i_phi_j, phi_neighbor, minima_m, minima_p):
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_list_without_boundary = np.array([exp_i_phi_j[i] * np.exp(1j * phi_bar[i]) for i in range(dim)])
        exp_i_sum_phi = (exp_i_phi_j[-1] * np.exp(1j * 2.0 * np.pi * self.flux)
                         * np.prod([np.exp(1j * self.boundary_coefficients[i] * phi_bar[i])
                                    for i in range(dim)]))
        potential = np.sum([-0.5 * self.EJlist[junction] * (exp_i_phi_list_without_boundary[junction]
                                                            + exp_i_phi_list_without_boundary[junction].conjugate())
                            for junction in range(dim)])
        potential = potential - 0.5 * self.EJlist[-1] * (exp_i_sum_phi + exp_i_sum_phi.conjugate())
        potential = potential + np.sum(self.EJlist)
        return potential

    def _one_state_local_transfer(self, exp_i_phi_j, EC_mat_t, Xi_inv, phi_neighbor, minima_m, minima_p):
        return (self._one_state_local_kinetic(EC_mat_t, Xi_inv, phi_neighbor, minima_m, minima_p)
                + self._one_state_local_potential(exp_i_phi_j, phi_neighbor, minima_m, minima_p))

    def _local_potential_contribution_to_transfer_matrix(self, exp_i_phi_list, premultiplied_a_and_a_dagger, Xi,
                                                         phi_neighbor, minima_m, minima_p):
        """Calculating exp_i_phi operators is costly, which is why it is
        passed to this function in this way rather than calculated below"""
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_list_without_boundary = np.array([exp_i_phi_list[i] * np.exp(1j * phi_bar[i])
                                                    for i in range(dim)])
        exp_i_sum_phi = (exp_i_phi_list[-1] * np.exp(1j * 2.0 * np.pi * self.flux)
                         * np.prod(np.array([np.exp(1j * self.boundary_coefficients[i] * phi_bar[i])
                                             for i in range(dim)])))
        potential_matrix = np.sum(np.array([-0.5*self.EJlist[junction]
                                  * (exp_i_phi_list_without_boundary[junction]
                                      + exp_i_phi_list_without_boundary[junction].conjugate())
                                   for junction in range(dim)]), axis=0)
        potential_matrix = potential_matrix - 0.5*self.EJlist[-1]*(exp_i_sum_phi + exp_i_sum_phi.conjugate())
        potential_matrix = potential_matrix + np.sum(self.EJlist) * self.identity()
        return potential_matrix

    def _local_contribution_to_transfer_matrix(self, exp_i_phi_list, premultiplied_a_and_a_dagger, EC_mat_t,
                                               Xi, Xi_inv, phi_neighbor, minima_m, minima_p):
        return (self._local_kinetic_contribution_to_transfer_matrix(premultiplied_a_and_a_dagger, EC_mat_t, Xi_inv,
                                                                    phi_neighbor, minima_m, minima_p)
                + self._local_potential_contribution_to_transfer_matrix(exp_i_phi_list, premultiplied_a_and_a_dagger,
                                                                        Xi, phi_neighbor, minima_m, minima_p))

    def _inner_product_function(self, phi_neighbor, minima_m, minima_p):
        """The three arguments need to be passed in order to match the signature of
        operators that are functions of the raising and lowering operators, whose local
        contributions depend on the periodic continuation vector `phi_neighbor` as well
        as the minima where the states in question are located."""
        return self.identity()

    def _periodic_continuation(self, func, exp_a_list=None):
        """This function is the meat of the VCHOS method. Any operator whose matrix
        elements we want (the transfer matrix and inner product matrix are obvious examples)
        can be passed to this function, and the matrix elements of that operator
        will be returned.

        Parameters
        ----------
        func: method
            function that takes three arguments (phi_neighbor, minima_m, minima_p) and returns the
            relevant operator with dimension NxN, where N is the number of states
            displaced into each minimum. For instance to find the inner product matrix,
            we use the function self._inner_product_operator(phi_neighbor, minima_m, minima_p) -> self.identity

        Returns
        -------
        ndarray
        """
        Xi_inv = inv(self.Xi_matrix())
        a_operator_list = self.a_operator_list()
        if exp_a_list is None:
            exp_a_list = self._build_general_translation_operators(Xi_inv, a_operator_list)
        if not self.nearest_neighbors:
            self.find_relevant_periodic_continuation_vectors()
        minima_list = self.sorted_minima()
        hilbertdim = self.hilbertdim()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                matrix_element = self._periodic_continuation_for_minima_pair(minima_m, minima_list[p],
                                                                             self.nearest_neighbors[counter],
                                                                             func, exp_a_list, Xi_inv, a_operator_list)
                operator_matrix[m*num_states_min: (m + 1)*num_states_min,
                                p*num_states_min: (p + 1)*num_states_min] += matrix_element
                counter += 1
        operator_matrix = self._populate_hermitean_matrix(operator_matrix)
        return operator_matrix

    def _periodic_continuation_for_minima_pair(self, minima_m, minima_p, nearest_neighbors,
                                               func, exp_a_list, Xi_inv, a_operator_list):
        minima_diff = minima_p - minima_m
        exp_minima_difference = self._build_minima_dependent_translation_operators(minima_diff, Xi_inv, a_operator_list)
        dim = int(self.number_states_per_minimum())
        matrix_element = np.zeros((dim, dim), dtype=np.complex_)
        for neighbor in nearest_neighbors:
            phi_neighbor = 2.0 * np.pi * np.array(neighbor)
            exp_prod_coefficient = self._exp_product_coefficient(phi_neighbor + minima_diff, Xi_inv)
            exp_a_dagger, exp_a = self._translation_operator_builder(exp_a_list, exp_minima_difference,
                                                                     neighbor)
            neighbor_matrix_element = exp_prod_coefficient * func(phi_neighbor, minima_m, minima_p)
            matrix_element += exp_a_dagger @ neighbor_matrix_element @ exp_a
        return matrix_element

    def _one_state_periodic_continuation(self, global_min, nearest_neighbors, func, Xi_inv):
        matrix_element = 0.0+0j
        for neighbor in nearest_neighbors:
            phi_neighbor = 2.0 * np.pi * np.array(neighbor)
            exp_prod_coefficient = self._exp_product_coefficient(phi_neighbor, Xi_inv)
            neighbor_matrix_element = exp_prod_coefficient * func(phi_neighbor, global_min, global_min)
            matrix_element += neighbor_matrix_element
        return matrix_element

    def _populate_hermitean_matrix(self, mat):
        """Return a fully Hermitean matrix, assuming that the input matrix has been
        populated with the upper right blocks"""
        minima_list = self.sorted_minima()
        num_states_min = int(self.number_states_per_minimum())
        for m, minima_m in enumerate(minima_list):
            for p in range(m + 1, len(minima_list)):
                matrix_element = mat[m*num_states_min: (m + 1)*num_states_min,
                                     p*num_states_min: (p + 1)*num_states_min]
                mat[p*num_states_min: (p + 1)*num_states_min,
                    m*num_states_min: (m + 1)*num_states_min] += matrix_element.conjugate().T
        return mat

    def _evals_calc(self, evals_count):
        transfer_matrix, inner_product_matrix = self.transfer_matrix_and_inner_product()
        try:
            evals = eigh(transfer_matrix, b=inner_product_matrix,
                         eigvals_only=True, eigvals=(0, evals_count - 1))
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            evals = solve_generalized_eigenvalue_problem_with_QZ(transfer_matrix, inner_product_matrix,
                                                                 evals_count, eigvals_only=True)
        return evals

    def _esys_calc(self, evals_count):
        transfer_matrix, inner_product_matrix = self.transfer_matrix_and_inner_product()
        try:
            evals, evecs = eigh(transfer_matrix, b=inner_product_matrix,
                                eigvals_only=False, eigvals=(0, evals_count - 1))
            evals, evecs = order_eigensystem(evals, evecs)
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            evals, evecs = solve_generalized_eigenvalue_problem_with_QZ(transfer_matrix, inner_product_matrix,
                                                                        evals_count, eigvals_only=False)
        return evals, evecs

    def sorted_potential_values_and_minima(self):
        minima_holder = np.array(self.find_minima())
        value_of_potential = np.array([self.potential(minima) for minima in minima_holder])
        sorted_indices = np.argsort(value_of_potential)
        return value_of_potential[sorted_indices], minima_holder[sorted_indices, :]

    def sorted_minima(self):
        """Sort the minima based on the value of the potential at the minima """
        sorted_value_of_potential, sorted_minima_holder = self.sorted_potential_values_and_minima()
        # For efficiency purposes, don't want to displace states into minima
        # that are too high energy. Arbitrarily set a 40 GHz cutoff
        global_min = sorted_value_of_potential[0]
        dim = len(sorted_minima_holder)
        sorted_minima_holder = np.array([sorted_minima_holder[i] for i in range(dim)
                                         if sorted_value_of_potential[i] < global_min + 40.0])
        return sorted_minima_holder

    def normalize_minimum_inside_pi_range(self, minimum):
        num_extended = self.number_extended_degrees_freedom
        extended_coordinates = minimum[0:num_extended]
        periodic_coordinates = np.mod(minimum, 2*np.pi*np.ones_like(minimum))[num_extended:]
        periodic_coordinates = np.array([elem - 2*np.pi if elem > np.pi else elem for elem in periodic_coordinates])
        return np.concatenate((extended_coordinates, periodic_coordinates))

    def _check_if_new_minima(self, new_minima, minima_holder):
        """
        Helper function for find_minima, checking if new_minima is
        already represented in minima_holder. If so,
        _check_if_new_minima returns False.
        """
        num_extended = self.number_extended_degrees_freedom
        for minima in minima_holder:
            extended_coordinates = np.array(minima[0:num_extended] - new_minima[0:num_extended])
            periodic_coordinates = np.mod(minima - new_minima, 2*np.pi*np.ones_like(minima))[num_extended:]
            diff_array_bool_extended = [True if np.allclose(elem, 0.0, atol=1e-3) else False
                                        for elem in extended_coordinates]
            diff_array_bool_periodic = [True if (np.allclose(elem, 0.0, atol=1e-3)
                                                 or np.allclose(elem, 2*np.pi, atol=1e-3))
                                        else False for elem in periodic_coordinates]
            if np.all(diff_array_bool_extended) and np.all(diff_array_bool_periodic):
                return False
        return True

    def _filter_repeated_minima(self, minima_holder):
        filtered_minima_holder = [minima_holder[0]]
        for minima in minima_holder:
            if self._check_if_new_minima(minima, filtered_minima_holder):
                filtered_minima_holder.append(minima)
        return filtered_minima_holder

    def villain_minima_finder(self):
        result_holder = []
        for junction in range(len(self.EJlist)):
            for m in range(-5, 5, 1):
                m_list = 0.0*np.ones_like(self.EJlist)
                m_list[junction] = m
                villain_func = partial(self.villain_potential, m_list)
                villain_result = minimize(villain_func, np.array([0.0, 0.0]))
                result = minimize(self.potential, villain_result.x)
                if result.success:
                    result_holder.append(np.mod(result.x, 2.0*np.pi*np.ones_like(result.x)))
        result_holder = self._filter_repeated_minima(result_holder)
        return result_holder

    def wavefunction(self, esys=None, which=0):
        """
        Return a vchos wavefunction, assuming the qubit has 2 degrees of freedom

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self._esys_calc(evals_count)
        else:
            _, evecs = esys
        minima_list = self.sorted_minima()

        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        normalization = np.sqrt(np.abs(np.linalg.det(Xi))) ** (-1)

        dim_extended = self.number_extended_degrees_freedom
        dim_periodic = self.number_periodic_degrees_freedom
        phi_1_grid = self.periodic_grid
        phi_1_vec = phi_1_grid.make_linspace()
        phi_2_grid = self.periodic_grid
        phi_2_vec = phi_2_grid.make_linspace()

        if dim_extended != 0:
            phi_1_grid = self.extended_grid
            phi_1_vec = phi_1_grid.make_linspace()

        wavefunction_amplitudes = np.zeros_like(np.outer(phi_1_vec, phi_2_vec), dtype=np.complex_).T

        for i, minimum in enumerate(minima_list):
            neighbors = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=dim_periodic)
            neighbor = next(neighbors, -1)
            while neighbor != -1:
                phi_neighbor = 2.0 * np.pi * np.concatenate((np.zeros(dim_extended), neighbor))
                phi_offset = phi_neighbor - minimum
                state_amplitudes = self.state_amplitudes_function(i, evecs, which)
                phi_1_with_offset = phi_1_vec + phi_offset[0]
                phi_2_with_offset = phi_2_vec + phi_offset[1]
                normal_mode_1 = np.add.outer(Xi_inv[0, 0]*phi_1_with_offset, Xi_inv[0, 1]*phi_2_with_offset)
                normal_mode_2 = np.add.outer(Xi_inv[1, 0]*phi_1_with_offset, Xi_inv[1, 1]*phi_2_with_offset)
                wavefunction_amplitudes += (self.wavefunction_amplitudes_function(state_amplitudes,
                                                                                  normal_mode_1, normal_mode_2)
                                            * normalization * np.exp(-1j * np.dot(self.nglist, phi_offset)))
                neighbor = next(neighbors, -1)

        grid2d = discretization.GridSpec(np.asarray([[phi_1_grid.min_val, phi_1_grid.max_val, phi_1_grid.pt_count],
                                                     [phi_2_grid.min_val, phi_2_grid.max_val, phi_2_grid.pt_count]]))

        wavefunction_amplitudes = standardize_phases(wavefunction_amplitudes)

        return storage.WaveFunctionOnGrid(grid2d, wavefunction_amplitudes)

    def state_amplitudes_function(self, i, evecs, which):
        total_num_states = self.number_states_per_minimum()
        return np.real(np.reshape(evecs[i * total_num_states: (i + 1) * total_num_states, which],
                                  (self.num_exc + 1, self.num_exc + 1)))

    def wavefunction_amplitudes_function(self, state_amplitudes, normal_mode_1, normal_mode_2):
        return np.sum([plot.multiply_two_harm_osc_functions(s1, s2, normal_mode_1, normal_mode_2)
                       * state_amplitudes[s1, s2] for s2 in range(self.num_exc + 1)
                       for s1 in range(self.num_exc + 1)], axis=0).T

    def plot_wavefunction(self, esys=None, which=0, mode='abs', zero_calibrate=True, **kwargs):
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
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
        wavefunction = self.wavefunction(esys, which=which)
        wavefunction.amplitudes = amplitude_modifier(wavefunction.amplitudes)
        return plot.wavefunction2d(wavefunction, zero_calibrate=zero_calibrate, **kwargs)

    @abstractmethod
    def potential(self, phi_array):
        """returns a float that is the value of the potential at the location specified by phi_array"""

    @abstractmethod
    def villain_potential(self, phi_array, m_list):
        """returns a float that is the value of the linearized (harmonic) potential at the location
        specified by phi_array with villain parameters specified by m_list"""

    @abstractmethod
    def find_minima(self):
        """finds all minima in the potential energy landscape"""

    @abstractmethod
    def build_capacitance_matrix(self):
        """builds the capacitance matrix"""

    @abstractmethod
    def build_EC_matrix(self):
        """builds the charging energy matrix"""
