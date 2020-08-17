import itertools
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
from numpy.linalg import norm
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
                 number_periodic_degrees_freedom=0, num_exc=None):
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        self.Z0 = 1. / (2 * self.e)**2
        self.Phi0 = 1. / (2 * self.e)
        self.nearest_neighbor_cutoff = 180.0
        self.EJlist = EJlist
        self.nglist = nglist
        self.flux = flux
        self.kmax = kmax
        self.number_degrees_freedom = number_degrees_freedom
        self.number_periodic_degrees_freedom = number_periodic_degrees_freedom
        self.number_extended_degrees_freedom = number_degrees_freedom - number_periodic_degrees_freedom
        self.num_exc = num_exc
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

        gamma_diag = np.diag([gamma_list[j] * np.cos(min_loc[j]) for j in range(dim)])
        gamma_matrix += gamma_diag

        min_loc_bound_sum = np.sum([self.boundary_coefficients[j] * min_loc[j] for j in range(dim)])
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

    def compare_harmonic_lengths_with_minima_separations(self, optimized_lengths=None):
        """
        Returns
        -------
        ndarray
            ratio of harmonic lengths to minima separations
        """
        Xi = self.Xi_matrix(optimized_lengths)
        return self._wrapper_for_functions_comparing_minima(self._find_closest_periodic_minimum, Xi)

    def _wrapper_for_functions_comparing_minima(self, function, Xi):
        sorted_minima = self.sorted_minima()
        periodic_vectors = self._find_relevant_periodic_continuation_vectors(Xi)
        all_minima_pairs = list(itertools.combinations_with_replacement(sorted_minima, 2))
        return np.array([function(minima_pair, periodic_vectors[i], Xi)
                         for i, minima_pair in enumerate(all_minima_pairs)])

    def _find_closest_periodic_minimum(self, minima_pair, periodic_vectors, Xi):
        Xi_inv = inv(Xi)
        delta_inv = Xi_inv.T @ Xi_inv
        if np.allclose(minima_pair[1], minima_pair[0]):  # Do not include equivalent minima in the same unit cell
            periodic_vectors = np.array([vec for vec in periodic_vectors if not np.allclose(vec, np.zeros_like(vec))])
        minima_distances = np.array([np.linalg.norm(2.0*np.pi*vec + (minima_pair[1] - minima_pair[0])) / 2.0
                                     for vec in periodic_vectors])
        minima_vectors = np.array([2.0 * np.pi * vec + (minima_pair[1] - minima_pair[0])
                                   for i, vec in enumerate(periodic_vectors)])
        minima_unit_vectors = np.array([minima_vectors[i] / minima_distances[i] for i in range(len(minima_distances))])
        harmonic_lengths = np.array([4.0*(unit_vec @ delta_inv @ unit_vec)**(-1/2) for unit_vec in minima_unit_vectors])
        return np.max(harmonic_lengths / minima_distances)

    def Xi_matrix(self, optimized_lengths=None):
        """
        Returns
        -------
        ndarray
            Xi matrix of the normal mode eigenvectors normalized
            to encode the harmonic length
        """
        if optimized_lengths is None:
            optimized_lengths = np.ones(self.number_degrees_freedom)
        omega_squared, normal_mode_eigenvectors = self.eigensystem_normal_modes(0)
        # We introduce a normalization such that \Xi^T C \Xi = \Omega^{-1}/Z0
        Xi_matrix = np.array([normal_mode_eigenvectors[:, i] * optimized_lengths[i] * omega**(-1/4)
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
        identity_operator_list = [identity_operator for _ in range(self.number_degrees_freedom)]
        return operator_in_full_Hilbert_space([annihilation(self.num_exc + 1, dtype=np.complex_)],
                                              [mu], identity_operator_list, sparse=False)

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

    def _find_relevant_periodic_continuation_vectors(self, Xi):
        """
        We have found that specifically this part of the code is quite slow, that
        is finding the relevant nearest neighbor, next nearest neighbor, etc. lattice vectors
        that meaningfully contribute. This is a calculation that previously had to be done
        for the kinetic, potential and inner product matrices separately, even though
        the results were the same for all three matrices. This helper function allows us to only
        do it once.
        """
        Xi_inv = inv(Xi)
        minima_list = self.sorted_minima()
        nearest_neighbors = []
        nearest_neighbors_single_minimum = []
        dim_extended = self.number_extended_degrees_freedom
        dim_periodic = self.number_periodic_degrees_freedom
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                all_neighbors = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=dim_periodic)
                filtered_neighbors = itertools.filterfalse(lambda e: self._filter_neighbors(e, minima_diff, Xi_inv),
                                                           all_neighbors)
                neighbor = next(filtered_neighbors, -1)
                while neighbor != -1:
                    nearest_neighbors_single_minimum.append(np.concatenate((np.zeros(dim_extended, dtype=int),
                                                                            neighbor)))
                    neighbor = next(filtered_neighbors, -1)
                nearest_neighbors.append(nearest_neighbors_single_minimum)
                nearest_neighbors_single_minimum = []
        return nearest_neighbors

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
        return prod > self.nearest_neighbor_cutoff

    def _build_premultiplied_a_and_a_dagger(self):
        dim = self.number_degrees_freedom
        a = np.array([self.a_operator(i) for i in range(dim)])
        a_a = np.array([self.a_operator(i) @ self.a_operator(i) for i in range(dim)])
        a_dagger_a = np.array([self.a_operator(i).T @ self.a_operator(i) for i in range(dim)])
        return a, a_a, a_dagger_a

    def _build_single_exp_i_phi_j_operator(self, j, Xi):
        dim = self.number_degrees_freedom
        if j == dim:
            exp_i_phi_j_a_component = expm(np.sum([self.boundary_coefficients[i]
                                                   * 1j * Xi[i, k] * self.a_operator(k) / np.sqrt(2.0)
                                                   for i in range(dim) for k in range(dim)], axis=0))
            BCH_factor = self._BCH_factor_for_potential_boundary(Xi)
        else:
            exp_i_phi_j_a_component = expm(np.sum([1j * Xi[j, k] * self.a_operator(k) / np.sqrt(2.0)
                                                   for k in range(dim)], axis=0))
            BCH_factor = np.exp(-0.25*np.dot(Xi[j, :], Xi.T[:, j]))
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a_component.T
        return BCH_factor * exp_i_phi_j_a_dagger_component @ exp_i_phi_j_a_component

    def _build_all_exp_i_phi_j_operators(self, Xi):
        return np.array([self._build_single_exp_i_phi_j_operator(j, Xi)
                         for j in range(self.number_degrees_freedom + 1)])

    def _build_general_exponentiated_translation_operators(self, Xi_inv):
        dim = self.number_degrees_freedom
        exp_a_list = np.array([expm(np.sum([2.0 * np.pi * Xi_inv.T[i, j] * self.a_operator(j) / np.sqrt(2.0)
                                            for j in range(dim)], axis=0)) for i in range(dim)])
        return exp_a_list

    def _build_minima_dependent_exponentiated_translation_operators(self, minima_diff, Xi_inv):
        """In general this is a costly part of the code (expm is quite slow)"""
        dim = self.number_degrees_freedom
        exp_a_minima_difference = expm(np.sum([minima_diff[i] * Xi_inv.T[i, j] * self.a_operator(j) / np.sqrt(2.0)
                                               for i in range(dim) for j in range(dim)], axis=0))
        return exp_a_minima_difference

    def _translation_operator_builder(self, exp_a_list, exp_minima_difference, neighbor):
        """Build translation operators using matrix_power rather than the more costly expm"""
        dim = self.number_degrees_freedom
        translation_op_a_dagger = self.identity()
        translation_op_a = self.identity()
        for j in range(dim):
            translation_op_for_direction = matrix_power(exp_a_list[j].T, neighbor[j])
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
        return np.exp(-0.25*np.sum([self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                    * np.dot(Xi[j, :], Xi.T[:, k]) for j in range(dim) for k in range(dim)]))

    def hamiltonian(self):
        pass

    def kinetic_matrix(self, optimized_lengths=None):
        """
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        Xi = self.Xi_matrix(optimized_lengths)
        Xi_inv = inv(Xi)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger()
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        kinetic_function = partial(self._local_kinetic_contribution_to_transfer_matrix,
                                   premultiplied_a_and_a_dagger, EC_mat_t, Xi_inv)
        return self._periodic_continuation(kinetic_function, Xi)

    def potential_matrix(self, optimized_lengths=None):
        """
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        Xi = self.Xi_matrix(optimized_lengths)
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators(Xi)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger()
        potential_function = partial(self._local_potential_contribution_to_transfer_matrix, exp_i_phi_list,
                                     premultiplied_a_and_a_dagger, Xi)
        return self._periodic_continuation(potential_function, Xi)

    def transfer_matrix(self, optimized_lengths=None):
        """
        Returns
        -------
        ndarray
            Returns the transfer matrix
        """
        Xi = self.Xi_matrix(optimized_lengths)
        Xi_inv = inv(Xi)
        nearest_neighbors = self._find_relevant_periodic_continuation_vectors(Xi)
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators(Xi)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger()
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        transfer_matrix_function = partial(self._local_contribution_to_transfer_matrix, exp_i_phi_list,
                                           premultiplied_a_and_a_dagger, EC_mat_t, Xi, Xi_inv)
        return self._periodic_continuation(transfer_matrix_function, Xi, nearest_neighbors=nearest_neighbors)

    def inner_product_matrix(self, optimized_lengths=None):
        """
        Returns
        -------
        ndarray
            Returns the inner product matrix
        """
        Xi = self.Xi_matrix(optimized_lengths)
        return self._periodic_continuation(self._inner_product_operator, Xi)

    def transfer_matrix_and_inner_product(self, nearest_neighbors=None, optimized_lengths=None):
        Xi = self.Xi_matrix(optimized_lengths)
        Xi_inv = inv(Xi)
        if nearest_neighbors is None:
            nearest_neighbors = self._find_relevant_periodic_continuation_vectors(Xi)
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators(Xi)
        exp_a_list = self._build_general_exponentiated_translation_operators(Xi_inv)
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger()
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T

        transfer_matrix_function = partial(self._local_contribution_to_transfer_matrix, exp_i_phi_list,
                                           premultiplied_a_and_a_dagger, EC_mat_t, Xi, Xi_inv)
        transfer_matrix = self._periodic_continuation(transfer_matrix_function, Xi, exp_a_list=exp_a_list,
                                                      nearest_neighbors=nearest_neighbors)
        inner_product_matrix = self._periodic_continuation(self._inner_product_operator, Xi, exp_a_list=exp_a_list,
                                                           nearest_neighbors=nearest_neighbors)
        return transfer_matrix, inner_product_matrix

    def _local_kinetic_contribution_to_transfer_matrix(self, premultiplied_a_and_a_dagger, EC_mat_t, Xi_inv,
                                                       phi_neighbor, minima_m, minima_p):
        """Calculating products of a, a_dagger operators is costly,
        as well as repeatedly calculating Xi (or Xi_inv) which is why they are
        passed to this function in this way rather than calculated below"""
        a, a_a, a_dagger_a = premultiplied_a_and_a_dagger
        minima_diff = minima_p - minima_m
        delta_phi = phi_neighbor + minima_diff
        delta_phi_rotated = Xi_inv @ delta_phi
        kinetic_matrix = np.sum([(-0.5*4*a_a[i] - 0.5*4*a_a[i].T + 0.5*8*a_dagger_a[i]
                                  - 4*(a[i] - a[i].T)*delta_phi_rotated[i]/np.sqrt(2.0))
                                 * EC_mat_t[i, i]
                                 for i in range(self.number_degrees_freedom)], axis=0)
        identity_coefficient = 0.5 * 4 * np.trace(EC_mat_t)
        identity_coefficient += -0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated
        kinetic_matrix += identity_coefficient * self.identity()
        return kinetic_matrix

    def _local_potential_contribution_to_transfer_matrix(self, exp_i_phi_list, premultiplied_a_and_a_dagger,
                                                         Xi, phi_neighbor, minima_m, minima_p):
        """Calculating exp_i_phi operators is costly, which is why it is
        passed to this function in this way rather than calculated below"""
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_list_without_boundary = np.array([exp_i_phi_list[i] * np.exp(1j * phi_bar[i])
                                                    for i in range(dim)])
        exp_i_sum_phi = (exp_i_phi_list[-1] * np.exp(1j * 2.0 * np.pi * self.flux)
                         * np.prod([np.exp(1j * self.boundary_coefficients[i] * phi_bar[i]) for i in range(dim)]))
        potential_matrix = np.sum([-0.5*self.EJlist[junction]
                                   * (exp_i_phi_list_without_boundary[junction]
                                      + exp_i_phi_list_without_boundary[junction].conjugate())
                                   for junction in range(dim)], axis=0)
        potential_matrix += -0.5*self.EJlist[-1]*(exp_i_sum_phi + exp_i_sum_phi.conjugate())
        potential_matrix += np.sum(self.EJlist) * self.identity()
        return potential_matrix

    def _local_contribution_to_transfer_matrix(self, exp_i_phi_list, premultiplied_a_and_a_dagger, EC_mat_t,
                                               Xi, Xi_inv, phi_neighbor, minima_m, minima_p):
        return (self._local_kinetic_contribution_to_transfer_matrix(premultiplied_a_and_a_dagger, EC_mat_t, Xi_inv,
                                                                    phi_neighbor, minima_m, minima_p)
                + self._local_potential_contribution_to_transfer_matrix(exp_i_phi_list, premultiplied_a_and_a_dagger,
                                                                        Xi, phi_neighbor, minima_m, minima_p))

    def _inner_product_operator(self, phi_neighbor, minima_m, minima_p):
        """The three arguments need to be passed in order to match the signature of
        operators that are functions of the raising and lowering operators, whose local
        contributions depend on the periodic continuation vector `phi_neighbor` as well
        as the minima where the states in question are located."""
        return self.identity()

    def _periodic_continuation(self, func, Xi, exp_a_list=None, nearest_neighbors=None):
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
        nearest_neighbors: _find_relevant_periodic_continuation_vectors()
            list that encodes the nearest neighbors relevant when examining matrix elements
            between states in inequivalent minima.

        Returns
        -------
        ndarray
        """
        Xi_inv = inv(Xi)
        if exp_a_list is None:
            exp_a_list = self._build_general_exponentiated_translation_operators(Xi_inv)
        if nearest_neighbors is None:
            nearest_neighbors = self._find_relevant_periodic_continuation_vectors(Xi)
        minima_list = self.sorted_minima()
        hilbertdim = self.hilbertdim()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_list[p] - minima_m
                exp_minima_difference = self._build_minima_dependent_exponentiated_translation_operators(minima_diff,
                                                                                                         Xi_inv)
                for neighbor in nearest_neighbors[counter]:
                    phi_neighbor = 2.0 * np.pi * np.array(neighbor)
                    exp_prod_coefficient = self._exp_product_coefficient(phi_neighbor + minima_diff, Xi_inv)
                    exp_a_dagger, exp_a = self._translation_operator_builder(exp_a_list, exp_minima_difference,
                                                                             neighbor)
                    matrix_element = exp_prod_coefficient*func(phi_neighbor, minima_m, minima_p)
                    matrix_element = exp_a_dagger @ matrix_element @ exp_a
                    operator_matrix[m*num_states_min: (m + 1)*num_states_min,
                                    p*num_states_min: (p + 1)*num_states_min] += matrix_element
                counter += 1
        operator_matrix = self._populate_hermitean_matrix(operator_matrix)
        return operator_matrix

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
                mod_result = np.mod(result.x, 2.0*np.pi*np.ones_like(result.x))
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

        Xi = self.Xi_matrix(None)
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
            klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=dim_periodic)
            neighbor = next(klist, -1)
            while neighbor != -1:
                phik = 2.0 * np.pi * np.concatenate((np.zeros(dim_extended), neighbor))
                phi_offset = phik - minimum
                state_amplitudes = self.state_amplitudes_function(i, evecs, which)
                phi_1_with_offset = phi_1_vec + phi_offset[0]
                phi_2_with_offset = phi_2_vec + phi_offset[1]
                normal_mode_1 = np.add.outer(Xi_inv[0, 0]*phi_1_with_offset, Xi_inv[0, 1]*phi_2_with_offset)
                normal_mode_2 = np.add.outer(Xi_inv[1, 0]*phi_1_with_offset, Xi_inv[1, 1]*phi_2_with_offset)
                wavefunction_amplitudes += (self.wavefunction_amplitudes_function(state_amplitudes,
                                                                                  normal_mode_1, normal_mode_2)
                                            * normalization * np.exp(-1j * np.dot(self.nglist, phi_offset)))
                neighbor = next(klist, -1)

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


class OptimizeHarmonicLength:
    number_degrees_freedom: int
    Xi_matrix: Callable[[np.ndarray], np.ndarray]
    _find_relevant_periodic_continuation_vectors: Callable[[np.ndarray], np.ndarray]
    transfer_matrix_and_inner_product: Callable[..., np.ndarray]

    def optimize_Xi_variational(self):
        """
        Returns
        -------
        ndarray
            Returns a list of integers representing the optimal scaling factor for each mode to minimize the g.s.e.
        """
        default_lengths = np.ones(self.number_degrees_freedom)
        Xi = self.Xi_matrix(default_lengths)
        nearest_neighbors = self._find_relevant_periodic_continuation_vectors(Xi)
        evals_func = partial(self._evals_calc_variational, nearest_neighbors)
        optimized_lengths = minimize(evals_func, default_lengths, tol=1e-2)
        assert optimized_lengths.success
        self.optimized_lengths = optimized_lengths.x
        return optimized_lengths.x

    def _evals_calc_variational(self, nearest_neighbors, optimized_lengths):
        transfer_matrix, inner_product = self.transfer_matrix_and_inner_product(nearest_neighbors=nearest_neighbors,
                                                                                optimized_lengths=optimized_lengths)
        evals = eigh(transfer_matrix, b=inner_product, eigvals_only=True, eigvals=(0, 0))
        return evals

    def _evals_calc(self, evals_count):
        optimized_lengths = self.optimize_Xi_variational()
        transfer_matrix, inner_product = self.transfer_matrix_and_inner_product(optimized_lengths=optimized_lengths)
        try:
            evals = eigh(transfer_matrix, b=inner_product,
                         eigvals_only=True, eigvals=(0, evals_count - 1))
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            evals = solve_generalized_eigenvalue_problem_with_QZ(transfer_matrix, inner_product,
                                                                 evals_count, eigvals_only=True)
        return evals

    def _esys_calc(self, evals_count):
        optimized_lengths = self.optimize_Xi_variational()
        transfer_matrix, inner_product = self.transfer_matrix_and_inner_product(optimized_lengths=optimized_lengths)
        try:
            evals, evecs = eigh(transfer_matrix, b=inner_product,
                                eigvals_only=False, eigvals=(0, evals_count - 1))
            evals, evecs = order_eigensystem(evals, evecs)
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            evals, evecs = solve_generalized_eigenvalue_problem_with_QZ(transfer_matrix, inner_product,
                                                                        evals_count, eigvals_only=False)
        return evals, evecs
