import itertools
import warnings
from functools import partial, reduce

from typing import Callable, List, Tuple, Optional
import numpy as np
from scipy.linalg import LinAlgError, expm, inv, eigh
from scipy.optimize import minimize
import scipy.constants as const
from numpy import ndarray
from numpy.linalg import matrix_power

from scqubits.core import discretization, storage, descriptors
import scqubits.core.constants as constants
from scqubits.core.operators import annihilation, operator_in_full_Hilbert_space
from scqubits.core.hashing import generate_next_vector, reflect_vectors
import scqubits.utils.plotting as plot
from scqubits.utils.cpu_switch import get_map_method
from scqubits.utils.spectrum_utils import order_eigensystem, solve_generalized_eigenvalue_problem_with_QZ, \
    standardize_phases, standardize_sign


class VariationalTightBinding:
    r"""Variational Tight Binding

    This module allows for the diagonalization of superconducting circuit Hamiltonians using as a basis states
    that more closely approximate the low-energy eigenstates as compared to the charge basis.
    This module assumes that the potential is of the form

    .. math::

        U=-EJ[1]\cos(\phi_1)-EJ[2]\cos(\phi_2)-...-EJ[N]\cos(bc[1]\phi_1+bc[2]\phi_2+...-2\pi f),

    where the array :math:`bc` denotes the coefficients of terms in the boundary term.
    Extension of this module to circuits that include inductors is possible and is implemented for the
    zero-pi qubit in zero_pi_vtb.py. The user must define a new qubit class
    that inherits VariationalTightBinding, with all of the qubit specific information. Specifically, the user
    must provide in their child qubit class the functions build_capacitance_matrix(),
    build_EC_matrix(), find_minima(), which define the capacitance matrix,
    the charging energy matrix, and a method to find and find all inequivalent minima, respectively.
    See current_mirror_vtb.py, flux_qubit_vtb.py and zero_pi_vtb.py for examples.

    Parameters
    ----------
    EJlist: ndarray
        Josephson energies associated with each junction, which are allowed to vary.
    nglist: ndarray
        offset charge associated with each dynamical degree of freedom.
    flux: float
        magnetic flux through the circuit loop, measured in units of the flux quantum
    maximum_periodic_vector_length: int
        Maximum Manhattan length of a periodic continuation vector. This should be varied to ensure convergence.
    number_degrees_freedom: int
        number of degrees of freedom of the circuit
    number_periodic_degrees_freedom: int
        number of periodic degrees of freedom
    num_exc: int
        number of excitations kept in each mode
    nearest_neighbors: dict
        dictionary of the relevant periodic continuation vectors for each minima pair. It can
        be passed here to avoid constructing it later, a costly calculation
    harmonic_length_optimization: int
        flag denoting whether or not to optimize the harmonic length
    optimize_all_minima: int
        flag only relevant in the case of squeezing (see class VariationalTightBindingSqueezing) denoting whether or
        not to optimize the harmonic lengths in all minima
    quiet: int
        flag whether or not to print out information regarding completion of intermediate tasks
    """
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    maximum_periodic_vector_length = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    potential: Callable
    find_minima: Callable
    build_capacitance_matrix: Callable
    build_EC_matrix: Callable
    boundary_coefficients: ndarray
    EJlist: ndarray
    nglist: ndarray
    flux: float

    def __init__(self,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 number_degrees_freedom: int,
                 number_periodic_degrees_freedom: int,
                 harmonic_length_optimization: int = 0,
                 optimize_all_minima: int = 0,
                 quiet: int = 0,
                 num_cpus: int = 1,
                 ) -> None:
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        self.Z0 = 1. / (2 * self.e)**2
        self.Phi0 = 1. / (2 * self.e)
        self.nearest_neighbor_cutoff = 1e-15
        self.maximum_periodic_vector_length = maximum_periodic_vector_length
        self.maximum_site_length = 2
        self.number_degrees_freedom = number_degrees_freedom
        self.number_periodic_degrees_freedom = number_periodic_degrees_freedom
        self.number_extended_degrees_freedom = number_degrees_freedom - number_periodic_degrees_freedom
        self.num_exc = num_exc
        self.harmonic_length_optimization = harmonic_length_optimization
        self.optimize_all_minima = optimize_all_minima
        self.quiet = quiet
        self.num_cpus = num_cpus
        self.periodic_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 100)
        self.extended_grid = discretization.Grid1d(-6 * np.pi, 6 * np.pi, 200)
        self.optimized_lengths = np.array([])
        self._evec_dtype = np.complex_

    def build_gamma_matrix(self, minimum: int = 0) -> ndarray:
        """Returns linearized potential matrix

        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.

        We are imagining an arbitrary loop of JJs where we have
        changed variables to the difference variables, so that
        each junction is a function of just one variable, except for
        the last junction, which is a function of all of the variables

        Parameters
        ----------
        minimum: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom
        gamma_matrix = np.zeros((dim, dim))
        min_loc = self.sorted_minima()[minimum]
        gamma_list = self.EJlist / self.Phi0 ** 2

        gamma_diag = np.diag(np.array([gamma_list[j] * np.cos(min_loc[j]) for j in range(dim)]))
        gamma_matrix = gamma_matrix + gamma_diag

        min_loc_bound_sum = np.sum(np.array([self.boundary_coefficients[j] * min_loc[j] for j in range(dim)]))
        for j in range(dim):
            for k in range(dim):
                gamma_matrix[j, k] += (gamma_list[-1] * self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                       * np.cos(min_loc_bound_sum + 2*np.pi*self.flux))
        return gamma_matrix

    def eigensystem_normal_modes(self, minimum: int = 0) -> (ndarray, ndarray):
        """Returns squared normal mode frequencies, matrix of eigenvectors

        Parameters
        ----------
        minimum: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray, ndarray
        """
        C_matrix = self.build_capacitance_matrix()
        g_matrix = self.build_gamma_matrix(minimum)

        omega_squared, normal_mode_eigenvectors = eigh(g_matrix, b=C_matrix)
        standardized_normal_mode_eigenvectors = np.zeros_like(normal_mode_eigenvectors)
        for j, eigenvec in enumerate(normal_mode_eigenvectors.T):
            standardized_normal_mode_eigenvectors[:, j] = standardize_sign(eigenvec)
        return omega_squared, standardized_normal_mode_eigenvectors

    def omega_matrix(self, minimum: int = 0) -> ndarray:
        """Returns a diagonal matrix of the normal mode frequencies of a given minimum

        Parameters
        ----------
        minimum: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray
        """
        omega_squared, _ = self.eigensystem_normal_modes(minimum)
        return np.sqrt(omega_squared)

    def compare_harmonic_lengths_with_minima_separations(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            ratio of harmonic lengths to minima separations, providing a measure of the validity of tight binding.
            If any of the values in the returned array exceed unity, then the wavefunctions are relatively spread out
            as compared to the minima separations
        """
        return self._wrapper_for_functions_comparing_minima(self._find_closest_periodic_minimum)

    def _wrapper_for_functions_comparing_minima(self, function: Callable) -> ndarray:
        """Helper function for functions comparing minima"""
        minima_list = self.sorted_minima()
        minima_list_with_index = zip(minima_list, [m for m in range(len(minima_list))])
        all_minima_pairs = itertools.combinations_with_replacement(minima_list_with_index, 2)
        return np.array([function(minima_pair) for minima_pair in all_minima_pairs], dtype=object)

    def _find_closest_periodic_minimum(self, minima_pair: Tuple) -> ndarray:
        """Helper function comparing minima separation for given minima pair"""
        return self._find_closest_periodic_minimum_for_given_minima(minima_pair, 0)

    def _find_closest_periodic_minimum_for_given_minima(self, minima_pair: Tuple, minimum: int) -> ndarray:
        """Helper function comparing minima separation for given minima pair, along with the specification
        that we would like to use the Xi matrix as defined for `minimum`"""
        (minima_m, m), (minima_p, p) = minima_pair
        nearest_neighbors = self.nearest_neighbors
        if nearest_neighbors is None or np.allclose(nearest_neighbors, [np.zeros(self.number_degrees_freedom)]):
            return np.array([0.0])
        Xi_inv = inv(self.Xi_matrix(minimum=minimum))
        delta_inv = Xi_inv.T @ Xi_inv
        if m == p:  # Do not include equivalent minima in the same unit cell
            nearest_neighbors = np.array([vec for vec in nearest_neighbors if not np.allclose(vec, np.zeros_like(vec))])
        minima_distances = np.array([np.linalg.norm(2.0*np.pi*vec + (minima_p - minima_m)) / 2.0
                                     for vec in nearest_neighbors])
        minima_vectors = np.array([2.0 * np.pi * vec + (minima_p - minima_m)
                                   for i, vec in enumerate(nearest_neighbors)])
        minima_unit_vectors = np.array([minima_vectors[i] / minima_distances[i] for i in range(len(minima_distances))])
        harmonic_lengths = np.array([4.0*(unit_vec @ delta_inv @ unit_vec)**(-1/2)
                                     for unit_vec in minima_unit_vectors])
        return np.max(harmonic_lengths / minima_distances)

    def Xi_matrix(self, minimum: int = 0) -> ndarray:
        """ Returns Xi matrix of the normal mode eigenvectors normalized to encode the harmonic length.
        This matrix simultaneously diagonalizes the capacitance and effective inductance matrices.

        Parameters
        ----------
        minimum: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray
        """
        minima_list = self.sorted_minima()
        if self.optimized_lengths.size == 0 or self.harmonic_length_optimization == 0:
            self.optimized_lengths = np.ones((len(minima_list), self.number_degrees_freedom))
        omega_squared, normal_mode_eigenvectors = self.eigensystem_normal_modes(minimum)
        # We introduce a normalization such that \Xi^T C \Xi = \Omega^{-1}/Z0
        Xi_matrix = np.array([normal_mode_eigenvectors[:, i] * self.optimized_lengths[minimum, i] * omega**(-1/4)
                              * np.sqrt(1./self.Z0) for i, omega in enumerate(omega_squared)]).T
        return Xi_matrix

    def a_operator(self, mu: int) -> ndarray:
        """Returns the lowering operator associated with the mu^th d.o.f. in the full Hilbert space

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

    def _a_operator_list(self) -> ndarray:
        """Helper method to return a list of annihilation operator matrices for each mode"""
        return np.array([self.a_operator(i) for i in range(self.number_degrees_freedom)])

    @property
    def nearest_neighbors(self):
        target_map = get_map_method(self.num_cpus)
        dim_extended = self.number_extended_degrees_freedom
        zero_vec = np.concatenate((np.zeros(dim_extended, dtype=int), np.zeros(self.number_periodic_degrees_freedom)))
        periodic_vector_lengths = np.array([i for i in range(1, self.maximum_periodic_vector_length + 1)])
        filtered_vectors = list(target_map(self._sum_over_generated_vectors, periodic_vector_lengths))
        filtered_vectors.append(zero_vec)
        return self._stack_filtered_vectors(filtered_vectors)

    def _sum_over_generated_vectors(self, periodic_vector_length):
        sites = self.number_periodic_degrees_freedom
        prev_vec = np.zeros(sites, dtype=int)
        prev_vec[0] = periodic_vector_length
        new_vectors = []
        if periodic_vector_length <= self.maximum_site_length:
            reflected_vectors = reflect_vectors(prev_vec)
            for vecs in reflected_vectors:
                new_vectors.append(vecs)
        while prev_vec[-1] != periodic_vector_length:
            next_vec = generate_next_vector(prev_vec, periodic_vector_length)
            if len(np.argwhere(next_vec > self.maximum_site_length)) == 0:
                reflected_vectors = reflect_vectors(prev_vec)
                for vecs in reflected_vectors:
                    new_vectors.append(vecs)
            prev_vec = next_vec
        return np.array(new_vectors)

    @staticmethod
    def _stack_filtered_vectors(filtered_vectors: List) -> Optional[ndarray]:
        """Helper function for stacking together periodic continuation vectors of different Manhattan lengths"""
        filtered_vectors = list(filter(lambda x: len(x) != 0, filtered_vectors))
        if filtered_vectors:
            return np.vstack(filtered_vectors)
        else:
            return None

    def identity(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the identity matrix whose dimensions are the same as self.a_operator(mu)
        """
        return np.eye(int(self.number_states_per_minimum()))

    def number_states_per_minimum(self) -> int:
        """
        Returns
        -------
        int
            Returns the number of states displaced into each local minimum
        """
        return (self.num_exc + 1)**self.number_degrees_freedom

    def hilbertdim(self) -> int:
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension.
        """
        return int(len(self.sorted_minima()) * self.number_states_per_minimum())

    def _build_premultiplied_a_a_dagger(self, a_operator_list: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """Helper method for premultiplying creation and annihilation operators (multiplications are expensive)"""
        dim = self.number_degrees_freedom
        a_a = np.array([a_operator_list[i] @ a_operator_list[i] for i in range(dim)])
        a_dagger_a = np.array([a_operator_list[i].T @ a_operator_list[i] for i in range(dim)])
        return a_operator_list, a_a, a_dagger_a

    def _build_single_exp_i_phi_j_operator(self, j: int, Xi: ndarray, a_operator_list: ndarray) -> ndarray:
        r"""Returns operator :math:`\exp(i\phi_{j})`. If `j` specifies the boundary term, then that is constructed
        based on the boundary coefficients."""
        dim = self.number_degrees_freedom
        if j == dim:
            exp_i_phi_j_a_component = expm(np.sum([self.boundary_coefficients[i] * 1j * Xi[i, k] * a_operator_list[k]
                                                  for i in range(dim) for k in range(dim)], axis=0) / np.sqrt(2.0))
            BCH_factor = self._BCH_factor_for_potential_boundary(Xi)
        else:
            exp_i_phi_j_a_component = expm(np.sum([1j * Xi[j, k] * a_operator_list[k]
                                                  for k in range(dim)], axis=0) / np.sqrt(2.0))
            BCH_factor = np.exp(-0.25*np.dot(Xi[j, :], Xi.T[:, j]))
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a_component.T
        return BCH_factor * exp_i_phi_j_a_dagger_component @ exp_i_phi_j_a_component

    def _build_all_exp_i_phi_j_operators(self, Xi: ndarray, a_operator_list: ndarray) -> ndarray:
        """Helper method for building all potential operators"""
        return np.array([self._build_single_exp_i_phi_j_operator(j, Xi, a_operator_list)
                         for j in range(self.number_degrees_freedom + 1)])

    def _build_general_translation_operators(self, Xi_inv: ndarray,
                                             a_operator_list: ndarray) -> Tuple[ndarray, ndarray]:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. The resulting matrices yield a 2pi translation
        in each degree of freedom, so that any translation can be built from these by an appropriate
        call to np.matrix_power"""
        dim = self.number_degrees_freedom
        exp_a_list = np.array([expm(np.sum([2.0 * np.pi * Xi_inv.T[i, j] * a_operator_list[j] / np.sqrt(2.0)
                                    for j in range(dim)], axis=0)) for i in range(dim)])
        exp_a_dagger_list = np.array([expm(np.sum([2.0 * np.pi * Xi_inv.T[i, j] * a_operator_list[j].T / np.sqrt(2.0)
                                      for j in range(dim)], axis=0)) for i in range(dim)])
        return exp_a_list, exp_a_dagger_list

    def _build_minima_dependent_translation_operators(self, minima_diff: ndarray, Xi_inv: ndarray,
                                                      a_operator_list: ndarray) -> Tuple[ndarray, ndarray]:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. This part of the translation operator accounts
        for the differing location of minima within a single unit cell."""
        dim = self.number_degrees_freedom
        exp_a_minima_difference = expm(np.sum([-minima_diff[i] * Xi_inv.T[i, j] * a_operator_list[j] / np.sqrt(2.0)
                                               for i in range(dim) for j in range(dim)], axis=0))
        exp_a_dagger_minima_difference = expm(np.sum([minima_diff[i] * Xi_inv.T[i, j] * a_operator_list[j].T
                                              for i in range(dim) for j in range(dim)], axis=0) / np.sqrt(2.0))
        return exp_a_minima_difference, exp_a_dagger_minima_difference

    def _build_local_translation_operators(self, exp_a_list: Tuple[ndarray, ndarray],
                                           exp_minima_difference: Tuple[ndarray, ndarray],
                                           neighbor: ndarray) -> Tuple[ndarray, ndarray]:
        """Helper method that builds translation operators using matrix_power and the pre-exponentiated
        translation operators that define 2pi translations."""
        dim = self.number_degrees_freedom
        exp_a_list, exp_a_dagger_list = exp_a_list
        exp_a_minima_difference, exp_a_dagger_minima_difference = exp_minima_difference
        individual_a_dagger_op = np.array([matrix_power(exp_a_dagger_list[j], int(neighbor[j])) for j in range(dim)])
        individual_a_op = np.array([matrix_power(exp_a_list[j], -int(neighbor[j])) for j in range(dim)])
        translation_op_a_dagger = reduce((lambda x, y: x @ y), individual_a_dagger_op) @ exp_a_dagger_minima_difference
        translation_op_a = reduce((lambda x, y: x @ y), individual_a_op) @ exp_a_minima_difference
        return translation_op_a_dagger, translation_op_a

    def _exp_product_coefficient(self, delta_phi: ndarray, Xi_inv: ndarray) -> ndarray:
        """Returns overall multiplicative factor, including offset charge and Gaussian suppression BCH factor
        from the periodic continuation (translation) operators"""
        delta_phi_rotated = Xi_inv @ delta_phi
        return np.exp(-1j * self.nglist @ delta_phi) * np.exp(-0.25 * delta_phi_rotated @ delta_phi_rotated)

    def _BCH_factor_for_potential_boundary(self, Xi: ndarray) -> ndarray:
        """BCH factor obtained from the last potential operator"""
        dim = self.number_degrees_freedom
        return np.exp(-0.25*np.sum([self.boundary_coefficients[j] * self.boundary_coefficients[k]
                                   * np.dot(Xi[j, :], Xi.T[:, k]) for j in range(dim) for k in range(dim)]))

    def n_operator(self, j: int = 0) -> ndarray:
        Xi_inv = inv(self.Xi_matrix())
        a_operator_list = self._a_operator_list()
        premultiplied_a_a_dagger = self._build_premultiplied_a_a_dagger(a_operator_list)
        charge_function = partial(self._local_charge_operator, j, premultiplied_a_a_dagger, Xi_inv)
        return self._periodic_continuation(charge_function)

    def phi_operator(self, j: int = 0) -> ndarray:
        Xi = self.Xi_matrix()
        a_operator_list = self._a_operator_list()
        premultiplied_a_a_dagger = self._build_premultiplied_a_a_dagger(a_operator_list)
        phi_function = partial(self._local_phi_operator, j, premultiplied_a_a_dagger, Xi)
        return self._periodic_continuation(phi_function)

    def exp_i_phi_operator(self, j: int = 0) -> ndarray:
        exp_i_phi_j = self._build_all_exp_i_phi_j_operators(self.Xi_matrix(), self._a_operator_list())
        exp_i_phi_j_function = partial(self._local_exp_i_phi_operator, j, exp_i_phi_j)
        return self._periodic_continuation(exp_i_phi_j_function)

    def hamiltonian(self) -> ndarray:
        return self.transfer_matrix()

    def kinetic_matrix(self, nearest_neighbors: ndarray = None) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        Xi_inv = inv(self.Xi_matrix())
        a_operator_list = self._a_operator_list()
        premultiplied_a_a_dagger = self._build_premultiplied_a_a_dagger(a_operator_list)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        kinetic_function = partial(self._local_kinetic, premultiplied_a_a_dagger, EC_mat_t, Xi_inv)
        return self._periodic_continuation(kinetic_function, nearest_neighbors)

    def potential_matrix(self, nearest_neighbors: ndarray = None) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        Xi = self.Xi_matrix()
        a_operator_list = self._a_operator_list()
        exp_i_phi_j = self._build_all_exp_i_phi_j_operators(Xi, a_operator_list)
        premultiplied_a_a_dagger = self._build_premultiplied_a_a_dagger(a_operator_list)
        potential_function = partial(self._local_potential, exp_i_phi_j, premultiplied_a_a_dagger, Xi)
        return self._periodic_continuation(potential_function, nearest_neighbors)

    def transfer_matrix(self, nearest_neighbors: ndarray = None) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the transfer matrix
        """
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_list = self._a_operator_list()
        exp_i_phi_j = self._build_all_exp_i_phi_j_operators(Xi, a_operator_list)
        premultiplied_a_a_dagger = self._build_premultiplied_a_a_dagger(a_operator_list)
        EC_mat_t = Xi_inv @ self.build_EC_matrix() @ Xi_inv.T
        transfer_matrix_function = partial(self._local_kinetic_plus_potential, exp_i_phi_j,
                                           premultiplied_a_a_dagger, EC_mat_t, Xi, Xi_inv)
        return self._periodic_continuation(transfer_matrix_function, nearest_neighbors)

    def inner_product_matrix(self, nearest_neighbors: ndarray = None) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the inner product matrix
        """
        return self._periodic_continuation(lambda x, y, z: self.identity(), nearest_neighbors)

    def _local_charge_operator(self, j: int, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                               Xi_inv: ndarray, phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        constant_coefficient = -0.5 * 1j * (Xi_inv.T @ Xi_inv @ (phi_neighbor + minima_p - minima_m))[j]
        return (-(1j/np.sqrt(2.0)) * np.sum([Xi_inv.T[j, mu] * (a[mu] - a[mu].T) for mu in range(dim)], axis=0)
                + constant_coefficient * self.identity())

    def _local_phi_operator(self, j: int, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                            Xi: ndarray, phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        constant_coefficient = 0.5 * (phi_neighbor + (minima_m + minima_p))
        return ((1.0/np.sqrt(2.0)) * np.sum([Xi[j, mu] * (a[mu] + a[mu].T) for mu in range(dim)], axis=0)
                + constant_coefficient[j] * self.identity())

    def _exp_i_phi_j_operators_with_phi_bar(self, exp_i_phi_j: ndarray, phi_bar: ndarray) -> Tuple[ndarray, ndarray]:
        """Returns exp_i_phi_j operators including the local contribution of phi_bar"""
        exp_i_phi_j_without_boundary = np.array([exp_i_phi_j[i] * np.exp(1j * phi_bar[i])
                                                 for i in range(self.number_degrees_freedom)])
        exp_i_sum_phi = (exp_i_phi_j[-1] * np.exp(1j * 2.0 * np.pi * self.flux)
                         * np.exp(1j * self.boundary_coefficients @ phi_bar))
        return exp_i_phi_j_without_boundary, exp_i_sum_phi

    def _local_exp_i_phi_operator(self, j: int, exp_i_phi_j: ndarray, phi_neighbor: ndarray,
                                  minima_m: ndarray, minima_p: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_j_without_boundary, exp_i_sum_phi = self._exp_i_phi_j_operators_with_phi_bar(exp_i_phi_j, phi_bar)
        if j == dim:
            return exp_i_sum_phi
        else:
            return exp_i_phi_j_without_boundary[j]

    def _local_kinetic(self, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                       EC_mat_t: ndarray, Xi_inv: ndarray, phi_neighbor: ndarray,
                       minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Calculate the local kinetic contribution to the transfer matrix given two
        minima and a periodic continuation vector `phi_neighbor`"""
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        delta_phi = phi_neighbor + minima_p - minima_m
        delta_phi_rotated = Xi_inv @ delta_phi
        kinetic_matrix = np.sum([EC_mat_t[i, i]*(-0.5*4*a_a[i] - 0.5*4*a_a[i].T + 0.5*8*a_dagger_a[i]
                                                 - 4*(a[i] - a[i].T)*delta_phi_rotated[i]/np.sqrt(2.0))
                                 for i in range(self.number_degrees_freedom)], axis=0)
        identity_coefficient = 0.5*4*np.trace(EC_mat_t) - 0.25*4*delta_phi_rotated @ EC_mat_t @ delta_phi_rotated
        kinetic_matrix = kinetic_matrix + identity_coefficient*self.identity()
        return kinetic_matrix

    def _local_potential(self, exp_i_phi_j: ndarray, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                         Xi: ndarray, phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Calculate the local potential contribution to the transfer matrix given two
        minima and a periodic continuation vector `phi_neighbor`"""
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_j_without_boundary, exp_i_sum_phi = self._exp_i_phi_j_operators_with_phi_bar(exp_i_phi_j, phi_bar)
        potential_matrix = np.sum([-0.5*self.EJlist[junction]
                                   * (exp_i_phi_j_without_boundary[junction]
                                      + exp_i_phi_j_without_boundary[junction].conjugate())
                                   for junction in range(dim)], axis=0)
        potential_matrix = potential_matrix - 0.5*self.EJlist[-1]*(exp_i_sum_phi + exp_i_sum_phi.conjugate())
        potential_matrix = potential_matrix + np.sum(self.EJlist)*self.identity()
        return potential_matrix

    def _local_kinetic_plus_potential(self, exp_i_phi_j: ndarray,
                                      premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                                      EC_mat_t: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                      phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Calculate the local contribution to the transfer matrix given two
        minima and a periodic continuation vector `phi_neighbor`"""
        return (self._local_kinetic(premultiplied_a_a_dagger, EC_mat_t, Xi_inv, phi_neighbor, minima_m, minima_p)
                + self._local_potential(exp_i_phi_j, premultiplied_a_a_dagger, Xi, phi_neighbor, minima_m, minima_p))

    def _periodic_continuation(self, func: Callable, nearest_neighbors: ndarray = None) -> ndarray:
        """This function is the meat of the VariationalTightBinding method. Any operator whose matrix
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
        a_operator_list = self._a_operator_list()
        exp_a_list = self._build_general_translation_operators(Xi_inv, a_operator_list)
        minima_list = self.sorted_minima()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((self.hilbertdim(), self.hilbertdim()), dtype=np.complex128)
        minima_list_with_index = zip(minima_list, [m for m in range(len(minima_list))])
        all_minima_pairs = itertools.combinations_with_replacement(minima_list_with_index, 2)
        if nearest_neighbors is None:
            nearest_neighbors = self.nearest_neighbors
        for (minima_m, m), (minima_p, p) in all_minima_pairs:
            matrix_element = self._periodic_continuation_for_minima_pair(minima_m, minima_p,
                                                                         nearest_neighbors,
                                                                         func, exp_a_list, Xi_inv, a_operator_list)
            operator_matrix[m*num_states_min: (m + 1)*num_states_min,
                            p*num_states_min: (p + 1)*num_states_min] += matrix_element
        operator_matrix = self._populate_hermitian_matrix(operator_matrix)
        return operator_matrix

    def _periodic_continuation_for_minima_pair(self, minima_m: ndarray, minima_p: ndarray, nearest_neighbors: ndarray,
                                               func: Callable, exp_a_list: Tuple[ndarray, ndarray], Xi_inv: ndarray,
                                               a_operator_list: ndarray) -> ndarray:
        """Helper method for performing the periodic continuation calculation given a minima pair."""
        if nearest_neighbors is not None:
            minima_diff = minima_p - minima_m
            exp_minima_difference = self._build_minima_dependent_translation_operators(minima_diff, Xi_inv,
                                                                                       a_operator_list)
            return np.sum([self._neighbor_contribution(neighbor, func, minima_m, minima_p,
                                                       exp_a_list, exp_minima_difference, Xi_inv)
                           for neighbor in nearest_neighbors], axis=0)
        else:
            return np.zeros((self.number_states_per_minimum(), self.number_states_per_minimum()), dtype=np.complex_)

    def _neighbor_contribution(self, neighbor: ndarray, func: Callable, minima_m: ndarray, minima_p: ndarray,
                               exp_a_list: Tuple[ndarray, ndarray], exp_minima_difference: Tuple[ndarray, ndarray],
                               Xi_inv: ndarray) -> ndarray:
        """Helper method for calculating the contribution of a specific periodic continuation vector `neighbor`"""
        phi_neighbor = 2.0 * np.pi * np.array(neighbor)
        exp_prod_coefficient = self._exp_product_coefficient(phi_neighbor + minima_p - minima_m, Xi_inv)
        exp_a_dagger, exp_a = self._build_local_translation_operators(exp_a_list, exp_minima_difference,
                                                                      neighbor)
        neighbor_matrix_element = exp_prod_coefficient * func(phi_neighbor, minima_m, minima_p)
        return exp_a_dagger @ neighbor_matrix_element @ exp_a

    def _populate_hermitian_matrix(self, mat: ndarray) -> ndarray:
        """Return a fully Hermitian matrix, assuming that the input matrix has been
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

    def _transfer_matrix_and_inner_product(self) -> Tuple[ndarray, ndarray]:
        """Helper method called by _esys_calc and _evals_calc that returns the transfer matrix and inner product
        matrix but warns the user if the system is in a regime where tight-binding has questionable validity."""
        if self.harmonic_length_optimization:
            self.optimize_Xi_variational_wrapper()
#        harmonic_length_minima_comparison = self.compare_harmonic_lengths_with_minima_separations()
#        if np.max(harmonic_length_minima_comparison) > 1.0 and not self.quiet:
#            print("Warning: large harmonic length compared to minima separation "
#                  "(largest is 3*l/(d/2) = {ratio})".format(ratio=np.max(harmonic_length_minima_comparison)))
        nearest_neighbors = self.nearest_neighbors
        transfer_matrix = self.transfer_matrix(nearest_neighbors)
        inner_product_matrix = self.inner_product_matrix(nearest_neighbors)
        return transfer_matrix, inner_product_matrix

    def _evals_esys_calc(self, evals_count: int, eigvals_only: bool) -> ndarray:
        """Helper method that wraps the try and except regarding
        singularity/indefiniteness of the inner product matrix"""
        transfer_matrix, inner_product_matrix = self._transfer_matrix_and_inner_product()
        try:
            eigs = eigh(transfer_matrix, b=inner_product_matrix,
                        eigvals_only=eigvals_only, eigvals=(0, evals_count - 1))
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            eigs = solve_generalized_eigenvalue_problem_with_QZ(transfer_matrix, inner_product_matrix,
                                                                evals_count, eigvals_only=eigvals_only)
        return eigs

    def _evals_calc(self, evals_count: int) -> ndarray:
        """Overrides method from QubitBaseClass for calculating eigenvalues.
        Here it is clear that we are solving a generalized eigenvalue problem."""
        evals = self._evals_esys_calc(evals_count, True)
        return evals

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        """See _evals_calc. Here we calculate eigenvalues and eigenvectors."""
        evals, evecs = self._evals_esys_calc(evals_count, False)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def _sorted_potential_values_and_minima(self) -> Tuple[ndarray, ndarray]:
        """Returns the value of the potential at minima and the location of minima, in sorted order."""
        minima_holder = self.find_minima()
        value_of_potential = np.array([self.potential(minima) for minima in minima_holder])
        sorted_indices = np.argsort(value_of_potential)
        return value_of_potential[sorted_indices], minima_holder[sorted_indices, :]

    def sorted_minima(self) -> ndarray:
        """
        Return sorted array of the minima locations

        Returns
        -------
        ndarray
        """
        _, sorted_minima_holder = self._sorted_potential_values_and_minima()
        return sorted_minima_holder

    def _normalize_minimum_inside_pi_range(self, minimum: ndarray) -> ndarray:
        """Helper method for defining the unit cell from -pi to pi rather than the less symmetric 0 to 2pi"""
        num_extended = self.number_extended_degrees_freedom
        extended_coordinates = minimum[0:num_extended]
        periodic_coordinates = np.mod(minimum, 2*np.pi*np.ones_like(minimum))[num_extended:]
        periodic_coordinates = np.array([elem - 2*np.pi if elem > np.pi else elem for elem in periodic_coordinates])
        return np.concatenate((extended_coordinates, periodic_coordinates))

    def _check_if_new_minima(self, new_minima: ndarray, minima_holder: List) -> bool:
        """Helper method for find_minima, checking if new_minima is already represented in minima_holder. If so,
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

    def _filter_repeated_minima(self, minima_holder: List) -> List:
        """Eliminate repeated minima contained in minima_holder"""
        filtered_minima_holder = [minima_holder[0]]
        for minima in minima_holder:
            if self._check_if_new_minima(minima, filtered_minima_holder):
                filtered_minima_holder.append(minima)
        return filtered_minima_holder

    def optimize_Xi_variational_wrapper(self) -> None:
        """Optimize the Xi matrix by adjusting the harmonic lengths of the ground state to minimize its energy.
        For tight-binding without squeezing, this is only done for the ansatz ground state wavefunction
        localized in the global minimum."""
        minima_list = self.sorted_minima()
        self.optimized_lengths = np.ones((len(minima_list), self.number_degrees_freedom))
        self._optimize_Xi_variational(0, minima_list[0])
        for minimum, _ in enumerate(minima_list):
            self.optimized_lengths[minimum] = self.optimized_lengths[0]

    def _optimize_Xi_variational(self, minimum: int = 0, minimum_location: ndarray = None) -> None:
        """Perform the harmonic length optimization for a h.o. ground state wavefunction localized in a given minimum"""
        default_Xi = self.Xi_matrix(minimum)
        EC_mat = self.build_EC_matrix()
        optimized_lengths_result = minimize(self._evals_calc_variational, self.optimized_lengths[minimum],
                                            jac=self._gradient_evals_calc_variational,
                                            args=(minimum_location, minimum, EC_mat, default_Xi), tol=1e-1)
        assert optimized_lengths_result.success
        optimized_lengths = optimized_lengths_result.x
        if not self.quiet:
            print("completed harmonic length optimization for the m={m} minimum".format(m=minimum))
        self.optimized_lengths[minimum] = optimized_lengths

    def _update_Xi(self, default_Xi: ndarray, minimum: int) -> ndarray:
        """Helper method for updating Xi so that the Xi matrix is not constantly regenerated."""
        return np.array([row * self.optimized_lengths[minimum, i] for i, row in enumerate(default_Xi.T)]).T

    def _gradient_evals_calc_variational(self, optimized_lengths: ndarray, minimum_location: ndarray, minimum: int,
                                         EC_mat: ndarray, default_Xi: ndarray) -> ndarray:
        """Returns the gradient of evals_calc_variational to aid in the harmonic length optimization calculation"""
        self.optimized_lengths[minimum] = optimized_lengths
        Xi = self._update_Xi(default_Xi, minimum)
        Xi_inv = inv(Xi)
        exp_i_phi_j = self._one_state_exp_i_phi_j_operators(Xi)
        EC_mat_t = Xi_inv @ EC_mat @ Xi_inv.T
        nearest_neighbors = 2.0*np.pi*self.nearest_neighbors
        gradient_transfer = [np.sum([self._exp_product_coefficient(neighbor, Xi_inv)
                                    * self._gradient_one_state_local_transfer(exp_i_phi_j, EC_mat_t, Xi, Xi_inv,
                                                                              neighbor, minimum_location,
                                                                              minimum_location, which_length)
                                    + self._gradient_one_state_local_inner_product(neighbor, Xi_inv, which_length)
                                     * self._one_state_local_transfer(exp_i_phi_j, EC_mat_t, Xi, Xi_inv, neighbor,
                                                                      minimum_location, minimum_location)
                                    for neighbor in nearest_neighbors])
                             for which_length in range(self.number_degrees_freedom)]
        transfer = np.sum([self._exp_product_coefficient(neighbor, Xi_inv)
                           * self._one_state_local_transfer(exp_i_phi_j, EC_mat_t, Xi, Xi_inv, neighbor,
                                                            minimum_location, minimum_location)
                           for neighbor in nearest_neighbors])
        gradient_inner = [np.sum([self._gradient_one_state_local_inner_product(neighbor, Xi_inv, which_length)
                                  for neighbor in nearest_neighbors])
                          for which_length in range(self.number_degrees_freedom)]
        inner = np.sum([self._exp_product_coefficient(neighbor, Xi_inv) for neighbor in nearest_neighbors])
        return np.real((inner * np.array(gradient_transfer) - transfer * np.array(gradient_inner)) / inner**2)

    def _evals_calc_variational(self, optimized_lengths: ndarray, minimum_location: ndarray, minimum: int,
                                EC_mat: ndarray, default_Xi: ndarray) -> ndarray:
        """Function to be optimized in the minimization procedure, corresponding to the variational estimate of
        the ground state energy."""
        self.optimized_lengths[minimum] = optimized_lengths
        Xi = self._update_Xi(default_Xi, minimum)
        Xi_inv = inv(Xi)
        exp_i_phi_j = self._one_state_exp_i_phi_j_operators(Xi)
        EC_mat_t = Xi_inv @ EC_mat @ Xi_inv.T
        transfer, inner = self._one_state_construct_transfer_and_inner(Xi, Xi_inv, minimum_location, minimum,
                                                                       EC_mat_t, exp_i_phi_j)
        return np.real([transfer / inner])

    def _one_state_exp_i_phi_j_operators(self, Xi: ndarray) -> ndarray:
        r"""Helper method for building :math:`\exp(i\phi_{j})` when no excitations are kept."""
        dim = self.number_degrees_freedom
        exp_factors = np.array([np.exp(-0.25*np.dot(Xi[j, :], Xi.T[:, j])) for j in range(dim)])
        return np.append(exp_factors, self._BCH_factor_for_potential_boundary(Xi))

    def _gradient_one_state_local_inner_product(self, delta_phi: ndarray, Xi_inv: ndarray, which_length: int) -> float:
        """Returns gradient of the inner product matrix"""
        delta_phi_rotated = Xi_inv @ delta_phi
        return (self.optimized_lengths[0, which_length]**(-1)
                * (1j * self.nglist[which_length] * delta_phi_rotated[which_length]
                   + 0.5 * delta_phi_rotated[which_length]**2) * self._exp_product_coefficient(delta_phi, Xi_inv))

    @staticmethod
    def _one_state_local_kinetic(EC_mat_t: ndarray, Xi_inv: ndarray,
                                 phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> float:
        """Local kinetic contribution when considering only the ground state."""
        delta_phi_rotated = Xi_inv @ (phi_neighbor + minima_p - minima_m)
        return 0.5 * 4 * np.trace(EC_mat_t) - 0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated

    def _gradient_one_state_local_kinetic(self, EC_mat_t: ndarray, Xi_inv: ndarray, phi_neighbor: ndarray,
                                          minima_m: ndarray, minima_p: ndarray, which_length: int) -> ndarray:
        """Returns gradient of the kinetic matrix"""
        delta_phi_rotated = Xi_inv @ (phi_neighbor + minima_p - minima_m)
        return (-4.0*self.optimized_lengths[0, which_length]**(-1)
                * (EC_mat_t[which_length, which_length] - (delta_phi_rotated @ EC_mat_t)[which_length]
                   * delta_phi_rotated[which_length]))

    def _gradient_one_state_local_potential(self, exp_i_phi_j: ndarray, phi_neighbor: ndarray, minima_m: ndarray,
                                            minima_p: ndarray, Xi: ndarray, which_length: int) -> ndarray:
        """Returns gradient of the potential matrix"""
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_j_without_boundary, exp_i_sum_phi = self._exp_i_phi_j_operators_with_phi_bar(exp_i_phi_j, phi_bar)
        potential_gradient = np.sum([0.25 * self.EJlist[junction]
                                     * Xi[junction, which_length] * Xi.T[which_length, junction]
                                     * self.optimized_lengths[0, which_length]**(-1)
                                     * (exp_i_phi_j_without_boundary[junction]
                                        + exp_i_phi_j_without_boundary[junction].conjugate())
                                     for junction in range(self.number_degrees_freedom)])
        potential_gradient += (0.25 * self.EJlist[-1] * self.optimized_lengths[0, which_length]**(-1)
                               * (self.boundary_coefficients @ Xi[:, which_length])**2
                               * (exp_i_sum_phi + exp_i_sum_phi.conjugate()))
        return potential_gradient

    def _gradient_one_state_local_transfer(self, exp_i_phi_j: ndarray, EC_mat_t: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                           phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                           which_length: int) -> ndarray:
        """Returns gradient of the transfer matrix"""
        return (self._gradient_one_state_local_potential(exp_i_phi_j, phi_neighbor, minima_m,
                                                         minima_p, Xi, which_length)
                + self._gradient_one_state_local_kinetic(EC_mat_t, Xi_inv, phi_neighbor, minima_m,
                                                         minima_p, which_length))

    def _one_state_local_potential(self, exp_i_phi_j: ndarray, Xi: ndarray, phi_neighbor: ndarray,
                                   minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Local potential contribution when considering only the ground state."""
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_phi_j_without_boundary, exp_i_sum_phi = self._exp_i_phi_j_operators_with_phi_bar(exp_i_phi_j, phi_bar)
        potential = np.sum([-0.5 * self.EJlist[junction] * (exp_i_phi_j_without_boundary[junction]
                                                            + exp_i_phi_j_without_boundary[junction].conjugate())
                            for junction in range(self.number_degrees_freedom)])
        potential += np.sum(self.EJlist) - 0.5 * self.EJlist[-1] * (exp_i_sum_phi + exp_i_sum_phi.conjugate())
        return potential

    def _one_state_local_transfer(self, exp_i_phi_j: ndarray, EC_mat_t: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                  phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Local transfer contribution when considering only the ground state."""
        return (self._one_state_local_kinetic(EC_mat_t, Xi_inv, phi_neighbor, minima_m, minima_p)
                + self._one_state_local_potential(exp_i_phi_j, Xi, phi_neighbor, minima_m, minima_p))

    def _one_state_construct_transfer_and_inner(self, Xi: ndarray, Xi_inv: ndarray, minimum_location: ndarray,
                                                minimum: int, EC_mat_t: ndarray,
                                                exp_i_phi_j: ndarray) -> Tuple[ndarray, ndarray]:
        """Transfer matrix and inner product matrix when considering only the ground state."""
        nearest_neighbors = 2.0*np.pi*self.nearest_neighbors
        transfer_function = partial(self._one_state_local_transfer, exp_i_phi_j, EC_mat_t, Xi, Xi_inv)
        transfer = np.sum([self._exp_product_coefficient(neighbor, Xi_inv)
                           * transfer_function(neighbor, minimum_location, minimum_location)
                           for neighbor in nearest_neighbors])
        # Note need to include 2.0*np.pi*np.array(neighbor) + minimum_location - minimum_location for completeness
        inner_product = np.sum([self._exp_product_coefficient(neighbor, Xi_inv)
                                for neighbor in nearest_neighbors])
        return transfer, inner_product

    def wavefunction(self, esys=None, which=0):
        """
        Return a vtb wavefunction, assuming the qubit has 2 degrees of freedom

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
            neighbors = itertools.product(np.arange(-self.maximum_periodic_vector_length,
                                                    self.maximum_periodic_vector_length + 1), repeat=dim_periodic)
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
        """Helper method for wavefunction returning the matrix of state amplitudes
        that can be overridden in the case of a global excitation cutoff"""
        total_num_states = self.number_states_per_minimum()
        return np.real(np.reshape(evecs[i * total_num_states: (i + 1) * total_num_states, which],
                                  (self.num_exc + 1, self.num_exc + 1)))

    def wavefunction_amplitudes_function(self, state_amplitudes, normal_mode_1, normal_mode_2):
        """Helper method for wavefunction returning wavefunction amplitudes
        that can be overridden in the case of a global excitation cutoff"""
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
