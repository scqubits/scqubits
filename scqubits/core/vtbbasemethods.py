import itertools
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial, reduce
from typing import Callable, List, Tuple, Optional

import numpy as np
from numpy import ndarray
from numpy.linalg import matrix_power
from scipy.linalg import LinAlgError, expm, inv, eigh
from scipy.optimize import minimize

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core import discretization, storage, descriptors
from scqubits.core.discretization import Grid1d
from scqubits.core.hashing import generate_next_vector, reflect_vectors
from scqubits.core.operators import annihilation, identity_wrap
from scqubits.utils.cpu_switch import get_map_method
from scqubits.utils.spectrum_utils import (order_eigensystem, solve_generalized_eigenvalue_problem_with_QZ,
                                           standardize_phases)


class VTBBaseMethods(ABC):
    r"""Variational Tight Binding

    This class provides the methods necessary to construct operators and calculate spectra using tight-binding
    states, which in some cases more closely approximate the low-energy eigenstates of
    superconducting circuit Hamiltonians as compared to the charge basis.
    This class assumes that the potential is of the form

    .. math::

        U=-EJ[1]\cos(\phi_1)-EJ[2]\cos(\phi_2)-...-EJ[N]\cos(bc[1]\phi_1+bc[2]\phi_2+...-2\pi f),

    where the array :math:`bc` denotes the coefficients of terms in the stitching term.
    Extension of this module to circuits that include inductors is possible and is implemented for the
    zero-pi qubit in zero_pi_vtb.py.

    To implement a new qubit class using tight-binding states, the user's class
    must inherit VTBBaseMethods and also define the methods build_capacitance_matrix(),
    build_EC_matrix(), find_minima(), which contain the qubit specific information defining the capacitance matrix,
    the charging energy matrix, and a method to find all inequivalent minima, respectively.
    See current_mirror_vtb.py, flux_qubit_vtb.py and zero_pi_vtb.py for examples.

    Parameters
    ----------
    num_exc: int
        number of excitations kept in each mode
    maximum_periodic_vector_length: int
        Maximum Manhattan length of a periodic continuation vector. This should be varied to ensure convergence.
    number_degrees_freedom: int
        number of degrees of freedom of the circuit
    number_periodic_degrees_freedom: int
        number of periodic degrees of freedom
    number_junctions: int
        number of junctions in the loop
    retained_unit_cell_displacement_vectors: dict
        dictionary of the relevant periodic continuation vectors for each minima pair. It can
        be passed here to avoid constructing it later, a costly calculation
    harmonic_length_optimization: bool
        bool denoting whether or not to optimize the harmonic length
    optimize_all_minima: bool
        bool only relevant in the case of squeezing (see class VariationalTightBindingSqueezing) denoting whether or
        not to optimize the harmonic lengths in all minima
    quiet: int
        flag whether or not to print out information regarding completion of intermediate tasks
    grid: Grid1d
        grid for wavefunction plotting that will be used for extended d.o.f.
    displacement_vector_cutoff: float
        criteria for retaining periodic continuation vectors. If the overlap of two multidimensional Gaussian
        wavefunctions is less than this, the periodic continuation vector is not retained.
    maximum_site_length: int
        maximum displacement allowed for each coordinate of a periodic continuation vector.
    """
    num_exc = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    maximum_periodic_vector_length = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    number_degrees_freedom = descriptors.ReadOnlyProperty()
    number_periodic_degrees_freedom = descriptors.ReadOnlyProperty()
    number_junctions = descriptors.ReadOnlyProperty()
    harmonic_length_optimization = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    optimize_all_minima = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    extended_grid = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    displacement_vector_cutoff = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    maximum_site_length = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    stitching_coefficients = descriptors.ReadOnlyProperty()

    def __init__(self,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 number_degrees_freedom: int,
                 number_periodic_degrees_freedom: int,
                 number_junctions: int,
                 retained_unit_cell_displacement_vectors: Optional[dict] = None,
                 harmonic_length_optimization: bool = False,
                 optimize_all_minima: bool = False,
                 quiet: int = 0,
                 grid: Grid1d = Grid1d(-6 * np.pi, 6 * np.pi, 200),
                 displacement_vector_cutoff: float = 1e-15,
                 maximum_site_length: int = 2,
                 ) -> None:
        self.num_exc = num_exc
        self.maximum_periodic_vector_length = maximum_periodic_vector_length
        self._number_degrees_freedom = number_degrees_freedom
        self._number_periodic_degrees_freedom = number_periodic_degrees_freedom
        self._number_junctions = number_junctions
        self.retained_unit_cell_displacement_vectors = retained_unit_cell_displacement_vectors
        self.harmonic_length_optimization = harmonic_length_optimization
        self.optimize_all_minima = optimize_all_minima
        self.quiet = quiet
        self.extended_grid = grid
        self.displacement_vector_cutoff = displacement_vector_cutoff
        self.maximum_site_length = maximum_site_length
        self.periodic_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 100)
        self.optimized_lengths = None
        self._evec_dtype = np.complex_

    @property
    def number_extended_degrees_freedom(self):
        return self.number_degrees_freedom - self.number_periodic_degrees_freedom

    @abstractmethod
    def EC_matrix(self) -> ndarray:
        pass

    @abstractmethod
    def capacitance_matrix(self) -> ndarray:
        pass

    @abstractmethod
    def find_minima(self) -> ndarray:
        pass

    @abstractmethod
    def vtb_potential(self, phi_array: ndarray) -> float:
        pass

    @property
    @abstractmethod
    def EJlist(self):
        pass

    @property
    @abstractmethod
    def nglist(self):
        pass

    def gamma_matrix(self, minimum_index: int = 0) -> ndarray:
        """Returns linearized potential matrix

        Note that we must divide by Phi_0^2 since Ej/Phi_0^2 = 1/Lj,
        or one over the effective impedance of the junction.

        We are imagining an arbitrary loop of JJs where we have
        changed variables to the difference variables, so that
        each junction is a function of just one variable, except for
        the last junction, which is a function of all of the variables

        Parameters
        ----------
        minimum_index: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom
        minimum_location = self.sorted_minima[minimum_index]
        Phi0 = 1. / (2. * 1.)  # units where e_charge = 1
        diagonal_elements = np.diag(self.EJlist[0:dim] * np.cos(minimum_location))
        stitching_term_sum = np.sum(self.stitching_coefficients * minimum_location)
        gamma_matrix = (self.EJlist[-1] * np.cos(stitching_term_sum + 2 * np.pi * self.flux)
                        * np.outer(self.stitching_coefficients, self.stitching_coefficients)
                        + diagonal_elements) / Phi0**2
        return gamma_matrix

    def eigensystem_normal_modes(self, minimum_index: int = 0) -> (ndarray, ndarray):
        """Returns squared normal mode frequencies, matrix of eigenvectors

        Parameters
        ----------
        minimum_index: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray, ndarray
        """
        omega_squared, normal_mode_eigenvectors = eigh(self.gamma_matrix(minimum_index), b=self.capacitance_matrix())
        return omega_squared, normal_mode_eigenvectors

    def compare_harmonic_lengths_with_minima_separations(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            ratio of harmonic lengths to minima separations, providing a measure of the validity of tight binding.
            If any of the values in the returned array exceed unity, then the wavefunctions are relatively spread out
            as compared to the minima separations
        """
        if not self.retained_unit_cell_displacement_vectors:
            self.find_relevant_periodic_continuation_vectors()
        sorted_minima_dict = self.sorted_minima
        all_minima_index_pairs = itertools.combinations_with_replacement(sorted_minima_dict.items(), 2)
        return np.array(list(map(self._find_closest_periodic_minimum, all_minima_index_pairs)))

    def _find_closest_periodic_minimum(self, minima_index_pair: Tuple) -> float:
        """Helper function comparing minima separation for given minima pair"""
        return self._max_localization_ratio_for_minima_pair(minima_index_pair, 0)

    def _max_localization_ratio_for_minima_pair(self, minima_index_pair: Tuple, Xi_minimum_index_arg: int) -> float:
        """Helper function comparing minima separation for given minima pair, along with the specification
        that we would like to use the Xi matrix as defined for the minimum indexed by `Xi_arg`"""
        (m, minima_m), (p, minima_p) = minima_index_pair
        retained_unit_cell_displacement_vectors = self.retained_unit_cell_displacement_vectors[(m, p)]
        if retained_unit_cell_displacement_vectors is None or np.allclose(retained_unit_cell_displacement_vectors,
                                                                          [np.zeros(self.number_degrees_freedom)]):
            return 0.0
        Xi_inv = inv(self.Xi_matrix(minimum_index=Xi_minimum_index_arg))
        delta_inv = Xi_inv.T @ Xi_inv
        if m == p:  # Do not include equivalent minima in the same unit cell
            retained_unit_cell_displacement_vectors = np.array([vec for vec in retained_unit_cell_displacement_vectors
                                                                if not np.allclose(vec, np.zeros_like(vec))])
        displacement_vectors = 2.0 * np.pi * retained_unit_cell_displacement_vectors + (minima_p - minima_m)
        minima_distances = np.linalg.norm(displacement_vectors, axis=1)
        minima_unit_vectors = displacement_vectors / np.tile(minima_distances, (self.number_degrees_freedom, 1)).T
        harmonic_lengths = np.array([(unit_vec @ delta_inv @ unit_vec)**(-1/2) for unit_vec in minima_unit_vectors])
        return np.max(3.0 * harmonic_lengths / minima_distances / 2.0)

    def Xi_matrix(self, minimum_index: int = 0) -> ndarray:
        """ Returns Xi matrix of the normal mode eigenvectors normalized to encode the harmonic length.
        This matrix simultaneously diagonalizes the capacitance and effective inductance matrices.

        Parameters
        ----------
        minimum_index: int
            integer specifying which minimum to linearize around, 0<=minimum<= total number of minima

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom
        sorted_minima_dict = self.sorted_minima
        if self.optimized_lengths is None or not self.harmonic_length_optimization:
            self.optimized_lengths = np.ones((len(sorted_minima_dict), self.number_degrees_freedom))
        omega_squared, eigenvectors = self.eigensystem_normal_modes(minimum_index)
        # We introduce a normalization such that \Xi^T C \Xi = \Omega^{-1}/Z0
        Z0 = 1. / (2. * 1.)**2  # units where e_charge = 1
        return eigenvectors * np.tile(self.optimized_lengths[minimum_index] * omega_squared ** (-1 / 4),
                                      (dim, 1)) / np.sqrt(Z0)

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
        return identity_wrap(np.array([annihilation(self.num_exc + 1, dtype=np.complex_)]),
                             np.array([mu]), identity_operator_list, sparse=False)

    def _a_operator_array(self) -> ndarray:
        """Helper method to return a list of annihilation operator matrices for each mode"""
        return np.array([self.a_operator(i) for i in range(self.number_degrees_freedom)])

    def find_relevant_periodic_continuation_vectors(self, num_cpus: int = 1) -> None:
        """Constructs a dictionary of the relevant periodic continuation vectors for each pair of minima.

        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        """
        Xi_inv = inv(self.Xi_matrix())
        sorted_minima_dict = self.sorted_minima
        number_of_minima = len(sorted_minima_dict)
        retained_unit_cell_displacement_vectors = {}
        all_minima_index_pairs = itertools.combinations(sorted_minima_dict.items(), 2)
        retained_unit_cell_displacement_vectors[(0, 0)] = self._filter_for_minima_pair(np.zeros_like(sorted_minima_dict[0]),
                                                                                       Xi_inv, num_cpus)
        for ((m, minima_m), (p, minima_p)) in all_minima_index_pairs:
            minima_diff = Xi_inv @ (minima_p - minima_m)
            retained_unit_cell_displacement_vectors[(m, p)] = self._filter_for_minima_pair(minima_diff, Xi_inv,
                                                                                           num_cpus)
        for m in range(1, number_of_minima):
            retained_unit_cell_displacement_vectors[(m, m)] = retained_unit_cell_displacement_vectors[(0, 0)]
        self.retained_unit_cell_displacement_vectors = retained_unit_cell_displacement_vectors

    def _filter_for_minima_pair(self, minima_diff: ndarray, Xi_inv: ndarray, num_cpus: int) -> ndarray:
        """Given a minima pair, generate and then filter the periodic continuation vectors"""
        target_map = get_map_method(num_cpus)
        dim_extended = self.number_extended_degrees_freedom
        periodic_vector_lengths = np.array([i for i in range(1, self.maximum_periodic_vector_length + 1)])
        filter_function = partial(self._filter_periodic_vectors, minima_diff, Xi_inv)
        filtered_vectors = list(target_map(filter_function, periodic_vector_lengths))
        zero_vec = np.zeros(self.number_periodic_degrees_freedom)
        if self._filter_displacement_vectors(minima_diff, Xi_inv, zero_vec):
            filtered_vectors.append(np.concatenate((np.zeros(dim_extended, dtype=int), zero_vec)))
        return self._stack_filtered_vectors(filtered_vectors)

    @staticmethod
    def _stack_filtered_vectors(filtered_vectors: List) -> Optional[ndarray]:
        """Helper function for stacking together periodic continuation vectors of different Manhattan lengths"""
        filtered_vectors = list(filter(lambda x: len(x) != 0, filtered_vectors))
        if filtered_vectors:
            return np.vstack(filtered_vectors)
        else:
            return None

    def _filter_periodic_vectors(self, minima_diff: ndarray, Xi_inv: ndarray,
                                 periodic_vector_length: int) -> ndarray:
        """Helper function that generates and filters periodic vectors of a given Manhattan length"""
        sites = self.number_periodic_degrees_freedom
        filtered_vectors = []
        prev_vec = np.zeros(sites, dtype=int)
        prev_vec[0] = periodic_vector_length
        if periodic_vector_length <= self.maximum_site_length:
            self._filter_reflected_vectors(minima_diff, Xi_inv, prev_vec, filtered_vectors)
        while prev_vec[-1] != periodic_vector_length:
            next_vec = generate_next_vector(prev_vec, periodic_vector_length)
            if len(np.argwhere(next_vec > self.maximum_site_length)) == 0:
                self._filter_reflected_vectors(minima_diff, Xi_inv, next_vec, filtered_vectors)
            prev_vec = next_vec
        return np.array(filtered_vectors)

    def _filter_reflected_vectors(self, minima_diff: ndarray, Xi_inv: ndarray,
                                  vec: ndarray, filtered_vectors: List) -> None:
        """Helper function where given a specific vector, generate all possible reflections and filter those"""
        dim_extended = self.number_extended_degrees_freedom
        reflected_vectors = reflect_vectors(vec)
        filter_function = partial(self._filter_displacement_vectors, minima_diff, Xi_inv)
        new_vectors = filter(filter_function, reflected_vectors)
        for filtered_vec in new_vectors:
            filtered_vectors.append(np.concatenate((np.zeros(dim_extended, dtype=int), filtered_vec)))

    def _filter_displacement_vectors(self, minima_diff: ndarray, Xi_inv: ndarray, unit_cell_vector: ndarray) -> bool:
        """Helper function that does the filtering. Matrix elements are suppressed by a
        gaussian exponential factor, and we filter those that are suppressed below a cutoff.
        Assumption is that extended degrees of freedom precede the periodic d.o.f.
        """
        displacement_vector = 2.0 * np.pi * np.concatenate((np.zeros(self.number_extended_degrees_freedom),
                                                            unit_cell_vector))
        gaussian_overlap_argument = Xi_inv @ displacement_vector + minima_diff
        gaussian_overlap = np.exp(-0.25*np.dot(gaussian_overlap_argument, gaussian_overlap_argument))
        return gaussian_overlap > self.displacement_vector_cutoff

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
        return int(len(self.sorted_minima) * self.number_states_per_minimum())

    def _premultiplied_a_a_dagger(self, a_operator_array: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """Helper method for premultiplying creation and annihilation operators (multiplications are expensive)"""
        return (a_operator_array, a_operator_array @ a_operator_array,
                np.transpose(a_operator_array, (0, 2, 1)) @ a_operator_array)  # a, a * a, a^{\dagger} * a

    def _single_exp_i_phi_j_operator(self, j: int, Xi: ndarray, a_operator_array: ndarray) -> ndarray:
        r"""Returns operator :math:`\exp(i\phi_{j})`. If `j` specifies the stitching term, which is
        assumed to be the last junction, then that is constructed based on the stitching coefficients."""
        if j == self.number_junctions - 1:
            exp_i_phi_j_a = expm(1j * np.sum(self.stitching_coefficients @ Xi
                                             * np.transpose(a_operator_array, (1, 2, 0)), axis=2) / np.sqrt(2.0))
            BCH_factor = self._BCH_factor_for_potential_stitching(Xi)
        else:
            exp_i_phi_j_a = expm(1j * np.sum(Xi[j] * np.transpose(a_operator_array, (1, 2, 0)), axis=2) / np.sqrt(2.0))
            BCH_factor = np.exp(-0.25 * Xi[j] @ Xi[j])
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a.T
        return BCH_factor * exp_i_phi_j_a_dagger_component @ exp_i_phi_j_a

    def _all_exp_i_phi_j_operators(self, Xi: ndarray, a_operator_array: ndarray) -> ndarray:
        """Helper method for building all potential operators"""
        return np.array([self._single_exp_i_phi_j_operator(j, Xi, a_operator_array)
                         for j in range(self.number_junctions)])

    def _general_translation_operators(self, Xi_inv: ndarray,
                                       a_operator_array: ndarray) -> Tuple[ndarray, ndarray]:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. The resulting matrices yield a 2pi translation
        in each degree of freedom, so that any translation can be built from these by an appropriate
        call to np.matrix_power"""
        dim = self.number_degrees_freedom
        exp_a_list = np.array([expm(np.sum(2.0 * np.pi * Xi_inv.T[i] * np.transpose(a_operator_array, (1, 2, 0))
                                           / np.sqrt(2.0), axis=2)) for i in range(dim)])
        exp_a_dagger_list = np.array([expm(np.sum(2.0 * np.pi * Xi_inv.T[i] * a_operator_array.T / np.sqrt(2.0),
                                                  axis=2)) for i in range(dim)])
        return exp_a_list, exp_a_dagger_list

    def _minima_dependent_translation_operators(self, minima_diff: ndarray, Xi_inv: ndarray,
                                                a_operator_array: ndarray) -> Tuple[ndarray, ndarray]:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. This part of the translation operator accounts
        for the differing location of minima within a single unit cell."""
        exp_a_minima_difference = expm(np.sum((-minima_diff @ Xi_inv.T) * np.transpose(a_operator_array, (1, 2, 0))
                                              / np.sqrt(2.0), axis=2))
        exp_a_dagger_minima_difference = expm(np.sum((minima_diff @ Xi_inv.T) * a_operator_array.T
                                              / np.sqrt(2.0), axis=2))
        return exp_a_minima_difference, exp_a_dagger_minima_difference

    def _local_translation_operators(self, exp_a_list: Tuple[ndarray, ndarray],
                                     exp_minima_difference: Tuple[ndarray, ndarray],
                                     unit_cell_vector: ndarray) -> Tuple[ndarray, ndarray]:
        """Helper method that builds translation operators using matrix_power and the pre-exponentiated
        translation operators that define 2pi translations."""
        dim = self.number_degrees_freedom
        exp_a_list, exp_a_dagger_list = exp_a_list
        exp_a_minima_difference, exp_a_dagger_minima_difference = exp_minima_difference
        # Note: stacks of object matrices are not currently supported by numpy: must use listcomp
        individual_a_dagger_op = np.array([matrix_power(exp_a_dagger_list[j], int(unit_cell_vector[j])) for j in range(dim)])
        individual_a_op = np.array([matrix_power(exp_a_list[j], -int(unit_cell_vector[j])) for j in range(dim)])
        translation_op_a_dagger = reduce((lambda x, y: x @ y), individual_a_dagger_op) @ exp_a_dagger_minima_difference
        translation_op_a = reduce((lambda x, y: x @ y), individual_a_op) @ exp_a_minima_difference
        return translation_op_a_dagger, translation_op_a

    def _exp_product_coefficient(self, delta_phi: ndarray, Xi_inv: ndarray) -> ndarray:
        """Returns overall multiplicative factor, including offset charge and Gaussian suppression BCH factor
        from the periodic continuation (translation) operators"""
        delta_phi_rotated = Xi_inv @ delta_phi
        return np.exp(-1j * self.nglist @ delta_phi) * np.exp(-0.25 * delta_phi_rotated @ delta_phi_rotated)

    def _BCH_factor_for_potential_stitching(self, Xi: ndarray) -> ndarray:
        """BCH factor obtained from the last potential operator"""
        return np.exp(-0.25 * self.stitching_coefficients @ Xi @ Xi.T @ self.stitching_coefficients)

    def n_operator(self, j: int = 0) -> ndarray:
        Xi_inv = inv(self.Xi_matrix())
        a_operator_array = self._a_operator_array()
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        charge_function = partial(self._local_charge_operator, j, premultiplied_a_a_dagger, Xi_inv)
        return self._periodic_continuation(charge_function)

    def phi_operator(self, j: int = 0) -> ndarray:
        Xi = self.Xi_matrix()
        a_operator_array = self._a_operator_array()
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        phi_function = partial(self._local_phi_operator, j, premultiplied_a_a_dagger, Xi)
        return self._periodic_continuation(phi_function)

    def exp_i_phi_operator(self, j: int = 0) -> ndarray:
        exp_i_phi_j = self._all_exp_i_phi_j_operators(self.Xi_matrix(), self._a_operator_array())
        exp_i_phi_j_function = partial(self._local_exp_i_phi_operator, j, exp_i_phi_j)
        return self._periodic_continuation(exp_i_phi_j_function)

    def hamiltonian(self) -> ndarray:
        return self.transfer_matrix()

    def kinetic_matrix(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        Xi_inv = inv(self.Xi_matrix())
        a_operator_array = self._a_operator_array()
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        EC_mat_t = Xi_inv @ self.EC_matrix() @ Xi_inv.T
        kinetic_function = partial(self._local_kinetic, premultiplied_a_a_dagger, EC_mat_t, Xi_inv)
        return self._periodic_continuation(kinetic_function)

    def potential_matrix(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        Xi = self.Xi_matrix()
        a_operator_array = self._a_operator_array()
        exp_i_phi_j = self._all_exp_i_phi_j_operators(Xi, a_operator_array)
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        potential_function = partial(self._local_potential, exp_i_phi_j, premultiplied_a_a_dagger, Xi)
        return self._periodic_continuation(potential_function)

    def transfer_matrix(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the transfer matrix
        """
        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        exp_i_phi_j = self._all_exp_i_phi_j_operators(Xi, a_operator_array)
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        EC_mat_t = Xi_inv @ self.EC_matrix() @ Xi_inv.T
        transfer_matrix_function = partial(self._local_kinetic_plus_potential, exp_i_phi_j,
                                           premultiplied_a_a_dagger, EC_mat_t, Xi, Xi_inv)
        return self._periodic_continuation(transfer_matrix_function)

    def inner_product_matrix(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            Returns the inner product matrix
        """
        return self._periodic_continuation(lambda x, y, z: self.identity())

    def _local_charge_operator(self, j: int, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                               Xi_inv: ndarray, displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        constant_coefficient = -0.5 * 1j * (Xi_inv.T @ Xi_inv @ (displacement_vector + minima_p - minima_m))[j]
        return (-(1j / np.sqrt(2.0)) * np.sum(Xi_inv.T[j] * (np.transpose(a, (1, 2, 0)) - a.T), axis=2)
                + constant_coefficient * self.identity())

    def _local_phi_operator(self, j: int, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                            Xi: ndarray, displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        constant_coefficient = 0.5 * (displacement_vector + (minima_m + minima_p))
        return ((1.0 / np.sqrt(2.0)) * np.sum(Xi[j] * (np.transpose(a, (1, 2, 0)) + a.T), axis=2)
                + constant_coefficient[j] * self.identity())

    def _exp_i_phi_j_with_phi_bar(self, exp_i_phi_j: ndarray, phi_bar: ndarray) -> Tuple[ndarray, ndarray]:
        """Returns exp_i_phi_j operators including the local contribution of phi_bar"""
        exp_i_phi_j_phi_bar = np.transpose(exp_i_phi_j[:-1], (1, 2, 0)) * np.exp(1j * phi_bar)
        exp_i_stitching_phi_j_phi_bar = (exp_i_phi_j[-1] * np.exp(1j * 2.0 * np.pi * self.flux)
                                         * np.exp(1j * self.stitching_coefficients @ phi_bar))
        return np.transpose(exp_i_phi_j_phi_bar, (2, 0, 1)), exp_i_stitching_phi_j_phi_bar

    def _local_exp_i_phi_operator(self, j: int, exp_i_phi_j: ndarray, displacement_vector: ndarray,
                                  minima_m: ndarray, minima_p: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (displacement_vector + (minima_m + minima_p))
        exp_i_phi_j_phi_bar, exp_i_stitching_phi_j_phi_bar = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        if j == dim:
            return exp_i_stitching_phi_j_phi_bar
        else:
            return exp_i_phi_j_phi_bar[j]

    def _local_kinetic(self, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                       EC_mat_t: ndarray, Xi_inv: ndarray, displacement_vector: ndarray,
                       minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Calculate the local kinetic contribution to the transfer matrix given two
        minima and a periodic continuation vector `displacement_vector`"""
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        delta_phi = displacement_vector + minima_p - minima_m
        delta_phi_rotated = Xi_inv @ delta_phi
        kinetic_matrix = np.sum(np.diag(EC_mat_t)
                                * (- 0.5 * 4 * np.transpose(a_a, (1, 2, 0)) - 0.5 * 4 * a_a.T
                                   + 0.5 * 8 * np.transpose(a_dagger_a, (1, 2, 0))
                                   - 4 * (np.transpose(a, (1, 2, 0)) - a.T) * delta_phi_rotated / np.sqrt(2.0)), axis=2)
        identity_coefficient = (0.5 * 4 * np.trace(EC_mat_t)
                                - 0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated)
        kinetic_matrix = kinetic_matrix + identity_coefficient*self.identity()
        return kinetic_matrix

    def _local_potential(self, exp_i_phi_j: ndarray, premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                         Xi: ndarray, displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Calculate the local potential contribution to the transfer matrix given two
        minima and a periodic continuation vector `displacement_vector`"""
        phi_bar = 0.5 * (displacement_vector + (minima_m + minima_p))
        exp_i_phi_j_phi_bar, exp_i_stitching_phi_j_phi_bar = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        potential_matrix = np.sum(-0.5 * self.EJlist[: -1]
                                  * (np.transpose(exp_i_phi_j_phi_bar, (1, 2, 0))
                                     + np.transpose(exp_i_phi_j_phi_bar, (1, 2, 0)).conjugate()), axis=2)
        potential_matrix = potential_matrix - 0.5 * self.EJlist[-1] * (exp_i_stitching_phi_j_phi_bar
                                                                       + exp_i_stitching_phi_j_phi_bar.conjugate())
        potential_matrix = potential_matrix + np.sum(self.EJlist) * self.identity()
        return potential_matrix

    def _local_kinetic_plus_potential(self, exp_i_phi_j: ndarray,
                                      premultiplied_a_a_dagger: Tuple[ndarray, ndarray, ndarray],
                                      EC_mat_t: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                      displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Calculate the local contribution to the transfer matrix given two
        minima and a periodic continuation vector `displacement_vector`"""
        return (self._local_kinetic(premultiplied_a_a_dagger, EC_mat_t, Xi_inv, displacement_vector, minima_m, minima_p)
                + self._local_potential(exp_i_phi_j, premultiplied_a_a_dagger, Xi, displacement_vector, minima_m, minima_p))

    def _periodic_continuation(self, func: Callable) -> ndarray:
        """This function is the meat of the VariationalTightBinding method. Any operator whose matrix
        elements we want (the transfer matrix and inner product matrix are obvious examples)
        can be passed to this function, and the matrix elements of that operator
        will be returned.

        Parameters
        ----------
        func: method
            function that takes three arguments (displacement_vector, minima_m, minima_p) and returns the
            relevant operator with dimension NxN, where N is the number of states
            displaced into each minimum. For instance to find the inner product matrix,
            we use the function self._inner_product_operator(displacement_vector, minima_m, minima_p) -> self.identity

        Returns
        -------
        ndarray
        """
        if not self.retained_unit_cell_displacement_vectors:
            self.find_relevant_periodic_continuation_vectors()
        Xi_inv = inv(self.Xi_matrix())
        a_operator_array = self._a_operator_array()
        exp_a_list = self._general_translation_operators(Xi_inv, a_operator_array)
        sorted_minima_dict = self.sorted_minima
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((self.hilbertdim(), self.hilbertdim()), dtype=np.complex128)
        all_minima_index_pairs = itertools.combinations_with_replacement(sorted_minima_dict.items(), 2)
        for ((m, minima_m), (p, minima_p)) in all_minima_index_pairs:
            matrix_element = self._periodic_continuation_for_minima_pair(minima_m, minima_p,
                                                                         self.retained_unit_cell_displacement_vectors[(m, p)],
                                                                         func, exp_a_list, Xi_inv, a_operator_array)
            operator_matrix[m * num_states_min: (m + 1) * num_states_min,
                            p * num_states_min: (p + 1) * num_states_min] += matrix_element
        operator_matrix = self._populate_hermitian_matrix(operator_matrix)
        return operator_matrix

    def _periodic_continuation_for_minima_pair(self, minima_m: ndarray, minima_p: ndarray, retained_unit_cell_displacement_vectors: ndarray,
                                               func: Callable, exp_a_list: Tuple[ndarray, ndarray], Xi_inv: ndarray,
                                               a_operator_array: ndarray) -> ndarray:
        """Helper method for performing the periodic continuation calculation given a minima pair."""
        if retained_unit_cell_displacement_vectors is not None:
            minima_diff = minima_p - minima_m
            exp_minima_difference = self._minima_dependent_translation_operators(minima_diff, Xi_inv,
                                                                                 a_operator_array)
            return np.sum([self._displacement_vector_contribution(unit_cell_vector, func, minima_m, minima_p,
                                                                  exp_a_list, exp_minima_difference, Xi_inv)
                           for unit_cell_vector in retained_unit_cell_displacement_vectors], axis=0)
        else:
            return np.zeros((self.number_states_per_minimum(), self.number_states_per_minimum()), dtype=np.complex_)

    def _displacement_vector_contribution(self, unit_cell_vector: ndarray, func: Callable, minima_m: ndarray, minima_p: ndarray,
                                          exp_a_list: Tuple[ndarray, ndarray], exp_minima_difference: Tuple[ndarray, ndarray],
                                          Xi_inv: ndarray) -> ndarray:
        """Helper method for calculating the contribution of a specific periodic continuation vector `displacement_vector`"""
        displacement_vector = 2.0 * np.pi * np.array(unit_cell_vector)
        exp_prod_coefficient = self._exp_product_coefficient(displacement_vector + minima_p - minima_m, Xi_inv)
        exp_a_dagger, exp_a = self._local_translation_operators(exp_a_list, exp_minima_difference, unit_cell_vector)
        matrix_element = exp_prod_coefficient * func(displacement_vector, minima_m, minima_p)
        return exp_a_dagger @ matrix_element @ exp_a

    def _populate_hermitian_matrix(self, mat: ndarray) -> ndarray:
        """Return a fully Hermitian matrix, assuming that the input matrix has been
        populated with the upper right blocks"""
        sorted_minima_dict = self.sorted_minima
        num_states_min = self.number_states_per_minimum()
        for m, _ in sorted_minima_dict.items():
            for p in range(m + 1, len(sorted_minima_dict)):
                matrix_element = mat[m * num_states_min: (m + 1) * num_states_min,
                                     p * num_states_min: (p + 1) * num_states_min]
                mat[p * num_states_min: (p + 1) * num_states_min,
                    m * num_states_min: (m + 1) * num_states_min] += matrix_element.conjugate().T
        return mat

    def _transfer_matrix_and_inner_product(self) -> Tuple[ndarray, ndarray]:
        """Helper method called by _esys_calc and _evals_calc that returns the transfer matrix and inner product
        matrix but warns the user if the system is in a regime where tight-binding has questionable validity."""
        self.find_relevant_periodic_continuation_vectors()
        if self.harmonic_length_optimization:
            self.optimize_Xi_variational()
        harmonic_length_minima_comparison = self.compare_harmonic_lengths_with_minima_separations()
        if np.max(harmonic_length_minima_comparison) > 1.0 and not self.quiet:
            print("Warning: large harmonic length compared to minima separation "
                  "(largest is 3*l/(d/2) = {ratio})".format(ratio=np.max(harmonic_length_minima_comparison)))
        transfer_matrix = self.transfer_matrix()
        inner_product_matrix = self.inner_product_matrix()
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
        minima_list = self.find_minima()
        value_of_potential = np.array([self.vtb_potential(minima) for minima in minima_list])
        sorted_indices = np.argsort(value_of_potential)
        return value_of_potential[sorted_indices], minima_list[sorted_indices, :]

    @property
    def sorted_minima(self) -> OrderedDict:
        """
        Return sorted array of the minima locations

        Returns
        -------
        ndarray
        """
        _, sorted_minima_list = self._sorted_potential_values_and_minima()
        return OrderedDict({m: minimum for m, minimum in enumerate(sorted_minima_list)})

    def _normalize_minimum_inside_pi_range(self, minimum: ndarray) -> ndarray:
        """Helper method for defining the unit cell from -pi to pi rather than the less symmetric 0 to 2pi"""
        num_extended = self.number_extended_degrees_freedom
        extended_coordinates = minimum[0: num_extended]
        periodic_coordinates = np.mod(minimum, 2 * np.pi * np.ones_like(minimum))[num_extended:]
        periodic_coordinates = np.array([elem - 2 * np.pi if elem > np.pi else elem for elem in periodic_coordinates])
        return np.concatenate((extended_coordinates, periodic_coordinates))

    def _check_if_new_minima(self, new_minima: ndarray, minima_list: List) -> bool:
        """Helper method for find_minima, checking if new_minima is already represented in minima_list. If so,
        _check_if_new_minima returns False.
        """
        num_extended = self.number_extended_degrees_freedom
        for minimum in minima_list:
            extended_coordinates = np.array(minimum[0:num_extended] - new_minima[0:num_extended])
            periodic_coordinates = np.mod(minimum - new_minima, 2*np.pi*np.ones_like(minimum))[num_extended:]
            diff_array_bool_extended = [True if np.allclose(elem, 0.0, atol=1e-3) else False
                                        for elem in extended_coordinates]
            diff_array_bool_periodic = [True if (np.allclose(elem, 0.0, atol=1e-3)
                                                 or np.allclose(elem, 2 * np.pi, atol=1e-3))
                                        else False for elem in periodic_coordinates]
            if np.all(diff_array_bool_extended) and np.all(diff_array_bool_periodic):
                return False
        return True

    def _filter_repeated_minima(self, minima_list: List) -> List:
        """Eliminate repeated minima contained in minima_list"""
        filtered_minima_list = [minima_list[0]]
        for minima in minima_list:
            if self._check_if_new_minima(minima, filtered_minima_list):
                filtered_minima_list.append(minima)
        return filtered_minima_list

    def optimize_Xi_variational(self) -> None:
        """Optimize the Xi matrix by adjusting the harmonic lengths of the ground state to minimize its energy.
        For tight-binding without squeezing, this is only done for the ansatz ground state wavefunction
        localized in the global minimum."""
        sorted_minima_dict = self.sorted_minima
        self.optimized_lengths = np.ones((len(sorted_minima_dict), self.number_degrees_freedom))
        self._optimize_Xi_variational(0, sorted_minima_dict[0])
        for m, _ in sorted_minima_dict.items():
            self.optimized_lengths[m] = self.optimized_lengths[0]

    def _optimize_Xi_variational(self, minimum: int = 0, minimum_location: ndarray = None) -> None:
        """Perform the harmonic length optimization for a h.o. ground state wavefunction localized in a given minimum"""
        default_Xi = self.Xi_matrix(minimum)
        EC_mat = self.EC_matrix()
        if not self.retained_unit_cell_displacement_vectors:
            self.find_relevant_periodic_continuation_vectors()
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
        displacement_vectors = 2.0*np.pi*self.retained_unit_cell_displacement_vectors[(minimum, minimum)]
        gradient_transfer = [np.sum([self._exp_product_coefficient(displacement_vector, Xi_inv)
                                    * self._gradient_one_state_local_transfer(exp_i_phi_j, EC_mat_t, Xi, Xi_inv,
                                                                              displacement_vector, minimum_location,
                                                                              minimum_location, which_length)
                                    + self._gradient_one_state_local_inner_product(displacement_vector, Xi_inv, which_length)
                                     * self._one_state_local_transfer(exp_i_phi_j, EC_mat_t, Xi, Xi_inv, displacement_vector,
                                                                      minimum_location, minimum_location)
                                    for displacement_vector in displacement_vectors])
                             for which_length in range(self.number_degrees_freedom)]
        transfer = np.sum([self._exp_product_coefficient(displacement_vector, Xi_inv)
                           * self._one_state_local_transfer(exp_i_phi_j, EC_mat_t, Xi, Xi_inv, displacement_vector,
                                                            minimum_location, minimum_location)
                           for displacement_vector in displacement_vectors])
        gradient_inner = [np.sum([self._gradient_one_state_local_inner_product(displacement_vector, Xi_inv, which_length)
                                  for displacement_vector in displacement_vectors])
                          for which_length in range(self.number_degrees_freedom)]
        inner = np.sum([self._exp_product_coefficient(displacement_vector, Xi_inv) for displacement_vector in displacement_vectors])
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
        return np.append(exp_factors, self._BCH_factor_for_potential_stitching(Xi))

    def _gradient_one_state_local_inner_product(self, delta_phi: ndarray, Xi_inv: ndarray, which_length: int) -> float:
        """Returns gradient of the inner product matrix"""
        delta_phi_rotated = Xi_inv @ delta_phi
        return (self.optimized_lengths[0, which_length]**(-1)
                * (1j * self.nglist[which_length] * delta_phi_rotated[which_length]
                   + 0.5 * delta_phi_rotated[which_length] ** 2) * self._exp_product_coefficient(delta_phi, Xi_inv))

    @staticmethod
    def _one_state_local_kinetic(EC_mat_t: ndarray, Xi_inv: ndarray,
                                 displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray) -> float:
        """Local kinetic contribution when considering only the ground state."""
        delta_phi_rotated = Xi_inv @ (displacement_vector + minima_p - minima_m)
        return 0.5 * 4 * np.trace(EC_mat_t) - 0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated

    def _gradient_one_state_local_kinetic(self, EC_mat_t: ndarray, Xi_inv: ndarray, displacement_vector: ndarray,
                                          minima_m: ndarray, minima_p: ndarray, which_length: int) -> ndarray:
        """Returns gradient of the kinetic matrix"""
        delta_phi_rotated = Xi_inv @ (displacement_vector + minima_p - minima_m)
        return (-4.0*self.optimized_lengths[0, which_length]**(-1)
                * (EC_mat_t[which_length, which_length] - (delta_phi_rotated @ EC_mat_t)[which_length]
                   * delta_phi_rotated[which_length]))

    def _gradient_one_state_local_potential(self, exp_i_phi_j: ndarray, displacement_vector: ndarray, minima_m: ndarray,
                                            minima_p: ndarray, Xi: ndarray, which_length: int) -> ndarray:
        """Returns gradient of the potential matrix"""
        phi_bar = 0.5 * (displacement_vector + (minima_m + minima_p))
        exp_i_phi_j_phi_bar, exp_i_stitching_phi_j_phi_bar = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        potential_gradient = np.sum([0.25 * self.EJlist[junction]
                                     * Xi[junction, which_length] * Xi.T[which_length, junction]
                                     * self.optimized_lengths[0, which_length]**(-1)
                                     * (exp_i_phi_j_phi_bar[junction] + exp_i_phi_j_phi_bar[junction].conjugate())
                                     for junction in range(self.number_junctions - 1)])
        potential_gradient += (0.25 * self.EJlist[-1] * self.optimized_lengths[0, which_length] ** (-1)
                               * (self.stitching_coefficients @ Xi[:, which_length]) ** 2
                               * (exp_i_stitching_phi_j_phi_bar + exp_i_stitching_phi_j_phi_bar.conjugate()))
        return potential_gradient

    def _gradient_one_state_local_transfer(self, exp_i_phi_j: ndarray, EC_mat_t: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                           displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray,
                                           which_length: int) -> ndarray:
        """Returns gradient of the transfer matrix"""
        return (self._gradient_one_state_local_potential(exp_i_phi_j, displacement_vector, minima_m,
                                                         minima_p, Xi, which_length)
                + self._gradient_one_state_local_kinetic(EC_mat_t, Xi_inv, displacement_vector, minima_m,
                                                         minima_p, which_length))

    def _one_state_local_potential(self, exp_i_phi_j: ndarray, Xi: ndarray, displacement_vector: ndarray,
                                   minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Local potential contribution when considering only the ground state."""
        phi_bar = 0.5 * (displacement_vector + (minima_m + minima_p))
        exp_i_phi_j_phi_bar, exp_i_stitching_phi_j_phi_bar = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        potential = np.sum([-0.5 * self.EJlist[junction] * (exp_i_phi_j_phi_bar[junction]
                                                            + exp_i_phi_j_phi_bar[junction].conjugate())
                            for junction in range(self.number_junctions - 1)])
        potential += np.sum(self.EJlist) - 0.5 * self.EJlist[-1] * (exp_i_stitching_phi_j_phi_bar
                                                                    + exp_i_stitching_phi_j_phi_bar.conjugate())
        return potential

    def _one_state_local_transfer(self, exp_i_phi_j: ndarray, EC_mat_t: ndarray, Xi: ndarray, Xi_inv: ndarray,
                                  displacement_vector: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        """Local transfer contribution when considering only the ground state."""
        return (self._one_state_local_kinetic(EC_mat_t, Xi_inv, displacement_vector, minima_m, minima_p)
                + self._one_state_local_potential(exp_i_phi_j, Xi, displacement_vector, minima_m, minima_p))

    def _one_state_construct_transfer_and_inner(self, Xi: ndarray, Xi_inv: ndarray, minimum_location: ndarray,
                                                minimum: int, EC_mat_t: ndarray,
                                                exp_i_phi_j: ndarray) -> Tuple[ndarray, ndarray]:
        """Transfer matrix and inner product matrix when considering only the ground state."""
        retained_unit_cell_displacement_vectors = 2.0*np.pi*self.retained_unit_cell_displacement_vectors[(minimum, minimum)]
        transfer_function = partial(self._one_state_local_transfer, exp_i_phi_j, EC_mat_t, Xi, Xi_inv)
        transfer = np.sum([self._exp_product_coefficient(unit_cell_vector, Xi_inv)
                           * transfer_function(unit_cell_vector, minimum_location, minimum_location)
                           for unit_cell_vector in retained_unit_cell_displacement_vectors])
        # Note need to include 2.0*np.pi*np.array(unit_cell_vector) + minimum_location - minimum_location for completeness
        inner_product = np.sum([self._exp_product_coefficient(unit_cell_vector, Xi_inv)
                                for unit_cell_vector in retained_unit_cell_displacement_vectors])
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
        sorted_minima_dict = self.sorted_minima

        Xi = self.Xi_matrix()
        Xi_inv = inv(Xi)
        normalization = np.sqrt(np.abs(np.linalg.det(Xi))) ** (-1)

        dim_extended = self.number_extended_degrees_freedom
        dim_periodic = self.number_periodic_degrees_freedom
        phi_1_grid = self.periodic_grid
        phi_1_vec = phi_1_grid.make_linspace()
        phi_2_grid = self.periodic_grid
        phi_2_vec = phi_2_grid.make_linspace()

        if dim_extended == 1:
            phi_1_grid = self.extended_grid
            phi_1_vec = phi_1_grid.make_linspace()
        if dim_extended == 2:
            phi_2_grid = self.extended_grid
            phi_2_vec = phi_1_grid.make_linspace()

        wavefunction_amplitudes = np.zeros_like(np.outer(phi_1_vec, phi_2_vec), dtype=np.complex_).T

        for i, minimum in sorted_minima_dict.items():
            unit_cell_vectors = itertools.product(np.arange(-self.maximum_periodic_vector_length,
                                                    self.maximum_periodic_vector_length + 1), repeat=dim_periodic)
            unit_cell_vector = next(unit_cell_vectors, -1)
            while unit_cell_vector != -1:
                displacement_vector = 2.0 * np.pi * np.concatenate((np.zeros(dim_extended), unit_cell_vector))
                phi_offset = displacement_vector - minimum
                state_amplitudes = self.state_amplitudes_function(i, evecs, which)
                phi_1_with_offset = phi_1_vec + phi_offset[0]
                phi_2_with_offset = phi_2_vec + phi_offset[1]
                normal_mode_1 = np.add.outer(Xi_inv[0, 0]*phi_1_with_offset, Xi_inv[0, 1]*phi_2_with_offset)
                normal_mode_2 = np.add.outer(Xi_inv[1, 0]*phi_1_with_offset, Xi_inv[1, 1]*phi_2_with_offset)
                wavefunction_amplitudes += (self.wavefunction_amplitudes_function(state_amplitudes,
                                                                                  normal_mode_1, normal_mode_2)
                                            * normalization * np.exp(-1j * np.dot(self.nglist, phi_offset)))
                unit_cell_vector = next(unit_cell_vectors, -1)

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
        return plot.wavefunction2d(wavefunction, zero_calibrate=zero_calibrate,
                                   xlabel=r'$\phi$', ylabel=r'$\theta$', **kwargs)
