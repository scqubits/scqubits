import itertools
import warnings
from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy as sp
from numpy import ndarray
from numpy.linalg import matrix_power
from scipy.linalg import LinAlgError, eigh, expm, inv
from scipy.optimize import minimize

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core import descriptors, discretization, storage
from scqubits.core.discretization import Grid1d
from scqubits.core.hashing import generate_next_vector, reflect_vectors
from scqubits.core.operators import annihilation, identity_wrap_array
from scqubits.utils.cpu_switch import get_map_method
from scqubits.utils.spectrum_utils import (
    order_eigensystem,
    solve_generalized_eigenvalue_problem_with_QZ,
    standardize_phases,
)


class VTBBaseMethods(ABC):
    r"""Variational Tight Binding

    This class provides the methods necessary to construct
    operators and calculate spectra using tight-binding
    states, which in some cases more closely approximate the low-energy eigenstates of
    superconducting circuit Hamiltonians as compared to the charge basis.
    This class assumes that the potential is of the form

    .. math::

        U=-EJ[1]\cos(\phi_1)-EJ[2]\cos(\phi_2)-...-EJ[N]
        \cos(bc[1]\phi_1+bc[2]\phi_2+...-2\pi f),

    where the array :math:`bc` denotes the coefficients of terms in the stitching term.
    Extension of this module to circuits that include inductors
    is possible and is implemented for the
    zero-pi qubit in zero_pi_vtb.py.

    To implement a new qubit class using tight-binding states, the user's class
    must inherit VTBBaseMethods and also define the methods build_capacitance_matrix(),
    build_EC_matrix(), find_minima(), which contain the qubit
     specific information defining the capacitance matrix,
    the charging energy matrix, and a method to
     find all inequivalent minima, respectively.
    See current_mirror_vtb.py, flux_qubit_vtb.py and zero_pi_vtb.py for examples.

    Parameters
    ----------
    num_exc: int
        number of excitations kept in each mode
    maximum_unit_cell_vector_length: int
        Maximum Manhattan length of a unit cell vector.
        This should be varied to ensure convergence.
    number_degrees_freedom: int
        number of degrees of freedom of the circuit
    number_periodic_degrees_freedom: int
        number of periodic degrees of freedom
    number_junctions: int
        number of junctions in the loop
    harmonic_length_optimization: bool
        bool denoting whether or not to optimize the harmonic length
    optimize_all_minima: bool
        bool only relevant in the case of squeezing (see class
        VariationalTightBindingSqueezing) denoting whether or
        not to optimize the harmonic lengths in all minima
    quiet: int
        flag whether or not to print out information
        regarding completion of intermediate tasks
    grid: Grid1d
        grid for wavefunction plotting that will be used for extended d.o.f.
    displacement_vector_cutoff: float
        criteria for relevant unit cell vectors.
        If the overlap of two multidimensional Gaussian
        wavefunctions is less than this, the unit cell vector is not relevant.
    maximum_site_length: int
        maximum displacement allowed for each coordinate of a unit cell vector.
    """
    num_exc = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    maximum_unit_cell_vector_length = descriptors.WatchedProperty(
        int, "QUANTUMSYSTEM_UPDATE"
    )
    number_degrees_freedom = descriptors.ReadOnlyProperty(int)
    number_periodic_degrees_freedom = descriptors.ReadOnlyProperty(int)
    number_junctions = descriptors.ReadOnlyProperty(int)
    harmonic_length_optimization = descriptors.WatchedProperty(bool, "QUANTUMSYSTEM_UPDATE")
    optimize_all_minima = descriptors.WatchedProperty(bool, "QUANTUMSYSTEM_UPDATE")
    extended_grid = descriptors.WatchedProperty(Grid1d, "QUANTUMSYSTEM_UPDATE")
    displacement_vector_cutoff = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    maximum_site_length = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    stitching_coefficients = descriptors.ReadOnlyProperty(ndarray)

    def __init__(
        self,
        num_exc: int,
        maximum_unit_cell_vector_length: int,
        number_degrees_freedom: int,
        number_periodic_degrees_freedom: int,
        number_junctions: int,
        harmonic_length_optimization: bool = False,
        optimize_all_minima: bool = False,
        use_global_min_harmonic_lengths: bool = False,
        quiet: int = 0,
        grid: Grid1d = Grid1d(-6 * np.pi, 6 * np.pi, 200),
        displacement_vector_cutoff: float = 1e-15,
        maximum_site_length: int = 2,
        safe_run: bool = False,
        inner_prod_eval_tol: float = 1e-8,
    ) -> None:
        self.num_exc = num_exc
        self.maximum_unit_cell_vector_length = maximum_unit_cell_vector_length
        self._number_degrees_freedom = number_degrees_freedom
        self._number_periodic_degrees_freedom = number_periodic_degrees_freedom
        self._number_junctions = number_junctions
        self.harmonic_length_optimization = harmonic_length_optimization
        if optimize_all_minima and use_global_min_harmonic_lengths:
            raise ValueError(
                "Cannot optimize the harmonic lengths of all minima and use the "
                "harmonic lengths defined in the global minimum"
            )
        self.optimize_all_minima = optimize_all_minima
        self.use_global_min_harmonic_lengths = use_global_min_harmonic_lengths
        self.quiet = quiet
        self.extended_grid = grid
        self.displacement_vector_cutoff = displacement_vector_cutoff
        self.maximum_site_length = maximum_site_length
        self.periodic_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 100)
        self.safe_run = safe_run
        self.inner_prod_eval_tol = inner_prod_eval_tol
        self._evec_dtype = np.complex128

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
            integer specifying which minimum to linearize around,
            0<=minimum<= total number of minima

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom
        minimum_location = self.sorted_minima_dict[minimum_index]
        Phi0 = 0.5  # units where e_charge, hbar = 1; Phi0 = hbar / (2 * e)
        diagonal_elements = np.diag(self.EJlist[0:dim] * np.cos(minimum_location))
        stitching_term_sum = np.sum(self.stitching_coefficients * minimum_location)
        gamma_matrix = (
            self.EJlist[-1]
            * np.cos(stitching_term_sum + 2 * np.pi * self.flux)
            * np.outer(self.stitching_coefficients, self.stitching_coefficients)
            + diagonal_elements
        ) / Phi0 ** 2
        return gamma_matrix

    def find_invertible_submatrix(self, mat: ndarray, tol: float):
        n, m = mat.shape
        invertible_submatrix_mat = np.zeros((min(n, m), min(n, m)))
        append_counter = 0
        for i, row in enumerate(mat):
            tmp_mat = np.copy(invertible_submatrix_mat)
            tmp_mat[append_counter] = mat[i]
            rank = np.linalg.matrix_rank(tmp_mat, tol)
            if rank == append_counter + 1:
                invertible_submatrix_mat = np.copy(tmp_mat)
                append_counter += 1
            if append_counter == min(n, m):
                break
        return invertible_submatrix_mat

    def eigensystem_normal_modes(self, minimum_index: int = 0) -> (ndarray, ndarray):
        """Returns squared normal mode frequencies, matrix of eigenvectors

        Parameters
        ----------
        minimum_index: int
            integer specifying which minimum to linearize around,
            0<=minimum<= total number of minima

        Returns
        -------
        ndarray, ndarray
        """
        omega_squared, normal_mode_eigenvectors = eigh(
            self.gamma_matrix(minimum_index), b=self.capacitance_matrix()
        )
        return omega_squared, normal_mode_eigenvectors

    def compute_minimum_localization_ratio(
        self, relevant_unit_cell_vectors: Optional[dict] = None
    ) -> ndarray:
        """
        Returns
        -------
        ndarray
            minimum ratio of harmonic lengths to minima separations, providing
            a measure of the validity of tight binding.
            If the returned minimumvalue is much less than unity, then
            in certain directions the wavefunctions are relatively spread out
            as compared to the minima separations, leading to numerical stability
            issues. In this regime, the validity of tight-binding techniques is
            questionable.
        """
        if relevant_unit_cell_vectors is None:
            relevant_unit_cell_vectors = self.find_relevant_unit_cell_vectors()
        all_minima_index_pairs = itertools.combinations_with_replacement(
            self.sorted_minima_dict.items(), 2
        )
        find_closest_periodic_minimum = partial(
            self._find_closest_periodic_minimum, relevant_unit_cell_vectors
        )
        return np.array(
            list(map(find_closest_periodic_minimum, all_minima_index_pairs))
        )

    def _find_closest_periodic_minimum(
        self, relevant_unit_cell_vectors: dict, minima_index_pair: Tuple
    ) -> float:
        """Helper function comparing minima separation for given minima pair"""
        return self._min_localization_ratio_for_minima_pair(
            minima_index_pair, 0, relevant_unit_cell_vectors
        )

    def _min_localization_ratio_for_minima_pair(
        self,
        minima_index_pair: Tuple,
        Xi_minimum_index_arg: int,
        relevant_unit_cell_vectors: dict,
    ) -> float:
        """Helper function comparing minima separation for
        given minima pair, along with the specification
        that we would like to use the Xi matrix as defined
        for the minimum indexed by `Xi_arg`"""
        (m_prime_index, m_prime_location), (m_index, m_location) = minima_index_pair
        minima_pair_displacement_vectors = relevant_unit_cell_vectors[
            (m_prime_index, m_index)
        ]
        if minima_pair_displacement_vectors is None or np.allclose(
            minima_pair_displacement_vectors, 0.0
        ):
            return np.inf
        Xi_inv = inv(self.Xi_matrix(minimum_index=Xi_minimum_index_arg))
        delta_inv = Xi_inv.T @ Xi_inv
        if (
            m_prime_index == m_index
        ):  # Do not include equivalent minima in the same unit cell
            minima_pair_displacement_vectors = np.array(
                [
                    vec
                    for vec in minima_pair_displacement_vectors
                    if not np.allclose(vec, 0.0)
                ]
            )
        displacement_vectors = 2.0 * np.pi * minima_pair_displacement_vectors + (
            m_location - m_prime_location
        )
        minima_distances = np.linalg.norm(displacement_vectors, axis=1)
        minima_unit_vectors = (
            displacement_vectors
            / np.tile(minima_distances, (self.number_degrees_freedom, 1)).T
        )
        harmonic_lengths = np.array(
            [
                (unit_vec @ delta_inv @ unit_vec) ** (-1 / 2)
                for unit_vec in minima_unit_vectors
            ]
        )
        return np.min(minima_distances / 2.0 / harmonic_lengths)

    def Xi_matrix(
        self, minimum_index: int = 0, harmonic_lengths: Optional[ndarray] = None
    ) -> ndarray:
        """Returns Xi matrix of the normal-mode eigenvectors normalized
        according to \Xi^T C \Xi = \Omega^{-1}/Z0, or equivalently \Xi^T
        \Gamma \Xi = \Omega/Z0. The \Xi matrix
        simultaneously diagonalizes the capacitance and effective
        inductance matrices by a congruence transformation.

        Parameters
        ----------
        minimum_index: int
            integer specifying which minimum to linearize around,
            0<=minimum<= total number of minima
        harmonic_lengths: Optional[ndarray]
            ndarray specifying the harmonic lengths of each mode.
            Useful when considering optimized harmonic lengths.

        Returns
        -------
        ndarray
        """
        sorted_minima_dict = self.sorted_minima_dict
        if harmonic_lengths is None:
            harmonic_lengths = np.ones(
                (len(sorted_minima_dict), self.number_degrees_freedom)
            )
        omega_squared_array, eigenvectors = self.eigensystem_normal_modes(minimum_index)
        Z0 = 0.25  # units where e and hbar = 1; Z0 = hbar / (2 * e)**2
        return np.array(
            [
                eigenvectors[:, i]
                * harmonic_lengths[minimum_index, i]
                * omega_squared ** (-1 / 4)
                * np.sqrt(1.0 / Z0)
                for i, omega_squared in enumerate(omega_squared_array)
            ]
        ).T

    def a_operator(self, dof_index: int) -> ndarray:
        """Returns the lowering operator associated
        with the mu^th d.o.f. in the full Hilbert space

        Parameters
        ----------
        dof_index: int
            which degree of freedom, 0<=dof_index<=self.number_degrees_freedom

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom
        identity_operator_list = np.empty((dim, self.num_exc + 1, self.num_exc + 1))
        identity_operator_list[np.arange(dim)] = np.eye(self.num_exc + 1)
        return identity_wrap_array(
            np.array([annihilation(self.num_exc + 1)]),
            np.array([dof_index]),
            identity_operator_list,
            sparse=False,
        )

    def _a_operator_array(self) -> ndarray:
        """Helper method to return a list of annihilation
        operator matrices for each mode"""
        dim = self.number_degrees_freedom
        num_states_per_min = self.number_states_per_minimum()
        a_operator_array = np.empty((dim, num_states_per_min, num_states_per_min))
        for i in range(dim):
            a_operator_array[i] = self.a_operator(i)
        return a_operator_array

    def find_relevant_unit_cell_vectors(
        self, num_cpus: int = 1
    ) -> Dict[Tuple[int, int], ndarray]:
        """Constructs a dictionary of the relevant unit cell
        vectors for each pair of minima. By unit cell vector
        we are referring to the vector that points from the
         first unit cell to another unit cell. This is done
        by generating all possible unit cell vectors, given the cutoffs specified by
        self.maximum_unit_cell_vector_length, which specifies
        the maximum Manhattan length of a unit cell vector,
        and self.maximum_site_length, which specifies the maximum
        entry allowed in a unit cell vector. We first
        generate the unit cell vectors with all positive entries,
         then perform all possible reflections
        of those vectors and consider each vector individually.
        This is done for each pair of minima in turn.

        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        """
        Xi_inv = inv(self.Xi_matrix())
        sorted_minima_dict = self.sorted_minima_dict
        relevant_unit_cell_vectors = {
            (0, 0): self._unit_cell_vectors_minima_pair(
                np.zeros(self.number_degrees_freedom), Xi_inv, num_cpus
            )
        }
        for m_prime_index in range(1, len(sorted_minima_dict)):
            relevant_unit_cell_vectors[
                (m_prime_index, m_prime_index)
            ] = relevant_unit_cell_vectors[(0, 0)]
        all_minima_location_index_pairs = itertools.combinations(
            sorted_minima_dict.items(), 2
        )
        for (
            (m_prime_index, m_prime_location),
            (m_index, m_location),
        ) in all_minima_location_index_pairs:
            minima_diff = m_location - m_prime_location
            relevant_unit_cell_vectors[
                (m_prime_index, m_index)
            ] = self._unit_cell_vectors_minima_pair(minima_diff, Xi_inv, num_cpus)
        return relevant_unit_cell_vectors

    def _unit_cell_vectors_minima_pair(
        self, minima_diff: ndarray, Xi_inv: ndarray, num_cpus: int
    ) -> ndarray:
        """Given a minima pair, generate and then filter the unit cell vectors"""
        if self.number_periodic_degrees_freedom == 0:
            return np.zeros((1, self.number_degrees_freedom), dtype=int)
        target_map = get_map_method(num_cpus)
        relevant_vectors = list(
            target_map(
                partial(
                    self._generate_and_filter_unit_cell_vectors, minima_diff, Xi_inv
                ),
                np.arange(0, self.maximum_unit_cell_vector_length + 1),
            )
        )
        return self._stack_filtered_vectors(relevant_vectors)

    @staticmethod
    def _stack_filtered_vectors(filtered_vectors: List) -> Optional[ndarray]:
        """Helper function for stacking together unit
        cell vectors of different Manhattan lengths"""
        filtered_vectors = list(filter(lambda x: len(x) != 0, filtered_vectors))
        if filtered_vectors:
            return np.vstack(filtered_vectors)
        else:
            return None

    def _generate_and_filter_unit_cell_vectors(
        self, minima_diff: ndarray, Xi_inv: ndarray, unit_cell_vector_length: int
    ) -> ndarray:
        """Helper function that generates and filters periodic vectors of a
        given Manhattan length. Inspired by the algorithm described in J. M. Zhang
        and R. X. Dong, European Journal of Physics 31, 591 (2010) for generating
        all vectors of a given Manhattan length (specified here by
        periodic_vector_length) with only positive entries. We make two modifications
        here to the underlying algorithm: the first is that we need to also generate
        vectors with negative entries, to obtain all possible unit cell vectors. The
        second is that we don't consider vectors with entries larger than
        self.maximum_site_length. We have found empirically that nominally "short"
        vectors with entries in specific sites that are large don't contribute."""
        filtered_vectors = []
        prev_unit_cell_vec = np.zeros(self.number_periodic_degrees_freedom, dtype=int)
        prev_unit_cell_vec[0] = unit_cell_vector_length
        if unit_cell_vector_length <= self.maximum_site_length:
            self._reflect_and_filter_vector(
                prev_unit_cell_vec, minima_diff, Xi_inv, filtered_vectors
            )
        while prev_unit_cell_vec[-1] != unit_cell_vector_length:
            next_unit_cell_vec = generate_next_vector(
                prev_unit_cell_vec, unit_cell_vector_length
            )
            if (
                len(np.argwhere(next_unit_cell_vec > self.maximum_site_length)) == 0
            ):  # No element > maximum_site_length
                self._reflect_and_filter_vector(
                    next_unit_cell_vec, minima_diff, Xi_inv, filtered_vectors
                )
            prev_unit_cell_vec = next_unit_cell_vec
        return np.array(filtered_vectors)

    def _reflect_and_filter_vector(
        self,
        unit_cell_vector: ndarray,
        minima_diff: ndarray,
        Xi_inv: ndarray,
        filtered_vectors: List,
    ) -> None:
        """Helper function where given a specific vector, generate all
        possible reflections and filter those"""
        filter_function = partial(self._filter_single_vector, minima_diff, Xi_inv)
        relevant_vectors = filter(filter_function, reflect_vectors(unit_cell_vector))
        for filtered_vec in relevant_vectors:
            filtered_vectors.append(
                np.concatenate(
                    (
                        np.zeros(self.number_extended_degrees_freedom, dtype=int),
                        filtered_vec,
                    )
                )
            )

    def _filter_single_vector(
        self, minima_diff: ndarray, Xi_inv: ndarray, unit_cell_vector: ndarray
    ) -> bool:
        """Helper function that does the filtering. Matrix elements are suppressed by a
        gaussian exponential factor, and we filter those that are suppressed below a
        cutoff. Assumption is that extended degrees of freedom precede the periodic
        d.o.f.
        """
        displacement_vector = (
            2.0
            * np.pi
            * np.concatenate(
                (np.zeros(self.number_extended_degrees_freedom), unit_cell_vector)
            )
        )
        gaussian_overlap_argument = Xi_inv @ (displacement_vector + minima_diff)
        gaussian_overlap = np.exp(
            -0.25 * gaussian_overlap_argument @ gaussian_overlap_argument
        )
        return gaussian_overlap > self.displacement_vector_cutoff

    def _identity(self) -> ndarray:
        """Returns the identity matrix whose dimensions are the same as
        self.number_states_per_minimum()"""
        return np.eye(int(self.number_states_per_minimum()))

    def number_states_per_minimum(self) -> int:
        """
        Returns
        -------
        int
            Returns the number of states in each local minimum
        """
        return (self.num_exc + 1) ** self.number_degrees_freedom

    def hilbertdim(self) -> int:
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension.
        """
        return int(len(self.sorted_minima_dict) * self.number_states_per_minimum())

    @staticmethod
    def _premultiplied_a_a_dagger(
        a_operator_array: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Helper method for premultiplying creation and annihilation operators
        (multiplications are expensive)"""
        return (
            a_operator_array,
            a_operator_array @ a_operator_array,
            np.transpose(a_operator_array, (0, 2, 1)) @ a_operator_array,
        )  # a, a * a, a^{\dagger} * a

    def _single_exp_i_phi_j_operator(
        self, dof_index: int, Xi: ndarray, a_operator_array: ndarray
    ) -> ndarray:
        r"""Returns operator :math:`\exp(i\phi_{j})`. If `j` specifies the stitching
        term, which is assumed to be the last junction, then that is constructed
        based on the stitching coefficients."""
        if dof_index == self.number_junctions - 1:
            exp_i_phi_j_a = expm(
                1j
                * np.sum(
                    self.stitching_coefficients
                    @ Xi
                    * np.transpose(a_operator_array, (1, 2, 0)),
                    axis=2,
                )
                / np.sqrt(2.0)
            )
            BCH_factor = self._BCH_factor_for_potential_stitching(Xi)
        else:
            exp_i_phi_j_a = expm(
                1j
                * np.sum(
                    Xi[dof_index] * np.transpose(a_operator_array, (1, 2, 0)), axis=2
                )
                / np.sqrt(2.0)
            )
            BCH_factor = np.exp(-0.25 * Xi[dof_index] @ Xi[dof_index])
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a.T
        return BCH_factor * exp_i_phi_j_a_dagger_component @ exp_i_phi_j_a

    def _all_exp_i_phi_j_operators(
        self, Xi: ndarray, a_operator_array: ndarray
    ) -> ndarray:
        """Helper method for building all potential operators"""
        num_states_per_min = self.number_states_per_minimum()
        all_exp_i_phi_j = np.empty(
            (self.number_junctions, num_states_per_min, num_states_per_min),
            dtype=np.complex128,
        )
        for j in range(self.number_junctions):
            all_exp_i_phi_j[j] = self._single_exp_i_phi_j_operator(
                j, Xi, a_operator_array
            )
        return all_exp_i_phi_j

    def _general_translation_operators(
        self, Xi_inv: ndarray, a_operator_array: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. The resulting matrices yield a
        2pi translation in each degree of freedom, so that any translation can be
        built from these by an appropriate call to np.matrix_power. Note that we need
        only use lowering operators, because raising operators can be constructed by
        transposition. We construct both positive and negative 2pi translations to
        avoid costly calls to `inv` later, which happen implicitly in
        np.matrix_power."""
        dim = self.number_degrees_freedom
        num_states_per_min = self.number_states_per_minimum()
        exp_a_list = np.empty((dim, num_states_per_min, num_states_per_min))
        exp_a_minus_list = np.empty_like(exp_a_list)
        for i in range(dim):
            expm_argument = np.sum(
                2.0
                * np.pi
                * Xi_inv.T[i]
                * np.transpose(a_operator_array, (1, 2, 0))
                / np.sqrt(2.0),
                axis=2,
            )
            exp_a_list[i] = expm(expm_argument)
            exp_a_minus_list[i] = expm(-expm_argument)
        return exp_a_list, exp_a_minus_list

    def _minima_dependent_translation_operators(
        self, minima_diff: ndarray, Xi_inv: ndarray, a_operator_array: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Helper method that performs matrix exponentiation to aid in the
        future construction of translation operators. This part of the translation
        operator accounts for the differing location of minima within a single unit
        cell."""
        exp_a_minima_difference = expm(
            np.sum(
                (-minima_diff @ Xi_inv.T)
                * np.transpose(a_operator_array, (1, 2, 0))
                / np.sqrt(2.0),
                axis=2,
            )
        )
        exp_a_dagger_minima_difference = expm(
            np.sum((minima_diff @ Xi_inv.T) * a_operator_array.T / np.sqrt(2.0), axis=2)
        )
        return exp_a_minima_difference, exp_a_dagger_minima_difference

    def _local_translation_operators(
        self,
        exp_list: Tuple[ndarray, ndarray],
        exp_minima_difference: Tuple[ndarray, ndarray],
        unit_cell_vector: ndarray,
    ) -> Tuple[ndarray, ndarray]:
        """Helper method that builds translation operators using matrix_power and the
        pre-exponentiated translation operators that define 2pi translations. Note
        that this function is currently the speed bottleneck."""
        exp_a_list, exp_a_minus_list = exp_list
        exp_a_minima_difference, exp_a_dagger_minima_difference = exp_minima_difference
        ops_with_power_for_a_dagger = zip(
            exp_a_list,
            exp_a_minus_list,
            unit_cell_vector.astype(int),
        )
        ops_with_power_for_a = zip(
            exp_a_list,
            exp_a_minus_list,
            -unit_cell_vector.astype(int),
        )
        translation_op_a_dagger = (
            reduce(
                np.matmul, map(self._matrix_power_helper, ops_with_power_for_a_dagger)
            ).T
            @ exp_a_dagger_minima_difference
        )
        translation_op_a = (
            reduce(np.matmul, map(self._matrix_power_helper, ops_with_power_for_a))
            @ exp_a_minima_difference
        )
        return translation_op_a_dagger, translation_op_a

    @staticmethod
    def _matrix_power_helper(
        translation_op_with_power: Tuple[ndarray, ndarray, int]
    ) -> ndarray:
        """Helper method that actually returns translation operators. If the translation
        operator has been built before and stored, use that result. Additionally if
        the translation is given by a negative integer, take advantage of having
        built a pre-exponentiated operator with -2\pi argument to avoid a costly call to
        inv."""
        (exp_plus_list, exp_minus_list, unit_cell_vector) = translation_op_with_power
        if unit_cell_vector >= 0:
            translation_operator = matrix_power(exp_plus_list, unit_cell_vector)
        else:
            translation_operator = matrix_power(exp_minus_list, -unit_cell_vector)
        return translation_operator

    def _exp_product_coefficient(self, delta_phi: ndarray, Xi_inv: ndarray) -> ndarray:
        """Returns overall multiplicative factor, including offset charge and Gaussian
        suppression BCH factor from the translation operators"""
        delta_phi_rotated = Xi_inv @ delta_phi
        return np.exp(-1j * self.nglist @ delta_phi) * np.exp(
            -0.25 * delta_phi_rotated @ delta_phi_rotated
        )

    def _BCH_factor_for_potential_stitching(self, Xi: ndarray) -> ndarray:
        """BCH factor obtained from the last potential operator"""
        return np.exp(
            -0.25
            * self.stitching_coefficients
            @ Xi
            @ Xi.T
            @ self.stitching_coefficients
        )

    def _abstract_VTB_operator(
        self, local_func: Callable, num_cpus: int = 1
    ) -> ndarray:
        """Factory for building a VTB operator. local_func represents the local
        contribution to the operator, taking into account normal ordering. local func
        must have the signature
        local_func(precalculated_quantities: Tuple[ndarray, ndarray,
         Tuple, ndarray, ndarray], displacement_vector: ndarray, m_prime_location: ndarray,
         m_location: ndarray)
        where precalculated_quantities = (Xi, Xi_inv, premultiplied_a_a_dagger,
                                          exp_i_phi_j, EC_mat_t)
        defined below. These are all expensive quantities to calculate over and over
        again for each displacement_vector, but do not depend on the
        displacement_vector, therefore are calculated once and extracted when needed.
        Obviously, for example, exp_i_phi_j is not necessary for the kinetic
        calculation, but is passed nonetheless for consistency.
        """
        relevant_unit_cell_vectors, harmonic_lengths = self._initialize_VTB(num_cpus)
        Xi = self.Xi_matrix(minimum_index=0, harmonic_lengths=harmonic_lengths)
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        exp_i_phi_j = self._all_exp_i_phi_j_operators(Xi, self._a_operator_array())
        EC_mat_t = Xi_inv @ self.EC_matrix() @ Xi_inv.T
        partial_local_func = partial(
            local_func, (Xi, Xi_inv, premultiplied_a_a_dagger, exp_i_phi_j, EC_mat_t)
        )
        return self._periodic_continuation(
            partial_local_func, relevant_unit_cell_vectors, harmonic_lengths, num_cpus
        )

    def _initialize_VTB(
        self, num_cpus: int = 1
    ) -> Tuple[Dict[Tuple[int, int], ndarray], ndarray]:
        """Initialization when building a VTB operator"""
        relevant_unit_cell_vectors = self.find_relevant_unit_cell_vectors(
            num_cpus=num_cpus
        )
        if self.harmonic_length_optimization:
            harmonic_lengths = self._optimize_harmonic_lengths(
                relevant_unit_cell_vectors
            )
        else:
            harmonic_lengths = None
        return relevant_unit_cell_vectors, harmonic_lengths

    def n_operator(self, dof_index: int = 0, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        dof_index: int
            which degree of freedom, 0<=dof_index<=self.number_degrees_freedom
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
            Charge operator in the basis of VTB states
        """
        return self._abstract_VTB_operator(
            partial(self._local_charge_operator, dof_index), num_cpus
        )

    def phi_operator(self, dof_index: int = 0, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        dof_index: int
            which degree of freedom, 0<=dof_index<self.number_extended_degrees_freedom
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
            :math:`\\phi` operator in the basis of VTB states. Note that this is only
            defined for degrees of freedom dof_index that are extended
        """
        if dof_index >= self.number_extended_degrees_freedom:
            raise ValueError(
                "phi_operator is only defined for extended degrees of freedom"
            )
        return self._abstract_VTB_operator(
            partial(self._local_phi_operator, dof_index), num_cpus
        )

    def exp_i_phi_operator(self, dof_index: int = 0, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        dof_index: int
            which degree of freedom, 0<=dof_index<=self.number_degrees_freedom
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
            :math:`e^{i \\phi}` operator in the basis of VTB states.
        """
        return self._abstract_VTB_operator(
            partial(self._local_exp_i_phi_operator, dof_index), num_cpus
        )

    def hamiltonian(self, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
        ndarray
            Returns the transfer matrix
        """
        return self.transfer_matrix(num_cpus)

    def kinetic_matrix(self, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        return self._abstract_VTB_operator(self._local_kinetic, num_cpus)

    def potential_matrix(self, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        return self._abstract_VTB_operator(self._local_potential, num_cpus)

    def transfer_matrix(self, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
        ndarray
            Returns the transfer matrix
        """
        return self._abstract_VTB_operator(self._local_transfer, num_cpus)

    def inner_product_matrix(self, num_cpus: int = 1) -> ndarray:
        """
        Parameters
        ----------
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.
        Returns
        -------
        ndarray
            Returns the inner product matrix
        """
        return self._abstract_VTB_operator(
            lambda precalculated_quantities, displacement_vector, m_prime_location, m_location: self._identity(),
            num_cpus,
        )

    def _transfer_matrix(
        self,
        relevant_unit_cell_vectors: dict,
        harmonic_lengths: ndarray,
        num_cpus: int = 1,
    ):
        """To be called in conjunction with _inner_product_matrix, it is expected that
        _initialize_VTB has already been called. In an eigenvalue calculation,
        one must calculate both the transfer matrix and the inner product matrix,
        but the translation operators, relevant unit cell vectors and harmonic lengths
        will be the same for both. Therefore those calculations need only be performed
        once."""
        Xi = self.Xi_matrix(0, harmonic_lengths=harmonic_lengths)
        Xi_inv = inv(Xi)
        a_operator_array = self._a_operator_array()
        exp_i_phi_j = self._all_exp_i_phi_j_operators(Xi, a_operator_array)
        premultiplied_a_a_dagger = self._premultiplied_a_a_dagger(a_operator_array)
        EC_mat_t = Xi_inv @ self.EC_matrix() @ Xi_inv.T
        transfer_matrix_function = partial(
            self._local_transfer,
            (Xi, Xi_inv, premultiplied_a_a_dagger, exp_i_phi_j, EC_mat_t),
        )
        return self._periodic_continuation(
            transfer_matrix_function,
            relevant_unit_cell_vectors,
            harmonic_lengths,
            num_cpus,
        )

    def _inner_product_matrix(
        self,
        relevant_unit_cell_vectors: dict,
        harmonic_lengths: ndarray,
        num_cpus: int = 1,
    ):
        """See _transfer_matrix documentation. Calculate the inner product matrix with
        pre-calculated relevant unit cell vectors and harmonic lengths"""
        return self._periodic_continuation(
            lambda displacement_vector, m_prime_location, m_location: self._identity(),
            relevant_unit_cell_vectors,
            harmonic_lengths,
            num_cpus=num_cpus,
        )

    def _local_charge_operator(
        self,
        dof_index: int,
        precalculated_quantities: Tuple[ndarray, ndarray, Tuple, ndarray, ndarray],
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        r"""Calculate the local contribution to the charge operator given two
        minima and a unit cell vector `displacement_vector`"""
        _, Xi_inv, premultiplied_a_a_dagger, _, _ = precalculated_quantities
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        constant_coefficient = (
            -0.5
            * 1j
            * (
                Xi_inv.T
                @ Xi_inv
                @ (displacement_vector + m_location - m_prime_location)
            )[dof_index]
        )
        return (
            -(1j / np.sqrt(2.0))
            * np.sum(Xi_inv.T[dof_index] * (np.transpose(a, (1, 2, 0)) - a.T), axis=2)
            + constant_coefficient * self._identity()
        )

    def _local_phi_operator(
        self,
        dof_index: int,
        precalculated_quantities: Tuple[ndarray, ndarray, Tuple, ndarray, ndarray],
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        r"""Calculate the local contribution to the `\phi` operator given two
        minima and a unit cell vector `displacement_vector`"""
        Xi, _, premultiplied_a_a_dagger, _, _ = precalculated_quantities
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        constant_coefficient = 0.5 * (
            displacement_vector + (m_prime_location + m_location)
        )
        return (1.0 / np.sqrt(2.0)) * np.sum(
            Xi[dof_index] * (np.transpose(a, (1, 2, 0)) + a.T), axis=2
        ) + constant_coefficient[dof_index] * self._identity()

    def _local_exp_i_phi_operator(
        self,
        dof_index: int,
        precalculated_quantities: Tuple[ndarray, ndarray, Tuple, ndarray, ndarray],
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        r"""Calculate the local contribution to the :math:`e^{i \\phi}` operator given
        two minima and a unit cell vector `displacement_vector`"""
        _, _, _, exp_i_phi_j, _ = precalculated_quantities
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (displacement_vector + (m_prime_location + m_location))
        (
            exp_i_phi_j_phi_bar,
            exp_i_stitching_phi_j_phi_bar,
        ) = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        if dof_index == dim:
            return exp_i_stitching_phi_j_phi_bar
        else:
            return exp_i_phi_j_phi_bar[dof_index]

    def _exp_i_phi_j_with_phi_bar(
        self, exp_i_phi_j: ndarray, phi_bar: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Returns exp_i_phi_j operators including the local contribution of phi_bar"""
        if exp_i_phi_j.ndim > 1:  # Normal VTB
            _exp_i_phi_j_phi_bar = np.transpose(exp_i_phi_j[:-1], (1, 2, 0)) * np.exp(
                1j * phi_bar
            )
            exp_i_phi_j_phi_bar = np.transpose(_exp_i_phi_j_phi_bar, (2, 0, 1))
        else:  # One state VTB, each element of the array is the ground state value
            exp_i_phi_j_phi_bar = exp_i_phi_j[:-1] * np.exp(1j * phi_bar)
        exp_i_stitching_phi_j_phi_bar = (
            exp_i_phi_j[-1]
            * np.exp(1j * 2.0 * np.pi * self.flux)
            * np.exp(1j * self.stitching_coefficients @ phi_bar)
        )
        return exp_i_phi_j_phi_bar, exp_i_stitching_phi_j_phi_bar

    def _local_kinetic(
        self,
        precalculated_quantities: Tuple[ndarray, ndarray, Tuple, ndarray, ndarray],
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Calculate the local kinetic contribution to the transfer matrix given two
        minima and a unit cell vector `displacement_vector`"""
        _, Xi_inv, premultiplied_a_a_dagger, _, EC_mat_t = precalculated_quantities
        a, a_a, a_dagger_a = premultiplied_a_a_dagger
        delta_phi = displacement_vector + m_location - m_prime_location
        delta_phi_rotated = Xi_inv @ delta_phi
        local_kinetic_diagonal = (
            -2.0 * np.transpose(a_a, (1, 2, 0))
            - 2.0 * a_a.T
            + 4.0 * np.transpose(a_dagger_a, (1, 2, 0))
            - 4 * (np.transpose(a, (1, 2, 0)) - a.T) * delta_phi_rotated / np.sqrt(2.0)
        )
        kinetic_matrix = np.sum(np.diag(EC_mat_t) * local_kinetic_diagonal, axis=2)
        identity_coefficient = (
            0.5 * 4 * np.trace(EC_mat_t)
            - 0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated
        )
        kinetic_matrix = kinetic_matrix + identity_coefficient * self._identity()
        return kinetic_matrix

    def _local_potential(
        self,
        precalculated_quantities: Tuple[ndarray, ndarray, Tuple, ndarray, ndarray],
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Calculate the local potential contribution to the transfer matrix given two
        minima and a unit cell vector `displacement_vector`"""
        _, _, _, exp_i_phi_j, _ = precalculated_quantities
        phi_bar = 0.5 * (displacement_vector + (m_prime_location + m_location))
        (
            exp_i_phi_j_phi_bar,
            exp_i_stitching_phi_j_phi_bar,
        ) = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        potential_matrix = np.sum(
            -0.5
            * self.EJlist[:-1]
            * (
                np.transpose(exp_i_phi_j_phi_bar, (1, 2, 0))
                + np.transpose(exp_i_phi_j_phi_bar, (1, 2, 0)).conjugate()
            ),
            axis=2,
        )
        potential_matrix = potential_matrix - 0.5 * self.EJlist[-1] * (
            exp_i_stitching_phi_j_phi_bar + exp_i_stitching_phi_j_phi_bar.conjugate()
        )
        potential_matrix = potential_matrix + np.sum(self.EJlist) * self._identity()
        return potential_matrix

    def _local_transfer(
        self,
        precalculated_quantities: Tuple[ndarray, ndarray, Tuple, ndarray, ndarray],
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Calculate the local contribution to the transfer matrix given two
        minima and a unit cell vector `displacement_vector`"""
        return self._local_kinetic(
            precalculated_quantities, displacement_vector, m_prime_location, m_location
        ) + self._local_potential(
            precalculated_quantities, displacement_vector, m_prime_location, m_location
        )

    def _periodic_continuation(
        self,
        local_func: Callable,
        relevant_unit_cell_vectors: Dict[Tuple[int, int], ndarray],
        harmonic_lengths: ndarray,
        num_cpus: int = 1,
    ) -> ndarray:
        """This function is the meat of the VariationalTightBinding method, performing
        the summation over unit cell vectors for each pair of minima and constructing
        the resulting operator
        ::
                [M_{0, 0} M_{0, 1} ...  M_{0, N_{\text{min}}} ]
            M = [M_{1, 0} M_{1, 1} ...          \vdot         ]
                [ \vdot            \ddot                      ]
                [M_{N_{\text{min}}, 0} ...    M_{N_{\text{min}},N_{\text{min}} }]

        where `M_{ij}` refers to the matrix block calculated for the pair
        of minima `i`, `j`
        Parameters
        ----------
        local_func: Callable
            function that takes three arguments (displacement_vector, m_prime_location,
             m_location) and returns the relevant operator with dimension NxN,
             where N=self.number_states_per_minimum().
        relevant_unit_cell_vectors: Dict[Tuple[int, int], ndarray]
            Dictionary of the relevant unit cell vectors for each pair of minima
        harmonic_lengths: ndarray
            harmonic lengths to be passed to the Xi matrix
        num_cpus: int
            Number of CPUS/cores employed in underlying calculation.

        Returns
        -------
        ndarray
        """
        Xi_inv = inv(self.Xi_matrix(0, harmonic_lengths))
        target_map = get_map_method(num_cpus)
        a_operator_array = self._a_operator_array()
        exp_a_list = self._general_translation_operators(Xi_inv, a_operator_array)
        all_minima_index_pairs = list(
            itertools.combinations_with_replacement(self.sorted_minima_dict.items(), 2)
        )
        periodic_continuation_for_minima_pair = partial(
            self._periodic_continuation_for_minima_pair,
            local_func,
            exp_a_list,
            Xi_inv,
            a_operator_array,
            relevant_unit_cell_vectors,
        )
        matrix_elements = list(
            target_map(periodic_continuation_for_minima_pair, all_minima_index_pairs)
        )
        return self._construct_VTB_operator_given_blocks(
            matrix_elements, all_minima_index_pairs
        )

    def _construct_VTB_operator_given_blocks(
        self, matrix_elements, all_minima_index_pairs
    ):
        num_states_per_min = self.number_states_per_minimum()
        hilbertdim = self.hilbertdim()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        for i, ((m_prime_index, m_prime_location), (m_index, m_location)) in enumerate(
            all_minima_index_pairs
        ):
            operator_matrix[
                m_prime_index
                * num_states_per_min : (m_prime_index + 1)
                * num_states_per_min,
                m_index * num_states_per_min : (m_index + 1) * num_states_per_min,
            ] += matrix_elements[i]
        return self._populate_hermitian_matrix(operator_matrix)

    def _periodic_continuation_for_minima_pair(
        self,
        func: Callable,
        exp_a_list: Tuple[ndarray, ndarray],
        Xi_inv: ndarray,
        a_operator_array: ndarray,
        relevant_unit_cell_vectors: dict,
        minima_index_pair: Tuple,
    ) -> ndarray:
        """Helper method for performing the periodic continuation calculation given a
        minima pair."""
        ((m_prime_index, m_prime_location), (m_index, m_location)) = minima_index_pair
        minima_pair_displacement_vectors = relevant_unit_cell_vectors[
            (m_prime_index, m_index)
        ]
        num_states_per_min = self.number_states_per_minimum()
        if minima_pair_displacement_vectors is not None:
            minima_diff = m_location - m_prime_location
            exp_minima_difference = self._minima_dependent_translation_operators(
                minima_diff, Xi_inv, a_operator_array
            )
            displacement_vector_contribution = partial(
                self._displacement_vector_contribution,
                func,
                m_prime_location,
                m_location,
                exp_a_list,
                exp_minima_difference,
                Xi_inv,
            )
            relevant_vector_contributions = sum(
                map(displacement_vector_contribution, minima_pair_displacement_vectors)
            )
        else:
            relevant_vector_contributions = np.zeros(
                (num_states_per_min, num_states_per_min), dtype=np.complex128
            )
        return relevant_vector_contributions

    def _displacement_vector_contribution(
        self,
        func: Callable,
        m_prime_location: ndarray,
        m_location: ndarray,
        exp_a_list: Tuple[ndarray, ndarray],
        exp_minima_difference: Tuple[ndarray, ndarray],
        Xi_inv: ndarray,
        unit_cell_vector: ndarray,
    ) -> ndarray:
        """Helper method for calculating the contribution of a specific
        unit cell vector `displacement_vector`"""
        displacement_vector = 2.0 * np.pi * np.array(unit_cell_vector)
        exp_prod_coefficient = self._exp_product_coefficient(
            displacement_vector + m_location - m_prime_location, Xi_inv
        )
        exp_a_dagger, exp_a = self._local_translation_operators(
            exp_a_list, exp_minima_difference, unit_cell_vector
        )
        matrix_element = exp_prod_coefficient * func(
            displacement_vector, m_prime_location, m_location
        )
        return exp_a_dagger @ matrix_element @ exp_a

    def _populate_hermitian_matrix(self, mat: ndarray) -> ndarray:
        """Return a fully Hermitian matrix, assuming that the input matrix has been
        populated with the upper right blocks"""
        sorted_minima_dict = self.sorted_minima_dict
        num_states_per_min = self.number_states_per_minimum()
        for m_prime_index, _ in sorted_minima_dict.items():
            for p in range(m_prime_index + 1, len(sorted_minima_dict)):
                matrix_element = mat[
                    m_prime_index
                    * num_states_per_min : (m_prime_index + 1)
                    * num_states_per_min,
                    p * num_states_per_min : (p + 1) * num_states_per_min,
                ]
                mat[
                    p * num_states_per_min : (p + 1) * num_states_per_min,
                    m_prime_index
                    * num_states_per_min : (m_prime_index + 1)
                    * num_states_per_min,
                ] += matrix_element.conjugate().T
        return mat

    def _evals_esys_calc(
        self, evals_count: int, eigvals_only: bool, num_cpus: int = 1
    ) -> ndarray:
        """Helper method that wraps the try and except regarding
        singularity/indefiniteness of the inner product matrix"""
        relevant_unit_cell_vectors, harmonic_lengths = self._initialize_VTB(num_cpus)
        harmonic_length_minima_comparison = self.compute_minimum_localization_ratio()
        if np.min(harmonic_length_minima_comparison) < 1.0:
            warnings.warn(
                "Warning: small minima separation compared to harmonic length. "
                "smallest is (d/2)/(2*l) = {ratio})".format(
                    ratio=np.min(harmonic_length_minima_comparison)
                )
            )
        transfer_matrix = self._transfer_matrix(
            relevant_unit_cell_vectors, harmonic_lengths, num_cpus=num_cpus
        )
        inner_product_matrix = self._inner_product_matrix(
            relevant_unit_cell_vectors, harmonic_lengths, num_cpus=num_cpus
        )
        if self.safe_run:
            eval_tol = self.inner_prod_eval_tol
            evals_inn, evecs_inn = eigh(inner_product_matrix)
            evals_nonzero = list(filter(lambda x: x > eval_tol, evals_inn))
            if not np.allclose(evals_inn, evals_nonzero):
                idx = np.abs(evals_inn - evals_nonzero[0]).argmin()
                evecs_nonzero = evecs_inn[:, idx:]
                scale_factor = np.array(evals_nonzero) ** (-0.5)
                scale_factor_mat = np.outer(scale_factor, scale_factor)
                new_tran = evecs_nonzero.conj().T @ transfer_matrix @ evecs_nonzero
                new_tran = new_tran * scale_factor_mat
                eigs = eigh(
                    new_tran,
                    eigvals_only=eigvals_only,
                    eigvals=(0, min(evals_count - 1, self.hilbertdim() - idx - 1)),
                )
                return eigs
        try:
            eigs = eigh(
                transfer_matrix,
                b=inner_product_matrix,
                eigvals_only=eigvals_only,
                eigvals=(0, evals_count - 1),
            )
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            eigs = solve_generalized_eigenvalue_problem_with_QZ(
                transfer_matrix,
                inner_product_matrix,
                evals_count,
                eigvals_only=eigvals_only,
            )
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
        """Returns the value of the potential at minima_location and
        the location of minima_location, in sorted order."""
        minima_location_array = self.find_minima()
        value_of_potential_array = np.array(
            [
                self.vtb_potential(minima_location)
                for minima_location in minima_location_array
            ]
        )
        sorted_indices = np.argsort(value_of_potential_array)
        return (
            value_of_potential_array[sorted_indices],
            minima_location_array[sorted_indices, :],
        )

    @property
    def sorted_minima_dict(self) -> Dict[int, ndarray]:
        """
        Return sorted array of the minima locations

        Returns
        -------
        ndarray
        """
        _, sorted_minima_array = self._sorted_potential_values_and_minima()
        return {
            minimum_index: minimum_location
            for minimum_index, minimum_location in enumerate(sorted_minima_array)
        }

    def _normalize_minimum_inside_pi_range(self, minimum_location: ndarray) -> ndarray:
        """Helper method for defining the unit cell from -pi to
        pi rather than the less symmetric 0 to 2pi"""
        num_extended = self.number_extended_degrees_freedom
        extended_coordinates = minimum_location[0:num_extended]
        periodic_coordinates = np.mod(
            minimum_location, 2 * np.pi * np.ones_like(minimum_location)
        )[num_extended:]
        periodic_coordinates = np.array(
            [
                elem - 2 * np.pi if elem > np.pi else elem
                for elem in periodic_coordinates
            ]
        )
        return np.concatenate((extended_coordinates, periodic_coordinates))

    def _check_if_new_minima(
        self, new_minima_location: ndarray, minima_list: List
    ) -> bool:
        """Helper method for find_minima, checking if new_minima is
        already represented in minima_list. If so,
        _check_if_new_minima returns False.
        """
        num_extended = self.number_extended_degrees_freedom
        for minimum_location in minima_list:
            extended_coordinates = np.array(
                minimum_location[0:num_extended] - new_minima_location[0:num_extended]
            )
            periodic_coordinates = np.mod(
                minimum_location - new_minima_location,
                2 * np.pi * np.ones_like(minimum_location),
            )[num_extended:]
            diff_array_bool_extended = [
                True if np.allclose(elem, 0.0, atol=1e-3) else False
                for elem in extended_coordinates
            ]
            diff_array_bool_periodic = [
                True
                if (
                    np.allclose(elem, 0.0, atol=1e-3)
                    or np.allclose(elem, 2 * np.pi, atol=1e-3)
                )
                else False
                for elem in periodic_coordinates
            ]
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

    def _optimize_harmonic_lengths(self, relevant_unit_cell_vectors: dict) -> ndarray:
        """Optimize the Xi matrix by adjusting the harmonic lengths of the ground state
        to minimize its energy.  For tight-binding without squeezing, this is only
        done for the ansatz ground state wavefunction localized in the global
        minimum."""
        sorted_minima_dict = self.sorted_minima_dict
        num_minima = len(sorted_minima_dict)
        harmonic_lengths = np.ones((num_minima, self.number_degrees_freedom))
        optimized_harmonic_lengths = self._optimize_harmonic_lengths_minimum(
            0, sorted_minima_dict[0], relevant_unit_cell_vectors
        )
        harmonic_lengths[np.arange(num_minima)] = optimized_harmonic_lengths
        return harmonic_lengths

    def _optimize_harmonic_lengths_minimum(
        self,
        minimum_index: int,
        minimum_location: ndarray,
        relevant_unit_cell_vectors: dict,
    ) -> ndarray:
        """Perform the harmonic length optimization for a h.o. ground state wavefunction
        localized in a given minimum"""
        default_Xi = self.Xi_matrix(minimum_index)
        EC_mat = self.EC_matrix()
        optimized_lengths_result = minimize(
            self._evals_calc_variational,
            np.ones(self.number_degrees_freedom),
            jac=self._gradient_evals_calc_variational,
            args=(
                minimum_location,
                minimum_index,
                EC_mat,
                default_Xi,
                relevant_unit_cell_vectors,
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

    @staticmethod
    def _update_Xi(default_Xi: ndarray, harmonic_lengths) -> ndarray:
        """Helper method for updating Xi so that the Xi
        matrix is not constantly regenerated."""
        return np.array(
            [row * harmonic_lengths[i] for i, row in enumerate(default_Xi.T)]
        ).T

    def _gradient_evals_calc_variational(
        self,
        harmonic_lengths: ndarray,
        minimum_location: ndarray,
        minimum_index: int,
        EC_mat: ndarray,
        default_Xi: ndarray,
        relevant_unit_cell_vectors: dict,
    ) -> ndarray:
        """Returns the gradient of evals_calc_variational to aid in
        the harmonic length optimization calculation"""
        Xi = self._update_Xi(default_Xi, harmonic_lengths)
        Xi_inv = inv(Xi)
        exp_i_phi_j = self._one_state_exp_i_phi_j_operators(Xi)
        EC_mat_t = Xi_inv @ EC_mat @ Xi_inv.T
        displacement_vectors = (
            2.0 * np.pi * relevant_unit_cell_vectors[(minimum_index, minimum_index)]
        )
        gradient_transfer = [
            np.sum(
                [
                    self._exp_product_coefficient(displacement_vector, Xi_inv)
                    * self._gradient_one_state_local_transfer(
                        Xi,
                        Xi_inv,
                        exp_i_phi_j,
                        EC_mat_t,
                        harmonic_lengths,
                        which_length,
                        displacement_vector,
                        minimum_location,
                        minimum_location,
                    )
                    + self._gradient_one_state_local_inner_product(
                        Xi_inv, harmonic_lengths, which_length, displacement_vector
                    )
                    * self._one_state_local_transfer(
                        Xi,
                        Xi_inv,
                        exp_i_phi_j,
                        EC_mat_t,
                        displacement_vector,
                        minimum_location,
                        minimum_location,
                    )
                    for displacement_vector in displacement_vectors
                ]
            )
            for which_length in range(self.number_degrees_freedom)
        ]
        transfer = np.sum(
            [
                self._exp_product_coefficient(displacement_vector, Xi_inv)
                * self._one_state_local_transfer(
                    Xi,
                    Xi_inv,
                    exp_i_phi_j,
                    EC_mat_t,
                    displacement_vector,
                    minimum_location,
                    minimum_location,
                )
                for displacement_vector in displacement_vectors
            ]
        )
        gradient_inner = [
            np.sum(
                [
                    self._gradient_one_state_local_inner_product(
                        Xi_inv, harmonic_lengths, which_length, displacement_vector
                    )
                    for displacement_vector in displacement_vectors
                ]
            )
            for which_length in range(self.number_degrees_freedom)
        ]
        inner = np.sum(
            [
                self._exp_product_coefficient(displacement_vector, Xi_inv)
                for displacement_vector in displacement_vectors
            ]
        )
        return np.real(
            (inner * np.array(gradient_transfer) - transfer * np.array(gradient_inner))
            / inner ** 2
        )

    def _evals_calc_variational(
        self,
        harmonic_lengths: ndarray,
        minimum_location: ndarray,
        minimum_index: int,
        EC_mat: ndarray,
        default_Xi: ndarray,
        relevant_unit_cell_vectors: dict,
    ) -> ndarray:
        """Function to be optimized in the minimization procedure, corresponding to the
        variational estimate of the ground state energy."""
        Xi = self._update_Xi(default_Xi, harmonic_lengths)
        Xi_inv = inv(Xi)
        exp_i_phi_j = self._one_state_exp_i_phi_j_operators(Xi)
        EC_mat_t = Xi_inv @ EC_mat @ Xi_inv.T
        transfer, inner = self._one_state_construct_transfer_and_inner(
            Xi,
            Xi_inv,
            minimum_location,
            minimum_index,
            EC_mat_t,
            exp_i_phi_j,
            relevant_unit_cell_vectors,
        )
        return np.real([transfer / inner])

    def _one_state_exp_i_phi_j_operators(self, Xi: ndarray) -> ndarray:
        r"""Helper method for building :math:`\exp(i\phi_{j})` when no
        excitations are kept."""
        dim = self.number_degrees_freedom
        exp_factors = np.array(
            [np.exp(-0.25 * np.dot(Xi[j, :], Xi.T[:, j])) for j in range(dim)]
        )
        return np.append(exp_factors, self._BCH_factor_for_potential_stitching(Xi))

    def _gradient_one_state_local_inner_product(
        self,
        Xi_inv: ndarray,
        harmonic_lengths: ndarray,
        which_length: int,
        displacement_vector: ndarray,
    ) -> float:
        """Returns gradient of the inner product matrix"""
        delta_phi_rotated = Xi_inv @ displacement_vector
        return (
            harmonic_lengths[which_length] ** (-1)
            * (
                1j * self.nglist[which_length] * delta_phi_rotated[which_length]
                + 0.5 * delta_phi_rotated[which_length] ** 2
            )
            * self._exp_product_coefficient(displacement_vector, Xi_inv)
        )

    @staticmethod
    def _one_state_local_kinetic(
        Xi_inv: ndarray,
        EC_mat_t: ndarray,
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> float:
        """Local kinetic contribution when considering only the ground state."""
        delta_phi_rotated = Xi_inv @ (
            displacement_vector + m_location - m_prime_location
        )
        return (
            0.5 * 4 * np.trace(EC_mat_t)
            - 0.25 * 4 * delta_phi_rotated @ EC_mat_t @ delta_phi_rotated
        )

    @staticmethod
    def _gradient_one_state_local_kinetic(
        Xi_inv: ndarray,
        EC_mat_t: ndarray,
        harmonic_lengths: ndarray,
        which_length: int,
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Returns gradient of the kinetic matrix"""
        delta_phi_rotated = Xi_inv @ (
            displacement_vector + m_location - m_prime_location
        )
        return (
            -4.0
            * harmonic_lengths[which_length] ** (-1)
            * (
                EC_mat_t[which_length, which_length]
                - (delta_phi_rotated @ EC_mat_t)[which_length]
                * delta_phi_rotated[which_length]
            )
        )

    def _gradient_one_state_local_potential(
        self,
        Xi: ndarray,
        exp_i_phi_j: ndarray,
        harmonic_lengths: ndarray,
        which_length: int,
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Returns gradient of the potential matrix"""
        phi_bar = 0.5 * (displacement_vector + (m_prime_location + m_location))
        (
            exp_i_phi_j_phi_bar,
            exp_i_stitching_phi_j_phi_bar,
        ) = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        potential_gradient = np.sum(
            [
                0.25
                * self.EJlist[junction]
                * Xi[junction, which_length]
                * Xi.T[which_length, junction]
                * harmonic_lengths[which_length] ** (-1)
                * (
                    exp_i_phi_j_phi_bar[junction]
                    + exp_i_phi_j_phi_bar[junction].conjugate()
                )
                for junction in range(self.number_junctions - 1)
            ]
        )
        potential_gradient += (
            0.25
            * self.EJlist[-1]
            * harmonic_lengths[which_length] ** (-1)
            * (self.stitching_coefficients @ Xi[:, which_length]) ** 2
            * (
                exp_i_stitching_phi_j_phi_bar
                + exp_i_stitching_phi_j_phi_bar.conjugate()
            )
        )
        return potential_gradient

    def _gradient_one_state_local_transfer(
        self,
        Xi: ndarray,
        Xi_inv: ndarray,
        exp_i_phi_j: ndarray,
        EC_mat_t: ndarray,
        harmonic_lengths: ndarray,
        which_length: int,
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Returns gradient of the transfer matrix"""
        return self._gradient_one_state_local_potential(
            Xi,
            exp_i_phi_j,
            harmonic_lengths,
            which_length,
            displacement_vector,
            m_prime_location,
            m_location,
        ) + self._gradient_one_state_local_kinetic(
            Xi_inv,
            EC_mat_t,
            harmonic_lengths,
            which_length,
            displacement_vector,
            m_prime_location,
            m_location,
        )

    def _one_state_local_potential(
        self,
        Xi: ndarray,
        exp_i_phi_j: ndarray,
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Local potential contribution when considering only the ground state."""
        phi_bar = 0.5 * (displacement_vector + (m_prime_location + m_location))
        (
            exp_i_phi_j_phi_bar,
            exp_i_stitching_phi_j_phi_bar,
        ) = self._exp_i_phi_j_with_phi_bar(exp_i_phi_j, phi_bar)
        potential = np.sum(
            [
                -0.5
                * self.EJlist[junction]
                * (
                    exp_i_phi_j_phi_bar[junction]
                    + exp_i_phi_j_phi_bar[junction].conjugate()
                )
                for junction in range(self.number_junctions - 1)
            ]
        )
        potential += np.sum(self.EJlist) - 0.5 * self.EJlist[-1] * (
            exp_i_stitching_phi_j_phi_bar + exp_i_stitching_phi_j_phi_bar.conjugate()
        )
        return potential

    def _one_state_local_transfer(
        self,
        Xi: ndarray,
        Xi_inv: ndarray,
        exp_i_phi_j: ndarray,
        EC_mat_t: ndarray,
        displacement_vector: ndarray,
        m_prime_location: ndarray,
        m_location: ndarray,
    ) -> ndarray:
        """Local transfer contribution when considering only the ground state."""
        return self._one_state_local_kinetic(
            Xi_inv, EC_mat_t, displacement_vector, m_prime_location, m_location
        ) + self._one_state_local_potential(
            Xi, exp_i_phi_j, displacement_vector, m_prime_location, m_location
        )

    def _one_state_construct_transfer_and_inner(
        self,
        Xi: ndarray,
        Xi_inv: ndarray,
        minimum_location: ndarray,
        minimum_index: int,
        EC_mat_t: ndarray,
        exp_i_phi_j: ndarray,
        relevant_unit_cell_vectors: dict,
    ) -> Tuple:
        """Transfer matrix and inner product matrix when considering
        only the ground state."""
        minima_pair_displacement_vectors = (
            2.0 * np.pi * relevant_unit_cell_vectors[(minimum_index, minimum_index)]
        )
        transfer_function = partial(
            self._one_state_local_transfer, Xi, Xi_inv, exp_i_phi_j, EC_mat_t
        )
        transfer = np.sum(
            [
                self._exp_product_coefficient(displacement_vector, Xi_inv)
                * transfer_function(
                    displacement_vector, minimum_location, minimum_location
                )
                for displacement_vector in minima_pair_displacement_vectors
            ]
        )
        # Need to include displacement_vector + minimum_location -
        # minimum_location for completeness
        inner_product = np.sum(
            [
                self._exp_product_coefficient(displacement_vector, Xi_inv)
                for displacement_vector in minima_pair_displacement_vectors
            ]
        )
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
        sorted_minima_dict = self.sorted_minima_dict

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

        wavefunction_amplitudes = np.zeros_like(
            np.outer(phi_1_vec, phi_2_vec), dtype=np.complex128
        ).T

        for minimum_index, minimum_location in sorted_minima_dict.items():
            unit_cell_vectors = itertools.product(
                np.arange(
                    -self.maximum_unit_cell_vector_length,
                    self.maximum_unit_cell_vector_length + 1,
                ),
                repeat=dim_periodic,
            )
            unit_cell_vector = next(unit_cell_vectors, -1)
            while unit_cell_vector != -1:
                displacement_vector = (
                    2.0
                    * np.pi
                    * np.concatenate((np.zeros(dim_extended), unit_cell_vector))
                )
                phi_offset = displacement_vector - minimum_location
                state_amplitudes = self.state_amplitudes_function(
                    minimum_index, evecs, which
                )
                phi_1_with_offset = phi_1_vec + phi_offset[0]
                phi_2_with_offset = phi_2_vec + phi_offset[1]
                normal_mode_1 = np.add.outer(
                    Xi_inv[0, 0] * phi_1_with_offset, Xi_inv[0, 1] * phi_2_with_offset
                )
                normal_mode_2 = np.add.outer(
                    Xi_inv[1, 0] * phi_1_with_offset, Xi_inv[1, 1] * phi_2_with_offset
                )
                wavefunction_amplitudes += (
                    self.wavefunction_amplitudes_function(
                        state_amplitudes, normal_mode_1, normal_mode_2
                    )
                    * normalization
                    * np.exp(-1j * np.dot(self.nglist, phi_offset))
                )
                unit_cell_vector = next(unit_cell_vectors, -1)

        grid2d = discretization.GridSpec(
            np.asarray(
                [
                    [phi_1_grid.min_val, phi_1_grid.max_val, phi_1_grid.pt_count],
                    [phi_2_grid.min_val, phi_2_grid.max_val, phi_2_grid.pt_count],
                ]
            )
        )

        wavefunction_amplitudes = standardize_phases(wavefunction_amplitudes)

        return storage.WaveFunctionOnGrid(grid2d, wavefunction_amplitudes)

    def state_amplitudes_function(self, i, evecs, which):
        """Helper method for wavefunction returning the matrix of state amplitudes
        that can be overridden in the case of a global excitation cutoff"""
        num_states_per_min = self.number_states_per_minimum()
        return np.real(
            np.reshape(
                evecs[i * num_states_per_min : (i + 1) * num_states_per_min, which],
                (self.num_exc + 1, self.num_exc + 1),
            )
        )

    def wavefunction_amplitudes_function(
        self, state_amplitudes, normal_mode_1, normal_mode_2
    ):
        """Helper method for wavefunction returning wavefunction amplitudes
        that can be overridden in the case of a global excitation cutoff"""
        return np.sum(
            [
                plot.multiply_two_harm_osc_functions(
                    s1, s2, normal_mode_1, normal_mode_2
                )
                * state_amplitudes[s1, s2]
                for s2 in range(self.num_exc + 1)
                for s1 in range(self.num_exc + 1)
            ],
            axis=0,
        ).T

    # def plot_wavefunction(
    #     self, esys=None, which=0, mode="abs", zero_calibrate=True, **kwargs
    # ):
    #     """Plots 2d phase-basis wave function.
    #
    #     Parameters
    #     ----------
    #     esys: ndarray, ndarray
    #         eigenvalues, eigenvectors as obtained from `.eigensystem()`
    #     which: int, optional
    #         index of wave function to be plotted (default value = (0)
    #     mode: str, optional
    #         choices as specified in `constants.MODE_FUNC_DICT`
    #         (default value = 'abs_sqr')
    #     zero_calibrate: bool, optional
    #         if True, colors are adjusted to use zero wavefunction amplitude as the
    #         neutral color in the palette
    #     **kwargs:
    #         plot options
    #
    #     Returns
    #     -------
    #     Figure, Axes
    #     """
    #     amplitude_modifier = constants.MODE_FUNC_DICT[mode]
    #     wavefunction = self.wavefunction(esys, which=which)
    #     wavefunction.amplitudes = amplitude_modifier(wavefunction.amplitudes)
    #     return plot.wavefunction2d(
    #         wavefunction,
    #         zero_calibrate=zero_calibrate,
    #         xlabel=r"$\phi$",
    #         ylabel=r"$\theta$",
    #         **kwargs
    #     )
