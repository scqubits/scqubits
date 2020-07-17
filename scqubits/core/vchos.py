import itertools
import warnings

import numpy as np
from scipy.linalg import LinAlgError, expm, inv, eigh
import scipy.constants as const
from numpy.linalg import matrix_power

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.misc import kron_matrix_list
from scqubits.utils.spectrum_utils import order_eigensystem, solve_generalized_eigenvalue_problem_with_QZ


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

class VCHOS(base.QubitBaseClass, serializers.Serializable):
    def __init__(self, EJlist, nglist, flux, kmax, num_exc=None):
        self.e = np.sqrt(4.0*np.pi*const.alpha)
        self.Z0 = 1./(2*self.e)**2
        self.Phi0 = 1./(2*self.e)
        self.nearest_neighbor_cutoff = 180.0
        self.EJlist = EJlist
        self.nglist = nglist
        self.flux = flux
        self.kmax = kmax
        self.num_exc = num_exc
        # This must be set in the individual qubit class and
        # specifies the structure of the boundary term
        self.boundary_coeffs = np.array([])

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def nonfit_params():
        return []

    def potential(self, phi_array):
        """
        Potential evaluated at the location specified by phi_array

        Parameters
        ----------
        phi_array: ndarray
            float value of the phase variable `phi`

        Returns
        -------
        float
        """
        dim = self.number_degrees_freedom()
        pot_sum = np.sum([- self.EJlist[j] * np.cos(phi_array[j]) for j in range(dim)])
        pot_sum += (- self.EJlist[-1]*np.cos(np.sum([self.boundary_coeffs[i] * phi_array[i]
                                                     for i in range(dim)]) + 2*np.pi*self.flux))
        return pot_sum

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
        dim = self.number_degrees_freedom()
        gmat = np.zeros((dim, dim))
        min_loc = self.sorted_minima()[i]
        gamma_list = self.EJlist / self.Phi0 ** 2

        gamma_diag = np.diag([gamma_list[j]*np.cos(min_loc[j]) for j in range(dim)])
        gmat += gamma_diag

        min_loc_bound_sum = np.sum([self.boundary_coeffs[j]*min_loc[j] for j in range(dim)])
        for j in range(dim):
            for k in range(dim):
                gmat[j, k] += (gamma_list[-1]*self.boundary_coeffs[j]*self.boundary_coeffs[k]
                               * np.cos(min_loc_bound_sum + 2*np.pi*self.flux))
        return gmat

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
        return np.diag(np.sqrt(omega_squared))

    def oscillator_lengths(self, i):
        """Return oscillator lengths of the mode frequencies for a given minimum

        Parameters
        ----------
        i: int
            integer specifying which minimum to linearize around, 0<=i<= total number of minima

        Returns
        -------
        ndarray
        """
        dim = self.number_degrees_freedom()
        omega_squared, normal_mode_eigenvectors = self.eigensystem_normal_modes(i)
        omega = np.sqrt(omega_squared)
        diag_norm = np.matmul(normal_mode_eigenvectors.T, normal_mode_eigenvectors)
        normalized_eigenvectors = np.array([normal_mode_eigenvectors[:, mu]
                                            / np.sqrt(diag_norm[mu, mu]) for mu in range(dim)]).T
        C_matrix = self.build_capacitance_matrix()
        C_matrix_diagonal = np.matmul(normalized_eigenvectors.T, np.matmul(C_matrix, normalized_eigenvectors))
        EC_matrix_diagonal = 0.5*self.e**2*np.diag(C_matrix_diagonal)**(-1)
        oscillator_lengths = np.array([np.sqrt(8*EC_matrix_diagonal[mu]/omega[mu]) for mu in range(len(omega))])
        return oscillator_lengths

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
        Xi_matrix = np.array([normal_mode_eigenvectors[:, i]*(omega_squared[i]) ** (-1 / 4)
                          * np.sqrt(1. / self.Z0) for i in range(len(omega_squared))]).T
        return Xi_matrix

    def a_operator(self, mu):
        """Return the lowering operator associated with the mu^th d.o.f. in the full Hilbert space

        Parameters
        ----------
        mu: int
            which degree of freedom, 0<=mu<=self.number_degrees_freedom()

        Returns
        -------
        ndarray
        """
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a, k=1)
        return self._full_o([a_mat], [mu])

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
        return (self.num_exc + 1) ** self.number_degrees_freedom()

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension.
        """
        return int(len(self.sorted_minima())*self.number_states_per_minimum())

    def _find_nearest_neighbors_for_each_minimum(self):
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
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                all_neighbors = itertools.product(np.arange(-self.kmax, self.kmax + 1),
                                                  repeat=self.number_degrees_freedom())
                filtered_neighbors = itertools.filterfalse(lambda e: self._filter_neighbors(e, minima_diff, Xi_inv),
                                                           all_neighbors)
                neighbor = next(filtered_neighbors, -1)
                while neighbor != -1:
                    nearest_neighbors_single_minimum.append(neighbor)
                    neighbor = next(filtered_neighbors, -1)
                nearest_neighbors.append(nearest_neighbors_single_minimum)
                nearest_neighbors_single_minimum = []
        return nearest_neighbors

    def _filter_neighbors(self, neighbor, minima_diff, Xi_inv):
        """
        Want to eliminate periodic continuation terms that are irrelevant, i.e.,
        they add nothing to the Hamiltonian. These can be identified as each term
        is suppressed by a gaussian exponential factor. If the argument np.dot(dpkX, dpkX)
        of the exponential is greater than 180.0, this results in a suppression of ~10**(-20),
        and so can be safely neglected.
        """
        phi_neighbor = 2.0*np.pi*np.array(neighbor)
        dpkX = np.matmul(Xi_inv, phi_neighbor + minima_diff)
        prod = np.dot(dpkX, dpkX)
        return prod > self.nearest_neighbor_cutoff

    def _build_premultiplied_a_and_a_dagger(self):
        dim = self.number_degrees_freedom()
        a = np.array([self.a_operator(i) for i in range(dim)])
        a_a = np.array([np.matmul(self.a_operator(i), self.a_operator(i)) for i in range(dim)])
        a_dagger_a = np.array([np.matmul(self.a_operator(i).T, self.a_operator(i)) for i in range(dim)])
        return a, a_a, a_dagger_a

    def _build_single_exp_i_phi_j_operator(self, j):
        Xi = self.Xi_matrix()
        dim = self.number_degrees_freedom()
        if j == dim:
            exp_i_phi_j_a_component = expm(np.sum([self.boundary_coeffs[i]
                                                   * 1j*Xi[i, k]*self.a_operator(k)/np.sqrt(2.0)
                                                   for i in range(dim) for k in range(dim)], axis=0))
            BCH_factor = self._BCH_factor_for_potential_boundary()
        else:
            exp_i_phi_j_a_component = expm(np.sum([1j*Xi[j, k]*self.a_operator(k)/np.sqrt(2.0)
                                                   for k in range(dim)], axis=0))
            BCH_factor = self._BCH_factor_for_junction(j)
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a_component.T
        return BCH_factor*np.matmul(exp_i_phi_j_a_dagger_component, exp_i_phi_j_a_component)

    def _build_all_exp_i_phi_j_operators(self):
        return np.array([self._build_single_exp_i_phi_j_operator(j) for j in range(self.number_degrees_freedom()+1)])

    def _build_exponentiated_translation_operators(self, minima_diff, Xi_inv):
        """In general this is the costliest part of the code (expm is quite slow)"""
        dim = self.number_degrees_freedom()
        exp_a_list = np.array([expm(np.sum([(2.0*np.pi*Xi_inv.T[i, j]/np.sqrt(2.0))*self.a_operator(j)
                                            for j in range(dim)], axis=0)) for i in range(dim)])
        exp_a_minima_difference = expm(np.sum([(minima_diff[i]*Xi_inv.T[i, j]/np.sqrt(2.0))*self.a_operator(j)
                                               for i in range(dim) for j in range(dim)], axis=0))
        return exp_a_list, exp_a_minima_difference

    def _translation_operator_builder(self, exp_a_list_and_minima_difference, neighbor):
        """Build translation operators using matrix_power rather than the more costly expm"""
        dim = self.number_degrees_freedom()
        exp_a_list, exp_a_minima_difference = exp_a_list_and_minima_difference
        translation_op_a_dagger = self.identity()
        translation_op_a = self.identity()
        for j in range(dim):
            translation_op_a_dagger = np.matmul(translation_op_a_dagger, matrix_power(exp_a_list[j].T, neighbor[j]))
        for j in range(dim):
            translation_op_a = np.matmul(translation_op_a, matrix_power(exp_a_list[j], -neighbor[j]))
        translation_op_a_dagger = np.matmul(exp_a_minima_difference.T, translation_op_a_dagger)
        translation_op_a = np.matmul(translation_op_a, inv(exp_a_minima_difference))
        return translation_op_a_dagger, translation_op_a

    def _exp_product_coefficient(self, delta_phi, Xi_inv):
        """Overall multiplicative factor, including offset charge, Gaussian suppression factor"""
        delta_phi_rotated = np.matmul(Xi_inv, delta_phi)
        return (np.exp(-1j*np.dot(self.nglist, delta_phi))
                * np.exp(-0.25*np.dot(delta_phi_rotated, delta_phi_rotated)))

    def _BCH_factor_for_potential_boundary(self):
        Xi = self.Xi_matrix()
        dim = self.number_degrees_freedom()
        return np.exp(-0.25*np.sum([self.boundary_coeffs[j]*self.boundary_coeffs[k]
                                    * np.dot(Xi[j, :], Xi.T[:, k]) for j in range(dim) for k in range(dim)]))

    def _BCH_factor_for_junction(self, j):
        Xi = self.Xi_matrix()
        return np.exp(-0.25*np.dot(Xi[j, :], Xi.T[:, j]))

    def hamiltonian(self):
        """
        Returns
        -------
        ndarray
            Returns the Hamiltonian matrix
        """
        return self.kinetic_matrix() + self.potential_matrix()

    def kinetic_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the kinetic energy matrix
        """
        nearest_neighbors = self._find_nearest_neighbors_for_each_minimum()
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger()
        kinetic_function = self._kinetic_contribution_to_hamiltonian(premultiplied_a_and_a_dagger,
                                                                     inv(self.Xi_matrix()))
        return self.wrapper_for_operator_construction(kinetic_function, nearest_neighbors=nearest_neighbors)

    def potential_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the potential energy matrix
        """
        nearest_neighbors = self._find_nearest_neighbors_for_each_minimum()
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators()
        potential_function = self._potential_contribution_to_hamiltonian(exp_i_phi_list)
        return self.wrapper_for_operator_construction(potential_function, nearest_neighbors=nearest_neighbors)

    def _kinetic_contribution_to_hamiltonian(self, premultiplied_a_and_a_dagger, Xi_inv):
        """Calculating products of a, a_dagger operators is costly,
        as well as repeatedly calculating Xi (or Xi_inv) which is why they are
        passed to this function in this way rather than calculated below"""
        def _inner_kinetic_c_t_h(delta_phi, phi_bar):
            a, a_a, a_dagger_a = premultiplied_a_and_a_dagger
            EC_mat_transformed = np.matmul(Xi_inv, np.matmul(self.build_EC_matrix(), Xi_inv.T))
            delta_phi_rotated = np.matmul(Xi_inv, delta_phi)
            kinetic_matrix = np.sum([(-0.5*4*a_a[i] - 0.5*4*a_a[i].T + 0.5*8*a_dagger_a[i]
                                      - 4*(a[i] - a[i].T)*delta_phi_rotated[i]/np.sqrt(2.0))
                                     * EC_mat_transformed[i, i]
                                     for i in range(self.number_degrees_freedom())], axis=0)
            identity_coefficient = 0.5*4*np.trace(EC_mat_transformed)
            identity_coefficient += -0.25*4*np.matmul(delta_phi_rotated,
                                                      np.matmul(EC_mat_transformed, delta_phi_rotated))
            kinetic_matrix += identity_coefficient*self.identity()
            return kinetic_matrix
        return _inner_kinetic_c_t_h

    def _potential_contribution_to_hamiltonian(self, exp_i_phi_list):
        """Calculating exp_i_phi operators is costly, which is why it is
        passed to this function in this way rather than calculated below"""
        def _inner_potential_c_t_h(delta_phi, phi_bar):
            dim = self.number_degrees_freedom()
            exp_i_phi_list_without_boundary = np.array([exp_i_phi_list[i]*np.exp(1j * phi_bar[i])
                                                        for i in range(dim)])
            exp_i_sum_phi = (exp_i_phi_list[-1] * np.exp(1j*2.0*np.pi*self.flux)
                             * np.prod([np.exp(1j * self.boundary_coeffs[i] * phi_bar[i]) for i in range(dim)]))
            potential_matrix = np.sum([-0.5*self.EJlist[junction]
                                       * (exp_i_phi_list_without_boundary[junction]
                                          + exp_i_phi_list_without_boundary[junction].conjugate())
                                       for junction in range(dim)], axis=0)
            potential_matrix += -0.5*self.EJlist[-1]*(exp_i_sum_phi + exp_i_sum_phi.conjugate())
            potential_matrix += np.sum(self.EJlist)*self.identity()
            return potential_matrix
        return _inner_potential_c_t_h

    def inner_product_matrix(self):
        """
        Returns
        -------
        ndarray
            Returns the inner product matrix
        """
        nearest_neighbors = self._find_nearest_neighbors_for_each_minimum()
        return self.wrapper_for_operator_construction(self._inner_product_operator,
                                                      nearest_neighbors=nearest_neighbors)

    # TODO find a way to eliminate the arguments here, as they are unnecessary
    def _inner_product_operator(self, delta_phi, phi_bar):
        return self.identity()

    def wrapper_for_operator_construction(self, specific_function, nearest_neighbors=None):
        """This function is the meat of the VCHOS method. Any operator whose matrix
        elements we want (the Hamiltonian and inner product matrices are obvious examples)
        can be passed to this function, and the matrix elements of that operator
        will be returned.

        Parameters
        ----------
        specific_function: method
            function that takes two arguments (delta_phi, phi_bar) and returns the
            relevant operator with dimension NxN, where N is the number of states
            displaced into each minimum. For instance to find the inner product matrix,
            we use the function self._inner_product_operator(delta_phi, phi_bar) -> self.identity
        nearest_neighbors: self._find_nearest_neighbors_for_each_minimum()
            list that encodes the nearest neighbors relevant when examining matrix elements
            between states in inequivalent minima.

        Returns
        -------
        ndarray
        """
        if nearest_neighbors is None:
            nearest_neighbors = self._find_nearest_neighbors_for_each_minimum()
        Xi_inv = inv(self.Xi_matrix())
        minima_list = self.sorted_minima()
        hilbertdim = self.hilbertdim()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                exp_a_list_and_minima_difference = self._build_exponentiated_translation_operators(minima_diff, Xi_inv)
                for neighbor in nearest_neighbors[counter]:
                    phi_neighbor = 2.0*np.pi*np.array(neighbor)
                    delta_phi = phi_neighbor + minima_diff
                    phi_bar = 0.5*(phi_neighbor + (minima_m + minima_list[p]))
                    exp_prod_coefficient = self._exp_product_coefficient(delta_phi, Xi_inv)
                    exp_a_dagger, exp_a = self._translation_operator_builder(exp_a_list_and_minima_difference, neighbor)
                    matrix_element = exp_prod_coefficient*specific_function(delta_phi, phi_bar)
                    matrix_element = np.matmul(exp_a_dagger, np.matmul(matrix_element, exp_a))
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

    def _full_o(self, operators, indices):
        """Return operator in the full Hilbert space"""
        i_o = np.eye(self.num_exc + 1)
        i_o_list = [i_o for _ in range(self.number_degrees_freedom())]
        product_list = i_o_list[:]
        oi_list = zip(operators, indices)
        for oi in oi_list:
            product_list[oi[1]] = oi[0]
        full_op = kron_matrix_list(product_list)
        return full_op

    def _efficient_construction_of_hamiltonian_and_inner_product(self):
        nearest_neighbors = self._find_nearest_neighbors_for_each_minimum()
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators()
        premultiplied_a_and_a_dagger = self._build_premultiplied_a_and_a_dagger()
        kinetic_function = self._kinetic_contribution_to_hamiltonian(premultiplied_a_and_a_dagger,
                                                                     inv(self.Xi_matrix()))
        potential_function = self._potential_contribution_to_hamiltonian(exp_i_phi_list)

        def kinetic_plus_potential(delta_phi, phi_bar):
            return kinetic_function(delta_phi, phi_bar) + potential_function(delta_phi, phi_bar)
        hamiltonian_matrix = self.wrapper_for_operator_construction(kinetic_plus_potential,
                                                                    nearest_neighbors=nearest_neighbors)
        inner_product_matrix = self.wrapper_for_operator_construction(self._inner_product_operator,
                                                                      nearest_neighbors=nearest_neighbors)
        return hamiltonian_matrix, inner_product_matrix

    def _evals_calc(self, evals_count):
        hamiltonian_matrix, inner_product_matrix = self._efficient_construction_of_hamiltonian_and_inner_product()
        try:
            evals = eigh(hamiltonian_matrix, b=inner_product_matrix,
                         eigvals_only=True, eigvals=(0, evals_count - 1))
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            evals = solve_generalized_eigenvalue_problem_with_QZ(hamiltonian_matrix, inner_product_matrix,
                                                                 evals_count, eigvals_only=True)
        return evals

    def _esys_calc(self, evals_count):
        hamiltonian_matrix, inner_product_matrix = self._efficient_construction_of_hamiltonian_and_inner_product()
        try:
            evals, evecs = eigh(hamiltonian_matrix, b=inner_product_matrix,
                                eigvals_only=False, eigvals=(0, evals_count - 1))
            evals, evecs = order_eigensystem(evals, evecs)
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm")
            evals, evecs = solve_generalized_eigenvalue_problem_with_QZ(hamiltonian_matrix, inner_product_matrix,
                                                                        evals_count, eigvals_only=False)
        return evals, evecs

    def _check_if_new_minima(self, new_minima, minima_holder):
        """
        Helper function for find_minima, checking if new_minima is
        indeed a minimum and is already represented in minima_holder. If so,
        _check_if_new_minima returns False.
        """
        if -self.potential(new_minima) <= 0:  # maximum or saddle point then, not a minimum
            return False
        new_minima_bool = True
        for minima in minima_holder:
            diff_array = minima - new_minima
            diff_array_reduced = np.array([np.mod(x, 2*np.pi) for x in diff_array])
            elem_bool = True
            for elem in diff_array_reduced:
                # if every element is zero or 2pi, then we have a repeated minima
                elem_bool = elem_bool and (np.allclose(elem, 0.0, atol=1e-3)
                                           or np.allclose(elem, 2*np.pi, atol=1e-3))
            if elem_bool:
                new_minima_bool = False
                break
        return new_minima_bool

    # The following four methods must be overridden in child classes
    def sorted_minima(self):
        return []

    def build_capacitance_matrix(self):
        return []

    def build_EC_matrix(self):
        return []

    def number_degrees_freedom(self):
        return 0
