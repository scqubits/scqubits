import itertools
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError
from scipy.special import factorial
import scipy.constants as const
from numpy.linalg import matrix_power

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.fix_heiberger import fixheiberger
from scqubits.utils.misc import kron_matrix_list
from scqubits.utils.spectrum_utils import order_eigensystem


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
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.Z0 = 1. / (2 * self.e) ** 2
        self.Phi0 = 1. / (2 * self.e)
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

    def potential(self, phiarray):
        """
        Potential evaluated at the location specified by phiarray

        Parameters
        ----------
        phiarray: ndarray
            float value of the phase variable `phi`

        Returns
        -------
        float
        """
        pot_sum = np.sum([-self.EJlist[j] * np.cos(phiarray[j])
                          for j in range(self.number_degrees_freedom())])
        pot_sum += (-self.EJlist[-1]
                    * np.cos(np.sum([self.boundary_coeffs[i] * phiarray[i]
                                     for i in range(self.number_degrees_freedom())]) + 2 * np.pi * self.flux))
        return pot_sum

    def build_gamma_matrix(self, i):
        """
        Return linearized potential matrix
        
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
        gmat = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))

        min_loc = self.sorted_minima()[i]
        gamma_list = self.EJlist / self.Phi0 ** 2

        gamma_diag = np.diag([gamma_list[j] * np.cos(min_loc[j])
                              for j in range(self.number_degrees_freedom())])
        gmat += gamma_diag

        min_loc_bound_sum = np.sum([self.boundary_coeffs[j] * min_loc[j]
                                    for j in range(self.number_degrees_freedom())])
        for j in range(self.number_degrees_freedom()):
            for k in range(self.number_degrees_freedom()):
                gmat[j, k] += (gamma_list[-1] * self.boundary_coeffs[j] * self.boundary_coeffs[k]
                               * np.cos(min_loc_bound_sum + 2 * np.pi * self.flux))

        return gmat

    def _eigensystem_normal_modes(self, i):
        """Return squared normal mode frequencies, matrix of eigenvectors"""
        Cmat = self.build_capacitance_matrix()
        gmat = self.build_gamma_matrix(i)

        omegasq, eigvec = sp.linalg.eigh(gmat, b=Cmat)
        return omegasq, eigvec

    def omegamat(self, i):
        """Return a diagonal matrix of the normal mode frequencies of a given minimim """
        omegasq, _ = self._eigensystem_normal_modes(i)
        return np.diag(np.sqrt(omegasq))

    def oscillator_lengths(self, i):
        """Return oscillator lengths of the mode frequencies for a given minimum"""
        omegasq, eigvec = self._eigensystem_normal_modes(i)
        omega = np.sqrt(omegasq)
        diag_norm = np.matmul(eigvec.T, eigvec)
        norm_eigvec = np.array([eigvec[:, mu] / np.sqrt(diag_norm[mu, mu])
                                for mu in range(self.number_degrees_freedom())]).T
        Cmat = self.build_capacitance_matrix()
        Cmat_diag = np.matmul(norm_eigvec.T, np.matmul(Cmat, norm_eigvec))
        ECmat_diag = 0.5 * self.e ** 2 * np.diag(Cmat_diag) ** (-1)
        oscillator_lengths = np.array([np.sqrt(8 * ECmat_diag[mu] / omega[mu]) for mu in range(len(omega))])
        return oscillator_lengths

    def Xi_matrix(self):
        """Construct the Xi matrix, encoding the oscillator lengths of each dimension"""
        omegasq, eigvec = self._eigensystem_normal_modes(0)
        # We introduce a normalization such that \Xi^T C \Xi = \Omega^{-1}/Z0
        Ximat = np.array([eigvec[:, i] * (omegasq[i]) ** (-1 / 4)
                          * np.sqrt(1. / self.Z0) for i in range(len(omegasq))]).T

        return Ximat

    def a_operator(self, mu):
        """Return the lowering operator associated with the mu^th d.o.f. in the full Hilbert space"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a, k=1)
        return self._full_o([a_mat], [mu])

    def _identity(self):
        return np.eye(int(self.number_states_per_minimum()))

    def number_states_per_minimum(self):
        return (self.num_exc + 1) ** self.number_degrees_freedom()

    def hilbertdim(self):
        """Return N if the size of the Hamiltonian matrix is NxN"""
        return int(len(self.sorted_minima()) * self.number_states_per_minimum())

    def _find_k_values_for_different_minima(self):
        """
        We have found that specifically this part of the code is quite slow, that
        is finding the relevant nearest neighbor, next nearest neighbor, etc. lattice vectors
        that meaningfully contribute. This is a calculation that previously had to be done
        for the kinetic, potential and inner product matrices separately, even though
        the results were the same for all three matrices. This helper function allows us to only
        do it once.
        """
        Xi_inv = sp.linalg.inv(self.Xi_matrix())
        minima_list = self.sorted_minima()
        wrapper_klist_holder = []
        klist_holder = []
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.number_degrees_freedom())
                klist = itertools.filterfalse(lambda e: self._filter_jkvals(e, minima_diff, Xi_inv), klist)
                jkvals = next(klist, -1)
                while jkvals != -1:
                    klist_holder.append(jkvals)
                    jkvals = next(klist, -1)
                wrapper_klist_holder.append(klist_holder)
                klist_holder = []
        return wrapper_klist_holder

    def _filter_jkvals(self, jkvals, minima_diff, Xi_inv):
        """
        Want to eliminate periodic continuation terms that are irrelevant, i.e.,
        they add nothing to the Hamiltonian. These can be identified as each term
        is suppressed by a gaussian exponential factor. If the argument np.dot(dpkX, dpkX)
        of the exponential is greater than 180.0, this results in a suppression of ~10**(-20),
        and so can be safely neglected.
        """
        phik = 2.0 * np.pi * np.array(jkvals)
        dpkX = np.matmul(Xi_inv, phik + minima_diff)
        prod = np.dot(dpkX, dpkX)
        return prod > self.nearest_neighbor_cutoff

    def _build_single_exp_i_phi_j_operator(self, j):
        Xi = self.Xi_matrix()
        if j == self.number_degrees_freedom() - 1:
            exp_i_phi_j_a_component = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                             1j * Xi[j, k] * self.a_operator(k) / np.sqrt(2.0)
                                                             for j in range(self.number_degrees_freedom())
                                                             for k in range(self.number_degrees_freedom())], axis=0))
        else:
            exp_i_phi_j_a_component = sp.linalg.expm(np.sum([1j * Xi[j, k] * self.a_operator(k) / np.sqrt(2.0)
                                                             for k in range(self.number_degrees_freedom())], axis=0))
        exp_i_phi_j_a_dagger_component = exp_i_phi_j_a_component.T
        return np.matmul(exp_i_phi_j_a_dagger_component, exp_i_phi_j_a_component)

    def _build_all_exp_i_phi_j_operators(self):
        """
        as well as the exp(i\phi_{j}) operators for the potential
        :return:
        """
        return np.array([self._build_single_exp_i_phi_j_operator(j) for j in range(self.number_degrees_freedom())])

    def _build_exponentiated_translation_operators(self, minima_diff):
        """
        This routine builds the translation operators necessary for periodic continuation
        """
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        exp_a_list = np.array([sp.linalg.expm(np.sum([2.0 * np.pi * Xi_inv.T[i, j] / np.sqrt(2.0) * self.a_operator(j)
                                                      for j in range(self.number_degrees_freedom())], axis=0))
                               for i in range(self.number_degrees_freedom())])
        exp_a_minima_difference = sp.linalg.expm(np.sum([minima_diff[i] * Xi_inv.T[i, j] / np.sqrt(2.0)
                                                         * self.a_operator(j)
                                                         for i in range(self.number_degrees_freedom())
                                                         for j in range(self.number_degrees_freedom())], axis=0))
        return exp_a_list, exp_a_minima_difference

    def _translation_operator_builder(self, exp_a_list_and_minima_difference, jkvals):
        """
        Build translation operators using matrix_power rather than the
        more costly expm
        """
        exp_a_list, exp_a_minima_difference = exp_a_list_and_minima_difference
        translation_op_a_dagger = self._identity()
        for j in range(self.number_degrees_freedom()):
            translation_op_a_dagger_temp = matrix_power(exp_a_list[j].T, jkvals[j])
            translation_op_a_dagger = np.matmul(translation_op_a_dagger, translation_op_a_dagger_temp)

        translation_op_a = self._identity()
        for j in range(self.number_degrees_freedom()):
            translation_op_a_temp = matrix_power(exp_a_list[j], -jkvals[j])
            translation_op_a = np.matmul(translation_op_a, translation_op_a_temp)

        translation_op_a_dagger = np.matmul(exp_a_minima_difference.T, translation_op_a_dagger)
        translation_op_a = np.matmul(translation_op_a, sp.linalg.inv(exp_a_minima_difference))

        return translation_op_a_dagger, translation_op_a

    def _exp_prod_coeff(self, delta_phi_kpm):
        """
        Overall multiplicative factor. Includes offset charge,
        Gaussian suppression factor
        """
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        delta_phi_kpm_rotated = np.matmul(Xi_inv, delta_phi_kpm)
        return (np.exp(-1j * np.dot(self.nglist, delta_phi_kpm))
                * np.exp(-0.25 * np.dot(delta_phi_kpm_rotated, delta_phi_kpm_rotated)))

    def _exp_prod_bound_coeff(self):
        Xi = self.Xi_matrix()
        exp_prod_boundary_coeff = np.exp(-0.25 * np.sum([self.boundary_coeffs[j]
                                                         * self.boundary_coeffs[k]
                                                         * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                                         for j in range(self.number_degrees_freedom())
                                                         for k in range(self.number_degrees_freedom())]))
        return exp_prod_boundary_coeff

    def hamiltonian(self):
        wrapper_klist = self._find_k_values_for_different_minima()
        return self.wrapper_for_operator_construction(self._add_kinetic_and_potential_matrices,
                                                      wrapper_klist=wrapper_klist)

    def _add_kinetic_and_potential_matrices(self, delta_phi_km, phibar_kpm):
        exp_i_phi_list = self._build_all_exp_i_phi_j_operators()
        inner_potential_c_t_h_function = self._potential_contribution_to_hamiltonian(exp_i_phi_list)
        return self._kinetic_contribution_to_hamiltonian(delta_phi_km) + inner_potential_c_t_h_function(phibar_kpm)

    def _kinetic_contribution_to_hamiltonian(self, delta_phi_kpm):
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat_transformed = np.matmul(Xi_inv, np.matmul(self.build_EC_matrix(), Xi_inv.T))
        delta_phi_kpm_rotated = np.matmul(Xi_inv, delta_phi_kpm)
        kinetic_matrix = np.sum([- 0.5 * self.a_operator(i) * 4 * EC_mat_transformed[i, j] * self.a_operator(j)
                                 - 0.5 * self.a_operator(i).T * 4 * EC_mat_transformed[i, j] * self.a_operator(j).T
                                 + 0.5 * self.a_operator(i).T * 8 * EC_mat_transformed[i, j] * self.a_operator(j)
                                 + ((self.a_operator(i) - self.a_operator(i).T)*4*EC_mat_transformed[i, j]
                                    * delta_phi_kpm_rotated[j]/np.sqrt(2.0))
                                 for i in range(self.number_degrees_freedom())
                                 for j in range(self.number_degrees_freedom())], axis=0)
        identity_coefficient = 0.5*4*np.trace(EC_mat_transformed)
        identity_coefficient += -0.25*np.matmul(delta_phi_kpm_rotated,
                                                np.matmul(4*EC_mat_transformed, delta_phi_kpm_rotated))
        kinetic_matrix += identity_coefficient*self._identity()
        return kinetic_matrix

    def _potential_contribution_to_hamiltonian(self, exp_i_phi_list):
        dim = self.number_degrees_freedom()
        Xi = self.Xi_matrix()
        exp_prod_boundary_coeff = self._exp_prod_bound_coeff()

        def _inner_potential_c_t_h(phibar_kpm):
            exp_i_phi_list_without_boundary = np.array([exp_i_phi_list[i] * np.exp(1j * phibar_kpm[i])
                                                        for i in range(dim)])
            exp_i_sum_phi = (exp_i_phi_list[-1] * np.exp(1j * 2.0 * np.pi * self.flux)
                             * np.prod([np.exp(1j * self.boundary_coeffs[i] * phibar_kpm[i]) for i in range(dim)]))
            potential_matrix = np.sum([-0.5*self.EJlist[junction]
                                       * (exp_i_phi_list_without_boundary[junction]
                                          + exp_i_phi_list_without_boundary[junction].conjugate())
                                       * np.exp(-.25 * np.dot(Xi[junction, :], np.transpose(Xi)[:, junction]))
                                       for junction in range(dim)], axis=0)
            potential_matrix += (-0.5 * self.EJlist[-1] * (exp_i_sum_phi + exp_i_sum_phi.conjugate())
                                 * exp_prod_boundary_coeff)
            potential_matrix += np.sum(self.EJlist) * self._identity()
            return potential_matrix
        return _inner_potential_c_t_h

    def inner_product(self):
        wrapper_klist = self._find_k_values_for_different_minima()
        return self.wrapper_for_operator_construction(self._inner_product_operator, wrapper_klist=wrapper_klist)

    # TODO find a way to eliminate the arguments here, as they are unnecessary
    def _inner_product_operator(self, delta_phi_kpm, phibar_kpm):
        return self._identity()

    def wrapper_for_operator_construction(self, specific_function, wrapper_klist=None):
        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        minima_list = self.sorted_minima()
        hilbertdim = self.hilbertdim()
        num_states_min = self.number_states_per_minimum()
        operator_matrix = np.zeros((hilbertdim, hilbertdim), dtype=np.complex128)
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                exp_a_list_and_minima_difference = self._build_exponentiated_translation_operators(minima_diff)
                for jkvals in wrapper_klist[counter]:
                    phik = 2.0 * np.pi * np.array(jkvals)
                    delta_phi_kpm = phik + minima_diff
                    phibar_kpm = 0.5 * (phik + (minima_m + minima_list[p]))
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm)
                    exp_a_dagger, exp_a = self._translation_operator_builder(exp_a_list_and_minima_difference, jkvals)
                    matrix_element = exp_prod_coeff * specific_function(delta_phi_kpm, phibar_kpm)
                    matrix_element = np.matmul(exp_a_dagger, np.matmul(matrix_element, exp_a))
                    operator_matrix[m*num_states_min: (m + 1)*num_states_min,
                                    p*num_states_min: (p + 1)*num_states_min] += matrix_element
                counter += 1
        operator_matrix = self._populate_hermitian_matrix(operator_matrix)
        return operator_matrix

    def _populate_hermitian_matrix(self, mat):
        """
        Return a fully Hermitian matrix, assuming that the input matrix has been
        populated with the upper right blocks
        """
        minima_list = self.sorted_minima()
        dim = int(self.number_states_per_minimum())
        for m, minima_m in enumerate(minima_list):
            for p in range(m + 1, len(minima_list)):
                matrix_element = mat[m * dim: m * dim + dim, p * dim: p * dim + dim]
                mat[p * dim: p * dim + dim, m * dim: m * dim + dim] += matrix_element.conjugate().T
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

    def _evals_calc(self, evals_count):
        wrapper_klist = self._find_k_values_for_different_minima()
        hamiltonian_mat = self.wrapper_for_operator_construction(self._add_kinetic_and_potential_matrices,
                                                                 wrapper_klist=wrapper_klist)
        inner_product_mat = self.wrapper_for_operator_construction(self._inner_product_operator,
                                                                   wrapper_klist=wrapper_klist)
        try:
            evals = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat,
                                   eigvals_only=True, eigvals=(0, evals_count - 1))
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm and Fix-Heiberger, compare for convergence")
            evals = self._singular_inner_product_helper(hamiltonian_mat=hamiltonian_mat,
                                                        inner_product_mat=inner_product_mat,
                                                        evals_count=evals_count,
                                                        eigvals_only=True)
        return evals

    def _esys_calc(self, evals_count):
        wrapper_klist = self._find_k_values_for_different_minima()
        hamiltonian_mat = self.wrapper_for_operator_construction(self._add_kinetic_and_potential_matrices,
                                                                 wrapper_klist=wrapper_klist)
        inner_product_mat = self.wrapper_for_operator_construction(self._inner_product_operator,
                                                                   wrapper_klist=wrapper_klist)
        try:
            evals, evecs = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat,
                                          eigvals_only=False, eigvals=(0, evals_count - 1))
            evals, evecs = order_eigensystem(evals, evecs)
        except LinAlgError:
            warnings.warn("Singular inner product. Attempt QZ algorithm and Fix-Heiberger, compare for convergence")
            evals, evecs = self._singular_inner_product_helper(hamiltonian_mat=hamiltonian_mat,
                                                               inner_product_mat=inner_product_mat,
                                                               evals_count=evals_count,
                                                               eigvals_only=False)

        return evals, evecs

    def _singular_inner_product_helper(self, hamiltonian_mat, inner_product_mat, evals_count, eigvals_only=True):
        AA, BB, alpha, beta, Q, Z = sp.linalg.ordqz(hamiltonian_mat, inner_product_mat)
        a_max = np.max(np.abs(alpha))
        b_max = np.max(np.abs(beta))
        # filter ill-conditioned eigenvalues (alpha and beta values both small)
        alpha, beta = list(zip(*filter(lambda x: np.abs(x[0]) > 0.001 * a_max
                                       and np.abs(x[1]) > 0.001 * b_max, zip(alpha, beta))))
        evals_qz = np.array(alpha) / np.array(beta)
        evals_qz = np.sort(np.real(list(filter(lambda a: np.real(a) > 0, evals_qz))))[0: evals_count]
        evals_fh = fixheiberger(hamiltonian_mat, inner_product_mat, num_eigvals=evals_count, eigvals_only=True)
        assert (np.allclose(evals_qz, evals_fh))
        evals = evals_qz
        evecs = Z.T  # Need to ensure that this is the right way to produce eigenvectors
        if eigvals_only:
            return evals
        else:
            return evals, evecs

    # The following four methods must be overridden in child classes
    def sorted_minima(self):
        return []

    def build_capacitance_matrix(self):
        return []

    def build_EC_matrix(self):
        return []

    def number_degrees_freedom(self):
        return 0
