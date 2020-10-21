import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, inv, logm
import scipy.constants as const

from scqubits import VCHOSSqueezing
from scqubits.core.hashing import Hashing
from scqubits.core.vchos import VCHOS
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


class ZeroPiVCHOS(VCHOS, base.QubitBaseClass, serializers.Serializable):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=0, dCJ=0, truncated_dim=None, phi_extent=10, **kwargs):
        VCHOS.__init__(self, np.array([EJ, EJ]), np.array([0.0, ng]), flux,
                       number_degrees_freedom=2, number_periodic_degrees_freedom=1, **kwargs)
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ
        self.EC = EC
        self.ng = ng
        self.flux = flux
        self.phi_extent = phi_extent
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_

    @staticmethod
    def default_params():
        return {
            'EJ': 10.0,
            'EL': 0.04,
            'ECJ': 20.0,
            'EC': 0.04,
            'dEJ': 0.0,
            'dCJ': 0.0,
            'ng': 0.1,
            'flux': 0.23,
            'num_exc': 5,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['ng', 'flux', 'num_exc', 'truncated_dim']

    def build_capacitance_matrix(self):
        dim = self.number_degrees_freedom
        C_matrix = np.zeros((dim, dim))

        C = self.e ** 2 / (2. * self.EC)
        CJ = self.e ** 2 / (2. * self.ECJ)

        C_matrix[0, 0] = 2*CJ
        C_matrix[1, 1] = 2*(C + CJ)
        C_matrix[0, 1] = C_matrix[1, 0] = 2*self.dCJ

        return C_matrix

    def build_EC_matrix(self):
        C_matrix = self.build_capacitance_matrix()
        return 0.5 * self.e**2 * inv(C_matrix)

    def _check_second_derivative_positive(self, phi, theta):
        return (self.EL + 2 * self.EJ * np.cos(theta) * np.cos(phi - np.pi * self.flux)) > 0

    def potential(self, phi_theta_array):
        phi = phi_theta_array[0]
        theta = phi_theta_array[1]
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)
                + self.EL * phi ** 2 + 2.0 * self.EJ
                + self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def _append_new_minima(self, result, minima_holder):
        new_minimum = self._check_if_new_minima(result, minima_holder)
        if new_minimum:
            minima_holder.append(np.array([result[0], np.mod(result[1], 2 * np.pi)]))
        return minima_holder

    def find_minima(self):
        minima_holder = []
        guess = np.array([0.01, 0.01])
        result = minimize(self.potential, guess)
        minima_holder.append(np.array([result.x[0], np.mod(result.x[1], 2 * np.pi)]))
        for m in range(1, self.phi_extent):
            guess_positive_0 = np.array([np.pi * m, 0.0])
            guess_negative_0 = np.array([-np.pi * m, 0.0])
            guess_positive_pi = np.array([np.pi * m, np.pi])
            guess_negative_pi = np.array([-np.pi * m, np.pi])
            result_positive_0 = minimize(self.potential, guess_positive_0)
            result_negative_0 = minimize(self.potential, guess_negative_0)
            result_positive_pi = minimize(self.potential, guess_positive_pi)
            result_negative_pi = minimize(self.potential, guess_negative_pi)
            minima_holder = self._append_new_minima(result_positive_0.x, minima_holder)
            minima_holder = self._append_new_minima(result_negative_0.x, minima_holder)
            minima_holder = self._append_new_minima(result_positive_pi.x, minima_holder)
            minima_holder = self._append_new_minima(result_negative_pi.x, minima_holder)
        return minima_holder

    def build_gamma_matrix(self, minimum=0):
        dim = self.number_degrees_freedom
        gamma_matrix = np.zeros((dim, dim))
        min_loc = self.sorted_minima()[minimum]
        phi_location = min_loc[0]
        theta_location = min_loc[1]
        gamma_matrix[0, 0] = (2*self.EL + 2*self.EJ*np.cos(phi_location - np.pi*self.flux)*np.cos(theta_location)
                              - 2*self.dEJ*np.sin(theta_location)*np.sin(phi_location - np.pi*self.flux))
        gamma_matrix[1, 1] = (2*self.EJ*np.cos(phi_location - np.pi*self.flux)*np.cos(theta_location)
                              - 2*self.dEJ*np.sin(theta_location)*np.sin(phi_location - np.pi*self.flux))
        off_diagonal_term = (-2*self.EJ*np.sin(phi_location - np.pi*self.flux) * np.sin(theta_location)
                             + 2*self.EL*phi_location
                             + 2*self.dEJ*np.cos(theta_location)*np.cos(phi_location - np.pi*self.flux))
        gamma_matrix[1, 0] = off_diagonal_term
        gamma_matrix[0, 1] = off_diagonal_term
        return gamma_matrix/self.Phi0**2

    def _BCH_factor(self, j, Xi):
        dim = self.number_degrees_freedom
        boundary_coeffs = np.array([(-1)**j, 1])
        return np.exp(-0.25 * np.sum([boundary_coeffs[i]*boundary_coeffs[k]*np.dot(Xi[i, :], Xi.T[:, k])
                                      for i in range(dim) for k in range(dim)]))

    def _build_single_exp_i_phi_j_operator(self, j, Xi, a_operator_list):
        dim = self.number_degrees_freedom
        boundary_coeffs = np.array([(-1)**j, 1])
        exp_i_phi_theta_a_component = expm(np.sum([1j * boundary_coeffs[i] * Xi[i, k]
                                                   * a_operator_list[k] / np.sqrt(2.0)
                                                   for i in range(dim) for k in range(dim)], axis=0))
        return self._BCH_factor(j, Xi) * exp_i_phi_theta_a_component.T @ exp_i_phi_theta_a_component

    def _build_all_exp_i_phi_j_operators(self, Xi, a_operator_list):
        return np.array([self._build_single_exp_i_phi_j_operator(j, Xi, a_operator_list)
                         for j in range(self.number_degrees_freedom)])

    def _harmonic_contribution_to_potential(self, premultiplied_a_and_a_dagger, Xi, phi_bar):
        dim = self.number_degrees_freedom
        a, a_a, a_dagger_a = premultiplied_a_and_a_dagger
        harmonic_contribution = np.sum([0.5*self.EL*Xi[0, i]*Xi.T[i, 0]*(a_a[i] + a_a[i].T
                                                                         + 2.0*a_dagger_a[i] + self.identity())
                                        + np.sqrt(2.0)*self.EL*Xi[0, i]*(a[i] + a[i].T)*phi_bar[0]
                                        for i in range(dim)], axis=0)
        harmonic_contribution += self.EL * phi_bar[0]**2 * self.identity()
        return harmonic_contribution

    def _local_potential_contribution_to_transfer_matrix(self, exp_i_phi_list, premultiplied_a_and_a_dagger,
                                                         Xi, phi_neighbor, minima_m, minima_p):
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        potential_matrix = self._harmonic_contribution_to_potential(premultiplied_a_and_a_dagger, Xi, phi_bar)
        for j in range(dim):
            boundary_coeffs = np.array([(-1)**j, 1])
            exp_i_phi_theta = (exp_i_phi_list[j]*np.prod([np.exp(1j*boundary_coeffs[i]*phi_bar[i])
                                                         for i in range(dim)])
                               * np.exp((-1)**(j+1) * 1j * np.pi * self.flux))
            potential_matrix += -0.5 * self.EJlist[j] * (exp_i_phi_theta + exp_i_phi_theta.conjugate())
        potential_matrix += np.sum(self.EJlist) * self.identity()
        return potential_matrix


class ZeroPiVCHOSSqueezing(VCHOSSqueezing, ZeroPiVCHOS):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=0, dCJ=0, truncated_dim=None, phi_extent=10, **kwargs):
        VCHOSSqueezing.__init__(self, EJlist=np.array([EJ, EJ]), nglist=np.array([0.0, ng]), flux=flux,
                                number_degrees_freedom=2, number_periodic_degrees_freedom=1, **kwargs)
        ZeroPiVCHOS.__init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=dEJ, dCJ=dCJ, truncated_dim=truncated_dim,
                             phi_extent=phi_extent, **kwargs)

    def _build_potential_operators_squeezing(self, a_operator_list, Xi, exp_a_dagger_a,
                                             disentangled_squeezing_matrices, delta_rho_matrices):
        exp_i_list = []
        dim = self.number_degrees_freedom
        prefactor_a, prefactor_a_dagger = self._build_potential_exp_prefactors(disentangled_squeezing_matrices,
                                                                               delta_rho_matrices)
        for j in range(dim):
            boundary_coeffs = np.array([(-1) ** j, 1])
            exp_i_j_a_dagger_part = expm(np.sum([1j * boundary_coeffs[i]
                                                 * (Xi @ prefactor_a_dagger)[i, k] * a_operator_list[k].T
                                                 for i in range(dim) for k in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j_a_part = expm(np.sum([1j * boundary_coeffs[i] * (Xi @ prefactor_a)[i, k] * a_operator_list[k]
                                          for i in range(dim) for k in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j = exp_i_j_a_dagger_part @ exp_a_dagger_a @ exp_i_j_a_part
            exp_i_list.append(exp_i_j)
        return exp_i_list

    def _local_potential_squeezing_function(self, Xi, Xi_inv, phi_neighbor, minima_m, minima_p,
                                            disentangled_squeezing_matrices, delta_rho_matrices,
                                            exp_a_dagger_a, minima_pair_results):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_list, harmonic_minima_pair_results = minima_pair_results
        exp_i_phi_list = []
        for j in range(dim):
            boundary_coeffs = np.array([(-1) ** j, 1])
            exp_i_phi_list.append(exp_i_list[j]*np.prod([np.exp(1j*boundary_coeffs[i]*phi_bar[i]) for i in range(dim)])
                                  * np.exp((-1)**(j+1) * 1j * np.pi * self.flux))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      delta_rho_matrices,
                                                                                      exp_i_phi_list)
                                   for j in range(dim)], axis=0)
        potential_matrix += self._local_potential_harmonic_squeezing(Xi, Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                     disentangled_squeezing_matrices,
                                                                     delta_rho_matrices, exp_a_dagger_a,
                                                                     harmonic_minima_pair_results, phi_bar)
        potential_matrix += (self._local_contribution_identity_squeezing(Xi_inv, phi_neighbor, minima_m, minima_p,
                                                                         disentangled_squeezing_matrices,
                                                                         delta_rho_matrices, exp_a_dagger_a,
                                                                         minima_pair_results)
                             * np.sum(self.EJlist))
        return potential_matrix

    def _minima_pair_potential_squeezing_function(self, a_operator_list, Xi, exp_a_dagger_a,
                                                  disentangled_squeezing_matrices, delta_rho_matrices):
        """Return data necessary for constructing the potential matrix that only depends on the minima
        pair, and not on the specific periodic continuation operator."""
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        linear_coefficients_potential = self._linear_coefficient_matrices(rho_prime, delta_rho,
                                                                          Xi / np.sqrt(2.0), Xi / np.sqrt(2.0))
        return (self._build_potential_operators_squeezing(a_operator_list, Xi, exp_a_dagger_a,
                                                          disentangled_squeezing_matrices, delta_rho_matrices),
                self._minima_pair_potential_harmonic_squeezing(a_operator_list, exp_a_dagger_a, Xi,
                                                               disentangled_squeezing_matrices,
                                                               delta_rho_matrices, linear_coefficients_potential))

    def _minima_pair_potential_harmonic_squeezing(self, a_operator_list, exp_a_dagger_a, Xi,
                                                  disentangled_squeezing_matrices, delta_rho_matrices,
                                                  linear_coefficient_matrices):
        dim = self.number_degrees_freedom
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        a_coefficient, a_dagger_coefficient = linear_coefficient_matrices
        (xa, xaa, dxa, dx, ddx) = self._premultiplying_exp_a_dagger_a_with_a(exp_a_dagger_a, a_operator_list)
        sigma_delta_rho_bar_zpp = (expm(-sigma).T @ expm(delta_rho_bar) @ a_dagger_coefficient.T)[:, 0]
        xaa_coefficient = np.outer((a_coefficient @ expm(-sigma_prime)).T[:, 0],
                                   (a_coefficient @ expm(-sigma_prime))[0, :])
        dxa_coefficient = np.outer(sigma_delta_rho_bar_zpp, (a_coefficient @ expm(-sigma_prime))[0, :])
        ddx_coefficient = np.outer(sigma_delta_rho_bar_zpp, (expm(-sigma).T @ expm(delta_rho_bar)
                                                             @ a_dagger_coefficient.T).T[0, :])
        x_coefficient = a_dagger_coefficient.T[:, 0] @ a_coefficient[0, :]
        xa_coefficient = a_coefficient[0, :] @ expm(-sigma_prime)
        dx_coefficient = a_dagger_coefficient[0, :] @ (expm(-sigma).T @ expm(delta_rho_bar)).T
        potential_matrix = np.sum([xaa_coefficient[mu, nu] * exp_a_dagger_a @ a_operator_list[mu] @ a_operator_list[nu]
                                   + (2 * dxa_coefficient[mu, nu] * a_operator_list[mu].T
                                      @ exp_a_dagger_a @ a_operator_list[nu])
                                   + (ddx_coefficient[mu, nu] * a_operator_list[mu].T
                                      @ a_operator_list[nu].T @ exp_a_dagger_a)
                                   for mu in range(dim) for nu in range(dim)], axis=0) * self.EL
        potential_matrix += exp_a_dagger_a * x_coefficient * self.EL
        return potential_matrix, xa, dx, xa_coefficient, dx_coefficient

    def _local_contribution_single_junction_squeezing(self, j, delta_phi, Xi, Xi_inv, disentangled_squeezing_matrices,
                                                      delta_rho_matrices, exp_i_phi_list):
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = delta_rho_matrices
        boundary_coeffs = np.array([(-1) ** j, 1])
        arg_exp_a_dag = (delta_phi @ Xi_inv.T + 1j * (boundary_coeffs @ Xi)) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rho_prime, delta_rho)
        potential_matrix = -0.5 * self.EJlist[j] * (alpha * exp_i_phi_list[j] + (alpha * exp_i_phi_list[j]).conj())
        potential_matrix *= self._BCH_factor(j, Xi)
        return potential_matrix

    def _construct_potential_alpha_epsilon_squeezing(self, Xi, Xi_inv, delta_phi, rho_prime, delta_rho, phi_bar):
        arg_exp_a_dag = delta_phi @ Xi_inv.T / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho)
        A = Xi / np.sqrt(2.0)
        B = Xi / np.sqrt(2.0)
        epsilon = -(A @ rho_prime @ arg_exp_a + 0.5 * (B - A @ rho_prime) @ (delta_rho + delta_rho.T)
                    @ (arg_exp_a_dag - rho_prime @ arg_exp_a)) + phi_bar
        return alpha, epsilon

    def _local_potential_harmonic_squeezing(self, Xi, Xi_inv, phi_neighbor, minima_m, minima_p,
                                            disentangled_squeezing_matrices, delta_rho_matrices,
                                            exp_a_dagger_a, minima_pair_results, phi_bar):
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = delta_rho_matrices
        potential_matrix_minima_pair, xa, dx, xa_coefficient, dx_coefficient = minima_pair_results
        alpha, epsilon = self._construct_potential_alpha_epsilon_squeezing(Xi, Xi_inv, delta_phi,
                                                                           rho_prime, delta_rho, phi_bar)
        e_xa_coefficient = epsilon[0] * xa_coefficient
        e_dx_coefficient = epsilon[0] * dx_coefficient
        return alpha * (np.sum([2 * xa[mu] * e_xa_coefficient[mu] + 2 * dx[mu] * e_dx_coefficient[mu]
                                for mu in range(dim)], axis=0) * self.EL
                        + potential_matrix_minima_pair + exp_a_dagger_a * self.EL * epsilon[0]**2)


class ZeroPiVCHOSGlobal(Hashing, ZeroPiVCHOS):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=0, dCJ=0, truncated_dim=None, phi_extent=10,
                 global_exc=0, **kwargs):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2)
        ZeroPiVCHOS.__init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=dEJ, dCJ=dCJ,
                             truncated_dim=truncated_dim, phi_extent=phi_extent, **kwargs)
