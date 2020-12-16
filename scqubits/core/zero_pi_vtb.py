from itertools import product
from typing import Dict, Any, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from scipy.linalg import expm, inv

from scqubits import VariationalTightBindingSqueezing
from scqubits.core import descriptors
from scqubits.core.hashing import Hashing
from scqubits.core.variationaltightbinding import VariationalTightBinding
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.zeropi import ZeroPiFunctions


class ZeroPiVTB(ZeroPiFunctions, VariationalTightBinding,
                base.QubitBaseClass, serializers.Serializable):
    r""" Zero Pi using VTB

    See class ZeroPi for documentation on the qubit itself.

    Initialize in the same way as for ZeroPi, however now `num_exc` and `maximum_periodic_vector_length`
    must be set. See VTB for explanation of other kwargs.
    """
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dEJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    dCJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self,
                 EJ: float,
                 EL: float,
                 ECJ: float,
                 EC: float,
                 ng: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 dEJ: float = 0.0,
                 dCJ: float = 0.0,
                 truncated_dim: int = None,
                 phi_extent: int = 10,
                 **kwargs
                 ) -> None:
        ZeroPiFunctions.__init__(self, EJ, EL, flux, dEJ=dEJ)
        VariationalTightBinding.__init__(self, num_exc, maximum_periodic_vector_length,
                                         number_degrees_freedom=2, number_periodic_degrees_freedom=1, **kwargs)
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
    def default_params() -> Dict[str, Any]:
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

    @property
    def nglist(self):
        return np.array([0.0, self.ng])

    @property
    def EJlist(self):
        return np.array([self.EJ, self.EJ])

    def build_capacitance_matrix(self) -> ndarray:
        dim = self.number_degrees_freedom
        C_matrix = np.zeros((dim, dim))

        C = self.e**2 / (2. * self.EC)
        CJ = self.e**2 / (2. * self.ECJ)

        C_matrix[0, 0] = 2 * CJ
        C_matrix[1, 1] = 2 * (C + CJ)
        C_matrix[0, 1] = C_matrix[1, 0] = 4 * self.dCJ

        return C_matrix

    def build_EC_matrix(self) -> ndarray:
        C_matrix = self.build_capacitance_matrix()
        return 0.5 * self.e**2 * inv(C_matrix)

    def _check_second_derivative_positive(self, phi: ndarray, theta: ndarray) -> bool:
        return (self.EL + 2 * self.EJ * np.cos(theta) * np.cos(phi - np.pi * self.flux)) > 0

    def _append_new_minima(self, result: ndarray, minima_holder: List) -> List:
        new_minimum = self._check_if_new_minima(result, minima_holder)
        if new_minimum:
            minima_holder.append(np.array([result[0], np.mod(result[1], 2 * np.pi)]))
        return minima_holder

    def find_minima(self) -> ndarray:
        minima_holder = []
        guess = np.array([0.01, 0.01])
        result = minimize(self.potential, guess)
        minima_holder.append(np.array([result.x[0], np.mod(result.x[1], 2 * np.pi)]))
        for m in range(1, self.phi_extent):
            guesses = product(np.array([np.pi * m, -np.pi * m]), np.array([0.0, np.pi]))
            for guess in guesses:
                result = minimize(self.potential, guess)
                minima_holder = self._append_new_minima(result.x, minima_holder)
        return np.array(minima_holder)

    def build_gamma_matrix(self, minimum: int = 0) -> ndarray:
        dim = self.number_degrees_freedom
        gamma_matrix = np.zeros((dim, dim))
        min_loc = self.sorted_minima()[minimum]
        phi_location = min_loc[0]
        theta_location = min_loc[1]
        gamma_matrix[0, 0] = (2 * self.EL + 2 * self.EJ * np.cos(phi_location - np.pi * self.flux)
                              * np.cos(theta_location) - 2 * self.dEJ * np.sin(theta_location)
                              * np.sin(phi_location - np.pi * self.flux))
        gamma_matrix[1, 1] = (2 * self.EJ * np.cos(phi_location - np.pi * self.flux) * np.cos(theta_location)
                              - 2 * self.dEJ * np.sin(theta_location) * np.sin(phi_location - np.pi * self.flux))
        off_diagonal_term = (-2 * self.EJ * np.sin(phi_location - np.pi * self.flux) * np.sin(theta_location)
                             + 2 * self.EL * phi_location + 2 * self.dEJ * np.cos(theta_location)
                             * np.cos(phi_location - np.pi * self.flux))
        gamma_matrix[1, 0] = off_diagonal_term
        gamma_matrix[0, 1] = off_diagonal_term
        return gamma_matrix/self.Phi0**2

    def _BCH_factor(self, j: int, Xi: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        boundary_coeffs = np.array([(-1)**j, 1])
        return np.exp(-0.25 * np.sum([boundary_coeffs[i] * boundary_coeffs[k] * np.dot(Xi[i, :], Xi.T[:, k])
                                      for i in range(dim) for k in range(dim)]))

    def _build_single_exp_i_phi_j_operator(self, j: int, Xi: ndarray, a_operator_list: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        boundary_coeffs = np.array([(-1)**j, 1])
        exp_i_phi_theta_a_component = expm(np.sum([1j * boundary_coeffs[i] * Xi[i, k]
                                                   * a_operator_list[k] / np.sqrt(2.0)
                                                   for i in range(dim) for k in range(dim)], axis=0))
        return self._BCH_factor(j, Xi) * exp_i_phi_theta_a_component.T @ exp_i_phi_theta_a_component

    def _one_state_exp_i_phi_j_operators(self, Xi: ndarray) -> ndarray:
        r"""Helper method for building :math:`\exp(i\phi_{j})` when no excitations are kept."""
        dim = self.number_degrees_freedom
        exp_factors_list = np.zeros(dim)
        for j in range(dim):
            boundary_coeffs = np.array([(-1) ** j, 1])
            exp_factors_list[j] = np.exp(-0.25 * np.sum([boundary_coeffs[j] * boundary_coeffs[k]
                                                         * np.dot(Xi[j, :], Xi.T[:, k]) for j in range(dim)
                                                         for k in range(dim)]))
        return exp_factors_list

    def _build_all_exp_i_phi_j_operators(self, Xi: ndarray, a_operator_list: ndarray) -> ndarray:
        return np.array([self._build_single_exp_i_phi_j_operator(j, Xi, a_operator_list)
                         for j in range(self.number_degrees_freedom)])

    def _harmonic_contribution_to_potential(self, premultiplied_a_and_a_dagger: Tuple,
                                            Xi: ndarray, phi_bar: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        a, a_a, a_dagger_a = premultiplied_a_and_a_dagger
        harmonic_contribution = np.sum([0.5 * self.EL * Xi[0, i]
                                        * Xi.T[i, 0] * (a_a[i] + a_a[i].T + 2.0 * a_dagger_a[i] + self.identity())
                                        + np.sqrt(2.0) * self.EL * Xi[0, i] * (a[i] + a[i].T) * phi_bar[0]
                                        for i in range(dim)], axis=0)
        harmonic_contribution += self.EL * phi_bar[0]**2 * self.identity()
        return harmonic_contribution

    def _local_potential(self, exp_i_phi_list: ndarray, premultiplied_a_and_a_dagger: Tuple,
                         Xi: ndarray, phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        potential_matrix = self._harmonic_contribution_to_potential(premultiplied_a_and_a_dagger, Xi, phi_bar)
        for j in range(dim):
            exp_i_phi_theta = self._exp_i_phi_theta_with_phi_bar(j, exp_i_phi_list, phi_bar)
            potential_matrix += (-0.5 * self.EJ * (1.0 + (-1)**j * self.dEJ)
                                 * (exp_i_phi_theta + exp_i_phi_theta.conjugate()))
        potential_matrix += 2.0 * self.EJ * self.identity()
        return potential_matrix

    def _one_state_local_potential(self, exp_i_phi_j: ndarray, Xi: ndarray,
                                   phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        potential_matrix = self.EL * (phi_bar[0]**2 + 0.5*(Xi @ Xi.T)[0, 0])
        for j in range(dim):
            exp_i_phi_theta = self._exp_i_phi_theta_with_phi_bar(j, exp_i_phi_j, phi_bar)
            potential_matrix += (-0.5 * self.EJ * (1.0 + (-1)**j * self.dEJ)
                                 * (exp_i_phi_theta + exp_i_phi_theta.conjugate()))
        potential_matrix += 2.0 * self.EJ
        return potential_matrix

    def _exp_i_phi_theta_with_phi_bar(self, j: int, exp_i_phi_j: ndarray, phi_bar: ndarray) -> ndarray:
        dim = self.number_degrees_freedom
        boundary_coeffs = np.array([(-1) ** j, 1])
        exp_i_phi_theta_with_phi_bar = (exp_i_phi_j[j] * np.prod([np.exp(1j * boundary_coeffs[i] * phi_bar[i])
                                                                  for i in range(dim)])
                                        * np.exp((-1) ** (j + 1) * 1j * np.pi * self.flux))
        return exp_i_phi_theta_with_phi_bar

    def _gradient_one_state_local_potential(self, exp_i_phi_j: ndarray, phi_neighbor: ndarray,
                                            minima_m: ndarray, minima_p: ndarray, Xi: ndarray,
                                            which_length: int) -> ndarray:
        """Returns gradient of the potential matrix"""
        dim = self.number_degrees_freedom
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        potential_gradient = self.EL * Xi[0, which_length]**2 * self.optimized_lengths[0, which_length]**(-1)
        for j in range(dim):
            boundary_coeffs = np.array([(-1) ** j, 1])
            exp_i_phi_theta = self._exp_i_phi_theta_with_phi_bar(j, exp_i_phi_j, phi_bar)
            potential_gradient += (0.25 * self.EJ * (1.0 + (-1)**j * self.dEJ)
                                   * self.optimized_lengths[0, which_length]**(-1)
                                   * (boundary_coeffs @ Xi[:, which_length])**2
                                   * (exp_i_phi_theta + exp_i_phi_theta.conjugate()))
        return potential_gradient


class ZeroPiVTBSqueezing(VariationalTightBindingSqueezing, ZeroPiVTB):
    def __init__(self,
                 EJ: float,
                 EL: float,
                 ECJ: float,
                 EC: float,
                 ng: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 dEJ: float = 0.0,
                 dCJ: float = 0.0,
                 truncated_dim: int = None,
                 phi_extent: int = 10,
                 **kwargs
                 ) -> None:
        ZeroPiVTB.__init__(self, EJ, EL, ECJ, EC, ng, flux, num_exc, maximum_periodic_vector_length, dEJ=dEJ, dCJ=dCJ,
                           truncated_dim=truncated_dim, phi_extent=phi_extent, **kwargs)

    def _build_potential_operators_squeezing(self, a_operator_list: ndarray, Xi: ndarray,
                                             exp_a_dagger_a: ndarray,
                                             disentangled_squeezing_matrices: Tuple,
                                             delta_rho_matrices: Tuple) -> ndarray:
        exp_i_list = []
        dim = self.number_degrees_freedom
        prefactor_a, prefactor_a_dagger = self._build_potential_exp_prefactors(disentangled_squeezing_matrices,
                                                                               delta_rho_matrices)
        for j in range(dim):
            boundary_coeffs = np.array([(-1)**j, 1])
            exp_i_j_a_dagger_part = expm(np.sum([1j * boundary_coeffs[i]
                                                 * (Xi @ prefactor_a_dagger)[i, k] * a_operator_list[k].T
                                                 for i in range(dim) for k in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j_a_part = expm(np.sum([1j * boundary_coeffs[i] * (Xi @ prefactor_a)[i, k] * a_operator_list[k]
                                          for i in range(dim) for k in range(dim)], axis=0) / np.sqrt(2.0))
            exp_i_j = exp_i_j_a_dagger_part @ exp_a_dagger_a @ exp_i_j_a_part
            exp_i_list.append(exp_i_j)
        return np.array(exp_i_list)

    def _local_potential_squeezing_function(self, Xi: ndarray, Xi_inv: ndarray,
                                            phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                            disentangled_squeezing_matrices: Tuple,
                                            delta_rho_matrices: Tuple,
                                            exp_a_dagger_a: ndarray, minima_pair_results: Tuple
                                            ) -> ndarray:
        dim = self.number_degrees_freedom
        delta_phi = phi_neighbor + minima_p - minima_m
        phi_bar = 0.5 * (phi_neighbor + (minima_m + minima_p))
        exp_i_list, harmonic_minima_pair_results = minima_pair_results
        exp_i_phi_list = []
        for j in range(dim):
            boundary_coeffs = np.array([(-1)**j, 1])
            exp_i_phi_list.append(exp_i_list[j] * np.prod([np.exp(1j * boundary_coeffs[i] * phi_bar[i])
                                                           for i in range(dim)])
                                  * np.exp((-1)**(j+1) * 1j * np.pi * self.flux))
        potential_matrix = np.sum([self._local_contribution_single_junction_squeezing(j, delta_phi, Xi, Xi_inv,
                                                                                      disentangled_squeezing_matrices,
                                                                                      delta_rho_matrices,
                                                                                      np.array(exp_i_phi_list))
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

    def _minima_pair_potential_squeezing_function(self, a_operator_list: ndarray, Xi: ndarray,
                                                  exp_a_dagger_a: ndarray,
                                                  disentangled_squeezing_matrices: Tuple,
                                                  delta_rho_matrices: Tuple
                                                  ) -> Tuple:
        """Return data necessary for constructing the potential matrix that only depends on the minima
        pair, and not on the specific periodic continuation operator."""
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar = delta_rho_matrices
        linear_coefficients_potential = self._linear_coefficient_matrices(rho_prime, delta_rho,
                                                                          Xi / np.sqrt(2.0), Xi / np.sqrt(2.0))
        return (self._build_potential_operators_squeezing(a_operator_list, Xi, exp_a_dagger_a,
                                                          disentangled_squeezing_matrices, delta_rho_matrices),
                self._minima_pair_potential_harmonic_squeezing(a_operator_list, exp_a_dagger_a,
                                                               disentangled_squeezing_matrices,
                                                               delta_rho_matrices, linear_coefficients_potential))

    def _minima_pair_potential_harmonic_squeezing(self, a_operator_list: ndarray, exp_a_dagger_a: ndarray,
                                                  disentangled_squeezing_matrices: Tuple,
                                                  delta_rho_matrices: Tuple,
                                                  linear_coefficient_matrices: Tuple
                                                  ) -> Tuple:
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

    def _local_contribution_single_junction_squeezing(self, j: int, delta_phi: ndarray,
                                                      Xi: ndarray, Xi_inv: ndarray,
                                                      disentangled_squeezing_matrices: Tuple,
                                                      delta_rho_matrices: Tuple, exp_i_phi_list: ndarray
                                                      ) -> ndarray:
        rho, rho_prime, sigma, sigma_prime, tau, tau_prime = disentangled_squeezing_matrices
        delta_rho, delta_rho_prime, delta_rho_bar, zp, zpp = delta_rho_matrices
        boundary_coeffs = np.array([(-1)**j, 1])
        arg_exp_a_dag = (delta_phi @ Xi_inv.T + 1j * (boundary_coeffs @ Xi)) / np.sqrt(2.)
        alpha = self._alpha_helper(arg_exp_a_dag, -arg_exp_a_dag.conjugate(), rho_prime, delta_rho)
        potential_matrix = -0.5 * self.EJlist[j] * (alpha * exp_i_phi_list[j] + (alpha * exp_i_phi_list[j]).conj())
        potential_matrix *= self._BCH_factor(j, Xi)
        return potential_matrix

    def _construct_potential_alpha_epsilon_squeezing(self, Xi: ndarray, Xi_inv: ndarray,
                                                     delta_phi: ndarray, rho_prime: ndarray,
                                                     delta_rho: ndarray, phi_bar: ndarray) -> Tuple:
        arg_exp_a_dag = delta_phi @ Xi_inv.T / np.sqrt(2.)
        arg_exp_a = -arg_exp_a_dag
        alpha = self._alpha_helper(arg_exp_a_dag, arg_exp_a, rho_prime, delta_rho)
        A = Xi / np.sqrt(2.0)
        B = Xi / np.sqrt(2.0)
        epsilon = -(A @ rho_prime @ arg_exp_a + 0.5 * (B - A @ rho_prime) @ (delta_rho + delta_rho.T)
                    @ (arg_exp_a_dag - rho_prime @ arg_exp_a)) + phi_bar
        return alpha, epsilon

    def _local_potential_harmonic_squeezing(self, Xi: ndarray, Xi_inv: ndarray,
                                            phi_neighbor: ndarray, minima_m: ndarray, minima_p: ndarray,
                                            disentangled_squeezing_matrices: Tuple,
                                            delta_rho_matrices: Tuple,
                                            exp_a_dagger_a: ndarray, minima_pair_results: Tuple,
                                            phi_bar: ndarray) -> ndarray:
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


class ZeroPiVTBGlobal(Hashing, ZeroPiVTB):
    def __init__(self,
                 EJ: float,
                 EL: float,
                 ECJ: float,
                 EC: float,
                 ng: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 dEJ: float = 0.0,
                 dCJ: float = 0.0,
                 truncated_dim: int = None,
                 phi_extent: int = 10,
                 **kwargs
                 ) -> None:
        Hashing.__init__(self)
        ZeroPiVTB.__init__(self, EJ, EL, ECJ, EC, ng, flux, num_exc, maximum_periodic_vector_length, dEJ=dEJ, dCJ=dCJ,
                           truncated_dim=truncated_dim, phi_extent=phi_extent, **kwargs)


class ZeroPiVTBGlobalSqueezing(Hashing, ZeroPiVTBSqueezing):
    def __init__(self,
                 EJ: float,
                 EL: float,
                 ECJ: float,
                 EC: float,
                 ng: float,
                 flux: float,
                 num_exc: int,
                 maximum_periodic_vector_length: int,
                 dEJ: float = 0.0,
                 dCJ: float = 0.0,
                 truncated_dim: int = None,
                 phi_extent: int = 10,
                 **kwargs
                 ) -> None:
        Hashing.__init__(self)
        ZeroPiVTBSqueezing.__init__(self, EJ, EL, ECJ, EC, ng, flux, num_exc, maximum_periodic_vector_length,
                                    dEJ=dEJ, dCJ=dCJ, truncated_dim=truncated_dim, phi_extent=phi_extent, **kwargs)
