import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, inv
import scipy.constants as const

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
            potential_matrix += -0.5*self.EJ*(exp_i_phi_theta + exp_i_phi_theta.conjugate())
        potential_matrix += 2*self.EJ*self.identity()
        return potential_matrix


class ZeroPiVCHOSGlobal(Hashing, ZeroPiVCHOS):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=0, dCJ=0, truncated_dim=None, phi_extent=10,
                 global_exc=0, **kwargs):
        Hashing.__init__(self, global_exc, number_degrees_freedom=2)
        ZeroPiVCHOS.__init__(self, EJ, EL, ECJ, EC, ng, flux, dEJ=dEJ, dCJ=dCJ,
                             truncated_dim=truncated_dim, phi_extent=phi_extent, **kwargs)
