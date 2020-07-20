import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, inv
import scipy.constants as const

from scqubits.core.vchos import VCHOS, VCHOSMinimaFinder, VCHOSGlobal


class ZeroPiVCHOSFunctions(VCHOSMinimaFinder):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux):
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ
        self.EC = EC
        self.ng = ng
        self.flux = flux

    def build_capacitance_matrix(self):
        dim = self.number_degrees_freedom()
        C_matrix = np.zeros((dim, dim))

        C = self.e ** 2 / (2. * self.EC)
        CJ = self.e ** 2 / (2. * self.ECJ)
        Cs = C + CJ

        C_matrix[0, 0] = CJ
        C_matrix[1, 1] = Cs

        return C_matrix

    def build_EC_matrix(self):
        C_matrix = self.build_capacitance_matrix()
        return 0.5 * self.e**2 * inv(C_matrix)

    def number_degrees_freedom(self):
        return 2

    def number_periodic_degrees_freedom(self):
        return 1

    def _check_if_new_minima(self, result, minima_holder):
        """
        Helper function for find_minima, checking if new_minima is
        indeed a minimum and is already represented in minima_holder. If so,
        _check_if_new_minima returns False.
        """
        if not self._check_second_derivative_positive(result.x[0], result.x[1]):
            return False
        if self.potential(result.x) > 20.0 + self.potential(minima_holder[0]):
            return False
        new_minima_bool = True
        for minima in minima_holder:
            diff_array = minima - result.x
            diff_array_reduced = np.array([diff_array[0], np.mod(diff_array[1], 2 * np.pi)])
            if (np.allclose(diff_array_reduced[1], 0.0, atol=1e-3)
                    or np.allclose(diff_array_reduced[1], 2 * np.pi, atol=1e-3)):
                if np.allclose(diff_array_reduced[0], 0.0, atol=1e-3):
                    new_minima_bool = False
                    break
        return new_minima_bool

    def _check_second_derivative_positive(self, phi, theta):
        return (self.EL + 2 * self.EJ * np.cos(theta) * np.cos(phi - np.pi * self.flux)) > 0

    def _append_new_minima(self, result, minima_holder):
        new_minimum = self._check_if_new_minima(result, minima_holder)
        if new_minimum and result.success:
            minima_holder.append(np.array([result.x[0], np.mod(result.x[1], 2 * np.pi)]))
        return minima_holder

    def find_minima(self):
        minima_holder = []
        guess = np.array([0.01, 0.01])
        result = minimize(self.potential, guess)
        minima_holder.append(result.x)
        for m in range(1, 10):
            guess_positive_0 = np.array([np.pi * m, 0.0])
            guess_negative_0 = np.array([-np.pi * m, 0.0])
            guess_positive_pi = np.array([np.pi * m, np.pi])
            guess_negative_pi = np.array([-np.pi * m, np.pi])
            result_positive_0 = minimize(self.potential, guess_positive_0)
            result_negative_0 = minimize(self.potential, guess_negative_0)
            result_positive_pi = minimize(self.potential, guess_positive_pi)
            result_negative_pi = minimize(self.potential, guess_negative_pi)
            minima_holder = self._append_new_minima(result_positive_0, minima_holder)
            minima_holder = self._append_new_minima(result_negative_0, minima_holder)
            minima_holder = self._append_new_minima(result_positive_pi, minima_holder)
            minima_holder = self._append_new_minima(result_negative_pi, minima_holder)
        return minima_holder


class ZeroPiVCHOS(ZeroPiVCHOSFunctions, VCHOS):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux, kmax, num_exc, dEJ=0, dCJ=0, truncated_dim=None):
        VCHOS.__init__(self, np.array([EJ, EJ]), np.array([0.0, ng]), flux, kmax, num_exc)
        ZeroPiVCHOSFunctions.__init__(self, EJ, EL, ECJ, EC, ng, flux)
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

    def potential(self, phi_theta_array):
        phi = phi_theta_array[0]
        theta = phi_theta_array[1]
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)
                + self.EL * phi ** 2 + 2.0 * self.EJ
                + self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def build_gamma_matrix(self, i):
        dim = self.number_degrees_freedom()
        gamma_matrix = np.zeros((dim, dim))
        min_loc = self.sorted_minima()[i]
        phi_location = min_loc[0]
        theta_location = min_loc[1]
        gamma_matrix[0, 0] = 2*self.EL + 2*self.EJ*np.cos(phi_location - np.pi*self.flux)*np.cos(theta_location)
        gamma_matrix[1, 1] = 2*self.EJ*np.cos(phi_location - np.pi*self.flux)*np.cos(theta_location)
        gamma_matrix[1, 0] = gamma_matrix[0, 1] = (-2*self.EJ*np.sin(phi_location - np.pi*self.flux)
                                                   * np.sin(theta_location) + 2*self.EL*phi_location)
        return gamma_matrix/self.Phi0**2

    def _BCH_factor(self, j):
        Xi = self.Xi_matrix()
        dim = self.number_degrees_freedom()
        boundary_coeffs = np.array([(-1)**j, 1])
        return np.exp(-0.25 * np.sum([boundary_coeffs[j]*boundary_coeffs[k]*np.dot(Xi[j, :], Xi.T[:, k])
                                      for j in range(dim) for k in range(dim)]))

    def _build_single_exp_i_phi_j_operator(self, j):
        Xi = self.Xi_matrix()
        dim = self.number_degrees_freedom()
        boundary_coeffs = np.array([(-1)**j, 1])
        exp_i_phi_theta_a_component = expm(np.sum([1j*boundary_coeffs[i]*Xi[i, k]*self.a_operator(k)/np.sqrt(2.0)
                                                   for i in range(dim) for k in range(dim)], axis=0))
        exp_i_phi_theta_a_dagger_component = exp_i_phi_theta_a_component.T
        exp_i_phi_theta = np.matmul(exp_i_phi_theta_a_dagger_component, exp_i_phi_theta_a_component)
        exp_i_phi_theta *= np.exp((-1)**(j+1)*1j*np.pi*self.flux)
        exp_i_phi_theta *= self._BCH_factor(j)
        return exp_i_phi_theta

    def _harmonic_contribution_to_potential(self, premultiplied_a_and_a_dagger, phi_bar, Xi):
        dim = self.number_degrees_freedom()
        a, a_a, a_dagger_a = premultiplied_a_and_a_dagger
        harmonic_contribution = np.sum([0.5*self.EL*Xi[0, i]*Xi.T[i, 0]*(a_a[i] + a_a[i].T
                                                                         + 2.0*a_dagger_a[i] + self.identity())
                                        + np.sqrt(2.0)*self.EL*Xi[0, i]*(a[i] + a[i].T)*phi_bar[0]
                                        for i in range(dim)], axis=0)
        harmonic_contribution += self.EL*phi_bar[0]**2*self.identity()
        return harmonic_contribution

    def _potential_contribution_to_hamiltonian(self, exp_i_phi_list, premultiplied_a_and_a_dagger, Xi):
        def _inner_potential_c_t_h(delta_phi, phi_bar):
            dim = self.number_degrees_freedom()
            potential_matrix = self._harmonic_contribution_to_potential(premultiplied_a_and_a_dagger, phi_bar, Xi)
            for j in range(dim):
                boundary_coeffs = np.array([(-1)**j, 1])
                exp_i_phi_theta = exp_i_phi_list[j]*np.prod([np.exp(1j*boundary_coeffs[i]*phi_bar[i])
                                                             for i in range(dim)])
                potential_matrix += -0.5*self.EJ*(exp_i_phi_theta + exp_i_phi_theta.conjugate())
            potential_matrix += 2*self.EJ*self.identity()
            return potential_matrix
        return _inner_potential_c_t_h


class ZeroPiVCHOSGlobal(ZeroPiVCHOSFunctions, VCHOSGlobal):
    def __init__(self, EJ, EL, ECJ, EC, ng, flux, kmax, num_exc, dEJ=0, dCJ=0, truncated_dim=None):
        VCHOSGlobal.__init__(self, np.array([EJ, EJ]), np.array([0.0, ng]), flux, kmax, num_exc)
        ZeroPiVCHOSFunctions.__init__(self, EJ, EL, ECJ, EC, ng, flux)
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
            'global_exc': 5,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['ng', 'flux', 'global_exc', 'truncated_dim']
