import itertools
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError
from scipy.special import factorial

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.fix_heiberger import fixheiberger
from scqubits.utils.spectrum_utils import order_eigensystem
from scqubits.core.vchos import VCHOS


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

class VCHOSSqueezing(VCHOS):
    def __init__(self):
        # All of these parameters will be set in the individual qubit class
        self.e = 0.0
        self.squeezing = False
        self.kmax = 0
        self.Phi0 = 0.0
        self.num_exc = 0
        self.Z0 = 0.0
        self.flux = 0.0
        self.boundary_coeffs = np.array([])
        self.EJlist = np.array([])
        self.nglist = np.array([])

    def _build_U_squeezing_operator(self, i, Xi):
        """
        Return the rho, sigma, tau matrices that define the overall squeezing operator U

        Parameters
        ----------
        i: int
            integer representing the minimum for which to build the squeezing operator U,
            0<i<=total number of minima (no squeezing need be performed for the global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times

        Returns
        -------
        ndarray, ndarray, ndarray
        """
        uvmat = self._squeezing_M_builder(i, Xi)
        dim = uvmat.shape[0]
        u = uvmat[0: int(dim / 2), 0: int(dim / 2)]
        v = uvmat[int(dim / 2): dim, 0: int(dim / 2)]
        u_inv = sp.linalg.inv(u)
        rho = np.matmul(u_inv, v)
        sigma = sp.linalg.logm(u)
        tau = np.matmul(v, u_inv)
        return rho, sigma, tau

    def _define_squeezing_variables(self, rho, rhoprime, Xi):
        """Build variables helpful for constructing the Hamiltonian """
        Xi_inv = sp.linalg.inv(Xi)
        deltarhoprime = np.matmul(sp.linalg.inv(np.eye(self.number_degrees_freedom())
                                                - np.matmul(rhoprime, rho)), rhoprime)
        deltarho = np.matmul(sp.linalg.inv(np.eye(self.number_degrees_freedom())
                                           - np.matmul(rho, rhoprime)), rho)
        deltarhobar = sp.linalg.logm(sp.linalg.inv(np.eye(self.number_degrees_freedom()) - np.matmul(rhoprime, rho)))
        z = 1j * np.transpose(Xi_inv) / np.sqrt(2.)
        zp = (z + 0.5 * np.matmul(np.matmul(z, rhoprime), deltarho + deltarho.T)
              + 0.5 * np.matmul(z, deltarho + deltarho.T))
        zpp = np.matmul(z, rhoprime) + z
        return deltarho, deltarhoprime, deltarhobar, zp, zpp

    def n_operator(self, j, wrapper_klist=None):
        """Return the charge operator associated with the j^th node, neglecting squeezing"""
        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        a_op_list = np.array([self.a_operator(i) for i in range(self.number_degrees_freedom())])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        n_op_mat = np.zeros((dim, dim), dtype=np.complex128)
        counter = 0
        n_op = -1j * np.sum([np.transpose(Xi_inv)[j, mu] * (a_op_list[mu] - a_op_list[mu].T)
                             for mu in range(self.number_degrees_freedom())], axis=0) / np.sqrt(2.)
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, _, _, _, _, _, _, _, _) = self._build_squeezing_ops(m, p, minima_diff, Xi,
                                                                               a_op_list, potential=False)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, _, _) = exp_list
                id_op = np.copy(exp_a_a)
                for jkvals in wrapper_klist[counter]:
                    phik = 2.0 * np.pi * np.array(jkvals)
                    delta_phi_kpm = phik + minima_diff
                    dpkX = np.matmul(Xi_inv, delta_phi_kpm)
                    exp_prod_coeff = (np.exp(-1j * np.dot(self.nglist, delta_phi_kpm))
                                      * np.exp(-0.25 * np.dot(dpkX, dpkX)))

                    epsilon = - (1j / 2.) * np.matmul(Xi_inv.T, np.matmul(Xi_inv, delta_phi_kpm)) * id_op

                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    n_op_tmp = exp_prod_coeff * np.matmul(exp_adag, np.matmul(n_op + epsilon, exp_a))

                    n_op_mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                             p * num_exc_tot: p * num_exc_tot + num_exc_tot] += n_op_tmp

                counter += 1

        # fill in kinetic energy matrix according to hermiticity
        n_op_mat = self._populate_hermitian_matrix(n_op_mat, minima_list, num_exc_tot)

        return n_op_mat

    def cos_or_sin_phi_operator(self, j, which='cos', wrapper_klist=None):
        """Return the operator cos(\phi_j) or sin(\phi_j), or cos\sin(\sum_j \phi_j)
        if j corresponds to the boundary

        Parameters
        ----------
        j: int
            integer corresponding to the junction, 1<=j<=self.num_deg_freedom+1
        which: str
            string corresponding to which operator is desired, cos or sin
        wrapper_klist: ndarray
            optional argument for speeding up calculations, if relevant
            k vectors have been precomputed.

        Returns
        -------
        ndarray corresponding to cos or sin operator
        """

        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        a_op_list = np.array([self.a_operator(i) for i in range(self.number_degrees_freedom())])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        cos_or_sin_phi_j_mat = np.zeros((dim, dim), dtype=np.complex128)
        exp_prod_boundary_coeff = np.exp(-0.25 * np.sum([self.boundary_coeffs[j]
                                                         * self.boundary_coeffs[k]
                                                         * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                                         for j in range(self.number_degrees_freedom())
                                                         for k in range(self.number_degrees_freedom())]))
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi,
                                                                             a_op_list, potential=True)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum) = exp_list
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.number_degrees_freedom()) - np.matmul(rho, rhoprime)))
                for jkvals in wrapper_klist[counter]:
                    phik = 2.0 * np.pi * np.array(jkvals)
                    delta_phi_kpm = phik + minima_diff
                    phibar_kpm = 0.5 * (phik + (minima_m + minima_p))
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm, Xi_inv, sigma, sigmaprime)

                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    if j == (self.number_degrees_freedom() + 1):
                        exp_i_phi_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                                        * np.prod([np.exp(1j * self.boundary_coeffs[i] * phibar_kpm[i])
                                                   for i in range(self.number_degrees_freedom())])
                                        * exp_prod_boundary_coeff * exp_prod_coeff)
                        x = (np.matmul(delta_phi_kpm, Xi_inv.T)
                             + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                       for i in range(self.number_degrees_freedom())], axis=0)) / np.sqrt(2.)
                        y = (- np.matmul(delta_phi_kpm, Xi_inv.T)
                             + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                       for i in range(self.number_degrees_freedom())], axis=0)) / np.sqrt(2.)

                    else:
                        exp_i_phi_op = (exp_i_list[j-1] * np.exp(1j * phibar_kpm[j-1])
                                        * np.exp(-.25 * np.dot(Xi[j - 1, :], np.transpose(Xi)[:, j - 1]))
                                        * exp_prod_coeff)
                        x = (np.matmul(delta_phi_kpm, Xi_inv.T) + 1j * Xi[j-1, :]) / np.sqrt(2.)
                        y = (-np.matmul(delta_phi_kpm, Xi_inv.T) + 1j * Xi[j-1, :]) / np.sqrt(2.)

                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    alpha_con = scale * self._alpha_helper(x.conjugate(), y.conjugate(),
                                                           rhoprime, deltarho)
                    if which is 'cos':
                        cos_or_sin_phi_j = 0.5 * (alpha * exp_i_phi_op + alpha_con * exp_i_phi_op.conjugate())
                    elif which is 'sin':
                        cos_or_sin_phi_j = -1j*0.5*(alpha * exp_i_phi_op - alpha_con * exp_i_phi_op.conjugate())
                    else:
                        raise ValueError('which must be cos or sin')

                    cos_or_sin_phi_j = np.matmul(exp_adag, np.matmul(cos_or_sin_phi_j, exp_a))

                    cos_or_sin_phi_j_mat[m * num_exc_tot:m * num_exc_tot + num_exc_tot,
                                         p * num_exc_tot:p * num_exc_tot + num_exc_tot] += cos_or_sin_phi_j

                counter += 1

        cos_or_sin_phi_j_mat = self._populate_hermitian_matrix(cos_or_sin_phi_j_mat, minima_list, num_exc_tot)

        return cos_or_sin_phi_j_mat

    def _squeezing_M_builder(self, i, Xi):
        """
        Returns the M matrix as defined in G. Qin et. al “General multimodesqueezed states,”
        (2001) arXiv:quant-ph/0109020, M=[[u, v],[v, u]] where u and v are the matrices
        that define the Bogoliubov transformation
        Parameters
        ----------
        i: int
            integer representing the minimum for which to build the squeezing operator U,
            0<i<=total number of minima (no squeezing need be performed for the global min)
        Xi: ndarray
            Xi matrix, passed to avoid building multiple times

        Returns
        -------
        ndarray
        """
        gamma = self.build_gamma_matrix(i)
        gamma_prime = np.matmul(Xi.T, np.matmul(gamma, Xi))
        omegamat = self.omegamat(i)
        zeta = 0.25 * (self.Phi0 ** 2 * gamma_prime + omegamat)
        eta = 0.25 * (self.Phi0 ** 2 * gamma_prime - omegamat)
        hmat = np.block([[zeta, -eta],
                         [eta, -zeta]])
        eigvals, eigvec = sp.linalg.eig(hmat)
        eigvals, eigvec = self._order_eigensystem_squeezing(np.real(eigvals), eigvec)
        eigvec = eigvec.T  # since eigvec represents M.T
        # Normalization ensures that eigvec.T K eigvec = K, K = [[1, 0],[0, -1]] (1, 0 are matrices)
        _, eigvec = self._normalize_symplectic_eigensystem_squeezing(eigvals, eigvec)
        return eigvec

    def _order_eigensystem_squeezing(self, eigvals, eigvec):
        """Order eigensystem to have positive eigenvalues followed by negative, in same order"""
        dim2 = int(len(eigvals) / 2)
        eigval_holder = np.zeros(dim2)
        eigvec_holder = np.zeros_like(eigvec)
        count = 0
        for k, eigval in enumerate(eigvals):
            if eigval > 0:
                eigval_holder[count] = eigval
                eigvec_holder[:, count] = eigvec[:, k]
                count += 1
        index_array = np.argsort(eigval_holder)
        eigval_holder = eigval_holder[index_array]
        eigvec_holder[:, 0: dim2] = eigvec_holder[:, index_array]
        # Now attempt to deal with degenerate modes
        for k in range(0, len(eigval_holder) - 1):
            if np.allclose(eigval_holder[k], eigval_holder[k + 1], atol=1e-6):
                evec_1 = eigvec_holder[:, k]
                evec_2 = eigvec_holder[:, k + 1]
                mat = np.array([[evec_1[k], evec_1[k + 1]],  # Assume maximal elements are same as global min
                                [evec_2[k], evec_2[k + 1]]])
                sol = sp.linalg.inv(mat)  # Find linear transformation to get (1, 0) and (0, 1) vectors
                new_evec_1 = sol[0, 0] * evec_1 + sol[0, 1] * evec_2
                new_evec_2 = sol[1, 0] * evec_1 + sol[1, 1] * evec_2
                eigvec_holder[:, k] = new_evec_1
                eigvec_holder[:, k + 1] = new_evec_2
        dim = eigvec.shape[0]
        dim2 = int(dim / 2)
        u = eigvec_holder[0: dim2, 0: dim2]
        v = eigvec_holder[dim2: dim, 0: dim2]
        eigvec_holder[0: dim2, dim2: dim] = v
        eigvec_holder[dim2: dim, dim2: dim] = u
        return eigval_holder, eigvec_holder

    def _normalize_symplectic_eigensystem_squeezing(self, eigvals, eigvec):
        """Enforce commutation relations so that Bogoliubov transformation is symplectic """
        dim = eigvec.shape[0]
        dim2 = int(dim / 2)
        for col in range(dim2):
            a = np.sum([eigvec[row, col] for row in range(dim)])
            if a < 0.0:
                eigvec[:, col] *= -1
        A = eigvec[0: dim2, 0: dim2]
        B = eigvec[dim2: dim, 0: dim2]
        for vec in range(dim2):
            a = 1. / np.sqrt(np.sum([A[num, vec] * A[num, vec] - B[num, vec] * B[num, vec]
                                     for num in range(dim2)]))
            eigvec[:, vec] *= a
        A = eigvec[0: dim2, 0: dim2]
        B = eigvec[dim2: dim, 0: dim2]
        eigvec[dim2: dim, dim2: dim] = A
        eigvec[0: dim2, dim2: dim] = B
        return eigvals, eigvec

    def _build_exponentiated_operators(self, m, p, minima_diff, Xi, a_op_list, potential=True):
        """
        Build all operators relevant for building the Hamiltonian. If there is no squeezing,
        this routine then just builds the translation operators necessary for periodic
        continuation, as well as the exp(i\phi_{j}) operators for the potential
        """
        if self.squeezing:
            if m == 0:  # At the global minimum, no squeezing required
                rho = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))
                sigma = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))
                tau = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))
            else:
                rho, sigma, tau = self._build_U_squeezing_operator(m, Xi)
            if p == 0:
                rhoprime = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))
                sigmaprime = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))
                tauprime = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))
            elif p == m:
                rhoprime = np.copy(rho)
                sigmaprime = np.copy(sigma)
                tauprime = np.copy(tau)
            else:
                rhoprime, sigmaprime, tauprime = self._build_U_squeezing_operator(p, Xi)

            deltarho, deltarhoprime, deltarhobar, zp, zpp = self._define_squeezing_variables(rho, rhoprime, Xi)

            expsigma = sp.linalg.expm(-sigma)
            expsigmaprime = sp.linalg.expm(-sigmaprime)
            expdeltarhobar = sp.linalg.expm(deltarhobar)
            expdrbs = np.matmul(expdeltarhobar.T, expsigma)

            prefactor_adag_adag = 0.5 * (tau.T - np.matmul(expsigma.T, np.matmul(deltarhoprime, expsigma)))
            prefactor_a_a = 0.5 * (tauprime - np.matmul(expsigmaprime.T, np.matmul(deltarho, expsigmaprime)))
            prefactor_adag_a = sp.linalg.logm(np.matmul(expsigma.T, np.matmul(expdeltarhobar, expsigmaprime)))

            exp_adag_adag = sp.linalg.expm(np.sum([prefactor_adag_adag[i, j]
                                                   * np.matmul(a_op_list[i].T,
                                                               a_op_list[j].T)
                                                   for i in range(self.number_degrees_freedom())
                                                   for j in range(self.number_degrees_freedom())], axis=0))
            exp_a_a = sp.linalg.expm(np.sum([prefactor_a_a[i, j]
                                             * np.matmul(a_op_list[i],
                                                         a_op_list[j])
                                             for i in range(self.number_degrees_freedom())
                                             for j in range(self.number_degrees_freedom())], axis=0))
            exp_adag_a = self._normal_ordered_adag_a_exponential(prefactor_adag_a)
        else:
            # We will call squeezing operators later, so must define them to be 0 or 1 where appropriate
            N = self.number_degrees_freedom()
            dim = a_op_list[0].shape[0]
            rho, sigma, tau = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            rhoprime, sigmaprime, tauprime = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            deltarho, deltarhoprime, deltarhobar, zp, zpp = self._define_squeezing_variables(rho, rhoprime, Xi)
            expsigma, expsigmaprime = np.eye(N), np.eye(N)
            expdeltarhobar, expdrbs = np.eye(N), np.eye(N)

            exp_adag_adag, exp_a_a, exp_adag_a = np.eye(dim), np.eye(dim), np.eye(dim)

        Xi_inv = sp.linalg.inv(Xi)

        # For the case of the translation operators
        prefactor_adag = np.matmul(np.eye(self.number_degrees_freedom()) + rhoprime, expdrbs)
        a_temp_coeff = 0.5 * np.matmul(np.eye(self.number_degrees_freedom()) + rhoprime, deltarho + deltarho.T)
        prefactor_a = np.matmul(np.eye(self.number_degrees_freedom()) + a_temp_coeff, expsigmaprime)

        exp_adag_list = []
        for j in range(self.number_degrees_freedom()):
            exp_adag_j = sp.linalg.expm(np.sum([2.0 * np.pi * (np.matmul(Xi_inv.T, prefactor_adag)[j, i] / np.sqrt(2.0))
                                                * a_op_list[i].T for i in range(self.number_degrees_freedom())], axis=0))
            exp_adag_list.append(exp_adag_j)

        exp_adag_mindiff = sp.linalg.expm(np.sum([minima_diff[x] * np.matmul(Xi_inv.T, prefactor_adag)[x, i]
                                                  * a_op_list[i].T for x in range(self.number_degrees_freedom())
                                                  for i in range(self.number_degrees_freedom())], axis=0) / np.sqrt(2.0))
        exp_a_list = []
        for j in range(self.number_degrees_freedom()):
            exp_a_j = sp.linalg.expm(np.sum([2.0 * np.pi * (np.matmul(Xi_inv.T, prefactor_a)[j, i] / np.sqrt(2.0))
                                             * a_op_list[i] for i in range(self.number_degrees_freedom())], axis=0))
            exp_a_list.append(exp_a_j)

        exp_a_mindiff = sp.linalg.expm(np.sum([-minima_diff[x] * np.matmul(Xi_inv.T, prefactor_a)[x, i]
                                               * a_op_list[i] for x in range(self.number_degrees_freedom())
                                               for i in range(self.number_degrees_freedom())], axis=0) / np.sqrt(2.0))

        # Now the potential operators
        exp_i_list = []
        exp_i_sum = 0.0
        if potential:
            prefactor_adag = np.matmul(np.eye(self.number_degrees_freedom()) - rhoprime, expdrbs)
            a_temp_coeff = 0.5 * np.matmul(np.eye(self.number_degrees_freedom()) - rhoprime, deltarho + deltarho.T)
            prefactor_a = np.matmul(np.eye(self.number_degrees_freedom()) - a_temp_coeff, expsigmaprime)

            Xid = 1j * np.matmul(Xi, prefactor_adag) / np.sqrt(2.0)
            Xia = 1j * np.matmul(Xi, prefactor_a) / np.sqrt(2.0)
            for j in range(self.number_degrees_freedom()):
                exp_i_j_adag_part = sp.linalg.expm(np.sum([Xid[j, i] * a_op_list[i].T
                                                           for i in range(self.number_degrees_freedom())], axis=0))
                exp_i_j_a_part = sp.linalg.expm(np.sum([Xia[j, i] * a_op_list[i]
                                                        for i in range(self.number_degrees_freedom())], axis=0))
                exp_i_j = np.matmul(exp_i_j_adag_part, np.matmul(exp_adag_a, exp_i_j_a_part))
                exp_i_list.append(exp_i_j)

            exp_i_sum_adag_part = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                         Xid[j, i] * a_op_list[i].T
                                                         for i in range(self.number_degrees_freedom())
                                                         for j in range(self.number_degrees_freedom())], axis=0))
            exp_i_sum_a_part = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                      Xia[j, i] * a_op_list[i]
                                                      for i in range(self.number_degrees_freedom())
                                                      for j in range(self.number_degrees_freedom())], axis=0))
            exp_i_sum = np.matmul(exp_i_sum_adag_part, np.matmul(exp_adag_a, exp_i_sum_a_part))

        exp_list = [exp_adag_adag, exp_a_a, exp_adag_a,
                    exp_adag_list, exp_adag_mindiff,
                    exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum]

        return exp_list, rho, rhoprime, sigma, sigmaprime, deltarho, deltarhobar, zp, zpp

    def kineticmat(self, wrapper_klist=None):
        """Return the kinetic part of the hamiltonian"""
        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        a_op_list = np.array([self.a_operator(i) for i in range(self.number_degrees_freedom())])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        kinetic_mat = np.zeros((dim, dim), dtype=np.complex128)
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi,
                                                                             a_op_list, potential=False)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, _, _) = exp_list
                # Define all of these variables here and "premultiply" so that 
                # it does not have to be done inside of the costly while loop
                (xa, xaa, dxa, dx, ddx) = self._premultiplying_exp_adag_a_with_a(exp_adag_a, a_op_list)
                expsdrb = np.matmul(sp.linalg.expm(-sigma).T, sp.linalg.expm(deltarhobar))
                expsigmaprime = sp.linalg.expm(-sigmaprime)
                zpexpsp = np.matmul(zp, expsigmaprime)
                expsdrbzppT = np.matmul(expsdrb, zpp.T)
                xaa_coeff = np.matmul(zpexpsp.T, np.matmul(EC_mat, zpexpsp))
                dxa_coeff = np.matmul(expsdrbzppT, np.matmul(EC_mat, zpexpsp))
                ddx_coeff = np.matmul(expsdrbzppT, np.matmul(EC_mat, expsdrbzppT.T))
                x_coeff = np.matmul(zpp.T, np.matmul(EC_mat, zp))
                xa_coeff = np.matmul(np.matmul(EC_mat, zp), expsigmaprime)
                dx_coeff = np.matmul(np.matmul(EC_mat, zpp), expsdrb.T)
                kinetic_temp_dpk_independent = np.sum([+4 * xaa[mu] * xaa_coeff[mu, mu]
                                                       - 8 * dxa[mu] * dxa_coeff[mu, mu]
                                                       + 4 * ddx[mu] * ddx_coeff[mu, mu]
                                                       - 4 * exp_adag_a * x_coeff[mu, mu]
                                                       for mu in range(self.number_degrees_freedom())], axis=0)
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.number_degrees_freedom()) - np.matmul(rho, rhoprime)))
                for jkvals in wrapper_klist[counter]:
                    phik = 2.0 * np.pi * np.array(jkvals)
                    delta_phi_kpm = phik + minima_diff
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm, Xi_inv, sigma, sigmaprime)

                    # x is the vector that appears in exp(x_{i}a_{i}^{\dagger}) (Einstein summation)
                    x = np.matmul(delta_phi_kpm, Xi_inv.T) / np.sqrt(2.)
                    # y is the vector that appears in exp(y_{i}a_{i})
                    y = -x
                    z = 1j * Xi_inv.T / np.sqrt(2.)

                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    yrhop = np.matmul(y, rhoprime)
                    deltarhopp = 0.5 * np.matmul(x - yrhop, deltarho + deltarho.T)

                    # offset present even in the absence of squeezing,
                    # then equal to -i * 0.5 Xi^{-T} Xi^{-1} delta_phi_kpm
                    epsilon = (-np.matmul(z, np.matmul(rhoprime, deltarhopp) - yrhop + deltarhopp)
                               - (1j / 2.) * np.matmul(Xi_inv.T, np.matmul(Xi_inv, delta_phi_kpm)))
                    e_xa_coeff = np.matmul(epsilon, xa_coeff)
                    e_dx_coeff = np.matmul(epsilon, dx_coeff)

                    # use pre-exponentiated matrices to build the translation operators, using
                    # the (hopefully) less costly matrix power routines
                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    kinetic_temp = np.sum([-8 * xa[mu] * e_xa_coeff[mu] + 8 * dx[mu] * e_dx_coeff[mu]
                                           for mu in range(self.number_degrees_freedom())], axis=0)
                    kinetic_temp += kinetic_temp_dpk_independent

                    kinetic_temp += 4 * exp_adag_a * np.matmul(epsilon, np.matmul(EC_mat, epsilon))

                    kinetic_temp = (alpha * exp_prod_coeff
                                    * np.matmul(exp_adag, np.matmul(kinetic_temp, exp_a)))

                    kinetic_mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                                p * num_exc_tot: p * num_exc_tot + num_exc_tot] += kinetic_temp

                counter += 1

        # fill in kinetic energy matrix according to hermiticity
        kinetic_mat = self._populate_hermitian_matrix(kinetic_mat, minima_list, num_exc_tot)

        return kinetic_mat

    def _alpha_helper(self, x, y, rhoprime, deltarho):
        """Build the prefactor that arises due to squeezing. With no squeezing, alpha=1 (number, not matrix)"""
        yrhop = np.matmul(y, rhoprime)
        alpha = np.exp(-0.5 * np.matmul(y, yrhop) - 0.5 * np.matmul(x - yrhop, np.matmul(deltarho, x - yrhop)))
        return alpha

    def potentialmat(self, wrapper_klist=None):
        """Return the potential part of the hamiltonian"""
        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        a_op_list = np.array([self.a_operator(i) for i in range(self.number_degrees_freedom())])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        potential_mat = np.zeros((dim, dim), dtype=np.complex128)
        EJlist = self.EJlist
        exp_prod_boundary_coeff = np.exp(-0.25 * np.sum([self.boundary_coeffs[j]
                                                         * self.boundary_coeffs[k]
                                                         * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                                         for j in range(self.number_degrees_freedom())
                                                         for k in range(self.number_degrees_freedom())]))
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi,
                                                                             a_op_list, potential=True)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum) = exp_list
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.number_degrees_freedom()) - np.matmul(rho, rhoprime)))
                for jkvals in wrapper_klist[counter]:
                    phik = 2.0 * np.pi * np.array(jkvals)
                    delta_phi_kpm = phik + minima_diff
                    phibar_kpm = 0.5 * (phik + (minima_m + minima_p))
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm, Xi_inv, sigma, sigmaprime)

                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    exp_i_phi_list = np.array([exp_i_list[i] * np.exp(1j * phibar_kpm[i])
                                               for i in range(self.number_degrees_freedom())])
                    exp_i_phi_sum_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                                        * np.prod([np.exp(1j * self.boundary_coeffs[i] * phibar_kpm[i])
                                                   for i in range(self.number_degrees_freedom())]))

                    for num in range(self.number_degrees_freedom()):  # summing over potential terms cos(\phi_x)
                        x = (np.matmul(delta_phi_kpm, Xi_inv.T) + 1j * Xi[num, :]) / np.sqrt(2.)
                        y = (-np.matmul(delta_phi_kpm, Xi_inv.T) + 1j * Xi[num, :]) / np.sqrt(2.)

                        alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                        alpha_con = scale * self._alpha_helper(x.conjugate(), y.conjugate(),
                                                               rhoprime, deltarho)

                        potential_temp = -0.5 * EJlist[num] * alpha * exp_i_phi_list[num]
                        # No need to .T the h.c. term, all that is necessary is conjugation
                        potential_temp += -0.5 * EJlist[num] * alpha_con * exp_i_phi_list[num].conjugate()
                        potential_temp = np.matmul(exp_adag, np.matmul(potential_temp, exp_a))
                        potential_temp *= (np.exp(-.25 * np.dot(Xi[num, :], np.transpose(Xi)[:, num]))
                                           * exp_prod_coeff)

                        potential_mat[m * num_exc_tot:m * num_exc_tot + num_exc_tot,
                                      p * num_exc_tot:p * num_exc_tot + num_exc_tot] += potential_temp

                    # cos(sum-2\pi f)
                    x = (np.matmul(delta_phi_kpm, Xi_inv.T)
                         + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                   for i in range(self.number_degrees_freedom())], axis=0)) / np.sqrt(2.)
                    y = (- np.matmul(delta_phi_kpm, Xi_inv.T)
                         + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                   for i in range(self.number_degrees_freedom())], axis=0)) / np.sqrt(2.)
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    alpha_con = scale * self._alpha_helper(x.conjugate(), y.conjugate(),
                                                           rhoprime, deltarho)

                    potential_temp = -0.5 * EJlist[-1] * alpha * exp_i_phi_sum_op
                    potential_temp += -0.5 * EJlist[-1] * alpha_con * exp_i_phi_sum_op.conjugate()

                    potential_temp = (np.matmul(exp_adag, np.matmul(potential_temp, exp_a))
                                      * exp_prod_boundary_coeff * exp_prod_coeff)

                    # Identity term
                    x = np.matmul(delta_phi_kpm, Xi_inv.T) / np.sqrt(2.)
                    y = -x
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)

                    potential_temp += (alpha * np.sum(EJlist) * exp_prod_coeff
                                       * np.matmul(exp_adag, np.matmul(exp_adag_a, exp_a)))

                    potential_mat[m * num_exc_tot:m * num_exc_tot + num_exc_tot,
                                  p * num_exc_tot:p * num_exc_tot + num_exc_tot] += potential_temp

                counter += 1

        potential_mat = self._populate_hermitian_matrix(potential_mat, minima_list, num_exc_tot)

        return potential_mat

    def hamiltonian(self):
        """Construct the Hamiltonian"""
        wrapper_klist = self._find_k_values_for_different_minima()
        return self.kineticmat(wrapper_klist) + self.potentialmat(wrapper_klist)

    def inner_product(self, wrapper_klist=None):
        """Return the inner product matrix, which is nontrivial with tight-binding states"""
        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        a_op_list = np.array([self.a_operator(i) for i in range(self.number_degrees_freedom())])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        inner_product_mat = np.zeros((dim, dim), dtype=np.complex128)
        counter = 0
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi,
                                                                             a_op_list, potential=False)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, _, _) = exp_list
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.number_degrees_freedom()) - np.matmul(rho, rhoprime)))
                for jkvals in wrapper_klist[counter]:
                    phik = 2.0 * np.pi * np.array(jkvals)
                    delta_phi_kpm = phik + minima_diff
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm, Xi_inv, sigma, sigmaprime)

                    x = np.matmul(delta_phi_kpm, Xi_inv.T) / np.sqrt(2.)
                    y = -x
                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)

                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    inner_temp = alpha * exp_prod_coeff * np.matmul(np.matmul(exp_adag, exp_adag_a), exp_a)

                    inner_product_mat[m * num_exc_tot:m * num_exc_tot + num_exc_tot,
                                      p * num_exc_tot:p * num_exc_tot + num_exc_tot] += inner_temp
                counter += 1

        inner_product_mat = self._populate_hermitian_matrix(inner_product_mat, minima_list, num_exc_tot)

        return inner_product_mat
