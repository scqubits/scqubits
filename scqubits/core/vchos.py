from abc import ABC

import numpy as np
import scipy as sp
import itertools

from scqubits.core.qubit_base import QubitBaseClass
import scqubits.io_utils.fileio_serializers as serializers
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
# find and sort all inequivalent minima (based on the value of the 
# potential at that minimum), respectively. 

class VCHOS(QubitBaseClass, serializers.Serializable, ABC):
    def __init__(self):
        # All of these parameters will be set in the individual qubit class
        self.squeezing = False
        self.kmax = 1
        self.Phi0 = 0.0
        self.num_exc = 1
        self.Z0 = 0.0
        self.flux = 0.0
        self.num_deg_freedom = 0
        self.boundary_coeffs = np.array([])
        self.EJlist = np.array([])
        self.nglist = np.array([])

    def potential(self, phiarray):
        """
        Potential evaluated at the location specified by phiarray
        """
        pot_sum = np.sum([-self.EJlist[j] * np.cos(phiarray[j])
                          for j in range(self.num_deg_freedom)])
        pot_sum += (-self.EJlist[-1]
                    * np.cos(np.sum([self.boundary_coeffs[i] * phiarray[i]
                                     for i in range(self.num_deg_freedom)])
                             + 2 * np.pi * self.flux))
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
        
        """
        gmat = np.zeros((self.num_deg_freedom, self.num_deg_freedom))

        min_loc = self.sorted_minima()[i]
        gamma_list = self.EJlist / self.Phi0 ** 2

        gamma_diag = np.diag([gamma_list[j] * np.cos(min_loc[j])
                              for j in range(self.num_deg_freedom)])
        gmat += gamma_diag

        min_loc_bound_sum = np.sum([self.boundary_coeffs[i] * min_loc[i]
                                    for i in range(self.num_deg_freedom)])
        for j in range(self.num_deg_freedom):
            for k in range(self.num_deg_freedom):
                gmat[j, k] += (gamma_list[-1] * self.boundary_coeffs[j] * self.boundary_coeffs[k]
                               * np.cos(min_loc_bound_sum + 2 * np.pi * self.flux))

        return gmat

    def _eigensystem_normal_modes(self):
        Cmat = self.build_capacitance_matrix()
        gmat = self.build_gamma_matrix(0)

        omegasq, eigvec = sp.linalg.eigh(gmat, b=Cmat)
        return omegasq, eigvec

    def omegamat(self):
        """Return a diagonal matrix of the normal mode frequencies of the global min """
        omegasq, _ = self._eigensystem_normal_modes()
        return np.diag(np.sqrt(omegasq))
    
    def Xi_matrix(self):
        """Construct the Xi matrix, encoding the oscillator lengths of each dimension"""
        omegasq, eigvec = self._eigensystem_normal_modes()
        # We introduce a normalization such that \Xi^T C \Xi = \Omega^{-1}/Z0
        Ximat = np.array([eigvec[:, i] * (omegasq[i])**(-1/4)
                          * np.sqrt(1. / self.Z0) for i in range(len(omegasq))]).T
        
        return Ximat

    def _build_U_squeezing_operator(self, i, Xi):
        freq, uvmat = self._squeezing_M_builder(i, Xi)
        dim = uvmat.shape[0]
        u = uvmat[0: int(dim / 2), 0: int(dim / 2)]
        v = uvmat[int(dim / 2): dim, 0: int(dim / 2)]
        u_inv = sp.linalg.inv(u)
        rho = np.matmul(u_inv, v)
        sigma = sp.linalg.logm(u)
        tau = np.matmul(v, u_inv)
        return rho, sigma, tau

    def _define_squeezing_variables(self, rho, rhoprime, Xi):
        Xi_inv = sp.linalg.inv(Xi)
        deltarhoprime = np.matmul(sp.linalg.inv(np.eye(self.num_deg_freedom)
                                                - np.matmul(rhoprime, rho)), rhoprime)
        deltarho = np.matmul(sp.linalg.inv(np.eye(self.num_deg_freedom)
                                           - np.matmul(rho, rhoprime)), rho)
        deltarhobar = sp.linalg.logm(sp.linalg.inv(np.eye(self.num_deg_freedom) - np.matmul(rhoprime, rho)))
        z = 1j * np.transpose(Xi_inv) / np.sqrt(2.)
        zp = (z + 0.5 * np.matmul(np.matmul(z, rhoprime), deltarho + deltarho.T)
              + 0.5 * np.matmul(z, deltarho + deltarho.T))
        zpp = np.matmul(z, rhoprime) + z
        return deltarho, deltarhoprime, deltarhobar, zp, zpp

    def a_operator(self, mu):
        """Return the lowering operator associated with the xth d.o.f. in the full Hilbert space"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a, k=1)
        return self._full_o([a_mat], [mu])

    def _squeezing_M_builder(self, i, Xi):
        dim = Xi.shape[0]
        gamma = self.build_gamma_matrix(i)
        gamma_prime = np.matmul(Xi.T, np.matmul(gamma, Xi))
        omegamat = self.omegamat()
        zeta = 0.25 * (self.Phi0 ** 2 * gamma_prime + omegamat)
        eta = 0.25 * (self.Phi0 ** 2 * gamma_prime - omegamat)
        hmat = np.block([[zeta, -eta],
                         [eta, -zeta]])
        K = np.block([[np.eye(dim), np.zeros((dim, dim))],
                      [np.zeros((dim, dim)), -np.eye(dim)]])
        eigvals, eigvec = sp.linalg.eig(hmat)
        eigvals, eigvec = self._order_eigensystem_squeezing(np.real(eigvals), eigvec)
        eigvec = eigvec.T  # since eigvec represents M.T
        eigvals, eigvec = self._normalize_symplectic_eigensystem_squeezing(eigvals, eigvec)
        assert (np.allclose(np.matmul(eigvec.T, np.matmul(K, eigvec)), K))
        return eigvals, eigvec

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
        for k in range(0, len(eigval_holder) - 1, 2):
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

    def _normal_ordered_adag_a_exponential(self, x):
        """Expectation is that exp(a_{i}^{\dagger}x_{ij}a_{j}) needs to be normal ordered"""
        expx = sp.linalg.expm(x)
        dim = self.a_operator(0).shape[0]
        result = np.eye(dim, dtype=np.complex128)
        dim = result.shape[0]
        additionalterm = np.eye(dim, dtype=np.complex128)
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        k = 1
        while not np.allclose(additionalterm, np.zeros((dim, dim))):
            additionalterm = np.sum([((expx - np.eye(self.num_deg_freedom))[i, j]) ** k
                                     * (sp.special.factorial(k)) ** (-1)
                                     * np.matmul(np.linalg.matrix_power(a_op_list[i].T, k),
                                                 np.linalg.matrix_power(a_op_list[j], k))
                                     for i in range(self.num_deg_freedom)
                                     for j in range(self.num_deg_freedom)], axis=0)
            result += additionalterm
            k += 1
        return result

    def _build_squeezing_ops(self, m, p, minima_diff, Xi, a_op_list):
        if self.squeezing:
            if m == 0:  # At the global minimum, no squeezing required
                rho = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
                sigma = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
                tau = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
            else:
                rho, sigma, tau = self._build_U_squeezing_operator(m, Xi)
            if p == 0:
                rhoprime = np.zeros((self.num_deg_freedom, self.num_deg_freedom))  # 2 d.o.f.
                sigmaprime = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
                tauprime = np.zeros((self.num_deg_freedom, self.num_deg_freedom))
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
                                                   for i in range(self.num_deg_freedom)
                                                   for j in range(self.num_deg_freedom)], axis=0))
            exp_a_a = sp.linalg.expm(np.sum([prefactor_a_a[i, j]
                                             * np.matmul(a_op_list[i],
                                                         a_op_list[j])
                                             for i in range(self.num_deg_freedom)
                                             for j in range(self.num_deg_freedom)], axis=0))
            exp_adag_a = self._normal_ordered_adag_a_exponential(prefactor_adag_a)
        else:
            N = self.num_deg_freedom
            dim = a_op_list[0].shape[0]
            rho, sigma, tau = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            rhoprime, sigmaprime, tauprime = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            deltarho, deltarhoprime, deltarhobar, zp, zpp = self._define_squeezing_variables(rho, rhoprime, Xi)
            expsigma, expsigmaprime = np.eye(N), np.eye(N)
            expdeltarhobar, expdrbs = np.eye(N), np.eye(N)

            exp_adag_adag, exp_a_a, exp_adag_a = np.eye(dim), np.eye(dim), np.eye(dim)

        Xi_inv = sp.linalg.inv(Xi)

        prefactor_adag = np.matmul(np.eye(self.num_deg_freedom) + rhoprime, expdrbs)
        a_temp_coeff = 0.5 * np.matmul(np.eye(self.num_deg_freedom) + rhoprime, deltarho + deltarho.T)
        prefactor_a = np.matmul(np.eye(self.num_deg_freedom) + a_temp_coeff, expsigmaprime)

        exp_adag_list = []
        for j in range(self.num_deg_freedom):
            exp_adag_j = sp.linalg.expm(np.sum([2.0 * np.pi * (np.matmul(Xi_inv.T, prefactor_adag)[j, i] / np.sqrt(2.0))
                                                * a_op_list[i].T for i in range(self.num_deg_freedom)], axis=0))
            exp_adag_list.append(exp_adag_j)

        exp_adag_mindiff = sp.linalg.expm(np.sum([minima_diff[x] * np.matmul(Xi_inv.T, prefactor_adag)[x, i]
                                                  * a_op_list[i].T for x in range(self.num_deg_freedom)
                                                  for i in range(self.num_deg_freedom)], axis=0) / np.sqrt(2.0))
        exp_a_list = []
        for j in range(self.num_deg_freedom):
            exp_a_j = sp.linalg.expm(np.sum([2.0 * np.pi * (np.matmul(Xi_inv.T, prefactor_a)[j, i] / np.sqrt(2.0))
                                             * a_op_list[i] for i in range(self.num_deg_freedom)], axis=0))
            exp_a_list.append(exp_a_j)

        exp_a_mindiff = sp.linalg.expm(np.sum([-minima_diff[x] * np.matmul(Xi_inv.T, prefactor_a)[x, i]
                                               * a_op_list[i] for x in range(self.num_deg_freedom)
                                               for i in range(self.num_deg_freedom)], axis=0) / np.sqrt(2.0))

        prefactor_adag = np.matmul(np.eye(self.num_deg_freedom) - rhoprime, expdrbs)
        a_temp_coeff = 0.5 * np.matmul(np.eye(self.num_deg_freedom) - rhoprime, deltarho + deltarho.T)
        prefactor_a = np.matmul(np.eye(self.num_deg_freedom) - a_temp_coeff, expsigmaprime)

        exp_i_list = []
        Xid = 1j * np.matmul(Xi, prefactor_adag) / np.sqrt(2.0)
        Xia = 1j * np.matmul(Xi, prefactor_a) / np.sqrt(2.0)
        for j in range(self.num_deg_freedom):
            exp_i_j_adag_part = sp.linalg.expm(np.sum([Xid[j, i] * a_op_list[i].T
                                                       for i in range(self.num_deg_freedom)], axis=0))
            exp_i_j_a_part = sp.linalg.expm(np.sum([Xia[j, i] * a_op_list[i]
                                                    for i in range(self.num_deg_freedom)], axis=0))
            exp_i_j = np.matmul(exp_i_j_adag_part, np.matmul(exp_adag_a, exp_i_j_a_part))
            exp_i_list.append(exp_i_j)

        exp_i_sum_adag_part = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                     Xid[j, i] * a_op_list[i].T
                                                     for i in range(self.num_deg_freedom)
                                                     for j in range(self.num_deg_freedom)], axis=0))
        exp_i_sum_a_part = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                  Xia[j, i] * a_op_list[i]
                                                  for i in range(self.num_deg_freedom)
                                                  for j in range(self.num_deg_freedom)], axis=0))
        exp_i_sum = np.matmul(exp_i_sum_adag_part, np.matmul(exp_adag_a, exp_i_sum_a_part))

        exp_list = [exp_adag_adag, exp_a_a, exp_adag_a,
                    exp_adag_list, exp_adag_mindiff,
                    exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum]

        return exp_list, rho, rhoprime, sigma, sigmaprime, deltarho, deltarhobar, zp, zpp

    def _premultiplying_exp_adag_a_with_a(self, exp_adag_a, a_op_list):
        """
        Naming scheme is  x -> exp(A_{ij}a_{i}^{\dag}a_{j}) (for whatever matrix A is)
                          a -> a_{i}
                          d -> a_{i}^{\dag}
        """
        xa = np.array([np.matmul(exp_adag_a, a_op_list[mu])
                       for mu in range(self.num_deg_freedom)])
        xaa = np.array([np.matmul(xa[mu], a_op_list[mu])
                        for mu in range(self.num_deg_freedom)])
        dxa = np.array([np.matmul(a_op_list[mu].T, xa[mu])
                        for mu in range(self.num_deg_freedom)])
        dx = np.array([np.matmul(a_op_list[mu].T, exp_adag_a)
                       for mu in range(self.num_deg_freedom)])
        ddx = np.array([np.matmul(a_op_list[mu].T, dx[mu])
                        for mu in range(self.num_deg_freedom)])
        return xa, xaa, dxa, dx, ddx

    def _filter_jkvals(self, jkvals, minima_diff, Xi_inv):
        """ 
        Want to eliminate periodic continuation terms that are irrelevant, i.e.,
        they add nothing to the Hamiltonian. These can be identified as each term 
        is suppressed by a gaussian exponential factor. If the argument np.dot(dpkX, dpkX)
        of the exponential is greater than 180.0, this results in a suppression of ~10**(-20),
        and so can be safely neglected.
        """
        phik = 2.0 * np.pi * np.array([jkvals[i] for i in range(self.num_deg_freedom)])
        dpkX = np.matmul(Xi_inv, phik + minima_diff)
        prod = np.dot(dpkX, dpkX)
        return prod > 180.0

    def _exp_prod_coeff(self, delta_phi_kpm, Xi_inv, sigma, sigmaprime):
        dpkX = np.matmul(Xi_inv, delta_phi_kpm)
        nglist = self.nglist
        return (np.exp(-1j * np.dot(nglist, delta_phi_kpm))
                * np.exp(-0.5 * np.trace(sigma) - 0.5 * np.trace(sigmaprime))
                * np.exp(-0.25 * np.dot(dpkX, dpkX)))

    def kineticmat(self):
        """Return the kinetic part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        kinetic_mat = np.zeros((dim, dim), dtype=np.complex128)
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi, a_op_list)
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
                                                       for mu in range(self.num_deg_freedom)], axis=0)
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom) - np.matmul(rho, rhoprime)))
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.num_deg_freedom)
                klist = itertools.filterfalse(lambda e: self._filter_jkvals(e, minima_diff, Xi_inv), klist)
                jkvals = next(klist, -1)
                while jkvals != -1:
                    phik = 2.0 * np.pi * np.array([jkvals[i] for i in range(self.num_deg_freedom)])
                    delta_phi_kpm = phik - (minima_m - minima_p)
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm, Xi_inv, sigma, sigmaprime)

                    x = np.matmul(delta_phi_kpm, Xi_inv.T) / np.sqrt(2.)
                    y = -x
                    z = 1j * Xi_inv.T / np.sqrt(2.)

                    alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                    yrhop = np.matmul(y, rhoprime)
                    deltarhopp = 0.5 * np.matmul(x - yrhop, deltarho + deltarho.T)

                    epsilon = (-np.matmul(z, np.matmul(rhoprime, deltarhopp) - yrhop + deltarhopp)
                               - (1j / 2.) * np.matmul(Xi_inv.T, np.matmul(Xi_inv, delta_phi_kpm)))
                    e_xa_coeff = np.matmul(epsilon, xa_coeff)
                    e_dx_coeff = np.matmul(epsilon, dx_coeff)

                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    kinetic_temp = np.sum([-8 * xa[mu] * e_xa_coeff[mu] + 8 * dx[mu] * e_dx_coeff[mu]
                                           for mu in range(self.num_deg_freedom)], axis=0)
                    kinetic_temp += kinetic_temp_dpk_independent

                    kinetic_temp += 4 * exp_adag_a * np.matmul(epsilon, np.matmul(EC_mat, epsilon))

                    kinetic_temp = (alpha * exp_prod_coeff
                                    * np.matmul(exp_adag, np.matmul(kinetic_temp, exp_a)))

                    #                    if not np.allclose(kinetic_temp, np.zeros_like(kinetic_temp)):
                    #                        print("m, p = ", m, p, jkvals, np.dot(dpkX, dpkX))

                    kinetic_mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                                p * num_exc_tot: p * num_exc_tot + num_exc_tot] += kinetic_temp

                    jkvals = next(klist, -1)

        for m, minima_m in enumerate(minima_list):
            for p in range(m + 1, len(minima_list)):
                kinetic_temp = kinetic_mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                                           p * num_exc_tot: p * num_exc_tot + num_exc_tot]
                kinetic_mat[p * num_exc_tot: p * num_exc_tot + num_exc_tot,
                            m * num_exc_tot: m * num_exc_tot + num_exc_tot] += kinetic_temp.conjugate().T

        return kinetic_mat

    def _alpha_helper(self, x, y, rhoprime, deltarho):
        yrhop = np.matmul(y, rhoprime)
        alpha = np.exp(-0.5 * np.matmul(y, yrhop) - 0.5 * np.matmul(x - yrhop, np.matmul(deltarho, x - yrhop)))
        return alpha

    def potentialmat(self):
        """Return the potential part of the hamiltonian"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        potential_mat = np.zeros((dim, dim), dtype=np.complex128)
        EJlist = self.EJlist
        exp_prod_boundary_coeff = np.exp(-0.25 * np.sum([self.boundary_coeffs[j]
                                                         * self.boundary_coeffs[k]
                                                         * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                                         for j in range(self.num_deg_freedom)
                                                         for k in range(self.num_deg_freedom)]))
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi, a_op_list)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum) = exp_list
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom) - np.matmul(rho, rhoprime)))
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.num_deg_freedom)
                klist = itertools.filterfalse(lambda e: self._filter_jkvals(e, minima_diff, Xi_inv), klist)
                jkvals = next(klist, -1)
                while jkvals != -1:
                    phik = 2.0 * np.pi * np.array([jkvals[i] for i in range(self.num_deg_freedom)])
                    delta_phi_kpm = phik - (minima_m - minima_p)
                    phibar_kpm = 0.5 * (phik + (minima_m + minima_p))
                    exp_prod_coeff = self._exp_prod_coeff(delta_phi_kpm, Xi_inv, sigma, sigmaprime)

                    (exp_adag, exp_a) = self._V_op_builder(exp_adag_list, exp_a_list, jkvals)
                    exp_adag = np.matmul(exp_adag_mindiff, exp_adag)
                    exp_adag = np.matmul(exp_adag_adag, exp_adag)
                    exp_a = np.matmul(exp_a, exp_a_mindiff)
                    exp_a = np.matmul(exp_a, exp_a_a)

                    exp_i_phi_list = np.array([exp_i_list[i] * np.exp(1j * phibar_kpm[i])
                                               for i in range(self.num_deg_freedom)])
                    exp_i_phi_sum_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                                        * np.prod([np.exp(1j * self.boundary_coeffs[i] * phibar_kpm[i])
                                                   for i in range(self.num_deg_freedom)]))

                    for num in range(self.num_deg_freedom):  # summing over potential terms cos(\phi_x)
                        x = (np.matmul(delta_phi_kpm, Xi_inv.T) + 1j * Xi[num, :]) / np.sqrt(2.)
                        y = (-np.matmul(delta_phi_kpm, Xi_inv.T) + 1j * Xi[num, :]) / np.sqrt(2.)

                        alpha = scale * self._alpha_helper(x, y, rhoprime, deltarho)
                        alpha_con = scale * self._alpha_helper(x.conjugate(), y.conjugate(),
                                                               rhoprime, deltarho)

                        potential_temp = -0.5 * EJlist[num] * alpha * exp_i_phi_list[num]
                        potential_temp += -0.5 * EJlist[num] * alpha_con * exp_i_phi_list[num].conjugate()
                        potential_temp = np.matmul(exp_adag, np.matmul(potential_temp, exp_a))
                        potential_temp *= (np.exp(-.25 * np.dot(Xi[num, :], np.transpose(Xi)[:, num]))
                                           * exp_prod_coeff)

                        potential_mat[m * num_exc_tot:m * num_exc_tot + num_exc_tot,
                                      p * num_exc_tot:p * num_exc_tot + num_exc_tot] += potential_temp

                    # cos(sum-2\pi f)
                    x = (np.matmul(delta_phi_kpm, Xi_inv.T)
                         + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                   for i in range(self.num_deg_freedom)], axis=0)) / np.sqrt(2.)
                    y = (- np.matmul(delta_phi_kpm, Xi_inv.T)
                         + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                   for i in range(self.num_deg_freedom)], axis=0)) / np.sqrt(2.)
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

                    jkvals = next(klist, -1)

        for m, minima_m in enumerate(minima_list):
            for p in range(m + 1, len(minima_list)):
                potential_temp = potential_mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                                               p * num_exc_tot: p * num_exc_tot + num_exc_tot]
                potential_mat[p * num_exc_tot: p * num_exc_tot + num_exc_tot,
                              m * num_exc_tot: m * num_exc_tot + num_exc_tot] += potential_temp.conjugate().T

        return potential_mat

    def hamiltonian(self):
        """Construct the Hamiltonian"""
        return (self.kineticmat() + self.potentialmat())

    def inner_product(self):
        """Return the inner product matrix, which is nontrivial with VCHOS states"""
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom)])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        inner_product_mat = np.zeros((dim, dim), dtype=np.complex128)
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_p = minima_list[p]
                minima_diff = minima_p - minima_m
                (exp_list, rho, rhoprime, sigma, sigmaprime,
                 deltarho, deltarhobar, zp, zpp) = self._build_squeezing_ops(m, p, minima_diff, Xi, a_op_list)
                (exp_adag_adag, exp_a_a, exp_adag_a,
                 exp_adag_list, exp_adag_mindiff,
                 exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum) = exp_list
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom) - np.matmul(rho, rhoprime)))
                klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=self.num_deg_freedom)
                klist = itertools.filterfalse(lambda e: self._filter_jkvals(e, minima_diff, Xi_inv), klist)
                jkvals = next(klist, -1)
                while jkvals != -1:
                    phik = 2.0 * np.pi * np.array([jkvals[i] for i in range(self.num_deg_freedom)])
                    delta_phi_kpm = phik - (minima_m - minima_p)
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
                    jkvals = next(klist, -1)

        for m, minima_m in enumerate(minima_list):
            for p in range(m + 1, len(minima_list)):
                inner_temp = inner_product_mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                                               p * num_exc_tot: p * num_exc_tot + num_exc_tot]
                inner_product_mat[p * num_exc_tot: p * num_exc_tot + num_exc_tot,
                                  m * num_exc_tot: m * num_exc_tot + num_exc_tot] += inner_temp.conjugate().T

        return inner_product_mat

    def _V_op_builder(self, exp_adag_list, exp_a_list, jkvals):
        num_exc_tot = exp_adag_list[0].shape[0]
        V_op_dag = np.eye(num_exc_tot)
        for j in range(self.num_deg_freedom):
            V_op_dag_temp = np.linalg.matrix_power(exp_adag_list[j], jkvals[j])
            V_op_dag = np.matmul(V_op_dag, V_op_dag_temp)

        V_op = np.eye(num_exc_tot)
        for j in range(self.num_deg_freedom):
            V_op_temp = np.linalg.matrix_power(exp_a_list[j], -jkvals[j])
            V_op = np.matmul(V_op, V_op_temp)

        return V_op_dag, V_op

    def _full_o(self, operators, indices):
        i_o = np.eye(self.num_exc + 1)
        i_o_list = [i_o for _ in range(self.num_deg_freedom)]
        product_list = i_o_list[:]
        oi_list = zip(operators, indices)
        for oi in oi_list:
            product_list[oi[1]] = oi[0]
        full_op = self._kron_matrix_list(product_list)
        return full_op

    def _kron_matrix_list(self, matrix_list):
        output = matrix_list[0]
        for matrix in matrix_list[1:]:
            output = np.kron(output, matrix)
        return output

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        inner_product_mat = self.inner_product()
        evals = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat,
                               eigvals_only=True, eigvals=(0, evals_count - 1))
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        inner_product_mat = self.inner_product()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, b=inner_product_mat,
                                      eigvals_only=False, eigvals=(0, evals_count - 1))
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def sorted_minima(self):
        pass

    def build_capacitance_matrix(self):
        pass

    def build_EC_matrix(self):
        pass
