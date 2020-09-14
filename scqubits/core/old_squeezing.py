import itertools
import warnings
import math
from functools import partial

import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError
from scipy.special import factorial
from scipy.optimize import minimize
import scipy.constants as const

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.cpu_switch import get_map_method
from scqubits.utils.fix_heiberger import fixheiberger
from scqubits.utils.spectrum_utils import order_eigensystem, standardize_sign


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

class VCHOSOldSqueezing(base.QubitBaseClass, serializers.Serializable):
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
                          for j in range(self.num_deg_freedom())])
        pot_sum += (-self.EJlist[-1]
                    * np.cos(np.sum([self.boundary_coeffs[i] * phiarray[i]
                                     for i in range(self.num_deg_freedom())])
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
        Parameters
        ----------
        i: int
            integer specifying which minimum to linearize around, 0<=i<= total number of minima
        Returns
        -------
        ndarray

        """
        gmat = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))

        min_loc = self.sorted_minima()[i]
        gamma_list = self.EJlist / self.Phi0 ** 2

        gamma_diag = np.diag([gamma_list[j] * np.cos(min_loc[j])
                              for j in range(self.num_deg_freedom())])
        gmat += gamma_diag

        min_loc_bound_sum = np.sum([self.boundary_coeffs[j] * min_loc[j]
                                    for j in range(self.num_deg_freedom())])
        for j in range(self.num_deg_freedom()):
            for k in range(self.num_deg_freedom()):
                gmat[j, k] += (gamma_list[-1] * self.boundary_coeffs[j] * self.boundary_coeffs[k]
                               * np.cos(min_loc_bound_sum + 2 * np.pi * self.flux))

        return gmat

    def _eigensystem_normal_modes(self, i):
        """Return squared normal mode frequencies, matrix of eigenvectors"""
        Cmat = self.build_capacitance_matrix()
        gmat = self.build_gamma_matrix(i)

        omegasq, eigvec = sp.linalg.eigh(gmat, b=Cmat)
#        eigvec = self._standardize_sign_Xi(eigvec)
        return omegasq, eigvec

    @staticmethod
    def _standardize_sign_Xi(normal_mode_eigenvectors):
        new_normal_mode_eigenvector = np.zeros_like(normal_mode_eigenvectors)
        for i, eigenvector in enumerate(normal_mode_eigenvectors.T):
            for elem in eigenvector:
                if not np.allclose(np.abs(elem), 0.0, atol=1e-3):
                    new_normal_mode_eigenvector[i, :] = np.sign(elem) * eigenvector
                    break
        return new_normal_mode_eigenvector.T

    def omegamat(self, i):
        """Return a diagonal matrix of the normal mode frequencies of a given minimim """
        omegasq, _ = self._eigensystem_normal_modes(i)
        return np.diag(np.sqrt(omegasq))

    def oscillator_lengths(self, i):
        """Return oscillator lengths of the mode frequencies for a given minimum"""
        omegasq, eigvec = self._eigensystem_normal_modes(i)
        omega = np.sqrt(omegasq)
        diag_norm = np.matmul(eigvec.T, eigvec)
        norm_eigvec = np.array([eigvec[:, mu] / np.sqrt(diag_norm[mu, mu]) for mu in range(self.num_deg_freedom())]).T
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
        deltarhoprime = np.matmul(sp.linalg.inv(np.eye(self.num_deg_freedom())
                                                - np.matmul(rhoprime, rho)), rhoprime)
        deltarho = np.matmul(sp.linalg.inv(np.eye(self.num_deg_freedom())
                                           - np.matmul(rho, rhoprime)), rho)
        deltarhobar = sp.linalg.logm(sp.linalg.inv(np.eye(self.num_deg_freedom()) - np.matmul(rhoprime, rho)))
        z = 1j * np.transpose(Xi_inv) / np.sqrt(2.)
        zp = (z + 0.5 * np.matmul(np.matmul(z, rhoprime), deltarho + deltarho.T)
              + 0.5 * np.matmul(z, deltarho + deltarho.T))
        zpp = np.matmul(z, rhoprime) + z
        return deltarho, deltarhoprime, deltarhobar, zp, zpp

    def a_operator(self, mu):
        """Return the lowering operator associated with the mu^th d.o.f. in the full Hilbert space"""
        a = np.array([np.sqrt(num) for num in range(1, self.num_exc + 1)])
        a_mat = np.diag(a, k=1)
        return self._full_o([a_mat], [mu])

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
        omegamat = self.omegamat(0)
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

    def _normal_ordered_adag_a_exponential(self, x):
        """Return normal ordered exponential matrix of exp(a_{i}^{\dagger}x_{ij}a_{j})"""
        expx = sp.linalg.expm(x)
        dim = self.a_operator(0).shape[0]
        result = np.eye(dim, dtype=np.complex128)
        additionalterm = np.eye(dim, dtype=np.complex128)
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom())])
        k = 1
        while not np.allclose(additionalterm, np.zeros((dim, dim))):
            additionalterm = np.sum([((expx - np.eye(self.num_deg_freedom()))[i, j]) ** k
                                     * (factorial(k)) ** (-1)
                                     * np.matmul(np.linalg.matrix_power(a_op_list[i].T, k),
                                                 np.linalg.matrix_power(a_op_list[j], k))
                                     for i in range(self.num_deg_freedom())
                                     for j in range(self.num_deg_freedom())], axis=0)
            result += additionalterm
            k += 1
        return result

    def _find_k_values_for_different_minima(self, num_cpus=1):
        """
        We have found that specifically this part of the code is quite slow, that
        is finding the relevant nearest neighbor, next nearest neighbor, etc. lattice vectors
        that meaningfully contribute. This is a calculation that previously had to be done
        for the kinetic, potential and inner product matrices separately, even though
        the results were the same for all three matrices. This helper function allows us to only
        do it once.
        """
        target_map = get_map_method(num_cpus)
        Xi_inv = sp.linalg.inv(self.Xi_matrix())
        minima_list = self.sorted_minima()
        nearest_neighbors = []
        for m, minima_m in enumerate(minima_list):
            for p in range(m, len(minima_list)):
                minima_diff = minima_list[p] - minima_m
                if (m == p) and (m != 0):  # vectors will be the same as m=p=0
                    nearest_neighbors.append(nearest_neighbors[0])
                else:
                    periodic_vector_lengths = np.array([i for i in range(1, self.kmax + 1)])
                    filter_function = partial(self._filter_periodic_vectors, minima_diff, Xi_inv)
                    filtered_vectors = list(target_map(filter_function, periodic_vector_lengths))
                    zero_vec = np.zeros(self.num_deg_freedom())
                    if self._filter_neighbors(minima_diff, Xi_inv, zero_vec):
                        filtered_vectors.append(zero_vec)
                    nearest_neighbors_single = self._stack_filtered_vectors(filtered_vectors)
                    nearest_neighbors.append(nearest_neighbors_single)
                print("completed m={m}, p={p} minima pair computation".format(m=m, p=p))
        return nearest_neighbors

    @staticmethod
    def _stack_filtered_vectors(filtered_vectors):
        filtered_vectors = list(filter(lambda x: len(x) != 0, filtered_vectors))
        return np.vstack(filtered_vectors)

    def _filter_periodic_vectors(self, minima_diff, Xi_inv, periodic_vector_length):
        sites = self.num_deg_freedom()
        filtered_vectors = []
        prev_vec = np.zeros(sites, dtype=int)
        prev_vec[0] = periodic_vector_length
        if periodic_vector_length <= 2:
            self._filter_reflected_vectors(minima_diff, Xi_inv, prev_vec, filtered_vectors)
        while prev_vec[-1] != periodic_vector_length:
            next_vec = self._generate_next_vec(prev_vec, periodic_vector_length)
            if len(np.argwhere(next_vec > 2)) == 0:
                self._filter_reflected_vectors(minima_diff, Xi_inv, next_vec, filtered_vectors)
            prev_vec = next_vec
        return np.array(filtered_vectors)

    def _filter_reflected_vectors(self, minima_diff, Xi_inv, vec, filtered_vectors):
        reflected_vectors = self._reflect_vectors(vec)
        filter_function = partial(self._filter_neighbors, minima_diff, Xi_inv)
        new_vectors = filter(filter_function, reflected_vectors)
        for filtered_vec in new_vectors:
            filtered_vectors.append(filtered_vec)

    @staticmethod
    def _reflect_vectors(vec):
        reflected_vec_list = []
        nonzero_indices = np.nonzero(vec)
        nonzero_vec = vec[nonzero_indices]
        multiplicative_factors = itertools.product(np.array([1, -1]), repeat=len(nonzero_vec))
        for factor in multiplicative_factors:
            reflected_vec = np.copy(vec)
            np.put(reflected_vec, nonzero_indices, np.multiply(nonzero_vec, factor))
            reflected_vec_list.append(reflected_vec)
        return reflected_vec_list

    def _filter_neighbors(self, minima_diff, Xi_inv, neighbor):
        """
        Want to eliminate periodic continuation terms that are irrelevant, i.e.,
        they add nothing to the transfer matrix. These can be identified as each term
        is suppressed by a gaussian exponential factor. If the argument np.dot(dpkX, dpkX)
        of the exponential is greater than 180.0, this results in a suppression of ~10**(-20),
        and so can be safely neglected.

        Assumption is that extended degrees of freedom precede the periodic d.o.f.
        """
        phi_neighbor = 2.0 * np.pi * neighbor
        dpkX = Xi_inv @ (phi_neighbor + minima_diff)
        prod = np.exp(-0.25 * np.dot(dpkX, dpkX))
        return prod > 1e-15

    @staticmethod
    def _generate_next_vec(prev_vec, radius):
        k = 0
        for num in range(len(prev_vec) - 2, -1, -1):
            if prev_vec[num] != 0:
                k = num
                break
        next_vec = np.zeros_like(prev_vec)
        next_vec[0:k] = prev_vec[0:k]
        next_vec[k] = prev_vec[k] - 1
        next_vec[k + 1] = radius - np.sum([next_vec[i] for i in range(k + 1)])
        return next_vec

    def _build_squeezing_ops(self, m, p, minima_diff, Xi, a_op_list, potential=True):
        """
        Build all operators relevant for building the Hamiltonian. If there is no squeezing,
        this routine then just builds the translation operators necessary for periodic
        continuation, as well as the exp(i\phi_{j}) operators for the potential
        """
        if self.squeezing:
            if m == 0:  # At the global minimum, no squeezing required
                rho = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))
                sigma = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))
                tau = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))
            else:
                rho, sigma, tau = self._build_U_squeezing_operator(m, Xi)
            if p == 0:
                rhoprime = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))
                sigmaprime = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))
                tauprime = np.zeros((self.num_deg_freedom(), self.num_deg_freedom()))
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
                                                   for i in range(self.num_deg_freedom())
                                                   for j in range(self.num_deg_freedom())], axis=0))
            exp_a_a = sp.linalg.expm(np.sum([prefactor_a_a[i, j]
                                             * np.matmul(a_op_list[i],
                                                         a_op_list[j])
                                             for i in range(self.num_deg_freedom())
                                             for j in range(self.num_deg_freedom())], axis=0))
            exp_adag_a = self._normal_ordered_adag_a_exponential(prefactor_adag_a)
        else:
            # We will call squeezing operators later, so must define them to be 0 or 1 where appropriate
            N = self.num_deg_freedom()
            dim = a_op_list[0].shape[0]
            rho, sigma, tau = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            rhoprime, sigmaprime, tauprime = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            deltarho, deltarhoprime, deltarhobar, zp, zpp = self._define_squeezing_variables(rho, rhoprime, Xi)
            expsigma, expsigmaprime = np.eye(N), np.eye(N)
            expdeltarhobar, expdrbs = np.eye(N), np.eye(N)

            exp_adag_adag, exp_a_a, exp_adag_a = np.eye(dim), np.eye(dim), np.eye(dim)

        Xi_inv = sp.linalg.inv(Xi)

        # For the case of the translation operators
        prefactor_adag = np.matmul(np.eye(self.num_deg_freedom()) + rhoprime, expdrbs)
        a_temp_coeff = 0.5 * np.matmul(np.eye(self.num_deg_freedom()) + rhoprime, deltarho + deltarho.T)
        prefactor_a = np.matmul(np.eye(self.num_deg_freedom()) + a_temp_coeff, expsigmaprime)

        exp_adag_list = []
        for j in range(self.num_deg_freedom()):
            exp_adag_j = sp.linalg.expm(np.sum([2.0 * np.pi * (np.matmul(Xi_inv.T, prefactor_adag)[j, i] / np.sqrt(2.0))
                                                * a_op_list[i].T for i in range(self.num_deg_freedom())], axis=0))
            exp_adag_list.append(exp_adag_j)

        exp_adag_mindiff = sp.linalg.expm(np.sum([minima_diff[x] * np.matmul(Xi_inv.T, prefactor_adag)[x, i]
                                                  * a_op_list[i].T for x in range(self.num_deg_freedom())
                                                  for i in range(self.num_deg_freedom())], axis=0) / np.sqrt(2.0))
        exp_a_list = []
        for j in range(self.num_deg_freedom()):
            exp_a_j = sp.linalg.expm(np.sum([2.0 * np.pi * (np.matmul(Xi_inv.T, prefactor_a)[j, i] / np.sqrt(2.0))
                                             * a_op_list[i] for i in range(self.num_deg_freedom())], axis=0))
            exp_a_list.append(exp_a_j)

        exp_a_mindiff = sp.linalg.expm(np.sum([-minima_diff[x] * np.matmul(Xi_inv.T, prefactor_a)[x, i]
                                               * a_op_list[i] for x in range(self.num_deg_freedom())
                                               for i in range(self.num_deg_freedom())], axis=0) / np.sqrt(2.0))

        # Now the potential operators
        exp_i_list = []
        exp_i_sum = 0.0
        if potential:
            prefactor_adag = np.matmul(np.eye(self.num_deg_freedom()) - rhoprime, expdrbs)
            a_temp_coeff = 0.5 * np.matmul(np.eye(self.num_deg_freedom()) - rhoprime, deltarho + deltarho.T)
            prefactor_a = np.matmul(np.eye(self.num_deg_freedom()) - a_temp_coeff, expsigmaprime)

            Xid = 1j * np.matmul(Xi, prefactor_adag) / np.sqrt(2.0)
            Xia = 1j * np.matmul(Xi, prefactor_a) / np.sqrt(2.0)
            for j in range(self.num_deg_freedom()):
                exp_i_j_adag_part = sp.linalg.expm(np.sum([Xid[j, i] * a_op_list[i].T
                                                           for i in range(self.num_deg_freedom())], axis=0))
                exp_i_j_a_part = sp.linalg.expm(np.sum([Xia[j, i] * a_op_list[i]
                                                        for i in range(self.num_deg_freedom())], axis=0))
                exp_i_j = np.matmul(exp_i_j_adag_part, np.matmul(exp_adag_a, exp_i_j_a_part))
                exp_i_list.append(exp_i_j)

            exp_i_sum_adag_part = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                         Xid[j, i] * a_op_list[i].T
                                                         for i in range(self.num_deg_freedom())
                                                         for j in range(self.num_deg_freedom())], axis=0))
            exp_i_sum_a_part = sp.linalg.expm(np.sum([self.boundary_coeffs[j] *
                                                      Xia[j, i] * a_op_list[i]
                                                      for i in range(self.num_deg_freedom())
                                                      for j in range(self.num_deg_freedom())], axis=0))
            exp_i_sum = np.matmul(exp_i_sum_adag_part, np.matmul(exp_adag_a, exp_i_sum_a_part))

        exp_list = [exp_adag_adag, exp_a_a, exp_adag_a,
                    exp_adag_list, exp_adag_mindiff,
                    exp_a_list, exp_a_mindiff, exp_i_list, exp_i_sum]

        return exp_list, rho, rhoprime, sigma, sigmaprime, deltarho, deltarhobar, zp, zpp

    def _premultiplying_exp_adag_a_with_a(self, exp_adag_a, a_op_list):
        """
        Helper function for building the kinetic part of the Hamiltonian.
        Naming scheme is  x -> exp(A_{ij}a_{i}^{\dag}a_{j}) (for whatever matrix A is)
                          a -> a_{i}
                          d -> a_{i}^{\dag}
        """
        xa = np.array([np.matmul(exp_adag_a, a_op_list[mu])
                       for mu in range(self.num_deg_freedom())])
        xaa = np.array([np.matmul(xa[mu], a_op_list[mu])
                        for mu in range(self.num_deg_freedom())])
        dxa = np.array([np.matmul(a_op_list[mu].T, xa[mu])
                        for mu in range(self.num_deg_freedom())])
        dx = np.array([np.matmul(a_op_list[mu].T, exp_adag_a)
                       for mu in range(self.num_deg_freedom())])
        ddx = np.array([np.matmul(a_op_list[mu].T, dx[mu])
                        for mu in range(self.num_deg_freedom())])
        return xa, xaa, dxa, dx, ddx

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
        return prod > 136.0 #92.0  # 180

    def _exp_prod_coeff(self, delta_phi_kpm, Xi_inv, sigma, sigmaprime):
        """
        Overall multiplicative factor. Includes offset charge,
        Gaussian suppression factor in the absence of squeezing. With squeezing,
        also includes exponential of trace over sigma and sigmaprime, see Qin et. al
        """
        dpkX = np.matmul(Xi_inv, delta_phi_kpm)
        nglist = self.nglist
        return (np.exp(-1j * np.dot(nglist, delta_phi_kpm))
                * np.exp(-0.5 * np.trace(sigma) - 0.5 * np.trace(sigmaprime))
                * np.exp(-0.25 * np.dot(dpkX, dpkX)))

    def kineticmat(self, wrapper_klist=None):
        """Return the kinetic part of the hamiltonian"""
        if wrapper_klist is None:
            wrapper_klist = self._find_k_values_for_different_minima()
        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        EC_mat = self.build_EC_matrix()
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom())])
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
                                                       for mu in range(self.num_deg_freedom())], axis=0)
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom()) - np.matmul(rho, rhoprime)))
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
                                           for mu in range(self.num_deg_freedom())], axis=0)
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
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom())])
        num_exc_tot = a_op_list[0].shape[0]
        minima_list = self.sorted_minima()
        dim = len(minima_list) * num_exc_tot
        potential_mat = np.zeros((dim, dim), dtype=np.complex128)
        EJlist = self.EJlist
        exp_prod_boundary_coeff = np.exp(-0.25 * np.sum([self.boundary_coeffs[j]
                                                         * self.boundary_coeffs[k]
                                                         * np.dot(Xi[j, :], np.transpose(Xi)[:, k])
                                                         for j in range(self.num_deg_freedom())
                                                         for k in range(self.num_deg_freedom())]))
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
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom()) - np.matmul(rho, rhoprime)))
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
                                               for i in range(self.num_deg_freedom())])
                    exp_i_phi_sum_op = (exp_i_sum * np.exp(1j * 2.0 * np.pi * self.flux)
                                        * np.prod([np.exp(1j * self.boundary_coeffs[i] * phibar_kpm[i])
                                                   for i in range(self.num_deg_freedom())]))

                    for num in range(self.num_deg_freedom()):  # summing over potential terms cos(\phi_x)
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
                                   for i in range(self.num_deg_freedom())], axis=0)) / np.sqrt(2.)
                    y = (- np.matmul(delta_phi_kpm, Xi_inv.T)
                         + np.sum([1j * Xi[i, :] * self.boundary_coeffs[i]
                                   for i in range(self.num_deg_freedom())], axis=0)) / np.sqrt(2.)
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
        a_op_list = np.array([self.a_operator(i) for i in range(self.num_deg_freedom())])
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
                scale = 1. / np.sqrt(sp.linalg.det(np.eye(self.num_deg_freedom()) - np.matmul(rho, rhoprime)))
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

    def _populate_hermitian_matrix(self, mat, minima_list, num_exc_tot):
        """
        Return a fully Hermitian matrix, assuming that the input matrix has been
        populated with the upper right blocks
        """
        for m, minima_m in enumerate(minima_list):
            for p in range(m + 1, len(minima_list)):
                mat_temp = mat[m * num_exc_tot: m * num_exc_tot + num_exc_tot,
                           p * num_exc_tot: p * num_exc_tot + num_exc_tot]
                mat[p * num_exc_tot: p * num_exc_tot + num_exc_tot,
                m * num_exc_tot: m * num_exc_tot + num_exc_tot] += mat_temp.conjugate().T
        return mat

    def _V_op_builder(self, exp_adag_list, exp_a_list, jkvals):
        """
        Build translation operators using matrix_power rather than the
        more costly expm
        """
        num_exc_tot = exp_adag_list[0].shape[0]
        V_op_dag = np.eye(num_exc_tot)
        for j in range(self.num_deg_freedom()):
            V_op_dag_temp = np.linalg.matrix_power(exp_adag_list[j], int(jkvals[j]))
            V_op_dag = np.matmul(V_op_dag, V_op_dag_temp)

        V_op = np.eye(num_exc_tot)
        for j in range(self.num_deg_freedom()):
            V_op_temp = np.linalg.matrix_power(exp_a_list[j], -int(jkvals[j]))
            V_op = np.matmul(V_op, V_op_temp)

        return V_op_dag, V_op

    def _full_o(self, operators, indices):
        """Return operator in the full Hilbert space"""
        i_o = np.eye(self.num_exc + 1)
        i_o_list = [i_o for _ in range(self.num_deg_freedom())]
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
        wrapper_klist = self._find_k_values_for_different_minima()
        hamiltonian_mat = self.kineticmat(wrapper_klist) + self.potentialmat(wrapper_klist)
        inner_product_mat = self.inner_product(wrapper_klist)
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
        hamiltonian_mat = self.kineticmat(wrapper_klist) + self.potentialmat(wrapper_klist)
        inner_product_mat = self.inner_product(wrapper_klist)
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
        AA, BB, alpha, beta, Q, Z = sp.linalg.ordqz(hamiltonian_mat, inner_product_mat, sort=self._ordqz_sorter)
        a_max = np.max(np.abs(alpha))
        b_max = np.max(np.abs(beta))
        # filter ill-conditioned eigenvalues (alpha and beta values both small)
        alpha, beta = list(zip(*filter(lambda x: np.abs(x[0]) > 0.001 * a_max
                                                 and np.abs(x[1]) > 0.001 * b_max, zip(alpha, beta))))
        evals_qz = np.array(alpha) / np.array(beta)
        evals_qz = np.sort(np.real(list(filter(self._ordqz_filter, evals_qz))))[0: evals_count]
        evals_fh = fixheiberger(hamiltonian_mat, inner_product_mat, num_eigvals=evals_count, eigvals_only=True)
        assert (np.allclose(evals_qz, evals_fh))
        evals = evals_qz
        evecs = Z.T  # Need to ensure that this is the right way to produce eigenvectors
        if eigvals_only:
            return evals
        else:
            return evals, evecs

    def _ordqz_filter(self, a):
        if np.real(a) < 0:
            return False
        else:
            return True

    def _ordqz_sorter(self, alpha, beta):
        x = alpha / beta
        out = np.logical_and(np.real(x) > 0, np.abs(np.imag(x)) < 10 ** (-12))
        return out

    # The following four methods must be overridden in child classes
    def sorted_minima(self):
        return []

    def build_capacitance_matrix(self):
        return []

    def build_EC_matrix(self):
        return []

    def num_deg_freedom(self):
        return 0


# Current Mirror using VCHOS. Truncation scheme used is defining a cutoff num_exc
# of the number of excitations kept for each mode. The dimension of the hilbert space
# is then m*(num_exc+1)**(2*N - 1), where m is the number of inequivalent minima in
# the first unit cell and N is the number of big capacitors.

class CurrentMirrorVCHOS(VCHOSOldSqueezing):
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, kmax, num_exc, squeezing=False, truncated_dim=None):
        self.N = N
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist

        V_m = self._build_V_m()
        self.nglist = np.dot(sp.linalg.inv(V_m).T, nglist)[0:-1]

        self.flux = flux
        self.kmax = kmax
        self.num_exc = num_exc
        self.squeezing = squeezing
        self.hGHz = const.h * 10 ** 9
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.Z0 = 1. / (2 * self.e) ** 2
        self.Phi0 = 1. / (2 * self.e)
        self.boundary_coeffs = np.ones(2 * N - 1)

        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self.truncated_dim = truncated_dim

    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0 / 2.7,
            'ECg': 20.0,
            'EJlist': np.array(5 * [18.95]),
            'nglist': np.array(5 * [0.0]),
            'flux': 0.0,
            'kmax': 1,
            'num_exc': 2,
            'squeezing': False,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'kmax', 'num_exc', 'truncated_dim']

    def build_Cmat_full(self):
        N = self.N
        CB = self.e ** 2 / (2. * self.ECB)
        CJ = self.e ** 2 / (2. * self.ECJ)
        Cg = self.e ** 2 / (2. * self.ECg)

        Cmat = np.diagflat([Cg + 2 * CJ + CB for _ in range(2 * N)], 0)
        Cmat += np.diagflat([- CJ for _ in range(2 * N - 1)], +1)
        Cmat += np.diagflat([- CJ for _ in range(2 * N - 1)], -1)
        Cmat += np.diagflat([- CB for _ in range(N)], +N)
        Cmat += np.diagflat([- CB for _ in range(N)], -N)
        Cmat[0, -1] = Cmat[-1, 0] = - CJ

        return Cmat

    def build_capacitance_matrix(self):
        Cmat = self.build_Cmat_full()

        V_m_inv = sp.linalg.inv(self._build_V_m())
        Cmat = np.matmul(V_m_inv.T, np.matmul(Cmat, V_m_inv))

        return Cmat[0:-1, 0:-1]

    def _build_V_m(self):
        N = self.N
        V_m = np.diagflat([-1 for _ in range(2 * N)], 0)
        V_m += np.diagflat([1 for _ in range(2 * N - 1)], 1)
        V_m[-1] = np.array([1 for _ in range(2 * N)])

        return V_m

    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        Cmat = self.build_capacitance_matrix()
        return 0.5 * self.e ** 2 * sp.linalg.inv(Cmat)

    def hilbertdim(self):
        """Return N if the size of the Hamiltonian matrix is NxN"""
        return len(self.sorted_minima()) * (self.num_exc + 1) ** (2 * self.N - 1)

    def num_deg_freedom(self):
        return 2 * self.N - 1

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
            diff_array_reduced = np.array([np.mod(x, 2 * np.pi) for x in diff_array])
            elem_bool = True
            for elem in diff_array_reduced:
                # if every element is zero or 2pi, then we have a repeated minima
                elem_bool = elem_bool and (np.allclose(elem, 0.0, atol=1e-3)
                                           or np.allclose(elem, 2 * np.pi, atol=1e-3))
            if elem_bool:
                new_minima_bool = False
                break
        return new_minima_bool

    def find_minima(self):
        """
        Index all minima
        """
        minima_holder = []
        N = self.N
        for m in range(int(math.ceil(N / 2 - np.abs(self.flux))) + 1):
            guess_pos = np.array([np.pi * (m + self.flux) / N for _ in range(self.num_deg_freedom())])
            guess_neg = np.array([np.pi * (-m + self.flux) / N for _ in range(self.num_deg_freedom())])
            result_pos = minimize(self.potential, guess_pos)
            result_neg = minimize(self.potential, guess_neg)
            new_minimum_pos = self._check_if_new_minima(result_pos.x, minima_holder)
            if new_minimum_pos and result_pos.success:
                minima_holder.append(np.array([np.mod(elem, 2 * np.pi) for elem in result_pos.x]))
            new_minimum_neg = self._check_if_new_minima(result_neg.x, minima_holder)
            if new_minimum_neg and result_neg.success:
                minima_holder.append(np.array([np.mod(elem, 2 * np.pi) for elem in result_neg.x]))
        return minima_holder

    def sorted_minima(self):
        """Sort the minima based on the value of the potential at the minima """
        minima_holder = self.find_minima()
        value_of_potential = np.array([self.potential(minima_holder[x])
                                       for x in range(len(minima_holder))])
        sorted_value_holder = np.array([x for x, _ in
                                        sorted(zip(value_of_potential, minima_holder), key=lambda x: x[0])])
        sorted_minima_holder = np.array([x for _, x in
                                         sorted(zip(value_of_potential, minima_holder), key=lambda x: x[0])])
        # For efficiency purposes, don't want to displace states into minima
        # that are too high energy. Arbitrarily set a 40 GHz cutoff
        global_min = sorted_value_holder[0]
        dim = len(sorted_minima_holder)
        sorted_minima_holder = np.array([sorted_minima_holder[i] for i in range(dim)
                                         if sorted_value_holder[i] < global_min + 100.0])
        return sorted_minima_holder


class Hashing:
    def __init__(self):
        self.prime_list = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23,
                                    29, 31, 37, 41, 43, 47, 53, 59,
                                    61, 67, 71, 73, 79, 83, 89, 97,
                                    101, 103, 107, 109, 113, 127,
                                    131, 137, 139, 149, 151, 157,
                                    163, 167, 173, 179, 181, 191,
                                    193, 197, 199, 211, 223, 227,
                                    229, 233, 239, 241, 251, 257,
                                    263, 269, 271, 277, 281, 283,
                                    293, 307, 311, 313, 317, 331,
                                    337, 347, 349, 353, 359, 367,
                                    373, 379, 383, 389, 397, 401,
                                    409, 419, 421, 431, 433, 439,
                                    443, 449, 457, 461, 463, 467,
                                    479, 487, 491, 499, 503, 509,
                                    521, 523, 541, 547, 557, 563,
                                    569, 571, 577, 587, 593, 599,
                                    601, 607, 613, 617, 619, 631,
                                    641, 643, 647, 653, 659, 661,
                                    673, 677, 683, 691, 701, 709,
                                    719, 727, 733, 739, 743, 751,
                                    757, 761, 769, 773, 787, 797,
                                    809, 811, 821, 823, 827, 829,
                                    839, 853, 857, 859, 863, 877,
                                    881, 883, 887, 907, 911, 919,
                                    929, 937, 941, 947, 953, 967,
                                    971, 977, 983, 991, 997])

    def _hash(self, vec):
        dim = len(vec)
        return np.sum([np.sqrt(self.prime_list[i]) * vec[i] for i in range(dim)])

    def _gen_tags(self, basis_vecs):
        dim = basis_vecs.shape[0]
        tag_list = np.array([self._hash(basis_vecs[i, :]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return (tag_list, index_array)

    def _gen_basis_vecs(self):
        sites = self.num_deg_freedom()
        vec_list = []
        vec_list.append(np.zeros(sites))
        for total_exc in range(1, self.global_exc + 1):  # No excitation number conservation as in [1]
            prev_vec = np.zeros(sites)
            prev_vec[0] = total_exc
            vec_list.append(prev_vec)
            while prev_vec[-1] != total_exc:  # step through until the last entry is total_exc
                k = self._find_k(prev_vec)
                next_vec = np.zeros(sites)
                next_vec[0:k] = prev_vec[0:k]
                next_vec[k] = prev_vec[k] - 1
                next_vec[k + 1] = total_exc - np.sum([next_vec[i] for i in range(k + 1)])
                vec_list.append(next_vec)
                prev_vec = next_vec
        return np.array(vec_list)

    def eigvec_population(self, eigvec):
        basis_vecs = self._gen_basis_vecs()
        dim = len(basis_vecs)
        pop_list = []
        min_list = []
        vec_list = []
        for k, elem in enumerate(eigvec):
            if not np.allclose(elem, 0.0, atol=1e-4):
                minimum = math.floor(k / dim)
                pop_list.append(elem)
                min_list.append(minimum)
                vec_list.append(basis_vecs[np.mod(k, dim)])
        pop_list = np.abs(pop_list) ** 2
        index_array = np.argsort(np.abs(pop_list))
        pop_list = (pop_list[index_array])[::-1]
        min_list = (np.array(min_list)[index_array])[::-1]
        vec_list = (np.array(vec_list)[index_array])[::-1]
        return pop_list, zip(min_list, vec_list)

    def _find_k(self, vec):
        dim = len(vec)
        for num in range(dim - 2, -1, -1):
            if vec[num] != 0:
                return num


class CurrentMirrorVCHOSGlobalOld(CurrentMirrorVCHOS, Hashing):
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux,
                 kmax, global_exc, squeezing=False, truncated_dim=None):
        CurrentMirrorVCHOS.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux,
                                    kmax, num_exc=None, squeezing=squeezing, truncated_dim=truncated_dim)
        Hashing.__init__(self)
        self.global_exc = global_exc

    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0 / 2.7,
            'ECg': 20.0,
            'EJlist': np.array(5 * [18.95]),
            'nglist': np.array(5 * [0.0]),
            'flux': 0.0,
            'kmax': 1,
            'global_exc': 2,
            'squeezing': False,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'kmax', 'global_exc', 'truncated_dim']

    def a_operator(self, i):
        """
        This method for defining the a_operator is based on
        J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
        We ask the question, for each basis vector, what is the action of a_i
        on it? In this way, we can define a_i using a single for loop.
        """
        basis_vecs = self._gen_basis_vecs()
        tags, index_array = self._gen_tags(basis_vecs)
        dim = basis_vecs.shape[0]
        a = np.zeros((dim, dim))
        for w, vec in enumerate(basis_vecs):
            if vec[i] >= 1:
                temp_vec = np.copy(vec)
                temp_vec[i] = vec[i] - 1
                temp_coeff = np.sqrt(vec[i])
                temp_vec_tag = self._hash(temp_vec)
                index = np.searchsorted(tags, temp_vec_tag)
                basis_index = index_array[index]
                a[basis_index, w] = temp_coeff
        return a

    def hilbertdim(self):
        return len(self.sorted_minima()) * len(self._gen_basis_vecs())