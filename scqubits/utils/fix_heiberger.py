import numpy as np
import scipy as sp
from scipy import linalg


def fixheiberger(A, B, num_eigvals=6):
    """Implement the Fix-Heiberger method for dealing
    with ill-conditioned generalized eigenvalue problems

    Parameters
    ----------
    A : ndarray
        matrix that plays the role of the Hamiltonian
    B : ndarray
        matrix of the same dimension as A that plays the role
        of the inner-product matrix
    num_eigvals : int
        number of eigenvalues to return

    Returns
    -------
    ndarray
        matrix of reduced dimension that is to be diagonalized
    """
    global ind2
    n = A.shape[0]
    D0, Q1 = linalg.eigh(B)
    index_array_D0 = np.argsort(D0)[::-1]  # sort in descending order
    D0 = D0[index_array_D0]
    Q1 = Q1[:, index_array_D0]
    epsilon = 10.0 * np.abs(
        D0[-1] / D0[0]
    )  # Choose epsilon such that all negative eigenvalues are neglected
    evals_list = []
    converged = False
    n1 = 0
    j = 0
    while not converged:
        j += 1
        epsilon, n1, n2 = _epsilon_update(epsilon, D0, n1)
        if n2 > n1:  # This will only occur for large epsilon
            raise ConvergenceError("Convergence as a function of epsilon not achieved.")
        # partition D0 so that the offending eigenvalues are in the
        # bottom right corner of the matrix and set them to zero (F0_22 = 0)
        D0_11 = D0[:n1]
        # Apply the same transformation to A
        A0 = np.matmul(Q1.conjugate().T, np.matmul(A, Q1))
        # Apply the congruent transformation to A and B
        # that takes B to the identity matrix aside from the
        # offending B eigenvalues which are set to zero
        R1 = np.zeros((n, n))
        R1[0:n1, 0:n1] = np.diag(np.sqrt(1.0 / D0_11))
        R1[n1:n, n1:n] = np.eye(n2)
        A1 = np.matmul(R1.conjugate().T, np.matmul(A0, R1))
        A1_22 = A1[n1:n, n1:n]
        # Diagonalize A1_22, separate out offending
        # eigenvalues here as well
        D2, Q2_22 = linalg.eigh(A1_22)
        index_array_D2 = np.argsort(D2)[::-1]  # sort in descending order
        D2 = D2[index_array_D2]
        Q2_22 = Q2_22[:, index_array_D2]
        # Separate out eigenvalues of the A1_22 matrix that are
        # similarly below the epsilon threshold
        if np.abs(D2[0]) < epsilon:
            n3 = 0
            n4 = n2
        else:
            for num2 in range(n2 - 1, -1, -1):
                if np.abs(D2[num2]) > epsilon * np.abs(D2[0]):
                    ind2 = num2
                    break
                if num2 == 0:
                    ind2 = -1
            n3 = ind2 + 1
            n4 = n2 - n3
        Q2 = np.zeros((n, n), dtype=np.complex_)
        Q2[0:n1, 0:n1] = np.eye(n1)
        Q2[n1:n, n1:n] = Q2_22
        A2 = np.matmul(Q2.conjugate().T, np.matmul(A1, Q2))
        if n4 == 0:
            A2_11 = A2[0:n1, 0:n1]
            A2_12 = A2[0:n1, n1:n]
            D2_33 = A2[n1:n, n1:n]
            offsetmat = np.matmul(
                A2_12, np.matmul(sp.linalg.inv(D2_33), A2_12.conjugate().T)
            )
            fh_mat = A2_11 - offsetmat
            evals = sp.linalg.eigh(
                fh_mat, eigvals_only=True, eigvals=(0, num_eigvals - 1)
            )
            evals_list.append(evals)
            if j != 1:
                converged = np.allclose(
                    evals_list[j - 2], evals_list[j - 1], rtol=1e-3, atol=1e-8
                )
        elif n3 != 0:
            # different number of offending eigenvalues for A and B. Note that
            # this is not the ideal situation, since we end up folding matrix
            # elements of A associated with the ill-conditioned eigenvectors
            # back into the problem
            D2_33 = D2[:n3]
            D2 = np.zeros((n2, n2))
            D2[0:n3, 0:n3] = np.diag(D2_33)
            # Transform A1 and B1 according to Q2
            A2[n1 + n3 :, n1 + n3 :] = np.zeros((n4, n4))
            A2_13 = A2[0:n1, n1 + n3 :]
            #        assert(np.linalg.matrix_rank(A2_13)==n4)
            # Reduce A2_13 to triangular form by Householder reflections
            Q3_11, R3_11, P3_11 = linalg.qr(A2_13, pivoting=True)
            Q3 = np.zeros((n, n), dtype=np.complex_)
            Q3[0:n1, 0:n1] = Q3_11
            Q3[n1 : n1 + n3, n1 : n1 + n3] = np.eye(n3)
            Q3[
                n1 + n3 : n, n1 + n3 : n
            ] = P3_11  # Because A P = Q R for qr decomposition
            A3 = np.matmul(Q3.conjugate().T, np.matmul(A2, Q3))
            A3_22 = A3[n4:n1, n4:n1]
            A3_23 = A3[n4:n1, n1 : n1 + n3]
            A3_32 = A3[n1 : n1 + n3, n4:n1]
            A3_33 = A3[n1 : n1 + n3, n1 : n1 + n3]
            A3_33_inv = linalg.inv(A3_33)
            fh_mat = A3_22 - np.matmul(A3_23, np.matmul(A3_33_inv, A3_32))
            evals = sp.linalg.eigh(
                fh_mat, eigvals_only=True, eigvals=(0, num_eigvals - 1)
            )
            evals_list.append(evals)
            if j != 1:
                converged = np.allclose(
                    evals_list[j - 2], evals_list[j - 1], rtol=1e-3, atol=1e-8
                )
        else:  # same number of offending eigenvalues for A and B
            A2_13 = A2[0:n1, n1:n]
            # Reduce A2_13 to triangular form by Householder reflections
            Q3_11, R3_11, P3_11 = linalg.qr(A2_13, pivoting=True)
            Q3 = np.zeros((n, n), dtype=np.complex_)
            Q3[0:n1, 0:n1] = Q3_11
            Q3[n1:n, n1:n] = P3_11  # Because A P = Q R for qr decomposition
            A3 = np.matmul(Q3.conjugate().T, np.matmul(A2, Q3))
            A3_22 = A3[n2:n1, n2:n1]
            fh_mat = A3_22
            evals = sp.linalg.eigh(
                fh_mat, eigvals_only=True, eigvals=(0, num_eigvals - 1)
            )
            evals_list.append(evals)
            if j != 1:
                converged = np.allclose(
                    evals_list[j - 2], evals_list[j - 1], rtol=1e-3, atol=1e-8
                )
    return evals_list[-2]


def _epsilon_update(epsilon, D0, n1):
    # The purpose of this function is to ensure that two different choices
    # of epsilon do not result in eliminating the same eigenvalues,
    # thereby leading to the same result and giving the illusion of convergence
    n1_new = n1
    n = D0.shape[0]
    while n1 == n1_new:
        for num in range(n - 1, -1, -1):
            if D0[num] > epsilon * D0[0]:
                ind = num
                break
            if epsilon > D0[0]:
                raise ConvergenceError(
                    "Convergence as a function of epsilon not achieved."
                )
        n1_new = ind + 1
        n2_new = n - n1_new
        epsilon *= 10.0
    return 0.1 * epsilon, n1_new, n2_new


class ConvergenceError(Exception):
    pass
