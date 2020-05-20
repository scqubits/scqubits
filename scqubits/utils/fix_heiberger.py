import numpy as np
import scipy as sp
from scipy import linalg


def fixheiberger(A, B, epsilon):
    """Implement the Fix-Heiberger method for dealing
    with ill-conditioned generalized eigenvalue problems

    Parameters
    ----------
    A : ndarray
        matrix that plays the role of the Hamiltonian
    B : ndarray
        matrix of the same dimension as A that plays the role
        of the inner-product matrix
    epsilon : float
        tolerance level for neglecting eigenvalues

    Returns
    -------
    ndarray
        matrix of reduced dimension that is to be diagonalized
    """
    n = A.shape[0]
    D0, Q1 = linalg.eigh(B)
    index_array = np.argsort(D0)[:: -1]  # sort in descending order
    D0 = D0[index_array]
    Q1 = Q1[:, index_array]
    # eliminate all eigenvalues of the inner product matrix
    # that are below the threshold epsilon
    for num in range(n-1, -1, -1):
        if D0[num] > epsilon*D0[0]:
            ind = num
            break
    n1 = ind + 1
    n2 = n - n1
    # partition D0 so that the offending eigenvalues are in the
    # bottom right corner of the matrix and set them to zero (F0_22 = 0)
    D0_11 = D0[:n1]
    D0 = np.zeros((n, n))
    D0[0:n1, 0:n1] = np.diag(D0_11)
    # Apply the same transformation to A
    A0 = np.matmul(Q1.conjugate().T, np.matmul(A, Q1))
    # Apply the congruent transformation to A and B
    # that takes B to the identity matrix aside from the
    # offending B eigenvalues which are set to zero
    R1 = np.zeros((n, n))
    R1[0:n1, 0:n1] = np.diag(np.sqrt(1./D0_11))
    R1[n1:n, n1:n] = np.eye(n2)
    A1 = np.matmul(R1.conjugate().T, np.matmul(A0, R1))
    A1_22 = A1[n1:n, n1:n]
    # Diagonalize A1_22, separate out offending
    # eigenvalues here as well
    D2, Q2_22 = linalg.eigh(A1_22)
    index_array = np.argsort(D2)[:: -1]  # sort in descending order
    D2 = D2[index_array]
    Q2_22 = Q2_22[:, index_array]
    # Separate out eigenvalues of the A1_22 matrix that are
    # similarly below the epsilon threshold
    for num2 in range(n2-1, -1, -1):
        if D2[num2] > epsilon*D2[0]:
            ind2 = num2
            break
        if num2 == 0:  # All are below the threshold, n3 = 0
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
        offsetmat = np.matmul(A2_12, np.matmul(sp.linalg.inv(D2_33), A2_12.conjugate().T))
        return A2_11 - offsetmat
    if n3 != 0:
        # different number of offending eigenvalues for A and B. Note that
        # this is not the ideal situation, since we end up folding matrix
        # elements of A associated with the ill-conditioned eigenvectors
        # back into the problem
        D2_33 = D2[:n3]
        D2 = np.zeros((n2, n2))
        D2[0:n3, 0:n3] = np.diag(D2_33)
        # Transform A1 and B1 according to Q2
        A2[n1+n3:, n1+n3:] = np.zeros((n4, n4))
        A2_13 = A2[0:n1, n1+n3:]
#        assert(np.linalg.matrix_rank(A2_13)==n4)
        # Reduce A2_13 to triangular form by Householder reflections
        Q3_11, R3_11, P3_11 = linalg.qr(A2_13, pivoting=True)
        Q3 = np.zeros((n, n), dtype=np.complex_)
        Q3[0:n1, 0:n1] = Q3_11
        Q3[n1:n1+n3, n1:n1+n3] = np.eye(n3)
        Q3[n1+n3:n, n1+n3:n] = P3_11  # Because A P = Q R for qr decomposition
        A3 = np.matmul(Q3.conjugate().T, np.matmul(A2, Q3))
        A3_22 = A3[n4:n1, n4:n1]
        A3_23 = A3[n4:n1, n1:n1+n3]
        A3_32 = A3[n1:n1+n3, n4:n1]
        A3_33 = A3[n1:n1+n3, n1:n1+n3]
        A3_33_inv = linalg.inv(A3_33)
        return A3_22 - np.matmul(A3_23, np.matmul(A3_33_inv, A3_32))
    else:  # same number of offending eigenvalues for A and B
        A2_13 = A2[0:n1, n1:n]
        # Reduce A2_13 to triangular form by Householder reflections
        Q3_11, R3_11, P3_11 = linalg.qr(A2_13, pivoting=True)
        Q3 = np.zeros((n, n))
        Q3[0:n1, 0:n1] = Q3_11
        Q3[n1:n, n1:n] = P3_11  # Because A P = Q R for qr decomposition
        A3 = np.matmul(Q3.conjugate().T, np.matmul(A2, Q3))
        A3_22 = A3[n2:n1, n2:n1]
        return A3_22

# Tests/examples

#delta = 1e-9
#A = np.diag([6, 5, 4, 3, 2, 1, 0, 0])
#A[0, 6] = A[1, 7] = 1
#A[6, 0] = A[7, 1] = 1
#B = np.diag([1, 1, 1, 1, delta, delta, delta, delta])

#A = fixheiberger(A, B, 1e-7)
#evals, evecs = linalg.eigh(A)
#assert(np.allclose(np.array([3, 4]), evals))

#A = np.diag([1, -1, 2, 3, 4, -3, 0, 0, 0, 0])
#A += np.diagflat(np.array([1, 1, 1, 1]), 6)
#A += np.diagflat(np.array([1, 1, 1, 1]), -6)
#A[0, 8] = A[8, 0] = 2
#A[1, 9] = A[9, 1] = 1
#B = np.diag([1, 2, 3, 2, 1, 1, 2*delta, 3*delta, delta, 2*delta])
#evals_t, evecs_t = sp.linalg.eigh(A, B)

#A = fixheiberger(A, B, 1e-7)
#evals, evecs = linalg.eigh(A)
#print(evals)
