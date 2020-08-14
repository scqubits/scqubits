import os

import numpy as np
import scipy as sp
from scipy.sparse import eye, diags
from scipy.sparse.linalg import eigsh
import itertools
import scipy.constants as const

import scqubits.core.qubit_base as base
import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.spectrum_utils import order_eigensystem
from scqubits.core.operators import operator_in_full_Hilbert_space


class CurrentMirrorFunctions:
    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux):
        self.e = np.sqrt(4.0 * np.pi * const.alpha)
        self.N = N
        self.number_degrees_freedom = 2*N - 1
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist
        self.nglist = nglist
        self.flux = flux

    def build_capacitance_matrix(self):
        """Returns the capacitance matrix, transforming to coordinates where the variable corresponding
        to the total charge can be eliminated

        Returns
        -------
            ndarray
        """
        N = self.N
        CB = self.e**2 / (2.*self.ECB)
        CJ = self.e**2 / (2.*self.ECJ)
        Cg = self.e**2 / (2.*self.ECg)

        C_matrix = np.diagflat([Cg + 2*CJ + CB for _ in range(2*N)], 0)
        C_matrix += np.diagflat([-CJ for _ in range(2*N - 1)], +1)
        C_matrix += np.diagflat([-CJ for _ in range(2*N - 1)], -1)
        C_matrix += np.diagflat([-CB for _ in range(N)], +N)
        C_matrix += np.diagflat([-CB for _ in range(N)], -N)
        C_matrix[0, -1] = C_matrix[-1, 0] = - CJ

        V_m_inv = sp.linalg.inv(self._build_V_m())
        C_matrix = np.matmul(V_m_inv.T, np.matmul(C_matrix, V_m_inv))

        return C_matrix[0:-1, 0:-1]

    def build_EC_matrix(self):
        """Returns the charging energy matrix

        Returns
        -------
            ndarray
        """
        C_matrix = self.build_capacitance_matrix()
        return 0.5 * self.e**2 * sp.linalg.inv(C_matrix)

    def _build_V_m(self):
        """Builds the matrix necessary for the coordinate transformation"""
        N = self.N
        V_m = np.diagflat([-1 for _ in range(2*N)], 0)
        V_m += np.diagflat([1 for _ in range(2*N - 1)], 1)
        V_m[-1] = np.array([1 for _ in range(2*N)])
        return V_m


class CurrentMirror(CurrentMirrorFunctions, base.QubitBaseClass, serializers.Serializable):
    r"""Current Mirror Qubit

    | [1] A. Kitaev, arXiv:cond-mat/0609441. https://arxiv.org/abs/cond-mat/0609441
    | [2] D. K. Weiss et al., Physical Review B, 100, 224507 (2019). https://doi.org/10.1103/PhysRevB.100.224507

    The current-mirror qubit as first described in [1] and analyzed numerically in [2]. Here we diagonalize
    the current-mirror qubit with each degree of freedom in the charge basis. We find that
    diagonalizing the current-mirror qubit Hamiltonian is possible with :math:`N=2, 3`, where :math:`N`
    is the number of big capacitors. For :math:`N=4` or larger, the Hamiltonian is too large to
    store in memory. For a given :math:`N`, the number of degrees of freedom is :math:`2*N - 1`. This is because
    while there are :math:`2*N` nodes, one variable is cyclic (corresponding to the net charge on the circuit).
    Diagonalization then proceeds in the coordinates where this variable has been eliminated.
    The Hamiltonian is given by

    .. math::

       H=&\sum_{i, j=1}^{2N-1}(n_{i}-n_{gi})4(E_\text{C})_{ij}(n_{j}-n_{gj}) \\
        -\sum_{i=1}^{2N-1}&E_{Ji}\cos\phi_{i}-E_{J2N}\cos(\sum_{i=1}^{2N-1}\phi_{i}+2\pi f),

    where the charging energy matrix is solved for numerically by inverting the capacitance matrix.
    Initialize with, for example::

        N = 2
        ECB = 0.2
        ECJ = 20.0/2.7
        ECg = 20.0
        EJ = 20.0
        EJlist = np.array([EJ for _ in range(2*N)])
        nglist = np.array([0.0 for _ in range(2*N-1)])
        flux = 0.0
        current_mirror = qubit.CurrentMirror(N, ECB, ECJ, ECg, EJlist, nglist, flux, ncut=10, truncated_dim=6)

    Parameters
    ----------
    N: int
        number of big capacitors
    ECB: float
        charging energy associated with the nominally identical big capacitors
    ECJ: float
        charging energy associated with the Josephson junctions
    ECg: float
        charging energy associated with the capacitive coupling to ground on each node
    EJlist: ndarray
        Josephson energies associated with each junction, which are allowed to vary. Must have size `2\cdot N`
    nglist: ndarray
        offset charge associated with each dynamical degree of freedom. Must have size `2\cdot N - 1`
    flux: float
        magnetic flux through the circuit loop, measured in units of the flux quantum
    ncut: int
        charge number cutoff for each degree of freedom,  `n = -ncut, ..., ncut`
    truncated_dim: int, optional
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    """
    N = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECB = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJlist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    nglist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ncut = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, N, ECB, ECJ, ECg, EJlist, nglist, 
                 flux, ncut, truncated_dim=None):
        CurrentMirrorFunctions.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux)
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/currentmirror.png')

    @staticmethod
    def default_params():
        return {
            'N': 3,
            'ECB': 0.2,
            'ECJ': 20.0/2.7,
            'ECg': 20.0,
            'EJlist': np.array(6*[18.95]),
            'nglist': np.array(5*[0.0]),
            'flux': 0.0,
            'ncut': 10,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['N', 'nglist', 'flux', 'ncut', 'truncated_dim']

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=False)
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=True)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs
    
    def hilbertdim(self):
        """Return Hilbert space dimension."""
        return (2*self.ncut+1)**self.number_degrees_freedom

    def hamiltonian(self):
        """Returns the Hamiltonian employing the charge number basis for all :math:`2\cdot N - 1` d.o.f.

        Returns
        -------
            ndarray
        """
        dim = self.number_degrees_freedom
        EC_matrix = self.build_EC_matrix()
        number_op = self._charge_number_operator()
        identity_op = self._identity_operator()
        identity_operator_list = self._identity_operator_list()
        
        H = 0.*self.identity_operator()
        for j, k in itertools.product(range(dim), range(dim)):
            if j != k:
                H += 4*EC_matrix[j, k]*operator_in_full_Hilbert_space([number_op - self.nglist[j]*identity_op,
                                                                       number_op - self.nglist[k]*identity_op],
                                                                      [j, k], identity_operator_list, sparse=True)
            else:
                n_squared = (number_op - self.nglist[j]*identity_op).dot(number_op - self.nglist[j]*identity_op)
                H += 4 * EC_matrix[j, j] * operator_in_full_Hilbert_space([n_squared], [j],
                                                                          identity_operator_list, sparse=True)
        for j in range(dim):
            H += (-self.EJlist[j]/2.)*(self.exp_i_phi_j_operator(j) + self.exp_i_phi_j_operator(j).T)
            H += self.EJlist[j]*self.identity_operator()
        H += (-self.EJlist[-1] / 2.) * np.exp(1j*2*np.pi*self.flux) * self.exp_i_phi_boundary_term().T
        H += (-self.EJlist[-1] / 2.) * np.exp(-1j*2*np.pi*self.flux) * self.exp_i_phi_boundary_term()
        H += self.EJlist[-1]*self.identity_operator()
        
        return H

    def _identity_operator(self):
        return eye(2 * self.ncut + 1, k=0, format="csr", dtype=np.complex_)

    def _identity_operator_list(self):
        return [self._identity_operator() for _ in range(self.number_degrees_freedom)]

    def identity_operator(self):
        """Returns the identity operator in the full Hilbert space

        Returns
        -------
            ndarray
        """
        return operator_in_full_Hilbert_space([], [], self._identity_operator_list(), sparse=True)

    def _charge_number_operator(self):
        return diags([i for i in range(-self.ncut, self.ncut + 1, 1)], offsets=0, format="csr", dtype=np.complex_)

    def charge_number_operator(self, j=0):
        """Returns charge number operator :math:`n_{j}` in the full Hilbert space

        Parameters
        ----------
        j: int
            specifies the degree of freedom

        Returns
        -------
            ndarray
        """
        number_operator = self._charge_number_operator()
        return operator_in_full_Hilbert_space([number_operator], [j], self._identity_operator_list(), sparse=True)

    def _exp_i_phi_j_operator(self):
        return eye(2*self.ncut + 1, k=-1, format="csr", dtype=np.complex_)

    def exp_i_phi_j_operator(self, j=0):
        """Returns the operator :math:`\exp(i\phi_{j})` in the full Hilbert space

        Parameters
        ----------
        j: int
            specifies the degree of freedom

        Returns
        -------
            ndarray
        """
        exp_i_phi_j = self._exp_i_phi_j_operator()
        return operator_in_full_Hilbert_space([exp_i_phi_j], [j], self._identity_operator_list(), sparse=True)

    def exp_i_phi_boundary_term(self):
        """Returns the operator associated with the last Josephson junction,
         :math:`\exp(i\sum_{j=1}^{2N-1}\phi_{j})` in the full Hilbert space

        Returns
        -------
            ndarray
        """
        dim = self.number_degrees_freedom
        exp_i_phi_op = self._exp_i_phi_j_operator()
        identity_operator_list = self._identity_operator_list()
        return operator_in_full_Hilbert_space([exp_i_phi_op for _ in range(dim)],
                                              [j for j in range(dim)], identity_operator_list, sparse=True)
