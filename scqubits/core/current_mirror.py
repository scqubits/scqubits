import itertools
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import scipy as sp
from numpy import ndarray
from scipy.sparse import eye, diags
from scipy.sparse.linalg import eigsh

import scqubits.core.descriptors as descriptors
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.hashing_charge_basis import HashingChargeBasis
from scqubits.core.noise import NoisySystem
from scqubits.core.operators import operator_in_full_Hilbert_space
from scqubits.utils.spectrum_utils import order_eigensystem


class NoisyCurrentMirror(NoisySystem):
    pass


class CurrentMirror(base.QubitBaseClass, serializers.Serializable, NoisyCurrentMirror):
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
    number_degrees_freedom = descriptors.ReadOnlyProperty()
    ECB = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJlist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    nglist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ncut = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self,
                 N: int,
                 ECB: float,
                 ECJ: float,
                 ECg: float,
                 EJlist: ndarray,
                 nglist: ndarray,
                 flux: float,
                 ncut: int,
                 truncated_dim: int = None
                 ) -> None:
        self.N = N
        self._number_degrees_freedom = 2 * N - 1
        self.ECB = ECB
        self.ECJ = ECJ
        self.ECg = ECg
        self.EJlist = EJlist
        self.nglist = nglist
        self.flux = flux
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/currentmirror.png')

    @staticmethod
    def default_params() -> Dict[str, Any]:
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

    def supported_noise_channels(self) -> List[str]:
        """Return a list of supported noise channels"""
        return ['']

    def potential(self, phi_array: ndarray) -> ndarray:
        """Potential evaluated at the location specified by phi_array.

        Parameters
        ----------
        phi_array: ndarray
            float value of the phase variable `phi`

        Returns
        -------
        float
        """
        dim = self.number_degrees_freedom
        pot_sum = np.sum([- self.EJlist[j] * np.cos(phi_array[j]) for j in range(dim)])
        pot_sum += (-self.EJlist[-1] * np.cos(np.sum([phi_array[i] for i in range(dim)]) + 2*np.pi*self.flux))
        pot_sum += np.sum(self.EJlist)
        return pot_sum

    def capacitance_matrix(self) -> ndarray:
        """Returns the capacitance matrix, transforming to coordinates where the variable corresponding
        to the total charge can be eliminated

        Returns
        -------
            ndarray
        """
        N = self.N
        CB = 1. / (2.*self.ECB)
        CJ = 1. / (2.*self.ECJ)
        Cg = 1. / (2.*self.ECg)

        C_matrix = np.diagflat([Cg + 2*CJ + CB for _ in range(2*N)], 0)
        C_matrix += np.diagflat([-CJ for _ in range(2*N - 1)], +1)
        C_matrix += np.diagflat([-CJ for _ in range(2*N - 1)], -1)
        C_matrix += np.diagflat([-CB for _ in range(N)], +N)
        C_matrix += np.diagflat([-CB for _ in range(N)], -N)
        C_matrix[0, -1] = C_matrix[-1, 0] = - CJ

        V_m_inv = sp.linalg.inv(self._build_V_m())
        C_matrix = np.matmul(V_m_inv.T, np.matmul(C_matrix, V_m_inv))

        return C_matrix[0:-1, 0:-1]

    def EC_matrix(self) -> ndarray:
        """Returns the charging energy matrix

        Returns
        -------
            ndarray
        """
        return 0.5 * sp.linalg.inv(self.capacitance_matrix())

    def _build_V_m(self) -> ndarray:
        """Builds the matrix necessary for the coordinate transformation"""
        N = self.N
        V_m = np.diagflat([-1 for _ in range(2*N)], 0)
        V_m += np.diagflat([1 for _ in range(2*N - 1)], 1)
        V_m[-1] = np.array([1 for _ in range(2*N)])
        return V_m

    def harmonic_modes(self) -> ndarray:
        """Returns the harmonic modes associated with the linearized current mirror Hamiltonian.

        Returns
        -------
        ndarray
        """
        CB = 1. / (2. * self.ECB)
        CJ = 1. / (2. * self.ECJ)
        Cg = 1. / (2. * self.ECg)
        omega_list = np.zeros(self.number_degrees_freedom)
        for mu in range(1, self.number_degrees_freedom+1):
            potential_contribution = (4.0*self.EJlist[0]*np.sin(np.pi*mu/(2*self.N))**2)
            kinetic_contribution = Cg + 4*CJ*np.sin(np.pi*mu/(2*self.N))**2 + (1 - (-1)**mu)*CB
            omega_mu = 2*np.sqrt(potential_contribution/kinetic_contribution)
            omega_list[mu-1] = omega_mu
        return omega_list

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=False)
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, which='SA', return_eigenvectors=True)
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs
    
    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2*self.ncut+1)**self.number_degrees_freedom

    def hamiltonian(self) -> ndarray:
        """Returns the Hamiltonian employing the charge number basis for all :math:`2\cdot N - 1` d.o.f.

        Returns
        -------
            ndarray
        """
        dim = self.number_degrees_freedom
        EC_matrix = self.EC_matrix()
        H = 0.*self.identity_operator()
        for j, k in itertools.product(range(dim), range(dim)):
            H += 4*EC_matrix[j, k]*((self.n_operator(j) - self.nglist[j] * self.identity_operator())
                                    @ (self.n_operator(k) - self.nglist[k] * self.identity_operator()))
        for j in range(dim):
            H += (-self.EJlist[j]/2.)*(self.exp_i_phi_j_operator(j) + self.exp_i_phi_j_operator(j).conj().T)
            H += self.EJlist[j]*self.identity_operator()
        H += (-self.EJlist[-1] / 2.) * (np.exp(1j*2*np.pi*self.flux) * self.exp_i_phi_boundary_term().conj().T
                                        + np.exp(-1j*2*np.pi*self.flux) * self.exp_i_phi_boundary_term())
        H += self.EJlist[-1]*self.identity_operator()
        return H

    def _identity_operator(self) -> ndarray:
        return eye(2 * self.ncut + 1, k=0, format="csr", dtype=np.complex_)

    def _identity_operator_list(self) -> List[ndarray]:
        return [self._identity_operator() for _ in range(self.number_degrees_freedom)]

    def identity_operator(self) -> ndarray:
        """Returns the identity operator in the full Hilbert space

        Returns
        -------
            ndarray
        """
        return operator_in_full_Hilbert_space([], [], self._identity_operator_list(), sparse=True)

    def _n_operator(self) -> ndarray:
        return diags([i for i in range(-self.ncut, self.ncut + 1, 1)], offsets=0, format="csr", dtype=np.complex_)

    def n_operator(self, j: int = 0) -> ndarray:
        """Returns charge number operator :math:`n_{j}` in the full Hilbert space

        Parameters
        ----------
        j: int
            specifies the degree of freedom

        Returns
        -------
            ndarray
        """
        number_operator = self._n_operator()
        return operator_in_full_Hilbert_space([number_operator], [j], self._identity_operator_list(), sparse=True)

    def _exp_i_phi_j_operator(self) -> ndarray:
        return eye(2*self.ncut + 1, k=-1, format="csr", dtype=np.complex_)

    def exp_i_phi_j_operator(self, j: int = 0) -> ndarray:
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

    def exp_i_phi_boundary_term(self) -> ndarray:
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


class CurrentMirrorGlobal(HashingChargeBasis, CurrentMirror):
    def __init__(self,
                 N: int,
                 ECB: float,
                 ECJ: float,
                 ECg: float,
                 EJlist: ndarray,
                 nglist: ndarray,
                 flux: float,
                 num_exc: int,
                 truncated_dim: int = None
                 ) -> None:
        self.num_exc = num_exc
        HashingChargeBasis.__init__(self)
        CurrentMirror.__init__(self, N, ECB, ECJ, ECg, EJlist, nglist, flux, ncut=0, truncated_dim=truncated_dim)
