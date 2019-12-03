# zeropi_full.py


import numpy as np
from scipy import sparse

import sc_qubits.core.operators as op

from sc_qubits.core.qubit_base import QubitBaseClass
from sc_qubits.core.zeropi import ZeroPi
from sc_qubits.utils.spectrum_utils import order_eigensystem, get_matrixelement_table



class FullZeroPi(QubitBaseClass):
    """Full Zero-Pi Qubit, with all disorder types in circuit element parameters included. This couples
    the zeta degree of freedom, see Eqs. (4) or (5) in Groszkowski et al., New J. Phys. 20, 043053 (2018).

    Formulation of the Hamiltonian matrix proceeds in the product basis of the disordered (dEJ, dCJ)
    Zero-Pi qubit on one hand and the zeta LC oscillator on the other hand.

    Parameters
    ----------
    EJ: float
        mean Josephson energy of the two junctions
    EL: float
        inductive energy of the two (super-)inductors
    ECJ: float
        charging energy associated with the two junctions
    EC: float or None
        charging energy of the large shunting capacitances; set to `None` if `ECS` is provided instead
    dEJ: float
        relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dEL: float
        relative disorder in EL, i.e., (EL1-EL2)/EL(mean)
    dCJ: float
        relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    dC: float
        relative disorder in large capacitances, i.e., (C1-C2)/Cavg
    ng: float
        offset charge associated with theta
    zeropi_cutoff: int
        cutoff in the number of states of the disordered zero-pi qubit
    zeta_cutoff: int
        cutoff in the zeta oscillator basis (Fock state basis)
    flux: float
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid1d object
        specifies the range and spacing of the discretization lattice
    ncut: int
        charge number cutoff for `n_theta`,  `n_theta = -ncut, ..., ncut`
    ECS: float, optional
        total charging energy including large shunting capacitances and junction capacitances; may be provided instead
        of EC
    truncated_dim: int, optional
        desired dimension of the truncated quantum system
    """

    __initialized = False

    def __init__(self, EJ, EL, ECJ, EC, dEJ, dCJ, dC, dEL, flux, ng, zeropi_cutoff, zeta_cutoff, grid, ncut,
                 ECS=None, truncated_dim=None):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ

        if EC is None and ECS is None:
            raise ValueError("Argument missing: must either provide EC or ECS")
        elif EC and ECS:
            raise ValueError("Argument error: can only provide either EC or ECS")
        elif EC:
            self.EC = EC
        else:
            self.EC = 1 / (1 / ECS - 1 / self.ECJ)

        self.dEJ = dEJ
        self.dCJ = dCJ
        self.dC = dC
        self.dEL = dEL
        self.flux = flux
        self.ng = ng
        self.zeta_cutoff = zeta_cutoff
        self.zeropi_cutoff = zeropi_cutoff
        self.grid = grid
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._sys_type = 'full 0-Pi circuit (phi, theta, zeta) in 0pi - zeta product basis'

        self._zeropi = ZeroPi(
            EJ=self.EJ,
            EL=self.EL,
            ECJ=self.ECJ,
            EC=self.EC,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
            flux=self.flux,
            ng=self.ng,
            grid=self.grid,
            ncut=self.ncut,
            truncated_dim=self.zeropi_cutoff
        )
        self.__initialized = True

    def ECS(self):
        """ """
        return 1 / (1 / self.EC + 1 / self.ECJ)

    def set_EC_via_ECS(self, ECS):
        """Helper function to set `EC` by providing `ECS`, keeping `ECJ` constant."""
        self.EC = 1 / (1 / ECS - 1 / self.ECJ)
        return None

    def omega_zeta(self):
        """ """
        return (8.0 * self.EL * self.EC) ** 0.5

    def hamiltonian(self, return_parts=False):
        """"""
        zeropi_dim = self.zeropi_cutoff
        zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)
        zeropi_diag_hamiltonian = sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.float_)
        zeropi_diag_hamiltonian.setdiag(zeropi_evals)

        zeta_dim = self.zeta_cutoff
        prefactor = self.omega_zeta()
        zeta_diag_hamiltonian = op.number_sparse(zeta_dim, prefactor)

        hamiltonian_mat = sparse.kron(zeropi_diag_hamiltonian, sparse.identity(zeta_dim, format='dia', dtype=np.float_))
        hamiltonian_mat += sparse.kron(sparse.identity(zeropi_dim, format='dia', dtype=np.float_),
                                       zeta_diag_hamiltonian)

        gmat = self.g_coupling_matrix(zeropi_evecs)
        zeropi_coupling = sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.float_)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                zeropi_coupling += gmat[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)
        hamiltonian_mat += sparse.kron(zeropi_coupling,
                                       op.annihilation_sparse(zeta_dim) + op.creation_sparse(zeta_dim))

        if return_parts:
            return [hamiltonian_mat.tocsc(), zeropi_evals, zeropi_evecs, gmat]
        else:
            return hamiltonian_mat.tocsc()

    def _zeropi_operator_in_product_basis(self, zeropi_operator, zeropi_evecs=None):
        """Helper method that converts a zeropi operator into one in the product basis'

        TODO: Could update d_hamiltonian_EJ(),  d_hamiltonian_ng(),  d_hamiltonian_flux() to use this.
        """
        zeropi_dim = self.zeropi_cutoff
        zeta_dim = self.zeta_cutoff

        if zeropi_evecs is None:
            zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        op_eigen_basis = sparse.dia_matrix((zeropi_dim, zeropi_dim),
                                           dtype=np.complex_)  # is this guaranteed to be zero?

        op_zeropi = get_matrixelement_table(zeropi_operator, zeropi_evecs, real_valued=False)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                op_eigen_basis += op_zeropi[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)

        return sparse.kron(op_eigen_basis, sparse.identity(zeta_dim, format='csc', dtype=np.complex_), format='csc')

    def i_d_dphi_operator(self, zeropi_evecs=None):
        """"""
        return self._zeropi_operator_in_product_basis(self._zeropi.i_d_dphi_operator(), zeropi_evecs=zeropi_evecs)

    def n_theta_operator(self, zeropi_evecs=None):
        """"""
        return self._zeropi_operator_in_product_basis(self._zeropi.n_theta_operator(), zeropi_evecs=zeropi_evecs)

    def phi_operator(self, zeropi_evecs=None):
        """"""
        return self._zeropi_operator_in_product_basis(self._zeropi.phi_operator(), zeropi_evecs=zeropi_evecs)

    def hilbertdim(self):
        """ """
        return self.zeropi_cutoff * self.zeta_cutoff

    def _evals_calc(self, evals_count, hamiltonian_mat=None):
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    def _esys_calc(self, evals_count, hamiltonian_mat=None):
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def g_phi_coupling_matrix(self, zeropi_states):
        """Returns a matrix of coupling strengths g^\\phi_{ll'} [cmp. Dempster et al., Eq. (18)], using the states
        from the list `zeropi_states`. Most commonly, `zeropi_states` will contain eigenvectors of the
        `DisorderedZeroPi` type.
        """
        # prefactor = self.EL * self.dEL * (8.0 * self.EC / self.EL)**0.25
        prefactor = self.EL * (self.dEL / 2.0) * (8.0 * self.EC / self.EL) ** 0.25
        return prefactor * get_matrixelement_table(self._zeropi.phi_operator(), zeropi_states)

    def g_theta_coupling_matrix(self, zeropi_states):
        """Returns a matrix of coupling strengths i*g^\\theta_{ll'} [cmp. Dempster et al., Eq. (17)], using the states
        from the list 'zeropi_states'.
        """
        prefactor = 1j * self.ECS() * (self.dC / 2.0) * (32.0 * self.EL / self.EC) ** 0.25
        return prefactor * get_matrixelement_table(self._zeropi.n_theta_operator(), zeropi_states)

    def g_coupling_matrix(self, zeropi_states=None, evals_count=None):
        """Returns a matrix of coupling strengths g_{ll'} [cmp. Dempster et al., text above Eq. (17)], using the states
        from 'zeropi_states'. If `zeropi_states==None`, then a set of `self.zeropi` eigenstates is calculated. Only in
        that case is `evals_count` used for the eigenstate number (and hence the coupling matrix size).
        """
        if evals_count is None:
            evals_count = self._zeropi.truncated_dim
        if zeropi_states is None:
            _, zeropi_states = self._zeropi.eigensys(evals_count=evals_count)
        return self.g_phi_coupling_matrix(zeropi_states) + self.g_theta_coupling_matrix(zeropi_states)

