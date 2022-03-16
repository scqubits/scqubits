import numpy as np
import scipy as sp


class snailQubit(base.QubitBaseClass):

    def __init__(self, E_J1, E_J2, E_J3, E_J4, E_C1, E_C2, E_C3, E_C4, cg1, cg2, cg3, ng1, ng2, ng3, flux, n_cut):
        # Constant parameters for each Josephson Junction
        self.E_J1 = E_J1
        self.E_J2 = E_J2
        self.E_J3 = E_J3
        self.E_J4 = E_J4
        # Capacitance connected to each Josephson Junction
        self.c1 = 1 / (2 * E_C1)
        self.c2 = 1 / (2 * E_C2)
        self.c3 = 1 / (2 * E_C3)
        self.c4 = 1 / (2 * E_C4)
        # Capacitance connected to ground
        self.cg1 = 1 / (2 * cg1)
        self.cg2 = 1 / (2 * cg2)
        self.cg3 = 1 / (2 * cg3)
        # offset charges associated with each Josephson Junction
        self.ng1 = ng1
        self.ng2 = ng2
        self.ng3 = ng3
        # flux
        self.flux = flux
        # Truncation dimension
        self.n_cut = n_cut
        self.dim = 2 * n_cut + 1

    # Construct the Ec matrix, we need this to calculate the kinetic_energy matrix in the Hamiltonian
    def get_Ec_matrix(self):
        E_c = np.array([[self.c1 + self.c2 + self.cg1, - self.c2, 0],
                        [- self.c2, self.c2 + self.c3 + self.cg2, - self.c3],
                        [0, - self.c3, self.c3 + self.c4 + self.cg3]])
        E_c = np.linalg.inv(E_c) * (1 / 2)
        return E_c

    def get_kinetic(self):
        ec = self.get_Ec_matrix()
        identity = sparse.identity(self.dim, format="csc")

        n1 = np.arange(-self.n_cut, self.n_cut + 1) - self.ng1
        n1 = sparse.dia_matrix((n1, [0]), shape=(self.dim, self.dim)).tocsc()
        n1 = sparse.kron(sparse.kron(n1, identity, format="csc"), identity, format="csc")

        n2 = np.arange(-self.n_cut, self.n_cut + 1) - self.ng2
        n2 = sparse.dia_matrix((n2, [0]), shape=(self.dim, self.dim)).tocsc()
        n2 = sparse.kron(sparse.kron(identity, n2, format="csc"), identity, format="csc")

        n3 = np.arange(-self.n_cut, self.n_cut + 1) - self.ng3
        n3 = sparse.dia_matrix((n3, [0]), shape=(self.dim, self.dim)).tocsc()
        n3 = sparse.kron(sparse.kron(identity, identity, format="csc"), n3, format="csc")

        nvec = np.array([n1, n2, n3])

        return 4 * nvec.T @ ec @ nvec

    def get_potential(self):
        identity = sparse.identity(self.dim, format="csc")

        ones_on_diagonal = np.ones((1, self.dim))  # might cause problem
        e_positive_phi = sparse.dia_matrix((ones_on_diagonal, [1]), shape=(self.dim, self.dim)).tocsc()
        e_negative_phi = sparse.dia_matrix((ones_on_diagonal, [-1]), shape=(self.dim, self.dim)).tocsc()

        component1 = -(1 / 2) * self.E_J1 * sparse.kron(
            sparse.kron(e_positive_phi + e_negative_phi, identity, format="csc"), identity, format="csc")
        component2 = -(1 / 2) * self.E_J4 * sparse.kron(sparse.kron(identity, identity, format="csc"),
                                                        e_positive_phi + e_negative_phi, format="csc")
        component3 = -(1 / 2) * self.E_J2 * sparse.kron(sparse.kron(e_positive_phi, e_negative_phi, format="csc"),
                                                        identity, format="csc")
        component4 = -(1 / 2) * self.E_J2 * sparse.kron(sparse.kron(e_negative_phi, e_positive_phi, format="csc"),
                                                        identity, format="csc")
        component5 = -(1 / 2) * self.E_J3 * sparse.kron(sparse.kron(identity, e_positive_phi, format="csc"),
                                                        e_negative_phi, format="csc")
        component6 = -(1 / 2) * self.E_J3 * sparse.kron(sparse.kron(identity, e_negative_phi, format="csc"),
                                                        e_positive_phi, format="csc")

        return np.exp(-1j * 2 * np.pi * self.flux) * component1 + np.exp(
            -1j * 2 * np.pi * self.flux) * component2 + component3 + component4 + component5 + component6

    def hamiltonian(self):
        """Return the Hamiltonian."""
        return self.get_kinetic() + self.get_potential()

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.n_cut + 1) ** 2

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(
            hamiltonian_mat, eigvals_only=True, eigvals=(0, evals_count - 1)
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(
            hamiltonian_mat, eigvals_only=False, eigvals=(0, evals_count - 1)
        )
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs