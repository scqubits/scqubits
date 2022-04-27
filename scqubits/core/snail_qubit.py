import numpy as np
import scipy as sp
from scipy import sparse

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from scqubits.core.qubit_base import QubitBaseClass
import matplotlib.pyplot as plt
from numpy import ndarray


class SnailQubit(QubitBaseClass):

    def __init__(self,
                 E_J1: float,
                 E_J2: float,
                 E_J3: float,
                 E_J4: float,
                 E_C1: float,
                 E_C2: float,
                 E_C3: float,
                 E_C4: float,
                 E_cg1: float,
                 E_cg2: float,
                 E_cg3: float,
                 ng1: float,
                 ng2: float,
                 ng3: float,
                 flux: float,
                 n_cut: int) -> None:

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
        self.cg1 = 1 / (2 * E_cg1)
        self.cg2 = 1 / (2 * E_cg2)
        self.cg3 = 1 / (2 * E_cg3)
        # offset charges associated with each Josephson Junction
        self.ng1 = ng1
        self.ng2 = ng2
        self.ng3 = ng3
        # flux
        self.flux = flux
        # Truncation dimension
        self.n_cut = n_cut
        self.dim = 2 * n_cut + 1

    def default_params(self) -> Dict[str, Any]:
        return {
            "EJ1": 1.0,
            "EJ2": 1.0,
            "EJ3": 1.0,
            "EJ4":0.7,
            "ECJ1": 1,
            "ECJ2": 1,
            "ECJ3": 1,
            "ECJ4": 1/0.7,
            "ECg1": 10,
            "ECg2": 10,
            "ECg3": 10,
            "ng1": 0.0,
            "ng2": 0.0,
            "ng3": 0.0,
            "flux": 0.41,
            "ncut": 10
        }

    # Construct the Ec matrix, we need this to calculate the kinetic_energy matrix in the Hamiltonian
    def get_Ec_matrix(self) -> ndarray:
        capacitance = np.array([[self.c1 + self.c2 + self.cg1, - self.c2, 0],
                        [- self.c2, self.c2 + self.c3 + self.cg2, - self.c3],
                        [0, - self.c3, self.c3 + self.c4 + self.cg3]])
        E_c = np.linalg.inv(capacitance) * (1 / 2)
        return E_c


    def get_kinetic(self) -> ndarray:
        ec = self.get_Ec_matrix()
        identity = sparse.identity(self.dim, format="csc")

        n1 = np.arange(-self.n_cut, self.n_cut + 1, 1)
        n1 = sparse.diags(n1).tocsc()
        n1 = sparse.kron(sparse.kron(n1, identity, format="csc"), identity, format="csc")

        n2 = np.arange(-self.n_cut, self.n_cut + 1, 1)
        n2 = sparse.diags(n2).tocsc()
        n2 = sparse.kron(sparse.kron(identity, n2, format="csc"), identity, format="csc")

        n3 = np.arange(-self.n_cut, self.n_cut + 1, 1)
        n3 = sparse.diags(n3).tocsc()
        n3 = sparse.kron(sparse.kron(identity, identity, format="csc"), n3, format="csc")

        nvec = np.array([n1, n2, n3])

        return 4 * nvec.T @ ec @ nvec


    def potential(self, phi1, phi2, phi3) -> float:
        """Return value of the potential energy at phi1 and phi2 and phi3, disregarding
        constants."""
        return (
            - self.E_J1 * np.cos(phi1)
            - self.E_J2 * np.cos(phi2-phi1)
            - self.E_J3 * np.cos(phi3-phi2)
            - self.E_J4 * np.cos(2.0 * np.pi * self.flux - phi3)
        )

    def get_potential(self) -> ndarray:
        identity = sparse.identity(self.dim, format="csc")

        ones_on_diagonal = np.ones((1, self.dim-1))  # might cause problem
        e_positive_phi = sparse.diags(ones_on_diagonal, [ 1]).tocsc()
        e_negative_phi = sparse.diags(ones_on_diagonal, [-1]).tocsc()

        component1 = -(0.5) * self.E_J1 * sparse.kron(sparse.kron(e_positive_phi + e_negative_phi, identity, format="csc"), identity, format="csc")
        component2 = -(0.5) * self.E_J4 * sparse.kron(sparse.kron(identity, identity, format="csc"),
                                                        np.exp(-1j * 2 * np.pi * self.flux)* e_positive_phi
                                                        + np.exp(+1j * 2 * np.pi * self.flux)* e_negative_phi,
                                                        format="csc")
        component3 = -(0.5) * self.E_J2 * sparse.kron(sparse.kron(e_positive_phi, e_negative_phi, format="csc"),
                                                        identity, format="csc")
        component4 = -(0.5) * self.E_J2 * sparse.kron(sparse.kron(e_negative_phi, e_positive_phi, format="csc"),
                                                        identity, format="csc")
        component5 = -(0.5) * self.E_J3 * sparse.kron(sparse.kron(identity, e_positive_phi, format="csc"),
                                                        e_negative_phi, format="csc")
        component6 = -(0.5) * self.E_J3 * sparse.kron(sparse.kron(identity, e_negative_phi, format="csc"),
                                                        e_positive_phi, format="csc")


        return  component1 +  component2 + component3 + component4 + component5 + component6

    def hamiltonian(self) -> ndarray:
        """Return the Hamiltonian."""
        return self.get_kinetic() + self.get_potential()

    def hermitian(self) -> float:
        h = self.hamiltonian()
        return np.max(np.abs(h - h.T.conjugate()))

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.n_cut + 1) ** 3

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(
                hamiltonian_mat, which='SM',k = evals_count, return_eigenvectors=False
            )
        return np.sort(evals)

    def plot_potential(
        self,
        phi_grid = None,
        contour_vals= None,
        **kwargs
    ) :
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        contour_vals:
            specific contours to draw
        **kwargs:
            plot options
        """
        x_vals = 0.003 * np.arange(-800,800)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5, 5)
        return plt.plot(
            x_vals, self.potential(x_vals,x_vals,x_vals)/self.E_J4)

test = SnailQubit(1, 1, 1, 0.7, 1, 1, 1, 1/0.7, 10, 10, 10, 0, 0, 0, 0.41, 10)
#print("\nmat_mul H:\n------------------------------------------------------\n")
#print(test.hamiltonian())
print(test._evals_calc(8))
#print(test.hilbertdim())
#plt.title("Nonlinear Asymmetric Potential")
#plt.xlabel("Flux reduced to 1 DoF")
#plt.ylabel("Potential Energy (U)")
#test.plot_potential()
#plt.show()
