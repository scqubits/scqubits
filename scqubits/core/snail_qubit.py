from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from numpy import ndarray
from scipy import sparse

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers


class SnailQubit(base.QubitBaseClass, serializers.Serializable):
    def __init__(
        self,
        EJ1: float,
        EJ2: float,
        EJ3: float,
        EJ4: float,
        EC1: float,
        EC2: float,
        EC3: float,
        EC4: float,
        ECg1: float,
        ECg2: float,
        ECg3: float,
        ng1: float,
        ng2: float,
        ng3: float,
        flux: float,
        ncut: int,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        # Constant parameters for each Josephson Junction
        self.EJ1 = EJ1
        self.EJ2 = EJ2
        self.EJ3 = EJ3
        self.EJ4 = EJ4
        # Capacitance connected to each Josephson Junction
        self.EC1 = EC1
        self.EC2 = EC2
        self.EC3 = EC3
        self.EC4 = EC4

        # Capacitance connected to ground
        self.ECg1 = ECg1
        self.ECg2 = ECg2
        self.ECg3 = ECg3

        # offset charges associated with each Josephson Junction
        self.ng1 = ng1
        self.ng2 = ng2
        self.ng3 = ng3
        # flux
        self.flux = flux
        # Truncation dimension
        self.ncut = ncut
        self._image_filename = None

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ1": 887.3,
            "EJ2": 887.3,
            "EJ3": 887.3,
            "EJ4": 117.5,
            "EC1": 0.2873,
            "EC2": 0.2873,
            "EC3": 0.2873,
            "EC4": 1.437,
            "ECg1": 193.7,
            "ECg2": 193.7,
            "ECg3": 193.7,
            "ng1": 0.0,
            "ng2": 0.0,
            "ng3": 0.0,
            "flux": 0.0,
            "ncut": 30,
        }

    # Construct the Ec matrix, we need this to calculate the kinetic_energy matrix in
    # the Hamiltonian
    def get_Ec_matrix(self) -> ndarray:
        c1 = 1 / (2 * self.EC1)
        c2 = 1 / (2 * self.EC2)
        c3 = 1 / (2 * self.EC3)
        c4 = 1 / (2 * self.EC4)

        cg1 = 1 / (2 * self.ECg1)
        cg2 = 1 / (2 * self.ECg2)
        cg3 = 1 / (2 * self.ECg3)
        capacitance = np.array(
            [
                [c1 + c2 + cg1, -c2, 0],
                [-c2, c2 + c3 + cg2, -c3],
                [0, -c3, c3 + c4 + cg3],
            ]
        )
        E_c = 0.5 * np.linalg.inv(capacitance)
        return E_c

    def get_kinetic(self) -> ndarray:
        ec = self.get_Ec_matrix()
        identity = sparse.identity(2 * self.ncut + 1, format="csc")

        n_op = np.arange(-self.ncut, self.ncut + 1, 1)
        n_op1 = sparse.diags(n_op + self.ng1).tocsc()
        n_op2 = sparse.diags(n_op + self.ng2).tocsc()
        n_op3 = sparse.diags(n_op + self.ng3).tocsc()

        n1 = sparse.kron(
            sparse.kron(n_op1, identity, format="csc"), identity, format="csc"
        )

        n2 = sparse.kron(
            sparse.kron(identity, n_op2, format="csc"), identity, format="csc"
        )

        n3 = sparse.kron(
            sparse.kron(identity, identity, format="csc"), n_op3, format="csc"
        )

        nvec = np.array([n1 , n2 , n3 ]) # added offset gate charges

        return 4 * nvec.T @ ec @ nvec

    def potential(self, phi1, phi2, phi3) -> float:
        """Return value of the potential energy at phi1 and phi2 and phi3, disregarding
        constants."""
        return (
            -self.EJ1 * np.cos(phi1)
            - self.EJ2 * np.cos(phi2 - phi1)
            - self.EJ3 * np.cos(phi3 - phi2)
            - self.EJ4 * np.cos(2.0 * np.pi * self.flux - phi3)
        )

    def get_potential(self) -> ndarray:
        identity = sparse.identity(2 * self.ncut + 1, format="csc")

        ones_on_diagonal = np.ones((1, 2 * self.ncut))
        e_positive_phi = sparse.diags(ones_on_diagonal, [1]).tocsc()
        e_negative_phi = sparse.diags(ones_on_diagonal, [-1]).tocsc()

        component1 = (
            -0.5
            * self.EJ1
            * sparse.kron(
                sparse.kron(e_positive_phi + e_negative_phi, identity, format="csc"),
                identity,
                format="csc",
            )
        )
        component2 = (
            -0.5
            * self.EJ4
            * sparse.kron(
                sparse.kron(identity, identity, format="csc"),
                np.exp(-1j * 2 * np.pi * self.flux) * e_positive_phi
                + np.exp(+1j * 2 * np.pi * self.flux) * e_negative_phi,
                format="csc",
            )
        )
        component3 = (
            -0.5
            * self.EJ2
            * sparse.kron(
                sparse.kron(e_positive_phi, e_negative_phi, format="csc"),
                identity,
                format="csc",
            )
        )
        component4 = (
            -0.5
            * self.EJ2
            * sparse.kron(
                sparse.kron(e_negative_phi, e_positive_phi, format="csc"),
                identity,
                format="csc",
            )
        )
        component5 = (
            -0.5
            * self.EJ3
            * sparse.kron(
                sparse.kron(identity, e_positive_phi, format="csc"),
                e_negative_phi,
                format="csc",
            )
        )
        component6 = (
            -0.5
            * self.EJ3
            * sparse.kron(
                sparse.kron(identity, e_negative_phi, format="csc"),
                e_positive_phi,
                format="csc",
            )
        )

        return (
            component1 + component2 + component3 + component4 + component5 + component6
        )

    def hamiltonian(self) -> ndarray:
        """Return the Hamiltonian."""
        return self.get_kinetic() + self.get_potential()

    # def hermitian(self) -> float:
    #     h = self.hamiltonian()
    #     return np.max(np.abs(h - h.T.conjugate()))

    def hilbertdim(self) -> int:
        """Return Hilbert space dimension."""
        return (2 * self.ncut + 1) ** 3

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(
            hamiltonian_mat, which="SA", k=evals_count, return_eigenvectors=False
        )
        return np.sort(evals)

    def plot_potential(self, phi_grid=None, contour_vals=None, **kwargs):
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
        x_vals = 0.003 * np.arange(-800, 800)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5, 5)
        return plt.plot(x_vals, self.potential(x_vals, x_vals, x_vals) / self.EJ4)
