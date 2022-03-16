# zeropi.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import os
import warnings

from typing import Any, Dict, Optional

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse.csc import csc_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem


# - ZeroPi noise class

class NoisyZeroPi(NoisySystem):
    pass


# -Symmetric 0-pi qubit, phi discretized, theta in charge basis-------------------------


class ZeroPi(base.QubitBaseClass2dExtPer, serializers.Serializable, NoisyZeroPi):
    r"""Zero-Pi Qubit

    | [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    | [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    | [3] Groszkowski et al., New J. Phys. 20, 043053 (2018). https://doi.org/10.1088/1367-2630/aab7cd

    Zero-Pi qubit without coupling to the `zeta` mode, i.e., no disorder in `EC` and
    `EL`, see Eq. (4) in Groszkowski et al., New J. Phys. 20, 043053 (2018),

    .. math::

        H &= -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
               +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta
               -2E_\text{J}\cos\theta\cos(\phi-\varphi_\text{ext}/2)+E_L\phi^2\\
          &\qquad +2E_\text{J} + E_J dE_J \sin\theta\sin(\phi-\phi_\text{ext}/2).

    Formulation of the Hamiltonian matrix proceeds by discretization of the `phi`
    variable, and using charge basis for the `theta` variable.

    Parameters
    ----------
    EJ:
        mean Josephson energy of the two junctions
    EL:
        inductive energy of the two (super-)inductors
    ECJ:
        charging energy associated with the two junctions
    EC:
        charging energy of the large shunting capacitances; set to `None` if `ECS` is
        provided instead
    dEJ:
        relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dCJ:
        relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    ng:
        offset charge associated with theta
    flux:
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid:
        specifies the range and spacing of the discretization lattice
    ncut:
        charge number cutoff for `n_theta`,  `n_theta = -ncut, ..., ncut`
    ECS:
        total charging energy including large shunting capacitances and junction
        capacitances; may be provided instead of EC
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
   """
    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dEJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dCJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        EL: float,
        ECJ: float,
        EC: Optional[float],
        ng: float,
        flux: float,
        grid: Grid1d,
        ncut: int,
        dEJ: float = 0.0,
        dCJ: float = 0.0,
        ECS: float = None,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)

        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ

        if EC is None and ECS is None:
            raise ValueError("Argument missing: must either provide EC or ECS")
        if EC and ECS:
            raise ValueError("Argument error: can only provide either EC or ECS")
        if EC:
            self.EC = EC
        elif ECS:
            self.EC = 1 / (1 / ECS - 1 / self.ECJ)
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.ng = ng
        self.flux = flux
        self.grid = grid
        self.ncut = ncut
        self.truncated_dim = truncated_dim

        # _default_grid is for *theta*, needed for plotting wavefunction
        self._default_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 200)

        self._init_params.remove(
            "ECS"
        )  # used in for file Serializable purposes; remove ECS as init parameter
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/zeropi.jpg"
        )
        dispatch.CENTRAL_DISPATCH.register("GRID_UPDATE", self)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 10.0,
            "EL": 0.04,
            "ECJ": 20.0,
            "EC": 0.04,
            "dEJ": 0.0,
            "dCJ": 0.0,
            "ng": 0.1,
            "flux": 0.23,
            "ncut": 30,
            "truncated_dim": 10,
        }

    def get_ECS(self) -> float:
        return 1 / (1 / self.EC + 1 / self.ECJ)

    def set_ECS(self, value) -> None:
        warnings.warn(
            "It is not possible to directly set ECS (except in initialization)."
            " Instead, set EC or ECJ, or use set_EC_via_ECS() to update EC indirectly.",
            Warning,
        )

    ECS = property(get_ECS, set_ECS)

    def set_EC_via_ECS(self, ECS: float) -> None:
        """Helper function to set `EC` by providing `ECS`, keeping `ECJ` constant."""
        self.EC = 1 / (1 / ECS - 1 / self.ECJ)

    def potential(self, phi: ndarray, theta: ndarray) -> ndarray:
        """
        Returns
        -------
            value of the potential energy evaluated at phi, theta
        """
        return (
            -2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)
            + self.EL * phi ** 2
            + 2.0 * self.EJ
            + self.EJ
            * self.dEJ
            * np.sin(theta)
            * np.sin(phi - 2.0 * np.pi * self.flux / 2.0)
        )

    def sparse_kinetic_mat(self) -> csc_matrix:
        """
        Kinetic energy portion of the Hamiltonian.

        Returns
        -------
            matrix representing the kinetic energy operator
        """
        pt_count = self.grid.pt_count
        dim_theta = 2 * self.ncut + 1
        identity_phi = sparse.identity(pt_count, format="csc")
        identity_theta = sparse.identity(dim_theta, format="csc")
        kinetic_matrix_phi = self.grid.second_derivative_matrix(
            prefactor=-2.0 * self.ECJ
        )
        diag_elements = (
            2.0
            * self.ECS
            * np.square(np.arange(-self.ncut + self.ng, self.ncut + 1 + self.ng))
        )
        kinetic_matrix_theta = sparse.dia_matrix(
            (diag_elements, [0]), shape=(dim_theta, dim_theta)
        ).tocsc()
        kinetic_matrix = sparse.kron(
            kinetic_matrix_phi, identity_theta, format="csc"
        ) + sparse.kron(identity_phi, kinetic_matrix_theta, format="csc")
        if self.dCJ != 0:
            kinetic_matrix -= (
                2.0
                * self.ECS
                * self.dCJ
                * self.i_d_dphi_operator()
                * self.n_theta_operator()
            )

        return kinetic_matrix

    def sparse_potential_mat(self) -> csc_matrix:
        """
        Potential energy portion of the Hamiltonian.

        Returns
        -------
            matrix representing the potential energy operator
        """
        pt_count = self.grid.pt_count
        grid_linspace = self.grid.make_linspace()
        dim_theta = 2 * self.ncut + 1

        phi_inductive_vals = self.EL * np.square(grid_linspace)
        phi_inductive_potential = sparse.dia_matrix(
            (phi_inductive_vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        phi_cos_vals = np.cos(grid_linspace - 2.0 * np.pi * self.flux / 2.0)
        phi_cos_potential = sparse.dia_matrix(
            (phi_cos_vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        phi_sin_vals = np.sin(grid_linspace - 2.0 * np.pi * self.flux / 2.0)
        phi_sin_potential = sparse.dia_matrix(
            (phi_sin_vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()

        theta_cos_potential = (
            -self.EJ
            * (
                sparse.dia_matrix(
                    ([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                )
                + sparse.dia_matrix(
                    ([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)
                )
            )
        ).tocsc()
        potential_mat = (
            sparse.kron(phi_cos_potential, theta_cos_potential, format="csc")
            + sparse.kron(phi_inductive_potential, self._identity_theta(), format="csc")
            + 2
            * self.EJ
            * sparse.kron(self._identity_phi(), self._identity_theta(), format="csc")
        )
        if self.dEJ != 0:
            potential_mat += (
                self.EJ
                * self.dEJ
                * sparse.kron(phi_sin_potential, self._identity_theta(), format="csc")
                * self.sin_theta_operator()
            )
        return potential_mat

    def hamiltonian(self) -> csc_matrix:
        """Calculates Hamiltonian in basis obtained by discretizing phi and employing
        charge basis for theta.

        Returns
        -------
            matrix representing the potential energy operator
        """
        return self.sparse_kinetic_mat() + self.sparse_potential_mat()

    def sparse_d_potential_d_flux_mat(self) -> csc_matrix:
        r"""Calculates a of the potential energy w.r.t flux, at the current value of
        flux, as stored in the object.

        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0.
        So if \frac{\partial U}{ \partial \Phi_{\rm ext}}, is needed, the expr returned
        by this function, needs to be multiplied by 1/\Phi_0.

        Returns
        -------
            matrix representing the derivative of the potential energy
        """
        op_1 = sparse.kron(
            self._sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
            self._cos_theta_operator(),
            format="csc",
        )
        op_2 = sparse.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
            self._sin_theta_operator(),
            format="csc",
        )
        return -2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2

    def d_hamiltonian_d_flux(self) -> csc_matrix:
        r"""Calculates a derivative of the Hamiltonian w.r.t flux, at the current value
        of flux, as stored in the object.

        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0.
        So if \frac{\partial H}{ \partial \Phi_{\rm ext}}, is needed, the expr returned
        by this function, needs to be multiplied by 1/\Phi_0.

        Returns
        -------
            matrix representing the derivative of the Hamiltonian
        """
        return self.sparse_d_potential_d_flux_mat()

    def sparse_d_potential_d_EJ_mat(self) -> csc_matrix:
        r"""Calculates a of the potential energy w.r.t EJ.

        Returns
        -------
            matrix representing the derivative of the potential energy
        """
        return -2.0 * sparse.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
            self._cos_theta_operator(),
            format="csc",
        )

    def d_hamiltonian_d_EJ(self) -> csc_matrix:
        r"""Calculates a derivative of the Hamiltonian w.r.t EJ.

        Returns
        -------
            matrix representing the derivative of the Hamiltonian
        """
        return self.sparse_d_potential_d_EJ_mat()

    def d_hamiltonian_d_ng(self) -> csc_matrix:
        r"""Calculates a derivative of the Hamiltonian w.r.t ng.
        as stored in the object.

        Returns
        -------
            matrix representing the derivative of the Hamiltonian
        """
        return -8 * self.EC * self.n_theta_operator()
