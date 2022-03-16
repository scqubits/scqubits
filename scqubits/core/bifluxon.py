# bifluxon.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import os

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse.csc import csc_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.ui.qubit_widget as ui
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem
from scqubits.core.storage import WaveFunctionOnGrid


class NoisyBifluxon(NoisySystem):
    pass


# Bifluxon qubit, fluxonium-like phi mode discretized, small island connected with two
# junctions solved in the charge basis--------------------------------------------------


class Bifluxon(base.QubitBaseClass2dExtPer, serializers.Serializable, NoisyBifluxon):
    r"""Bifluxon Qubit

    | [1] Kalashnikov et al., PRX Quantum 1, 010307 (2020). https://doi.org/10.1103/PRXQuantum.1.010307


    Bifluxon qubit without considering disorder in the small Josephson junctions,
    based on Eq. (1) in [1],

    .. math::

        H &= 4E_{\text{C}}(-i\partial_\theta-n_g)^2-2E_\text{J}\cos\theta\cos(\phi/2) \\
        -4E_\text{CL}\partial_\phi^2+E_L(\phi -\varphi_\text{ext})^2.

    Formulation of the Hamiltonian matrix proceeds by discretization of the `phi`
    variable, and uses charge basis for the `theta` variable.

    Parameters
    ----------
    EJ:
        mean Josephson energy of the two junctions
    EL:
        inductive energy of the inductors
    EC:
        charging energy of the superconducting island connected to the two junctions
    ECL:
        charging energy of the large superinductor
    dEJ:
        relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    ng:
        offset charge at the small superconducting island
    flux:
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid:
        specifies the range and spacing of the discretization lattice
    ncut:
        charge number cutoff for the superconducting island, `n_theta = -ncut,...,ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
   """
    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dEJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        EL: float,
        EC: float,
        ECL: float,
        ng: float,
        flux: float,
        grid: Grid1d,
        ncut: int,
        dEJ: float = 0.0,
        truncated_dim: int = 6,
    ) -> None:
        self.EJ = EJ
        self.EL = EL
        self.EC = EC
        self.ECL = ECL
        self.dEJ = dEJ
        self.ng = ng
        self.flux = flux
        self.grid = grid
        self.ncut = ncut
        self.truncated_dim = truncated_dim

        # _default_grid is for *theta*, needed for plotting wavefunction
        self._default_grid = discretization.Grid1d(-np.pi / 2, 3 * np.pi / 2, 200)

        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/bifluxon.jpg"
        )
        dispatch.CENTRAL_DISPATCH.register("GRID_UPDATE", self)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 27.2,
            "EL": 0.94,
            "EC": 7.7,
            "ECL": 10.0,
            "dEJ": 0.0,
            "ng": 0.5,
            "flux": 0.23,
            "ncut": 30,
            "truncated_dim": 10,
        }

    def potential(self, phi: ndarray, theta: ndarray) -> ndarray:
        """
        Returns
        -------
            value of the potential energy evaluated at phi, theta for Bifluxon
        """
        return (
            -2.0
            * self.EJ
            * np.cos(theta)
            * np.cos(phi / 2.0 + 2.0 * np.pi * self.flux / 2.0)
            + (1 / 2.0) * self.EL * phi ** 2
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
            prefactor=-4.0 * self.ECL
        )
        diag_elements = (
            4.0
            * self.EC
            * np.square(np.arange(-self.ncut + self.ng, self.ncut + 1 + self.ng))
        )
        kinetic_matrix_theta = sparse.dia_matrix(
            (diag_elements, [0]), shape=(dim_theta, dim_theta)
        ).tocsc()
        kinetic_matrix = sparse.kron(
            kinetic_matrix_phi, identity_theta, format="csc"
        ) + sparse.kron(identity_phi, kinetic_matrix_theta, format="csc")

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

        phi_inductive_vals = (1 / 2.0) * self.EL * np.square(grid_linspace)
        phi_inductive_potential = sparse.dia_matrix(
            (phi_inductive_vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        phi_cosby2_vals = np.cos(grid_linspace / 2.0 + 2.0 * np.pi * self.flux / 2.0)
        phi_cosby2_potential = sparse.dia_matrix(
            (phi_cosby2_vals, [0]), shape=(pt_count, pt_count)
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
            sparse.kron(phi_cosby2_potential, theta_cos_potential, format="csc")
            + sparse.kron(phi_inductive_potential, self._identity_theta(), format="csc")
            + 2
            * self.EJ
            * sparse.kron(self._identity_phi(), self._identity_theta(), format="csc")
        )

        # TODO - remove or enable
        # if self.dEJ != 0:
        #     potential_mat += (
        #         self.EJ
        #         * self.dEJ
        #         * sparse.kron(phi_sin_potential, self._identity_theta(), format="csc")
        #         * self.sin_theta_operator()
        #     )
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
        r"""Calculates derivative of the potential energy w.r.t flux, at the current
        value of flux, as stored in the object.

        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0.
        So if \frac{\partial U}{ \partial \Phi_{\rm ext}}, is needed, the expression
        returned by this function, needs to be multiplied by 1/\Phi_0.

        Returns
        -------
            matrix representing the derivative of the potential energy.
            TODO
            NEED TO UPDATE FOR BIFLUXON with phi/2 operators
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
        So if \frac{\partial H}{ \partial \Phi_{\rm ext}}, is needed, the expression
        returned by this function, needs to be multiplied by 1/\Phi_0.

        Returns
        -------
            matrix representing the derivative of the Hamiltonian
        """
        return self.sparse_d_potential_d_flux_mat()

    def sparse_d_potential_d_EJ_mat(self) -> csc_matrix:
        r"""Calculates derivative of the potential energy w.r.t EJ.

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
        r"""Calculates the derivative of the Hamiltonian w.r.t EJ.

        Returns
        -------
            matrix representing the derivative of the Hamiltonian
        """
        return self.sparse_d_potential_d_EJ_mat()

    def d_hamiltonian_d_ng(self) -> csc_matrix:
        r"""Calculates the derivative of the Hamiltonian w.r.t ng.

        Returns
        -------
            matrix representing the derivative of the Hamiltonian
        """
        return -8 * self.EC * self.n_theta_operator()

    # TODO: should be able to lift the following two into QubitBaseClass2dExtPer?
    def _sin_phiby2_operator(self, x: float = 0) -> csc_matrix:
        r"""
        Operator :math:`\sin(\phi/2 + x)`, acting only on the `\phi` Hilbert subspace.x
        """
        pt_count = self.grid.pt_count

        vals = np.sin(self.grid.make_linspace() / 2.0 + x)
        sin_phiby2_matrix = sparse.dia_matrix(
            (vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        return sin_phiby2_matrix

    def _cos_phiby2_operator(self, x: float = 0) -> csc_matrix:
        r"""
        Operator :math:`\cos(\phi/2.0 + x)`, acting only on the `\phi` Hilbert subspace.
        """
        pt_count = self.grid.pt_count

        vals = np.cos(self.grid.make_linspace() / 2.0 + x)
        cos_phiby2_matrix = sparse.dia_matrix(
            (vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        return cos_phiby2_matrix
