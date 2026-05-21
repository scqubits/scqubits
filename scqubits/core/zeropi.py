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

from __future__ import annotations

import warnings

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix, dia_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.ui.qubit_widget as ui
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as utils

from scqubits.core.convergence import ConvergenceCheckable
from scqubits.core.convergence_report import TruncationChannel
from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem
from scqubits.core.storage import WaveFunctionOnGrid

# - ZeroPi noise class


class NoisyZeroPi(NoisySystem):
    """Noise mixin for the :class:`ZeroPi` qubit."""

    pass


# -Symmetric 0-pi qubit, phi discretized, theta in charge basis


class ZeroPi(
    base.QubitBaseClass, serializers.Serializable, NoisyZeroPi, ConvergenceCheckable
):
    r"""Zero-Pi Qubit.

    | [1] Brooks et al., Physical Review A, 87(5), 052306 (2013).
    |     http://doi.org/10.1103/PhysRevA.87.052306
    | [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    |     http://doi.org/10.1103/PhysRevB.90.094518
    | [3] Groszkowski et al., New J. Phys. 20, 043053 (2018).
    |     https://doi.org/10.1088/1367-2630/aab7cd

    Zero-Pi qubit without coupling to the :math:`\zeta` mode, i.e., no disorder in `EC` and
    `EL`, see Eq. (4) in Groszkowski et al., New J. Phys. 20, 043053 (2018),

    .. math::

        H &= -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
               +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta
               -2E_\text{J}\cos\theta\cos(\phi-\varphi_\text{ext}/2)+E_L\phi^2\\
          &\qquad +2E_\text{J} + E_J dE_J \sin\theta\sin(\phi-\phi_\text{ext}/2).

    Formulation of the Hamiltonian matrix proceeds by discretization of the :math:`\phi`
    variable, and using charge basis for the :math:`\theta` variable.

    Parameters
    ----------
    EJ:
        mean Josephson energy of the two junctions
    EL:
        inductive energy of the two (super-)inductors
    ECJ:
        charging energy associated with the two junctions
    EC:
        charging energy of the large shunting capacitances; set to ``None`` if `ECS` is
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
        charge number cutoff for :math:`n_\theta`,
        :math:`n_\theta = -{\rm ncut}, \dots, {\rm ncut}`
    ECS:
        total charging energy including large shunting capacitances and junction
        capacitances; may be provided instead of EC
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in :class:`HilbertSpace`
        and :class:`ParameterSweep`. If not provided, an id is auto-generated.
    esys_method:
        method for esys diagonalization, callable or string representation
    esys_method_options:
        dictionary with esys diagonalization options
    evals_method:
        method for evals diagonalization, callable or string representation
    evals_method_options:
        dictionary with evals diagonalization options
    """

    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ECJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dEJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dCJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    _convergence_axes: tuple[str, ...] = ("grid", "ncut")
    _convergence_basis: str = "discretized_phi+charge"

    def __init__(
        self,
        EJ: float,
        EL: float,
        ECJ: float,
        EC: float | None,
        ng: float,
        flux: float,
        grid: Grid1d,
        ncut: int,
        dEJ: float = 0.0,
        dCJ: float = 0.0,
        ECS: float | None = None,
        truncated_dim: int = 6,
        id_str: str | None = None,
        evals_method: Callable | str | None = None,
        evals_method_options: dict | None = None,
        esys_method: Callable | str | None = None,
        esys_method_options: dict | None = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )

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
        dispatch.CENTRAL_DISPATCH.register("GRID_UPDATE", self)

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return a default-parameter dict suitable for instantiating the class."""
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

    @classmethod
    def create(cls) -> "ZeroPi":
        """Use ipywidgets to create a new class instance."""
        phi_grid = discretization.Grid1d(-19.0, 19.0, 200)
        init_params = cls.default_params()
        init_params["grid"] = phi_grid
        zeropi = cls(**init_params)
        zeropi.widget()
        return zeropi

    @classmethod
    def supported_noise_channels(cls) -> list[str]:
        """Return a list of supported noise channels."""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            "t1_flux_bias_line",
            # 't1_capacitive',
            "t1_inductive",
        ]

    def widget(self, params: dict[str, Any] | None = None) -> None:
        """Use ipywidgets to modify parameters of class instance.

        Parameters
        ----------
        params:
            optional dict overriding the parameters used to populate the widget;
            if ``None``, current instance parameters are used
        """
        init_params = params or self.get_initdata()
        init_params.pop("id_str", None)
        del init_params["grid"]
        init_params["grid_max_val"] = self.grid.max_val
        init_params["grid_min_val"] = self.grid.min_val
        init_params["grid_pt_count"] = self.grid.pt_count
        ui.create_widget(
            self.set_params_from_gui, init_params, image_filename=self._image_filename
        )

    def set_params_from_gui(self, **kwargs) -> None:  # type: ignore[override]
        """Set instance parameters and grid from GUI-provided keyword arguments.

        Mutates instance state: rebuilds :attr:`grid` from ``grid_min_val``,
        ``grid_max_val``, ``grid_pt_count`` and assigns remaining ``kwargs``
        as attributes on the instance.
        """
        phi_grid = discretization.Grid1d(
            kwargs.pop("grid_min_val"),
            kwargs.pop("grid_max_val"),
            kwargs.pop("grid_pt_count"),
        )
        self.grid = phi_grid
        for param_name, param_val in kwargs.items():
            setattr(self, param_name, param_val)

    def receive(self, event: str, sender: object, **kwargs: Any) -> None:
        """Handle dispatched events; rebroadcast a grid change as a system update.

        Parameters
        ----------
        event:
            name of the dispatched event
        sender:
            object that originated the event
        """
        if sender is self.grid:
            self.broadcast("QUANTUMSYSTEM_UPDATE")

    def _evals_calc(self, evals_count: int) -> ndarray:
        """Return the lowest ``evals_count`` eigenvalues.

        Eigenvalues are obtained from a sparse shift-invert ``eigsh_safe`` call
        and post-sorted via :func:`numpy.sort`, since Lanczos solvers do not
        guarantee ordered output.

        Parameters
        ----------
        evals_count:
            number of eigenvalues to return
        """
        hamiltonian_mat = self.hamiltonian()
        evals = utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            sigma=0.0,
            which="LM",
            return_eigenvectors=False,
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> tuple[ndarray, ndarray]:
        """Return the lowest ``evals_count`` eigenvalues and corresponding eigenvectors.

        Uses sparse shift-invert ``eigsh_safe``; the eigensystem is reordered via
        :func:`~scqubits.utils.spectrum_utils.order_eigensystem` since Lanczos
        solvers do not guarantee ordered output.

        Parameters
        ----------
        evals_count:
            number of eigenvalues and eigenvectors to return
        """
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            sigma=0.0,
            which="LM",
            return_eigenvectors=True,
        )
        evals, evecs = utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def get_ECS(self) -> float:
        """Return the total effective charging energy ``ECS = 1 / (1/EC + 1/ECJ)``."""
        return 1 / (1 / self.EC + 1 / self.ECJ)

    def set_ECS(self, value: float) -> None:
        """Reject direct assignment to :attr:`ECS` and emit a warning.

        Parameters
        ----------
        value:
            ignored; included only to satisfy the property setter signature
        """
        warnings.warn(
            "It is not possible to directly set ECS (except in initialization)."
            " Instead, set EC or ECJ, or use set_EC_via_ECS() to update EC indirectly.",
            Warning,
        )

    ECS = property(get_ECS, set_ECS)

    def set_EC_via_ECS(self, ECS: float) -> None:
        """Set :attr:`EC` by providing `ECS`, keeping `ECJ` constant.

        Mutates instance state: assigns to :attr:`EC`.

        Parameters
        ----------
        ECS:
            target value of the total effective charging energy
        """
        self.EC = 1 / (1 / ECS - 1 / self.ECJ)

    def hilbertdim(self) -> int:
        """Return the Hilbert space dimension."""
        return self.grid.pt_count * (2 * self.ncut + 1)

    # ----- Convergence-diagnostics hooks ----------------------------------------------

    def _convergence_axis_value(self, axis: str) -> int:
        """Return the integer size of a truncation axis.

        ``"grid"`` reports the phi-grid point count; ``"ncut"`` the theta charge
        cutoff.
        """
        if axis == "grid":
            return self.grid.pt_count
        return int(getattr(self, axis))

    def _convergence_set_axis(
        self, clone: "ConvergenceCheckable", axis: str, value: int
    ) -> None:
        """Set a truncation axis on ``clone``.

        For ``"grid"`` a new :class:`Grid1d` with ``value`` points is built over
        the same phi window; ``"ncut"`` is assigned directly.
        """
        if axis == "grid":
            clone.grid = Grid1d(  # type: ignore[attr-defined]
                self.grid.min_val, self.grid.max_val, value
            )
        else:
            setattr(clone, axis, value)

    def _convergence_truncation_channel(self, axis: str) -> TruncationChannel:
        """Report the FD phi-grid channel for ``"grid"`` and charge for ``"ncut"``."""
        return "FD_grid" if axis == "grid" else "charge"

    def _convergence_boundary_diagnostic(
        self,
        esys: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        axis: str,
    ) -> npt.NDArray[np.float64] | None:
        """Per-level boundary-amplitude diagnostic for the requested axis.

        Eigenvectors flatten as ``(grid.pt_count, 2*ncut+1) = [phi, theta]``. For
        ``"grid"`` the squared amplitude on the phi-window edges (summed over
        theta) is returned; for ``"ncut"`` the amplitude on the charge edges
        (summed over phi). A large value flags appreciable support at the basis
        boundary. Returns ``None`` for an unrecognized axis.
        """
        if axis not in ("grid", "ncut"):
            return None
        _, evecs = esys
        dim_phi = self.grid.pt_count
        dim_theta = 2 * self.ncut + 1
        n_cols = evecs.shape[1]
        boundary = np.empty(n_cols, dtype=np.float64)
        for k in range(n_cols):
            amp_sq = np.abs(evecs[:, k].reshape(dim_phi, dim_theta)) ** 2
            if axis == "grid":
                boundary[k] = float(amp_sq[0, :].sum() + amp_sq[-1, :].sum())
            else:  # ncut
                boundary[k] = float(amp_sq[:, 0].sum() + amp_sq[:, -1].sum())
        return boundary

    def potential(self, phi: ndarray, theta: ndarray) -> ndarray:
        r"""Return the zero-pi potential energy evaluated at :math:`\phi,\theta`.

        Parameters
        ----------
        phi:
            value(s) of the :math:`\phi` variable
        theta:
            value(s) of the :math:`\theta` variable
        """
        return (
            -2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)
            + self.EL * phi**2
            + 2.0 * self.EJ
            + self.EJ
            * self.dEJ
            * np.sin(theta)
            * np.sin(phi - 2.0 * np.pi * self.flux / 2.0)
        )

    def sparse_kinetic_mat(self) -> csc_matrix:
        """Return the kinetic-energy part of the Hamiltonian.

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
        """Return the potential-energy part of the Hamiltonian.

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

    def hamiltonian(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the Hamiltonian in the native or eigenenergy basis.

        The native basis discretizes :math:`\phi` and uses the charge basis for
        :math:`\theta`.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the Hamiltonian in the basis obtained
            by discretizing :math:`\phi` and employing the charge basis for
            :math:`\theta`. If ``True``, the energy eigenspectrum is computed and
            the Hamiltonian is returned in the energy eigenbasis. If
            ``energy_esys = esys``, where ``esys`` is a tuple ``(evals, evecs)``
            of eigenvalues and eigenvectors, the Hamiltonian is expressed in that
            eigenbasis without recomputing the spectrum.

        Returns
        -------
        Hamiltonian in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        hamiltonian_mat = self.sparse_kinetic_mat() + self.sparse_potential_mat()
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )

    def sparse_d_potential_d_flux_mat(self) -> csc_matrix:
        r"""Return the derivative of the potential energy w.r.t. flux at the stored flux value.

        The flux is assumed to be given in the units of the ratio
        :math:`\Phi_{\rm ext}/\Phi_0`. So if
        :math:`\frac{\partial U}{\partial \Phi_{\rm ext}}` is needed, the
        expression returned by this function needs to be multiplied by
        :math:`1/\Phi_0`.

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

    def d_hamiltonian_d_flux(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the derivative of the Hamiltonian w.r.t. flux at the stored flux value.

        The flux is assumed to be given in the units of the ratio
        :math:`\Phi_{ext}/\Phi_0`.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        native = self.sparse_d_potential_d_flux_mat()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def sparse_d_potential_d_EJ_mat(self) -> csc_matrix:
        r"""Return the derivative of the potential energy w.r.t. EJ.

        Returns
        -------
        matrix representing the derivative of the potential energy
        """
        return -2.0 * sparse.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
            self._cos_theta_operator(),
            format="csc",
        )

    def d_hamiltonian_d_EJ(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the derivative of the Hamiltonian w.r.t. EJ.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        native = self.sparse_d_potential_d_EJ_mat()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_ng(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the derivative of the Hamiltonian w.r.t. ng at the stored ng value.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        native = -8 * self.EC * self.n_theta_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _identity_phi(self) -> csc_matrix:
        r"""Return the identity operator on the :math:`\phi` subspace."""
        pt_count = self.grid.pt_count
        return sparse.identity(pt_count, format="csc")  # type: ignore[return-value]

    def _identity_theta(self) -> csc_matrix:
        r"""Return the identity operator on the :math:`\theta` subspace."""
        dim_theta = 2 * self.ncut + 1
        return sparse.identity(dim_theta, format="csc")  # type: ignore[return-value]

    def i_d_dphi_operator(self) -> csc_matrix:
        r"""Return the operator :math:`i\,d/d\phi`."""
        return sparse.kron(
            self.grid.first_derivative_matrix(prefactor=1j),
            self._identity_theta(),
            format="csc",
        )

    def _phi_operator(self) -> dia_matrix:
        r"""Return the :math:`\phi` operator on the :math:`\phi` Hilbert subspace."""
        pt_count = self.grid.pt_count

        phi_matrix = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = self.grid.make_linspace()
        phi_matrix.setdiag(diag_elements)
        return phi_matrix

    def phi_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`\phi` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        native = sparse.kron(self._phi_operator(), self._identity_theta(), format="csc")
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def n_theta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`n_\theta` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        dim_theta = 2 * self.ncut + 1
        diag_elements = np.arange(-self.ncut, self.ncut + 1)
        n_theta_matrix = sparse.dia_matrix(
            (diag_elements, [0]), shape=(dim_theta, dim_theta)
        ).tocsc()  # type: ignore[type-var,misc]
        native = sparse.kron(self._identity_phi(), n_theta_matrix, format="csc")
        return self.process_op(native_op=native, energy_esys=energy_esys)  # type: ignore[arg-type]

    def _sin_phi_operator(self, x: float = 0) -> csc_matrix:
        r"""Operator :math:`\sin(\phi + x)` acting only on the :math:`\phi` subspace.

        Parameters
        ----------
        x:
            additive phase offset inside the sine
        """
        pt_count = self.grid.pt_count

        vals = np.sin(self.grid.make_linspace() + x)
        sin_phi_matrix = sparse.dia_matrix(
            (vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        return sin_phi_matrix

    def _cos_phi_operator(self, x: float = 0) -> csc_matrix:
        r"""Operator :math:`\cos(\phi + x)` acting only on the :math:`\phi` subspace.

        Parameters
        ----------
        x:
            additive phase offset inside the cosine
        """
        pt_count = self.grid.pt_count

        vals = np.cos(self.grid.make_linspace() + x)
        cos_phi_matrix = sparse.dia_matrix(
            (vals, [0]), shape=(pt_count, pt_count)
        ).tocsc()
        return cos_phi_matrix

    def _cos_theta_operator(self) -> csc_matrix:
        r"""Return the :math:`\cos(\theta)` operator on the :math:`\theta` subspace."""
        dim_theta = 2 * self.ncut + 1
        cos_theta_matrix = (
            0.5
            * (
                sparse.dia_matrix(
                    ([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                )
                + sparse.dia_matrix(
                    ([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)
                )
            ).tocsc()
        )
        return cos_theta_matrix

    def cos_theta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`\cos(\theta)` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        native = sparse.kron(
            self._identity_phi(), self._cos_theta_operator(), format="csc"
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _sin_theta_operator(self) -> csc_matrix:
        r"""Return the :math:`\sin(\theta)` operator on the :math:`\theta` subspace."""
        dim_theta = 2 * self.ncut + 1
        sin_theta_matrix = (
            -0.5
            * 1j
            * (
                sparse.dia_matrix(
                    ([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                )
                - sparse.dia_matrix(
                    ([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)
                )
            ).tocsc()
        )
        return sin_theta_matrix

    def sin_theta_operator(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return the :math:`\sin(\theta)` operator in the native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns the operator in the native basis.
            If ``True``, the energy eigenspectrum is computed and the operator
            is returned in the energy eigenbasis. If ``energy_esys = esys``,
            where ``esys`` is a tuple ``(evals, evecs)`` of eigenvalues and
            eigenvectors, the operator is expressed in that eigenbasis without
            recomputing the spectrum.

        Returns
        -------
        Operator in the chosen basis. In the native basis it is returned as a
        :class:`scipy.sparse.csc_matrix`. In the eigenenergy basis it is returned
        as an ndarray of shape ``truncated_dim x truncated_dim``; if an explicit
        eigensystem of ``m`` eigenvectors is passed, the shape is ``m x m``.
        """
        native = sparse.kron(
            self._identity_phi(), self._sin_theta_operator(), format="csc"
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def plot_potential(
        self,
        theta_grid: Grid1d | None = None,
        contour_vals: list[float] | ndarray | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        r"""Draw a contour plot of the potential energy.

        Parameters
        ----------
        theta_grid:
            custom grid for :math:`\theta`; if ``None``, uses :attr:`_default_grid`
        contour_vals:
            optional list of contour level values
        **kwargs:
            plotting parameters
        """
        theta_grid = theta_grid or self._default_grid

        x_vals = self.grid.make_linspace()
        y_vals = theta_grid.make_linspace()
        return plot.contours(
            x_vals,
            y_vals,
            self.potential,
            contour_vals=contour_vals,
            xlabel=r"$\phi$",
            ylabel=r"$\theta$",
            **kwargs,
        )

    def wavefunction(
        self,
        esys: tuple[ndarray, ndarray] | None = None,
        which: int = 0,
        theta_grid: Grid1d | None = None,
    ) -> WaveFunctionOnGrid:
        r"""Return a zero-pi wave function in the :math:`\phi,\theta` basis.

        Parameters
        ----------
        esys:
            eigenvalues and eigenvectors; if ``None``, the eigensystem is recomputed
        which:
            index of the desired wave function (default: 0)
        theta_grid:
            custom grid for :math:`\theta`; if ``None``, uses :attr:`_default_grid`
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count=evals_count)
        else:
            _, evecs = esys

        theta_grid = theta_grid or self._default_grid
        dim_theta = 2 * self.ncut + 1
        state_amplitudes = evecs[:, which].reshape(self.grid.pt_count, dim_theta)

        # Calculate psi_{phi, theta} = sum_n state_amplitudes_{phi, n} A_{n, theta}
        # where a_{n, theta} = 1/sqrt(2 pi) e^{i n theta}
        n_vec = np.arange(-self.ncut, self.ncut + 1)
        theta_vec = theta_grid.make_linspace()
        a_n_theta = np.exp(1j * np.outer(n_vec, theta_vec)) / (2 * np.pi) ** 0.5
        wavefunc_amplitudes = np.matmul(state_amplitudes, a_n_theta).T
        wavefunc_amplitudes = utils.standardize_phases(wavefunc_amplitudes)

        grid2d = discretization.GridSpec(
            np.asarray(
                [
                    [self.grid.min_val, self.grid.max_val, self.grid.pt_count],
                    [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                ]
            )
        )
        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(
        self,
        esys: tuple[ndarray, ndarray] | None = None,
        which: int = 0,
        theta_grid: Grid1d | None = None,
        mode: str = "abs",
        zero_calibrate: bool = True,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot the 2D phase-basis wave function.

        Parameters
        ----------
        esys:
            eigenvalues and eigenvectors as obtained from :meth:`eigensys`; if
            ``None``, the eigensystem is recomputed
        which:
            index of the wave function to be plotted (default: 0)
        theta_grid:
            custom grid for :math:`\\theta`; if ``None``, uses :attr:`_default_grid`
        mode:
            amplitude-modifier choice from
            :data:`scqubits.core.constants.MODE_FUNC_DICT` (default: ``'abs'``)
        zero_calibrate:
            if ``True``, colors are calibrated so that zero amplitude maps to
            the palette's neutral color
        **kwargs:
            plot options
        """
        theta_grid = theta_grid or self._default_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, theta_grid=theta_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)  # type: ignore[operator]
        return plot.wavefunction2d(
            wavefunc,
            zero_calibrate=zero_calibrate,
            xlabel=r"$\phi$",
            ylabel=r"$\theta$",
            **kwargs,
        )
