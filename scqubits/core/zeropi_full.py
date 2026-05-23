# zeropi_full.py
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

import dataclasses

from collections.abc import Callable
from typing import Any

import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

import scqubits
import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.ui.qubit_widget as ui
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.convergence import ConvergenceCheckable, _status_rank
from scqubits.core.convergence_report import ConvergenceReport, TruncationChannel
from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem

# - ZeroPi noise class


class NoisyFullZeroPi(NoisySystem):
    """Noise mixin for :class:`FullZeroPi`."""

    pass


class FullZeroPi(
    base.QubitBaseClass,
    serializers.Serializable,
    NoisyFullZeroPi,
    ConvergenceCheckable,
):
    r"""Zero-Pi qubit including coupling to the :math:`\zeta` mode.

    See [Brooks2013]_ and [Dempster2014]_. The circuit is described by the
    Hamiltonian :math:`H = H_{0-\pi} + H_\text{int} + H_\zeta`, where

    .. math::

        &H_{0-\pi} = -2E_\text{CJ}\partial_\phi^2
                     +2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
                     +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta\\
        &\qquad\qquad\qquad+2E_{C\Sigma}(\delta C_J/C_J)\partial_\phi\partial_\theta
                     +2\,\delta E_J \sin\theta\sin(\phi-\varphi_\text{ext}/2)\\
        &H_\text{int} = 2E_{C\Sigma}dC\,\partial_\theta\partial_\zeta
                     + E_L dE_L \phi\,\zeta\\
        &H_\zeta = E_{\zeta} a^\dagger a

    expressed in phase basis. The definition of the relevant charging energies
    :math:`E_\text{CJ}`, :math:`E_{\text{C}\Sigma}`,
    Josephson energies :math:`E_\text{J}`, inductive energies :math:`E_\text{L}`,
    and relative amounts of disorder :math:`dC_\text{J}`, :math:`dE_\text{J}`,
    :math:`dC`, :math:`dE_\text{L}` follows [Groszkowski2018]_. Internally,
    the :class:`FullZeroPi` class formulates the Hamiltonian matrix via the
    product basis of the decoupled Zero-Pi qubit (see :class:`ZeroPi`)  on one hand, and the
    zeta LC oscillator on the other hand.

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
    dEL:
        relative disorder in EL, i.e., (EL1-EL2)/ELavg
    dCJ:
        relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    dC:
        relative disorder in large capacitances, i.e., (C1-C2)/Cavg
    ng:
        offset charge associated with theta
    zeropi_cutoff:
        cutoff in the number of states of the disordered zero-pi qubit
    zeta_cutoff:
        cutoff in the zeta oscillator basis (Fock state basis)
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

    EJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    EL = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    ECJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    EC = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    ECS = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    dEJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    dCJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    dC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dEL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    flux = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    grid = descriptors.WatchedProperty(
        Grid1d, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    ncut = descriptors.WatchedProperty(
        int, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    zeropi_cutoff = descriptors.WatchedProperty(
        int,
        "QUANTUMSYSTEM_UPDATE",
        inner_object_name="_zeropi",
        attr_name="truncated_dim",
    )

    def __init__(
        self,
        EJ: float,
        EL: float,
        ECJ: float,
        EC: float,
        dEJ: float,
        dCJ: float,
        dC: float,
        dEL: float,
        flux: float,
        ng: float,
        zeropi_cutoff: int,
        zeta_cutoff: int,
        grid: Grid1d,
        ncut: int,
        ECS: float | None = None,
        truncated_dim: int = 6,
        id_str: str | None = None,
        evals_method: Callable | str | None = None,
        evals_method_options: dict | None = None,
        esys_method: Callable | str | None = None,
        esys_method_options: dict | None = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self._zeropi = scqubits.ZeroPi(
            EJ=EJ,
            EL=EL,
            ECJ=ECJ,
            EC=EC,
            ng=ng,
            flux=flux,
            grid=grid,
            ncut=ncut,
            dEJ=dEJ,
            dCJ=dCJ,
            ECS=ECS,
            # the zeropi_cutoff defines the truncated_dim of the "base" zeropi object
            truncated_dim=zeropi_cutoff,
            id_str=self._id_str + " [interior ZeroPi]",
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.dC = dC
        self.dEL = dEL
        self.zeta_cutoff = zeta_cutoff
        self.truncated_dim = truncated_dim
        self._init_params.remove(
            "ECS"
        )  # used for file IO Serializable purposes; remove ECS as init parameter

        dispatch.CENTRAL_DISPATCH.register("GRID_UPDATE", self)

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return a default-parameter dict suitable for instantiating the class."""
        return {
            "EJ": 10.0,
            "EL": 0.04,
            "ECJ": 20.0,
            "EC": 0.04,
            "dEJ": 0.05,
            "dCJ": 0.05,
            "dC": 0.08,
            "dEL": 0.05,
            "ng": 0.1,
            "flux": 0.23,
            "ncut": 30,
            "zeropi_cutoff": 10,
            "zeta_cutoff": 40,
            "truncated_dim": 10,
        }

    @classmethod
    def create(cls) -> "FullZeroPi":
        """Use ipywidgets to instantiate a :class:`FullZeroPi` interactively."""
        phi_grid = discretization.Grid1d(-25.0, 25.0, 360)
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
            optional initial parameters; if None, uses :meth:`get_initdata`
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
        """Set instance parameters from GUI-provided keyword arguments."""
        for param_name, param_val in kwargs.items():
            if "grid_" in param_name:
                setattr(self.grid, param_name[5:], param_val)
            else:
                setattr(self, param_name, param_val)

    def receive(self, event: str, sender: object, **kwargs) -> None:
        """Re-broadcast a ``QUANTUMSYSTEM_UPDATE`` when the inner grid changes.

        Parameters
        ----------
        event:
            name of the dispatched event
        sender:
            object that fired the event
        """
        if sender is self._zeropi.grid:
            self.broadcast("QUANTUMSYSTEM_UPDATE")

    def __str__(self) -> str:
        """Return human-readable summary including subsystem dimensions."""
        output = super().__str__()
        output = output[: output.rfind("\n")]
        output = output[: output.rfind("\n")]
        line_indent = output[output.rfind("\n") :]

        output += (
            line_indent
            + " dim: {0}   --[ (theta, phi): {1} total, {2} truncated;  (zeta): {3}"
            " ]--\n".format(
                self.hilbertdim(),
                self._zeropi.hilbertdim(),
                self.zeropi_cutoff,
                self.zeta_cutoff,
            )
        )
        return output

    def set_EC_via_ECS(self, ECS: float) -> None:
        """Set :attr:`EC` by providing `ECS`, keeping `ECJ` constant.

        Parameters
        ----------
        ECS:
            target value of the total charging energy
        """
        self._zeropi.set_EC_via_ECS(ECS)

    @property
    def E_zeta(self) -> float:
        """Return the energy quantum of the zeta mode."""
        return (8.0 * self.EL * self.EC) ** 0.5

    def set_E_zeta(self, value: float) -> None:
        """Reject attempts to set :attr:`E_zeta` directly; raises :class:`ValueError`.

        Parameters
        ----------
        value:
            ignored; provided for setter signature compatibility
        """
        raise ValueError(
            "Cannot directly set `E_zeta`. Instead one can set its value through :attr:`EL`"
            " or :attr:`EC`."
        )

    @property
    def esys_method(self) -> Callable | str | None:
        """Method used for esys diagonalization on the inner :class:`ZeroPi`."""
        return self._zeropi.esys_method

    @esys_method.setter
    def esys_method(self, value: Callable | str | None = None) -> None:
        """Forward the new esys diagonalization method to the inner :class:`ZeroPi`.

        Parameters
        ----------
        value:
            method for esys diagonalization, callable or string representation
        """
        self._zeropi.esys_method = value

    @property
    def esys_method_options(self) -> dict | None:
        """Options dictionary for the esys diagonalization method."""
        return self._zeropi.esys_method_options

    @esys_method_options.setter
    def esys_method_options(self, value: dict | None = None) -> None:
        """Forward the new esys options dict to the inner :class:`ZeroPi`.

        Parameters
        ----------
        value:
            dictionary with esys diagonalization options
        """
        self._zeropi.esys_method_options = value

    @property
    def evals_method(self) -> Callable | str | None:
        """Method used for evals diagonalization on the inner :class:`ZeroPi`."""
        return self._zeropi.evals_method

    @evals_method.setter
    def evals_method(self, value: Callable | str | None = None) -> None:
        """Forward the new evals diagonalization method to the inner :class:`ZeroPi`.

        Parameters
        ----------
        value:
            method for evals diagonalization, callable or string representation
        """
        self._zeropi.evals_method = value

    @property
    def evals_method_options(self) -> dict | None:
        """Options dictionary for the evals diagonalization method."""
        return self._zeropi.evals_method_options

    @evals_method_options.setter
    def evals_method_options(self, value: dict | None = None) -> None:
        """Forward the new evals options dict to the inner :class:`ZeroPi`.

        Parameters
        ----------
        value:
            dictionary with evals diagonalization options
        """
        self._zeropi.evals_method_options = value

    def hamiltonian(
        self,
        return_parts: bool = False,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix | tuple[ndarray | csc_matrix, ndarray, ndarray, ndarray]:
        r"""Return Hamiltonian in product or eigenenergy basis.

        The product basis is obtained by discretizing :math:`\phi`, employing
        charge basis for :math:`\theta`, and Fock basis for :math:`\zeta`.

        Parameters
        ----------
        return_parts:
            If set to true, `hamiltonian` returns
            [hamiltonian, evals, evecs, g_coupling_matrix]
        energy_esys:
            If ``False`` (default), returns Hamiltonian in native Hamiltonian basis.
            If ``True``, the energy eigenspectrum is computed, returns Hamiltonian in
            the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns Hamiltonian in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Hamiltonian in chosen basis as csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, the Hamiltonian has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`. Otherwise, if eigenenergy basis
        is chosen, Hamiltonian has dimensions of m x m, for m given eigenvectors.
        """
        zeropi_dim = self.zeropi_cutoff
        zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)
        zeropi_diag_hamiltonian = sparse.dia_matrix(
            (zeropi_dim, zeropi_dim), dtype=np.complex128
        )
        zeropi_diag_hamiltonian.setdiag(zeropi_evals)

        zeta_dim = self.zeta_cutoff
        prefactor = self.E_zeta

        zeta_diag_hamiltonian = op.number_sparse(zeta_dim, prefactor)

        hamiltonian_mat = sparse.kron(
            zeropi_diag_hamiltonian,
            sparse.identity(zeta_dim, format="dia", dtype=np.complex128),
        )
        hamiltonian_mat += sparse.kron(
            sparse.identity(zeropi_dim, format="dia", dtype=np.complex128),
            zeta_diag_hamiltonian,
        )

        gmat = self.g_coupling_matrix(zeropi_evecs)
        zeropi_coupling = sparse.dia_matrix(
            (zeropi_dim, zeropi_dim), dtype=np.complex128
        )
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                zeropi_coupling += gmat[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)
        hamiltonian_mat += sparse.kron(
            zeropi_coupling, op.annihilation_sparse(zeta_dim)
        ) + sparse.kron(zeropi_coupling.conjugate().T, op.creation_sparse(zeta_dim))
        hmtocsc = hamiltonian_mat.tocsc()
        if return_parts:
            return (
                self.process_hamiltonian(
                    native_hamiltonian=hmtocsc, energy_esys=energy_esys
                ),
                zeropi_evals,
                zeropi_evecs,
                gmat,
            )
        return self.process_hamiltonian(
            native_hamiltonian=hmtocsc, energy_esys=energy_esys
        )

    def d_hamiltonian_d_flux(
        self,
        zeropi_evecs: ndarray | None = None,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return derivative of the Hamiltonian w.r.t flux in product or eigenenergy basis.

        Helper method :meth:`_zeropi_operator_in_product_basis` is employed which
        converts a zeropi operator into one in the product basis. If the user already
        has zeropi eigenvectors, they can be supplied via ``zeropi_evecs=ndarray``.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method :meth:`_zeropi_operator_in_product_basis`
            calculates zeropi eigenvectors and uses them to convert operator to the
            product basis, if product basis is chosen.
            If ``zeropi_evecs = zeropievecs``, where ``zeropievecs`` is an ndarray,
            and product basis is chosen, helper method
            :meth:`_zeropi_operator_in_product_basis` uses ``zeropievecs`` to convert
            the operator to the product basis.
        energy_esys:
            If ``False`` (default), returns operator in the product basis.
            If ``True``, the energy eigenspectrum is computed, returns operator in the
            energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Operator in chosen basis. If product basis is chosen, operator
        returned as a csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, operator has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`, and is returned as an ndarray.
        Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
        for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.d_hamiltonian_d_flux(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ(
        self,
        zeropi_evecs: ndarray | None = None,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return derivative of the Hamiltonian w.r.t EJ in product or eigenenergy basis.

        Helper method :meth:`_zeropi_operator_in_product_basis` is employed which
        converts a zeropi operator into one in the product basis. If the user already
        has zeropi eigenvectors, they can be supplied via ``zeropi_evecs=ndarray``.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method :meth:`_zeropi_operator_in_product_basis`
            calculates zeropi eigenvectors and uses them to convert operator to the
            product basis, if product basis is chosen.
            If ``zeropi_evecs = zeropievecs``, where ``zeropievecs`` is an ndarray,
            and product basis is chosen, helper method
            :meth:`_zeropi_operator_in_product_basis` uses ``zeropievecs`` to convert
            the operator to the product basis.
        energy_esys:
            If ``False`` (default), returns operator in the product basis.
            If ``True``, the energy eigenspectrum is computed, returns operator in the
            energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Operator in chosen basis. If product basis chosen, operator
        returned as a csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, operator has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`, and is returned as an ndarray.
        Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
        for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.d_hamiltonian_d_EJ(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_ng(
        self, energy_esys: bool | tuple[ndarray, ndarray] = False
    ) -> ndarray | csc_matrix:
        r"""Return derivative of the Hamiltonian w.r.t ng in native or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If ``False`` (default), returns operator in the native basis.
            If ``True``, the energy eigenspectrum is computed, returns operator in the
            energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Operator in chosen basis. If native basis chosen, operator
        returned as a csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, operator has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`, and is returned as an ndarray.
        Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
        for m given eigenvectors, and is returned as an ndarray.
        """
        native = -8 * self.EC * self.n_theta_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _zeropi_operator_in_product_basis(
        self,
        zeropi_operator: ndarray | csc_matrix,
        zeropi_evecs: ndarray | None = None,
    ) -> csc_matrix:
        """Convert a zeropi operator into one in the product basis.

        Parameters
        ----------
        zeropi_operator:
            operator acting in the zeropi Hilbert space (sparse matrix or ndarray)
        zeropi_evecs:
            eigenvectors of the inner :class:`ZeroPi` to use when transforming;
            if None (default), they are calculated on the fly

        Returns
        -------
        operator written in the product basis
        """
        zeropi_dim = self.zeropi_cutoff
        zeta_dim = self.zeta_cutoff

        if zeropi_evecs is None:
            _, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        op_eigen_basis = sparse.dia_matrix(
            (zeropi_dim, zeropi_dim), dtype=np.complex128
        )  # guaranteed to be zero?

        op_zeropi = spec_utils.get_matrixelement_table(zeropi_operator, zeropi_evecs)
        for n in range(zeropi_dim):
            for m in range(zeropi_dim):
                op_eigen_basis += op_zeropi[n, m] * op.hubbard_sparse(n, m, zeropi_dim)

        return sparse.kron(  # type: ignore[return-value]
            op_eigen_basis,
            sparse.identity(zeta_dim, format="csc", dtype=np.complex128),
            format="csc",
        )

    def i_d_dphi_operator(
        self,
        zeropi_evecs: ndarray | None = None,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return :math:`i\,d/d\phi` operator in the product or eigenenergy basis.

        Helper method :meth:`_zeropi_operator_in_product_basis` is employed which
        converts a zeropi operator into one in the product basis. If the user already
        has zeropi eigenvectors, they can be supplied via ``zeropi_evecs=ndarray``.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method :meth:`_zeropi_operator_in_product_basis`
            calculates zeropi eigenvectors and uses them to convert operator to the
            product basis, if product basis is chosen.
            If ``zeropi_evecs = zeropievecs``, where ``zeropievecs`` is an ndarray,
            and product basis is chosen, helper method
            :meth:`_zeropi_operator_in_product_basis` uses ``zeropievecs`` to convert
            the operator to the product basis.
        energy_esys:
            If ``False`` (default), returns operator in the product basis.
            If ``True``, the energy eigenspectrum is computed, returns operator in the
            energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Operator in chosen basis. If product basis chosen, operator
        returned as a csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, operator has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`, and is returned as an ndarray.
        Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
        for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.i_d_dphi_operator(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def n_theta_operator(
        self,
        zeropi_evecs: ndarray | None = None,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return :math:`n_\theta` operator in the product or eigenenergy basis.

        Helper method :meth:`_zeropi_operator_in_product_basis` is employed which
        converts a zeropi operator into one in the product basis. If the user already
        has zeropi eigenvectors, they can be supplied via ``zeropi_evecs=ndarray``.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method :meth:`_zeropi_operator_in_product_basis`
            calculates zeropi eigenvectors and uses them to convert operator to the
            product basis, if product basis is chosen.
            If ``zeropi_evecs = zeropievecs``, where ``zeropievecs`` is an ndarray,
            and product basis is chosen, helper method
            :meth:`_zeropi_operator_in_product_basis` uses ``zeropievecs`` to convert
            the operator to the product basis.
        energy_esys:
            If ``False`` (default), returns operator in the product basis.
            If ``True``, the energy eigenspectrum is computed, returns operator in the
            energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Operator in chosen basis. If product basis chosen, operator
        returned as a csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, operator has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`, and is returned as an ndarray.
        Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
        for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.n_theta_operator(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def phi_operator(
        self,
        zeropi_evecs: ndarray | None = None,
        energy_esys: bool | tuple[ndarray, ndarray] = False,
    ) -> ndarray | csc_matrix:
        r"""Return :math:`\phi` operator in the product or eigenenergy basis.

        Helper method :meth:`_zeropi_operator_in_product_basis` is employed which
        converts a zeropi operator into one in the product basis. If the user already
        has zeropi eigenvectors, they can be supplied via ``zeropi_evecs=ndarray``.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method :meth:`_zeropi_operator_in_product_basis`
            calculates zeropi eigenvectors and uses them to convert operator to the
            product basis, if product basis is chosen.
            If ``zeropi_evecs = zeropievecs``, where ``zeropievecs`` is an ndarray,
            and product basis is chosen, helper method
            :meth:`_zeropi_operator_in_product_basis` uses ``zeropievecs`` to convert
            the operator to the product basis.
        energy_esys:
            If ``False`` (default), returns operator in the product basis.
            If ``True``, the energy eigenspectrum is computed, returns operator in the
            energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy
            eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
        Operator in chosen basis. If product basis chosen, operator
        returned as a csc_matrix. If the eigenenergy basis is chosen,
        unless `energy_esys` is specified, operator has dimensions of
        :attr:`truncated_dim` x :attr:`truncated_dim`, and is returned as an ndarray.
        Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
        for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.phi_operator(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """Return the Hilbert space dimension.

        Returns
        -------
        Product ``zeropi_cutoff * zeta_cutoff`` -- the dimension of the
        full ``ZeroPi`` Hilbert space (zero-pi sector tensor zeta-mode).
        """
        return self.zeropi_cutoff * self.zeta_cutoff

    # ----------------------------------------------------------- convergence
    # FullZeroPi is hierarchical: an interior ZeroPi (its grid / ncut) is
    # diagonalized and truncated to ``zeropi_cutoff`` levels, then coupled to a
    # zeta oscillator (``zeta_cutoff``). Convergence is therefore checked in two
    # layers, mirroring HilbertSpace: layer 1 delegates to the interior ZeroPi's
    # own estimate_convergence (verifying the 0-pi basis, with its FD box/stencil
    # and edge diagnostics); layer 2 refines the coupling cutoffs zeropi_cutoff
    # and zeta_cutoff and re-diagonalizes the full coupled system.
    _convergence_axes: tuple[str, ...] = ("zeropi_cutoff", "zeta_cutoff")
    _convergence_basis: str = "zeropi(discretized_phi+charge) x zeta_fock"

    def _convergence_truncation_channel(self, axis: str) -> TruncationChannel:
        """``zeropi_cutoff`` -> ``composite_coupling`` (how many 0-pi levels enter
        the coupling); ``zeta_cutoff`` -> ``HO_tail`` (zeta Fock cutoff)."""
        return "HO_tail" if axis == "zeta_cutoff" else "composite_coupling"

    def _convergence_step(self, axis: str) -> int:
        """Refinement step for a coupling cutoff.

        ``truncated_dim``-style step; ``zeropi_cutoff`` is capped at the interior
        ZeroPi's available levels (its bare Hilbert dimension), while the zeta
        Fock space is unbounded.
        """
        current = self._convergence_axis_value(axis)
        step = max(2, current // 4)
        if axis == "zeropi_cutoff":
            headroom = int(self._zeropi.hilbertdim()) - current
            step = min(step, max(1, headroom // 2))
        return step

    def estimate_convergence(  # type: ignore[override]
        self,
        n_levels: int = 6,
        mode: str = "verify",
        scope: str = "absolute",
        target_abs_GHz: float | None = None,
        target_gap_rel: float = 1e-3,
        g_floor_GHz: float = 1e-3,
        assume_inner_converged: bool = False,
    ) -> ConvergenceReport:
        """Estimate convergence of the coupled FullZeroPi spectrum in two layers.

        Layer 1 (interior basis): unless ``assume_inner_converged=True``, delegate
        to the interior :class:`.ZeroPi`'s own ``estimate_convergence`` (the phi
        grid box/spacing and theta charge cutoff) for the lowest ``zeropi_cutoff``
        plus refinement-reach levels, and attach it under
        ``report.derived["interior_zeropi"]``.

        Layer 2 (coupling truncation): refine ``zeropi_cutoff`` (how many 0-pi
        levels enter the coupling) and ``zeta_cutoff`` (the zeta Fock cutoff),
        re-diagonalizing the full coupled system and comparing cluster-matched
        spectra. The ``aggregate_status`` is the worse of the two layers.

        Parameters
        ----------
        n_levels:
            Number of lowest coupled eigenvalues to assess.
        mode:
            ``"quick"`` (no coupled re-diagonalization; verify-recommended),
            ``"verify"`` (one refinement; default), or ``"strict"`` (ratio test).
        scope, target_abs_GHz, target_gap_rel, g_floor_GHz:
            As in :meth:`ConvergenceCheckable.estimate_convergence`.
        assume_inner_converged:
            If True, skip the layer-1 interior-ZeroPi check.

        Returns
        -------
        :class:`~scqubits.core.convergence_report.ConvergenceReport`
        """
        if n_levels > self.hilbertdim():
            raise ValueError(
                f"n_levels={n_levels} exceeds the FullZeroPi dimension "
                f"{self.hilbertdim()}; reduce n_levels or raise zeropi_cutoff / "
                "zeta_cutoff"
            )

        if mode == "quick":
            composite = self._convergence_unverified_report(
                n_levels,
                scope,
                mode,
                recommendations=[
                    "coupled truncation has no cheap estimate; run mode='verify' "
                    "for a coupled-spectrum verdict"
                ],
                warning="composite_verify_recommended",
            )
        else:
            composite = ConvergenceCheckable.estimate_convergence(
                self,
                n_levels=n_levels,
                mode=mode,
                scope=scope,
                target_abs_GHz=target_abs_GHz,
                target_gap_rel=target_gap_rel,
                g_floor_GHz=g_floor_GHz,
            )

        if assume_inner_converged:
            return composite

        reach = self._convergence_step("zeropi_cutoff")
        if mode == "strict":
            reach *= 2
        n_inner = max(
            1, min(self.zeropi_cutoff + reach, int(self._zeropi.hilbertdim()))
        )
        inner = self._zeropi.estimate_convergence(
            n_levels=n_inner,
            mode=mode,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
            g_floor_GHz=g_floor_GHz,
        )

        recommendations = list(composite.recommendations)
        if _status_rank(inner.aggregate_status) >= _status_rank("marginal"):
            recommendations.append(
                f"the interior ZeroPi basis is {inner.aggregate_status} for the "
                "lowest coupled levels; converge it first (phi grid / theta ncut) "
                "-- the full ZeroPi cannot be more converged than its 0-pi sector"
            )
        aggregate = composite.aggregate_status
        if _status_rank(inner.aggregate_status) > _status_rank(aggregate):
            aggregate = inner.aggregate_status
        derived = dict(composite.derived or {})
        derived["interior_zeropi"] = inner
        return dataclasses.replace(
            composite,
            aggregate_status=aggregate,
            recommendations=recommendations,
            derived=derived or None,
        )

    def _evals_calc(
        self, evals_count: int, hamiltonian_mat: csc_matrix | None = None
    ) -> ndarray:
        """Return the lowest ``evals_count`` eigenvalues, sorted ascending.

        Uses :func:`scqubits.utils.spectrum_utils.eigsh_safe` with ``which="SA"``
        and an explicit :func:`numpy.sort` pass to enforce ascending order.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues
        hamiltonian_mat:
            optional precomputed Hamiltonian matrix; if None, :meth:`hamiltonian`
            is called to construct it
        """
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()  # type: ignore[assignment]
        evals = spec_utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            # sigma=0.0,
            which="SA",
            return_eigenvectors=False,
        )
        return np.sort(evals)

    def _esys_calc(
        self, evals_count: int, hamiltonian_mat: csc_matrix | None = None
    ) -> tuple[ndarray, ndarray]:
        """Return the lowest ``evals_count`` eigenvalues and corresponding eigenvectors.

        Uses :func:`scqubits.utils.spectrum_utils.eigsh_safe` with ``which="SA"``
        followed by :func:`scqubits.utils.spectrum_utils.order_eigensystem` to
        return ascending eigenvalues with matching eigenvectors.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues
        hamiltonian_mat:
            optional precomputed Hamiltonian matrix; if None, :meth:`hamiltonian`
            is called to construct it
        """
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()  # type: ignore[assignment]
        evals, evecs = spec_utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            # sigma=0.0,
            which="SA",
            return_eigenvectors=True,
        )
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def g_phi_coupling_matrix(self, zeropi_states: ndarray) -> ndarray:
        r"""Return matrix of coupling strengths :math:`g^\phi_{ll'}`.

        See [Dempster2014]_, Eq. (18). Most commonly, `zeropi_states` will contain
        eigenvectors of the `DisorderedZeroPi` type.

        Parameters
        ----------
        zeropi_states:
            array of zeropi eigenvectors used to evaluate the matrix elements
        """
        prefactor = self.EL * (self.dEL / 2.0) * (8.0 * self.EC / self.EL) ** 0.25
        return prefactor * spec_utils.get_matrixelement_table(
            self._zeropi.phi_operator(), zeropi_states
        )

    def g_theta_coupling_matrix(self, zeropi_states: ndarray) -> ndarray:
        r"""Return matrix of coupling strengths :math:`ig^\theta_{ll'}`.

        See [Dempster2014]_, Eq. (17).

        Parameters
        ----------
        zeropi_states:
            array of zeropi eigenvectors used to evaluate the matrix elements
        """
        prefactor = 1j * self.ECS * (self.dC / 2.0) * (32.0 * self.EL / self.EC) ** 0.25
        return prefactor * spec_utils.get_matrixelement_table(
            self._zeropi.n_theta_operator(), zeropi_states
        )

    def g_coupling_matrix(
        self, zeropi_states: ndarray | None = None, evals_count: int | None = None
    ) -> ndarray:
        r"""Return matrix of coupling strengths :math:`g_{ll'}`.

        See [Dempster2014]_, text above Eq. (17). If `zeropi_states==None`, then a set
        of `self.zeropi` eigenstates is calculated. Only in that case is `evals_count`
        used for the eigenstate number (and hence the coupling matrix size).

        Parameters
        ----------
        zeropi_states:
            array of zeropi eigenvectors used to evaluate the matrix elements; if
            None, eigenstates are computed from the inner :class:`ZeroPi`
        evals_count:
            number of zeropi eigenstates to compute when `zeropi_states` is None;
            defaults to ``self._zeropi.truncated_dim``
        """
        if evals_count is None:
            evals_count = self._zeropi.truncated_dim
        if zeropi_states is None:
            _, zeropi_states = self._zeropi.eigensys(evals_count=evals_count)
        return self.g_phi_coupling_matrix(zeropi_states) + self.g_theta_coupling_matrix(
            zeropi_states
        )
