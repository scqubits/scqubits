# routines.py
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

import functools
import operator as builtin_op
import re

from collections.abc import Callable
from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem

from abc import ABC

import dill
import numpy as np
import qutip as qt
import scipy as sp
import sympy as sm

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

import scqubits.core.circuit as circuit
import scqubits.core.diag as diag
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.utils.spectrum_utils as utils

from scqubits import HilbertSpace, settings
from scqubits.core import descriptors
from scqubits.core import operators as op
from scqubits.core.circuit_internals.charge_basis_operators import (
    _cos_theta,
    _exp_i_theta_operator,
    _n_theta_operator,
    _sin_theta,
)
from scqubits.core.circuit_internals.utils import get_trailing_number
from scqubits.core.circuit_internals.sawtooth import sawtooth_potential
from scqubits.core.circuit_internals.discretized_phi_operators import (
    _cos_phi,
    _i_d2_dphi2_operator,
    _i_d_dphi_operator,
    _phi_operator,
    _sin_phi,
)
from scqubits.core.circuit_internals.matrix_helpers import (
    _cos_dia,
    _cos_dia_dense,
    _sin_dia,
    _sin_dia_dense,
    matrix_power_sparse,
)
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.circuit_internals.operator_factories import (
    grid_operator_func_factory,
    hierarchical_diagonalization_func_factory,
    operator_func_factory,
)
from scqubits.core.circuit_internals.sympy_helpers import (
    _generate_symbols_list,
    round_symbolic_expr,
)
from scqubits.io_utils.fileio import IOData
from scqubits.io_utils.fileio_serializers import dict_serialize
from scqubits.utils.misc import (
    Qobj_to_scipy_csc_matrix,
    check_sync_status_circuit,
    flatten_list_recursive,
    unique_elements_in_list,
)
from scqubits.utils.spectrum_utils import (
    convert_matrix_to_qobj,
    identity_wrap,
    order_eigensystem,
)

__all__ = [
    "CircuitRoutines",
]


@contextmanager
def _dispatch_suspended():
    """Temporarily disable :data:`settings.DISPATCH_ENABLED`, restoring on exit."""
    old_status = settings.DISPATCH_ENABLED
    settings.DISPATCH_ENABLED = False
    try:
        yield
    finally:
        settings.DISPATCH_ENABLED = old_status


_PROPERTY_SETTER_BY_TYPE: dict[str, str] = {
    "update_param_vars": "_set_property_and_update_param_vars",
    "update_external_flux_or_charge": "_set_property_and_update_ext_flux_or_charge",
    "update_cutoffs": "_set_property_and_update_cutoffs",
}


class CircuitRoutines(ABC):
    """Mixin/ABC providing shared routines for :class:`Circuit` and :class:`Subsystem`.

    Holds the bulk of the circuit-quantization machinery: parameter syncing,
    operator construction, hierarchical diagonalization helpers, and
    Hamiltonian evaluation. Concrete subclasses are responsible for
    initializing the declared attributes; this class only mixes in the
    behaviour that operates on them.
    """

    # Attributes set by concrete subclasses (Circuit, Subsystem) in their __init__.
    # Declared here so mypy can resolve cross-subclass access patterns in shared
    # methods defined on this ABC.
    hierarchical_diagonalization: bool
    var_categories: dict[str, list[int]]
    dynamic_var_indices: list[int]
    external_fluxes: list[Any]
    symbolic_params: dict[Any, Any]
    offset_charges: list[Any]
    free_charges: list[Any]
    type_of_matrices: str
    system_hierarchy: list[Any]
    parent: Any
    is_purely_harmonic: bool
    subsystem_trunc_dims: list[Any]
    is_child: bool
    subsystems: list[Any]
    symbolic_circuit: Any
    closure_branches: list[Any]
    cutoff_names: list[str]
    discretized_phi_range: dict[int, Any]
    ext_basis: Any
    transformation_matrix: Any
    use_dynamic_flux_grouping: bool
    affine_transformation_matrix: Any
    _default_grid_phi: Any
    evals_method: Any
    evals_method_options: Any
    esys_method: Any
    esys_method_options: Any
    _hamiltonian_sym_for_numerics: Any
    _out_of_sync_with_parent: bool
    hamiltonian_symbolic: Any
    potential_symbolic: Any
    subsystem_interactions: Any
    vars: dict[str, Any]
    broadcast: Callable[..., Any]
    _generate_sym_potential: Callable[..., Any]
    _find_and_set_sym_attrs: Callable[..., Any]
    _generate_hamiltonian_sym_for_numerics: Callable[..., Any]
    _sym_subsystem_hamiltonian_and_interactions: Callable[..., Any]
    _is_expression_purely_harmonic: Callable[..., Any]
    _basis_for_var_index: Callable[..., Any]
    _evaluate_symbolic_expr: Callable[..., Any]
    truncated_dim: int
    hilbert_space: Any
    operators_by_name: Any
    _id_str: str
    eigensys: Callable[..., Any]
    generate_bare_esys: Callable[..., Any]
    _get_eval_hamiltonian_string: Callable[..., Any]
    _is_diagonalization_necessary: Callable[..., Any]

    _read_only_attributes = [
        "ext_basis",
        "transformation_matrix",
        "hierarchical_diagonalization",
        "system_hierarchy",
        "subsystem_trunc_dims",
        "discretized_phi_range",
        "cutoff_names",
        "closure_branches",
        "external_fluxes",
        "use_dynamic_flux_grouping",
    ]

    @classmethod
    def create(cls) -> base.QuantumSystem:
        """Factory method placeholder; concrete subclasses must override."""
        raise NotImplementedError

    # methods for serialization
    def serialize(self) -> "IOData":
        """Return an :class:`IOData` representation of ``self`` for file I/O."""
        obj_in_bytes = dill.dumps(self)
        initdata = {"subsystem_in_hex": obj_in_bytes.hex()}
        if hasattr(self, "_id_str"):
            initdata["id_str"] = self._id_str
        iodata = dict_serialize(initdata)
        iodata.typename = type(self).__name__
        return iodata

    @classmethod
    def deserialize(cls, io_data: "IOData"):
        """Reconstruct a Circuit/Subsystem instance from :class:`IOData`.

        Parameters
        ----------
        io_data:
            serialized payload as produced by :meth:`serialize`
        """
        obj_in_bytes = bytes.fromhex(io_data.as_kwargs()["subsystem_in_hex"])
        return dill.loads(obj_in_bytes)

    def return_root_child(self, var_index: int):
        """Return the root child subsystem holding the given variable index.

        Parameters
        ----------
        var_index:
            index of one of the dynamical degrees of freedom (from
            :attr:`dynamic_var_indices`)

        Returns
        -------
        :class:`Subsystem` instance with ``var_index`` in its
        :attr:`dynamic_var_indices`.
        """
        if (
            not self.hierarchical_diagonalization
            and var_index in self.dynamic_var_indices
        ):
            return self
        for subsys in self.subsystems:
            if var_index in subsys.dynamic_var_indices:
                return subsys.return_root_child(var_index)

    def return_parent_circuit(self):
        """Return the parent Circuit instance."""
        if not self.is_child:
            return self
        return self.parent.return_parent_circuit()

    def _diagonalize_purely_harmonic_hamiltonian(self, return_osc_dict: bool = False):
        """Decouple harmonic oscillators in purely harmonic Hamiltonians.

        Mutates several instance attributes (``normal_mode_freqs``,
        ``_hamiltonian_sym_for_numerics``, ``osc_lengths``, ``osc_freqs``,
        ``osc_eigvecs``, ``undiagonalized_osc_params``). Uses
        :func:`numpy.linalg.eig` on a non-Hermitian product, so the returned
        eigenvalues may in principle be complex; their square roots are taken
        as normal-mode frequencies.

        Parameters
        ----------
        return_osc_dict:
            if ``True``, return a dictionary with the diagonalization data
            (normal-mode frequencies, eigenvectors and oscillator lengths).
        """
        if not self.is_purely_harmonic:
            raise ValueError("The Subsystem Hamiltonian is not purely harmonic.")
        num_oscs = len(self.var_categories["extended"])
        # Construct capacitance and inductance matrices from the symbolic hamiltonian
        EC = np.zeros([num_oscs, num_oscs])
        EL = np.zeros([num_oscs, num_oscs])
        # substitute all external fluxes in the symbolic Hamiltonian
        hamiltonian = self.hamiltonian_symbolic
        for param in (
            self.external_fluxes
            + list(self.symbolic_params.keys())
            + self.offset_charges
            + self.free_charges
        ):
            hamiltonian = hamiltonian.subs(param, getattr(self, param.name))
        ext_var_indices = self.var_categories["extended"]
        # filling the matrices
        for i in range(num_oscs):
            for j in range(num_oscs):
                if i == j:
                    EC[i, j] = hamiltonian.coeff(f"Q{ext_var_indices[i]}**2") / 4
                    EL[i, j] = hamiltonian.coeff(f"θ{ext_var_indices[i]}**2") * 2
                else:
                    EC[i, j] = (
                        hamiltonian.coeff(
                            f"Q{ext_var_indices[i]}*Q{ext_var_indices[j]}"
                        )
                        / 8
                    )
                    EL[i, j] = hamiltonian.coeff(
                        f"θ{ext_var_indices[i]}*θ{ext_var_indices[j]}"
                    )
        # diagonalizing the matrices
        normal_mode_freqs_sq, eig_vecs = np.linalg.eig(8 * EC @ EL)

        self.normal_mode_freqs = normal_mode_freqs_sq**0.5

        self._hamiltonian_sym_for_numerics = round_symbolic_expr(
            self._transform_hamiltonian(self.hamiltonian_symbolic, eig_vecs).expand(),
            12,
        )
        for flux in self.external_fluxes:
            self._hamiltonian_sym_for_numerics = (
                self._hamiltonian_sym_for_numerics.subs(flux, flux * np.pi * 2)
            )
        # storing the annihilation operators in the eigenbasis
        osc_lengths = (
            np.diagonal(
                8
                * np.linalg.inv(eig_vecs.T @ np.linalg.inv(EC) @ eig_vecs)
                @ np.linalg.inv(eig_vecs.T @ EL @ eig_vecs)
            )
            ** 0.25
        )
        old_osc_lengths, old_osc_freqs = self._set_harmonic_basis_osc_params(
            hamiltonian=self.hamiltonian_symbolic
        )
        self.osc_lengths = dict(zip(self.var_categories["extended"], osc_lengths))
        self.osc_freqs = dict(
            zip(self.var_categories["extended"], normal_mode_freqs_sq**0.5)
        )
        self.osc_eigvecs = eig_vecs
        self.undiagonalized_osc_params = {
            "osc_freqs": old_osc_freqs,
            "osc_lengths": old_osc_lengths,
        }

        if return_osc_dict:
            osc_dict = {
                "normal_mode_freqs": normal_mode_freqs_sq**0.5,
                "eig_vecs": eig_vecs,
                "osc_lengths": osc_lengths,
            }
            return osc_dict

    def _transform_hamiltonian(
        self,
        hamiltonian: sm.Expr,
        transformation_matrix: ndarray,
        return_transformed_exprs: bool = False,
    ):
        """Transform the Hamiltonian to a new set of variables.

        Parameters
        ----------
        hamiltonian:
            symbolic Hamiltonian to transform
        transformation_matrix:
            matrix relating the original to the new (extended) coordinates
        return_transformed_exprs:
            if ``True``, return only the transformed ``Q`` and ``θ``
            expressions; otherwise return the substituted Hamiltonian.
        """
        ext_var_indices = self.var_categories["extended"]
        num_vars = len(ext_var_indices)
        Q_vars = [sm.symbols(f"Q{var_idx}") for var_idx in ext_var_indices]
        θ_vars = [sm.symbols(f"θ{var_idx}") for var_idx in ext_var_indices]

        Qn_vars = [sm.symbols(f"Qn{var_idx}") for var_idx in ext_var_indices]
        θn_vars = [sm.symbols(f"θn{var_idx}") for var_idx in ext_var_indices]

        Q_exprs = np.linalg.inv(transformation_matrix.T).dot(Qn_vars)
        θ_exprs = transformation_matrix.dot(θn_vars)
        if return_transformed_exprs:
            return np.linalg.inv(transformation_matrix.T).dot(
                Q_vars
            ), transformation_matrix.dot(θ_vars)

        for idx in range(num_vars):
            hamiltonian = hamiltonian.subs(Q_vars[idx], Q_exprs[idx]).subs(
                θ_vars[idx], θ_exprs[idx]
            )
        for idx in range(num_vars):
            hamiltonian = hamiltonian.subs(Qn_vars[idx], Q_vars[idx]).subs(
                θn_vars[idx], θ_vars[idx]
            )

        return hamiltonian

    def __setattr__(self, name: str, value: Any) -> None:
        """Restrict attribute creation/modification once the instance is frozen.

        Parameters
        ----------
        name:
            attribute name being set
        value:
            value to assign
        """
        if self._frozen and name in self._read_only_attributes:
            raise AttributeError(
                f"{name} is a read only attribute. Please use configure method to change this property of Circuit/Subsystem instance."
            )
        if not self._frozen or name in dir(self):
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Creating new attributes is disabled: [{name}, {value}]."
            )

    def __reduce__(self):
        """Custom ``__reduce__`` for pickling, also preserving dynamic properties."""
        # needed for multiprocessing / proper pickling
        pickle_func, pickle_args, pickled_state = object.__reduce__(self)
        pickled_dict = self.__dict__
        pickled_properties = {
            property_name: property_obj
            for property_name, property_obj in self.__class__.__dict__.items()
            if isinstance(
                property_obj, (property, descriptors.WatchedProperty)
            )  # WatchedProperty is not a child of property
        }
        return pickle_func, pickle_args, (pickled_dict, pickled_properties)

    def __setstate__(self, state: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Restore instance ``__dict__`` and dynamic properties when unpickling.

        Parameters
        ----------
        state:
            tuple ``(pickled_dict, pickled_properties)`` produced by
            :meth:`__reduce__`.
        """
        pickled_dict, pickled_properties = state
        object.__setattr__(self, "_frozen", False)
        self.__dict__ = pickled_dict

        for property_name, property_obj in pickled_properties.items():
            setattr(self.__class__, property_name, property_obj)

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return an empty dict; concrete circuit subclasses have no defaults."""
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}
        return {}

    def cutoffs_dict(self) -> dict[int, int]:
        """Map each dynamic variable index to its associated cutoff.

        Returns
        -------
        Dictionary ``{var_index: cutoff}``.
        """
        cutoffs_dict = {}

        for var_index in self.dynamic_var_indices:
            for cutoff_name in self.cutoff_names:
                if str(var_index) in cutoff_name:
                    cutoffs_dict[var_index] = getattr(self, cutoff_name)
        return cutoffs_dict

    ##############################################
    ####### Methods for parameter updates ########
    ##############################################

    def _sync_parameters_with_parent(self):
        """Method syncs the parameters of the subsystem with the parent instance."""
        for param_var in (
            self.external_fluxes
            + self.offset_charges
            + self.free_charges
            + list(self.symbolic_params.keys())
        ):
            setattr(self, param_var.name, getattr(self.parent, param_var.name))

        # sync discretized phi range
        for var_index in self.var_categories["extended"]:
            self.discretized_phi_range[var_index] = self.parent.discretized_phi_range[
                var_index
            ]
        # sync ext_basis
        subsys_index_in_parent = self.parent.subsystems.index(self)
        self.ext_basis = self.parent.ext_basis[subsys_index_in_parent]

    def _sync_parameters_with_subsystems(self):
        """Re-assign all parameter attributes to trigger subsystem-side setters."""
        for param_var in (
            self.external_fluxes
            + self.offset_charges
            + self.free_charges
            + list(self.symbolic_params.keys())
        ):
            setattr(self, param_var.name, getattr(self, param_var.name))
        # the setters will make sure to sync the parameters with the subsystems

    def _set_sync_status_to_True(self, reset_affected_subsystem_indices: bool = False):
        """Mark the instance and nested subsystems as in-sync.

        Parameters
        ----------
        reset_affected_subsystem_indices:
            if ``True``, also empty :attr:`affected_subsystem_indices`.
        """
        if not self.hierarchical_diagonalization:
            return None
        self._out_of_sync = False
        if reset_affected_subsystem_indices:
            self.affected_subsystem_indices: list[int] = []
        for subsys in self.subsystems:
            if subsys.hierarchical_diagonalization:
                subsys._set_sync_status_to_True()
                subsys._out_of_sync = False

    def receive(self, event: str, sender: object, **kwargs) -> None:
        """Help :mod:`central_dispatch` track sync status in Circuit/Subsystem.

        Parameters
        ----------
        event:
            event name dispatched by :mod:`central_dispatch`
        sender:
            object that emitted the event
        """
        if sender is self:
            self.broadcast("QUANTUMSYSTEM_UPDATE")
            if self.hierarchical_diagonalization:
                self.hilbert_space._out_of_sync = True
        if self.hierarchical_diagonalization and (sender in self.subsystems):
            sender._out_of_sync_with_parent = True  # type: ignore[attr-defined]
            self._store_updated_subsystem_index(self.subsystems.index(sender))
            self.broadcast("CIRCUIT_UPDATE")
            self._out_of_sync = True
            self.hilbert_space._out_of_sync = True

    def _store_updated_subsystem_index(self, index: int) -> None:
        """Append a modified subsystem index to :attr:`affected_subsystem_indices`.

        Parameters
        ----------
        index:
            position of the modified subsystem in :attr:`subsystems`.
        """
        if not self.hierarchical_diagonalization:
            raise RuntimeError(f"{self} has no subsystems.")
        if index not in self.affected_subsystem_indices:
            self.affected_subsystem_indices.append(index)

    def _fetch_symbolic_hamiltonian(self):
        """Method to fetch the symbolic hamiltonian of an instance."""
        if isinstance(self, circuit.Circuit):
            # when the Circuit instance is created from a symbolic Hamiltonian, or nothing is updated or changed
            if not hasattr(self, "symbolic_circuit") or self._out_of_sync:
                return self.hamiltonian_symbolic

            self.symbolic_circuit.configure(
                transformation_matrix=self.symbolic_circuit.transformation_matrix,
                closure_branches=self.symbolic_circuit.closure_branches,
            )
            hamiltonian_symbolic = self.symbolic_circuit.hamiltonian_symbolic

            # if the flux is static, remove the linear terms from the potential
            if not self.symbolic_circuit.use_dynamic_flux_grouping:
                hamiltonian_symbolic = self._shift_harmonic_oscillator_potential(
                    hamiltonian_symbolic
                )

            return hamiltonian_symbolic
        else:
            full_hamiltonian = self.parent._fetch_symbolic_hamiltonian()
            non_operator_symbols = (
                self.offset_charges
                + self.free_charges
                + self.external_fluxes
                + list(self.symbolic_params.keys())
                + [sm.symbols("I")]
            )
            hamiltonian_list, _ = self._sym_subsystem_hamiltonian_and_interactions(
                full_hamiltonian,
                [self.dynamic_var_indices],
                non_operator_symbols,
            )
            return hamiltonian_list[0]

    def update(self, calculate_bare_esys: bool = True):
        """Sync subsystem parameters with the current instance.

        Parameters
        ----------
        calculate_bare_esys:
            if ``True``, also recompute the bare eigensystems via
            :meth:`_update_bare_esys` after syncing.
        """
        if not self.hierarchical_diagonalization:
            return None
        self._frozen = False
        if self._out_of_sync:
            self._sync_parameters_with_subsystems()
            self._set_sync_status_to_True()
        if calculate_bare_esys:
            self._update_bare_esys()
        self._frozen = True

    def _perform_internal_updates(
        self,
        fetch_hamiltonian: bool = True,
    ):
        """Refresh symbolic expressions, operators, and subsystems after a change.

        Parameters
        ----------
        fetch_hamiltonian:
            if ``True`` (default), re-fetch the symbolic Hamiltonian from the
            parent before regenerating downstream quantities.
        """
        self._frozen = False
        # Regenerate the symbolic hamiltonians from the Circuit module
        if fetch_hamiltonian:
            self.hamiltonian_symbolic = self._fetch_symbolic_hamiltonian()
        self.potential_symbolic = self._generate_sym_potential()

        if self.is_child:
            self._find_and_set_sym_attrs()

        self._generate_hamiltonian_sym_for_numerics()
        # copy the transformation matrix and normal_mode_freqs if self is a Circuit instance.
        if self.is_purely_harmonic and self.ext_basis == "harmonic":
            if not self.is_child:
                self.transformation_matrix = self.symbolic_circuit.transformation_matrix
            self._diagonalize_purely_harmonic_hamiltonian()

        self.operators_by_name = self._set_operators()
        if self.hierarchical_diagonalization:
            # regenerate subsystem hamiltonians
            self._generate_subsystems(only_update_subsystems=True)
            self._update_interactions()
            # keep track of regenerating all subsystems
            self.affected_subsystem_indices = list(range(len(self.subsystems)))
            # making internal updates in all subsystems
            for subsys in self.subsystems:
                subsys._perform_internal_updates(fetch_hamiltonian=False)
        self._frozen = True

    def _update_bare_esys(self):
        """Recompute bare eigensystems for affected subsystems and reset flags."""
        if not self.hierarchical_diagonalization:
            raise RuntimeError(
                "Hierarchical diagonalization is not used in the current instance of Subsystem/Circuit."
            )
        _ = self.hilbert_space.generate_bare_esys(
            update_subsystem_indices=self.affected_subsystem_indices
        )
        for subsys in self.subsystems:
            if subsys.hierarchical_diagonalization:
                subsys._update_bare_esys()
        self._out_of_sync = False
        self.hilbert_space._out_of_sync = False
        self.affected_subsystem_indices = []

    def _is_internal_update_required(self, param_name: str) -> bool:
        """Check whether changing ``param_name`` requires an internal rebuild.

        Parameters
        ----------
        param_name:
            name of the symbolic parameter that has been changed.
        """
        # this update is only necessary when Circuit instance is created with circuit graph, i.e. with SymbolicCircuit
        is_circuit = hasattr(self, "symbolic_circuit")
        if not is_circuit:
            return False
        num_nodes_threshold = (
            len(self.symbolic_circuit.nodes)
        ) >= settings.SYM_INVERSION_MAX_NODES
        frozen_vars = len(self.var_categories["frozen"]) > 0
        # check to see if it is a junction symbolic param
        if (
            not self.is_purely_harmonic
            and sm.symbols(param_name) in self.junction_potential.free_symbols
        ):
            return False

        if num_nodes_threshold or frozen_vars or self.is_purely_harmonic:
            return True
        return False

    def _mark_all_subsystems_as_affected(self):
        """Method to mark all subsystems as affected."""
        if not self.hierarchical_diagonalization:
            return None
        self.affected_subsystem_indices = list(range(len(self.subsystems)))
        for subsys in self.subsystems:
            subsys._mark_all_subsystems_as_affected()

    def _propagate_param_to_affected_subsystems(
        self, param_name: str, value: Any
    ) -> None:
        """Forward ``param_name = value`` to every subsystem that owns the attribute."""
        if not self.hierarchical_diagonalization:
            return
        for subsys_idx, subsys in enumerate(self.subsystems):
            if hasattr(subsys, param_name):
                self._store_updated_subsystem_index(subsys_idx)
                setattr(subsys, param_name, value)

    def _set_property_and_update_param_vars(
        self, param_name: str, value: float
    ) -> None:
        """Setter method to set parameter variables which are instance properties.

        Parameters
        ----------
        param_name:
            Name of the symbol which is updated
        value:
            The value to which the instance property is updated.
        """
        # update the attribute for the current instance
        # first check if the input value is valid.
        if not (np.isrealobj(value)):
            raise AttributeError(
                f"'{value}' is invalid. Branch parameters must be real."
            )
        setattr(self, f"_{param_name}", value)

        _user_changed_parameter = False
        if self._is_internal_update_required(param_name):
            self.symbolic_circuit.update_param_init_val(param_name, value)
            _user_changed_parameter = True
            self._perform_internal_updates()

        if self.ext_basis == "harmonic":
            # set the oscillator parameters, for the extended variables (taking the coefficient of Q^2 and theta^2)
            self._set_harmonic_basis_osc_params()

        if (
            self.hierarchical_diagonalization
            and isinstance(self, circuit.Circuit)
            and _user_changed_parameter
        ):
            self._mark_all_subsystems_as_affected()
        self._propagate_param_to_affected_subsystems(param_name, value)

    def _set_property_and_update_ext_flux_or_charge(
        self, param_name: str, value: float
    ) -> None:
        """Setter for external flux or offset charge instance properties.

        Parameters
        ----------
        param_name:
            Name of the symbol which is updated
        value:
            The value to which the instance property is updated.
        """
        # first check if the input value is valid.
        if not np.isrealobj(value):
            raise AttributeError(
                f"'{value}' is invalid. External flux and offset charges must be real valued."
            )

        # update the attribute for the current instance
        setattr(self, f"_{param_name}", value)

        if self.is_purely_harmonic and self.ext_basis == "harmonic":
            self._set_operators()

        self._propagate_param_to_affected_subsystems(param_name, value)

    def _set_property_and_update_cutoffs(self, param_name: str, value: int) -> None:
        """Setter method to set cutoffs which are instance properties.

        Parameters
        ----------
        param_name:
            Name of the symbol which is updated
        value:
            The value to which the instance property is updated.
        """
        if not (isinstance(value, int) and value > 0):
            raise AttributeError(
                f"{value} is invalid. Basis cutoffs can only be positive integers."
            )

        setattr(self, f"_{param_name}", value)

        self._propagate_param_to_affected_subsystems(param_name, value)

    def _make_property(
        self,
        attrib_name: str,
        init_val: int | float,
        property_update_type: str,
        use_central_dispatch: bool = True,
    ) -> None:
        """Create a class-level property with a name- and update-type-aware setter.

        Parameters
        ----------
        attrib_name:
            Name of the property that needs to be created.
        init_val:
            The value to which the property is initialized.
        property_update_type:
            The string which sets the kind of setter used for this instance
            property.
        use_central_dispatch:
            if ``True`` (default), wrap the property as a
            :class:`~scqubits.core.descriptors.WatchedProperty` so changes are
            broadcast through :mod:`central_dispatch`; otherwise use a plain
            :class:`property`.
        """
        setattr(self, f"_{attrib_name}", init_val)

        def getter(obj, name=attrib_name):
            return getattr(obj, f"_{name}")

        setter_method_name = _PROPERTY_SETTER_BY_TYPE[property_update_type]

        def setter(obj, value, name=attrib_name, method=setter_method_name):
            with _dispatch_suspended():
                getattr(obj, method)(name, value)

        if use_central_dispatch:
            setattr(
                self.__class__,
                attrib_name,
                descriptors.WatchedProperty(
                    float,
                    "CIRCUIT_UPDATE",
                    fget=getter,
                    fset=setter,
                    attr_name=attrib_name,
                ),
            )
        else:
            setattr(self.__class__, attrib_name, property(fget=getter, fset=setter))

    ##############################################
    ##############################################

    def set_discretized_phi_range(
        self, var_indices: tuple[int], phi_range: tuple[float]
    ) -> None:
        """Set the flux range for discretized phi basis or for plotting.

        Parameters
        ----------
        var_indices:
            list of var_indices whose range needs to be changed
        phi_range:
            The desired range for each of the discretized phi variables
        """
        if self.hierarchical_diagonalization:
            for var_index in var_indices:
                subsys_index = self.get_subsystem_index(var_index)
                self.subsystems[subsys_index].set_discretized_phi_range(
                    (var_index,), phi_range
                )
                self._store_updated_subsystem_index(subsys_index)

        for var_index in var_indices:
            if var_index not in self.var_categories["extended"]:
                raise ValueError(
                    f"Variable with index {var_index} is not an extended variable."
                )
            self.discretized_phi_range[var_index] = phi_range
        self.operators_by_name = self._set_operators()

    def set_and_return(self, attr_name: str, value: Any) -> "CircuitRoutines":
        """Set an attribute and return ``self`` to enable fluent chaining.

        Useful for doing something like example::

            qubit.set_and_return('flux', 0.23).some_method()

        instead of example::

            qubit.flux=0.23
            qubit.some_method()

        Parameters
        ----------
        attr_name:
            name of class attribute in string form
        value:
            value that the attribute is to be set to

        Returns
        -------
        self
        """
        setattr(self, attr_name, value)
        return self

    def get_ext_basis(self) -> str | list[str]:
        """Return the ext_basis for this Circuit, descending into subsystems if any."""
        if not self.hierarchical_diagonalization:
            return self.ext_basis
        else:
            ext_basis: list[str | list[str]] = []
            for subsys in self.subsystems:
                ext_basis.append(subsys.get_ext_basis())
            return ext_basis  # type: ignore[return-value]

    # *****************************************************************
    # **** Functions to construct the operators for the Hamiltonian ****
    # *****************************************************************
    def discretized_grids_dict_for_vars(self):
        """Build a ``{var_index: Grid1d}`` mapping for extended/periodic variables."""
        cutoffs_dict = self.cutoffs_dict()
        grids = {}
        for i in self.var_categories["extended"]:
            grids[i] = discretization.Grid1d(
                self.discretized_phi_range[i][0],
                self.discretized_phi_range[i][1],
                cutoffs_dict[i],
            )
        for i in self.var_categories["periodic"]:
            grids[i] = discretization.Grid1d(
                -np.pi, np.pi, self._default_grid_phi.pt_count
            )
        return grids

    def _check_truncation_indices(self):
        """Validate that subsystem truncation indices fit within their Hilbert dim."""
        if not self.hierarchical_diagonalization:
            return

        for subsystem_idx, subsystem in enumerate(self.subsystems):
            if subsystem.truncated_dim >= subsystem.hilbertdim() - 1:
                # find the correct position of the subsystem where the truncation
                # index  is too big
                subsystem_position = f"subsystem {subsystem_idx} "
                parent = subsystem.parent
                while parent.is_child:
                    grandparent = parent.parent
                    # find the subsystem position of the parent system
                    subsystem_position += f"of subsystem {grandparent.get_subsystem_index(parent.dynamic_var_indices[0])} "
                    parent = grandparent
                raise ValueError(
                    f"The truncation index for {subsystem_position} exceeds the maximum"
                    f" size of {subsystem.hilbertdim() - 1}."
                )
            elif not (
                isinstance(subsystem.truncated_dim, int)
                and (subsystem.truncated_dim > 0)
            ):
                raise ValueError(
                    "Invalid value encountered in subsystem_trunc_dims. "
                    "Truncated dimension must be a positive integer."
                )

    def _generate_subsystems(
        self,
        only_update_subsystems: bool = False,
        subsys_dict: dict[str, Any] | None = None,
    ):
        """Generate (or refresh) child subsystems following :attr:`system_hierarchy`.

        Parameters
        ----------
        only_update_subsystems:
            if ``True``, update existing subsystems in place instead of
            constructing new ones.
        subsys_dict:
            optional pre-computed ``{"systems_sym": ..., "interaction_sym": ...}``
            decomposition; if ``None``, it is recomputed from the current
            symbolic Hamiltonian.
        """
        hamiltonian = self.hamiltonian_symbolic
        systems_sym, interaction_sym = self._get_systems_and_interactions(
            hamiltonian, subsys_dict
        )

        if only_update_subsystems:
            self._update_existing_subsystems(systems_sym, interaction_sym)
        else:
            self._create_new_subsystems(systems_sym, interaction_sym)

    def _get_systems_and_interactions(
        self, hamiltonian: sm.Expr, subsys_dict: dict[str, Any] | None
    ) -> tuple[list[sm.Expr], list[sm.Expr]]:
        """Split ``hamiltonian`` into per-subsystem and interaction symbolic terms.

        Parameters
        ----------
        hamiltonian:
            full symbolic Hamiltonian to decompose.
        subsys_dict:
            optional pre-computed decomposition; if provided, returned as-is.
        """
        non_operator_symbols = (
            self.offset_charges
            + self.free_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
            + [sm.symbols("I")]
        )
        if subsys_dict:
            return subsys_dict["systems_sym"], subsys_dict["interaction_sym"]
        return self._sym_subsystem_hamiltonian_and_interactions(
            hamiltonian, self.system_hierarchy, non_operator_symbols
        )

    def _update_existing_subsystems(
        self, systems_sym: list[sm.Expr], interaction_sym: list[sm.Expr]
    ):
        """Refresh the symbolic Hamiltonians of existing child subsystems.

        Parameters
        ----------
        systems_sym:
            list of per-subsystem symbolic Hamiltonians, one per entry of
            :attr:`system_hierarchy`.
        interaction_sym:
            list of symbolic interaction terms, one per entry of
            :attr:`system_hierarchy`.
        """
        for subsys_index, subsys in enumerate(self.subsystems):
            subsys.hamiltonian_symbolic = systems_sym[subsys_index]
            subsys._frozen = False
            subsys._find_and_set_sym_attrs()
            self.subsystem_interactions[subsys_index] = interaction_sym[subsys_index]

    def _create_new_subsystems(
        self, systems_sym: list[sm.Expr], interaction_sym: list[sm.Expr]
    ):
        """Construct fresh :class:`Subsystem` instances and a :class:`HilbertSpace`.

        Parameters
        ----------
        systems_sym:
            list of per-subsystem symbolic Hamiltonians, one per entry of
            :attr:`system_hierarchy`.
        interaction_sym:
            list of symbolic interaction terms, one per entry of
            :attr:`system_hierarchy`.
        """
        self.subsystem_hamiltonians = dict(
            zip(range(len(self.system_hierarchy)), systems_sym)
        )
        self.subsystem_interactions = dict(
            zip(range(len(self.system_hierarchy)), interaction_sym)
        )
        self.subsystems = []
        for index in range(len(self.system_hierarchy)):
            is_purely_harmonic = self._is_expression_purely_harmonic(systems_sym[index])
            if isinstance(self.ext_basis, list):
                ext_basis = self.ext_basis[index]
            else:
                ext_basis = "harmonic" if is_purely_harmonic else self.ext_basis
            self.subsystems.append(
                circuit.Subsystem(
                    self,  # type: ignore[arg-type]
                    systems_sym[index],
                    system_hierarchy=self.system_hierarchy[index],
                    truncated_dim=(
                        self.subsystem_trunc_dims[index][0]
                        if isinstance(self.subsystem_trunc_dims[index], list)
                        else self.subsystem_trunc_dims[index]
                    ),
                    ext_basis=ext_basis,
                    subsystem_trunc_dims=(
                        self.subsystem_trunc_dims[index][1]
                        if isinstance(self.subsystem_trunc_dims[index], list)
                        else None
                    ),
                    evals_method=(
                        self.evals_method if not is_purely_harmonic else None
                    ),
                    evals_method_options=(
                        self.evals_method_options if not is_purely_harmonic else None
                    ),
                    esys_method=(self.esys_method if not is_purely_harmonic else None),
                    esys_method_options=(
                        self.esys_method_options if not is_purely_harmonic else None
                    ),
                )
            )
        self.hilbert_space = HilbertSpace(self.subsystems)

    def get_subsystem_index(self, var_index: int) -> int:
        """Return the subsystem index that owns the given variable index.

        Parameters
        ----------
        var_index:
            variable index in integer starting from 1.

        Returns
        -------
        subsystem index which can be used to identify the subsystem index in the
        list self.subsystems.
        """
        for index, system_hierarchy in enumerate(self.system_hierarchy):
            if var_index in flatten_list_recursive(system_hierarchy):
                return index
        raise ValueError(
            f"The var_index={var_index} could not be identified with any subsystem."
        )

    def _update_interactions(self, recursive: bool = False) -> None:
        """Rebuild HilbertSpace interactions when hierarchical diagonalization is on.

        Parameters
        ----------
        recursive:
            if ``True``, also recurse into subsystems that themselves use
            hierarchical diagonalization.
        """
        self.hilbert_space.interaction_list = []

        # Adding interactions using the symbolic interaction term
        for sys_index in range(len(self.system_hierarchy)):
            interaction = self.subsystem_interactions[sys_index].expand()
            if interaction == 0:  # if the interaction term is zero
                continue

            interaction = interaction.subs("I", 1)
            # adding a factor of 2pi for external flux
            for sym_param in interaction.free_symbols:
                if sym_param in self.external_fluxes:
                    interaction = interaction.subs(sym_param, 2 * np.pi * sym_param)

            expr_dict = interaction.as_coefficients_dict()
            interaction_terms = list(expr_dict.keys())

            for idx, term in enumerate(interaction_terms):
                coefficient_sympy = expr_dict[term]

                branch_sym_params = [
                    symbol
                    for symbol in term.free_symbols
                    if symbol in list(self.symbolic_params.keys())
                ]
                operator_expr, param_expr = term.as_independent(
                    *branch_sym_params, as_Mul=True
                )

                param_expr = coefficient_sympy * param_expr
                for param in list(self.symbolic_params.keys()):
                    param_expr = param_expr.subs(
                        param, sm.symbols("self." + param.name)
                    )
                param_expr_str = str(param_expr)
                self.hilbert_space.add_interaction(
                    expr=param_expr_str + "*operator_expr",
                    const={"self": self},
                    op1=(
                        "operator_expr",
                        self._operator_from_sym_expr_wrapper(operator_expr),
                    ),
                    check_validity=False,
                )
        if recursive:
            for subsys in self.subsystems:
                if subsys.hierarchical_diagonalization:
                    subsys._update_interactions(recursive=recursive)

    def _operator_from_sym_expr_wrapper(self, sym_expr: sm.Expr):
        """Return a closure that evaluates ``sym_expr`` to a numerical operator.

        Parameters
        ----------
        sym_expr:
            sympy expression to be evaluated when the closure is called.
        """

        def wrapper_func(self=self, sym_expr=sym_expr, bare_esys=None):
            # The bare esys here is a dict of esys for each of the subsystem present under hilbert_space
            return self._evaluate_symbolic_expr(sym_expr, bare_esys=bare_esys)

        return wrapper_func

    def _set_vars(self):
        """Set :attr:`vars` to map operator types to their sympy ``Symbol`` lists."""
        if not self.hierarchical_diagonalization:
            return self._set_vars_no_hd()
        vars = {"periodic": {}, "extended": {}, "identity": [sm.symbols("I")]}
        for subsys in self.subsystems:
            subsys._set_vars()
            for var_type in ["periodic", "extended"]:
                for operator_type in subsys.vars[var_type]:
                    if operator_type not in vars[var_type]:
                        vars[var_type][operator_type] = subsys.vars[var_type][
                            operator_type
                        ]
                    else:
                        vars[var_type][operator_type] = unique_elements_in_list(
                            vars[var_type][operator_type]
                            + subsys.vars[var_type][operator_type]
                        )
        self.vars = vars

    def _set_vars_no_hd(self):
        """Populate :attr:`vars` for the no-hierarchical-diagonalization case."""
        # Defining the list of variables for periodic operators
        periodic_symbols_sin = _generate_symbols_list(
            "sinθ", self.var_categories["periodic"]
        )

        periodic_symbols_cos = _generate_symbols_list(
            "cosθ", self.var_categories["periodic"]
        )
        periodic_symbols_n = _generate_symbols_list(
            "n", self.var_categories["periodic"]
        )

        # Defining the list of discretized_ext variables
        y_symbols = _generate_symbols_list("θ", self.var_categories["extended"])
        p_symbols = _generate_symbols_list("Q", self.var_categories["extended"])

        if self.ext_basis == "discretized":
            ps_symbols = [
                sm.symbols("Qs" + str(i)) for i in self.var_categories["extended"]
            ]
            sin_symbols = [
                sm.symbols(f"sinθ{i}") for i in self.var_categories["extended"]
            ]
            cos_symbols = [
                sm.symbols(f"cosθ{i}") for i in self.var_categories["extended"]
            ]

        elif self.ext_basis == "harmonic":
            a_symbols = [sm.symbols(f"a{i}") for i in self.var_categories["extended"]]
            ad_symbols = [sm.symbols(f"ad{i}") for i in self.var_categories["extended"]]
            Nh_symbols = [sm.symbols(f"Nh{i}") for i in self.var_categories["extended"]]
            pos_symbols = [sm.symbols(f"θ{i}") for i in self.var_categories["extended"]]
            sin_symbols = [
                sm.symbols(f"sinθ{i}") for i in self.var_categories["extended"]
            ]
            cos_symbols = [
                sm.symbols(f"cosθ{i}") for i in self.var_categories["extended"]
            ]
            momentum_symbols = [
                sm.symbols(f"Q{i}") for i in self.var_categories["extended"]
            ]

        # setting the attribute self.vars
        self.vars: dict[str, Any] = {
            "periodic": {
                "sin": periodic_symbols_sin,
                "cos": periodic_symbols_cos,
                "number": periodic_symbols_n,
            },
            "identity": [sm.symbols("I")],
        }

        if self.ext_basis == "discretized":
            self.vars["extended"] = {
                "position": y_symbols,
                "momentum": p_symbols,
                "momentum_squared": ps_symbols,
                "sin": sin_symbols,
                "cos": cos_symbols,
            }
        elif self.ext_basis == "harmonic":
            self.vars["extended"] = {
                "annihilation": a_symbols,
                "creation": ad_symbols,
                "number": Nh_symbols,
                "position": pos_symbols,
                "momentum": momentum_symbols,
                "sin": sin_symbols,
                "cos": cos_symbols,
            }

    # #################################################################
    # ############## Functions to construct the operators #############
    # #################################################################
    def get_cutoffs(self) -> dict[str, list]:
        """Method to get the cutoffs for each of the circuit's degree of freedom."""
        cutoffs_dict: dict[str, list[Any]] = {
            "cutoff_n": [],
            "cutoff_ext": [],
        }

        for cutoff_type in cutoffs_dict:
            attr_list = [x for x in self.cutoff_names if cutoff_type in x]

            if attr_list:
                attr_list.sort()
                cutoffs_dict[cutoff_type] = [getattr(self, attr) for attr in attr_list]

        return cutoffs_dict

    def _collect_cutoff_values(self):
        """Yield per-degree-of-freedom Hilbert-space dimensions for this instance."""
        if not self.hierarchical_diagonalization:
            cutoff_dict = self.get_cutoffs()
            for cutoff_name in cutoff_dict:
                for cutoff in cutoff_dict[cutoff_name]:
                    if "cutoff_n" in cutoff_name:
                        yield 2 * cutoff + 1
                    elif "cutoff_ext" in cutoff_name:
                        yield cutoff
        else:
            for idx, _ in enumerate(self.system_hierarchy):
                if isinstance(self.subsystem_trunc_dims[idx], list):
                    yield self.subsystem_trunc_dims[idx][0]
                else:
                    yield self.subsystem_trunc_dims[idx]

    def hilbertdim(self):
        """Return the Hilbert dimension of the Circuit instance."""
        cutoff_values = np.fromiter(self._collect_cutoff_values(), dtype=int)
        return np.prod(cutoff_values)

    # helper functions
    def _kron_operator(
        self, operator: csc_matrix | ndarray, var_index: int
    ) -> csc_matrix | ndarray:
        """Identity-wrap ``operator`` along the other variable indices of the subsystem.

        Parameters
        ----------
        operator:
            The operator belonging to the variable index ``var_index``.
        var_index:
            Variable index to which the operator belongs.

        Returns
        -------
        Operator identity-wrapped for the current subsystem.
        """
        dynamic_var_indices = self.dynamic_var_indices.copy()
        var_index_pos = dynamic_var_indices.index(var_index)

        cutoffs_dict = self.cutoffs_dict()
        for var_idx in cutoffs_dict:
            if var_idx in self.var_categories["periodic"]:
                cutoffs_dict[var_idx] = 2 * cutoffs_dict[var_idx] + 1

        var_dim_list = [cutoffs_dict[var_idx] for var_idx in dynamic_var_indices]

        if self.type_of_matrices == "dense":
            matrix_format = "array"
        elif self.type_of_matrices == "sparse":
            matrix_format = "csc"

        if len(dynamic_var_indices) > 1:
            if var_index_pos > 0:
                identity_left = sparse.identity(  # type: ignore[call-overload]
                    int(np.prod(var_dim_list[:var_index_pos])),
                    format=matrix_format,
                )
            if var_index_pos < len(dynamic_var_indices) - 1:
                identity_right = sparse.identity(  # type: ignore[call-overload]
                    int(np.prod(var_dim_list[var_index_pos + 1 :])),
                    format=matrix_format,
                )

            if var_index == dynamic_var_indices[0]:
                return sparse.kron(operator, identity_right, format=matrix_format)  # type: ignore[call-overload]
            elif var_index == dynamic_var_indices[-1]:
                return sparse.kron(identity_left, operator, format=matrix_format)  # type: ignore[call-overload]
            else:
                return sparse.kron(
                    sparse.kron(identity_left, operator, format=matrix_format),  # type: ignore[call-overload]
                    identity_right,
                    format=matrix_format,
                )
        else:
            return self._sparsity_adaptive(operator)

    def _sparsity_adaptive(self, matrix: csc_matrix | ndarray) -> csc_matrix | ndarray:
        """Changes the type of matrix depending on the attribute :attr:`type_of_matrices`.

        Parameters
        ----------
        matrix:
            The operator or matrix whose type needs to be changed

        Returns
        -------
        Returns the matrix in sparse or dense version depending on the type of
        matrices used.
        """
        #  all of this can be simplified.
        if sparse.issparse(matrix):
            if self.type_of_matrices == "sparse":
                return matrix
            return matrix.toarray()

        if self.type_of_matrices == "sparse":
            return sparse.csc_matrix(matrix)
        return matrix

    def _identity_qobj(self):
        """Return the Qobj of the identity matrix of the right dimensions."""
        if not self.hierarchical_diagonalization:
            return qt.identity(self.hilbertdim())

        subsys_trunc_dims = [subsys.truncated_dim for subsys in self.subsystems]

        return qt.tensor([qt.identity(truncdim) for truncdim in subsys_trunc_dims])

    def _identity(self):
        """Return the Identity operator for the entire Hilbert space of the circuit."""
        if (
            hasattr(self, "hierarchical_diagonalization")
            and self.hierarchical_diagonalization
        ):
            return self._identity_qobj()
        dim = self.hilbertdim()
        if self.type_of_matrices == "sparse":
            op = sparse.identity(dim, format="csc")
            return op
        elif self.type_of_matrices == "dense":
            return np.identity(dim)

    def exp_i_operator(
        self, var_sym: sm.Symbol, prefactor: float
    ) -> csc_matrix | ndarray:
        r"""Return the bare operator :math:`\exp(i\,\theta\,\text{prefactor})`.

        The result is *not* identity-wrapped (no Kronecker product applied).
        Requires the oscillator lengths to be set in :attr:`osc_lengths` when
        ``ext_basis`` is set to ``"harmonic"``.

        Parameters
        ----------
        var_sym:
            sympy symbol identifying the variable (e.g. ``θ1`` or ``Q1``).
        prefactor:
            real prefactor multiplying the variable inside the exponential.
        """
        var_index = get_trailing_number(var_sym.name)
        var_basis = self._basis_for_var_index(var_index)

        if var_basis == "periodic":
            # Periodic-variable invariant: prefactor must be an integer (the
            # diagonal offset of the resulting dia_matrix). The commented-out
            # check below documents the historical assertion.
            # if abs(prefactor) != 1:
            #     raise Exception("Prefactor for periodic variable should be 1.")
            # if prefactor > 0:
            exp_i_theta = _exp_i_theta_operator(
                self.cutoffs_dict()[var_index], int(prefactor)
            )
        elif var_basis == "discretized":
            phi_grid = discretization.Grid1d(
                self.discretized_phi_range[var_index][0],
                self.discretized_phi_range[var_index][1],
                self.cutoffs_dict()[var_index],
            )
            if "θ" in var_sym.name:
                diagonal = np.exp(phi_grid.make_linspace() * prefactor * 1j)
                exp_i_theta = sparse.dia_matrix(
                    (diagonal, [0]), shape=(phi_grid.pt_count, phi_grid.pt_count)
                ).tocsc()
            elif "Q" in var_sym.name:
                exp_i_theta = sp.linalg.expm(  # type: ignore[assignment]
                    _i_d_dphi_operator(phi_grid).toarray() * prefactor * 1j
                )
        elif var_basis == "harmonic":
            osc_length = self.get_osc_param(var_index, which_param="length")
            if "θ" in var_sym.name:
                exp_argument_op = op.a_plus_adag_sparse(
                    self.cutoffs_dict()[var_index],
                    prefactor=(osc_length / 2**0.5),
                )
            elif "Q" in var_sym.name:
                exp_argument_op = op.iadag_minus_ia_sparse(
                    self.cutoffs_dict()[var_index],
                    prefactor=(osc_length * 2**0.5) ** -1,
                )
            exp_i_theta = sparse.linalg.expm(exp_argument_op * prefactor * 1j)

        return self._sparsity_adaptive(exp_i_theta)

    def _evaluate_matrix_sawtooth_terms(
        self, saw_expr: sm.Expr, bare_esys: dict[int, tuple] | None = None
    ) -> qt.Qobj:
        """Evaluate symbolic sawtooth-potential terms to a :class:`qutip.Qobj`.

        Parameters
        ----------
        saw_expr:
            symbolic expression containing :func:`sawtooth_potential` terms.
        bare_esys:
            optional precomputed dict of bare eigensystems for subsystems.
        """
        if self.hierarchical_diagonalization:
            subsystem_list = self.subsystems
            identity = qt.tensor(
                [qt.identity(subsystem.truncated_dim) for subsystem in subsystem_list]
            )
        else:
            identity = qt.identity(self.hilbertdim())

        saw_potential_matrix = identity * 0

        saw = sm.Function("saw", real=True)
        for saw_term in saw_expr.as_ordered_terms():
            coefficient = float(list(saw_expr.as_coefficients_dict().values())[0])
            saw_argument_expr = [
                arg.args[0] for arg in (1.0 * saw_term).args if (arg.has(saw))
            ][0]

            saw_argument_operator = self._evaluate_symbolic_expr(
                saw_argument_expr, bare_esys
            )

            # since this operator only works for discretized phi basis

            diagonal_elements = sawtooth_potential(saw_argument_operator.diag())
            saw_potential_matrix += coefficient * qt.qdiags(
                diagonal_elements, 0, dims=saw_potential_matrix.dims
            )

        return saw_potential_matrix

    def _evaluate_matrix_cosine_terms(
        self, junction_potential: sm.Expr, bare_esys: dict[int, tuple] | None = None
    ) -> qt.Qobj:
        """Evaluate symbolic Josephson cosine/sine terms to a :class:`qutip.Qobj`.

        Parameters
        ----------
        junction_potential:
            symbolic expression containing ``cos`` and/or ``sin`` terms.
        bare_esys:
            optional precomputed dict of bare eigensystems for subsystems.
        """
        if self.hierarchical_diagonalization:
            subsystem_list = self.subsystems
            identity = qt.tensor(
                [qt.identity(subsystem.truncated_dim) for subsystem in subsystem_list]
            )
        else:
            identity = qt.identity(self.hilbertdim())

        junction_potential_matrix = identity * 0

        if (
            isinstance(junction_potential, (int, float))
            or len(junction_potential.free_symbols) == 0
        ):
            return junction_potential_matrix

        for cos_term in junction_potential.as_ordered_terms():
            coefficient = float(list(cos_term.as_coefficients_dict().values())[0])
            cos_argument_expr = [
                arg.args[0]
                for arg in (1.0 * cos_term).args
                if (arg.has(sm.cos) or arg.has(sm.sin))
            ][0]

            var_indices = [
                get_trailing_number(var_symbol.name)
                for var_symbol in cos_argument_expr.free_symbols
            ]

            # removing any constant terms
            for term in cos_argument_expr.as_ordered_terms():
                if not term.free_symbols:
                    cos_argument_expr -= term
                    coefficient *= np.exp(float(term) * 1j)

            operator_list = []
            for idx, var_symbol in enumerate(cos_argument_expr.free_symbols):
                prefactor = float(cos_argument_expr.coeff(var_symbol))
                child_circuit = self.return_root_child(var_indices[idx])
                operator_bare = child_circuit._kron_operator(
                    self.exp_i_operator(var_symbol, prefactor), var_indices[idx]
                )
                operator_list.append(
                    self.identity_wrap_for_hd(
                        operator_bare,
                        child_circuit,
                        bare_esys=bare_esys,
                    )
                )
            cos_term_operator = coefficient * functools.reduce(
                builtin_op.mul,
                operator_list,
            )
            if any([arg.has(sm.cos) for arg in (1.0 * cos_term).args]):
                junction_potential_matrix += (
                    cos_term_operator + cos_term_operator.dag()
                ) * 0.5
            elif any([arg.has(sm.sin) for arg in (1.0 * cos_term).args]):
                junction_potential_matrix += (
                    (cos_term_operator - cos_term_operator.dag()) * 0.5 * (-1j)
                )
        return junction_potential_matrix

    def _set_harmonic_basis_osc_params(self, hamiltonian: sm.Expr | None = None):
        """Compute oscillator lengths and frequencies for the extended variables.

        Parameters
        ----------
        hamiltonian:
            optional symbolic Hamiltonian to extract :math:`E_C` and
            :math:`E_L` coefficients from. If provided, the computed
            ``(osc_lengths, osc_freqs)`` are returned without mutating
            :attr:`osc_lengths`/:attr:`osc_freqs`. If ``None``, the cached
            ``_hamiltonian_sym_for_numerics`` is used and the instance
            attributes are updated in place.
        """
        osc_lengths = {}
        osc_freqs = {}
        hamiltonian_sym = hamiltonian or self._hamiltonian_sym_for_numerics
        # substitute all the parameter values
        hamiltonian_sym = hamiltonian_sym.subs(
            [
                (param, getattr(self, str(param)))
                for param in list(self.symbolic_params.keys())
                + self.external_fluxes
                + self.offset_charges
                + self.free_charges
            ]
        )
        for list_idx, var_index in enumerate(self.var_categories["extended"]):
            ECi = float(hamiltonian_sym.coeff(f"Q{var_index}**2").cancel()) / 4
            ELi = float(hamiltonian_sym.coeff(f"θ{var_index}**2").cancel()) * 2
            osc_freqs[var_index] = (8 * ELi * ECi) ** 0.5
            osc_lengths[var_index] = (8.0 * ECi / ELi) ** 0.25
        if hamiltonian is not None:
            return osc_lengths, osc_freqs
        self.osc_lengths = osc_lengths
        self.osc_freqs = osc_freqs

    def _purely_harmonic_operator_func_factory(self, operator_name: str):
        """Build the operator method ``<operator_name>_operator`` for a harmonic system.

        Parameters
        ----------
        operator_name:
            name of the symbolic operator (e.g. ``"Q1"``, ``"θ2"``, ``"a1"``,
            ``"ad1"`` or ``"Nh1"``) to produce a method for.
        """

        def purely_harmonic_operator_func(
            self: "Subsystem", energy_esys: bool | tuple[ndarray, ndarray] = False
        ):
            """Returns the operator <op_name> (corresponds to the name of the method
            "<op_name>_operator") for the Circuit/Subsystem instance.

            Parameters
            ----------
            energy_esys:
                If `False` (default), returns charge operator n in the charge basis.
                If `True`, energy eigenspectrum is computed, returns charge operator n in the energy eigenbasis.
                If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
                eigenvectors), returns charge operator n in the energy eigenbasis, and does not have to recalculate the
                eigenspectrum.

            Returns
            -------
                Returns the operator <op_name>(corresponds to the name of the method "<op_name>_operator").
                For `energy_esys=True`, n has dimensions of :attr:`truncated_dim` x :attr:`truncated_dim`.
                If an actual eigensystem is handed to `energy_sys`, then `n` has dimensions of m x m,
                where m is the number of given eigenvectors.
            """
            var_index = get_trailing_number(operator_name)
            Q_new, θ_new = self._transform_hamiltonian(
                hamiltonian=self.hamiltonian_symbolic,
                transformation_matrix=self.osc_eigvecs,
                return_transformed_exprs=True,
            )
            main_op_type = [
                optypename
                for optypename in ["Q", "θ", "ad", "a", "Nh"]
                if optypename in operator_name
            ][0]
            op_exprs = dict(
                zip(
                    ["Q", "θ"],
                    [
                        Q_new[self.var_categories["extended"].index(var_index)],
                        θ_new[self.var_categories["extended"].index(var_index)],
                    ],
                )
            )
            ops: dict[str, Any] = dict.fromkeys(["Q", "θ"])
            for optype in ops:
                terms = op_exprs[optype].as_ordered_terms()
                operator: Any = 0
                for term in terms:
                    sym_var_index = get_trailing_number(term.free_symbols.pop().name)
                    term_op: csc_matrix | ndarray
                    if optype == "Q":
                        term_op = op.iadag_minus_ia_sparse(
                            getattr(self, f"cutoff_ext_{sym_var_index}"),
                            prefactor=1 / (self.osc_lengths[sym_var_index] * 2**0.5),
                        ) * float(term.as_coeff_Mul()[0])
                    if optype == "θ":
                        term_op = op.a_plus_adag(
                            getattr(self, f"cutoff_ext_{sym_var_index}"),
                            prefactor=self.osc_lengths[sym_var_index] / 2**0.5,
                        ) * float(term.as_coeff_Mul()[0])
                    operator += self._kron_operator(term_op, sym_var_index)
                if optype == main_op_type:
                    return operator
                ops[optype] = operator
            old_osc_length = self.undiagonalized_osc_params["osc_lengths"][var_index]
            annihilation_operator = (
                1
                / 2**0.5
                * (ops["θ"] / old_osc_length + 1j * ops["Q"] * old_osc_length)
            )
            if main_op_type == "a":
                operator = annihilation_operator
            elif main_op_type == "ad":
                operator = annihilation_operator.T
            elif main_op_type == "Nh":
                operator = annihilation_operator.T * annihilation_operator
            return self.process_op(native_op=operator, energy_esys=energy_esys)

        return purely_harmonic_operator_func

    def _generate_operator_methods(self) -> dict[str, Callable]:
        """Build the dict of operator functions to attach to a :class:`Circuit`."""
        return {
            **self._build_periodic_operator_methods(),
            **self._build_extended_operator_methods(),
            "I_operator": CircuitRoutines._identity,
        }

    def _build_extended_operator_methods(self) -> dict[str, Callable]:
        """Dispatch to the basis-appropriate builder for extended-variable operators."""
        if self.hierarchical_diagonalization:
            return self._extended_operators_hierarchical()
        if self.ext_basis == "discretized":
            return self._extended_operators_discretized()
        if self.ext_basis == "harmonic":
            if self.is_purely_harmonic:
                return self._extended_operators_purely_harmonic()
            return self._extended_operators_harmonic()
        return {}

    def _build_periodic_operator_methods(self) -> dict[str, Callable]:
        """Dispatch for periodic-variable operators."""
        if self.hierarchical_diagonalization:
            return self._periodic_operators_hierarchical()
        return self._periodic_operators_charge_basis()

    def _extended_operators_hierarchical(self) -> dict[str, Callable]:
        """Extended-variable operators when hierarchical diagonalization is active."""
        extended_vars = self.vars["extended"]
        return {
            sym_variable.name
            + "_operator": hierarchical_diagonalization_func_factory(sym_variable.name)
            for var_type in extended_vars
            for sym_variable in extended_vars[var_type]
        }

    def _extended_operators_discretized(self) -> dict[str, Callable]:
        """Extended-variable operators in the discretized-phi basis."""
        nonwrapped_ops: dict[str, Callable[..., Any]] = {
            "position": _phi_operator,
            "cos": _cos_phi,
            "sin": _sin_phi,
            "momentum": _i_d_dphi_operator,
            "momentum_squared": _i_d2_dphi2_operator,
        }
        extended_vars = self.vars["extended"]
        return {
            sym_variable.name
            + "_operator": grid_operator_func_factory(
                op_func, int(get_trailing_number(sym_variable.name))
            )
            for short_op_name, op_func in nonwrapped_ops.items()
            for sym_variable in extended_vars[short_op_name]
        }

    def _extended_operators_harmonic(self) -> dict[str, Callable]:
        """Extended-variable operators in the (non-purely-harmonic) harmonic basis."""
        self._set_harmonic_basis_osc_params()
        nonwrapped_ops: dict[str, Callable[..., Any]] = {
            "creation": op.creation_sparse,
            "annihilation": op.annihilation_sparse,
            "number": op.number_sparse,
            "position": op.a_plus_adag_sparse,
            "sin": op.sin_theta_harmonic,
            "cos": op.cos_theta_harmonic,
            "momentum": op.iadag_minus_ia_sparse,
        }
        extended_vars = self.vars["extended"]
        operators: dict[str, Callable] = {}
        for list_idx, var_index in enumerate(self.var_categories["extended"]):
            for short_op_name, op_func in nonwrapped_ops.items():
                sym_variable = extended_vars[short_op_name][list_idx]
                operators[sym_variable.name + "_operator"] = operator_func_factory(
                    op_func, var_index, op_type=short_op_name
                )
        return operators

    def _extended_operators_purely_harmonic(self) -> dict[str, Callable]:
        """Extended-variable operators when the system is purely harmonic."""
        self._set_harmonic_basis_osc_params()
        extended_vars = self.vars["extended"]
        operators: dict[str, Callable] = {}
        for list_idx, _var_index in enumerate(self.var_categories["extended"]):
            for short_op_name in (
                "position",
                "momentum",
                "number",
                "annihilation",
                "creation",
            ):
                sym_variable = extended_vars[short_op_name][list_idx]
                operators[sym_variable.name + "_operator"] = (
                    self._purely_harmonic_operator_func_factory(sym_variable.name)
                )
        return operators

    @staticmethod
    def _periodic_op_table() -> dict[str, Callable[..., Any]]:
        """Return the (op-name -> bare-operator-function) table for periodic vars."""
        return {
            "sin": _sin_theta,
            "cos": _cos_theta,
            "number": _n_theta_operator,
        }

    def _periodic_operators_hierarchical(self) -> dict[str, Callable]:
        """Periodic-variable operators when hierarchical diagonalization is active."""
        periodic_vars = self.vars["periodic"]
        return {
            sym_variable.name
            + "_operator": hierarchical_diagonalization_func_factory(sym_variable.name)
            for short_op_name in self._periodic_op_table()
            for sym_variable in periodic_vars[short_op_name]
        }

    def _periodic_operators_charge_basis(self) -> dict[str, Callable]:
        """Periodic-variable operators in the charge basis."""
        periodic_vars = self.vars["periodic"]
        return {
            sym_variable.name
            + "_operator": operator_func_factory(
                op_func, get_trailing_number(sym_variable.name)
            )
            for short_op_name, op_func in self._periodic_op_table().items()
            for sym_variable in periodic_vars[short_op_name]
        }

    # #################################################################
    # ############### Functions for parameter queries #################
    # #################################################################
    def offset_free_charge_values(self) -> list[float]:
        """Return the current values of all offset and free charge attributes."""
        return [
            getattr(self, charge_var.name)
            for charge_var in self.offset_charges + self.free_charges
        ]

    def _set_operators(self) -> dict[str, Callable]:
        """Create the operator methods `<name>_operator` for the circuit."""

        if self.hierarchical_diagonalization:
            for subsys in self.subsystems:
                subsys.operators_by_name = subsys._set_operators()

        op_func_by_name = self._generate_operator_methods()
        for op_name, op_func in op_func_by_name.items():
            setattr(self, op_name, MethodType(op_func, self))

        return {func_name: getattr(self, func_name) for func_name in op_func_by_name}

    def is_subsystem(self, instance: "CircuitRoutines") -> bool:
        """Return ``True`` if ``instance`` shares any variable index with ``self``.

        Parameters
        ----------
        instance:
            another Circuit/Subsystem instance to test against ``self``.
        """
        if len(set(self.dynamic_var_indices) & set(instance.dynamic_var_indices)) > 0:
            return True
        return False

    def identity_wrap_for_hd(
        self,
        operator: csc_matrix | ndarray | None,
        child_instance: "CircuitRoutines",
        bare_esys: dict[int, tuple] | None = None,
    ) -> qt.Qobj:
        r"""Return an identity-wrapped operator of size :meth:`hilbertdim`.

        Only converts an operator that belongs to a specific variable index.
        For example, :math:`Q_1` or :math:`\cos(\theta_1)`, but not
        :math:`Q_1 Q_2`.

        Parameters
        ----------
        operator:
            operator in the form of :class:`scipy.sparse.csc_matrix` or
            :class:`numpy.ndarray`.
        child_instance:
            the subsystem to which the operator belongs.
        bare_esys:
            dict containing subsystem indices starting from 0, paired with the
            bare eigensystem for each of the subsystems.

        Returns
        -------
        Identity-wrapped operator as a :class:`qutip.Qobj`.
        """
        if not self.hierarchical_diagonalization:
            return qt.Qobj(operator)

        subsystem_index = [
            subsys_index
            for subsys_index, subsys in enumerate(self.subsystems)
            if subsys.is_subsystem(child_instance)
        ][0]
        subsystem = self.subsystems[subsystem_index]
        operator = subsystem.identity_wrap_for_hd(operator, child_instance)

        if isinstance(operator, qt.Qobj):
            operator = operator.full()

        operator = convert_matrix_to_qobj(
            operator,
            subsystem,
            op_in_eigenbasis=False,
            evecs=(
                bare_esys[subsystem_index][1]
                if bare_esys
                else subsystem.eigensys(evals_count=subsystem.truncated_dim)[1]
            ),
        )
        return identity_wrap(
            operator,
            subsystem,
            self.subsystems,
            evecs=(
                bare_esys[subsystem_index][1]
                if bare_esys
                else subsystem.eigensys(evals_count=subsystem.truncated_dim)[1]
            ),
            op_in_eigenbasis=True,
        )

    @check_sync_status_circuit
    def get_operator_by_name(
        self,
        operator_name: str,
        power: int | None = None,
        bare_esys: dict[int, tuple] | None = None,
    ) -> qt.Qobj:
        """Return the operator named ``operator_name`` for this instance.

        The returned operator has dimension :meth:`hilbertdim`.

        Parameters
        ----------
        operator_name:
            Name of a sympy ``Symbol`` object which should be one of the
            symbols collected in the attribute :attr:`vars`.
        power:
            If requesting an operator raised to a certain power; ``None``
            defaults to 1.
        bare_esys:
            optional precomputed dict of bare eigensystems for nested
            subsystems, keyed by subsystem index.

        Returns
        -------
        Operator identified by ``operator_name``.
        """
        if not self.hierarchical_diagonalization:
            # if the operator_name is a Qsn operator (which is possible when self is a
            # purely harmonic subsystem when using HD) then return the operator
            # constructed using ladder operators
            if re.fullmatch(r"Qs\d+", operator_name) and self.is_purely_harmonic:
                var_index = get_trailing_number(operator_name)
                return self.get_operator_by_name(f"Q{var_index}") ** 2

            return qt.Qobj(getattr(self, operator_name + "_operator")()) ** (
                power if power else 1
            )

        var_index = get_trailing_number(operator_name)
        assert var_index
        subsystem_index = self.get_subsystem_index(var_index)
        subsystem = self.subsystems[subsystem_index]
        subsys_bare_esys = None
        if bare_esys and subsystem.hierarchical_diagonalization:
            subsys_bare_esys = {
                sys_index: (
                    subsystem.hilbert_space["bare_evals"][sys_index][0],
                    subsystem.hilbert_space["bare_evecs"][sys_index][0],
                )
                for sys_index, sys in enumerate(subsystem.hilbert_space.subsystem_list)
            }

        operator = subsystem.get_operator_by_name(
            operator_name, power=power, bare_esys=subsys_bare_esys
        )

        if isinstance(operator, qt.Qobj):
            operator = Qobj_to_scipy_csc_matrix(operator)

        operator = convert_matrix_to_qobj(
            operator,
            subsystem,
            op_in_eigenbasis=False,
            evecs=(
                bare_esys[subsystem_index][1]
                if bare_esys
                else subsystem.eigensys(evals_count=subsystem.truncated_dim)[1]
            ),
        )
        return identity_wrap(
            operator,
            subsystem,
            self.subsystems,
            op_in_eigenbasis=True,
        )

    # #################################################################
    # ############ Functions for eigenvalues and matrices ############
    # #################################################################

    def _hamiltonian_for_harmonic_extended_vars(self) -> csc_matrix | ndarray:
        """Build the matrix Hamiltonian when extended vars use the harmonic basis."""
        hamiltonian = self._hamiltonian_sym_for_numerics
        # substitute all parameter values
        all_sym_parameters = (
            list(self.symbolic_params.keys())
            + self.external_fluxes
            + self.offset_charges
            + self.free_charges
        )
        hamiltonian = hamiltonian.subs(
            [
                (sym_param, getattr(self, sym_param.name))
                for sym_param in all_sym_parameters
            ]
        )
        hamiltonian = hamiltonian.subs("I", 1)
        # add an identity operator for the constant in the symbolic expression
        constant = float(hamiltonian.as_coefficients_dict()[1])
        hamiltonian -= hamiltonian.as_coefficients_dict()[1]
        hamiltonian = hamiltonian.expand() + constant * sm.symbols("I")

        # replace the extended degrees of freedom with harmonic oscillators
        for var_index in self.var_categories["extended"]:
            ECi = float(hamiltonian.coeff(f"Q{var_index}" + "**2").cancel()) / 4
            ELi = float(hamiltonian.coeff(f"θ{var_index}" + "**2").cancel()) * 2
            osc_freq = (8 * ELi * ECi) ** 0.5
            hamiltonian = (
                (
                    hamiltonian
                    - ECi * 4 * sm.symbols(f"Q{var_index}") ** 2
                    - ELi / 2 * sm.symbols(f"θ{var_index}") ** 2
                    + osc_freq
                    * (sm.symbols("Nh" + str(var_index)) + 0.5 * sm.symbols("I"))
                )
                .cancel()
                .expand()
            )

        # separating cosine and LC part of the Hamiltonian
        junction_potential = sum(
            [term for term in hamiltonian.as_ordered_terms() if "cos" in str(term)]
        )

        self.junction_potential = junction_potential
        hamiltonian_LC = hamiltonian - junction_potential

        H_LC_str = self._get_eval_hamiltonian_string(hamiltonian_LC)

        offset_free_charge_names = [
            charge_var.name for charge_var in self.offset_charges + self.free_charges
        ]
        offset_free_var_dict = dict(
            zip(offset_free_charge_names, self.offset_free_charge_values())
        )
        external_flux_names = [
            external_flux.name for external_flux in self.external_fluxes
        ]
        external_flux_dict = dict(
            zip(
                external_flux_names,
                [getattr(self, flux) for flux in external_flux_names],
            )
        )

        replacement_dict: dict[str, Any] = {
            **self.operators_by_name,
            **offset_free_var_dict,
            **external_flux_dict,
        }

        # adding matrix power to the dict
        if self.type_of_matrices == "dense":
            replacement_dict["matrix_power"] = np.linalg.matrix_power
            replacement_dict["cos"] = _cos_dia_dense
            replacement_dict["sin"] = _sin_dia_dense
        else:
            replacement_dict["matrix_power"] = matrix_power_sparse
            replacement_dict["cos"] = _cos_dia
            replacement_dict["sin"] = _sin_dia

        # adding self to the list
        replacement_dict["self"] = self

        junction_potential_matrix = self._evaluate_matrix_cosine_terms(
            junction_potential
        )
        junction_potential_matrix = Qobj_to_scipy_csc_matrix(junction_potential_matrix)

        if H_LC_str:
            return eval(H_LC_str, replacement_dict) + junction_potential_matrix
        else:
            return junction_potential_matrix

    def _evaluate_hamiltonian(self) -> csc_matrix | ndarray:
        """Substitute parameter values and return the numerical Hamiltonian matrix."""
        hamiltonian = self._hamiltonian_sym_for_numerics
        hamiltonian = hamiltonian.subs(
            [
                (param, getattr(self, str(param)))
                for param in list(self.symbolic_params.keys())
                + self.external_fluxes
                + self.offset_charges
                + self.free_charges
            ]
        )
        hamiltonian = hamiltonian.subs("I", 1)

        return self._sparsity_adaptive(
            Qobj_to_scipy_csc_matrix(self._evaluate_symbolic_expr(hamiltonian))
        )

    @check_sync_status_circuit
    def _hamiltonian_for_purely_harmonic(
        self, return_unsorted: bool = False
    ) -> csc_matrix | ndarray:
        """Hamiltonian for purely harmonic systems when ext_basis is set to harmonic.

        Parameters
        ----------
        return_unsorted:
            currently unused; retained for API compatibility.

        Returns
        -------
        :class:`scipy.sparse.csc_matrix` or dense :class:`numpy.ndarray`,
        depending on :attr:`type_of_matrices`.
        """
        hamiltonian = self._hamiltonian_sym_for_numerics
        # substitute parameters
        for sym_param in (
            self.offset_charges
            + self.free_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
        ):
            hamiltonian = hamiltonian.subs(sym_param, getattr(self, sym_param.name))
        hamiltonian = hamiltonian.subs("I", 1)
        operator_dict = {}
        for var_index in self.dynamic_var_indices:
            cutoff = getattr(self, f"cutoff_ext_{var_index}")
            theta_operator: csc_matrix | ndarray = op.a_plus_adag_sparse(
                cutoff,
                prefactor=(self.osc_lengths[var_index] / 2**0.5),
            )
            theta_operator = self._kron_operator(theta_operator, var_index)
            Q_operator: csc_matrix | ndarray = op.iadag_minus_ia_sparse(
                cutoff,
                prefactor=1 / (self.osc_lengths[var_index] * 2**0.5),
            )
            Q_operator = self._kron_operator(Q_operator, var_index)
            operator_dict[f"Q{var_index}"] = qt.Qobj(Q_operator)
            operator_dict[f"θ{var_index}"] = qt.Qobj(theta_operator)
        return self._sparsity_adaptive(
            Qobj_to_scipy_csc_matrix(eval(str(hamiltonian), operator_dict))
        )

    def _eigenvals_for_purely_harmonic(self, evals_count: int):
        """Eigenenergies for purely harmonic circuits (hierarchical diag disabled).

        Parameters
        ----------
        evals_count:
            Number of eigenenergies to return.
        """
        operator_for_var_index: list[csc_matrix | ndarray] = []
        for idx, var_index in enumerate(self.var_categories["extended"]):
            cutoff = getattr(self, f"cutoff_ext_{var_index}")
            evals = (0.5 + np.arange(0, cutoff)) * self.normal_mode_freqs[idx]
            H_osc = sp.sparse.dia_matrix(
                (evals, [0]), shape=(cutoff, cutoff), dtype=np.float64
            ).tocsc()
            operator_for_var_index.append(self._kron_operator(H_osc, var_index))
        # sum() of sparse/dense matrices produces the same sparse/dense sum at runtime;
        # mypy's Iterable[SupportsAdd] stub can't narrow this — safe to ignore.
        H = sum(operator_for_var_index)  # type: ignore[arg-type]
        unsorted_eigs = H.diagonal()  # type: ignore[attr-defined]
        dressed_indices = np.argsort(unsorted_eigs)[:evals_count]
        return unsorted_eigs[dressed_indices]

    @check_sync_status_circuit
    def hamiltonian(self) -> csc_matrix | ndarray:
        """Return the Hamiltonian of the Circuit."""
        if not self.hierarchical_diagonalization:
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                return self._hamiltonian_for_purely_harmonic()
            else:
                return self._evaluate_hamiltonian()

        else:
            bare_esys = {
                sys_index: (
                    self.hilbert_space["bare_evals"][sys_index][0],
                    self.hilbert_space["bare_evecs"][sys_index][0],
                )
                for sys_index, sys in enumerate(self.hilbert_space.subsystem_list)
            }
            hamiltonian = self.hilbert_space.hamiltonian(bare_esys=bare_esys)
            if self.type_of_matrices == "dense":
                return hamiltonian.full()
            if self.type_of_matrices == "sparse":
                return Qobj_to_scipy_csc_matrix(hamiltonian)
            raise ValueError(f"Unexpected type_of_matrices: {self.type_of_matrices!r}")

    def _evals_calc(self, evals_count: int) -> ndarray:
        """Compute the lowest ``evals_count`` eigenenergies (sorted, real).

        Parameters
        ----------
        evals_count:
            number of eigenvalues to return.
        """
        if self.is_child and not self._is_diagonalization_necessary():
            subsys_index = self.parent.subsystems.index(self)
            return self.parent.hilbert_space["bare_evals"][subsys_index][0][
                :evals_count
            ]

        if self.is_purely_harmonic and not self.hierarchical_diagonalization:
            return self._eigenvals_for_purely_harmonic(evals_count=evals_count)

        hamiltonian_mat = self.hamiltonian()
        if self.evals_method is None:
            if self.type_of_matrices == "sparse":
                evals_method = "evals_scipy_sparse"
            elif self.type_of_matrices == "dense":
                evals_method = "evals_scipy_dense"

        evals_method = self.evals_method or evals_method

        diagonalizer = (
            diag.DIAG_METHODS[evals_method]
            if isinstance(evals_method, str)
            else evals_method
        )
        evals = diagonalizer(
            hamiltonian_mat,
            evals_count=evals_count,
            **({} if self.evals_method_options is None else self.evals_method_options),
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> tuple[ndarray, ndarray]:
        """Compute eigenenergies and eigenvectors via the configured diagonalizer.

        Parameters
        ----------
        evals_count:
            number of eigenvalues/eigenvectors to return.
        """
        if self.is_child and not self._is_diagonalization_necessary():
            subsys_index = self.parent.subsystems.index(self)
            return (
                self.parent.hilbert_space["bare_evals"][subsys_index][0][:evals_count],
                self.parent.hilbert_space["bare_evecs"][subsys_index][0][
                    :, :evals_count
                ],
            )

        hamiltonian_mat = self.hamiltonian()

        if self.esys_method is None:
            if self.type_of_matrices == "sparse":
                esys_method = "esys_scipy_sparse"
            elif self.type_of_matrices == "dense":
                esys_method = "esys_scipy_dense"

        esys_method = self.esys_method or esys_method

        diagonalizer = (
            diag.DIAG_METHODS[esys_method]
            if isinstance(esys_method, str)
            else esys_method
        )
        evals, evecs = diagonalizer(
            hamiltonian_mat,
            evals_count=evals_count,
            **({} if self.esys_method_options is None else self.esys_method_options),
        )
        return order_eigensystem(evals, evecs, standardize_phase=True)

    def generate_bare_eigensys(self):
        """Return the bare-basis eigensystems of the Circuit and its subsystems.

        Output is in the truncated basis of size :attr:`truncated_dim`.
        """
        if not self.hierarchical_diagonalization:
            return self.eigensys(evals_count=self.truncated_dim)

        self._update_bare_esys()
        subsys_eigensys = dict.fromkeys([i for i in range(len(self.subsystems))])
        for idx, subsys in enumerate(self.subsystems):
            if subsys.hierarchical_diagonalization:
                subsys_eigensys[idx] = subsys.generate_bare_eigensys()
            else:
                subsys_eigensys[idx] = subsys.eigensys(evals_count=subsys.truncated_dim)
        return self.eigensys(evals_count=self.truncated_dim), subsys_eigensys

    def set_bare_eigensys(self, eigensys: tuple) -> None:
        """Store a precomputed bare eigensystem in the :attr:`hilbert_space` lookup.

        Has no effect unless :attr:`hierarchical_diagonalization` is ``True``.

        Parameters
        ----------
        eigensys:
            eigensystem in the format produced by :meth:`generate_bare_eigensys`.
        """
        if not self.hierarchical_diagonalization:
            return None
        bare_evals = np.empty((len(self.subsystems),), dtype=object)
        bare_evecs = np.empty((len(self.subsystems),), dtype=object)

        for subsys_idx, subsys in enumerate(self.subsystems):
            if subsys.hierarchical_diagonalization:
                sub_eigsys, _ = eigensys[1][subsys_idx]
                subsys.set_bare_eigensys(eigensys[1][subsys_idx])
            else:
                sub_eigsys = eigensys[1][subsys_idx]
            bare_evals[subsys_idx] = NamedSlotsNdarray(
                np.asarray([sub_eigsys[0].tolist()]),
                self.hilbert_space._parameters.paramvals_by_name,
            )
            bare_evecs[subsys_idx] = NamedSlotsNdarray(
                np.asarray([sub_eigsys[1].tolist()]),
                self.hilbert_space._parameters.paramvals_by_name,
            )
        # store eigensys of the subsystem in the HilbertSpace Lookup table
        self.hilbert_space._data["bare_evals"] = NamedSlotsNdarray(
            bare_evals, {"subsys": np.arange(len(self.subsystems))}
        )
        self.hilbert_space._data["bare_evecs"] = NamedSlotsNdarray(
            bare_evecs, {"subsys": np.arange(len(self.subsystems))}
        )
        # empty the affected_subsystem_indices
        self.affected_subsystem_indices = []

    def get_osc_param(self, var_index: int, which_param: str = "length") -> float:
        """Return an oscillator parameter from the harmonic-basis diagonalization.

        Parameters
        ----------
        var_index:
            variable index whose oscillator parameter is requested.
        which_param:
            ``"length"`` or ``"freq"``; selects which parameter to return.
            Defaults to ``"length"``.

        Returns
        -------
        Float value: the oscillator length or frequency of the oscillator
        corresponding to ``var_index``, depending on ``which_param``.
        """
        if not self.hierarchical_diagonalization:
            return eval("self.osc_" + which_param + "s[" + str(var_index) + "]")

        subsystem = self.subsystems[self.get_subsystem_index(var_index)]
        return subsystem.get_osc_param(var_index, which_param=which_param)

    def _get_cutoff_value(self, var_index: int) -> int:
        """Return the cutoff value associated with variable index ``var_index``.

        Parameters
        ----------
        var_index:
            integer variable index whose cutoff is requested.
        """
        for cutoff_name in self.parent.cutoff_names:
            if str(var_index) in cutoff_name:
                return getattr(self.parent, cutoff_name)
        raise ValueError(
            f"No cutoff found for var_index={var_index}; "
            f"available cutoff_names: {self.parent.cutoff_names!r}"
        )
