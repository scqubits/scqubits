# lifecycle.py
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
"""Parameter sync, dispatch handling, watched-property installation.

This mixin owns the runtime *lifecycle* of a Circuit / Subsystem
instance: how it reacts to parameter mutations, how cached state is
invalidated, and how the dynamically-installed
:class:`~scqubits.core.descriptors.WatchedProperty` setters route
through the central dispatch system.

* parameter syncing across the subsystem tree
  (``_sync_parameters_with_parent``,
  ``_sync_parameters_with_subsystems``,
  ``_set_sync_status_to_True``);
* central-dispatch event reception
  (``receive``, ``_store_updated_subsystem_index``,
  ``_mark_all_subsystems_as_affected``,
  ``_propagate_param_to_affected_subsystems``);
* the update pipeline
  (``update``, ``_perform_internal_updates``, ``_update_bare_esys``,
  ``_is_internal_update_required``, ``_fetch_symbolic_hamiltonian``);
* watched-property setter installation
  (``_make_property``, the three
  ``_set_property_and_update_<param_kind>`` methods, the
  ``_dispatch_suspended`` context manager, and the
  ``_PROPERTY_SETTER_BY_TYPE`` lookup that routes
  ``PropertyUpdateType`` values to the right setter).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import sympy as sm

import scqubits.core.circuit as circuit
import scqubits.settings as settings

from scqubits.core.circuit_internals._protocols import CircuitProtocol

from scqubits.core import descriptors
from scqubits.utils.misc import (
    check_sync_status_circuit,
    flatten_list_recursive,
    unique_elements_in_list,
)

__all__ = [
    "LifecycleMixin",
    "PropertyUpdateType",
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


# Type alias for the kinds of WatchedProperty updates `_make_property` knows
# how to install on Circuit/Subsystem instances.
PropertyUpdateType = Literal[
    "update_param_vars",
    "update_external_flux_or_charge",
    "update_cutoffs",
]

_PROPERTY_SETTER_BY_TYPE: dict[PropertyUpdateType, str] = {
    "update_param_vars": "_set_property_and_update_param_vars",
    "update_external_flux_or_charge": "_set_property_and_update_ext_flux_or_charge",
    "update_cutoffs": "_set_property_and_update_cutoffs",
}


class LifecycleMixin(ABC, CircuitProtocol):
    """Mixin: parameter sync + watched-property dispatch + update pipeline.

    Cross-mixin attributes and methods are inherited from
    :class:`~scqubits.core.circuit_internals._protocols.CircuitProtocol`,
    which is the single source of truth for the cross-mixin surface. At
    runtime ``CircuitProtocol`` is an empty class (its body is gated
    under ``TYPE_CHECKING``); the inheritance is a no-op.
    """

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
        property_update_type: PropertyUpdateType,
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
