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
from typing import TYPE_CHECKING, Any, Literal

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
from scqubits.core.circuit_internals.discretized_phi_operators import (
    _cos_phi,
    _i_d2_dphi2_operator,
    _i_d_dphi_operator,
    _phi_operator,
    _sin_phi,
)
from scqubits.core.circuit_internals.hamiltonian_assembly import (
    HamiltonianAssemblyMixin,
)
from scqubits.core.circuit_internals.matrix_helpers import (
    _cos_dia,
    _cos_dia_dense,
    _sin_dia,
    _sin_dia_dense,
    matrix_power_sparse,
)
from scqubits.core.circuit_internals.operator_factories import (
    make_basis_operator_method,
    make_grid_operator_method,
    make_hierarchical_diag_method,
)
from scqubits.core.circuit_internals.sawtooth import sawtooth_potential
from scqubits.core.circuit_internals.subsystem_tree import SubsystemTreeMixin
from scqubits.core.circuit_internals.sympy_helpers import (
    _generate_symbols_list,
    round_symbolic_expr,
)
from scqubits.core.circuit_internals.utils import get_trailing_number
from scqubits.core.namedslots_array import NamedSlotsNdarray
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


class CircuitRoutines(SubsystemTreeMixin, HamiltonianAssemblyMixin, ABC):
    """Mixin/ABC providing shared routines for :class:`Circuit` and :class:`Subsystem`.

    The remaining responsibilities here are: lifecycle / parameter-sync
    bookkeeping, the ``WatchedProperty`` plumbing, and a handful of
    Hilbert-space basics (cutoffs, kron, identity).  Subsystem-tree
    construction lives on :class:`SubsystemTreeMixin`; per-variable
    operator construction, Hamiltonian assembly, and eigensystem
    computation live on :class:`HamiltonianAssemblyMixin`. Concrete
    subclasses are responsible for initializing the declared attributes;
    this class only mixes in the behaviour that operates on them.
    """

    # Attributes set by concrete subclasses (Circuit, Subsystem) in their __init__.
    # Declared here so mypy can resolve cross-subclass access patterns in shared
    # methods defined on this ABC.
    hierarchical_diagonalization: bool
    var_categories: dict[
        Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]
    ]
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

    # #################################################################
    # ############### Functions for parameter queries #################
    # #################################################################
    # #################################################################
    # ############ Functions for eigenvalues and matrices ############
    # #################################################################

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
