# circuit.py
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
import copy
import functools
import itertools
import operator as builtin_op
import re
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import qutip as qt
import scipy as sp
import sympy as sm
import dill
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix
from sympy import latex

try:
    from IPython.display import display, Latex
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

import scqubits.core.discretization as discretization
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core import descriptors
import scqubits.core.oscillator as osc
import scqubits.core.storage as storage
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as utils
from scqubits import get_units
import scqubits.core.qubit_base as base

from scqubits import HilbertSpace, settings
from scqubits.io_utils.fileio_serializers import dict_serialize
from scqubits.io_utils.fileio import IOData
from scqubits.core import operators as op
from scqubits.core.circuit_utils import (
    is_potential_term,
    sawtooth_operator,
    sawtooth_potential,
    _cos_dia,
    _cos_dia_dense,
    _cos_phi,
    _cos_theta,
    _exp_i_theta_operator,
    _exp_i_theta_operator_conjugate,
    _generate_symbols_list,
    _i_d2_dphi2_operator,
    _i_d_dphi_operator,
    _n_theta_operator,
    _phi_operator,
    _sin_dia,
    _sin_dia_dense,
    _sin_phi,
    _sin_theta,
    get_operator_number,
    get_trailing_number,
    grid_operator_func_factory,
    hierarchical_diagonalization_func_factory,
    matrix_power_sparse,
    operator_func_factory,
    round_symbolic_expr,
)
from scqubits.utils.misc import (
    flatten_list_recursive,
    list_intersection,
    check_sync_status_circuit,
    unique_elements_in_list,
    Qobj_to_scipy_csc_matrix,
)
from scqubits.utils.plot_utils import _process_options
from scqubits.utils.spectrum_utils import (
    convert_matrix_to_qobj,
    identity_wrap,
    order_eigensystem,
)
import scqubits.core.circuit as circuit
from abc import ABC


class CircuitRoutines(ABC):
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

    # methods for serialization
    def serialize(self) -> "IOData":
        obj_in_bytes = dill.dumps(self)
        initdata = {"subsystem_in_hex": obj_in_bytes.hex()}
        if hasattr(self, "_id_str"):
            initdata["id_str"] = self._id_str  # type:ignore
        iodata = dict_serialize(initdata)
        iodata.typename = type(self).__name__
        return iodata

    @classmethod
    def deserialize(cls, io_data: "IOData"):
        obj_in_bytes = bytes.fromhex(io_data.as_kwargs()["subsystem_in_hex"])
        return dill.loads(obj_in_bytes)

    def return_root_child(self, var_index: int):
        if (
            not self.hierarchical_diagonalization
            and var_index in self.dynamic_var_indices
        ):
            return self
        for subsys in self.subsystems:
            if var_index in subsys.dynamic_var_indices:
                return subsys.return_root_child(var_index)

    def return_parent_circuit(self):
        """
        Returns the parent Circuit instance.
        """
        if not self.is_child:
            return self
        return self.parent.return_parent_circuit()

    @staticmethod
    def _is_expression_purely_harmonic(hamiltonian):
        """
        Method used to check if the hamiltonian is purely harmonic.
        """
        # if the hamiltonian contains any cos or sin term, return False
        if (
            len(
                set.union(
                    *[
                        (hamiltonian).atoms(operator)
                        for operator in [sm.cos, sm.sin, sm.Function("saw", real=True)]
                    ]
                )
            )
            > 0
        ):
            return False
        # further, if the hamiltonian contains any charge operator of periodic variables
        # return False
        periodic_charge_variable_index = set()
        extended_charge_variable_index = set()
        phase_variable_index = set()
        variable_str_list = [str(symbol) for symbol in list(hamiltonian.free_symbols)]
        for variable_str in variable_str_list:
            if variable_str[0] == "n" and variable_str[1:].isnumeric():
                periodic_charge_variable_index.add(variable_str[1:])
            if variable_str[0] == "Q" and variable_str[1:].isnumeric():
                extended_charge_variable_index.add(variable_str[1:])
            if variable_str[0] == "θ" and variable_str[1:].isnumeric():
                phase_variable_index.add(variable_str[1:])
        if len(periodic_charge_variable_index) > 0:
            return False
        # further, if the hamiltonian has any DoF where only its charge
        # or flux operator is present, return False
        if extended_charge_variable_index != phase_variable_index:
            return False
        return True

    def _diagonalize_purely_harmonic_hamiltonian(self, return_osc_dict: bool = False):
        """
        Method used to decouple harmonic oscillators in purely harmonic Hamiltonians.
        """
        if not self.is_purely_harmonic:
            raise Exception("The Subsystem Hamiltonian is not purely harmonic.")
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
        """
        Transforms the hamiltonian to a set of new variables using the transformation matrix.
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

    def offset_charge_transformation(self) -> None:
        """
        Prints the variable transformation between offset charges of transformed variables and the node charges.
        """
        if not hasattr(self, "symbolic_circuit"):
            raise Exception(
                f"{self._id_str} instance is not generated from a SymbolicCircuit instance, and hence does not have any associated branches."
            )
        trans_mat = np.linalg.inv(self.transformation_matrix.T)
        node_offset_charge_vars = [
            sm.symbols(f"q_n{index}")
            for index in range(
                1, len(self.symbolic_circuit.nodes) - self.is_grounded + 1
            )
        ]
        periodic_offset_charge_vars = [
            sm.symbols(f"ng{index}")
            for index in self.symbolic_circuit.var_categories["periodic"]
        ]
        periodic_offset_charge_eqns = []
        for idx, node_var in enumerate(periodic_offset_charge_vars):
            periodic_offset_charge_eqns.append(
                self._make_expr_human_readable(
                    sm.Eq(
                        periodic_offset_charge_vars[idx],
                        np.sum(trans_mat[idx, :] * node_offset_charge_vars),
                    )
                )
            )
        if _HAS_IPYTHON:
            self.print_expr_in_latex(periodic_offset_charge_eqns)
        else:
            print(periodic_offset_charge_eqns)

    def __setattr__(self, name, value):
        """
        Modifying the __setattr__ method to prevent creation of new attributes using the _frozen attribute.
        """
        if self._frozen and name in self._read_only_attributes:
            raise Exception(
                f"{name} is a read only attribute. Please use configure method to change this property of Circuit/Subsystem instance."
            )
        if not self._frozen or name in dir(self):
            super().__setattr__(name, value)
        else:
            raise Exception(f"Creating new attributes is disabled: [{name}, {value}].")

    def __reduce__(self):
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

    def __setstate__(self, state):
        pickled_dict, pickled_properties = state
        object.__setattr__(self, "_frozen", False)
        self.__dict__ = pickled_dict

        for property_name, property_obj in pickled_properties.items():
            setattr(self.__class__, property_name, property_obj)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}
        return {}

    def cutoffs_dict(self) -> Dict[int, int]:
        """
        Returns a dictionary, where each variable is associated with its respective
        cutoff.

        Returns
        -------
            Cutoffs dictionary; {var_index: cutoff}
        """
        cutoffs_dict = {}

        for var_index in self.dynamic_var_indices:
            for cutoff_name in self.cutoff_names:
                if str(var_index) in cutoff_name:
                    cutoffs_dict[var_index] = getattr(self, cutoff_name)
        return cutoffs_dict

    def _set_property_and_update_param_vars(
        self, param_name: str, value: float
    ) -> None:
        """
        Setter method to set parameter variables which are instance properties.

        Parameters
        ----------
        param_name:
            Name of the symbol which is updated
        value:
            The value to which the instance property is updated.
        """
        # update the attribute for the current instance
        # first check if the input value is valid.
        if not (np.isrealobj(value) and value >= 0):
            raise AttributeError(
                f"'{value}' is invalid. Branch parameters must be positive and real."
            )
        setattr(self, f"_{param_name}", value)

        # update the attribute for the instance in symbolic_circuit
        # generate _hamiltonian_sym_for_numerics if not already generated, delayed for
        # large circuits

        if (hasattr(self, "symbolic_circuit")) and (
            (
                (len(self.symbolic_circuit.nodes)) > settings.SYM_INVERSION_MAX_NODES
                or len(self.var_categories["frozen"]) > 0
            )
            or self.is_purely_harmonic
        ):
            capacitance_branches = [
                branch
                for branch in self.branches
                if (branch.type == "C" or "JJ" in branch.type)
            ]
            capacitance_params = [
                (
                    branch.parameters["EC"]
                    if branch.type == "C"
                    else branch.parameters["ECJ"]
                )
                for branch in capacitance_branches
            ]
            capacitance_sym_params = [
                param for param in capacitance_params if isinstance(param, sm.Expr)
            ]

            self.symbolic_circuit.update_param_init_val(param_name, value)
            # if param_name in [param.name for param in capacitance_sym_params]:
            self._user_changed_parameter = True
        # regenerate symbolic hamiltonian if purely harmonic
        if self.is_child and self.is_purely_harmonic:
            # copy the Hamiltonian from the parent
            subsys_index = self.parent.subsystems.index(self)
            self.hamiltonian_symbolic = self.parent.subsystem_hamiltonians[subsys_index]
            self._configure()

        if self.ext_basis == "harmonic":
            # set the oscillator parameters, for the extended variables (taking the coefficient of Q^2 and theta^2)
            self._set_harmonic_basis_osc_params()

        # update all subsystem instances
        if self.hierarchical_diagonalization:
            if isinstance(self, circuit.Circuit) and self._user_changed_parameter:
                self.affected_subsystem_indices = list(range(len(self.subsystems)))
            for subsys_idx, subsys in enumerate(self.subsystems):
                subsys._user_changed_parameter = self._user_changed_parameter
                if not subsys._user_changed_parameter and hasattr(subsys, param_name):
                    self._store_updated_subsystem_index(subsys_idx)
                    setattr(subsys, param_name, value)

    def _set_property_and_update_ext_flux_or_charge(
        self, param_name: str, value: float
    ) -> None:
        """
        Setter method to set external flux or offset charge variables which are instance
        properties.

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
            self.set_operators()

        # update all subsystem instances
        if self.hierarchical_diagonalization:
            for subsys_idx, subsys in enumerate(self.subsystems):
                if hasattr(subsys, param_name):
                    self._store_updated_subsystem_index(subsys_idx)
                    setattr(subsys, param_name, value)

    def _set_property_and_update_cutoffs(self, param_name: str, value: int) -> None:
        """
        Setter method to set cutoffs which are instance properties.

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

        # set operators and rebuild the HilbertSpace object
        if self.hierarchical_diagonalization:
            for subsys_idx, subsys in enumerate(self.subsystems):
                if hasattr(subsys, param_name):
                    self._store_updated_subsystem_index(subsys_idx)
                    setattr(subsys, param_name, value)

    def _set_property_and_update_user_changed_parameter(
        self, param_name: str, value: str
    ) -> None:
        """
        Setter method for changing the attrubute _user_changed_parameter
        """
        setattr(self, f"_{param_name}", value)
        if self.hierarchical_diagonalization:
            for subsys in self.subsystems:
                setattr(subsys, param_name, value)

    def _make_property(
        self,
        attrib_name: str,
        init_val: Union[int, float],
        property_update_type: str,
        use_central_dispatch: bool = True,
    ) -> None:
        """
        Creates a class instance property with the name attrib_name which is initialized
        to `init_val`. The setter is set depending on the string in the
        `property_update_type`.

        Parameters
        ----------
        attrib_name:
            Name of the property that needs to be created.
        init_val:
            The value to which the property is initialized.
        property_update_type:
            The string which sets the kind of setter used for this instance property.
        """
        setattr(self, f"_{attrib_name}", init_val)

        def getter(obj, name=attrib_name):
            return getattr(obj, f"_{name}")

        if property_update_type == "update_param_vars":

            def setter(obj, value, name=attrib_name):
                old_dispatch_status = settings.DISPATCH_ENABLED
                if old_dispatch_status:
                    settings.DISPATCH_ENABLED = False
                obj._set_property_and_update_param_vars(name, value)
                if old_dispatch_status:
                    settings.DISPATCH_ENABLED = True

        elif property_update_type == "update_external_flux_or_charge":

            def setter(obj, value, name=attrib_name):
                old_dispatch_status = settings.DISPATCH_ENABLED
                if old_dispatch_status:
                    settings.DISPATCH_ENABLED = False
                obj._set_property_and_update_ext_flux_or_charge(name, value)
                if old_dispatch_status:
                    settings.DISPATCH_ENABLED = True

        elif property_update_type == "update_cutoffs":

            def setter(obj, value, name=attrib_name):
                old_dispatch_status = settings.DISPATCH_ENABLED
                if old_dispatch_status:
                    settings.DISPATCH_ENABLED = False
                obj._set_property_and_update_cutoffs(name, value)
                if old_dispatch_status:
                    settings.DISPATCH_ENABLED = True

        elif property_update_type == "update_user_changed_parameter":

            def setter(obj, value, name=attrib_name):
                obj._set_property_and_update_user_changed_parameter(name, value)

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

    def set_discretized_phi_range(
        self, var_indices: Tuple[int], phi_range: Tuple[float]
    ) -> None:
        """
        Sets the flux range for discretized phi basis or for plotting

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
                raise Exception(
                    f"Variable with index {var_index} is not an extended variable."
                )
            self.discretized_phi_range[var_index] = phi_range
        self.operators_by_name = self.set_operators()

    def set_and_return(self, attr_name: str, value: Any) -> base.QubitBaseClass:
        """
        Allows to set an attribute after which self is returned. This is useful for
        doing something like example::

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
        if hasattr(self, "hierarchical_diagonalization"):
            self.update()
        return self

    def get_ext_basis(self):
        """
        Get the ext_basis object for the Circuit instance, according to the setting in self.hierarchical_diagonalization
        """
        if not self.hierarchical_diagonalization:
            return self.ext_basis
        else:
            ext_basis = []
            for subsys in self.subsystems:
                ext_basis.append(subsys.get_ext_basis())
            return ext_basis

    def sync_parameters_with_parent(self):
        """
        Method syncs the parameters of the subsystem with the parent instance.
        """
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

    def _set_sync_status_to_True(self, reset_affected_subsystem_indices: bool = False):
        if not self.hierarchical_diagonalization:
            return None
        self._out_of_sync = False
        if reset_affected_subsystem_indices:
            self.affected_subsystem_indices = []
        for subsys in self.subsystems:
            if subsys.hierarchical_diagonalization:
                subsys._set_sync_status_to_True()
                subsys._out_of_sync = False

    def receive(self, event: str, sender: object, **kwargs) -> None:
        """
        Method to help the CentralDispatch keep track of the sync status in Circuit and SubSystem modules
        """
        if sender is self:
            self.broadcast("QUANTUMSYSTEM_UPDATE")
            if self.hierarchical_diagonalization:
                self.hilbert_space._out_of_sync = True
        if self.hierarchical_diagonalization and (sender in self.subsystems):
            self._store_updated_subsystem_index(self.subsystems.index(sender))
            self.broadcast("CIRCUIT_UPDATE")
            self._out_of_sync = True
            self.hilbert_space._out_of_sync = True

    def _store_updated_subsystem_index(self, index: int) -> None:
        """
        Stores the index of the subsystem which is modified in affected_subsystem_indices
        """
        if not self.hierarchical_diagonalization:
            raise Exception(f"The subsystem provided to {self} has no subsystems.")
        if index not in self.affected_subsystem_indices:
            self.affected_subsystem_indices.append(index)

    def fetch_symbolic_hamiltonian(self):
        """
        Method to fetch the symbolic hamiltonian of an instance.
        """
        if isinstance(self, circuit.Circuit):
            # when the Circuit instance is created from a symbolic Hamiltonian, or nothing is updated or changed
            if not hasattr(self, "symbolic_circuit") or not (
                self._out_of_sync or self._user_changed_parameter
            ):
                return self.hamiltonian_symbolic

            if self._user_changed_parameter:
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
            full_hamiltonian = self.parent.fetch_symbolic_hamiltonian()
            hamiltonian, _ = self._sym_hamiltonian_for_var_indices(
                full_hamiltonian,
                self.dynamic_var_indices,
            )
            return hamiltonian

    def update(self, calculate_bare_esys: bool = True):
        """
        Syncs all the parameters of the subsystems with the current instance.
        """
        self._frozen = False
        self._perform_internal_updates(calculate_bare_esys=calculate_bare_esys)
        self._set_sync_status_to_True()
        self._frozen = True

    def _perform_internal_updates(
        self, fetch_hamiltonian: bool = True, calculate_bare_esys: bool = True
    ):
        # if purely harmonic the circuit attributes should change
        if self._user_changed_parameter:
            if fetch_hamiltonian:
                self.hamiltonian_symbolic = self.fetch_symbolic_hamiltonian()
            self.potential_symbolic = self.generate_sym_potential()

            if self.is_child:
                self._frozen = False
                self._find_and_set_sym_attrs()
                self._configure()

            self.generate_hamiltonian_sym_for_numerics()
            # copy the transformation matrix and normal_mode_freqs if self is a Circuit instance.
            if self.is_purely_harmonic:
                if not self.is_child:
                    self.transformation_matrix = (
                        self.symbolic_circuit.transformation_matrix
                    )
                self._diagonalize_purely_harmonic_hamiltonian()

            self.operators_by_name = self.set_operators()
            if self.hierarchical_diagonalization:
                self.generate_subsystems(only_update_subsystems=True)
                self.update_interactions()

        if self.hierarchical_diagonalization:
            self.affected_subsystem_indices = list(range(len(self.subsystems)))
            for subsys in self.subsystems:
                subsys._perform_internal_updates(
                    fetch_hamiltonian=False, calculate_bare_esys=calculate_bare_esys
                )
            if calculate_bare_esys:
                self._update_bare_esys()
        self._user_changed_parameter = False

    def _update_bare_esys(self):
        if not self.hierarchical_diagonalization:
            raise Exception(
                "Hierarchical diagonalization is not used in the current instance of Subsystem/Circuit."
            )
        _ = self.hilbert_space.generate_bare_esys(
            update_subsystem_indices=self.affected_subsystem_indices
        )
        self._out_of_sync = False
        self.hilbert_space._out_of_sync = False
        self.affected_subsystem_indices = []

    # *****************************************************************
    # **** Functions to construct the operators for the Hamiltonian ****
    # *****************************************************************
    def discretized_grids_dict_for_vars(self):
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

    def _constants_in_subsys(self, H_sys: sm.Expr, constants_expr: sm.Expr) -> sm.Expr:
        """
        Returns an expression of constants that belong to the subsystem with the
        Hamiltonian H_sys

        Parameters
        ----------
        H_sys:
            subsystem hamiltonian

        Returns
        -------
            expression of constants belonging to the subsystem
        """
        constant_expr = 0
        subsys_free_symbols = set(H_sys.free_symbols)
        constant_terms = constants_expr.copy()
        for term in constant_terms:
            if set(term.free_symbols) & subsys_free_symbols == set(term.free_symbols):
                constant_expr += term
        return constant_expr

    def _list_of_constants_from_expr(self, expr: sm.Expr) -> List[sm.Expr]:
        ordered_terms = expr.as_ordered_terms()
        constants = [
            term
            for term in ordered_terms
            if (
                set(
                    self.external_fluxes
                    + self.offset_charges
                    + self.free_charges
                    + list(self.symbolic_params.keys())
                    + [sm.symbols("I")]
                )
                & set(term.free_symbols)
            )
            == set(term.free_symbols)
        ]
        return constants

    def _check_truncation_indices(self):
        """
        Checks to see if the truncation indices for subsystems are not out of the range.
        """
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
                raise Exception(
                    f"The truncation index for {subsystem_position} exceeds the maximum"
                    f" size of {subsystem.hilbertdim() - 1}."
                )
            elif not (
                isinstance(subsystem.truncated_dim, int)
                and (subsystem.truncated_dim > 0)
            ):
                raise Exception(
                    "Invalid value encountered in subsystem_trunc_dims. "
                    "Truncated dimension must be a positive integer."
                )

    def _sym_hamiltonian_for_var_indices(
        self, hamiltonian_expr: sm.Expr, subsys_index_list: List[int]
    ) -> sm.Expr:
        """
        Returns the symbolic Hamiltonian and interaction terms of the subsystem corresponding to the set of variable indices in subsys_index_list

        Args:
            hamiltonian_expr (sm.Expr): The full Hamiltonian expression
            subsys_index_list (List[int]): A list or nested list of variable indices

        Returns:
            sm.Expr: Subsystem Hamiltonian for the given variable indices
        """
        # collecting constants to remove them for processing the Hamiltonian
        constants = self._list_of_constants_from_expr(hamiltonian_expr)
        for const in constants:
            hamiltonian_expr -= const

        non_operator_symbols = (
            self.offset_charges
            + self.free_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
            + [sm.symbols("I")]
        )

        subsys_index_list = flatten_list_recursive(subsys_index_list)

        hamiltonian_terms = hamiltonian_expr.as_ordered_terms()

        H_sys = 0 * sm.symbols("x")  # making an empty symbolic expression
        H_int = 0 * sm.symbols("x")
        for term in hamiltonian_terms:
            term_operator_indices = [
                get_trailing_number(var_sym.name)
                for var_sym in term.free_symbols
                if var_sym not in non_operator_symbols
            ]
            term_operator_indices_unique = unique_elements_in_list(
                term_operator_indices
            )

            if len(set(term_operator_indices_unique) - set(subsys_index_list)) == 0:
                H_sys += term

            if (
                len(set(term_operator_indices_unique) - set(subsys_index_list)) > 0
                and len(set(term_operator_indices_unique) & set(subsys_index_list)) > 0
            ):
                H_int += term

        return H_sys + self._constants_in_subsys(H_sys, constants), H_int

    def generate_subsystems(self, only_update_subsystems: bool = False):
        """
        Generates the subsystems (child instances of Circuit) depending on the attribute
        `self.system_hierarchy`
        """
        hamiltonian = self.hamiltonian_symbolic

        # collecting constants to remove them for processing the Hamiltonian
        constants = self._list_of_constants_from_expr(hamiltonian)
        # self._constant_terms_in_hamiltonian = constants
        for const in constants:
            hamiltonian -= const

        systems_sym = []
        interaction_sym = []

        non_operator_symbols = (
            self.offset_charges
            + self.free_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
            + [sm.symbols("I")]
        )

        for subsys_index_list in self.system_hierarchy:
            subsys_index_list = flatten_list_recursive(subsys_index_list)

            hamiltonian_terms = hamiltonian.as_ordered_terms()

            H_sys = 0 * sm.symbols("x")  # making an empty symbolic expression
            H_int = 0 * sm.symbols("x")
            for term in hamiltonian_terms:
                term_operator_indices = [
                    get_trailing_number(var_sym.name)
                    for var_sym in term.free_symbols
                    if var_sym not in non_operator_symbols
                ]
                term_operator_indices_unique = unique_elements_in_list(
                    term_operator_indices
                )

                if len(set(term_operator_indices_unique) - set(subsys_index_list)) == 0:
                    H_sys += term

                if (
                    len(set(term_operator_indices_unique) - set(subsys_index_list)) > 0
                    and len(set(term_operator_indices_unique) & set(subsys_index_list))
                    > 0
                ):
                    H_int += term

            # adding constants
            subsys_const = self._constants_in_subsys(H_sys, constants)
            if subsys_const in constants:
                constants.remove(subsys_const)
            systems_sym.append(H_sys + subsys_const)
            interaction_sym.append(H_int)
            hamiltonian -= H_sys + H_int  # removing the terms added to a subsystem

        if len(constants) > 0:
            systems_sym[0] += sum(constants)

        if only_update_subsystems:
            for subsys_index, subsys in enumerate(self.subsystems):
                subsys.hamiltonian_symbolic = systems_sym[subsys_index]
                subsys._frozen = False
                subsys._find_and_set_sym_attrs()
                subsys._configure()
                self.subsystem_interactions[subsys_index] = interaction_sym[
                    subsys_index
                ]
        else:
            # storing data in class attributes
            self.subsystem_hamiltonians: Dict[int, sm.Expr] = dict(
                zip(
                    range(len(self.system_hierarchy)),
                    [systems_sym[index] for index in range(len(self.system_hierarchy))],
                )
            )

            self.subsystem_interactions: Dict[int, sm.Expr] = dict(
                zip(
                    range(len(self.system_hierarchy)),
                    [
                        interaction_sym[index]
                        for index in range(len(self.system_hierarchy))
                    ],
                )
            )
            self.subsystems: List["circuit.Subsystem"] = []
            for index in range(len(self.system_hierarchy)):
                is_purely_harmonic = self._is_expression_purely_harmonic(
                    systems_sym[index]
                )
                ext_basis = (
                    ("harmonic" if is_purely_harmonic else self.ext_basis)
                    if not isinstance(self.ext_basis, list)
                    else self.ext_basis[index]
                )
                self.subsystems.append(
                    circuit.Subsystem(
                        self,
                        systems_sym[index],
                        system_hierarchy=self.system_hierarchy[index],
                        truncated_dim=(
                            self.subsystem_trunc_dims[index][0]
                            if type(self.subsystem_trunc_dims[index]) == list
                            else self.subsystem_trunc_dims[index]
                        ),
                        ext_basis=ext_basis,
                        subsystem_trunc_dims=(
                            self.subsystem_trunc_dims[index][1]
                            if type(self.subsystem_trunc_dims[index]) == list
                            else None
                        ),
                        evals_method=(
                            self.evals_method if not is_purely_harmonic else None
                        ),
                        evals_method_options=(
                            self.evals_method_options
                            if not is_purely_harmonic
                            else None
                        ),
                        esys_method=(
                            self.esys_method if not is_purely_harmonic else None
                        ),
                        esys_method_options=(
                            self.esys_method_options if not is_purely_harmonic else None
                        ),
                    )
                )

            self.hilbert_space = HilbertSpace(self.subsystems)

    def get_eigenstates(self) -> ndarray:
        """
        Returns the eigenstates for the SubSystem instance
        """
        if self.is_child:
            subsys_index = self.parent.hilbert_space.subsystem_list.index(self)
            return self.parent.hilbert_space["bare_evecs"][subsys_index][0]
        else:
            return self.eigensys()[1]

    def get_subsystem_index(self, var_index: int) -> int:
        """
        Returns the subsystem index for the subsystem to which the given var_index
        belongs.

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
        raise Exception(
            f"The var_index={var_index} could not be identified with any subsystem."
        )

    def update_interactions(self, recursive=False) -> None:
        """
        Update interactions of the HilbertSpace object for the `Circuit` instance if
        `hierarchical_diagonalization` is set to true.
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
                    subsys.update_interactions(recursive=recursive)

    @check_sync_status_circuit
    def _evaluate_symbolic_expr(self, sym_expr, bare_esys=None) -> qt.Qobj:
        # substitute circuit parameters
        param_symbols = (
            self.external_fluxes
            + self.offset_charges
            + self.free_charges
            + list(self.symbolic_params.keys())
        )
        for param in param_symbols:
            sym_expr = sym_expr.subs(param, getattr(self, param.name))

        # if the expression is zero
        if sym_expr == 0:
            return 0
        expr_dict = sym_expr.as_coefficients_dict()
        terms = list(expr_dict.keys())

        eval_matrix_list = []
        for idx, term in enumerate(terms):
            coefficient_sympy = float(expr_dict[term])
            if term == 1:
                eval_matrix_list.append(self._identity_qobj() * (coefficient_sympy))
                continue

            factors = term.as_ordered_factors()
            factor_op_list = []

            for factor in factors:
                if any(
                    [arg.has(sm.cos) or arg.has(sm.sin) for arg in (1.0 * factor).args]
                ):
                    factor_op_list.append(
                        self._evaluate_matrix_cosine_terms(factor, bare_esys=bare_esys)
                    )
                elif any(
                    [
                        arg.has(sm.Function("saw", real=True))
                        for arg in (1.0 * factor).args
                    ]
                ):
                    factor_op_list.append(
                        self._evaluate_matrix_sawtooth_terms(
                            factor, bare_esys=bare_esys
                        )
                    )

                else:
                    power_dict = dict(factor.as_powers_dict())
                    free_sym = list(factor.free_symbols)[0]
                    if not self.hierarchical_diagonalization:
                        factor_op_list.append(
                            self.get_operator_by_name(
                                free_sym.name,
                                bare_esys=bare_esys,
                                power=power_dict[free_sym],
                            )
                        )
                    else:
                        subsys = self.return_root_child(
                            get_trailing_number(free_sym.name)
                        )
                        operator = subsys.get_operator_by_name(
                            free_sym.name,
                            bare_esys=None,
                            power=power_dict[free_sym],
                        )
                        factor_op_list.append((subsys, operator))
            operators_per_subsys = {}
            operator_list = []
            for factor_op in factor_op_list:
                if not isinstance(factor_op, tuple):
                    operator_list.append(factor_op)
                    continue
                subsys, operator = factor_op
                if subsys not in operators_per_subsys:
                    operators_per_subsys[subsys] = [operator]
                else:
                    operators_per_subsys[subsys].append(operator)

            operator_list += [
                self.identity_wrap_for_hd(
                    functools.reduce(builtin_op.mul, operators_per_subsys[subsys]),
                    subsys,
                )
                for subsys in operators_per_subsys
            ]

            eval_matrix_list.append(
                functools.reduce(builtin_op.mul, operator_list) * coefficient_sympy
            )
        return sum(eval_matrix_list)

    def _operator_from_sym_expr_wrapper(self, sym_expr):
        def wrapper_func(self=self, sym_expr=sym_expr, bare_esys=None):
            return self._evaluate_symbolic_expr(sym_expr, bare_esys=bare_esys)

        return wrapper_func

    def _generate_symbols_list(
        self, var_str: str, iterable_list: Union[List[int], ndarray]
    ) -> List[sm.Symbol]:
        """
        Returns the list of symbols generated using the var_str + iterable as the name
        of the symbol.

        Parameters
        ----------
        var_str:
            name of the variable which needs to be generated
        iterable_list:
            The list of indices which generates the symbols
        """
        return [sm.symbols(var_str + str(iterable)) for iterable in iterable_list]

    def _set_vars(self):
        """
        Sets the attribute vars which is a dictionary containing all the Sympy Symbol
        objects for all the operators present in the circuit with no HD
        """
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
        """
        Method to set the attribute vars when hierarchical diagonalization is not used.
        """
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
        self.vars: Dict[str, Any] = {
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

    def _shift_harmonic_oscillator_potential(self, hamiltonian: sm.Expr) -> sm.Expr:
        # shifting the harmonic oscillator potential to the point of external fluxes
        flux_shift_vars = {}
        for var_index in self.var_categories["extended"]:
            flux_shift_vars[var_index] = sm.symbols("Δθ" + str(var_index))
            hamiltonian = hamiltonian.replace(
                sm.symbols(f"θ{var_index}"),
                sm.symbols(f"θ{var_index}") + flux_shift_vars[var_index],
            )  # substituting the flux offset variable offsets to collect the
            # coefficients later
        hamiltonian = hamiltonian.expand()

        flux_shift_equations = [
            hamiltonian.coeff(f"θ{var_index}").subs(
                [(f"θ{i}", 0) for i in self.var_categories["extended"]]
            )
            for var_index in flux_shift_vars.keys()
        ]  # finding the coefficients of the linear terms

        A, b = sm.linear_eq_to_matrix(
            flux_shift_equations, tuple(flux_shift_vars.values())
        )
        flux_shifts = sm.linsolve(
            (A, b), tuple(flux_shift_vars.values())
        )  # solving for the flux offsets

        if len(flux_shifts) != 0:
            flux_shifts = list(list(flux_shifts)[0])
        else:
            flux_shifts = []

        flux_shifts_dict = dict(zip(list(flux_shift_vars.values()), list(flux_shifts)))

        hamiltonian = hamiltonian.subs(
            list(flux_shifts_dict.items())
        )  # substituting the flux offsets to remove the linear terms
        hamiltonian = hamiltonian.subs(
            [(var, 0) for var in flux_shift_vars.values()]
        )  # removing the shift vars from the Hamiltonian
        # remove constants from Hamiltonian
        hamiltonian -= hamiltonian.as_coefficients_dict()[1]
        return round_symbolic_expr(hamiltonian.expand(), 16)
        # * ##########################################################################

    def generate_sym_potential(self):
        # and bringing the potential into the same form as for the class Circuit
        potential_symbolic = 0 * sm.symbols("x")
        for term in self.hamiltonian_symbolic.as_ordered_terms():
            if is_potential_term(term):
                potential_symbolic += term
        for i in self.dynamic_var_indices:
            potential_symbolic = (
                potential_symbolic.replace(
                    sm.symbols(f"cosθ{i}"), sm.cos(1.0 * sm.symbols(f"θ{i}"))
                )
                .replace(sm.symbols(f"sinθ{i}"), sm.sin(1.0 * sm.symbols(f"θ{i}")))
                .subs(sm.symbols("I"), 1 / (2 * np.pi))
            )
        return potential_symbolic

    def generate_hamiltonian_sym_for_numerics(
        self,
        hamiltonian: Optional[sm.Expr] = None,
        return_exprs=False,
    ):
        """
        Generates a symbolic expression which is ready for numerical evaluation starting
        from the expression stored in the attribute hamiltonian_symbolic. Stores the
        result in the attribute _hamiltonian_sym_for_numerics.
        """

        hamiltonian = hamiltonian or (
            self.hamiltonian_symbolic.expand()
        )  # applying expand is critical; otherwise the replacement of p^2 with ps2
        # would not succeed

        if self.ext_basis == "discretized":
            # marking the squared momentum operators with a separate symbol
            for i in self.var_categories["extended"]:
                hamiltonian = hamiltonian.replace(
                    sm.symbols(f"Q{i}") ** 2, sm.symbols("Qs" + str(i))
                )

        # associate an identity matrix with the external flux vars
        for ext_flux in self.external_fluxes:
            hamiltonian = hamiltonian.subs(
                ext_flux, ext_flux * sm.symbols("I") * 2 * np.pi
            )

        # associate an identity matrix with offset and free charge vars
        for charge_var in self.offset_charges + self.free_charges:
            hamiltonian = hamiltonian.subs(charge_var, charge_var * sm.symbols("I"))

        # finding the cosine terms
        cos_terms = sum(
            [term for term in hamiltonian.as_ordered_terms() if "cos" in str(term)]
        )
        if return_exprs:
            return hamiltonian, cos_terms
        setattr(self, "_hamiltonian_sym_for_numerics", hamiltonian)
        setattr(self, "junction_potential", cos_terms)

    # #################################################################
    # ############## Functions to construct the operators #############
    # #################################################################
    def get_cutoffs(self) -> Dict[str, list]:
        """
        Method to get the cutoffs for each of the circuit's degree of freedom.
        """
        cutoffs_dict: Dict[str, List[Any]] = {
            "cutoff_n": [],
            "cutoff_ext": [],
        }

        for cutoff_type in cutoffs_dict.keys():
            attr_list = [x for x in self.cutoff_names if cutoff_type in x]

            if len(attr_list) > 0:
                attr_list.sort()
                cutoffs_dict[cutoff_type] = [getattr(self, attr) for attr in attr_list]

        return cutoffs_dict

    def _collect_cutoff_values(self):
        if not self.hierarchical_diagonalization:
            cutoff_dict = self.get_cutoffs()
            for cutoff_name in cutoff_dict.keys():
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
        """
        Returns the Hilbert dimension of the Circuit instance
        """
        cutoff_values = np.fromiter(self._collect_cutoff_values(), dtype=int)
        return np.prod(cutoff_values)

    # helper functions
    def _kron_operator(
        self, operator: Union[csc_matrix, ndarray], var_index: int
    ) -> Union[csc_matrix, ndarray]:
        """
        Identity wraps the operator with identities generated for all the other variable
        indices present in the current Subsystem.

        Parameters
        ----------
        operator:
            The operator belonging to the variable index set in the argument index.
        index:
            Variable index to which the operator belongs

        Returns
        -------
            Returns the operator which is identity wrapped for the current subsystem.
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
                identity_left = sparse.identity(
                    np.prod(var_dim_list[:var_index_pos]),
                    format=matrix_format,
                )
            if var_index_pos < len(dynamic_var_indices) - 1:
                identity_right = sparse.identity(
                    np.prod(var_dim_list[var_index_pos + 1 :]),
                    format=matrix_format,
                )

            if var_index == dynamic_var_indices[0]:
                return sparse.kron(operator, identity_right, format=matrix_format)
            elif var_index == dynamic_var_indices[-1]:
                return sparse.kron(identity_left, operator, format=matrix_format)
            else:
                return sparse.kron(
                    sparse.kron(identity_left, operator, format=matrix_format),
                    identity_right,
                    format=matrix_format,
                )
        else:
            return self._sparsity_adaptive(operator)

    def _sparsity_adaptive(
        self, matrix: Union[csc_matrix, ndarray]
    ) -> Union[csc_matrix, ndarray]:
        """
        Changes the type of matrix depending on the attribute
        type_of_matrices

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
        """
        Returns the Qobj of the identity matrix of the right dimensions
        """
        if not self.hierarchical_diagonalization:
            return qt.identity(self.hilbertdim())

        subsys_trunc_dims = [subsys.truncated_dim for subsys in self.subsystems]

        return qt.tensor([qt.identity(truncdim) for truncdim in subsys_trunc_dims])

    def _identity(self):
        """
        Returns the Identity operator for the entire Hilbert space of the circuit.
        """
        if (
            hasattr(self, "hierarchical_diagonalization")
            and self.hierarchical_diagonalization
        ):
            return None
        dim = self.hilbertdim()
        if self.type_of_matrices == "sparse":
            op = sparse.identity(dim, format="csc")
            return op
        elif self.type_of_matrices == "dense":
            return np.identity(dim)

    def exp_i_operator(
        self, var_sym: sm.Symbol, prefactor: float
    ) -> Union[csc_matrix, ndarray]:
        """
        Returns the bare operator exp(i*\theta*prefactor), without the kron product.
        Needs the oscillator lengths to be set in the attribute, `osc_lengths`,
        when `ext_basis` is set to "harmonic".
        """
        var_index = get_trailing_number(var_sym.name)
        var_basis = self._basis_for_var_index(var_index)

        if var_basis == "periodic":
            # if abs(prefactor) != 1:
            #     raise Exception("Prefactor for periodic variable should be 1.")
            # if prefactor > 0:
            exp_i_theta = _exp_i_theta_operator(
                self.cutoffs_dict()[var_index], prefactor
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
                exp_i_theta = sp.linalg.expm(
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
        self, saw_expr: sm.Expr, bare_esys=None
    ) -> qt.Qobj:
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
        self, junction_potential: sm.Expr, bare_esys=None
    ) -> qt.Qobj:
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
                if len(term.free_symbols) == 0:
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

    def _set_harmonic_basis_osc_params(self, hamiltonian: Optional[sm.Expr] = None):
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

    def _wrapper_operator_for_purely_harmonic_system(self, operator_name: str):
        def purely_harmonic_operator_func(self=self, operator_name=operator_name):
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
            ops = dict.fromkeys(["Q", "θ"])
            for optype in ops:
                terms = op_exprs[optype].as_ordered_terms()
                operator = 0
                for term in terms:
                    sym_var_index = get_trailing_number(term.free_symbols.pop().name)
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
            return operator

        return purely_harmonic_operator_func

    def _generate_operator_methods(self) -> Dict[str, Callable]:
        """
        Returns the set of operator functions to be turned into methods of the `Circuit`
        class.
        """
        periodic_vars = self.vars["periodic"]
        extended_vars = self.vars["extended"]

        # constructing the operators for extended variables
        extended_operators = {}

        if self.hierarchical_diagonalization:
            for var_type in extended_vars:
                for sym_variable in extended_vars[var_type]:
                    op_name = sym_variable.name + "_operator"
                    extended_operators[op_name] = (
                        hierarchical_diagonalization_func_factory(sym_variable.name)
                    )

        elif self.ext_basis == "discretized":
            nonwrapped_ops = {
                "position": _phi_operator,
                "cos": _cos_phi,
                "sin": _sin_phi,
                "momentum": _i_d_dphi_operator,
                "momentum_squared": _i_d2_dphi2_operator,
            }
            for short_op_name in nonwrapped_ops.keys():
                for sym_variable in extended_vars[short_op_name]:
                    index = int(get_trailing_number(sym_variable.name))
                    op_func = nonwrapped_ops[short_op_name]
                    op_name = sym_variable.name + "_operator"
                    extended_operators[op_name] = grid_operator_func_factory(
                        op_func, index
                    )

        elif self.ext_basis == "harmonic":
            self._set_harmonic_basis_osc_params()
            nonwrapped_ops = {
                "creation": op.creation_sparse,
                "annihilation": op.annihilation_sparse,
                "number": op.number_sparse,
                "position": None,  # need to set for each variable separately
                "sin": None,
                "cos": None,
                "momentum": None,
            }

            for list_idx, var_index in enumerate(self.var_categories["extended"]):
                if self.is_purely_harmonic and self.ext_basis == "harmonic":
                    for short_op_name in [
                        "position",
                        "momentum",
                        "number",
                        "annihilation",
                        "creation",
                    ]:
                        sym_variable = extended_vars[short_op_name][list_idx]
                        op_func = self._wrapper_operator_for_purely_harmonic_system(
                            sym_variable.name
                        )
                        op_name = sym_variable.name + "_operator"
                        extended_operators[op_name] = op_func

                else:
                    nonwrapped_ops["position"] = op.a_plus_adag_sparse
                    nonwrapped_ops["sin"] = op.sin_theta_harmonic
                    nonwrapped_ops["cos"] = op.cos_theta_harmonic
                    nonwrapped_ops["momentum"] = op.iadag_minus_ia_sparse

                    for short_op_name in nonwrapped_ops.keys():
                        op_func = nonwrapped_ops[short_op_name]
                        sym_variable = extended_vars[short_op_name][list_idx]
                        op_name = sym_variable.name + "_operator"
                        extended_operators[op_name] = operator_func_factory(
                            op_func, var_index, op_type=short_op_name
                        )

        # constructing the operators for periodic variables
        periodic_operators = {}
        nonwrapped_ops = {
            "sin": _sin_theta,
            "cos": _cos_theta,
            "number": _n_theta_operator,
        }
        for short_op_name, op_func in nonwrapped_ops.items():
            for sym_variable in periodic_vars[short_op_name]:
                var_index = get_operator_number(sym_variable.name)
                op_name = sym_variable.name + "_operator"
                if self.hierarchical_diagonalization:
                    periodic_operators[op_name] = (
                        hierarchical_diagonalization_func_factory(sym_variable.name)
                    )
                else:
                    periodic_operators[op_name] = operator_func_factory(
                        op_func, var_index
                    )
        return {
            **periodic_operators,
            **extended_operators,
            "I_operator": CircuitRoutines._identity,
        }

    # #################################################################
    # ############### Functions for parameter queries #################
    # #################################################################
    def get_params(self) -> List[float]:
        """
        Method to get the circuit parameters set for all the branches.
        """
        params = []
        for param in self.symbolic_params:
            params.append(getattr(self, param.name))
        return params

    def offset_free_charge_values(self) -> List[float]:
        """
        Returns all the offset charges set using the circuit attributes for each of the
        periodic degree of freedom.
        """
        return [
            getattr(self, charge_var.name)
            for charge_var in self.offset_charges + self.free_charges
        ]

    def set_operators(self) -> Dict[str, Callable]:
        """
        Creates the operator methods `<name>_operator` for the circuit.
        """

        if self.hierarchical_diagonalization:
            for subsys in self.subsystems:
                subsys.operators_by_name = subsys.set_operators()

        op_func_by_name = self._generate_operator_methods()
        for op_name, op_func in op_func_by_name.items():
            setattr(self, op_name, MethodType(op_func, self))

        return {func_name: getattr(self, func_name) for func_name in op_func_by_name}

    def is_subsystem(self, instance):
        """
        Returns true if the instance is a subsystem of self (regardless of the hierarchy)
        """
        if len(set(self.dynamic_var_indices) & set(instance.dynamic_var_indices)) > 0:
            return True
        return False

    def identity_wrap_for_hd(
        self,
        operator: Optional[Union[csc_matrix, ndarray]],
        child_instance,
        bare_esys: Optional[Dict[int, Tuple]] = None,
    ) -> qt.Qobj:
        """
        Returns an identity wrapped operator whose size is equal to the
        `self.hilbertdim()`. Only converts operator which belongs to a specific variable
        index. For example, operator Q_1 or cos(\theta_1). But not, Q1*Q2.

        Parameters
        ----------
        operator:
            operator in the form of csc_matrix, ndarray
        instance:
            The subsystem to which the operator belongs

        Returns
        -------
            identity wrapped operator.
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
                else subsystem.get_eigenstates()
            ),
        )
        return identity_wrap(
            operator,
            subsystem,
            self.subsystems,
            evecs=(
                bare_esys[subsystem_index][1]
                if bare_esys
                else subsystem.get_eigenstates()
            ),
        )

    @check_sync_status_circuit
    def get_operator_by_name(
        self, operator_name: str, power: Optional[int] = None, bare_esys=None
    ) -> qt.Qobj:
        """
        Returns the operator for the given operator symbol which has the same dimension
        as the hilbertdim of the instance from which the operator is requested.

        Parameters
        ----------
        operator_name:
            Name of a sympy Symbol object which should be one among the symbols in the
            attribute vars
        power:
            If asking for an operator raised to a certain power. Which wen set to None
            defaults to 1

        Returns
        -------
            operator identified by `operator_name`
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
                else subsystem.get_eigenstates()
            ),
        )
        return identity_wrap(
            operator,
            subsystem,
            self.subsystems,
            evecs=(
                bare_esys[subsystem_index][1]
                if bare_esys
                else subsystem.get_eigenstates()
            ),
        )

    # #################################################################
    # ############ Functions for eigenvalues and matrices ############
    # #################################################################
    def _is_mat_mul_replacement_necessary(self, term):
        return (
            set(self.var_categories["extended"])
            & set([get_trailing_number(str(i)) for i in term.free_symbols])
        ) and "*" in str(term)

    def _replace_mat_mul_operator(self, term: sm.Expr):
        if not self._is_mat_mul_replacement_necessary(term):
            return str(term)

        if self.ext_basis == "discretized":
            term_string = str(term)
            term_var_categories = [
                get_trailing_number(str(i)) for i in term.free_symbols
            ]
            if len(set(term_var_categories) & set(self.var_categories["extended"])) > 1:
                if all(["Q" in var.name for var in term.free_symbols]):
                    term_string = str(term).replace(
                        "*", "@"
                    )  # replacing all the * with @

        elif self.ext_basis == "harmonic":
            # replace ** with np.matrix_power
            if "**" in str(term):
                operators = [
                    match.replace("**", "")
                    for match in re.findall(r"[^*]+\*{2}", str(term), re.MULTILINE)
                ]
                exponents = re.findall(r"(?<=\*{2})\d", str(term), re.MULTILINE)

                new_string_list = []
                for idx, operator in enumerate(operators):
                    if get_trailing_number(operator) in self.var_categories["extended"]:
                        new_string_list.append(
                            f"matrix_power({operator},{exponents[idx]})"
                        )
                    else:
                        new_string_list.append(operator + "**" + exponents[idx])
                term_string = "*".join(new_string_list)
            else:
                term_string = str(term)

            # replace * with @ in the entire term
            if len(term.free_symbols) > 1:
                term_string = re.sub(
                    r"(?<=[^*])\*(?!\*)", "@", term_string, re.MULTILINE
                )
        return term_string

    def _get_eval_hamiltonian_string(self, H: sm.Expr) -> str:
        """
        Returns the string which defines the expression for Hamiltonian in harmonic
        oscillator basis
        """
        expr_dict = H.as_coefficients_dict()
        # removing zero terms
        expr_dict = {key: expr_dict[key] for key in expr_dict if expr_dict[key] != 0}
        terms_list = list(expr_dict.keys())
        coeff_list = list(expr_dict.values())

        H_string = ""
        for idx, term in enumerate(terms_list):
            term_string = f"{coeff_list[idx]}*{self._replace_mat_mul_operator(term)}"
            if float(coeff_list[idx]) > 0:
                term_string = "+" + term_string
            H_string += term_string

        # replace all position, sin and cos operators with methods
        H_string = re.sub(r"(?P<x>(θ\d)|(cosθ\d))", r"\g<x>_operator()", H_string)

        # replace all other operators with methods
        operator_symbols_list = flatten_list_recursive(
            [
                (
                    list(short_op_dict.values())
                    if isinstance(short_op_dict, dict)
                    else short_op_dict
                )
                for short_op_dict in list(self.vars.values())
            ]
        )
        operator_name_list = [symbol.name for symbol in operator_symbols_list]
        for operator_name in operator_name_list:
            if "θ" not in operator_name:
                H_string = H_string.replace(
                    operator_name, operator_name + "_operator()"
                )
        return H_string

    def _hamiltonian_for_harmonic_extended_vars(self) -> Union[csc_matrix, ndarray]:
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

        replacement_dict: Dict[str, Any] = {
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

    def _evaluate_hamiltonian(self) -> csc_matrix:
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
    ) -> csc_matrix:
        """Hamiltonian for purely harmonic systems when ext_basis is set to harmonic

        Returns:
            csc_matrix:
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
        H_diag_basis = self._identity() * 0
        identity = self._identity()
        operator_dict = {}
        for var_index in self.dynamic_var_indices:
            cutoff = getattr(self, f"cutoff_ext_{var_index}")
            theta_operator = op.a_plus_adag_sparse(
                cutoff,
                prefactor=(self.osc_lengths[var_index] / 2**0.5),
            )
            theta_operator = self._kron_operator(theta_operator, var_index)
            Q_operator = op.iadag_minus_ia_sparse(
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
        """
        Returns Hamiltonian for purely harmonic circuits. Hierarchical diagonalization
        is disabled for such circuits.

        Parameters
        ----------
        evals_count:
            Number of eigenenergies
        """
        operator_for_var_index = []
        for idx, var_index in enumerate(self.var_categories["extended"]):
            cutoff = getattr(self, f"cutoff_ext_{var_index}")
            evals = (0.5 + np.arange(0, cutoff)) * self.normal_mode_freqs[idx]
            H_osc = sp.sparse.dia_matrix(
                (evals, [0]), shape=(cutoff, cutoff), dtype=np.float_
            )
            operator_for_var_index.append(self._kron_operator(H_osc, var_index))
        H = sum(operator_for_var_index)
        unsorted_eigs = H.diagonal()
        dressed_indices = np.argsort(unsorted_eigs)[:evals_count]
        return unsorted_eigs[dressed_indices]

    @check_sync_status_circuit
    def hamiltonian(self) -> Union[csc_matrix, ndarray]:
        """
        Returns the Hamiltonian of the Circuit.
        """
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

    @check_sync_status_circuit
    def hamiltonian_for_qutip_dynamics(
        self,
        free_var_func_dict: Dict[str, Callable],
        prefactor: float = 1.0,
        extra_terms: Optional[str] = None,
    ) -> Tuple[List[Union[qt.Qobj, Tuple[qt.Qobj, Callable]]], sm.Expr, List[sm.Expr]]:
        """
        Returns the Hamiltonian in a format amenable to be forwarded to mesolve in
        Qutip. Also returns the symbolic expressions of time independent and time
        dependent terms of the Hamiltonian.
        `free_var_func_dict` is a dictionary which has a tuple for every time dependent variable with the follwoing two elements:
        1 - time dependent function for the variable
        2 - the order to which the Hamiltonian needs to be expanded around the drive amplitude
        For example, to get the Hamiltonian for a circuit where Φ1 is the time varying parameter and the drive is expanded to second order,
        this method can be called in the following way:

        ```
        def flux_t(t, args):
            return np.sin(t*2)
        def EJ_t(t, args):
            return (1-np.exp(-t/1))*0.2
        free_var_func_dict = {"Φ1": (flux_t, 2), "EJ":(EJ_t, 1)}

        mesolve_input_H = self.hamiltonian_for_qutip_dynamics(free_var_func_dict)
        ```
        Parameters
        ----------
        free_var_func_dict:
            Dict, as defined in the description above
        prefactor:
            float, value with which the Hamiltonian and corresponding operators are multiplied with
        extra_terms:
            str, a string which will be converted into sympy expression, containing terms which are not present in the Circuit Hamiltonian. Is useful to define custom drive operators.
        """
        free_var_names = list(free_var_func_dict.keys())
        free_var_symbols = [sm.symbols(sym_name) for sym_name in free_var_names]

        fixed_hamiltonian = 0 * sm.symbols("x")
        time_varying_hamiltonian = []

        all_circuit_params = (
            list(self.symbolic_params.keys())
            + self.offset_charges
            + self.external_fluxes
            + self.free_charges
        )
        # adding extra terms to the Hamiltonian
        if extra_terms:
            extra_terms = sm.parse_expr(extra_terms)
            for extra_sym in extra_terms.free_symbols:
                if (
                    extra_sym not in self._hamiltonian_sym_for_numerics.free_symbols
                    and extra_sym not in free_var_symbols
                ):
                    raise Exception(f"{extra_sym.name} is unknown.")
        else:
            extra_terms = 0

        sym_hamiltonian = self._hamiltonian_sym_for_numerics + extra_terms
        sym_hamiltonian = sym_hamiltonian.subs("I", 1)
        # series expand Hamiltonian around the bias
        for free_var in free_var_symbols:
            if free_var in all_circuit_params:
                sym_hamiltonian = sym_hamiltonian.subs(
                    free_var, free_var + getattr(self, free_var.name)
                ).expand()

        expr_dict = sym_hamiltonian.expand().as_coefficients_dict()
        terms = list(expr_dict.keys())
        time_dep_terms = {}

        for term in terms:
            if len(list_intersection(list(term.free_symbols), free_var_symbols)) == 0:
                fixed_hamiltonian = fixed_hamiltonian + term * expr_dict[term]
                continue
            # if the term does have a free variable
            ### expand trigonometrically
            should_trig_expand = any(
                [
                    (free_sym in term.free_symbols and term.coeff(free_sym) == 0)
                    for free_sym in free_var_symbols
                ]
            )
            term_expanded = term.expand(trig=should_trig_expand)

            term_expr_dict = term_expanded.as_coefficients_dict()
            terms_in_term = list(term_expr_dict.keys())
            for inner_term in terms_in_term:
                operator_expr, parameter_expr = inner_term.as_independent(
                    *free_var_symbols, as_Mul=True
                )

                if parameter_expr in time_dep_terms:
                    time_dep_terms[parameter_expr] = (
                        operator_expr * expr_dict[term] * term_expr_dict[inner_term]
                        + time_dep_terms[parameter_expr]
                    )
                else:
                    time_dep_terms[parameter_expr] = (
                        operator_expr * expr_dict[term] * term_expr_dict[inner_term]
                    )

        for parameter_expr in time_dep_terms:
            # separating the time independent constants
            for sym in parameter_expr.free_symbols:
                if sym not in free_var_symbols:
                    parameter_expr = parameter_expr.subs(sym, getattr(self, sym.name))

            lambdify_func = sm.lambdify(
                list(parameter_expr.free_symbols), parameter_expr, "numpy"
            )

            def parameter_func(
                t,
                args,
                parameter_expr=parameter_expr,
                self=self,
                free_var_func_dict=free_var_func_dict,
                lambdify_func=lambdify_func,
            ):
                return lambdify_func(
                    *[
                        free_var_func_dict[sym.name](t, args)
                        for sym in parameter_expr.free_symbols
                    ]
                )

            operator_matrix = (
                self._evaluate_symbolic_expr(time_dep_terms[parameter_expr]) * prefactor
            )  # also multiplying the constant to the operator
            if operator_matrix == 0:
                continue
            time_varying_hamiltonian.append([operator_matrix, parameter_func])

        fixed_hamiltonian = fixed_hamiltonian.subs("I", 1)
        return (
            [self._evaluate_symbolic_expr(fixed_hamiltonian) * prefactor]
            + time_varying_hamiltonian,
            fixed_hamiltonian,
            dict((val, key) for key, val in time_dep_terms.items()),
        )

    def _evals_calc(self, evals_count: int) -> ndarray:
        # dimension of the hamiltonian
        hilbertdim = self.hilbertdim()

        if self.is_purely_harmonic and not self.hierarchical_diagonalization:
            return self._eigenvals_for_purely_harmonic(evals_count=evals_count)

        hamiltonian_mat = self.hamiltonian()
        if self.type_of_matrices == "sparse":
            evals = utils.eigsh_safe(
                hamiltonian_mat,
                return_eigenvectors=False,
                k=evals_count,
                which="SA",
            )
        elif self.type_of_matrices == "dense":
            evals = sp.linalg.eigvalsh(
                hamiltonian_mat, subset_by_index=[0, evals_count - 1]
            )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        # dimension of the hamiltonian

        hamiltonian_mat = self.hamiltonian()
        if self.type_of_matrices == "sparse":
            evals, evecs = utils.eigsh_safe(
                hamiltonian_mat,
                return_eigenvectors=True,
                k=evals_count,
                which="SA",
            )
        elif self.type_of_matrices == "dense":
            evals, evecs = sp.linalg.eigh(
                hamiltonian_mat,
                eigvals_only=False,
                subset_by_index=[0, evals_count - 1],
            )
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def generate_bare_eigensys(self):
        if not self.hierarchical_diagonalization:
            return self.eigensys(evals_count=self.truncated_dim)

        subsys_eigensys = dict.fromkeys([i for i in range(len(self.subsystems))])
        for idx, subsys in enumerate(self.subsystems):
            if subsys.hierarchical_diagonalization:
                subsys_eigensys[idx] = subsys.generate_bare_eigensys()
            else:
                subsys_eigensys[idx] = subsys.eigensys(evals_count=subsys.truncated_dim)
        return self.eigensys(evals_count=self.truncated_dim), subsys_eigensys

    def set_bare_eigensys(self, eigensys):
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

    # ****************************************************************
    # ***** Functions for pretty display of symbolic expressions *****
    # ****************************************************************
    @staticmethod
    def print_expr_in_latex(expr: Union[sm.Expr, List["sm.Equality"]]) -> None:
        """
        Print a sympy expression or a list of equalities in LaTeX

        Parameters
        ----------
        expr:
            a sympy expressions or a list of equalities
        """
        if isinstance(expr, sm.Expr):
            display(Latex("$ " + sm.printing.latex(expr) + " $"))
        elif isinstance(expr, list):
            equalities_in_latex = "$ "
            for eqn in expr:
                equalities_in_latex += sm.printing.latex(eqn) + " \\\ "
            equalities_in_latex = equalities_in_latex[:-4] + " $"
            display(Latex(equalities_in_latex))

    def __repr__(self) -> str:
        # string to describe the Circuit
        return self._id_str

    def info(self):
        """
        Describes the Circuit instance, its parameters, and the symbolic Hamiltonian.
        """
        # string to describe the Circuit
        if not _HAS_IPYTHON:
            return self._id_str
        # Hamiltonian string
        H_latex_str = (
            "$H=" + sm.printing.latex(self.sym_hamiltonian(return_expr=True)) + "$"
        )
        # describe the variables
        cutoffs_dict = self.cutoffs_dict()
        var_str = "Operators (flux, charge) - cutoff: "
        if len(self.var_categories["periodic"]) > 0:
            var_str += "  \n Discrete Charge Basis:  "
            for var_index in self.var_categories["periodic"]:
                var_str += (
                    f"$(θ{var_index}, n{var_index}) - {cutoffs_dict[var_index]}$, "
                )

        var_str_discretized = "  \nDiscretized Phi basis:  "
        var_str_harmonic = "  \nHarmonic oscillator basis:  "

        for var_index in self.var_categories["extended"]:
            var_index_basis = self._basis_for_var_index(var_index)
            if var_index_basis == "discretized":
                var_str_discretized += (
                    f"$(θ{var_index}, Q{var_index}) - {cutoffs_dict[var_index]}$, "
                )
            if var_index_basis == "harmonic":
                var_str_harmonic += (
                    f"$(θ{var_index}, Q{var_index}) - {cutoffs_dict[var_index]}$, "
                )

        if var_str_discretized == "  \nDiscretized Phi basis:  ":
            var_str_discretized = ""
        if var_str_harmonic == "  \nHarmonic oscillator basis:  ":
            var_str_harmonic = ""

        display(Latex(H_latex_str))
        display(
            Latex(
                var_str
                + var_str_discretized
                + ("  \n" if var_str_discretized else "")
                + var_str_harmonic
            )
        )
        # symbolic parameters
        if len(self.symbolic_params) > 0:
            sym_params_str = "Symbolic parameters (symbol, default value):  "
            for sym, val in self.symbolic_params.items():
                sym_params_str += f"$({sym.name}, {val})$, "
            display(Latex(sym_params_str))
        if len(self.external_fluxes) > 0:
            sym_params_str = "External fluxes (symbol, default value):  "
            for sym in self.external_fluxes:
                sym_params_str += f"$({sym.name}, {getattr(self, sym.name)})$, "
            display(Latex(sym_params_str))
        if len(self.offset_charges) > 0:
            sym_params_str = "Offset charges (symbol, default value):  "
            for sym in self.offset_charges:
                sym_params_str += f"$({sym.name}, {getattr(self, sym.name)})$, "
            display(Latex(sym_params_str))
        if len(self.free_charges) > 0:
            sym_params_str = "Free charges (symbol, default value):  "
            for sym in self.free_charges:
                sym_params_str += f"$({sym.name}, {getattr(self, sym.name)})$, "
            display(Latex(sym_params_str))
        if self.hierarchical_diagonalization:
            display(Latex(f"System hierarchy: {self.system_hierarchy}"))
            display(Latex(f"Truncated Dimensions: {self.subsystem_trunc_dims}"))

    def _make_expr_human_readable(self, expr: sm.Expr, float_round: int = 6) -> sm.Expr:
        """
        Method returns a user readable symbolic expression for the current instance

        Parameters
        ----------
        expr:
            A symbolic sympy expression
        float_round:
            Number of digits after the decimal to which floats are rounded

        Returns
        -------
            Sympy expression which is simplified to make it human readable.
        """
        expr_modified = expr
        # rounding the decimals in the coefficients
        # citation:
        # https://stackoverflow.com/questions/43804701/round-floats-within-an-expression
        # accepted answer
        for term in sm.preorder_traversal(expr):
            if isinstance(term, sm.Float):
                expr_modified = expr_modified.subs(term, round(term, float_round))

        for var_index in self.dynamic_var_indices:
            # replace sinθ with sin(..) and similarly with cos
            expr_modified = (
                expr_modified.replace(
                    sm.symbols(f"cosθ{var_index}"),
                    sm.cos(1.0 * sm.symbols(f"θ{var_index}")),
                )
                .replace(
                    sm.symbols(f"sinθ{var_index}"),
                    sm.sin(1.0 * sm.symbols(f"θ{var_index}")),
                )
                .replace(
                    (1.0 * sm.symbols(f"θ{var_index}")),
                    (sm.symbols(f"θ{var_index}")),
                )
            )
            # replace Qs with Q^2 etc
            expr_modified = expr_modified.replace(
                sm.symbols("Qs" + str(var_index)), sm.symbols(f"Q{var_index}") ** 2
            )
            expr_modified = expr_modified.replace(
                sm.symbols("ng" + str(var_index)), sm.symbols("n_g" + str(var_index))
            )
            # replace I by 1
            expr_modified = expr_modified.replace(sm.symbols("I"), 1)
        for ext_flux_var in self.external_fluxes:
            # removing 1.0 decimals from flux vars
            expr_modified = expr_modified.replace(1.0 * ext_flux_var, ext_flux_var)
        return expr_modified

    def sym_potential(
        self, float_round: int = 6, print_latex: bool = False, return_expr: bool = False
    ) -> Union[sm.Expr, None]:
        """
        Method prints a user readable symbolic potential for the current instance

        Parameters
        ----------
        float_round:
            Number of digits after the decimal to which floats are rounded
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
                if set to True, all printing is suppressed and the function will silently
                return the sympy expression
        """
        potential = self._make_expr_human_readable(
            self.potential_symbolic, float_round=float_round
        )

        for external_flux in self.external_fluxes:
            potential = potential.replace(
                external_flux,
                sm.symbols(
                    "(2π" + "Φ_{" + str(get_trailing_number(str(external_flux))) + "})"
                ),
            )

        if print_latex:
            print(latex(potential))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(potential)
        else:
            print(potential)

    def sym_hamiltonian(
        self,
        subsystem_index: Optional[int] = None,
        float_round: int = 6,
        print_latex: bool = False,
        return_expr: bool = False,
    ) -> Union[sm.Expr, None]:
        """
        Prints a user readable symbolic Hamiltonian for the current instance

        Parameters
        ----------
        subsystem_index:
            when set to an index, the Hamiltonian for the corresponding subsystem is
            returned.
        float_round:
            Number of digits after the decimal to which floats are rounded
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
            if set to True, all printing is suppressed and the function will silently
            return the sympy expression
        """
        if subsystem_index is not None:
            if not self.hierarchical_diagonalization:
                raise Exception(
                    "Hierarchical diagonalization was not enabled. Hence there "
                    "are no identified subsystems addressable by "
                    "subsystem_index."
                )
            # start with the raw system hamiltonian
            sym_hamiltonian = self._make_expr_human_readable(
                self.subsystems[subsystem_index].hamiltonian_symbolic.expand(),
                float_round=float_round,
            )
            # create PE symbolic expressions
            sym_hamiltonian_PE = self._make_expr_human_readable(
                self.subsystems[subsystem_index].potential_symbolic.expand(),
                float_round=float_round,
            )
            # obtain the KE of hamiltonian
            pot_symbols = (
                self.external_fluxes
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["extended"]
                ]
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["periodic"]
                ]
            )
            sym_hamiltonian_KE = 0 * sm.Symbol("x")
            for term in sym_hamiltonian.args:
                if term.free_symbols.isdisjoint(pot_symbols):
                    sym_hamiltonian_KE = sm.Add(sym_hamiltonian_KE, term)

            # add a symbolic 2pi
            for external_flux in self.external_fluxes:
                sym_hamiltonian_PE = self._make_expr_human_readable(
                    sym_hamiltonian_PE.replace(
                        external_flux,
                        sm.symbols(
                            "(2π"
                            + "Φ_{"
                            + str(get_trailing_number(str(external_flux)))
                            + "})"
                        ),
                    ),
                    float_round=float_round,
                )
            # substitute free charges
            for free_charge in self.free_charges:
                sym_hamiltonian_PE = sym_hamiltonian_PE.subs(
                    free_charge, getattr(self, free_charge.name)
                )
            # obtain system symbolic hamiltonian by glueing KE and PE
            sym_hamiltonian = sm.Add(
                sym_hamiltonian_KE, sym_hamiltonian_PE, evaluate=False
            )
        else:
            # create KE and PE symbolic expressions
            sym_hamiltonian = self._make_expr_human_readable(
                self.hamiltonian_symbolic.expand(),
                float_round=float_round,
            )
            # substitute free charges
            for free_charge in self.free_charges:
                sym_hamiltonian = sym_hamiltonian.subs(
                    free_charge, getattr(self, free_charge.name)
                )
            pot_symbols = (
                self.external_fluxes
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["extended"]
                ]
                + [
                    sm.symbols("θ" + str(idx))
                    for idx in self.var_categories["periodic"]
                ]
            )
            sym_hamiltonian_KE = 0 * sm.Symbol("x")
            for term in sym_hamiltonian.args:
                if term.free_symbols.isdisjoint(pot_symbols):
                    sym_hamiltonian_KE = sm.Add(sym_hamiltonian_KE, term)
            sym_hamiltonian_PE = self._make_expr_human_readable(
                self.potential_symbolic.expand(), float_round=float_round
            )
            # add a 2pi coefficient in front of external fluxes, since the the external
            # fluxes are measured in 2pi numerically
            for external_flux in self.external_fluxes:
                sym_hamiltonian_PE = sym_hamiltonian_PE.replace(
                    external_flux,
                    sm.symbols(
                        "(2π"
                        + "Φ_{"
                        + str(get_trailing_number(str(external_flux)))
                        + "})"
                    ),
                )
            # add the KE and PE and suppress the evaluation
            sym_hamiltonian = sm.Add(
                sym_hamiltonian_KE, sym_hamiltonian_PE, evaluate=False
            )
        if return_expr:
            return sym_hamiltonian
        if print_latex:
            print(latex(sym_hamiltonian))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(sym_hamiltonian)
        else:
            print(sym_hamiltonian)

    def sym_interaction(
        self,
        subsystem_indices: Tuple[int],
        float_round: int = 6,
        print_latex: bool = False,
        return_expr: bool = False,
    ) -> Union[sm.Expr, None]:
        """
        Print the interaction between any set of subsystems for the current instance.
        It would print the interaction terms having operators from all the subsystems
        mentioned in the tuple.

        Parameters
        ----------
        subsystem_indices:
            Tuple of subsystem indices
        float_round:
            Number of digits after the decimal to which floats are rounded
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
            if set to True, all printing is suppressed and the function will silently
            return the sympy expression
        """
        interaction = sm.symbols("x") * 0
        for subsys_index_pair in itertools.combinations(subsystem_indices, 2):
            for term in self.subsystem_interactions[
                min(subsys_index_pair)
            ].as_ordered_terms():
                term_mod = term.subs(
                    [
                        (symbol, 1)
                        for symbol in self.external_fluxes
                        + self.offset_charges
                        + self.free_charges
                        + list(self.symbolic_params.keys())
                        + [sm.symbols("I")]
                    ]
                )
                interaction_var_indices = [
                    self.get_subsystem_index(get_trailing_number(symbol.name))
                    for symbol in term_mod.free_symbols
                ]
                if np.array_equal(
                    np.sort(interaction_var_indices), np.sort(subsystem_indices)
                ):
                    interaction += term
        for external_flux in self.external_fluxes:
            interaction = self._make_expr_human_readable(
                interaction.replace(external_flux, external_flux / (2 * np.pi)),
                float_round=float_round,
            )
            interaction = interaction.replace(
                external_flux,
                sm.symbols(
                    "(2π" + "Φ_{" + str(get_trailing_number(str(external_flux))) + "})"
                ),
            )
        if return_expr:
            return interaction
        if print_latex:
            print(latex(interaction))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(interaction)
        else:
            print(interaction)

    # ****************************************************************
    # ************* Functions for plotting potential *****************
    # ****************************************************************
    def potential_energy(self, **kwargs) -> ndarray:
        """
        Returns the full potential of the circuit evaluated in a grid of points as
        chosen by the user or using default variable ranges.

        Parameters
        ----------
        θ<index>:
            value(s) for variable :math:`\theta_i` in the potential.
        """
        periodic_indices = self.var_categories["periodic"]
        discretized_ext_indices = self.var_categories["extended"]
        var_categories = discretized_ext_indices + periodic_indices

        # substituting the parameters
        potential_sym = self.potential_symbolic.subs("I", 1)
        for ext_flux in self.external_fluxes:
            potential_sym = potential_sym.subs(ext_flux, ext_flux * 2 * np.pi)

        # constructing the grids
        parameters = dict.fromkeys(
            [f"θ{index}" for index in var_categories]
            + [var.name for var in self.external_fluxes]
            + [var.name for var in self.symbolic_params]
        )

        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                parameters[var_name] = kwargs[var_name]
            elif isinstance(kwargs[var_name], (int, float)):
                parameters[var_name] = kwargs[var_name]
            else:
                raise AttributeError(
                    "Only float, int or Numpy ndarray assignments are allowed."
                )

        for var_name in parameters.keys():
            if parameters[var_name] is None:
                if var_name in [
                    var.name
                    for var in list(self.symbolic_params.keys()) + self.external_fluxes
                ]:
                    parameters[var_name] = getattr(self, var_name)
                elif var_name in [f"θ{index}" for index in var_categories]:
                    raise AttributeError(var_name + " is not set.")

        # creating a meshgrid for multiple dimensions
        sweep_vars = {}
        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                sweep_vars[var_name] = kwargs[var_name]
        if len(sweep_vars) > 1:
            sweep_vars.update(
                zip(
                    sweep_vars,
                    np.meshgrid(*[grid for grid in sweep_vars.values()]),
                )
            )
            for var_name in sweep_vars:
                parameters[var_name] = sweep_vars[var_name]

        potential_func = sm.lambdify(
            parameters.keys(), potential_sym, [{"saw": sawtooth_potential}, "numpy"]
        )

        return potential_func(*parameters.values())

    def plot_potential(self, **kwargs) -> Tuple[Figure, Axes]:
        r"""
        Returns the plot of the potential for the circuit instance. Make sure to not set
        more than two variables in the instance.potential to a Numpy array, as the the
        code cannot plot with more than 3 dimensions.

        Parameters
        ----------
        θ<index>:
            value(s) for the variable :math:`\theta_i` occurring in the potential.

        Returns
        -------
            Returns a axes and figure for further editing.
        """

        periodic_indices = self.var_categories["periodic"]
        discretized_ext_indices = self.var_categories["extended"]
        var_categories = discretized_ext_indices + periodic_indices

        # constructing the grids
        parameters = dict.fromkeys(
            [f"θ{index}" for index in var_categories]
            + [var.name for var in self.external_fluxes]
            + [var.name for var in self.symbolic_params]
        )

        # filtering the plotting options
        plot_kwargs = {}
        list_of_keys = list(kwargs.keys())
        for key in list_of_keys:
            if key not in parameters:
                plot_kwargs[key] = kwargs[key]
                del kwargs[key]

        sweep_vars = {}
        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                sweep_vars[var_name] = kwargs[var_name]
        if len(sweep_vars) > 1:
            sweep_vars.update(zip(sweep_vars, np.meshgrid(*list(sweep_vars.values()))))
            for var_name in sweep_vars:
                parameters[var_name] = sweep_vars[var_name]

        if len(sweep_vars) > 2:
            raise AttributeError(
                "Cannot plot with a dimension greater than 3; Only give a maximum of "
                "two grid inputs"
            )

        potential_energies = self.potential_energy(**kwargs)

        fig, axes = kwargs.get("fig_ax") or plt.subplots()

        if len(sweep_vars) == 1:
            axes.plot(*(list(sweep_vars.values()) + [potential_energies]))
            axes.set_xlabel(
                r"$\theta_{{{}}}$".format(
                    get_trailing_number(list(sweep_vars.keys())[0])
                )
            )
            axes.set_ylabel("Potential energy in " + get_units())

        if len(sweep_vars) == 2:
            contourset = axes.contourf(
                *(list(sweep_vars.values()) + [potential_energies])
            )
            var_indices = [
                get_trailing_number(var_name) for var_name in list(sweep_vars.keys())
            ]
            axes.set_xlabel(r"$\theta_{{{}}}$".format(var_indices[0]))
            axes.set_ylabel(r"$\theta_{{{}}}$".format(var_indices[1]))
            cbar = plt.colorbar(contourset, ax=axes)
            cbar.set_label("Potential energy in " + get_units())
        _process_options(fig, axes, **plot_kwargs)
        return fig, axes

    # ****************************************************************
    # ************* Functions for plotting wave function *************
    # ****************************************************************
    def get_osc_param(self, var_index: int, which_param: str = "length") -> float:
        """
        Returns the oscillator parameters based on the oscillator used to diagonalize
        the Hamiltonian in the harmonic oscillator basis.

        Parameters
        ----------
        var_index:
            var index whose oscillator parameter needs to be fetched
        which_param:
            "length" or "freq" - decides which parameter is returned, by default
            "length"

        Returns
        -------
            returns the float value which is the oscillator length or the frequency of
            the oscillator corresponding to var_index depending on the string
            `which_param`.
        """
        if not self.hierarchical_diagonalization:
            return eval("self.osc_" + which_param + "s[" + str(var_index) + "]")

        subsystem = self.subsystems[self.get_subsystem_index(var_index)]
        return subsystem.get_osc_param(var_index, which_param=which_param)

    def _recursive_basis_change(
        self, wf_reshaped, wf_dim, subsystem, relevant_indices=None
    ):
        """
        Method to change the basis recursively, to reverse hierarchical diagonalization
        and get to the basis in which the variables were initially defined.

        Parameters
        ----------
        wf_dim:
            The dimension of the wave function which needs to be rewritten in terms of
            the initial basis

        """
        U_subsys = (
            subsystem.get_eigenstates()
        )  # eigensys(evals_count=subsystem.truncated_dim)
        wf_sublist = list(range(len(wf_reshaped.shape)))
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_new_basis = np.einsum(
            wf_reshaped, wf_sublist, U_subsys.T, U_sublist, target_sublist
        )
        if subsystem.hierarchical_diagonalization:
            wf_shape = list(wf_new_basis.shape)
            wf_shape[wf_dim] = [
                sub_subsys.truncated_dim for sub_subsys in subsystem.subsystems
            ]
            wf_new_basis = wf_new_basis.reshape(flatten_list_recursive(wf_shape))
            for sub_subsys_index, sub_subsys in enumerate(subsystem.subsystems):
                if len(set(relevant_indices) & set(sub_subsys.dynamic_var_indices)) > 0:
                    wf_new_basis = self._recursive_basis_change(
                        wf_new_basis,
                        wf_dim + sub_subsys_index,
                        sub_subsys,
                        relevant_indices=relevant_indices,
                    )
        else:
            if len(set(relevant_indices) & set(subsystem.dynamic_var_indices)) > 0:
                wf_shape = list(wf_new_basis.shape)
                wf_shape[wf_dim] = [
                    (
                        getattr(subsystem, cutoff_attrib)
                        if "ext" in cutoff_attrib
                        else (2 * getattr(subsystem, cutoff_attrib) + 1)
                    )
                    for cutoff_attrib in subsystem.cutoff_names
                ]
                wf_new_basis = wf_new_basis.reshape(flatten_list_recursive(wf_shape))
        return wf_new_basis

    def _basis_change_harm_osc_to_n(
        self, wf_original_basis, wf_dim, var_index, grid_n: discretization.Grid1d
    ):
        """
        Method to change the basis from harmonic oscillator to n basis
        """
        U_ho_n = np.array(
            [
                osc.harm_osc_wavefunction(
                    n,
                    grid_n.make_linspace(),
                    abs(self.get_osc_param(var_index, which_param="length")),
                )
                for n in range(getattr(self, "cutoff_ext_" + str(var_index)))
            ]
        )
        wf_sublist = [idx for idx, _ in enumerate(wf_original_basis.shape)]
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_new_basis = np.einsum(
            wf_original_basis, wf_sublist, U_ho_n.T, U_sublist, target_sublist
        )
        return wf_new_basis

    def _basis_change_harm_osc_to_phi(
        self, wf_original_basis, wf_dim, var_index, grid_phi: discretization.Grid1d
    ):
        """
        Method to change the basis from harmonic oscillator to phi basis
        """
        U_ho_phi = np.array(
            [
                osc.harm_osc_wavefunction(
                    n,
                    grid_phi.make_linspace(),
                    abs(self.get_osc_param(var_index, which_param="length")),
                )
                for n in range(getattr(self, "cutoff_ext_" + str(var_index)))
            ]
        )
        wf_sublist = [idx for idx, _ in enumerate(wf_original_basis.shape)]
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_ext_basis = np.einsum(
            wf_original_basis, wf_sublist, U_ho_phi, U_sublist, target_sublist
        )
        return wf_ext_basis

    def _basis_change_n_to_phi(
        self, wf_original_basis, wf_dim, var_index, grid_phi: discretization.Grid1d
    ):
        """
        Method to change the basis from harmonic oscillator to phi basis
        """
        U_n_phi = np.array(
            [
                np.exp(n * grid_phi.make_linspace() * 1j)
                for n in range(
                    -getattr(self, "cutoff_n_" + str(var_index)),
                    getattr(self, "cutoff_n_" + str(var_index)) + 1,
                )
            ]
        )
        wf_sublist = list(range(len(wf_original_basis.shape)))
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_ext_basis = np.einsum(
            wf_original_basis, wf_sublist, U_n_phi, U_sublist, target_sublist
        )
        return wf_ext_basis

    def _get_var_dim_for_reshaped_wf(self, wf_var_indices, var_index):
        wf_dim = 0
        if not self.hierarchical_diagonalization:
            return self.dynamic_var_indices.index(var_index)
        for subsys in self.subsystems:
            intersection = list_intersection(subsys.dynamic_var_indices, wf_var_indices)
            if len(intersection) > 0 and var_index not in intersection:
                if subsys.hierarchical_diagonalization:
                    wf_dim += subsys._get_var_dim_for_reshaped_wf(
                        wf_var_indices, var_index
                    )
                else:
                    wf_dim += len(subsys.dynamic_var_indices)
            elif len(intersection) > 0 and var_index in intersection:
                if subsys.hierarchical_diagonalization:
                    wf_dim += subsys._get_var_dim_for_reshaped_wf(
                        wf_var_indices, var_index
                    )
                else:
                    wf_dim += subsys.dynamic_var_indices.index(var_index)
                break
            else:
                wf_dim += 1
        return wf_dim

    def _dims_to_be_summed(self, var_indices: Tuple[int], num_wf_dims) -> List[int]:
        all_var_indices = self.dynamic_var_indices
        non_summed_dims = []
        for var_index in all_var_indices:
            if var_index in var_indices:
                non_summed_dims.append(
                    self._get_var_dim_for_reshaped_wf(var_indices, var_index)
                )
        return [dim for dim in range(num_wf_dims) if dim not in non_summed_dims]

    def _reshape_and_change_to_variable_basis(
        self, wf: ndarray, var_indices: Tuple[int]
    ) -> ndarray:
        """
        This method changes the basis of the wavefunction when hierarchical diagonalization is used.
        Then reshapes the wavefunction to represent each of the variable indices as a separate dimension.
        """
        if self.hierarchical_diagonalization:
            subsys_index_for_var_index = unique_elements_in_list(
                [self.get_subsystem_index(index) for index in var_indices]
            )  # getting the subsystem index for each of the variable indices
            subsys_index_for_var_index.sort()

            subsys_trunc_dims = [sys.truncated_dim for sys in self.subsystems]
            # reshaping the wave functions to truncated dims of subsystems
            wf_hd_reshaped = wf.reshape(*subsys_trunc_dims)

            # **** Converting to the basis in which the variables are defined *****
            wf_original_basis = wf_hd_reshaped
            for subsys_index in subsys_index_for_var_index:
                wf_dim = 0
                for sys_index in range(subsys_index):
                    if sys_index in subsys_index_for_var_index:
                        wf_dim += len(self.subsystems[sys_index].dynamic_var_indices)
                    else:
                        wf_dim += 1
                wf_original_basis = self._recursive_basis_change(
                    wf_original_basis,
                    wf_dim,
                    self.subsystems[subsys_index],
                    relevant_indices=var_indices,
                )
        else:
            wf_original_basis = wf.reshape(
                *[
                    (
                        getattr(self, cutoff_attrib)
                        if "ext" in cutoff_attrib
                        else (2 * getattr(self, cutoff_attrib) + 1)
                    )
                    for cutoff_attrib in self.cutoff_names
                ]
            )
        return wf_original_basis

    def _basis_for_var_index(self, var_index: int) -> str:
        """
        Returns the ext_basis of the subsystem with no further subsystems to which the
        var_index belongs.
        """
        if self.hierarchical_diagonalization:
            subsys = self.subsystems[self.get_subsystem_index(var_index)]
            return subsys._basis_for_var_index(var_index)
        else:
            if var_index in self.var_categories["extended"]:
                return self.ext_basis
            else:
                return "periodic"

    def _change_to_phi_basis(
        self,
        wf_original_basis: ndarray,
        var_indices: Tuple[int],
        grids_dict: Dict[int, Union[discretization.Grid1d, ndarray]],
        change_discrete_charge_to_phi: bool,
    ):
        """
        Changes the basis of the varaible indices to discretized phi basis which is amenable to plotting
        """
        wf_ext_basis = wf_original_basis
        for var_index in var_indices:
            # finding the dimension corresponding to the var_index
            if not self.hierarchical_diagonalization:
                wf_dim = self.dynamic_var_indices.index(var_index)
            else:
                wf_dim = self._get_var_dim_for_reshaped_wf(var_indices, var_index)

            var_basis = self._basis_for_var_index(var_index)

            if var_basis == "harmonic":
                wf_ext_basis = self._basis_change_harm_osc_to_phi(
                    wf_ext_basis, wf_dim, var_index, grids_dict[var_index]
                )
            elif var_basis == "periodic" and change_discrete_charge_to_phi:
                wf_ext_basis = self._basis_change_n_to_phi(
                    wf_ext_basis, wf_dim, var_index, grids_dict[var_index]
                )
        return wf_ext_basis

    def generate_wf_plot_data(
        self,
        which: int = 0,
        mode: str = "abs-sqr",
        var_indices: Tuple[int] = (1,),
        eigensys: ndarray = None,
        change_discrete_charge_to_phi: bool = True,
        grids_dict: Dict[int, discretization.Grid1d] = None,
    ):
        """
        Returns treated wave function of the current Circuit instance for the
        specified variables.

        Parameters
        ----------
        which:
            integer to choose which wave function to plot
        mode:
            "abs", "real", "imag", "abs-sqr" - decides which part of the wave
            function is plotted.
        var_indices:
            A tuple containing the indices of the variables chosen to plot the
            wave function in. Should not have more than 2 entries.
        eigensys:
            eigenvalues and eigenstates of the Circuit instance; if not provided,
            calling this method will perform a diagonalization to obtain these.
        extended_variable_basis: str
            The basis in which the extended variables are plotted. Can be either
            "phi" or "charge".
        periodic_variable_basis: str
            The basis in which the periodic variables are plotted. Can be either
            "phi" or "charge".
        grids_dict:
            A dictionary which pairs var indices with the requested grids used to create
            the plot.
        """
        # checking to see if eigensys needs to be generated
        if eigensys is None:
            _, wfs = self.eigensys(evals_count=which + 1)
        else:
            _, wfs = eigensys

        wf = wfs[:, which]
        # change the wf to the basis in which the variables were initially defined
        wf_original_basis = self._reshape_and_change_to_variable_basis(
            wf=wf, var_indices=var_indices
        )

        # making a basis change to the desired basis for every var_index
        wf_ext_basis = self._change_to_phi_basis(
            wf_original_basis,
            var_indices=var_indices,
            grids_dict=grids_dict,
            change_discrete_charge_to_phi=change_discrete_charge_to_phi,
        )

        # sum over the dimensions not relevant to the ones in var_indices
        # finding the dimensions which needs to be summed over
        dims_to_be_summed = self._dims_to_be_summed(
            var_indices, len(wf_ext_basis.shape)
        )
        # summing over the dimensions
        # summing over the dimensions
        if mode == "abs-sqr":
            wf_plot = np.sum(
                np.abs(wf_ext_basis) ** 2,
                axis=tuple(dims_to_be_summed),
            )
            return wf_plot
        if mode == "abs":
            if len(dims_to_be_summed) == 0:
                return np.abs(wf_ext_basis)
            else:
                raise AttributeError(
                    "Cannot plot the absolute value of the wave function in more than 2 dimensions."
                )
        elif mode == "real":
            if len(dims_to_be_summed) == 0:
                return np.real(wf_ext_basis)
            else:
                raise AttributeError(
                    "Cannot plot the real part of the wave function in more than 2 dimensions."
                )
        elif mode == "imag":
            if len(dims_to_be_summed) == 0:
                return np.imag(wf_ext_basis)
            else:
                raise AttributeError(
                    "Cannot plot the imaginary part of the wave function in more than 2 dimensions."
                )

    def plot_wavefunction(
        self,
        which=0,
        mode: str = "abs-sqr",
        var_indices: Tuple[int] = (1,),
        esys: Tuple[ndarray, ndarray] = None,
        change_discrete_charge_to_phi: bool = True,
        zero_calibrate: bool = True,
        grids_dict: Dict[int, discretization.Grid1d] = {},
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Returns the plot of the wavefunction in the requested variables.
        At most 2 numbers of variables for
        wavefunction can be specified as plotting axis. If the number of
        plotting variables for wave function is smaller than the number of
        variables in the circuit, the marginal probability distribution of the
        state with respect to the specified variables is plotted. This means
        the norm square of the wave function is integrated over the rest of the
        variables and is then plotted.

        Parameters
        ----------
        which:
            integer to choose which wave function to plot
        mode:
            "abs", "real", "imag", "abs-sqr" - decides which part of the wave function is plotted,
            by default "abs-sqr"
        var_indices:
            A tuple containing the indices of the variables chosen to plot the
            wave function in. It should not have more than 2 entries.
        esys:
            The object returned by the method `.eigensys`, is used to avoid the
            re-evaluation of the eigen systems if already evaluated.
        change_discrete_charge_to_phi:
            If True, the wave function is plotted in the phi basis for the periodic
            variables. If False, the wave function is plotted in the charge basis
            for the periodic variables.
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the
            neutral color in the palette
        grids_dict:
            A dictionary which pairs var indices with the grids used to create
            the plot. The way to specify the grids is as follows:
            1. For extended variables, the grids should be of type `discretization.Grid1d`.
            2. When the discretized phi basis is used for the extended variable, the grids
            used in the diagonalization is used to plot the wave function instead of
            the grids specified here.
            3. For periodic variables, only if `change_discrete_charge_to_phi` is True,
            the grid specified here will used for plotting. The grid is specified as an integer
            which is the number of points in the grid. The grid has a minimum and maximum value
            of -pi and pi respectively.
            4. If the grid is not specified for a variable that requires a grid for plotting (i.e.
            extended variable with harmonic oscillator basis, or periodic variable with
            `change_discrete_charge_to_phi` set to True), the default grid is used.

        **kwargs:
            plotting parameters

        Returns
        -------
            Returns a axes and figure for further editing.
        """
        if len(var_indices) > 2:
            raise AttributeError(
                "Cannot plot wave function in more than 2 dimensions. The number of "
                "dimensions should be less than 2."
            )
        var_indices = np.sort(var_indices)
        grids_per_varindex_dict = grids_dict or self.discretized_grids_dict_for_vars()

        plot_data = self.generate_wf_plot_data(
            which=which,
            mode=mode,
            var_indices=var_indices,
            eigensys=esys,
            change_discrete_charge_to_phi=change_discrete_charge_to_phi,
            grids_dict=grids_per_varindex_dict,
        )

        var_types = []

        for var_index in var_indices:
            if var_index in self.var_categories["periodic"]:
                if not change_discrete_charge_to_phi:
                    var_types.append("Charge in units of 2e, periodic variable:")
                else:
                    var_types.append("Dimensionless flux, periodic variable:")
            if var_index in self.var_categories["extended"]:
                var_types.append("Dimensionless flux, extended variable:")

        if len(var_indices) == 1:
            return self._plot_wf_pdf_1D(
                plot_data,
                mode,
                var_indices,
                grids_per_varindex_dict,
                change_discrete_charge_to_phi,
                kwargs,
            )

        elif len(var_indices) == 2:
            return self._plot_wf_pdf_2D(
                plot_data,
                var_indices,
                grids_per_varindex_dict,
                change_discrete_charge_to_phi,
                zero_calibrate=zero_calibrate,
                kwargs=kwargs,
            )

    def _plot_wf_pdf_2D(
        self,
        wf_plot: ndarray,
        var_indices,
        grids_per_varindex_dict,
        change_discrete_charge_to_phi: bool,
        zero_calibrate: bool,
        kwargs,
    ) -> Tuple[Figure, Axes]:
        # check if each variable is periodic
        grids = []
        labels = []
        for index_order in [1, 0]:
            if not change_discrete_charge_to_phi and (
                var_indices[index_order] in self.var_categories["periodic"]
            ):
                grids.append(
                    [
                        -getattr(self, "cutoff_n_" + str(var_indices[index_order])),
                        getattr(self, "cutoff_n_" + str(var_indices[index_order])),
                        2 * getattr(self, "cutoff_n_" + str(var_indices[index_order]))
                        + 1,
                    ]
                )
                labels.append(r"$n_{{{}}}$".format(str(var_indices[index_order])))
            else:
                grids.append(
                    list(
                        grids_per_varindex_dict[var_indices[index_order]]
                        .get_initdata()
                        .values()
                    ),
                )
                labels.append(r"$\theta_{{{}}}$".format(str(var_indices[index_order])))
        wavefunc_grid = discretization.GridSpec(np.asarray(grids))

        wavefunc = storage.WaveFunctionOnGrid(wavefunc_grid, wf_plot)
        # obtain fig and axes from
        fig, axes = plot.wavefunction2d(
            wavefunc,
            zero_calibrate=zero_calibrate,
            ylabel=labels[1],
            xlabel=labels[0],
            **kwargs,
        )
        # change frequency of tick mark for variables in charge basis
        # also force the tick marks to be integers
        if not change_discrete_charge_to_phi:
            if var_indices[0] in self.var_categories["periodic"]:
                if getattr(self, "cutoff_n_" + str(var_indices[0])) >= 6:
                    axes.yaxis.set_major_locator(plt.MaxNLocator(13, integer=True))
                else:
                    axes.yaxis.set_major_locator(
                        plt.MaxNLocator(
                            1 + 2 * getattr(self, "cutoff_n_" + str(var_indices[0])),
                            integer=True,
                        )
                    )
            if var_indices[1] in self.var_categories["periodic"]:
                if getattr(self, "cutoff_n_" + str(var_indices[1])) >= 15:
                    axes.xaxis.set_major_locator(plt.MaxNLocator(31, integer=True))
                else:
                    axes.xaxis.set_major_locator(
                        plt.MaxNLocator(
                            1 + 2 * getattr(self, "cutoff_n_" + str(var_indices[1])),
                            integer=True,
                        )
                    )

        return fig, axes

    def _plot_wf_pdf_1D(
        self,
        wf_plot: ndarray,
        mode: str,
        var_indices,
        grids_per_varindex_dict,
        change_discrete_charge_to_phi: bool,
        kwargs,
    ) -> Tuple[Figure, Axes]:
        var_index = var_indices[0]

        if not change_discrete_charge_to_phi and (
            var_indices[0] in self.var_categories["periodic"]
        ):
            ncut = self.cutoffs_dict()[var_indices[0]]
            wavefunc = storage.WaveFunction(
                basis_labels=np.linspace(-ncut, ncut, 2 * ncut + 1),
                amplitudes=wf_plot,
            )
            kwargs = {
                **defaults.wavefunction1d_discrete("abs_sqr"),
                **kwargs,
            }
            wavefunc.basis_labels = np.arange(
                -getattr(self, "cutoff_n_" + str(var_index)),
                getattr(self, "cutoff_n_" + str(var_index)) + 1,
            )
            fig, axes = plot.wavefunction1d_discrete(wavefunc, **kwargs)
            # changing the tick frequency for axes
            if getattr(self, "cutoff_n_" + str(var_index)) >= 7:
                axes.xaxis.set_major_locator(plt.MaxNLocator(15, integer=True))
            else:
                axes.xaxis.set_major_locator(
                    plt.MaxNLocator(1 + 2 * getattr(self, "cutoff_n_" + str(var_index)))
                )
        else:
            wavefunc = storage.WaveFunction(
                basis_labels=grids_per_varindex_dict[var_indices[0]].make_linspace(),
                amplitudes=wf_plot,
            )
            if mode == "abs":
                ylabel = r"$|\psi(\theta_{{{}}})|$".format(str(var_indices[0]))
            elif mode == "abs-sqr":
                ylabel = r"$|\psi(\theta_{{{}}})|^2$".format(str(var_indices[0]))
            elif mode == "real":
                ylabel = r"$\mathrm{{Re}}(\psi(\theta_{{{}}}))$".format(
                    str(var_indices[0])
                )
            elif mode == "imag":
                ylabel = r"$\mathrm{{Im}}(\psi(\theta_{{{}}}))$".format(
                    str(var_indices[0])
                )
            fig, axes = plot.wavefunction1d_nopotential(
                wavefunc,
                0,
                xlabel=r"$\theta_{{{}}}$".format(str(var_indices[0])),
                ylabel=ylabel,
                **kwargs,
            )
        return fig, axes

    def _get_cutoff_value(self, var_index: int) -> int:
        """Return the cutoff value associated with the variable with integer index
        `var_index`."""
        for cutoff_name in self.parent.cutoff_names:
            if str(var_index) in cutoff_name:
                return getattr(self.parent, cutoff_name)

    def operator_names_in_hamiltonian_symbolic(self) -> List[str]:
        """
        Returns a list of the names (strings) of all operators
        occurring in the symbolic Hamiltonian.
        """
        return [
            symbol.name
            for symbol in self.hamiltonian_symbolic.free_symbols
            if ("ng" not in symbol.name and "Φ" not in symbol.name)
            and symbol not in self.symbolic_params
        ]
