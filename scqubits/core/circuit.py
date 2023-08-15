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
import warnings
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import qutip as qt
import scipy as sp
import sympy as sm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix
from sympy import latex

try:
    from IPython.display import Latex, display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

import scqubits as scq
import scqubits.core.discretization as discretization
import scqubits.core.oscillator as osc
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as utils
from scqubits import HilbertSpace, settings
from scqubits.core import operators as op
from scqubits.core.circuit_utils import (
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
    compose,
    get_operator_number,
    get_trailing_number,
    grid_operator_func_factory,
    is_potential_term,
    matrix_power_sparse,
    operator_func_factory,
)
from scqubits.core.symbolic_circuit import Branch, SymbolicCircuit
from scqubits.io_utils.fileio import IOData
from scqubits.io_utils.fileio_serializers import dict_serialize
from scqubits.utils.misc import (
    flatten_list,
    flatten_list_recursive,
    list_intersection,
    number_of_lists_in_list,
)
from scqubits.utils.plot_utils import _process_options
from scqubits.utils.spectrum_utils import (
    convert_matrix_to_qobj,
    identity_wrap,
    order_eigensystem,
)


class Subsystem(base.QubitBaseClass, serializers.Serializable):
    """
    Defines a subsystem for a circuit, which can further be used recursively to define
    subsystems within subsystem.

    Parameters
    ----------
    parent: Subsystem
        the instance under which the new subsystem is defined.
    hamiltonian_symbolic: sm.Expr
        The symbolic expression which defines the Hamiltonian for the new subsystem
    system_hierarchy: Optional[List], optional
        Defines the hierarchy of the new subsystem, is set to None when hierarchical
        diagonalization is not required. by default None
    subsystem_trunc_dims: Optional[List], optional
        Defines the truncated dimensions for the subsystems inside the current
        subsystem, is set to None when hierarchical diagonalization is not required,
        by default `None`
    truncated_dim: Optional[int], optional
        sets the truncated dimension for the current subsystem, set to 10 by default.
    """

    # switch used in protecting the class from erroneous addition of new attributes
    _frozen = False

    def __init__(
        self,
        parent: "Subsystem",
        hamiltonian_symbolic: sm.Expr,
        system_hierarchy: Optional[List] = None,
        subsystem_trunc_dims: Optional[List] = None,
        truncated_dim: Optional[int] = 10,
    ):
        base.QuantumSystem.__init__(self, id_str=None)

        # This class does not yet support custom diagonalization options, but these
        # still have to be defined
        self.evals_method = None
        self.evals_method_options = None
        self.esys_method = None
        self.esys_method_options = None

        self.system_hierarchy = system_hierarchy
        self.truncated_dim = truncated_dim
        self.subsystem_trunc_dims = subsystem_trunc_dims

        self.is_child = True
        self.parent = parent
        self.hamiltonian_symbolic = hamiltonian_symbolic
        self._hamiltonian_sym_for_numerics = hamiltonian_symbolic
        self._default_grid_phi = self.parent._default_grid_phi

        self.junction_potential = None
        self._H_LC_str_harmonic = None

        self.ext_basis: str = self.parent.ext_basis
        self.external_fluxes = [
            var
            for var in self.parent.external_fluxes
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.offset_charges = [
            var
            for var in self.parent.offset_charges
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.symbolic_params = {
            var: self.parent.symbolic_params[var]
            for var in self.parent.symbolic_params
            if var in self.hamiltonian_symbolic.free_symbols
        }

        self.var_categories_list: List[int] = []
        cutoffs: List[int] = []
        for var_name in self.operator_names_in_hamiltonian_symbolic():
            var_index = get_trailing_number(var_name)
            if var_index not in self.var_categories_list and var_index is not None:
                self.var_categories_list.append(var_index)
                cutoffs += [self.parent.cutoffs_dict()[var_index]]

        self.var_categories_list.sort()

        self.var_categories: Dict[str, List[int]] = {}
        for var_type in self.parent.var_categories:
            self.var_categories[var_type] = [
                var_index
                for var_index in self.parent.var_categories[var_type]
                if var_index in self.var_categories_list
            ]

        self.cutoff_names: List[str] = []
        for var_type in self.var_categories.keys():
            if var_type == "periodic":
                for var_index in self.var_categories["periodic"]:
                    self.cutoff_names.append(f"cutoff_n_{var_index}")
            if var_type == "extended":
                for var_index in self.var_categories["extended"]:
                    self.cutoff_names.append(f"cutoff_ext_{var_index}")

        self.discretized_phi_range: Dict[int, Tuple[float]] = {
            idx: self.parent.discretized_phi_range[idx]
            for idx in self.parent.discretized_phi_range
            if idx in self.var_categories_list
        }

        # storing the potential terms separately
        # and bringing the potential into the same form as for the class Circuit
        potential_symbolic = 0 * sm.symbols("x")
        for term in self.hamiltonian_symbolic.as_ordered_terms():
            if is_potential_term(term):
                potential_symbolic += term
        for i in self.var_categories_list:
            potential_symbolic = (
                potential_symbolic.replace(
                    sm.symbols(f"cosθ{i}"), sm.cos(1.0 * sm.symbols(f"θ{i}"))
                )
                .replace(sm.symbols(f"sinθ{i}"), sm.sin(1.0 * sm.symbols(f"θ{i}")))
                .subs(sm.symbols("I"), 1 / (2 * np.pi))
            )

        self.potential_symbolic = potential_symbolic

        self.hierarchical_diagonalization: bool = (
            system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
        )

        if len(self.var_categories_list) == 1 and self.ext_basis == "harmonic":
            self.type_of_matrices = "dense"
        else:
            self.type_of_matrices = "sparse"

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []

        # attributes for purely harmonic
        self.normal_mode_freqs = []

        self._configure()
        self._frozen = True

    def __setattr__(self, name, value):
        if not self._frozen or name in dir(self):
            super().__setattr__(name, value)
        else:
            raise Exception("Creating new attributes is disabled.")

    def __repr__(self) -> str:
        return self._id_str

    def __reduce__(self):
        # needed for multiprocessing / proper pickling
        pickle_func, pickle_args, pickled_state = super().__reduce__()
        new_pickled_state = {
            key: value for key, value in pickled_state.items() if "_operator" not in key
        }
        new_pickled_state["_frozen"] = False

        pickled_properties = {
            property_name: property_obj
            for property_name, property_obj in self.__class__.__dict__.items()
            if isinstance(property_obj, property)
        }

        return pickle_func, pickle_args, (new_pickled_state, pickled_properties)

    def __setstate__(self, state):
        # needed for multiprocessing / proper unpickling
        pickled_attribs, pickled_properties = state
        self._frozen = False

        self.__dict__.update(pickled_attribs)
        self.operators_by_name = self.set_operators()

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

        for var_index in self.var_categories_list:
            for cutoff_name in self.cutoff_names:
                if str(var_index) in cutoff_name:
                    cutoffs_dict[var_index] = getattr(self, cutoff_name)
        return cutoffs_dict

    def _regenerate_sym_hamiltonian(self) -> None:
        """
        Regenerates the system Hamiltonian from the symbolic circuit when needed (for
        example when the circuit is large and circuit parameters are changed).
        """
        if (
            not self.is_child
            and (len(self.symbolic_circuit.nodes)) > settings.SYM_INVERSION_MAX_NODES
        ):
            self.hamiltonian_symbolic = (
                self.symbolic_circuit.generate_symbolic_hamiltonian(
                    substitute_params=True
                )
            )
            self.generate_hamiltonian_sym_for_numerics()

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
        if not (np.isrealobj(value) and value > 0):
            raise AttributeError(
                f"'{value}' is invalid. Branch parameters must be positive and real."
            )
        setattr(self, f"_{param_name}", value)

        # update the attribute for the instance in symbolic_circuit
        # generate _hamiltonian_sym_for_numerics if not already generated, delayed for
        # large circuits
        if (
            not self.is_child
            and (len(self.symbolic_circuit.nodes)) > settings.SYM_INVERSION_MAX_NODES
        ) or self.is_purely_harmonic:
            self.symbolic_circuit.update_param_init_val(param_name, value)
            self._regenerate_sym_hamiltonian()

        # update Circuit instance

        # if purely harmonic the circuit attributes should change
        if self.is_purely_harmonic and isinstance(self, Circuit):
            self.potential_symbolic = self.symbolic_circuit.potential_symbolic
            self.transformation_matrix = self.symbolic_circuit.transformation_matrix
            self.normal_mode_freqs = self.symbolic_circuit.normal_mode_freqs

        if self.hierarchical_diagonalization:
            self.generate_subsystems()
            self.operators_by_name = self.set_operators()
            self.updated_subsystem_indices = list(range(len(self.subsystems)))
        else:
            self.operators_by_name = self.set_operators()

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

        # update all subsystem instances
        if self.hierarchical_diagonalization:
            for subsys_idx, subsys in enumerate(self.subsystems.values()):
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
            for subsys_idx, subsys in enumerate(self.subsystems.values()):
                if hasattr(subsys, param_name):
                    self._store_updated_subsystem_index(subsys_idx)
                    setattr(subsys, param_name, value)

    def _make_property(
        self, attrib_name: str, init_val: Union[int, float], property_update_type: str
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
                return obj._set_property_and_update_param_vars(name, value)

        elif property_update_type == "update_external_flux_or_charge":

            def setter(obj, value, name=attrib_name):
                return obj._set_property_and_update_ext_flux_or_charge(name, value)

        elif property_update_type == "update_cutoffs":

            def setter(obj, value, name=attrib_name):
                return obj._set_property_and_update_cutoffs(name, value)

        setattr(self.__class__, attrib_name, property(fget=getter, fset=setter))

    def _configure(self) -> None:
        """
        Function which is used to initiate the subsystem instance.
        """

        for x, param in enumerate(self.symbolic_params):
            # if harmonic oscillator basis is used, param vars become class properties.
            self._make_property(
                param.name, getattr(self.parent, param.name), "update_param_vars"
            )

        # getting attributes from parent
        for flux in self.external_fluxes:
            self._make_property(
                flux.name,
                getattr(self.parent, flux.name),
                "update_external_flux_or_charge",
            )
        for offset_charge in self.offset_charges:
            self._make_property(
                offset_charge.name,
                getattr(self.parent, offset_charge.name),
                "update_external_flux_or_charge",
            )

        for cutoff_str in self.cutoff_names:
            self._make_property(
                cutoff_str, getattr(self.parent, cutoff_str), "update_cutoffs"
            )

        # Creating the attributes for purely harmonic circuits
        self.is_purely_harmonic = self.parent.is_purely_harmonic
        if (
            self.is_purely_harmonic
        ):  # assuming that the parent has only extended variables and are ordered
            # starting from 1, 2, 3, ...
            self.normal_mode_freqs = self.parent.normal_mode_freqs[
                [var_idx - 1 for var_idx in self.var_categories["extended"]]
            ]

        self._set_vars()
        if self.hierarchical_diagonalization:
            # attribute to note updated subsystem indices
            self.updated_subsystem_indices = []

            self.generate_subsystems()
            self._check_truncation_indices()
            self.operators_by_name = self.set_operators()
            self.updated_subsystem_indices = list(range(len(self.subsystems)))
        else:
            self.operators_by_name = self.set_operators()

    def _store_updated_subsystem_index(self, index: int) -> None:
        if not self.hierarchical_diagonalization:
            raise Exception(f"The subsystem provided to {self} has no subsystems.")
        if index not in self.updated_subsystem_indices:
            self.updated_subsystem_indices.append(index)

    # *****************************************************************
    # **** Functions to construct the operators for the Hamiltonian ****
    # *****************************************************************
    def grids_dict_for_discretized_extended_vars(self):
        cutoffs_dict = self.cutoffs_dict()
        grids = {}
        for i in self.var_categories["extended"]:
            grids[i] = discretization.Grid1d(
                self.discretized_phi_range[i][0],
                self.discretized_phi_range[i][1],
                cutoffs_dict[i],
            )
        return grids

    def _constants_in_subsys(self, H_sys: sm.Expr) -> sm.Expr:
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
        constant_terms = self._constant_terms_in_hamiltonian.copy()
        for term in constant_terms:
            if set(term.free_symbols) & subsys_free_symbols == set(term.free_symbols):
                constant_expr += term
                self._constant_terms_in_hamiltonian.remove(term)
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

        for subsystem_idx, subsystem in self.subsystems.items():
            if subsystem.truncated_dim >= subsystem.hilbertdim() - 1:
                # find the correct position of the subsystem where the truncation
                # index  is too big
                subsystem_position = f"subsystem {subsystem_idx} "
                parent = subsystem.parent
                while parent.is_child:
                    grandparent = parent.parent
                    # find the subsystem position of the parent system
                    subsystem_position += f"of subsystem {grandparent.get_subsystem_index(parent.var_categories_list[0])} "
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

    def generate_subsystems(self):
        """
        Generates the subsystems (child instances of Circuit) depending on the attribute
        `self.system_hierarchy`
        """
        hamiltonian = self._hamiltonian_sym_for_numerics
        # collecting constants
        constants = self._list_of_constants_from_expr(hamiltonian)
        self._constant_terms_in_hamiltonian = constants
        for const in constants:
            hamiltonian -= const

        systems_sym = []
        interaction_sym = []

        non_operator_symbols = (
            self.offset_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
            + [sm.symbols("I")]
        )

        for subsys_index_list in self.system_hierarchy:
            subsys_index_list = flatten_list_recursive(subsys_index_list)

            hamiltonian_terms = hamiltonian.as_ordered_terms()

            H_sys = 0 * sm.symbols("x")
            H_int = 0 * sm.symbols("x")
            for term in hamiltonian_terms:
                term_operator_indices = [
                    get_trailing_number(var_sym.name)
                    for var_sym in term.free_symbols
                    if var_sym not in non_operator_symbols
                ]
                term_operator_indices_unique = list(set(term_operator_indices))

                if len(set(term_operator_indices_unique) - set(subsys_index_list)) == 0:
                    H_sys += term

                if (
                    len(set(term_operator_indices_unique) - set(subsys_index_list)) > 0
                    and len(set(term_operator_indices_unique) & set(subsys_index_list))
                    > 0
                ):
                    H_int += term

            # adding constants
            systems_sym.append(H_sys + self._constants_in_subsys(H_sys))
            interaction_sym.append(H_int)
            hamiltonian -= H_sys + H_int  # removing the terms added to a subsystem

        if len(constants) > 0:
            systems_sym[0] += sum(constants)
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
                [interaction_sym[index] for index in range(len(self.system_hierarchy))],
            )
        )

        self.subsystems: Dict[int, "Subsystem"] = dict(
            zip(
                range(len(self.system_hierarchy)),
                [
                    Subsystem(
                        self,
                        systems_sym[index],
                        system_hierarchy=self.system_hierarchy[index],
                        truncated_dim=self.subsystem_trunc_dims[index][0]
                        if type(self.subsystem_trunc_dims[index]) == list
                        else self.subsystem_trunc_dims[index],
                        subsystem_trunc_dims=self.subsystem_trunc_dims[index][1]
                        if type(self.subsystem_trunc_dims[index]) == list
                        else None,
                    )
                    for index in range(len(self.system_hierarchy))
                ],
            )
        )

        self.hilbert_space = HilbertSpace(
            [self.subsystems[i] for i in range(len(self.system_hierarchy))]
        )

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

    def build_hilbertspace(
        self, update_subsystem_indices: Optional[List[int]] = None
    ) -> None:
        """
        Builds the HilbertSpace object for the `Circuit` instance if
        `hierarchical_diagonalization` is set to true.

        Parameters
        ----------
        update_subsystem_indices:
            List of subsystem indices which need to be updated. If set to None, all the
           are updated.
        """
        # generate lookup table in HilbertSpace
        _ = self.hilbert_space.generate_bare_esys(
            update_subsystem_indices=update_subsystem_indices
        )

        self.hilbert_space.interaction_list = []

        # Adding interactions using the symbolic interaction term
        for sys_index in range(len(self.system_hierarchy)):
            interaction = self.subsystem_interactions[sys_index].expand()
            if interaction == 0:  # if the interaction term is zero
                continue
            # modifying interaction terms:
            # substituting all the external flux, offset charge and branch parameters.
            interaction = interaction.subs(
                [
                    (param, getattr(self, str(param)))
                    for param in list(self.symbolic_params.keys())
                    + self.external_fluxes
                    + self.offset_charges
                ]
            )
            #   - substituting Identity with 1
            interaction = interaction.subs("I", 1)

            expr_dict = interaction.as_coefficients_dict()
            interaction_terms = list(expr_dict.keys())

            for idx, term in enumerate(interaction_terms):
                coefficient_sympy = expr_dict[term]
                self.hilbert_space.add_interaction(
                    qobj=float(coefficient_sympy)
                    * self._interaction_operator_from_expression(term),
                    check_validity=False,
                )

    def _interaction_operator_from_expression(self, symbolic_interaction_term: sm.Expr):
        """
        Returns the matrix which has the hilbert dimension equal to the hilbert
        dimension of the parent. Note that this method cannot deal with a coefficient
        which is different from 1. That should be dealt with externally.

        Parameters
        ----------
        symbolic_interaction_term:
            The symbolic expression which has the interaction terms.
        """

        non_operator_symbols = (
            self.offset_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
        )

        # substitute all non_operator_symbols
        for var_sym in non_operator_symbols:
            symbolic_interaction_term = symbolic_interaction_term.subs(
                var_sym, getattr(self, var_sym.name)
            )

        if symbolic_interaction_term.has(sm.cos):
            return self._evaluate_matrix_cosine_terms(symbolic_interaction_term)

        term_var_indices = [
            get_trailing_number(var_sym.name)
            for var_sym in symbolic_interaction_term.free_symbols
            if var_sym not in non_operator_symbols
        ]

        term_operator_syms = [
            var_sym
            for var_sym in symbolic_interaction_term.free_symbols
            if var_sym not in non_operator_symbols
        ]

        interacting_subsystem_indices = set(
            [self.get_subsystem_index(idx) for idx in term_var_indices]
        )

        operator_dict = dict.fromkeys([idx for idx, _ in enumerate(self.subsystems)])

        for subsys_index in operator_dict:
            operator_dict[subsys_index] = qt.tensor(
                [
                    qt.identity(subsys.truncated_dim)
                    for subsys in list(self.subsystems.values())
                ]
            )
            if subsys_index in interacting_subsystem_indices:
                for operator_sym in term_operator_syms:
                    if (
                        self.get_subsystem_index(get_trailing_number(operator_sym.name))
                        == subsys_index
                    ):
                        operator_matrix = self.subsystems[
                            subsys_index
                        ].get_operator_by_name(operator_sym.name)
                        if isinstance(operator_matrix, qt.Qobj):
                            operator_matrix = operator_matrix.data.tocsc()
                        operator_dict[subsys_index] *= identity_wrap(
                            operator_matrix,
                            self.subsystems[subsys_index],
                            list(self.subsystems.values()),
                            evecs=self.subsystems[subsys_index].get_eigenstates(),
                        )
        operator_list = list(operator_dict.values())
        return functools.reduce(builtin_op.mul, operator_list)

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
        objects for all the operators present in the circuit
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
            if hamiltonian.coeff(f"θ{var_index}") != 0:
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

        flux_shifts = sm.linsolve(
            flux_shift_equations, tuple(flux_shift_vars.values())
        )  # solving for the flux offsets

        if len(flux_shifts) != 0:
            flux_shifts = list(list(flux_shifts)[0])
        else:
            flux_shifts = []

        flux_shifts_dict = dict(zip(list(flux_shift_vars.keys()), list(flux_shifts)))

        hamiltonian = hamiltonian.subs(
            [
                (sm.symbols("Δθ" + str(var_index)), flux_shifts_dict[var_index])
                for var_index in flux_shifts_dict.keys()
            ]
        )  # substituting the flux offsets to remove the linear terms
        hamiltonian = hamiltonian.subs(
            [(var, 0) for var in flux_shift_vars.values()]
        )  # removing the constants from the Hamiltonian

        flux_shifts_dict.update(
            {
                var_index: 0
                for var_index in self.var_categories["extended"]
                if var_index not in flux_shifts_dict
            }
        )
        # remove constants from Hamiltonian
        hamiltonian -= hamiltonian.as_coefficients_dict()[1]
        return hamiltonian.expand()
        # * ##########################################################################

    def generate_hamiltonian_sym_for_numerics(self):
        """
        Generates a symbolic expression which is ready for numerical evaluation starting
        from the expression stored in the attribute hamiltonian_symbolic. Stores the
        result in the attribute _hamiltonian_sym_for_numerics.
        """

        hamiltonian = (
            self.hamiltonian_symbolic.expand()
        )  # applying expand is critical; otherwise the replacement of p^2 with ps2
        # would not succeed

        # shifting the potential to the point of external fluxes
        hamiltonian = self._shift_harmonic_oscillator_potential(hamiltonian)

        if self.ext_basis == "discretized":
            # marking the squared momentum operators with a separate symbol
            for i in self.var_categories["extended"]:
                hamiltonian = hamiltonian.replace(
                    sm.symbols(f"Q{i}") ** 2, sm.symbols("Qs" + str(i))
                )

        # removing the constants from the Hamiltonian
        ordered_terms = hamiltonian.as_ordered_terms()
        constants = [
            term
            for term in ordered_terms
            if (
                set(
                    self.external_fluxes
                    + self.offset_charges
                    + list(self.symbolic_params.keys())
                    + [sm.symbols("I")]
                )
                & set(term.free_symbols)
            )
            == set(term.free_symbols)
        ]
        self._constant_terms_in_hamiltonian = constants
        for const in constants:
            hamiltonian -= const

        # associate an identity matrix with the external flux vars
        for ext_flux in self.external_fluxes:
            hamiltonian = hamiltonian.subs(
                ext_flux, ext_flux * sm.symbols("I") * 2 * np.pi
            )

        # associate an identity matrix with offset charge vars
        for offset_charge in self.offset_charges:
            hamiltonian = hamiltonian.subs(
                offset_charge, offset_charge * sm.symbols("I")
            )

        # finding the cosine terms
        cos_terms = sum(
            [term for term in hamiltonian.as_ordered_terms() if "cos" in str(term)]
        )
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
        self, operator: Union[csc_matrix, ndarray], index: int
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
        var_index_list = (
            self.var_categories["periodic"] + self.var_categories["extended"]
        )

        cutoff_names = np.fromiter(self._collect_cutoff_values(), dtype=int)  # [

        if self.type_of_matrices == "dense":
            matrix_format = "array"
        elif self.type_of_matrices == "sparse":
            matrix_format = "csc"

        if len(var_index_list) > 1:
            if index > var_index_list[0]:
                identity_left = sparse.identity(
                    np.prod(cutoff_names[: var_index_list.index(index)]),
                    format=matrix_format,
                )
            if index < var_index_list[-1]:
                identity_right = sparse.identity(
                    np.prod(cutoff_names[var_index_list.index(index) + 1 :]),
                    format=matrix_format,
                )

            if index == var_index_list[0]:
                return sparse.kron(operator, identity_right, format=matrix_format)
            elif index == var_index_list[-1]:
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

        subsys_trunc_dims = [
            subsys.truncated_dim for subsys in list(self.subsystems.values())
        ]

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

    def exp_i_pos_operator(
        self, var_sym: sm.Symbol, prefactor: float
    ) -> Union[csc_matrix, ndarray]:
        """
        Returns the bare operator exp(i*\theta*prefactor), without the kron product.
        Needs the oscillator lengths to be set in the attribute, `osc_lengths`,
        when `ext_basis` is set to "harmonic".
        """
        var_index = get_trailing_number(var_sym.name)

        if var_index in self.var_categories["periodic"]:
            if abs(prefactor) != 1:
                raise Exception("Prefactor for periodic variable should be 1.")
            if prefactor > 0:
                exp_i_theta = _exp_i_theta_operator(self.cutoffs_dict()[var_index])
            else:
                exp_i_theta = _exp_i_theta_operator_conjugate(
                    self.cutoffs_dict()[var_index]
                )
        elif var_index in self.var_categories["extended"]:
            if self.ext_basis == "discretized":
                phi_grid = discretization.Grid1d(
                    self.discretized_phi_range[var_index][0],
                    self.discretized_phi_range[var_index][1],
                    self.cutoffs_dict()[var_index],
                )
                diagonal = np.exp(phi_grid.make_linspace() * prefactor * 1j)
                exp_i_theta = sparse.dia_matrix(
                    (diagonal, [0]), shape=(phi_grid.pt_count, phi_grid.pt_count)
                ).tocsc()
            elif self.ext_basis == "harmonic":
                osc_length = self.osc_lengths[var_index]
                pos_operator = (osc_length / 2**0.5) * (
                    op.creation(self.cutoffs_dict()[var_index])
                    + op.annihilation(self.cutoffs_dict()[var_index])
                )
                exp_i_theta = sp.linalg.expm(pos_operator * prefactor * 1j)

        return self._sparsity_adaptive(exp_i_theta)

    def _evaluate_matrix_cosine_terms(self, junction_potential: sm.Expr) -> qt.Qobj:
        if self.hierarchical_diagonalization:
            subsystem_list = list(self.subsystems.values())
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
                arg.args[0] for arg in (1.0 * cos_term).args if arg.has(sm.cos)
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
                operator_list.append(
                    self.identity_wrap_for_hd(
                        self.exp_i_pos_operator(var_symbol, prefactor), var_indices[idx]
                    )
                )

            cos_term_operator = coefficient * functools.reduce(
                builtin_op.mul,
                operator_list,
            )

            junction_potential_matrix += (
                cos_term_operator + cos_term_operator.dag()
            ) * 0.5

        return junction_potential_matrix

    def circuit_operator_functions(self) -> Dict[str, Callable]:
        """
        Returns the set of operator functions to be turned into methods of the `Circuit`
        class.
        """
        periodic_vars = self.vars["periodic"]
        extended_vars = self.vars["extended"]

        # constructing the operators for extended variables
        extended_operators = {}
        if self.ext_basis == "discretized":
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

        else:  # expect that self.ext_basis is "harmonic":
            hamiltonian = self._hamiltonian_sym_for_numerics
            # substitute all the parameter values
            hamiltonian = hamiltonian.subs(
                [
                    (param, getattr(self, str(param)))
                    for param in list(self.symbolic_params.keys())
                    + self.external_fluxes
                    + self.offset_charges
                ]
            )

            osc_lengths = {}
            osc_freqs = {}
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
                ECi = float(hamiltonian.coeff(f"Q{var_index}**2").cancel()) / 4
                ELi = float(hamiltonian.coeff(f"θ{var_index}**2").cancel()) * 2
                osc_freqs[var_index] = (8 * ELi * ECi) ** 0.5
                osc_lengths[var_index] = (8.0 * ECi / ELi) ** 0.25
                nonwrapped_ops["position"] = functools.partial(
                    op.a_plus_adag_sparse, prefactor=osc_lengths[var_index] / (2**0.5)
                )
                nonwrapped_ops["sin"] = compose(
                    sp.linalg.sinm,
                    functools.partial(
                        op.a_plus_adag, prefactor=osc_lengths[var_index] / (2**0.5)
                    ),
                )
                nonwrapped_ops["cos"] = compose(
                    sp.linalg.cosm,
                    functools.partial(
                        op.a_plus_adag, prefactor=osc_lengths[var_index] / (2**0.5)
                    ),
                )
                nonwrapped_ops["momentum"] = functools.partial(
                    op.ia_minus_iadag_sparse,
                    prefactor=-1 / (osc_lengths[var_index] * 2**0.5),
                )

                for short_op_name in nonwrapped_ops.keys():
                    op_func = nonwrapped_ops[short_op_name]
                    sym_variable = extended_vars[short_op_name][list_idx]
                    op_name = sym_variable.name + "_operator"
                    extended_operators[op_name] = operator_func_factory(
                        op_func, var_index
                    )

            self.osc_lengths = osc_lengths
            self.osc_freqs = osc_freqs

        # constructing the operators for periodic variables
        periodic_operators = {}
        nonwrapped_ops = {
            "sin": _sin_theta,
            "cos": _cos_theta,
            "number": _n_theta_operator,
        }
        for short_op_name, op_func in nonwrapped_ops.items():
            for sym_variable in periodic_vars[short_op_name]:
                index = get_operator_number(sym_variable.name)
                op_name = sym_variable.name + "_operator"
                periodic_operators[op_name] = operator_func_factory(op_func, index)

        return {
            **periodic_operators,
            **extended_operators,
            "I_operator": Circuit._identity,
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

    def external_flux_values(self) -> List[float]:
        """
        Returns all the time independent external flux set using the circuit attributes
        for each of the closure branches.
        """
        return [getattr(self, flux.name) for flux in self.external_fluxes]

    def offset_charge_values(self) -> List[float]:
        """
        Returns all the offset charges set using the circuit attributes for each of the
        periodic degree of freedom.
        """
        return [
            getattr(self, offset_charge.name) for offset_charge in self.offset_charges
        ]

    def set_operators(self) -> Dict[str, Callable]:
        """
        Creates the operator methods `<name>_operator` for the circuit.
        """

        if self.hierarchical_diagonalization:
            for subsys in self.subsystems.values():
                subsys.operators_by_name = subsys.set_operators()

        op_func_by_name = self.circuit_operator_functions()
        for op_name, op_func in op_func_by_name.items():
            setattr(self, op_name, MethodType(op_func, self))

        return op_func_by_name

    def identity_wrap_for_hd(
        self,
        operator: Optional[Union[csc_matrix, ndarray]],
        var_index: int,
    ) -> qt.Qobj:
        """
        Returns an identity wrapped operator whose size is equal to the
        `self.hilbertdim()`. Only converts operator which belongs to a specific variable
        index. For example, operator Q_1 or cos(\theta_1). But not, Q1*Q2.

        Parameters
        ----------
        operator:
            operator in the form of csc_matrix, ndarray
        var_index:
            integer which represents which variable index the given operator belongs to

        Returns
        -------
            identity wrapped operator.
        """
        if not self.hierarchical_diagonalization:
            return qt.Qobj(self._kron_operator(operator, var_index))

        subsystem_index = self.get_subsystem_index(var_index)
        subsystem = self.subsystems[subsystem_index]
        operator = subsystem.identity_wrap_for_hd(operator, var_index=var_index)

        if isinstance(operator, qt.Qobj):
            operator = operator.full()

        operator = convert_matrix_to_qobj(
            operator,
            subsystem,
            op_in_eigenbasis=False,
            evecs=subsystem.get_eigenstates(),
        )
        return identity_wrap(
            operator,
            subsystem,
            list(self.subsystems.values()),
            evecs=subsystem.get_eigenstates(),
        )

    def get_operator_by_name(self, operator_name: str) -> qt.Qobj:
        """
        Returns the operator for the given operator symbol which has the same dimension
        as the hilbertdim of the instance from which the operator is requested.

        Parameters
        ----------
        operator_name:
            Name of a sympy Symbol object which should be one among the symbols in the
            attribute vars

        Returns
        -------
            operator identified by `operator_name`
        """
        if not self.hierarchical_diagonalization:
            return getattr(self, operator_name + "_operator")()

        var_index = get_trailing_number(operator_name)
        assert var_index
        subsystem_index = self.get_subsystem_index(var_index)
        subsystem = self.subsystems[subsystem_index]
        operator = subsystem.get_operator_by_name(operator_name)

        if isinstance(operator, qt.Qobj):
            operator = operator.data.tocsc()

        operator = convert_matrix_to_qobj(
            operator,
            subsystem,
            op_in_eigenbasis=False,
            evecs=subsystem.get_eigenstates(),
        )
        return identity_wrap(
            operator,
            subsystem,
            list(self.subsystems.values()),
            evecs=subsystem.get_eigenstates(),
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
        H_string = re.sub(r"(?P<x>(θ\d)|(cosθ\d))", r"\g<x>_operator(self)", H_string)

        # replace all other operators with methods
        operator_symbols_list = flatten_list_recursive(
            [
                list(short_op_dict.values())
                if isinstance(short_op_dict, dict)
                else short_op_dict
                for short_op_dict in list(self.vars.values())
            ]
        )
        operator_name_list = [symbol.name for symbol in operator_symbols_list]
        for operator_name in operator_name_list:
            if "θ" not in operator_name:
                H_string = H_string.replace(
                    operator_name, operator_name + "_operator(self)"
                )
        return H_string

    def _hamiltonian_for_harmonic_extended_vars(self) -> Union[csc_matrix, ndarray]:
        hamiltonian = self._hamiltonian_sym_for_numerics
        # substitute all parameter values
        all_sym_parameters = (
            list(self.symbolic_params.keys())
            + self.external_fluxes
            + self.offset_charges
        )
        hamiltonian = hamiltonian.subs(
            [
                (sym_param, getattr(self, sym_param.name))
                for sym_param in all_sym_parameters
            ]
        )
        hamiltonian = hamiltonian.subs("I", 1)
        # remove constants from the Hamiltonian
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

        offset_charge_names = [
            offset_charge.name for offset_charge in self.offset_charges
        ]
        offset_charge_dict = dict(zip(offset_charge_names, self.offset_charge_values()))
        external_flux_names = [
            external_flux.name for external_flux in self.external_fluxes
        ]
        external_flux_dict = dict(zip(external_flux_names, self.external_flux_values()))

        replacement_dict: Dict[str, Any] = {
            **self.operators_by_name,
            **offset_charge_dict,
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
        ).data.tocsc()

        return eval(H_LC_str, replacement_dict) + junction_potential_matrix

    def _hamiltonian_for_discretized_extended_vars(self) -> csc_matrix:
        hamiltonian = self._hamiltonian_sym_for_numerics
        hamiltonian = hamiltonian.subs(
            [
                (param, getattr(self, str(param)))
                for param in list(self.symbolic_params.keys())
                + self.external_fluxes
                + self.offset_charges
            ]
        )
        hamiltonian = hamiltonian.subs("I", 1)
        # # remove constants from the Hamiltonian
        constant = float(hamiltonian.as_coefficients_dict()[1])
        hamiltonian -= hamiltonian.as_coefficients_dict()[1]
        hamiltonian = hamiltonian.expand() + constant * sm.symbols("I")

        junction_potential = sum(
            [term for term in hamiltonian.as_ordered_terms() if "cos" in str(term)]
        )

        self.junction_potential = junction_potential
        hamiltonian_LC = hamiltonian - junction_potential

        H_LC_str = self._get_eval_hamiltonian_string(hamiltonian_LC)

        replacement_dict: Dict[str, Any] = copy.deepcopy(self.operators_by_name)

        # adding self to the list
        replacement_dict["self"] = self

        junction_potential_matrix = self._evaluate_matrix_cosine_terms(
            junction_potential
        ).data.tocsc()

        return eval(H_LC_str, replacement_dict) + junction_potential_matrix

    def _eigenvals_for_purely_harmonic(self, evals_count: int):
        """
        Returns Hamiltonian for purely harmonic circuits. Hierarchical diagonalization
        is disabled for such circuits.

        Parameters
        ----------
        evals_count:
            Number of eigenenergies
        """
        normal_mode_freqs = self.normal_mode_freqs
        excitations = [np.arange(evals_count) for i in self.var_categories["extended"]]
        energy_array = sum(
            [
                (grid + 0.5) * normal_mode_freqs[idx]
                for idx, grid in enumerate(np.meshgrid(*excitations, indexing="ij"))
            ]
        )
        excitation_indices = []
        energies = []
        num_oscs = len(self.var_categories["extended"])
        for energy in np.unique(energy_array.flatten()):
            if energy not in energies:
                indices = np.where(energy_array == energy)
                for idx in range(len(indices[0])):
                    configuration = [
                        indices[osc_index][idx] for osc_index in range(num_oscs)
                    ]
                    excitation_indices.append(configuration)
                    energies.append(energy)
                    if len(excitation_indices) == evals_count:
                        break
            if len(excitation_indices) >= evals_count:
                break

        return energies, excitation_indices

    def _eigensys_for_purely_harmonic(self, evals_count: int):
        eigenvals, excitation_numbers = self._eigenvals_for_purely_harmonic(
            evals_count=evals_count
        )
        eigen_vectors = []
        for eig_idx, energy in enumerate(eigenvals):
            eigen_vector = []
            for osc_idx, var_index in enumerate(self.var_categories["extended"]):
                evec = np.zeros(self.cutoffs_dict()[var_index])
                evec[excitation_numbers[eig_idx][osc_idx]] = 1
                eigen_vector.append(evec)
            eigen_vectors.append(functools.reduce(np.kron, eigen_vector))
        return eigenvals, np.array(eigen_vectors).T

    def hamiltonian(self) -> Union[csc_matrix, ndarray]:
        """
        Returns the Hamiltonian of the Circuit.
        """
        if not self.hierarchical_diagonalization:
            if isinstance(self, Circuit) and self.is_purely_harmonic:
                return self._hamiltonian_for_harmonic_extended_vars()
            elif self.ext_basis == "harmonic":
                return self._hamiltonian_for_harmonic_extended_vars()
            elif self.ext_basis == "discretized":
                return self._hamiltonian_for_discretized_extended_vars()

        else:
            # update the hilbertspace
            self.build_hilbertspace(
                update_subsystem_indices=self.updated_subsystem_indices
            )
            self.updated_subsystem_indices = []

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
                return hamiltonian.data.tocsc()

    def _evals_calc(self, evals_count: int) -> ndarray:
        # dimension of the hamiltonian
        hilbertdim = self.hilbertdim()

        if (
            isinstance(self, Circuit)
            and self.is_purely_harmonic
            and not self.hierarchical_diagonalization
        ):
            return self._eigenvals_for_purely_harmonic(evals_count=evals_count)[0]

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
        if (
            isinstance(self, Circuit)
            and self.is_purely_harmonic
            and not self.hierarchical_diagonalization
        ):
            return self._eigensys_for_purely_harmonic(evals_count=evals_count)

        # dimension of the hamiltonian
        hilbertdim = self.hilbertdim()

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

        for var_index in self.var_categories_list:
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
        for var in self.external_fluxes:
            potential_sym = potential_sym.subs(var, var * np.pi * 2)

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

        potential_func = sm.lambdify(parameters.keys(), potential_sym, "numpy")

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
            axes.set_ylabel("Potential energy in " + scq.get_units())

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
            cbar.set_label("Potential energy in " + scq.get_units())
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
                sub_subsys.truncated_dim for sub_subsys in subsystem.subsystems.values()
            ]
            wf_new_basis = wf_new_basis.reshape(flatten_list_recursive(wf_shape))
            for sub_subsys_index, sub_subsys in enumerate(
                subsystem.subsystems.values()
            ):
                if len(set(relevant_indices) & set(sub_subsys.var_categories_list)) > 0:
                    wf_new_basis = self._recursive_basis_change(
                        wf_new_basis,
                        wf_dim + sub_subsys_index,
                        sub_subsys,
                        relevant_indices=relevant_indices,
                    )
        else:
            if len(set(relevant_indices) & set(subsystem.var_categories_list)) > 0:
                wf_shape = list(wf_new_basis.shape)
                wf_shape[wf_dim] = [
                    getattr(subsystem, cutoff_attrib)
                    if "ext" in cutoff_attrib
                    else (2 * getattr(subsystem, cutoff_attrib) + 1)
                    for cutoff_attrib in subsystem.cutoff_names
                ]
                wf_new_basis = wf_new_basis.reshape(flatten_list_recursive(wf_shape))
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
            return self.var_categories_list.index(var_index)
        for subsys in self.subsystems.values():
            intersection = list_intersection(subsys.var_categories_list, wf_var_indices)
            if len(intersection) > 0 and var_index not in intersection:
                if subsys.hierarchical_diagonalization:
                    wf_dim += subsys._get_var_dim_for_reshaped_wf(
                        wf_var_indices, var_index
                    )
                else:
                    wf_dim += len(subsys.var_categories_list)
            elif len(intersection) > 0 and var_index in intersection:
                if subsys.hierarchical_diagonalization:
                    wf_dim += subsys._get_var_dim_for_reshaped_wf(
                        wf_var_indices, var_index
                    )
                else:
                    wf_dim += subsys.var_categories_list.index(var_index)
                break
            else:
                wf_dim += 1
        return wf_dim

    def _dims_to_be_summed(self, var_indices: Tuple[int], num_wf_dims) -> List[int]:
        all_var_indices = self.var_categories_list
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
        if self.hierarchical_diagonalization:
            system_hierarchy_for_vars_chosen = list(
                set([self.get_subsystem_index(index) for index in var_indices])
            )  # getting the subsystem index for each of the variable indices

            subsys_trunc_dims = [sys.truncated_dim for sys in self.subsystems.values()]
            # reshaping the wave functions to truncated dims of subsystems
            wf_hd_reshaped = wf.reshape(*subsys_trunc_dims)

            # **** Converting to the basis in which the variables are defined *****
            wf_original_basis = wf_hd_reshaped
            for subsys_index in system_hierarchy_for_vars_chosen:
                wf_dim = 0
                for sys_index in range(subsys_index):
                    if sys_index in system_hierarchy_for_vars_chosen:
                        wf_dim += len(self.subsystems[sys_index].var_categories_list)
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
                    getattr(self, cutoff_attrib)
                    if "ext" in cutoff_attrib
                    else (2 * getattr(self, cutoff_attrib) + 1)
                    for cutoff_attrib in self.cutoff_names
                ]
            )
        return wf_original_basis

    def _change_to_phi_basis(
        self,
        wf_original_basis: ndarray,
        var_indices: Tuple[int],
        grids_dict: Dict[int, Union[discretization.Grid1d, ndarray]],
        change_discrete_charge_to_phi: bool,
    ):
        wf_ext_basis = wf_original_basis
        for var_index in var_indices:
            # finding the dimension corresponding to the var_index
            if not self.hierarchical_diagonalization:
                wf_dim = self.var_categories_list.index(var_index)
            else:
                wf_dim = self._get_var_dim_for_reshaped_wf(var_indices, var_index)

            if (
                var_index in self.var_categories["extended"]
                and self.ext_basis == "harmonic"
            ):
                wf_ext_basis = self._basis_change_harm_osc_to_phi(
                    wf_ext_basis, wf_dim, var_index, grids_dict[var_index]
                )
            if (
                var_index in self.var_categories["periodic"]
                and change_discrete_charge_to_phi
            ):
                wf_ext_basis = self._basis_change_n_to_phi(
                    wf_ext_basis, wf_dim, var_index, grids_dict[var_index]
                )
        return wf_ext_basis

    def generate_wf_plot_data(
        self,
        which: int = 0,
        var_indices: Tuple[int] = (1,),
        eigensys: ndarray = None,
        change_discrete_charge_to_phi: bool = True,
        grids_dict: Dict[int, discretization.Grid1d] = None,
    ):
        """
        Returns the plot of the probability density of the wave function in the
        requested variables for the current Circuit instance.

        Parameters
        ----------
        which:
            integer to choose which wave function to plot
        var_indices:
            A tuple containing the indices of the variables chosen to plot the
            wave function in. Should not have more than 2 entries.
        eigensys:
            The object returned by the method instance. `eigensys` is used to avoid the
            re-evaluation of the eigensystems if already evaluated.
        change_discrete_charge_to_phi: bool
            boolean to choose if the discrete charge basis for the periodic variable
            needs to be changed to phi basis.
        grids_dict:
            A dictionary which pairs var indices with the requested grids used to create
            the plot.
        """
        # checking to see if eigensys needs to be generated
        if eigensys is None:
            evals_count = 6 if which < 6 else which
            _, wfs = self.eigensys(evals_count=which + 1)
        else:
            _, wfs = eigensys

        wf = wfs[:, which]
        # change the wf to the basis in which the variables were initially defined
        wf_original_basis = self._reshape_and_change_to_variable_basis(
            wf=wf, var_indices=var_indices
        )

        # making a basis change to phi for every var_index
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
        wf_plot = np.sum(
            np.abs(wf_ext_basis) ** 2,
            axis=tuple(dims_to_be_summed),
        )
        return wf_plot

    def plot_wavefunction(
        self,
        which=0,
        var_indices: Tuple[int] = (1,),
        esys: Tuple[ndarray, ndarray] = None,
        change_discrete_charge_to_phi: bool = True,
        zero_calibrate: bool = True,
        grids_dict: Dict[int, discretization.Grid1d] = {},
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Returns the plot of the probability density of the wave function in the
        requested variables for the current Circuit instance.

        Parameters
        ----------
        which:
            integer to choose which wave function to plot
        var_indices:
            A tuple containing the indices of the variables chosen to plot the
            wave function in. Should not have more than 2 entries.
        esys:
            The object returned by the method `.eigensys`, is used to avoid the
            re-evaluation of the eigen systems if already evaluated.
        change_discrete_charge_to_phi:
            chooses if the discrete charge basis for the periodic variable
            needs to be changed to phi basis.
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the
            neutral color in the palette
        grids_dict:
            A dictionary which pairs var indices with the requested grids used to create
            the plot.
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
        cutoffs_dict = (
            self.cutoffs_dict()
        )  # dictionary for cutoffs for each variable index
        grids_per_varindex_dict = {}
        var_index_dims_dict = {}
        for cutoff_attrib in self.cutoff_names:
            var_index = get_trailing_number(cutoff_attrib)
            if "cutoff_n" in cutoff_attrib:
                grids_per_varindex_dict[var_index] = (
                    grids_per_varindex_dict[var_index]
                    if var_index in grids_per_varindex_dict
                    else discretization.Grid1d(
                        -np.pi, np.pi, self._default_grid_phi.pt_count
                    )
                )
            else:
                var_index_dims_dict[var_index] = getattr(self, cutoff_attrib)
                if self.ext_basis == "harmonic":
                    grid = (
                        grids_per_varindex_dict[var_index]
                        if var_index in grids_per_varindex_dict
                        else self._default_grid_phi
                    )
                elif self.ext_basis == "discretized":
                    grid = discretization.Grid1d(
                        self.discretized_phi_range[var_index][0],
                        self.discretized_phi_range[var_index][1],
                        cutoffs_dict[var_index],
                    )
                grids_per_varindex_dict[var_index] = grid

        wf_plot = self.generate_wf_plot_data(
            which=which,
            var_indices=var_indices,
            eigensys=esys,
            change_discrete_charge_to_phi=change_discrete_charge_to_phi,
            grids_dict=grids_per_varindex_dict,
        )

        var_types = []

        for var_index in np.sort(var_indices):
            if var_index in self.var_categories["periodic"]:
                if not change_discrete_charge_to_phi:
                    var_types.append("Charge in units of 2e, variable:")
                else:
                    var_types.append("Dimensionless flux, discrete charge variable:")
            else:
                var_types.append("Dimensionless flux, variable:")

        if len(var_indices) == 1:
            return self._plot_wavefunction_1D(
                wf_plot,
                var_indices,
                grids_per_varindex_dict,
                change_discrete_charge_to_phi,
                kwargs,
            )

        elif len(var_indices) == 2:
            return self._plot_wavefunction_2D(
                wf_plot,
                var_indices,
                grids_per_varindex_dict,
                change_discrete_charge_to_phi,
                zero_calibrate=zero_calibrate,
                kwargs=kwargs,
            )

    def _plot_wavefunction_2D(
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

    def _plot_wavefunction_1D(
        self,
        wf_plot: ndarray,
        var_indices,
        grids_per_varindex_dict,
        change_discrete_charge_to_phi: bool,
        kwargs,
    ) -> Tuple[Figure, Axes]:
        var_index = var_indices[0]
        wavefunc = storage.WaveFunction(
            basis_labels=grids_per_varindex_dict[var_indices[0]].make_linspace(),
            amplitudes=wf_plot,
        )

        if not change_discrete_charge_to_phi and (
            var_indices[0] in self.var_categories["periodic"]
        ):
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
                    plt.MaxNLocator(
                        1
                        + 2 * getattr(self, "cutoff_n_" + str(var_index), integer=True)
                    )
                )
        else:
            fig, axes = plot.wavefunction1d_nopotential(
                wavefunc,
                0,
                xlabel=r"$\theta_{{{}}}$".format(str(var_indices[0])),
                ylabel=r"$|\psi(\theta_{{{}}})|^2$".format(str(var_indices[0])),
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


class Circuit(Subsystem):
    """
    Class for analysis of custom superconducting circuits.

    Parameters
    ----------
    input_string: str
        String describing the number of nodes and branches connecting then along
        with their parameters
    from_file: bool
        Set to True by default, when a file name should be provided to
        `input_string`, else the circuit graph description in YAML should be
        provided as a string.
    basis_completion: str
        either "heuristic" or "canonical", defines the matrix used for completing the
        transformation matrix. Sometimes used to change the variable transformation
        to result in a simpler symbolic Hamiltonian, by default "heuristic"
    ext_basis: str
        can be "discretized" or "harmonic" which chooses whether to use discretized
        phi or harmonic oscillator basis for extended variables,
        by default "discretized"
    initiate_sym_calc: bool
        attribute to initiate Circuit instance, by default `True`
    truncated_dim: Optional[int]
        truncated dimension if the user wants to use this circuit instance in
        HilbertSpace, by default `None`
    """

    # switch used in protecting the class from erroneous addition of new attributes
    _frozen = False

    def __init__(
        self,
        input_string: str,
        from_file: bool = True,
        basis_completion="heuristic",
        ext_basis: str = "discretized",
        initiate_sym_calc: bool = True,
        truncated_dim: int = None,
    ):
        base.QuantumSystem.__init__(self, id_str=None)
        if basis_completion not in ["heuristic", "canonical"]:
            raise Exception(
                "Invalid choice for basis_completion: must be 'heuristic' or "
                "'canonical'."
            )

        symbolic_circuit = SymbolicCircuit.from_yaml(
            input_string,
            from_file=from_file,
            basis_completion=basis_completion,
            initiate_sym_calc=True,
        )

        # This class does not yet support custom diagonalization options, but these
        # still have to be defined
        self.evals_method = None
        self.evals_method_options = None
        self.esys_method = None
        self.esys_method_options = None

        sm.init_printing(pretty_print=False, order="none")
        self.is_child = False
        self.symbolic_circuit: SymbolicCircuit = symbolic_circuit

        self.ext_basis: str = ext_basis
        self.truncated_dim: int = truncated_dim
        self.system_hierarchy: list = None
        self.subsystem_trunc_dims: list = None
        self.operators_by_name = None

        self.discretized_phi_range: Dict[int, Tuple[float, float]] = {}
        self.cutoff_names: List[str] = []

        # setting default grids for plotting
        self._default_grid_phi: discretization.Grid1d = discretization.Grid1d(
            -6 * np.pi, 6 * np.pi, 200
        )

        self.type_of_matrices: str = (
            "sparse"  # type of matrices used to construct the operators
        )
        # copying all the required attributes
        required_attributes = [
            "branches",
            "closure_branches",
            "external_fluxes",
            "ground_node",
            "hamiltonian_symbolic",
            "input_string",
            "is_grounded",
            "lagrangian_node_vars",
            "lagrangian_symbolic",
            "nodes",
            "offset_charges",
            "potential_symbolic",
            "potential_node_vars",
            "symbolic_params",
            "transformation_matrix",
            "var_categories",
        ]
        for attr in required_attributes:
            setattr(self, attr, getattr(self.symbolic_circuit, attr))

        if initiate_sym_calc:
            self.configure()

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []
        self._frozen = True

    def __setattr__(self, name, value):
        if not self._frozen or name in dir(self):
            super().__setattr__(name, value)
        else:
            raise Exception(f"Creating new attributes is disabled [{name}, {value}].")

    def set_discretized_phi_range(
        self, var_indices: Tuple[int], phi_range: Tuple[float]
    ) -> None:
        """
        Sets the flux range for discretized phi basis when ext_basis is set to
        'discretized'.

        Parameters
        ----------
        var_indices:
            list of var_indices whose range needs to be changed
        phi_range:
            The desired range for each of the discretized phi variables
        """
        if self.ext_basis != "discretized":
            raise Exception(
                "Discretized phi range is only used when ext_basis is set to "
                "'discretized'."
            )
        for var_index in var_indices:
            if var_index not in self.var_categories["extended"]:
                raise Exception(
                    f"Variable with index {var_index} is not an extended variable."
                )
            self.discretized_phi_range[var_index] = phi_range
        self.operators_by_name = self.set_operators()

    @classmethod
    def from_yaml(
        cls,
        input_string: str,
        from_file: bool = True,
        basis_completion="heuristic",
        ext_basis: str = "discretized",
        initiate_sym_calc: bool = True,
        truncated_dim: int = None,
    ):
        """
        Wrapper to Circuit __init__ to create a class instance. This is deprecated and
        will not be supported in future releases.

        Parameters
        ----------
        input_string:
            String describing the number of nodes and branches connecting then along
            with their parameters
        from_file:
            Set to True by default, when a file name should be provided to
            `input_string`, else the circuit graph description in YAML should be
            provided as a string.
        basis_completion:
            either "heuristic" or "canonical", defines the matrix used for completing
            the transformation matrix. Sometimes used to change the variable
            transformation to result in a simpler symbolic Hamiltonian, by default
            "heuristic"
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use discretized
            phi or harmonic oscillator basis for extended variables,
            by default "discretized"
        initiate_sym_calc:
            attribute to initiate Circuit instance, by default `True`
        truncated_dim:
            truncated dimension if the user wants to use this circuit instance in
            HilbertSpace, by default `None`
        """
        warnings.warn(
            "Initializing Circuit instances with `from_yaml` will not be "
            "supported in the future. Use `Circuit` to initialize a Circuit instance.",
            np.VisibleDeprecationWarning,
        )
        return Circuit(
            input_string=input_string,
            from_file=from_file,
            basis_completion=basis_completion,
            ext_basis=ext_basis,
            initiate_sym_calc=initiate_sym_calc,
            truncated_dim=truncated_dim,
        )

    def dict_for_serialization(self):
        # setting the __init__params attribute
        modified_attrib_keys = (
            [param.name for param in self.symbolic_params]
            + [flux.name for flux in self.external_fluxes]
            + [offset_charge.name for offset_charge in self.offset_charges]
            + self.cutoff_names
            + ["system_hierarchy", "subsystem_trunc_dims", "transformation_matrix"]
        )
        modified_attrib_dict = {key: getattr(self, key) for key in modified_attrib_keys}
        init_params_dict = {}
        init_params_list = ["ext_basis", "input_string", "truncated_dim"]

        for param in init_params_list:
            init_params_dict[param] = getattr(self, param)
        init_params_dict["from_file"] = False
        init_params_dict["basis_completion"] = self.symbolic_circuit.basis_completion

        # storing which branches are used for closure_branches
        closure_branches_data = []
        for branch in self.closure_branches:  # store symbolic param as string
            branch_params = branch.parameters.copy()
            for param in branch_params:
                if isinstance(branch_params[param], sm.Symbol):
                    branch_params[param] = branch_params[param].name
            branch_data = [
                branch.nodes[0].index,
                branch.nodes[1].index,
                branch.type,
                branch_params,
            ]
            closure_branches_data.append(branch_data)
        modified_attrib_dict["closure_branches_data"] = closure_branches_data

        init_params_dict["_modified_attributes"] = modified_attrib_dict
        return init_params_dict

    def serialize(self):
        iodata = dict_serialize(self.dict_for_serialization())
        iodata.typename = type(self).__name__
        return iodata

    @classmethod
    def deserialize(cls, iodata: "IOData") -> "Circuit":
        """
        Take the given IOData and return an instance of the described class, initialized
        with the data stored in io_data.

        Parameters
        ----------
        iodata:

        Returns
        -------
            Circuit instance
        """
        init_params = iodata.as_kwargs()
        _modified_attributes = init_params.pop("_modified_attributes")

        circuit = cls(**init_params)

        closure_branches = [
            circuit._find_branch(*branch_data)
            for branch_data in _modified_attributes["closure_branches_data"]
        ]
        del _modified_attributes["closure_branches_data"]

        # removing parameters that are not defined
        configure_attribs = [
            # "transformation_matrix",
            "system_hierarchy",
            "subsystem_trunc_dims",
        ]
        configure_attribs = [
            attrib for attrib in configure_attribs if attrib in _modified_attributes
        ]

        circuit.configure(
            closure_branches=closure_branches,
            **{key: _modified_attributes[key] for key in configure_attribs},
        )
        # modifying the attributes if necessary
        for attrib in _modified_attributes:
            if attrib not in configure_attribs:
                setattr(circuit, "_" + attrib, _modified_attributes[attrib])
        if circuit.hierarchical_diagonalization:
            circuit.generate_subsystems()
            circuit.build_hilbertspace()
        return circuit

    def _find_branch(
        self, node_id_1: int, node_id_2: int, branch_type: str, branch_params: dict
    ):
        for branch in self.symbolic_circuit.branches:
            branch_node_ids = [node.index for node in branch.nodes]
            branch_params_circ = branch.parameters.copy()
            for param in branch_params_circ:
                if isinstance(branch_params_circ[param], sm.Symbol):
                    branch_params_circ[param] = branch_params_circ[param].name
            if node_id_1 not in branch_node_ids or node_id_2 not in branch_node_ids:
                continue
            if branch.type != branch_type:
                continue
            if branch_params != branch_params_circ:
                continue
            return branch
        return None

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return self._id_str

    def clear_unnecessary_attribs(self):
        """
        Clear all the attributes which are not part of the circuit description
        """
        necessary_attrib_names = (
            self.cutoff_names
            + [flux_symbol.name for flux_symbol in self.external_fluxes]
            + [
                offset_charge_symbol.name
                for offset_charge_symbol in self.offset_charges
            ]
            + ["cutoff_names"]
        )
        attrib_keys = list(self.__dict__.keys()).copy()
        for attrib in attrib_keys:
            if attrib[1:] not in necessary_attrib_names:
                if (
                    "cutoff_n_" in attrib
                    or "Φ" in attrib
                    or "cutoff_ext_" in attrib
                    or attrib[1:3] == "ng"
                ):
                    delattr(self, attrib)

    def configure(
        self,
        transformation_matrix: ndarray = None,
        system_hierarchy: list = None,
        subsystem_trunc_dims: list = None,
        closure_branches: List[Branch] = None,
    ):
        """
        Method which re-initializes a circuit instance to update, hierarchical
        diagonalization parameters or closure branches or the variable transformation
        used to describe the circuit.

        Parameters
        ----------
        transformation_matrix:
            A user defined variable transformation which has the dimensions of the
            number nodes (not counting the ground node), by default `None`
        system_hierarchy:
            A list of lists which is provided by the user to define subsystems,
            by default `None`
        subsystem_trunc_dims:
            dict object which can be generated for a specific system_hierarchy using the
            method `truncation_template`, by default `None`
        closure_branches:
            List of branches where external flux variables will be specified, by default
            `None` which then chooses closure branches by an internally generated
            spanning tree.

        Raises
        ------
        Exception
            when system_hierarchy is set and subsystem_trunc_dims is not set.
        """
        old_transformation_matrix = self.transformation_matrix
        old_system_hierarchy = self.system_hierarchy
        old_subsystem_trunc_dims = self.subsystem_trunc_dims
        old_closure_branches = self.closure_branches
        try:
            self._configure(
                transformation_matrix=transformation_matrix,
                system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims,
                closure_branches=closure_branches,
            )
        except:
            # resetting the necessary attributes
            self.system_hierarchy = old_system_hierarchy
            self.subsystem_trunc_dims = old_subsystem_trunc_dims
            self.transformation_matrix = old_transformation_matrix
            self.closure_branches = old_closure_branches
            # Calling configure
            self._configure(
                transformation_matrix=old_transformation_matrix,
                system_hierarchy=old_system_hierarchy,
                subsystem_trunc_dims=old_subsystem_trunc_dims,
                closure_branches=old_closure_branches,
            )
            raise Exception("Configure failed due to incorrect parameters.")

    def _configure(
        self,
        transformation_matrix: ndarray = None,
        system_hierarchy: list = None,
        subsystem_trunc_dims: list = None,
        closure_branches: List[Branch] = None,
    ):
        """
        Method which re-initializes a circuit instance to update, hierarchical
        diagonalization parameters or closure branches or the variable transformation
        used to describe the circuit.

        Parameters
        ----------
        transformation_matrix:
            A user defined variable transformation which has the dimensions of the
            number nodes (not counting the ground node), by default `None`
        system_hierarchy:
            A list of lists which is provided by the user to define subsystems,
            by default `None`
        subsystem_trunc_dims:
            dict object which can be generated for a specific system_hierarchy using the
            method `truncation_template`, by default `None`
        closure_branches:
            List of branches where external flux variables will be specified, by default
            `None` which then chooses closure branches by an internally generated
            spanning tree.

        Raises
        ------
        Exception
            when system_hierarchy is set and subsystem_trunc_dims is not set.
        """
        self._frozen = False
        system_hierarchy = system_hierarchy or self.system_hierarchy
        subsystem_trunc_dims = subsystem_trunc_dims or self.subsystem_trunc_dims
        closure_branches = closure_branches or self.closure_branches
        if transformation_matrix is None:
            if hasattr(
                self, "transformation_matrix"
            ):  # checking to see if configure is being called outside of init
                transformation_matrix = self.transformation_matrix

        self.hierarchical_diagonalization = (
            True if system_hierarchy is not None else False
        )

        self.symbolic_circuit.configure(
            transformation_matrix=transformation_matrix,
            closure_branches=closure_branches,
        )

        # copying all the required attributes
        required_attributes = [
            "branches",
            "closure_branches",
            "external_fluxes",
            "ground_node",
            "hamiltonian_symbolic",
            "input_string",
            "is_grounded",
            "lagrangian_node_vars",
            "lagrangian_symbolic",
            "nodes",
            "offset_charges",
            "potential_symbolic",
            "potential_node_vars",
            "symbolic_params",
            "transformation_matrix",
            "var_categories",
            "is_purely_harmonic",
        ]
        for attr in required_attributes:
            setattr(self, attr, getattr(self.symbolic_circuit, attr))

        if self.is_purely_harmonic:
            self.normal_mode_freqs = self.symbolic_circuit.normal_mode_freqs
            if self.ext_basis != "harmonic":
                warnings.warn(
                    "Purely harmonic circuits need ext_basis to be set to 'harmonic'"
                )
                self.ext_basis = "harmonic"

        # initiating the class properties
        if not hasattr(self, "cutoff_names"):
            self.cutoff_names = []
        for var_type in self.var_categories.keys():
            if var_type == "periodic":
                for idx, var_index in enumerate(self.var_categories["periodic"]):
                    if not hasattr(self, f"_cutoff_n_{var_index}"):
                        self._make_property(
                            f"cutoff_n_{var_index}", 5, "update_cutoffs"
                        )
                        self.cutoff_names.append(f"cutoff_n_{var_index}")
            if var_type == "extended":
                for idx, var_index in enumerate(self.var_categories["extended"]):
                    if not hasattr(self, f"_cutoff_ext_{var_index}"):
                        self._make_property(
                            f"cutoff_ext_{var_index}", 30, "update_cutoffs"
                        )
                        self.cutoff_names.append(f"cutoff_ext_{var_index}")

        self.var_categories_list = flatten_list(list(self.var_categories.values()))

        # default values for the parameters
        for idx, param in enumerate(self.symbolic_params):
            if not hasattr(self, param.name):
                self._make_property(
                    param.name, self.symbolic_params[param], "update_param_vars"
                )

        # setting the ranges for flux ranges used for discrete phi vars
        for var_index in self.var_categories["extended"]:
            if var_index not in self.discretized_phi_range:
                self.discretized_phi_range[var_index] = (-6 * np.pi, 6 * np.pi)
        # external flux vars
        for flux in self.external_fluxes:
            # setting the default to zero external flux
            if not hasattr(self, flux.name):
                self._make_property(flux.name, 0.0, "update_external_flux_or_charge")
        # offset charges
        for offset_charge in self.offset_charges:
            # default to zero offset charge
            if not hasattr(self, offset_charge.name):
                self._make_property(
                    offset_charge.name, 0.0, "update_external_flux_or_charge"
                )

        # changing the matrix type if necessary
        if (
            len(flatten_list(self.var_categories.values())) == 1
            and self.ext_basis == "harmonic"
        ):
            self.type_of_matrices = "dense"

        self._set_vars()  # setting the attribute vars to store operator symbols

        if (len(self.symbolic_circuit.nodes)) > settings.SYM_INVERSION_MAX_NODES:
            self.hamiltonian_symbolic = (
                self.symbolic_circuit.generate_symbolic_hamiltonian(
                    substitute_params=True
                )
            )

        if system_hierarchy is not None:
            self.hierarchical_diagonalization = (
                system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
            )

        if not self.hierarchical_diagonalization:
            self.generate_hamiltonian_sym_for_numerics()
            self.operators_by_name = self.set_operators()
        else:
            # list for updating necessary subsystems when calling build hilbertspace
            self.updated_subsystem_indices = []
            self.operators_by_name = None
            self.system_hierarchy = system_hierarchy
            if subsystem_trunc_dims is None:
                raise Exception(
                    "The truncated dimensions attribute for hierarchical "
                    "diagonalization is not set."
                )

            self.subsystem_trunc_dims = subsystem_trunc_dims
            self.generate_hamiltonian_sym_for_numerics()
            self.generate_subsystems()
            self._check_truncation_indices()
            self.operators_by_name = self.set_operators()
            self.updated_subsystem_indices = list(range(len(self.subsystems)))
        # clear unnecessary attribs
        self.clear_unnecessary_attribs()
        self._frozen = True

    def variable_transformation(self) -> None:
        """
        Prints the variable transformation used in this circuit
        """
        trans_mat = self.transformation_matrix
        theta_vars = [
            sm.symbols(f"θ{index}")
            for index in range(
                1, len(self.symbolic_circuit._node_list_without_ground) + 1
            )
        ]
        node_vars = [
            sm.symbols(f"φ{index}")
            for index in range(
                1, len(self.symbolic_circuit._node_list_without_ground) + 1
            )
        ]
        node_var_eqns = []
        for idx, node_var in enumerate(node_vars):
            node_var_eqns.append(
                sm.Eq(node_vars[idx], np.sum(trans_mat[idx, :] * theta_vars))
            )
        if _HAS_IPYTHON:
            self.print_expr_in_latex(node_var_eqns)
        else:
            print(node_var_eqns)

    def sym_lagrangian(
        self,
        vars_type: str = "node",
        print_latex: bool = False,
        return_expr: bool = False,
    ) -> Union[sm.Expr, None]:
        """
        Method that gives a user readable symbolic Lagrangian for the current instance

        Parameters
        ----------
        vars_type:
            "node" or "new", fixes the kind of lagrangian requested, by default "node"
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
            if set to True, all printing is suppressed and the function will silently
            return the sympy expression
        """
        if vars_type == "node":
            lagrangian = self.lagrangian_node_vars
            # replace v\theta with \theta_dot
            for var_index in range(
                1, 1 + len(self.symbolic_circuit._node_list_without_ground)
            ):
                lagrangian = lagrangian.replace(
                    sm.symbols(f"vφ{var_index}"),
                    sm.symbols("\\dot{φ_" + str(var_index) + "}"),
                )
            # break down the lagrangian into kinetic and potential part, and rejoin
            # with evaluate=False to force the kinetic terms together and appear first
            sym_lagrangian_PE_node_vars = self.potential_node_vars
            for external_flux in self.external_fluxes:
                sym_lagrangian_PE_node_vars = sym_lagrangian_PE_node_vars.replace(
                    external_flux,
                    sm.symbols(
                        "(2π"
                        + "Φ_{"
                        + str(get_trailing_number(str(external_flux)))
                        + "})"
                    ),
                )
            lagrangian = sm.Add(
                (self._make_expr_human_readable(lagrangian + self.potential_node_vars)),
                (self._make_expr_human_readable(-sym_lagrangian_PE_node_vars)),
                evaluate=False,
            )

        elif vars_type == "new":
            lagrangian = self.lagrangian_symbolic
            # replace v\theta with \theta_dot
            for var_index in self.var_categories_list:
                lagrangian = lagrangian.replace(
                    sm.symbols(f"vθ{var_index}"),
                    sm.symbols("\\dot{θ_" + str(var_index) + "}"),
                )
            # break down the lagrangian into kinetic and potential part, and rejoin
            # with evaluate=False to force the kinetic terms together and appear first
            sym_lagrangian_PE_new = self.potential_symbolic.expand()
            for external_flux in self.external_fluxes:
                sym_lagrangian_PE_new = sym_lagrangian_PE_new.replace(
                    external_flux,
                    sm.symbols(
                        "(2π"
                        + "Φ_{"
                        + str(get_trailing_number(str(external_flux)))
                        + "})"
                    ),
                )
            lagrangian = sm.Add(
                (
                    self._make_expr_human_readable(
                        lagrangian + self.potential_symbolic.expand()
                    )
                ),
                (self._make_expr_human_readable(-sym_lagrangian_PE_new)),
                evaluate=False,
            )
        if return_expr:
            return lagrangian
        if print_latex:
            print(latex(lagrangian))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(lagrangian)
        else:
            print(lagrangian)

    def offset_charge_transformation(self) -> None:
        """
        Prints the variable transformation between offset charges of periodic variables
        and the offset node charges
        """
        trans_mat = self.transformation_matrix
        node_offset_charge_vars = [
            sm.symbols(f"q_g{index}")
            for index in range(
                1, len(self.symbolic_circuit._node_list_without_ground) + 1
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

    def sym_external_fluxes(self) -> Dict[sm.Expr, Tuple["Branch", List["Branch"]]]:
        """
        Method returns a dictionary of Human readable external fluxes with associated
        branches and loops (represented as lists of branches) for the current instance

        Returns
        -------
            A dictionary of Human readable external fluxes with their associated
            branches and loops
        """
        return {
            self._make_expr_human_readable(self.external_fluxes[ibranch]): (
                self.closure_branches[ibranch],
                self.symbolic_circuit._find_loop(self.closure_branches[ibranch]),
            )
            for ibranch in range(len(self.external_fluxes))
        }
