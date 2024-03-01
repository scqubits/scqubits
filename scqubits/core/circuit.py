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

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import sympy as sm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from sympy import latex

try:
    from IPython.display import Latex, display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

import scqubits.core.discretization as discretization
import scqubits.core.central_dispatch as dispatch
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers

from scqubits import settings
from scqubits.core.circuit_utils import (
    get_trailing_number,
    is_potential_term,
)
from scqubits.core.symbolic_circuit import Branch, SymbolicCircuit
from scqubits.utils.misc import (
    flatten_list,
    flatten_list_recursive,
    is_string_float,
    number_of_lists_in_list,
)

from scqubits.core.circuit_routines import CircuitRoutines
from scqubits.core.circuit_noise import NoisyCircuit


class Subsystem(
    CircuitRoutines,
    base.QubitBaseClass,
    serializers.Serializable,
    dispatch.DispatchClient,
    NoisyCircuit,
):
    """
    Defines a subsystem for a circuit, which can further be used recursively to define
    subsystems within subsystem.

    Parameters
    ----------
    parent: Subsystem
        the instance under which the new subsystem is defined.
    ext_basis: str
        The basis that should be used for extended variables
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

    def __init__(
        self,
        parent: "Subsystem",
        hamiltonian_symbolic: sm.Expr,
        ext_basis: Union[str, List],
        system_hierarchy: Optional[List] = None,
        subsystem_trunc_dims: Optional[List] = None,
        truncated_dim: Optional[int] = 10,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ):
        # switch used in protecting the class from erroneous addition of new attributes
        object.__setattr__(self, "_frozen", False)

        base.QubitBaseClass.__init__(
            self,
            id_str=None,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )

        self.system_hierarchy = system_hierarchy
        self.truncated_dim = truncated_dim
        self.subsystem_trunc_dims = subsystem_trunc_dims

        self.is_child = True
        self.parent = parent
        self.hamiltonian_symbolic = hamiltonian_symbolic
        self._default_grid_phi = self.parent._default_grid_phi

        self.junction_potential = None
        self._H_LC_str_harmonic = None
        # attribute to keep track if the symbolic Hamiltonian needs to be updated
        self._make_property(
            "_user_changed_parameter",
            False,
            "update_user_changed_parameter",
            use_central_dispatch=False,
        )

        self.ext_basis = ext_basis
        self._find_and_set_sym_attrs()

        self.dynamic_var_indices: List[int] = flatten_list_recursive(
            [self.system_hierarchy]
        )
        parent_cutoffs_dict = self.parent.cutoffs_dict()
        cutoffs: List[int] = [
            parent_cutoffs_dict[var_index] for var_index in self.dynamic_var_indices
        ]

        self.var_categories: Dict[str, List[int]] = {}
        for var_type in self.parent.var_categories:
            self.var_categories[var_type] = [
                var_index
                for var_index in self.parent.var_categories[var_type]
                if var_index in self.dynamic_var_indices
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
            if idx in self.dynamic_var_indices
        }

        # storing the potential terms separately

        self.potential_symbolic = self.generate_sym_potential()

        self.hierarchical_diagonalization: bool = (
            system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
        )

        if len(self.dynamic_var_indices) == 1:
            self.type_of_matrices = "dense"
        else:
            self.type_of_matrices = "sparse"

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []

        # attributes for purely harmonic
        self.normal_mode_freqs = []

        self._configure()
        self._frozen = True

    def _find_and_set_sym_attrs(self):
        """
        Finds the symbolic and other circuit params from the symbolic Hamiltonian, and sets the attribs
        external_fluxes, offset_charges and symbolic_params. Only works when _frozen is set to False, or
        the above attribs are already set.
        """
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
        self.free_charges = [
            var
            for var in self.parent.free_charges
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.symbolic_params = {
            var: self.parent.symbolic_params[var]
            for var in self.parent.symbolic_params
            if var in self.hamiltonian_symbolic.free_symbols
        }

    def _configure(self) -> None:
        """
        Function which is used to initiate the subsystem instance.
        """
        self._frozen = False
        for idx, param in enumerate(self.symbolic_params):
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
        for charge_var in self.offset_charges + self.free_charges:
            self._make_property(
                charge_var.name,
                getattr(self.parent, charge_var.name),
                "update_external_flux_or_charge",
            )

        for cutoff_str in self.cutoff_names:
            self._make_property(
                cutoff_str, getattr(self.parent, cutoff_str), "update_cutoffs"
            )
        # if subsystem hamiltonian is purely harmonic
        if (
            self._is_expression_purely_harmonic(self.hamiltonian_symbolic)
            and self.ext_basis == "harmonic"
        ):
            self.is_purely_harmonic = True
            self._annihilation_operator_in_eigenbasis = None
        else:
            self.is_purely_harmonic = False

        # Creating the attributes for purely harmonic circuits
        if (
            isinstance(self, Circuit) and self.parent.is_purely_harmonic
        ):  # assuming that the parent has only extended variables and are ordered
            # starting from 1, 2, 3, ...
            self.is_purely_harmonic = self.parent.is_purely_harmonic
            self.normal_mode_freqs = self.parent.normal_mode_freqs[
                [var_idx - 1 for var_idx in self.var_categories["extended"]]
            ]

        if self.hierarchical_diagonalization:
            # attribute to note updated subsystem indices
            self.affected_subsystem_indices = []
            self._hamiltonian_sym_for_numerics = self.hamiltonian_symbolic.copy()
            self.generate_subsystems()
            self.ext_basis = self.get_ext_basis()
            self.update_interactions()
            self._check_truncation_indices()
            self.affected_subsystem_indices = list(range(len(self.subsystems)))
        else:
            self.generate_hamiltonian_sym_for_numerics()
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                self._diagonalize_purely_harmonic_hamiltonian()

        self._set_vars()
        self.operators_by_name = self.set_operators()

        if self.hierarchical_diagonalization:
            self._out_of_sync = False  # for use with CentralDispatch
            dispatch.CENTRAL_DISPATCH.register("CIRCUIT_UPDATE", self)
        self._frozen = True


class Circuit(
    CircuitRoutines,
    base.QubitBaseClass,
    serializers.Serializable,
    dispatch.DispatchClient,
    NoisyCircuit,
):
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
    use_dynamic_flux_grouping: bool
        set to False by default. Indicates if the flux allocation is done by assuming
        that flux is time dependent. When set to True, it disables the option to change
        the closure branches.
    initiate_sym_calc: bool
        attribute to initiate Circuit instance, by default `True`
    truncated_dim: Optional[int]
        truncated dimension if the user wants to use this circuit instance in
        HilbertSpace, by default `None`
    """

    def __init__(
        self,
        input_string: Optional[str] = None,
        from_file: bool = True,
        basis_completion="heuristic",
        ext_basis: str = "discretized",
        use_dynamic_flux_grouping: bool = False,
        generate_noise_methods: bool = False,
        initiate_sym_calc: bool = True,
        truncated_dim: int = 10,
        symbolic_param_dict: Dict[str, float] = None,
        symbolic_hamiltonian: sm.Expr = None,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ):
        # switch used in protecting the class from erroneous addition of new attributes
        object.__setattr__(self, "_frozen", False)
        base.QubitBaseClass.__init__(
            self,
            id_str=None,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        if symbolic_hamiltonian and input_string:
            raise Exception(
                "Circuit instance cannot be initialized with both input_string and symbolic_hamiltonian."
            )
        if input_string:
            self.from_yaml(
                input_string=input_string,
                from_file=from_file,
                basis_completion=basis_completion,
                ext_basis=ext_basis,
                use_dynamic_flux_grouping=use_dynamic_flux_grouping,
                generate_noise_methods=generate_noise_methods,
                initiate_sym_calc=initiate_sym_calc,
                truncated_dim=truncated_dim,
            )

        else:
            if use_dynamic_flux_grouping or generate_noise_methods:
                raise Exception(
                    "Circuit instance initialized using symbolic Hamiltonian cannot be configured with closure_branches, use_dynamic_flux_grouping, transformation_matrix or generate_noise_methods."
                )
            self.from_symbolic_hamiltonian(
                symbolic_hamiltonian=symbolic_hamiltonian,
                symbolic_param_dict=symbolic_param_dict,
                initiate_sym_calc=initiate_sym_calc,
                truncated_dim=truncated_dim,
                ext_basis=ext_basis,
            )

    def from_symbolic_hamiltonian(
        self,
        symbolic_hamiltonian: sm.Expr,
        symbolic_param_dict: Dict[str, float],
        initiate_sym_calc: bool,
        truncated_dim: int,
        ext_basis: str,
    ):
        self.hamiltonian_symbolic = symbolic_hamiltonian

        self.symbolic_params = {}
        for param_str in symbolic_param_dict:
            if "ng" in param_str or "Φ" in param_str:
                continue
            self.symbolic_params[sm.symbols(param_str)] = symbolic_param_dict[param_str]

        sm.init_printing(pretty_print=False, order="none")
        self.is_child = False

        self.ext_basis = ext_basis
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

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []
        self._out_of_sync = False  # for use with CentralDispatch
        self._make_property(
            "_user_changed_parameter",
            False,
            "update_user_changed_parameter",
            use_central_dispatch=False,
        )

        if initiate_sym_calc:
            self.configure()
        self._frozen = True
        dispatch.CENTRAL_DISPATCH.register("CIRCUIT_UPDATE", self)

    def from_yaml(
        self,
        input_string: str,
        from_file: bool = True,
        basis_completion="heuristic",
        ext_basis: str = "discretized",
        use_dynamic_flux_grouping: bool = False,
        generate_noise_methods: bool = False,
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
        use_dynamic_flux_grouping: bool
            set to False by default. Indicates if the flux allocation is done by assuming that flux is time dependent. When set to True, it disables the
            option to change the closure branches.
        """
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
            use_dynamic_flux_grouping=use_dynamic_flux_grouping,
        )
        sm.init_printing(pretty_print=False, order="none")
        self.is_child = False
        self.symbolic_circuit: SymbolicCircuit = symbolic_circuit

        self.ext_basis = ext_basis
        self.hierarchical_diagonalization: bool = False
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
            "free_charges",
            "potential_symbolic",
            "potential_node_vars",
            "symbolic_params",
            "transformation_matrix",
            "var_categories",
        ]
        for attr in required_attributes:
            setattr(self, attr, getattr(self.symbolic_circuit, attr))

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []
        self._out_of_sync = False  # for use with CentralDispatch
        self._make_property(
            "_user_changed_parameter",
            False,
            "update_user_changed_parameter",
            use_central_dispatch=False,
        )

        if initiate_sym_calc:
            self.configure()
        if generate_noise_methods:
            self.generate_noise_methods()
        self._frozen = True
        dispatch.CENTRAL_DISPATCH.register("CIRCUIT_UPDATE", self)

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

    def _clear_unnecessary_attribs(self):
        """
        Clear all the attributes which are not part of the circuit description
        """
        necessary_attrib_names = (
            self.cutoff_names
            + [flux_symbol.name for flux_symbol in self.external_fluxes]
            + [
                charge_symbol.name
                for charge_symbol in self.offset_charges + self.free_charges
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
        transformation_matrix: Optional[ndarray] = None,
        system_hierarchy: Optional[list] = None,
        subsystem_trunc_dims: Optional[list] = None,
        closure_branches: Optional[List[Branch]] = None,
        ext_basis: Optional[str] = None,
        use_dynamic_flux_grouping: Optional[bool] = None,
        generate_noise_methods: bool = False,
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
            spanning tree. For this option, Circuit should be initialized with `use_dynamic_flux_grouping` set to False.
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use discretized
            phi or harmonic oscillator basis for extended variables,
            by default `None`
        use_dynamic_flux_grouping:
            set to False by default. Indicates if the flux allocation is done by assuming that flux is time dependent. When set to True, it disables the option to change the closure branches.
        generate_noise_methods:
            set to False by default. Indicates if the noise methods should be generated for the circuit instance.
        Raises
        ------
        Exception
            When system_hierarchy is set and subsystem_trunc_dims is not set.
        Exception
            When closure_branches is set and the Circuit instance is initialized with the setting
            `use_dynamic_flux_grouping=True`.
        """

        old_system_hierarchy = self.system_hierarchy
        old_subsystem_trunc_dims = self.subsystem_trunc_dims
        old_ext_basis = self.ext_basis
        if hasattr(self, "symbolic_circuit"):
            old_transformation_matrix = self.transformation_matrix
            old_use_dynamic_flux_grouping = (
                self.symbolic_circuit.use_dynamic_flux_grouping
            )
            old_closure_branches = (
                self.closure_branches if not old_use_dynamic_flux_grouping else None
            )
            old_generate_noise_methods = hasattr(self, "_noise_methods_generated")
        try:
            if hasattr(self, "symbolic_circuit"):
                self._configure(
                    transformation_matrix=transformation_matrix,
                    system_hierarchy=system_hierarchy,
                    subsystem_trunc_dims=subsystem_trunc_dims,
                    closure_branches=closure_branches,
                    ext_basis=ext_basis,
                    use_dynamic_flux_grouping=use_dynamic_flux_grouping,
                    generate_noise_methods=generate_noise_methods,
                )
            else:
                if (
                    closure_branches is not None
                    or use_dynamic_flux_grouping
                    or generate_noise_methods
                ):
                    raise Exception(
                        "Circuit instance initialized using symbolic Hamiltonian cannot be configured with closure_branches, use_dynamic_flux_grouping, transformation_matrix or generate_noise_methods."
                    )
                self._configure_sym_hamiltonian(
                    system_hierarchy=system_hierarchy,
                    subsystem_trunc_dims=subsystem_trunc_dims,
                    ext_basis=ext_basis,
                )
        except:
            # resetting the necessary attributes
            self.system_hierarchy = old_system_hierarchy
            self.subsystem_trunc_dims = old_subsystem_trunc_dims
            if hasattr(self, "symbolic_circuit"):
                self.transformation_matrix = old_transformation_matrix
                self.closure_branches = old_closure_branches
            # Calling configure
            if hasattr(self, "symbolic_circuit"):
                self._configure(
                    transformation_matrix=old_transformation_matrix,
                    system_hierarchy=old_system_hierarchy,
                    subsystem_trunc_dims=old_subsystem_trunc_dims,
                    closure_branches=old_closure_branches,
                    ext_basis=old_ext_basis,
                    use_dynamic_flux_grouping=old_use_dynamic_flux_grouping,
                    generate_noise_methods=old_generate_noise_methods,
                )
            else:
                self._configure_sym_hamiltonian(
                    system_hierarchy=old_system_hierarchy,
                    subsystem_trunc_dims=old_subsystem_trunc_dims,
                    ext_basis=old_ext_basis,
                )
            raise Exception("Configure failed due to incorrect parameters.")

    def _read_symbolic_hamiltonian(
        self, symbolic_hamiltonian: sm.Expr
    ) -> Tuple[List[sm.Expr], List[sm.Expr], List[sm.Expr], Dict[str, List[int]]]:
        free_symbols = symbolic_hamiltonian.free_symbols
        external_fluxes = []
        offset_charges = []
        free_charges = []
        var_categories = {"periodic": [], "extended": [], "free": [], "frozen": []}
        for var_sym in free_symbols:
            if re.match(r"^ng\d+$", var_sym.name):
                offset_charges.append(var_sym)
            elif re.match(r"^Qf\d+$", var_sym.name):
                free_charges.append(var_sym)
            elif re.match(r"^Φ\d+$", var_sym.name):
                external_fluxes.append(var_sym)
            elif re.match(r"^n\d+$", var_sym.name):
                var_index = get_trailing_number(var_sym.name)
                var_categories["periodic"].append(var_index)
            elif re.match(r"^Q\d+$", var_sym.name):
                var_index = get_trailing_number(var_sym.name)
                var_categories["extended"].append(var_index)
        var_categories = {
            category: sorted(var_categories[category]) for category in var_categories
        }
        return external_fluxes, offset_charges, free_charges, var_categories

    def _configure_sym_hamiltonian(
        self,
        system_hierarchy: Optional[list] = None,
        subsystem_trunc_dims: Optional[list] = None,
        ext_basis: Optional[str] = None,
    ):
        """
        Method which re-initializes a circuit instance to update, hierarchical
        diagonalization parameters or closure branches or the variable transformation
        used to describe the circuit.

        Parameters
        ----------
        system_hierarchy:
            A list of lists which is provided by the user to define subsystems,
            by default `None`
        subsystem_trunc_dims:
            dict object which can be generated for a specific system_hierarchy using the
            method `truncation_template`, by default `None`
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use discretized
            phi or harmonic oscillator basis for extended variables,
            by default `None`

        Raises
        ------
        Exception
            when system_hierarchy is set and subsystem_trunc_dims is not set.
        """
        self._frozen = False
        system_hierarchy = system_hierarchy or self.system_hierarchy
        subsystem_trunc_dims = subsystem_trunc_dims or self.subsystem_trunc_dims
        self.ext_basis = ext_basis or self.ext_basis

        self.hierarchical_diagonalization = (
            True if system_hierarchy is not None else False
        )

        self.is_purely_harmonic = self._is_expression_purely_harmonic(
            self.hamiltonian_symbolic
        )

        (
            self.external_fluxes,
            self.offset_charges,
            self.free_charges,
            self.var_categories,
        ) = self._read_symbolic_hamiltonian(self.hamiltonian_symbolic)

        # initiating the class properties
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

        self.dynamic_var_indices = (
            self.var_categories["periodic"] + self.var_categories["extended"]
        )

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
        for charge_var in self.offset_charges + self.free_charges:
            # default to zero offset charge
            if not hasattr(self, charge_var.name):
                self._make_property(
                    charge_var.name, 0.0, "update_external_flux_or_charge"
                )

        self.potential_symbolic = self.generate_sym_potential()

        # changing the matrix type if necessary
        if len(flatten_list(self.var_categories.values())) == 1:
            self.type_of_matrices = "dense"

        if system_hierarchy is not None:
            self.hierarchical_diagonalization = (
                system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
            )

        if not self.hierarchical_diagonalization:
            if self.is_purely_harmonic and not ext_basis:
                if self.ext_basis != "harmonic":
                    warnings.warn(
                        "Purely harmonic circuits need ext_basis to be set to 'harmonic'"
                    )
                    self.ext_basis = "harmonic"
            self.ext_basis = ext_basis or self.ext_basis
            self.generate_hamiltonian_sym_for_numerics()
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                # using the default methods
                self.evals_method = None
                self.evals_method_options = None
                self._annihilation_operator_in_eigenbasis = None
                self._diagonalize_purely_harmonic_hamiltonian()
            self._set_vars()  # setting the attribute vars to store operator symbols
            self.operators_by_name = self.set_operators()
        else:
            # list for updating necessary subsystems when calling build hilbertspace
            self.affected_subsystem_indices = []
            self.operators_by_name = None
            self.system_hierarchy = system_hierarchy
            if subsystem_trunc_dims is None:
                raise Exception(
                    "The truncated dimensions attribute for hierarchical "
                    "diagonalization is not set."
                )

            self.subsystem_trunc_dims = subsystem_trunc_dims
            if ext_basis:
                self.ext_basis = ext_basis
            self.generate_hamiltonian_sym_for_numerics()
            self.generate_subsystems()
            self.ext_basis = (
                self.get_ext_basis()
            )  # update the ext_basis after generating subsystems
            self._set_vars()  # setting the attribute vars to store operator symbols
            self._check_truncation_indices()
            self.operators_by_name = self.set_operators()
            self.affected_subsystem_indices = list(range(len(self.subsystems)))
            self.update_interactions()

        # clear unnecessary attribs
        self._clear_unnecessary_attribs()
        self._frozen = True
        self.update()

    def _configure(
        self,
        transformation_matrix: Optional[ndarray] = None,
        system_hierarchy: Optional[list] = None,
        subsystem_trunc_dims: Optional[list] = None,
        closure_branches: Optional[List[Branch]] = None,
        ext_basis: Optional[str] = None,
        use_dynamic_flux_grouping: Optional[bool] = None,
        generate_noise_methods: bool = False,
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
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use discretized, or can be a list of lists of lists, when hierarchical diagonalization is used.
        use_dynamic_flux_grouping:
            set to False by default. Indicates if the flux allocation is done by assuming that flux is time dependent. When set to True, it disables the option to change the closure branches.
        generate_noise_methods:
            set to False by default. Indicates if the noise methods should be generated for the circuit instance.

        Raises
        ------
        Exception
            when system_hierarchy is set and subsystem_trunc_dims is not set.
        """
        self._frozen = False

        # reinitiate the symbolic circuit when the transformation matrix and closure branches are provided
        if (
            transformation_matrix is not None
            or closure_branches is not None
            or use_dynamic_flux_grouping is not None
        ):
            self.symbolic_circuit.configure(
                transformation_matrix=transformation_matrix,
                closure_branches=closure_branches,
                use_dynamic_flux_grouping=use_dynamic_flux_grouping,
            )

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
            "free_charges",
            "potential_symbolic",
            "potential_node_vars",
            "symbolic_params",
            "transformation_matrix",
            "var_categories",
            "is_purely_harmonic",
        ]
        for attr in required_attributes:
            setattr(self, attr, getattr(self.symbolic_circuit, attr))

        # initiating the class properties
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

        self.dynamic_var_indices = (
            self.var_categories["periodic"] + self.var_categories["extended"]
        )

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
        # offset and free charges
        for charge_var in self.offset_charges + self.free_charges:
            # default to zero offset charge
            if not hasattr(self, charge_var.name):
                self._make_property(
                    charge_var.name, 0.0, "update_external_flux_or_charge"
                )

        # changing the matrix type if necessary
        if (
            len((self.var_categories["extended"] + self.var_categories["periodic"]))
            == 1
        ):
            self.type_of_matrices = "dense"

        self.hamiltonian_symbolic = self.symbolic_circuit.hamiltonian_symbolic
        # if the flux is static, remove the linear terms from the potential
        if not self.symbolic_circuit.use_dynamic_flux_grouping:
            self.hamiltonian_symbolic = self._shift_harmonic_oscillator_potential(
                self.hamiltonian_symbolic
            )
            self.potential_symbolic = self._shift_harmonic_oscillator_potential(
                self.potential_symbolic.expand()
            )

        if system_hierarchy is not None:
            self.hierarchical_diagonalization = (
                system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
            )

        if not self.hierarchical_diagonalization:
            if self.is_purely_harmonic and not ext_basis:
                self.normal_mode_freqs = self.symbolic_circuit.normal_mode_freqs
                if self.ext_basis != "harmonic":
                    warnings.warn(
                        "Purely harmonic circuits need ext_basis to be set to 'harmonic'"
                    )
                    self.ext_basis = "harmonic"
            self.generate_hamiltonian_sym_for_numerics()
            self.ext_basis = ext_basis or self.ext_basis
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                # using the default methods
                self.evals_method = None
                self.evals_method_options = None
                self._annihilation_operator_in_eigenbasis = None
                self._diagonalize_purely_harmonic_hamiltonian()
        else:
            # list for updating necessary subsystems when calling build hilbertspace
            self.affected_subsystem_indices = []
            self.operators_by_name = None
            self.system_hierarchy = system_hierarchy
            if subsystem_trunc_dims is None:
                raise Exception(
                    "The truncated dimensions attribute for hierarchical "
                    "diagonalization is not set."
                )

            self.subsystem_trunc_dims = subsystem_trunc_dims
            self.generate_hamiltonian_sym_for_numerics()
            self.ext_basis = ext_basis or self.ext_basis
            self.generate_subsystems()
            self.ext_basis = self.get_ext_basis()
            self.update_interactions()
            self._check_truncation_indices()
            self.affected_subsystem_indices = list(range(len(self.subsystems)))

        self._set_vars()  # setting the attribute vars to store operator symbols
        self.operators_by_name = self.set_operators()
        # clear unnecessary attribs
        self._clear_unnecessary_attribs()
        if generate_noise_methods:
            self.generate_noise_methods()
        self._frozen = True
        self.update()

    def supported_noise_channels(self) -> List[str]:
        """Return a list of supported noise channels"""
        if not hasattr(self, "_noise_methods_generated"):
            raise Exception(
                "Noise methods are not generated, please use configure() with generate_noise_methods=True to generate them."
            )
        return [
            method_name
            for method_name in self.__dict__
            if "tphi_1_over_f" in method_name or "t1_" in method_name
        ]

    def effective_noise_channels(self):
        if not hasattr(self, "_noise_methods_generated"):
            raise Exception(
                "Noise methods are not generated, please use configure() with generate_noise_methods=True to generate them."
            )
        return [
            method_name
            for method_name in self.supported_noise_channels()
            if not is_string_float(method_name[-1])
        ]

    def variable_transformation(self, new_vars_to_node_vars=True) -> None:
        """
        Prints the variable transformation used in this circuit
        """
        trans_mat = self.transformation_matrix
        if new_vars_to_node_vars:
            trans_mat = np.linalg.inv(trans_mat)
        theta_vars = [
            sm.symbols(f"θ{index}")
            for index in range(
                1, len(self.symbolic_circuit.nodes) - self.is_grounded + 1
            )
        ]
        node_vars = [
            sm.symbols(f"φ{index}")
            for index in range(
                1, len(self.symbolic_circuit.nodes) - self.is_grounded + 1
            )
        ]
        var_eqns = []
        for idx, node_var in enumerate(node_vars):
            if not new_vars_to_node_vars:
                var_eqns.append(
                    sm.Eq(node_vars[idx], np.sum(trans_mat[idx, :] * theta_vars))
                )
            else:
                var_eqns.append(
                    sm.Eq(theta_vars[idx], np.sum(trans_mat[idx, :] * node_vars))
                )
        if _HAS_IPYTHON:
            self.print_expr_in_latex(var_eqns)
        else:
            print(var_eqns)

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
                1, 1 + len(self.symbolic_circuit.nodes) - self.is_grounded
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
            for var_index in self.dynamic_var_indices:
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

    def oscillator_list(self, osc_index_list: List[int]):
        """
        If hierarchical diagonalization is used, specify subsystems that corresponds to
        single-mode oscillators, if there is any. The attributes `_osc_subsys_list` and
        `osc_subsys_list` of the `hilbert_space` attribute of the Circuit instance will
        be assigned accordingly, enabling the correct identification of harmonic modes
        for the dispersive regime analysis in ParameterSweep.

        Parameters
        ----------
        osc_index_list:
            a list of indices of subsystems that are single-mode harmonic oscillators
        """
        # identify if each nominated subsystem indeed have a single harmonic oscillator
        osc_subsys_list = []
        for subsystem_index in osc_index_list:
            subsystem = self.subsystems[subsystem_index]
            if not subsystem.is_purely_harmonic:
                raise Exception(
                    f"the subsystem {subsystem_index} is not purely harmonic"
                )
            elif len(subsystem.var_categories["extended"]) != 1:
                raise Exception(
                    f"the subsystem has more than one harmonic oscillator mode"
                )
            else:
                osc_subsys_list.append(subsystem)
        self.hilbert_space._osc_subsys_list = osc_subsys_list

    def qubit_list(self, qbt_index_list: List[int]):
        """
        If hierarchical diagonalization is used, specify subsystems that corresponds to
        single-mode oscillators, if there is any. The attributes `_osc_subsys_list` and
        `osc_subsys_list` of the `hilbert_space` attribute of the Circuit instance will
        be assigned accordingly, enabling the correct identification of harmonic modes
        for the dispersive regime analysis in ParameterSweep.

        Parameters
        ----------
        qbt_index_list:
            a list of indices of subsystems that are single-mode harmonic oscillators
        """
        # identify if each naminated subsystem indeed have a single harmonic oscillator
        qbt_subsys_list = []
        for subsystem_index in qbt_index_list:
            subsystem = self.subsystems[subsystem_index]
            qbt_subsys_list.append(subsystem)
        self.hilbert_space._qbt_subsys_list = qbt_subsys_list
