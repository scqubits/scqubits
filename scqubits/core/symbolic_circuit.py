# symbolic_circuit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import copy
import itertools
import warnings
from symtable import Symbol
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import scipy as sp
import sympy
import sympy as sm
import random

from numpy import ndarray
from sympy import symbols, use
from scqubits.core.circuit_utils import (
    round_symbolic_expr,
    _capactiance_variable_for_branch,
    _junction_order,
    get_trailing_number,
)

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings

from scqubits.utils.misc import (
    flatten_list_recursive,
    is_string_float,
    unique_elements_in_list,
)
from scqubits.core.circuit_input import (
    remove_comments,
    remove_branchline,
    strip_empty_lines,
    parse_code_line,
    process_param,
)


class Node:
    """Class representing a circuit node, and handled by `Circuit`. The attribute
    `branches` is a list of `Branch` objects containing all branches connected to the
    node.

    Parameters
    ----------
    id: int
        integer identifier of the node
    marker: int
        An internal attribute used to group nodes and identify sub-circuits in the
        method independent_modes.
    """

    def __init__(self, index: int, marker: int):
        self.index = index
        self.marker = marker
        self._init_params = {"id": self.index, "marker": self.marker}
        self.branches: List[Branch] = []

    def __str__(self) -> str:
        return "Node {}".format(self.index)

    def __repr__(self) -> str:
        return "Node({})".format(self.index)

    def connected_nodes(self, branch_type: str) -> List["Node"]:
        """Returns a list of all nodes directly connected by branches to the current
        node, either considering all branches or a specified `branch_type` - ("C", "L", "JJ", "all") for capacitive, inductive, Josephson junction, or all types of branches.
        """
        result = []
        if branch_type == "all":
            branch_list = self.branches
        else:
            branch_list = [
                branch for branch in self.branches if branch.type == branch_type
            ]
        for branch in branch_list:
            if branch.nodes[0].index == self.index:
                result.append(branch.nodes[1])
            else:
                result.append(branch.nodes[0])
        return result

    def is_ground(self) -> bool:
        """Returns a bool if the node is a ground node.

        It is ground if the id is set to 0.
        """
        return True if self.index == 0 else False

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class Branch:
    """Class describing a circuit branch, used in the Circuit class.

    Parameters
    ----------
    n_i, n_f:
        initial and final nodes connected by this branch;
    branch_type:
        is the type of this Branch, example `"C"`,`"JJ"` or `"L"`
    parameters:
        list of parameters for the branch, namely for
        capacitance: {"EC":  <value>};
        for inductance: {"EL": <value>};
        for Josephson Junction: {"EJ": <value>, "ECJ": <value>}
        for phase-slip Junction: {"EQ": <value>}
    aux_params:
        Dictionary of auxiliary parameters which map a symbol from the input file a numeric parameter.

    Examples
    --------
    capacitive branch::
        Branch("C", Node(1, 0), Node(2, 0))
    is a capacitive branch connecting the nodes with indices 0 and 1.
    """

    def __init__(
        self,
        n_i: Node,
        n_f: Node,
        branch_type: str,
        parameters: Optional[List[Union[float, Symbol, int]]] = None,
        index: Optional[int] = None,
        aux_params: Dict[Symbol, float] = {},
    ):
        self.nodes = (n_i, n_f)
        self.type = branch_type
        self.parameters = parameters
        self.index = index
        # store info of current branch inside the provided nodes
        # setting the parameters if it is provided
        if parameters is not None:
            self.set_parameters(parameters)
        self.aux_params = aux_params
        self.nodes[0].branches.append(self)
        self.nodes[1].branches.append(self)

    def __str__(self) -> str:
        return (
            "Branch "
            + self.type
            + " connecting nodes: ("
            + str(self.nodes[0].index)
            + ","
            + str(self.nodes[1].index)
            + "); "
            + str(self.parameters)
        )

    def __repr__(self) -> str:
        return f"Branch({self.type}, {self.nodes[0].index}, {self.nodes[1].index}, index: {self.index})"

    def set_parameters(self, parameters) -> None:
        if self.type in ["C", "L"]:
            self.parameters = {f"E{self.type}": parameters[0]}
        elif "JJ" in self.type:
            number_of_junc_params = _junction_order(self.type)
            self.parameters = {}
            for junc_order in range(1, number_of_junc_params + 1):
                if junc_order == 1:
                    self.parameters["EJ"] = parameters[0]
                else:
                    self.parameters[f"EJ{junc_order}"] = parameters[junc_order - 1]
            self.parameters["ECJ"] = parameters[number_of_junc_params]
        elif "PSJ" in self.type:
            number_of_junc_params = _junction_order(self.type)
            self.parameters = {}
            for junc_order in range(1, number_of_junc_params + 1):
                if junc_order == 1:
                    self.parameters["EQ"] = parameters[0]
                else:
                    self.parameters[f"EQ{junc_order}"] = parameters[junc_order - 1]

    def node_ids(self) -> Tuple[int, int]:
        """Returns the indices of the nodes connected by the branch."""
        return self.nodes[0].index, self.nodes[1].index

    def is_connected(self, branch) -> bool:
        """Returns a boolean indicating whether the current branch is connected to the
        given `branch`"""
        distinct_node_count = len(set(self.nodes + branch.nodes))
        if distinct_node_count < 4:
            return True
        return False

    def common_node(self, branch) -> Set[Node]:
        """Returns the common nodes between self and the `branch` given as input."""
        return set(self.nodes) & set(branch.nodes)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class Coupler:
    """Coupler class is used to define elements which couple two existing branches in
    the Circuit class.

    Parameters
    ----------
    branch1, branch2:
        Branch objects which are being coupled.
    coupling_type:
        The type of coupling between the branches - allowed is mutual inductance - "ML"
    parameters:
        List of parameters for the coupling, namely for mutual inductance: {"EM": <value>}
    aux_params:
        Dictionary of auxiliary parameters which map a symbol from the input file a numeric parameter.

    Examples
    --------
    `Coupler("ML", branch1, branch2, "ML", 1e2)`
    """

    def __init__(
        self,
        branch1: Branch,
        branch2: Branch,
        coupling_type: str,
        parameters: Optional[List[Union[float, Symbol, int]]] = None,
        index: Optional[int] = None,
        aux_params: Dict[Symbol, float] = {},
    ):
        self.branches = (branch1, branch2)
        self.type = coupling_type
        if parameters is not None:
            self.set_parameters(parameters)
        self.aux_params = aux_params
        self.index = index

    def set_parameters(self, parameters) -> None:
        if self.type in ["ML"]:
            self.parameters = {f"E{self.type}": parameters[0]}

    def __repr__(self) -> str:
        return f"Coupler({self.type}, ({self.branches[0].type}, {self.branches[0].node_ids()}), ({self.branches[1].type}, {self.branches[1].node_ids()}), index: {self.index})"

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


def make_coupler(
    branches_list: List[Branch],
    coupler_type: str,
    idx1: int,
    idx2: int,
    params,
    aux_params,
    _branch_count: int,
):
    params_dict = {}
    params = [process_param(param) for param in params]
    if coupler_type == "ML":
        params_dict[sm.symbols("EML")] = (
            params[0][0] if params[0][0] is not None else params[0][1]
        )
        for idx in [idx1, idx2]:
            if branches_list[idx].type != "L":
                raise ValueError(
                    "Mutual inductance coupling is only allowed between inductive branches."
                )
        branch1 = branches_list[idx1]
        branch2 = branches_list[idx2]
    sym_params_dict = {
        param[0]: param[1] for param in params if param[0] is not None
    }  # dictionary of symbolic params and the default values
    return (
        Coupler(
            branch1,
            branch2,
            coupler_type,
            list(params_dict.values()),
            _branch_count,
            process_param(aux_params),
        ),
        sym_params_dict,
    )


def make_branch(
    nodes_list: List[Node],
    branch_type: str,
    idx1: int,
    idx2: int,
    params,
    aux_params,
    _branch_count: int,
):
    params_dict = {}
    params = [process_param(param) for param in params]

    if "JJ" in branch_type:
        for idx, param in enumerate(
            params[:-1]
        ):  # getting EJi for all orders i specified
            params_dict[sm.symbols(f"EJ{idx + 1}" if idx > 0 else "EJ")] = (
                param[0] if param[0] is not None else param[1]
            )
        params_dict[sm.symbols("EC")] = (
            params[-1][0] if params[-1][0] is not None else params[-1][1]
        )

    if "PSJ" in branch_type:
        for idx, param in enumerate(params):  # getting EQi for all orders i specified
            params_dict[sm.symbols(f"EQ{idx + 1}" if idx > 0 else "EQ")] = (
                param[0] if param[0] is not None else param[1]
            )

    if branch_type == "C":
        params_dict[sm.symbols("EC")] = (
            params[-1][0] if params[-1][0] is not None else params[-1][1]
        )
    elif branch_type == "L":
        params_dict[sm.symbols("EL")] = (
            params[-1][0] if params[-1][0] is not None else params[-1][1]
        )

    # return idx1, idx2, branch_type, list(params_dict.keys()), str(_branch_count), process_param(aux_params)
    is_grounded = True if any([node.is_ground() for node in nodes_list]) else False
    node_1 = nodes_list[idx1 if is_grounded else idx1 - 1]
    node_2 = nodes_list[idx2 if is_grounded else idx2 - 1]
    sym_params_dict = {
        param[0]: param[1] for param in params if param[0] is not None
    }  # dictionary of symbolic params and the default values
    return (
        Branch(
            node_1,
            node_2,
            branch_type,
            list(params_dict.values()),
            _branch_count,
            process_param(aux_params),
        ),
        sym_params_dict,
    )


class SymbolicCircuit(serializers.Serializable):
    r"""Describes a circuit consisting of nodes and branches.

    Examples
    --------
    For a transmon qubit, the input file reads::

        # file_name: transmon_num.inp
        nodes: 2
        branches:
        C       1,2     1
        JJ      1,2     1       10

    The `Circuit` object can be initiated using::
        `Circuit.from_input_file("transmon_num.inp")`

    Parameters
    ----------
    nodes_list: List[Node]
        List of nodes in the circuit
    branches_list: List[Branch]
        List of branches connecting the above set of nodes.
    couplers_list: List[Coupler]
        List of couplers connecting the branches.
    basis_completion: str
        choices are: "heuristic" (default) or "canonical"; selects type of basis for
        completing the transformation matrix.
    use_dynamic_flux_grouping: bool
        set to False by default. Indicates if the flux allocation is done by assuming
        that flux is time dependent. When set to True, it disables the option to change
        the closure branches.
    initiate_sym_calc: bool
        set to True by default. Initiates the object attributes by calling the
        function initiate_symboliccircuit method when set to True.
    """

    def __init__(
        self,
        nodes_list: List[Node],
        branches_list: List[Branch],
        couplers_list: List[Coupler],
        branch_var_dict: Dict[Union[Any, Symbol], Union[Any, float]],
        basis_completion: str = "heuristic",
        use_dynamic_flux_grouping: bool = False,
        initiate_sym_calc: bool = True,
        input_string: str = "",
    ):
        self.branches = branches_list
        self.nodes = nodes_list
        self.couplers = couplers_list
        self.input_string = input_string

        self._sys_type = type(self).__name__  # for object description

        # attributes set by methods
        self.transformation_matrix: Optional[ndarray] = None

        self.var_categories: Optional[List[int]] = None
        self.external_fluxes: List[Symbol] = []
        self.closure_branches: Optional[List[Union[Branch, Dict[Branch, float]]]] = []

        self.symbolic_params: Dict[Symbol, float] = branch_var_dict

        self.hamiltonian_symbolic: Optional[sympy.Expr] = None
        # to store the internally used lagrangian
        self._lagrangian_symbolic: Optional[sympy.Expr] = None
        self.lagrangian_symbolic: Optional[sympy.Expr] = None
        # symbolic lagrangian in terms of untransformed generalized flux variables
        self.lagrangian_node_vars: Optional[sympy.Expr] = None
        # symbolic expression for potential energy
        self.potential_symbolic: Optional[sympy.Expr] = None
        self.potential_node_vars: Optional[sympy.Expr] = None

        # parameters for grounding the circuit
        self.is_grounded = False
        self.ground_node = None
        for node in self.nodes:
            if node.is_ground():
                self.ground_node = node
                self.is_grounded = True

        # switch to control the dynamic flux allocation in the loops
        self.use_dynamic_flux_grouping = use_dynamic_flux_grouping

        # parameter for choosing matrix used for basis completion in the variable
        # transformation matrix
        self.basis_completion = (
            basis_completion  # default, the other choice is standard
        )

        self.initiate_sym_calc = initiate_sym_calc

        # Calling the function to initiate the class variables
        if initiate_sym_calc:
            self.configure()

    def is_any_branch_parameter_symbolic(self):
        return True if len(self.symbolic_params) > 0 else False

    @staticmethod
    def _gram_schmidt(initial_vecs: ndarray, metric: ndarray) -> ndarray:
        def inner_product(u, v, metric):
            return u @ metric @ v

        def projection(u, v, metric):
            """Projection of u on v.

            Parameters
            ----------
            u : ndarray
            v : ndarray
            """
            return v * inner_product(v, u, metric) / inner_product(v, v, metric)

        orthogonal_vecs = [initial_vecs[0]]
        for i in range(1, len(initial_vecs)):
            vec = initial_vecs[i]
            projection_on_orthovecs = sum(
                [projection(vec, ortho_vec, metric) for ortho_vec in orthogonal_vecs]
            )
            orthogonal_vecs.append(vec - projection_on_orthovecs)
        return np.array(orthogonal_vecs).T

    def _orthogonalize_degenerate_eigen_vecs(
        self, evecs: ndarray, eigs: ndarray, relevant_eig_indices, cap_matrix: ndarray
    ) -> ndarray:
        relevant_eigs = eigs[relevant_eig_indices]
        unique_eigs = np.unique(np.round(relevant_eigs, 10))
        close_eigs = [
            list(np.where(np.abs(eigs - eig) < 1e-10)[0]) for eig in unique_eigs
        ]
        degenerate_indices_list = [
            indices for indices in close_eigs if len(indices) > 1
        ]

        orthogonal_evecs = evecs.copy()

        for degenerate_set in degenerate_indices_list:
            orthogonal_evecs[:, degenerate_set] = self._gram_schmidt(
                evecs[:, degenerate_set].T, metric=cap_matrix
            )

        return orthogonal_evecs

    def purely_harmonic_transformation(self) -> Tuple[ndarray, ndarray]:
        trans_mat, _ = self.variable_transformation_matrix()
        c_mat = (
            trans_mat.T @ self._capacitance_matrix(substitute_params=True) @ trans_mat
        )
        l_inv_mat = (
            trans_mat.T
            @ self._inductance_inverse_matrix(substitute_params=True)
            @ trans_mat
        )
        if not self.is_grounded:
            c_mat = c_mat[:-1, :-1]
            l_inv_mat = l_inv_mat[:-1, :-1]
        normal_mode_freqs, normal_mode_vecs = sp.linalg.eig(l_inv_mat, c_mat)
        normal_mode_freqs = normal_mode_freqs.round(10)
        # rounding to the tenth digit to remove numerical errors in eig calculation
        # rearranging the vectors
        idx = normal_mode_freqs.argsort()
        normal_freq_ids = [
            id
            for id in idx
            if normal_mode_freqs[id] != 0 and not np.isinf(normal_mode_freqs[id])
        ]
        zero_freq_ids = [id for id in idx if normal_mode_freqs[id] == 0]
        inf_freq_ids = [id for id in idx if np.isinf(normal_mode_freqs[id])]
        idx = normal_freq_ids + zero_freq_ids + inf_freq_ids
        # sorting so that all the zero frequencies show up at the end

        normal_mode_freqs = normal_mode_freqs[idx]
        normal_mode_vecs = normal_mode_vecs[:, idx]

        orthogonalized_normal_mode_vecs = self._orthogonalize_degenerate_eigen_vecs(
            normal_mode_vecs, normal_mode_freqs, range(len(normal_freq_ids)), c_mat
        )

        # constructing the new transformation
        trans_mat_new = trans_mat.copy()
        trans_mat_new[:, : len(c_mat)] = (
            trans_mat[:, : len(c_mat)] @ orthogonalized_normal_mode_vecs
        )

        return (
            np.real(
                np.sqrt(
                    [
                        freq
                        for freq in normal_mode_freqs
                        if not np.isinf(freq) and freq != 0
                    ]
                )
            ),
            trans_mat_new,
        )

    def configure(
        self,
        transformation_matrix: Optional[ndarray] = None,
        closure_branches: Optional[List[Union[Branch, Dict[Branch, float]]]] = None,
        use_dynamic_flux_grouping: Optional[bool] = None,
    ):
        """Method to initialize the CustomQCircuit instance and initialize all the
        attributes needed before it can be passed on to AnalyzeQCircuit.

        Parameters
        ----------
        transformation_matrix:
            array used to set a transformation matrix other than the one generated by
            the method `variable_transformation_matrix`.
        closure_branches:
            Each element of the list corresponds to one external flux variable. If the element is a branch
            the external flux will be associated with that branch. If the element is a dictionary, the external flux variable
            will be distributed across the branches according to the dictionary with the factor given as a key value.
        """
        # if the circuit is purely harmonic, then store the eigenfrequencies
        branch_type_list = [branch.type for branch in self.branches]
        self.is_purely_harmonic = not any(
            branch_type in "".join(branch_type_list) for branch_type in ["JJ", "PSJ"]
        )
        if use_dynamic_flux_grouping is not None:
            self.use_dynamic_flux_grouping = use_dynamic_flux_grouping

        ################# setting the spanning tree and closure branches #################
        # if the user provides a transformation matrix
        if transformation_matrix is not None:
            self.var_categories = self.check_transformation_matrix(
                transformation_matrix, enable_warnings=not self.is_purely_harmonic
            )
            self.transformation_matrix = transformation_matrix
        # calculate the transformation matrix and identify the boundary conditions if
        # the user does not provide a custom transformation matrix
        else:
            (
                self.transformation_matrix,
                self.var_categories,
            ) = self.variable_transformation_matrix()

        # find the closure branches in the circuit
        default_spanning_tree_dict = self._spanning_tree(
            consider_capacitive_loops=self.use_dynamic_flux_grouping,
            use_closure_branches=False,
        )
        if closure_branches:
            if len(closure_branches) != len(
                flatten_list_recursive(
                    default_spanning_tree_dict["closure_branches_for_trees"]
                )
            ):
                raise ValueError(
                    """The number of closure branches should be equal to the number of loops present in the circuit. Please check the attribute spanning_tree_dict of the SymbolicCircuit instance."""
                )
            self.closure_branches = closure_branches
            self.spanning_tree_dict = self._spanning_tree(
                consider_capacitive_loops=self.use_dynamic_flux_grouping
            )
        else:
            self.spanning_tree_dict = default_spanning_tree_dict
            self.closure_branches = self._closure_branches()

        # setting external flux and offset charge variables
        self._set_external_fluxes(closure_branches=self.closure_branches)
        #####################################################################

        ################# setting the transformation matrix #################
        if self.is_purely_harmonic:
            (
                self.normal_mode_freqs,
                transformation_matrix_normal_mode,
            ) = self.purely_harmonic_transformation()
            if transformation_matrix is None:
                transformation_matrix = transformation_matrix_normal_mode

        # if the user provides a transformation matrix
        if transformation_matrix is not None:
            self.var_categories = self.check_transformation_matrix(
                transformation_matrix, enable_warnings=not self.is_purely_harmonic
            )
            self.transformation_matrix = transformation_matrix
        # calculate the transformation matrix and identify the boundary conditions if
        # the user does not provide a custom transformation matrix
        else:
            (
                self.transformation_matrix,
                self.var_categories,
            ) = self.variable_transformation_matrix()
        #####################################################################

        # find the closure branches in the circuit
        default_spanning_tree_dict = self._spanning_tree(
            consider_capacitive_loops=self.use_dynamic_flux_grouping,
            use_closure_branches=False,
        )
        if closure_branches:
            if len(closure_branches) != len(
                flatten_list_recursive(
                    default_spanning_tree_dict["closure_branches_for_trees"]
                )
            ):
                raise ValueError(
                    """The number of closure branches should be equal to the number of loops present in the circuit. Please check the attribute spanning_tree_dict of the SymbolicCircuit instance."""
                )
            self.closure_branches = closure_branches
            self.spanning_tree_dict = self._spanning_tree(
                consider_capacitive_loops=self.use_dynamic_flux_grouping
            )
        else:
            self.spanning_tree_dict = default_spanning_tree_dict
            self.closure_branches = self._closure_branches()

        # setting external flux and offset charge variables
        self._set_external_fluxes(closure_branches=self.closure_branches)
        self.offset_charges = [
            symbols(f"ng{index}") for index in self.var_categories["periodic"]
        ]
        self.free_charges = [
            symbols(f"Qf{index}") for index in self.var_categories["free"]
        ]
        # store the flux allocation for each branches
        self.branch_flux_allocations = self._generate_branch_flux_allocations()
        # calculating the Hamiltonian directly when the number of nodes is less than 3
        substitute_params = False
        if (
            len(self.nodes) >= settings.SYM_INVERSION_MAX_NODES
            or len(self.var_categories["frozen"]) > 0
        ):  # only calculate the symbolic hamiltonian when the number of nodes is less
            # than 3. Else, the calculation will be skipped to the end when numerical
            # Hamiltonian of the circuit is requested.
            substitute_params = True
            self.frozen_var_exprs = {}

        # Calculate the Lagrangian
        (
            self._lagrangian_symbolic,
            self.potential_symbolic,
            self.lagrangian_node_vars,
            self.potential_node_vars,
        ) = self.generate_symbolic_lagrangian(substitute_params=substitute_params)

        # replacing energies with capacitances in the kinetic energy of the Lagrangian
        (
            self.lagrangian_symbolic,
            self.lagrangian_node_vars,
        ) = self._replace_energies_with_capacitances_L()

        self.hamiltonian_symbolic = self.generate_symbolic_hamiltonian(
            substitute_params=substitute_params
        )

    def _replace_energies_with_capacitances_L(self):
        """Method replaces the energies in the Lagrangian with capacitances which are
        arbitrarily generated to make sure that the Lagrangian looks dimensionally
        correct."""
        # Replacing energies with capacitances if any branch parameters are symbolic
        L = self._lagrangian_symbolic.expand()
        L_old = self.lagrangian_node_vars
        if self.is_any_branch_parameter_symbolic():
            # finding the unique capacitances
            uniq_capacitances = []
            for c, b in enumerate(
                [t for t in self.branches if t.type == "C" or "JJ" in t.type]
            ):
                if len(set(b.nodes)) > 1:  # check to see if branch is shorted
                    if (
                        b.parameters[_capactiance_variable_for_branch(b.type)]
                        not in uniq_capacitances
                    ):
                        uniq_capacitances.append(
                            b.parameters[_capactiance_variable_for_branch(b.type)]
                        )

            for index, var in enumerate(uniq_capacitances):
                L = L.subs(var, 1 / (8 * symbols(f"C{index + 1}")))
                L_old = L_old.subs(var, 1 / (8 * symbols(f"C{index + 1}")))
        return L, L_old

    # Serialize will not currently work for the Circuit class.
    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

        return {}

    @staticmethod
    def are_branchsets_disconnected(
        branch_list1: List[Branch], branch_list2: List[Branch]
    ) -> bool:
        """Determines whether two sets of branches are disconnected.

        Parameters
        ----------
        branch_list1:
            first list of branches
        branch_list2:
            second list of branches

        Returns
        -------
        bool
            Returns True if the branches have a connection, else False
        """
        node_array1 = np.array([branch.node_ids() for branch in branch_list1]).flatten()
        node_array2 = np.array([branch.node_ids() for branch in branch_list2]).flatten()
        return np.intersect1d(node_array1, node_array2).size == 0

    @staticmethod
    def _parse_nodes(branches_list) -> Tuple[Optional[Node], List[Node]]:
        node_index_list = []
        for branch_list_input in [
            branch for branch in branches_list if branch[0] != "ML"
        ]:
            for idx in [1, 2]:
                node_idx = branch_list_input[idx]
                if node_idx not in node_index_list:
                    node_index_list.append(node_idx)
        node_index_list.sort()
        return [Node(idx, 0) for idx in node_index_list]

    @classmethod
    def from_yaml(
        cls,
        input_string: str,
        from_file: bool = True,
        basis_completion: str = "heuristic",
        use_dynamic_flux_grouping: Optional[bool] = None,
        initiate_sym_calc: bool = True,
    ):
        """
        Constructs the instance of Circuit from an input string. Here is an example of
        an input string that is used to initiate an object of the
        class `SymbolicCircuit`::
            #zero-pi.yaml
            nodes    : 4
            # zero-pi
            branches:
            - [JJ, 1,2, EJ = 10, 20]
            - [JJ, 3,4, 10, 20]
            - [L, 2,3, 0.008]
            - [L, 4,1, 0.008]
            - [C, 1,3, 0.02]
            - [C, 2,4, 0.02]

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
            choices: "heuristic" or "canonical"; used to choose a type of basis
            for completing the transformation matrix. Set to "heuristic" by default.
        use_dynamic_flux_grouping: bool
            set to False by default. Indicates if the flux allocation is done by
            assuming that flux is time dependent. When set to True, it disables the
            option to change the closure branches.
        initiate_sym_calc:
            set to True by default. Initiates the object attributes by calling
            the function `initiate_symboliccircuit` method when set to True.
            Set to False for debugging.

        Returns
        -------
            Instance of the class `SymbolicCircuit`
        """
        if from_file:
            file = open(input_string, "r")
            circuit_desc = file.read()
            file.close()
        else:
            circuit_desc = input_string

        input_str = remove_comments(circuit_desc)
        input_str = remove_branchline(input_str)
        input_str = strip_empty_lines(input_str)

        parsed_branches = [
            parse_code_line(code_line, branch_count)
            for branch_count, code_line in enumerate(input_str.split("\n"))
        ]

        # find and create the nodes
        nodes_list = cls._parse_nodes(parsed_branches)
        # if the node indices do not start from 0, raise an error
        node_ids = [node.index for node in nodes_list]
        if min(node_ids) not in [0, 1]:
            raise ValueError("The node indices should start from 0 or 1.")
        # parse branches and couplers
        branches_list = []
        couplers_list = []
        branch_var_dict = {}
        # make individual branches
        individual_branches = [
            branch for branch in parsed_branches if branch[0] != "ML"
        ]
        for parsed_branch in individual_branches:
            branch, sym_params = make_branch(nodes_list, *parsed_branch)
            for sym_param in sym_params:
                if sym_param in branch_var_dict and sym_params[sym_param] is not None:
                    raise Exception(
                        f"Symbol {sym_param} has already been assigned a value."
                    )
                if sym_params[sym_param] is not None:
                    branch_var_dict[sym_param] = sym_params[sym_param]
            branches_list.append(branch)
        # make couplers
        coupler_branches = [
            branch for branch in parsed_branches if branch not in individual_branches
        ]
        for parsed_branch in coupler_branches:
            coupler, sym_params = make_coupler(branches_list, *parsed_branch)
            for sym_param in sym_params:
                if sym_param in branch_var_dict and sym_params[sym_param] is not None:
                    raise Exception(
                        f"Symbol {sym_param} has already been assigned a value."
                    )
                if sym_params[sym_param] is not None:
                    branch_var_dict[sym_param] = sym_params[sym_param]
            couplers_list.append(coupler)

        circuit = cls(
            nodes_list,
            branches_list,
            couplers_list,
            use_dynamic_flux_grouping=use_dynamic_flux_grouping,
            branch_var_dict=branch_var_dict,
            basis_completion=basis_completion,
            initiate_sym_calc=initiate_sym_calc,
            input_string=circuit_desc,
        )
        return circuit

    def _independent_modes(
        self,
        branch_subset: List[Branch],
        single_nodes: bool = True,
        basisvec_entries: Optional[List[int]] = None,
    ):
        """Returns the vectors which span a subspace where there is no generalized flux
        difference across the branches present in the branch_subset.

        Parameters
        ----------
        single_nodes:
            if the single nodes are taken into consideration for basis vectors.
        """
        if basisvec_entries is None:
            basisvec_entries = [1, 0]

        nodes_copy = copy.copy(self.nodes)  # copying self.nodes as it is being modified

        # making sure that the ground node is placed at the end of the list
        if self.is_grounded:
            nodes_copy.pop(0)  # removing the ground node
            nodes_copy = nodes_copy + [
                copy.copy(self.ground_node)
            ]  # reversing the order of the nodes

        for node in nodes_copy:  # reset the node markers
            node.marker = 0

        # step 2: finding the maximum connected set of independent branches in
        # branch_subset, then identifying the sets of nodes in each of those sets
        branch_subset_copy = branch_subset.copy()

        max_connected_subgraphs = []  # list containing the maximum connected subgraphs

        while (
            len(branch_subset_copy) > 0
        ):  # while loop ends when all the branches are sorted
            b_0 = branch_subset_copy.pop(0)
            max_connected_subgraph = [b_0]

            while not self.are_branchsets_disconnected(
                max_connected_subgraph, branch_subset_copy
            ):
                for b1 in branch_subset_copy:
                    for b2 in max_connected_subgraph:
                        if b1.is_connected(b2):
                            max_connected_subgraph.append(b1)
                            branch_subset_copy.remove(b1)
                            break
            max_connected_subgraphs.append(max_connected_subgraph)

        # finding the nodes in each of the maximum connected subgraph
        nodes_in_max_connected_branchsets = [
            unique_elements_in_list(sum([branch.nodes for branch in branch_set], ()))
            for branch_set in max_connected_subgraphs
        ]

        # using node.marker to mark the maximum connected subgraph to which a node
        # belongs
        for node_set_index, node_set in enumerate(nodes_in_max_connected_branchsets):
            for node in node_set:
                if any([n.is_ground() for n in node_set]):
                    node.marker = -1
                else:
                    node.marker = node_set_index + 1

        # marking ground nodes separately
        for node in nodes_copy:
            if node.is_ground():
                node.marker = -1

        node_branch_set_indices = [
            node.marker for node in nodes_copy
        ]  # identifies which node belongs to which maximum connected subgraphs;
        # different numbers on two nodes indicates that they are not connected through
        # any of the branches in branch_subset. 0 implies the node does not belong to
        # any of the branches in max connected branch subsets and -1 implies the max
        # connected branch set is connected to ground.

        # step 3: Finding the linearly independent vectors spanning the vector space
        # represented by branch_set_index
        basis = []

        unique_branch_set_markers = unique_elements_in_list(node_branch_set_indices)
        # removing the marker -1 as it is grounded.
        branch_set_markers_ungrounded = [
            marker for marker in unique_branch_set_markers if marker != -1
        ]

        for index in branch_set_markers_ungrounded:
            basis.append(
                [
                    basisvec_entries[0] if i == index else basisvec_entries[1]
                    for i in node_branch_set_indices
                ]
            )

        if single_nodes:  # taking the case where the node_branch_set_index is 0
            single_node_modes = []
            if node_branch_set_indices.count(0) > 0:
                ref_vector = [
                    basisvec_entries[0] if i == 0 else basisvec_entries[1]
                    for i in node_branch_set_indices
                ]
                positions = [
                    index
                    for index, num in enumerate(ref_vector)
                    if num == basisvec_entries[0]
                ]
                for pos in positions:
                    single_node_modes.append(
                        [
                            basisvec_entries[0] if x == pos else basisvec_entries[1]
                            for x, num in enumerate(node_branch_set_indices)
                        ]
                    )

            for mode in single_node_modes:
                mat = np.array(basis + [mode])
                if np.linalg.matrix_rank(mat) == len(mat):
                    basis.append(mode)

        if (
            self.is_grounded
        ):  # if grounded remove the last column and first row corresponding to the
            basis = [i[:-1] for i in basis]

        return basis

    @staticmethod
    def _mode_in_subspace(mode, subspace) -> bool:
        """Method to check if the vector mode is a part of the subspace provided as a
        set of vectors.

        Parameters
        ----------
        mode:
            numpy ndarray of one dimension.
        subspace:
            numpy ndarray which represents a collection of basis vectors for a vector
            subspace
        """
        if len(subspace) == 0:
            return False
        matrix = np.vstack([subspace, np.array(mode)])
        return np.linalg.matrix_rank(matrix) == len(subspace)

    def _mode_dictionary(
        self, return_linearly_independent_modes=False, spanning_tree_dict=None
    ) -> Union[Tuple[Union[dict, ndarray], ...], dict]:
        """
        Returns a dictionary of modes present in the circuit. This will return all the possible modes that can be found,
        other than extended modes: discrete, periodic, frozen, free, sigma and LC modes. Also, union of all the modes might not be linearly independent.


        Returns
        -------
        dict
            Dictionary of modes with keys "discrete", "periodic", "frozen" and "free" modes.
        """
        modes_dict: Dict[str, list] = {
            "discrete": [],
            "periodic": [],
            "frozen": [],
            "free": [],
        }
        # *************************** Finding the Discrete Modes **********************
        # these are discrete flux modes coming from QPS in a loop with inductive branches
        selected_branches = [
            branch
            for branch in self.branches
            if branch.type == "C" or "JJ" in branch.type
        ]
        modes_dict["discrete"] = self._independent_modes(
            selected_branches, single_nodes=True
        )

        # ****************  Finding the Frozen modes ****************
        selected_branches = [branch for branch in self.branches if branch.type != "L"]
        modes_dict["frozen"] = self._independent_modes(
            selected_branches, single_nodes=True
        )

        # ****************  Finding the Periodic Modes ****************
        selected_branches = [
            branch
            for branch in self.branches
            if branch.type == "L" or "PSJ" in branch.type
        ]
        modes_dict["periodic"] = self._independent_modes(
            selected_branches, single_nodes=True
        )

        # **************** Finding the Free Modes ****************
        selected_branches = [branch for branch in self.branches if branch.type != "C"]
        modes_dict["free"] = self._independent_modes(
            selected_branches, single_nodes=True
        )

        # **************** including the Σ mode ****************
        modes_dict["sigma"] = []
        Σ = [1] * (len(self.nodes) - self.is_grounded)
        if not self.is_grounded:  # only append if the circuit is not grounded
            modes_dict["sigma"] = [Σ]
        # identify sigma mode for galvanically disconnected subcircuits
        # find the sigma mode for all the subcircuits with different node sets
        node_sets = self._node_sets(self)
        for node_set in node_sets:
            node_set = flatten_list_recursive(node_set)
            if self.ground_node in node_set:
                continue
            modes_dict["sigma"].append(
                [
                    1 if node in node_set else 0
                    for node in self.nodes if node.index!=0
                ]
            )


        # Do note that:
        # 1. Discrete modes form a superset of the frozen modes
        # 2. Periodic modes form a superset of the free modes

        # **************** Finding the LC Modes ****************
        selected_branches = [
            branch
            for branch in self.branches
            if "JJ" in branch.type or "PSJ" in branch.type
        ]
        modes_dict["LC"] = self._independent_modes(selected_branches, single_nodes=True)

        if return_linearly_independent_modes:
            return modes_dict, *self._find_linearly_independent_modes(modes_dict)
        return modes_dict

    def _find_linearly_independent_modes(
        self, modes_dict
    ) -> Tuple[Dict[str, list], ndarray]:
        """
        Finds the dynamical linearly independent modes from the modes_dict. The non-dynamical modes are the free, frozen, and sigma modes.
        """
        # **************** Making the basis set with linearly independent vectors *********
        mode_types_in_order = [
            "sigma",
            "frozen",
            "free",
            "periodic",
            "discrete",
            "LC",
        ]  # this order is important

        linearly_independent_modes = []  # starting with an empty list
        mode_category_index_dict = {
            mode_type: [] for mode_type in modes_dict
        }  # saving the final modes
        for mode_type in mode_types_in_order:
            for m in modes_dict[mode_type]:
                if not self._mode_in_subspace(m, linearly_independent_modes):
                    linearly_independent_modes.append(m)
                    mode_category_index_dict[mode_type].append(
                        len(linearly_independent_modes) - 1
                    )

        # basis used to find the rest of the extended modes
        standard_basis = self._standard_basis()
        mode_category_index_dict["extended"] = []
        # filling the basis
        for m in standard_basis:  # completing the basis
            if not self._mode_in_subspace(m, linearly_independent_modes):
                linearly_independent_modes.append(m)
                mode_category_index_dict["extended"].append(
                    len(linearly_independent_modes) - 1
                )
        linearly_independent_modes = np.array(linearly_independent_modes)

        # sort the modes in the order periodic, discrete, extended, free, frozen, sigma
        mode_types_in_order = [
            "periodic",
            "discrete",
            "LC",
            "extended",
            "free",
            "frozen",
            "sigma",
        ]
        linearly_independent_modes = linearly_independent_modes[
            flatten_list_recursive(
                [
                    mode_category_index_dict[mode_type]
                    for mode_type in mode_types_in_order
                ]
            )
        ]
        last_idx = 1
        for mode_type in mode_types_in_order:
            mode_category_index_dict[mode_type] = list(
                range(last_idx, last_idx + len(mode_category_index_dict[mode_type]))
            )
            last_idx += len(mode_category_index_dict[mode_type])
        # add LC modes to extended category
        mode_category_index_dict["extended"] += mode_category_index_dict["LC"]
        mode_category_index_dict["extended"].sort()

        return mode_category_index_dict, linearly_independent_modes

    def _standard_basis(self) -> List[List[int]]:
        """
        Returns the standard basis, which spans the space of variable transformations for this circuit, in node variables.
        """
        import itertools

        # constructing a standard basis
        if self.basis_completion == "heuristic":
            node_count = len(self.nodes) - self.is_grounded
            standard_basis = [[1] * node_count]

            vector_ref = [1] * node_count
            vector_ref[-1] = 0
            if node_count > 2:
                vector_ref[-2] = 0

            vector_set = (
                permutation
                for permutation in itertools.permutations(vector_ref, node_count)
            )  # making a generator
            while np.linalg.matrix_rank(np.array(standard_basis)) < node_count:
                a = next(vector_set)
                mat = np.array(standard_basis + [a])
                if np.linalg.matrix_rank(mat) == len(mat):
                    standard_basis = standard_basis + [list(a)]

        elif self.basis_completion == "canonical":
            node_count = len(self.nodes) - self.is_grounded
            standard_basis = [
                [1 if j == i else 0 for j in range(node_count)]
                for i in range(node_count)
            ]
        return standard_basis

    def check_transformation_matrix(
        self, transformation_matrix: ndarray, enable_warnings: bool = True
    ) -> Dict[str, list]:
        """Method to identify the different modes in the transformation matrix provided
        by the user.

        Parameters
        ----------
        transformation_matrix:
            numpy ndarray which is a square matrix having the dimensions of the number
            of nodes present in the circuit.
        warnings:
            If False, will not raise the warnings regarding any unidentified modes. It
            is set to True by default.

        Returns
        -------
            A dictionary of lists which has the variable indices classified with
            var indices corresponding to the rows of the transformation matrix
        """
        # basic check to see if the matrix is invertible
        if np.linalg.det(transformation_matrix) == 0:
            raise Exception("The transformation matrix provided is not invertible.")

        # get all the modes present in the circuit
        modes_dict, var_categories, _ = self._mode_dictionary(
            return_linearly_independent_modes=True
        )

        # Classifying the modes given in the transformation by the user
        user_given_modes = transformation_matrix.transpose()
        modes_dict_user = {mode_type: [] for mode_type in var_categories}
        mode_types_in_order = [
            "sigma",
            "frozen",
            "free",
            "periodic",
            "discrete",
            "LC",
        ]  # this order is important

        def identify_mode(mode):
            for mode_type in mode_types_in_order:
                # calculate the number of periodic modes
                if (mode_type == "sigma"):
                    if (not self.is_grounded
                    and self._mode_in_subspace(modes_dict["sigma"], [mode])):
                        return "sigma"

                elif self._mode_in_subspace(mode, modes_dict[mode_type]):
                    return mode_type
            return "extended"

        for idx, mode in enumerate(user_given_modes):
            mode_type = identify_mode(mode)
            modes_dict_user[mode_type].append(idx + 1)

        # add LC modes to the extended modes
        modes_dict_user["extended"] += modes_dict_user["LC"]
        modes_dict_user["extended"].sort()

        # comparing the modes in the user defined and the code generated transformation
        for mode_type in mode_types_in_order:
            num_extra_modes = len(var_categories[mode_type]) - len(
                modes_dict_user[mode_type]
            )
            if num_extra_modes > 0 and enable_warnings:
                warnings.warn(
                    "Number of extra "
                    + mode_type
                    + " modes found: "
                    + str(num_extra_modes)
                    + "\n"
                )
        if not self.is_grounded and len(modes_dict_user["sigma"]) == 0:
            raise Exception(
                "This circuit is not grounded, and so has a sigma mode. This transformation does not have a sigma mode."
            )

        return modes_dict_user

    def variable_transformation_matrix(self) -> Tuple[ndarray, Dict[str, List[int]]]:
        """Evaluates the boundary conditions and constructs the variable transformation
        matrix, which is returned along with the dictionary `var_categories` which
        classifies the types of variables present in the circuit.

        Returns
        -------
            tuple of transformation matrix for the node variables and `var_categories`
            dict which classifies the variable types for each variable index
        """
        _, var_categories, linearly_independent_modes = self._mode_dictionary(
            return_linearly_independent_modes=True
        )

        return np.array(linearly_independent_modes).astype(float).T, var_categories

    def update_param_init_val(self, param_name, value):
        """Updates the param init val for param_name."""
        for index, param in enumerate(list(self.symbolic_params.keys())):
            if param_name == param.name:
                self.symbolic_params[param] = value
                break
        if self.is_purely_harmonic:
            (
                self.normal_mode_freqs,
                self.transformation_matrix,
            ) = self.purely_harmonic_transformation()
            self.configure()

    def _josephson_junction_terms(self):
        terms = 0
        # looping over all the junction terms
        junction_branches = [
            branch
            for branch in self.branches
            if "JJ" in branch.type and "JJs" not in branch.type
        ]
        junction_branch_order = [
            _junction_order(branch.type) for branch in junction_branches
        ]

        for branch_idx, jj_branch in enumerate(junction_branches):
            # adding external flux
            phi_ext = self.branch_flux_allocations[jj_branch.index]

            # if loop to check for the presence of ground node
            for order in range(1, junction_branch_order[branch_idx] + 1):
                junction_param = "EJ" if order == 1 else f"EJ{order}"
                if jj_branch.nodes[1].index == 0:
                    terms += -jj_branch.parameters[junction_param] * sympy.cos(
                        (order)
                        * (-sympy.symbols(f"φ{jj_branch.nodes[0].index}") + phi_ext)
                    )
                elif jj_branch.nodes[0].index == 0:
                    terms += -jj_branch.parameters[junction_param] * sympy.cos(
                        (order)
                        * (sympy.symbols(f"φ{jj_branch.nodes[1].index}") + phi_ext)
                    )
                else:
                    terms += -jj_branch.parameters[junction_param] * sympy.cos(
                        (order)
                        * (
                            (
                                sympy.symbols(f"φ{jj_branch.nodes[1].index}")
                                - sympy.symbols(f"φ{jj_branch.nodes[0].index}")
                            )
                            + phi_ext
                        )
                    )
        return terms

    def _phase_slip_junction_terms(self):
        terms = 0
        # looping over all the junction terms
        junction_branches = [
            branch
            for branch in self.branches
            if "PSJ" in branch.type and "PSJs" not in branch.type
        ]
        junction_branch_order = [
            _junction_order(branch.type) for branch in junction_branches
        ]
        flux_branch_assignment = self.branch_flux_allocations
        # Qb denotes the branch variable, and Qbd denotes Qb_dot or time derivative of Qb
        for branch_idx, psj_branch in enumerate(junction_branches):
            # adding external flux
            phi_ext = flux_branch_assignment[psj_branch.index]

            # if loop to check for the presence of ground node
            for order in range(1, junction_branch_order[branch_idx] + 1):
                junction_param = "EQ" if order == 1 else f"EQ{order}"
                terms += -psj_branch.parameters[junction_param] * sympy.cos(
                    (order) * sympy.symbols(f"Qb{psj_branch.index}")
                )
        return terms

    def _JJs_terms(self):
        """To add terms for the sawtooth josephson junction."""
        terms = 0
        # looping over all the junction terms
        junction_branches = [branch for branch in self.branches if "JJs" in branch.type]

        # defining a function for sawtooth
        saw = sympy.Function("saw", real=True)

        for branch_idx, jj_branch in enumerate(junction_branches):
            # adding external flux
            phi_ext = self.branch_flux_allocations[jj_branch.index]

            # if loop to check for the presence of ground node
            junction_param = "EJ"
            if jj_branch.nodes[1].index == 0:
                terms += jj_branch.parameters[junction_param] * saw(
                    (-sympy.symbols(f"φ{jj_branch.nodes[0].index}") + phi_ext)
                )
            elif jj_branch.nodes[0].index == 0:
                terms += jj_branch.parameters[junction_param] * saw(
                    (sympy.symbols(f"φ{jj_branch.nodes[1].index}") + phi_ext)
                )
            else:
                terms += jj_branch.parameters[junction_param] * saw(
                    (
                        (
                            sympy.symbols(f"φ{jj_branch.nodes[1].index}")
                            - sympy.symbols(f"φ{jj_branch.nodes[0].index}")
                        )
                        + phi_ext
                    )
                )
        return terms

    def _inductance_inverse_matrix(self, substitute_params: bool = False):
        """Generate a inductance matrix for the circuit.

        Parameters
        ----------
        substitute_params:
            when set to True all the symbolic branch parameters are substituted with
            their corresponding attributes in float, by default False

        Returns
        -------
        _type_
            _description_
        """
        branches_with_inductance = [
            branch for branch in self.branches if branch.type == "L"
        ]

        param_init_vals_dict = self.symbolic_params

        # filling the non-diagonal entries
        if not self.is_grounded:
            num_nodes = len(self.nodes) - self.is_grounded
        else:
            num_nodes = len(self.nodes) - self.is_grounded + 1
        if not self.is_any_branch_parameter_symbolic() or substitute_params:
            L_mat = np.zeros([num_nodes, num_nodes])
        else:
            L_mat = sympy.zeros(num_nodes)

        for branch in branches_with_inductance:
            if len(set(branch.nodes)) > 1:  # branch if shorted is not considered
                inductance = branch.parameters["EL"]
                if type(inductance) != float and substitute_params:
                    inductance = param_init_vals_dict[inductance]
                if self.is_grounded:
                    L_mat[branch.nodes[0].index, branch.nodes[1].index] += -inductance
                else:
                    L_mat[
                        branch.nodes[0].index - 1, branch.nodes[1].index - 1
                    ] += -inductance

        if not self.is_any_branch_parameter_symbolic() or substitute_params:
            L_mat = L_mat + L_mat.T - np.diag(L_mat.diagonal())
        else:
            L_mat = L_mat + L_mat.T - sympy.diag(*L_mat.diagonal())

        for i in range(L_mat.shape[0]):  # filling the diagonal entries
            L_mat[i, i] = -np.sum(L_mat[i, :])

        if self.is_grounded:  # if grounded remove the 0th column and row from L_mat
            L_mat = L_mat[1:, 1:]
        return L_mat

    def _capacitance_matrix(self, substitute_params: bool = False, in_node_vars: bool = False, remove_non_dynamical: bool = False, invert_matrix: bool = False) -> Union[ndarray, sympy.Matrix]:
        """_summary_

        Args:
            substitute_params (bool, optional): Whether symbolic parameters should be substituted. Defaults to False.
            in_node_vars (bool, optional): Set to True if the capacitance matrix is in terms of node variables, when set to True there are no non_dynamical variables and matrix inversion is not always possible. Defaults to False.
            remove_non_dynamical (bool, optional): If non dynamical variables, discrete and frozen need to be removed. Defaults to True. It is automatically set to true when invert_matrix is set to True.
            invert_matrix (bool, optional): Set to True if an inverted capacitance matrix is needed. Defaults to False.

        Raises:
            ValueError: When in_node_vars is set to True and remove_non_dynamical or invert_matrix is set to True.

        Returns:
            _type_: ndarray or sympy symbolic matrix
        """
        if in_node_vars and (remove_non_dynamical or invert_matrix):
            raise ValueError("in_node_vars can only be set to True when remove_non_dynamical and invert_matrix are set to False.")
        
        if invert_matrix:
            remove_non_dynamical = True

        branches_with_capacitance = [
            branch
            for branch in self.branches
            if ("C" == branch.type or "JJ" in branch.type)
        ]

        param_init_vals_dict = self.symbolic_params

        # filling the non-diagonal entries
        if not self.is_grounded:
            num_nodes = len(self.nodes) - self.is_grounded
        else:
            num_nodes = len(self.nodes) - self.is_grounded + 1
        if not self.is_any_branch_parameter_symbolic() or substitute_params:
            C_mat = np.zeros([num_nodes, num_nodes])
        else:
            C_mat = sympy.zeros(num_nodes)

        for branch in branches_with_capacitance:
            if len(set(branch.nodes)) > 1:  # branch if shorted is not considered
                capacitance = branch.parameters[
                    _capactiance_variable_for_branch(branch.type)
                ]
                if type(capacitance) != float and substitute_params:
                    capacitance = param_init_vals_dict[capacitance]
                if self.is_grounded:
                    C_mat[branch.nodes[0].index, branch.nodes[1].index] += -1 / (
                        capacitance * 8
                    )
                else:
                    C_mat[
                        branch.nodes[0].index - 1, branch.nodes[1].index - 1
                    ] += -1 / (capacitance * 8)

        if not self.is_any_branch_parameter_symbolic() or substitute_params:
            C_mat = C_mat + C_mat.T - np.diag(C_mat.diagonal())
        else:
            C_mat = C_mat + C_mat.T - sympy.diag(*C_mat.diagonal())

        for i in range(C_mat.shape[0]):  # filling the diagonal entries
            C_mat[i, i] = -np.sum(C_mat[i, :])

        if self.is_grounded:  # if grounded remove the 0th column and row from C_mat
            C_mat = C_mat[1:, 1:]
            
        if in_node_vars:
            return C_mat
            
        transformation_matrix = self.transformation_matrix
        # generating the C_mat_θ by inverting the capacitance matrix
        is_sym_matrix = self.is_any_branch_parameter_symbolic() and not substitute_params
        if is_sym_matrix:
            C_mat_θ = (
                transformation_matrix.T
                * C_mat
                * transformation_matrix
            )
        else:
            C_mat_θ = (
                transformation_matrix.T
                @ C_mat
                @ transformation_matrix
            )
            
        if remove_non_dynamical:
            null_capacitance_indices = [
                i - 1
                for i in self.var_categories["frozen"]
                + self.var_categories["discrete"]
                + self.var_categories["sigma"]
            ]
            if is_sym_matrix:
                relevant_indices = [
                    i for i in range(C_mat_θ.shape[0]) if i not in null_capacitance_indices
                ]
                C_mat_θ = C_mat_θ[relevant_indices, relevant_indices]
            else:
                C_mat_θ = np.delete(C_mat_θ, null_capacitance_indices, 0) # remove both the rows and columns for the null indices
                C_mat_θ = np.delete(C_mat_θ, null_capacitance_indices, 1)
            
        if invert_matrix:
            if is_sym_matrix:
                return C_mat_θ.inv()
            else:
                return np.linalg.inv(C_mat_θ)
        return C_mat_θ

    def _capacitor_terms(self):
        terms = 0
        branches_with_capacitance = [
            branch
            for branch in self.branches
            if branch.type == "C" or "JJ" in branch.type
        ]
        for c_branch in branches_with_capacitance:
            if c_branch.nodes[1].index == 0:
                terms += (
                    1
                    / (
                        16
                        * c_branch.parameters[
                            _capactiance_variable_for_branch(c_branch.type)
                        ]
                    )
                    * (symbols(f"vφ{c_branch.nodes[0].index}")) ** 2
                )
            elif c_branch.nodes[0].index == 0:
                terms += (
                    1
                    / (
                        16
                        * c_branch.parameters[
                            _capactiance_variable_for_branch(c_branch.type)
                        ]
                    )
                    * (-symbols(f"vφ{c_branch.nodes[1].index}")) ** 2
                )
            else:
                terms += (
                    1
                    / (
                        16
                        * c_branch.parameters[
                            _capactiance_variable_for_branch(c_branch.type)
                        ]
                    )
                    * (
                        symbols(f"vφ{c_branch.nodes[1].index}")
                        - symbols(f"vφ{c_branch.nodes[0].index}")
                    )
                    ** 2
                )
        return terms

    def _inductance_matrix_branch_vars(
        self, substitute_params: bool = False, return_inverse: bool = False
    ):
        """Generate a inductance matrix for the circuit, including the mutual
        inductances.

        Returns
        -------
        _type_
            _description_
        """
        num_branches = len(self.branches)
        if not self.is_any_branch_parameter_symbolic() or substitute_params:
            L_mat = np.zeros([num_branches, num_branches])
        else:
            L_mat = sympy.zeros(num_branches)
        # filling the diagonal entries
        for branch in [b for b in self.branches if b.type == "L"]:
            EL = branch.parameters["EL"]
            if type(EL) != float and substitute_params:
                EL = self.symbolic_params[EL]
            L_mat[branch.index, branch.index] = 1 / EL
        # filling the non-diagonal entries
        for idx, coupler in enumerate([c for c in self.couplers if c.type == "ML"]):
            EML = coupler.parameters["EML"]
            if type(EML) != float and substitute_params:
                EML = self.symbolic_params[EML]
            L_mat[coupler.branches[0].index, coupler.branches[1].index] = 1 / EML
            L_mat[coupler.branches[1].index, coupler.branches[0].index] = 1 / EML
        # remove all zero rows and columns
        irrelevant_indices = [
            idx for idx in range(num_branches) if self.branches[idx].type != "L"
        ]
        L_mat = np.delete(L_mat, irrelevant_indices, axis=0)
        L_mat = np.delete(L_mat, irrelevant_indices, axis=1)
        if return_inverse:
            if substitute_params or not self.is_any_branch_parameter_symbolic():
                return np.linalg.inv(L_mat)
            else:
                # return sympy matrix inverse
                return sympy.Matrix(L_mat).inv()

        return L_mat

    def _generate_branch_flux_allocations(self):
        """Returns an array of the flux allocation for each branch in the circuit."""
        if self.use_dynamic_flux_grouping:
            return self._time_dependent_flux_distribution()

        if not self.closure_branches or len(self.closure_branches) == 0:
            return np.zeros(len(self.branches))
        flux_allocation_array = np.zeros(
            (len(self.branches), len(self.closure_branches))
        )
        if not self.closure_branches:
            return flux_allocation_array
        for flux_idx, element in enumerate(self.closure_branches):
            if isinstance(element, Branch):
                flux_allocation_array[element.index, flux_idx] = 1
                continue
            for branch in element:
                flux_allocation_array[branch.index, flux_idx] = element[branch]
        return np.dot(flux_allocation_array, self.external_fluxes)

    def _inductor_terms(self, substitute_params: bool = False):
        """Returns terms corresponding to purely inductive branches in the circuit."""
        inverse_inductance_mat = self._inductance_matrix_branch_vars(
            substitute_params, return_inverse=True
        )
        branch_fluxes = []
        for l_branch in [branch for branch in self.branches if branch.type == "L"]:
            # adding external flux
            phi_ext = self.branch_flux_allocations[l_branch.index]

            if l_branch.nodes[0].index == 0:
                branch_flux = symbols(f"φ{l_branch.nodes[1].index}") + phi_ext
            elif l_branch.nodes[1].index == 0:
                branch_flux = -symbols(f"φ{l_branch.nodes[0].index}") + phi_ext
            else:
                branch_flux = (
                    symbols(f"φ{l_branch.nodes[1].index}")
                    - symbols(f"φ{l_branch.nodes[0].index}")
                    + phi_ext
                )
            branch_fluxes.append(branch_flux)
        branch_currents = inverse_inductance_mat @ np.array(branch_fluxes)
        terms = 0
        for idx, branch in enumerate(
            [branch for branch in self.branches if branch.type == "L"]
        ):
            terms += 0.5 * 1 / (branch.parameters["EL"]) * branch_currents[idx] ** 2
        # substitute params if necessary
        if substitute_params and terms != 0:
            for symbol in self.symbolic_params:
                terms = terms.subs(symbol.name, self.symbolic_params[symbol])
        return terms
    
    @staticmethod
    def _node_sets(circuit):
        node_sets_for_trees = []  # seperate node sets for separate trees
        list_of_nodes = circuit.nodes.copy()
        if circuit.is_grounded:
            node_sets = [[circuit.ground_node]]
            list_of_nodes.remove(circuit.ground_node)
        else:
            node_sets = [
                [list_of_nodes[0]]
            ]  # starting with the first set that has the first node as the only element
        node_sets_for_trees.append(node_sets)

        num_nodes = len(list_of_nodes)
        # this needs to be done as the ground node is not included in self.nodes
        if circuit.is_grounded:
            num_nodes += 1

        # finding all the sets of nodes and filling node_sets
        node_set_index = 0
        tree_index = 0
        while (
            len(flatten_list_recursive(node_sets_for_trees))
            < num_nodes  # checking to see if all the nodes are present in node_sets
        ):
            node_set = []

            for node in node_sets_for_trees[tree_index][node_set_index]:
                node_set += node.connected_nodes("all")

            node_set = [
                x
                for x in unique_elements_in_list(node_set)
                if x
                not in flatten_list_recursive(
                    node_sets_for_trees[tree_index][: node_set_index + 1]
                )
            ]
            if node_set:
                node_set.sort(key=lambda node: node.index)

            # code to handle two different capacitive islands in the circuit.
            if node_set == []:
                node_sets_for_trees.append([])
                for node in list_of_nodes:
                    if node not in flatten_list_recursive(
                        node_sets_for_trees[tree_index]
                    ):
                        tree_index += 1
                        node_sets_for_trees[tree_index].append([node])
                        node_set_index = 0
                        break
                continue

            node_sets_for_trees[tree_index].append(node_set)
            node_set_index += 1
        return node_sets_for_trees


    def _spanning_tree(self, consider_capacitive_loops: bool = False, use_closure_branches: bool = True):
        r"""
        Returns a spanning tree (as a list of branches) for the given instance. Notice that
        if the circuit contains multiple capacitive islands, the returned spanning tree will
        not include the capacitive twig between two capacitive islands.

        This function also returns all the branches that form superconducting loops, and a
        list of lists of nodes (node_sets), which keeps the generation info for nodes, e.g.,
        for the following spanning tree:

                   /---Node(2)
        Node(1)---'
                   '---Node(3)---Node(4)

        has the node_sets returned as [[Node(1)], [Node(2),Node(3)], [Node(4)]]

        Returns
        -------
            A list of spanning trees in the circuit, which does not include capacitor branches,
            a list of branches that forms superconducting loops for each tree, and a list of lists of nodes
            (node_sets) for each tree (which keeps the generation info for nodes of branches on the path)
            and list of closure branches for each tree.
        """

        # Make a copy of self; do not need symbolic expressions etc., so do a minimal
        # initialization only
        circ_copy = copy.deepcopy(self)

        # adding an attribute for node list without ground
        # circ_copy.nodes = circ_copy.nodes.copy()
        if circ_copy.is_grounded:
            circ_copy.nodes.remove(circ_copy.ground_node)

        # **************** removing all the capacitive branches and updating the nodes *
        # identifying capacitive branches
        branches_to_be_removed = []
        if not consider_capacitive_loops:
            branches_to_be_removed = [
                branch for branch in list(circ_copy.branches) if branch.type == "C"
            ]
        for c_branch in branches_to_be_removed:
            for (
                node
            ) in (
                c_branch.nodes
            ):  # updating the branches attribute for each node that this branch
                # connects
                node.branches = [b for b in node.branches if b is not c_branch]
            circ_copy.branches.remove(c_branch)  # removing the branch

        num_float_nodes = 1
        while num_float_nodes > 0:  # breaks when no floating nodes are detected
            num_float_nodes = 0  # setting
            for node in circ_copy.nodes:
                if len(node.branches) == 0:
                    circ_copy.nodes.remove(node)
                    num_float_nodes += 1
                    continue
                if len(node.branches) == 1:
                    branches_connected_to_node = node.branches[0]
                    circ_copy.branches.remove(branches_connected_to_node)
                    for new_node in branches_connected_to_node.nodes:
                        if new_node != node:
                            new_node.branches = [
                                i
                                for i in new_node.branches
                                if i is not branches_connected_to_node
                            ]
                            num_float_nodes += 1
                            continue
                        else:
                            circ_copy.nodes.remove(node)

        if circ_copy.nodes == []:
            return {
                "list_of_trees": [],
                "loop_branches_for_trees": [],
                "node_sets_for_trees": [],
                "closure_branches_for_trees": [],
            }
        if circ_copy.is_grounded:
            circ_copy.nodes = [circ_copy.ground_node] + circ_copy.nodes
        # *****************************************************************************

        # **************** Constructing the node_sets ***************
        node_sets_for_trees = self._node_sets(circ_copy)
        # ***************************

        # **************** constructing the spanning tree ##########
        def connecting_branches(n1: Node, n2: Node):
            return [branch for branch in n1.branches if branch in n2.branches]

        def is_same_branch(branch_1: Branch, branch_2: Branch):
            return branch_1.index == branch_2.index

        def fetch_same_branch_from_circ(branch: Branch, circ: SymbolicCircuit):
            for b in circ.branches:
                if is_same_branch(b, branch):
                    return b

        def fetch_same_node_from_circ(node: Node, circ: SymbolicCircuit):
            for n in circ.nodes:
                if n.index == node.index:
                    return n

        list_of_trees = []
        for node_sets in node_sets_for_trees:
            tree = []  # tree having branches of the instance that is copied

            # find the branch connecting this node to another node in a previous node set.
            for index, node_set in enumerate(node_sets):
                if index == 0:
                    continue
                for node in node_set:
                    for prev_node in node_sets[index - 1]:
                        if len(connecting_branches(node, prev_node)) != 0:
                            # excluding pure PSJ from the tree. TODO
                            relevant_branches = connecting_branches(node, prev_node)
                            PSJ_branches = [branch for branch in relevant_branches if "PSJ" in branch.type]
                            tree.append(
                                PSJ_branches[0] if len(PSJ_branches) > 0 else relevant_branches[0]
                            )
                            break
            list_of_trees.append(tree)

        # as the capacitors are removed to form the spanning tree, and as a result
        # floating branches as well, the set of all branches which form the
        # superconducting loops would be in circ_copy.
        closure_branches_for_trees = [[] for tree in list_of_trees]
        loop_branches_for_trees = []
        for tree_idx, tree in enumerate(list_of_trees):
            loop_branches = tree.copy()
            nodes_in_tree = flatten_list_recursive(node_sets_for_trees[tree_idx])
            for branch in [
                branch for branch in circ_copy.branches if branch not in tree
            ]:
                if len([node for node in branch.nodes if node in nodes_in_tree]) == 2:
                    loop_branches.append(branch)
                    closure_branches_for_trees[tree_idx].append(branch)
            loop_branches_for_trees.append(loop_branches)

        # get branches from the original circuit
        for tree_idx, tree in enumerate(list_of_trees):
            list_of_trees[tree_idx] = [
                fetch_same_branch_from_circ(branch, self) for branch in tree
            ]
            loop_branches_for_trees[tree_idx] = [
                fetch_same_branch_from_circ(branch, self)
                for branch in loop_branches_for_trees[tree_idx]
            ]
            closure_branches_for_trees[tree_idx] = [
                fetch_same_branch_from_circ(branch, self)
                for branch in closure_branches_for_trees[tree_idx]
            ]
            node_sets_for_trees[tree_idx] = [
                [fetch_same_node_from_circ(node, self) for node in node_set]
                for node_set in node_sets_for_trees[tree_idx]
            ]

        # if the closure branches are manually set, then the spanning tree would be all
        # the superconducting loop branches except the closure branches
        if (self.closure_branches != [] and use_closure_branches) and np.all(
            [isinstance(elem, Branch) for elem in self.closure_branches]
        ):
            closure_branches_for_trees = [
                [] for loop_branches in loop_branches_for_trees
            ]
            list_of_trees = []
            for tree_idx, loop_branches in enumerate(loop_branches_for_trees):
                list_of_trees.append(
                    [
                        branch
                        for branch in loop_branches
                        if branch not in self.closure_branches
                    ]
                )
                closure_branches_for_trees[tree_idx] = [
                    branch
                    for branch in loop_branches
                    if branch in self.closure_branches
                ]

        return {
            "list_of_trees": list_of_trees,
            "loop_branches_for_trees": loop_branches_for_trees,
            "node_sets_for_trees": node_sets_for_trees,
            "closure_branches_for_trees": closure_branches_for_trees,
        }

    def _closure_branches(self, spanning_tree_dict=None):
        r"""Returns and stores the closure branches in the circuit."""
        return flatten_list_recursive(
            (spanning_tree_dict or self.spanning_tree_dict)[
                "closure_branches_for_trees"
            ]
        )

    def _time_dependent_flux_distribution(self):
        closure_branches = self._closure_branches()
        # constructing the constraint matrix
        R = np.zeros([len(self.branches), len(closure_branches)])
        # constructing branch capacitance matrix
        C_diag = np.identity(len(self.branches)) * 0
        # constructing the matrix which transforms node to branch variables
        W = np.zeros([len(self.branches), len(self.nodes) - self.is_grounded])

        for closure_brnch_idx, closure_branch in enumerate(closure_branches):
            loop_branches = self._find_loop(closure_branch)
            # setting the loop direction from the direction of the closure branch
            R_prev_brnch = 1
            for b_idx, branch in enumerate(loop_branches):
                R_elem = 1
                if b_idx == 0:
                    start_node = list(branch.common_node(loop_branches[1]))[0]
                    start_node_idx = branch.nodes.index(start_node)
                    if start_node_idx == 0:
                        R_elem *= -1
                if b_idx > 0:
                    start_node_idx = 1 if R_prev_brnch > 0 else 0
                    start_node = loop_branches[b_idx - 1].nodes[start_node_idx]
                    R_elem = R_prev_brnch
                    if branch.node_ids()[start_node_idx] == start_node.index:
                        R_elem *= -1
                R_prev_brnch = R_elem
                R[self.branches.index(branch), closure_brnch_idx] = R_elem
            if R[self.branches.index(closure_branch), closure_brnch_idx] < 0:
                R[:, closure_brnch_idx] = R[:, closure_brnch_idx] * -1

        for idx, branch in enumerate(self.branches):
            if branch.type == "C" or "JJ" in branch.type:
                EC = (
                    branch.parameters["EC"]
                    if branch.type == "C"
                    else branch.parameters["ECJ"]
                )
                if isinstance(EC, sympy.Expr):
                    EC = self.symbolic_params[EC]
                C_diag[idx, idx] = 1 / (EC * 8)
            for node_idx, node in enumerate(branch.nodes):
                if node.is_ground():
                    continue
                n_id = self.nodes.index(node) - self.is_grounded
                W[idx, n_id] = (-1) ** node_idx

        M = np.vstack([(W.T @ C_diag), R.T])

        I = np.vstack(
            [
                np.zeros([len(self.nodes) - self.is_grounded, len(closure_branches)]),
                np.identity(len(closure_branches)),
            ]
        )

        B = (np.linalg.pinv(M)) @ I
        return B.round(10) @ self.external_fluxes

    def _find_path_to_root(
        self, node: Node, spanning_tree_dict=None
    ) -> Tuple[int, List["Node"], List["Branch"]]:
        r"""Returns all the nodes and branches in the spanning tree path between the
        input node and the root of the spanning tree. Also returns the distance
        (generation) between the input node and the root node. The root of the spanning
        tree is node 0 if there is a physical ground node, otherwise it is node 1.

        Notice that the branches that sit on the boundaries of capacitive islands are
        not included in the branch list.

        Parameters
        ----------
        node: Node
            Node variable which is the input

        Returns
        -------
            An integer for the generation number, a list of ancestor nodes, and a list
            of branches on the path
        """
        # extract spanning trees node_sets (to determine the generation of the node)
        tree_info_dict = spanning_tree_dict or self.spanning_tree_dict
        # find out the generation number of the node in the spanning tree
        for tree_idx, tree in enumerate(tree_info_dict["list_of_trees"]):
            node_sets = tree_info_dict["node_sets_for_trees"][tree_idx]
            tree = tree_info_dict["list_of_trees"][tree_idx]
            # generation number begins from 0
            for igen, nodes in enumerate(node_sets):
                nodes_id = [node.index for node in nodes]
                if node.index in nodes_id:
                    generation = igen
                    break
            # find out the path from the node to the root
            current_node = node
            ancestor_nodes_list = []
            branch_path_to_root = []
            root_node = node_sets[0][0]
            if root_node == node:
                return (0, [], [], tree_idx)

            tree_perm_gen = (perm for perm in itertools.permutations(tree))
            while root_node not in ancestor_nodes_list:
                ancestor_nodes_list = []
                branch_path_to_root = []
                current_node = node
                try:
                    tree_perm = next(tree_perm_gen)
                except StopIteration:
                    break
                # finding the parent of the current_node, and the branch that links the
                # parent and current_node
                for branch in tree_perm:
                    common_node_list = [
                        n for n in branch.nodes if n not in [current_node]
                    ]
                    if (
                        len(common_node_list) == 1
                        and common_node_list[0] not in ancestor_nodes_list
                    ):
                        second_node = common_node_list[0]
                        ancestor_nodes_list.append(second_node)
                        branch_path_to_root.append(branch)
                        current_node = second_node
                        if current_node.index == root_node.index:
                            break
            if root_node in ancestor_nodes_list:
                break

        ancestor_nodes_list.reverse()
        branch_path_to_root.reverse()
        return generation, ancestor_nodes_list, branch_path_to_root, tree_idx

    def _find_loop(
        self, closure_branch: Branch, spanning_tree_dict=None
    ) -> List["Branch"]:
        r"""Find out the loop that is closed by the closure branch.

        Parameters
        ----------
        closure_branch: Branch
            The input closure branch

        Returns
        -------
            A list of branches that corresponds to the loop closed by the closure branch
        """
        # find out ancestor nodes, path to root and generation number for each node in the
        # closure branch
        tree_info_dict = spanning_tree_dict or self.spanning_tree_dict
        _, _, path_1, tree_idx_0 = self._find_path_to_root(
            closure_branch.nodes[0], tree_info_dict
        )
        _, _, path_2, tree_idx_1 = self._find_path_to_root(
            closure_branch.nodes[1], tree_info_dict
        )
        # find branches that are not common in the paths, and then add the closure
        # branch to form the loop
        path_1 = unique_elements_in_list(path_1)
        path_2 = unique_elements_in_list(path_2)
        loop = (
            [branch for branch in path_1 if branch not in path_2]
            + [branch for branch in path_2 if branch not in path_1]
            + [closure_branch]
        )
        return self._order_branches_in_loop(loop)

    def _order_branches_in_loop(self, loop_branches):
        branches_in_order = [loop_branches[0]]
        branch_node_ids = [branch.node_ids() for branch in loop_branches]
        prev_node_id = branch_node_ids[0][0]
        while len(branches_in_order) < len(loop_branches):
            for branch in [
                brnch for brnch in loop_branches if brnch not in branches_in_order
            ]:
                if prev_node_id in branch.node_ids():
                    branches_in_order.append(branch)
                    break
            prev_node_id = [idx for idx in branch.node_ids() if idx != prev_node_id][0]
        return branches_in_order

    def _set_external_fluxes(
        self,
        closure_branches: Optional[List[Union[Branch, Dict[Branch, float]]]] = None,
    ):
        # setting the class properties

        if self.is_purely_harmonic and not self.use_dynamic_flux_grouping:
            self.external_fluxes = []
            self.closure_branches = []
            return 0

        if closure_branches:
            closure_branch_list = [
                branch if isinstance(branch, Branch) else list(branch.keys())
                for branch in closure_branches
            ]
            closure_branch_list = flatten_list_recursive(closure_branch_list)
            for branch in closure_branch_list:
                if branch.type == "C" and not self.use_dynamic_flux_grouping:
                    raise ValueError(
                        "The closure branch cannot be a capacitive branch, when dynamic flux grouping is not used."
                    )

        closure_branches = closure_branches or self._closure_branches()

        if len(closure_branches) > 0:
            self.closure_branches = closure_branches
            self.external_fluxes = [
                symbols("Φ" + str(i + 1)) for i in range(len(closure_branches))
            ]

    def _branch_sym_expr(
        self,
        branch: Branch,
        return_charge: bool = False,
        substitute_params: bool = True,
    ):
        """Returns the voltage across the branch in terms of the charge operators.

        Args:
            branch (Branch): A branch of the instance
        """
        transformation_matrix = self.transformation_matrix

        if return_charge:
            frozen_indices = [
                i - 1
                for i in self.var_categories["frozen"] + self.var_categories["sigma"]
            ]
            # generating the C_mat_θ by inverting the capacitance matrix
            EC_mat_θ = self._capacitance_matrix(invert_matrix=True)
            p_θ_vars = [
                (
                    symbols(f"Q{i}")
                    if i not in self.var_categories["free"]
                    else symbols(f"Qf{i}")
                )
                for i in np.sort(
                    self.var_categories["periodic"]
                    + self.var_categories["extended"]
                    + self.var_categories["free"]
                )
                # replacing the free charge with 0, as it would not affect the circuit
                # Lagrangian.
            ]
            node_id1, node_id2 = [
                node.index - (1 if not self.is_grounded else 0) for node in branch.nodes
            ]
            voltages = list(EC_mat_θ * sympy.Matrix(p_θ_vars))

            # insert the voltages for frozen modes
            for index in self.var_categories["sigma"]:
                voltages.insert(index, 0)
            # substitute expressions for frozen variables
            for index in self.var_categories["frozen"]:
                frozen_var_expr = self.frozen_var_exprs[index]
                frozen_var_expr = frozen_var_expr.subs(
                    [
                        (var_sym, f"Q{get_trailing_number(var_sym.name)}")
                        for var_sym in frozen_var_expr.free_symbols
                    ]
                )
                voltages.insert(index, round_symbolic_expr(frozen_var_expr, 10))

            node_voltages = list(transformation_matrix * sympy.Matrix(voltages))

            if self.is_grounded:
                node_voltages = [0] + node_voltages

            branch_voltage_expr = node_voltages[node_id1] - node_voltages[node_id2]
            # adding the offset charge variables
            for var_index in self.var_categories["periodic"]:
                branch_voltage_expr = branch_voltage_expr.subs(
                    symbols(f"Q{var_index}"),
                    symbols(f"n{var_index}") + symbols(f"ng{var_index}"),
                )
            return branch_voltage_expr * (
                1 / (8 * branch.parameters["EC"])
                if branch.type == "C"
                else 1 / (8 * branch.parameters["ECJ"])
            )

        node_id1, node_id2 = [node.index for node in branch.nodes]
        expr_node_vars = symbols(f"φ{node_id2}") - symbols(f"φ{node_id1}")
        expr_node_vars = expr_node_vars.subs(
            "φ0", 0
        )  # substituting node flux of ground to zero
        num_vars = len(self.nodes) - self.is_grounded
        new_vars = [
            (
                symbols(f"θ{index}")
                if index not in self.var_categories["frozen"]
                else self.frozen_var_exprs[index]
            )
            for index in range(1, 1 + num_vars)
        ]
        # free variables do not show up in the branch flux expression for inductors, assuming capacitances do not depend on the flux, but charge expression
        old_vars = [symbols(f"φ{index}") for index in range(1, 1 + num_vars)]
        transformed_expr = transformation_matrix.dot(new_vars)
        # add external flux
        phi_ext = self.branch_flux_allocations[branch.index]
        for idx, var in enumerate(old_vars):
            expr_node_vars = expr_node_vars.subs(var, transformed_expr[idx])
        return round_symbolic_expr(expr_node_vars + phi_ext, 12)

    def generate_symbolic_lagrangian(
        self, substitute_params: bool = False
    ) -> Tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
        r"""
        Returns four symbolic expressions: lagrangian_θ, potential_θ, lagrangian_φ,
        potential_φ, where θ represents the set of new variables and φ represents
        the set of node variables
        """
        transformation_matrix = self.transformation_matrix

        # defining the φ variables
        φ_dot_vars = [
            symbols(f"vφ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]

        # defining the θ variables
        θ_vars = [
            symbols(f"θ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]
        # defining the θ dot variables
        θ_dot_vars = [
            symbols(f"vθ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]
        # writing φ in terms of θ variables
        φ_vars_θ = transformation_matrix.dot(θ_vars)
        # writing φ dot vars in terms of θ variables
        φ_dot_vars_θ = transformation_matrix.dot(θ_dot_vars)

        # C_terms = self._C_terms()
        C_mat = self._capacitance_matrix(in_node_vars=True)
        if not self.is_any_branch_parameter_symbolic():
            # in terms of node variables
            C_terms_φ = C_mat.dot(φ_dot_vars).dot(φ_dot_vars) * 0.5
            # in terms of new variables
            C_terms_θ = C_mat.dot(φ_dot_vars_θ).dot(φ_dot_vars_θ) * 0.5
        else:
            C_terms_φ = (sympy.Matrix(φ_dot_vars).T * C_mat * sympy.Matrix(φ_dot_vars))[
                0
            ] * 0.5  # in terms of node variables
            C_terms_θ = (
                sympy.Matrix(φ_dot_vars_θ).T * C_mat * sympy.Matrix(φ_dot_vars_θ)
            )[
                0
            ] * 0.5  # in terms of new variables

        inductor_terms_φ = self._inductor_terms(substitute_params=substitute_params)

        JJ_terms_φ = self._josephson_junction_terms() + self._JJs_terms()

        PSJ_terms_Q = self._phase_slip_junction_terms()

        lagrangian_φ = C_terms_φ + PSJ_terms_Q - inductor_terms_φ - JJ_terms_φ

        potential_φ = inductor_terms_φ + JJ_terms_φ + PSJ_terms_Q
        potential_θ = (
            potential_φ.copy() if potential_φ != 0 else symbols("x") * 0
        )  # copying the potential in terms of the old variables to make substitutions

        for index in range(
            len(self.nodes) - self.is_grounded
        ):  # converting potential to new variables
            potential_θ = potential_θ.subs(symbols(f"φ{index + 1}"), φ_vars_θ[index])

        # eliminating the frozen variables
        if len(self.var_categories["frozen"]) > 0:
            frozen_eom_list = []
            for frozen_var_index in self.var_categories["frozen"]:
                frozen_eom_list.append(
                    potential_θ.diff(symbols(f"θ{frozen_var_index}"))
                )

            frozen_exprs = sympy.solve(
                frozen_eom_list,
                [
                    symbols(f"θ{frozen_var_index}")
                    for frozen_var_index in self.var_categories["frozen"]
                ],
            )

            for frozen_var, frozen_expr in frozen_exprs.items():
                potential_θ = potential_θ.replace(frozen_var, frozen_expr).expand()
            self.frozen_var_exprs = {
                get_trailing_number(frozen_var.name): frozen_expr
                for frozen_var, frozen_expr in frozen_exprs.items()
            }

        lagrangian_θ = C_terms_θ - potential_θ

        return lagrangian_θ, potential_θ, lagrangian_φ, potential_φ

    def generate_symbolic_hamiltonian(
        self, substitute_params=False, reevaluate_lagrangian: bool = False
    ) -> sympy.Expr:
        r"""Returns the Hamiltonian of the circuit in terms of the new variables
        :math:`\theta_i`.

        Parameters
        ----------
        substitute_params:
            When set to True, the symbols defined for branch parameters will be
            substituted with the numerical values in the respective Circuit attributes.
        """
        if reevaluate_lagrangian:
            _, potential_symbolic, _, _ = self.generate_symbolic_lagrangian(
                substitute_params=substitute_params
            )
        else:
            potential_symbolic = self.potential_symbolic

        transformation_matrix = self.transformation_matrix
        # generating the C_mat_θ by inverting the capacitance matrix
        C_inv_mat_θ = self._capacitance_matrix(invert_matrix=True, remove_non_dynamical=True)

        p_φ_vars = [
            (
                symbols(f"Q{i}")
                if i not in self.var_categories["free"]
                else symbols(f"Qf{i}")
            )
            for i in np.sort(
                self.var_categories["periodic"]
                + self.var_categories["extended"]
                + self.var_categories["free"]
            )
            # replacing the free charge with 0, as it would not affect the circuit
            # Lagrangian.
        ]
        ### NOTE: sorting the variable indices in the above step is important as the transformation
        ### matrix already takes care of defining the appropriate momenta in the new variables. So, the above variables should be
        ### in the node variable order.
        # generating the kinetic energy terms for the Hamiltonian
        if not self.is_any_branch_parameter_symbolic():
            C_terms_new = (
                C_inv_mat_θ.dot(p_φ_vars).dot(p_φ_vars) * 0.5
            )  # in terms of new variables
        else:
            C_terms_new = (sympy.Matrix(p_φ_vars).T * C_inv_mat_θ * sympy.Matrix(p_φ_vars))[
                0
            ] * 0.5  # in terms of new variables

        hamiltonian_symbolic = C_terms_new + potential_symbolic

        # adding the offset charge variables
        for var_index in self.var_categories["periodic"]:
            hamiltonian_symbolic = hamiltonian_symbolic.subs(
                symbols(f"Q{var_index}"),
                symbols(f"n{var_index}") + symbols(f"ng{var_index}"),
            )

        ## deal with branch charge variables
        transformation_matrix = self.transformation_matrix
        # defining the θ dot variables
        θ_dot_vars = [
            symbols(f"vθ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]
        # writing φ dot vars in terms of θ variables
        φ_dot_vars_θ = transformation_matrix.dot(θ_dot_vars)
        
        lagrangian_terms_PSJ = sum(
            [
                sm.symbols(f"Qb{branch.index}")
                * sum(
                    [
                        φ_dot_vars_θ[node.index - 1] * (-1) ** idx
                        for idx, node in enumerate([n for n in branch.nodes if n.index])
                    ]
                )
                for branch in self.branches
                if "PSJ" in branch.type
            ]
        )
        discrete_charge_symbols = [
            sm.symbols(f"Q{idx}") for idx in self.var_categories["discrete"]
        ]
        discrete_charge_exprs = [
            lagrangian_terms_PSJ.diff(f"vθ{var_idx}") - discrete_charge_symbols[idx]
            for idx, var_idx in enumerate(self.var_categories["discrete"])
        ]
        # solve for branch charges from charge expressions
        branch_charge_in_discrete_charge = sm.solve(
            discrete_charge_exprs,
            [
                sm.symbols(f"Qb{branch.index}")
                for branch in self.branches
                if "PSJ" in branch.type
            ],
        )
        if len(branch_charge_in_discrete_charge) == 0:
            branch_charge_in_discrete_charge = {}
        hamiltonian_symbolic = hamiltonian_symbolic.subs(
            list(branch_charge_in_discrete_charge.items())
        ).expand()

        return round_symbolic_expr(hamiltonian_symbolic.expand(), 12)
