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
from sympy import symbols
from scqubits.core.circuit_utils import (
    round_symbolic_expr,
    _capactiance_variable_for_branch,
    _junction_order,
)

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings

from scqubits.utils.misc import flatten_list, is_string_float, unique_elements_in_list
from scqubits.core.circuit_input import (
    remove_comments,
    remove_branchline,
    strip_empty_lines,
    parse_code_line,
    process_param,
)


class Node:
    """
    Class representing a circuit node, and handled by `Circuit`. The attribute
    `<Node>.branches` is a list of `Branch` objects containing all branches connected to
    the node.

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
        """
        Returns a list of all nodes directly connected by branches to the current
        node, either considering all branches or a specified `branch_type`:
        "C", "L", "JJ", "all" for capacitive, inductive, Josephson junction,
        or all types of branches.
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
        """
        Returns a bool if the node is a ground node. It is ground if the id is set to 0.
        """
        return True if self.index == 0 else False

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


def make_branch(
    nodes_list: List[Node],
    branch_type: str,
    node_idx1: int,
    node_idx2: int,
    params,
    aux_params,
    _branch_count: int,
):
    params_dict = {}
    params = [process_param(param) for param in params]

    if "JJ" in branch_type:
        for idx, param in enumerate(params[:-1]):
            params_dict[sm.symbols(f"EJ{idx + 1}" if idx > 0 else "EJ")] = (
                param[0] if param[0] is not None else param[1]
            )

        params_dict[sm.symbols("EC")] = (
            params[-1][0] if params[-1][0] is not None else params[-1][1]
        )
    if branch_type == "C":
        params_dict[sm.symbols("EC")] = (
            params[-1][0] if params[-1][0] is not None else params[-1][1]
        )
    elif branch_type == "L":
        params_dict[sm.symbols("EL")] = (
            params[-1][0] if params[-1][0] is not None else params[-1][1]
        )

    # return node_idx1, node_idx2, branch_type, list(params_dict.keys()), str(_branch_count), process_param(aux_params)
    is_grounded = True if any([node.is_ground() for node in nodes_list]) else False
    node_1 = nodes_list[node_idx1 if is_grounded else node_idx1 - 1]
    node_2 = nodes_list[node_idx2 if is_grounded else node_idx2 - 1]
    sym_params_dict = {
        param[0]: param[1] for param in params if param[0] is not None
    }  # dictionary of symbolic params and the default values
    return (
        Branch(
            node_1,
            node_2,
            branch_type,
            list(params_dict.values()),
            str(_branch_count),
            process_param(aux_params),
        ),
        sym_params_dict,
    )


class Branch:
    """
    Class describing a circuit branch, used in the Circuit class.

    Parameters
    ----------
    n_i, n_f:
        initial and final nodes connected by this branch;
    branch_type:
        is the type of this Branch, example "C","JJ" or "L"
    parameters:
        list of parameters for the branch, namely for
        capacitance: {"EC":  <value>};
        for inductance: {"EL": <value>};
        for Josephson Junction: {"EJ": <value>, "ECJ": <value>}
    aux_params:
        Dictionary of auxiliary parameters which map a symbol from the input file a numeric parameter.

    Examples
    --------
    `Branch("C", Node(1, 0), Node(2, 0))`
    is a capacitive branch connecting the nodes with indices 0 and 1.
    """

    def __init__(
        self,
        n_i: Node,
        n_f: Node,
        branch_type: str,
        parameters: Optional[List[Union[float, Symbol, int]]] = None,
        id_str: str = None,
        aux_params: Dict[Symbol, float] = {},
    ):
        self.nodes = (n_i, n_f)
        self.type = branch_type
        self.parameters = parameters
        self.id_str = id_str
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
        return f"Branch({self.type}, {self.nodes[0].index}, {self.nodes[1].index}, id_str: {self.id_str})"

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
                    self.parameters[f"EJ{junc_order}"] = parameters[junc_order]
            self.parameters["ECJ"] = parameters[number_of_junc_params]

    def node_ids(self) -> Tuple[int, int]:
        return self.nodes[0].index, self.nodes[1].index

    def is_connected(self, branch) -> bool:
        """Returns a boolean indicating whether the current branch is
        connected to the given `branch`"""
        distinct_node_count = len(set(self.nodes + branch.nodes))
        if distinct_node_count < 4:
            return True
        return False

    def common_node(self, branch) -> Set[Node]:
        """Returns the common nodes between self and the `branch` given as input"""
        return set(self.nodes) & set(branch.nodes)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class SymbolicCircuit(serializers.Serializable):
    r"""
    Describes a circuit consisting of nodes and branches.

    Examples
    --------
    For a transmon qubit, the input file reads:
        ```
        # file_name: transmon_num.inp
        nodes: 2
        branches:
        C	1,2	1
        JJ	1,2	1	10
        ```

    The `Circuit` object can be initiated using:
        `Circuit.from_input_file("transmon_num.inp")`

    Parameters
    ----------
    nodes_list: List[Nodes]
        List of nodes in the circuit
    branches_list: List[Branch]
        List of branches connecting the above set of nodes.
    basis_completion: str
        choices are: "heuristic" (default) or "canonical"; selects type of basis for
        completing the transformation matrix.
    is_flux_dynamic: bool
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
        branch_var_dict: Dict[Union[Any, Symbol], Union[Any, float]],
        basis_completion: str = "heuristic",
        is_flux_dynamic: bool = True,
        initiate_sym_calc: bool = True,
        input_string: str = "",
    ):
        self.branches = branches_list
        self.nodes = nodes_list
        self.input_string = input_string

        self._sys_type = type(self).__name__  # for object description

        # attributes set by methods
        self.transformation_matrix: Optional[ndarray] = None

        self.var_categories: Optional[List[int]] = None
        self.external_fluxes: List[Symbol] = []
        self.closure_branches: List[Branch] = []

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
        self.is_flux_dynamic = is_flux_dynamic

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
            """
            Projection of u on v

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
        l_mat = (
            trans_mat.T @ self._inductance_matrix(substitute_params=True) @ trans_mat
        )
        if not self.is_grounded:
            c_mat = c_mat[:-1, :-1]
            l_mat = l_mat[:-1, :-1]
        normal_mode_freqs, normal_mode_vecs = sp.linalg.eig(l_mat, c_mat)
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
        transformation_matrix: ndarray = None,
        closure_branches: List[Branch] = None,
    ):
        """
        Method to initialize the CustomQCircuit instance and initialize all the
        attributes needed before it can be passed on to AnalyzeQCircuit.

        Parameters
        ----------
        transformation_matrix:
            array used to set a transformation matrix other than the one generated by
            the method `variable_transformation_matrix`.
        closure_branches:
            List of branches for which the external flux variables will be defined.
        """
        # if the circuit is purely harmonic, then store the eigenfrequencies
        branch_type_list = [branch.type for branch in self.branches]
        self.is_purely_harmonic = "JJ" not in "".join(branch_type_list)

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

        # find the closure branches in the circuit
        self.closure_branches = closure_branches or self._closure_branches()
        # setting external flux and offset charge variables
        self._set_external_fluxes(closure_branches=closure_branches)
        self._set_offset_charges()
        # setting the branch parameter variables

        # calculating the Hamiltonian directly when the number of nodes is less than 3
        substitute_params = False
        if (
            len(self.nodes) >= settings.SYM_INVERSION_MAX_NODES
            or len(self.var_categories["frozen"]) > 0
        ):  # only calculate the symbolic hamiltonian when the number of nodes is less
            # than 3. Else, the calculation will be skipped to the end when numerical
            # Hamiltonian of the circuit is requested.
            substitute_params = True

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
        """
        Method replaces the energies in the Lagrangian with capacitances which are
        arbitrarily generated to make sure that the Lagrangian looks dimensionally
        correct.
        """
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
        """
        Determines whether two sets of branches are disconnected.

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
        for branch_list_input in branches_list:
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
        is_flux_dynamic: bool = True,
        initiate_sym_calc: bool = True,
    ):
        """
        Constructs the instance of Circuit from an input string. Here is an example of
        an input string that is used to initiate an object of the
        class `SymbolicCircuit`:

            ```
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
            ```

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
        is_flux_dynamic: bool
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
        # parse branches
        branches_list = []
        branch_var_dict = {}
        for parsed_branch in parsed_branches:
            branch, sym_params = make_branch(nodes_list, *parsed_branch)
            for sym_param in sym_params:
                if sym_param in branch_var_dict and sym_params[sym_param] is not None:
                    raise Exception(
                        f"Symbol {sym_param} has already been assigned a value."
                    )
                if sym_params[sym_param] is not None:
                    branch_var_dict[sym_param] = sym_params[sym_param]
            branches_list.append(branch)

        circuit = cls(
            nodes_list,
            branches_list,
            is_flux_dynamic=is_flux_dynamic,
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
        """
        Returns the vectors which span a subspace where there is no generalized flux
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
            list(set(sum([branch.nodes for branch in branch_set], ())))
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

        unique_branch_set_markers = list(set(node_branch_set_indices))
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
        """
        Method to check if the vector mode is a part of the subspace provided as a set
        of vectors

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

    def check_transformation_matrix(
        self, transformation_matrix: ndarray, enable_warnings: bool = True
    ):
        """
        Method to identify the different modes in the transformation matrix provided by
        the user.

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

        # find all the different types of modes present in the circuit.

        # *************************** Finding the Periodic Modes **********************
        selected_branches = [branch for branch in self.branches if branch.type == "L"]
        periodic_modes = self._independent_modes(selected_branches)

        # *************************** Finding the frozen modes **********************
        selected_branches = [branch for branch in self.branches if branch.type != "L"]
        frozen_modes = self._independent_modes(selected_branches, single_nodes=True)

        # *************************** Finding the Cyclic Modes ****************
        selected_branches = [branch for branch in self.branches if branch.type != "C"]
        free_modes = self._independent_modes(selected_branches)

        # ***************************# Finding the LC Modes ****************
        selected_branches = [branch for branch in self.branches if "JJ" in branch.type]
        LC_modes = self._independent_modes(selected_branches, single_nodes=False)

        # ******************* including the Σ mode ****************
        Σ = [1] * (len(self.nodes) - self.is_grounded)
        if not self.is_grounded:  # only append if the circuit is not grounded
            mat = np.array(frozen_modes + [Σ])
            # check to see if the vectors are still independent
            if np.linalg.matrix_rank(mat) < len(frozen_modes) + 1:
                frozen_modes = frozen_modes[1:] + [Σ]
            else:
                frozen_modes.append(Σ)

        # *********** Adding periodic, free and extended modes to frozen ************
        modes = []  # starting with the frozen modes

        for m in (
            frozen_modes + free_modes + periodic_modes + LC_modes  # + extended_modes
        ):  # This order is important
            if not self._mode_in_subspace(m, modes):
                modes.append(m)

        for m in LC_modes:  # adding the LC modes to the basis
            if not self._mode_in_subspace(m, modes):
                modes.append(m)

        var_categories_circuit: Dict[str, list] = {
            "periodic": [],
            "extended": [],
            "free": [],
            "frozen": [],
        }

        for x, mode in enumerate(modes):
            # calculate the number of periodic modes
            if self._mode_in_subspace(Σ, [mode]) and not self.is_grounded:
                continue

            if self._mode_in_subspace(mode, frozen_modes):
                var_categories_circuit["frozen"].append(x + 1)
                continue

            if self._mode_in_subspace(mode, free_modes):
                var_categories_circuit["free"].append(x + 1)
                continue

            if self._mode_in_subspace(mode, periodic_modes):
                var_categories_circuit["periodic"].append(x + 1)
                continue

            # Any mode which survived the above conditionals is an extended mode
            var_categories_circuit["extended"].append(x + 1)

        # Classifying the modes given in the transformation by the user

        user_given_modes = transformation_matrix.transpose()

        var_categories_user: Dict[str, list] = {
            "periodic": [],
            "extended": [],
            "free": [],
            "frozen": [],
            "sigma": [],
        }
        sigma_mode_found = False
        for x, mode in enumerate(user_given_modes):
            # calculate the number of periodic modes
            if self._mode_in_subspace(Σ, [mode]) and not self.is_grounded:
                sigma_mode_found = True
                var_categories_user["sigma"].append(x + 1)
                continue

            if self._mode_in_subspace(mode, frozen_modes):
                var_categories_user["frozen"].append(x + 1)
                continue

            if self._mode_in_subspace(mode, free_modes):
                var_categories_user["free"].append(x + 1)
                continue

            if self._mode_in_subspace(mode, periodic_modes):
                var_categories_user["periodic"].append(x + 1)
                continue

            # Any mode which survived the above conditionals is an extended mode
            var_categories_user["extended"].append(x + 1)

        # comparing the modes in the user defined and the code generated transformation

        mode_types = ["periodic", "extended", "free", "frozen"]

        for mode_type in mode_types:
            num_extra_modes = len(var_categories_circuit[mode_type]) - len(
                var_categories_user[mode_type]
            )
            if num_extra_modes > 0 and enable_warnings:
                warnings.warn(
                    "Number of extra "
                    + mode_type
                    + " modes found: "
                    + str(num_extra_modes)
                    + "\n"
                )
        if not self.is_grounded and not sigma_mode_found:
            raise Exception(
                "This circuit is not grounded, and so has a sigma mode. This transformation does not have a sigma mode."
            )

        return var_categories_user

    def variable_transformation_matrix(self) -> Tuple[ndarray, Dict[str, List[int]]]:
        """
        Evaluates the boundary conditions and constructs the variable transformation
        matrix, which is returned along with the dictionary `var_categories` which
        classifies the types of variables present in the circuit.

        Returns
        -------
            tuple of transformation matrix for the node variables and `var_categories`
            dict which classifies the variable types for each variable index
        """

        # ****************  Finding the Periodic Modes ****************
        selected_branches = [branch for branch in self.branches if branch.type == "L"]
        periodic_modes = self._independent_modes(selected_branches)

        # ****************  Finding the frozen modes ****************
        selected_branches = [branch for branch in self.branches if branch.type != "L"]
        frozen_modes = self._independent_modes(selected_branches, single_nodes=True)

        # **************** Finding the Cyclic Modes ****************
        selected_branches = [branch for branch in self.branches if branch.type != "C"]
        free_modes = self._independent_modes(selected_branches)

        # ****************  including the Σ mode ****************
        Σ = [1] * (len(self.nodes) - self.is_grounded)
        if not self.is_grounded:  # only append if the circuit is not grounded
            mat = np.array(frozen_modes + [Σ])
            # check to see if the vectors are still independent
            if np.linalg.matrix_rank(mat) < len(frozen_modes) + 1:
                frozen_modes = frozen_modes[1:] + [Σ]
            else:
                frozen_modes.append(Σ)

        # **************** Finding the LC Modes ****************
        selected_branches = [branch for branch in self.branches if "JJ" in branch.type]
        LC_modes = self._independent_modes(
            selected_branches, single_nodes=False, basisvec_entries=[-1, 1]
        )

        # **************** Adding frozen, free, periodic , LC and extended modes ****
        modes = []  # starting with an empty list

        for m in (
            frozen_modes + free_modes + periodic_modes + LC_modes  # + extended_modes
        ):  # This order is important
            mat = np.array(modes + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                modes.append(m)

        # ********** Completing the Basis ****************
        # step 4: construct the new set of basis vectors

        # constructing a standard basis
        node_count = len(self.nodes) - self.is_grounded
        standard_basis = [np.ones(node_count)]

        vector_ref = np.zeros(node_count)
        if node_count > 2:
            vector_ref[: node_count - 2] = 1
        else:
            vector_ref[: node_count - 1] = 1

        vector_set = list((itertools.permutations(vector_ref, node_count)))
        item = 0
        while np.linalg.matrix_rank(np.array(standard_basis)) < node_count:
            a = vector_set[item]
            item += 1
            mat = np.array(standard_basis + [a])
            if np.linalg.matrix_rank(mat) == len(mat):
                standard_basis = standard_basis + [list(a)]

        standard_basis = np.array(standard_basis)

        if self.basis_completion == "canonical":
            standard_basis = np.identity(len(self.nodes) - self.is_grounded)

        new_basis = modes.copy()

        for m in standard_basis:  # completing the basis
            mat = np.array([i for i in new_basis] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                new_basis.append(m)

        new_basis = np.array(new_basis)

        # sorting the basis so that the free, periodic and frozen variables occur at
        # the beginning.
        if not self.is_grounded:
            pos_Σ = [i for i in range(len(new_basis)) if new_basis[i].tolist() == Σ]
        else:
            pos_Σ = []

        pos_free = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if new_basis[i].tolist() in free_modes
        ]
        pos_periodic = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_free
            if new_basis[i].tolist() in periodic_modes
        ]
        pos_frozen = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_free
            if i not in pos_periodic
            if new_basis[i].tolist() in frozen_modes
        ]
        pos_rest = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_free
            if i not in pos_periodic
            if i not in pos_frozen
        ]
        pos_list = pos_periodic + pos_rest + pos_free + pos_frozen + pos_Σ
        # transforming the new_basis matrix
        new_basis = new_basis[pos_list].T

        # saving the variable identification to a dict
        var_categories = {
            "periodic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_periodic
            ],
            "extended": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_rest
            ],
            "free": [i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_free],
            "frozen": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_frozen
            ],
            "sigma": [i + 1 for i in range(len(pos_list)) if pos_list[i] == pos_Σ[0]]
            if not self.is_grounded
            else [],
        }

        return np.array(new_basis), var_categories

    def update_param_init_val(self, param_name, value):
        """
        Updates the param init val for param_name
        """
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

    def _junction_terms(self):
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
            phi_ext = 0
            if jj_branch in self.closure_branches:
                if not self.is_flux_dynamic:
                    index = self.closure_branches.index(jj_branch)
                    phi_ext += self.external_fluxes[index]
            if self.is_flux_dynamic:
                flux_branch_assignment = self._time_dependent_flux_distribution()
                phi_ext += flux_branch_assignment[int(jj_branch.id_str)]

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

    def _JJs_terms(self):
        """To add terms for the sawtooth josephson junction"""
        terms = 0
        # looping over all the junction terms
        junction_branches = [branch for branch in self.branches if "JJs" in branch.type]

        # defining a function for sawtooth
        saw = sympy.Function("saw", real=True)

        for branch_idx, jj_branch in enumerate(junction_branches):
            # adding external flux
            phi_ext = 0
            if jj_branch in self.closure_branches:
                if not self.is_flux_dynamic:
                    index = self.closure_branches.index(jj_branch)
                    phi_ext += self.external_fluxes[index]
            if self.is_flux_dynamic:
                flux_branch_assignment = self._time_dependent_flux_distribution()
                phi_ext += flux_branch_assignment[int(jj_branch.id_str)]

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

    # def _JJ2_terms(self):
    #     terms = 0
    #     # looping over all the JJ2 branches
    #     for jj2_branch in [t for t in self.branches if t.type == "JJ2"]:
    #         # adding external flux
    #         phi_ext = 0
    #         if jj2_branch in self.closure_branches:
    #             if not self.is_flux_dynamic:
    #                 index = self.closure_branches.index(jj2_branch)
    #                 phi_ext += self.external_fluxes[index]
    #         if self.is_flux_dynamic:
    #             flux_branch_assignment = self._time_dependent_flux_distribution()
    #             phi_ext += flux_branch_assignment[int(jj2_branch.id_str)]

    #         # if loop to check for the presence of ground node
    #         if jj2_branch.nodes[1].index == 0:
    #             terms += -jj2_branch.parameters["EJ"] * sympy.cos(
    #                 2 * (-symbols(f"φ" + str(jj2_branch.nodes[0].index)) + phi_ext)
    #             )
    #         elif jj2_branch.nodes[0].index == 0:
    #             terms += -jj2_branch.parameters["EJ"] * sympy.cos(
    #                 2 * (symbols(f"φ{jj2_branch.nodes[1].index}") + phi_ext)
    #             )
    #         else:
    #             terms += -jj2_branch.parameters["EJ"] * sympy.cos(
    #                 2
    #                 * (
    #                     symbols(f"φ{jj2_branch.nodes[1].index}")
    #                     - symbols(f"φ{jj2_branch.nodes[0].index}")
    #                     + phi_ext
    #                 )
    #             )
    #     return terms

    def _inductance_matrix(self, substitute_params: bool = False):
        """
        Generate a inductance matrix for the circuit

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

    def _capacitance_matrix(self, substitute_params: bool = False):
        """
        Generate a capacitance matrix for the circuit

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
        return C_mat

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

    def _inductor_terms(self, substitute_params: bool = False):
        terms = 0
        for l_branch in [branch for branch in self.branches if branch.type == "L"]:
            # adding external flux
            phi_ext = 0
            if l_branch in self.closure_branches:
                if not self.is_flux_dynamic:
                    index = self.closure_branches.index(l_branch)
                    phi_ext += self.external_fluxes[index]
            if self.is_flux_dynamic:
                flux_branch_assignment = self._time_dependent_flux_distribution()
                phi_ext += flux_branch_assignment[int(l_branch.id_str)]

            if l_branch.nodes[0].index == 0:
                terms += (
                    0.5
                    * l_branch.parameters["EL"]
                    * (symbols(f"φ{l_branch.nodes[1].index}") + phi_ext) ** 2
                )
            elif l_branch.nodes[1].index == 0:
                terms += (
                    0.5
                    * l_branch.parameters["EL"]
                    * (-symbols(f"φ{l_branch.nodes[0].index}") + phi_ext) ** 2
                )
            else:
                terms += (
                    0.5
                    * l_branch.parameters["EL"]
                    * (
                        symbols(f"φ{l_branch.nodes[1].index}")
                        - symbols(f"φ{l_branch.nodes[0].index}")
                        + phi_ext
                    )
                    ** 2
                )
        # substitute params if necessary
        if substitute_params and terms != 0:
            for symbol in self.symbolic_params:
                terms = terms.subs(symbol.name, self.symbolic_params[symbol])
        return terms

    def _spanning_tree(self):
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
            A spanning tree as a list of branches, which does not include capacitor branches,
            a list of branches that forms superconducting loops, and a list of lists of nodes
            (node_sets), which keeps the generation info for nodes of branches on the path.
        """

        # Make a copy of self; do not need symbolic expressions etc., so do a minimal
        # initialization only
        circ_copy = copy.deepcopy(self)

        # adding an attribute for node list without ground
        circ_copy._node_list_without_ground = circ_copy.nodes
        if circ_copy.is_grounded:
            circ_copy._node_list_without_ground.remove(circ_copy.ground_node)

        # **************** removing all the capacitive branches and updating the nodes *
        # identifying capacitive branches
        capacitor_branches = [
            branch for branch in list(circ_copy.branches) if branch.type == "C"
        ]
        for c_branch in capacitor_branches:
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
            for node in circ_copy._node_list_without_ground:
                if len(node.branches) == 0:
                    circ_copy._node_list_without_ground.remove(node)
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
                            circ_copy._node_list_without_ground.remove(node)

        if circ_copy._node_list_without_ground == []:
            return [], [], []
        # *****************************************************************************

        # **************** Constructing the node_sets ***************
        if circ_copy.is_grounded:
            node_sets = [[circ_copy.ground_node]]
        else:
            node_sets = [
                [circ_copy._node_list_without_ground[0]]
            ]  # starting with the first set that has the first node as the only element

        num_nodes = len(circ_copy._node_list_without_ground)
        # this needs to be done as the ground node is not included in self.nodes
        if circ_copy.is_grounded:
            num_nodes += 1

        # finding all the sets of nodes and filling node_sets
        node_set_index = 0
        while (
            len(sum(node_sets, []))
            < num_nodes  # checking to see if all the nodes are present in node_sets
        ):
            node_set = []

            # code to handle two different capacitive islands in the circuit.
            if node_sets[node_set_index] == []:
                for node in circ_copy._node_list_without_ground:
                    if node not in flatten_list(node_sets):
                        node_sets[node_set_index].append(node)
                        break

            for node in node_sets[node_set_index]:
                node_set += node.connected_nodes("all")

            node_set = [
                x
                for x in list(set(node_set))
                if x not in flatten_list(node_sets[: node_set_index + 1])
            ]
            if node_set:
                node_set.sort(key=lambda node: node.index)

            node_sets.append(node_set)
            node_set_index += 1
        # ***************************

        # **************** constructing the spanning tree ##########
        tree_copy = []  # tree having branches of the instance that is copied

        def connecting_branches(n1: Node, n2: Node):
            return [branch for branch in n1.branches if branch in n2.branches]

        # find the branch connecting this node to another node in a previous node set.
        for index, node_set in enumerate(node_sets):
            if index == 0:
                continue
            for node in node_set:
                for prev_node in node_sets[index - 1]:
                    if len(connecting_branches(node, prev_node)) != 0:
                        tree_copy.append(connecting_branches(node, prev_node)[0])
                        break

        # ************* selecting the appropriate branches from circ as from circ_copy #
        def is_same_branch(branch_1: Branch, branch_2: Branch):
            return branch_1.id_str == branch_2.id_str

        tree = []  # tree having branches of the current instance
        for c_branch in tree_copy:
            tree += [b for b in self.branches if is_same_branch(b, c_branch)]

        # as the capacitors are removed to form the spanning tree, and as a result
        # floating branches as well, the set of all branches which form the
        # superconducting loops would be in circ_copy.
        superconducting_loop_branches = []
        for branch_copy in circ_copy.branches:
            superconducting_loop_branches += [
                branch
                for branch in self.branches
                if is_same_branch(branch, branch_copy)
            ]
        # if the closure branches are manually set, then the spanning tree would be all
        # the superconducting loop branches except the closure branches
        if self.closure_branches != []:
            tree = [
                branch
                for branch in superconducting_loop_branches
                if branch not in self.closure_branches
            ]

        return tree, superconducting_loop_branches, node_sets

    def _closure_branches(self):
        r"""
        Returns and stores the closure branches in the circuit.
        """
        tree, superconducting_loop_branches, node_sets = self._spanning_tree()
        if tree == []:
            closure_branches = []
        else:
            closure_branches = [
                branch for branch in superconducting_loop_branches if branch not in tree
            ]
        return closure_branches

    def _time_dependent_flux_distribution(self):
        # constructing the constraint matrix
        R = np.zeros([len(self.branches), len(self.closure_branches)])
        # constructing branch capacitance matrix
        C_diag = np.identity(len(self.branches)) * 0
        # constructing the matrix which transforms node to branch variables
        W = np.zeros([len(self.branches), len(self.nodes) - self.is_grounded])

        for closure_brnch_idx, closure_branch in enumerate(self.closure_branches):
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
                np.zeros(
                    [len(self.nodes) - self.is_grounded, len(self.closure_branches)]
                ),
                np.identity(len(self.closure_branches)),
            ]
        )

        B = (np.linalg.pinv(M)) @ I
        return B.round(10) @ self.external_fluxes

    def _find_path_to_root(
        self, node: Node
    ) -> Tuple[int, List["Node"], List["Branch"]]:
        r"""
        Returns all the nodes and branches in the spanning tree path between the
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
        # extract spanning tree node_sets (to determine the generation of the node)
        tree, superconducting_loop_branches, node_sets = self._spanning_tree()
        # find out the generation number of the node in the spanning tree
        # generation number begins from 0
        for igen, nodes in enumerate(node_sets):
            nodes_id = [node.index for node in nodes]
            if node.index in nodes_id:
                generation = igen
                break
        # find out the path from the node to the root
        current_node = node
        ancestor_nodes_list: List[Node] = []
        branch_path_to_root: List[Branch] = []
        root_node = self.nodes[0]
        if root_node == node:
            return (0, [], [])
        while root_node not in ancestor_nodes_list:
            ancestor_nodes_list = []
            branch_path_to_root = []
            current_node = node
            random.shuffle(tree)
            # finding the parent of the current_node, and the branch that links the
            # parent and current_node
            for branch in tree:
                common_node_list = list(set(branch.nodes) - set([current_node]))
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

        ancestor_nodes_list.reverse()
        branch_path_to_root.reverse()
        return generation, ancestor_nodes_list, branch_path_to_root

    def _find_loop(self, closure_branch: Branch) -> List["Branch"]:
        r"""
        Find out the loop that is closed by the closure branch

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
        _, _, path_1 = self._find_path_to_root(closure_branch.nodes[0])
        _, _, path_2 = self._find_path_to_root(closure_branch.nodes[1])
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

    def _set_external_fluxes(self, closure_branches: List[Branch] = None):
        # setting the class properties

        if self.is_purely_harmonic:
            self.external_fluxes = []
            self.closure_branches = []
            return 0

        closure_branches = closure_branches or self._closure_branches()
        closure_branches = [branch for branch in closure_branches if branch.type != "C"]

        if len(closure_branches) > 0:
            self.closure_branches = closure_branches
            self.external_fluxes = [
                symbols("Φ" + str(i + 1)) for i in range(len(closure_branches))
            ]

    def _set_offset_charges(self):
        """
        Create the offset charge variables and store in class attribute offset_charges
        """
        self.offset_charges = []
        for p in self.var_categories["periodic"]:
            self.offset_charges = self.offset_charges + [symbols(f"ng{p}")]

    def _branch_sym_expr(
        self,
        branch: Branch,
        return_charge: bool = False,
        substitute_params: bool = True,
    ):
        """
        Returns the voltage across the branch in terms of the charge operators

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
            if self.is_any_branch_parameter_symbolic() and not substitute_params:
                C_mat_θ = (
                    transformation_matrix.T
                    * self._capacitance_matrix()
                    * transformation_matrix
                )
                relevant_indices = [
                    i for i in range(C_mat_θ.shape[0]) if i not in frozen_indices
                ]
                C_mat_θ = C_mat_θ[relevant_indices, relevant_indices]
                C_mat_θ = C_mat_θ.inv()
            else:
                C_mat_θ = (
                    transformation_matrix.T
                    @ self._capacitance_matrix(substitute_params=substitute_params)
                    @ transformation_matrix
                )
                C_mat_θ = np.delete(C_mat_θ, frozen_indices, 0)
                C_mat_θ = np.delete(C_mat_θ, frozen_indices, 1)
                C_mat_θ = np.linalg.inv(C_mat_θ)
            p_θ_vars = [
                symbols(f"Q{i}")
                for i in self.var_categories["periodic"]
                + self.var_categories["extended"]
                + self.var_categories["free"]
                # replacing the free charge with 0, as it would not affect the circuit
                # Lagrangian.
            ]
            node_id1, node_id2 = [
                node.index - (1 if not self.is_grounded else 0) for node in branch.nodes
            ]
            voltages = C_mat_θ * sympy.Matrix(p_θ_vars)

            if not self.is_grounded:
                voltages = list(voltages) + [0]

            node_voltages = list(
                np.linalg.inv(self.transformation_matrix) * sympy.Matrix(voltages)
            )
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
        new_vars = [symbols(f"θ{index}") for index in range(1, 1 + num_vars)]
        old_vars = [symbols(f"φ{index}") for index in range(1, 1 + num_vars)]
        transformed_expr = transformation_matrix.dot(new_vars)
        # add external flux
        phi_ext = 0
        if branch in self.closure_branches:
            if not self.is_flux_dynamic:
                index = self.closure_branches.index(branch)
                phi_ext += self.external_fluxes[index]
        if self.is_flux_dynamic:
            flux_branch_assignment = self._time_dependent_flux_distribution()
            phi_ext += flux_branch_assignment[int(branch.id_str)]
        for idx, var in enumerate(old_vars):
            expr_node_vars = expr_node_vars.subs(var, transformed_expr[idx])
        return expr_node_vars + phi_ext

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
        C_mat = self._capacitance_matrix()
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

        JJ_terms_φ = self._junction_terms() + self._JJs_terms()

        lagrangian_φ = C_terms_φ - inductor_terms_φ - JJ_terms_φ

        potential_φ = inductor_terms_φ + JJ_terms_φ
        potential_θ = (
            potential_φ.copy() if potential_φ != 0 else symbols("x") * 0
        )  # copying the potential in terms of the old variables to make substitutions

        for index in range(
            len(self.nodes) - self.is_grounded
        ):  # converting potential to new variables
            potential_θ = potential_θ.subs(symbols(f"φ{index + 1}"), φ_vars_θ[index])

        # eliminating the frozen variables
        for frozen_var_index in self.var_categories["frozen"]:
            sub = sympy.solve(
                potential_θ.diff(symbols(f"θ{frozen_var_index}")),
                symbols(f"θ{frozen_var_index}"),
            )
            potential_θ = potential_θ.replace(symbols(f"θ{frozen_var_index}"), sub[0])

        lagrangian_θ = C_terms_θ - potential_θ

        return lagrangian_θ, potential_θ, lagrangian_φ, potential_φ

    def generate_symbolic_hamiltonian(
        self, substitute_params=False, reevaluate_lagrangian: bool = False
    ) -> sympy.Expr:
        r"""
        Returns the Hamiltonian of the circuit in terms of the new variables
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

        frozen_indices = [
            i - 1 for i in self.var_categories["frozen"] + self.var_categories["sigma"]
        ]
        # generating the C_mat_θ by inverting the capacitance matrix
        if self.is_any_branch_parameter_symbolic() and not substitute_params:
            C_mat_θ = (
                transformation_matrix.T
                * self._capacitance_matrix()
                * transformation_matrix
            )
            relevant_indices = [
                i for i in range(C_mat_θ.shape[0]) if i not in frozen_indices
            ]
            C_mat_θ = C_mat_θ[relevant_indices, relevant_indices]
            C_mat_θ = C_mat_θ.inv()
        else:
            C_mat_θ = (
                transformation_matrix.T
                @ self._capacitance_matrix(substitute_params=substitute_params)
                @ transformation_matrix
            )
            C_mat_θ = np.delete(C_mat_θ, frozen_indices, 0)
            C_mat_θ = np.delete(C_mat_θ, frozen_indices, 1)
            C_mat_θ = np.linalg.inv(C_mat_θ)

        p_θ_vars = [
            symbols(f"Q{i}")
            for i in self.var_categories["periodic"] + self.var_categories["extended"]
            # replacing the free charge with 0, as it would not affect the circuit
            # Lagrangian.
        ] + [
            0 for i in self.var_categories["free"]
        ]  # defining the momentum variables

        # generating the kinetic energy terms for the Hamiltonian
        if not self.is_any_branch_parameter_symbolic():
            C_terms_new = (
                C_mat_θ.dot(p_θ_vars).dot(p_θ_vars) * 0.5
            )  # in terms of new variables
        else:
            C_terms_new = (sympy.Matrix(p_θ_vars).T * C_mat_θ * sympy.Matrix(p_θ_vars))[
                0
            ] * 0.5  # in terms of new variables

        hamiltonian_symbolic = C_terms_new + potential_symbolic

        # adding the offset charge variables
        for var_index in self.var_categories["periodic"]:
            hamiltonian_symbolic = hamiltonian_symbolic.subs(
                symbols(f"Q{var_index}"),
                symbols(f"n{var_index}") + symbols(f"ng{var_index}"),
            )

        return round_symbolic_expr(hamiltonian_symbolic.expand(), 14)
