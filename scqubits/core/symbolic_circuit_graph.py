import copy
import itertools
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy import ndarray
import scipy as sp
import sympy
import sympy as sm
from sympy import symbols, Symbol

from scqubits.core.circuit_utils import (
    round_symbolic_expr,
    _capacitance_variable_for_branch,
    _junction_order,
    get_trailing_number,
)

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
from itertools import chain

from scqubits.utils.misc import (
    flatten_list_recursive,
    unique_elements_in_list,
)
from scqubits.core.circuit_input import (
    remove_comments,
    remove_branchline,
    strip_empty_lines,
    parse_code_line,
    process_param,
)

from abc import ABC


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

    def __init__(self, index: int, marker: int = 0):
        self.index: int = index
        self.marker: int = marker
        self._init_params: Dict[str, int] = {"id": self.index, "marker": self.marker}
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
        """Returns a bool if the node is a ground node (Node with index set to 0)."""
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
    n_i:
        initial `Node` of the branch
    n_f:
        final `Node` of the branch
    branch_type:
        is the type of this Branch, example `"C"`, `"JJ"` or `"L"`
    parameters:
        list of parameters for the branch, namely for
        capacitance: `[EC]`;
        for inductance: `[10]`;
        for Josephson Junction: `[EJ, 1]`
    aux_params:
        Dictionary of auxiliary parameters which map a symbol from the input file a numeric parameter.

    Examples
    --------
    `Branch("C", Node(1, 0), Node(2, 0))` is a capacitive branch connecting the nodes with indices 0 and 1.
    """

    def __init__(
        self,
        n_i: Node,
        n_f: Node,
        branch_type: str,
        parameters: List[Union[float, Symbol, int]],
        index: Optional[int] = None,
        aux_params: Dict[Symbol, float] = {},
    ):
        self.nodes = (n_i, n_f)
        self.type = branch_type
        self.index = index
        # store info of current branch inside the provided nodes
        # setting the parameters if it is provided
        self._set_parameters(parameters)
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

    def _set_parameters(self, parameters) -> None:
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
        parameters: List[Union[float, Symbol, int]],
        index: Optional[int] = None,
        aux_params: Dict[Symbol, float] = {},
    ):
        self.branches = (branch1, branch2)
        self.type = coupling_type
        self._set_parameters(parameters)
        self.aux_params = aux_params
        self.index = index

    def _set_parameters(self, parameters) -> None:
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
        params_dict[sm.symbols("EML")] = params[0][0] or params[0][1]
        for idx in [idx1, idx2]:
            if branches_list[idx].type != "L":
                raise ValueError(
                    "Mutual inductance coupling is only allowed between inductive branches."
                )
        branch1 = branches_list[idx1]
        branch2 = branches_list[idx2]
    sym_params_dict = {
        param[0]: param[1] for param in params if param[0]
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
                param[0] or param[1]
            )

        params_dict[sm.symbols("EC")] = params[-1][0] or params[-1][1]
    if branch_type == "C":
        params_dict[sm.symbols("EC")] = params[-1][0] or params[-1][1]
    elif branch_type == "L":
        params_dict[sm.symbols("EL")] = params[-1][0] or params[-1][1]

    # return idx1, idx2, branch_type, list(params_dict.keys()), str(_branch_count), process_param(aux_params)
    is_grounded = True if any([node.is_ground() for node in nodes_list]) else False
    node_1 = nodes_list[idx1 if is_grounded else idx1 - 1]
    node_2 = nodes_list[idx2 if is_grounded else idx2 - 1]
    sym_params_dict = {
        param[0]: param[1] for param in params if param[0]
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


class SymbolicCircuitGraph(ABC):

    def _spanning_tree(
        self, consider_capacitive_loops: bool = False, use_closure_branches: bool = True
    ):
        r"""Returns a spanning tree (as a list of branches) for the given instance.
        Notice that if the circuit contains multiple capacitive islands, the returned
        spanning tree will not include the capacitive twig between two capacitive
        islands. Option `use_closure_branches` can be set to `False` if one does not
        want to use the internally set closure_branches.

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
        # circ_copy.nodes = circ_copy.nodes
        if circ_copy.ground_node:
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
        # *****************************************************************************

        # **************** Constructing the node_sets ***************
        node_sets_for_trees = []  # seperate node sets for separate trees
        if circ_copy.is_grounded:
            node_sets = [[circ_copy.ground_node]]
        else:
            node_sets = [
                [circ_copy.nodes[0]]
            ]  # starting with the first set that has the first node as the only element
        node_sets_for_trees.append(node_sets)

        num_nodes = len(circ_copy.nodes)
        # this needs to be done as the ground node is not included in self.nodes
        if circ_copy.is_grounded:
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
                for node in circ_copy.nodes:
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
        # ***************************

        # **************** constructing the spanning tree ##########
        def connecting_branches(n1: Node, n2: Node):
            return [branch for branch in n1.branches if branch in n2.branches]

        def is_same_branch(branch_1: Branch, branch_2: Branch):
            return branch_1.index == branch_2.index

        def fetch_same_branch_from_circ(branch: Branch, circ):
            for b in circ.branches:
                if is_same_branch(b, branch):
                    return b

        def fetch_same_node_from_circ(node: Node, circ):
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
                            tree.append(connecting_branches(node, prev_node)[0])
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
                if isinstance(EC, sympy.Symbol):
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
    ) -> Tuple[int, List[Node], List[Branch], int]:
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
    def _parse_nodes(branches_list) -> List[Node]:
        node_index_list = []
        for branch_list_input in [
            branch for branch in branches_list if branch[0] != "ML"
        ]:
            for idx in [1, 2]:
                node_idx = branch_list_input[idx]
                if node_idx not in node_index_list:
                    node_index_list.append(node_idx)
        node_index_list.sort()
        return [Node(idx) for idx in node_index_list]

    @classmethod
    def from_yaml(
        cls,
        input_string: str,
        from_file: bool = True,
        basis_completion: str = "heuristic",
        use_dynamic_flux_grouping: bool = False,
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
                if sym_param in branch_var_dict and sym_params[sym_param]:
                    raise Exception(
                        f"Symbol {sym_param} has already been assigned a value."
                    )
                if sym_params[sym_param]:
                    branch_var_dict[sym_param] = sym_params[sym_param]
            branches_list.append(branch)
        # make couplers
        coupler_branches = [
            branch for branch in parsed_branches if branch not in individual_branches
        ]
        for parsed_branch in coupler_branches:
            coupler, sym_params = make_coupler(branches_list, *parsed_branch)
            for sym_param in sym_params:
                if sym_param in branch_var_dict and sym_params[sym_param]:
                    raise Exception(
                        f"Symbol {sym_param} has already been assigned a value."
                    )
                if sym_params[sym_param]:
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
        if self.ground_node:
            nodes_copy.pop(0)  # removing the ground node
            nodes_copy = nodes_copy + [
                copy.copy(self.ground_node)
            ]  # reversing the order of the nodes

        for node in nodes_copy:  # reset the node markers
            node.marker = 0

        # step 2: finding the maximum connected set of independent branches in
        # branch_subset, then identifying the sets of nodes in each of those sets
        branch_subset_copy = branch_subset.copy()

        max_connected_subgraphs: List[List[Branch]] = (
            []
        )  # list containing the maximum connected subgraphs

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
            unique_elements_in_list(
                list(chain(*[branch.nodes for branch in branch_set]))
            )
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

    def check_transformation_matrix(
        self, transformation_matrix: ndarray, enable_warnings: bool = True
    ) -> Dict[str, List[int]]:
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
        """Evaluates the boundary conditions and constructs the variable transformation
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
        if self.basis_completion == "heuristic":
            node_count = len(self.nodes) - self.is_grounded
            standard_basis = [np.ones(node_count)]

            vector_ref = np.zeros(node_count)
            if node_count > 2:
                vector_ref[: node_count - 2] = 1
            else:
                vector_ref[: node_count - 1] = 1

            vector_set = (
                permutation
                for permutation in itertools.permutations(vector_ref, node_count)
            )  # making a generator
            while np.linalg.matrix_rank(np.array(standard_basis)) < node_count:
                a = next(vector_set)
                mat = np.array(standard_basis + [a])
                if np.linalg.matrix_rank(mat) == len(mat):
                    standard_basis = standard_basis + [list(a)]

            standard_basis = np.array(standard_basis)

        elif self.basis_completion == "canonical":
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
            "sigma": (
                [i + 1 for i in range(len(pos_list)) if pos_list[i] == pos_Σ[0]]
                if not self.is_grounded
                else []
            ),
        }

        return np.array(new_basis), var_categories
