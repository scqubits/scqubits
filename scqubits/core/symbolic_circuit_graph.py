from __future__ import annotations

import copy
import itertools
import warnings

from abc import ABC
from itertools import chain
from typing import Any

import numpy as np
import pyparsing as pp
import scipy as sp
import sympy
import sympy as sm

from numpy import ndarray
from sympy import Symbol, symbols

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings

from scqubits.core.branch_metadata import (
    _capacitance_variable_for_branch,
    _junction_order,
)
from scqubits.core.circuit_input import (
    parse_code_line,
    process_param,
    remove_branchline,
    remove_comments,
    strip_empty_lines,
)
from scqubits.core.circuit_utils import (
    get_trailing_number,
    round_symbolic_expr,
)
from scqubits.utils.misc import (
    flatten_list_recursive,
    unique_elements_in_list,
)


class Node:
    """Circuit node handled by :class:`SymbolicCircuit`.

    The attribute :attr:`branches` is a list of :class:`Branch` objects
    containing all branches connected to the node.

    Parameters
    ----------
    index:
        integer identifier of the node
    marker:
        internal attribute used to group nodes and identify sub-circuits in
        :meth:`SymbolicCircuitGraph._independent_modes`
    """

    def __init__(self, index: int, marker: int = 0):
        self.index: int = index
        self.marker: int = marker
        self._init_params: dict[str, int] = {"id": self.index, "marker": self.marker}
        self.branches: list[Branch] = []

    def __str__(self) -> str:
        """Return a human-readable label of the form ``"Node <index>"``."""
        return "Node {}".format(self.index)

    def __repr__(self) -> str:
        """Return a developer-readable representation of the node."""
        return "Node({})".format(self.index)

    def connected_nodes(self, branch_type: str) -> list["Node"]:
        """Return all nodes directly connected to this node by branches.

        Parameters
        ----------
        branch_type:
            type of branch used to filter neighbors; one of ``"C"``,
            ``"L"``, ``"JJ"``, or ``"all"`` for capacitive, inductive,
            Josephson, or all branch types
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
        """Return a bool if the node is a ground node (Node with index set to 0)."""
        return True if self.index == 0 else False

    def __deepcopy__(self, memo: dict[int, Any]) -> "Node":
        """Return a deep copy of the node, recursing into all instance attributes.

        Parameters
        ----------
        memo:
            standard ``copy.deepcopy`` memoization dictionary
        """
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
        initial :class:`Node` of the branch
    n_f:
        final :class:`Node` of the branch
    branch_type:
        type of this branch, e.g. ``"C"``, ``"JJ"``, or ``"L"``
    parameters:
        list of parameters for the branch, namely for
        capacitance: ``[EC]``;
        for inductance: ``[EL]``;
        for Josephson Junction: ``[EJ, ECJ]``
    index:
        optional integer label identifying the branch within its parent
        circuit (default: ``None``)
    aux_params:
        dictionary of auxiliary parameters mapping a symbol from the input
        file to a numeric parameter

    Examples
    --------
    ``Branch("C", Node(1, 0), Node(2, 0))`` is a capacitive branch connecting
    the nodes with indices 0 and 1.
    """

    def __init__(
        self,
        n_i: Node,
        n_f: Node,
        branch_type: str,
        parameters: list[float | Symbol | int],
        index: int | None = None,
        aux_params: dict[Symbol, float] = {},
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
        """Return a human-readable label of the branch."""
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
        """Return a developer-readable representation of the branch."""
        return f"Branch({self.type}, {self.nodes[0].index}, {self.nodes[1].index}, index: {self.index})"

    def _set_parameters(self, parameters: list[float | Symbol | int]) -> None:
        """Populate :attr:`parameters` from the given raw parameter list.

        For capacitive (``"C"``) and inductive (``"L"``) branches the first
        element is stored under the key ``"EC"`` or ``"EL"``. For Josephson
        branches (``"JJ"``, ``"JJ2"``, ...), one ``"EJ"``, ``"EJ2"``, ...
        entry is stored per junction order, and the trailing element is stored
        under ``"ECJ"``.

        Parameters
        ----------
        parameters:
            list of branch parameters whose layout depends on
            :attr:`type`
        """
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

    def node_ids(self) -> tuple[int, int]:
        """Return the indices of the nodes connected by the branch."""
        return self.nodes[0].index, self.nodes[1].index

    def is_connected(self, branch: "Branch") -> bool:
        """Return whether the current branch shares any node with `branch`.

        Parameters
        ----------
        branch:
            other branch to test for shared nodes
        """
        distinct_node_count = len(set(self.nodes + branch.nodes))
        if distinct_node_count < 4:
            return True
        return False

    def common_node(self, branch: "Branch") -> set[Node]:
        """Return the set of nodes shared with the given `branch`.

        Parameters
        ----------
        branch:
            other branch with which to take the intersection of nodes
        """
        return set(self.nodes) & set(branch.nodes)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Branch":
        """Return a deep copy of the branch, recursing into all instance attributes.

        Parameters
        ----------
        memo:
            standard ``copy.deepcopy`` memoization dictionary
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class Coupler:
    """Class describing an element coupling two existing branches in a circuit.

    Parameters
    ----------
    branch1:
        first :class:`Branch` being coupled
    branch2:
        second :class:`Branch` being coupled
    coupling_type:
        type of coupling between the branches; mutual inductance ``"ML"`` is
        currently the only supported value
    parameters:
        list of parameters for the coupling, e.g. for mutual inductance the
        value ``EML`` is expected at index ``0``
    index:
        optional integer label identifying the coupler within its parent
        circuit (default: ``None``)
    aux_params:
        dictionary of auxiliary parameters mapping a symbol from the input
        file to a numeric parameter

    Examples
    --------
    ``Coupler(branch1, branch2, "ML", [1e2])``
    """

    def __init__(
        self,
        branch1: Branch,
        branch2: Branch,
        coupling_type: str,
        parameters: list[float | Symbol | int],
        index: int | None = None,
        aux_params: dict[Symbol, float] = {},
    ):
        self.branches = (branch1, branch2)
        self.type = coupling_type
        self._set_parameters(parameters)
        self.aux_params = aux_params
        self.index = index

    def _set_parameters(self, parameters: list[float | Symbol | int]) -> None:
        """Populate :attr:`parameters` from the given raw parameter list.

        For mutual-inductance couplings the first element is stored under the
        key ``"EML"``.

        Parameters
        ----------
        parameters:
            list of coupler parameters whose layout depends on :attr:`type`
        """
        if self.type in ["ML"]:
            self.parameters = {f"E{self.type}": parameters[0]}

    def __repr__(self) -> str:
        """Return a developer-readable representation of the coupler."""
        return f"Coupler({self.type}, ({self.branches[0].type}, {self.branches[0].node_ids()}), ({self.branches[1].type}, {self.branches[1].node_ids()}), index: {self.index})"

    def __deepcopy__(self, memo: dict[int, Any]) -> "Coupler":
        """Return a deep copy of the coupler, recursing into all instance attributes.

        Parameters
        ----------
        memo:
            standard ``copy.deepcopy`` memoization dictionary
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


def make_coupler(
    branches_list: list[Branch],
    coupler_type: str,
    idx1: int,
    idx2: int,
    params: list,
    aux_params: pp.ParseResults,
    _branch_count: int,
) -> tuple[Coupler, dict[Symbol, float]]:
    """Build a :class:`Coupler` from parsed pyparsing results.

    Parameters
    ----------
    branches_list:
        list of branches already constructed for the circuit; the coupler is
        attached to ``branches_list[idx1]`` and ``branches_list[idx2]``
    coupler_type:
        type of coupling (currently only ``"ML"`` for mutual inductance)
    idx1:
        index of the first branch in `branches_list`
    idx2:
        index of the second branch in `branches_list`
    params:
        list of pyparsing parse results for the coupler parameters
    aux_params:
        pyparsing parse results for the auxiliary-parameter block
    _branch_count:
        running branch count used as the coupler's :attr:`Coupler.index`

    Returns
    -------
    Tuple ``(coupler, sym_params_dict)`` where ``coupler`` is the constructed
    :class:`Coupler` and ``sym_params_dict`` maps any newly seen symbolic
    parameters to their default numeric values.
    """
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
            process_param(aux_params),  # type: ignore[arg-type]
        ),
        sym_params_dict,
    )


def make_branch(
    nodes_list: list[Node],
    branch_type: str,
    idx1: int,
    idx2: int,
    params: list,
    aux_params: pp.ParseResults,
    _branch_count: int,
) -> tuple[Branch, dict[Symbol, float]]:
    """Build a :class:`Branch` from parsed pyparsing results.

    Parameters
    ----------
    nodes_list:
        list of all :class:`Node` objects already constructed for the circuit
    branch_type:
        type of branch, e.g. ``"C"``, ``"L"``, or a Josephson type such as
        ``"JJ"``
    idx1:
        index identifying the first endpoint node within `nodes_list`
        (interpreted with or without the ground-node offset)
    idx2:
        index identifying the second endpoint node within `nodes_list`
    params:
        list of pyparsing parse results for the branch parameters
    aux_params:
        pyparsing parse results for the auxiliary-parameter block
    _branch_count:
        running branch count used as the branch's :attr:`Branch.index`

    Returns
    -------
    Tuple ``(branch, sym_params_dict)`` where ``branch`` is the constructed
    :class:`Branch` and ``sym_params_dict`` maps any newly seen symbolic
    parameters to their default numeric values.
    """
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
            process_param(aux_params),  # type: ignore[arg-type]
        ),
        sym_params_dict,
    )


class SymbolicCircuitGraph(ABC):
    """Mixin providing graph-theoretic helpers for :class:`SymbolicCircuit`.

    Encapsulates spanning-tree construction, closure-branch detection,
    loop discovery, mode classification, and the node-variable-to-new-variable
    transformation matrix. The concrete circuit data (nodes, branches,
    grounding, etc.) is provided by the subclass and declared below as
    class-level type annotations.
    """

    # class-level annotations for attributes provided by subclass SymbolicCircuit
    nodes: list[Node]
    branches: list[Branch]
    ground_node: Node | None
    is_grounded: bool
    use_dynamic_flux_grouping: bool
    is_purely_harmonic: bool
    basis_completion: str
    spanning_tree_dict: dict[str, Any]
    closure_branches: list[Branch | dict[Branch, float]]
    external_fluxes: list[Symbol]

    def _spanning_tree(
        self,
        consider_capacitive_loops: bool = False,
        use_closure_branches: bool = True,
    ) -> dict[str, list]:
        r"""Return a spanning tree (as a list of branches) for the given instance.

        If the circuit contains multiple capacitive islands, the returned
        spanning tree will not include the capacitive twig between two
        capacitive islands.

        Within node-set construction the candidate nodes are sorted in place
        with ``list.sort`` so that the order of the spanning tree is
        deterministic across runs.

        For the example spanning tree::

                       /---Node(2)
            Node(1)---'
                       '---Node(3)---Node(4)

        the ``node_sets`` is returned as ``[[Node(1)], [Node(2), Node(3)],
        [Node(4)]]``.

        Parameters
        ----------
        consider_capacitive_loops:
            when ``True``, capacitive branches are kept while building the
            spanning tree, so loops formed purely by capacitors are detected
            (default: ``False``)
        use_closure_branches:
            when ``True`` (default), the internally set
            :attr:`closure_branches` are used to decide which branches lie
            outside the spanning tree

        Returns
        -------
        Dictionary with keys ``"list_of_trees"``, ``"loop_branches_for_trees"``,
        ``"node_sets_for_trees"``, and ``"closure_branches_for_trees"``,
        each mapping to a per-tree list. ``list_of_trees`` excludes capacitor
        branches, ``loop_branches_for_trees`` collects all branches forming
        superconducting loops for each tree, ``node_sets_for_trees`` records
        the generation info for the nodes on the path, and
        ``closure_branches_for_trees`` lists the closure branches for each
        tree.
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
        node_sets_for_trees: list[list[list[Node]]] = (
            []
        )  # seperate node sets for separate trees
        if circ_copy.is_grounded:
            node_sets = [[circ_copy.ground_node]]
        else:
            node_sets = [
                [circ_copy.nodes[0]]
            ]  # starting with the first set that has the first node as the only element
        node_sets_for_trees.append(node_sets)  # type: ignore[arg-type]

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
        for node_sets in node_sets_for_trees:  # type: ignore[assignment]
            tree = []  # tree having branches of the instance that is copied

            # find the branch connecting this node to another node in a previous node set.
            for index, node_set in enumerate(node_sets):  # type: ignore[assignment]
                if index == 0:
                    continue
                for node in node_set:
                    for prev_node in node_sets[index - 1]:
                        if len(connecting_branches(node, prev_node)) != 0:  # type: ignore[arg-type]
                            tree.append(connecting_branches(node, prev_node)[0])  # type: ignore[arg-type]
                            break
            list_of_trees.append(tree)

        # as the capacitors are removed to form the spanning tree, and as a result
        # floating branches as well, the set of all branches which form the
        # superconducting loops would be in circ_copy.
        closure_branches_for_trees: list[list[Branch]] = [[] for tree in list_of_trees]
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

    def _closure_branches(
        self, spanning_tree_dict: dict[str, list] | None = None
    ) -> list[Branch | dict[Branch, float]]:
        r"""Return the flattened list of closure branches in the circuit.

        Parameters
        ----------
        spanning_tree_dict:
            spanning-tree dictionary from which the closure branches are
            extracted; if ``None`` (default), the cached
            :attr:`spanning_tree_dict` is used
        """
        return flatten_list_recursive(
            (spanning_tree_dict or self.spanning_tree_dict)[
                "closure_branches_for_trees"
            ]
        )

    def _time_dependent_flux_distribution(self):
        r"""Return the symbolic time-dependent flux distribution across branches.

        Constructs the constraint matrix :math:`R` from the loops generated
        by each closure branch, the diagonal branch-capacitance matrix
        :math:`C_\text{diag}` (with :math:`1/(8\,E_C)` entries), and the
        node-to-branch incidence matrix :math:`W`. Solves the resulting
        least-squares problem ``M @ B = I`` via the Moore-Penrose pseudoinverse
        ``np.linalg.pinv``, then projects ``B`` onto the symbolic external
        fluxes.

        Returns
        -------
        Array of length ``len(self.branches)`` whose entries are the symbolic
        per-branch flux contributions in terms of the external-flux symbols.
        """
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
        self, node: Node, spanning_tree_dict: dict[str, list] | None = None
    ) -> tuple[int, list[Node], list[Branch], int]:
        r"""Return spanning-tree path data from the input node to the root.

        The root of the spanning tree is node 0 if there is a physical ground
        node, otherwise it is node 1. Branches sitting on the boundaries of
        capacitive islands are not included in the returned branch list.

        Parameters
        ----------
        node:
            input node whose path to the root is sought
        spanning_tree_dict:
            spanning-tree dictionary as returned by :meth:`_spanning_tree`;
            if ``None`` (default), the cached :attr:`spanning_tree_dict` is
            used

        Returns
        -------
        Tuple ``(generation, ancestor_nodes_list, branch_path_to_root,
        tree_idx)`` containing the generation number, the list of ancestor
        nodes from `node` up to the root, the list of branches on that path,
        and the index of the tree containing `node`.
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
            ancestor_nodes_list: list[Node] = []
            branch_path_to_root: list[Branch] = []
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
        self,
        closure_branch: Branch,
        spanning_tree_dict: dict[str, list] | None = None,
    ) -> list["Branch"]:
        r"""Find out the loop that is closed by the closure branch.

        Parameters
        ----------
        closure_branch:
            input closure branch
        spanning_tree_dict:
            spanning-tree dictionary as returned by :meth:`_spanning_tree`;
            if ``None`` (default), the cached :attr:`spanning_tree_dict` is
            used

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

    def _order_branches_in_loop(self, loop_branches: list[Branch]) -> list[Branch]:
        """Reorder a set of loop branches into a contiguous traversal sequence.

        Walks the branches by chaining each one to the next via the shared
        node, producing an ordering in which consecutive branches share a
        node.

        Parameters
        ----------
        loop_branches:
            unordered branches that together form a single loop
        """
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
        closure_branches: list[Branch | dict[Branch, float]] | None = None,
    ):
        r"""Populate :attr:`external_fluxes` and :attr:`closure_branches`.

        For each user-supplied or auto-detected closure branch, a symbolic
        external flux ``Φi`` is created and attached to the instance. Raises
        :exc:`ValueError` if a capacitive branch is used as a closure branch
        without :attr:`use_dynamic_flux_grouping`.

        Parameters
        ----------
        closure_branches:
            optional list of closure branches; each entry is either a
            :class:`Branch` or a mapping from branches to weights describing
            how the external flux is distributed
        """
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
                if branch.type == "C" and not self.use_dynamic_flux_grouping:  # type: ignore[union-attr]
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
        branch_list1: list[Branch], branch_list2: list[Branch]
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
    def _parse_nodes(branches_list: list) -> list[Node]:
        """Construct the list of unique :class:`Node` instances from branch specs.

        Coupler entries (those starting with ``"ML"``) are skipped. The
        resulting node indices are sorted in place via ``list.sort`` so the
        output node ordering is deterministic.

        Parameters
        ----------
        branches_list:
            sequence of parsed branch specifications, each beginning with the
            branch type followed by two node indices
        """
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
        """Construct an instance of :class:`SymbolicCircuit` from an input string.

        Example input string used to initiate an object::

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
            string describing the number of nodes and branches connecting them
            along with their parameters
        from_file:
            when ``True`` (default), `input_string` is interpreted as a file
            name whose contents are loaded; otherwise it is treated as the
            circuit graph description in YAML
        basis_completion:
            choice of basis used to complete the transformation matrix; one
            of ``"heuristic"`` (default) or ``"canonical"``
        use_dynamic_flux_grouping:
            when ``True``, the flux allocation is performed assuming
            time-dependent flux, which disables the option to change the
            closure branches (default: ``False``)
        initiate_sym_calc:
            when ``True`` (default), the object attributes are initialized by
            calling :meth:`SymbolicCircuit.configure`; set to ``False`` for
            debugging

        Returns
        -------
        Instance of :class:`SymbolicCircuit`.
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

        circuit = cls(  # type: ignore[call-arg]
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
        branch_subset: list[Branch],
        single_nodes: bool = True,
        basisvec_entries: list[int] | None = None,
    ):
        """Return vectors spanning the no-flux-difference subspace for the branches.

        The returned basis vectors correspond to node-variable combinations
        across which there is no generalized flux difference for any branch
        in `branch_subset`.

        Parameters
        ----------
        branch_subset:
            list of branches across which the no-flux-difference condition is
            imposed
        single_nodes:
            when ``True`` (default), single-node modes are also taken into
            consideration when constructing the basis vectors
        basisvec_entries:
            two-element sequence ``[on, off]`` giving the values used for
            "in-set" and "out-of-set" entries of the basis vectors; defaults
            to ``[1, 0]``
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

        max_connected_subgraphs: list[list[Branch]] = (
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
    def _mode_in_subspace(mode: ndarray | list, subspace: ndarray | list) -> bool:
        """Check whether the vector `mode` is a part of the given `subspace`.

        Parameters
        ----------
        mode:
            one-dimensional vector to be tested for membership
        subspace:
            collection of basis vectors representing the vector subspace
        """
        if len(subspace) == 0:
            return False
        matrix = np.vstack([subspace, np.array(mode)])
        return np.linalg.matrix_rank(matrix) == len(subspace)

    def check_transformation_matrix(
        self, transformation_matrix: ndarray, enable_warnings: bool = True
    ) -> dict[str, list[int]]:
        """Identify the different modes in a user-provided transformation matrix.

        Parameters
        ----------
        transformation_matrix:
            square ndarray with the dimensions of the number of nodes present
            in the circuit
        enable_warnings:
            when ``True`` (default), warnings are emitted regarding any
            unidentified modes

        Returns
        -------
        Dictionary of lists classifying the variable indices, where the
        indices correspond to the rows of the transformation matrix.
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
        modes: list[ndarray] = []  # starting with the frozen modes

        for m in (
            frozen_modes + free_modes + periodic_modes + LC_modes  # + extended_modes
        ):  # This order is important
            if not self._mode_in_subspace(m, modes):
                modes.append(m)

        for m in LC_modes:  # adding the LC modes to the basis
            if not self._mode_in_subspace(m, modes):
                modes.append(m)

        var_categories_circuit: dict[str, list] = {
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

        var_categories_user: dict[str, list] = {
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

    def variable_transformation_matrix(self) -> tuple[ndarray, dict[str, list[int]]]:
        """Construct the variable transformation matrix and variable categories.

        Evaluates the boundary conditions and builds the node-to-new-variable
        transformation matrix, returning it along with the ``var_categories``
        dictionary that classifies the types of variables present in the
        circuit.

        Returns
        -------
        Tuple ``(transformation_matrix, var_categories)`` where
        ``transformation_matrix`` maps node variables to new variables and
        ``var_categories`` classifies the variable types (periodic, extended,
        free, frozen, sigma) for each variable index.
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
        modes: list[ndarray] = []  # starting with an empty list

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
                    standard_basis = standard_basis + [list(a)]  # type: ignore[list-item]

            standard_basis = np.array(standard_basis)  # type: ignore[assignment]

        elif self.basis_completion == "canonical":
            standard_basis = np.identity(len(self.nodes) - self.is_grounded)  # type: ignore[assignment]

        new_basis = modes.copy()

        for m in standard_basis:  # completing the basis
            mat = np.array([i for i in new_basis] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                new_basis.append(m)

        new_basis = np.array(new_basis)  # type: ignore[assignment]

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
        new_basis = new_basis[pos_list].T  # type: ignore[call-overload]

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
