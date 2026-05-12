from __future__ import annotations

import copy
import itertools
import warnings

from abc import ABC
from itertools import chain
from typing import Any, Literal

import numpy as np
import pyparsing as pp
import scipy as sp
import sympy
import sympy as sm

from numpy import ndarray
from sympy import Symbol, symbols

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings

from scqubits.core.circuit_internals.branch_metadata import (
    _capacitance_variable_for_branch,
    _junction_order,
)
from scqubits.core.circuit_internals.input import (
    parse_code_line,
    process_param,
    remove_branchline,
    remove_comments,
    strip_empty_lines,
)
from scqubits.core.circuit_internals.sympy_helpers import round_symbolic_expr
from scqubits.core.circuit_internals.utils import get_trailing_number
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
    """

    def __init__(self, index: int):
        self.index: int = index
        self._init_params: dict[str, int] = {"id": self.index}
        self.branches: list[Branch] = []

    def __str__(self) -> str:
        """Return a human-readable label of the form ``"Node <index>"``."""
        return "Node {}".format(self.index)

    def __repr__(self) -> str:
        """Return a developer-readable representation of the node."""
        return "Node({})".format(self.index)

    def __eq__(self, other: object) -> bool:
        """Two nodes are equal iff they share the same ``index``."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        """Hash by ``index`` so equal nodes hash equal; required for set/dict use."""
        return hash(self.index)

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
            if branch.nodes[0] == self:
                result.append(branch.nodes[1])
            else:
                result.append(branch.nodes[0])
        return result

    def is_ground(self) -> bool:
        """Return a bool if the node is a ground node (Node with index set to 0)."""
        return self.index == 0

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

    def __eq__(self, other: object) -> bool:
        """Two branches are equal iff they share the same ``index``.

        This makes membership / containment checks across deep-copies of
        the circuit graph (e.g. inside ``_spanning_tree``) behave by
        circuit-topology identity rather than Python object identity.
        Use ``is`` / ``is not`` when you need actual object identity.
        """
        if not isinstance(other, Branch):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        """Hash by ``index`` so equal branches hash equal; required for set/dict use."""
        return hash(self.index)

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


def _branch_param_dict(branch_type: str, params: list) -> dict[Symbol, float]:
    """Build the ``{Symbol: numeric default}`` dict for a branch.

    Encodes the per-branch-type convention for parameter naming:
    ``"C"`` -> {EC}, ``"L"`` -> {EL}, ``"JJ[n]"`` -> {EJ, EJ2, ..., EJn,
    EC}.  The trailing element of ``params`` is always the
    junction/branch capacitance for JJ branches.
    """
    params_dict: dict[Symbol, float] = {}
    if "JJ" in branch_type:
        for idx, param in enumerate(params[:-1]):
            symbol = sm.symbols(f"EJ{idx + 1}" if idx > 0 else "EJ")
            params_dict[symbol] = param[0] or param[1]
        params_dict[sm.symbols("EC")] = params[-1][0] or params[-1][1]
    elif branch_type == "C":
        params_dict[sm.symbols("EC")] = params[-1][0] or params[-1][1]
    elif branch_type == "L":
        params_dict[sm.symbols("EL")] = params[-1][0] or params[-1][1]
    return params_dict


def make_branch(
    nodes_list: list[Node],
    branch_type: str,
    idx1: int,
    idx2: int,
    params: list,
    aux_params: pp.ParseResults,
    _branch_count: int,
    *,
    node_index_offset: int | None = None,
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
        node-ID of the first endpoint as it appears in the YAML
    idx2:
        node-ID of the second endpoint as it appears in the YAML
    params:
        list of pyparsing parse results for the branch parameters
    aux_params:
        pyparsing parse results for the auxiliary-parameter block
    _branch_count:
        running branch count used as the branch's :attr:`Branch.index`
    node_index_offset:
        offset to subtract from ``idx1`` / ``idx2`` to land on the
        correct position in ``nodes_list``.  Use ``0`` when the YAML
        is 0-indexed (ground node present at position 0) and ``1``
        when the YAML is 1-indexed (no ground node, lowest node-ID is
        1).  When ``None`` (default), the offset is inferred from
        ``nodes_list`` for backward compatibility — passing it
        explicitly is the recommended path.

    Returns
    -------
    Tuple ``(branch, sym_params_dict)`` where ``branch`` is the constructed
    :class:`Branch` and ``sym_params_dict`` maps any newly seen symbolic
    parameters to their default numeric values.
    """
    params = [process_param(param) for param in params]
    params_dict = _branch_param_dict(branch_type, params)

    if node_index_offset is None:
        node_index_offset = 0 if any(n.is_ground() for n in nodes_list) else 1
    node_1 = nodes_list[idx1 - node_index_offset]
    node_2 = nodes_list[idx2 - node_index_offset]

    sym_params_dict = {param[0]: param[1] for param in params if param[0]}
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


class _AdjacencyIndex:
    """Per-spanning-tree adjacency index supporting O(depth) path-to-root.

    Built once from the ``list_of_trees`` and ``node_sets_for_trees`` of a
    :meth:`SymbolicCircuitGraph._spanning_tree` result.  Each component
    (entry of ``list_of_trees``) gets a parent map computed by BFS from
    the root (``node_sets[0][0]``).

    The previous ``_find_path_to_root`` implementation enumerated
    ``itertools.permutations(tree)`` of the spanning-tree branches and
    tried each permutation until one assembled a valid ancestor chain —
    O(tree_size!) in the worst case.  Because a spanning tree is acyclic,
    the path between any two nodes is unique; DFS / BFS finds the same
    path the brute-force search would have found, in O(depth).
    """

    __slots__ = (
        "_node_to_tree_idx",
        "_node_to_generation",
        "_parent_in_tree",
    )

    def __init__(
        self,
        list_of_trees: list[list[Branch]],
        node_sets_for_trees: list[list[list[Node]]],
    ) -> None:
        self._node_to_tree_idx: dict[Node, int] = {}
        self._node_to_generation: dict[Node, int] = {}
        # parent_in_tree[node] = (parent_node, connecting_branch); root maps to None.
        self._parent_in_tree: dict[Node, tuple[Node, Branch] | None] = {}
        for tree_idx, tree in enumerate(list_of_trees):
            node_sets = node_sets_for_trees[tree_idx]
            for generation, nodes_in_layer in enumerate(node_sets):
                for n in nodes_in_layer:
                    self._node_to_tree_idx[n] = tree_idx
                    self._node_to_generation[n] = generation
            if not node_sets:
                continue
            root = node_sets[0][0]
            self._parent_in_tree[root] = None
            self._build_parent_map_via_bfs(root, tree)

    def _build_parent_map_via_bfs(self, root: Node, tree: list[Branch]) -> None:
        """Populate ``self._parent_in_tree`` via BFS from ``root``.

        Builds an adjacency dict from the branches in ``tree``, then walks
        outward from ``root``.  For every newly-discovered node, records
        the ``(parent_node, connecting_branch)`` pair under the node's key
        in ``_parent_in_tree``; ``root`` itself was already mapped to
        ``None`` by the caller.  Each node is visited exactly once because
        ``tree`` is acyclic.
        """
        # adjacency: node -> list of (neighbour, branch) pairs, restricted
        # to the tree's branches
        adj: dict[Node, list[tuple[Node, Branch]]] = {}
        for branch in tree:
            n0, n1 = branch.nodes
            adj.setdefault(n0, []).append((n1, branch))
            adj.setdefault(n1, []).append((n0, branch))
        frontier = [root]
        seen: set[Node] = {root}
        while frontier:
            next_frontier: list[Node] = []
            for current in frontier:
                for neighbour, branch in adj.get(current, []):
                    if neighbour in seen:
                        continue
                    self._parent_in_tree[neighbour] = (current, branch)
                    seen.add(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier

    def path_to_root(self, node: Node) -> tuple[int, list[Node], list[Branch], int]:
        """Return ``(generation, ancestor_nodes, branch_path, tree_idx)``.

        ``ancestor_nodes`` lists the path *up* to but not including the
        root, in root-to-node order; ``branch_path`` lists the branches
        traversed in the same direction.  When ``node`` is itself the
        root, both lists are empty and ``generation == 0``.

        Raises ``ValueError`` if ``node`` is not present in any tree.
        """
        if node not in self._node_to_tree_idx:
            raise ValueError(
                f"Node {node!r} is not present in any spanning tree of this "
                f"adjacency index."
            )
        tree_idx = self._node_to_tree_idx[node]
        generation = self._node_to_generation[node]

        ancestors: list[Node] = []
        branches: list[Branch] = []
        current = node
        # Defensive cycle guard (should be unreachable for an actual tree)
        seen_nodes: set[Node] = {current}
        while True:
            parent_link = self._parent_in_tree.get(current)
            if parent_link is None:
                break
            parent, branch = parent_link
            if parent in seen_nodes:
                raise RuntimeError(
                    "Spanning-tree adjacency contained a cycle; this should "
                    "be impossible for a tree.  Inputs to _AdjacencyIndex "
                    "are inconsistent."
                )
            ancestors.append(parent)
            branches.append(branch)
            seen_nodes.add(parent)
            current = parent

        ancestors.reverse()
        branches.reverse()
        return generation, ancestors, branches, tree_idx

    @classmethod
    def from_spanning_tree_dict(
        cls, spanning_tree_dict: dict[str, list]
    ) -> "_AdjacencyIndex":
        """Build an index from a :meth:`SymbolicCircuitGraph._spanning_tree` result.

        Reads the ``"list_of_trees"`` and ``"node_sets_for_trees"`` keys
        of the dict and forwards them to ``__init__``.  The other keys
        (``"loop_branches_for_trees"``, ``"closure_branches_for_trees"``)
        are not needed by the index.
        """
        return cls(
            spanning_tree_dict["list_of_trees"],
            spanning_tree_dict["node_sets_for_trees"],
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

    def _local_copy_of_branch(self, branch: Branch) -> Branch | None:
        """Return ``self``'s branch matching ``branch`` (equal ``.index``), or ``None``.

        Lets the spanning-tree construction work on a deepcopy and then
        remap results back to ``self``'s actual ``Branch`` objects.
        """
        return next((b for b in self.branches if b == branch), None)

    def _local_copy_of_node(self, node: Node) -> Node | None:
        """Return ``self``'s node matching ``node`` (equal ``.index``), or ``None``.

        Sibling of :meth:`_local_copy_of_branch`; used by
        :meth:`_remap_spanning_tree_to_self` to substitute deepcopy-copied
        node references with their counterparts on ``self``.
        """
        return next((n for n in self.nodes if n == node), None)

    def _remap_spanning_tree_to_self(
        self,
        list_of_trees: list,
        loop_branches_for_trees: list,
        closure_branches_for_trees: list,
        node_sets_for_trees: list,
    ) -> None:
        """Replace branches/nodes from a working circuit copy with their counterparts on ``self``.

        The spanning-tree construction in ``_spanning_tree`` operates on a
        ``copy.deepcopy(self)``; this helper walks the four output lists
        and substitutes each branch / node with its matching one on
        ``self`` (``Branch`` and ``Node`` define ``__eq__`` by ``.index``).
        """
        for tree_idx in range(len(list_of_trees)):
            list_of_trees[tree_idx] = [
                self._local_copy_of_branch(b) for b in list_of_trees[tree_idx]
            ]
            loop_branches_for_trees[tree_idx] = [
                self._local_copy_of_branch(b) for b in loop_branches_for_trees[tree_idx]
            ]
            closure_branches_for_trees[tree_idx] = [
                self._local_copy_of_branch(b)
                for b in closure_branches_for_trees[tree_idx]
            ]
            node_sets_for_trees[tree_idx] = [
                [self._local_copy_of_node(n) for n in node_set]
                for node_set in node_sets_for_trees[tree_idx]
            ]

    @staticmethod
    def _remove_capacitive_branches(circ: "SymbolicCircuitGraph") -> None:
        """Drop every capacitor branch from ``circ`` and detach it from its nodes.

        Mutates ``circ`` in place. Used by ``_spanning_tree`` when
        capacitive loops are NOT being considered, so the spanning-tree
        construction sees only inductive / Josephson branches.
        """
        capacitive = [b for b in list(circ.branches) if b.type == "C"]
        for c_branch in capacitive:
            for node in c_branch.nodes:
                node.branches = [b for b in node.branches if b is not c_branch]
            circ.branches.remove(c_branch)

    @staticmethod
    def _prune_floating_nodes(circ: "SymbolicCircuitGraph") -> None:
        """Iteratively remove nodes (and their unique branch, if any) that aren't part of a loop.

        A node with zero branches is dropped outright. A node with exactly
        one branch is a leaf of the graph: its branch can't appear in any
        loop, so the node-and-branch pair is removed and the neighbour's
        branch list is updated. Repeats until no floating nodes remain.

        Mutates ``circ`` in place.
        """
        while True:
            num_float_nodes = 0
            for node in list(circ.nodes):
                if not node.branches:
                    circ.nodes.remove(node)
                    num_float_nodes += 1
                    continue
                if len(node.branches) == 1:
                    leaf_branch = node.branches[0]
                    circ.branches.remove(leaf_branch)
                    for neighbour in leaf_branch.nodes:
                        if neighbour is not node:
                            neighbour.branches = [
                                b for b in neighbour.branches if b is not leaf_branch
                            ]
                            num_float_nodes += 1
                        else:
                            circ.nodes.remove(node)
            if num_float_nodes == 0:
                break

    @staticmethod
    def _build_node_sets_for_trees(
        circ: "SymbolicCircuitGraph",
    ) -> list[list[list[Node]]]:
        """BFS-layer the nodes of ``circ`` into per-tree, per-generation node sets.

        Returns one list of generation-layered node sets per connected
        component (each component is a separate "tree" in the
        spanning-tree sense; multiple components arise when capacitive
        branches are removed and the circuit splits into disjoint
        capacitive islands).

        Each layer holds the nodes one BFS step further from the root
        than the previous layer. Layers within a component are sorted by
        ``Node.index`` for deterministic output.

        Caller must ensure ``circ.nodes`` does not include the ground
        node when ``circ.is_grounded``.
        """
        node_sets_for_trees: list[list[list[Node]]] = []
        if circ.is_grounded:
            if circ.ground_node is None:
                raise ValueError(
                    "Inconsistent circuit state: is_grounded=True but "
                    "ground_node is None."
                )
            initial_layer: list[Node] = [circ.ground_node]
        else:
            initial_layer = [circ.nodes[0]]
        node_sets_for_trees.append([initial_layer])

        # ground node lives outside circ.nodes when grounded; count it
        # explicitly so the BFS termination check covers it
        num_nodes = len(circ.nodes) + (1 if circ.is_grounded else 0)

        # ``visited`` tracks the set of nodes already placed into some
        # node-set across all components, used by the BFS termination
        # check below.
        visited: set[Node] = {n for n in initial_layer}

        node_set_index = 0
        tree_index = 0
        while len(visited) < num_nodes:
            current_layer = node_sets_for_trees[tree_index][node_set_index]

            neighbours: list[Node] = []
            for node in current_layer:
                neighbours += node.connected_nodes("all")

            next_layer = [
                n for n in unique_elements_in_list(neighbours) if n not in visited
            ]
            if next_layer:
                next_layer.sort(key=lambda node: node.index)

            if not next_layer:
                # current component exhausted; start a new tree on the
                # next un-visited node (handles disjoint capacitive islands)
                node_sets_for_trees.append([])
                for node in circ.nodes:
                    if node not in visited:
                        tree_index += 1
                        node_sets_for_trees[tree_index].append([node])
                        node_set_index = 0
                        visited.add(node)
                        break
                continue

            node_sets_for_trees[tree_index].append(next_layer)
            visited.update(next_layer)
            node_set_index += 1

        return node_sets_for_trees

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

        if not consider_capacitive_loops:
            self._remove_capacitive_branches(circ_copy)
        self._prune_floating_nodes(circ_copy)

        if not circ_copy.nodes:
            return {
                "list_of_trees": [],
                "loop_branches_for_trees": [],
                "node_sets_for_trees": [],
                "closure_branches_for_trees": [],
            }

        node_sets_for_trees = self._build_node_sets_for_trees(circ_copy)
        list_of_trees = self._construct_initial_tree_per_component(node_sets_for_trees)
        loop_branches_for_trees, closure_branches_for_trees = (
            self._loop_branches_and_closures_per_component(
                circ_copy, list_of_trees, node_sets_for_trees
            )
        )

        self._remap_spanning_tree_to_self(
            list_of_trees,
            loop_branches_for_trees,
            closure_branches_for_trees,
            node_sets_for_trees,
        )

        list_of_trees, closure_branches_for_trees = self._apply_user_closure_policy(
            list_of_trees,
            loop_branches_for_trees,
            closure_branches_for_trees,
            use_closure_branches,
        )

        return {
            "list_of_trees": list_of_trees,
            "loop_branches_for_trees": loop_branches_for_trees,
            "node_sets_for_trees": node_sets_for_trees,
            "closure_branches_for_trees": closure_branches_for_trees,
        }

    @staticmethod
    def _construct_initial_tree_per_component(
        node_sets_for_trees: list[list[list[Node]]],
    ) -> list[list[Branch]]:
        """Greedy spanning-tree per connected component, layer-by-layer.

        For each connected component (one entry of ``node_sets_for_trees``),
        walk the BFS layers and pick exactly one branch per layer transition
        — the first branch found connecting any node in the current layer to
        any node in the previous layer.  Returns one list of branches per
        component, in tree-construction order.
        """

        def connecting_branches(n1: Node, n2: Node) -> list[Branch]:
            return [branch for branch in n1.branches if branch in n2.branches]

        list_of_trees: list[list[Branch]] = []
        for node_sets in node_sets_for_trees:
            tree: list[Branch] = []
            for index, node_set in enumerate(node_sets):
                if index == 0:
                    continue
                for node in node_set:
                    for prev_node in node_sets[index - 1]:
                        connecting = connecting_branches(node, prev_node)
                        if connecting:
                            tree.append(connecting[0])
                            break
            list_of_trees.append(tree)
        return list_of_trees

    @staticmethod
    def _loop_branches_and_closures_per_component(
        circ_copy: "SymbolicCircuitGraph",
        list_of_trees: list[list[Branch]],
        node_sets_for_trees: list[list[list[Node]]],
    ) -> tuple[list[list[Branch]], list[list[Branch]]]:
        """Compute ``(loop_branches, closure_branches)`` per component.

        For each component, the loop branches are the tree branches plus
        every other branch whose two endpoints both lie inside the
        component's node set; the closure branches are exactly the
        non-tree subset of those.  Order: tree branches first (in
        tree-construction order), then closure candidates (in
        ``circ_copy.branches`` iteration order) — preserved from the
        legacy implementation because downstream methods iterate these
        lists.
        """
        loop_branches_for_trees: list[list[Branch]] = []
        closure_branches_for_trees: list[list[Branch]] = [[] for _ in list_of_trees]
        for tree_idx, tree in enumerate(list_of_trees):
            loop_branches = tree.copy()
            nodes_in_tree = flatten_list_recursive(node_sets_for_trees[tree_idx])
            for branch in circ_copy.branches:
                if branch in tree:
                    continue
                if all(node in nodes_in_tree for node in branch.nodes):
                    loop_branches.append(branch)
                    closure_branches_for_trees[tree_idx].append(branch)
            loop_branches_for_trees.append(loop_branches)
        return loop_branches_for_trees, closure_branches_for_trees

    def _apply_user_closure_policy(
        self,
        list_of_trees: list[list[Branch]],
        loop_branches_for_trees: list[list[Branch]],
        closure_branches_for_trees: list[list[Branch]],
        use_closure_branches: bool,
    ) -> tuple[list[list[Branch]], list[list[Branch]]]:
        """If the user has supplied explicit ``Branch`` closures, override the
        auto-derived tree/closure split: each component's tree becomes
        ``loop_branches \\ user_closures`` and its closures become
        ``loop_branches ∩ user_closures``.

        Otherwise return the auto-derived split unchanged.
        """
        user_set_explicit_closures = (
            self.closure_branches != []
            and use_closure_branches
            and np.all([isinstance(elem, Branch) for elem in self.closure_branches])
        )
        if not user_set_explicit_closures:
            return list_of_trees, closure_branches_for_trees

        user_closures = self.closure_branches
        new_list_of_trees = [
            [b for b in loop_branches if b not in user_closures]
            for loop_branches in loop_branches_for_trees
        ]
        new_closures = [
            [b for b in loop_branches if b in user_closures]
            for loop_branches in loop_branches_for_trees
        ]
        return new_list_of_trees, new_closures

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
            loop_branches = self._find_loop(closure_branch, self.spanning_tree_dict)
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
        self, node: Node, spanning_tree_dict: dict[str, list]
    ) -> tuple[int, list[Node], list[Branch], int]:
        r"""Return spanning-tree path data from the input node to the root.

        The root of each spanning tree is the first node of the tree's
        ``node_sets[0]`` (node 0 when grounded, node 1 otherwise).
        Implemented as O(depth) DFS via :class:`_AdjacencyIndex`; the
        previous brute-force ``itertools.permutations(tree)`` enumeration
        was O(tree_size!).

        A spanning tree is by definition acyclic, so the path between any
        two nodes is unique — DFS finds the same path the permutation
        search would have returned.

        Parameters
        ----------
        node:
            input node whose path to the root is sought
        spanning_tree_dict:
            spanning-tree dictionary as returned by :meth:`_spanning_tree`.
            Required; pass ``self.spanning_tree_dict`` explicitly when the
            cached version is wanted.  (The implicit-fallback ``None``
            default was removed because it could mask staleness — a
            caller that passed ``None`` by accident would silently use
            an out-of-date cache and get wrong answers.)

        Returns
        -------
        Tuple ``(generation, ancestor_nodes_list, branch_path_to_root,
        tree_idx)`` containing the generation number, the list of ancestor
        nodes from ``node`` up to the root, the list of branches on that
        path, and the index of the tree containing ``node``.

        Raises
        ------
        ValueError
            if ``node`` is not present in any spanning tree of
            ``spanning_tree_dict`` (the legacy code silently returned
            stale data in this case).
        """
        adjacency = self._adjacency_for_spanning_tree_dict(spanning_tree_dict)
        return adjacency.path_to_root(node)

    def _adjacency_for_spanning_tree_dict(
        self, tree_info_dict: dict[str, list]
    ) -> "_AdjacencyIndex":
        """Return a cached :class:`_AdjacencyIndex` keyed by dict identity.

        ``_find_path_to_root`` and ``_find_loop`` are called repeatedly
        against the same ``self.spanning_tree_dict`` (e.g. once per
        closure branch in :meth:`_time_dependent_flux_distribution`).
        Building the index is O(|nodes| + |branches|); caching it here
        avoids rebuilding on every call.  The cache holds a strong
        reference to the dict it was built from, so an ``is`` comparison
        is both correct and safe — id-recycling cannot produce a stale
        hit because the previously-cached dict is kept alive by the
        cache itself.
        """
        cached = getattr(self, "_adjacency_cache", None)
        if cached is None or cached[0] is not tree_info_dict:
            adjacency = _AdjacencyIndex.from_spanning_tree_dict(tree_info_dict)
            object.__setattr__(
                self, "_adjacency_cache", (tree_info_dict, adjacency)
            )
            return adjacency
        return cached[1]

    def _find_loop(
        self,
        closure_branch: Branch,
        spanning_tree_dict: dict[str, list],
    ) -> list["Branch"]:
        r"""Find out the loop that is closed by the closure branch.

        Parameters
        ----------
        closure_branch:
            input closure branch
        spanning_tree_dict:
            spanning-tree dictionary as returned by :meth:`_spanning_tree`.
            Required; pass ``self.spanning_tree_dict`` explicitly when
            the cached version is wanted.  (Same rationale as
            :meth:`_find_path_to_root`: the implicit-fallback ``None``
            default could mask staleness.)

        Returns
        -------
        A list of branches that corresponds to the loop closed by the closure branch
        """
        _, _, path_1, _ = self._find_path_to_root(
            closure_branch.nodes[0], spanning_tree_dict
        )
        _, _, path_2, _ = self._find_path_to_root(
            closure_branch.nodes[1], spanning_tree_dict
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
        ``True`` if the two branch sets share no nodes (are disconnected),
        ``False`` if they share at least one node.
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

    @staticmethod
    def _merge_branch_symbols(branch_var_dict: dict, sym_params: dict) -> None:
        """Merge ``sym_params`` into ``branch_var_dict``, raising on redefinition.

        A symbol ``S`` is recorded in ``branch_var_dict[S]`` the first time
        a branch declares it with a non-empty value.  Re-declaring the same
        symbol with a (different) non-empty value on a later branch is a
        user error and raises ``ValueError``.  Re-declaring with an empty
        value is silently allowed (it just references the prior value).
        """
        for sym_param, value in sym_params.items():
            if not value:
                continue
            if sym_param in branch_var_dict:
                raise ValueError(
                    f"Symbol {sym_param} has already been assigned a value."
                )
            branch_var_dict[sym_param] = value

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
            with open(input_string, "r") as file:
                circuit_desc = file.read()
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
        branches_list: list[Branch] = []
        couplers_list: list[Coupler] = []
        branch_var_dict: dict = {}

        # YAML node-IDs are 0-based when a ground node (id 0) is present,
        # 1-based otherwise; ``_parse_nodes`` already validated that
        # ``min(node_ids) in {0, 1}`` above.
        node_index_offset = nodes_list[0].index

        individual_branches = [
            branch for branch in parsed_branches if branch[0] != "ML"
        ]
        for parsed_branch in individual_branches:
            branch, sym_params = make_branch(
                nodes_list, *parsed_branch, node_index_offset=node_index_offset
            )
            cls._merge_branch_symbols(branch_var_dict, sym_params)
            branches_list.append(branch)

        coupler_branches = [
            branch for branch in parsed_branches if branch not in individual_branches
        ]
        for parsed_branch in coupler_branches:
            coupler, sym_params = make_coupler(branches_list, *parsed_branch)
            cls._merge_branch_symbols(branch_var_dict, sym_params)
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

    @classmethod
    def _max_connected_branch_subgraphs(
        cls, branch_subset: list[Branch]
    ) -> list[list[Branch]]:
        """Partition ``branch_subset`` into maximum connected subgraphs.

        Two branches belong to the same subgraph if they share a node,
        transitively (the relation generated by :meth:`Branch.is_connected`).
        Used by :meth:`_independent_modes` to identify the
        no-flux-difference subspaces.
        """
        remaining = branch_subset.copy()
        subgraphs: list[list[Branch]] = []
        while remaining:
            seed = remaining.pop(0)
            subgraph = [seed]
            while not cls.are_branchsets_disconnected(subgraph, remaining):
                for candidate in remaining:
                    for member in subgraph:
                        if candidate.is_connected(member):
                            subgraph.append(candidate)
                            remaining.remove(candidate)
                            break
            subgraphs.append(subgraph)
        return subgraphs

    @staticmethod
    def _compute_subgraph_membership(
        nodes_in_max_connected_branchsets: list[list[Node]],
        all_nodes: list[Node],
    ) -> list[int]:
        """Return the per-node subgraph index (parallel to ``all_nodes``).

        Pure replacement for the ``Node.marker`` mutation pattern: returns a
        list ``markers`` such that ``markers[i]`` is

        * ``-1`` if ``all_nodes[i]`` is the ground node, or belongs to a
          subgraph that contains a ground node;
        * the 1-based subgraph index if the node belongs to a non-grounded
          subgraph;
        * ``0`` if the node belongs to no subgraph in
          ``nodes_in_max_connected_branchsets``.

        No ``Node`` object is mutated.
        """
        node_to_marker: dict[Node, int] = {n: 0 for n in all_nodes}
        for subgraph_idx, node_set in enumerate(nodes_in_max_connected_branchsets):
            grounded = any(n.is_ground() for n in node_set)
            label = -1 if grounded else subgraph_idx + 1
            for node in node_set:
                node_to_marker[node] = label
        for node in all_nodes:
            if node.is_ground():
                node_to_marker[node] = -1
        return [node_to_marker[n] for n in all_nodes]

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

        # Order the nodes with the ground node (if any) at the end.
        # This is the column ordering of the returned basis vectors;
        # downstream slicing at L1483 (``basis = [i[:-1] for i in basis]``)
        # drops the trailing column on the assumption that ground sits there.
        all_nodes = list(self.nodes)
        if self.ground_node and self.ground_node in all_nodes:
            all_nodes.remove(self.ground_node)
            all_nodes.append(self.ground_node)
        elif self.ground_node:
            all_nodes.append(self.ground_node)

        # step 2: partition branch_subset into max-connected subgraphs and
        # collect the node set for each
        max_connected_subgraphs = self._max_connected_branch_subgraphs(branch_subset)
        nodes_in_max_connected_branchsets = [
            unique_elements_in_list(
                list(chain(*[branch.nodes for branch in branch_set]))
            )
            for branch_set in max_connected_subgraphs
        ]

        # Compute subgraph membership directly (pure return; no mutation
        # of ``Node.marker`` on ``self.nodes``).  Indices are parallel
        # to ``all_nodes`` and follow the convention:
        #   ``-1``  : node belongs to a subgraph containing the ground
        #             node (i.e. is connected to ground via the chosen
        #             branch subset);
        #   ``0``   : node is not in any of the max-connected subgraphs;
        #   ``k>0`` : node belongs to subgraph index ``k - 1``.
        node_branch_set_indices = self._compute_subgraph_membership(
            nodes_in_max_connected_branchsets, all_nodes
        )

        # step 3: build a basis vector per non-grounded subgraph
        basis = self._subgraph_basis_vectors(node_branch_set_indices, basisvec_entries)

        # step 3b: optionally extend with single-node modes (one per
        # node not in any subgraph), keeping only those that strictly
        # increase the rank
        if single_nodes:
            for mode in self._single_node_basis_candidates(
                node_branch_set_indices, basisvec_entries
            ):
                if self._mode_strictly_increases_rank(basis, mode):
                    basis.append(mode)

        if self.is_grounded:
            # drop the trailing column corresponding to the ground node
            basis = [vec[:-1] for vec in basis]

        return basis

    @staticmethod
    def _subgraph_basis_vectors(
        node_branch_set_indices: list[int], basisvec_entries: list[int]
    ) -> list[list[int]]:
        """Build one basis vector per non-grounded max-connected subgraph.

        Each subgraph corresponds to a "no-flux-difference" mode: every
        node in the subgraph has the same generalized flux, so the
        characteristic vector marks those nodes with ``basisvec_entries[0]``
        (default ``1``) and all other nodes with ``basisvec_entries[1]``
        (default ``0``).  Subgraphs whose marker is ``-1`` (touching the
        ground node) are skipped because they are pinned to ground and
        contribute no degree of freedom.
        """
        unique_markers = unique_elements_in_list(node_branch_set_indices)
        return [
            [
                basisvec_entries[0] if marker == k else basisvec_entries[1]
                for marker in node_branch_set_indices
            ]
            for k in unique_markers
            if k != -1
        ]

    @staticmethod
    def _single_node_basis_candidates(
        node_branch_set_indices: list[int], basisvec_entries: list[int]
    ) -> list[list[int]]:
        """Yield candidate basis vectors for nodes outside every subgraph.

        Nodes with marker ``0`` belong to no max-connected branch subgraph
        (they sit on the boundary of capacitive islands or have no
        relevant branch).  Each such node contributes a one-hot
        candidate basis vector that the caller can adopt if it
        strictly increases the rank of the existing basis.  Returns
        an empty list when every node is already in some subgraph.
        """
        if 0 not in node_branch_set_indices:
            return []
        unmarked_positions = [
            i for i, marker in enumerate(node_branch_set_indices) if marker == 0
        ]
        return [
            [
                basisvec_entries[0] if i == pos else basisvec_entries[1]
                for i in range(len(node_branch_set_indices))
            ]
            for pos in unmarked_positions
        ]

    @staticmethod
    def _mode_strictly_increases_rank(basis: list[list[int]], mode: list[int]) -> bool:
        """Return ``True`` iff ``mode`` is linearly independent of ``basis``.

        Used by :meth:`_independent_modes` to decide whether a candidate
        single-node mode adds a new degree of freedom or is already in the
        span of the existing basis vectors.  Equivalent to the rank check
        ``rank(basis ∪ {mode}) == len(basis) + 1``.
        """
        mat = np.array(basis + [mode])
        return np.linalg.matrix_rank(mat) == len(mat)

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

    def _canonical_modes_periodic_frozen_free_with_sigma(
        self,
    ) -> tuple[list, list, list, list[int]]:
        """Compute periodic / frozen / free mode bases plus the Σ vector.

        Shared prelude for :meth:`check_transformation_matrix` and
        :meth:`variable_transformation_matrix`.  The returned
        ``frozen_modes`` already includes Σ when the circuit is
        non-grounded (with the leading frozen vector dropped if Σ is
        not linearly independent of the existing frozen basis).

        Returns
        -------
        ``(periodic_modes, frozen_modes_with_sigma, free_modes, Σ)``
        """
        periodic_modes = self._independent_modes(
            [branch for branch in self.branches if branch.type == "L"]
        )
        frozen_modes = self._independent_modes(
            [branch for branch in self.branches if branch.type != "L"],
            single_nodes=True,
        )
        free_modes = self._independent_modes(
            [branch for branch in self.branches if branch.type != "C"]
        )

        Σ = [1] * (len(self.nodes) - self.is_grounded)
        if not self.is_grounded:
            mat = np.array(frozen_modes + [Σ])
            if np.linalg.matrix_rank(mat) < len(frozen_modes) + 1:
                frozen_modes = frozen_modes[1:] + [Σ]
            else:
                frozen_modes.append(Σ)

        return periodic_modes, frozen_modes, free_modes, Σ

    def _classify_modes_into_categories(
        self,
        modes,
        Σ: list[int],
        frozen_modes: list,
        free_modes: list,
        periodic_modes: list,
        *,
        track_sigma: bool,
    ) -> dict[Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]]:
        """Classify modes into ``periodic`` / ``extended`` / ``free`` / ``frozen``
        (and optionally ``sigma``).

        Each mode is checked against the canonical subspaces in
        precedence order ``Σ → frozen → free → periodic → extended``.
        When ``track_sigma=True`` and the circuit is non-grounded, a
        mode lying in the Σ subspace is recorded in the ``"sigma"``
        bucket; otherwise it is silently skipped (matching the original
        behaviour for the auto-generated basis vs. the user-provided
        transformation respectively).

        Returns indices are 1-based to mirror the legacy convention.
        """
        categories: dict[
            Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]
        ] = {"periodic": [], "extended": [], "free": [], "frozen": []}
        if track_sigma:
            categories["sigma"] = []

        for x, mode in enumerate(modes):
            if self._mode_in_subspace(Σ, [mode]) and not self.is_grounded:
                if track_sigma:
                    categories["sigma"].append(x + 1)
                continue
            if self._mode_in_subspace(mode, frozen_modes):
                categories["frozen"].append(x + 1)
            elif self._mode_in_subspace(mode, free_modes):
                categories["free"].append(x + 1)
            elif self._mode_in_subspace(mode, periodic_modes):
                categories["periodic"].append(x + 1)
            else:
                categories["extended"].append(x + 1)
        return categories

    @staticmethod
    def _warn_unmatched_mode_counts(
        circuit_cats: dict[
            Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]
        ],
        user_cats: dict[
            Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]
        ],
        enable_warnings: bool,
    ) -> None:
        """Warn when the user-supplied transformation has fewer modes of any kind
        than the canonical (auto-generated) basis."""
        for mode_type in ("periodic", "extended", "free", "frozen"):
            num_extra_modes = len(circuit_cats[mode_type]) - len(user_cats[mode_type])
            if num_extra_modes > 0 and enable_warnings:
                warnings.warn(
                    "Number of extra "
                    + mode_type
                    + " modes found: "
                    + str(num_extra_modes)
                    + "\n"
                )

    def check_transformation_matrix(
        self, transformation_matrix: ndarray, enable_warnings: bool = True
    ) -> dict[Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]]:
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
        if np.linalg.det(transformation_matrix) == 0:
            raise ValueError("The transformation matrix provided is not invertible.")

        periodic_modes, frozen_modes, free_modes, Σ = (
            self._canonical_modes_periodic_frozen_free_with_sigma()
        )

        # LC modes use the default basisvec_entries=[1, 0] here (the
        # variable_transformation_matrix variant uses [-1, 1] instead).
        LC_modes = self._independent_modes(
            [branch for branch in self.branches if "JJ" in branch.type],
            single_nodes=False,
        )

        # Build the canonical mode basis by adding (in this order)
        # frozen, free, periodic, then LC modes that are not yet in the
        # span. The double-pass over LC_modes mirrors the original code.
        modes: list[ndarray] = []
        for m in frozen_modes + free_modes + periodic_modes + LC_modes:
            if not self._mode_in_subspace(m, modes):
                modes.append(m)
        for m in LC_modes:
            if not self._mode_in_subspace(m, modes):
                modes.append(m)

        var_categories_circuit = self._classify_modes_into_categories(
            modes,
            Σ,
            frozen_modes,
            free_modes,
            periodic_modes,
            track_sigma=False,
        )

        user_given_modes = transformation_matrix.transpose()
        var_categories_user = self._classify_modes_into_categories(
            user_given_modes,
            Σ,
            frozen_modes,
            free_modes,
            periodic_modes,
            track_sigma=True,
        )
        sigma_mode_found = bool(var_categories_user["sigma"])

        self._warn_unmatched_mode_counts(
            var_categories_circuit, var_categories_user, enable_warnings
        )
        if not self.is_grounded and not sigma_mode_found:
            raise ValueError(
                "This circuit is not grounded, and so has a sigma mode. This transformation does not have a sigma mode."
            )

        return var_categories_user

    def _complete_basis_with_standard_vectors(
        self, modes: list[ndarray]
    ) -> list[ndarray]:
        """Extend ``modes`` to a full basis using a standard-basis completion.

        Adds vectors from a "standard basis" until the resulting matrix has
        rank ``len(self.nodes) - self.is_grounded``.  The standard basis
        depends on :attr:`basis_completion`:

        * ``"heuristic"`` (default): start from the all-ones vector and
          greedily fill from permutations of an "almost-ones" reference
          vector, accepting each candidate that increases the rank.
        * ``"canonical"``: use the identity matrix.

        Returns the completed basis as a list of ndarray rows; conversion
        to a 2-D ndarray is left to the caller.
        """
        standard_basis: ndarray
        if self.basis_completion == "heuristic":
            node_count = len(self.nodes) - self.is_grounded
            heuristic_basis: list = [np.ones(node_count)]

            # Heuristic basis: candidate vectors are ``n``-tuples of 0
            # and 1 with a fixed number of zeros (2 for ``n > 2``, 1
            # otherwise).  ``reversed(combinations(range(n),
            # zeros_count))`` emits zero-position tuples in
            # lex-decreasing order, which matches the first-occurrence
            # order of the candidates each accepted by rank check.
            zeros_count = 2 if node_count > 2 else 1
            zero_positions_in_legacy_order = reversed(
                list(itertools.combinations(range(node_count), zeros_count))
            )
            for zero_positions in zero_positions_in_legacy_order:
                if len(heuristic_basis) == node_count:
                    break
                zero_set = set(zero_positions)
                candidate = [0 if i in zero_set else 1 for i in range(node_count)]
                mat = np.array(heuristic_basis + [candidate])
                if np.linalg.matrix_rank(mat) == len(mat):
                    heuristic_basis.append(candidate)
            standard_basis = np.array(heuristic_basis)
        elif self.basis_completion == "canonical":
            standard_basis = np.identity(len(self.nodes) - self.is_grounded)
        else:
            raise ValueError(
                f"Unknown basis_completion {self.basis_completion!r}; "
                f"expected 'heuristic' or 'canonical'."
            )

        new_basis = modes.copy()
        for m in standard_basis:
            mat = np.array([i for i in new_basis] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                new_basis.append(m)
        return new_basis

    @staticmethod
    def _build_var_categories_from_positions(
        pos_list: list[int],
        pos_periodic: list[int],
        pos_rest: list[int],
        pos_free: list[int],
        pos_frozen: list[int],
        pos_Σ: list[int],
        is_grounded: bool,
    ) -> dict[Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]]:
        """Translate the ``pos_*`` index buckets into the ``var_categories`` dict.

        Each output index is the 1-based position of the variable inside
        ``pos_list`` (the row order of the transformation matrix).  The
        ``"sigma"`` bucket is non-empty only for non-grounded circuits, in
        which case it contains the single position whose ``pos_list``
        entry equals ``pos_Σ[0]``.
        """
        return {
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
                if not is_grounded
                else []
            ),
        }

    def variable_transformation_matrix(
        self,
    ) -> tuple[
        ndarray,
        dict[Literal["periodic", "extended", "free", "frozen", "sigma"], list[int]],
    ]:
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

        periodic_modes, frozen_modes, free_modes, Σ = (
            self._canonical_modes_periodic_frozen_free_with_sigma()
        )

        # LC modes use basisvec_entries=[-1, 1] here (the
        # check_transformation_matrix variant uses the default [1, 0]).
        LC_modes = self._independent_modes(
            [branch for branch in self.branches if "JJ" in branch.type],
            single_nodes=False,
            basisvec_entries=[-1, 1],
        )

        # **************** Adding frozen, free, periodic , LC and extended modes ****
        modes: list[ndarray] = []  # starting with an empty list

        for m in (
            frozen_modes + free_modes + periodic_modes + LC_modes  # + extended_modes
        ):  # This order is important
            mat = np.array(modes + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                modes.append(m)

        # step 4: extend the modes list to a full basis using the standard
        # basis chosen by self.basis_completion ("heuristic" or "canonical")
        new_basis = self._complete_basis_with_standard_vectors(modes)
        new_basis_arr = np.array(new_basis)

        # Classify each basis row in a single pass via a precomputed lookup,
        # replacing 5 chained O(n^2) comprehensions with O(n).
        # Precedence (highest priority first): sigma > free > periodic > frozen
        # > rest. The dict is built in reverse precedence so later entries
        # overwrite earlier ones.
        mode_to_label: dict[tuple, str] = {}
        for m in frozen_modes:
            mode_to_label[tuple(m)] = "frozen"
        for m in periodic_modes:
            mode_to_label[tuple(m)] = "periodic"
        for m in free_modes:
            mode_to_label[tuple(m)] = "free"
        if not self.is_grounded:
            mode_to_label[tuple(Σ)] = "sigma"

        buckets: dict[str, list[int]] = {
            "sigma": [],
            "free": [],
            "periodic": [],
            "frozen": [],
            "rest": [],
        }
        for i, row in enumerate(new_basis_arr):
            buckets[mode_to_label.get(tuple(row.tolist()), "rest")].append(i)

        pos_Σ = buckets["sigma"]
        pos_free = buckets["free"]
        pos_periodic = buckets["periodic"]
        pos_frozen = buckets["frozen"]
        pos_rest = buckets["rest"]
        pos_list = pos_periodic + pos_rest + pos_free + pos_frozen + pos_Σ
        # transforming the new_basis matrix into the order
        # (periodic, extended, free, frozen, sigma)
        transformation_matrix = new_basis_arr[pos_list].T

        var_categories = self._build_var_categories_from_positions(
            pos_list,
            pos_periodic,
            pos_rest,
            pos_free,
            pos_frozen,
            pos_Σ,
            self.is_grounded,
        )

        return np.array(transformation_matrix), var_categories
