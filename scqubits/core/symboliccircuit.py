# circuit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import itertools
import copy

import sympy
import numpy as np
from numpy import ndarray
from sympy import symbols

from scqubits.utils.misc import is_float_string


def process_word(word: str) -> Union[float, symbols]:
    if is_float_string(word):
        return float(word)
    return symbols(word)


def are_disconnected(branch_list1: List[Branch], branch_list2: List[Branch]) -> bool:
    """
    Determines whether two sets of branches are disconnected
    """
    node_array1 = np.array([branch.node_ids() for branch in branch_list1]).flatten()
    node_array2 = np.array([branch.node_ids() for branch in branch_list2]).flatten()
    return np.intersect1d(node_array1, node_array2).size == 0


class Node:
    """
    Class to represent a Node in a circuit handled by Circuit. The attribute
    `<Node>.branches` is a list of Branch objects containing all branches connected
    to the node.
    Parameters
    ----------
    id:
        integer identifier of the node
    marker:
        TODO: this description is not useful
        marker used to identify the Node with respect to other nodes in any situation,
        can be set as 0 if not used.
    """

    # TODO: fix type of marker
    def __init__(self, id: int, marker: Any):
        self.id = id
        self.marker = marker
        self.branches: List[Branch] = []

    def __str__(self) -> str:
        return "Node {}".format(self.id)

    def __repr__(self) -> str:
        return "Node({})".format(self.id)

    def connected_nodes(self, branch_type: str) -> List["Node"]:
        """
        Returns a list of all nodes directly connected by branches to the current
        node, either considering all branches or a specified branch type
        branch_type:
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
            if branch.nodes[0].id == self.id:
                result.append(branch.nodes[1])
            else:
                result.append(branch.nodes[0])
        return result


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
        dictionary of parameters for the branch, namely for
        capacitance: {"EC":  <value>};
        for inductance: {"EL": <value>};
        for Josephson Junction: {"EJ": <value>, "ECJ": <value>}
    Example: `Branch("C", Node(1, 0), Node(2, 0))`
    is a capacitive branch connecting the nodes with indices 0 and 1.
    """

    def __init__(
        self,
        n_i: Node,
        n_f: Node,
        branch_type: str,
        parameters: Optional[Dict[str, float]] = None,
    ):
        self.nodes = (n_i, n_f)
        self.type = branch_type
        self.parameters = parameters
        # store info of current branch inside the provided nodes
        # TODO: discuss - this is an unexpected side effect
        self.nodes[0].branches.append(self)
        self.nodes[1].branches.append(self)

    def __str__(self) -> str:
        return (
            "Branch "
            + self.type
            + " connecting nodes: ("
            + str(self.nodes[0].id)
            + ","
            + str(self.nodes[1].id)
            + "); "
            + str(self.parameters)
        )

    def __repr__(self) -> str:
        return "Branch({}, {}, {})".format(
            self.type, self.nodes[0].id, self.nodes[1].id
        )

    def set_parameters(self, parameters: ndarray) -> None:
        if self.type in ["C", "L"]:
            self.parameters = {self.type: parameters[0]}
        elif self.type in ["JJ", "JJ2"]:
            self.parameters = {"EJ": parameters[0], "ECJ": parameters[1]}

    def node_ids(self) -> Tuple[int, int]:
        return (self.nodes[0].id, self.nodes[1].id)

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

class SymbolicCircuit(serializers.Serializable):
    r"""
    Describes a circuit consisting of nodes and branches.
    Can be initialized using an input file. Examples: for a Transmon qubit the
    input file would read:
    # file_name: transmon_num.inp
        nodes: 2
        branches:
        C	1,2	1
        JJ	1,2	1	10
    Circuit object can be initiated using:
        Circuit.from_input_file("transmon_num.inp", mode="num")
    # TODO entries incomplete
    Parameters
    ----------
    nodes_list:
        List of nodes in the circuit
    branches_list:
        List of branches connecting the above set of nodes.
    mode:
        "num" or "sym" switches between numeric and symbolic representation of
        parameters in the input file.
    """

    def __init__(
        self,
        nodes_list: List[Node],
        branches_list: List[Branch],
        mode: str = "sym",
        basis: str = "simple",
        ground_node=None,
        initiate_sym_calc: bool = True,
    ):
        self.branches = branches_list
        self.nodes = nodes_list
        self.mode = mode
        self.input_string = None

        # TODO the following line is unexpected, needs explanation if correct
        self._init_params = ["input_string"]  # for saving the init data
        self._sys_type = type(self).__name__  # for object description

        # TODO all the following attributes require type annotations
        # attributes set by methods
        self.trans_mat = None

        self.var_indices = None
        self.external_flux_vars = []
        self.closure_branches = []

        self.param_vars = []

        self.hamiltonian = None
        self._lagrangian = None  # to store the internally used lagrangian
        self.lagrangian = None
        # TODO: naming -- "old" is not informative
        self.lagrangian_old = None  # symbolic lagrangian in terms of untransformed generalized flux variables
        self.potential = None  # symbolic expression for potential energy

        # parameters for grounding the circuit
        self.ground_node = ground_node
        self.is_grounded = bool(self.ground_node)

        # TODO comments in the following two lines are not helpful
        # paramater for chosing the basis
        self.basis = basis  # default, the other choice is standard

        self.initiate_sym_calc = initiate_sym_calc

        # Calling the function to initiate the calss variables
        if initiate_sym_calc:
            self.initiate_symboliccircuit()

    def initiate_symboliccircuit(self, transformation_matrix=None):
        """
        Method to initialize the CustomQCircuit instance and initialize all the attributes needed before it can be passed on to AnalyzeQCircuit.

        Parameters
        ----------
        transformation_matrix:
            Takes an ndarray and is used to set an alternative transformation matrix than the one generated by the method variable_transformation_matrix.
        """
        # calculate the transformation matrix and identify the boundary conditions
        self.trans_mat, self.var_indices = self.variable_transformation_matrix()

        # find the closure branches in the circuit
        self.closure_branches = self._closure_branches()
        # setting external flux and offset charge variables
        self._set_external_flux_vars()
        self._set_offset_charge_vars()
        # setting the branch parameter variables
        self._set_param_vars()
        # Calculate the Lagrangian
        self._lagrangian, self.potential, self.lagrangian_old = self.lagrangian_sym(
            transformation_matrix=transformation_matrix
        )

        # replacing energies with capacitances in the kinetic energy of the Lagrangian
        self.lagrangian, self.lagrangian_old = self.replace_energies_with_capacitances_L()

        # calculating the Hamiltonian
        self.hamiltonian = self.hamiltonian_sym(transformation_matrix=transformation_matrix)

    def replace_energies_with_capacitances_L(self):
        """
        Method replaces the energies in the Lagrangian with capacitances which are arbitrarily generated to make sure that the Lagrangian looks dimensionally correct.
        """
        # Replacing energies with capacitances if the circuit mode is symbolic
        L = self._lagrangian.expand()
        L_old = self.lagrangian_old
        if self.mode == "sym":
            # finding the unique capacitances
            uniq_capacitances = []
            element_param = {"C": "EC", "JJ": "ECJ", "JJ2": "ECJ"}
            for c, b in enumerate(
                [
                    t
                    for t in self.branches
                    if t.type == "C" or t.type == "JJ" or t.type == "JJ2"
                ]
            ):
                if len(set(b.nodes)) > 1:  # check to see if branch is shorted
                    if b.parameters[element_param[b.type]] not in uniq_capacitances:
                        uniq_capacitances.append(b.parameters[element_param[b.type]])

            for index, var in enumerate(uniq_capacitances):
                L = L.subs(var, 1 / (8 * symbols("C" + str(index + 1))))
                L_old = L_old.subs(var, 1 / (8 * symbols("C" + str(index + 1))))
        return L, L_old

    # TODO: what's going on here?
    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

        return {}

    @staticmethod
    def are_branchsets_disconnected(l1: List[branch], l2: List[branch]) -> bool:
        """
        function to determine if two sets of branches are disconnected
        """
        for b1 in l1:
            for b2 in l2:
                if b1.is_connected(b2):
                    return False
        return True

        @staticmethod
    def parse_nodes(code_lines: List[str]) -> List[Node]:
        node_count = int(code_lines[0].split(":")[-1])
        return [Node(id, 0) for id in range(1, node_count + 1)]

    @staticmethod
    def parse_branches(
        code_lines: List[str], nodes: List[Node]
    ) -> Tuple[List[Branch], Node]:
        start_index = code_lines.index("branches:") + 1
        end_index = len(code_lines)

        node_count = len(nodes)
        is_grounded = False
        ground_node = None

        branches = []
        for line_index in range(start_index, end_index):
            if code_lines[line_index] == "":
                break

            current_code_line = code_lines[line_index].replace("\t", " ")
            words = [word for word in current_code_line.split(" ") if word != ""]
            # TODO: naming -- completely unclear what n1,n2 represent
            node_id1, node_id2 = [int(num) for num in words[1].split(",")]

            if node_id1 * node_id2 == 0 and not is_grounded:
                # TODO: following explanation is unhelpful
                # make a ground node in case Node zero is used for any Branch in the
                # input file
                node_count += 1  # TODO: explain why?
                ground_node = Node(0, 0)
                is_grounded = True

            branch_type = words[0]

            if branch_type in ["JJ", "JJ2"]:
                if len(words) <= 3:
                    raise Exception("Cannot parse input: too few parameters for JJ.")
                parameter1 = process_word(words[2])
                parameter2 = process_word(words[3])
                parameters = {branch_type: [parameter1, parameter2]}
            else:
                if len(words) <= 2:
                    raise Exception("Cannot parse input: too few parameters for C/L.")
                parameters = {branch_type: self.process_word(words[2])}

            if node_id1 == 0:
                branches.append(
                    Branch(ground_node, nodes[node_id2 - 1], branch_type, parameters)
                )
            elif node_id2 == 0:
                branches.append(
                    Branch(nodes[node_id1 - 1], ground_node, branch_type, parameters)
                )
            else:
                branches.append(
                    Branch(
                        nodes[node_id1 - 1],
                        nodes[node_id2 - 1],
                        branch_type,
                        parameters,
                    )
                )
        return branches, ground_node

        @classmethod
    def from_input_string(
        cls,
        input_string: str,
        mode: str = "sym",
        basis: str = "simple",
        initiate_sym_calc: bool = True,
    ) -> None:
        """
        Constructs the instance of Circuit from an input string.
        TODO: info incomplete
        Parameters
        ----------
        input_string:
            String describing the number of nodes and branches connecting then along with their parameters
        """
        code_lines = input_string.split("\n")

        nodes = cls.parse_nodes(code_lines)
        branches, ground_node = cls.parse_branches(code_lines, nodes)

        circuit = cls(
            nodes,
            branches,
            ground_node=ground_node,
            mode=mode,
            basis=basis,
            initiate_sym_calc=initiate_sym_calc,
        )
        circuit.input_string = input_string
        return circuit
    @classmethod
    def from_input_file(
        cls,
        filename: str,
        mode: str = "sym",
        basis="simple",
        initiate_sym_calc=True,
    ):
        """
        Constructs the instance of Circuit from an input string.
        Parameters
        ----------
        filename:
            name of the file containing the text describing the number of nodes and branches connecting then along with their parameters
        """
        file = open(filename, "r")
        input_string = file.read()
        file.close()
        return cls.from_input_string(
            input_string,
            mode=mode,
            basis=basis,
            initiate_sym_calc=initiate_sym_calc,
        )

    """
    Methods to find the cyclic variables of the circuit
    """

    def independent_modes(self, branch_subset: List[branch], single_nodes: bool = True):
        """
        Returns the vectors which span a subspace of /mathcal{G} where there is no generalized flux difference across the branches present in the branch_subset.
        Optional Variables:
        single_nodes: Boolean, if the single nodes are taken into consideration for basis vectors.
        """
        nodes_copy = self.nodes.copy()  # copying self.nodes as it is being modified

        if self.is_grounded:  # needed as gorund node is not included in self.nodes
            nodes_copy.append(self.ground_node)

        for node in nodes_copy:  # reset the node markers
            node.marker = 0

        # step 2: finding the maximum connected set of independent branches in branch_subset, then identifying the sets of nodes in each of thoses sets
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

        # using node.marker to mark the maximum connected subgraph to which a node belongs
        for node_set_index, node_set in enumerate(nodes_in_max_connected_branchsets):
            for node in node_set:
                node.marker = (
                    -1 if any([n.is_ground() for n in node_set]) else node_set_index + 1
                )

        node_branch_set_indices = [
            node.marker for node in nodes_copy
        ]  # identifies which node belongs to which maximum connected subgraphs; different numbers on two nodes indicates that they are not connected through any of the branches in branch_subset. 0 implies the node does not belong to any of the branches in max connected branch subsets and -1 implies the max connected branch set is connected to ground.

        # step 3: Finding the linearly independent vectors spanning the vector space represented by branch_set_index
        basis = []
        basisvec_entries = [1, 0]  # numbers used to make basis vectors

        num_branch_sets = max(node_branch_set_indices)

        range_branch_sets = [
            1 if min(node_branch_set_indices) <= 0 else 2,
            num_branch_sets + 1,
        ]

        for index in range(*range_branch_sets):
            basis.append([basisvec_entries[0] if i == index else basisvec_entries[1] for i in node_branch_set_indices])

        if single_nodes == True:  # taking the case where the node_branch_set_index is 0
            if node_branch_set_indices.count(0) > 0:
                ref_vector = [basisvec_entries[0] if i==0 else basisvec_entries[1] for i in node_branch_set_indices]
                positions = [index for index,num in enumerate(ref_vector) if num==basisvec_entries[0]]
                for pos in positions:
                    basis.append([basisvec_entries[0] if x==pos else basisvec_entries[1] for x,num in enumerate(node_branch_set_indices)])

        if self.is_grounded:  # if grounded remove the first row and column
            basis = [i[:-1] for i in basis]

        return basis

    def variable_transformation_matrix(self) -> ndarray:
        r"""
        Generates a transformation matrix which transforms the old to the new variables. Returns the variable identification along with the transformation matrix in the format: transformation_matrix, var_indices
        """

        ##################### Finding the Periodic Modes ##################
        selected_branches = [branch for branch in self.branches if branch.type == "L"]
        periodic_modes = self.independent_modes(selected_branches)

        ##################### Finding the frozen modes ##################
        selected_branches = [branch for branch in self.branches if branch.type != "L"]
        frozen_modes = self.independent_modes(selected_branches, single_nodes=True)

        ##################### Finding the Cyclic Modes ##################
        selected_branches = [branch for branch in self.branches if branch.type != "C"]
        cyclic_modes = self.independent_modes(selected_branches)
        #################### including the Σ mode #################
        Σ = [1 for n in self.nodes]
        if not self.is_grounded: # only append if the circuit is not grounded
            mat = np.array(frozen_modes + [Σ])
            if np.linalg.matrix_rank(mat) < len(frozen_modes) + 1: # check to see if the vectors are still independent
                frozen_modes = frozen_modes[1:] + [Σ]
            else:
                frozen_modes.append(Σ)


        ###################### Finding the LC Modes ##################
        selected_branches = [branch for branch in self.branches if branch.type == "JJ"]
        LC_modes = self.independent_modes(selected_branches, single_nodes=False)

        ################ Adding periodic and cyclic modes to frozen ones #############
        modes = frozen_modes.copy()  # starting with the frozen modes

        for m in (
            cyclic_modes + periodic_modes
        ):  # adding the ones which are periodic such that all vectors in modes are LI
            mat = np.array(modes + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                modes.append(m)

        for m in LC_modes:  # adding the LC modes to the basis
            mat = np.array(modes + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                modes.append(m)

        ####################### Completing the Basis ######################
        # step 4: construct the new set of basis vectors

        # constructing a standard basis
        l = len(self.nodes)
        standard_basis = [np.ones(l)]

        vector_ref = np.zeros(l)
        if l > 2:
            vector_ref[: l - 2] = 1
        else:
            vector_ref[: l - 1] = 1

        vector_set = list((itertools.permutations(vector_ref, l)))
        item = 0
        while np.linalg.matrix_rank(np.array(standard_basis)) < l:
            a = vector_set[item]
            item += 1
            mat = np.array(standard_basis + [a])
            if np.linalg.matrix_rank(mat) == len(mat):
                standard_basis = standard_basis + [list(a)]

        standard_basis = np.array(standard_basis)

        if self.basis == "standard":
            standard_basis = np.identity(len(self.nodes))

        new_basis = modes.copy()

        for m in standard_basis:  # completing the basis
            mat = np.array([i for i in new_basis] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                new_basis.append(m)
        new_basis = np.array(new_basis)

        # sorting the basis so that the cyclic, periodic and frozen variables occur at the beginning.
        if not self.is_grounded:
            pos_Σ = [i for i in range(len(new_basis)) if new_basis[i].tolist() == Σ]
        else:
            pos_Σ = []

        pos_cyclic = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if new_basis[i].tolist() in cyclic_modes
        ]
        pos_periodic = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_cyclic
            if new_basis[i].tolist() in periodic_modes
        ]
        pos_frozen = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_cyclic
            if i not in pos_periodic
            if new_basis[i].tolist() in frozen_modes
        ]
        pos_osc = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_cyclic
            if i not in pos_periodic
            if i not in pos_frozen
            if new_basis[i].tolist() in LC_modes
        ]
        pos_rest = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_cyclic
            if i not in pos_periodic
            if i not in pos_frozen
        ]
        pos_list = pos_periodic + pos_rest + pos_cyclic + pos_frozen + pos_Σ
        # transforming the new_basis matrix
        new_basis = new_basis[pos_list].T

        # Updating the class properties
        var_indices = {
            "periodic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_periodic
            ],
            "discretized_phi": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_rest
            ],
            "cyclic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_cyclic
            ],
            "frozen": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_frozen
            ],
            "osc": [i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_osc],
        }

        return np.array(new_basis), var_indices

    def _set_param_vars(self):
        """
        Sets the attribute param_vars.
        """
        # creating a class attribute for conserved charges corresponding to cyclic variables
        for c in self.var_indices["cyclic"]:
            setattr(self, "Qc" + str(c), 0)

        # set param_vars
        if self.mode == "sym":
            parameters = []  # showing three sublists, Ec's, El's ,Ej's and Ecj's
            for b in self.branches:
                if b.type == "JJ" or b.type == "JJ2":
                    parameters.append(b.parameters["EJ"])
                    parameters.append(b.parameters["ECJ"])
                elif b.type == "L":
                    parameters.append(b.parameters["EL"])
                elif b.type == "C":
                    parameters.append(b.parameters["EC"])
            parameters = [
                param
                for param in list(set(parameters))
                if type(param) == sympy.core.symbol.Symbol
            ]
        elif self.mode == "num":
            parameters = []

        for c in self.var_indices["cyclic"]:
            parameters.append(symbols("Qc" + str(c)))

        self.param_vars = parameters

    """
    Methods used to construct the Lagrangian of the circuit
    """

    def _junction_terms(self):
        terms = 0
        # looping over all the junction terms
        junction_branches = [branch for branch in self.branches if branch.type == "JJ"]
        for jj_branch in junction_branches:
            # adding external flux
            phi_ext = 0
            if jj_branch in self.closure_branches:
                index = self.closure_branches.index(jj_branch)
                phi_ext += self.external_flux_vars[index]

            if jj_branch.nodes[1].id == 0:  # if loop to check for the presence of ground node
                terms += -jj_branch.parameters["EJ"] * sympy.cos(
                    -symbols("φ" + str(jj_branch.nodes[0].id)) + phi_ext
                )
            elif jj_branch.nodes[0].id == 0:
                terms += -jj_branch.parameters["EJ"] * sympy.cos(
                    symbols("φ" + str(jj_branch.nodes[1].id)) + phi_ext
                )
            else:
                terms += -jj_branch.parameters["EJ"] * sympy.cos(
                    symbols("φ" + str(jj_branch.nodes[1].id))
                    - symbols("φ" + str(jj_branch.nodes[0].id))
                    + phi_ext
                )
        return terms

    def _JJ2_terms(self):
        terms = 0
        # looping over all the JJ2 branches
        for jj2_branch in [t for t in self.branches if t.type == "JJ2"]:
            # adding external flux
            phi_ext = 0
            if jj2_branch in self.closure_branches:
                index = self.closure_branches.index(jj2_branch)
                phi_ext += self.external_flux_vars[index]

            if jj2_branch.nodes[1].id == 0:  # if loop to check for the presence of ground node
                terms += -jj2_branch.parameters["EJ"] * sympy.cos(
                    2 * (-symbols("φ" + str(jj2_branch.nodes[0].id)) + phi_ext)
                )
            elif jj2_branch.nodes[0].id == 0:
                terms += -jj2_branch.parameters["EJ"] * sympy.cos(
                    2 * (symbols("φ" + str(jj2_branch.nodes[1].id)) + phi_ext)
                )
            else:
                terms += -jj2_branch.parameters["EJ"] * sympy.cos(
                    2
                    * (
                        symbols("φ" + str(jj2_branch.nodes[1].id))
                        - symbols("φ" + str(jj2_branch.nodes[0].id))
                        + phi_ext
                    )
                )
        return terms

    def _capacitance_matrix(self):
        branches_with_capacitance = [
            branch
            for branch in self.branches
            if branch.type == "C" or branch.type == "JJ" or branch.type == "JJ2"
        ]
        capacitance_param_for_branch_type = {"C": "EC", "JJ": "ECJ", "JJ2": "ECJ"}

        # filling the non-diagonal entries
        if not self.is_grounded:
            num_nodes = len(self.nodes)
            if self.mode == "num":
                C_mat = np.zeros([num_nodes, num_nodes])
            elif self.mode == "sym":
                C_mat = sympy.zeros(num_nodes)
            for branch in branches_with_capacitance:
                if len(set(branch.nodes)) > 1:  # branch if shorted is not considered
                    C_mat[branch.nodes[0].id - 1, branch.nodes[1].id - 1] += -1 / (
                        branch.parameters[
                            capacitance_param_for_branch_type[branch.type]
                        ]
                        * 8
                    )
        else:
            num_nodes = len(self.nodes) + 1
            if self.mode == "num":
                C_mat = np.zeros([num_nodes, num_nodes])
            elif self.mode == "sym":
                C_mat = sympy.zeros(num_nodes)
            for branch in branches_with_capacitance:
                if len(set(branch.nodes)) > 1:  # branch if shorted is not considered
                    C_mat[branch.nodes[0].id, branch.nodes[1].id] += -1 / (
                        branch.parameters[
                            capacitance_param_for_branch_type[branch.type]
                        ]
                        * 8
                    )

        if self.mode == "num":
            C_mat = C_mat + C_mat.T - np.diag(C_mat.diagonal())
        elif self.mode == "sym":
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
            if branch.type == "C" or branch.type == "JJ" or branch.type == "JJ2"
        ]
        for c_branch in branches_with_capacitance:
            element_param = {"C": "EC", "JJ": "ECJ", "JJ2": "ECJ"}

            if c_branch.nodes[1].id == 0:
                terms += (
                    1
                    / (16 * c_branch.parameters[element_param[c_branch.type]])
                    * (symbols("vφ" + str(c_branch.nodes[0].id))) ** 2
                )
            elif c_branch.nodes[0].id == 0:
                terms += (
                    1
                    / (16 * c_branch.parameters[element_param[c_branch.type]])
                    * (-symbols("vφ" + str(c_branch.nodes[1].id))) ** 2
                )
            else:
                terms += (
                    1
                    / (16 * c_branch.parameters[element_param[c_branch.type]])
                    * (
                        symbols("vφ" + str(c_branch.nodes[1].id))
                        - symbols("vφ" + str(c_branch.nodes[0].id))
                    )
                    ** 2
                )
        return terms

    def _inductor_terms(self):
        terms = 0
        for l_branch in [branch for branch in self.branches if branch.type == "L"]:
            # adding external flux
            phi_ext = 0
            if l_branch in self.closure_branches:
                index = self.closure_branches.index(l_branch)
                phi_ext += self.external_flux_vars[index]

            if l_branch.nodes[0].id == 0:
                terms += (
                    0.5
                    * l_branch.parameters["EL"]
                    * (symbols("φ" + str(l_branch.nodes[1].id)) + phi_ext) ** 2
                )
            elif l_branch.nodes[1].id == 0:
                terms += (
                    0.5
                    * l_branch.parameters["EL"]
                    * (-symbols("φ" + str(l_branch.nodes[0].id)) + phi_ext) ** 2
                )
            else:
                terms += (
                    0.5
                    * l_branch.parameters["EL"]
                    * (
                        symbols("φ" + str(l_branch.nodes[1].id))
                        - symbols("φ" + str(l_branch.nodes[0].id))
                        + phi_ext
                    )
                    ** 2
                )
        return terms

    def _spanning_tree(self):
        r"""
        returns a spanning tree for the given instance
        """

        # making a deep copy to make sure that the original instance is unaffected
        circ_copy = copy.deepcopy(self)

        ############# removing all the capacitive branches and updating the nodes ################
        # identifying capacitive branches
        for branch in [b for b in list(circ_copy.branches) if b.type == "C"]:
            for (
                node
            ) in (
                branch.nodes
            ):  # updating the branches attribute for each node that this branch connects
                node.branches = [b for b in node.branches if b is not branch]
            circ_copy.branches.remove(branch)  # removing the branch

        num_float_nodes = 1
        while num_float_nodes > 0:  # breaks when no floating nodes are detected
            num_float_nodes = 0  # setting
            for n in circ_copy.nodes:
                if len(n.branches) == 0:
                    circ_copy.nodes.remove(n)
                    num_float_nodes += 1
                    continue
                if len(n.branches) == 1:
                    b = n.branches[0]
                    circ_copy.branches.remove(b)
                    for n1 in b.nodes:
                        if n1 != n:
                            n1.branches = [i for i in n1.branches if i is not b]
                            num_float_nodes += 1
                            continue
                        else:
                            circ_copy.nodes.remove(n)

        if circ_copy.nodes == []:
            return []
        #######################################################################################

        ################### Constructing the node_sets ###############
        if circ_copy.is_grounded:
            node_sets = [[circ_copy.ground_node]]
        else:
            node_sets = [
                [circ_copy.nodes[0]]
            ]  # starting with the first set which has the first node as the only element

        num_nodes = len(circ_copy.nodes)
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

            if node_sets[node_set_index] == []:
                for n in circ_copy.nodes:
                    if n not in [q for p in node_sets for q in p]:
                        node_sets[node_set_index].append(n)

            for n in node_sets[node_set_index]:
                node_set += n.connected_nodes("all")

            node_set = [
                x
                for x in list(set(node_set))
                if x not in [q for p in node_sets[: node_set_index + 1] for q in p]
            ]
            if node_set != []:
                node_set.sort(key=lambda x: x.id)

            node_sets.append(node_set)
            node_set_index += 1
        #############################################################

        ########## constructing the spanning tree ##########
        tree_copy = []  # tree having branches of the instance that is copied

        def connecting_branches(n1, n2):
            return list(set(n1.branches).intersection(set(n2.branches)))

        # finding the branch which connects the node to another node in a previous node set.
        for index, node_set in enumerate(node_sets):
            if index == 0:
                continue
            for node in node_set:
                for prev_node in node_sets[index - 1]:
                    if len(connecting_branches(node, prev_node)) != 0:
                        tree_copy.append(connecting_branches(node, prev_node)[0])
                        break

        ############## selecting the appropriate branches from circ as from circ_copy #######
        def is_same_branch(b1, b2):
            d1 = b1.__dict__
            d2 = b2.__dict__
            if d1["type"] == d2["type"] and d1["parameters"] == d2["parameters"]:
                if [i.id for i in d1["nodes"]] == [i.id for i in d2["nodes"]]:
                    return True
                else:
                    return False
            else:
                False

        tree = []  # tree having branches of the current instance
        for branch in tree_copy:
            tree += [b for b in self.branches if is_same_branch(branch, b)]

        return tree

    def _closure_branches(self):
        r"""
        Returns and stores the closure branches in the circuit.
        """
        tree = self._spanning_tree()
        if tree == []:
            closure_branches = []
        else:
            closure_branches = list(set(self.branches) - set(tree))
        return closure_branches

    def _set_external_flux_vars(self):
        # setting the class properties
        closure_branches = [branch for branch in self._closure_branches() if branch.type!="C"]
        if len(closure_branches) > 0:
            self.closure_branches = closure_branches
            self.external_flux_vars = [
                symbols("Φ" + str(i + 1)) for i in range(len(closure_branches))
            ]

    def _set_offset_charge_vars(self):
        """
        Create the offset charge variables and store in class attribute offset_charge_vars
        """
        self.offset_charge_vars = []
        for p in self.var_indices[
            "periodic"
        ]:  # same as above for periodic variables and adding the offset charge variables
            self.offset_charge_vars = self.offset_charge_vars + [
                symbols("ng_" + str(p))
            ]

    def lagrangian_sym(self, transformation_matrix: ndarray = None):
        r"""
        Returns three symbolic expressions: L_θ, potential_θ, L_φ
        where θ represents the set of new variables and φ represents the set of old variables

        Parameters
        ----------
        transformation_matrix:
            None or an alternative transformation matrix to the one returned by the method variable_transformation_matrix
        """
        # if the user is not using any custom transformation_matrix
        if transformation_matrix is None:
            transformation_matrix = self.trans_mat.astype(int)

        # defining the φ variables
        φ_dot_vars = [symbols("vφ" + str(i)) for i in range(1, len(self.nodes) + 1)]

        # defining the θ variables
        θ_vars = [symbols("θ" + str(i)) for i in range(1, len(self.nodes) + 1)]
        # defining the θ dot variables
        θ_dot_vars = [symbols("vθ" + str(i)) for i in range(1, len(self.nodes) + 1)]
        # writing φ in terms of θ variables
        φ_vars_θ = (transformation_matrix).dot(θ_vars)
        # writing φ dot vars in terms of θ variables
        φ_dot_vars_θ = (transformation_matrix).dot(θ_dot_vars)

        # C_terms = self._C_terms()
        C_mat = self._capacitance_matrix()
        if self.mode == "num":
            C_terms_φ = ((C_mat).dot(φ_dot_vars)).dot(
                φ_dot_vars
            ) * 0.5  # interms of node variables
            C_terms_θ = ((C_mat).dot(φ_dot_vars_θ)).dot(
                φ_dot_vars_θ
            ) * 0.5  # interms of new variables
        elif self.mode == "sym":
            C_terms_φ = (sympy.Matrix(φ_dot_vars).T * C_mat * sympy.Matrix(φ_dot_vars))[
                0
            ] * 0.5  # interms of node variables
            C_terms_θ = (
                sympy.Matrix(φ_dot_vars_θ).T * C_mat * sympy.Matrix(φ_dot_vars_θ)
            )[
                0
            ] * 0.5  # interms of new variables

        L_terms_φ = self._inductor_terms()

        JJ_terms_φ = self._junction_terms() + self._JJ2_terms()

        L_φ = C_terms_φ - L_terms_φ - JJ_terms_φ

        potential_φ = L_terms_φ + JJ_terms_φ
        potential_θ = (
            potential_φ.copy()
        )  # copying the potential in terms of the old variables to make substitutions

        for i in range(len(self.nodes)):  # converting potential to new variables
            potential_θ = potential_θ.subs(symbols("φ" + str(i + 1)), φ_vars_θ[i])

        # eliminating the frozen variables
        for i in self.var_indices["frozen"]:
            sub = sympy.solve(
                potential_θ.diff(symbols("θ" + str(i))), symbols("θ" + str(i))
            )
            potential_θ = potential_θ.replace(symbols("θ" + str(i)), sub[0])

        L_θ = C_terms_θ - potential_θ

        return L_θ, potential_θ, L_φ

    def hamiltonian_sym(self, transformation_matrix: ndarray = None):
        r"""
        Returns the Hamiltonian of the circuit in terms of the new variables :math:`\theta_i`.

        Parameters
        ----------
        transformation_matrix:
            None or an alternative transformation matrix to the one returned by the method variable_transformation_matrix
        """

        if (
            transformation_matrix is None
        ):  # if user did not provide transformation_matrix
            transformation_matrix = self.trans_mat
        # basis_inv = np.linalg.inv(basis)[0 : N - n, 0 : N - n]

        # Excluding the frozen modes based on how they are organized in the method variable_transformation_matrix
        if self.is_grounded:
            num_frozen_modes = len(self.var_indices["frozen"])
        else:
            num_frozen_modes = len(self.var_indices["frozen"]) + 1
        num_nodes = len(self.nodes)

        # generating the C_mat_θ by inverting the capacitance matrix
        if self.mode == "sym":
            C_mat_θ = (
                transformation_matrix.T
                * self._capacitance_matrix()
                * transformation_matrix
            )[
                0 : num_nodes - num_frozen_modes, 0 : num_nodes - num_frozen_modes
            ].inv()  # exlcluding the frozen modes
        elif self.mode == "num":
            C_mat_θ = np.linalg.inv(
                (
                    transformation_matrix.T
                    @ self._capacitance_matrix()
                    @ transformation_matrix
                )[0 : num_nodes - num_frozen_modes, 0 : num_nodes - num_frozen_modes]
            )  # exlcluding the frozen modes

        p_θ_vars = [
            symbols("Q" + str(i))
            if i not in self.var_indices["cyclic"]
            else symbols("Qc" + str(i))
            for i in range(1, len(self.nodes) + 1 - num_frozen_modes)
        ]  # defining the momentum variables
        # p_φ_vars_θ = basis.dot(p_θ_vars) # writing φ in terms of θ variables

        # generating the kinetic energy terms for the Hamiltonian
        if self.mode == "num":
            C_terms_new = (
                C_mat_θ.dot(p_θ_vars).dot(p_θ_vars) * 0.5
            )  # interms of new variables
        elif self.mode == "sym":
            C_terms_new = (sympy.Matrix(p_θ_vars).T * C_mat_θ * sympy.Matrix(p_θ_vars))[
                0
            ] * 0.5  # interms of new variables

        H = C_terms_new + self.potential

        # adding the offset charge variables
        for p in self.var_indices["periodic"]:
            H = H.subs(
                symbols("Q" + str(p)), symbols("n" + str(p)) + symbols("ng_" + str(p))
            )

        return H
