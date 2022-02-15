# customqcircuit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import itertools
import copy

import sympy
import numpy as np
from numpy import ndarray
from sympy import symbols
from scipy import sparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix
from matplotlib import pyplot as plt
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean


import scqubits.core.discretization as discretization
import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot


class node:
    """
    Class to represent a node in a circuit handled by CustomQCircuit.

    id: identifier of the node represented by a number,
    marker: marker used to identify the node with respect to other nodes in any situation, can be set as 0 if not used.

    Properties:
    branches: list of branches connected to this node
    """

    def __init__(self, id: int, marker: any):
        self.id = id
        self.marker = marker
        self.branches = []

    def __str__(self):
        return "Node " + str(self.id)

    def __repr__(self):
        return "node({})".format(self.id)

    def connected_nodes(self, branch_type: str):
        """
        Returns the list of all the nodes where the branches connected to this node end

        branch_type: "C", "L" and "JJ"; type: str
        """
        result = []
        if branch_type != "all":
            for b in [x for x in self.branches if x.type == branch_type]:
                if b.nodes[0].id == self.id:
                    result.append(b.nodes[1])
                else:
                    result.append(b.nodes[0])
        else:
            for b in self.branches:
                if b.nodes[0].id == self.id:
                    result.append(b.nodes[1])
                else:
                    result.append(b.nodes[0])

        return result

    def is_ground(self):
        """
        Returns a bool if the node is a ground node. It is ground if the id is set to 0.
        """
        return True if self.id == 0 else False


class branch:
    """
    Class to describe a branch in a circuit, used by CustomQCircuit.

    n_i and n_f represent the initial and final node this branch connects;
    element is the type of this branch, example "C","JJ" or "L"
    parameters is the list of parameters for the branch element, which are the following
    capacitance: parameter "E_C"
    inductance: parameter "E_L"
    Josephson Junction: parameter "E_J", parameter "E_CJ"

    Example: branch("C", node(1, 0), node(2, 0))
    gives the branch connecting nodes with index 0 and 1.
    """

    def __init__(self, n_i: node, n_f: node, element: str, parameters=None):
        self.nodes = (n_i, n_f)
        self.type = element
        # setting the parameters if it is provided
        if parameters != None:
            if element == "C" or element == "L":
                self.parameters = {"E_" + element: parameters[0]}
            elif element == "JJ" or element == "JJ2":
                self.parameters = {"E_J": parameters[0], "E_CJ": parameters[1]}
        # updating the nodes
        self.nodes[0].branches.append(self)
        self.nodes[1].branches.append(self)

    def set_parameters(self, parameters: ndarray):
        if self.type == "C" or self.type == "L":
            self.parameters = {self.type: parameters[0]}
        elif self.type == "JJ" or self.type == "JJ2":
            self.parameters = {"E_J": parameters[0], "E_CJ": parameters[1]}

    def __str__(self):
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

    def __repr__(self):
        return "Branch({}, {}, {})".format(
            self.type, self.nodes[0].id, self.nodes[1].id
        )

    def is_connected(self, branch):
        """Returns a boolean if self is connected to another branch given as input"""
        distinct_nodes = len(set(self.nodes + branch.nodes))
        if distinct_nodes < 4:
            return True
        else:
            return False

    def common_node(self, branch):
        """Returns the common nodes between self and the branch given as input"""
        return set(self.nodes) & set(branch.nodes)


class CustomQCircuit(serializers.Serializable):
    r"""
    Class to describe a circuit using the classes node and branch.

    Can be initialized using an input file. For a Transmon qubit for example the following input file can be used.
    # file_name: transmon_num.inp
        nodes: 2
        branches:
        C	1,2	1
        JJ	1,2	1	10

    Circuit object can be initiated using:
        CustomQCircuit.from_input_file("transmon_num.inp", mode="num")

    A set of nodes with branches connecting them forms a circuit.
    Parameters
    ----------
    nodes_list:
        List of nodes in the circuit
    branches_list:
        List of branches connecting the above set of nodes.
    mode:
        "num" or "sym" correspondingly to numeric or symbolic representation of input parameters in the input file.
    """

    def __init__(
        self,
        nodes_list: list,
        branches_list: list,
        mode: str = "num",
        basis: str = "simple",
        ground_node=None,
        initiate_sym_calc: bool = True,
    ):

        self.branches = branches_list
        self.nodes = nodes_list
        self.mode = mode
        self.input_string: Optional[str] = None

        self._init_params = ["input_string"]  # for saving the init data
        self._sys_type = type(self).__name__  # for object description

        # properties set by methods
        self.trans_mat = None

        self.var_indices = None

        self.external_flux_vars = []
        self.closure_branches = []

        self.param_vars = []

        self.H = None
        self._L = (
            None  # to store the internally used Lagrangian passed over to Hamiltonian
        )
        self.L = (
            None  # to store the Human readable Lagrangian in terms of the new variables
        )
        self.L_old = None  # symbolic Lagrangian in terms of untransformed generalized flux variables
        self.potential = (
            None  # symbolic expression for the potential energy of the circuit
        )
        # parameters for grounding the circuit
        if ground_node != None:
            self.is_grounded = True
            self.ground_node = ground_node
        else:
            self.is_grounded = False
            self.ground_node = None

        # paramater for chosing the basis
        self.basis = basis  # default, the other choice is standard

        self.initiate_sym_calc = initiate_sym_calc

        # Calling the function to initiate the calss variables
        if initiate_sym_calc:
            self.initiate_customqcircuit()

    def initiate_customqcircuit(self, transformation_matrix=None):
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
        self._L, self.potential, self.L_old = self.lagrangian_sym(
            transformation_matrix=transformation_matrix
        )

        # replacing energies with capacitances in the kinetic energy of the Lagrangian
        self.L, self.L_old = self.replace_energies_with_capacitances_L()

        # calculating the Hamiltonian
        self.H = self.hamiltonian_sym(transformation_matrix=transformation_matrix)

    def replace_energies_with_capacitances_L(self):
        """
        Method replaces the energies in the Lagrangian with capacitances which are arbitrarily generated to make sure that the Lagrangian looks dimensionally correct.
        """
        # Replacing energies with capacitances if the circuit mode is symbolic
        L = self._L.expand()
        L_old = self.L_old
        if self.mode == "sym":
            # finding the unique capacitances
            uniq_capacitances = []
            element_param = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}
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

    @classmethod
    def from_input_string(
        cls,
        input_string: str,
        mode: str = "num",
        basis="simple",
        initiate_sym_calc=True,
    ):
        """
        Constructs the instance of CustomQCircuit from an input string.
        Parameters
        ----------
        input_string:
            String describing the number of nodes and branches connecting then along with their parameters

        """
        lines = (input_string).split("\n")
        num_nodes = int(lines[0].split(":")[-1])
        nodes = [node(i, 0) for i in range(1, num_nodes + 1)]
        branches = []

        first_branch = lines.index("branches:") + 1

        is_grounded = (
            False  # paramater to identify the presence of ground node in the circuit.
        )
        ground_node = None

        def is_str_float(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        for l in range(first_branch, len(lines)):

            if lines[l] == "":
                break

            words = [
                word for word in lines[l].replace("\t", " ").split(" ") if word != ""
            ]
            n1, n2 = [int(num) for num in words[1].split(",")]

            if (
                n1 * n2 == 0 and is_grounded == False
            ):  # Making a ground node in case node zero is used for any branch in the input file
                num_nodes += 1
                ground_node = node(0, 0)
                is_grounded = True

            element = words[0]

            if element == "JJ" or element == "JJ2":
                if (
                    len(words) > 3
                ):  # check to see if all the required parameters are defined
                    parameters = None

                if mode == "sym":
                    if is_str_float(words[2]):
                        p1 = float(words[2])
                    else:
                        p1 = symbols(words[2])
                    if is_str_float(words[3]):
                        p2 = float(words[3])
                    else:
                        p2 = symbols(words[3])

                    parameters = [p1, p2]
                elif mode == "num":
                    parameters = [float(words[2]), float(words[3])]

            else:
                if (
                    len(words) > 2
                ):  # check to see if all the required parameters are defined
                    parameters = None

                if mode == "sym":
                    if is_str_float(words[2]):
                        p = float(words[2])
                    else:
                        p = symbols(words[2])
                    parameters = [p]
                elif mode == "num":
                    parameters = [float(words[2])]

            if n1 == 0:
                branches.append(branch(ground_node, nodes[n2 - 1], element, parameters))
            elif n2 == 0:
                branches.append(branch(nodes[n1 - 1], ground_node, element, parameters))
            else:
                branches.append(
                    branch(nodes[n1 - 1], nodes[n2 - 1], element, parameters)
                )

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
        mode: str = "num",
        basis="simple",
        initiate_sym_calc=True,
    ):
        """
        Constructs the instance of CustomQCircuit from an input string.
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
        basis_numbers = [1, 0]  # numbers used to make basis vectors

        num_branch_sets = max(node_branch_set_indices)

        range_branch_sets = [
            1 if min(node_branch_set_indices) <= 0 else 2,
            num_branch_sets + 1,
        ]

        for index in range(*range_branch_sets):
            basis.append([basis_numbers[0] if i == index else basis_numbers[1] for i in node_branch_set_indices])

        if single_nodes == True:  # taking the case where the node_branch_set_index is 0
            if node_branch_set_indices.count(0) > 0:
                ref_vector = [basis_numbers[0] if i==0 else basis_numbers[1] for i in node_branch_set_indices]
                positions = [index for index,num in enumerate(ref_vector) if num==basis_numbers[0]]
                for pos in positions:
                    basis.append([basis_numbers[0] if x==pos else basis_numbers[1] for x,num in enumerate(node_branch_set_indices)])

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
                    parameters.append(b.parameters["E_J"])
                    parameters.append(b.parameters["E_CJ"])
                elif b.type == "L":
                    parameters.append(b.parameters["E_L"])
                elif b.type == "C":
                    parameters.append(b.parameters["E_C"])
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
        for b in [t for t in self.branches if t.type == "JJ"]:
            # adding external flux
            phi_ext = 0
            if b in self.closure_branches:
                index = self.closure_branches.index(b)
                phi_ext += self.external_flux_vars[index]

            if b.nodes[1].id == 0:  # if loop to check for the presence of ground node
                terms += -b.parameters["E_J"] * sympy.cos(
                    -symbols("φ" + str(b.nodes[0].id)) + phi_ext
                )
            elif b.nodes[0].id == 0:
                terms += -b.parameters["E_J"] * sympy.cos(
                    symbols("φ" + str(b.nodes[1].id)) + phi_ext
                )
            else:
                terms += -b.parameters["E_J"] * sympy.cos(
                    symbols("φ" + str(b.nodes[1].id))
                    - symbols("φ" + str(b.nodes[0].id))
                    + phi_ext
                )
        return terms

    def _JJ2_terms(self):
        terms = 0
        # looping over all the JJ2 branches
        for b in [t for t in self.branches if t.type == "JJ2"]:
            # adding external flux
            phi_ext = 0
            if b in self.closure_branches:
                index = self.closure_branches.index(b)
                phi_ext += self.external_flux_vars[index]

            if b.nodes[1].id == 0:  # if loop to check for the presence of ground node
                terms += -b.parameters["E_J"] * sympy.cos(
                    2 * (-symbols("φ" + str(b.nodes[0].id)) + phi_ext)
                )
            elif b.nodes[0].id == 0:
                terms += -b.parameters["E_J"] * sympy.cos(
                    2 * (symbols("φ" + str(b.nodes[1].id)) + phi_ext)
                )
            else:
                terms += -b.parameters["E_J"] * sympy.cos(
                    2
                    * (
                        symbols("φ" + str(b.nodes[1].id))
                        - symbols("φ" + str(b.nodes[0].id))
                        + phi_ext
                    )
                )
        return terms

    def _capacitance_matrix(self):
        branches_with_capacitance = [
            t
            for t in self.branches
            if t.type == "C" or t.type == "JJ" or t.type == "JJ2"
        ]
        capacitance_param_for_branch_type = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}

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
            t
            for t in self.branches
            if t.type == "C" or t.type == "JJ" or t.type == "JJ2"
        ]
        for b in branches_with_capacitance:
            element_param = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}

            if b.nodes[1].id == 0:
                terms += (
                    1
                    / (16 * b.parameters[element_param[b.type]])
                    * (symbols("vφ" + str(b.nodes[0].id))) ** 2
                )
            elif b.nodes[0].id == 0:
                terms += (
                    1
                    / (16 * b.parameters[element_param[b.type]])
                    * (-symbols("vφ" + str(b.nodes[1].id))) ** 2
                )
            else:
                terms += (
                    1
                    / (16 * b.parameters[element_param[b.type]])
                    * (
                        symbols("vφ" + str(b.nodes[1].id))
                        - symbols("vφ" + str(b.nodes[0].id))
                    )
                    ** 2
                )
        return terms

    def _inductor_terms(self):
        terms = 0
        for b in [t for t in self.branches if t.type == "L"]:
            # adding external flux
            phi_ext = 0
            if b in self.closure_branches:
                index = self.closure_branches.index(b)
                phi_ext += self.external_flux_vars[index]

            if b.nodes[0].id == 0:
                terms += (
                    0.5
                    * b.parameters["E_L"]
                    * (symbols("φ" + str(b.nodes[1].id)) + phi_ext) ** 2
                )
            elif b.nodes[1].id == 0:
                terms += (
                    0.5
                    * b.parameters["E_L"]
                    * (-symbols("φ" + str(b.nodes[0].id)) + phi_ext) ** 2
                )
            else:
                terms += (
                    0.5
                    * b.parameters["E_L"]
                    * (
                        symbols("φ" + str(b.nodes[1].id))
                        - symbols("φ" + str(b.nodes[0].id))
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
        where θ represents the set of new variables and φ representst the set of old variables

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
