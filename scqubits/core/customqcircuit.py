# transmon.py
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

    def __init__(self, i: int, marker: any):
        self.id = i
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
    """
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
    list_nodes:
        List of nodes in the circuit
    list_branches:
        List of branches connecting the above set of nodes.
    mode:
        String either "num" or "sym" correspondingly to numeric or symbolic representation of input parameters in the input file.
    """

    def __init__(
        self,
        list_nodes: list,
        list_branches: list,
        ground_node=None,
        mode: str = "sym",
        basis: str = "simple",
        initiate_sym_calc: bool = True
    ):

        self.branches = list_branches
        self.nodes = list_nodes
        self.mode = mode
        self.input_string = None

        self._init_params = ["input_string"]  # for saving the init data
        self._sys_type = type(self).__name__  # for object description

        # properties set by methods
        self.trans_mat = None

        self.var_indices = None

        self.external_flux_vars = []
        self.flux_branches = []

        self.param_vars = []

        self.H = None
        self._L = None  # to store the internally used Lagrangian
        self.L = None
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

        # Calling the function to initiate the calss variables
        if initiate_sym_calc:
            self.hamiltonian_sym()

    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

        return {}

    @staticmethod
    def is_disconnected(l1: List[branch], l2: List[branch]):
        """
        function to determine if two sets of branches are disconnected
        """
        for b1 in l1:
            for b2 in l2:
                if b1.is_connected(b2):
                    return False
        return True

    @classmethod
    def from_input_string(cls, input_string: str, mode: str = "sym", basis="simple", initiate_sym_calc=True):
        """
        Constructor of class CustomQCircuit:
        - Constructing the instance from an input string
        - mode parameter to specify the use of symbolic or numerical circuit parameters
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
        def isfloat(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        for l in range(first_branch, len(lines)):
            if lines[l] != "":
                line = [i for i in lines[l].replace("\t", " ").split(" ") if i != ""]
                a, b = [int(i) for i in line[1].split(",")]

                if (
                    a * b == 0 and is_grounded == False
                ):  # Making a ground node in case node zero is used for any branch in the input file
                    num_nodes += 1
                    ground_node = node(0, 0)
                    is_grounded = True

                element = line[0]

                if element == "JJ" or element == "JJ2":
                    if (
                        len(line) > 3
                    ):  # check to see if all the required parameters are defined
                        if mode == "sym":
                            if isfloat(line[2]):
                                p1 = float(line[2])
                            else:
                                p1 = symbols(line[2])
                            if isfloat(line[3]):
                                p2 = float(line[3])
                            else:
                                p2 = symbols(line[3])
                            
                            parameters = [p1, p2]
                        elif mode == "num":
                            parameters = [float(line[2]), float(line[3])]
                    else:
                        parameters = None
                else:
                    if (
                        len(line) > 2
                    ):  # check to see if all the required parameters are defined
                        if mode == "sym":
                            if isfloat(line[2]):
                                p = float(line[2])
                            else:
                                p = symbols(line[2])
                            parameters = [p]
                        elif mode == "num":
                            parameters = [float(line[2])]
                    else:
                        parameters = None
                if a == 0:
                    branches.append(
                        branch(ground_node, nodes[b - 1], element, parameters)
                    )
                elif b == 0:
                    branches.append(
                        branch(nodes[a - 1], ground_node, element, parameters)
                    )
                else:
                    branches.append(
                        branch(nodes[a - 1], nodes[b - 1], element, parameters)
                    )
            else:
                break

        circuit = cls(nodes, branches, ground_node=ground_node, mode=mode, basis=basis, initiate_sym_calc=initiate_sym_calc)
        circuit.input_string = input_string

        return circuit

    @classmethod
    def from_input_file(cls, filename: str, mode: str = "sym", basis="simple", initiate_sym_calc = True):
        """
        Constructor of class CustomQCircuit:
        - Constructing the instance from an input file
        - mode parameter to specify the use of symbolic or numerical circuit parameters
        """
        file = open(filename, "r")
        input_string = file.read()
        file.close()
        return cls.from_input_string(input_string, mode=mode, basis=basis, initiate_sym_calc=initiate_sym_calc)

    """
    Methods to find the cyclic variables of the circuit
    """

    def variable_transformation_matrix(self):
        """
        Method to construct a new set of flux variables of the circuit, such that all the possible cyclic and periodic variables are included.
        """
        

        def independent_modes(branch_subset, single_nodes=True):
            """
            Returns the vectors which span a subspace of /mathcal{G} where there is no generalized flux difference across the branches present in the branch_subset.
            Optional Variables:
            single_nodes: Boolean, if the single nodes are taken into consideration for basis vectors.
            """
            nodes = self.nodes.copy()
            if self.is_grounded:
                nodes.append(self.ground_node)
            branches = self.branches
            basis_params = [1, 0]  # numbers used to make basis vectors

            for n in nodes:  # reset the node markers
                n.marker = 0

            # step 2: finding maximum connected subgraphs of branches in branch_subset, then identifying the sets of nodes in each of thoses sets
            # step 2: finding the maximum connected set of independent branches in branch_subset, then identifying the sets of nodes in each of thoses sets
            branch_subset_ref = branch_subset.copy()

            trees = []  # list containing the maximum connected subgraphs

            while len(branch_subset_ref) > 0:
                b_0 = branch_subset_ref[0]
                tree = [b_0]
                branch_subset_ref.remove(b_0)

                while not self.is_disconnected(tree, branch_subset_ref):
                    for b1 in branch_subset_ref:
                        for b2 in tree:
                            if b1.is_connected(b2):
                                tree.append(b1)
                                branch_subset_ref.remove(b1)
                                break
                trees.append(tree)

            # finding the nodes in each of the maximum connected subgraph
            ind_nodes = [
                list(set(sum([branch.nodes for branch in tree], ()))) for tree in trees
            ]

            for x, node_set in enumerate(ind_nodes):
                if sum([n.id == 0 for n in node_set]) > 0:
                    for n in node_set:
                        n.marker = -1
                else:
                    for n in node_set:
                        n.marker = (
                            x + 1
                        )  # marking the nodes depending on which tree they belong to

            pos = [
                node.marker for node in nodes
            ]  # identifies the positions of the sets of nodes connected by the branches in branch_subset; different numbers on two nodes indicates that they are not connected through any of the branches in branch_subset. 0 implies the node does not belong to any of the branches in branch_subset

            # step 3: Finding the linearly independent vectors spanning the vector space of pos
            basis = []

            for marker in range(max(pos)):
                if marker == 0 and pos.count(0) == 0:
                    if pos.count(-1) == 0:
                        continue
                    else:
                        basis.append(
                            [
                                basis_params[0] if t == marker + 1 else basis_params[1]
                                for t in pos
                            ]
                        )
                else:
                    basis.append(
                        [
                            basis_params[0] if t == marker + 1 else basis_params[1]
                            for t in pos
                        ]
                    )

            if single_nodes == True:
                if pos.count(0) > 0:
                    for marker in range(len(pos)):
                        if pos[marker] == 0:
                            basis.append(
                                [
                                    basis_params[0] if t == marker else basis_params[1]
                                    for t in range(len(pos))
                                ]
                            )
            if self.is_grounded:
                basis = [i[:-1] for i in basis]

            return basis

        ##################### Finding the Periodic Modes ##################
        selected_branches = [branch for branch in self.branches if branch.type == "L"]
        periodic_modes = independent_modes(selected_branches)

        ##################### Finding the Zombie modes ##################
        selected_branches = [branch for branch in self.branches if branch.type != "L"]
        zombie_modes = independent_modes(selected_branches, single_nodes=False)

        ##################### Finding the Cyclic Modes ##################
        selected_branches = [branch for branch in self.branches if branch.type != "C"]
        cyclic_modes = independent_modes(selected_branches)
        # including the Σ mode
        Σ = [1 for n in self.nodes]
        if not self.is_grounded:  # only append if the circuit is not grounded
            # cyclic_modes.append(Σ)
            zombie_modes.append(Σ)

        # ##################### Finding the LC Modes ##################
        # selected_branches = [branch for branch in branches if branch.type == "JJ"]
        # LC_modes = independent_modes(selected_branches, single_nodes=False)

        ################ Adding periodic and zombie modes to cyclic ones #############
        # modes = cyclic_modes.copy()  # starting with the cyclic modes
        modes = zombie_modes.copy()  # starting with the cyclic modes

        for m in (
            # periodic_modes + zombie_modes
            cyclic_modes
            + periodic_modes
        ):  # adding the ones which are periodic such that all vectors in modes are LI
            mat = np.array(modes + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                modes.append(m)

        # for m in (LC_modes): # adding the LC modes to the basis
        #     mat = np.array(modes+[m])
        #     if np.linalg.matrix_rank(mat)==len(mat):
        #         modes.append(m)

        ####################### Completing the Basis ######################
        # step 4: construct the new set of basis vectors

        # constructing the standard basis
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
        #         standard_basis = np.ones([len(nodes),len(nodes)]) - 2*np.identity(len(nodes))

        new_basis = modes.copy()  # starting with the cyclic modes

        for m in standard_basis:  # completing the basis
            mat = np.array([i for i in new_basis] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                new_basis.append(m)
        new_basis = np.array(new_basis)

        # sorting the basis so that the cyclic, periodic and zombie variables occur at the beginning.
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
        pos_zombie = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_cyclic
            if i not in pos_periodic
            if new_basis[i].tolist() in zombie_modes
        ]
        pos_rest = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_cyclic
            if i not in pos_periodic
            if i not in pos_zombie
        ]
        pos_list =  pos_periodic + pos_rest + pos_cyclic + pos_zombie +  pos_Σ
        # transforming the new_basis matrix
        new_basis = new_basis[pos_list].T

        # Updating the class properties
        self.var_indices = {
            "periodic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_periodic
            ],
            "discretized_phi": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_rest
            ],
            "cyclic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_cyclic
            ],
            "zombie": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_zombie
            ],
        }
        # creating a class attribute for conserved charges corresponding to cyclic variables
        for c in self.var_indices["cyclic"]:
            setattr(self, "Qc" + str(c), 0)


        # set param_vars
        if self.mode == "sym":
            parameters = [
                [],
                [],
                [],
                [],
            ]  # showing three sublists, Ec's, El's ,Ej's and Ecj's
            for b in self.branches:
                if b.type == "JJ" or b.type == "JJ2":
                    parameters[2].append(b.parameters["E_J"])
                    parameters[3].append(b.parameters["E_CJ"])
                elif b.type == "L":
                    parameters[1].append(b.parameters["E_L"])
                elif b.type == "C":
                    parameters[0].append(b.parameters["E_C"])
            parameters = [list(set(i)) for i in parameters]
            param_list = []
            for i in parameters:
                param_list += [j for j in i if type(j) == sympy.core.symbol.Symbol]
        elif self.mode == "num":
            param_list = []

        for c in self.var_indices["cyclic"]:
            param_list.append(symbols("Qc" + str(c)))
        self.param_vars = param_list
        self.trans_mat = np.array(new_basis)

        return np.array(new_basis)

    """
    Methods used to construct the Lagrangian of the circuit
    """

    def _JJ_terms(self):
        J = 0
        for b in [t for t in self.branches if t.type == "JJ"]:
            if len(set(b.nodes)) > 1:  # branch if shorted is not considered
                # adding external flux
                phi_ext = 0
                if b in self.flux_branches:
                    index = self.flux_branches.index(b)
                    phi_ext += self.external_flux_vars[index]

                if (
                    b.nodes[1].id == 0
                ):  # if loop to check for the presence of ground node
                    J += -b.parameters["E_J"] * sympy.cos(
                        -symbols("φ" + str(b.nodes[0].id)) + phi_ext
                    )
                elif b.nodes[0].id == 0:
                    J += -b.parameters["E_J"] * sympy.cos(
                        symbols("φ" + str(b.nodes[1].id)) + phi_ext
                    )
                else:
                    J += -b.parameters["E_J"] * sympy.cos(
                        symbols("φ" + str(b.nodes[1].id))
                        - symbols("φ" + str(b.nodes[0].id))
                        + phi_ext
                    )
        return J

    def _JJ2_terms(self):
        J = 0
        for b in [t for t in self.branches if t.type == "JJ2"]:
            if len(set(b.nodes)) > 1:  # branch if shorted is not considered
                # adding external flux
                phi_ext = 0
                if b in self.flux_branches:
                    index = self.flux_branches.index(b)
                    phi_ext += self.external_flux_vars[index]

                if (
                    b.nodes[1].id == 0
                ):  # if loop to check for the presence of ground node
                    J += -b.parameters["E_J"] * sympy.cos(
                        2*(-symbols("φ" + str(b.nodes[0].id)) + phi_ext)
                    )
                elif b.nodes[0].id == 0:
                    J += -b.parameters["E_J"] * sympy.cos(
                        2*(symbols("φ" + str(b.nodes[1].id)) + phi_ext)
                    )
                else:
                    J += -b.parameters["E_J"] * sympy.cos(
                        2*(symbols("φ" + str(b.nodes[1].id))
                        - symbols("φ" + str(b.nodes[0].id))
                        + phi_ext)
                    )
        return J

    def _C_matrix(self):
        
        if not self.is_grounded:
            N = len(self.nodes)
            if self.mode == "num":
                C_mat = np.zeros([N,N])
            elif self.mode == "sym":
                C_mat = sympy.zeros(N)
            for b in [t for t in self.branches if t.type=="C" or t.type=="JJ" or t.type=="JJ2"]:
                if len(set(b.nodes)) > 1: # branch if shorted is not considered
                    element_param = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}
                    C_mat[b.nodes[0].id - 1, b.nodes[1].id - 1] += -1/(b.parameters[element_param[b.type]]*8)
        else:
            N = len(self.nodes) + 1
            if self.mode == "num":
                C_mat = np.zeros([N,N])
            elif self.mode == "sym":
                C_mat = sympy.zeros(N)
            for b in [t for t in self.branches if t.type=="C" or t.type=="JJ" or t.type=="JJ2"]:
                if len(set(b.nodes)) > 1: # branch if shorted is not considered
                    element_param = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}
                    C_mat[b.nodes[0].id, b.nodes[1].id] += -1/(b.parameters[element_param[b.type]]*8)

        if self.mode == "num":
            C_mat = C_mat + C_mat.T - np.diag(C_mat.diagonal())
        elif self.mode == "sym":
            C_mat = C_mat + C_mat.T - sympy.diag(*C_mat.diagonal())
        
        for i in range(C_mat.shape[0]):
            C_mat[i, i] = -np.sum(C_mat[i, :])
        
        if self.is_grounded: # if grounded remove the 0th column and row from C_mat
            C_mat =  C_mat[1:,1:]
        return C_mat

    def _C_terms(self):
        C = 0
        for b in [t for t in self.branches if t.type == "C" or t.type == "JJ" or t.type == "JJ2"]:
            if len(set(b.nodes)) > 1:  # branch if shorted is not considered
                element_param = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}

                if b.nodes[1].id == 0:
                    C += (
                        1
                        / (16 * b.parameters[element_param[b.type]])
                        * (symbols("vφ" + str(b.nodes[0].id))) ** 2
                    )
                elif b.nodes[0].id == 0:
                    C += (
                        1
                        / (16 * b.parameters[element_param[b.type]])
                        * (-symbols("vφ" + str(b.nodes[1].id))) ** 2
                    )
                else:
                    C += (
                        1
                        / (16 * b.parameters[element_param[b.type]])
                        * (
                            symbols("vφ" + str(b.nodes[1].id))
                            - symbols("vφ" + str(b.nodes[0].id))
                        )
                        ** 2
                    )
        return C

    def _L_terms(self):
        L = 0
        for b in [t for t in self.branches if t.type == "L"]:
            if len(set(b.nodes)) > 1:  # branch if shorted is not considered
                # adding external flux
                phi_ext = 0
                if b in self.flux_branches:
                    index = self.flux_branches.index(b)
                    phi_ext += self.external_flux_vars[index]

                if b.nodes[0].id == 0:
                    L += (
                        0.5
                        * b.parameters["E_L"]
                        * (symbols("φ" + str(b.nodes[1].id)) + phi_ext) ** 2
                    )
                elif b.nodes[1].id == 0:
                    L += (
                        0.5
                        * b.parameters["E_L"]
                        * (-symbols("φ" + str(b.nodes[0].id)) + phi_ext) ** 2
                    )
                else:
                    L += (
                        0.5
                        * b.parameters["E_L"]
                        * (
                            symbols("φ" + str(b.nodes[1].id))
                            - symbols("φ" + str(b.nodes[0].id))
                            + phi_ext
                        )
                        ** 2
                    )
        return L

    def _flux_loops(self):

        circ_copy = copy.deepcopy(self)
        ################### removing all the capacitive branches and updating the nodes ################
        for b in list(circ_copy.branches):
            if b.type == "C":
                for n in b.nodes:
                    n.branches = [i for i in n.branches if i is not b]
                circ_copy.branches.remove(b)

        i = 2
        while i > 1:
            i = 1
            for n in circ_copy.nodes:
                if len(n.branches) == 0:
                    circ_copy.nodes.remove(n)
                    i += 1
                    continue
                if len(n.branches) == 1:
                    b = n.branches[0]
                    circ_copy.branches.remove(b)
                    for n1 in b.nodes:
                        if n1 != n:
                            n1.branches = [i for i in n1.branches if i is not b]
                            i += 1
                            continue
                        else:
                            circ_copy.nodes.remove(n)

        if circ_copy.nodes == []:
            return []
        ##################################################

        ################### Constructing the node_sets ###############
        if circ_copy.is_grounded:
            node_sets = [[circ_copy.ground_node]]
        else:
            node_sets = [
                [circ_copy.nodes[0]]
            ]  # starting with the first set which has the first node as the only element

        num_nodes = len(circ_copy.nodes)
        if circ_copy.is_grounded:
            num_nodes += 1

        i = 0
        while (
            len([q for p in node_sets for q in p]) < num_nodes
        ):  # finding all the sets of nodes and filling node_sets
            node_set = []

            if node_sets[i] == []:
                for n in circ_copy.nodes:
                    if n not in [q for p in node_sets for q in p]:
                        node_sets[i].append(n)

            for n in node_sets[i]:
                node_set += n.connected_nodes("all")

            node_sets.append(
                [
                    x
                    for x in list(set(node_set))
                    if x not in [q for p in node_sets[: i + 1] for q in p]
                ]
            )
            i += 1
        ###############################################

        ############### Identifying the flux branches ######################
        flux_branches = []  # set of branches where external flux is associated

        flux_branches = []
        for i in range(0, len(node_sets)):
            for n in node_sets[i]:
                next_branches = []
                next_nodes = []
                loop_branches = []

                for b in n.branches:
                    if b.type != "C":
                        if b.nodes[0] != n:
                            next_nodes.append(b.nodes[0])
                        else:
                            next_nodes.append(b.nodes[1])
                        next_branches.append(b)
                # Indexing which set the nodes belong to
                next_nodes_set = []
                for k in next_nodes:
                    for p in range(0, len(node_sets)):
                        if k in node_sets[p]:
                            next_nodes_set.append(p + 1)

                # identifying the branches accordingly
                for j in range(len(next_nodes)):
                    if next_nodes_set[j] > i + 1:
                        loop_branches.append(next_branches[j])

                if len(loop_branches) > 1:
                    flux_branches.append(
                        loop_branches[:-1]
                    )  # selecting n-1 elements in the list for external flux @ basis

                # identifying the loops in the same set
                self_nodes = []
                self_branches = []
                for x, s in enumerate(next_nodes_set):
                    if s == i + 1:
                        self_nodes.append(next_nodes[x])
                        self_branches.append(next_branches[x])

                self_nodes = list(set(self_nodes))
                self_branches = list(set(self_branches))

                if len(self_nodes) == 1 and len(self_branches) > 1:
                    flux_branches.append(
                        self_branches[:-1]
                    )  # selecting n-1 branches for external flux

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

        flux_branches = list(set([i for j in flux_branches for i in j]))
        flux_branches_circ = []
        for b in flux_branches:
            flux_branches_circ += [i for i in self.branches if is_same_branch(i, b)]
        ########################################

        # setting the class properties
        if len(flux_branches) > 0:
            self.flux_branches = flux_branches_circ
            self.external_flux_vars = [
                symbols("Φ" + str(i + 1)) for i in range(len(flux_branches_circ))
            ]

        return self.flux_branches

    def lagrangian_sym(self, basis=None):
        """
        Outputs the Lagrangian of the circuit in terms of the new variables
        output: (number of cyclic variables, periodic variables, Sympy expression)
        """
        if basis is None:  # using the Lagrangian for a different transformation matrix
            basis = self.variable_transformation_matrix().astype(int)
        flux_branches = self._flux_loops()

        φ_vars = [
            symbols("φ" + str(i)) for i in range(1, len(self.nodes) + 1)
        ]  # defining the φ variables
        φ_dot_vars = [
            symbols("vφ" + str(i)) for i in range(1, len(self.nodes) + 1)
        ]  # defining the φ variables
        
        θ_vars = [
            symbols("θ" + str(i)) for i in range(1, len(self.nodes) + 1)
        ]  # defining the θ variables
        θ_dot_vars = [symbols("vθ" + str(i)) for i in range(1, len(self.nodes) + 1)]
        φ_vars_θ = (basis).dot(θ_vars)  # writing φ in terms of θ variables
        φ_dot_vars_θ = (basis).dot(θ_dot_vars)

        # C_terms = self._C_terms()
        C_mat = self._C_matrix()
        if self.mode == "num":
            C_terms_φ = ((C_mat).dot(φ_dot_vars)).dot(φ_dot_vars)*0.5 # interms of node variables
            C_terms_θ = ((C_mat).dot(φ_dot_vars_θ)).dot(φ_dot_vars_θ)*0.5 # interms of new variables
        elif self.mode == "sym":
            C_terms_φ = (sympy.Matrix(φ_dot_vars).T * C_mat * sympy.Matrix(φ_dot_vars))[0] * 0.5 # interms of node variables
            C_terms_θ = (sympy.Matrix(φ_dot_vars_θ).T * C_mat * sympy.Matrix(φ_dot_vars_θ))[0] * 0.5 # interms of new variables
        
        L_terms_φ = self._L_terms()

        JJ_terms_φ = self._JJ_terms() + self._JJ2_terms()

        L_φ = C_terms_φ - L_terms_φ - JJ_terms_φ

        potential_φ = L_terms_φ + JJ_terms_φ
        potential_θ = potential_φ.copy()

        for i in range(len(self.nodes)):  # converting potential to new variables
            potential_θ = potential_θ.subs(symbols("φ" + str(i + 1)), φ_vars_θ[i])

        # eliminating the zombie variables
        for i in self.var_indices["zombie"]:
            sub = sympy.solve(potential_θ.diff(symbols("θ" + str(i))), symbols("θ" + str(i)))
            potential_θ = potential_θ.replace(symbols("θ" + str(i)), sub[0])

        self.potential = potential_θ
        L_θ = C_terms_θ - potential_θ
        self._L = L_θ  # using a separate variable to store Lagrangian as used by code internally

        ############# Updating the class properties ###################

        self.L = L_θ.expand()
        self.L_old = L_φ

        # Replacing energies with capacitances if the circuit mode is symbolic
        if self.mode == "sym":
            # finding the unique capacitances
            uniq_capacitances = []
            element_param = {"C": "E_C", "JJ": "E_CJ", "JJ2": "E_CJ"}
            for c, b in enumerate(
                [t for t in self.branches if t.type == "C" or t.type == "JJ" or t.type == "JJ2"]
            ):
                if len(set(b.nodes)) > 1:  # check to see if branch is shorted
                    if b.parameters[element_param[b.type]] not in uniq_capacitances:
                        uniq_capacitances.append(b.parameters[element_param[b.type]])

            for index, var in enumerate(uniq_capacitances):
                self.L = self.L.subs(var, 1 / (8 * symbols("C" + str(index + 1))))
                self.L_old = self.L_old.subs(
                    var, 1 / (8 * symbols("C" + str(index + 1)))
                )

        return self.L

    def hamiltonian_sym(self, basis=None):
        """
        Outputs the Hamiltonian of the circuit in terms of the new variables
        output: (number of cyclic variables, periodic variables, Sympy expression)
        """
        self.lagrangian_sym(basis=basis)
        
        # Excluding the zombie modes
        if self.is_grounded:
            n = len(self.var_indices["zombie"])
        else:
            n = len(self.var_indices["zombie"]) + 1 

        N = len(self.nodes)
        if basis is None:
            basis = self.trans_mat
        basis_inv = np.linalg.inv(basis)[0:N-n, 0:N-n]

        if self.mode == "sym":
            C_mat_θ = (basis.T * self._C_matrix() * basis)[0:N-n, 0:N-n].inv()  # exlcluding the zombie modes
        elif self.mode == "num":
            C_mat_θ = np.linalg.inv((basis.T @ self._C_matrix() @ basis)[0:N-n, 0:N-n])  # exlcluding the zombie modes

        # x_vars = (self.trans_mat).dot(y_vars) # writing φ in terms of θ variables
        # x_dot_vars = (self.trans_mat).dot(y_dot_vars)

        p_θ_vars = [
            symbols("Q" + str(i)) if i not in self.var_indices["cyclic"] else symbols("Qc" + str(i)) for i in range(1, len(self.nodes) + 1 - n)
        ]  # defining the momentum variables
        # p_φ_vars_θ = basis.dot(p_θ_vars) # writing φ in terms of θ variables 

        if self.mode == "num":
            C_terms_new = C_mat_θ.dot(p_θ_vars).dot(p_θ_vars) * 0.5 # interms of new variables
        elif self.mode == "sym":
            C_terms_new = (sympy.Matrix(p_θ_vars).T * C_mat_θ  * sympy.Matrix(p_θ_vars))[0] * 0.5 # interms of new variables

        H = C_terms_new + self.potential

        self.offset_charge_vars = []
        for p in self.var_indices[
            "periodic"
        ]:  # same as above for periodic variables and adding the offset charge variables
            H = H.subs(
                symbols("Q" + str(p)), symbols("n" + str(p)) + symbols("ng_" + str(p))
            )
            self.offset_charge_vars = self.offset_charge_vars + [
                symbols("ng_" + str(p))
            ]

        # Updating the class property
        self.H = H  # .cancel()  # .expand()

        return self.H