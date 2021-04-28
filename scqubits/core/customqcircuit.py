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

import sympy
import numpy as np
from numpy import ndarray
from sympy import symbols, lambdify, MatrixSymbol
from scipy import sparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix
from matplotlib import pyplot as plt

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
            elif element == "JJ":
                self.parameters = {"E_J": parameters[0], "E_CJ": parameters[1]}
        # updating the nodes
        self.nodes[0].branches.append(self)
        self.nodes[1].branches.append(self)

    def set_parameters(self, parameters: ndarray):
        if self.type == "C" or self.type == "L":
            self.parameters = {self.type: parameters[0]}
        elif self.type == "JJ":
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


class CustomQCircuit:
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

    def __init__(self, list_nodes: list, list_branches: list = None, mode: str = "sym"):

        self.branches = list_branches
        self.nodes = list_nodes
        self.mode = mode
        self._input_string = None
        # properties set by methods
        self.trans_mat = None

        self.var_indices = None

        self.external_flux_vars = []
        self.flux_branches = []

        self.param_vars = []

        self.H = None
        self.L = None
        self.L_old = (
            None  # Lagrangian in terms of untransformed generalized flux variables
        )

        # Calling the function to initiate the calss variables
        self.hamiltonian_sym()

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
    def from_input_string(cls, input_string: str, mode: str = "sym"):
        """
        Constructor of class CustomQCircuit:
        - Constructing the instance from an input string
        - mode parameter to specify the use of symbolic or numerical circuit parameters
        """
        lines = (input_string).split("\n")
        num_nodes = int(lines[0].split(":")[-1])
        nodes = [node(i, 0) for i in range(1, num_nodes + 1)]
        branches = []

        first_branch = lines.index("branches:")
        for l in range(first_branch + 1, len(lines)):
            if lines[l] != "":
                line = lines[l].split("\t")
                a, b = [int(i) for i in line[1].split(",")]
                element = line[0]

                if element == "JJ":
                    if (
                        len(line) > 3
                    ):  # check to see if all the required parameters are defined
                        if mode == "sym":
                            parameters = [symbols(line[2]), symbols(line[3])]
                        elif mode == "num":
                            parameters = [float(line[2]), float(line[3])]
                    else:
                        parameters = None
                else:
                    if (
                        len(line) > 2
                    ):  # check to see if all the required parameters are defined
                        if mode == "sym":
                            parameters = [symbols(line[2])]
                        elif mode == "num":
                            parameters = [float(line[2])]
                    else:
                        parameters = None
                branches.append(branch(nodes[a - 1], nodes[b - 1], element, parameters))
            else:
                break
        circuit = cls(nodes, branches, mode=mode)
        circuit._input_string = input_string

        return circuit

    @classmethod
    def from_input_file(cls, filename: str, mode: str = "sym"):
        """
        Constructor of class CustomQCircuit:
        - Constructing the instance from an input file
        - mode parameter to specify the use of symbolic or numerical circuit parameters
        """
        file = open(filename, "r")
        return cls.from_input_string(cls, file.read(), mode=mode)

    """
    Methods to find the cyclic variables of the circuit
    """

    def new_basis(self):
        """
        Method to construct a new set of flux variables of the circuit, such that all the possible cyclic and periodic variables are included.
        This method returns a integer l(denoting the number of cyclic variables), and a matrix expressing the old in terms of the new variables
        The first l variables of the output are cyclic.
        periodic_vars: indices of the index of new variables which are periodic
        Returns: (l,periodic_vars,basis)

        """
        basis_params = [1, 0]  # numbers used to construct the basis vectors

        nodes = self.nodes
        branches = self.branches
        for n in nodes:
            n.marker = 0
        # first values is used for the selected node, and the second for the rest of the nodes
        ##################### Finding the Periodic Modes ##################
        # step 1
        s_1 = [i for i in branches if i.type == "L"]

        # step 2: finding the independent sets of branches in s_1, then identifying the sets of nodes in each of thoses sets
        s_1_ref = s_1.copy()

        trees = []  # list of independet sets of branches, in s_1

        while len(s_1_ref) > 0:
            b_0 = s_1_ref[0]
            tree = [b_0]
            s_1_ref.remove(b_0)

            while not self.is_disconnected(tree, s_1_ref):
                for b1 in s_1_ref:
                    for b2 in tree:
                        if b1.is_connected(b2):
                            tree.append(b1)
                            s_1_ref.remove(b1)
                            break
            trees.append(tree)

        ind_nodes = [
            list(set(sum([i.nodes for i in t], ()))) for t in trees
        ]  # finding the nodes in each of the independent sets of branches found earlier

        ref = list(
            range(1, len(nodes) + 1)
        )  # using numbers as variables, maximum needed is the number of nodes in the circuit

        for i in range(len(ind_nodes)):
            for n in ind_nodes[i]:
                n.marker = ref[i]

        pos = (
            []
        )  # identifies the positions of the sets of nodes connected by the branches in s_1; different numbers on two nodes indicates that they are not connected through any of the branches in s_1. 0 implies the node does not belong to any of the branches in s_1
        for i in nodes:
            pos.append(i.marker)

        # step 3: Finding the linearly independent vectors spanning the vector space of pos, this gives us the cyclic modes
        #         modes_num = pos.count(0) + max(pos) # total number of cyclic modes
        periodic_modes = []

        for i in range(max(pos)):
            if i == 0 and pos.count(0) == 0:
                continue
            else:
                periodic_modes.append(
                    [basis_params[0] if t == ref[i] else basis_params[1] for t in pos]
                )
        if pos.count(0) > 0:
            for i in range(len(pos)):
                if pos[i] == 0:
                    periodic_modes.append(
                        [
                            basis_params[0] if t == i else basis_params[1]
                            for t in range(len(pos))
                        ]
                    )

        ##################### Finding the Zombie modes ##################
        for n in nodes:
            n.marker = 0
        # step 1
        s_1 = [i for i in branches if i.type != "L"]

        # step 2: finding the independent sets of branches in s_1, then identifying the sets of nodes in each of thoses sets
        s_1_ref = s_1.copy()

        trees = []  # list of independet sets of branches, in s_1

        while len(s_1_ref) > 0:
            b_0 = s_1_ref[0]
            tree = [b_0]
            s_1_ref.remove(b_0)

            while not self.is_disconnected(tree, s_1_ref):
                for b1 in s_1_ref:
                    for b2 in tree:
                        if b1.is_connected(b2):
                            tree.append(b1)
                            s_1_ref.remove(b1)
                            break
            trees.append(tree)

        ind_nodes = [
            list(set(sum([i.nodes for i in t], ()))) for t in trees
        ]  # finding the nodes in each of the independent sets of branches found earlier

        ref = list(
            range(1, len(nodes) + 1)
        )  # using numbers as variables, maximum needed is the number of nodes in the circuit

        for i in range(len(ind_nodes)):
            for n in ind_nodes[i]:
                n.marker = ref[i]

        pos = (
            []
        )  # identifies the positions of the sets of nodes connected by the branches in s_1; different numbers on two nodes indicates that they are not connected through any of the branches in s_1. 0 implies the node does not belong to any of the branches in s_1
        for i in nodes:
            pos.append(i.marker)

        # step 3: Finding the linearly independent vectors spanning the vector space of pos, this gives us the cyclic modes
        #         modes_num = pos.count(0) + max(pos) # total number of cyclic modes
        zombie_modes = []

        for i in range(max(pos)):
            if i == 0 and pos.count(0) == 0:
                continue
            else:
                zombie_modes.append(
                    [basis_params[0] if t == ref[i] else basis_params[1] for t in pos]
                )
        if pos.count(0) > 0:
            for i in range(len(pos)):
                if pos[i] == 0:
                    zombie_modes.append(
                        [
                            basis_params[0] if t == i else basis_params[1]
                            for t in range(len(pos))
                        ]
                    )

        ##################### Finding the Cyclic Modes ##################
        for n in nodes:
            n.marker = 0
        # step 1
        s_1 = [i for i in branches if i.type != "C"]

        # step 2: finding the independent sets of branches in s_1, then identifying the sets of nodes in each of thoses sets
        s_1_ref = s_1.copy()

        trees = []  # list of independet sets of branches, in s_1

        while len(s_1_ref) > 0:
            b_0 = s_1_ref[0]
            tree = [b_0]
            s_1_ref.remove(b_0)

            while not self.is_disconnected(tree, s_1_ref):
                for b1 in s_1_ref:
                    for b2 in tree:
                        if b1.is_connected(b2):
                            tree.append(b1)
                            s_1_ref.remove(b1)
                            break
            trees.append(tree)

        ind_nodes = [
            list(set(sum([i.nodes for i in t], ()))) for t in trees
        ]  # finding the nodes in each of the independent sets of branches found earlier

        ref = list(
            range(1, len(nodes) + 1)
        )  # using numbers as variables, maximum needed is the number of nodes in the circuit

        for i in range(len(ind_nodes)):
            for n in ind_nodes[i]:
                n.marker = ref[i]

        pos = (
            []
        )  # identifies the positions of the sets of nodes connected by the branches in s_1; different numbers on two nodes indicates that they are not connected through any of the branches in s_1. 0 implies the node does not belong to any of the branches in s_1
        for i in nodes:
            pos.append(i.marker)

        # step 3: Finding the linearly independent vectors spanning the vector space of pos, this gives us the cyclic modes
        modes_num = pos.count(0) + max(pos)  # total number of cyclic modes
        cyclic_modes = []

        for i in range(max(pos)):
            if i == 0 and pos.count(0) == 0:
                continue
            else:
                cyclic_modes.append(
                    [basis_params[0] if t == ref[i] else basis_params[1] for t in pos]
                )

        Σ = [1 for t in pos]
        cyclic_modes.append(Σ)  # including the Σ mode

        ################ Adding cyclic and zombie modes to cyclic ones #############
        modes = cyclic_modes.copy()  # starting with the cyclic modes
        for m in (
            periodic_modes + zombie_modes
        ):  # adding the ones which are periodic such that all vectors in modes are LI
            mat = np.array([i for i in modes] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                modes.append(m)

        ####################### Completing the Basis ######################
        # step 4: construct the new set of basis vectors
        l = len(np.array(modes))
        standard_basis = np.identity(len(nodes))
        #         standard_basis = np.ones([len(nodes),len(nodes)]) - 2*np.identity(len(nodes))

        new_basis = modes.copy()  # starting with the cyclic modes

        for m in standard_basis:  # completing the basis
            mat = np.array([i for i in new_basis] + [m])
            if np.linalg.matrix_rank(mat) == len(mat):
                new_basis.append(m)
        new_basis = np.array(new_basis)

        # sorting the basis so that the cyclic, periodic and zombie variables occur at the beginning.
        pos_Σ = [i for i in range(len(new_basis)) if new_basis[i].tolist() == Σ]
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
        pos_list = pos_cyclic + pos_periodic + pos_rest + pos_zombie + pos_Σ
        # transforming the new_basis matrix
        new_basis = new_basis[pos_list].T

        # Updating the class properties
        self.var_indices = {
            "cyclic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_cyclic
            ],
            "periodic": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_periodic
            ],
            "discretized_phi": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_rest
            ],
            "zombie": [
                i + 1 for i in range(len(pos_list)) if pos_list[i] in pos_zombie
            ],
        }
        # updating param_vals
        if self.mode == "sym":
            parameters = [
                [],
                [],
                [],
                [],
            ]  # showing three sublists, Ec's, El's ,Ej's and Ecj's
            for b in self.branches:
                if b.type == "JJ":
                    parameters[2].append(b.parameters["E_J"])
                    parameters[3].append(b.parameters["E_CJ"])
                elif b.type == "L":
                    parameters[1].append(b.parameters["E_L"])
                elif b.type == "C":
                    parameters[0].append(b.parameters["E_C"])
            parameters = [list(set(i)) for i in parameters]
            param_list = []
            for i in parameters:
                param_list += i
        elif self.mode == "num":
            param_list = []
        self.param_vars = param_list
        # if self.var_cutoffs == None: # set cutoffs to default value if it is not already set
        #     self.var_cutoffs = {"cyclic":[2 for i in range(len(pos_list)) if pos_list[i] in pos_cyclic],
        #                         "periodic":[4 for i in range(len(pos_list)) if pos_list[i] in pos_periodic],
        #                         "discretized_phi":[30 for i in range(len(pos_list)) if pos_list[i] in pos_rest]}
        self.trans_mat = new_basis

        return new_basis

    """
    Methods used to construct the Lagrangian of the circuit
     - C_matrix
     - L_matrix
     - J_matrix
     These matrices are constructed in a similar fashion as compared to [C], [L^(-1)] matrices in 
     Intro to Quantum Electromagnetic Circuits, Uri Vool and Michel Devoret
    """

    def _C_matrix(self):
        N = len(self.nodes)

        C_mat = np.zeros([N, N], dtype="object")
        for b in [t for t in self.branches if t.type == "C" or t.type == "JJ"]:
            if len(set(self.nodes)) > 1:  # branch if shorted is not considered
                if b.type == "JJ":
                    C_mat[b.nodes[0].id - 1, b.nodes[1].id - 1] += -1 / (
                        b.parameters["E_CJ"] * 8
                    )
                else:
                    C_mat[b.nodes[0].id - 1, b.nodes[1].id - 1] += -1 / (
                        b.parameters["E_C"] * 8
                    )

        C_mat = C_mat + C_mat.T - np.diag(C_mat.diagonal())
        for i in range(len(self.nodes)):
            C_mat[i, i] = -np.sum(C_mat[i, :])
        return C_mat

    def _L_matrix(self):
        N = len(self.nodes)

        L_mat = np.zeros([N, N], dtype="object")
        for b in [t for t in self.branches if t.type == "L"]:
            if len(set(self.nodes)) > 1:  # branch if shorted is not considered
                L_mat[b.nodes[0].id - 1, b.nodes[1].id - 1] += -b.parameters["E_L"]
        L_mat = L_mat + L_mat.T - np.diag(L_mat.diagonal())
        for i in range(len(self.nodes)):
            L_mat[i, i] = -np.sum(L_mat[i, :])
        return L_mat

    def _JJ_terms(self):
        J = 0
        for b in [t for t in self.branches if t.type == "JJ"]:
            if len(set(self.nodes)) > 1:  # branch if shorted is not considered
                # adding external flux
                phi_ext = 0
                if b in self.flux_branches:
                    index = self.flux_branches.index(b)
                    phi_ext += self.external_flux_vars[index]

                J += -b.parameters["E_J"] * sympy.cos(
                    symbols("x" + str(b.nodes[1].id))
                    - symbols("x" + str(b.nodes[0].id))
                    + phi_ext
                )
        return J

    def _L_terms(self):
        L = 0
        for b in [t for t in self.branches if t.type == "L"]:
            if len(set(self.nodes)) > 1:  # branch if shorted is not considered
                # adding external flux
                phi_ext = 0
                if b in self.flux_branches:
                    index = self.flux_branches.index(b)
                    phi_ext += self.external_flux_vars[index]

                L += (
                    0.5
                    * b.parameters["E_L"]
                    * (
                        symbols("x" + str(b.nodes[1].id))
                        - symbols("x" + str(b.nodes[0].id))
                        + phi_ext
                    )
                    ** 2
                )
        return L

    def _flux_loops(self):
        node_sets = [
            [self.nodes[0]]
        ]  # starting with the first set which has the first node as the only element

        i = 0
        while len([q for p in node_sets for q in p]) < len(
            self.nodes
        ):  # finding all the sets of nodes and filling node_sets
            node_set = []
            for n in node_sets[i]:
                node_set += n.connected_nodes("L") + n.connected_nodes("JJ")

            node_sets.append(
                [
                    x
                    for x in list(set(node_set))
                    if x not in [q for p in node_sets[: i + 1] for q in p]
                ]
            )
            i += 1

        flux_branches = []  # set of branches where external flux is associated

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
                    if k in node_sets[0]:
                        next_nodes_set.append(1)
                    else:
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
                    )  # selecting n-1 elements in the list for external flux

        # setting the class property
        if len(flux_branches) > 0:
            self.flux_branches = [i for j in flux_branches for i in j]
            self.external_flux_vars = [
                symbols("Φ" + str(i + 1)) for i in range(len(self.flux_branches))
            ]

        return self.flux_branches

    def lagrangian_sym(self):
        """
        Outputs the Lagrangian of the circuit in terms of the new variables
        output: (number of cyclic variables, periodic variables, Sympy expression)
        """
        basis = self.new_basis()
        flux_branches = self._flux_loops()

        y_vars = [
            symbols("y" + str(i)) for i in range(1, len(self.nodes) + 1)
        ]  # defining the θ variables
        y_dot_vars = [symbols("vy" + str(i)) for i in range(1, len(self.nodes) + 1)]
        x_vars = (basis).dot(y_vars)  # writing φ in terms of θ variables
        x_dot_vars = (basis).dot(y_dot_vars)

        C_mat = self._C_matrix()
        C_terms = ((C_mat).dot(x_dot_vars)).dot(x_dot_vars) * 0.5

        #         L_mat = self.L_matrix()
        #         L_terms = ((L_mat).dot(x_vars)).dot(x_vars)*0.5

        L_terms = self._L_terms()

        JJ_terms = self._JJ_terms()

        L_old = (C_terms - L_terms - JJ_terms).expand()

        L_new = L_old.copy()

        for i in range(len(self.nodes)):  # converting to the new variables
            L_new = L_new.subs(symbols("x" + str(i + 1)), x_vars[i])
        #             JJ_terms = JJ_terms.subs(symbols("x" + str(i+1)), x_vars[i])
        #             if L_ex_flux_terms != 0:
        #                 L_ex_flux_terms = L_ex_flux_terms.subs(symbols("x" + str(i+1)), x_vars[i])

        #         L = (C_terms - L_terms - L_ex_flux_terms - JJ_terms).expand()

        # eliminating the zombie variables
        for i in self.var_indices["zombie"]:
            sub = sympy.solve(L_new.diff(symbols("y" + str(i))), symbols("y" + str(i)))
            L_new = L_new.replace(symbols("y" + str(i)), sub[0])
        # Updating the class properties
        self.L = L_new.expand()
        self.L_old = L_old

        return self.L

    def hamiltonian_sym(self):
        """
        Outputs the Hamiltonian of the circuit in terms of the new variables
        output: (number of cyclic variables, periodic variables, Sympy expression)
        """
        L = self.lagrangian_sym()
        y_vars = [
            symbols("y" + str(i)) for i in range(1, len(self.nodes) + 1)
        ]  # defining the θ variables
        y_dot_vars = [symbols("vy" + str(i)) for i in range(1, len(self.nodes) + 1)]
        # x_vars = (self.trans_mat).dot(y_vars) # writing φ in terms of θ variables
        # x_dot_vars = (self.trans_mat).dot(y_dot_vars)

        p_y_vars = [
            symbols("p" + str(i)) for i in range(1, len(self.nodes) + 1)
        ]  # defining the momentum variables
        p_y = np.array(
            [sympy.diff(L, i) for i in y_dot_vars]
        )  # finding the momentum expression in terms of y_dot

        n_vars_p = len(
            self.var_indices["cyclic"]
            + self.var_indices["periodic"]
            + self.var_indices["discretized_phi"]
        )
        y_dot_py = sympy.linsolve(
            (p_y - np.array(p_y_vars)).tolist()[:n_vars_p], tuple(y_dot_vars[:n_vars_p])
        )
        y_dot_py = list(list(y_dot_py)[0])

        H = (p_y[:n_vars_p].dot(y_dot_vars[:n_vars_p]) - L).subs(
            [(y_dot_vars[i], y_dot_py[i]) for i in range(len(y_dot_py))]
        )

        for c in self.var_indices[
            "cyclic"
        ]:  # To make it clear that the charge basis is used for cyclic variables
            H = H.subs(symbols("p" + str(c)), symbols("n" + str(c)))

        for p in self.var_indices["periodic"]:  # same as above for periodic variables
            H = H.subs(symbols("p" + str(p)), symbols("n" + str(p)))

        # Updating the class property
        self.H = H.expand()

        return self.H
