# analyze_circuit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


from ast import operator
from selectors import EpollSelector
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from matplotlib.cbook import flatten

import sympy
import sympy as sm
import numpy as np
import scipy as sp
import regex as re

from numpy import ndarray
from numpy.linalg import matrix_power
from sympy import symbols, lambdify, parse_expr
from scipy import sparse
from scipy.sparse.csc import csc_matrix
from matplotlib import pyplot as plt
from scqubits.core import operators as op
from scqubits import HilbertSpace, settings

from scqubits.core.symboliccircuit import SymbolicCircuit
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
from scqubits.core.storage import DataStore
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.utils.misc import list_intersection, flatten_list

from scqubits.utils.spectrum_utils import (
    get_matrixelement_table,
    order_eigensystem,
)

# Causing a circular import
# if TYPE_CHECKING:
#     from scqubits.core.symboliccircuit import Circuit


def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else None


class Circuit(base.QubitBaseClass, SymbolicCircuit, serializers.Serializable):
    r"""
    Class to numerically analyze an instance of CustomQCircuit.

    Can be initialized using an input file. For a Transmon qubit for example the following input file can be used.
    # file_name: transmon_num.inp
        nodes: 2
        branches:
        C	1,2	1
        JJ	1,2	1	10

    Circuit object can be initiated using:
        CustomQCircuit.from_input_file("transmon_num.inp")

    A set of nodes with branches connecting them forms a circuit.
    Parameters
    ----------
    nodes_list:
        List of nodes in the circuit
    branches_list:
        List of branches connecting the above set of nodes.
    phi_basis:
        "sparse" or "harmonic": Choose whether to use discretized phi or harmonic oscillator basis for extended variables.
    hierarchical_diagonalization:
        Boolean which indicates if the HilbertSpace from scqubits is used for simplification
    """

    def __init__(
        self,
        symbolic_circuit: SymbolicCircuit,
        initiate_sym_calc: bool = True,
        phi_basis: str = "sparse",
        hierarchical_diagonalization: bool = False,
    ):
        # inheriting all the attributes from SymbolicCircuit instance
        self.__dict__.update(symbolic_circuit.__dict__)

        # defining additional class properties

        self.vars = None
        self.external_flux = []

        self.H_func = None

        # setting truncated_dim for dispersion calculations
        self.truncated_dim = 6

        # setting default grids for plotting
        self._default_grid_phi = discretization.Grid1d(-6 * np.pi, 6 * np.pi, 200)

        self.discretized_phi_range = {}
        self.cutoffs_list = []
        self.phi_basis = phi_basis
        self.hierarchical_diagonalization = hierarchical_diagonalization

        # Hamiltonian function
        if initiate_sym_calc:
            self.initiate_symboliccircuit()
            self.initiate_circuit()

    # constructor to initiate using a CustomQCircuit object
    @classmethod
    def from_CustomQCircuit(
        cls,
        symbolic_circuit: SymbolicCircuit,
        hierarchical_diagonalization: bool = False,
        phi_basis: str = "sparse",
    ):
        """
        Initialize AnalyzeQCircuit using an instance of CustomQCircuit.

        Parameters
        ----------
        circuit:
            An instance of CustomQCircuit
        """
        return cls(
            symbolic_circuit,
            initiate_sym_calc=symbolic_circuit.initiate_sym_calc,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
        )

    @classmethod
    def from_input_string(
        cls,
        input_string: str,
        phi_basis="sparse",
        basis="simple",
        initiate_sym_calc=True,
        hierarchical_diagonalization: bool = False,
    ):

        circuit = SymbolicCircuit.from_input_string(
            input_string, basis=basis, initiate_sym_calc=initiate_sym_calc
        )

        return cls.from_CustomQCircuit(
            circuit,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
        )

    @classmethod
    def from_input_file(
        cls,
        filename: str,
        phi_basis="sparse",
        basis="simple",
        initiate_sym_calc=True,
        hierarchical_diagonalization: bool = False,
    ):

        circuit = SymbolicCircuit.from_input_file(
            filename, basis=basis, initiate_sym_calc=initiate_sym_calc
        )

        return cls.from_CustomQCircuit(
            circuit,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
        )

    def initiate_circuit(self, transformation_matrix=None):
        """
        Method to initialize the Circuit instance and initialize all the attributes needed before it can be passed on to AnalyzeQCircuit.

        Parameters
        ----------
        transformation_matrix:
            Takes an ndarray and is used to set an alternative transformation matrix than the one generated by the method variable_transformation_matrix.
        """
        self.initiate_symboliccircuit(transformation_matrix=transformation_matrix)
        # initiating the class properties
        self.cutoffs_list = []
        for var_type in self.var_indices.keys():
            if var_type == "periodic":
                for x, var_index in enumerate(self.var_indices["periodic"]):
                    setattr(self, "cutoff_n_" + str(var_index), 5)
                    self.cutoffs_list.append("cutoff_n_" + str(var_index))
            if var_type == "discretized_phi":
                for x, var_index in enumerate(self.var_indices["discretized_phi"]):
                    setattr(self, "cutoff_phi_" + str(var_index), 30)
                    self.cutoffs_list.append("cutoff_phi_" + str(var_index))
        # default values for the parameters
        for param in self.param_vars:
            # setting the default parameters as 1
            setattr(self, param.name, 1.0)
        # setting the ranges for floux ranges used for discrete phi vars
        for v in self.var_indices["discretized_phi"]:
            self.discretized_phi_range[v] = (-6 * np.pi, 6 * np.pi)
        # default values for the external flux vars
        for flux in self.external_flux_vars:
            # setting the default to zero external flux
            setattr(self, flux.name, 0.0)
        # default values for the offset charge vars
        for offset_charge in self.offset_charge_vars:
            # default to zero offset charge
            setattr(self, offset_charge.name, 0.0)

        # setting the __init__params attribute
        self._init_params = (
            [param.name for param in self.param_vars]
            + [flux.name for flux in self.external_flux_vars]
            + [offset_charge.name for offset_charge in self.offset_charge_vars]
            + self.cutoffs_list
            + ["input_string"]
        )

        self.set_vars()  # setting the attribute vars to store operator symbols

        if not self.hierarchical_diagonalization:
            self.hamiltonian_function()
        else:
            # self.HD_indices = []
            self.subsystem_truncated_dims = {}
            self.subsystem_truncated_dims["main"] = 10  # default value
            for sys_index in self.HD_indices:
                # default value
                self.subsystem_truncated_dims["sys_" + str(sys_index)] = 10
            self.hamiltonian_function()
            self.hierarchical_diagonalization_func()

        # initilizing attributes for operators
        self.set_operators()

        if self.hierarchical_diagonalization:
            self.main_subsystem.set_operators()
            for var_index in self.HD_indices:
                self.subsystems[var_index][0].set_operators()
            self.complete_hilbert_space()

    ##################################################################
    ##### Functions to construct the function for the Hamiltonian ####
    ##################################################################
    @staticmethod
    def _exp_dia(x):
        """
        This is a special function to calculate the expm of sparse diagonal matrices
        """
        return sparse.diags(np.exp((x.todia()).diagonal())).tocsc()

    @staticmethod
    def _cos_dia(x):
        """
        This is a special function to calculate the expm of sparse diagonal matrices
        """
        return sparse.diags(np.cos((x.todia()).diagonal())).tocsc()

    @staticmethod
    def _sin_dia(x):
        """
        This is a special function to calculate the expm of sparse diagonal matrices
        """
        return sparse.diags(np.sin((x.todia()).diagonal())).tocsc()

    def hierarchical_diagonalization_func(self):
        # H = self.hamiltonian_symbolic.expand()
        H = self.H_f

        systems = []
        interaction = []

        if len(self.HD_indices) == 0:
            raise Exception(
                "No oscillator has been detected in this circuit, hierarchcal diagonalization has only been implemented for oscillators."
            )

        for var_index in self.HD_indices:
            expr_dict = H.as_coefficients_dict()
            terms_list = list(expr_dict.keys())

            H_sys = 0 * symbols("x")
            H_int = 0 * symbols("x")
            for term in terms_list:
                var_indices = []
                for var in term.free_symbols:
                    # remove any branch parameters or flux and offset charge symbols
                    if (
                        "Φ" not in str(var)
                        and "ng" not in str(var)
                        and len(list_intersection(self.param_vars, [var])) == 0
                    ):
                        index = get_trailing_number(str(var))
                        if index not in var_indices:
                            var_indices.append(index)

                if len(var_indices) == 1:
                    if var_indices[0] == var_index:
                        H_sys = H_sys + expr_dict[term] * term

                if len(var_indices) > 1 and var_index in var_indices:
                    H_int = H_int + expr_dict[term] * term
            systems.append(H_sys)
            interaction.append(H_int)
            H = H - H_sys - H_int  # removing the terms added to a subsystem

        # storing data in class attributes
        self.subsystems_sym = dict(
            zip(
                self.HD_indices,
                [
                    [systems[index], interaction[index]]
                    for index in range(len(self.HD_indices))
                ],
            )
        )
        self.main_subsystem_sym = H  # just what is left of H

        self.main_subsystem = CircuitSubsystem(self, self.main_subsystem_sym)
        self.subsystems = dict(
            zip(
                self.HD_indices,
                [
                    [CircuitSubsystem(self, systems[index]), interaction[index]]
                    for index in range(len(self.HD_indices))
                ],
            )
        )

        # updating truncated dims
        self.main_subsystem.truncated_dim = self.subsystem_truncated_dims["main"]
        for sys_index in self.HD_indices:
            self.subsystems[sys_index][0].truncated_dim = self.subsystem_truncated_dims[
                "sys_" + str(sys_index)
            ]

    def complete_hilbert_space(self):
        hilbert_space = HilbertSpace(
            [self.main_subsystem] + [self.subsystems[i][0] for i in self.HD_indices]
        )

        # Adding interactions using the symbolic interaction term
        for sys_index in self.HD_indices:
            interaction = self.subsystems[sys_index][1].expand()
            if interaction == 0:  # if the interaction term is zero
                continue
            # modifying interaction terms
            #   - substituting all the external flux, offset charge and branch parameters.
            interaction = interaction.subs(
                [
                    (param, getattr(self, str(param)))
                    for param in self.param_vars
                    + self.external_flux_vars
                    + self.offset_charge_vars
                ]
            )
            #   - substituting Identity with 1
            interaction = interaction.subs("I", 1)
            #   - substituting cos and sin operators with their own symbols
            for i in self.var_indices["discretized_phi"]:
                interaction = interaction.replace(
                    sympy.cos(1.0 * symbols("θ" + str(i))), symbols("θc" + str(i))
                ).replace(
                    sympy.sin(1.0 * symbols("θ" + str(i))), symbols("θs" + str(i))
                )

            expr_dict = interaction.as_coefficients_dict()
            terms_str = list(expr_dict.keys())
            # coeff_str = list(expr_dict.values())

            for i, x in enumerate(terms_str):
                coefficient = expr_dict[x]

                # adding external flux, offset charge and branch parameters to coefficient
                for var in x.free_symbols:
                    if "Φ" in str(var) or "ng" in str(var) or var in self.param_vars:
                        coefficient = coefficient * getattr(self, str(var))

                operator_symbols = [
                    var
                    for var in x.free_symbols
                    if (("Φ" not in str(var)) and ("ng" not in str(var)))
                    and (var not in self.param_vars)
                ]

                main_op_list = []
                sys_op_dict = {index: [] for index in self.HD_indices}
                for var in operator_symbols:
                    var_index = get_trailing_number(str(var))
                    if var_index not in self.HD_indices and "I" not in str(var):
                        main_op_list.append(getattr(self.main_subsystem, str(var)))
                    elif var_index in self.HD_indices and "I" not in str(var):
                        sys_op_dict[var_index].append(
                            getattr(self.subsystems[var_index][0], str(var))
                        )
                    elif "I" in str(var):
                        main_op_list.append(self.main_subsystem._identity())

                operator_dict = {}
                for op_index, op in enumerate(main_op_list):
                    operator_dict["op" + str(op_index + 1)] = (
                        op,
                        self.main_subsystem,
                    )

                for index in self.HD_indices:
                    for op_index, op in enumerate(sys_op_dict[index]):
                        operator_dict["op" + str(len(main_op_list) + op_index + 1)] = (
                            op,
                            self.subsystems[sys_index][0],
                        )
                hilbert_space.add_interaction(g=float(coefficient), **operator_dict)

        self.hilbert_space = hilbert_space

    def set_vars(self):
        """
        Sets the attribute vars which is a dictionary containing all the Sympy symbol objects for all the operators present in the circuit
        """
        # Defining the list of variables for periodic operators
        periodic_symbols_sin = [
            symbols("θs" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols_cos = [
            symbols("θc" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols_n = [
            symbols("n" + str(i)) for i in self.var_indices["periodic"]
        ]

        # Defining the list of discretized_phi variables
        y_symbols = [symbols("θ" + str(i)) for i in self.var_indices["discretized_phi"]]
        p_symbols = [symbols("Q" + str(i)) for i in self.var_indices["discretized_phi"]]

        if self.phi_basis == "sparse":

            ps_symbols = [
                symbols("Qs" + str(i)) for i in self.var_indices["discretized_phi"]
            ]

        elif self.phi_basis == "harmonic":

            a_symbols = [
                symbols("a" + str(i)) for i in self.var_indices["discretized_phi"]
            ]
            ad_symbols = [
                symbols("ad" + str(i)) for i in self.var_indices["discretized_phi"]
            ]
            Nh_symbols = [
                symbols("Nh" + str(i)) for i in self.var_indices["discretized_phi"]
            ]

            extended_symbols = a_symbols + ad_symbols + Nh_symbols

        # setting the attribute self.vars
        self.vars = {
            "periodic": {
                "sin": periodic_symbols_sin,
                "cos": periodic_symbols_cos,
                "number": periodic_symbols_n,
            },
            "identity": [symbols("I")],
        }

        if self.phi_basis == "sparse":
            self.vars["discretized_phi"] = {
                "position": y_symbols,
                "momentum": p_symbols,
                "momentum_squared": ps_symbols,
            }
        elif self.phi_basis == "harmonic":
            self.vars["discretized_phi"] = {
                "annihilation": a_symbols,
                "creation": ad_symbols,
                "number": Nh_symbols,
            }

    def hamiltonian_function(self):
        """
        Outputs the function using lambdify in Sympy, which returns a Hamiltonian matrix by using the circuit attributes set in either the input file or the instance attributes.
        """
        H = (
            self.hamiltonian_symbolic.expand()
        )  # this expand method is critical to be applied, otherwise the replacemnt of the variables p^2 with ps2 will not be successful and the results would be incorrect

        # marking the sin and cos terms of the periodic variables with different symbols
        if len(self.var_indices["periodic"]) > 0:
            H = sympy.expand_trig(H).expand()

        for i in self.var_indices["periodic"]:
            H = H.replace(
                sympy.cos(1.0 * symbols("θ" + str(i))), symbols("θc" + str(i))
            ).replace(sympy.sin(1.0 * symbols("θ" + str(i))), symbols("θs" + str(i)))

        if self.phi_basis == "sparse":

            # marking the squared momentum operators with a separate symbol
            for i in self.var_indices["discretized_phi"]:
                H = H.replace(symbols("Q" + str(i)) ** 2, symbols("Qs" + str(i)))

        elif self.phi_basis == "harmonic":

            for i in self.var_indices["discretized_phi"]:
                H = H.replace(
                    sympy.cos(1.0 * symbols("θ" + str(i))),
                    symbols("θc" + str(i)),
                ).replace(
                    sympy.sin(1.0 * symbols("θ" + str(i))),
                    symbols("θs" + str(i)),
                )

        # removing the constants from the Hamiltonian
        coeff_dict = H.as_coefficients_dict()
        constants = [
            i
            for i in coeff_dict
            if "Q" not in str(i)
            and "θ" not in str(i)
            and "n" not in str(i)
            and "a" not in str(i)
            and "Nh" not in str(i)
        ]
        for i in constants:
            # + i*coeff_dict[i]*symbols("I")).expand()
            H = H - i * coeff_dict[i]

        # associate a identity matrix with the external flux vars
        for phi in self.external_flux_vars:
            H = H.subs(phi, phi * symbols("I") * 2 * np.pi)

        # associate a identity matrix with offset charge vars
        for offset_charge in self.offset_charge_vars:
            H = H.subs(offset_charge, offset_charge * symbols("I"))
        setattr(self, "H_f", H)

    ##################################################################
    ############### Functions to construct the operators #############
    ##################################################################
    def hilbertdim(self):
        """
        Returns the Hilbert dimension of the circuit used for calculations
        """
        cutoff_list = []
        for cutoffs in self.get_cutoffs().keys():
            if "cutoff_n" in cutoffs:
                cutoff_list.append([2 * k + 1 for k in self.get_cutoffs()[cutoffs]])
            elif "cutoff_phi" in cutoffs:
                cutoff_list.append([k for k in self.get_cutoffs()[cutoffs]])

        cutoff_list = [
            j for i in list(cutoff_list) for j in i
        ]  # concatenating the sublists
        return np.prod(cutoff_list)

    # helper functions
    def _kron_operator(self, operator, index):
        """
        Returns the final operator
        """
        if type(self) == Circuit and self.hierarchical_diagonalization:
            if index in self.main_subsystem.var_indices_list:
                var_index_list = self.main_subsystem.var_indices_list
            else:
                var_index_list = [index]
        else:
            var_index_list = (
                self.var_indices["periodic"] + self.var_indices["discretized_phi"]
            )

        var_index_list.sort()  # important to make sure that right cutoffs are chosen
        cutoff_dict = self.get_cutoffs()

        # if len(self.var_indices["periodic"]) != len(cutoff_dict["cutoff_n"]) or len(
        #     self.var_indices["discretized_phi"]
        # ) != len(cutoff_dict["cutoff_phi"]):
        #     raise AttributeError(
        #         "Make sure the cutoffs are only defined for the circuit variables in the class property var_indices, except for frozen variables. "
        #     )

        cutoff_list = []
        for cutoff_type in cutoff_dict.keys():
            if "cutoff_n" in cutoff_type:
                cutoff_list.append([2 * k + 1 for k in cutoff_dict[cutoff_type]])
            elif "cutoff_phi" in cutoff_type:
                cutoff_list.append([k for k in cutoff_dict[cutoff_type]])

        cutoffs = [
            j for i in list(cutoff_list) for j in i
        ]  # concatenating the sublists
        cutoffs_index_dict = dict(
            zip(
                self.var_indices["periodic"] + self.var_indices["discretized_phi"],
                cutoffs,
            )
        )
        cutoff_list = [
            cutoffs_index_dict[i] for i in var_index_list
        ]  # selecting the cutoffs present in

        if self.phi_basis == "sparse":
            matrix_format = "csc"
        elif self.phi_basis == "harmonic":
            matrix_format = "array"

        if len(var_index_list) > 1:
            if index > var_index_list[0]:
                Identity_l = sparse.identity(
                    np.prod(cutoff_list[: var_index_list.index(index)]),
                    format=matrix_format,
                )
            if index < var_index_list[-1]:
                Identity_r = sparse.identity(
                    np.prod(cutoff_list[var_index_list.index(index) + 1 :]),
                    format=matrix_format,
                )

            if index == var_index_list[0]:
                return sparse.kron(operator, Identity_r, format=matrix_format)
            elif index == var_index_list[-1]:
                return sparse.kron(Identity_l, operator, format=matrix_format)
            else:
                return sparse.kron(
                    sparse.kron(Identity_l, operator, format=matrix_format),
                    Identity_r,
                    format=matrix_format,
                )
        else:
            if self.phi_basis == "sparse":
                return sparse.csc_matrix(operator)
            elif self.phi_basis == "harmonic":
                return operator

    def _change_sparsity(self, x):
        if self.phi_basis == "harmonic":
            return x.toarray() * (1 + 0j)
        elif self.phi_basis == "sparse":
            return x

    # Identity Operator
    def _identity(self) -> csc_matrix:
        """
        Returns the Identity operator for the entire Hilber space of the circuit.
        """
        dim = self.hilbertdim()
        op = sparse.identity(dim)
        return op.tocsc()

    # Phi basis
    def _identity_phi(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator Identity in the discretized_phi basis.
        """
        pt_count = grid.pt_count
        return sparse.identity(pt_count, format="csc")

    def _phi_operator(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math: `\\varphi` in the discretized_phi basis.
        """
        pt_count = grid.pt_count

        phi_matrix = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = grid.make_linspace()
        phi_matrix.setdiag(diag_elements)
        return phi_matrix

    def _i_d_dphi_operator(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`-i * d/d\varphi` in the discretized_phi basis.
        """
        return grid.first_derivative_matrix(prefactor=-1j)

    def _i_d2_dphi2_operator(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`-1 * d^2/d\varphi^2`in the discretized_phi basis.
        """
        return grid.second_derivative_matrix(prefactor=-1.0)

    def _cos_phi(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`\\cos \\varphi` in the discretized_phi basis.
        """
        pt_count = grid.pt_count

        cos_op = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = np.cos(grid.make_linspace())
        cos_op.setdiag(diag_elements)
        return cos_op.tocsc()

    def _sin_phi(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`\\sin \\varphi` in the discretized_phi basis.
        """
        pt_count = grid.pt_count

        sin_op = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = np.cos(grid.make_linspace())
        sin_op.setdiag(diag_elements)
        return sin_op.tocsc()

    # charge basis

    def _identity_theta(self, ncut: int) -> csc_matrix:
        """
        Returns Operator identity in the charge basis.
        """
        dim_theta = 2 * ncut + 1
        return sparse.identity(dim_theta, format="csc")

    def _n_theta_operator(self, ncut: int) -> csc_matrix:
        """
        Returns charge operator `n` in the charge basis.
        """
        dim_theta = 2 * ncut + 1
        diag_elements = np.arange(-ncut, ncut + 1)
        n_theta_matrix = sparse.dia_matrix(
            (diag_elements, [0]), shape=(dim_theta, dim_theta)
        ).tocsc()
        return n_theta_matrix

    def _exp_i_theta_operator(self, ncut) -> csc_matrix:
        r"""
        Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.
        """
        dim_theta = 2 * ncut + 1
        matrix = (
            sparse.dia_matrix(([-1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))
        ).tocsc()
        return matrix

    def _exp_i_theta_operator_conjugate(self, ncut) -> csc_matrix:
        r"""
        Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.
        """
        dim_theta = 2 * ncut + 1
        matrix = (
            sparse.dia_matrix(([-1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta))
        ).tocsc()
        return matrix

    def _cos_theta(self, ncut: int) -> csc_matrix:
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * (
            self._exp_i_theta_operator(ncut)
            + self._exp_i_theta_operator_conjugate(ncut)
        )
        return cos_op

    def _sin_theta(self, ncut: int) -> csc_matrix:
        """Returns operator :math:`\\sin \\varphi` in the charge basis"""
        sin_op = (
            -1j
            * 0.5
            * (
                self._exp_i_theta_operator(ncut)
                - self._exp_i_theta_operator_conjugate(ncut)
            )
        )
        return sin_op

    def circuit_operators(self):
        """
        Returns the set of operators which can be given to the Hamiltonian function to construct the Hamiltonian matrix
        """
        import scqubits.core.discretization as discretization
        from scipy import sparse

        periodic_vars = self.vars["periodic"]
        normal_vars = self.vars["discretized_phi"]

        index_list = [j for i in list(self.var_indices.values()) for j in i]
        cutoff_list = [j for i in list(self.get_cutoffs().values()) for j in i]
        cutoffs = dict(zip(index_list, cutoff_list))

        grids = {}
        for i in self.var_indices["discretized_phi"]:
            grids[i] = discretization.Grid1d(
                self.discretized_phi_range[i][0],
                self.discretized_phi_range[i][1],
                cutoffs[i],
            )

        # constructing the operators for extended variables

        if self.phi_basis == "sparse":
            extended_operators = {
                "position": [],
                "momentum": [],
                "momentum_squared": [],
            }
            for v in normal_vars["position"]:  # position operators
                index = int(get_trailing_number(v.name))
                a_operator = self._phi_operator(grids[index])
                extended_operators["position"].append(
                    self._kron_operator(a_operator, index)
                )
            for v in normal_vars["momentum"]:  # momentum operators
                index = int(get_trailing_number(v.name))
                a_operator = self._i_d_dphi_operator(grids[index])
                extended_operators["momentum"].append(
                    self._kron_operator(a_operator, index)
                )
            for v in normal_vars["momentum_squared"]:  # squared momentum operators
                index = int(get_trailing_number(v.name))
                ps_operator = self._i_d2_dphi2_operator(grids[index])
                extended_operators["momentum_squared"].append(
                    self._kron_operator(ps_operator, index)
                )
        elif self.phi_basis == "harmonic":
            extended_operators = {
                "creation": [],
                "annihilation": [],
                "number": [],
            }
            for v in normal_vars["creation"]:  # a or annihilation operators
                index = int(get_trailing_number(v.name))
                ad_operator = op.creation(cutoffs[index])
                extended_operators["creation"].append(
                    self._kron_operator(ad_operator, index)
                )
            for v in normal_vars["annihilation"]:  # ad or creation operators
                index = int(get_trailing_number(v.name))
                a_operator = op.annihilation(cutoffs[index])
                extended_operators["annihilation"].append(
                    self._kron_operator(a_operator, index)
                )
            for v in normal_vars["number"]:  # Nh or number operators
                index = int(get_trailing_number(v.name))
                n_operator = (
                    op.creation(cutoffs[index])
                    @ op.annihilation(cutoffs[index])
                    * (1 + 0j)
                )
                extended_operators["number"].append(
                    self._kron_operator(n_operator, index)
                )

        # constructing the operators for periodic variables
        periodic_operators = {"sin": [], "cos": [], "number": []}
        for v in periodic_vars["sin"]:  # exp(ix) operators; ys
            index = int(v.name[2:])
            a_operator = self._change_sparsity(self._sin_theta(cutoffs[index]))
            periodic_operators["sin"].append(self._kron_operator(a_operator, index))
        for v in periodic_vars["cos"]:  # exp(-ix) operators; yc
            index = int(v.name[2:])
            a_operator = self._change_sparsity(self._cos_theta(cutoffs[index]))
            periodic_operators["cos"].append(self._kron_operator(a_operator, index))
        for v in periodic_vars["number"]:  # n operators; n
            index = int(v.name[1:])
            n_operator = self._change_sparsity(self._n_theta_operator(cutoffs[index]))
            periodic_operators["number"].append(self._kron_operator(n_operator, index))

        return {
            "periodic": periodic_operators,
            "discretized_phi": extended_operators,
            "identity": [self._change_sparsity(self._identity())],
        }

    ##################################################################
    ################ Functions for parameter queries #################
    ##################################################################
    def get_params(self):
        """
        Method to get the circuit parameters set using the instance attributes.
        """
        params = []
        for param in self.param_vars:
            params.append(getattr(self, param.name))
        return params

    def get_cutoffs(self):
        """
        Method to get the cutoffs for each of the circuit's degree of freedom.
        """
        cutoffs_dict = {
            "cutoff_n": [],
            "cutoff_phi": [],
        }

        for cutoff_type in cutoffs_dict.keys():
            attr_list = [x for x in self.cutoffs_list if cutoff_type in x]

            if len(attr_list) > 0:
                attr_list.sort()
                cutoffs_dict[cutoff_type] = [getattr(self, attr) for attr in attr_list]

        return cutoffs_dict

    def get_external_flux(self):
        """
        Returns all the time independent external flux set using the circuit attributes for each of the independent loops detected.
        """
        return [getattr(self, flux.name) for flux in self.external_flux_vars]

    def get_offset_charges(self):
        """
        Returns all the offset charges set using the circuit attributes for each of the periodic degree of freedom.
        """
        return [
            getattr(self, offset_charge.name)
            for offset_charge in self.offset_charge_vars
        ]

    def get_operators(self):
        """
        Returns a list of operators which can be given as an argument to self.H_func. These operators are not calculated again and are fetched directly from the circuit attibutes. Use set_attributes instead if the paramaters, expecially cutoffs, are changed.
        """
        variable_symbols_list = flatten_list(
            self.vars["periodic"].values()
        ) + flatten_list(self.vars["discretized_phi"].values())
        operator_list = []
        for operator in variable_symbols_list:
            operator_list.append(getattr(self, operator.name))

        # adding the identity operator
        operator_list.append(self._identity())

        return operator_list

    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

        return {}

    def set_operators(self, return_dict=False):
        """
        Sets the operator attributes of the circuit with new operators calculated using the paramaters set in the circuit attributes. Returns a list of operators similar to the method get_operators.
        Returns nothing.
        """

        variable_symbols_list = (
            flatten_list(self.vars["periodic"].values())
            + flatten_list(self.vars["discretized_phi"].values())
            + self.vars["identity"]
        )

        if type(self) == Circuit:
            ops = self.circuit_operators()
            operator_list = flatten_list(ops["periodic"].values()) + flatten_list(
                ops["discretized_phi"].values()
            )
        else:
            operator_list = [
                getattr(self.parent, str(var))
                for var in variable_symbols_list
                if "I" not in str(var)
            ]
        operator_list = operator_list + [self._identity()]

        for x, operator in enumerate(variable_symbols_list):
            setattr(self, operator.name, operator_list[x])

        if return_dict:
            return dict(
                zip(
                    [operator.name for operator in variable_symbols_list],
                    operator_list,
                )
            )

    ##################################################################
    ############# Functions for eigen values and matrices ############
    ##################################################################
    def is_mat_mul_replacement_necessary(self, term):
        return (
            set(self.var_indices["discretized_phi"])
            & set([get_trailing_number(str(i)) for i in term.free_symbols])
        ) and "*" in str(term)

    def replace_mat_mul_operator(self, term):

        if not self.is_mat_mul_replacement_necessary(term):
            return str(term)

        if self.phi_basis == "sparse":
            var_indices = [get_trailing_number(str(i)) for i in term.free_symbols]
            if len(set(var_indices) & set(self.var_indices["discretized_phi"])) > 1:
                return str(term).replace("*", "@")
            else:
                return str(term)

        term_string = ""
        # replace ** with np.matrix_power
        if "**" in str(term):
            operators = [
                match.replace("**", "")
                for match in re.findall(r"[^*]+\*{2}", str(term), re.MULTILINE)
            ]
            exponents = re.findall(r"\*{2}\K[0-9]", str(term), re.MULTILINE)

            new_string_list = []
            for x, operator in enumerate(operators):
                if get_trailing_number(operator) in self.var_indices["discretized_phi"]:
                    new_string_list.append(
                        "matrix_power(" + operator + "," + exponents[x] + ")"
                    )
                else:
                    new_string_list.append(operator + "**" + exponents[x])
            term_string = "*".join(new_string_list)
        else:
            term_string = str(term)

        # replace * with @ in the entire term
        term_string = re.sub(r"[^*]\K\*{1}(?!\*)", "@", term_string, re.MULTILINE)

        # replace @ with * for offset charge and flux variables
        matches = [
            "".join(match)
            for match in re.findall(
                r"(\@|)(ng_[0-9]+|Φ[0-9]+|I)(\@|)", term_string, re.MULTILINE
            )
        ]
        for match in matches:
            term_string = term_string.replace(match, match.replace("@", "*"))
        return term_string

    def get_eval_hamiltonian_string(self, H):
        """
        Returns the string which defines the expression for Hamiltonian in harmonic oscillator basis
        """
        expr_dict = H.as_coefficients_dict()
        terms_list = list(expr_dict.keys())
        coeff_list = list(expr_dict.values())

        H_string = ""
        for x, term in enumerate(terms_list):
            term_string = str(coeff_list[x]) + "*" + self.replace_mat_mul_operator(term)
            if float(coeff_list[x]) > 0:
                term_string = "+" + term_string
            H_string += term_string

        return H_string

    @staticmethod
    def _cos_dia_dense(x):
        """
        This is a special function to calculate the cos of dense diagonal matrices
        """
        return np.diag(np.cos(x.diagonal()))

    @staticmethod
    def _sin_dia_dense(x):
        """
        This is a special function to calculate the sin of dense diagonal matrices
        """
        return np.diag(np.sin(x.diagonal()))

    def hamiltonian_harmonic(self):

        H = self.H_f
        index_list = [j for i in list(self.var_indices.values()) for j in i]
        cutoff_list = [j for i in list(self.get_cutoffs().values()) for j in i]
        cutoffs_dict = dict(zip(index_list, cutoff_list))
        # substitute all the parameter values
        H = H.subs(
            [
                (param, getattr(self, str(param)))
                for param in self.param_vars
                + self.external_flux_vars
                + self.offset_charge_vars
            ]
        )

        # set the operators to circuit attributes
        self.set_operators()

        # calculate oscillator frequencies and use harmonic oscillator basis
        osc_lengths = {}
        osc_freqs = {}
        position_operators = {}
        momentum_operators = {}
        sin_operators = {}
        cos_operators = {}
        for i in self.var_indices["discretized_phi"]:
            ECi = float(H.coeff("Q" + str(i) + "**2").cancel()) / 4
            ELi = float(H.coeff("θ" + str(i) + "**2").cancel()) * 2
            osc_freqs[i] = (8 * ELi * ECi) ** 0.5
            osc_lengths[i] = (8.0 * ECi / ELi) ** 0.25
            ad_operator = op.creation(cutoffs_dict[i])
            a_operator = op.annihilation(cutoffs_dict[i])
            x_operator = (ad_operator + a_operator) * osc_lengths[i] / (2 ** 0.5)
            position_operators[i] = self._kron_operator(x_operator, i)
            momentum_operators[i] = self._kron_operator(
                1j * (ad_operator - a_operator) / (osc_lengths[i] * 2 ** 0.5), i
            )
            sin_operators[i] = self._kron_operator(sp.linalg.sinm(x_operator), i)
            cos_operators[i] = self._kron_operator(sp.linalg.cosm(x_operator), i)
            H = (
                (
                    H
                    - ECi * 4 * symbols("Q" + str(i)) ** 2
                    - ELi / 2 * symbols("θ" + str(i)) ** 2
                    + osc_freqs[i] * (symbols("Nh" + str(i)))
                )
                .cancel()
                .expand()
            )

        self.osc_lengths = osc_lengths
        self.osc_freqs = osc_freqs
        self.harmonic_operators = {
            "position": position_operators,
            "momentum": momentum_operators,
            "sin": sin_operators,
            "cos": cos_operators,
        }

        H_str = self.get_eval_hamiltonian_string(H)
        self.H_str_harmonic = H_str

        variable_symbols_list = (
            flatten_list(self.vars["periodic"].values())
            + flatten_list(self.vars["discretized_phi"].values())
            + self.vars["identity"]
        )

        harmonic_symbols_list = (
            [sm.symbols("θ" + str(i)) for i in self.var_indices["discretized_phi"]]
            + [sm.symbols("Q" + str(i)) for i in self.var_indices["discretized_phi"]]
            + [sm.symbols("θs" + str(i)) for i in self.var_indices["discretized_phi"]]
            + [sm.symbols("θc" + str(i)) for i in self.var_indices["discretized_phi"]]
        )

        harmonic_operators = (
            list(self.harmonic_operators["position"].values())
            + list(self.harmonic_operators["momentum"].values())
            + list(self.harmonic_operators["sin"].values())
            + list(self.harmonic_operators["cos"].values())
        )
        variable_str_list = [
            str(operator)
            for operator in variable_symbols_list
            + harmonic_symbols_list
            + self.offset_charge_vars
            + self.external_flux_vars
        ]
        variable_values_list = (
            self.get_operators()
            + harmonic_operators
            + self.get_offset_charges()
            + self.get_external_flux()
        )
        variable_dict = dict(zip(variable_str_list, variable_values_list))

        self.operator_dict = variable_dict

        # adding matrix power to the dict
        variable_dict["matrix_power"] = matrix_power
        variable_dict["cos"] = self._cos_dia_dense
        variable_dict["sin"] = self._sin_dia_dense

        return eval(H_str, variable_dict)

    def hamiltonian_sparse(self):

        H = self.H_f
        H = H.subs(
            [
                (param, getattr(self, str(param)))
                for param in self.param_vars
                + self.external_flux_vars
                + self.offset_charge_vars
            ]
        )

        H_str = self.get_eval_hamiltonian_string(H)

        variable_dict = self.set_operators(return_dict=True)
        variable_dict["cos"] = self._cos_dia
        variable_dict["sin"] = self._sin_dia

        return eval(H_str, variable_dict)

        # replace * with @ for non-diagonal operators

    def hamiltonian(self):
        """
        Returns the Hamiltonian of the Circuit bu using the parameters set in the class properties.
        """
        if self.phi_basis == "harmonic":
            if type(self) == Circuit:
                if not self.hierarchical_diagonalization:
                    return self.hamiltonian_harmonic()
            else:
                return self.hamiltonian_harmonic()

        # check on params class property
        if self.get_params() == None and self.is_any_branch_parameter_symbolic():
            raise AttributeError(
                "Set the params property of the circuit before calling this method."
            )
        if self.is_any_branch_parameter_symbolic():
            if len(self.param_vars) != len(self.get_params()):
                raise ValueError(
                    "Invalid number of parameters given, please check the number of parameters."
                )

        if type(self) == Circuit and self.hierarchical_diagonalization:
            self.set_operators()
            self.hierarchical_diagonalization_func()
            self.main_subsystem.set_operators()
            for var_index in self.HD_indices:
                self.subsystems[var_index][0].set_operators()
            self.complete_hilbert_space()
            return self.hilbert_space.hamiltonian()
        else:
            return self.hamiltonian_sparse()

    ##################################################################
    ############### Functions for plotting potential #################
    ##################################################################
    def potential_energy(self, **kwargs):
        """
        Returns the full potential of the circuit evaluated in a grid of points as chosen by the user or using default variable ranges.
        """
        periodic_indices = self.var_indices["periodic"]
        discretized_phi_indices = self.var_indices["discretized_phi"]
        var_indices = discretized_phi_indices + periodic_indices

        # method to concatenate sublists
        potential_sym = self.potential_symbolic

        # constructing the grids
        parameters = dict.fromkeys(
            ["θ" + str(index) for index in var_indices]
            + [var.name for var in self.external_flux_vars]
            + [var.name for var in self.param_vars]
        )

        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                parameters[var_name] = kwargs[var_name]
            elif isinstance(kwargs[var_name], int) or isinstance(
                kwargs[var_name], float
            ):
                parameters[var_name] = kwargs[var_name]
            else:
                raise AttributeError(
                    "Only float, Numpy ndarray or int assignments are allowed."
                )

        for var_name in parameters.keys():
            if parameters[var_name] is None:
                if var_name in [var.name for var in self.param_vars] + [
                    var.name for var in self.external_flux_vars
                ]:
                    parameters[var_name] = getattr(self, var_name)
                elif var_name in ["θ" + str(index) for index in var_indices]:
                    raise AttributeError(var_name + " is not set.")

        # creating a meshgrid for multiple dimensions
        sweep_vars = {}
        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                sweep_vars[var_name] = kwargs[var_name]
        if len(sweep_vars) > 1:
            sweep_vars.update(
                zip(
                    sweep_vars,
                    np.meshgrid(*[grid for grid in sweep_vars.values()]),
                )
            )
            for var_name in sweep_vars:
                parameters[var_name] = sweep_vars[var_name]

        potential_func = lambdify(parameters.keys(), potential_sym, "numpy")

        return potential_func(*parameters.values())

    def plot_potential(self, **kwargs):
        r"""
        Returns the plot of the potential for the circuit instance. Make sure to not set more than two variables in the instance.potential to a Numpy array, as the the code cannot plot with more than 3 dimensions.

        Parameters
        ----------
        :math:`\theta_i`:
            Numpy array or a Float, is the value set to the variable :math:`\theta_i` in the potential.
        """

        periodic_indices = self.var_indices["periodic"]
        discretized_phi_indices = self.var_indices["discretized_phi"]
        var_indices = discretized_phi_indices + periodic_indices

        # constructing the grids
        parameters = dict.fromkeys(
            ["θ" + str(index) for index in var_indices]
            + [var.name for var in self.external_flux_vars]
            + [var.name for var in self.param_vars]
        )

        sweep_vars = {}
        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                sweep_vars[var_name] = kwargs[var_name]
        if len(sweep_vars) > 1:
            sweep_vars.update(zip(sweep_vars, np.meshgrid(*list(sweep_vars.values()))))
            for var_name in sweep_vars:
                parameters[var_name] = sweep_vars[var_name]

        if len(sweep_vars) > 2:
            raise AttributeError(
                "Cannot plot with a dimension greater than 3; Only give a maximum of two grid inputs"
            )

        potential_energies = self.potential_energy(**kwargs)

        if len(sweep_vars) == 1:
            plot = plt.plot(*(list(sweep_vars.values()) + [potential_energies]))
            plt.xlabel(list(sweep_vars.keys())[0])
            plt.ylabel("Potential energy in GHz")

        if len(sweep_vars) == 2:
            plot = plt.contourf(*(list(sweep_vars.values()) + [potential_energies]))
            var_names = list(sweep_vars.keys())
            plt.xlabel(var_names[0])
            plt.ylabel(var_names[1])
            cbar = plt.colorbar()
            cbar.set_label("Potential energy in GHz")
        return plot

    ##################################################################
    ############# Functions for plotting wavefunction ################
    ##################################################################
    def plot_wavefunction(self, n=0, var_indices=(1,), mode="abs", eigensys=None):
        """
        Returns the plot of the wavefunction in the requested variables and for a specific eigen system calculation.

        Parameters
        ----------
        var_indices:
            A tuple containing the indices of the variables chosen to plot the wavefunction in. Should not have more than 2 entries.
        mode:
            "abs" or "real" or "imag" for absolute, real or imaginary parts of the wavefunction.
        eigensys:
            The object returned by the method instance.eigensys, is used to avoid the re-evaluation of the eigen systems if already evaluated.
        """
        dims = tuple(
            np.sort(var_indices) - 1
        )  # taking the var indices and identifying the dimensions.

        if eigensys is None:
            eigs, wf = self.eigensys()
        else:
            eigs, wf = eigensys

        cutoffs_dict = self.get_cutoffs()

        cutoff_list = []
        grids = []
        for cutoff_type in cutoffs_dict.keys():
            if "cutoff_n" in cutoff_type:
                cutoff_list.append([2 * k + 1 for k in cutoffs_dict[cutoff_type]])
                grids.append(
                    [list(range(-k, k + 1)) for k in cutoffs_dict[cutoff_type]]
                )
            elif "cutoff_phi" in cutoff_type:
                cutoff_list.append([k for k in cutoffs_dict[cutoff_type]])
                grids.append(
                    [
                        np.linspace(
                            self.discretized_phi_range[k][0],
                            self.discretized_phi_range[k][1],
                            cutoffs_dict[cutoff_type][i],
                        )
                        for i, k in enumerate(self.var_indices["discretized_phi"])
                    ]
                )
        # concatenating the sublists
        cutoff_list = [i for j in cutoff_list for i in j]
        grids = [i for j in grids for i in j]  # concatenating the sublists

        var_types = []

        for var_index in np.sort(var_indices):
            if var_index in self.var_indices["periodic"]:
                var_types.append("Charge in units of 2e, Variable:")
            else:
                var_types.append("Dimensionless Flux, Variable:")

        # selecting the n wave funciton according to the input
        if not self.hierarchical_diagonalization:
            wf_reshaped = wf[:, n].reshape(*cutoff_list)
        else:
            wf_reshaped = wf[n].full().reshape(*cutoff_list)

        if len(dims) > 2:
            raise AttributeError(
                "Cannot plot wavefunction in more than 2 dimensions. The number of dimensions should be less than 2."
            )

        wf_plot = (
            np.sum(
                eval("np." + mode + "(wf_reshaped)"),
                axis=tuple([i for i in range(len(cutoff_list)) if i not in dims]),
            )
        ).T

        if len(dims) == 1:
            if "Charge" in var_types[0]:
                plt.bar(grids[dims[0]], eval("np." + mode + "(wf_plot)"))
            else:
                plt.plot(grids[dims[0]], eval("np." + mode + "(wf_plot)"))
            plt.xlabel(var_types[0] + str(var_indices[0]))
        elif len(dims) == 2:
            x, y = np.meshgrid(grids[dims[0]], grids[dims[1]])
            plt.contourf(x, y, wf_plot)
            plt.xlabel(var_types[0] + str(np.sort(var_indices)[0]))
            plt.ylabel(var_types[1] + str(np.sort(var_indices)[1]))
            plt.colorbar()
        plt.title("Distribution of Wavefuntion along variables " + str(var_indices))

    ##################################################################
    ########### Functions from scqubits.core.qubit_base ##############
    ##################################################################
    def _evals_calc(self, evals_count: int) -> ndarray:

        if type(self) == Circuit and self.hierarchical_diagonalization:
            self.set_operators()
            self.hierarchical_diagonalization_func()
            self.main_subsystem.set_operators()
            for var_index in self.HD_indices:
                self.subsystems[var_index][0].set_operators()
            self.complete_hilbert_space()
            return self.hilbert_space.eigenvals(evals_count=evals_count)

        hamiltonian_mat = self.hamiltonian()
        if self.phi_basis == "sparse":
            evals = sparse.linalg.eigsh(
                hamiltonian_mat,
                return_eigenvectors=False,
                k=evals_count,
                v0=settings.RANDOM_ARRAY[: self.hilbertdim()],
                which="SA",
            )
        elif self.phi_basis == "harmonic":
            evals = sp.linalg.eigvalsh(
                hamiltonian_mat, subset_by_index=[0, evals_count - 1]
            )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:

        if type(self) == Circuit:
            if self.hierarchical_diagonalization:
                self.set_operators()
                self.hierarchical_diagonalization_func()
                self.main_subsystem.set_operators()
                for var_index in self.HD_indices:
                    self.subsystems[var_index][0].set_operators()
                self.complete_hilbert_space()
                return self.hilbert_space.eigensys(evals_count=evals_count)

        hamiltonian_mat = self.hamiltonian()
        if self.phi_basis == "sparse":
            evals, evecs = sparse.linalg.eigsh(
                hamiltonian_mat,
                return_eigenvectors=True,
                k=evals_count,
                which="SA",
                v0=settings.RANDOM_ARRAY[: self.hilbertdim()],
            )
        elif self.phi_basis == "harmonic":
            evals, evecs = sp.linalg.eigh(
                hamiltonian_mat,
                eigvals_only=False,
                subset_by_index=[0, evals_count - 1],
            )
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def matrixelement_table(
        self,
        operator: str,
        evecs: ndarray = None,
        evals_count: int = 6,
        filename: str = None,
        return_datastore: bool = False,
    ) -> ndarray:
        """Returns table of matrix elements for `operator` with respect to the
        eigenstates of the qubit. The operator is given as a string matching a class
        method returning an operator matrix. E.g., for an instance `trm` of Transmon,
        the matrix element table for the charge operator is given by
        `trm.op_matrixelement_table('n_operator')`. When `esys` is set to `None`,
        the eigensystem is calculated on-the-fly.

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix in
            qubit-internal basis.
        evecs:
            if not provided, then the necessary eigenstates are calculated on the fly
        evals_count:
            number of desired matrix elements, starting with ground state
            (default value = 6)
        filename:
            output file name
        return_datastore:
            if set to true, the returned data is provided as a DataStore object
            (default value = False)
        """
        if evecs is None:
            _, evecs = self.eigensys(evals_count=evals_count)
        operator_matrix = getattr(self, operator)
        table = get_matrixelement_table(operator_matrix, evecs)
        if filename or return_datastore:
            data_store = DataStore(
                system_params=self.get_initdata(), matrixelem_table=table
            )
        if filename:
            data_store.filewrite(filename)
        return data_store if return_datastore else table


class DataBuffer(Circuit):
    def __init__(self, parent: Circuit):
        self.parent = parent
        self.stored_cutoffs = parent.get_cutoffs()
        self.stored_param_vars = parent.param_vars
        self.stored_param_vals = parent.get_params()

        # operator dictionary to store operators
        self.stored_operators = None

        # setting class attributes
        self.store_operators()

    def did_cutoffs_change(self):
        """
        Method which checks if the cutoffs changed compared to the last time operators were stored in this instance of DataBuffer.
        """
        return all(
            flatten_list(self.stored_cutoffs.values())
            == flatten_list(self.parent.get_cutoffs())
        )

    def store_operators(self):
        """
        Method to store the operators from the parent, based on the current parameters.
        """
        self.parent.set_operators()
        self.stored_operators = self.parent.get_operators()

        if self.parent.phi_basis == "harmonic":
            position_operators = []
            momentum_operators = []
            sin_operators = []
            cos_operators = []
            for index in self.parent.var_indices["discretized_phi"]:
                θ = (
                    (
                        getattr(self.parent, "ad" + str(index))
                        + getattr(self.parent, "a" - str(index))
                    )
                    * self.parent.osc_lengths[index]
                    / (2 ** 0.5)
                )
                Q = (
                    1j
                    * (
                        getattr(self.parent, "ad" + str(index))
                        - getattr(self.parent, "a" - str(index))
                    )
                    / (self.parent.osc_lengths[index] * (2 ** 0.5))
                )
                sin_operators.append(sp.linalg.sinm(θ))
                cos_operators.append(sp.linalg.cosm(θ))
                position_operators.append(θ)
                momentum_operators.append(Q)

            self.stored_operators += (
                position_operators + momentum_operators + sin_operators + cos_operators
            )

    def update_buffer(self):
        """
        Updates the instance of DataBuffer only if the cutoffs of the parent were updated.
        """
        if self.did_cutoffs_change():
            self.store_operators()

    def retrieve_operators(self):
        """
        Retrieve operators from the DataBuffer instance or the parent depending on the Bool returned by did_cutoffs_change method.
        """
        if not self.did_cutoffs_change():
            return self.stored_operators
        else:
            self.update_buffer()
            return self.store_operators()


class CircuitSubsystem(Circuit, base.QubitBaseClass, serializers.Serializable):
    r"""
    Class to initiate a sub-system for a circuit just from a symbolic Hamiltonian.
    Circuit object can be initiated using:
        CircuitSubsystem(parent, H_sym)
    Parameters
    ----------
    parent:
        the Circuit object containing this subsystem.
    hamiltonian_symbolic:
        Symbolic Hamiltonian describing the system.
    """

    def __init__(self, parent, hamiltonian_symbolic):
        self.parent = parent
        self.hamiltonian_symbolic = hamiltonian_symbolic
        self.H_f = hamiltonian_symbolic

        # TODO what is this / why is it hardcoded to 10? - I followied this from the transmon.py file, as a starting parameter.
        self.truncated_dim = 10
        self._sys_type = type(self).__name__  # for object description
        # TODO we talked about this... did you not fix this meanwhile? -  I think this is somehow needed for some method call. I will need to test it further
        self.phi_basis = self.parent.phi_basis
        self.external_flux_vars = [
            var
            for var in self.parent.external_flux_vars
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.offset_charge_vars = [
            var
            for var in self.parent.offset_charge_vars
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.param_vars = [
            var
            for var in self.parent.param_vars
            if var in self.hamiltonian_symbolic.free_symbols
        ]

        for var in self.param_vars + self.offset_charge_vars + self.external_flux_vars:
            setattr(self, str(var), getattr(self.parent, str(var)))

        var_indices_list = []
        cutoffs = []
        for var in list(hamiltonian_symbolic.free_symbols):
            if "I" not in str(var):
                filtered_var = re.findall(
                    "[0-9]+", re.sub(r"ng_[0-9]+|Φ[0-9]+", "", str(var))
                )  # filtering offset charges and external flux
                if filtered_var == []:
                    continue
                else:
                    var_index = int(filtered_var[0])
                # var_index = (int(re.findall('[0-9]+', str(v))[0]))
                if var_index not in var_indices_list:
                    for cutoff_name in self.parent.cutoffs_list:
                        if str(var_index) in cutoff_name:
                            cutoffs.append(getattr(self.parent, cutoff_name))
                    var_indices_list.append(var_index)
        # setting some class attributes
        self.var_indices_list = var_indices_list
        self.var_indices = {}
        for var_type in self.parent.var_indices:
            self.var_indices[var_type] = [
                var_index
                for var_index in self.parent.var_indices[var_type]
                if var_index in var_indices_list
            ]

        self.cutoffs_list = []
        for var_type in self.var_indices.keys():
            if var_type == "periodic":
                for x, var_index in enumerate(self.var_indices["periodic"]):
                    setattr(
                        self,
                        "cutoff_n_" + str(var_index),
                        getattr(self.parent, "cutoff_n_" + str(var_index)),
                    )
                    self.cutoffs_list.append("cutoff_n_" + str(var_index))
            if var_type == "discretized_phi":
                for x, var_index in enumerate(self.var_indices["discretized_phi"]):
                    setattr(
                        self,
                        "cutoff_phi_" + str(var_index),
                        getattr(self.parent, "cutoff_phi_" + str(var_index)),
                    )
                    self.cutoffs_list.append("cutoff_phi_" + str(var_index))

        cutoffs_dict = {}
        for var_index in self.var_indices_list:
            for cutoff_name in self.parent.cutoffs_list:
                if str(var_index) in cutoff_name:
                    cutoffs_dict[var_index] = getattr(self.parent, cutoff_name)
        self.cutoffs_dict = cutoffs_dict
        self.discretized_phi_range = {
            i: self.parent.discretized_phi_range[i]
            for i in self.parent.discretized_phi_range
            if i in self.var_indices_list
        }

        self._id_str = (
            self._autogenerate_id_str()
        )  # Hilbert Space raises an error saying an instance of this class does not have an attribute _id_str

        self.set_vars()
        # self.set_operators()

        # def set_operators(self):
        #     Circuit.set_operators(self)
        # generate operators for harmonic oscillator basis
        if self.phi_basis == "harmonic":
            for var_index in self.var_indices["discretized_phi"]:
                H = self.hamiltonian_symbolic
                ECi = float(H.coeff("Q" + str(var_index) + "**2").cancel()) / 4
                ELi = float(H.coeff("θ" + str(var_index) + "**2").cancel()) * 2
                osc_freq = (8 * ELi * ECi) ** 0.5
                osc_length = (8.0 * ECi / ELi) ** 0.25
                ad_operator = op.creation(self.cutoffs_dict[var_index])
                a_operator = op.annihilation(self.cutoffs_dict[var_index])
                x_operator = (ad_operator + a_operator) * osc_length / (2 ** 0.5)
                position_operator = self._kron_operator(x_operator, var_index)
                momentum_operator = self._kron_operator(
                    1j * (ad_operator - a_operator) / (osc_length * 2 ** 0.5), var_index
                )
                sin_operator = self._kron_operator(
                    sp.linalg.sinm(x_operator), var_index
                )
                cos_operator = self._kron_operator(
                    sp.linalg.cosm(x_operator), var_index
                )

                setattr(self, "θ" + str(var_index), position_operator)
                setattr(self, "θc" + str(var_index), cos_operator)
                setattr(self, "θs" + str(var_index), sin_operator)
                setattr(self, "Q" + str(var_index), momentum_operator)

        if self.phi_basis == "sparse":
            grids = {}
            for var_index in self.var_indices["discretized_phi"]:
                grids[var_index] = discretization.Grid1d(
                    self.discretized_phi_range[var_index][0],
                    self.discretized_phi_range[var_index][1],
                    self.cutoffs_dict[var_index],
                )
            for var_index in self.var_indices["discretized_phi"]:
                setattr(
                    self,
                    "θc" + str(var_index),
                    self._cos_phi(grids[var_index]),
                )
                setattr(
                    self,
                    "θs" + str(var_index),
                    self._cos_phi(grids[var_index]),
                )

    def generate_symbols(self, prefix: str, index_type: str) -> List[symbols]:
        return [
            symbols(prefix + str(index))
            for index in list_intersection(
                self.parent.var_indices[index_type], self.var_indices_list
            )
        ]


# function to find the differences in the energy levels
def energy_split(levels):  # gives the energy splits given the energy levels
    splits = []
    for i in range(1, len(levels)):
        splits.append(levels[i] - levels[i - 1])
    return splits


# example input strings
def example_circuit(qubit):
    """
    Returns example input strings for AnalyzeQCircuit and CustomQCircuit for some of the popular qubits.

    Parameters
    ----------
    qubit:
        "fluxonium" or "transmon" or "zero_pi" or "cos2phi" chosing the respective example input strings.
    """

    # example input strings for popular qubits
    fluxonium = "nodes: 2\nbranches:\nJJ	1,2	Ej	Ecj\nL	1,2	El\nC	1,2	Ec"

    transmon = "nodes: 2\nbranches:\nC\t1,2\tEc\nJJ\t1,2\tEj\tEcj\n"

    cos2phi = "nodes: 4\nbranches:\nC\t1,3\tEc\nJJ\t1,2\tEj\tEcj\nJJ\t3,4\tEj\tEcj\nL\t1,4\tEl\nL\t2,3\tEl\n\n"

    zero_pi = "nodes: 4\nbranches:\nJJ\t1,2\tEj\tEcj\nL\t2,3\tEl\nJJ\t3,4\tEj\tEcj\nL\t4,1\tEl\nC\t1,3\tEc\nC\t2,4\tEc\n"

    if qubit == "transmon":
        return transmon
    elif qubit == "cos2phi":
        return cos2phi
    elif qubit == "zero_pi":
        return zero_pi
    elif qubit == "fluxonium":
        return fluxonium
    else:
        raise (AttributeError()("Qubit not available or invalid input."))
