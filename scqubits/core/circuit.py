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


from multiprocessing.spawn import old_main_modules
from sys import settrace
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
from attr import attrib
from matplotlib.text import OffsetFrom
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
from scqubits.utils.misc import list_intersection, flatten_list, flatten_list_recursive


from scqubits.utils.spectrum_utils import (
    get_matrixelement_table,
    identity_wrap,
    order_eigensystem,
    standardize_phases,
    standardize_sign,
    convert_evecs_to_ndarray,
)

# Causing a circular import
# if TYPE_CHECKING:
#     from scqubits.core.symboliccircuit import Circuit


def generate_default_trunc_dims(index_list):
    trunc_dims = {}
    for x, subsystem_indices in enumerate(index_list):
        if subsystem_indices == flatten_list_recursive(subsystem_indices):
            trunc_dims[x] = [10, {}]
        else:
            trunc_dims[x] = [
                50,
                generate_default_trunc_dims(subsystem_indices),
            ]
    return trunc_dims


def get_trailing_number(input_str: str) -> int:
    """
    Retuns the number trailing a string given as input. Example:
        $ get_trailing_number("a23")
        $ 23

    Parameters
    ----------
    input_str : str
        String which trails any number

    Returns
    -------
    int
        returns the trailing integer as int, else returns None
    """
    match = re.search(r"\d+$", input_str)
    return int(match.group()) if match else None


class Circuit(base.QubitBaseClass, serializers.Serializable):
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
        "discretized" or "harmonic": Choose whether to use discretized phi or harmonic oscillator basis for extended variables.
    hierarchical_diagonalization:
        Boolean which indicates if the HilbertSpace from scqubits is used for simplification
    """

    def __init__(
        self,
        circuit_data: dict,
        initiate_sym_calc: bool = True,
        phi_basis: str = "discretized",
        hierarchical_diagonalization: bool = True,
        HD_indices=[],
        HD_trunc_dims=[],
        truncated_dim: int = None,
    ):
        # attribute to check if this is a child circuit instance
        self.is_child = False
        # inheriting all the attributes from SymbolicCircuit instance
        self._sys_type = type(self).__name__
        # defining additional class properties

        self.vars = None
        self.external_flux = []

        # setting truncated_dim for calculating energy dispersion
        self.truncated_dim = truncated_dim

        # setting default grids for plotting
        self._default_grid_phi: discretization.Grid1d = discretization.Grid1d(
            -6 * np.pi, 6 * np.pi, 200
        )

        self.discretized_phi_range = {}
        self.cutoffs_list: List[int] = []
        self.phi_basis = phi_basis
        self.hierarchical_diagonalization = hierarchical_diagonalization
        self.HD_indices = HD_indices
        self.HD_trunc_dims = HD_trunc_dims

        self.__dict__.update(circuit_data)

        self._id_str = (
            self._autogenerate_id_str()
        )  # Hilbert Space raises an error saying an instance of this class does not have an attribute _id_str

        # Hamiltonian function
        if initiate_sym_calc:
            if hasattr(self, "parent"):
                self.initiate_child()
            else:
                self.initiate_circuit()

    def __repr__(self) -> str:
        return self._id_str

    # constructor to initiate using a CustomQCircuit object
    @classmethod
    def from_symbolic_hamiltonian(
        cls,
        parent,
        hamiltonian_symbolic,
        HD_indices,
        truncated_dim=None,
        HD_trunc_dims=[],
    ):
        HD_indices = HD_indices
        truncated_dim = truncated_dim
        HD_trunc_dims = HD_trunc_dims

        is_child = True
        parent = parent
        hamiltonian_symbolic = hamiltonian_symbolic
        H_f = hamiltonian_symbolic

        # _sys_type = type(self).__name__  # for object description
        # TODO we talked about this... did you not fix this meanwhile? -  I think this is somehow needed for some method call. I will need to test it further
        phi_basis = parent.phi_basis
        external_flux_vars = [
            var
            for var in parent.external_flux_vars
            if var in hamiltonian_symbolic.free_symbols
        ]
        offset_charge_vars = [
            var
            for var in parent.offset_charge_vars
            if var in hamiltonian_symbolic.free_symbols
        ]
        param_vars = [
            var for var in parent.param_vars if var in hamiltonian_symbolic.free_symbols
        ]

        # for var in param_vars + offset_charge_vars + external_flux_vars:
        #     setattr(self, str(var), getattr(parent, str(var)))

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
                    for cutoff_name in parent.cutoffs_list:
                        if str(var_index) in cutoff_name:
                            cutoffs.append(getattr(parent, cutoff_name))
                    var_indices_list.append(var_index)
        # setting some class attributes
        var_indices_list = var_indices_list
        var_indices = {}
        for var_type in parent.var_indices:
            var_indices[var_type] = [
                var_index
                for var_index in parent.var_indices[var_type]
                if var_index in var_indices_list
            ]

        cutoffs_list = []
        for var_type in var_indices.keys():
            if var_type == "periodic":
                for x, var_index in enumerate(var_indices["periodic"]):
                    cutoffs_list.append("cutoff_n_" + str(var_index))
            if var_type == "extended":
                for x, var_index in enumerate(var_indices["extended"]):
                    cutoffs_list.append("cutoff_phi_" + str(var_index))

        cutoffs_dict = {}
        for var_index in var_indices_list:
            for cutoff_name in parent.cutoffs_list:
                if str(var_index) in cutoff_name:
                    cutoffs_dict[var_index] = getattr(parent, cutoff_name)
        cutoffs_dict = cutoffs_dict
        discretized_phi_range = {
            i: parent.discretized_phi_range[i]
            for i in parent.discretized_phi_range
            if i in var_indices_list
        }

        # self.set_vars()
        hierarchical_diagonalization = (
            HD_indices != [] and HD_indices != flatten_list_recursive(HD_indices)
        )

        circuit_data = {
            "var_indices": var_indices,
            "var_indices_list": var_indices_list,
            "external_flux_vars": external_flux_vars,
            "offset_charge_vars": offset_charge_vars,
            "hamiltonian_symbolic": hamiltonian_symbolic,
            "H_f": H_f,
            "param_vars": param_vars,
            "cutoffs_dict": cutoffs_dict,
            "cutoffs_list": cutoffs_list,
            "parent": parent,
            "discretized_phi_range": discretized_phi_range,
            "is_child": is_child,
        }

        return cls(
            circuit_data,
            initiate_sym_calc=True,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
            HD_indices=HD_indices,
            HD_trunc_dims=HD_trunc_dims,
            truncated_dim=truncated_dim,
        )

    @classmethod
    def from_CustomQCircuit(
        cls,
        symbolic_circuit: SymbolicCircuit,
        initiate_sym_calc=True,
        hierarchical_diagonalization: bool = False,
        phi_basis: str = "discretized",
        HD_indices=None,
        HD_trunc_dims=None,
        truncated_dim: int = None,
    ):
        """
        Initialize AnalyzeQCircuit using an instance of CustomQCircuit.

        Parameters
        ----------
        circuit:
            An instance of CustomQCircuit
        """
        circuit_data = {
            "var_indices": symbolic_circuit.var_indices,
            "external_flux_vars": symbolic_circuit.external_flux_vars,
            "offset_charge_vars": symbolic_circuit.offset_charge_vars,
            "hamiltonian_symbolic": symbolic_circuit.hamiltonian_symbolic,
            "param_vars": symbolic_circuit.param_vars,
            "branches": symbolic_circuit.branches,
            "nodes": symbolic_circuit.nodes,
            "lagrangian_symbolic": symbolic_circuit.lagrangian_symbolic,
            "hamiltonian_symbolic": symbolic_circuit.hamiltonian_symbolic,
            "lagrangian_node_vars": symbolic_circuit.lagrangian_node_vars,
        }
        circuit_data["symbolic_circuit"] = symbolic_circuit

        return cls(
            circuit_data,
            initiate_sym_calc=initiate_sym_calc,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
            HD_indices=HD_indices,
            HD_trunc_dims=HD_trunc_dims,
            truncated_dim=truncated_dim,
        )

    @classmethod
    def from_input_string(
        cls,
        input_string: str,
        phi_basis="discretized",
        basis_completion="simple",
        initiate_sym_calc=True,
        hierarchical_diagonalization: bool = False,
        HD_indices=None,
        HD_trunc_dims=None,
        truncated_dim: int = None,
    ):
        """
        Creates an instance of Circuit from a circuit described using an input string.

        Parameters
        ----------
        input_string : str
            string describing the graph of a circuit
        phi_basis : str, optional
            Choses the kind of basis used to construct the operators for extended variables. Can be "discretized" or "harmonic".
        basis : str, optional
            _description_, by default "simple"
        initiate_sym_calc : bool, optional
            _description_, by default True
        hierarchical_diagonalization : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """

        circuit = SymbolicCircuit.from_input_string(
            input_string, basis_completion=basis_completion, initiate_sym_calc=True
        )

        return cls.from_CustomQCircuit(
            circuit,
            initiate_sym_calc=initiate_sym_calc,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
            HD_indices=HD_indices,
            HD_trunc_dims=HD_trunc_dims,
            truncated_dim=truncated_dim,
        )

    @classmethod
    def from_input_file(
        cls,
        filename: str,
        phi_basis="discretized",
        basis_completion="simple",
        initiate_sym_calc=True,
        hierarchical_diagonalization: bool = False,
        HD_indices=None,
        HD_trunc_dims=None,
        truncated_dim: int = None,
    ):

        circuit = SymbolicCircuit.from_input_file(
            filename, basis_completion=basis_completion, initiate_sym_calc=True
        )

        return cls.from_CustomQCircuit(
            circuit,
            initiate_sym_calc=initiate_sym_calc,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
            HD_indices=HD_indices,
            HD_trunc_dims=HD_trunc_dims,
            truncated_dim=truncated_dim,
        )

    # def set_param(self, param_name, value):
    #     if self.phi_basis == "harmonic":
    #         setattr(self, "_" + param_name, value)
    #         self.hierarchical_diagonalization_func()
    #         self.set_operators()
    #         self.complete_hilbert_space()

    def initiate_child(self):
        for var in self.param_vars + self.offset_charge_vars + self.external_flux_vars:
            setattr(self, str(var), getattr(self.parent, str(var)))

        for cutoff_str in self.cutoffs_list:
            setattr(self, cutoff_str, getattr(self.parent, cutoff_str))

        self._init_params = (
            [param.name for param in self.param_vars]
            + [flux.name for flux in self.external_flux_vars]
            + [offset_charge.name for offset_charge in self.offset_charge_vars]
            + self.cutoffs_list
        )

        self.set_vars()
        if self.hierarchical_diagonalization:
            self.hierarchical_diagonalization_func()
            self.set_operators()
            self.complete_hilbert_space()
        else:
            self.set_operators()

    def initiate_circuit(
        self,
        transformation_matrix=None,
        HD_indices=None,
        HD_trunc_dims=None,
        closure_branches=None,
    ):
        """
        Method to initialize the Circuit instance and initialize all the attributes needed before it can be passed on to AnalyzeQCircuit.

        Parameters
        ----------
        transformation_matrix:
            Takes an ndarray and is used to set an alternative transformation matrix than the one generated by the method variable_transformation_matrix.
        """
        HD_indices = HD_indices or self.HD_indices
        HD_trunc_dims = HD_trunc_dims or self.HD_trunc_dims

        self.hierarchical_diagonalization = True if HD_indices is not None else False

        self.symbolic_circuit.initiate_symboliccircuit(
            transformation_matrix=transformation_matrix,
            closure_branches=closure_branches,
        )
        self.__dict__.update(self.symbolic_circuit.__dict__)

        # removing any of the old cutoffs
        old_cutoffs = []
        for attr in self.__dict__:
            if "cutoff_" in attr:
                old_cutoffs.append(attr)
        for attr in old_cutoffs:
            delattr(self, attr)

        # initiating the class properties
        self.cutoffs_list = []
        for var_type in self.var_indices.keys():
            if var_type == "periodic":
                for x, var_index in enumerate(self.var_indices["periodic"]):
                    cutoff = (
                        5
                        if not hasattr(self, "parent")
                        else getattr(self.parent, "cutoff_n_" + str(var_index))
                    )
                    setattr(self, "cutoff_n_" + str(var_index), cutoff)
                    self.cutoffs_list.append("cutoff_n_" + str(var_index))
            if var_type == "extended":
                for x, var_index in enumerate(self.var_indices["extended"]):
                    cutoff = (
                        30
                        if not hasattr(self, "parent")
                        else getattr(self.parent, "cutoff_phi_" + str(var_index))
                    )
                    setattr(self, "cutoff_phi_" + str(var_index), cutoff)
                    self.cutoffs_list.append("cutoff_phi_" + str(var_index))

        # default values for the parameters
        for x, param in enumerate(self.param_vars):
            # storing the parameter values into a class attribute.
            # if harmonic oscillator basis is used, param vars become class properties.
            setattr(self, param.name, self.params_init_vals[x])

            # param_gettr = lambda self, param_name=param.name: getattr(self,  "_" + param_name)
            # param_settr = lambda self, value, param_name=param.name: self.set_param(param_name, value)
            # setattr(self.__class__, param.name, property(fget=param_gettr, fset=param_settr))

        # setting the ranges for floux ranges used for discrete phi vars
        for v in self.var_indices["extended"]:
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
            if HD_indices == None:
                self.HD_indices = [
                    self.var_indices["periodic"] + self.var_indices["extended"]
                ]
            else:
                self.HD_indices = HD_indices

            if HD_trunc_dims == None:
                raise Exception(
                    "The truncated dimensions attribute for hierarchical diagonalization is not set."
                )
            else:
                self.HD_trunc_dims = HD_trunc_dims
            self.hamiltonian_function()
            self.hierarchical_diagonalization_func()

        # initilizing attributes for operators
        self.set_operators()

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

        systems_sym = []
        interaction_sym = []

        for subsys_index_list in self.HD_indices:
            subsys_index_list = flatten_list_recursive(subsys_index_list)
            expr_dict = H.as_coefficients_dict()
            terms_list = list(expr_dict.keys())

            H_sys = 0 * symbols("x")
            H_int = 0 * symbols("x")
            for term in terms_list:
                term_var_indices = []
                for var in term.free_symbols:
                    # remove any branch parameters or flux and offset charge symbols
                    if (
                        "Φ" not in str(var)
                        and "ng" not in str(var)
                        and len(list_intersection(self.param_vars, [var])) == 0
                    ):
                        index = get_trailing_number(str(var))
                        if index not in term_var_indices and index is not None:
                            term_var_indices.append(index)

                if len(set(term_var_indices) - set(subsys_index_list)) == 0:
                    H_sys = H_sys + expr_dict[term] * term

                if (
                    len(set(term_var_indices) - set(subsys_index_list)) > 0
                    and len(set(term_var_indices) & set(subsys_index_list)) > 0
                ):
                    H_int = H_int + expr_dict[term] * term
            systems_sym.append(H_sys)
            interaction_sym.append(H_int)
            H = H - H_sys - H_int  # removing the terms added to a subsystem

        # storing data in class attributes
        self.subsystems_sym = dict(
            zip(
                range(len(self.HD_indices)),
                [
                    [systems_sym[index], interaction_sym[index]]
                    for index in range(len(self.HD_indices))
                ],
            )
        )

        self.subsystems = dict(
            zip(
                range(len(self.HD_indices)),
                [
                    Circuit.from_symbolic_hamiltonian(
                        self,
                        systems_sym[index],
                        HD_indices=self.HD_indices[index],
                        truncated_dim=self.HD_trunc_dims[index][0],
                        HD_trunc_dims=self.HD_trunc_dims[index][1],
                    )
                    for index in range(len(self.HD_indices))
                ],
            )
        )

    def get_subsystem_index(self, var_index):
        """
        Returns the index of the subsystem to which the var_index belongs to

        :param var_index: _description_
        :type var_index: _type_
        :return: int showing the subsystem
        :rtype: _type_
        """
        for index, subsystem_indices in enumerate(self.HD_indices):
            if var_index in flatten_list_recursive(subsystem_indices):
                return index

    def identity_wrap_operator(self, system, operator_symbol):

        operator = getattr(system, operator_symbol.name)
        subsystem_index = system.get_subsystem_index(
            get_trailing_number(operator_symbol.name)
        )
        subsystem = system.subsystems[subsystem_index]

        operator_identity_wrapped = identity_wrap(
            operator, subsystem, system.hilbert_space.subsys_list  # , evecs=evecs_bare
        )
        return operator_identity_wrapped.full()

    def complete_hilbert_space(self):
        hilbert_space = HilbertSpace(
            [self.subsystems[i] for i in range(len(self.HD_indices))]
        )

        # Adding interactions using the symbolic interaction term
        for sys_index in range(len(self.HD_indices)):
            interaction = self.subsystems_sym[sys_index][1].expand()
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
            for i in self.var_indices["extended"]:
                interaction = interaction.replace(
                    sympy.cos(1.0 * symbols("θ" + str(i))), symbols("θc" + str(i))
                ).replace(
                    sympy.sin(1.0 * symbols("θ" + str(i))), symbols("θs" + str(i))
                )

            expr_dict = interaction.as_coefficients_dict()
            terms_str = list(expr_dict.keys())
            # coeff_str = list(expr_dict.values())

            for i, term in enumerate(terms_str):
                coefficient = expr_dict[term]

                # adding external flux, offset charge and branch parameters to coefficient
                for var in term.free_symbols:
                    if "Φ" in str(var) or "ng" in str(var) or var in self.param_vars:
                        coefficient = coefficient * getattr(self, str(var))

                operator_symbols = [
                    var
                    for var in term.free_symbols
                    if (("Φ" not in str(var)) and ("ng" not in str(var)))
                    and (var not in self.param_vars)
                ]

                sys_op_dict = {index: [] for index in range(len(self.HD_indices))}
                for var in operator_symbols:
                    var_index = get_trailing_number(str(var))
                    subsystem_index = self.get_subsystem_index(var_index)
                    if "I" not in str(var):
                        operator = getattr(self.subsystems[subsystem_index], str(var))
                        subsystem = self.subsystems[subsystem_index]
                        if subsystem.hierarchical_diagonalization and hasattr(
                            subsystem, "parent"
                        ):
                            operator = self.identity_wrap_operator(subsystem, var)

                        sys_op_dict[subsystem_index].append(operator)
                    else:
                        sys_op_dict[0].append(self.subsystems[0]._identity())

                operator_dict = {}

                for index in range(len(self.HD_indices)):
                    for op_index, operator in enumerate(sys_op_dict[index]):
                        operator_dict["op" + str(len(operator_dict) + 1)] = (
                            operator,
                            self.subsystems[index],
                        )
                hilbert_space.add_interaction(
                    g=float(coefficient), **operator_dict, check_validity=False
                )

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
        y_symbols = [symbols("θ" + str(i)) for i in self.var_indices["extended"]]
        p_symbols = [symbols("Q" + str(i)) for i in self.var_indices["extended"]]

        if self.phi_basis == "discretized":

            ps_symbols = [symbols("Qs" + str(i)) for i in self.var_indices["extended"]]
            sin_symbols = [symbols("θs" + str(i)) for i in self.var_indices["extended"]]
            cos_symbols = [symbols("θc" + str(i)) for i in self.var_indices["extended"]]

        elif self.phi_basis == "harmonic":

            a_symbols = [symbols("a" + str(i)) for i in self.var_indices["extended"]]
            ad_symbols = [symbols("ad" + str(i)) for i in self.var_indices["extended"]]
            Nh_symbols = [symbols("Nh" + str(i)) for i in self.var_indices["extended"]]
            pos_symbols = [symbols("θ" + str(i)) for i in self.var_indices["extended"]]
            sin_symbols = [symbols("θs" + str(i)) for i in self.var_indices["extended"]]
            cos_symbols = [symbols("θc" + str(i)) for i in self.var_indices["extended"]]
            momentum_symbols = [
                symbols("Q" + str(i)) for i in self.var_indices["extended"]
            ]

            extended_symbols = (
                a_symbols
                + ad_symbols
                + Nh_symbols
                + pos_symbols
                + sin_symbols
                + cos_symbols
                + momentum_symbols
            )

        # setting the attribute self.vars
        self.vars = {
            "periodic": {
                "sin": periodic_symbols_sin,
                "cos": periodic_symbols_cos,
                "number": periodic_symbols_n,
            },
            "identity": [symbols("I")],
        }

        if self.phi_basis == "discretized":
            self.vars["extended"] = {
                "position": y_symbols,
                "momentum": p_symbols,
                "momentum_squared": ps_symbols,
                "sin": sin_symbols,
                "cos": cos_symbols,
            }
        elif self.phi_basis == "harmonic":
            self.vars["extended"] = {
                "annihilation": a_symbols,
                "creation": ad_symbols,
                "number": Nh_symbols,
                "position": pos_symbols,
                "momentum": momentum_symbols,
                "sin": sin_symbols,
                "cos": cos_symbols,
            }

    def hamiltonian_function(self):
        """
        Outputs the function using lambdify in Sympy, which returns a Hamiltonian matrix by using the circuit attributes set in either the input file or the instance attributes.
        """
        H = (
            self.hamiltonian_symbolic.expand()
        )  # this expand method is critical to be applied, otherwise the replacemnt of the variables p^2 with ps2 will not be successful and the results would be incorrect

        ######## shifting the harmonic oscillator potential to the point of external fluxes #############
        flux_shift_vars = {}
        for var_index in self.var_indices["extended"]:
            if H.coeff("θ" + str(var_index)) != 0:
                flux_shift_vars[var_index] = symbols("Δθ" + str(var_index))
                H = H.replace(
                    symbols("θ" + str(var_index)),
                    symbols("θ" + str(var_index)) + flux_shift_vars[var_index],
                )  # substituting the flux offset variable offsets to collect the coefficients later
        H = H.expand()

        flux_shift_equations = [
            H.coeff("θ" + str(var_index)).subs(
                [("θ" + str(i), 0) for i in self.var_indices["extended"]]
            )
            for var_index in flux_shift_vars.keys()
        ]  # finding the coefficients of the linear terms

        flux_shifts = sm.linsolve(
            flux_shift_equations, tuple(flux_shift_vars.values())
        )  # solving for the flux offsets

        if len(flux_shifts) != 0:
            flux_shifts = list(list(flux_shifts)[0])
        else:
            flux_shifts = []

        flux_shifts_dict = dict(zip(self.var_indices["extended"], list(flux_shifts)))

        H = H.subs(
            [
                (symbols("Δθ" + str(var_index)), flux_shifts_dict[var_index])
                for var_index in flux_shifts_dict.keys()
            ]
        )  # substituting the flux offsets to remove the linear terms
        H = H.subs(
            [(var, 0) for var in flux_shift_vars.values()]
        )  # removing the constants from the Hamiltonian

        flux_shifts_dict.update(
            {
                var_index: 0
                for var_index in self.var_indices["extended"]
                if var_index not in flux_shifts_dict
            }
        )
        # remove constants from Hamiltonian
        H = H - H.as_coefficients_dict()[1]
        H = H.expand()
        #############################

        # marking the sin and cos terms of the periodic variables with different symbols
        if len(self.var_indices["periodic"]) > 0:
            H = sympy.expand_trig(H).expand()

        for i in self.var_indices["periodic"]:
            H = H.replace(
                sympy.cos(1.0 * symbols("θ" + str(i))), symbols("θc" + str(i))
            ).replace(sympy.sin(1.0 * symbols("θ" + str(i))), symbols("θs" + str(i)))

        if self.phi_basis == "discretized":

            # marking the squared momentum operators with a separate symbol
            for i in self.var_indices["extended"]:
                H = H.replace(symbols("Q" + str(i)) ** 2, symbols("Qs" + str(i)))

        elif self.phi_basis == "harmonic":
            H = sympy.expand_trig(H).expand()

            for i in self.var_indices["extended"]:
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
        if (
            hasattr(self, "hierarchical_diagonalization")
            and self.hierarchical_diagonalization
        ):
            subsystem_index = self.get_subsystem_index(index)
            var_index_list = flatten_list_recursive(self.HD_indices[subsystem_index])
        else:
            var_index_list = self.var_indices["periodic"] + self.var_indices["extended"]

        var_index_list.sort()  # important to make sure that right cutoffs are chosen
        cutoff_dict = self.get_cutoffs()

        # if len(self.var_indices["periodic"]) != len(cutoff_dict["cutoff_n"]) or len(
        #     self.var_indices["extended"]
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
                self.var_indices["periodic"] + self.var_indices["extended"],
                cutoffs,
            )
        )
        cutoff_list = [
            cutoffs_index_dict[i] for i in var_index_list
        ]  # selecting the cutoffs present in

        if self.phi_basis == "discretized":
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
            if self.phi_basis == "discretized":
                return sparse.csc_matrix(operator)
            elif self.phi_basis == "harmonic":
                return operator

    def _change_sparsity(self, x):
        if self.phi_basis == "harmonic":
            return x.toarray()
        elif self.phi_basis == "discretized":
            return x

    # Identity Operator
    def _identity(self):
        """
        Returns the Identity operator for the entire Hilber space of the circuit.
        """
        if (
            hasattr(self, "hierarchical_diagonalization")
            and self.hierarchical_diagonalization
        ):
            return None
        dim = self.hilbertdim()
        if self.phi_basis == "discretized":
            op = sparse.identity(dim)
            return op.tocsc()
        elif self.phi_basis == "harmonic":
            return np.identity(dim)

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
        normal_vars = self.vars["extended"]

        index_list = [j for i in list(self.var_indices.values()) for j in i]
        cutoff_list = [j for i in list(self.get_cutoffs().values()) for j in i]
        cutoffs = dict(zip(index_list, cutoff_list))

        grids = {}
        for i in self.var_indices["extended"]:
            grids[i] = discretization.Grid1d(
                self.discretized_phi_range[i][0],
                self.discretized_phi_range[i][1],
                cutoffs[i],
            )

        # constructing the operators for extended variables

        if self.phi_basis == "discretized":
            extended_operators = {
                "position": [],
                "momentum": [],
                "momentum_squared": [],
                "cos": [],
                "sin": [],
            }
            for v in normal_vars["position"]:  # position operators
                index = int(get_trailing_number(v.name))
                phi_operator = self._phi_operator(grids[index])
                extended_operators["position"].append(
                    self._kron_operator(phi_operator, index)
                )
                extended_operators["sin"].append(
                    self._kron_operator(self._sin_phi(grids[index]), index)
                )
                extended_operators["cos"].append(
                    self._kron_operator(self._cos_phi(grids[index]), index)
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

            # calculate oscillator frequencies and use harmonic oscillator basis
            osc_lengths = {}
            osc_freqs = {}
            extended_operators = {
                "annihilation": [],
                "creation": [],
                "number": [],
                "position": [],
                "momentum": [],
                "sin": [],
                "cos": [],
            }
            for var_index in self.var_indices["extended"]:
                ECi = float(H.coeff("Q" + str(var_index) + "**2").cancel()) / 4
                ELi = float(H.coeff("θ" + str(var_index) + "**2").cancel()) * 2
                osc_freqs[var_index] = (8 * ELi * ECi) ** 0.5
                osc_lengths[var_index] = (8.0 * ECi / ELi) ** 0.25
                ad_operator = op.creation(cutoffs_dict[var_index])
                a_operator = op.annihilation(cutoffs_dict[var_index])
                extended_operators["creation"].append(
                    self._kron_operator(a_operator, var_index)
                )
                extended_operators["annihilation"].append(
                    self._kron_operator(ad_operator, var_index)
                )
                extended_operators["number"].append(
                    self._kron_operator(op.number(cutoffs_dict[var_index]), var_index)
                )
                x_operator = (
                    (ad_operator + a_operator) * osc_lengths[var_index] / (2 ** 0.5)
                )
                extended_operators["position"].append(
                    self._kron_operator(x_operator, var_index)
                )
                extended_operators["momentum"].append(
                    self._kron_operator(
                        1j
                        * (ad_operator - a_operator)
                        / (osc_lengths[var_index] * 2 ** 0.5),
                        var_index,
                    )
                )

                extended_operators["sin"].append(
                    self._kron_operator(
                        sp.linalg.sinm(x_operator),
                        var_index,
                    )
                )

                extended_operators["cos"].append(
                    self._kron_operator(
                        sp.linalg.cosm(x_operator),
                        var_index,
                    )
                )

            self.osc_lengths = osc_lengths
            self.osc_freqs = osc_freqs

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
            "extended": extended_operators,
            "identity": [self._identity()],
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

    def get_operators(self, return_dict=False):
        """
        Returns a list of operators which can be given as an argument to self.H_func. These operators are not calculated again and are fetched directly from the circuit attibutes. Use set_attributes instead if the paramaters, expecially cutoffs, are changed.
        """
        variable_symbols_list = flatten_list(
            self.vars["periodic"].values()
        ) + flatten_list(self.vars["extended"].values())
        operator_list = []
        for operator in variable_symbols_list:
            operator_list.append(getattr(self, operator.name))

        # adding the identity operator
        operator_list.append(self._identity())

        if return_dict:
            variable_symbols_list.append(symbols("I"))
            operator_dict = dict(zip(variable_symbols_list, operator_list))
            return operator_dict

        return operator_list

    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

        return {}

    def set_operators(self):
        """
        Sets the operator attributes of the circuit with new operators calculated using the paramaters set in the circuit attributes. Returns a list of operators similar to the method get_operators.
        Returns nothing.
        """

        variable_symbols_list = (
            flatten_list(self.vars["periodic"].values())
            + flatten_list(self.vars["extended"].values())
            + self.vars["identity"]
        )

        # if not self.is_child:
        ops = self.circuit_operators()
        operator_list = flatten_list(ops["periodic"].values()) + flatten_list(
            ops["extended"].values()
        )

        operator_list = operator_list + [self._identity()]

        for x, operator in enumerate(variable_symbols_list):
            setattr(self, operator.name, operator_list[x])

        # if (
        #     hasattr(self, "hierarchical_diagonalization")
        #     and self.hierarchical_diagonalization
        # ):
        #     for index in range(len(self.HD_indices)):
        #         self.subsystems[index][0].set_operators()

    ##################################################################
    ############# Functions for eigen values and matrices ############
    ##################################################################
    def is_mat_mul_replacement_necessary(self, term):
        return (
            set(self.var_indices["extended"])
            & set([get_trailing_number(str(i)) for i in term.free_symbols])
        ) and "*" in str(term)

    def replace_mat_mul_operator(self, term):

        if not self.is_mat_mul_replacement_necessary(term):
            return str(term)

        if self.phi_basis == "discretized":
            var_indices = [get_trailing_number(str(i)) for i in term.free_symbols]
            if len(set(var_indices) & set(self.var_indices["extended"])) > 1:
                term_string = str(term)  # .replace("*", "@")
            else:
                return str(term)
        elif self.phi_basis == "harmonic":
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
                    if get_trailing_number(operator) in self.var_indices["extended"]:
                        new_string_list.append(
                            "matrix_power(" + operator + "," + exponents[x] + ")"
                        )
                    else:
                        new_string_list.append(operator + "**" + exponents[x])
                term_string = "*".join(new_string_list)
            else:
                term_string = str(term)

            # replace * with @ in the entire term
            if len(term.free_symbols) > 1:
                term_string = re.sub(
                    r"[^*]\K\*{1}(?!\*)", "@", term_string, re.MULTILINE
                )

        # # replace * with @ in the entire term
        # term_string = re.sub(r"[^*]\K\*{1}(?!\*)", "@", term_string, re.MULTILINE)

        # # replace @ with * for all the multiplications where constants are involved
        # term_string = re.sub(
        #     r"(?<=[0-9])\@(?=[0-9])|(\@(?=[0-9]))|((?<=[0-9])\@)",
        #     "*",
        #     term_string,
        #     re.MULTILINE,
        # )
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
        H = H.subs("I", 1)

        # replace the extended degrees of freedom with harmonic oscillators
        for var_index in self.var_indices["extended"]:
            ECi = float(H.coeff("Q" + str(var_index) + "**2").cancel()) / 4
            ELi = float(H.coeff("θ" + str(var_index) + "**2").cancel()) * 2
            osc_freq = (8 * ELi * ECi) ** 0.5
            osc_length = (8.0 * ECi / ELi) ** 0.25
            H = (
                (
                    H
                    - ECi * 4 * symbols("Q" + str(var_index)) ** 2
                    - ELi / 2 * symbols("θ" + str(var_index)) ** 2
                    + osc_freq * (symbols("Nh" + str(var_index)))
                )
                .cancel()
                .expand()
            )

        H_str = self.get_eval_hamiltonian_string(H)
        self.H_str_harmonic = H_str

        variable_symbols_list = (
            flatten_list(self.vars["periodic"].values())
            + flatten_list(self.vars["extended"].values())
            + self.vars["identity"]
        )

        variable_str_list = [
            str(operator)
            for operator in variable_symbols_list
            + self.offset_charge_vars
            + self.external_flux_vars
        ]
        variable_values_list = (
            self.get_operators() + self.get_offset_charges() + self.get_external_flux()
        )
        variable_dict = dict(zip(variable_str_list, variable_values_list))

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

        variable_dict = self.get_operators(return_dict=True)
        # changing variables to strings
        variable_dict_str = dict(
            zip([var.name for var in variable_dict.keys()], variable_dict.values())
        )
        variable_dict_str["cos"] = self._cos_dia
        variable_dict_str["sin"] = self._sin_dia

        return eval(H_str, variable_dict_str)

        # replace * with @ for non-diagonal operators

    def hamiltonian(self):
        """
        Returns the Hamiltonian of the Circuit.
        """
        if self.hierarchical_diagonalization and not self.is_child:
            self.hierarchical_diagonalization_func()
            self.set_operators()
            self.complete_hilbert_space()
        elif not self.is_child:
            self.set_operators()

        if not self.hierarchical_diagonalization:
            if self.phi_basis == "harmonic":
                return self.hamiltonian_harmonic()
            elif self.phi_basis == "discretized":
                return self.hamiltonian_sparse()

        else:
            hamiltonian = self.hilbert_space.hamiltonian()
            if self.phi_basis == "harmonic":
                return hamiltonian.full()
            elif self.phi_basis == "discretized":
                return sp.sparse.csc_matrix(hamiltonian)

    ##################################################################
    ########### Functions from scqubits.core.qubit_base ##############
    ##################################################################
    def _evals_calc(self, evals_count: int) -> ndarray:

        # dimension of the hamiltonian
        if self.hierarchical_diagonalization:
            hilbertdim = np.prod([self.HD_trunc_dims[i][0] for i in self.HD_trunc_dims])
        else:
            hilbertdim = self.hilbertdim()

        hamiltonian_mat = self.hamiltonian()
        if self.phi_basis == "discretized" or self.hierarchical_diagonalization:
            evals = sparse.linalg.eigsh(
                hamiltonian_mat,
                return_eigenvectors=False,
                k=evals_count,
                v0=settings.RANDOM_ARRAY[:hilbertdim],
                which="SA",
            )
        elif self.phi_basis == "harmonic":
            evals = sp.linalg.eigvalsh(
                hamiltonian_mat, subset_by_index=[0, evals_count - 1]
            )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:

        # dimension of the hamiltonian
        if self.hierarchical_diagonalization:
            hilbertdim = np.prod([self.HD_trunc_dims[i][0] for i in self.HD_trunc_dims])
        else:
            hilbertdim = self.hilbertdim()

        hamiltonian_mat = self.hamiltonian()
        if self.phi_basis == "discretized":
            evals, evecs = sparse.linalg.eigsh(
                hamiltonian_mat,
                return_eigenvectors=True,
                k=evals_count,
                which="SA",
                v0=settings.RANDOM_ARRAY[:hilbertdim],
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

    ##################################################################
    ############### Functions for plotting potential #################
    ##################################################################
    def potential_energy(self, **kwargs):
        """
        Returns the full potential of the circuit evaluated in a grid of points as chosen by the user or using default variable ranges.
        """
        periodic_indices = self.var_indices["periodic"]
        discretized_phi_indices = self.var_indices["extended"]
        var_indices = discretized_phi_indices + periodic_indices

        # method to concatenate sublists
        potential_sym = self.potential_symbolic
        for var in self.external_flux_vars:
            potential_sym = potential_sym.subs(var, var * np.pi * 2)

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
                if var_name in [
                    var.name for var in self.param_vars + self.external_flux_vars
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
        discretized_phi_indices = self.var_indices["extended"]
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
                        for i, k in enumerate(self.var_indices["extended"])
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
                plt.bar(
                    np.array(grids[dims[0]]) / (2 * np.pi),
                    eval("np." + mode + "(wf_plot)"),
                )
            else:
                plt.plot(
                    np.array(grids[dims[0]]) / (2 * np.pi),
                    eval("np." + mode + "(wf_plot)"),
                )
            plt.xlabel(var_types[0] + str(var_indices[0]))
        elif len(dims) == 2:
            x, y = np.meshgrid(
                np.array(grids[dims[0]]) / (2 * np.pi),
                np.array(grids[dims[1]]) / (2 * np.pi),
            )
            plt.contourf(x, y, wf_plot)
            plt.xlabel(var_types[0] + str(np.sort(var_indices)[0]))
            plt.ylabel(var_types[1] + str(np.sort(var_indices)[1]))
            plt.colorbar()
        plt.title("Distribution of Wavefuntion along variables " + str(var_indices))


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
            for index in self.parent.var_indices["extended"]:
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
