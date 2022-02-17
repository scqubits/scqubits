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

import sympy
import numpy as np
import scipy as sp
import re

from numpy import ndarray
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
from scqubits.utils.misc import list_intersection

from scqubits.utils.spectrum_utils import (
    get_matrixelement_table,
    order_eigensystem,
)

# Causing a circular import
# if TYPE_CHECKING:
#     from scqubits.core.symboliccircuit import Circuit


class CircuitSubsystem(base.QubitBaseClass, serializers.Serializable):
    r"""
    Class to initiate a sub-system for a circuit just from a symbolic Hamiltonian.
    Circuit object can be initiated using:
        CircuitSubsystem(parent, H_sym)
    Parameters
    ----------
    parent:
        the Circuit object containing this subsystem.
    H_sym:
        Symbolic Hamiltonian describing the system.
    """

    def __init__(self, parent, H_sym):
        self.parent = parent
        self.H_sym = H_sym
        self.variables = list(H_sym.free_symbols)

        # TODO what is this / why is it hardcoded to 10?
        self.truncated_dim = 10
        self._sys_type = type(self).__name__  # for object description
        # TODO we talked about this... did you not fix this meanwhile?
        self._id_str = (
            self._autogenerate_id_str()
        )  # generating a class attribute to avoid error by parameter sweeps

        # TODO what on earth is this doing here?
        self.hilbertdim()
        # TODO what on earth is this doing here?
        self.hamiltonian_func()

    def generate_symbols(self, prefix: str, index_type: str) -> List[symbols]:
        return [
            symbols(prefix + str(index))
            for index in list_intersection(
                self.parent.var_indices[index_type], self.var_indices
            )
        ]

    # TODO docstring
    def hamiltonian_func(self) -> Callable:
        hamiltonian = self.H_sym.expand()

        periodic_var_indices = list_intersection(
            self.parent.var_indices["periodic"], self.var_indices
        )
        if len(periodic_var_indices) > 0:
            hamiltonian = sympy.expand_trig(hamiltonian).expand()

        periodic_symbols_ys = self.generate_symbols("θs", "periodic")
        periodic_symbols_yc = self.generate_symbols("θc", "periodic")
        # TODO what's up with this? generated then never used?
        periodic_symbols_n = self.generate_symbols("n", "periodic")

        for index in periodic_var_indices:
            hamiltonian = hamiltonian.replace(
                sympy.cos(1.0 * symbols("θ" + str(index))), symbols("θc" + str(index))
            )
            hamiltonian = hamiltonian.replace(
                sympy.sin(1.0 * symbols("θ" + str(index))), symbols("θs" + str(index))
            )

        # TODO naming obscure - what is ps??
        ps_symbols = [
            symbols("Qs" + str(index))
            for index in list_intersection(
                self.parent.var_indices["discretized_phi"], self.var_indices
            )
        ]

        # marking the squared momentum operators with a separate symbol
        for index in list_intersection(
            self.parent.var_indices["discretized_phi"], self.var_indices
        ):
            hamiltonian = hamiltonian.replace(
                symbols("Q" + str(index)) ** 2, symbols("Qs" + str(index))
            )

        if len(periodic_var_indices) == 0:
            # TODO meaning of "I" is completely obscure at this point in the code
            variables = [var for var in self.variables if "I" not in str(var)]
        else:
            # TODO what's up with the "[] +"?
            variables = [] + periodic_symbols_ys + periodic_symbols_yc
            for var in self.variables:
                for index in periodic_var_indices:
                    if "θ" + str(index) not in str(var) and "I" not in str(var):
                        variables.append(var)

        variables = variables + ps_symbols

        self.H_func_vars = variables.copy()

        if symbols("I") in hamiltonian.free_symbols:
            variables.append(symbols("I"))

        self.H_func = lambdify(
            variables,
            hamiltonian,
            [
                {"exp": self.parent._exp_dia},
                {"cos": self.parent._cos_dia},
                {"sin": self.parent._sin_dia},
                "scipy",
            ],
        )
        # TODO this looks peculiar: the result is stored and returned???
        return self.H_func

    def _identity(self) -> csc_matrix:
        """
        Returns the Identity operator for the entire Hilbert space of the circuit.
        """
        dim = self.hilbertdim()
        return sparse.identity(dim, format="csc")

    def hamiltonian(self):
        self.parent.set_operators()
        variables = [getattr(self.parent, str(var)) for var in self.H_func_vars]
        if symbols("I") in self.variables:
            variables.append(self._identity())
        return self.H_func(*variables)

    def hilbertdim(self):
        var_indices = []
        cutoffs = []
        for var in self.variables:
            if "I" not in str(var):
                filtered_var = re.findall(
                    "[0-9]+", re.sub(r"ng_[0-9]+|Φ[0-9]+", "", str(var))
                )  # filtering offset charges and external flux
                if filtered_var == []:
                    continue
                else:
                    var_index = int(filtered_var[0])
                # var_index = (int(re.findall('[0-9]+', str(v))[0]))
                if var_index not in var_indices:
                    for cutoff_name in self.parent.cutoffs_list:
                        if str(var_index) in cutoff_name:
                            cutoffs.append(getattr(self.parent, cutoff_name))
                    var_indices.append(var_index)
        # setting some class attributes
        # TODO do not declare attributes outside of _init_
        # TODO and why would this need to be set if hilbertdim was called repeatedly?
        self.var_indices = var_indices
        self.cutoffs = cutoffs
        dimensions = []
        for x, index in enumerate(var_indices):
            if index in self.parent.var_indices["periodic"]:
                dimensions.append(cutoffs[x] * 2 + 1)
            elif index in self.parent.var_indices["discretized_phi"]:
                dimensions.append(cutoffs[x])
        # returning the hilbertdim
        return np.prod(dimensions)

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()

        evals = sparse.linalg.eigsh(
            hamiltonian_mat,
            return_eigenvectors=False,
            k=evals_count,
            v0=settings.RANDOM_ARRAY[: self.hilbertdim()],
            which="SA",
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(
            hamiltonian_mat,
            return_eigenvectors=True,
            k=evals_count,
            v0=settings.RANDOM_ARRAY[: self.hilbertdim()],
            which="SA",
        )
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    # TODO need to discuss
    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}
        return {}


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
        list_nodes: list,
        list_branches: list = None,
        ground_node=None,
        basis: str = "simple",
        initiate_sym_calc: bool = True,
        phi_basis: str = "sparse",
        hierarchical_diagonalization: bool = False,
    ):
        SymbolicCircuit.__init__(
            self,
            list_nodes,
            list_branches,
            ground_node=ground_node,
            basis=basis,
            initiate_sym_calc=initiate_sym_calc,
        )

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
        if phi_basis != "sparse" and self.hierarchical_diagonalization:
            raise Exception(
                "Heirarchical Diagonalization only works with discretized phi basis for extended degrees of freedom. Please change `phi_basis` to sparse to use heirarchical diagonalization."
            )

        # Hamiltonian function
        if initiate_sym_calc:
            self.initiate_circuit()

    # constructor to initiate using a CustomQCircuit object
    @classmethod
    def from_CustomQCircuit(
        cls,
        circuit: SymbolicCircuit,
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
            circuit.nodes,
            circuit.branches,
            ground_node=circuit.ground_node,
            basis=circuit.basis,
            initiate_sym_calc=circuit.initiate_sym_calc,
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
        circuit.hierarchical_diagonalization = hierarchical_diagonalization
        circuit.phi_basis = phi_basis

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
        circuit.hierarchical_diagonalization = hierarchical_diagonalization
        circuit.phi_basis = phi_basis

        return cls.from_CustomQCircuit(
            circuit,
            hierarchical_diagonalization=hierarchical_diagonalization,
            phi_basis=phi_basis,
        )

    def initiate_circuit(self):
        """
        Function to initiate the instance attributes by calling the appropriate methods.
        """
        self.hamiltonian_sym()
        # initiating the class properties
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
            setattr(self, param.name, 1.0)  # setting the default parameters as 1
        # setting the ranges for floux ranges used for discrete phi vars
        for v in self.var_indices["discretized_phi"]:
            self.discretized_phi_range[v] = (-6 * np.pi, 6 * np.pi)
        # default values for the external flux vars
        for flux in self.external_flux_vars:
            setattr(self, flux.name, 0.0)  # setting the default to zero external flux
        # default values for the offset charge vars
        for offset_charge in self.offset_charge_vars:
            setattr(self, offset_charge.name, 0.0)  # default to zero offset charge

        # setting the __init__params attribute
        self._init_params = (
            [param.name for param in self.param_vars]
            + [flux.name for flux in self.external_flux_vars]
            + [offset_charge.name for offset_charge in self.offset_charge_vars]
            + self.cutoffs_list
            + ["input_string"]
        )

        self._id_str = (
            self._autogenerate_id_str()
        )  # generating a class attribute to avoid error by parameter sweeps

        if not self.hierarchical_diagonalization:
            self.hamiltonian_function()
        else:
            self.hierarchical_diagonalization_func()

        # initilizing attributes for operators
        self.set_operators()

        if self.hierarchical_diagonalization:
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
        periodic_symbols_ys = [
            symbols("θs" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols_yc = [
            symbols("θc" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols_n = [
            symbols("n" + str(i)) for i in self.var_indices["periodic"]
        ]

        y_symbols = [symbols("θ" + str(i)) for i in self.var_indices["discretized_phi"]]
        p_symbols = [symbols("Q" + str(i)) for i in self.var_indices["discretized_phi"]]
        ps_symbols = [
            symbols("Qs" + str(i)) for i in self.var_indices["discretized_phi"]
        ]

        self.vars = {
            "periodic": [
                periodic_symbols_ys,
                periodic_symbols_yc,
                periodic_symbols_n,
            ],
            "discretized_phi": [y_symbols, p_symbols, ps_symbols],
            "identity": [symbols("I")],
        }

        H = self.hamiltonian.expand()

        # terms_str = list(expr_dict.keys())
        # coeff_str = list(expr_dict.values())

        for phi in self.external_flux_vars:
            H = H.subs(phi, phi * symbols("I") * 2 * np.pi)

        # associate a identity matrix with offset charge vars
        for offset_charge in self.offset_charge_vars:
            H = H.subs(offset_charge, offset_charge * symbols("I"))

        expr_dict = H.as_coefficients_dict()
        terms_str = list(expr_dict.keys())

        oscs = []
        interaction = []

        Hf = H.copy()

        if len(self.var_indices["osc"]) == 0:
            raise Exception(
                "No oscillator has been detected in this circuit, hierarchcal diagonalization has only been implemented for oscillators."
            )

        for var_index in self.var_indices["osc"]:
            H_osc = 0 * symbols("x")
            H_int = 0 * symbols("x")
            for i, term in enumerate(terms_str):
                if ("θ" + str(var_index) + "**2") in str(term) or (
                    "Q" + str(var_index) + "**2"
                ) in str(term):
                    H_osc = H_osc + expr_dict[term] * term
                # mat = re.search("θ" + str(var_index), str(term))
                # mat1 = re.search("Q" + str(var_index), str(term))

                if ("θ" + str(var_index)) in str(term) and (
                    "θ" + str(var_index) + "**2"
                ) not in str(term):
                    H_int = H_int + expr_dict[term] * term

                if ("Q" + str(var_index)) in str(term) and (
                    "Q" + str(var_index) + "**2"
                ) not in str(term):
                    if (
                        len(
                            re.findall(
                                "[0-9]", re.sub(r"ng_[0-9]+|Φ[0-9]+", "", str(term))
                            )
                        )
                        > 1
                    ):
                        H_int = H_int + expr_dict[term] * term
                    else:
                        H_osc = H_osc + expr_dict[term] * term
            oscs.append(H_osc)
            interaction.append(H_int)

        # storing data in class attributes
        self.osc_subsystems_sym = dict(
            zip(
                self.var_indices["osc"],
                [
                    [oscs[index], interaction[index]]
                    for index in range(len(self.var_indices["osc"]))
                ],
            )
        )
        self.main_subsystem_sym = H - sum(oscs) - sum(interaction)

        self.main_subsystem = CircuitSubsystem(self, self.main_subsystem_sym)
        self.osc_subsystems = dict(
            zip(
                self.var_indices["osc"],
                [
                    [CircuitSubsystem(self, oscs[index]), interaction[index]]
                    for index in range(len(self.var_indices["osc"]))
                ],
            )
        )

    def complete_hilbert_space(self):
        hilbert_space = HilbertSpace(
            [self.main_subsystem]
            + [self.osc_subsystems[i][0] for i in self.var_indices["osc"]]
        )

        # Adding interactions using the symbolic interaction term
        for osc in self.var_indices["osc"]:
            interaction = self.osc_subsystems[osc][1].expand()
            expr_dict = interaction.as_coefficients_dict()
            terms_str = list(expr_dict.keys())
            # coeff_str = list(expr_dict.values())

            for i, x in enumerate(terms_str):
                coefficient = expr_dict[x]

                # adding external flux and offset charge to coefficient
                for var in x.free_symbols:
                    if "Φ" in str(var) or "ng" in str(var):
                        coefficient = coefficient*getattr(self, str(var))
                        
                operator_symbols = [var for var in x.free_symbols if (("Φ" not in str(var)) and ("ng" not in str(var)))]

                main_sub_op_list = []
                osc_sub_op_list = []
                for var in operator_symbols:
                    if var in self.main_subsystem.H_func_vars and "I" not in str(var):
                        main_sub_op_list.append(getattr(self, str(var)))
                    elif var in self.osc_subsystems[osc][
                        0
                    ].H_func_vars and "I" not in str(var):
                        osc_sub_op_list.append(getattr(self, str(var)))
                    elif "I" in str(var):
                        osc_sub_op_list.append(self.osc_subsystems[osc][0]._identity())

                operator_dict = {}
                for op_index, op in enumerate(main_sub_op_list):
                    operator_dict["op" + str(op_index + 1)] = (op, self.main_subsystem)

                for op_index, op in enumerate(osc_sub_op_list):
                    operator_dict["op" + str(len(main_sub_op_list) + op_index +1)] = (
                        op,
                        self.osc_subsystems[osc][0],
                    )
                hilbert_space.add_interaction(g=float(coefficient), **operator_dict)

        self.hilbert_space = hilbert_space

    def hamiltonian_function(self):
        """
        Outputs the function using lambdify in Sympy, which returns a Hamiltonian matrix by using the circuit attributes set in either the input file or the instance attributes.
        """
        H = (
            self.hamiltonian.expand()
        )  # this expand method is critical to be applied, otherwise the replacemnt of the variables p^2 with ps2 will not be successful and the results would be incorrect

        # Defining the list of variables for periodic operators
        periodic_symbols_ys = [
            symbols("θs" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols_yc = [
            symbols("θc" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols_n = [
            symbols("n" + str(i)) for i in self.var_indices["periodic"]
        ]
        periodic_symbols = (
            periodic_symbols_ys + periodic_symbols_yc + periodic_symbols_n
        )
        # marking the sin and cos terms of the periodic variables differently
        if len(self.var_indices["periodic"]) > 0:
            H = sympy.expand_trig(H).expand()

        for i in self.var_indices["periodic"]:
            H = H.replace(
                sympy.cos(1.0 * symbols("θ" + str(i))), symbols("θc" + str(i))
            ).replace(sympy.sin(1.0 * symbols("θ" + str(i))), symbols("θs" + str(i)))

        # Defining the list of discretized_phi variables
        y_symbols = [symbols("θ" + str(i)) for i in self.var_indices["discretized_phi"]]
        p_symbols = [symbols("Q" + str(i)) for i in self.var_indices["discretized_phi"]]

        if self.phi_basis == "sparse":

            ps_symbols = [
                symbols("Qs" + str(i)) for i in self.var_indices["discretized_phi"]
            ]

            # marking the squared momentum operators with a separate symbol
            for i in self.var_indices["discretized_phi"]:
                H = H.replace(symbols("Q" + str(i)) ** 2, symbols("Qs" + str(i)))

        elif self.phi_basis == "harmonic":
            osc_freqs = dict.fromkeys(self.var_indices["discretized_phi"])
            osc_lengths = dict.fromkeys(self.var_indices["discretized_phi"])
            a_symbols = [
                symbols("a" + str(i)) for i in self.var_indices["discretized_phi"]
            ]
            ad_symbols = [
                symbols("ad" + str(i)) for i in self.var_indices["discretized_phi"]
            ]
            Nh_symbols = [
                symbols("Nh" + str(i)) for i in self.var_indices["discretized_phi"]
            ]
            for i in self.var_indices["discretized_phi"]:
                ECi = H.coeff("Q" + str(i) + "**2").cancel() / 4
                ELi = H.coeff("θ" + str(i) + "**2").cancel() * 2
                osc_freqs[i] = (8 * ELi * ECi) ** 0.5
                osc_lengths[i] = (8.0 * ECi / ELi) ** 0.25
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
            # H = H.rewrite((sympy.cos, sympy.sin),sympy.exp)
            self.osc_freqs = osc_freqs
            self.osc_lengths = osc_lengths

            for i in self.var_indices["discretized_phi"]:
                H = H.replace(
                    symbols("θ" + str(i)),
                    (symbols("ad" + str(i)) + symbols("a" + str(i)))
                    * osc_lengths[i]
                    / np.sqrt(2),
                )
                H = H.replace(
                    symbols("Q" + str(i)),
                    1j
                    * (symbols("ad" + str(i)) - symbols("a" + str(i)))
                    / (osc_lengths[i] * np.sqrt(2)),
                )

            H = H.expand()
            expr_dict = H.as_coefficients_dict()
            terms_str = list(expr_dict.keys())
            coeff_str = list(expr_dict.values())
            # # from sympy.utilities.iterables import flatten
            for i, x in enumerate(terms_str):
                mat = re.search(r"a\d\*[\w+]*", str(x))
                mat1 = re.search(r"ad\d\*[\w+]*", str(x))

                if mat and "cos" not in str(x) and "sin" not in str(x):
                    orig = coeff_str[i] * x
                    x = coeff_str[i] * parse_expr(
                        str(x).replace(
                            mat.group(), "F(" + mat.group().replace("*", ",") + ")"
                        )
                    )
                    H = H - orig + x
                elif mat1 and "cos" not in str(x) and "sin" not in str(x):
                    orig = coeff_str[i] * x
                    x = coeff_str[i] * parse_expr(
                        str(x).replace(
                            mat1.group(), "F(" + mat1.group().replace("*", ",") + ")"
                        )
                    )
                    H = H - orig + x

                if "a" in str(x) and ("*cos" in str(x) or "*sin" in str(x)):
                    orig = coeff_str[i] * x
                    x = coeff_str[i] * parse_expr(
                        "F("
                        + str(x).replace("*cos", ",cos").replace("*sin", ",sin")
                        + ")"
                    )
                    H = H - orig + x

        # To include the circuit parameters as parameters for the function if the method is called in "sym" or symbolic mode

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
            H = H - i * coeff_dict[i]  # + i*coeff_dict[i]*symbols("I")).expand()

        # associate a identity matrix with the external flux vars
        for phi in self.external_flux_vars:
            H = H.subs(phi, phi * symbols("I") * 2 * np.pi)

        # associate a identity matrix with offset charge vars
        for offset_charge in self.offset_charge_vars:
            H = H.subs(offset_charge, offset_charge * symbols("I"))

        # Updating the class properties
        if self.phi_basis == "sparse":
            # defining the function from the Hamiltonian
            func = lambdify(
                (
                    periodic_symbols
                    + y_symbols
                    + p_symbols
                    + ps_symbols
                    + [symbols("I")]
                    + self.param_vars
                    + self.external_flux_vars
                    + self.offset_charge_vars
                ),
                H,
                [
                    {"exp": self._exp_dia},
                    {"cos": self._cos_dia},
                    {"sin": self._sin_dia},
                    "scipy",
                ],
            )
            self.vars = {
                "periodic": [
                    periodic_symbols_ys,
                    periodic_symbols_yc,
                    periodic_symbols_n,
                ],
                "discretized_phi": [y_symbols, p_symbols, ps_symbols],
                "identity": [symbols("I")],
            }
        elif self.phi_basis == "harmonic":
            func = lambdify(
                (
                    periodic_symbols
                    + a_symbols
                    + ad_symbols
                    + Nh_symbols
                    + [symbols("I")]
                    + self.param_vars
                    + self.external_flux_vars
                    + self.offset_charge_vars
                ),
                H,
                [
                    {"F": lambda *args: np.linalg.multi_dot(args)},
                    {"cos": sp.linalg.cosm},
                    {"sin": sp.linalg.sinm},
                    "scipy",
                ],
            )
            self.vars = {
                "periodic": [
                    periodic_symbols_ys,
                    periodic_symbols_yc,
                    periodic_symbols_n,
                ],
                "discretized_phi": [a_symbols, ad_symbols, Nh_symbols],
                "identity": [symbols("I")],
            }
        self.H_func = func
        setattr(self, "H_f", H)
        return func

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

        if self.hierarchical_diagonalization:
            if index in self.main_subsystem.var_indices:
                var_index_list = self.main_subsystem.var_indices
            else:
                var_index_list = [index]
        else:
            var_index_list = (
                self.var_indices["periodic"] + self.var_indices["discretized_phi"]
            )
        var_index_list.sort()  # important to make sure that right cutoffs are chosen
        cutoff_dict = self.get_cutoffs()

        if len(self.var_indices["periodic"]) != len(cutoff_dict["cutoff_n"]) or len(
            self.var_indices["discretized_phi"]
        ) != len(cutoff_dict["cutoff_phi"]):
            raise AttributeError(
                "Make sure the cutoffs are only defined for the circuit variables in the class property var_indices, except for frozen variables. "
            )

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

    ## Identity Operator
    def _identity(self) -> csc_matrix:
        """
        Returns the Identity operator for the entire Hilber space of the circuit.
        """
        dim = self.hilbertdim()
        op = sparse.identity(dim)
        return op.tocsc()

    ## Phi basis
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

    ## charge basis

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
        """
        Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.
        """
        dim_theta = 2 * ncut + 1
        matrix = (
            sparse.dia_matrix(([-1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))
        ).tocsc()
        return matrix

    def _exp_i_theta_operator_conjugate(self, ncut) -> csc_matrix:
        """
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
        identity_vars = self.vars["identity"]

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

        # constructing the operators for normal variables
        normal_operators = [[], [], []]
        if self.phi_basis == "sparse":
            for v in normal_vars[0]:  # position operators
                index = int(v.name[1:])
                x_operator = self._phi_operator(grids[index])
                normal_operators[0].append(self._kron_operator(x_operator, index))
            for v in normal_vars[1]:  # momentum operators
                index = int(v.name[1:])
                p_operator = self._i_d_dphi_operator(grids[index])
                normal_operators[1].append(self._kron_operator(p_operator, index))
            for v in normal_vars[2]:  # squared momentum operators
                index = int(v.name[2:])
                ps_operator = self._i_d2_dphi2_operator(grids[index])
                normal_operators[2].append(self._kron_operator(ps_operator, index))
        elif self.phi_basis == "harmonic":
            for v in normal_vars[0]:  # a or annihilation operators
                index = int(v.name[1:])
                x_operator = op.annihilation(cutoffs[index]) * (1 + 0j)
                normal_operators[0].append(self._kron_operator(x_operator, index))
            for v in normal_vars[1]:  # ad or creation operators
                index = int(v.name[2:])
                p_operator = op.creation(cutoffs[index]) * (1 + 0j)
                normal_operators[1].append(self._kron_operator(p_operator, index))
            for v in normal_vars[2]:  # Nh or number operators
                index = int(v.name[2:])
                ps_operator = (
                    op.creation(cutoffs[index])
                    @ op.annihilation(cutoffs[index])
                    * (1 + 0j)
                )
                normal_operators[2].append(self._kron_operator(ps_operator, index))

        # constructing the operators for periodic variables
        periodic_operators = [[], [], []]
        for v in periodic_vars[0]:  # exp(ix) operators; ys
            index = int(v.name[2:])
            x_operator = self._change_sparsity(self._sin_theta(cutoffs[index]))
            periodic_operators[0].append(self._kron_operator(x_operator, index))
        for v in periodic_vars[1]:  # exp(-ix) operators; yc
            index = int(v.name[2:])
            x_operator = self._change_sparsity(self._cos_theta(cutoffs[index]))
            periodic_operators[1].append(self._kron_operator(x_operator, index))
        for v in periodic_vars[2]:  # n operators; n
            index = int(v.name[1:])
            n_operator = self._change_sparsity(self._n_theta_operator(cutoffs[index]))
            periodic_operators[2].append(self._kron_operator(n_operator, index))

        return {
            "periodic": periodic_operators,
            "discretized_phi": normal_operators,
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
        syms = self.vars
        syms_list = (
            syms["periodic"][0]
            + syms["periodic"][1]
            + syms["periodic"][2]
            + syms["discretized_phi"][0]
            + syms["discretized_phi"][1]
            + syms["discretized_phi"][2]
            + syms["identity"]
        )
        operator_list = []
        for operator in syms_list:
            operator_list.append(getattr(self, operator.name))

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

        ops = self.circuit_operators()
        operator_list = (
            ops["periodic"][0]
            + ops["periodic"][1]
            + ops["periodic"][2]
            + ops["discretized_phi"][0]
            + ops["discretized_phi"][1]
            + ops["discretized_phi"][2]
            + ops["identity"]
        )

        syms = self.vars
        syms_list = (
            syms["periodic"][0]
            + syms["periodic"][1]
            + syms["periodic"][2]
            + syms["discretized_phi"][0]
            + syms["discretized_phi"][1]
            + syms["discretized_phi"][2]
            + syms["identity"]
        )

        for x, operator in enumerate(syms_list):
            setattr(self, operator.name, operator_list[x])

        return dict(zip([operator.name for operator in syms_list], operator_list))

    ##################################################################
    ############# Functions for eigen values and matrices ############
    ##################################################################
    def hamiltonian(self):
        """
        Returns the Hamiltonian of the Circuit bu using the parameters set in the class properties.
        """
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
        self.set_operators()  # updating the operators
        if not self.hierarchical_diagonalization:
            hamiltonian_matrix = self.H_func(
                *(
                    self.get_operators()
                    + self.get_params()
                    + self.get_external_flux()
                    + self.get_offset_charges()
                )
            )
        else:
            self.complete_hilbert_space()
            hamiltonian_matrix = self.hilbert_space.hamiltonian()

        return hamiltonian_matrix

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
        potential_sym = self.potential

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
                zip(sweep_vars, np.meshgrid(*[grid for grid in sweep_vars.values()]))
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

        # method to concatenate sublists
        potential_sym = self.potential

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
        cutoff_list = [i for j in cutoff_list for i in j]  # concatenating the sublists
        grids = [i for j in grids for i in j]  # concatenating the sublists

        var_types = []

        for var_index in np.sort(var_indices):
            if var_index in self.var_indices["periodic"]:
                var_types.append("Charge in units of 2e, Variable:")
            else:
                var_types.append("Dimensionless Flux, Variable:")

        # selecting the n wave funciton according to the input
        wf_reshaped = wf[:, n].reshape(*cutoff_list)

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

        if self.hierarchical_diagonalization:
            self.set_operators()
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

        if self.hierarchical_diagonalization:
            self.set_operators()
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
