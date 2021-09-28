from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from numpy.core.fromnumeric import var
from numpy.core.function_base import linspace

import sympy
import numpy as np
from numpy import ndarray
from sympy import symbols, lambdify
from scipy import sparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix
from matplotlib import pyplot as plt

from scqubits.core.customqcircuit import node, branch
from scqubits.core.customqcircuit import CustomQCircuit
import scqubits.core.discretization as discretization
import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
from scqubits.core.storage import DataStore, SpectrumData
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot

from scqubits.utils.spectrum_utils import (
    get_matrixelement_table,
    order_eigensystem,
    recast_esys_mapdata,
    standardize_sign,
)


class AnalyzeQCircuit(base.QubitBaseClass, CustomQCircuit, serializers.Serializable):
    """
    Class to make numerical analysis on the CustomQCircuit instance. Subclass of CustomQCircuit and can be initiated using the same input file.
    """

    def __init__(
        self,
        list_nodes: list,
        list_branches: list = None,
        ground_node=None,
        mode: str = "sym",
        basis: str = "simple",
        initiate_sym_calc: bool = True
    ):
        CustomQCircuit.__init__(
            self,
            list_nodes,
            list_branches,
            ground_node=ground_node,
            mode=mode,
            basis=basis,
            initiate_sym_calc=initiate_sym_calc
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
        self.cutoffs_list=[]

        # Hamiltonian function
        if initiate_sym_calc:
            self.initiate()

    def initiate(self):
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
            self.discretized_phi_range[v] = (-6*np.pi, 6*np.pi)
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
        
        self._id_str = self._autogenerate_id_str() # generating a class attribute to avoid error by parameter sweeps

        self.hamiltonian_function()
        # initilizing attributes for operators
        self.set_operators()

    # constructor to initiate using a CustomQCircuit object
    @classmethod
    def from_CustomQCircuit(cls, circuit: CustomQCircuit):
        """
        Initialize AnalyzeQCircuit using an instance of CustomQCircuit.
        """
        return cls(
            circuit.nodes,
            circuit.branches,
            ground_node=circuit.ground_node,
            mode=circuit.mode,
            basis=circuit.basis,
        )

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

    def hamiltonian_function(self):
        """
        Outputs the function which can be used to calculate the operator form of the Matrix given the ncut.
        mode:-
        "num" : The circuit parameters are given as numbers in the input file
        "sym" : The circuit parameters are given as symbols in the input file
        """
        H = (
            self.H.expand()
        )  # this expand method is critical to be applied, otherwise the replacemnt of the variables p^2 with ps2 will not be successful and the results would be incorrect

        # Defining the list of discretized_phi variables
        y_symbols = [symbols("θ" + str(i)) for i in self.var_indices["discretized_phi"]]
        p_symbols = [symbols("Q" + str(i)) for i in self.var_indices["discretized_phi"]]
        ps_symbols = [
            symbols("Qs" + str(i)) for i in self.var_indices["discretized_phi"]
        ]

        # marking the squared momentum operators with a separate symbol
        for i in self.var_indices["discretized_phi"]:
            H = H.replace(symbols("Q" + str(i)) ** 2, symbols("Qs" + str(i)))

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
        H = sympy.expand_trig(H).expand()
        for i in self.var_indices["periodic"]:
            H = H.replace(
                sympy.cos(1.0 * symbols("θ" + str(i))), symbols("θc" + str(i))
            ).replace(sympy.sin(1.0 * symbols("θ" + str(i))), symbols("θs" + str(i)))
        
        # # Defining the list of variables for cyclic operators
        cyclic_symbols = [symbols("Qc" + str(i)) for i in self.var_indices["cyclic"]]
        for c in cyclic_symbols:
            H = H.subs(c, c*symbols("I"))

        # To include the circuit parameters as parameters for the function if the method is called in "sym" or symbolic mode

        # removing the constants from the Hamiltonian
        coeff_dict = H.as_coefficients_dict()
        constants = [
            i
            for i in coeff_dict
            if "Q" not in str(i) and "θ" not in str(i) and "n" not in str(i)
        ]
        for i in constants:
            H = H - i * coeff_dict[i]  # + i*coeff_dict[i]*symbols("I")).expand()

        # associate a identity matrix with the external flux vars
        for phi in self.external_flux_vars:
            H = H.subs(phi, phi * symbols("I") * 2 * np.pi)

        # associate a identity matrix with offset charge vars
        for offset_charge in self.offset_charge_vars:
            H = H.subs(offset_charge, offset_charge * symbols("I"))

        # number_variables = (self.var_indices["cyclic"] + self.var_indices["periodic"] + self.var_indices["discretized_phi"]) # eliminating the Σ and zombie vars

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

        # Updating the class properties
        self.vars = {
            "periodic":[periodic_symbols_yc, periodic_symbols_ys, periodic_symbols_n],
            "discretized_phi":[y_symbols, p_symbols, ps_symbols],
            "identity":[symbols("I")]
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
        var_index_list = (
            self.var_indices["periodic"]
            + self.var_indices["discretized_phi"]
        )
        var_index_list.sort() # just a precaution
        cutoff_dict = self.get_cutoffs()

        if (
            len(self.var_indices["periodic"]) != len(cutoff_dict["cutoff_n"])
            or len(self.var_indices["discretized_phi"])
            != len(cutoff_dict["cutoff_phi"])
        ):
            raise AttributeError(
                "Make sure the cutoffs are only defined for the circuit variables in the class property var_indices, except for zombie variables. "
            )

        cutoff_list = []
        for cutoff_type in cutoff_dict.keys():
            if "cutoff_n" in cutoff_type:
                cutoff_list.append([2 * k + 1 for k in cutoff_dict[cutoff_type]])
            elif "cutoff_phi" in cutoff_type:
                cutoff_list.append([k for k in cutoff_dict[cutoff_type]])

        cutoff_list = [
            j for i in list(cutoff_list) for j in i
        ]  # concatenating the sublists

        if len(var_index_list) > 1:
            if index > var_index_list[0]:
                Identity_l = sparse.identity(
                    np.prod(cutoff_list[: index - 1]), format="csr"
                )
            if index < var_index_list[-1]:
                Identity_r = sparse.identity(np.prod(cutoff_list[index:]), format="csr")

            if index == var_index_list[0]:
                return sparse.kron(operator, Identity_r, format="csc")
            elif index == var_index_list[-1]:
                return sparse.kron(Identity_l, operator, format="csc")
            else:
                return sparse.kron(
                    sparse.kron(Identity_l, operator, format="csc"),
                    Identity_r,
                    format="csc",
                )
        else:
            return sparse.csc_matrix(operator)

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
            grids[i] = discretization.Grid1d(self.discretized_phi_range[i][0], self.discretized_phi_range[i][1], cutoffs[i])

        # constructing the operators for normal variables
        normal_operators = [[], [], []]
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

        # constructing the operators for periodic variables
        periodic_operators = [[], [], []]
        for v in periodic_vars[0]:  # exp(ix) operators; ys
            index = int(v.name[2:])
            x_operator = self._sin_theta(cutoffs[index])
            periodic_operators[0].append(self._kron_operator(x_operator, index))
        for v in periodic_vars[1]:  # exp(-ix) operators; yc
            index = int(v.name[2:])
            x_operator = self._cos_theta(cutoffs[index])
            periodic_operators[1].append(self._kron_operator(x_operator, index))
        for v in periodic_vars[2]:  # n operators; n
            index = int(v.name[1:])
            n_operator = self._n_theta_operator(cutoffs[index])
            periodic_operators[2].append(self._kron_operator(n_operator, index))

        # # constructing the operators for cyclic variables
        # cyclic_operators = []
        # for v in cyclic_vars:  # momentum; there's no position for cyclic variables
        #     index = int(v.name[1:])
        #     n_operator = self._n_theta_operator(
        #         cutoffs[index]
        #     )  # using the same operator as the periodic variables
        #     cyclic_operators.append(self._kron_operator(n_operator, index))

        return {"periodic":periodic_operators, "discretized_phi":normal_operators, "identity": [self._identity()]
        }

    ##################################################################
    ################ Functions for parameter queries #################
    ##################################################################
    def get_params(self):
        params = []
        for param in self.param_vars:
            params.append(getattr(self, param.name))
        return params

    def get_cutoffs(self):
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
        return [getattr(self, flux.name) for flux in self.external_flux_vars]

    def get_offset_charges(self):
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
        if self.get_params() == None and self.mode == "sym":
            raise AttributeError(
                "Set the params property of the circuit before calling this method."
            )
        if self.mode == "sym":
            if len(self.param_vars) != len(self.get_params()):
                raise ValueError(
                    "Invalid number of parameters given, please check the number of parameters."
                )
        self.set_operators()  # updating the operators
        hamiltonian_matrix = self.H_func(
            *(
                self.get_operators()
                + self.get_params()
                + self.get_external_flux()
                + self.get_offset_charges()
            )
        )

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

        # adding external fluxes
        for var_name in self.external_flux_vars:
            parameters

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
    def plot_wavefunction(self, n=0, var_indices=(1,), mode="abs"):
        dims = tuple(
            np.sort(var_indices) - 1
        )  # taking the var indices and identifying the dimensions.

        eigs, wf = self.eigensys()

        cutoffs_dict = self.get_cutoffs()

        cutoff_list = []
        grids = []
        for cutoff_type in cutoffs_dict.keys():
            if "cutoff_periodic" in cutoff_type:
                cutoff_list.append([2 * k + 1 for k in cutoffs_dict[cutoff_type]])
                grids.append(
                    [list(range(-k, k + 1)) for k in cutoffs_dict[cutoff_type]]
                )
            elif "cutoff_discretized_phi" in cutoff_type:
                cutoff_list.append([k for k in cutoffs_dict[cutoff_type]])
                grids.append(
                    [
                        np.linspace(self._discretized_phi_range[0], self._discretized_phi_range[1], k)
                        for k in cutoffs_dict[cutoff_type]
                    ]
                )
        cutoff_list = [i for j in cutoff_list for i in j]  # concatenating the sublists
        grids = [i for j in grids for i in j]  # concatenating the sublists

        var_types = []

        for var_index in np.sort(var_indices):
            if (
                var_index in self.var_indices["periodic"]
            ):
                var_types.append("Charge in units of 2e, Variable:")
            else:
                var_types.append("Dimensionless Flux, Variable:")

        # selecting the n wave funciton according to the input
        wf_reshaped = wf[:, n].reshape(*cutoff_list)

        if len(dims) > 2:
            raise AttributeError(
                "Cannot plot wavefunction in more than 2 dimensions. The number of dimensions in dims should be less than 2."
            )

        wf_plot = (
            np.sum(
                eval("np." + mode + "(wf_reshaped)"),
                axis=tuple([i for i in range(len(cutoff_list)) if i not in dims]),
            )
        ).T

        if len(dims) == 1:
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
        hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(
            hamiltonian_mat, return_eigenvectors=False, k=evals_count, which="SA"
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(
            hamiltonian_mat, return_eigenvectors=True, k=evals_count, which="SA"
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
