from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import sympy
import numpy as np
from numpy import ndarray
from sympy import symbols,lambdify, MatrixSymbol
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

class AnalyzeQCircuit(base.QubitBaseClass, CustomQCircuit):
    """
    Class to make numerical analysis on the CustomQCircuit instance. Subclass of CustomQCircuit and can be initiated using the same input file.
    """
    def __init__(self, list_nodes: list, list_branches: list = None, mode: str = "sym"):
        CustomQCircuit.__init__(self, list_nodes, list_branches, mode)

        # defining additional class properties
        self.operator_symbol_dictionary = None # setting a dictionary to save the operator list

        self.vars = None
        self.external_flux = []
        self.params = []
        self.var_cutoffs = None

        self.H_func = None

        # initiating the class properties
        cutoffs = self.var_indices.copy()
        for i in self.var_indices.keys():
            if i == "cyclic":
                cutoffs["cyclic"] = [5 for v in self.var_indices["cyclic"]]
            if i == "periodic":
                cutoffs["periodic"] = [5 for v in self.var_indices["periodic"]]
            if i == "discretized_phi":
                cutoffs["discretized_phi"] = [30 for v in self.var_indices["discretized_phi"]]

        self.var_cutoffs = cutoffs.copy()
        # default values for the parameters
        self.params = [1 for v in self.param_vars]
        # Hamiltonian function
        self.H_func = self.hamiltonian_function() 

    # constructor to initiate using a CustomQCircuit object
    @classmethod
    def from_CustomQCircuit(cls, circuit: CustomQCircuit):
        """
        Initialize AnalyzeQCircuit using an instance of CustomQCircuit.
        """
        return cls(circuit.nodes, circuit.branches, mode = circuit.mode)

    @classmethod
    def from_input_file(cls,  filename: str, mode: str = "sym"):
        """
        Constructor:
        - Constructing the instance from an input file
        - separate methods for numerical and symbolic parameters
        """  
        file = open(filename, "r")
        lines = (file.read()).split("\n")
        num_nodes = int(lines[0].split(":")[-1])
        nodes = [node(i, 0) for i in range(1, num_nodes+1)]
        branches = []

        first_branch = lines.index("branches:") 
        for l in range(first_branch + 1, len(lines)):
            if lines[l] != "":
                line = lines[l].split("\t")
                a,b = [int(i) for i in line[1].split(",")]
                element = line[0]
                
                if element == "JJ":
                    if len(line) > 3: # check to see if all the required parameters are defined 
                        if mode == "sym":
                            parameters = [symbols(line[2]), symbols(line[3])]
                        elif mode == "num":
                            parameters = [float(line[2]), float(line[3])]
                    else:
                        parameters = None
                else:
                    if len(line) > 2: # check to see if all the required parameters are defined
                        if mode == "sym":
                            parameters = [symbols(line[2])]
                        elif mode == "num":
                            parameters = [float(line[2])]
                    else:
                        parameters = None
                branches.append(branch(nodes[a-1], nodes[b-1], element, parameters))
            else:
                break
        
        return cls(nodes, branches, mode = mode)
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
        H = self.hamiltonian_sym()
        
        # Defining the list of discretized_phi variables
        y_symbols = [symbols("y" + str(i)) for i in self.var_indices["discretized_phi"]]
        p_symbols = [symbols("p" + str(i)) for i in self.var_indices["discretized_phi"]]
        ps_symbols = [symbols("ps" + str(i)) for i in self.var_indices["discretized_phi"]]
        
            # marking the squared momentum operators with a separate symbol
        for i in self.var_indices["discretized_phi"]:
            H = H.replace(symbols("p" + str(i))**2, symbols("ps" + str(i)))
            
        # Defining the list of variables for periodic operators
        periodic_symbols_ys = [symbols("ys" + str(i)) for i in self.var_indices["periodic"]]
        periodic_symbols_yc = [symbols("yc" + str(i)) for i in self.var_indices["periodic"]]
        periodic_symbols_n = [symbols("n" + str(i)) for i in self.var_indices["periodic"]]
        periodic_symbols = periodic_symbols_ys + periodic_symbols_yc + periodic_symbols_n
            # marking the sin and cos terms of the periodic variables differently
        H = sympy.expand_trig(H).expand()
        for i in self.var_indices["periodic"]:
            H = H.replace(sympy.cos(1.0*symbols("y" + str(i))), symbols("yc" + str(i))).replace(sympy.sin(1.0*symbols("y" + str(i))), symbols("ys" + str(i)))
        # Defining the list of variables for cyclic operators
        cyclic_symbols = [symbols("n" + str(i)) for i in self.var_indices["cyclic"]]
                    
        # To include the circuit parameters as parameters for the function if the method is called in "sym" or symbolic mode
        
        # removing the constants from the Hamiltonian
        coeff_dict = H.as_coefficients_dict()
        constants = [i for i in coeff_dict if "p" not in str(i) and "y" not in str(i) and "n" not in str(i)]
        for i in constants:
            H = (H - i*coeff_dict[i]) #+ i*coeff_dict[i]*symbols("I")).expand()
        
        # associate a identity matrix with the external flux variables
        for phi in self.external_flux_vars:
            H = H.subs(phi, phi * symbols("I") * 2 * np.pi)
        
        
        # number_variables = (self.var_indices["cyclic"] + self.var_indices["periodic"] + self.var_indices["discretized_phi"]) # eliminating the Σ and zombie vars

        # defining the function from the Hamiltonian
        func = lambdify(
            (cyclic_symbols + periodic_symbols + y_symbols + p_symbols + ps_symbols + [symbols("I")] + self.param_vars + self.external_flux_vars), H,  [{'exp':self._exp_dia}, {'cos':self._cos_dia}, {'sin':self._sin_dia}, "scipy"]
                       )
        
        # Updating the class properties
        self.vars = [cyclic_symbols, 
                     [periodic_symbols_ys, periodic_symbols_yc, periodic_symbols_n], 
                     [y_symbols, p_symbols, ps_symbols]
                    ]
        self.H_func = func
        
        return (func)
    
    ##################################################################
    ############### Functions to construct the operators #############
    ##################################################################
    def hilbertdim(self):
        """
        Returns the Hilbert dimension of the circuit used for calculations
        """
        cutoff_list = []
        for i in self.var_cutoffs:
            if i == "cyclic" or i == "periodic":
                cutoff_list.append([2*k+1 for k in self.var_cutoffs[i]])
            elif i == "discretized_phi":
                cutoff_list.append([k for k in self.var_cutoffs[i]])
                
        cutoff_list = [j for i in list(cutoff_list) for j in i] # concatenating the sublists
        return np.prod(cutoff_list)
    
    # helper functions
    def _kron_operator(self, operator, index):
        """
        Returns the final operator
        """
        var_index_list = self.var_indices["cyclic"] + self.var_indices["periodic"] + self.var_indices["discretized_phi"]
        
        if len(self.var_indices["cyclic"]) != len(self.var_cutoffs["cyclic"]) or len(self.var_indices["periodic"]) != len(self.var_cutoffs["periodic"]) or len(self.var_indices["discretized_phi"]) != len(self.var_cutoffs["discretized_phi"]):
            raise AttributeError("Make sure the cutoffs are only defined for the circuit variables in the class property var_indices, except for zombie variables. ")
        
        cutoff_list = []
        for i in self.var_cutoffs:
            if i == "cyclic" or i == "periodic":
                cutoff_list.append([2*k+1 for k in self.var_cutoffs[i]])
            elif (i == "discretized_phi"):
                cutoff_list.append([k for k in self.var_cutoffs[i]])
                
        cutoff_list = [j for i in list(cutoff_list) for j in i] # concatenating the sublists
        
        if len(var_index_list) > 1:
            if index > var_index_list[0]: 
                Identity_l = sparse.identity(np.prod(cutoff_list[:index-1]), format="csr")
            if index < var_index_list[-1]:
                Identity_r = sparse.identity(np.prod(cutoff_list[index:]), format="csr")

            if index == var_index_list[0]:
                return (sparse.kron(operator, Identity_r, format="csc"))
            elif index == var_index_list[-1]:
                return (sparse.kron(Identity_l, operator, format="csc"))
            else:
                return (sparse.kron(sparse.kron(Identity_l, operator, format="csc"), Identity_r, format="csc"))
        else:
            return (sparse.csc_matrix(operator))
    
    
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
        return (grid.first_derivative_matrix(prefactor = -1j))
    
    def _i_d2_dphi2_operator(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`-1 * d^2/d\varphi^2`in the discretized_phi basis.
        """
        return (grid.second_derivative_matrix(prefactor = -1.0))
    def _cos_phi(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`\\cos \\varphi` in the discretized_phi basis.
        """
        pt_count = grid.pt_count

        cos_op = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = np.cos(grid.make_linspace())
        cos_op.setdiag(diag_elements)
        return cos_op
    def _sin_phi(self, grid: discretization.Grid1d) -> csc_matrix:
        """
        Returns Operator :math:`\\sin \\varphi` in the discretized_phi basis.
        """
        pt_count = grid.pt_count

        sin_op = sparse.dia_matrix((pt_count, pt_count))
        diag_elements = np.cos(grid.make_linspace())
        sin_op.setdiag(diag_elements)
        return sin_op
    
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
    
    def _exp_i_theta_operator(self, ncut: int) -> csc_matrix:
        """
        Returns operator :math:`e^{i\\varphi}` in the charge basis
        """
        dim_theta = 2 * ncut + 1
        matrix = (
            (sparse.dia_matrix(
                    ([-1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                )
            ).tocsc()
        )
        return matrix
    def _cos_theta(self, ncut: int) -> csc_matrix:
        """Returns operator :math:`\\cos \\varphi` in the charge basis"""
        cos_op = 0.5 * self._exp_i_theta_operator(ncut)
        cos_op += cos_op.T
        return cos_op.tocsc()
    def _sin_theta(self, ncut: int) -> csc_matrix:
        """Returns operator :math:`\\sin \\varphi` in the charge basis"""
        sin_op = -1j * 0.5 * self._exp_i_theta_operator(ncut)
        sin_op += sin_op.conjugate().T
        return sin_op.tocsc()

    
    def circuit_operators(self):
        """
        Returns the set of operators which can be given to the Hamiltonian function to construct the Hamiltonian matrix
        """
        import scqubits.core.discretization as discretization
        from scipy import sparse
        
        cyclic_vars, periodic_vars, normal_vars = self.vars 
    
        index_list = [j for i in list(self.var_indices.values()) for j in i]
        cutoff_list = [j for i in list(self.var_cutoffs.values()) for j in i]
        cutoffs = dict(zip(index_list,cutoff_list))
        
        grids = {}
        for i in self.var_indices["discretized_phi"]:
            grids[i] = discretization.Grid1d(-6*np.pi, 6*np.pi, cutoffs[i])
        
        # constructing the operators for normal variables
        normal_operators = [[], [], []]
        for v in normal_vars[0]: # position operators
            index = int(v.name[1:])
            x_operator = self._phi_operator(grids[index])
            normal_operators[0].append(self._kron_operator(x_operator, index))
        for v in normal_vars[1]: # momentum operators
            index = int(v.name[1:])
            p_operator = self._i_d_dphi_operator(grids[index])
            normal_operators[1].append(self._kron_operator(p_operator, index))
        for v in normal_vars[2]: # squared momentum operators
            index = int(v.name[2:])
            ps_operator = self._i_d2_dphi2_operator(grids[index])
            normal_operators[2].append(self._kron_operator(ps_operator, index))
            
        # constructing the operators for periodic variables
        periodic_operators = [[], [], []]
        for v in periodic_vars[0]: # exp(ix) operators; yc
            index = int(v.name[2:])
            x_operator = self._cos_theta(cutoffs[index])
            periodic_operators[0].append(self._kron_operator(x_operator, index))
        for v in periodic_vars[1]: # exp(-ix) operators; ys
            index = int(v.name[2:])
            x_operator = self._sin_theta(cutoffs[index])
            periodic_operators[1].append(self._kron_operator(x_operator, index))
        for v in periodic_vars[2]: # n operators; n
            index = int(v.name[1:])
            n_operator = self._n_theta_operator(cutoffs[index])
            periodic_operators[2].append(self._kron_operator(n_operator, index))
            
        # constructing the operators for cyclic variables
        cyclic_operators = []
        for v in cyclic_vars: # momentum; there's no position for cyclic variables
            index = int(v.name[1:])
            n_operator = self._n_theta_operator(cutoffs[index]) # using the same operator as the periodic variables
            cyclic_operators.append(self._kron_operator(n_operator, index))
        
        return ([cyclic_operators, periodic_operators, normal_operators] + [[self._identity()]])
    ##################################################################
    ############# Functions for eigen values and matrices ############
    ##################################################################
    def hamiltonian(self):
        """
        Returns the Hamiltonian of the Circuit bu using the parameters set in the class properties.
        """
        # check on params class property
        if self.params == None and self.mode == "sym":
            raise AttributeError("Set the params property of the circuit before calling this method.")
        if self.mode == "sym":
            if len(self.param_vars) != len(self.params):
                raise ValueError("Invalid number of parameters given, please check the number of parameters.")
        
        ops = self.circuit_operators()
        operator_list = ops[0] + ops[1][0] + ops[1][1] + ops[1][2] + ops[2][0] + ops[2][1] + ops[2][2] + ops[3]
        
        hamiltonian_matrix = self.H_func(*(operator_list + self.params + self.external_flux))

        return hamiltonian_matrix
    
    ##################################################################
    ########### Functions to plot the circuit properties #############
    ##################################################################
    def get_spectrum_vs_paramvals(self, param, param_range, n_eigs=6):
        """
        Returns a spectrum, of n_eigs eigen values and eigen vectors, for a selected parameter sweep.
        param: one of the strings with one of the symbols from self.param_vars
        param_range: numpy array for the parameter sweep
        n_eigs: (default:6) number of energy states whose energy and eigen vectors need to be calculated.
        """
        if type(param) != str:
            print("Invalid input for param: Required input of type string")
            return 1
        if len(self.param_vars) != len(self.params):
            print("Invalid parameter template, unequal number of parameters given.")
            return 1
        if symbols(param) not in self.param_vars + self.external_flux_vars and param[:-1] not in ["cutoff_c", "cutoff_p", "cutoff_d"]:
            print("Parameter input is not available for this circuit.")
            return 1
        elif symbols(param) in self.param_vars:
            param_index = (self.param_vars).index(symbols(param))
        elif symbols(param) in self.external_flux_vars:
            flux_index = (self.external_flux_vars).index(symbols(param))
            
        #check on params
        if self.params == None and self.mode == "sym":
            raise AttributeError("Set the params property of the circuit before calling this method.")

        # circuit parameter sweep
        ops = self.circuit_operators()
        operator_list = ops[0] + ops[1][0] + ops[1][1] + ops[1][2] + ops[2][0] + ops[2][1] + ops[2][2] + ops[3]
        eigen_vals = []
        eigen_wfs = []
        if symbols(param) in self.param_vars:
            for i in param_range:
                if param_index == 0:
                    params = [i] + self.params[1:]
                else:
                    params = self.params[0:param_index] + [i] + self.params[param_index + 1:]

                H = self.H_func(*operator_list, *params, *self.external_flux)
                e, w = sparse.linalg.eigsh(H, n_eigs, which="SA")
                eigen_vals.append(np.sort(e))
                eigen_wfs.append(w)
        # flux sweep
        elif symbols(param) in self.external_flux_vars:
            for i in param_range:
                if flux_index == 0:
                    flux = [i] + self.external_flux[1:]
                else:
                    flux = self.external_flux[0:flux_index] + [i] + self.external_flux[flux_index + 1:]
                    
                H = self.H_func(*operator_list, *self.params, *flux)
                e, w = sparse.linalg.eigsh(H, n_eigs, which="SA")
                eigen_vals.append(np.sort(e))
                eigen_wfs.append(w)
        # cutoff sweep
        elif param[:-1] in ["cutoff_c", "cutoff_p", "cutoff_d"]:
            var_cutoffs_old = self.var_cutoffs.copy()
            for i in param_range:
                param_index = int(param[-1]) - 1
                if param[: -1] == "cutoff_c":
                    self.var_cutoffs["cyclic"][param_index] = i
                elif param[: -1] == "cutoff_p":
                    self.var_cutoffs["periodic"][param_index] = i
                elif param[: -1] == "cutoff_d":
                    self.var_cutoffs["discretized_phi"][param_index] = i
                    
                ops = self.circuit_operators()
                operator_list = ops[0] + ops[1][0] + ops[1][1] + ops[1][2] + ops[2][0] + ops[2][1] + ops[2][2] + ops[3]
                
                H = self.H_func(*operator_list, *self.params, *self.external_flux)
                e, w = sparse.linalg.eigsh(H, n_eigs, which="SA")
                eigen_vals.append(np.sort(e))
                eigen_wfs.append(w)
            self.var_cutoffs = var_cutoffs_old
        return [np.array(eigen_vals), np.array(eigen_wfs)]
    
    def plot_evals_vs_paramvals(self, param, param_range, n_eigs=6):
        evals, wf = self.get_spectrum_vs_paramvals(param, param_range, n_eigs)
        plt.plot(param_range, evals)
        plt.title("Eigen Values vs " + param)
        plt.xlabel(param)
        plt.ylabel("Eigen Values")
    ##################################################################
    ################## Functions for user queries ####################
    ##################################################################
    @staticmethod
    def default_params() -> Dict[str, Any]:
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

        return {}

    def operator_sym_dict(self):
        """
        Returns a dictionary which has the symbol names as the keys for the corresponding matrix operators used in the circuit.
        """
        if self.operator_symbol_dictionary == None:
            ops = self.circuit_operators()
            operator_list = ops[0] + ops[1][0] + ops[1][1] + ops[1][2] + ops[2][0] + ops[2][1] + ops[2][2] + ops[3]
            
            syms = self.vars
            syms_list = syms[0] + syms[1][0] + syms[1][1] + syms[1][2] + syms[2][0] + syms[2][1] + syms[2][2] + [symbols("I")] # adding the identity variable at the end

            self.operator_symbol_dictionary = dict(zip([i.name for i in syms_list], operator_list))

        return self.operator_symbol_dictionary

    def set_attr(self, paramval: float, param_name: str):
        """
        Function sets the attribute of a given variable in string. Returns the new set of parameters set for the circuit. The following are the acceptable string inputs for the param_name:
        - One of the symbols names present in param_vars
        - One of the symbols names present in external_flux_vars
        - One of the following ["cutoff_ci", "cutoff_pi", "cutoff_di"] where i is the ith cutoff value for the following categories:
            - cutoff_c : cutoffs for cyclic variables.
            - cutoff_p : cutoffs for periodic variables.
            - cutoff_d : cutoffs for the discretized_phi variables.
        """
        # setting the circuit parameters
        if symbols(param_name) in self.param_vars:
            for i in range(len(self.param_vars)):
                if self.param_vars[i].name == param_name:
                    self.params[i] = paramval
        # setting the external flux values
        if symbols(param_name) in self.external_flux_vars:
            for i in range(len(self.external_flux_vars)):
                if self.external_flux_vars[i].name == param_name:
                    self.external_flux[i] = paramval
        # setting the cutoff values
        if param_name[: -1] in ["cutoff_c", "cutoff_p", "cutoff_d"]:
            param_index = int(param_name[-1]) - 1
            if param_name[: -1] == "cutoff_c":
                self.var_cutoffs["cyclic"][param_index] = i
            elif param_name[: -1] == "cutoff_p":
                self.var_cutoffs["periodic"][param_index] = i
            elif param_name[: -1] == "cutoff_d":
                self.var_cutoffs["discretized_phi"][param_index] = i

    ##################################################################
    ########### Functions from scqubits.core.qubit_base ##############
    ##################################################################
    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sparse.linalg.eigsh(
            hamiltonian_mat, return_eigenvectors=False, k = evals_count)
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sparse.linalg.eigsh(
            hamiltonian_mat, return_eigenvectors=True, k = evals_count)
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
            name of the operator symbol in string form, should be one of the symbols in self.vars
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
        
        operator_matrix = self.operator_sym_dict()[operator]

        table = get_matrixelement_table(operator_matrix, evecs)
        if filename or return_datastore:
            data_store = DataStore(
                system_params=self.get_initdata(), matrixelem_table=table
            )
        if filename:
            data_store.filewrite(filename)
        return data_store if return_datastore else table


    def _esys_for_paramval(
        self, paramval: float, param_name: str, evals_count: int
    ) -> Union[Tuple[ndarray, ndarray], SpectrumData]:
        self.set_attr(paramval, param_name)
        return self.eigensys(evals_count)

    def _evals_for_paramval(
        self, paramval: float, param_name: str, evals_count: int
    ) -> ndarray:
        self.set_attr(paramval, param_name)
        return self.eigenvals(evals_count)

# function to find the differences in the energy levels
def energy_split(levels): # gives the energy splits given the energy levels
    splits = []
    for i in range(1, len(levels)):
        splits.append(levels[i] - levels[i-1])
    return splits