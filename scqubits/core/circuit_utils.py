# circuit-utils.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import re
from typing import TYPE_CHECKING, Any, Callable, List, Union, Optional, Tuple, Dict

import numpy as np
import scipy as sp
import sympy as sm
import qutip as qt
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

from scqubits.core import discretization as discretization
from scqubits.utils.misc import (
    flatten_list_recursive,
    is_string_float,
    unique_elements_in_list,
    Qobj_to_scipy_csc_matrix,
)
from scqubits.core import circuit_input

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem


def _junction_order(branch_type: str) -> int:
    """
    Returns the order of the branch if it is a JJ branch,
    if the order is n its energy is given by cos(phi) + cos(2*phi) + ... + cos(n*phi)

    Args:
        branch (_type_): Branch

    Raises:
        ValueError: when the branch is not a josephson junction

    Returns:
        _type_: int, order of the josephson junction
    """
    if "JJ" not in branch_type:
        raise ValueError("The branch is not a JJ branch")
    if len(branch_type) > 2:
        if (
            branch_type[2] == "s"
        ):  # adding "JJs" which is a junction with sawtooth current phase relationship
            return 1
        return int(branch_type[2:])
    else:
        return 1


def sawtooth_operator(x: Union[ndarray, csc_matrix]):
    """
    Returns the operator evaluated using applying the sawtooth_potential function on the
    diagonal elements of the operator x

    Args:
        x (Union[ndarray, csc_matrix]): argument of the sawtooth operator in the Hamiltonian
    """
    diagonal_elements = sawtooth_potential(x.diagonal())

    operator = sp.sparse.dia_matrix(
        (diagonal_elements, 0), shape=(len(diagonal_elements), len(diagonal_elements))
    )
    return operator.tocsc()


# def sawtooth_potential(x: float) -> float:
#     """
#     Is the function which returns the potential of a sawtooth junction,
#     i.e. a junction with a sawtooth current phase relationship, only in the discretized phi basis.
#     """
#     x_rel = (x - np.pi) % (2*np.pi) - np.pi
#     return (x_rel)**2/(np.pi)**2 # normalized to have a maximum of 1


def sawtooth_potential(phi_pts):
    # definition from Andras
    skewness = 0.99
    N = 1000
    V = np.zeros_like(phi_pts)
    for idx in range(1, N + 1):
        V += (skewness + 1) * (-skewness) ** (idx - 1) * np.cos(idx * phi_pts) / idx**2
    return -V


def _capactiance_variable_for_branch(branch_type: str):
    """
    Returns the parameter name that stores the capacitance of the branch
    """
    if "C" in branch_type:
        return "EC"
    elif "JJ" in branch_type:
        return "ECJ"
    else:
        raise ValueError("Branch type is not a capacitor or a JJ")


def truncation_template(
    system_hierarchy: list, individual_trunc_dim: int = 6, combined_trunc_dim: int = 30
) -> list:
    """
    Function to generate a template for defining the truncated dimensions for subsystems
    when hierarchical diagonalization is used.

    Parameters
    ----------
    system_hierarchy:
        list which sets the system hierarchy
    individual_trunc_dim:
        The default used to set truncation dimension for subsystems which do not
        use hierarchical diagonalization, by default 6
    combined_trunc_dim:
        The default used to set the truncated dim for subsystems which use hierarchical
        diagonalization, by default 30

    Returns
    -------
        The template for setting the truncated dims for the Circuit instance when
        hierarchical diagonalization is used.
    """
    trunc_dims: List[Union[int, list]] = []
    for subsystem_hierarchy in system_hierarchy:
        if subsystem_hierarchy == flatten_list_recursive(subsystem_hierarchy):
            trunc_dims.append(individual_trunc_dim)
        else:
            trunc_dims.append(
                [combined_trunc_dim, truncation_template(subsystem_hierarchy)]
            )
    return trunc_dims


def get_trailing_number(input_str: str) -> Union[int, None]:
    """
    Returns the number trailing a string given as input. Example:
        $ get_trailing_number("a23")
        $ 23

    Parameters
    ----------
    input_str:
        String which ends with a number

    Returns
    -------
        returns the trailing integer as int, else returns None
    """
    match = re.search(r"\d+$", input_str)
    return int(match.group()) if match else None


def get_operator_number(input_str: str) -> int:
    """
    Returns the number inside an operator name. Example:
        $ get_operator_number("annihilation9_operator")
        $ 9

    Parameters
    ----------
    input_str:
        operator name (one of the methods ending with `_operator`)

    Returns
    -------
        returns the integer as int, else returns None
    """
    match = re.search(r"(\d+)", input_str)
    number = int(match.group())
    if not number:
        raise Exception(f"{input_str} is not a valid operator name.")
    return number


def _identity_phi(grid: discretization.Grid1d) -> csc_matrix:
    """
    Returns identity operator in the discretized_phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the identity operator

    Returns
    -------
        identity operator in the discretized phi basis
    """
    pt_count = grid.pt_count
    return sparse.identity(pt_count, format="csc")


def _phi_operator(grid: discretization.Grid1d) -> csc_matrix:
    """
    Returns phi operator in the discretized_phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the phi operator

    Returns
    -------
        phi operator in the discretized phi basis
    """
    pt_count = grid.pt_count

    phi_matrix = sparse.dia_matrix((pt_count, pt_count))
    diag_elements = grid.make_linspace()
    phi_matrix.setdiag(diag_elements)
    return phi_matrix.tocsc()


def _i_d_dphi_operator(grid: discretization.Grid1d) -> csc_matrix:
    """
    Returns i*d/dphi operator in the discretized_phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the identity operator

    Returns
    -------
        i*d/dphi operator in the discretized phi basis
    """
    return grid.first_derivative_matrix(prefactor=-1j)


def _i_d2_dphi2_operator(grid: discretization.Grid1d) -> csc_matrix:
    """
    Returns i*d2/dphi2 operator in the discretized_phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the identity operator

    Returns
    -------
        i*d2/dphi2 operator in the discretized phi basis
    """
    return grid.second_derivative_matrix(prefactor=-1.0)


def _cos_phi(grid: discretization.Grid1d) -> csc_matrix:
    """
    Returns cos operator in the discretized_phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the identity operator

    Returns
    -------
        cos operator in the discretized phi basis
    """
    pt_count = grid.pt_count

    cos_op = sparse.dia_matrix((pt_count, pt_count))
    diag_elements = np.cos(grid.make_linspace())
    cos_op.setdiag(diag_elements)
    return cos_op.tocsc()


def _sin_phi(grid: discretization.Grid1d) -> csc_matrix:
    """
    Returns sin operator in the discretized_phi basis.

    Parameters
    ----------
    grid:
        Grid used to generate the identity operator

    Returns
    -------
        sin operator in the discretized phi basis
    """
    pt_count = grid.pt_count

    sin_op = sparse.dia_matrix((pt_count, pt_count))
    diag_elements = np.sin(grid.make_linspace())
    sin_op.setdiag(diag_elements)
    return sin_op.tocsc()


def _identity_theta(ncut: int) -> csc_matrix:
    """
    Returns Operator identity in the charge basis.
    """
    dim_theta = 2 * ncut + 1
    return sparse.identity(dim_theta, format="csc")


def _n_theta_operator(ncut: int) -> csc_matrix:
    """
    Returns charge operator `n` in the charge basis.
    """
    dim_theta = 2 * ncut + 1
    diag_elements = np.arange(-ncut, ncut + 1)
    n_theta_matrix = sparse.dia_matrix(
        (diag_elements, [0]), shape=(dim_theta, dim_theta)
    ).tocsc()
    return n_theta_matrix


def _exp_i_theta_operator(ncut, prefactor=1) -> csc_matrix:
    r"""
    Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.
    """
    # if type(prefactor) != int:
    #     raise ValueError("Prefactor must be an integer")
    dim_theta = 2 * ncut + 1
    matrix = sparse.dia_matrix(
        (np.ones(dim_theta), [-prefactor]),
        shape=(dim_theta, dim_theta),
    ).tocsc()
    return matrix


def _exp_i_theta_operator_conjugate(ncut) -> csc_matrix:
    r"""
    Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.
    """
    dim_theta = 2 * ncut + 1
    matrix = sparse.dia_matrix(
        (np.ones(dim_theta), [1]),
        shape=(dim_theta, dim_theta),
    ).tocsc()
    return matrix


def _cos_theta(ncut: int) -> csc_matrix:
    """Returns operator :math:`\\cos \\varphi` in the charge basis"""
    cos_op = 0.5 * (_exp_i_theta_operator(ncut) + _exp_i_theta_operator_conjugate(ncut))
    return cos_op


def _sin_theta(ncut: int) -> csc_matrix:
    """Returns operator :math:`\\sin \\varphi` in the charge basis"""
    sin_op = (
        -1j
        * 0.5
        * (_exp_i_theta_operator(ncut) - _exp_i_theta_operator_conjugate(ncut))
    )
    return sin_op


def _generate_symbols_list(
    var_str: str, iterable_list: List[int] or ndarray
) -> List[sm.Symbol]:
    """
    Returns the list of symbols generated using the var_str + iterable as the name
    of the symbol.

    Parameters
    ----------
    var_str:
        name of the variable which needs to be generated
    iterable_list:
        The list of indices which generates the symbols
    """
    return [sm.symbols(var_str + str(iterable)) for iterable in iterable_list]


def is_potential_term(term: sm.Expr) -> bool:
    """
    Determines if a given sympy expression term is part of the potential

    Parameters
    ----------
    term: sm.Expr
        a single terms in the form of Sympy expression.

    Returns
    -------
    bool
        True if the term is part of the potential of this instance's Hamiltonian
    """
    for symbol in term.free_symbols:
        if "θ" in symbol.name or "Φ" in symbol.name:
            return True
    return False


def example_circuit(qubit: str) -> str:
    """
    Returns example input strings for AnalyzeQCircuit and CustomQCircuit for some of the
    popular qubits.

    Parameters
    ----------
    qubit:
        "fluxonium" or "transmon" or "zero_pi" or "cos2phi" choosing the respective
        example input strings.
    """

    # example input strings for popular qubits
    inputs_by_qubit_name = dict(
        fluxonium="nodes: 2\nbranches:\nJJ	1,2	Ej	Ecj\nL	1,2	El\nC	1,2	Ec",
        transmon="nodes: 2\nbranches:\nC\t1,2\tEc\nJJ\t1,2\tEj\tEcj\n",
        cos2phi="nodes: 4\nbranches:\nC\t1,3\tEc\nJJ\t1,2\tEj\tEcj\nJJ\t3, "
        "4\tEj\tEcj\nL\t1,4\tEl\nL\t2,3\tEl\n\n",
        zero_pi="nodes: 4\nbranches:\nJJ\t1,2\tEj\tEcj\nL\t2,3\tEl\nJJ\t3,"
        "4\tEj\tEcj\nL\t4,1\tEl\nC\t1,3\tEc\nC\t2,4\tEc\n",
    )

    if qubit in inputs_by_qubit_name:
        return inputs_by_qubit_name[qubit]
    else:
        raise AttributeError("Qubit not available or invalid input.")


def grid_operator_func_factory(inner_op: Callable, index: int) -> Callable:
    def operator_func(self: "Subsystem"):
        return self._kron_operator(
            inner_op(self.discretized_grids_dict_for_vars()[index]), index
        )

    return operator_func


def hierarchical_diagonalization_func_factory(symbol_name: str) -> Callable:
    def operator_func(self: "Subsystem"):
        return Qobj_to_scipy_csc_matrix(self.get_operator_by_name(symbol_name))

    return operator_func


def keep_terms_for_subsystem(sym_expr, subsys, substitute_zero=False):
    if substitute_zero:
        for var_sym in sym_expr.free_symbols:
            sym_expr = sym_expr.subs(var_sym, 0)
        return sym_expr
    terms = sym_expr.as_ordered_terms()
    for term in terms:
        var_indices = [
            get_trailing_number(sym_var.name) for sym_var in list(term.free_symbols)
        ]
        if len(set(var_indices) & set(subsys.dynamic_var_indices)) == 0:
            sym_expr = sym_expr - term
    return sym_expr


def operator_func_factory(
    inner_op: Callable, index: int, op_type: Optional[str] = None
) -> Callable:
    def operator_func(self, op_type=op_type):
        prefactor = None
        if self.ext_basis == "harmonic":
            if op_type in ["position", "sin", "cos"]:
                prefactor = self.osc_lengths[index] / (2**0.5)
            elif op_type == "momentum":
                prefactor = 1 / (self.osc_lengths[index] * 2**0.5)
        if prefactor:
            return self._kron_operator(
                inner_op(self.cutoffs_dict()[index], prefactor=prefactor), index
            )
        else:
            return self._kron_operator(inner_op(self.cutoffs_dict()[index]), index)

    return operator_func


def compose(f: Callable, g: Callable) -> Callable:
    """Returns the function f o g:  x |-> f(g(x))"""

    def g_after_f(x: Any) -> Any:
        return f(g(x))

    return g_after_f


def _cos_dia(x: csc_matrix) -> csc_matrix:
    """
    Take the diagonal of the array x, compute its cosine, and fill the result into
    the diagonal of a sparse matrix
    """
    return sparse.diags(np.cos(x.diagonal())).tocsc()


def _sin_dia(x: csc_matrix) -> csc_matrix:
    """
    Take the diagonal of the array x, compute its sine, and fill the result into
    the diagonal of a sparse matrix.
    """
    return sparse.diags(np.sin(x.diagonal())).tocsc()


def _sin_dia_dense(x: ndarray) -> ndarray:
    """
    This is a special function to calculate the sin of dense diagonal matrices
    """
    return np.diag(np.sin(x.diagonal()))


def _cos_dia_dense(x: ndarray) -> ndarray:
    """
    This is a special function to calculate the cos of dense diagonal matrices
    """
    return np.diag(np.cos(x.diagonal()))


def matrix_power_sparse(dense_mat: ndarray, n: int) -> csc_matrix:
    sparse_mat = sparse.csc_matrix(dense_mat)
    return sparse_mat**n


def round_symbolic_expr(expr: sm.Expr, number_of_digits: int) -> sm.Expr:
    rounded_expr = expr.expand()
    for term in sm.preorder_traversal(expr.expand()):
        if isinstance(term, sm.Float):
            rounded_expr = rounded_expr.subs(term, round(term, number_of_digits))
    return rounded_expr


def yaml_like_out_with_pp(circuit_yaml):
    import pyparsing as pp

    code = circuit_input.remove_comments(circuit_yaml)
    code = circuit_input.remove_branchline(code)
    code = circuit_input.strip_empty_lines(code)

    bp = [
        circuit_input.BRANCHES.parse_string(branch)
        for branch in [branch for branch in code.splitlines()]
    ]

    yaml_like_out = []
    for branch in bp:
        yaml_branch = []
        for idx, param in enumerate(branch):
            if isinstance(param, pp.ParseResults):
                parse_type = param.getName()
                if parse_type == "ASSIGN" or parse_type == "AUX_PARAM":
                    yaml_branch.append(
                        param[0] + "=" + "".join([str(x) for x in param[1:] if x])
                    )
                elif parse_type == "SYMBOL":
                    yaml_branch.append(str(param[0]))
                else:
                    yaml_branch.append("".join([str(x) for x in param if x]))
            else:
                yaml_branch.append(param)
        yaml_like_out.append(yaml_branch)
    return yaml_like_out


def assemble_circuit(
    circuit_list: List[str],
    couplers: str,
    rename_parameters=False,
) -> Tuple[str, List[Dict[int, int]]]:
    """
    Assemble a yaml string for a large circuit that are made of smaller
    sub-circuits and coupling elements. This method takes a list of Sub-circuit yaml
    strings as the first argument, and a yaml string that characterize the coupler
    branches as the second argument. For example, if one wish to make a yaml string for
    a circuit consist of a grounded fluxonium capacitively coupled to another fluxonium,
    then one need to define:

    circuit_1 = '''
    branches:
    - [C, 0, 1, EC = 1]
    - [JJ, 0, 1, EJ = 20, ECJ = 3]
    - [L, 0, 1, EL = 10]
    '''
    circuit_2 = '''
    branches:
    - [C, 0, 1, EC = 3]
    - [JJ, 0, 1, EJ = 1, ECJ = 2]
    - [L, 0, 1, EL = 0.5]
    '''
    circuit_list = [circuit_1, circuit_2]
    couplers = '''
    branches:
    - [C, 1: 1, 2: 1, E_coup = 1]
    '''

    If rename_parameters argument set to False, the resulting yaml string for the assembled
    composite circuit is:
    branches:
    - [C, 0, 1, EC = 1]
    - [JJ, 0, 1, EJ = 20, ECJ = 3]
    - [L, 0, 1, EL = 10]
    - [C, 0, 2, EC]
    - [JJ, 0, 2, EJ, ECJ]
    - [L, 0, 2, EL]
    - [C, 1, 2, E_coup = 1]

    If rename_parameters argument set to True, the resulting yaml string for the assembled
    composite circuit is:
    branches:
    - [C, 0, 1, EC_1 = 1]
    - [JJ, 0, 1, EJ_1 = 20, ECJ_1 = 3]
    - [L, 0, 1, EL_1 = 10]
    - [C, 0, 2, EC_2 = 3]
    - [JJ, 0, 2, EJ_2 = 1, ECJ_2 = 2]
    - [L, 0, 2, EL_2 = 0.5]
    - [C, 1, 2, E_coup_12 = 1]

    The yaml strings for each coupler circuit follow the syntax of input strings used in the
    custom circuit module, whereas the syntax for coupler branches is different. Each coupler
    branch is represented by:
    <branch-type>, <subcircuit-index>:<node-index-of-the-circuit>, <subcircuit-index>:<node-index-of-the-circuit>, <param-1> [, <param-2>]]

    All the grounded sub-circuit share the same ground node in the composite circuit.
    The parameter symbols are global, i.e. the same parameter symbol that appears in different
    subcircuit yaml strings will be treated as one parameter in the composite circuit. The
    symbolic parameters are only initialized once, with the value specified at the first
    instance of appearance (notice the initial value for EC in the above example).

    Parameters
    ----------
    circuit_list:
        A list of yaml strings encoding branches of sub-circuits.
    couplers:
        A yaml string that encodes information of coupler branches
    rename_parameters:
        If set to True, parameters in the sub-circuits will be renamed as <original-parameter-name>_<sub-circuit-index>
        parameters in the couplers will be renamed as <original-parameter-name>_<sub-circuit-1-index><sub-circuit-2-index>

    Returns
    -------
        A yaml string for the composite circuit, which can be used as the input for the custom
        circuit module, and a list of dictionaries which provides the mapping between the
        node indices of the sub-circuits (keys) and those of the composite circuit (values).
    """
    # identify numbers of subcircuits
    subcircuit_number = len(circuit_list)
    subcircuit_branches_list = []
    subcircuit_nodes_list = []
    subcircuit_is_grounded_list = []
    subcircuit_node_number_list = []
    subcircuit_node_index_dict_list = []
    for circuit_yaml in circuit_list:
        # load subcircuit yaml strings
        subcircuit_branches = yaml_like_out_with_pp(circuit_yaml)
        # append the dictionary for each subcircuit
        subcircuit_branches_list.append(subcircuit_branches)
        # for each subcircuit, extract their node indices
        subcircuit_nodes = [
            [subcircuit_branch[1], subcircuit_branch[2]]
            for subcircuit_branch in subcircuit_branches
        ]
        subcircuit_nodes = [
            *unique_elements_in_list(flatten_list_recursive(subcircuit_nodes))
        ]
        # add node indices of each subcircuit to a single list
        subcircuit_nodes_list.append(subcircuit_nodes)
        # add total node number of each subcircuit to a single list
        subcircuit_node_number_list.append(len(subcircuit_nodes))
        subcircuit_is_grounded_list.append(True if (0 in subcircuit_nodes) else False)
    # generate a dictionary for each subcircuit which has subcircuit node indices as keys and
    # the assembled circuit node indices as values
    node_index_offset = 0
    for subcircuit_index in range(subcircuit_number):
        subcircuit_node_index_dict = {}
        for subcircuit_node_index in subcircuit_nodes_list[subcircuit_index]:
            subcircuit_node_index_dict[subcircuit_node_index] = (
                subcircuit_node_index + node_index_offset
            )
        if subcircuit_is_grounded_list[subcircuit_index]:
            subcircuit_node_index_dict[0] = 0
        node_index_offset += (
            subcircuit_node_number_list[subcircuit_index]
            - subcircuit_is_grounded_list[subcircuit_index]
        )
        subcircuit_node_index_dict_list.append(subcircuit_node_index_dict)
    # create new yaml string for the composite circuit
    composite_circuit_yaml = "\nbranches:\n"
    # initialize parameter dictionary
    param_dict = {}
    # write all the subcircuit branch info into the composite circuit yaml,
    # converting their node indices
    for subcircuit_index in range(subcircuit_number):
        for subcircuit_branch in subcircuit_branches_list[subcircuit_index]:
            composite_circuit_yaml += " - ["
            # identify branch type
            branch_type = subcircuit_branch[0]
            composite_circuit_yaml += branch_type + " ,"
            # include the converted first node index
            composite_circuit_yaml += (
                str(
                    subcircuit_node_index_dict_list[subcircuit_index][
                        subcircuit_branch[1]
                    ]
                )
                + " ,"
            )
            # include the converted second node index
            composite_circuit_yaml += (
                str(
                    subcircuit_node_index_dict_list[subcircuit_index][
                        subcircuit_branch[2]
                    ]
                )
                + " ,"
            )
            # identify parameter numbers
            num_params = 2 if "JJ" in branch_type else 1
            # include parameters
            for word in subcircuit_branch[3 : 3 + num_params]:
                if not is_string_float(word):
                    if not rename_parameters:
                        if len(word.split("=")) == 2:
                            param_str, init_val = word.split("=")
                            param_str, init_val = param_str.strip(), float(
                                init_val.strip()
                            )
                            # if the parameter is already initialized, the subsequent initialization
                            # is neglected
                            if param_str in param_dict:
                                composite_circuit_yaml += str(param_str) + ", "
                            else:
                                composite_circuit_yaml += str(word) + ", "
                                param_dict[param_str] = init_val
                        elif len(word.split("=")) == 1:
                            composite_circuit_yaml += str(word) + ", "
                    else:
                        if len(word.split("=")) == 2:
                            param_str, init_val = word.split("=")
                            param_str, init_val = param_str.strip(), float(
                                init_val.strip()
                            )
                            composite_circuit_yaml += (
                                param_str
                                + "_"
                                + str(subcircuit_index + 1)
                                + " = "
                                + str(init_val)
                                + ", "
                            )
                        elif len(word.split("=")) == 1:
                            composite_circuit_yaml += (
                                str(word.strip())
                                + "_"
                                + str(subcircuit_index + 1)
                                + ", "
                            )
                else:
                    composite_circuit_yaml += str(word) + ", "
            composite_circuit_yaml += "]\n"
    # add coupling branches to the composite circuit yaml string
    # load coupler yaml strings
    coupler_branches = yaml_like_out_with_pp(couplers)
    for coupler_branch in coupler_branches:
        composite_circuit_yaml += " - ["
        branch_type = coupler_branch[0]
        composite_circuit_yaml += branch_type + ", "
        # include the converted first node index
        composite_circuit_yaml += (
            str(
                subcircuit_node_index_dict_list[list(coupler_branch[1].keys())[0] - 1][
                    list(coupler_branch[1].values())[0]
                ]
            )
            + " ,"
        )
        # include the converted second node index
        composite_circuit_yaml += (
            str(
                subcircuit_node_index_dict_list[list(coupler_branch[2].keys())[0] - 1][
                    list(coupler_branch[2].values())[0]
                ]
            )
            + " ,"
        )
        # identify parameter numbers
        num_params = 2 if "JJ" in branch_type else 1
        # include parameters
        for word in coupler_branch[3 : 3 + num_params]:
            if not is_string_float(word):
                if not rename_parameters:
                    if len(word.split("=")) == 2:
                        param_str, init_val = word.split("=")
                        param_str, init_val = param_str.strip(), float(init_val.strip())
                        # if the parameter is already initialized, the subsequent initialization
                        # is neglected
                        if param_str in param_dict:
                            composite_circuit_yaml += str(param_str) + ", "
                        else:
                            composite_circuit_yaml += str(word) + ", "
                            param_dict[param_str] = init_val
                    elif len(word.split("=")) == 1:
                        composite_circuit_yaml += str(word) + ", "
                else:
                    if len(word.split("=")) == 2:
                        param_str, init_val = word.split("=")
                        param_str, init_val = param_str.strip(), float(init_val.strip())
                        composite_circuit_yaml += (
                            param_str
                            + "_"
                            + str(list(coupler_branch[1].keys())[0])
                            + str(list(coupler_branch[2].keys())[0])
                            + " = "
                            + str(init_val)
                            + ", "
                        )
                    elif len(word.split("=")) == 1:
                        composite_circuit_yaml += (
                            str(word.strip())
                            + "_"
                            + str(list(coupler_branch[1].keys())[0])
                            + str(list(coupler_branch[2].keys())[0])
                            + ", "
                        )
            else:
                composite_circuit_yaml += str(word) + ", "
        composite_circuit_yaml += "]\n"
    return composite_circuit_yaml, subcircuit_node_index_dict_list


def assemble_transformation_matrix(
    transformation_matrix_list: List[ndarray],
) -> ndarray:
    """
    Assemble a transformation matrix for a large circuit that are made of smaller sub-circuits
    and coupling elements. This method takes a list of sub-circuit transformation matrices as
    the argument.

    Parameters
    ----------
    transformation_matrix_list:
        A list of transformation matrices as numpy ndarray.

    Returns
    -------
        A numpy ndarray for the composite circuit.
    """

    return sp.linalg.block_diag(*transformation_matrix_list)
