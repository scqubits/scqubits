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
from typing import TYPE_CHECKING, Any, Callable, List, Union

import numpy as np
import sympy as sm
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

from scqubits.core import discretization as discretization
from scqubits.utils.misc import flatten_list_recursive

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem


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
    diag_elements = np.cos(grid.make_linspace())
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


def _exp_i_theta_operator(ncut) -> csc_matrix:
    r"""
    Operator :math:`\cos(\theta)`, acting only on the `\theta` Hilbert subspace.
    """
    dim_theta = 2 * ncut + 1
    matrix = sparse.dia_matrix(
        (np.ones(dim_theta), [-1]),
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
        if not self.hierarchical_diagonalization:
            return self._kron_operator(
                inner_op(self.grids_dict_for_discretized_extended_vars()[index]), index
            )
        else:
            return self.identity_wrap_for_hd(
                inner_op(self.grids_dict_for_discretized_extended_vars()[index]), index
            ).data.tocsc()

    return operator_func


def operator_func_factory(inner_op: Callable, index: int) -> Callable:
    def operator_func(self):
        if not self.hierarchical_diagonalization:
            return self._kron_operator(inner_op(self.cutoffs_dict()[index]), index)
        else:
            return self.identity_wrap_for_hd(
                inner_op(self.cutoffs_dict()[index]), index
            ).data.tocsc()

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
