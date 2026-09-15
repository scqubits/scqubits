# yaml_assembly.py
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
"""Composite-circuit YAML assembly: parse sub-circuit YAML strings, merge
them into one composite YAML, and block-diagonally combine transformation
matrices.
"""

from __future__ import annotations

import scipy as sp

from numpy import ndarray

from scqubits.core.circuit_internals import input as circuit_input
from scqubits.utils.misc import (
    flatten_list_recursive,
    is_string_float,
    unique_elements_in_list,
)


def yaml_like_out_with_pp(circuit_yaml: str) -> list[list]:
    """Parse a circuit YAML string into a list of branch token lists.

    The input is preprocessed (comments, branch-line markers, and empty
    lines stripped) and each remaining line is parsed with
    :data:`scqubits.core.circuit_input.BRANCHES`. Each parsed branch is
    converted into a flat list of strings/values suitable for further
    YAML emission.

    Parameters
    ----------
    circuit_yaml:
        circuit YAML string in the syntax accepted by the custom-circuit
        module

    Returns
    -------
    A list of branches, where each branch is a list of token strings (or
    raw values for non-:class:`pyparsing.ParseResults` items).
    """
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
    circuit_list: list[str],
    couplers: str,
    rename_parameters: bool = False,
) -> tuple[str, list[dict[int, int]]]:
    """Assemble a YAML string for a composite circuit from sub-circuits and couplers.

    This method takes a list of sub-circuit YAML strings as the first argument
    and a YAML string characterizing the coupler branches as the second
    argument. For example, to assemble a grounded fluxonium capacitively
    coupled to another fluxonium, define::

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

    With ``rename_parameters=False``, the resulting YAML string for the
    assembled composite circuit is::

        branches:
        - [C, 0, 1, EC = 1]
        - [JJ, 0, 1, EJ = 20, ECJ = 3]
        - [L, 0, 1, EL = 10]
        - [C, 0, 2, EC]
        - [JJ, 0, 2, EJ, ECJ]
        - [L, 0, 2, EL]
        - [C, 1, 2, E_coup = 1]

    With ``rename_parameters=True``, the resulting YAML string for the
    assembled composite circuit is::

        branches:
        - [C, 0, 1, EC_1 = 1]
        - [JJ, 0, 1, EJ_1 = 20, ECJ_1 = 3]
        - [L, 0, 1, EL_1 = 10]
        - [C, 0, 2, EC_2 = 3]
        - [JJ, 0, 2, EJ_2 = 1, ECJ_2 = 2]
        - [L, 0, 2, EL_2 = 0.5]
        - [C, 1, 2, E_coup_12 = 1]

    The YAML string for each sub-circuit follows the syntax of input strings
    used in the custom circuit module, whereas the syntax for coupler branches
    is different. Each coupler branch is represented by::

        <branch-type>, <subcircuit-index>:<node-index>,
        <subcircuit-index>:<node-index>, <param-1> [, <param-2>]

    All grounded sub-circuits share the same ground node in the composite
    circuit. The parameter symbols are global, i.e., the same parameter symbol
    appearing in different sub-circuit YAML strings is treated as one
    parameter in the composite circuit. The symbolic parameters are only
    initialized once, with the value specified at the first instance of
    appearance (notice the initial value for ``EC`` in the above example).

    Parameters
    ----------
    circuit_list:
        a list of YAML strings encoding branches of sub-circuits
    couplers:
        a YAML string that encodes information about coupler branches
    rename_parameters:
        if ``True``, parameters in the sub-circuits will be renamed as
        ``<original-parameter-name>_<sub-circuit-index>`` and parameters in
        the couplers will be renamed as
        ``<original-parameter-name>_<sub-circuit-1-index><sub-circuit-2-index>``

    Returns
    -------
    A YAML string for the composite circuit, which can be used as the input
    for the custom circuit module, together with a list of dictionaries
    providing the mapping between the node indices of the sub-circuits (keys)
    and those of the composite circuit (values).
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
                subcircuit_node_index + node_index_offset  # type: ignore[operator]
            )
        if subcircuit_is_grounded_list[subcircuit_index]:
            subcircuit_node_index_dict[0] = 0  # type: ignore[index]
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
    return composite_circuit_yaml, subcircuit_node_index_dict_list  # type: ignore[return-value]


def assemble_transformation_matrix(
    transformation_matrix_list: list[ndarray],
) -> ndarray:
    """Assemble a transformation matrix for a composite circuit from sub-circuits.

    This method takes a list of sub-circuit transformation matrices as the
    argument.

    Parameters
    ----------
    transformation_matrix_list:
        A list of transformation matrices as numpy ndarray.

    Returns
    -------
    A numpy ndarray for the composite circuit.
    """

    return sp.linalg.block_diag(*transformation_matrix_list)
