# sweeps.py
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

from typing import TYPE_CHECKING, Tuple

import numpy as np

from qutip import Qobj

from scqubits.core.qubit_base import QubitBaseClass

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep


def bare_matrixelement(
    sweep: "ParameterSweep",
    paramindex_tuple: Tuple[int, ...],
    paramvals_tuple: Tuple[float, ...],
    operator_name: str,
    subsystem: QubitBaseClass,
) -> np.ndarray:
    """
    Given parameter sweep data, compute and return a matrix element table using the bare
    states of the specified subsystem.

    Parameters
    ----------
    sweep:
        `ParameterSweep` object to be used for the computation
    paramindex_tuple:
        a complete set of parameter indices (i.e. a single point in the multi-dim
        parameter space)
    paramvals_tuple:
        [not used, but required by `generator` interface]
    operator_name:
        operator for which matrix elements are requested, given in string form
    subsystem:
        subsystem belonging to the underlying Hilbert space and compatible with the
        specified operator name

    Returns
    -------
        ndarray of matrix elements, in general complex-valued; shape: square array of
        size set by the truncated_dim of the subsystem
    """
    subsys_index = sweep.get_subsys_index(subsystem)
    bare_evecs = sweep["bare_evecs"][subsys_index][paramindex_tuple]
    return subsystem.matrixelement_table(
        operator=operator_name,
        evecs=bare_evecs,
        evals_count=subsystem.truncated_dim,
    )


def dressed_matrixelement(
    sweep: "ParameterSweep",
    paramindex_tuple: Tuple[int, ...],
    paramvals_tuple: Tuple[float, ...],
    operator: Qobj,
) -> np.ndarray:
    """
    Given parameter sweep data, compute and return a matrix element table using the
    dressed states of the composite Hilbert space.

    Parameters
    ----------
    sweep:
        `ParameterSweep` object to be used for the computation
    paramindex_tuple:
        a complete set of parameter indices (i.e. a single point in the multi-dim
        parameter space)
    paramvals_tuple:
        [not used, but required by `generator` interface]
    operator:
        given as `Qobj`, valid operator in the full Hilbert space

    Returns
    -------
        ndarray of matrix elements, in general complex-valued; shape: square array of
        size set by the truncated_dim of the subsystem
    """
    evecs = sweep["evecs"][paramindex_tuple]
    return np.asarray(
        [[operator.matrix_element(evec1, evec2) for evec1 in evecs] for evec2 in evecs]
    )
