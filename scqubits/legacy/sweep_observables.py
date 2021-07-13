# sweep_observables.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
from numpy import ndarray

import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

if TYPE_CHECKING:
    from scqubits import Oscillator
    from scqubits.core.qubit_base import QubitBaseClass
    from scqubits.legacy._param_sweep import _ParameterSweep


def dispersive_chi(
    sweep: "_ParameterSweep",
    param_index: int,
    qubit_subsys: "QubitBaseClass",
    osc_subsys: "Oscillator",
    chi_indices: Tuple[int, int] = None,
) -> Union[float, ndarray]:
    r"""For a given ParameterSweep, calculate dispersive shift data for a single value
    of the external parameter. The dispersive shift relates to a qubit subsystem
    coupled to an oscillator subsystem. :math:`\chi_j` is the shift of
    qubit level :math:`j` due to the addition of a photon in the oscillator. It is
    calculated here from the exact spectrum by means of
    :math:`\chi_j = E_{n=1,j} - E_{n=0,j} - \hbar\omega_\text{osc}`.

    Parameters
    ----------
    sweep: ParameterSweep
    param_index:
        index of the parameter value for which chis should be calculated
    qubit_subsys:
    osc_subsys:
    chi_indices:
        If specified, calculate chi_i - chi_j; otherwise return table of all chis in subspace of qubit_subsys

    Returns
    -------
        chi_i - chi_j   or   chi_0, chi_1, ...
    """
    qubitsys_index = sweep.get_subsys_index(qubit_subsys)
    oscsys_index = sweep.get_subsys_index(osc_subsys)
    if isinstance(chi_indices, tuple):
        chi_count = 2
        chi_range: Union[Tuple[int, int], List[int]] = chi_indices
    else:
        chi_count = qubit_subsys.truncated_dim
        chi_range = list(range(chi_count))

    chi_values = np.empty(chi_count, dtype=np.float_)
    omega = osc_subsys.E_osc
    # chi_j = E_1j - E_0j - omega
    for j in chi_range:
        bare_0j = utils.make_bare_labels(
            sweep.subsystem_count, (qubitsys_index, j), (oscsys_index, 0)
        )
        bare_1j = utils.make_bare_labels(
            sweep.subsystem_count, (qubitsys_index, j), (oscsys_index, 1)
        )
        energy_0j = sweep.lookup.energy_bare_index(bare_0j, param_index)
        energy_1j = sweep.lookup.energy_bare_index(bare_1j, param_index)
        if energy_0j and energy_1j:
            chi_values[j] = energy_1j - energy_0j - omega
        else:
            chi_values[j] = np.NaN

    if chi_indices is not None:
        return chi_values[1] - chi_values[0]
    return chi_values


def qubit_matrixelement(
    sweep: "_ParameterSweep",
    param_index: int,
    qubit_subsys: "QubitBaseClass",
    qubit_operator: ndarray,
) -> ndarray:
    """
    For given ParameterSweep and parameter_index, calculate the matrix elements for
    the provided qubit operator.

    Parameters
    ----------
    sweep:
    param_index:
    qubit_subsys:
    qubit_operator:
       operator within the qubit subspace
    """
    bare_evecs = sweep.lookup.bare_eigenstates(qubit_subsys, param_index=param_index)
    return spec_utils.get_matrixelement_table(qubit_operator, bare_evecs)
