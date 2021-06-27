# sweeps.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

import scqubits.settings as settings

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep


if settings.IN_IPYTHON:
    pass
else:
    pass


def matelem_by_name(
        sweep: "ParameterSweep",
        param_indices: Tuple[int, ...],
        param_vals: Tuple[float, ...],
        operator_name: Union[str, None] = None,
        subsystem=None,
) -> np.ndarray:
    subsys_index = sweep.get_subsys_index(subsystem)
    bare_evecs = sweep["bare_evecs"][subsys_index][param_indices]
    return subsystem.matrixelement_table(
        operator=operator_name,
        evecs=bare_evecs,
        evals_count=subsystem.truncated_dim,
    )