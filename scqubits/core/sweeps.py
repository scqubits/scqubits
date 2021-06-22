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

import functools
import itertools

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

import scqubits.settings as settings

from scqubits.core.namedslots_array import NamedSlotsNdarray

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep


if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def generator(sweep: "ParameterSweep", func: callable, **kwargs) -> np.ndarray:
    """Method for computing custom data as a function of the external parameter,
    calculated via the function `func`.

    Parameters
    ----------
    sweep:
        ParameterSweep object containing HilbertSpace and spectral information
    func:
        signature: `func(parametersweep, [paramindex_tuple, paramvals_tuple,
        **kwargs])`, specifies how to calculate the data for a single choice of
        parameter(s)
    **kwargs:
        keyword arguments to be included in func

    Returns
    -------
        array of custom data
    """
    reduced_parameters = sweep._parameters.create_sliced(
        sweep._current_param_indices, remove_fixed=False
    )
    total_count = np.prod(reduced_parameters.counts)

    def func_effective(paramindex_tuple: Tuple[int], params, **kw) -> Any:
        paramvals_tuple = params[paramindex_tuple]
        return func(
            sweep,
            paramindex_tuple=paramindex_tuple,
            paramvals_tuple=paramvals_tuple,
            **kw,
        )

    if hasattr(func, "__name__"):
        func_name = func.__name__
    else:
        func_name = ""

    data_array = list(
        tqdm(
            map(
                functools.partial(func_effective, params=reduced_parameters, **kwargs,),
                itertools.product(*reduced_parameters.ranges),
            ),
            total=total_count,
            desc="sweeping " + func_name,
            leave=False,
            disable=settings.PROGRESSBAR_DISABLED,
        )
    )
    data_array = np.asarray(data_array)
    return NamedSlotsNdarray(
        data_array.reshape(reduced_parameters.counts),
        reduced_parameters.paramvals_by_name,
    )
