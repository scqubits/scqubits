# cpu_switch.py
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

from typing import Callable

import scqubits.settings as settings


def get_map_method(num_cpus: int) -> Callable:
    """
    Selects the correct `.map` method depending on the specified number of desired
    cores. If num_cpus>1, the multiprocessing/pathos pool is started here.

    Parameters
    ----------
    num_cpus: int

    Returns
    -------
    function
        `.map` method to be used by caller
    """
    if num_cpus == 1:
        return map

    # num_cpus > 1 -----------------
    # user is asking for more than 1 cpu; start pool from here
    if settings.MULTIPROC == "pathos":
        try:
            import dill
            import pathos
        except ImportError:
            raise ImportError(
                "scqubits multiprocessing mode set to 'pathos'. Need but cannot find"
                " 'pathos'/'dill'!"
            )
        else:
            dill.settings["recurse"] = True
            settings.POOL = pathos.pools.ProcessPool(nodes=num_cpus)
            return settings.POOL.map
    if settings.MULTIPROC == "multiprocessing":
        import multiprocessing

        settings.POOL = multiprocessing.Pool(processes=num_cpus)
        return settings.POOL.map
    else:
        raise ValueError(
            "Unknown multiprocessing type: settings.MULTIPROC = {}".format(
                settings.MULTIPROC
            )
        )
