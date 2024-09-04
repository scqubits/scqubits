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

import os
from typing import Callable, Optional
import warnings
import scqubits.settings as settings


def get_map_method(num_nodes: int, cpu_per_node: Optional[int] = None) -> Callable:
    """
    Selects the correct `.map` method depending on the specified number of desired
    cores. If num_cpus>1, the multiprocessing/pathos/ray pool is started here.

    Parameters
    ----------
    num_nodes: int
        Number of nodes to use. 
    cpu_per_node: int = 1
        Number of CPUs per node, only used if use ray as backend. By default, 
        will

    Returns
    -------
    function
        `.map` method to be used by caller
    """
    if num_nodes == 1:
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
            settings.POOL = pathos.pools.ProcessPool(nodes=num_nodes)
            return settings.POOL.map
    if settings.MULTIPROC == "multiprocessing":
        import multiprocessing

        settings.POOL = multiprocessing.Pool(processes=num_nodes)
        return settings.POOL.map
    elif settings.MULTIPROC == "ray":
        try:
            import ray
        except ImportError:
            raise ImportError(
                "scqubits multiprocessing mode set to 'ray'. Need but cannot find 'ray'!"
            )
        else:
            
            # determine the number of cpus per task
            if cpu_per_node is None:
                # inspect number of cpus available
                total_cpus = os.cpu_count()
                if total_cpus is None:
                    cpu_per_node = 1
                    warnings.warn("Cannot determine number of CPUs available. "
                                  "Cpu per task set to 1.")
                else:
                    cpu_per_node = total_cpus // num_nodes
            if cpu_per_node < 1:    
                raise ValueError("Number of CPUs per task is less than 1.")
            
            ray.shutdown()
            ray.init(num_cpus=num_nodes * cpu_per_node)
            
            def ray_map(func, iterable):
                @ray.remote(num_cpus=cpu_per_node)
                def remote_func(x):
                    return func(x)

                result_refs = [remote_func.remote(x) for x in iterable]
                return ray.get(result_refs)

            return ray_map
    else:
        raise ValueError(
            "Unknown multiprocessing type: settings.MULTIPROC = {}".format(
                settings.MULTIPROC
            )
        )
