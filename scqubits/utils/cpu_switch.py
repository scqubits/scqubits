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
from typing import Any, Callable, Dict, Optional
import warnings
import scqubits.settings as settings


_RAY_OBJECT_CACHE: Dict[str, Any] = {}


def _ray_object_cache_key(obj_ref: Any) -> str:
    return obj_ref.hex() if hasattr(obj_ref, "hex") else repr(obj_ref)


def get_cached_ray_object(obj_ref: Any) -> Any:
    import ray

    key = _ray_object_cache_key(obj_ref)
    if key not in _RAY_OBJECT_CACHE:
        _RAY_OBJECT_CACHE[key] = ray.get(obj_ref)
    return _RAY_OBJECT_CACHE[key]


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
            
            # Initialize Ray only if it isn't already
            if not ray.is_initialized():
                ray.init(num_cpus=num_nodes * cpu_per_node)
            
            def ray_map(func, iterable):
                # Ray has a different execution model from multiprocessing/pathos.
                #
                # With multiprocessing/pathos, `pool.map(func, iterable)` can hand
                # the pool a Python callable directly and the pool implementation
                # takes care of shipping it to worker processes. Ray instead runs
                # explicitly-declared "remote" functions. Arguments to those remote
                # calls are serialized and sent through Ray's distributed object
                # store. That is why the Ray version below looks more involved than
                # a normal pool.map wrapper.
                #
                # Put the user callable in Ray's object store once. This is
                # especially important when `func` closes over large arrays or other
                # heavy objects: passing `func` directly to every task could serialize
                # the same captured data again and again. In ParameterSweep, for
                # example, custom sweep callables often need access to a sweep object
                # whose `_data` dictionary contains large eigenvalue/eigenvector
                # arrays. The Ray-specific ParameterSweep generator stores that sweep
                # object separately in the object store for the same reason.
                # Store func in the Ray object store to avoid serializing large
                # captured arrays into the remote function definition itself.
                func_ref = ray.put(func)

                # Ray automatically dereferences ObjectRefs when they are passed as
                # top-level task arguments. Wrapping the reference in a tuple keeps it
                # as a reference, so each worker can fetch and cache the callable on
                # demand instead of receiving a freshly deserialized copy per task.
                func_ref_box = (func_ref,)

                @ray.remote(num_cpus=cpu_per_node)
                def remote_func(func_ref_box, task_index, x):
                    # This code runs inside a Ray worker process, not in the caller.
                    # `func_ref_box[0]` is a handle to the callable stored above, not
                    # the callable itself. We retrieve it with ray.get through
                    # get_cached_ray_object(), which memoizes the result in the worker
                    # process so repeated tasks on the same worker reuse the same
                    # Python callable.
                    # Keep the dereferenced callable cached once per worker process.
                    func = get_cached_ray_object(func_ref_box[0])

                    # Ray tasks can finish in any order. Return the submission index
                    # together with the result so the generator below can yield values
                    # in the same order as Python's map/pool.map.
                    return task_index, func(x)

                # Keep only a bounded number of Ray tasks in flight. Submitting the
                # entire iterable at once can consume a large amount of memory if the
                # iterable is long or if each `x` is large. A small multiple of the
                # worker count keeps workers busy while preserving backpressure.
                max_pending = max(num_nodes * 2, 1)

                def result_generator():
                    pending = []
                    next_submit_index = 0
                    next_yield_index = 0
                    ready_results = {}
                    iterable_iter = iter(iterable)
                    iterable_done = False

                    def submit_until_full():
                        nonlocal next_submit_index, iterable_done
                        while not iterable_done and len(pending) < max_pending:
                            try:
                                x = next(iterable_iter)
                            except StopIteration:
                                iterable_done = True
                                break
                            pending.append(
                                remote_func.remote(
                                    func_ref_box,
                                    next_submit_index,
                                    x
                                )
                            )
                            next_submit_index += 1

                    submit_until_full()
                    while pending:
                        # Wait for at least one task to finish, then submit more work
                        # to keep the pipeline full. Unlike pool.map, Ray exposes this
                        # scheduling step explicitly via object references.
                        done_refs, pending[:] = ray.wait(
                            pending,
                            num_returns=1
                        )
                        for task_index, result in ray.get(done_refs):
                            ready_results[task_index] = result

                        submit_until_full()

                        # Yield only contiguous results starting at
                        # next_yield_index. Faster later tasks are held in
                        # ready_results until earlier tasks complete, matching the
                        # ordered behavior expected from map().
                        while next_yield_index in ready_results:
                            yield ready_results.pop(next_yield_index)
                            next_yield_index += 1

                return result_generator()

            return ray_map
    else:
        raise ValueError(
            "Unknown multiprocessing type: settings.MULTIPROC = {}".format(
                settings.MULTIPROC
            )
        )
