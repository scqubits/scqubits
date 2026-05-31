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

from __future__ import annotations

from collections.abc import Callable

import scqubits.settings as settings


def get_map_method(num_cpus: int) -> Callable:
    """Selects the correct `.map` method depending on the specified number of desired
    cores. If num_cpus>1, a multiprocessing/pathos pool is used.

    A pool created here is cached in ``settings.POOL`` and reused on subsequent
    calls that request the same worker count and backend, rather than spawning a
    fresh pool every time (a single ``ParameterSweep.run`` otherwise starts one
    pool per bare-subsystem sweep plus the dressed sweep). When the cached pool no
    longer matches the request it is shut down before a new one is started.

    Parameters
    ----------
    num_cpus:
        number of worker processes requested by the caller; ``1``
        returns the built-in ``map``, ``> 1`` uses a multiprocessing
        pool of the configured kind.

    Returns
    -------
    A ``.map``-style callable to be used by the caller.  For
    ``num_cpus == 1`` this is the built-in ``map``; otherwise it is
    the bound ``map`` method of a cached or freshly-started pathos or
    multiprocessing pool (selected by ``settings.MULTIPROC``).
    """
    if num_cpus == 1:
        return map

    # num_cpus > 1 -----------------
    existing_pool = settings.POOL
    if existing_pool is not None:
        if _pool_is_reusable(existing_pool, num_cpus):
            return existing_pool.map
        _shutdown_pool(existing_pool)
        settings.POOL = None

    settings.POOL = _new_pool(num_cpus)
    return settings.POOL.map


def _pool_is_reusable(pool: object, num_cpus: int) -> bool:
    """Return True if the cached pool matches the configured backend and cpu count."""
    if settings.MULTIPROC == "pathos":
        return getattr(pool, "nodes", None) == num_cpus
    if settings.MULTIPROC == "multiprocessing":
        return getattr(pool, "_processes", None) == num_cpus
    return False


def _shutdown_pool(pool: object) -> None:
    """Terminate a pool that is about to be discarded; never raise on cleanup."""
    for method_name in ("terminate", "close", "clear"):
        method = getattr(pool, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception:
            pass


def _new_pool(num_cpus: int) -> object:
    """Start a fresh pool of the configured kind."""
    if settings.MULTIPROC == "pathos":
        try:
            import dill
            import pathos
        except ImportError:
            raise ImportError(
                "scqubits multiprocessing mode set to 'pathos'. Need but cannot find"
                " 'pathos'/'dill'!"
            )
        dill.settings["recurse"] = True
        return pathos.pools.ProcessPool(nodes=num_cpus)
    if settings.MULTIPROC == "multiprocessing":
        import multiprocessing

        return multiprocessing.Pool(processes=num_cpus)
    raise ValueError(
        "Unknown multiprocessing type: settings.MULTIPROC = {}".format(
            settings.MULTIPROC
        )
    )
