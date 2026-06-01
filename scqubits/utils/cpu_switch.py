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

import atexit
import logging
import os

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Optional

import scqubits.settings as settings

LOGGER = logging.getLogger(__name__)

# Environment variables read by the common BLAS/OpenMP backends at numpy import.
_BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
    """Terminate a pool that is about to be discarded.

    Cleanup failures are logged rather than raised, so discarding a pool never
    interrupts the caller, while a failure that could leak worker processes still
    leaves a diagnostic trail.

    Parameters
    ----------
    pool:
        pool object to shut down; missing cleanup methods are skipped.
    """
    for method_name in ("terminate", "close", "clear"):
        method = getattr(pool, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception:
            LOGGER.warning(
                "scqubits: worker-pool cleanup via %s() failed; worker processes "
                "may linger.",
                method_name,
                exc_info=True,
            )


def _validated_blas_thread_cap() -> Optional[int]:
    """Return the validated ``settings.MULTIPROC_BLAS_THREADS`` value.

    Returns
    -------
    The configured positive thread cap, or ``None`` when capping is disabled.
    """
    threads = getattr(settings, "MULTIPROC_BLAS_THREADS", None)
    if threads is None:
        return None
    # bool is an int subclass; reject it explicitly to avoid True -> 1 surprises.
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError(
            "settings.MULTIPROC_BLAS_THREADS must be a positive int or None, got "
            "{!r}".format(threads)
        )
    if threads < 1:
        raise ValueError(
            "settings.MULTIPROC_BLAS_THREADS must be a positive int or None, got "
            "{}".format(threads)
        )
    return threads


@contextmanager
def _capped_blas_threads() -> Iterator[None]:
    """Temporarily cap worker BLAS/OpenMP threads via the environment.

    The thread-count environment variables are set on entry and restored to their
    prior values on exit, so the cap applies only to worker processes created
    inside the ``with`` block (they capture the capped environment as they fork or
    spawn) and never permanently alters the parent process's environment. No-op
    when ``settings.MULTIPROC_BLAS_THREADS`` is ``None``.
    """
    threads = _validated_blas_thread_cap()
    if threads is None:
        yield
        return
    value = str(threads)
    saved = {var: os.environ.get(var) for var in _BLAS_THREAD_ENV_VARS}
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ[var] = value
    try:
        yield
    finally:
        for var, original in saved.items():
            if original is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original


def _new_pool(num_cpus: int) -> object:
    """Start a fresh pool of the configured kind.

    The BLAS/OpenMP thread cap is applied only while the worker processes are
    created (they capture the capped environment as they spawn or fork); the parent
    process's environment is restored immediately afterwards.

    Parameters
    ----------
    num_cpus:
        number of worker processes for the new pool.

    Returns
    -------
    A freshly started pathos or multiprocessing pool, selected by
    ``settings.MULTIPROC``.
    """
    with _capped_blas_threads():
        if settings.MULTIPROC == "pathos":
            try:
                import dill
                import pathos
            except ImportError:
                raise ImportError(
                    "scqubits multiprocessing mode set to 'pathos'. Need but cannot"
                    " find 'pathos'/'dill'!"
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


@atexit.register
def _shutdown_cached_pool() -> None:
    """Shut down the worker pool cached in ``settings.POOL`` at interpreter exit.

    Without this, the reused pool's worker processes outlive the interpreter and
    are only reaped by the operating system.
    """
    pool = settings.POOL
    if pool is not None:
        _shutdown_pool(pool)
        settings.POOL = None
