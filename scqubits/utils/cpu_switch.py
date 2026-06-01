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
import warnings

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Optional

import scqubits.settings as settings

LOGGER = logging.getLogger(__name__)

# Environment variables read by the common BLAS/OpenMP backends at numpy import.
_BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# The "BLAS-thread cap cannot take effect" warning is emitted at most once.
_blas_cap_ineffective_warned = False


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


def _import_threadpoolctl() -> Optional[Any]:
    """Return the ``threadpoolctl`` module if installed, else ``None``."""
    try:
        import threadpoolctl
    except ImportError:
        return None
    return threadpoolctl


def _worker_start_method() -> str:
    """Return the process start method of the configured multiprocessing backend.

    Returns
    -------
    ``'fork'``, ``'spawn'``, ``'forkserver'``, or ``''`` if it cannot be queried.
    The pathos backend uses ``multiprocess``, whose default can differ from the
    standard-library ``multiprocessing`` default (notably fork on macOS).
    """
    module_name = (
        "multiprocess" if settings.MULTIPROC == "pathos" else "multiprocessing"
    )
    try:
        import importlib

        return importlib.import_module(module_name).get_start_method() or ""
    except Exception:
        return ""


def _warn_blas_cap_ineffective(threads: int) -> None:
    """Warn (once) when the requested BLAS-thread cap cannot take effect.

    Capping relies on either ``threadpoolctl`` (which retunes OpenBLAS/MKL/OpenMP
    in the parent before workers fork) or spawn-based workers that re-read the
    environment. It cannot work when numpy's BLAS exposes no controller (e.g. Apple
    Accelerate) or when workers are fork-based and ``threadpoolctl`` is unavailable.

    Parameters
    ----------
    threads:
        the requested per-worker thread cap, used only in the warning message.
    """
    global _blas_cap_ineffective_warned
    if _blas_cap_ineffective_warned:
        return
    threadpoolctl = _import_threadpoolctl()
    reason = ""
    if threadpoolctl is not None:
        if not threadpoolctl.threadpool_info():
            reason = (
                "numpy's BLAS backend exposes no thread control (e.g. Apple "
                "Accelerate)"
            )
    elif _worker_start_method() == "fork":
        reason = (
            "workers are fork-based and 'threadpoolctl' is not installed; install "
            "it (pip install threadpoolctl) to cap threads in forked workers, or "
            "export OPENBLAS_NUM_THREADS/MKL_NUM_THREADS before importing scqubits"
        )
    if reason:
        warnings.warn(
            "settings.MULTIPROC_BLAS_THREADS={} has no effect here: {}.".format(
                threads, reason
            ),
            stacklevel=3,
        )
        _blas_cap_ineffective_warned = True


@contextmanager
def _capped_blas_threads() -> Iterator[None]:
    """Temporarily cap worker BLAS/OpenMP threads while a pool is created.

    Two mechanisms are combined so the cap reaches workers regardless of start
    method:

    - the thread-count environment variables are set on entry and restored on
      exit, which spawn-based workers re-read when they re-import numpy;
    - if ``threadpoolctl`` is installed, the parent's BLAS/OpenMP thread count is
      limited for the duration of the block, so fork-based workers inherit the
      reduced count.

    Neither the parent environment nor its thread-pool settings are altered after
    the block exits. No-op when ``settings.MULTIPROC_BLAS_THREADS`` is ``None``; a
    warning is emitted when the cap cannot take effect on the current platform.
    """
    threads = _validated_blas_thread_cap()
    if threads is None:
        yield
        return
    _warn_blas_cap_ineffective(threads)
    value = str(threads)
    saved = {var: os.environ.get(var) for var in _BLAS_THREAD_ENV_VARS}
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ[var] = value
    threadpoolctl = _import_threadpoolctl()
    limiter: ContextManager[Any] = (
        threadpoolctl.threadpool_limits(limits=threads)
        if threadpoolctl is not None
        else nullcontext()
    )
    try:
        with limiter:
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
