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
import importlib
import logging
import os
import sys
import warnings

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Optional

import scqubits.settings as settings

LOGGER = logging.getLogger(__name__)

# Backend + start method + cpu count of the pool cached in settings.POOL, so reuse
# does not depend on reading private pool attributes.
_pool_signature: Optional[tuple] = None

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
    """Return True if the cached pool matches the requested backend, start method, cpu count.

    Parameters
    ----------
    pool:
        the cached pool object (unused; reuse is tracked via ``_pool_signature``).
    num_cpus:
        number of worker processes requested by the caller.
    """
    return _pool_signature == (settings.MULTIPROC, _resolve_start_method(), num_cpus)


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


def _backend_module() -> Any:
    """Return the multiprocessing backend module.

    ``multiprocess`` (the dill-backed engine) for the pathos backend, otherwise the
    standard-library ``multiprocessing``.
    """
    name = "multiprocess" if settings.MULTIPROC == "pathos" else "multiprocessing"
    return importlib.import_module(name)


def _resolve_start_method() -> str:
    """Return the process start method the worker pool will use.

    Uses ``settings.MULTIPROC_START_METHOD`` when set, otherwise a platform default:
    ``'fork'`` on Linux, ``'spawn'`` on macOS and Windows (fork-after-threads is
    unsafe on macOS). Falls back to an available method if the requested one is not
    supported by the backend on this platform.

    Returns
    -------
    One of ``'fork'``, ``'spawn'``, ``'forkserver'``.
    """
    try:
        available = set(_backend_module().get_all_start_methods())
    except Exception:
        available = set()
    requested = getattr(settings, "MULTIPROC_START_METHOD", None)
    if requested is None:
        requested = "fork" if sys.platform.startswith("linux") else "spawn"
    if available and requested not in available:
        requested = "spawn" if "spawn" in available else next(iter(available))
    return requested


def _worker_start_method() -> str:
    """Return the start method the worker pool will use, or ``''`` if it can't be queried."""
    try:
        return _resolve_start_method()
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
    # spawn/forkserver workers re-import numpy/scipy and read the thread env vars at
    # import, so the env-var cap is effective regardless of the BLAS backend.
    if _worker_start_method() in ("spawn", "forkserver"):
        return
    # fork: workers inherit the parent's already-initialized BLAS; the env-var cap is
    # inert, so capping relies on threadpoolctl being able to retune the loaded BLAS.
    threadpoolctl = _import_threadpoolctl()
    reason = ""
    if threadpoolctl is None:
        reason = (
            "workers are fork-based and 'threadpoolctl' is not installed; install it "
            "(pip install threadpoolctl), export OPENBLAS_NUM_THREADS/MKL_NUM_THREADS "
            "before importing scqubits, or set "
            "scqubits.settings.MULTIPROC_START_METHOD = 'spawn'"
        )
    elif not threadpoolctl.threadpool_info():
        reason = (
            "workers are fork-based and numpy's BLAS exposes no thread control (e.g. "
            "Apple Accelerate); set scqubits.settings.MULTIPROC_START_METHOD = 'spawn'"
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
    """Start a fresh worker pool of the configured backend and start method.

    The pool is created from an explicit process context (see
    :func:`_resolve_start_method`) so the start method is *chosen* rather than
    inherited -- notably ``'spawn'`` on macOS, where fork-after-threads is unsafe.
    The pathos backend uses ``multiprocess`` (dill-backed), so worker closures still
    pickle under spawn.

    The BLAS/OpenMP thread cap is applied only while the worker processes are created
    (they capture the capped environment as they spawn or fork); the parent process's
    environment is restored immediately afterwards.

    Parameters
    ----------
    num_cpus:
        number of worker processes for the new pool.

    Returns
    -------
    A freshly started pool whose ``.map`` method is used by the caller.
    """
    global _pool_signature
    if settings.MULTIPROC not in ("pathos", "multiprocessing"):
        raise ValueError(
            "Unknown multiprocessing type: settings.MULTIPROC = {}".format(
                settings.MULTIPROC
            )
        )
    method = _resolve_start_method()
    with _capped_blas_threads():
        if settings.MULTIPROC == "pathos":
            try:
                import dill
                import multiprocess as backend
            except ImportError:
                raise ImportError(
                    "scqubits multiprocessing mode set to 'pathos'. Need but cannot"
                    " find 'pathos'/'dill'!"
                )
            dill.settings["recurse"] = True
        else:
            import multiprocessing as backend
        pool = backend.get_context(method).Pool(processes=num_cpus)
    _pool_signature = (settings.MULTIPROC, method, num_cpus)
    return pool


@atexit.register
def _shutdown_cached_pool() -> None:
    """Shut down the worker pool cached in ``settings.POOL`` at interpreter exit.

    Without this, the reused pool's worker processes outlive the interpreter and
    are only reaped by the operating system.
    """
    global _pool_signature
    pool = settings.POOL
    if pool is not None:
        _shutdown_pool(pool)
        settings.POOL = None
        _pool_signature = None
