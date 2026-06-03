# parallel_tuning.py
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
"""Heuristic recommendations for parameter-sweep parallelization.

Choosing good values for ``num_cpus`` (worker processes) and the per-worker
BLAS-thread cap is workload-dependent: small or cheap sweeps run fastest serially,
many cheap points reward many workers each with a single BLAS thread, and a few
expensive *dense* points are better served by one process whose linear algebra uses
all cores. Getting it wrong is costly -- oversubscribing the cores
(``num_cpus`` x BLAS-threads far above the core count) on large dense matrices can
be ~90x slower.

:func:`recommend_parallelization` turns the workload descriptors -- Hilbert-space
dimension, number of grid points, eigenvalue count, and whether sparse
diagonalization will be used -- into a :class:`ParallelConfig`, optionally applying
it live to :mod:`scqubits.settings` (no kernel restart needed; the existing pool
machinery reads the new values at the next pool creation). The recommendation uses
documented dimension/point-count tiers rather than fragile absolute-time estimates;
the thresholds are deliberately conservative defaults.

The function is *pure* -- it never starts worker processes -- so it is safe to call
from Jupyter and from plain Python scripts alike. Only a subsequent sweep with
``num_cpus > 1`` starts workers, which under the spawn start method (macOS, Windows)
requires the usual ``if __name__ == "__main__":`` guard in a plain script.
"""

from __future__ import annotations

import os
import warnings

from dataclasses import dataclass
from typing import Any, Optional, Union

import scqubits.settings as settings

# Dimension tiers (Hilbert-space dimension) used to gauge per-point cost. Sparse
# diagonalization of a few eigenstates is far cheaper per point, so it shifts the
# effective tier down by one.
_SMALL_DIM = 64
_MEDIUM_DIM = 512
_LARGE_DIM = 2048

# Dense systems at or above this dimension with fewer grid points than cores are
# better parallelized through BLAS (one process, all cores on each big solve) than
# across worker processes. Tied to the auto-sparse threshold when present, with a
# matching default so the heuristic stands alone if auto-sparse diag is not installed.
_LARGE_DENSE_DIM = getattr(settings, "SPARSE_DIAG_MIN_DIM", 1000)

# Minimum number of grid points worth parallelizing across workers, per cost tier
# (cheaper per-point work needs more points to amortize the per-task overhead).
# Tier index: 0 tiny, 1 small, 2 medium, 3 large.
_POINTS_BREAK_EVEN = {0: 500, 1: 64, 2: 16, 3: 8}

# Target grid points handled per worker; sets how aggressively to add workers.
_POINTS_PER_WORKER = 4


@dataclass(frozen=True)
class ParallelConfig:
    """Recommended parallelization settings for a parameter sweep.

    Attributes
    ----------
    num_cpus:
        recommended number of worker processes (``>= 1``).
    blas_threads:
        recommended per-worker BLAS/OpenMP thread cap, or ``None`` to leave BLAS
        uncapped. ``None`` accompanies ``num_cpus == 1`` (one process should use all
        cores for its linear algebra); a parallel recommendation always names an
        explicit cap so workers do not oversubscribe the cores.
    reason:
        short human-readable explanation of the recommendation.
    """

    num_cpus: int
    blas_threads: Optional[int]
    reason: str


def _resolve_cores(cores: Optional[int]) -> int:
    """Return the core count to plan against.

    Parameters
    ----------
    cores:
        explicit core count, or ``None`` to detect via ``os.cpu_count()``.
    """
    if cores is not None:
        return max(1, int(cores))
    return os.cpu_count() or 1


def _default_is_sparse(dimension: int, evals_count: int) -> bool:
    """Return whether sparse diagonalization would be used for this workload.

    Mirrors the decision in
    :func:`scqubits.core.hilbert_space._auto_sparse_diag_method` without importing
    it (to avoid an import cycle): sparse is used when automatic sparse
    diagonalization is enabled, the dimension is large enough, and only a small
    fraction of the spectrum is requested.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension being diagonalized.
    evals_count:
        number of eigenvalues/eigenstates requested per grid point.
    """
    return bool(
        getattr(settings, "AUTO_SPARSE_DIAG", False)
        and dimension >= getattr(settings, "SPARSE_DIAG_MIN_DIM", 1000)
        and evals_count
        <= max(1, int(dimension * getattr(settings, "SPARSE_DIAG_MAX_EVALS_FRAC", 0.1)))
    )


def _cost_tier(dimension: int, is_sparse: bool) -> int:
    """Return a per-point cost tier (0 tiny .. 3 large) from the dimension.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension being diagonalized.
    is_sparse:
        whether sparse diagonalization will be used; sparse few-eigenstate solves
        are much cheaper per point and drop the effective tier by one.
    """
    if dimension < _SMALL_DIM:
        tier = 0
    elif dimension < _MEDIUM_DIM:
        tier = 1
    elif dimension < _LARGE_DIM:
        tier = 2
    else:
        tier = 3
    if is_sparse and tier > 0:
        tier -= 1
    return tier


def _recommend(
    dimension: int,
    total_points: int,
    evals_count: int,
    is_sparse: bool,
    cores: int,
) -> ParallelConfig:
    """Return a :class:`ParallelConfig` from explicit workload descriptors.

    The pure core of the heuristic. See the module docstring for the rationale; the
    decision uses dimension/point-count tiers (constants ``_SMALL_DIM`` etc.) and
    always keeps ``num_cpus * blas_threads <= cores``.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension being diagonalized at each grid point.
    total_points:
        total number of grid points in the sweep.
    evals_count:
        number of eigenvalues/eigenstates requested per grid point.
    is_sparse:
        whether sparse diagonalization will be used per point.
    cores:
        number of cores to plan against.
    """
    if total_points <= 1:
        return ParallelConfig(1, None, "single grid point: serial")
    if cores <= 1:
        return ParallelConfig(1, None, "only one core available: serial")

    dense_big = (not is_sparse) and dimension >= _LARGE_DENSE_DIM
    if dense_big and total_points < cores:
        return ParallelConfig(
            1,
            None,
            "{} points of a large dense system (dim {}): one process with uncapped "
            "BLAS lets the linear algebra use all {} cores".format(
                total_points, dimension, cores
            ),
        )

    tier = _cost_tier(dimension, is_sparse)
    break_even = _POINTS_BREAK_EVEN[tier]
    if total_points < break_even:
        return ParallelConfig(
            1,
            None,
            "{} points below the parallel break-even (~{}) for this per-point cost "
            "(dim {}{}): serial".format(
                total_points,
                break_even,
                dimension,
                ", sparse" if is_sparse else ", dense",
            ),
        )

    num_cpus = min(cores, total_points, max(2, total_points // _POINTS_PER_WORKER))
    light_per_point = is_sparse or dimension < _MEDIUM_DIM
    blas_threads = 1 if light_per_point else max(1, cores // num_cpus)
    while num_cpus > 1 and num_cpus * blas_threads > cores:
        num_cpus -= 1
        if not light_per_point:
            blas_threads = max(1, cores // num_cpus)

    reason = (
        "{} points x per-point cost (dim {}, {}) clears the break-even: "
        "{} workers x {} BLAS thread(s) (<= {} cores)".format(
            total_points,
            dimension,
            "sparse" if is_sparse else "dense",
            num_cpus,
            blas_threads,
            cores,
        )
    )
    return ParallelConfig(num_cpus, blas_threads, reason)


def _descriptors_from_system(
    system: Any,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract (dimension, total_points, evals_count) from a scqubits object.

    Recognizes a :class:`.HilbertSpace`, a bare qubit (:class:`.QubitBaseClass`), or
    a :class:`.ParameterSweep`. Unknown attributes come back as ``None`` so explicit
    keyword arguments can fill them in.

    Parameters
    ----------
    system:
        object to introspect for workload descriptors.
    """
    import numpy as np

    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.param_sweep import ParameterSweep
    from scqubits.core.qubit_base import QubitBaseClass

    if isinstance(system, ParameterSweep):
        warnings.warn(
            "recommend_parallelization received a constructed ParameterSweep. With "
            "the default autorun, the sweep has already run; build it with "
            "autorun=False (or pass the HilbertSpace plus num_points) to tune before "
            "the sweep runs.",
            UserWarning,
        )
        dimension = system._hilbertspace.dimension
        total_points = int(np.prod(system._parameters.counts))
        return dimension, total_points, system._evals_count
    if isinstance(system, HilbertSpace):
        return system.dimension, None, None
    if isinstance(system, QubitBaseClass):
        return system.hilbertdim(), None, None
    raise TypeError(
        "recommend_parallelization does not know how to read a workload from a {}; "
        "pass a HilbertSpace, a qubit, a ParameterSweep, or explicit dimension and "
        "num_points.".format(type(system).__name__)
    )


# --------------------------------------------------------------------------------------
# Machine-calibration consumption (optional; see scqubits.utils.parallel_calibration)
# --------------------------------------------------------------------------------------
# Cache the loaded calibration keyed by file modification time so the per-sweep auto
# hook does not re-read the file on every call.
_calibration_cache: dict[str, tuple[float, Any]] = {}


def _get_calibration() -> Any:
    """Return the persisted machine calibration, or ``None`` if absent.

    The result is cached and invalidated when the calibration file's modification
    time changes.
    """
    from scqubits.utils.parallel_calibration import (
        default_calibration_path,
        load_calibration,
    )

    path = default_calibration_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _calibration_cache.pop(path, None)
        return None
    cached = _calibration_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    calibration = load_calibration(path)
    _calibration_cache[path] = (mtime, calibration)
    return calibration


def _recommend_calibrated(
    dimension: int,
    total_points: int,
    evals_count: int,
    is_sparse: bool,
    cores: int,
    calibration: Any,
) -> Optional[ParallelConfig]:
    """Return a recommendation from measured overhead/per-point cost, or ``None``.

    Returns ``None`` when the calibration has no per-point cost sample matching the
    regime, so the caller can fall back to the default tiered heuristic. Parallelism
    is chosen only when the per-point work removed by adding workers exceeds the
    measured per-task dispatch overhead.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension diagonalized per grid point.
    total_points:
        total number of grid points in the sweep.
    evals_count:
        number of eigenvalues/eigenstates requested per point.
    is_sparse:
        whether sparse diagonalization will be used per point.
    cores:
        number of cores to plan against.
    calibration:
        a ``MachineCalibration`` providing measured overhead and per-point cost.
    """
    if total_points <= 1 or cores <= 1:
        return ParallelConfig(1, None, "serial: trivial workload")

    cost = calibration.estimated_cost_per_point(dimension, is_sparse)
    if cost is None:
        return None

    dense_big = (not is_sparse) and dimension >= _LARGE_DENSE_DIM
    if dense_big and total_points < cores:
        return ParallelConfig(
            1,
            None,
            "{} points of a large dense system (dim {}): one process with uncapped "
            "BLAS uses all {} cores".format(total_points, dimension, cores),
        )

    num_cpus = min(cores, total_points, max(2, total_points // _POINTS_PER_WORKER))
    overhead = calibration.overhead_s
    startup = getattr(calibration, "pool_startup_s", 0.0)
    # Per-point gain from adding workers, and the total it must accumulate to repay
    # the one-time pool startup (the spawn re-import paid once per sweep).
    per_point_gain = cost * (1.0 - 1.0 / num_cpus) - overhead
    if per_point_gain <= 0 or total_points * per_point_gain <= startup:
        return ParallelConfig(
            1,
            None,
            "measured: {} points x per-point gain {:.4f}s does not repay the "
            "{:.3f}s pool startup: serial".format(
                total_points, max(0.0, per_point_gain), startup
            ),
        )

    light_per_point = is_sparse or dimension < _MEDIUM_DIM
    blas_threads = 1 if light_per_point else max(1, cores // num_cpus)
    while num_cpus > 1 and num_cpus * blas_threads > cores:
        num_cpus -= 1
        if not light_per_point:
            blas_threads = max(1, cores // num_cpus)
    reason = (
        "measured: {} points x per-point gain {:.4f}s repays the {:.3f}s startup "
        "-> {} workers x {} BLAS thread(s) (<= {} cores)".format(
            total_points, per_point_gain, startup, num_cpus, blas_threads, cores
        )
    )
    return ParallelConfig(num_cpus, blas_threads, reason)


def _decide(
    dimension: int,
    total_points: int,
    evals_count: int,
    is_sparse: bool,
    cores: int,
) -> ParallelConfig:
    """Return a recommendation, using a machine calibration when one is available.

    Falls back to the default tiered heuristic (:func:`_recommend`) when no
    calibration is present or it lacks a matching cost sample.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension diagonalized per grid point.
    total_points:
        total number of grid points in the sweep.
    evals_count:
        number of eigenvalues/eigenstates requested per point.
    is_sparse:
        whether sparse diagonalization will be used per point.
    cores:
        number of cores to plan against.
    """
    calibration = _get_calibration()
    if calibration is not None:
        calibrated = _recommend_calibrated(
            dimension, total_points, evals_count, is_sparse, cores, calibration
        )
        if calibrated is not None:
            return calibrated
    return _recommend(dimension, total_points, evals_count, is_sparse, cores)


def recommend_parallelization(
    system: Any = None,
    *,
    hilbertspace: Any = None,
    qubit: Any = None,
    param_sweep: Any = None,
    num_points: Optional[int] = None,
    dimension: Optional[int] = None,
    evals_count: Optional[int] = None,
    is_sparse: Optional[bool] = None,
    cores: Optional[int] = None,
    apply: bool = False,
    explain: bool = False,
    measure: bool = False,
) -> ParallelConfig:
    """Recommend ``num_cpus`` and a per-worker BLAS-thread cap for a sweep.

    Intended to be called *before* building a sweep, e.g.::

        cfg = scqubits.recommend_parallelization(hilbertspace=hs, num_points=384,
                                                 evals_count=20)
        sweep = scqubits.ParameterSweep(..., num_cpus=cfg.num_cpus)

    The recommendation is a pure function of the workload; it starts no worker
    processes. With ``apply=True`` it sets ``settings.NUM_CPUS`` and
    ``settings.MULTIPROC_BLAS_THREADS`` live, taking effect at the next pool creation
    without a kernel restart.

    Parameters
    ----------
    system:
        a :class:`.HilbertSpace`, a bare qubit, or (for ``autorun=False`` sweeps) a
        :class:`.ParameterSweep` to read the dimension/points from. Optional if
        ``dimension`` (and ``num_points``) are given explicitly. The keyword aliases
        ``hilbertspace``, ``qubit``, and ``param_sweep`` are equivalent ways to pass
        this object.
    hilbertspace:
        alias for ``system`` when passing a :class:`.HilbertSpace`.
    qubit:
        alias for ``system`` when passing a bare qubit.
    param_sweep:
        alias for ``system`` when passing an ``autorun=False`` :class:`.ParameterSweep`.
    num_points:
        total number of grid points in the sweep; overrides a value read from
        ``system``.
    dimension:
        Hilbert-space dimension diagonalized per point; overrides ``system``.
    evals_count:
        number of eigenvalues/eigenstates requested per point; overrides ``system``.
        Defaults to 6 when neither given nor available.
    is_sparse:
        whether sparse diagonalization will be used; defaults to the same rule
        scqubits applies automatically (see ``settings.AUTO_SPARSE_DIAG``).
    cores:
        core count to plan against; defaults to ``os.cpu_count()``.
    apply:
        if ``True``, write the recommendation to ``settings.NUM_CPUS`` and
        ``settings.MULTIPROC_BLAS_THREADS`` for the current session.
    explain:
        if ``True``, print the recommendation and its reason.
    measure:
        if ``True`` and no machine calibration exists yet, run
        :func:`scqubits.calibrate_parallelization` first (which times a short battery
        of sweeps in subprocesses) so the recommendation uses measured break-evens.
        Off by default; the calibration spawns processes, so in a plain script guard
        the entry point with ``if __name__ == "__main__":``.

    Returns
    -------
    The recommended :class:`ParallelConfig`.

    Raises
    ------
    ValueError
        if neither ``system`` nor ``dimension`` and ``num_points`` are provided.
    """
    system = system if system is not None else (hilbertspace or qubit or param_sweep)
    if system is not None:
        sys_dim, sys_points, sys_evals = _descriptors_from_system(system)
        dimension = dimension if dimension is not None else sys_dim
        num_points = num_points if num_points is not None else sys_points
        evals_count = evals_count if evals_count is not None else sys_evals

    if dimension is None or num_points is None:
        raise ValueError(
            "recommend_parallelization needs a workload: pass a HilbertSpace/qubit/"
            "ParameterSweep, or explicit dimension and num_points."
        )
    if evals_count is None:
        evals_count = 6

    if is_sparse is None:
        is_sparse = _default_is_sparse(dimension, evals_count)
    resolved_cores = _resolve_cores(cores)

    if measure:
        from scqubits.utils.parallel_calibration import (
            calibrate_parallelization,
            default_calibration_path,
        )

        if not os.path.exists(default_calibration_path()):
            calibrate_parallelization(persist=True, explain=explain)

    config = _decide(
        int(dimension), int(num_points), int(evals_count), is_sparse, resolved_cores
    )

    if apply:
        settings.NUM_CPUS = config.num_cpus
        settings.MULTIPROC_BLAS_THREADS = config.blas_threads
    if explain:
        print(
            "scqubits parallelization: num_cpus={}, blas_threads={} -- {}".format(
                config.num_cpus, config.blas_threads, config.reason
            )
        )
    return config


def _auto_config(dimension: int, total_points: int, evals_count: int) -> ParallelConfig:
    """Return the heuristic recommendation for a sweep about to run.

    Helper used by the sweep/spectrum methods when ``num_cpus="auto"`` (or
    ``settings.AUTO_PARALLEL`` with ``num_cpus`` left unset). Detects the sparse
    regime, plans against ``os.cpu_count()``, and uses a machine calibration when one
    is present.

    Parameters
    ----------
    dimension:
        Hilbert-space dimension diagonalized per point.
    total_points:
        total number of grid points in the sweep.
    evals_count:
        number of eigenvalues/eigenstates requested per point.
    """
    is_sparse = _default_is_sparse(dimension, evals_count)
    return _decide(
        dimension, total_points, evals_count, is_sparse, _resolve_cores(None)
    )


# Sentinel accepted by num_cpus arguments to trigger the heuristic before a sweep runs.
AUTO: str = "auto"

NumCpusArg = Union[int, str, None]
