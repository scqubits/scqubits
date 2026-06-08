# parallel_calibration.py
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
"""One-time machine calibration for the parallelization heuristic.

:func:`recommend_parallelization` (see :mod:`scqubits.utils.parallel_tuning`) decides
whether a sweep is worth parallelizing from documented default thresholds. Those
defaults are deliberately conservative because the true break-even depends on the
machine: the per-task dispatch overhead and the per-point diagonalization cost vary
with CPU, BLAS library, and OS.

:func:`calibrate_parallelization` measures those two quantities **once** on the
current machine and writes a small JSON model to a per-user location. The heuristic
then loads that model and replaces its tiered point-count break-even with a measured
one (parallelize a point only when the per-point work it saves exceeds the measured
dispatch overhead). The calibration parameterizes the *workload-aware* heuristic; it
does not cache a single fixed configuration (which would be wrong for differently
sized sweeps).

Each measurement runs in a fresh subprocess (launched as
``python -m scqubits.utils.parallel_calibration``) with the BLAS-thread environment
preset before numpy is imported. Subprocess isolation gives honest BLAS-thread
control and avoids worker-pool contamination; launching by module (not by re-running
the caller's script) means the calibration call itself needs no
``if __name__ == "__main__":`` guard, in Jupyter or in a plain script alike.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import scqubits.settings as settings

# Environment variables the common BLAS/OpenMP backends read at numpy import.
_BLAS_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Probe workloads. Each entry is (subsystem_count, truncated_dim, kind); the dressed
# dimension is truncated_dim ** subsystem_count. "tiny" isolates dispatch overhead
# (near-zero compute); the others sample per-point cost in the dense and sparse
# regimes.
_PROFILES: dict[str, dict[str, Any]] = {
    "tiny": {"n_subsys": 2, "trunc": 3, "kind": "transmon"},  # dim 9
    "dense_small": {"n_subsys": 3, "trunc": 6, "kind": "transmon"},  # dim 216
    "dense_large": {"n_subsys": 4, "trunc": 6, "kind": "transmon"},  # dim 1296
    "sparse": {"n_subsys": 4, "trunc": 6, "kind": "fluxonium"},  # dim 1296, few evals
}


@dataclass
class MachineCalibration:
    """Measured machine model used to refine the parallelization heuristic.

    Attributes
    ----------
    cores:
        logical core count of the calibrated machine.
    overhead_s:
        measured per-task dispatch overhead in seconds (pickling plus inter-process
        hand-off for one grid point).
    pool_startup_s:
        measured one-time cost in seconds of starting the worker pool for a sweep
        (dominated by the ``spawn`` re-import on macOS/Windows); must be amortized
        before parallelism pays off. Measured at the full core count; for a
        worker-count-aware estimate use the base/per-worker fit below.
    pool_startup_base_s:
        fixed part of the pool-startup cost (seconds), independent of worker count.
    pool_startup_per_worker_s:
        marginal pool-startup cost (seconds) added per worker. Under ``spawn`` each
        worker re-imports, so startup grows with the worker count; the heuristic
        estimates it as ``pool_startup_base_s + pool_startup_per_worker_s * n``.
    cost_samples:
        list of ``{"dimension", "is_sparse", "seconds_per_point"}`` entries sampling
        the per-point diagonalization cost.
    platform:
        platform string of the calibrated machine.
    blas_backend:
        short description of the BLAS backend seen during calibration.
    scqubits_version:
        scqubits version at calibration time.
    timestamp:
        ISO-8601 calibration time.
    """

    cores: int
    overhead_s: float
    pool_startup_s: float = 0.0
    pool_startup_base_s: float = 0.0
    pool_startup_per_worker_s: float = 0.0
    cost_samples: list[dict[str, Any]] = field(default_factory=list)
    platform: str = ""
    blas_backend: str = ""
    scqubits_version: str = ""
    timestamp: str = ""

    def estimated_cost_per_point(
        self, dimension: int, is_sparse: bool
    ) -> Optional[float]:
        """Return an estimated per-point cost (seconds) for a workload, or ``None``.

        Uses the calibration sample of matching sparsity whose dimension is closest
        to ``dimension`` (no extrapolation model -- a coarse but measured estimate).

        Parameters
        ----------
        dimension:
            Hilbert-space dimension of the target workload.
        is_sparse:
            whether the target workload diagonalizes sparsely.
        """
        matching = [s for s in self.cost_samples if bool(s["is_sparse"]) == is_sparse]
        if not matching:
            return None
        dims = [int(s["dimension"]) for s in matching]
        # Trust the measured cost only within a generous factor of the sampled
        # dimension range; outside it (e.g. a much cheaper or much larger system),
        # return None so the caller falls back to the default tiered heuristic rather
        # than over/under-estimating from an unrelated sample.
        if dimension < min(dims) / 4 or dimension > max(dims) * 4:
            return None
        nearest = min(matching, key=lambda s: abs(int(s["dimension"]) - dimension))
        return float(nearest["seconds_per_point"])


def default_calibration_path() -> str:
    """Return the path the machine calibration is read from and written to.

    Uses ``settings.PARALLEL_CALIBRATION_PATH`` when set, else
    ``~/.scqubits/parallel_calibration.json``.
    """
    configured = getattr(settings, "PARALLEL_CALIBRATION_PATH", None)
    if configured:
        return os.path.expanduser(str(configured))
    return os.path.join(
        os.path.expanduser("~"), ".scqubits", "parallel_calibration.json"
    )


def load_calibration(path: Optional[str] = None) -> Optional[MachineCalibration]:
    """Return the persisted :class:`MachineCalibration`, or ``None`` if absent/unreadable.

    Parameters
    ----------
    path:
        file to read; defaults to :func:`default_calibration_path`.
    """
    path = path or default_calibration_path()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return MachineCalibration(**data)
    except (OSError, ValueError, TypeError):
        return None


def _save_calibration(calibration: MachineCalibration, path: str) -> None:
    """Write the calibration to ``path`` as JSON, creating parent directories.

    Parameters
    ----------
    calibration:
        the model to persist.
    path:
        destination file.
    """
    directory = os.path.dirname(path)
    if directory:  # empty for a bare filename; os.makedirs("") would raise
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(calibration), handle, indent=2)


# --------------------------------------------------------------------------------------
# Probe construction and one in-process measurement (run inside the -m subprocess)
# --------------------------------------------------------------------------------------
def _build_and_run(profile: str, num_cpus: int, n_points: int) -> None:
    """Build the probe sweep, run it, and print its wall time as ``__RESULT__``.

    Executed inside a measurement subprocess (BLAS env already preset by the
    launcher). Not intended to be called directly.

    Parameters
    ----------
    profile:
        key into ``_PROFILES`` selecting the probe system.
    num_cpus:
        worker count for the sweep.
    n_points:
        number of grid points to sweep.
    """
    import numpy as np

    import scqubits as scq

    spec = _PROFILES[profile]
    settings.PROGRESSBAR_DISABLED = True
    subsystems: list[Any]
    if spec["kind"] == "transmon":
        subsystems = [
            scq.TunableTransmon(
                EJmax=30.0,
                EC=0.2,
                d=0.1,
                flux=0.0,
                ng=0.0,
                ncut=50,
                truncated_dim=spec["trunc"],
                id_str=f"s{i}",
            )
            for i in range(spec["n_subsys"])
        ]
    else:
        subsystems = [
            scq.Fluxonium(
                EJ=4.0,
                EC=1.0,
                EL=1.0,
                flux=0.5,
                cutoff=110,
                truncated_dim=spec["trunc"],
                id_str=f"s{i}",
            )
            for i in range(spec["n_subsys"])
        ]
    hs = scq.HilbertSpace(subsystems)
    for i in range(spec["n_subsys"] - 1):
        hs.add_interaction(
            g_strength=0.1,
            op1=subsystems[i].n_operator,
            op2=subsystems[i + 1].n_operator,
        )

    def update(flux):
        subsystems[0].flux = flux

    def run_once() -> None:
        scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"flux": np.linspace(0.0, 0.33, n_points)},
            update_hilbertspace=update,
            evals_count=6,
            num_cpus=num_cpus,
            autorun=True,
        )

    t_cold = time.perf_counter()
    run_once()  # cold: pays the one-time pool startup (spawn re-import)
    cold = time.perf_counter() - t_cold
    t_warm = time.perf_counter()
    run_once()  # warm: the cached pool is reused
    warm = time.perf_counter() - t_warm
    print(
        "__RESULT__"
        + json.dumps({"warm_s": warm, "cold_s": cold, "n_points": n_points})
    )


def _measure(profile: str, num_cpus: int, n_points: int, blas: int) -> dict[str, float]:
    """Return the warm/cold wall times of one probe sweep, measured in a subprocess.

    Parameters
    ----------
    profile:
        key into ``_PROFILES``.
    num_cpus:
        worker count for the sweep.
    n_points:
        number of grid points.
    blas:
        BLAS-thread cap, preset in the subprocess environment before numpy import.
    """
    env = dict(os.environ)
    for var in _BLAS_ENV_VARS:
        env[var] = str(blas)
    cmd = [
        sys.executable,
        "-m",
        "scqubits.utils.parallel_calibration",
        "--measure",
        profile,
        str(num_cpus),
        str(n_points),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__") :])
    raise RuntimeError(
        "calibration measurement failed (profile={}, num_cpus={}, n_points={}, "
        "exit {}).\nstderr:\n{}".format(
            profile, num_cpus, n_points, proc.returncode, proc.stderr
        )
    )


def _blas_backend() -> str:
    """Return a short description of numpy's BLAS backend, or ``'unknown'``."""
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
        if pools:
            return ", ".join(str(p.get("internal_api")) for p in pools)
    except Exception:
        pass
    return "unknown"


def calibrate_parallelization(
    *,
    persist: bool = True,
    path: Optional[str] = None,
    explain: bool = True,
    timestamp: Optional[str] = None,
) -> MachineCalibration:
    """Measure the machine once and return (and persist) a :class:`MachineCalibration`.

    Runs a short battery of real sweeps in isolated subprocesses to measure the
    per-task dispatch overhead and the per-point diagonalization cost in the dense
    and sparse regimes, then writes the model so future
    :func:`recommend_parallelization` calls use a measured break-even.

    Safe to call from Jupyter or a plain script -- measurements run as
    ``python -m`` subprocesses, so no ``if __name__ == "__main__":`` guard is needed
    for the calibration itself.

    Parameters
    ----------
    persist:
        if ``True``, write the result to ``path``.
    path:
        destination file; defaults to :func:`default_calibration_path`.
    explain:
        if ``True``, print progress and the resulting model.
    timestamp:
        ISO-8601 string recorded in the model; defaults to an empty string (callers
        may stamp it, since scqubits avoids wall-clock calls in some contexts).

    Returns
    -------
    The measured :class:`MachineCalibration`.
    """
    cores = os.cpu_count() or 1
    # Use all cores as the parallel worker count; on a single-core machine this is 1
    # (a serial measurement) rather than oversubscribing with 2 workers on 1 core.
    workers = cores

    if explain:
        print("scqubits: calibrating parallelization on this machine...")

    # Per-task overhead and one-time pool startup: a near-zero-cost system swept
    # across many points. With compute negligible, the warm parallel per-point time
    # is dispatch overhead, and the cold-minus-warm gap is the pool-startup cost.
    overhead_points = 200
    tiny_par = _measure("tiny", workers, overhead_points, blas=1)
    tiny_ser = _measure("tiny", 1, overhead_points, blas=1)
    overhead_s = max(
        0.0, (tiny_par["warm_s"] - tiny_ser["warm_s"] / workers) / overhead_points
    )
    pool_startup_s = max(0.0, tiny_par["cold_s"] - tiny_par["warm_s"])
    # Pool startup grows with the worker count under spawn (each worker re-imports),
    # so a flat figure overcharges small pools. Measure a second, small-worker point
    # and fit startup(n) = base + per_worker * n so the heuristic can price the cost
    # of whatever worker count it is actually considering.
    low_workers = min(2, workers)
    if low_workers < workers:
        tiny_lo = _measure("tiny", low_workers, overhead_points, blas=1)
        startup_lo = max(0.0, tiny_lo["cold_s"] - tiny_lo["warm_s"])
        per_worker = max(0.0, (pool_startup_s - startup_lo) / (workers - low_workers))
        startup_base = max(0.0, startup_lo - per_worker * low_workers)
    else:  # single-core machine: no parallelism to model
        per_worker = 0.0
        startup_base = pool_startup_s

    cost_samples: list[dict[str, Any]] = []
    cost_points = 24
    for profile, is_sparse in (
        ("dense_small", False),
        ("dense_large", False),
        ("sparse", True),
    ):
        spec = _PROFILES[profile]
        dimension = spec["trunc"] ** spec["n_subsys"]
        t_serial = _measure(profile, 1, cost_points, blas=max(1, cores))
        seconds_per_point = t_serial["warm_s"] / cost_points
        cost_samples.append(
            {
                "dimension": dimension,
                "is_sparse": is_sparse,
                "seconds_per_point": seconds_per_point,
            }
        )
        if explain:
            print(
                "  {:>12} dim {:>4} {:>6}: {:.4f} s/point".format(
                    profile,
                    dimension,
                    "sparse" if is_sparse else "dense",
                    seconds_per_point,
                )
            )

    calibration = MachineCalibration(
        cores=cores,
        overhead_s=overhead_s,
        pool_startup_s=pool_startup_s,
        pool_startup_base_s=startup_base,
        pool_startup_per_worker_s=per_worker,
        cost_samples=cost_samples,
        platform=platform.platform(),
        blas_backend=_blas_backend(),
        scqubits_version=getattr(__import__("scqubits"), "__version__", ""),
        timestamp=timestamp or "",
    )

    if explain:
        print(
            "  per-task overhead: {:.4f} s  pool startup: {:.3f}s + {:.3f}s/worker  "
            "(cores={}, BLAS={})".format(
                overhead_s, startup_base, per_worker, cores, calibration.blas_backend
            )
        )

    if persist:
        target = path or default_calibration_path()
        _save_calibration(calibration, target)
        if explain:
            print("  wrote {}".format(target))

    return calibration


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point: ``--measure`` for one subprocess run, else calibrate.

    Parameters
    ----------
    argv:
        argument vector; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate scqubits parallelization for this machine."
    )
    parser.add_argument(
        "--measure",
        nargs=3,
        metavar=("PROFILE", "NUM_CPUS", "N_POINTS"),
        default=None,
        help=argparse.SUPPRESS,  # internal: time one probe sweep in this process
    )
    parser.add_argument(
        "--no-persist", action="store_true", help="do not write the calibration file"
    )
    args = parser.parse_args(argv)

    if args.measure is not None:
        profile, num_cpus, n_points = args.measure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _build_and_run(profile, int(num_cpus), int(n_points))
        return 0

    calibrate_parallelization(persist=not args.no_persist, explain=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
