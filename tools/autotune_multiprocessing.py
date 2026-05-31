#!/usr/bin/env python
# autotune_multiprocessing.py
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
"""Adaptive auto-tuner for scqubits parameter-sweep parallelism.

Finds the fastest ``(num_cpus, BLAS-threads-per-worker)`` configuration for a
sweep *on the machine it runs on*, because the optimum depends on core count,
BLAS library, OS, and matrix size -- a single quoted number is not portable.

Why both knobs: scqubits' diagonalizations call a multithreaded BLAS. Running
``num_cpus`` worker processes that each spawn a full BLAS thread pool
oversubscribes the cores, while a single process on tiny matrices wastes time in
BLAS thread overhead. The fastest config is a point on the
``num_cpus x BLAS-threads ~ cores`` frontier, and it must be measured.

Design notes addressing common benchmarking pitfalls:
* BLAS threads are set in each config's *environment before the subprocess
  starts*, so numpy reads them at import. This controls the ``num_cpus=1`` path
  (the parent does the math) as well as worker pools, on both spawn and fork.
* Each config runs in a fresh subprocess -> no leftover-pool contamination.
* CV-based early stopping: stop repeating once the run-to-run coefficient of
  variation is small; report median + IQR, and warn when a config stays noisy.
* Full environment (BLAS backend, thread caps, start method) is recorded.

Examples
--------
    python tools/autotune_multiprocessing.py --profile heavy --grid 24
    python tools/autotune_multiprocessing.py --num-cpus 1,2,4,8 --blas 1,auto --json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

from typing import Any, Callable

# Make the in-repo scqubits source and the sibling benchmark harness importable
# when run directly (sys.path[0] is tools/, not the repo root).
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_REPO_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_multiprocessing as bench  # system builders + pool cleanup


_BLAS_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


# --------------------------------------------------------------------------------------
# Statistics with CV-based early stopping
# --------------------------------------------------------------------------------------
def _stats(times: list[float]) -> dict[str, float]:
    """Summary statistics for a list of wall times."""
    n = len(times)
    mean = statistics.fmean(times)
    stdev = statistics.stdev(times) if n > 1 else 0.0
    if n >= 4:
        quartiles = statistics.quantiles(times, n=4)
        q25, q75 = quartiles[0], quartiles[-1]
    else:
        q25, q75 = min(times), max(times)
    return {
        "n_used": n,
        "min_s": min(times),
        "median_s": statistics.median(times),
        "q25_s": q25,
        "q75_s": q75,
        "mean_s": mean,
        "stdev_s": stdev,
        "cv": (stdev / mean) if mean else 0.0,
    }


def _timed(make_and_run: Callable[[], Any], min_repeats: int, max_repeats: int,
           cv_target: float) -> dict[str, float]:
    """Warm up once, then repeat until CV < cv_target (>= min_repeats) or max_repeats."""
    bench._cleanup_pool()
    make_and_run()  # warm-up: pays first-spawn / import cost, discarded

    times: list[float] = []
    for _ in range(max_repeats):
        bench._cleanup_pool()
        start = time.perf_counter()
        make_and_run()
        times.append(time.perf_counter() - start)
        if len(times) >= min_repeats:
            mean = statistics.fmean(times)
            cv = (statistics.stdev(times) / mean) if mean else 0.0
            if cv < cv_target:
                break
    bench._cleanup_pool()
    return _stats(times)


# --------------------------------------------------------------------------------------
# Config search space
# --------------------------------------------------------------------------------------
def _parse_int_list(raw: str, cores: int) -> list[int]:
    return sorted({min(int(tok), cores) for tok in raw.split(",") if tok.strip()})


def _build_configs(num_cpus_list: list[int], blas_spec: str, cores: int) -> list[tuple[int, int]]:
    """Return (num_cpus, blas_threads) configs, pruned to product <= cores.

    ``blas_spec`` is a comma list of ints and/or the token ``auto`` (=
    cores // num_cpus). The default frontier keeps total threads near the core
    count: heavy oversubscription is known-bad and is skipped.
    """
    configs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for num_cpus in num_cpus_list:
        blas_values: set[int] = set()
        for tok in blas_spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            blas_values.add(max(1, cores // num_cpus) if tok == "auto" else int(tok))
        for blas in sorted(blas_values):
            if num_cpus * blas > cores:  # skip oversubscription
                continue
            key = (num_cpus, blas)
            if key not in seen:
                seen.add(key)
                configs.append(key)
    return configs


# --------------------------------------------------------------------------------------
# One isolated measurement (run in a subprocess with BLAS env preset)
# --------------------------------------------------------------------------------------
def _measure_in_process(args: argparse.Namespace, num_cpus: int) -> dict[str, float]:
    """Time the sweep for one config in THIS process (BLAS env already applied)."""
    import scqubits.settings as settings

    bench.PROFILE = args.profile
    for flag in ("PROGRESSBAR_DISABLED", "DISABLE_PROGRESSBAR"):
        if hasattr(settings, flag):
            setattr(settings, flag, True)

    def make_and_run() -> None:
        sweep = bench._build_sweep(args.grid, args.evals_count, num_cpus)
        sweep.run()

    return _timed(make_and_run, args.min_repeats, args.max_repeats, args.cv_target)


def _run_config_isolated(args: argparse.Namespace, num_cpus: int, blas: int) -> dict[str, Any]:
    """Launch a fresh subprocess with BLAS threads preset; return its stats."""
    env = dict(os.environ)
    for var in _BLAS_ENV_VARS:
        env[var] = str(blas)
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--measure", str(num_cpus),
        "--profile", args.profile,
        "--grid", str(args.grid),
        "--evals-count", str(args.evals_count),
        "--min-repeats", str(args.min_repeats),
        "--max-repeats", str(args.max_repeats),
        "--cv-target", str(args.cv_target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError(
        f"measurement failed for num_cpus={num_cpus} blas={blas} (exit {proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# --------------------------------------------------------------------------------------
# Environment metadata
# --------------------------------------------------------------------------------------
def _blas_backend() -> str:
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        if pools:
            return ", ".join(
                f"{p.get('internal_api')}({p.get('num_threads')}t)" for p in pools
            )
    except Exception:
        pass
    return "unknown"


def _start_method() -> str:
    try:
        import multiprocessing
        return multiprocessing.get_start_method(allow_none=True) or "default"
    except Exception:
        return "unknown"


def _env_metadata(args: argparse.Namespace, cores: int) -> dict[str, Any]:
    import scqubits as scq

    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": cores,
        "blas_backend": _blas_backend(),
        "start_method": _start_method(),
        "scqubits_version": scq.__version__,
        "profile": args.profile,
        "grid_flux_points": args.grid,
        "grid_total_points": args.grid * 3,
        "evals_count": args.evals_count,
        "cv_target": args.cv_target,
        "min_repeats": args.min_repeats,
        "max_repeats": args.max_repeats,
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    cores = os.cpu_count() or 1
    parser = argparse.ArgumentParser(
        description="Auto-tune num_cpus x BLAS-threads for scqubits sweeps."
    )
    parser.add_argument("--profile", default="heavy", choices=["light", "heavy"])
    parser.add_argument("--grid", type=int, default=24, help="flux points (ng axis = 3)")
    parser.add_argument("--evals-count", type=int, default=20)
    parser.add_argument("--num-cpus", default=None,
                        help="comma list (default: 1,2,4,8,... up to cpu_count)")
    parser.add_argument("--blas", default="1,auto",
                        help="comma list of BLAS threads/worker and/or 'auto' (=cores//num_cpus)")
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--max-repeats", type=int, default=8)
    parser.add_argument("--cv-target", type=float, default=0.05,
                        help="stop repeating a config once CV drops below this")
    parser.add_argument("--json", nargs="?", const="<auto>", default=None)
    parser.add_argument("--measure", type=int, default=None,
                        help=argparse.SUPPRESS)  # internal: one config in a preset env
    args = parser.parse_args(argv)

    # Internal worker: BLAS env was preset by the launcher; time one config.
    if args.measure is not None:
        stats = _measure_in_process(args, args.measure)
        print("__RESULT__" + json.dumps(stats))
        return 0

    if args.num_cpus is None:
        candidates = [1, 2, 4, 8, 16, cores]
        num_cpus_list = sorted({n for n in candidates if 1 <= n <= cores})
    else:
        num_cpus_list = _parse_int_list(args.num_cpus, cores)
    configs = _build_configs(num_cpus_list, args.blas, cores)

    meta = _env_metadata(args, cores)
    print(f"autotune: {meta['platform']}  cores={cores}  BLAS={meta['blas_backend']}  "
          f"start_method={meta['start_method']}  scqubits={meta['scqubits_version']}")
    print(f"profile={args.profile}  grid={args.grid} ({args.grid * 3} pts)  "
          f"evals_count={args.evals_count}  cv_target={args.cv_target}  "
          f"repeats={args.min_repeats}-{args.max_repeats}")
    print(f"searching {len(configs)} configs (num_cpus x BLAS, product <= {cores}):")

    results: list[dict[str, Any]] = []
    for num_cpus, blas in configs:
        stats = _run_config_isolated(args, num_cpus, blas)
        row = {"num_cpus": num_cpus, "blas_threads": blas, **stats}
        results.append(row)
        warn = "  [!] noisy" if stats["cv"] >= 0.10 else ""
        print(f"  num_cpus={num_cpus:>3}  blas={blas:>3}  "
              f"median={stats['median_s']:7.3f}s  min={stats['min_s']:7.3f}s  "
              f"cv={stats['cv']:.1%}  n={stats['n_used']}{warn}")

    # Default a user gets out of the box: num_cpus=1, BLAS uncapped (= cores).
    default = next(
        (r for r in results if r["num_cpus"] == 1 and r["blas_threads"] == cores), None
    )
    default_median = default["median_s"] if default else max(r["median_s"] for r in results)
    for r in results:
        r["speedup_vs_default"] = default_median / r["median_s"]

    best = min(results, key=lambda r: r["median_s"])
    print("\n=== recommendation (this machine + workload only) ===")
    print(f"  num_cpus={best['num_cpus']}, BLAS-threads/worker={best['blas_threads']}  "
          f"-> {best['median_s']:.3f}s")
    if default:
        print(f"  {best['speedup_vs_default']:.2f}x faster than the default "
              f"(num_cpus=1, BLAS={cores}): {default_median:.3f}s")
    print(f"  Apply with: scqubits.settings.NUM_CPUS = {best['num_cpus']}  and set "
          f"OPENBLAS_NUM_THREADS={best['blas_threads']} before importing scqubits.")
    if best["cv"] >= 0.10:
        print("  [!] best config was noisy (CV >= 10%); rerun with higher --max-repeats.")

    if args.json is not None:
        report = {"meta": meta, "configs": results,
                  "recommendation": {k: best[k] for k in
                                     ("num_cpus", "blas_threads", "median_s", "speedup_vs_default")}}
        results_dir = os.path.join(_TOOLS_DIR, "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)
        if args.json == "<auto>":
            stamp = meta["timestamp"].replace(":", "").replace("-", "")
            path = os.path.join(results_dir, f"autotune_{stamp}.json")
        else:
            path = args.json
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
