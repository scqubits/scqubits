#!/usr/bin/env python
# benchmark_multiprocessing.py
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
"""Baseline benchmark harness for scqubits multiprocessing.

Standalone developer tool (not collected by pytest). It measures the wall time
of the parallelized code paths across worker counts and instruments two known
overhead sources so future optimizations can be compared against a baseline:

* pool lifecycle: how many times ``cpu_switch.get_map_method`` is invoked per
  run (``map_calls``), how many real pools are actually constructed
  (``pools_created`` -- these differ once pool reuse lands), and the wall time
  spent constructing pools;
* per-task serialization payload: the dill-pickled size of the ``HilbertSpace``
  and of the user ``update_hilbertspace`` closure shipped to workers.

The harness does NOT modify library code; instrumentation is done by monkey-
patching within this process only.

Scenarios
---------
sweep   ``ParameterSweep.run()`` over a 2-D flux x ng grid (bare + dressed).
hspace  ``HilbertSpace.get_spectrum_vs_paramvals`` (composite 1-D scan).
qubit   ``TunableTransmon.get_spectrum_vs_paramvals`` (single-qubit 1-D scan).

Profiles
--------
light   matches scqubits/tests/test_parametersweep.py (tiny; --verify uses it).
heavy   enlarged truncated dims / ncut so multiprocessing can actually pay off.

Examples
--------
    python tools/benchmark_multiprocessing.py --scenario all --num-cpus 1,2,4
    python tools/benchmark_multiprocessing.py --profile heavy --grid 32 --json
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

import numpy as np

# Ensure the in-repo scqubits source is importable when this script is run
# directly: `python tools/benchmark_multiprocessing.py` puts tools/ on
# sys.path[0], not the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import scqubits as scq
import scqubits.core.qubit_base as qubit_base
import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch


# Reference eigenvalues for sweep["evals"][1, 1] with the "light" system below
# and evals_count=20. Copied from scqubits/tests/test_parametersweep.py so the
# benchmarked computation can be sanity-checked for correctness.
_REFERENCE_EVALS_11 = np.array(
    [
        -38.24067872, -34.80636811, -33.70287551, -31.50027076, -31.23467726,
        -30.28547786, -29.16544659, -27.80039104, -27.08317259, -26.69778263,
        -25.76389721, -24.59864846, -24.49435409, -24.43639473, -23.2803766,
        -22.63621727, -22.16084701, -21.00224623, -21.00053542, -20.07811164,
    ]
)


# --------------------------------------------------------------------------------------
# System under test (mirrors tests/test_parametersweep.py initialize())
# --------------------------------------------------------------------------------------
# Workload profiles. "light" matches tests/test_parametersweep.py exactly (used
# by the --verify correctness guard). "heavy" enlarges the truncated dimensions
# (and ncut) so the dressed diagonalization is costly enough for multiprocessing
# to pay off, locating the regime where parallelism actually helps.
PROFILES = {
    "light": {"ncut1": 40, "ncut2": 30, "dim1": 3, "dim2": 3, "res_dim": 4},
    "heavy": {"ncut1": 110, "ncut2": 110, "dim1": 8, "dim2": 8, "res_dim": 10},
}
PROFILE = "light"


def _build_components() -> tuple[Any, Any, Any]:
    """Return the two tunable transmons and the oscillator for the active profile."""
    p = PROFILES[PROFILE]
    tmon1 = scq.TunableTransmon(
        EJmax=40.0, EC=0.2, d=0.1, flux=0.0, ng=0.3,
        ncut=p["ncut1"], truncated_dim=p["dim1"],
    )
    tmon2 = scq.TunableTransmon(
        EJmax=15.0, EC=0.15, d=0.2, flux=0.0, ng=0.0,
        ncut=p["ncut2"], truncated_dim=p["dim2"],
    )
    resonator = scq.Oscillator(E_osc=4.5, truncated_dim=p["res_dim"])
    return tmon1, tmon2, resonator


def _build_hilbertspace(tmon1: Any, tmon2: Any, resonator: Any) -> Any:
    """Assemble the coupled HilbertSpace used throughout the benchmark."""
    hilbertspace = scq.HilbertSpace([tmon1, tmon2, resonator])
    hilbertspace.add_interaction(
        g_strength=0.1, op1=tmon1.n_operator, op2=resonator.creation_operator, add_hc=True
    )
    hilbertspace.add_interaction(
        g_strength=0.2, op1=tmon2.n_operator, op2=resonator.creation_operator, add_hc=True
    )
    return hilbertspace


def _build_sweep(flux_count: int, evals_count: int, num_cpus: int) -> Any:
    """Build a (non-autorun) 2-D ParameterSweep over flux x ng."""
    tmon1, tmon2, resonator = _build_components()
    hilbertspace = _build_hilbertspace(tmon1, tmon2, resonator)
    flux_vals = np.linspace(0.0, 2.0, flux_count)
    ng_vals = np.linspace(-0.5, 0.5, 3)
    area_ratio = 1.2

    def update_hilbertspace(flux: float, ng: float) -> None:
        tmon1.flux = flux
        tmon2.flux = area_ratio * flux
        tmon2.ng = ng

    return scq.ParameterSweep(
        hilbertspace=hilbertspace,
        paramvals_by_name={"flux": flux_vals, "ng": ng_vals},
        update_hilbertspace=update_hilbertspace,
        evals_count=evals_count,
        subsys_update_info={"flux": [tmon1, tmon2], "ng": [tmon2]},
        num_cpus=num_cpus,
        autorun=False,
    )


def _build_hspace_scan(flux_count: int) -> tuple[Any, np.ndarray, Callable]:
    """Build the composite-system 1-D scan inputs (HilbertSpace path)."""
    tmon1, tmon2, resonator = _build_components()
    hilbertspace = _build_hilbertspace(tmon1, tmon2, resonator)
    flux_vals = np.linspace(0.0, 2.0, flux_count)

    def update_hilbertspace(flux: float) -> None:
        tmon1.flux = flux
        tmon2.flux = 1.2 * flux

    return hilbertspace, flux_vals, update_hilbertspace


# --------------------------------------------------------------------------------------
# Instrumentation (process-local monkeypatch of get_map_method + pool factory)
# --------------------------------------------------------------------------------------
class _MapInstrument:
    """Counts map-method calls, real pool constructions, and pool-build time."""

    def __init__(self) -> None:
        self._orig = cpu_switch.get_map_method
        self._pathos_pools: Any = None
        self._orig_pool_cls: Any = None
        self.calls = 0
        self.pool_spawn_s = 0.0
        self.pools_created = 0

    def _wrapped(self, num_cpus: int) -> Callable:
        self.calls += 1
        start = time.perf_counter()
        method = self._orig(num_cpus)
        self.pool_spawn_s += time.perf_counter() - start
        return method

    def __enter__(self) -> "_MapInstrument":
        cpu_switch.get_map_method = self._wrapped
        qubit_base.get_map_method = self._wrapped  # qubit_base imported it by name
        try:
            import pathos.pools as pathos_pools

            self._pathos_pools = pathos_pools
            self._orig_pool_cls = pathos_pools.ProcessPool
            instrument = self

            def _counting_pool(*args: Any, **kwargs: Any) -> Any:
                instrument.pools_created += 1
                return instrument._orig_pool_cls(*args, **kwargs)

            pathos_pools.ProcessPool = _counting_pool
        except Exception:
            self._pathos_pools = None
        return self

    def __exit__(self, *exc: Any) -> None:
        cpu_switch.get_map_method = self._orig
        qubit_base.get_map_method = self._orig
        if self._pathos_pools is not None and self._orig_pool_cls is not None:
            self._pathos_pools.ProcessPool = self._orig_pool_cls

    def reset(self) -> None:
        self.calls = 0
        self.pool_spawn_s = 0.0
        self.pools_created = 0


def _cleanup_pool() -> None:
    """Terminate, join, and forget any global pool so the next config spawns fresh.

    pathos caches pools; ``clear()`` removes the cached server and reaps its
    workers, which is what stops leftover processes from contaminating later
    timings. For true isolation use ``--isolate`` (a fresh subprocess per config).
    """
    pool = settings.POOL
    if pool is not None:
        for method_name in ("terminate", "join", "clear", "close"):
            try:
                getattr(pool, method_name)()
            except Exception:
                pass
    settings.POOL = None


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
def _time_callable(
    make_and_run: Callable[[], Any], repeats: int, instrument: _MapInstrument
) -> dict[str, float]:
    """Warm up once, then time `repeats` runs; return min/median + instrument data."""
    _cleanup_pool()
    make_and_run()  # warm-up (pays first-spawn / import cost)

    durations: list[float] = []
    last_calls = 0
    last_spawn = 0.0
    last_pools = 0
    for _ in range(repeats):
        _cleanup_pool()
        instrument.reset()
        start = time.perf_counter()
        make_and_run()
        durations.append(time.perf_counter() - start)
        last_calls = instrument.calls
        last_spawn = instrument.pool_spawn_s
        last_pools = instrument.pools_created
    _cleanup_pool()

    return {
        "wall_min_s": min(durations),
        "wall_median_s": statistics.median(durations),
        "map_calls": last_calls,
        "pools_created": last_pools,
        "pool_spawn_s": last_spawn,
    }


def _run_isolated(scenario: str, num_cpus: int, args: argparse.Namespace) -> dict[str, Any]:
    """Measure one config in a fresh subprocess (no leftover-pool contamination)."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single", scenario, str(num_cpus),
        "--profile", PROFILE,
        "--grid", str(args.grid),
        "--repeats", str(args.repeats),
        "--evals-count", str(args.evals_count),
    ]
    if args.blas_threads is not None:
        cmd += ["--blas-threads", str(args.blas_threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError(
        f"isolated run failed for {scenario}/{num_cpus} (exit {proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _runner_for(scenario: str, args: argparse.Namespace, num_cpus: int) -> Callable[[], Any]:
    """Return a zero-arg callable that builds + runs one scenario at `num_cpus`."""
    if scenario == "sweep":
        def run() -> Any:
            sweep = _build_sweep(args.grid, args.evals_count, num_cpus)
            sweep.run()
            return sweep
        return run
    if scenario == "hspace":
        def run() -> Any:
            hspace, flux_vals, update = _build_hspace_scan(args.grid)
            return hspace.get_spectrum_vs_paramvals(
                flux_vals, update, evals_count=args.evals_count, num_cpus=num_cpus
            )
        return run
    if scenario == "qubit":
        def run() -> Any:
            tmon, _, _ = _build_components()
            flux_vals = np.linspace(0.0, 2.0, max(args.grid, 50))
            return tmon.get_spectrum_vs_paramvals(
                "flux", flux_vals, evals_count=args.evals_count, num_cpus=num_cpus
            )
        return run
    raise ValueError(f"unknown scenario: {scenario}")


# --------------------------------------------------------------------------------------
# Serialization payload measurement
# --------------------------------------------------------------------------------------
def _measure_serialization(args: argparse.Namespace) -> dict[str, Any]:
    """Quantify the dill payload of the HilbertSpace and the update closure."""
    try:
        import dill
    except ImportError:
        return {"available": False}

    tmon1, tmon2, resonator = _build_components()
    hilbertspace = _build_hilbertspace(tmon1, tmon2, resonator)

    def update_hilbertspace(flux: float, ng: float) -> None:
        tmon1.flux = flux
        tmon2.flux = 1.2 * flux
        tmon2.ng = ng

    dill.settings["recurse"] = True
    start = time.perf_counter()
    hs_blob = dill.dumps(hilbertspace)
    hs_dumps_s = time.perf_counter() - start
    update_blob = dill.dumps(update_hilbertspace)

    return {
        "available": True,
        "hilbertspace_bytes": len(hs_blob),
        "hilbertspace_dumps_s": hs_dumps_s,
        "update_func_bytes": len(update_blob),
    }


# --------------------------------------------------------------------------------------
# Correctness guard
# --------------------------------------------------------------------------------------
def _verify(num_cpus: int) -> bool:
    """Run the canonical light 11x3 sweep and compare evals[1, 1] to the reference."""
    global PROFILE
    saved_profile = PROFILE
    PROFILE = "light"  # reference evals are only valid for the light system
    try:
        sweep = _build_sweep(flux_count=11, evals_count=20, num_cpus=num_cpus)
        sweep.run()
        calculated = np.asarray(sweep["evals"][1, 1])
    finally:
        PROFILE = saved_profile
    ok = bool(np.allclose(_REFERENCE_EVALS_11, calculated))
    print(f"[verify] sweep evals[1,1] vs reference (num_cpus={num_cpus}): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------------------
# Orchestration / reporting
# --------------------------------------------------------------------------------------
def _parse_num_cpus(raw: str, available: bool) -> list[int]:
    requested = sorted({int(token) for token in raw.split(",") if token.strip()})
    cap = os.cpu_count() or 1
    capped = [n for n in requested if 1 <= n <= cap]
    if not capped:
        capped = [1]
    if not available:
        # pathos/dill missing: only the serial path is runnable
        capped = [1]
        print("[warn] pathos/dill unavailable; restricting to num_cpus=1.")
    return capped


def _print_table(scenario: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n=== scenario: {scenario} ===")
    header = (
        f"{'cpus':>4}  {'min(s)':>9}  {'median(s)':>10}  {'speedup':>8}  "
        f"{'eff':>6}  {'map_calls':>9}  {'pools':>6}  {'pool_spawn(s)':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['num_cpus']:>4}  {row['wall_min_s']:>9.3f}  "
            f"{row['wall_median_s']:>10.3f}  {row['speedup_vs_1']:>8.2f}  "
            f"{row['efficiency']:>6.2f}  {row['map_calls']:>9}  "
            f"{row['pools_created']:>6}  {row['pool_spawn_s']:>13.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    global PROFILE

    parser = argparse.ArgumentParser(description="scqubits multiprocessing benchmark.")
    parser.add_argument("--scenario", default="all",
                        choices=["sweep", "hspace", "qubit", "all"])
    parser.add_argument("--profile", default="light", choices=["light", "heavy"],
                        help="workload size (heavy => MP can pay off)")
    parser.add_argument("--num-cpus", default="1,2,4",
                        help="comma-separated worker counts, e.g. '1,2,4'")
    parser.add_argument("--grid", type=int, default=11,
                        help="number of flux points (ng axis fixed at 3 for sweep)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--evals-count", type=int, default=20)
    parser.add_argument("--verify", action="store_true",
                        help="run the light 11x3 correctness guard against reference evals")
    parser.add_argument("--json", nargs="?", const="<auto>", default=None,
                        help="write results JSON (optional path; default under "
                             "tools/benchmark_results/)")
    parser.add_argument("--isolate", action="store_true",
                        help="run each (scenario, num_cpus) in a fresh subprocess "
                             "for contamination-free wall times")
    parser.add_argument("--blas-threads", type=int, default=None,
                        help="cap BLAS/OpenMP threads per worker via "
                             "settings.MULTIPROC_BLAS_THREADS during num_cpus>1 runs")
    parser.add_argument("--single", nargs=2, default=None,
                        metavar=("SCENARIO", "NUMCPUS"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    PROFILE = args.profile
    if args.blas_threads is not None:
        settings.MULTIPROC_BLAS_THREADS = args.blas_threads

    try:
        import dill  # noqa: F401
        import pathos  # noqa: F401
        backend_available = True
    except ImportError:
        backend_available = False

    # Quiet the per-call tqdm progress bars if the installed version exposes a switch.
    for flag in ("PROGRESSBAR_DISABLED", "DISABLE_PROGRESSBAR"):
        if hasattr(settings, flag):
            setattr(settings, flag, True)

    # Internal single-config worker used by --isolate: measure one config in this
    # fresh process and emit a single machine-readable result line.
    if args.single is not None:
        single_scenario, single_cpus = args.single[0], int(args.single[1])
        with _MapInstrument() as instrument:
            timing = _time_callable(
                _runner_for(single_scenario, args, single_cpus), args.repeats, instrument
            )
        print("__RESULT__" + json.dumps(timing))
        return 0

    num_cpus_list = _parse_num_cpus(args.num_cpus, backend_available)
    scenarios = ["sweep", "hspace", "qubit"] if args.scenario == "all" else [args.scenario]

    print(f"platform={platform.platform()}  cpu_count={os.cpu_count()}  "
          f"MULTIPROC={settings.MULTIPROC}  scqubits={scq.__version__}")
    print(f"profile={PROFILE}  num_cpus={num_cpus_list}  grid={args.grid}  "
          f"repeats={args.repeats}  evals_count={args.evals_count}  "
          f"blas_threads_per_worker={settings.MULTIPROC_BLAS_THREADS}")

    if args.verify:
        _verify(num_cpus=1)

    all_results: list[dict[str, Any]] = []
    with _MapInstrument() as instrument:
        for scenario in scenarios:
            rows: list[dict[str, Any]] = []
            baseline_median: float | None = None
            for num_cpus in num_cpus_list:
                if args.isolate:
                    timing = _run_isolated(scenario, num_cpus, args)
                else:
                    timing = _time_callable(
                        _runner_for(scenario, args, num_cpus), args.repeats, instrument
                    )
                if baseline_median is None:
                    baseline_median = timing["wall_median_s"]
                speedup = baseline_median / timing["wall_median_s"]
                row = {
                    "scenario": scenario,
                    "profile": PROFILE,
                    "num_cpus": num_cpus,
                    "grid_points": args.grid * 3 if scenario == "sweep" else max(args.grid, 50),
                    "evals_count": args.evals_count,
                    "repeats": args.repeats,
                    "speedup_vs_1": speedup,
                    "efficiency": speedup / num_cpus,
                    **timing,
                }
                rows.append(row)
                all_results.append(row)
            _print_table(scenario, rows)

    serialization = _measure_serialization(args)
    print("\n=== serialization payload (dill) ===")
    if serialization.get("available"):
        print(f"HilbertSpace: {serialization['hilbertspace_bytes']:,} bytes  "
              f"(dumps {serialization['hilbertspace_dumps_s'] * 1e3:.1f} ms)")
        print(f"update closure: {serialization['update_func_bytes']:,} bytes")
    else:
        print("dill unavailable; skipped.")

    if args.json is not None:
        report = {
            "meta": {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "multiproc": settings.MULTIPROC,
                "profile": PROFILE,
                "scqubits_version": scq.__version__,
            },
            "results": all_results,
            "serialization": serialization,
        }
        results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)
        if args.json == "<auto>":
            stamp = report["meta"]["timestamp"].replace(":", "").replace("-", "")
            path = os.path.join(results_dir, f"{PROFILE}_{stamp}.json")
        else:
            path = args.json
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote results to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
