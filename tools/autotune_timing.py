#!/usr/bin/env python
"""Manual evaluation: is ``num_cpus="auto"`` close to the *best* config? (needs a quiet machine)

For each workload this times a frontier of ``num_cpus`` in {1, 2, 4, cores} -- all with
the default ``"auto"`` BLAS cap, so none oversubscribe -- and ``num_cpus="auto"``, each
in its own process group (isolated; the group is killed wholesale on timeout so a slow
pool can't orphan workers that poison later timings).  It reports:

  auto/serial  -- did auto help, or at least not hurt?   (want <= ~1.3x)
  auto/best    -- how close to the fastest frontier point? (want <= ~1.3x)

IMPORTANT: wall-time benchmarks are meaningless on a loaded machine.  Quiesce the box
(close other heavy work; check ``uptime``) before running.  If configs time out, the
machine is too busy -- use ``tools/benchmark_multiprocessing.py --isolate`` instead.

The naive footgun (num_cpus=cores, BLAS uncapped) is deliberately NOT run here: it
oversubscribes, is already known to be ~1-2 orders of magnitude slower, and a runaway
pool risks orphaning workers.  ``autotune_decisions.py`` already shows auto never picks
an uncapped high-worker config, which is the footgun-avoidance evidence.

Run:  python tools/autotune_timing.py [--quick]
"""

import json
import os
import signal
import subprocess
import sys
import time

QUICK = "--quick" in sys.argv


def workloads(quick):
    pts = lambda full, q: q if quick else full
    return {  # name: (n_subsys, truncated_dim, ncut, num_points)
        "tiny": (3, 4, 30, pts(8, 8)),
        "many_cheap": (3, 6, 40, pts(192, 48)),
        "few_large_dense": (3, 12, 50, pts(8, 4)),
        "many_large": (3, 11, 50, pts(64, 16)),
    }


def build(spec):
    import scqubits as scq

    n, trunc, ncut, _ = spec
    qubits = [
        scq.TunableTransmon(
            EJmax=30.0,
            EC=0.2,
            d=0.1,
            flux=0.1 * i,
            ng=0.0,
            ncut=ncut,
            truncated_dim=trunc,
            id_str=f"t{i}",
        )
        for i in range(n)
    ]
    hilbertspace = scq.HilbertSpace(qubits)
    for i in range(n - 1):
        hilbertspace.add_interaction(
            g_strength=0.1, op1=qubits[i].n_operator, op2=qubits[i + 1].n_operator
        )

    def update(flux):
        qubits[0].flux = flux

    return hilbertspace, update


def measure(spec, num_cpus):
    """Time a single sweep at the given num_cpus (run inside an isolated subprocess)."""
    import numpy as np
    import scqubits as scq

    nc = "auto" if num_cpus == "auto" else int(num_cpus)
    flux_vals = np.linspace(0.0, 0.5, spec[3])
    warm_hs, warm_upd = build(spec)
    scq.ParameterSweep(
        hilbertspace=warm_hs,
        paramvals_by_name={"flux": flux_vals[:2]},
        update_hilbertspace=warm_upd,
        evals_count=20,
        num_cpus=1,
    )  # warm up imports / BLAS
    hilbertspace, update = build(spec)
    start = time.perf_counter()
    scq.ParameterSweep(
        hilbertspace=hilbertspace,
        paramvals_by_name={"flux": flux_vals},
        update_hilbertspace=update,
        evals_count=20,
        num_cpus=nc,
    )
    return time.perf_counter() - start


def run_child(wl, num_cpus, timeout=300):
    """Run one measurement in an isolated process group; return its wall time or inf."""
    args = [sys.executable, __file__, "--measure", wl, num_cpus] + (
        ["--quick"] if QUICK else []
    )
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        for line in out.splitlines():
            if line.startswith("__R__"):
                return json.loads(line[5:])["dt"]
        return float("inf")
    except subprocess.TimeoutExpired:
        return float("inf")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # reap pool workers too
        except Exception:
            pass
        time.sleep(2)  # let the machine settle before the next measurement


def main():
    cores = os.cpu_count() or 1
    frontier_nc = sorted({1, 2, 4, cores})
    print(f"cores={cores}  quick={QUICK}  frontier num_cpus={frontier_nc}\n")
    header = "workload".ljust(16) + "".join(f"nc={n}".rjust(9) for n in frontier_nc)
    header += "     auto   best@   auto/serial  auto/best"
    print(header)
    print("-" * len(header))
    for wl in workloads(QUICK):
        front = {n: run_child(wl, str(n)) for n in frontier_nc}
        auto = run_child(wl, "auto")
        serial = front[1]
        finite = {n: v for n, v in front.items() if v != float("inf")}
        if not finite:
            print(
                wl.ljust(16)
                + "  all configs timed out -- machine too loaded; quiesce and retry"
            )
            continue
        best_nc, best = min(finite.items(), key=lambda kv: kv[1])
        row = wl.ljust(16) + "".join(f"{front[n]:8.1f}s" for n in frontier_nc)

        def ratio(value):
            if serial in (0, float("inf")):
                return "n/a"
            return f"{value / serial:.2f}x"

        auto_best = "n/a" if best in (0, float("inf")) else f"{auto / best:.2f}x"
        row += f" {auto:8.1f}s  nc={best_nc:<3} {ratio(auto):>10}  {auto_best:>8}"
        print(row)


if __name__ == "__main__":
    # The --measure dispatch MUST live under this guard. A measurement subprocess
    # creates a worker Pool; under the 'spawn' start method (macOS/Windows) each
    # worker re-imports this module *with the parent's sys.argv*. If the dispatch
    # ran at module level it would re-run measure() in every worker -> each spawns
    # its own Pool -> recursive pool creation (a fork bomb). Importing as
    # __mp_main__ skips this block, so workers only pick up the function defs.
    if len(sys.argv) >= 4 and sys.argv[1] == "--measure":
        dt = measure(workloads(QUICK)[sys.argv[2]], sys.argv[3])
        print("__R__" + json.dumps({"dt": dt}))
        sys.exit(0)
    main()
