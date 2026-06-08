#!/usr/bin/env python
"""Manual evaluation: do the autotuning *decisions* make sense?  (load-insensitive)

This prints what ``scqubits.recommend_parallelization`` chooses (num_cpus, BLAS-thread
cap, and its reason) for a spread of workloads, and checks that ``num_cpus="auto"``
gives the same spectrum as serial.  It does no heavy timing, so it is reliable even on
a busy machine -- use ``autotune_timing.py`` (and a quiesced machine) for wall times.

Before running, on each machine:
  * check out the code under test and verify the line below prints ``'auto'``
    (``None`` means a stale checkout -- stop):
        python -c "import scqubits as scq; print(scq.settings.MULTIPROC_BLAS_THREADS)"
  * control the calibration file (HOME-global, shared across venvs):
        rm -f ~/.scqubits/parallel_calibration.json     # default heuristic
    A reason starting with ``measured:`` means a calibration file is in effect.

Run:  python tools/autotune_decisions.py
"""

import contextlib
import io
import os

import numpy as np

import scqubits as scq
from scqubits import settings

# (name, n_subsys, truncated_dim, ncut, num_points, evals_count, expectation)
WORKLOADS = [
    ("tiny           ", 3, 4, 30, 8, 10, "serial (too small)"),
    ("many_cheap     ", 3, 6, 40, 192, 20, "serial or few workers (points are cheap)"),
    (
        "few_large_dense",
        3,
        12,
        50,
        8,
        20,
        "num_cpus=1, BLAS uncapped (big dense, few pts)",
    ),
    ("many_large     ", 3, 11, 50, 48, 20, "parallel, BLAS capped to a small value"),
]


def build_hilbertspace(n_subsys, truncated_dim, ncut):
    """Return a chain of capacitively coupled tunable transmons and a flux updater."""
    qubits = [
        scq.TunableTransmon(
            EJmax=30.0,
            EC=0.2,
            d=0.1,
            flux=0.1 * i,
            ng=0.0,
            ncut=ncut,
            truncated_dim=truncated_dim,
            id_str=f"t{i}",
        )
        for i in range(n_subsys)
    ]
    hilbertspace = scq.HilbertSpace(qubits)
    for i in range(n_subsys - 1):
        hilbertspace.add_interaction(
            g_strength=0.1, op1=qubits[i].n_operator, op2=qubits[i + 1].n_operator
        )

    def update(flux):
        qubits[0].flux = flux

    return hilbertspace, update


def main():
    print(
        f"cores={os.cpu_count()}  "
        f"MULTIPROC_BLAS_THREADS default={settings.MULTIPROC_BLAS_THREADS!r}"
    )
    if settings.MULTIPROC_BLAS_THREADS is None:
        print("  WARNING: default is None, not 'auto' -- you are on a stale checkout.")
    print(
        f"\n{'workload':16}{'dim':>6}{'pts':>5}{'ev':>4}  "
        f"{'num_cpus':>9}{'blas':>6}   reason"
    )
    print("-" * 100)
    for name, n, trunc, ncut, pts, ev, expect in WORKLOADS:
        cfg = scq.recommend_parallelization(
            dimension=trunc**n, num_points=pts, evals_count=ev
        )
        print(
            f"{name:16}{trunc ** n:6d}{pts:5d}{ev:4d}  "
            f"{cfg.num_cpus:9d}{str(cfg.blas_threads):>6}   {cfg.reason}"
        )
        print(f"{'':16}expect: {expect}")

    print("\ncorrectness check: num_cpus='auto' vs serial")
    hilbertspace, update = build_hilbertspace(3, 11, 50)
    flux_vals = np.linspace(0.0, 0.4, 24)
    # The sweeps mutate the qubit flux, so scqubits emits a forced "parameters have
    # been changed / spectrum data could be outdated" warning (and a spawn-guard
    # notice) to stderr. They are expected here and only clutter the report, so run
    # the comparison with stderr captured and print just the result.
    with contextlib.redirect_stderr(io.StringIO()):
        serial = scq.ParameterSweep(
            hilbertspace=hilbertspace,
            paramvals_by_name={"flux": flux_vals},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus=1,
        )
        auto = scq.ParameterSweep(
            hilbertspace=hilbertspace,
            paramvals_by_name={"flux": flux_vals},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus="auto",
        )
        identical = np.allclose(
            np.sort(serial["evals"][:].ravel()), np.sort(auto["evals"][:].ravel())
        )
    print("  spectra identical (auto == serial):", bool(identical))


if __name__ == "__main__":
    main()
