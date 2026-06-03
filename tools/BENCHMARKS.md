# Multiprocessing benchmarks

`benchmark_multiprocessing.py` is a standalone developer tool (not run by pytest)
for measuring the parallelized code paths in scqubits and tracking the effect of
optimizations.

## For users: the shipped heuristic and calibration

End users do not need this maintainer harness. scqubits ships a workload-aware
recommendation, `scqubits.recommend_parallelization(...)` (plus the per-sweep
`num_cpus="auto"` sentinel and the `settings.AUTO_PARALLEL` flag), and a one-time
machine calibration, `scqubits.calibrate_parallelization()`, which measures per-task
dispatch overhead, pool-startup cost, and per-point diagonalization cost and feeds the
recommendation a measured break-even. See the *Parallel Processing* page of the user
guide. Portable lesson it encodes: **cap BLAS threads, and add worker processes only
once the grid is large enough to amortize the pool startup** -- with the crossover
measured per machine.

`benchmark_multiprocessing.py` below is the deeper maintainer tool for measuring the
parallelized code paths and tracking the effect of optimizations.

## Running

```bash
# trustworthy wall times: each (scenario, num_cpus) in a fresh subprocess
python tools/benchmark_multiprocessing.py --profile heavy --num-cpus 1,2,4 --repeats 3 --isolate

# heavier per-task workload / more grid points
python tools/benchmark_multiprocessing.py --profile heavy --grid 24 --isolate

# correctness guard (light 11x3 sweep vs test reference evals)
python tools/benchmark_multiprocessing.py --scenario qubit --verify

# write results JSON under tools/benchmark_results/ (gitignored)
python tools/benchmark_multiprocessing.py --profile heavy --isolate --json
```

Scenarios: `sweep` (`ParameterSweep.run`), `hspace`
(`HilbertSpace.get_spectrum_vs_paramvals`), `qubit`
(`TunableTransmon.get_spectrum_vs_paramvals`). Profiles: `light` (matches
`tests/test_parametersweep.py`; used by `--verify`) and `heavy` (enlarged dims so
MP can pay off). The benchmarked system mirrors `tests/test_parametersweep.py`.

The harness instruments, without editing library code: `map_calls` (get_map_method
calls per run), `pools_created` (real pool constructions per run), `pool_spawn(s)`,
and the dill payload of the work shipped to workers.

> **Always use `--isolate` for wall times.** Repeated pool-spawning runs in one
> process leave worker processes that contaminate later timings (an early non-
> isolated serial run read ~19 s; the same run in a clean process is ~5.6 s). The
> `--isolate` numbers below are contamination-free; the deterministic metrics
> (payload sizes, pools-per-run) are reliable in either mode.

## The dominant factor: BLAS thread oversubscription

scqubits' diagonalizations are *small* dense matrices (set by truncated dims), and
numpy/scipy run them on a multithreaded BLAS (OpenBLAS here, 20 threads by
default). Two problems follow:

- On small matrices, spreading one `eigh` across 20 BLAS threads is pure overhead
  -- often *slower* than single-threaded.
- With `num_cpus > 1`, every worker process launches its own full BLAS pool, so
  `num_cpus x BLAS-threads` vastly oversubscribes the cores.

Capping BLAS threads per worker via `settings.MULTIPROC_BLAS_THREADS` (default
`None` = unchanged; sets `OPENBLAS/MKL/OMP/NUMEXPR_NUM_THREADS` for spawned
workers in `cpu_switch._new_pool`) removes both. Same heavy sweep, 72 points,
`--isolate`, this box:

| num_cpus | default (BLAS=20/worker) | MULTIPROC_BLAS_THREADS=1 |
|---------:|-------------------------:|-------------------------:|
| 1        | 16.8 s                   | 17.4 s (unaffected*)     |
| 2        | 13.5 s                   | 5.9 s                    |
| 4        | 10.2 s (eff 0.41)        | 4.6 s (eff 0.95)         |
| 8        | 8.6 s                    | 4.6 s                    |

Capping nearly halves the best achievable time (8.6 s -> 4.6 s) and turns poor
parallel efficiency (0.41) into near-linear (0.95) at 4 cores. **This is a far
bigger runtime lever than pool reuse or serialization**, which move wall time
negligibly.

(*) `num_cpus=1` uses no pool, so this worker-side setting does not touch it; the
single-core `eigh` still runs on the parent's BLAS pool. To speed up *serial* runs
on small-matrix systems, set `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` in the
environment *before* importing scqubits (here that took serial 16.8 s -> 6.4 s).

Caveats (prototype): machine/library/workload specific -- large matrices (big
Circuits) *do* benefit from BLAS threads, so 1 is not universally optimal
(`cores // num_cpus` is a reasonable rule of thumb). The env mechanism only takes
effect for spawn-based workers (the Windows default); fork-based workers (Linux)
would need a `threadpoolctl`-based worker initializer instead.

## Where multiprocessing actually helps

Clean `--isolate` runs, heavy profile, this box (Windows 11, 20 cores, scqubits
4.3.1). Median wall time (s):

| sweep grid        | num_cpus=1 | num_cpus=2 | num_cpus=4 |
|-------------------|-----------:|-----------:|-----------:|
| grid=8  (24 pts)  | 5.60       | 6.06 (0.93x) | 4.95 (1.13x) |
| grid=24 (72 pts)  | 16.62      | --           | 10.04 (1.66x) |

MP is net-negative to marginal on small grids (spawn + per-task overhead dominate)
and only starts paying off with enough grid points to amortize: ~1.7x on 4 cores
at 72 points. Bigger grids / heavier per-point cost push efficiency higher.

## Pool-reuse optimization (implemented)

`cpu_switch.get_map_method` now caches the pool in `settings.POOL` and reuses it
when backend + `num_cpus` match (stale pools are shut down first), instead of
building one per parallel phase. **Deterministic:** pools constructed per
`ParameterSweep.run()` drop from **4 to 1**. Wall-time effect is negligible in
clean runs (pathos already caches pools internally, so the old code was not paying
a full 4x worker-spawn cost); the real benefit is not orphaning pools.

## Per-task serialization optimization (implemented, prototype)

The dressed-spectrum worker `_update_and_compute_dressed_esys` was a *bound*
method, so `pool.map` pickled the whole `ParameterSweep` into every task,
including `self._data` (the bare eigendata for the **entire grid**). Shipped once
per task, that is O(N^2) IPC. The dressed sweep now detaches those arrays during
dispatch and passes each grid point only its own bare slice; `self._data` is
restored afterwards.

**Deterministic** (dill, heavy, 72 points):

| quantity                      | before   | after  |
|-------------------------------|---------:|-------:|
| dressed-task function payload | 3406 KB  | 3.8 KB |
| total dressed-phase IPC       | ~240 MB  | ~4 MB  |
| scaling in grid size N        | O(N^2)   | O(N)   |

**Wall time is unchanged.** Clean `--isolate` before/after at 72 points, 4 cores:
old 10.08 s vs new 10.04 s (serial 16.89 vs 16.62) -- within noise. At these
scales compute dominates and the pickling overlaps it, so this is a **memory/IPC
footprint** fix (relevant for very large sweeps and heavy Circuit subsystems where
per-point bare data is large), not a speedup.

Correctness verified: light sweep matches reference evals (serial and
`num_cpus=2`); `pytest test_parametersweep.py test_circuit.py
test_spectrumlookup.py` pass (`test_circuit` covers hierarchical diagonalization).

> Prototype caveats: an advanced `update_hilbertspace(self, ...)` that reads
> `sweep._data["bare_evals"/...]` mid-sweep would now see `None` during dispatch
> (standard value-only update functions are unaffected). The circuit/HD path is
> covered by `test_circuit` but warrants a domain review before release.

## Remaining optimization targets
- **BLAS-thread coordination — largely addressed.** `MULTIPROC_BLAS_THREADS` now
  reaches fork-based (Linux) workers via `threadpoolctl`, and the shipped
  `recommend_parallelization` heuristic supplies a sensible auto cap
  (~`cores // num_cpus`, scoped per sweep). Still open: a documented cap for the
  serial `num_cpus=1` path (the parent's BLAS is currently left uncapped).
- **Chunking — investigated, no lever.** `pool.map` (unlike `imap`) already
  auto-computes `chunksize = ceil(n_tasks / (4 * num_workers))` when none is given
  (see `multiprocess.pool.Pool._map_async`), so scqubits already gets sensible
  batching for free. Passing an explicit chunksize only reproduces that default; a
  hand-tuned value is workload-dependent, not a general win.
- **Bare-sweep granularity / grid size** — MP only pays off past ~tens of points;
  worth documenting guidance for users on when to set `num_cpus > 1`.

Constraint: `dill` is required for Windows spawn-based multiprocessing and must
not be removed; optimizations must preserve the public `num_cpus` API.
