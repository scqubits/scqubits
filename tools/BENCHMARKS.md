# Multiprocessing benchmarks

`benchmark_multiprocessing.py` is a standalone developer tool (not run by pytest)
for measuring the parallelized code paths in scqubits and tracking the effect of
future optimizations against a baseline.

## Running

```bash
# full baseline across worker counts, written to tools/benchmark_results/
python tools/benchmark_multiprocessing.py --scenario all --num-cpus 1,2,4 --repeats 3 --json

# single scenario / heavier per-task workload
python tools/benchmark_multiprocessing.py --scenario sweep --grid 64 --evals-count 30

# correctness guard (compares the 11x3 sweep against test reference evals)
python tools/benchmark_multiprocessing.py --scenario qubit --verify
```

Scenarios: `sweep` (`ParameterSweep.run`), `hspace`
(`HilbertSpace.get_spectrum_vs_paramvals`), `qubit`
(`TunableTransmon.get_spectrum_vs_paramvals`). The benchmarked system mirrors
`scqubits/tests/test_parametersweep.py`.

The harness instruments two known overhead sources without editing library code
(process-local monkeypatch): `map_calls` / `pool_spawn(s)` = how many pools are
spawned per run and the time spent constructing them; plus the dill payload size
of the `HilbertSpace` and the `update_hilbertspace` closure shipped to workers.

## Baseline (2026-05-30)

Windows 11, 20 logical cores, `MULTIPROC=pathos`, scqubits 4.3.1, default test
system (two TunableTransmons + Oscillator), 11x3 grid, evals_count=20, repeats=3.
Median wall time in seconds:

| scenario | num_cpus=1 | num_cpus=2 | num_cpus=4 | map_calls |
|----------|-----------:|-----------:|-----------:|----------:|
| sweep    | 0.127      | 2.509      | 2.729      | 4         |
| hspace   | 0.033      | 2.420      | 2.539      | 1         |
| qubit    | 0.028      | 2.384      | 2.406      | 1         |

dill payload: HilbertSpace ~2.0 KB, update closure ~1.4 KB.

### Observations
1. **For this small system, multiprocessing is net-negative in every scenario.**
   Per-task compute is sub-millisecond to a few ms, while there is a ~2.4 s floor
   dominated by Windows spawn-based worker startup and per-task dill IPC. To see
   any MP benefit you must raise the per-task cost (larger `ncut` /
   `truncated_dim` / `evals_count`, or more grid points); the harness is designed
   to find that crossover.
2. **A fresh pool is spawned for every parallel phase and never reused.** One
   `ParameterSweep.run()` reports `map_calls=4` (one bare-spectrum sweep per
   subsystem, plus the dressed sweep). `cpu_switch.get_map_method` always builds
   a new `pathos.ProcessPool` and overwrites `settings.POOL`, ignoring the cached
   pool and never terminating the orphaned one.

## Pool-reuse optimization (implemented)

`cpu_switch.get_map_method` now caches the pool in `settings.POOL` and reuses it
when a later call requests the same backend and worker count, instead of building
a fresh pool every time; a stale pool (different `num_cpus`) is shut down first.
This realizes the intent already noted at `settings.py` ("store processing pool
once generated"). Public `num_cpus` API unchanged; `dill`/pathos retained.

Before vs after, heavy profile, sweep, grid=8 (24 points), repeats=2, this box.
Median wall time (s) and pools constructed per `ParameterSweep.run()`:

| num_cpus | before (s) | after (s) | before speedup | after speedup | pools/run |
|---------:|-----------:|----------:|---------------:|--------------:|----------:|
| 1        | 18.75      | 19.16     | 1.00           | 1.00          | 0         |
| 2        | 13.28      | 12.82     | 1.41           | 1.49          | 4 -> 1    |
| 4        | 9.66       | 8.80      | 1.94           | 2.18          | 4 -> 1    |

**Finding:** pools constructed per run drop 4 -> 1 and the 4-core run is ~9%
faster, but the gain is modest. pathos already caches pools internally, so the
old code was *not* paying a full 4x worker-spawn cost; explicit reuse mainly
removes pool-object/cache churn and stops orphaning pools. The larger remaining
costs are the diagonalizations themselves and per-task serialization. Verified
no regression: `pytest test_parametersweep.py test_transmon.py --num_cpus 2`.

## Remaining optimization targets
- **Chunking** — the `.map` calls pass no `chunksize` (`cpu_switch.py`).
- **Per-task serialization** — avoid re-pickling the `HilbertSpace`/subsystem for
  every task (`scqubits/core/param_sweep.py`); likely the biggest lever here.

Constraint: `dill` is required for Windows spawn-based multiprocessing and must
not be removed; optimizations must preserve the public `num_cpus` API.
