# test_parallel_tuning.py
# meant to be run with 'pytest'
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

import numpy as np
import pytest

import scqubits as scq
import scqubits.settings as settings

from scqubits.utils.parallel_tuning import (
    ParallelConfig,
    _auto_config,
    _recommend,
    recommend_parallelization,
)


def _make_hilbertspace():
    """Three weakly coupled tunable transmons (dressed dimension 6**3 = 216, dense)."""
    qubits = [
        scq.TunableTransmon(
            EJmax=30.0,
            EC=0.2,
            d=0.1,
            flux=0.0,
            ng=0.0,
            ncut=50,
            truncated_dim=6,
            id_str=f"t{i}",
        )
        for i in range(3)
    ]
    hs = scq.HilbertSpace(qubits)
    for i in range(2):
        hs.add_interaction(
            g_strength=0.1, op1=qubits[i].n_operator, op2=qubits[i + 1].n_operator
        )

    def update(flux):
        qubits[0].flux = flux

    return hs, update


@pytest.fixture(autouse=True)
def _isolate_calibration(monkeypatch, tmp_path):
    """Point the calibration at a nonexistent file so the auto-config path uses the
    pure tier heuristic deterministically, regardless of any real calibration written
    on the developer's machine (otherwise auto-config == pure-heuristic comparisons
    are machine-dependent and flaky)."""
    monkeypatch.setattr(
        settings, "PARALLEL_CALIBRATION_PATH", str(tmp_path / "no-calibration.json")
    )
    import scqubits.utils.parallel_tuning as _pt

    _pt._calibration_cache.clear()


class TestRecommendHeuristic:
    """The pure heuristic core, with cores injected for determinism."""

    def test_single_point_is_serial(self):
        cfg = _recommend(216, 1, 20, False, 8)
        assert cfg.num_cpus == 1 and cfg.blas_threads is None

    def test_single_core_is_serial(self):
        cfg = _recommend(3125, 400, 10, False, 1)
        assert cfg.num_cpus == 1 and cfg.blas_threads is None

    def test_small_sweep_stays_serial(self):
        # dim 216 dense, only 16 points: below the parallel break-even
        cfg = _recommend(216, 16, 20, False, 10)
        assert cfg.num_cpus == 1 and cfg.blas_threads is None

    def test_many_points_uses_workers_with_single_blas(self):
        # dim 216 dense (< MEDIUM): many points -> workers, one BLAS thread each
        cfg = _recommend(216, 384, 20, False, 10)
        assert cfg.num_cpus > 1
        assert cfg.blas_threads == 1

    def test_large_dense_few_points_serial_uncapped(self):
        # large dense matrix, fewer points than cores: parallelize the BLAS, not points
        cfg = _recommend(3125, 4, 10, False, 10)
        assert cfg.num_cpus == 1 and cfg.blas_threads is None

    def test_large_dense_many_points_uses_blas_threads_per_worker(self):
        cfg = _recommend(3125, 16, 10, False, 10)
        assert cfg.num_cpus > 1
        assert cfg.blas_threads is not None and cfg.blas_threads > 1

    def test_oversubscription_guard_never_exceeds_cores(self):
        for dim in (64, 216, 1024, 3125):
            for points in (8, 64, 384, 2048):
                for cores in (1, 2, 4, 8, 10, 16):
                    cfg = _recommend(dim, points, 20, False, cores)
                    cap = cfg.blas_threads if cfg.blas_threads is not None else cores
                    assert cfg.num_cpus * cap <= cores
                    assert cfg.num_cpus >= 1

    def test_sparse_is_cheaper_than_dense(self):
        # same large system: sparse (few evals) is cheaper per point than dense
        sparse = _recommend(4096, 32, 10, True, 10)
        dense = _recommend(4096, 32, 200, False, 10)
        assert dense.num_cpus >= sparse.num_cpus


class TestRecommendParallelizationPublic:
    def test_explicit_descriptors(self):
        cfg = recommend_parallelization(dimension=216, num_points=384, evals_count=20)
        assert isinstance(cfg, ParallelConfig)
        assert cfg.num_cpus >= 1

    def test_missing_inputs_raise(self):
        with pytest.raises(ValueError):
            recommend_parallelization()
        with pytest.raises(ValueError):
            recommend_parallelization(dimension=216)  # no num_points

    def test_apply_updates_settings_live(self, monkeypatch):
        monkeypatch.setattr(settings, "NUM_CPUS", 1)
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", None)
        cfg = recommend_parallelization(
            dimension=216, num_points=384, evals_count=20, cores=8, apply=True
        )
        assert settings.NUM_CPUS == cfg.num_cpus
        assert settings.MULTIPROC_BLAS_THREADS == cfg.blas_threads

    def test_apply_serial_keeps_global_blas_cap(self, monkeypatch):
        # a serial recommendation has blas_threads=None; apply must not clobber the
        # global cap (default "auto") that governs later parallel sweeps
        monkeypatch.setattr(settings, "NUM_CPUS", 4)
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", "auto")
        cfg = recommend_parallelization(
            dimension=216, num_points=4, evals_count=20, cores=8, apply=True
        )
        assert cfg.num_cpus == 1 and cfg.blas_threads is None
        assert settings.NUM_CPUS == 1
        assert settings.MULTIPROC_BLAS_THREADS == "auto"  # left intact

    def test_no_apply_leaves_settings_untouched(self, monkeypatch):
        monkeypatch.setattr(settings, "NUM_CPUS", 1)
        recommend_parallelization(dimension=216, num_points=384, evals_count=20)
        assert settings.NUM_CPUS == 1

    def test_introspects_hilbertspace(self):
        hs, _ = _make_hilbertspace()
        cfg = recommend_parallelization(hilbertspace=hs, num_points=16, evals_count=20)
        # dimension is read from the HilbertSpace (216); 16 points -> serial
        assert cfg.num_cpus == 1

    def test_introspects_bare_qubit(self):
        qubit = scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=30)
        cfg = recommend_parallelization(qubit=qubit, num_points=400, evals_count=6)
        assert cfg.num_cpus >= 1

    def test_constructed_autorun_sweep_warns(self):
        hs, update = _make_hilbertspace()
        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"flux": np.linspace(0.0, 0.5, 4)},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus=1,
            autorun=True,
        )
        with pytest.warns(UserWarning, match="already run"):
            recommend_parallelization(param_sweep=sweep)


class TestAutoHookBeforeRun:
    """The num_cpus='auto' / AUTO_PARALLEL hook resolves before any work runs."""

    def test_auto_config_matches_pure_heuristic(self):
        # _auto_config detects sparsity and cores; for dim 216 it is dense
        import os

        cores = os.cpu_count() or 1
        cfg = _auto_config(216, 384, 20)
        assert cfg.num_cpus == _recommend(216, 384, 20, False, cores).num_cpus

    def test_auto_parallel_default_off(self):
        assert settings.AUTO_PARALLEL is False

    def test_sweep_auto_picks_serial_for_tiny_grid(self):
        hs, update = _make_hilbertspace()
        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"flux": np.linspace(0.0, 0.5, 6)},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus="auto",
            autorun=False,  # only exercise the decision, not the run
        )
        assert sweep._num_cpus == 1
        assert sweep._blas_threads is None

    def test_sweep_auto_picks_workers_for_large_grid(self):
        hs, update = _make_hilbertspace()
        expected = _auto_config(hs.dimension, 200, 20)
        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"flux": np.linspace(0.0, 0.5, 200)},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus="auto",
            autorun=False,
        )
        assert sweep._num_cpus == expected.num_cpus
        assert sweep._blas_threads == expected.blas_threads

    def test_auto_parallel_flag_routes_through_when_num_cpus_unset(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTO_PARALLEL", True)
        hs, update = _make_hilbertspace()
        expected = _auto_config(hs.dimension, 200, 20)
        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"flux": np.linspace(0.0, 0.5, 200)},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus=None,  # unset -> AUTO_PARALLEL takes over
            autorun=False,
        )
        assert sweep._num_cpus == expected.num_cpus

    def test_explicit_num_cpus_bypasses_heuristic(self):
        hs, update = _make_hilbertspace()
        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"flux": np.linspace(0.0, 0.5, 200)},
            update_hilbertspace=update,
            evals_count=20,
            num_cpus=1,
            autorun=False,
        )
        assert sweep._num_cpus == 1
        assert sweep._blas_threads is None


class TestNumCpusValidation:
    """An explicit num_cpus is validated at the boundary, not silently coerced."""

    def test_resolve_explicit_rejects_bool(self):
        from scqubits.utils.cpu_switch import _resolve_explicit_num_cpus

        with pytest.raises(TypeError):
            _resolve_explicit_num_cpus(True)

    def test_resolve_explicit_rejects_negative(self):
        from scqubits.utils.cpu_switch import _resolve_explicit_num_cpus

        with pytest.raises(ValueError):
            _resolve_explicit_num_cpus(-2)

    def test_resolve_explicit_rejects_zero(self):
        from scqubits.utils.cpu_switch import _resolve_explicit_num_cpus

        with pytest.raises(ValueError):
            _resolve_explicit_num_cpus(0)

    def test_resolve_explicit_rejects_typo_sentinel(self):
        from scqubits.utils.cpu_switch import _resolve_explicit_num_cpus

        with pytest.raises(ValueError):
            _resolve_explicit_num_cpus("Auto")

    def test_resolve_explicit_none_uses_global_default(self, monkeypatch):
        from scqubits.utils.cpu_switch import _resolve_explicit_num_cpus

        monkeypatch.setattr(settings, "NUM_CPUS", 3)
        assert _resolve_explicit_num_cpus(None) == 3

    def test_resolve_explicit_passes_positive_int(self):
        from scqubits.utils.cpu_switch import _resolve_explicit_num_cpus

        assert _resolve_explicit_num_cpus(4) == 4

    def test_get_map_method_rejects_zero(self):
        from scqubits.utils.cpu_switch import get_map_method

        with pytest.raises(ValueError):
            get_map_method(0)

    def test_sweep_rejects_bool_num_cpus(self):
        hs, update = _make_hilbertspace()
        with pytest.raises(TypeError):
            scq.ParameterSweep(
                hilbertspace=hs,
                paramvals_by_name={"flux": np.linspace(0.0, 0.5, 6)},
                update_hilbertspace=update,
                evals_count=20,
                num_cpus=True,
                autorun=False,
            )

    def test_sweep_rejects_typo_sentinel(self):
        hs, update = _make_hilbertspace()
        with pytest.raises(ValueError):
            scq.ParameterSweep(
                hilbertspace=hs,
                paramvals_by_name={"flux": np.linspace(0.0, 0.5, 6)},
                update_hilbertspace=update,
                evals_count=20,
                num_cpus="Auto",
                autorun=False,
            )

    def test_recommend_rejects_nonpositive_dimension(self):
        with pytest.raises(ValueError):
            recommend_parallelization(dimension=-1, num_points=100)
        with pytest.raises(ValueError):
            recommend_parallelization(dimension=216, num_points=0)
