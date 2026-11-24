# test_parallel_calibration.py
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

import json

import pytest

import scqubits as scq
import scqubits.settings as settings
import scqubits.utils.parallel_calibration as pc
import scqubits.utils.parallel_tuning as pt

from scqubits.utils.parallel_calibration import (
    MachineCalibration,
    calibrate_parallelization,
    default_calibration_path,
    load_calibration,
)


def _make_calibration(overhead=0.001, startup=1.5):
    return MachineCalibration(
        cores=10,
        overhead_s=overhead,
        pool_startup_s=startup,
        cost_samples=[
            {"dimension": 216, "is_sparse": False, "seconds_per_point": 0.30},
            {"dimension": 1296, "is_sparse": False, "seconds_per_point": 0.05},
            {"dimension": 1296, "is_sparse": True, "seconds_per_point": 0.04},
        ],
    )


class TestMachineCalibration:
    def test_estimated_cost_nearest_matching_sample(self):
        cal = _make_calibration()
        assert cal.estimated_cost_per_point(216, False) == 0.30
        assert cal.estimated_cost_per_point(1296, True) == 0.04

    def test_estimated_cost_out_of_range_returns_none(self):
        cal = _make_calibration()
        # far below the smallest dense sample (216 / 4 = 54) -> untrusted
        assert cal.estimated_cost_per_point(30, False) is None
        # far above the largest dense sample (1296 * 4) -> untrusted
        assert cal.estimated_cost_per_point(20000, False) is None

    def test_estimated_cost_no_matching_sparsity_returns_none(self):
        cal = MachineCalibration(
            cores=8,
            overhead_s=0.001,
            cost_samples=[
                {"dimension": 216, "is_sparse": False, "seconds_per_point": 0.1}
            ],
        )
        assert cal.estimated_cost_per_point(1296, True) is None


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        path = str(tmp_path / "cal.json")
        cal = _make_calibration()
        pc._save_calibration(cal, path)
        back = load_calibration(path)
        assert back is not None
        assert back.cores == cal.cores
        assert back.pool_startup_s == cal.pool_startup_s
        assert back.cost_samples == cal.cost_samples

    def test_save_bare_filename_no_dir(self, tmp_path, monkeypatch):
        # a path with no directory component must not crash on os.makedirs("")
        monkeypatch.chdir(tmp_path)
        pc._save_calibration(_make_calibration(), "cal.json")
        assert load_calibration("cal.json") is not None

    def test_load_missing_returns_none(self, tmp_path):
        assert load_calibration(str(tmp_path / "absent.json")) is None

    def test_load_garbage_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {")
        assert load_calibration(str(path)) is None

    def test_default_path_honors_setting(self, monkeypatch, tmp_path):
        target = str(tmp_path / "custom.json")
        monkeypatch.setattr(settings, "PARALLEL_CALIBRATION_PATH", target)
        assert default_calibration_path() == target

    def test_default_path_fallback(self, monkeypatch):
        monkeypatch.setattr(settings, "PARALLEL_CALIBRATION_PATH", None)
        assert default_calibration_path().endswith("parallel_calibration.json")


class TestCalibrateOrchestration:
    def test_derives_overhead_startup_and_costs(self, monkeypatch):
        # stub the subprocess measurement so the test is fast and deterministic
        def fake_measure(profile, num_cpus, n_points, blas):
            if profile == "tiny":
                if num_cpus == 1:
                    return {"warm_s": 0.10, "cold_s": 0.10}
                return {"warm_s": 0.04, "cold_s": 1.60}  # cold includes pool startup
            return {"warm_s": 0.24, "cold_s": 0.30}  # 0.24 / 24 points = 0.01 s/point

        monkeypatch.setattr(pc, "_measure", fake_measure)
        cal = calibrate_parallelization(persist=False, explain=False)
        assert cal.pool_startup_s == pytest.approx(1.56, abs=1e-6)  # 1.60 - 0.04
        assert cal.overhead_s >= 0.0
        assert len(cal.cost_samples) == 3
        for sample in cal.cost_samples:
            assert sample["seconds_per_point"] == pytest.approx(0.01, abs=1e-9)

    def test_fits_startup_growth_with_worker_count(self, monkeypatch):
        # cold pool startup grows with worker count: startup(n) = 0.5 + 0.1 * n.
        monkeypatch.setattr(pc.os, "cpu_count", lambda: 8)

        def fake_measure(profile, num_cpus, n_points, blas):
            if profile == "tiny":
                if num_cpus == 1:
                    return {"warm_s": 0.10, "cold_s": 0.10}
                warm = 0.04
                return {"warm_s": warm, "cold_s": warm + 0.5 + 0.1 * num_cpus}
            return {"warm_s": 0.24, "cold_s": 0.30}

        monkeypatch.setattr(pc, "_measure", fake_measure)
        cal = calibrate_parallelization(persist=False, explain=False)
        # two-point fit from startup(2)=0.7 and startup(8)=1.3 -> base 0.5, slope 0.1
        assert cal.pool_startup_per_worker_s == pytest.approx(0.1, abs=1e-6)
        assert cal.pool_startup_base_s == pytest.approx(0.5, abs=1e-6)


class TestHeuristicConsumesCalibration:
    def test_measured_model_drives_break_even(self, monkeypatch, tmp_path):
        path = str(tmp_path / "cal.json")
        with open(path, "w") as handle:
            json.dump(pc.asdict(_make_calibration(startup=1.5)), handle)
        monkeypatch.setattr(settings, "PARALLEL_CALIBRATION_PATH", path)
        pt._calibration_cache.clear()

        # dim 216 dense, per-point 0.30 s, startup 1.5 s: few points stay serial,
        # many points repay the startup and parallelize
        few = scq.recommend_parallelization(dimension=216, num_points=4, evals_count=20)
        many = scq.recommend_parallelization(
            dimension=216, num_points=200, evals_count=20
        )
        assert few.num_cpus == 1
        assert many.num_cpus > 1
        assert "measured" in many.reason

    def test_scaled_startup_picks_moderate_worker_count(self, monkeypatch, tmp_path):
        # startup grows with workers (2.0s + 0.4s/worker): for many cheap *light*
        # points the heuristic should pick a moderate count, not all cores, because
        # extra workers cost more startup than they save.
        cal = MachineCalibration(
            cores=20,
            overhead_s=0.001,
            pool_startup_s=10.0,
            pool_startup_base_s=2.0,
            pool_startup_per_worker_s=0.4,
            cost_samples=[
                {"dimension": 216, "is_sparse": False, "seconds_per_point": 0.03}
            ],
        )
        path = str(tmp_path / "cal.json")
        with open(path, "w") as handle:
            json.dump(pc.asdict(cal), handle)
        monkeypatch.setattr(settings, "PARALLEL_CALIBRATION_PATH", path)
        pt._calibration_cache.clear()
        cfg = scq.recommend_parallelization(
            dimension=216, num_points=192, evals_count=20, cores=20
        )
        assert 1 < cfg.num_cpus < 20  # moderate, not max and not serial
        assert cfg.blas_threads == 1

    def test_cheap_startup_uses_all_cores_for_light_work(self, monkeypatch, tmp_path):
        # fork-like: startup ~0.1s flat -> light many-point work should use all cores.
        cal = MachineCalibration(
            cores=8,
            overhead_s=0.0005,
            pool_startup_s=0.1,
            pool_startup_base_s=0.1,
            pool_startup_per_worker_s=0.0,
            cost_samples=[
                {"dimension": 216, "is_sparse": False, "seconds_per_point": 0.05}
            ],
        )
        path = str(tmp_path / "cal.json")
        with open(path, "w") as handle:
            json.dump(pc.asdict(cal), handle)
        monkeypatch.setattr(settings, "PARALLEL_CALIBRATION_PATH", path)
        pt._calibration_cache.clear()
        cfg = scq.recommend_parallelization(
            dimension=216, num_points=400, evals_count=20, cores=8
        )
        assert cfg.num_cpus == 8

    def test_falls_back_to_tiers_without_calibration(self, monkeypatch, tmp_path):
        # point at a non-existent calibration file -> static tier heuristic
        monkeypatch.setattr(
            settings, "PARALLEL_CALIBRATION_PATH", str(tmp_path / "none.json")
        )
        pt._calibration_cache.clear()
        cfg = scq.recommend_parallelization(
            dimension=216, num_points=16, evals_count=20
        )
        assert cfg.num_cpus == 1
        assert "break-even" in cfg.reason  # tier-heuristic phrasing

    def test_out_of_range_dim_falls_back_to_tiers(self, monkeypatch, tmp_path):
        path = str(tmp_path / "cal.json")
        with open(path, "w") as handle:
            json.dump(pc.asdict(_make_calibration()), handle)
        monkeypatch.setattr(settings, "PARALLEL_CALIBRATION_PATH", path)
        pt._calibration_cache.clear()
        # dim 30 is far below the smallest dense sample -> tier heuristic, serial
        cfg = scq.recommend_parallelization(dimension=30, num_points=80, evals_count=6)
        assert cfg.num_cpus == 1


class TestMeasureSubprocess:
    def test_measure_returns_warm_and_cold(self):
        # one real subprocess measurement of the tiny probe (serial, few points)
        result = pc._measure("tiny", 1, 8, blas=1)
        assert "warm_s" in result and "cold_s" in result
        assert result["warm_s"] >= 0.0


class TestCostInterpolation:
    """estimated_cost_per_point interpolates instead of jumping at the midpoint."""

    def test_interpolates_strictly_between_samples(self):
        cal = _make_calibration()  # dense: dim 216 -> 0.30, dim 1296 -> 0.05
        mid = cal.estimated_cost_per_point(540, False)
        assert mid is not None
        assert 0.05 < mid < 0.30

    def test_no_discontinuity_at_old_nearest_neighbor_boundary(self):
        # nearest-neighbor jumped 6x at the arithmetic midpoint (~756) of 216 and
        # 1296; log-log interpolation is continuous there.
        cal = _make_calibration()
        lo = cal.estimated_cost_per_point(755, False)
        hi = cal.estimated_cost_per_point(757, False)
        assert lo is not None and hi is not None
        assert abs(hi - lo) < 0.02

    def test_endpoints_return_sampled_cost(self):
        cal = _make_calibration()
        assert cal.estimated_cost_per_point(216, False) == 0.30
        assert cal.estimated_cost_per_point(1296, False) == 0.05

    def test_zero_cost_sample_ignored(self):
        cal = MachineCalibration(
            cores=8,
            overhead_s=0.001,
            cost_samples=[
                {"dimension": 216, "is_sparse": False, "seconds_per_point": 0.0}
            ],
        )
        # a 0 s/point sample would imply infinite speedup; it is skipped -> None
        assert cal.estimated_cost_per_point(216, False) is None


class TestValueValidation:
    """Corrupt calibration values fail fast (and the loader maps that to None)."""

    def test_rejects_zero_cores(self):
        with pytest.raises(ValueError):
            MachineCalibration(cores=0, overhead_s=0.001)

    def test_rejects_negative_overhead(self):
        with pytest.raises(ValueError):
            MachineCalibration(cores=8, overhead_s=-0.1)

    def test_rejects_negative_cost(self):
        with pytest.raises(ValueError):
            MachineCalibration(
                cores=8,
                overhead_s=0.001,
                cost_samples=[
                    {"dimension": 216, "is_sparse": False, "seconds_per_point": -1.0}
                ],
            )

    def test_rejects_cost_sample_missing_key(self):
        with pytest.raises(ValueError):
            MachineCalibration(
                cores=8,
                overhead_s=0.001,
                cost_samples=[{"dimension": 216, "is_sparse": False}],
            )

    def test_load_corrupt_values_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "cores": 8,
                    "overhead_s": 0.001,
                    "cost_samples": [
                        {
                            "dimension": 216,
                            "is_sparse": False,
                            "seconds_per_point": -5,
                        }
                    ],
                }
            )
        )
        assert load_calibration(str(path)) is None

    def test_load_wrong_type_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"cores": "eight", "overhead_s": 0.001}))
        assert load_calibration(str(path)) is None


class TestStartMethodGuard:
    """A calibration from a different process start method is not reused."""

    def test_mismatched_start_method_ignored(self, tmp_path):
        cal_dict = pc.asdict(_make_calibration())
        cal_dict["start_method"] = "not-this-machines-method"
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(cal_dict))
        with pytest.warns(RuntimeWarning, match="start method"):
            assert load_calibration(str(path)) is None

    def test_empty_start_method_accepted(self, tmp_path):
        # files predating the field (start_method == "") load fine
        cal_dict = pc.asdict(_make_calibration())
        cal_dict["start_method"] = ""
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(cal_dict))
        assert load_calibration(str(path)) is not None

    def test_matching_start_method_accepted(self, tmp_path):
        from scqubits.utils.cpu_switch import _resolve_start_method

        cal_dict = pc.asdict(_make_calibration())
        cal_dict["start_method"] = _resolve_start_method()
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(cal_dict))
        assert load_calibration(str(path)) is not None


class TestStartupFit:
    """The two-point pool-startup fit handles noisy/inverted measurements."""

    def test_normal_fit(self):
        base, per = pc._fit_pool_startup(
            pool_startup_s=1.3, startup_lo=0.7, workers=8, low_workers=2
        )
        assert per == pytest.approx(0.1, abs=1e-9)
        assert base == pytest.approx(0.5, abs=1e-9)

    def test_inverted_fit_warns_and_flattens(self):
        with pytest.warns(RuntimeWarning, match="throttling"):
            base, per = pc._fit_pool_startup(
                pool_startup_s=0.9, startup_lo=1.2, workers=8, low_workers=2
            )
        assert per == 0.0
        assert base == pytest.approx(1.2, abs=1e-9)

    def test_single_core_is_flat(self):
        base, per = pc._fit_pool_startup(
            pool_startup_s=0.5, startup_lo=0.5, workers=1, low_workers=1
        )
        assert per == 0.0
        assert base == 0.5
