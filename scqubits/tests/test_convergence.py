# test_convergence.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
"""Integration tests for the convergence-diagnostics framework.

Covers dataclass construction; cheap / moderate / strict energy convergence
and degradation across the qubit classes; per-level checks; recommendations;
and the top-level ``check_convergence`` shim.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

import scqubits as sq

from scqubits.core.convergence import _assign_level_statuses, _status_rank
from scqubits.core.convergence_report import (
    ConvergenceReport,
    ImplementationAudit,
    LevelVerdict,
)

pytestmark = pytest.mark.slow


# ----------------------------------------------------------------- dataclass tests


class TestDataclasses:
    def _make_audit(self) -> ImplementationAudit:
        return ImplementationAudit(
            scqubits_version="x.y.z",
            scqubits_commit=None,
            qubit_class="Transmon",
            basis="charge",
            diagonalization_method="default",
            cutoff_parameters={"ncut": 31},
            fd_stencil_order=None,
            fd_box=None,
            nonpoly_backend=None,
            n_levels_requested=5,
            n_levels_buffer=0,
            mode="moderate",
            refinement="one_step",
        )

    def test_level_verdict_is_frozen(self):
        verdict = LevelVerdict(
            level_index=0,
            status="maybe_converged",
            status_scope="absolute",
            abs_err_est_GHz=1e-10,
            eps_gap_est=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.level_index = 1

    def test_implementation_audit_is_frozen(self):
        audit = self._make_audit()
        with pytest.raises(dataclasses.FrozenInstanceError):
            audit.qubit_class = "Foo"

    def test_convergence_report_asdict_roundtrip(self):
        verdict = LevelVerdict(
            level_index=0,
            status="maybe_converged",
            status_scope="absolute",
            abs_err_est_GHz=1e-10,
            eps_gap_est=None,
        )
        report = ConvergenceReport(
            per_level=[verdict],
            aggregate_status="maybe_converged",
            worst_level=0,
            channel_breakdown_GHz={"charge_tail": 1e-10},
            clusters=[(0,)],
            recommendations=[],
            implementation_audit=self._make_audit(),
        )
        d = dataclasses.asdict(report)
        assert d["aggregate_status"] == "maybe_converged"
        assert d["per_level"][0]["level_index"] == 0
        assert d["implementation_audit"]["mode"] == "moderate"

    def test_summary_renders_and_str_delegates(self):
        verdict = LevelVerdict(
            level_index=0,
            status="maybe_converged",
            status_scope="absolute",
            abs_err_est_GHz=1e-10,
            eps_gap_est=None,
        )
        report = ConvergenceReport(
            per_level=[verdict],
            aggregate_status="maybe_converged",
            worst_level=0,
            channel_breakdown_GHz={"charge_tail": 1e-10},
            clusters=[(0,)],
            recommendations=["do X"],
            implementation_audit=self._make_audit(),
        )
        text = report.summary()
        assert "aggregate: maybe_converged" in text
        assert "lvl" in text and "charge_tail" in text
        assert "-> do X" in text
        # The inapplicable observed-gap metric is omitted, not shown as a placeholder.
        assert "eps_gap" not in text
        # __str__ delegates to summary(), so print(report) shows the same text.
        assert str(report) == text

    def test_level_accessor_looks_up_by_index(self):
        verdicts = [
            LevelVerdict(
                level_index=k,
                status="maybe_converged",
                status_scope="absolute",
                abs_err_est_GHz=1e-10,
                eps_gap_est=None,
            )
            for k in range(3)
        ]
        report = ConvergenceReport(
            per_level=verdicts,
            aggregate_status="maybe_converged",
            worst_level=2,
            channel_breakdown_GHz={},
            clusters=[(0,), (1,), (2,)],
            recommendations=[],
            implementation_audit=self._make_audit(),
        )
        assert report.level(1).level_index == 1
        assert report.level(report.worst_level).level_index == 2
        with pytest.raises(KeyError):
            report.level(99)


# ---------------------------------------------------------------- API validation


class TestAPIValidation:
    def test_check_convergence_on_non_checkable_raises(self):
        # Oscillator is not (yet) ConvergenceCheckable.
        osc = sq.Oscillator(E_osc=5.0, truncated_dim=3)
        with pytest.raises(TypeError, match="does not support convergence checking"):
            sq.check_convergence(osc, n_levels=2, mode="moderate")

    def test_invalid_mode_raises(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="mode must be"):
            tmon.check_convergence(n_levels=3, mode="turbo")

    def test_invalid_scope_raises(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="scope must be"):
            tmon.check_convergence(n_levels=3, scope="LC_scale")

    def test_include_derived_requires_derived_quantities(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="requires derived_quantities"):
            tmon.check_convergence(n_levels=3, include_derived=True)

    def test_include_derived_rejects_cheap_mode(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="moderate"):
            tmon.check_convergence(
                n_levels=3,
                mode="cheap",
                include_derived=True,
                derived_quantities=["wavefunctions"],
            )

    def test_unknown_derived_quantity_raises(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="unknown derived_quantities"):
            tmon.check_convergence(
                n_levels=3, include_derived=True, derived_quantities=["bogus"]
            )


# ----------------------------------------------------------- moderate-mode tests


class TestEnergyModerateMode:
    def test_well_converged_transmon_is_maybe_converged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=5, mode="moderate", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "maybe_converged"
        # All five levels should be maybe_converged (best in moderate mode).
        statuses = {v.status for v in report.per_level}
        assert statuses == {"maybe_converged"}
        # The audit records the requested mode and the cutoff parameter.
        assert report.implementation_audit.mode == "moderate"
        assert report.implementation_audit.cutoff_parameters["ncut"] == 31

    def test_undersized_transmon_is_distrust(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=5, mode="moderate", target_abs_GHz=1e-6
        )
        assert report.aggregate_status == "distrust"
        # Recommendation should propose increasing ncut.
        assert any("ncut" in r for r in report.recommendations)

    @pytest.mark.parametrize("mode", ["cheap", "moderate", "strict"])
    def test_abs_err_estimate_is_unit_invariant(self, mode):
        # The same physical transmon expressed in GHz vs MHz must yield the same
        # abs_err_est_GHz and the same verdict: the energy diagnostics report in
        # GHz regardless of the active unit system (regression for the
        # GHz-hardcoding raised in issue #306). under-converged (ncut=5) so the
        # error is a real truncation error, well above the eigensolver floor.
        original_units = sq.get_units()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sq.set_units("GHz")
                rep_ghz = sq.Transmon(
                    EJ=15.0, EC=0.3, ng=0.0, ncut=5, truncated_dim=6
                ).check_convergence(n_levels=4, mode=mode, target_abs_GHz=1e-3)
                sq.set_units("MHz")
                rep_mhz = sq.Transmon(
                    EJ=15000.0, EC=300.0, ng=0.0, ncut=5, truncated_dim=6
                ).check_convergence(n_levels=4, mode=mode, target_abs_GHz=1e-3)
        finally:
            sq.set_units(original_units)
        err_ghz = [v.abs_err_est_GHz for v in rep_ghz.per_level]
        err_mhz = [v.abs_err_est_GHz for v in rep_mhz.per_level]
        assert err_ghz and all(e is not None for e in err_ghz)
        assert all(e is not None for e in err_mhz)
        np.testing.assert_allclose(err_ghz, err_mhz, rtol=1e-9)
        # the verdict must not depend on the unit system either
        assert rep_ghz.aggregate_status == rep_mhz.aggregate_status

    def test_eps_gap_is_unit_invariant_when_floor_binds(self):
        # observed-gap-scale eps = err / gap, with a large g_floor_GHz that floors
        # the isolation gap. The GHz floor must be rescaled into the active units
        # so eps stays unit-invariant even when the floor binds (regression for
        # the floor-rescaling raised in #316 review).
        original_units = sq.get_units()
        kw = dict(
            n_levels=4,
            mode="moderate",
            scope="observed_gap_scale",
            g_floor_GHz=10.0,
            target_gap_rel=1e-3,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sq.set_units("GHz")
                rep_ghz = sq.Transmon(
                    EJ=15.0, EC=0.3, ng=0.0, ncut=5, truncated_dim=6
                ).check_convergence(**kw)
                sq.set_units("MHz")
                rep_mhz = sq.Transmon(
                    EJ=15000.0, EC=300.0, ng=0.0, ncut=5, truncated_dim=6
                ).check_convergence(**kw)
        finally:
            sq.set_units(original_units)
        eps_ghz = [
            v.eps_gap_est for v in rep_ghz.per_level if v.eps_gap_est is not None
        ]
        eps_mhz = [
            v.eps_gap_est for v in rep_mhz.per_level if v.eps_gap_est is not None
        ]
        assert eps_ghz
        np.testing.assert_allclose(eps_ghz, eps_mhz, rtol=1e-9)

    def test_absolute_scope_without_target_returns_unverified(self):
        # The report still contains abs_err_est_GHz per level, but status
        # defaults to unverified when no target is supplied.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(tmon, n_levels=3, mode="moderate")
        for v in report.per_level:
            assert v.status == "unverified"
            assert v.abs_err_est_GHz is not None
        assert any("target_abs_GHz" in r for r in report.recommendations)

    def test_observed_gap_scope_uses_local_isolation(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=4, mode="moderate", scope="observed_gap_scale"
        )
        # eps_gap_est is populated for every level (buffer ensures
        # upper-gap is available for the topmost requested level).
        for v in report.per_level:
            assert v.eps_gap_est is not None
            assert v.eps_gap_est >= 0.0
        # The audit records the buffer.
        assert report.implementation_audit.n_levels_buffer == 1

    def test_per_level_count_matches_request(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=4, mode="moderate", target_abs_GHz=1e-4
        )
        assert len(report.per_level) == 4

    def test_transition_err_is_triangle_inequality(self):
        # Each level reports transition-error estimates to the other levels,
        # equal to the sum of the two absolute estimates (triangle inequality).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=4, mode="moderate", target_abs_GHz=1e-4
        )
        err = [v.abs_err_est_GHz for v in report.per_level]
        v0 = report.per_level[0]
        # Keys are (0, j) for every other level j.
        assert set(v0.transition_err_est_GHz) == {(0, 1), (0, 2), (0, 3)}
        for (i, j), te in v0.transition_err_est_GHz.items():
            assert te == pytest.approx(err[i] + err[j])

    def test_recommendation_is_channel_specific(self):
        # An undersized transmon's dominant channel is the charge tail, so the
        # recommendation must name the charge cutoff (not a generic axis bump).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=4, mode="moderate", target_abs_GHz=1e-6
        )
        assert report.aggregate_status == "distrust"
        assert any("charge" in r and "ncut" in r for r in report.recommendations)

    def test_boundary_probability_large_warning_in_moderate_mode(self):
        # At a small ncut the higher levels reach the charge boundary: they must
        # carry the boundary_probability_large warning (their dropped tail is
        # non-perturbative), and a recommendation must call it out.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=5, truncated_dim=6)
        report = sq.check_convergence(tmon, n_levels=5, target_abs_GHz=1e-6)
        assert any("boundary_probability_large" in v.warnings for v in report.per_level)
        assert any("boundary_probability_large" in r for r in report.recommendations)


# ------------------------------------------------------------- strict-mode tests


class TestEnergyStrictMode:
    def test_strict_mode_ratio_test_for_underconverged_input(self):
        # At small ncut the spectrum is far from converged; strict mode
        # should still produce a coherent report (the ratio test may or
        # may not detect asymptoticity at this small a problem).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=8, truncated_dim=4)
        report = sq.check_convergence(
            tmon, n_levels=3, mode="strict", target_abs_GHz=1e-6
        )
        assert report.implementation_audit.mode == "strict"
        assert report.implementation_audit.refinement == "ratio_test"
        # The estimator method records whether the ratio test succeeded.
        methods = {v.estimator_method for v in report.per_level}
        assert methods.issubset(
            {
                "ratio_test",
                "ratio_test_failed_fallback_one_step",
                "ratio_test_noise_floor",
                "one_step",
            }
        )

    def test_strict_mode_asymptotic_ratio_test_is_likely_converged(self):
        # A well-converged transmon in strict mode must reach 'likely_converged'
        # (the best verdict strict mode can return). Regression guard: an earlier
        # inversion mislabeled the asymptotic ratio test, so the strict-mode gate
        # wrongly downgraded every ratio-tested level to 'marginal'.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=16, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=4, mode="strict", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "likely_converged"
        for v in report.per_level:
            assert v.status == "likely_converged"


# ---------------------------------------------------------------- cheap-mode tests


class TestEnergyCheapMode:
    def test_cheap_mode_never_says_converged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(tmon, n_levels=5, mode="cheap")
        # Best possible cheap-mode status is "unverified".
        for v in report.per_level:
            assert v.status in {
                "marginal",
                "distrust",
                "unverified",
            }
            assert v.status != "maybe_converged"
            assert v.status != "likely_converged"

    def test_cheap_mode_well_converged_is_unverified(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(tmon, n_levels=4, mode="cheap")
        assert report.aggregate_status == "unverified"
        # The charge finite-tail gives a perturbative estimate (not just a
        # boundary diagnostic), so an abs_err_est is provided even in cheap mode.
        for v in report.per_level:
            assert v.estimator_method == "finite_tail_resolvent"
            assert v.abs_err_est_GHz is not None

    def test_cheap_mode_perturbative_estimate_tracks_true_error(self):
        # The perturbative tail estimate should grade an undersized transmon as
        # distrust for a tight target, and a level's estimate must be
        # comparable to the true charge-truncation error.
        ref = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=200).eigenvals(evals_count=5)
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=5, mode="cheap", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "distrust"
        e_user = tmon.eigenvals(evals_count=5)
        # The worst-level estimate is within a small factor of the true error
        # (the reported estimate includes the safety factor, so true/est ~ 0.5).
        v = report.per_level[report.worst_level]
        true_err = abs(float(e_user[v.level_index] - ref[v.level_index]))
        assert 0.2 <= true_err / v.abs_err_est_GHz <= 2.0

    def test_cheap_mode_undersized_is_distrust(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=4, truncated_dim=4)
        report = sq.check_convergence(tmon, n_levels=3, mode="cheap")
        assert report.aggregate_status == "distrust"
        # The dominant levels sit hard against the charge boundary.
        assert any("boundary_probability_large" in v.warnings for v in report.per_level)

    def test_cheap_mode_has_no_extra_diagonalization(self):
        # The audit field n_levels_buffer is 0 for cheap mode at
        # absolute scope (no buffer needed).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(tmon, n_levels=3, mode="cheap")
        assert report.implementation_audit.mode == "cheap"
        assert report.implementation_audit.n_levels_buffer == 0


# ---------------------------------------------------------------- cluster tests


class TestMonotonicityCheck:
    """Nested-basis variational monotonicity falsification check."""

    def test_assign_level_statuses_forces_distrust(self):
        # A flagged level is dismissed to distrust regardless of how small its
        # error estimate is; an unflagged level keeps its (passing) verdict.
        statuses = _assign_level_statuses(
            n_levels=2,
            per_level_abs_err=np.array([1e-9, 1e-9]),
            eps_gap_est=[None, None],
            per_level_warnings=[[], []],
            scope="absolute",
            target_abs_GHz=1e-3,
            target_gap_rel=1e-3,
            mode="strict",
            non_asymptotic=[False, False],
            monotonicity_violation=[True, False],
        )
        assert statuses == ["distrust", "likely_converged"]

    def test_no_false_alarm_on_converged_transmon(self):
        tmon = sq.Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30)
        report = tmon.check_convergence(n_levels=5, mode="strict", target_abs_GHz=1e-3)
        assert all("monotonicity_violation" not in v.warnings for v in report.per_level)

    def test_ho_basis_axis_is_excluded(self):
        # A harmonic-oscillator basis is not a nested Galerkin truncation (its
        # potential is built from matrix functions of truncated quadratures), so
        # it is excluded from the check even at a low, non-monotone cutoff.
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=14)
        report = flx.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=0.05)
        assert all("monotonicity_violation" not in v.warnings for v in report.per_level)

    def test_no_false_alarm_on_tensored_charge_axis(self):
        # A flat Circuit's periodic-charge axis (cutoff_n_*) is a nested charge
        # basis tensored with the other, fixed coordinates; growing it lowers the
        # energies and must not raise a monotonicity false alarm.
        circ = _zero_pi_circuit(cutoff_n=9, cutoff_ext=8)
        report = circ.check_convergence(
            n_levels=4, mode="moderate", target_abs_GHz=1e-3
        )
        assert all("monotonicity_violation" not in v.warnings for v in report.per_level)

    def test_violation_propagates_to_report(self, monkeypatch):
        # Force the utility to flag the first cluster; the full pipeline should
        # surface the warning, dismiss the level, and emit the dedicated advice.
        import scqubits.utils.convergence_utils as cutils

        def fake(coarse, fine, clusters, rel_tol, floor):
            flags = np.zeros(len(clusters), dtype=np.bool_)
            if len(clusters):
                flags[0] = True
            return flags

        monkeypatch.setattr(cutils, "galerkin_monotonicity_violation", fake)
        tmon = sq.Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30)
        report = tmon.check_convergence(
            n_levels=3, mode="moderate", target_abs_GHz=1e-3
        )
        assert report.per_level[0].status == "distrust"
        assert "monotonicity_violation" in report.per_level[0].warnings
        assert any("variational bound" in rec for rec in report.recommendations)


class TestCheckOutcomes:
    """Structured per-check record (LevelVerdict.checks)."""

    @staticmethod
    def _by_name(verdict):
        return {check.name: check.status for check in verdict.checks}

    def test_moderate_records_boundary_and_monotonicity(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=3, mode="moderate", target_abs_GHz=1e-4
        )
        names = self._by_name(report.level(0))
        # The strict-only asymptoticity test is recorded as not applicable.
        assert names.get("asymptoticity") == "not_applicable"
        assert names.get("boundary") == "pass"
        assert names.get("monotonicity") == "pass"

    def test_strict_runs_asymptoticity_when_there_is_movement(self):
        # Under-resolved: the refinement differences sit above the noise floor,
        # so the ratio test actually runs (and here passes -- the series is
        # contracting, even though the level fails the target grading).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        report = tmon.check_convergence(n_levels=3, mode="strict", target_abs_GHz=1e-6)
        assert self._by_name(report.level(2)).get("asymptoticity") in ("pass", "fail")

    def test_cheap_records_perturbative_tail(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(n_levels=3, mode="cheap", target_abs_GHz=1e-4)
        assert "perturbative_tail" in self._by_name(report.level(0))

    def test_ho_basis_omits_monotonicity_check(self):
        # Fluxonium's HO basis is not a nested charge basis, so the monotonicity
        # check does not apply and is omitted (not reported as not_applicable).
        flx = sq.Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=6
        )
        report = flx.check_convergence(n_levels=3, mode="moderate", target_abs_GHz=1e-4)
        assert "monotonicity" not in self._by_name(report.level(0))

    def test_monotonicity_violation_recorded_as_fail(self, monkeypatch):
        import scqubits.utils.convergence_utils as cutils

        def fake(coarse, fine, clusters, rel_tol, floor):
            flags = np.zeros(len(clusters), dtype=np.bool_)
            if len(clusters):
                flags[0] = True
            return flags

        monkeypatch.setattr(cutils, "galerkin_monotonicity_violation", fake)
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=3, mode="moderate", target_abs_GHz=1e-4
        )
        assert self._by_name(report.level(0)).get("monotonicity") == "fail"

    def test_summary_table_omits_eps_gap_and_shows_checks(self):
        # The aligned table drops the inapplicable eps_gap field, and shows the
        # per-check record beneath a dismissed/borderline level.
        small = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        text = small.check_convergence(
            n_levels=3, mode="moderate", target_abs_GHz=1e-6
        ).summary()
        assert "eps_gap" not in text
        assert "lvl" in text  # table header
        assert "checks" in text  # per-check line under a dismissed/marginal level


class TestClusterIntegration:
    def test_clusters_field_partitions_levels(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.check_convergence(
            tmon, n_levels=4, mode="moderate", target_abs_GHz=1e-4
        )
        flattened = [k for c in report.clusters for k in c]
        assert sorted(flattened) == list(range(4))


# --------------------------------------------------------- derived-channel tests


class TestDerivedChannels:
    def test_wavefunctions_converged_at_high_cutoff(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=4,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        wf = report.derived["wavefunctions"]
        assert wf.aggregate_status == "maybe_converged"
        assert len(wf.per_level) == 4
        for v in wf.per_level:
            # Derived metrics report rel_change_est (dimensionless), not the
            # gap-normalized eps_gap_est, and carry no GHz error estimate.
            assert v.rel_change_est is not None
            assert v.rel_change_est < 1e-3
            assert v.eps_gap_est is None
            assert v.abs_err_est_GHz is None
            assert v.estimator_method == "wavefunction_overlap"

    def test_matrix_elements_converged_at_high_cutoff(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=4,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["matrix_elements"],
        )
        me = report.derived["matrix_elements"]
        assert me.aggregate_status == "maybe_converged"
        for v in me.per_level:
            assert v.rel_change_est is not None
            assert v.eps_gap_est is None
            assert v.estimator_method == "matrix_element_frobenius"

    def test_both_channels_attached_together(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=3,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements"],
        )
        assert set(report.derived.keys()) == {"wavefunctions", "matrix_elements"}

    def test_wavefunction_refinement_diff_shrinks_with_cutoff(self):
        # The overlap deficit at a larger base cutoff is no larger than at a
        # clearly undersized one.
        small = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=5, truncated_dim=4)
        large = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=25, truncated_dim=4)
        rep_s = small.check_convergence(
            n_levels=3,
            mode="moderate",
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        rep_l = large.check_convergence(
            n_levels=3,
            mode="moderate",
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        worst_s = max(
            v.rel_change_est for v in rep_s.derived["wavefunctions"].per_level
        )
        worst_l = max(
            v.rel_change_est for v in rep_l.derived["wavefunctions"].per_level
        )
        assert worst_l <= worst_s + 1e-12

    def test_strict_mode_derived_runs_ratio_test(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=3,
            mode="strict",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        wf = report.derived["wavefunctions"]
        # Strict mode runs the ratio test; the estimator method records either
        # the successful ratio test or its one-step fallback.
        for v in wf.per_level:
            assert "ratio_test" in v.estimator_method


# -------------------------------------------------------------- coherence tests


class TestCoherenceChannel:
    def test_coherence_report_is_per_channel(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=4,
            mode="moderate",
            include_derived=True,
            derived_quantities=["coherence"],
        )
        co = report.derived["coherence"]
        methods = {v.estimator_method for v in co.per_level}
        # The qubit's effective noise channels plus the aggregate t1/t2 rates.
        assert "t1_effective_rate" in methods
        assert "t2_effective_rate" in methods
        assert "t1_capacitive_rate" in methods
        for v in co.per_level:
            assert v.estimator_method.endswith("_rate")
            # Coherence is a rate metric, not an energy: rel_change_est, not
            # eps_gap_est or abs_err_est_GHz.
            assert v.rel_change_est is not None
            assert v.eps_gap_est is None
            assert v.abs_err_est_GHz is None

    def test_coherence_converged_at_high_cutoff(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=4,
            mode="moderate",
            include_derived=True,
            derived_quantities=["coherence"],
        )
        assert report.derived["coherence"].aggregate_status == "maybe_converged"

    def test_symmetric_zero_channel_flagged_noise_floor(self):
        # At ng=0 the 1/f charge-noise dephasing rate vanishes by symmetry, so
        # its rate sits at the noise floor while a real channel does not.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=3,
            mode="moderate",
            include_derived=True,
            derived_quantities=["coherence"],
        )
        by_method = {
            v.estimator_method: v for v in report.derived["coherence"].per_level
        }
        assert "noise_floor" in by_method["tphi_1_over_f_ng_rate"].warnings
        assert "noise_floor" not in by_method["t1_capacitive_rate"].warnings

    def test_all_three_derived_channels_together(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.check_convergence(
            n_levels=3,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert set(report.derived.keys()) == {
            "wavefunctions",
            "matrix_elements",
            "coherence",
        }


# ------------------------------------------------------- TunableTransmon


class TestTunableTransmon:
    def test_inherits_all_channels(self):
        # TunableTransmon subclasses Transmon, so it inherits the charge-basis
        # convergence machinery (ncut axis) for all four channels.
        tt = sq.TunableTransmon(
            EJmax=30.0, EC=0.3, d=0.1, flux=0.1, ng=0.0, ncut=31, truncated_dim=6
        )
        report = tt.check_convergence(
            n_levels=4,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert report.aggregate_status == "maybe_converged"
        assert set(report.derived.keys()) == {
            "wavefunctions",
            "matrix_elements",
            "coherence",
        }
        for sub in report.derived.values():
            assert sub.aggregate_status == "maybe_converged"
        # The audit identifies the concrete subclass and its charge cutoff.
        assert report.implementation_audit.qubit_class == "TunableTransmon"
        assert report.implementation_audit.cutoff_parameters == {"ncut": 31}

    def test_undersized_is_not_converged(self):
        tt = sq.TunableTransmon(
            EJmax=30.0, EC=0.3, d=0.1, flux=0.3, ng=0.0, ncut=5, truncated_dim=6
        )
        report = tt.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-8)
        assert report.aggregate_status in {"marginal", "distrust"}


# ------------------------------------------------------------- Fluxonium


class TestFluxonium:
    def test_all_channels_converge(self):
        # Fluxonium uses a harmonic-oscillator (Fock) basis controlled by cutoff.
        flx = sq.Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=6
        )
        report = flx.check_convergence(
            n_levels=5,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert report.aggregate_status == "maybe_converged"
        # The truncation channel is the HO (Fock) tail, not a charge tail.
        assert report.per_level[0].truncation_channel == "HO_tail"
        assert report.implementation_audit.qubit_class == "Fluxonium"
        assert report.implementation_audit.cutoff_parameters == {"cutoff": 110}
        for sub in report.derived.values():
            assert sub.aggregate_status == "maybe_converged"

    def test_undersized_is_underconverged(self):
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=12, truncated_dim=6)
        report = flx.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-8)
        assert report.aggregate_status in {"marginal", "distrust"}
        assert any("cutoff" in r for r in report.recommendations)

    def test_pad_eigenvectors_appends_high_fock_zeros(self):
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=20, truncated_dim=6)
        evecs = np.zeros((20, 2), dtype=np.float64)
        evecs[0, 0] = 1.0
        evecs[5, 1] = 1.0
        padded = flx._convergence_pad_eigenvectors(evecs, 20, 30)
        assert padded.shape == (30, 2)
        # Existing Fock amplitudes are preserved; the added high-Fock rows are 0.
        assert padded[0, 0] == 1.0
        assert padded[5, 1] == 1.0
        assert np.all(padded[20:, :] == 0.0)

    def test_near_degenerate_doublet_warns_cluster_ambiguity(self):
        # At half flux with a deep double well the lowest two levels form a
        # near-degenerate doublet; they must be grouped into one cluster and
        # carry the cluster_index_ambiguity warning (their labels are not
        # individually reliable).
        flx = sq.Fluxonium(
            EJ=8.9, EC=2.5, EL=0.2, flux=0.5, cutoff=110, truncated_dim=8
        )
        report = flx.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-4)
        assert (0, 1) in report.clusters  # the doublet is one cluster
        for k in (0, 1):
            assert "cluster_index_ambiguity" in report.per_level[k].warnings

    def test_cheap_mode_uses_finite_window_perturbative(self):
        # Fluxonium cheap mode reports the finite-window block-resolvent estimate
        # (a perturbative abs_err_est), not a bare boundary band.
        flx = sq.Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=6
        )
        report = flx.check_convergence(n_levels=4, mode="cheap", target_abs_GHz=1e-4)
        assert report.aggregate_status == "unverified"
        for v in report.per_level:
            assert v.estimator_method == "finite_tail_resolvent"
            assert v.abs_err_est_GHz is not None

    def test_cheap_mode_undersized_is_distrust(self):
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=12, truncated_dim=6)
        report = flx.check_convergence(n_levels=4, mode="cheap", target_abs_GHz=1e-4)
        assert report.aggregate_status == "distrust"


# ------------------------------------------------------------- FluxQubit


class TestFluxQubit:
    @staticmethod
    def _make(ncut):
        ej, alpha = 35.0, 0.6
        return sq.FluxQubit(
            EJ1=ej,
            EJ2=ej,
            EJ3=alpha * ej,
            ECJ1=1.0,
            ECJ2=1.0,
            ECJ3=1.0 / alpha,
            ECg1=50.0,
            ECg2=50.0,
            ng1=0.0,
            ng2=0.0,
            flux=0.5,
            ncut=ncut,
            truncated_dim=6,
        )

    def test_all_channels_converge(self):
        # FluxQubit uses a two-island charge basis of dimension (2*ncut+1)**2.
        fq = self._make(14)
        report = fq.check_convergence(
            n_levels=4,
            mode="moderate",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert report.aggregate_status == "maybe_converged"
        assert report.per_level[0].truncation_channel == "charge_tail"
        assert report.implementation_audit.qubit_class == "FluxQubit"
        assert report.implementation_audit.cutoff_parameters == {"ncut": 14}
        for sub in report.derived.values():
            assert sub.aggregate_status == "maybe_converged"

    def test_undersized_is_underconverged(self):
        fq = self._make(4)
        report = fq.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-8)
        assert report.aggregate_status in {"marginal", "distrust"}

    def test_pad_eigenvectors_pads_both_charge_axes(self):
        fq = self._make(2)  # 5x5 charge grid, flattened to length 25
        evecs = np.zeros((25, 1), dtype=np.complex128)
        evecs[12, 0] = 1.0  # center state |n1=0, n2=0| (row 2, col 2)
        padded = fq._convergence_pad_eigenvectors(evecs, 2, 4)  # to 9x9 grid
        assert padded.shape == (81, 1)
        # The center maps to (row 4, col 4) -> flat index 40; nothing else.
        assert padded[40, 0] == 1.0
        assert abs(complex(padded.sum()) - 1.0) < 1e-12


# ---------------------------------------------------------------- ZeroPi


class TestZeroPi:
    @staticmethod
    def _make(window=6 * np.pi, pts=120, ncut=12):
        grid = sq.Grid1d(-window, window, pts)
        return sq.ZeroPi(
            grid=grid,
            EJ=10.0,
            EL=0.04,
            ECJ=20.0,
            EC=0.04,
            ng=0.1,
            flux=0.23,
            ncut=ncut,
            truncated_dim=6,
        )

    def test_box_refinement_sized_to_turning_point(self):
        # The grid_box refinement is widened to enclose the classically allowed
        # region for the computed window (its outermost turning points), so a
        # too-small box cannot be silently accepted; a non-box axis keeps the
        # ordinary step.
        zp = self._make()
        evals = zp.eigenvals(evals_count=4)
        current = zp._convergence_axis_value("grid_box")
        step = zp._convergence_box_refine_step("grid_box", evals, 4)
        spacing = (zp.grid.max_val - zp.grid.min_val) / (zp.grid.pt_count - 1)
        widened_half = 0.5 * spacing * (current + step - 1)
        box_half = 0.5 * (zp.grid.max_val - zp.grid.min_val)
        assert widened_half > box_half
        assert zp._convergence_box_refine_step(
            "grid_spacing", evals, 4
        ) == zp._convergence_step("grid_spacing")

    def test_too_small_box_is_distrusted(self):
        # A shallow parabola makes the classical region far larger than a small
        # box; widening to the turning point reveals the truncation error, so the
        # verdict is distrust even at a loose target.
        grid = sq.Grid1d(-2 * np.pi, 2 * np.pi, 60)
        zp = sq.ZeroPi(
            grid=grid,
            EJ=10.0,
            EL=0.01,
            ECJ=20.0,
            EC=0.04,
            ng=0.1,
            flux=0.2,
            ncut=12,
            truncated_dim=6,
        )
        report = zp.check_convergence(n_levels=3, mode="moderate", target_abs_GHz=1e-2)
        assert report.aggregate_status == "distrust"

    def test_two_fd_channels_plus_charge(self):
        # ZeroPi's phi grid contributes two independent FD channels (finite box
        # and finite spacing); theta contributes a charge tail.
        zp = self._make()
        report = zp.check_convergence(n_levels=3, mode="moderate", target_abs_GHz=1e-2)
        assert report.aggregate_status == "maybe_converged"
        assert set(report.channel_breakdown_GHz) == {
            "FD_box",
            "FD_stencil",
            "charge_tail",
        }
        # The per-level channel is the dominant physical channel (composite_coupling
        # is reserved for coupled-subsystem HilbertSpace truncation, not a
        # multi-coordinate single qubit); the full split is in channel_breakdown.
        assert {v.truncation_channel for v in report.per_level} <= {
            "FD_box",
            "FD_stencil",
            "charge_tail",
        }
        # All three axes are recorded in the audit.
        assert set(report.implementation_audit.cutoff_parameters) == {
            "grid_box",
            "grid_spacing",
            "ncut",
        }

    def test_small_box_diagnosed_by_FD_box_not_FD_stencil(self):
        # A snug phi window with plenty of points: the box is too small but the
        # spacing is fine. The design spec's point is that adding grid points at
        # a fixed window cannot fix a box error, so the FD_box channel must
        # dominate the FD_stencil channel.
        snug = self._make(window=np.pi, pts=100, ncut=12)
        report = snug.check_convergence(
            n_levels=3, mode="moderate", target_abs_GHz=1e-3
        )
        assert report.aggregate_status == "distrust"
        cb = report.channel_breakdown_GHz
        assert cb["FD_box"] > cb["FD_stencil"]
        # The recommendation must say to enlarge the box, not just add points.
        assert any("box" in r for r in report.recommendations)

    def test_cheap_mode_uses_pedge_and_never_converges(self):
        zp = self._make()
        report = zp.check_convergence(n_levels=3, mode="cheap")
        # Cheap mode reports the FD_box edge-band (P_edge) diagnostic ...
        assert {v.truncation_channel for v in report.per_level} == {"FD_box"}
        # ... and never claims an unqualified 'maybe_converged' or 'likely_converged'.
        for v in report.per_level:
            assert v.status not in {"maybe_converged", "likely_converged"}

    def test_grid_box_grows_window_at_fixed_spacing(self):
        # grid_box must enlarge the phi window while holding the spacing h fixed;
        # grid_spacing must keep the window and shrink h.
        zp = self._make(window=6 * np.pi, pts=101, ncut=4)
        h0 = (zp.grid.max_val - zp.grid.min_val) / (zp.grid.pt_count - 1)

        box_clone = zp._convergence_clone_at({"grid_box": 121})
        h_box = (box_clone.grid.max_val - box_clone.grid.min_val) / (
            box_clone.grid.pt_count - 1
        )
        assert box_clone.grid.pt_count == 121
        assert box_clone.grid.max_val > zp.grid.max_val  # window grew
        assert abs(h_box - h0) < 1e-9  # spacing held fixed

        spacing_clone = zp._convergence_clone_at({"grid_spacing": 121})
        h_spacing = (spacing_clone.grid.max_val - spacing_clone.grid.min_val) / (
            spacing_clone.grid.pt_count - 1
        )
        assert spacing_clone.grid.pt_count == 121
        assert spacing_clone.grid.max_val == zp.grid.max_val  # window unchanged
        assert h_spacing < h0  # spacing shrank

    def test_strict_mode_uses_richardson_for_grid_spacing(self):
        # The grid_spacing channel reports the FD stencil order h**(STENCIL-1);
        # strict mode must therefore verify it with Richardson extrapolation,
        # not the geometric ratio test. The composite estimator method records
        # this, and a well-sized ZeroPi reaches 'likely_converged'.
        import scqubits.settings as settings

        zp = self._make()
        assert zp._convergence_richardson_order("grid_spacing") == settings.STENCIL - 1
        assert zp._convergence_richardson_order("grid_box") is None
        report = zp.check_convergence(n_levels=3, mode="strict", target_abs_GHz=1e-2)
        assert report.aggregate_status == "likely_converged"
        for v in report.per_level:
            assert v.status == "likely_converged"
            assert v.estimator_method.startswith("richardson_composite")


# ------------------------------------------------------- parameter-sweep coverage


class TestParamSweep:
    def test_sweep_samples_endpoints_and_restores_param(self):
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.0, cutoff=30, truncated_dim=6)
        sweep = flx.check_convergence_vs_paramvals(
            "flux",
            np.linspace(0.0, 0.5, 21),
            sample=5,
            n_levels=3,
            target_abs_GHz=1e-3,
        )
        # Five points were sampled, including both endpoints.
        assert len(sweep.param_vals) == 5
        assert sweep.param_vals[0] == 0.0
        assert sweep.param_vals[-1] == 0.5
        assert len(sweep.reports) == 5
        # The swept parameter is restored after the sweep.
        assert flx.flux == 0.0
        # Worst-case accessors are self-consistent.
        assert sweep.worst_param_val() == sweep.param_vals[sweep.worst_index]
        assert sweep.worst_report() is sweep.reports[sweep.worst_index]
        # The reported aggregate is the worst per-point aggregate.
        assert (
            sweep.aggregate_status == sweep.reports[sweep.worst_index].aggregate_status
        )

    def test_sweep_summary_and_str_delegate(self):
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.0, cutoff=30, truncated_dim=6)
        sweep = flx.check_convergence_vs_paramvals(
            "flux", np.linspace(0.0, 0.5, 11), sample=3, n_levels=3, target_abs_GHz=1e-3
        )
        text = sweep.summary()
        assert "convergence across sweep of (flux)" in text
        assert str(sweep) == text

    def test_sweep_none_checks_every_value(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        sweep = tmon.check_convergence_vs_paramvals(
            "ng",
            np.array([0.0, 0.25, 0.5]),
            sample=None,
            n_levels=3,
            target_abs_GHz=1e-4,
        )
        assert len(sweep.param_vals) == 3

    def test_top_level_sweep_shim(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        sweep = sq.check_convergence_vs_paramvals(
            tmon, "ng", np.array([0.0, 0.5]), n_levels=2, target_abs_GHz=1e-4
        )
        assert sweep.param_name == "ng"
        assert len(sweep.param_vals) == 2


# ----------------------------------------------------- HilbertSpace (composite)


def _transmon_resonator(td_t=6, td_o=6, ncut=31, g=0.1):
    """Build a transmon-resonator HilbertSpace with a number-charge coupling."""
    tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=ncut, truncated_dim=td_t)
    osc = sq.Oscillator(E_osc=5.0, truncated_dim=td_o)
    hs = sq.HilbertSpace([tmon, osc])
    hs.add_interaction(g=g, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True)
    return hs, tmon, osc


class TestHilbertSpaceConvergence:
    def test_well_converged_composite_is_maybe_converged(self):
        hs, _, _ = _transmon_resonator(td_t=6, td_o=6)
        report = hs.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-3)
        assert report.aggregate_status == "maybe_converged"
        # The composite truncation is reported on the composite_coupling channel.
        assert {v.truncation_channel for v in report.per_level} == {
            "composite_coupling"
        }
        assert report.implementation_audit.mode == "moderate"

    def test_layer1_subsystem_reports_attached(self):
        # Layer 1 attaches each capable subsystem's own report under derived; the
        # oscillator has no internal cutoff and is skipped.
        hs, _, _ = _transmon_resonator(td_t=6, td_o=6)
        report = hs.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-3)
        assert report.derived is not None
        keys = list(report.derived)
        assert len(keys) == 1
        assert keys[0].startswith("subsystem:")
        assert "Transmon" in keys[0]

    def test_assume_subsystems_converged_skips_layer1(self):
        hs, _, _ = _transmon_resonator(td_t=6, td_o=6)
        report = hs.check_convergence(
            n_levels=5,
            mode="moderate",
            target_abs_GHz=1e-3,
            assume_subsystems_converged=True,
        )
        assert report.derived is None
        assert report.aggregate_status == "maybe_converged"

    def test_undersized_truncated_dim_is_distrust(self):
        # A too-small resonator truncated_dim leaves the coupled spectrum
        # unconverged; the dominant axis is the oscillator and a recommendation
        # must name it.
        hs, _, _ = _transmon_resonator(td_t=6, td_o=3, g=0.6)
        report = hs.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-5)
        assert report.aggregate_status == "distrust"
        assert any("Oscillator" in r for r in report.recommendations)

    def test_aggregate_widened_by_underconverged_subsystem(self):
        # If a subsystem is underconverged at its own cutoff, the composite
        # cannot be better: the aggregate is widened to the subsystem verdict.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=5, truncated_dim=6)
        osc = sq.Oscillator(E_osc=5.0, truncated_dim=6)
        hs = sq.HilbertSpace([tmon, osc])
        hs.add_interaction(
            g=0.1, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True
        )
        report = hs.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-6)
        sub = report.derived["subsystem:" + tmon.id_str]
        assert _status_rank(sub.aggregate_status) >= _status_rank("marginal")
        assert _status_rank(report.aggregate_status) >= _status_rank(
            sub.aggregate_status
        )

    def test_cheap_mode_is_moderate_recommended(self):
        hs, _, _ = _transmon_resonator(td_t=6, td_o=6)
        report = hs.check_convergence(n_levels=5, mode="cheap")
        # No cheap composite estimate: the composite verdict is unverified and a
        # recommendation points to moderate mode. Layer-1 subsystem cheap checks
        # are still attached.
        assert report.aggregate_status == "unverified"
        assert all(
            v.truncation_channel == "composite_coupling" for v in report.per_level
        )
        assert any("moderate" in r for r in report.recommendations)
        assert report.derived is not None and len(report.derived) == 1

    def test_n_levels_exceeding_dimension_raises(self):
        hs, _, _ = _transmon_resonator(td_t=2, td_o=2)  # composite dimension 4
        with pytest.raises(ValueError, match="exceeds the composite dimension"):
            hs.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-3)

    def test_strict_mode_uses_composite_coupling_channel(self):
        hs, _, _ = _transmon_resonator(td_t=6, td_o=6)
        report = hs.check_convergence(n_levels=5, mode="strict", target_abs_GHz=1e-3)
        assert {v.truncation_channel for v in report.per_level} == {
            "composite_coupling"
        }
        assert report.implementation_audit.mode == "strict"

    def test_top_level_shim_on_hilbertspace(self):
        hs, _, _ = _transmon_resonator(td_t=6, td_o=6)
        report = sq.check_convergence(
            hs, n_levels=5, mode="moderate", target_abs_GHz=1e-3
        )
        assert report.aggregate_status == "maybe_converged"

    def test_hybridization_screen_flags_near_resonance(self):
        # Resonator tuned to the transmon 0-1 transition: the bare product states
        # hybridize. The screen flags it as a labeling issue, independent of any
        # truncation error.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        evals = tmon.eigenvals(evals_count=2)
        f01 = float(evals[1] - evals[0])
        osc = sq.Oscillator(E_osc=f01, truncated_dim=6)
        hs = sq.HilbertSpace([tmon, osc])
        hs.add_interaction(
            g=0.2, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True
        )
        report = hs.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-3)
        assert any("hybridization" in r for r in report.recommendations)

    def test_hybridization_screen_silent_off_resonance(self):
        # A resonator far below all kept transmon transitions (which span
        # ~5-6.6 GHz here), weakly coupled, does not trip the screen.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        osc = sq.Oscillator(E_osc=2.0, truncated_dim=6)
        hs = sq.HilbertSpace([tmon, osc])
        hs.add_interaction(
            g=0.01, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True
        )
        report = hs.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-3)
        assert not any("hybridization" in r for r in report.recommendations)

    def test_fixed_matrix_oscillator_interaction_degrades_gracefully(self):
        # A fixed-matrix interaction operator on an oscillator cannot be resized
        # when truncated_dim is refined; the composite check degrades to
        # unverified with an actionable recommendation rather than crashing.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=4)
        osc = sq.Oscillator(E_osc=5.0, truncated_dim=4)
        hs = sq.HilbertSpace([tmon, osc])
        hs.add_interaction(
            g=0.1,
            op1=(tmon.n_operator(), tmon),
            op2=(osc.creation_operator(), osc),
            add_hc=True,
        )
        report = hs.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-3)
        assert report.aggregate_status == "unverified"
        assert any("callable" in r for r in report.recommendations)
        assert all(
            "composite_unrefinable_interaction" in v.warnings for v in report.per_level
        )


# ------------------------------------------------------- FullZeroPi (hierarchical)


def _full_zeropi(zeropi_cutoff=8, zeta_cutoff=30, ncut=20, npts=100):
    """Build a FullZeroPi (interior ZeroPi coupled to a zeta oscillator)."""
    grid = sq.Grid1d(-6 * np.pi, 6 * np.pi, npts)
    return sq.FullZeroPi(
        EJ=10.0,
        EL=0.04,
        ECJ=20.0,
        EC=0.04,
        dEJ=0.05,
        dCJ=0.05,
        dC=0.08,
        dEL=0.05,
        flux=0.23,
        ng=0.1,
        zeropi_cutoff=zeropi_cutoff,
        zeta_cutoff=zeta_cutoff,
        grid=grid,
        ncut=ncut,
        truncated_dim=10,
    )


class TestFullZeroPi:
    def test_converged_full_zeropi(self):
        fzp = _full_zeropi()
        report = fzp.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-2)
        assert report.aggregate_status == "maybe_converged"
        # Layer-2 coupling channels only (composite_coupling / HO_tail).
        assert {v.truncation_channel for v in report.per_level} <= {
            "composite_coupling",
            "HO_tail",
        }
        # Layer-1 interior ZeroPi report is attached.
        assert "interior_zeropi" in (report.derived or {})
        assert report.derived["interior_zeropi"].aggregate_status == "maybe_converged"

    def test_undersized_interior_is_flagged(self):
        # A too-small interior basis (ncut, grid) leaves the 0-pi sector
        # underconverged; layer 1 catches it and the aggregate reflects it.
        fzp = _full_zeropi(ncut=6, npts=60)
        report = fzp.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-5)
        assert _status_rank(report.aggregate_status) >= _status_rank("marginal")
        assert "interior_zeropi" in (report.derived or {})

    def test_assume_inner_converged_skips_layer1(self):
        fzp = _full_zeropi()
        report = fzp.check_convergence(
            n_levels=4,
            mode="moderate",
            target_abs_GHz=1e-2,
            assume_inner_converged=True,
        )
        assert report.derived is None

    def test_cheap_is_moderate_recommended(self):
        fzp = _full_zeropi()
        report = fzp.check_convergence(n_levels=4, mode="cheap")
        assert report.aggregate_status == "unverified"
        assert any("moderate" in r for r in report.recommendations)
        assert "interior_zeropi" in (report.derived or {})

    def test_n_levels_exceeding_dimension_raises(self):
        fzp = _full_zeropi(zeropi_cutoff=2, zeta_cutoff=2)  # dimension 4
        with pytest.raises(ValueError, match="exceeds the FullZeroPi dimension"):
            fzp.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-2)

    def test_top_level_shim(self):
        fzp = _full_zeropi()
        report = sq.check_convergence(fzp, n_levels=4, mode="cheap")
        assert report.aggregate_status == "unverified"


# --------------------------------------------------- Cos2PhiQubit (multi-coordinate)


def _cos2phi(ncut=8, phi_cut=7, zeta_cut=10):
    """Build a Cos2PhiQubit with the given (small) truncation cutoffs."""
    params = sq.Cos2PhiQubit.default_params()
    params.update(ncut=ncut, phi_cut=phi_cut, zeta_cut=zeta_cut)
    return sq.Cos2PhiQubit(**params)


class TestCos2PhiQubit:
    def test_converged(self):
        # A flat 3-axis qubit (theta charge + phi/zeta Fock); generous cutoffs at
        # a loose target are maybe_converged on the coupling-free channels.
        q = _cos2phi(ncut=8, phi_cut=15, zeta_cut=30)
        report = q.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=0.05)
        assert report.aggregate_status == "maybe_converged"
        assert {v.truncation_channel for v in report.per_level} <= {
            "charge_tail",
            "HO_tail",
        }

    def test_undersized_oscillator_is_distrust(self):
        # Too-small oscillator cutoffs dominate; the recommendation names an
        # oscillator cutoff, and both truncation families appear in the breakdown.
        q = _cos2phi(ncut=8, phi_cut=7, zeta_cut=10)
        report = q.check_convergence(n_levels=4, mode="moderate", target_abs_GHz=1e-5)
        assert report.aggregate_status == "distrust"
        assert any(
            "oscillator" in r and ("phi_cut" in r or "zeta_cut" in r)
            for r in report.recommendations
        )
        assert "HO_tail" in report.channel_breakdown_GHz
        assert "charge_tail" in report.channel_breakdown_GHz

    def test_cheap_is_not_converged(self):
        # Multi-coordinate cheap mode has no clean cheap estimate: never
        # maybe_converged or likely_converged.
        q = _cos2phi(ncut=8, phi_cut=7, zeta_cut=10)
        report = q.check_convergence(n_levels=4, mode="cheap")
        assert report.aggregate_status not in {"maybe_converged", "likely_converged"}

    def test_top_level_shim(self):
        q = _cos2phi(ncut=8, phi_cut=7, zeta_cut=10)
        report = sq.check_convergence(q, n_levels=4, mode="cheap")
        assert report.per_level


# ----------------------------------------------------- Circuit (non-hierarchical)

_ZERO_PI_YAML = """branches:
- ["JJ", 1, 2, 10, 20]
- ["JJ", 3, 4, 10, 20]
- ["L", 2, 3, 0.008]
- ["L", 4, 1, 0.008]
- ["C", 1, 3, 0.02]
- ["C", 2, 4, 0.02]
"""


def _zero_pi_circuit(ext_basis="discretized", cutoff_n=5, cutoff_ext=8):
    """Build a flat (non-HD) zero-pi Circuit with small cutoffs for fast tests."""
    circ = sq.Circuit.from_yaml_string(_ZERO_PI_YAML, ext_basis=ext_basis)
    circ.cutoff_n_1 = cutoff_n
    circ.cutoff_ext_2 = cutoff_ext
    circ.cutoff_ext_3 = cutoff_ext
    return circ


class TestCircuit:
    def test_axes_are_cutoff_names(self):
        circ = _zero_pi_circuit()
        # Refinement axes are exactly the per-variable cutoffs (sigma var 4 has none).
        assert tuple(circ._convergence_axes) == tuple(circ.cutoff_names)
        assert "cutoff_n_1" in circ._convergence_axes

    def test_discretized_underconverged_channels_and_recommendation(self):
        # Small discretized grid: distrust, grid-spacing (FD_stencil)
        # dominates, both channel families appear, and the recommendation names a
        # cutoff_ext grid.
        circ = _zero_pi_circuit(cutoff_n=5, cutoff_ext=8)
        report = circ.check_convergence(
            n_levels=4, mode="moderate", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "distrust"
        assert {v.truncation_channel for v in report.per_level} <= {
            "charge_tail",
            "FD_stencil",
        }
        assert "FD_stencil" in report.channel_breakdown_GHz
        assert "charge_tail" in report.channel_breakdown_GHz
        assert any("cutoff_ext" in r for r in report.recommendations)

    def test_harmonic_basis_uses_ho_tail_channel(self):
        # An extended variable in the harmonic basis maps to the HO_tail channel.
        circ = _zero_pi_circuit(ext_basis="harmonic", cutoff_n=5, cutoff_ext=10)
        report = circ.check_convergence(
            n_levels=4, mode="moderate", target_abs_GHz=1e-4
        )
        assert "HO_tail" in report.channel_breakdown_GHz
        assert all(
            v.truncation_channel in {"charge_tail", "HO_tail"} for v in report.per_level
        )

    def test_cheap_is_not_converged(self):
        circ = _zero_pi_circuit(cutoff_n=5, cutoff_ext=8)
        report = circ.check_convergence(n_levels=4, mode="cheap")
        assert report.aggregate_status not in {"maybe_converged", "likely_converged"}

    def test_n_levels_exceeding_truncated_dim_raises(self):
        # A Circuit returns at most truncated_dim eigenvalues.
        circ = _zero_pi_circuit(cutoff_n=5, cutoff_ext=8)
        circ.truncated_dim = 4
        with pytest.raises(ValueError, match="truncated_dim"):
            circ.check_convergence(n_levels=5, mode="moderate", target_abs_GHz=1e-3)

    def test_top_level_shim(self):
        circ = _zero_pi_circuit(cutoff_n=5, cutoff_ext=8)
        report = sq.check_convergence(circ, n_levels=4, mode="cheap")
        assert report.per_level


# ------------------------------------------------- Circuit (hierarchical)


def _hd_zero_pi_circuit():
    """Build a hierarchically-diagonalized zero-pi Circuit (two leaf subsystems)."""
    circ = sq.Circuit.from_yaml_string(_ZERO_PI_YAML, ext_basis="discretized")
    circ.configure(
        system_hierarchy=[[1], [2, 3]],
        subsystem_trunc_dims=sq.truncation_template([[1], [2, 3]]),
    )
    return circ


class TestCircuitHD:
    def test_two_layer_report(self):
        # Layer 2: the dressed truncation is the composite_coupling channel;
        # layer 1: one report per subsystem under derived.
        circ = _hd_zero_pi_circuit()
        report = circ.check_convergence(n_levels=3, mode="moderate", target_abs_GHz=0.5)
        assert {v.truncation_channel for v in report.per_level} == {
            "composite_coupling"
        }
        assert report.derived is not None
        keys = list(report.derived)
        assert len(keys) == len(circ.subsystems)
        assert all(k.startswith("subsystem:") for k in keys)

    def test_cheap_is_moderate_recommended(self):
        circ = _hd_zero_pi_circuit()
        report = circ.check_convergence(n_levels=3, mode="cheap")
        assert report.aggregate_status == "unverified"
        assert any("moderate" in r for r in report.recommendations)
        # layer-1 subsystem cheap checks are still attached
        assert report.derived is not None

    def test_assume_subsystems_converged_skips_layer1(self):
        circ = _hd_zero_pi_circuit()
        report = circ.check_convergence(
            n_levels=3, mode="cheap", assume_subsystems_converged=True
        )
        assert report.derived is None


# ------------------------------------------------------------- ParameterSweep


def _ng_sweep(npoints=5):
    """Build a transmon-resonator ParameterSweep over the transmon offset charge."""
    tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
    osc = sq.Oscillator(E_osc=5.0, truncated_dim=6)
    hs = sq.HilbertSpace([tmon, osc])
    hs.add_interaction(
        g=0.1, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True
    )

    def update(ng):
        tmon.ng = ng

    sweep = sq.ParameterSweep(
        hs, {"ng": np.linspace(0.0, 0.5, npoints)}, update, autorun=False
    )
    return sweep, tmon


class TestParameterSweepConvergence:
    def test_single_param_worst_case(self):
        sweep, _ = _ng_sweep()
        res = sweep.check_convergence(
            n_levels=4, mode="moderate", target_abs_GHz=1e-3, sample=3
        )
        assert isinstance(res, sq.ParameterSweepConvergence)
        assert res.param_names == ("ng",)
        assert len(res.reports) == 3
        assert all(set(point) == {"ng"} for point in res.param_points)
        assert res.worst_report() is res.reports[res.worst_index]
        assert res.worst_point() == res.param_points[res.worst_index]
        assert res.aggregate_status == res.reports[res.worst_index].aggregate_status

    def test_extremes_sampled(self):
        sweep, _ = _ng_sweep(npoints=5)
        res = sweep.check_convergence(n_levels=4, mode="cheap", sample=3)
        vals = [point["ng"] for point in res.param_points]
        assert 0.0 in vals and 0.5 in vals  # both endpoints (extremes)

    def test_two_parameter_corners_included(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        osc = sq.Oscillator(E_osc=5.0, truncated_dim=6)
        hs = sq.HilbertSpace([tmon, osc])
        hs.add_interaction(
            g=0.1, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True
        )

        def update(ej, ng):
            tmon.EJ = ej
            tmon.ng = ng

        sweep = sq.ParameterSweep(
            hs,
            {"EJ": np.linspace(18.0, 22.0, 3), "ng": np.linspace(0.0, 0.5, 3)},
            update,
            autorun=False,
        )
        res = sweep.check_convergence(n_levels=4, mode="cheap", sample=6)
        assert res.param_names == ("EJ", "ng")
        corners = {(18.0, 0.0), (18.0, 0.5), (22.0, 0.0), (22.0, 0.5)}
        got = {(point["EJ"], point["ng"]) for point in res.param_points}
        assert corners <= got

    def test_sample_none_checks_every_point(self):
        sweep, _ = _ng_sweep(npoints=3)
        res = sweep.check_convergence(n_levels=4, mode="cheap", sample=None)
        assert len(res.param_points) == 3

    def test_sweep_state_restored(self):
        sweep, tmon = _ng_sweep(npoints=5)
        out_of_sync_before = sweep._out_of_sync
        sweep.check_convergence(n_levels=4, mode="cheap", sample=3)
        assert tmon.ng == 0.0  # restored to the initial grid point
        assert sweep._out_of_sync == out_of_sync_before

    def test_assume_subsystems_converged_passthrough(self):
        sweep, _ = _ng_sweep(npoints=3)
        res = sweep.check_convergence(
            n_levels=4,
            mode="moderate",
            target_abs_GHz=1e-3,
            sample=2,
            assume_subsystems_converged=True,
        )
        # the per-point HilbertSpace reports skip the layer-1 subsystem checks
        assert all(report.derived is None for report in res.reports)

    def test_summary_and_str(self):
        sweep, _ = _ng_sweep(npoints=3)
        res = sweep.check_convergence(n_levels=4, mode="cheap", sample=2)
        text = res.summary()
        assert "convergence across sweep" in text
        assert str(res) == text
