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

PR-1 scope: dataclass construction; verify / strict / quick mode energy
convergence on Transmon; quick-mode degradation; recommendations; the
top-level ``estimate_convergence`` shim.
"""

from __future__ import annotations

import dataclasses

import pytest

import scqubits as sq

from scqubits.core.convergence_report import (
    EVIDENCE_ORDER,
    ConvergenceReport,
    ImplementationAudit,
    LevelVerdict,
    evidence_at_least,
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
            mode="verify",
            refinement="one_step",
        )

    def test_level_verdict_is_frozen(self):
        verdict = LevelVerdict(
            level_index=0,
            status="converged",
            status_scope="absolute",
            evidence="verified_empirical",
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
            status="converged",
            status_scope="absolute",
            evidence="verified_empirical",
            abs_err_est_GHz=1e-10,
            eps_gap_est=None,
        )
        report = ConvergenceReport(
            per_level=[verdict],
            aggregate_status="converged",
            worst_level=0,
            channel_breakdown_GHz={"charge": 1e-10},
            clusters=[(0,)],
            recommendations=[],
            implementation_audit=self._make_audit(),
        )
        d = dataclasses.asdict(report)
        assert d["aggregate_status"] == "converged"
        assert d["per_level"][0]["level_index"] == 0
        assert d["implementation_audit"]["mode"] == "verify"

    def test_evidence_ordering(self):
        # certified is the strongest; unverified the weakest.
        assert evidence_at_least("certified", "unverified")
        assert evidence_at_least("verified_empirical", "calibrated")
        assert not evidence_at_least("diagnostic", "calibrated")
        # Sanity: the order tuple matches the docstring.
        assert EVIDENCE_ORDER[0] == "certified"
        assert EVIDENCE_ORDER[-1] == "unverified"


# ---------------------------------------------------------------- API validation


class TestAPIValidation:
    def test_estimate_convergence_on_non_checkable_raises(self):
        # Oscillator is not (yet) ConvergenceCheckable.
        osc = sq.Oscillator(E_osc=5.0, truncated_dim=3)
        with pytest.raises(TypeError, match="does not implement convergence checking"):
            sq.estimate_convergence(osc, n_levels=2, mode="verify")

    def test_invalid_mode_raises(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="mode must be"):
            tmon.estimate_convergence(n_levels=3, mode="turbo")

    def test_invalid_scope_raises(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="scope must be"):
            tmon.estimate_convergence(n_levels=3, scope="LC_scale")

    def test_include_derived_requires_derived_quantities(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="requires derived_quantities"):
            tmon.estimate_convergence(n_levels=3, include_derived=True)

    def test_include_derived_rejects_quick_mode(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="verify"):
            tmon.estimate_convergence(
                n_levels=3,
                mode="quick",
                include_derived=True,
                derived_quantities=["wavefunctions"],
            )

    def test_unknown_derived_quantity_raises(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(ValueError, match="unknown derived_quantities"):
            tmon.estimate_convergence(
                n_levels=3, include_derived=True, derived_quantities=["bogus"]
            )

    def test_coherence_derived_not_implemented(self):
        tmon = sq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        with pytest.raises(NotImplementedError, match="coherence"):
            tmon.estimate_convergence(
                n_levels=3, include_derived=True, derived_quantities=["coherence"]
            )


# ------------------------------------------------------------- verify-mode tests


class TestEnergyVerifyMode:
    def test_well_converged_transmon_is_converged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=5, mode="verify", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "converged"
        # All five levels should be converged.
        statuses = {v.status for v in report.per_level}
        assert statuses == {"converged"}
        # Evidence should be verified_empirical (one-step refinement).
        evidences = {v.evidence for v in report.per_level}
        assert evidences == {"verified_empirical"}
        # The audit records the requested mode and the cutoff parameter.
        assert report.implementation_audit.mode == "verify"
        assert report.implementation_audit.cutoff_parameters["ncut"] == 31

    def test_undersized_transmon_is_underconverged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=5, mode="verify", target_abs_GHz=1e-6
        )
        assert report.aggregate_status == "underconverged"
        # Recommendation should propose increasing ncut.
        assert any("ncut" in r for r in report.recommendations)

    def test_absolute_scope_without_target_returns_unverified(self):
        # The report still contains abs_err_est_GHz per level, but status
        # defaults to unverified when no target is supplied.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(tmon, n_levels=3, mode="verify")
        for v in report.per_level:
            assert v.status == "unverified"
            assert v.abs_err_est_GHz is not None
        assert any("target_abs_GHz" in r for r in report.recommendations)

    def test_observed_gap_scope_uses_local_isolation(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=4, mode="verify", scope="observed_gap_scale"
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
        report = sq.estimate_convergence(
            tmon, n_levels=4, mode="verify", target_abs_GHz=1e-4
        )
        assert len(report.per_level) == 4


# ------------------------------------------------------------- strict-mode tests


class TestEnergyStrictMode:
    def test_strict_mode_ratio_test_for_underconverged_input(self):
        # At small ncut the spectrum is far from converged; strict mode
        # should still produce a coherent report (the ratio test may or
        # may not detect asymptoticity at this small a problem).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=8, truncated_dim=4)
        report = sq.estimate_convergence(
            tmon, n_levels=3, mode="strict", target_abs_GHz=1e-6
        )
        assert report.implementation_audit.mode == "strict"
        assert report.implementation_audit.refinement == "ratio_test"
        # The estimator method records whether the ratio test succeeded.
        methods = {v.estimator_method for v in report.per_level}
        assert methods.issubset(
            {"ratio_test", "ratio_test_failed_fallback_one_step", "one_step"}
        )


# --------------------------------------------------------------- quick-mode tests


class TestEnergyQuickMode:
    def test_quick_mode_never_says_converged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(tmon, n_levels=5, mode="quick")
        # Best possible quick-mode status is likely_converged.
        for v in report.per_level:
            assert v.status in {
                "likely_converged",
                "marginal",
                "underconverged",
                "unverified",
            }
            assert v.status != "converged"

    def test_quick_mode_well_converged_is_likely_converged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(tmon, n_levels=4, mode="quick")
        assert report.aggregate_status == "likely_converged"
        # No abs_err_est is provided in quick mode.
        for v in report.per_level:
            assert v.abs_err_est_GHz is None

    def test_quick_mode_undersized_is_unverified(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=4, truncated_dim=4)
        report = sq.estimate_convergence(tmon, n_levels=3, mode="quick")
        assert report.aggregate_status == "unverified"
        # Each underconverged level carries the relevant warning.
        for v in report.per_level:
            assert "boundary_amplitude_above_threshold" in v.warnings

    def test_quick_mode_has_no_extra_diagonalization(self):
        # The audit field n_levels_buffer is 0 for quick mode at
        # absolute scope (no buffer needed).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(tmon, n_levels=3, mode="quick")
        assert report.implementation_audit.mode == "quick"
        assert report.implementation_audit.n_levels_buffer == 0


# ---------------------------------------------------------------- cluster tests


class TestClusterIntegration:
    def test_clusters_field_partitions_levels(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=4, mode="verify", target_abs_GHz=1e-4
        )
        flattened = [k for c in report.clusters for k in c]
        assert sorted(flattened) == list(range(4))


# --------------------------------------------------------- derived-channel tests


class TestDerivedChannels:
    def test_wavefunctions_converged_at_high_cutoff(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
            n_levels=4,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        wf = report.derived["wavefunctions"]
        assert wf.aggregate_status == "converged"
        assert len(wf.per_level) == 4
        for v in wf.per_level:
            assert v.eps_gap_est is not None
            assert v.eps_gap_est < 1e-3
            # Derived metrics are dimensionless: no GHz error estimate.
            assert v.abs_err_est_GHz is None
            assert v.evidence == "verified_empirical"
            assert v.estimator_method == "wavefunction_overlap"

    def test_matrix_elements_converged_at_high_cutoff(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
            n_levels=4,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["matrix_elements"],
        )
        me = report.derived["matrix_elements"]
        assert me.aggregate_status == "converged"
        for v in me.per_level:
            assert v.eps_gap_est is not None
            assert v.estimator_method == "matrix_element_frobenius"

    def test_both_channels_attached_together(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
            n_levels=3,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements"],
        )
        assert set(report.derived.keys()) == {"wavefunctions", "matrix_elements"}

    def test_wavefunction_movement_shrinks_with_cutoff(self):
        # The overlap deficit at a larger base cutoff is no larger than at a
        # clearly undersized one.
        small = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=5, truncated_dim=4)
        large = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=25, truncated_dim=4)
        rep_s = small.estimate_convergence(
            n_levels=3,
            mode="verify",
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        rep_l = large.estimate_convergence(
            n_levels=3,
            mode="verify",
            include_derived=True,
            derived_quantities=["wavefunctions"],
        )
        worst_s = max(v.eps_gap_est for v in rep_s.derived["wavefunctions"].per_level)
        worst_l = max(v.eps_gap_est for v in rep_l.derived["wavefunctions"].per_level)
        assert worst_l <= worst_s + 1e-12

    def test_strict_mode_derived_runs_ratio_test(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
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
