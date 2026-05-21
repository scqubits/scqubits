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

import numpy as np
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
            channel_breakdown_GHz={"charge_tail": 1e-10},
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

    def test_transition_err_is_triangle_inequality(self):
        # Each level reports transition-error estimates to the other levels,
        # equal to the sum of the two absolute estimates (triangle inequality).
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=4, mode="verify", target_abs_GHz=1e-4
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
        report = sq.estimate_convergence(
            tmon, n_levels=4, mode="verify", target_abs_GHz=1e-6
        )
        assert report.aggregate_status == "underconverged"
        assert any("charge" in r and "ncut" in r for r in report.recommendations)


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

    def test_strict_mode_asymptotic_ratio_test_is_verified_empirical(self):
        # A well-converged transmon in strict mode must reach 'converged' backed
        # by 'verified_empirical' evidence -- the design spec reserves a strict
        # 'converged' for certified or ratio-tested verified_empirical evidence.
        # Regression guard: an earlier inversion mislabeled the asymptotic ratio
        # test 'calibrated', which the evidence ladder ranks *below*
        # verified_empirical, so the strict-mode gate wrongly downgraded every
        # ratio-tested 'converged' level to 'marginal'.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=16, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=4, mode="strict", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "converged"
        for v in report.per_level:
            assert v.evidence == "verified_empirical"
        # The refinement engine never emits 'calibrated' (reserved for
        # grid-calibrated cheap estimators, not runtime refinement).
        assert all(v.evidence != "calibrated" for v in report.per_level)


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
        # The charge finite-tail gives a perturbative estimate (not just a
        # boundary diagnostic), so an abs_err_est is provided even in quick mode.
        for v in report.per_level:
            assert v.evidence == "perturbative"
            assert v.estimator_method == "finite_tail_resolvent"
            assert v.abs_err_est_GHz is not None

    def test_quick_mode_perturbative_estimate_tracks_true_error(self):
        # The perturbative tail estimate should grade an undersized transmon as
        # underconverged for a tight target, and a level's estimate must be
        # comparable to the true charge-truncation error.
        ref = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=200).eigenvals(evals_count=5)
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=6, truncated_dim=6)
        report = sq.estimate_convergence(
            tmon, n_levels=5, mode="quick", target_abs_GHz=1e-4
        )
        assert report.aggregate_status == "underconverged"
        e_user = tmon.eigenvals(evals_count=5)
        # The worst-level estimate is within a small factor of the true error.
        v = report.per_level[report.worst_level]
        true_err = abs(float(e_user[v.level_index] - ref[v.level_index]))
        assert 0.5 <= true_err / v.abs_err_est_GHz <= 2.0

    def test_quick_mode_undersized_is_underconverged(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=4, truncated_dim=4)
        report = sq.estimate_convergence(tmon, n_levels=3, mode="quick")
        assert report.aggregate_status == "underconverged"
        # The dominant levels sit hard against the charge boundary.
        assert any("boundary_probability_large" in v.warnings for v in report.per_level)

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


# -------------------------------------------------------------- coherence tests


class TestCoherenceChannel:
    def test_coherence_report_is_per_channel(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
            n_levels=4,
            mode="verify",
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
            assert v.eps_gap_est is not None
            # Coherence is a rate metric, not an energy.
            assert v.abs_err_est_GHz is None

    def test_coherence_converged_at_high_cutoff(self):
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
            n_levels=4,
            mode="verify",
            include_derived=True,
            derived_quantities=["coherence"],
        )
        assert report.derived["coherence"].aggregate_status == "converged"

    def test_symmetric_zero_channel_flagged_noise_floor(self):
        # At ng=0 the 1/f charge-noise dephasing rate vanishes by symmetry, so
        # its rate sits at the noise floor while a real channel does not.
        tmon = sq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=6)
        report = tmon.estimate_convergence(
            n_levels=3,
            mode="verify",
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
        report = tmon.estimate_convergence(
            n_levels=3,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert set(report.derived.keys()) == {
            "wavefunctions",
            "matrix_elements",
            "coherence",
        }


# ------------------------------------------------------- TunableTransmon (Stage 2)


class TestTunableTransmon:
    def test_inherits_all_channels(self):
        # TunableTransmon subclasses Transmon, so it inherits the charge-basis
        # convergence machinery (ncut axis) for all four channels.
        tt = sq.TunableTransmon(
            EJmax=30.0, EC=0.3, d=0.1, flux=0.1, ng=0.0, ncut=31, truncated_dim=6
        )
        report = tt.estimate_convergence(
            n_levels=4,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert report.aggregate_status == "converged"
        assert set(report.derived.keys()) == {
            "wavefunctions",
            "matrix_elements",
            "coherence",
        }
        for sub in report.derived.values():
            assert sub.aggregate_status == "converged"
        # The audit identifies the concrete subclass and its charge cutoff.
        assert report.implementation_audit.qubit_class == "TunableTransmon"
        assert report.implementation_audit.cutoff_parameters == {"ncut": 31}

    def test_undersized_is_not_converged(self):
        tt = sq.TunableTransmon(
            EJmax=30.0, EC=0.3, d=0.1, flux=0.3, ng=0.0, ncut=5, truncated_dim=6
        )
        report = tt.estimate_convergence(n_levels=5, mode="verify", target_abs_GHz=1e-8)
        assert report.aggregate_status in {"marginal", "underconverged"}


# ------------------------------------------------------------- Fluxonium (Stage 2)


class TestFluxonium:
    def test_all_channels_converge(self):
        # Fluxonium uses a harmonic-oscillator (Fock) basis controlled by cutoff.
        flx = sq.Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=6
        )
        report = flx.estimate_convergence(
            n_levels=5,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert report.aggregate_status == "converged"
        # The truncation channel is the HO (Fock) tail, not a charge tail.
        assert report.per_level[0].truncation_channel == "HO_tail"
        assert report.implementation_audit.qubit_class == "Fluxonium"
        assert report.implementation_audit.cutoff_parameters == {"cutoff": 110}
        for sub in report.derived.values():
            assert sub.aggregate_status == "converged"

    def test_undersized_is_underconverged(self):
        flx = sq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=12, truncated_dim=6)
        report = flx.estimate_convergence(
            n_levels=5, mode="verify", target_abs_GHz=1e-8
        )
        assert report.aggregate_status in {"marginal", "underconverged"}
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
        report = flx.estimate_convergence(
            n_levels=4, mode="verify", target_abs_GHz=1e-4
        )
        assert (0, 1) in report.clusters  # the doublet is one cluster
        for k in (0, 1):
            assert "cluster_index_ambiguity" in report.per_level[k].warnings


# ------------------------------------------------------------- FluxQubit (Stage 2)


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
        report = fq.estimate_convergence(
            n_levels=4,
            mode="verify",
            target_abs_GHz=1e-4,
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        assert report.aggregate_status == "converged"
        assert report.per_level[0].truncation_channel == "charge_tail"
        assert report.implementation_audit.qubit_class == "FluxQubit"
        assert report.implementation_audit.cutoff_parameters == {"ncut": 14}
        for sub in report.derived.values():
            assert sub.aggregate_status == "converged"

    def test_undersized_is_underconverged(self):
        fq = self._make(4)
        report = fq.estimate_convergence(n_levels=4, mode="verify", target_abs_GHz=1e-8)
        assert report.aggregate_status in {"marginal", "underconverged"}

    def test_pad_eigenvectors_pads_both_charge_axes(self):
        fq = self._make(2)  # 5x5 charge grid, flattened to length 25
        evecs = np.zeros((25, 1), dtype=np.complex128)
        evecs[12, 0] = 1.0  # center state |n1=0, n2=0| (row 2, col 2)
        padded = fq._convergence_pad_eigenvectors(evecs, 2, 4)  # to 9x9 grid
        assert padded.shape == (81, 1)
        # The center maps to (row 4, col 4) -> flat index 40; nothing else.
        assert padded[40, 0] == 1.0
        assert abs(complex(padded.sum()) - 1.0) < 1e-12


# ---------------------------------------------------------------- ZeroPi (Stage 2)


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

    def test_two_fd_channels_plus_charge(self):
        # ZeroPi's phi grid contributes two independent FD channels (finite box
        # and finite spacing); theta contributes a charge tail.
        zp = self._make()
        report = zp.estimate_convergence(n_levels=3, mode="verify", target_abs_GHz=1e-2)
        assert report.aggregate_status == "converged"
        assert set(report.channel_breakdown_GHz) == {
            "FD_box",
            "FD_stencil",
            "charge_tail",
        }
        # The per-level channel is the multi-axis composite.
        assert {v.truncation_channel for v in report.per_level} == {
            "composite_coupling"
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
        report = snug.estimate_convergence(
            n_levels=3, mode="verify", target_abs_GHz=1e-3
        )
        assert report.aggregate_status == "underconverged"
        cb = report.channel_breakdown_GHz
        assert cb["FD_box"] > cb["FD_stencil"]
        # The recommendation must say to enlarge the box, not just add points.
        assert any("box" in r for r in report.recommendations)

    def test_quick_mode_uses_pedge_and_never_converges(self):
        zp = self._make()
        report = zp.estimate_convergence(n_levels=3, mode="quick")
        # Quick mode reports the FD_box edge-band (P_edge) diagnostic ...
        assert {v.truncation_channel for v in report.per_level} == {"FD_box"}
        # ... and never claims an unqualified 'converged'.
        for v in report.per_level:
            assert v.status != "converged"

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
        # this, and a well-sized ZeroPi reaches 'converged'.
        import scqubits.settings as settings

        zp = self._make()
        assert zp._convergence_richardson_order("grid_spacing") == settings.STENCIL - 1
        assert zp._convergence_richardson_order("grid_box") is None
        report = zp.estimate_convergence(n_levels=3, mode="strict", target_abs_GHz=1e-2)
        assert report.aggregate_status == "converged"
        for v in report.per_level:
            assert v.evidence == "verified_empirical"
            assert v.estimator_method.startswith("richardson_composite")
