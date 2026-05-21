# test_convergence_utils.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
"""Unit tests for the pure-numerics helpers in ``scqubits.utils.convergence_utils``."""

from __future__ import annotations

import numpy as np
import pytest

from scqubits.utils.convergence_utils import (
    cluster_safe_match_energies,
    detect_clusters,
    geometric_ratio_test,
)

# Mark all tests in this module as slow so they participate in the
# `-m "not slow"` quick suite opt-out.
pytestmark = pytest.mark.slow


# --------------------------------------------------------------- detect_clusters


class TestDetectClusters:
    def test_empty_array_returns_empty_list(self):
        assert detect_clusters(np.asarray([], dtype=np.float64)) == []

    def test_single_level_returns_singleton(self):
        assert detect_clusters(np.asarray([1.5], dtype=np.float64)) == [(0,)]

    def test_well_separated_levels_are_all_singletons(self):
        # Gaps grow rapidly; no cluster should form.
        evals = np.asarray([0.0, 1.0, 3.0, 7.0, 15.0], dtype=np.float64)
        clusters = detect_clusters(evals, gap_ratio_threshold=0.1)
        assert clusters == [(0,), (1,), (2,), (3,), (4,)]

    def test_doublet_merged_when_internal_gap_below_threshold(self):
        # Levels 1 and 2 are tightly spaced relative to the external gap
        # of order 1.0 above and 0.97 below.
        evals = np.asarray([0.0, 1.0, 1.001, 2.0, 3.0], dtype=np.float64)
        clusters = detect_clusters(evals, gap_ratio_threshold=0.1)
        # (1, 2) should be one cluster; others singletons.
        assert (1, 2) in clusters
        # Every index is covered exactly once.
        flattened = [k for c in clusters for k in c]
        assert sorted(flattened) == list(range(len(evals)))

    def test_partition_invariant_random(self):
        # Property test: for any sorted evals, partition is exhaustive
        # and disjoint.
        rng = np.random.default_rng(seed=42)
        for trial in range(20):
            n = int(rng.integers(2, 30))
            evals = np.sort(rng.uniform(-5.0, 5.0, size=n))
            clusters = detect_clusters(
                evals, gap_ratio_threshold=float(rng.uniform(0.01, 0.5))
            )
            flat = [k for c in clusters for k in c]
            assert sorted(flat) == list(
                range(n)
            ), f"Trial {trial}: partition not exhaustive or has duplicates"

    def test_threshold_zero_yields_only_singletons(self):
        # With threshold 0, no internal gap can be strictly smaller than
        # 0 times anything finite, so all levels are singletons.
        evals = np.asarray([0.0, 1e-12, 2e-12, 1.0], dtype=np.float64)
        clusters = detect_clusters(evals, gap_ratio_threshold=0.0)
        assert clusters == [(0,), (1,), (2,), (3,)]


# ------------------------------------------------------ cluster_safe_match_energies


class TestClusterSafeMatchEnergies:
    def test_identical_spectra_zero_max_diff(self):
        evals = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clusters = [(0,), (1,), (2,), (3,)]
        out_clusters, max_diffs = cluster_safe_match_energies(evals, evals, clusters)
        assert out_clusters == clusters
        np.testing.assert_array_equal(max_diffs, np.zeros(4))

    def test_singleton_diff_returns_absolute_value(self):
        evals_a = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        evals_b = np.asarray([0.01, 1.0, 1.97], dtype=np.float64)
        clusters = [(0,), (1,), (2,)]
        _, max_diffs = cluster_safe_match_energies(evals_a, evals_b, clusters)
        np.testing.assert_allclose(max_diffs, [0.01, 0.0, 0.03])

    def test_cluster_sort_then_compare_handles_intracluster_swap(self):
        # Two near-degenerate levels swap order between the two spectra;
        # cluster-aware sort-then-compare should report ~zero diff.
        evals_a = np.asarray([0.0, 1.0, 1.001, 5.0], dtype=np.float64)
        # If b's two near-degenerate levels are swapped in label but the
        # cluster sort-then-compare treats them as a set, the diff is 0.
        evals_b_sorted = np.asarray([0.0, 1.0, 1.001, 5.0], dtype=np.float64)
        clusters = [(0,), (1, 2), (3,)]
        _, max_diffs = cluster_safe_match_energies(evals_a, evals_b_sorted, clusters)
        np.testing.assert_allclose(max_diffs, [0.0, 0.0, 0.0])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            cluster_safe_match_energies(
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0, 2.0]),
                [(0,), (1,)],
            )

    def test_max_picks_largest_within_cluster(self):
        # Cluster of three levels with small but distinct shifts.
        evals_a = np.asarray([0.0, 1.0, 1.001, 1.002, 5.0], dtype=np.float64)
        evals_b = np.asarray([0.0, 1.0, 1.001, 1.005, 5.0], dtype=np.float64)
        clusters = [(0,), (1, 2, 3), (4,)]
        _, max_diffs = cluster_safe_match_energies(evals_a, evals_b, clusters)
        assert max_diffs[0] == 0.0
        # Within the cluster, only the third member moved (by 0.003).
        np.testing.assert_allclose(max_diffs[1], 0.003, rtol=1e-12)
        assert max_diffs[2] == 0.0


# ----------------------------------------------------------- geometric_ratio_test


class TestGeometricRatioTest:
    def test_asymptotic_cluster_extrapolates_geometric_tail(self):
        # Movement halves each step (R = 0.5): geometric tail = d0 / (1 - R) = 2.
        diff_first = np.asarray([1.0], dtype=np.float64)
        diff_second = np.asarray([0.5], dtype=np.float64)
        ratios, tail, is_asymptotic = geometric_ratio_test(diff_first, diff_second)
        np.testing.assert_allclose(ratios, [0.5])
        np.testing.assert_allclose(tail, [2.0])
        assert bool(is_asymptotic[0]) is True

    def test_non_asymptotic_cluster_flags_inf_tail(self):
        # Movement grows (R = 2): not asymptotic, tail undefined (inf).
        diff_first = np.asarray([1.0], dtype=np.float64)
        diff_second = np.asarray([2.0], dtype=np.float64)
        ratios, tail, is_asymptotic = geometric_ratio_test(diff_first, diff_second)
        np.testing.assert_allclose(ratios, [2.0])
        assert np.isinf(tail[0])
        assert bool(is_asymptotic[0]) is False

    def test_zero_first_movement_gives_inf_ratio(self):
        # d0 already at the floor: ratio undefined (inf), so not asymptotic.
        diff_first = np.asarray([0.0], dtype=np.float64)
        diff_second = np.asarray([1.0], dtype=np.float64)
        ratios, tail, is_asymptotic = geometric_ratio_test(diff_first, diff_second)
        assert np.isinf(ratios[0])
        assert np.isinf(tail[0])
        assert bool(is_asymptotic[0]) is False
