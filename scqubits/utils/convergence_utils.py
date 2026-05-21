# convergence_utils.py
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
"""Pure-numerics helpers for the convergence-diagnostics framework.

This module contains backend-agnostic functions used by
:mod:`scqubits.core.convergence`. Keeping them separate makes them
unit-testable without instantiating a qubit.

PR-1 scope: cluster detection, cluster-safe energy matching, and the
geometric ratio test.
PR-2 will add: subspace-angle, wavefunction-overlap.
PR-3 will add: rate-relative-error.
"""

from __future__ import annotations


import numpy as np
import numpy.typing as npt


def detect_clusters(
    evals: npt.NDArray[np.float64], gap_ratio_threshold: float = 0.1
) -> list[tuple[int, ...]]:
    """Group consecutive eigenvalues into near-degenerate clusters.

    A consecutive run of levels forms a cluster when the maximum internal
    gap is below ``gap_ratio_threshold`` times the external gap (to the
    nearest non-cluster level). Singleton clusters are also returned (so
    every level index appears in exactly one tuple).

    Parameters
    ----------
    evals:
        One-dimensional array of eigenvalues, assumed sorted ascending.
    gap_ratio_threshold:
        Maximum allowed internal-to-external gap ratio. Smaller values
        merge fewer levels into clusters.

    Returns
    -------
    A partition of ``range(len(evals))`` into tuples of consecutive
    indices. The union of the tuples is ``set(range(len(evals)))`` and
    they are pairwise disjoint.

    Notes
    -----
    The algorithm is greedy and left-to-right: a singleton starts each
    cluster; extending the cluster requires the next inter-level gap to
    be below ``gap_ratio_threshold * external_gap``, where the external
    gap is the larger of (a) the gap from the current cluster's lowest
    level to its predecessor, and (b) the gap from the candidate's
    successor (the level after the candidate). For terminal levels
    (first / last), the missing side is treated as infinite.
    """
    n = len(evals)
    if n == 0:
        return []
    if n == 1:
        return [(0,)]

    clusters: list[tuple[int, ...]] = []
    i = 0
    while i < n:
        cluster = [i]
        # Try to extend the cluster.
        while cluster[-1] + 1 < n:
            j = cluster[-1] + 1
            internal_gap = float(evals[j] - evals[j - 1])

            # External-gap candidates: below the cluster's first level
            # and above the cluster-extension candidate. Only finite
            # sides are usable as external references.
            ext_below: float | None = (
                float(evals[cluster[0]] - evals[cluster[0] - 1])
                if cluster[0] > 0
                else None
            )
            ext_above: float | None = (
                float(evals[j + 1] - evals[j]) if j + 1 < n else None
            )

            # External reference = the closest (smallest) non-cluster
            # gap on either side; this is the strictest comparison
            # against which the internal gap must be small to count as
            # a cluster. If neither side is available (terminal
            # singleton), don't extend.
            if ext_below is None and ext_above is None:
                break
            if ext_below is None:
                external_gap: float = ext_above  # type: ignore[assignment]
            elif ext_above is None:
                external_gap = ext_below
            else:
                external_gap = min(ext_below, ext_above)

            if internal_gap < gap_ratio_threshold * external_gap:
                cluster.append(j)
            else:
                break

        clusters.append(tuple(cluster))
        i = cluster[-1] + 1

    return clusters


def cluster_safe_match_energies(
    evals_a: npt.NDArray[np.float64],
    evals_b: npt.NDArray[np.float64],
    clusters_a: list[tuple[int, ...]],
) -> tuple[list[tuple[int, ...]], npt.NDArray[np.float64]]:
    """Map clusters from one spectrum to the same index ranges in another,
    and return per-cluster absolute energy differences.

    The two spectra are expected to correspond to the same physical
    system at two slightly different cutoffs, both sorted ascending and
    of the same length. Index identification is direct (cluster ``c`` in
    spectrum A corresponds to the same index tuple in spectrum B), which
    is appropriate when the cluster structure is stable across cutoffs.
    Within each cluster, the comparison is on sorted eigenvalue sets, so
    intra-cluster index swaps (eigenvalue rotations within a near-
    degenerate block) do not produce spurious differences.

    Parameters
    ----------
    evals_a, evals_b:
        Eigenvalue arrays of the same length, sorted ascending.
    clusters_a:
        Cluster partition for ``evals_a`` (as returned by
        :func:`detect_clusters`).

    Returns
    -------
    clusters_b : list of tuple of int
        The index partition for ``evals_b``; in this PR identical to
        ``clusters_a`` (direct-index mapping).
    max_diff_per_cluster : ndarray
        Entry ``k`` is the maximum of
        ``|sorted(evals_a[c]) - sorted(evals_b[c])|`` over the indices in
        cluster ``k``.

    Raises
    ------
    ValueError
        If the two spectra have different lengths.
    """
    if len(evals_a) != len(evals_b):
        raise ValueError(
            f"Spectra have different lengths: {len(evals_a)} vs {len(evals_b)}"
        )

    max_diffs = np.empty(len(clusters_a), dtype=np.float64)
    for idx, cluster in enumerate(clusters_a):
        sorted_a = np.sort(evals_a[list(cluster)])
        sorted_b = np.sort(evals_b[list(cluster)])
        max_diffs[idx] = float(np.max(np.abs(sorted_a - sorted_b)))

    # PR-1 uses direct-index mapping for cluster correspondence.
    # PR-2 may add a centroid-overlap-based matching for cases where the
    # cluster structure itself changes between cutoffs.
    return list(clusters_a), max_diffs


def geometric_ratio_test(
    diff_first: npt.NDArray[np.float64],
    diff_second: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Run a per-cluster geometric ratio test on successive refinement movements.

    Given the absolute spectral movements between consecutive cutoffs --
    ``diff_first`` for the base-to-first-refinement step and ``diff_second`` for
    the first-to-second-refinement step -- this characterizes whether each
    cluster is in the geometric (asymptotically converging) regime and, if so,
    extrapolates the remaining tail error from a geometric series.

    Parameters
    ----------
    diff_first:
        Per-cluster absolute eigenvalue movement of the first refinement step.
        Entries are non-negative.
    diff_second:
        Per-cluster absolute eigenvalue movement of the second refinement step,
        aligned element-wise with ``diff_first``.

    Returns
    -------
    ratios
        ``diff_second / diff_first`` per cluster, with ``inf`` wherever
        ``diff_first`` is zero (the movement is already at the floor and the
        ratio is undefined).
    geometric_tail
        The geometric-series tail estimate ``diff_first / (1 - ratios)`` for
        clusters with ``ratios < 1``, and ``inf`` elsewhere.
    is_asymptotic
        ``True`` for clusters with ``ratios < 1``, i.e. those whose movement is
        shrinking and for which the geometric extrapolation is meaningful.
    """
    ratios = np.divide(
        diff_second,
        diff_first,
        out=np.full_like(diff_first, np.inf),
        where=diff_first > 0,
    )
    geometric_tail = np.where(
        ratios < 1.0,
        diff_first / np.clip(1.0 - ratios, 1e-30, None),
        np.inf,
    )
    is_asymptotic = ratios < 1.0
    return ratios, geometric_tail, is_asymptotic
