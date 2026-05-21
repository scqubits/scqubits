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
PR-2 adds: charge-basis padding, subspace angle, and wavefunction overlap.
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


def richardson_estimate(
    diff_first: npt.NDArray[np.float64],
    diff_second: npt.NDArray[np.float64],
    n0: int,
    n1: int,
    n2: int,
    order: int,
    rel_tol: float = 0.5,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Richardson ``h**order`` error estimate and asymptoticity test for an FD grid.

    For a finite-difference coordinate the discretization error scales as
    ``h**order`` with the spacing ``h`` proportional to ``1 / (N - 1)`` at a fixed
    window, so the window cancels and only the grid-point counts ``n0 < n1 < n2``
    enter. With ``g_i = 1 / (n_i - 1)**order`` and the model
    ``E(N) = E_inf + C g(N)``, the error of the coarsest (user) grid extrapolated
    to the continuum is ``diff_first / (1 - g1 / g0)``.

    The asymptoticity test compares the observed movement ratio
    ``diff_second / diff_first`` against the model-predicted
    ``(g1 - g2) / (g0 - g1)``: a cluster is asymptotic when the two agree to
    within ``rel_tol`` (relative), or when ``diff_first`` is at the numerical
    floor (no measurable movement). Richardson is a verified estimate only in the
    asymptotic regime; the caller falls back to a one-step bound otherwise.

    Parameters
    ----------
    diff_first:
        Per-cluster absolute eigenvalue movement from the coarse grid ``n0`` to
        the first refinement ``n1``. Entries are non-negative.
    diff_second:
        Per-cluster absolute movement from ``n1`` to the second refinement
        ``n2``, aligned with ``diff_first``.
    n0, n1, n2:
        Grid-point counts of the coarse grid and the two refinements,
        ``n0 < n1 < n2``, all sharing the same window.
    order:
        Leading discretization-error order ``p`` (4 for a 5-point second-
        derivative stencil).
    rel_tol:
        Relative tolerance on the asymptoticity test.

    Returns
    -------
    estimate
        Per-cluster Richardson error estimate of the coarse grid ``n0``; ``0``
        where there is no movement.
    is_asymptotic
        ``True`` for clusters whose observed movement ratio matches the
        ``h**order`` model (or that are at the numerical floor).
    """
    g0 = 1.0 / (n0 - 1) ** order
    g1 = 1.0 / (n1 - 1) ** order
    g2 = 1.0 / (n2 - 1) ** order
    expected_ratio = (g1 - g2) / (g0 - g1)
    estimate = np.where(diff_first > 0, diff_first / (1.0 - g1 / g0), 0.0)
    observed_ratio = np.divide(
        diff_second,
        diff_first,
        out=np.full_like(diff_first, np.inf),
        where=diff_first > 0,
    )
    rel_dev = np.abs(observed_ratio - expected_ratio) / expected_ratio
    is_asymptotic = (rel_dev <= rel_tol) | (diff_first <= 0.0)
    return estimate, is_asymptotic


def _charge_tail_green_11(
    ncut: int,
    EJ: float,
    EC: float,
    ng: float,
    energy: float,
    side: int,
    depth: int,
) -> tuple[float, float]:
    """Return the ``(1, 1)`` Green-function entry and smallest eigenvalue of a tail.

    Builds the depth-``depth`` tridiagonal block of the dropped charge tail on one
    side (``side = +1`` for charges ``ncut+1 .. ncut+depth``, ``side = -1`` for
    ``-(ncut+1) .. -(ncut+depth)``), with diagonal ``4 EC (n - ng)**2`` and
    nearest-neighbor hopping ``-EJ/2``, and returns
    ``[(T - energy * I)^-1]_{0,0}`` together with the block's smallest eigenvalue
    (used to detect a non-perturbative tail).
    """
    idx = np.arange(1, depth + 1)
    n = side * (ncut + idx)
    diag = 4.0 * EC * (n - ng) ** 2
    block = np.diag(diag).astype(np.float64)
    if depth > 1:
        off = np.full(depth - 1, -EJ / 2.0)
        block += np.diag(off, 1) + np.diag(off, -1)
    min_eig = float(np.linalg.eigvalsh(block)[0])
    resolvent_col = np.linalg.solve(block - energy * np.eye(depth), np.eye(depth)[:, 0])
    return float(resolvent_col[0]), min_eig


def charge_finite_tail_estimate(
    evecs: npt.NDArray[np.float64],
    e_levels: npt.NDArray[np.float64],
    ncut: int,
    EJ: float,
    EC: float,
    ng: float,
    n_levels: int,
    depth_max: int = 16,
    tol: float = 0.01,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Finite-tail (Green-function) charge-truncation error estimate per level.

    Implements the design-spec second-order tail estimate for a 1D charge basis
    (transmon-like): for level ``k`` with boundary amplitudes
    ``c_R = <ncut|u_k>`` and ``c_L = <-ncut|u_k>``,

        errhat_k = | EJ^2 / 4 * (|c_R|^2 G_R + |c_L|^2 G_L) |,

    where ``G_R``/``G_L`` are the ``(1, 1)`` entries of the resolvents of the
    finite dropped tail blocks of depth ``d`` on each side. The depth is increased
    until the estimate changes by less than ``tol`` (relative) or ``depth_max`` is
    reached. At ``d = 1`` this reduces to the familiar boundary-denominator
    estimate. The estimate is a perturbative one and must not be used blindly for
    a multi-coordinate charge basis.

    Parameters
    ----------
    evecs:
        Charge-basis eigenvectors (columns are eigenvectors), rows ordered from
        charge ``-ncut`` to ``+ncut``.
    e_levels:
        Eigenvalues aligned with the columns of ``evecs``.
    ncut, EJ, EC, ng:
        Charge cutoff and transmon energies / offset charge.
    n_levels:
        Number of lowest levels to assess.
    depth_max:
        Maximum tail depth.
    tol:
        Relative change in the estimate at which depth refinement stops.

    Returns
    -------
    estimate
        Per-level perturbative error estimate (GHz).
    perturbative_ok
        ``False`` for a level whose tail block has an eigenvalue at or below the
        level energy -- the tail is not perturbative there and the estimate is
        unreliable.
    boundary_prob
        Per-level boundary probability ``|c_L|^2 + |c_R|^2``; a large value is an
        independent warning that the basis is too small.
    """
    c_left = evecs[0, :n_levels]
    c_right = evecs[-1, :n_levels]
    boundary_prob = (np.abs(c_left) ** 2 + np.abs(c_right) ** 2).astype(np.float64)
    estimate = np.zeros(n_levels, dtype=np.float64)
    perturbative_ok = np.ones(n_levels, dtype=np.bool_)

    for k in range(n_levels):
        energy = float(e_levels[k])
        previous: float | None = None
        est = 0.0
        for depth in range(1, depth_max + 1):
            g_right, min_right = _charge_tail_green_11(
                ncut, EJ, EC, ng, energy, 1, depth
            )
            g_left, min_left = _charge_tail_green_11(
                ncut, EJ, EC, ng, energy, -1, depth
            )
            est = abs(
                EJ**2
                / 4.0
                * (
                    abs(complex(c_right[k])) ** 2 * g_right
                    + abs(complex(c_left[k])) ** 2 * g_left
                )
            )
            if min(min_right, min_left) <= energy:
                perturbative_ok[k] = False
            if previous is not None and abs(est - previous) <= tol * max(
                previous, 1e-30
            ):
                break
            previous = est
        estimate[k] = est

    return estimate, perturbative_ok, boundary_prob


def pad_charge_basis(
    evecs: npt.NDArray[np.float64],
    ncut_old: int,
    ncut_new: int,
) -> npt.NDArray[np.float64]:
    """Embed charge-basis eigenvectors into a larger-cutoff basis by zero-padding.

    The transmon charge basis spans charge states ``-ncut .. +ncut`` (dimension
    ``2 * ncut + 1``). To compare eigenvectors computed at two cutoffs they must
    live in a common basis: the smaller-cutoff vectors are embedded into the
    larger basis by adding ``ncut_new - ncut_old`` zero rows at each end, which
    keeps the ``n = 0`` charge state aligned at the center.

    Parameters
    ----------
    evecs:
        Charge-basis eigenvectors with ``2 * ncut_old + 1`` rows; each column is
        one eigenvector.
    ncut_old:
        Charge cutoff the vectors were computed at.
    ncut_new:
        Target charge cutoff; must be at least ``ncut_old``.

    Returns
    -------
    The eigenvectors embedded in the ``2 * ncut_new + 1`` dimensional basis;
    returned unchanged when the two cutoffs are equal.

    Raises
    ------
    ValueError
        If ``ncut_new`` is smaller than ``ncut_old`` or ``evecs`` does not have
        ``2 * ncut_old + 1`` rows.
    """
    if ncut_new < ncut_old:
        raise ValueError(f"ncut_new ({ncut_new}) must be >= ncut_old ({ncut_old})")
    expected_rows = 2 * ncut_old + 1
    if evecs.shape[0] != expected_rows:
        raise ValueError(
            f"evecs has {evecs.shape[0]} rows; expected 2*ncut_old+1 = {expected_rows}"
        )
    pad = ncut_new - ncut_old
    if pad == 0:
        return evecs
    return np.pad(evecs, ((pad, pad), (0, 0)))


def subspace_angle(
    subspace_a: npt.NDArray[np.float64], subspace_b: npt.NDArray[np.float64]
) -> float:
    """Return the sine of the largest principal angle between two subspaces.

    The subspaces are the column spans of ``subspace_a`` and ``subspace_b`` --
    tall matrices with matching row dimension and orthonormal columns. The value
    ``||(I - A A^H) B||_2`` is 0 when the spans coincide and 1 when ``B`` has a
    direction orthogonal to all of ``A``. This is the cluster-level analogue of
    one minus a wavefunction overlap, robust to eigenvector rotations within a
    near-degenerate block.

    Parameters
    ----------
    subspace_a, subspace_b:
        Tall matrices with the same number of rows and orthonormal columns; the
        caller pads them to a common basis first if the cutoffs differ.

    Returns
    -------
    The sine of the largest principal angle, clamped to ``[0, 1]``.
    """
    residual = subspace_b - subspace_a @ (subspace_a.conj().T @ subspace_b)
    sin_angle = float(np.linalg.norm(residual, ord=2))
    return min(1.0, sin_angle)


def wavefunction_overlap(
    c_a: npt.NDArray[np.float64],
    c_b: npt.NDArray[np.float64],
    ncut_a: int,
    ncut_b: int,
) -> npt.NDArray[np.float64]:
    """Return per-level wavefunction overlap moduli across two charge cutoffs.

    Both eigenvector arrays are embedded into the larger charge basis (via
    :func:`pad_charge_basis`) and the modulus of the per-column inner product
    ``|<a_k | b_k>|`` is returned. The modulus is invariant under each
    eigenvector's arbitrary global phase, so no phase standardization is needed;
    a value near 1 means the wavefunction is stable across the cutoff change.
    This per-level measure is meaningful only for isolated (non-degenerate)
    levels; near-degenerate clusters must be compared with
    :func:`subspace_angle`.

    Parameters
    ----------
    c_a, c_b:
        Charge-basis eigenvector arrays (columns are eigenvectors) at cutoffs
        ``ncut_a`` and ``ncut_b``, with the same number of columns.
    ncut_a, ncut_b:
        Charge cutoffs of ``c_a`` and ``c_b``.

    Returns
    -------
    The per-level overlap moduli ``|<a_k | b_k>|``, one entry per column.
    """
    ncut_max = max(ncut_a, ncut_b)
    a = pad_charge_basis(c_a, ncut_a, ncut_max)
    b = pad_charge_basis(c_b, ncut_b, ncut_max)
    n_cols = a.shape[1]
    overlaps = np.empty(n_cols, dtype=np.float64)
    for k in range(n_cols):
        overlaps[k] = float(np.abs(np.vdot(a[:, k], b[:, k])))
    return overlaps
