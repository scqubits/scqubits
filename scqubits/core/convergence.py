# convergence.py
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
"""ConvergenceCheckable mixin and the energy-channel verified-refinement engine.

PR-1 scope: a concrete qubit that adds ``ConvergenceCheckable`` to its MRO and
declares ``_convergence_axes`` gains an :meth:`estimate_convergence` method
returning a :class:`~scqubits.core.convergence_report.ConvergenceReport`.
Only the energy sub-channel is implemented here; PR-2 adds wavefunctions and
matrix elements, PR-3 adds coherence.

The mixin's refinement engine clones the qubit, bumps a cutoff axis by a
step, re-diagonalizes, and compares cluster-matched eigenvalues. Cheap-mode
diagnostics never claim ``converged`` (per the published design specification).
"""

from __future__ import annotations

import copy

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

import scqubits.utils.convergence_utils as cutils

from scqubits import settings
from scqubits.core.convergence_report import (
    ConvergenceReport,
    Evidence,
    ImplementationAudit,
    LevelVerdict,
    Status,
    TruncationChannel,
    evidence_at_least,
)


class ConvergenceCheckable:
    """Mixin providing :meth:`estimate_convergence` for qubit classes.

    Concrete qubits opt in by:

    1. adding this class to their MRO,
    2. declaring ``_convergence_axes`` (a tuple of attribute names that name
       the truncation knobs to refine), and
    3. optionally overriding the hooks below.

    The mixin owns the refinement engine; per-class hooks supply the small
    amount of qubit-specific knowledge needed (step size, channel label,
    cheap diagnostic).
    """

    # Per-class override. A class attribute is the common case (Transmon:
    # ``("ncut",)``); ``Circuit`` will use a ``@property`` to introspect its
    # dynamic ``cutoff_n_*`` / ``cutoff_ext_*`` attributes.
    _convergence_axes: tuple[str, ...] = ()

    # ------------------------------------------------------------------ hooks
    # Per-class overrides. Defaults are reasonable for transmon-like
    # single-knob qubits; multi-axis qubits override.

    def _convergence_step(self, axis: str) -> int:
        """Return the refinement step (axis units) for ``axis``.

        Default: ``max(4, current_value // 4)``. Concrete qubits override
        if a different heuristic better matches their convergence law.
        """
        current = getattr(self, axis)
        return max(4, current // 4)

    def _convergence_truncation_channel(self, axis: str) -> TruncationChannel:
        """Return the physical channel label for ``axis``.

        Defaults to ``"charge"``. Concrete qubits override for HO bases
        (``"HO_phi"``), finite-difference grids (``"FD_grid"``), etc.
        """
        return "charge"

    def _convergence_boundary_diagnostic(
        self, esys: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], axis: str
    ) -> npt.NDArray[np.float64] | None:
        """Return a per-level cheap boundary-amplitude diagnostic.

        For Transmon: ``|c_{-ncut, k}|^2 + |c_{+ncut, k}|^2`` for each kept
        level ``k``. Returns ``None`` if no cheap diagnostic is available
        for this axis; quick mode then falls back to ``unverified``.
        """
        return None

    # ----------------------------------------------------------------- public

    def estimate_convergence(
        self,
        n_levels: int = 6,
        mode: str = "verify",
        scope: str = "absolute",
        target_abs_GHz: float | None = None,
        target_gap_rel: float = 1e-3,
        g_floor_GHz: float = 1e-3,
        include_derived: bool = False,
        derived_quantities: Sequence[str] | None = None,
        refinement: str = "one_step",
    ) -> ConvergenceReport:
        """Estimate the convergence of the lowest ``n_levels`` eigenvalues.

        Parameters
        ----------
        n_levels:
            Number of lowest eigenvalues to assess.
        mode:
            One of ``"quick"`` (cheap diagnostics only, no extra
            diagonalizations), ``"verify"`` (one refinement at a bumped
            cutoff; default), or ``"strict"`` (ratio test across two
            successive refinements).
        scope:
            ``"absolute"`` (verdict applied to ``abs_err_est_GHz``) or
            ``"observed_gap_scale"`` (verdict applied to error normalized
            by the local isolation gap).
        target_abs_GHz:
            Required for absolute-scope status assignment. If ``None``,
            the report still includes ``abs_err_est_GHz`` per level but
            statuses default to ``"unverified"``.
        target_gap_rel:
            Threshold for the observed-gap scope (default ``1e-3``).
        g_floor_GHz:
            Floor on the local isolation gap (default ``1e-3`` GHz = 1 MHz)
            to avoid divide-by-tiny-numbers when the gap is very small.
        include_derived:
            PR-1: ignored; raises ``NotImplementedError`` if set to True
            in this PR (the wavefunctions/matrix-elements/coherence
            channels arrive in PR-2 and PR-3).
        derived_quantities:
            PR-1: ignored.
        refinement:
            ``"one_step"`` for verify mode; ``"ratio_test"`` for strict
            mode. Coerced to match ``mode`` if inconsistent.

        Returns
        -------
        :class:`~scqubits.core.convergence_report.ConvergenceReport`
        """
        # -------- input validation
        if mode not in ("quick", "verify", "strict"):
            raise ValueError(
                f"mode must be 'quick', 'verify', or 'strict'; got {mode!r}"
            )
        if scope not in ("absolute", "observed_gap_scale"):
            raise ValueError(
                f"scope must be 'absolute' or 'observed_gap_scale'; got {scope!r}"
            )
        if refinement not in ("one_step", "ratio_test"):
            raise ValueError(
                f"refinement must be 'one_step' or 'ratio_test'; " f"got {refinement!r}"
            )
        if include_derived:
            raise NotImplementedError(
                "include_derived=True is not yet available in this version. "
                "Wavefunctions and matrix elements arrive in PR-2; coherence "
                "in PR-3."
            )
        if not self._convergence_axes:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare _convergence_axes; "
                "convergence checking is not implemented for this class."
            )

        # In strict mode, ratio_test is the implied refinement.
        if mode == "strict" and refinement == "one_step":
            refinement = "ratio_test"

        # -------- diagonalize at the current cutoff
        # Buffer one extra level so the topmost reported level still has an
        # upper gap available for the observed-gap-scale denominator.
        n_buffer = 1 if scope == "observed_gap_scale" else 0
        n_eigs = n_levels + n_buffer
        evals_n0, evecs_n0 = self.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]

        # -------- dispatch by mode
        if mode == "quick":
            return self._convergence_quick(
                evals_n0=evals_n0,
                evecs_n0=evecs_n0,
                n_levels=n_levels,
                scope=scope,
                target_abs_GHz=target_abs_GHz,
                target_gap_rel=target_gap_rel,
                g_floor_GHz=g_floor_GHz,
            )
        # verify / strict
        return self._convergence_refine(
            evals_n0=evals_n0,
            evecs_n0=evecs_n0,
            n_levels=n_levels,
            n_buffer=n_buffer,
            mode=mode,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
            g_floor_GHz=g_floor_GHz,
            refinement=refinement,
        )

    # ------------------------------------------------------------ engine helpers

    def _convergence_clone_at(
        self, axis_values: dict[str, int]
    ) -> "ConvergenceCheckable":
        """Return a deep-copy of this qubit with the named axes set to new values.

        Uses ``copy.deepcopy`` plus attribute assignment. scqubits'
        ``WatchedProperty`` descriptors invalidate any internal caches on
        attribute change, so the clone re-computes the spectrum from scratch.
        """
        clone = copy.deepcopy(self)
        for axis, value in axis_values.items():
            setattr(clone, axis, value)
        return clone

    def _convergence_audit(
        self,
        n_levels: int,
        n_buffer: int,
        mode: str,
        refinement: str,
    ) -> ImplementationAudit:
        """Construct an :class:`ImplementationAudit` snapshot."""
        try:
            from scqubits import __version__ as scq_version
        except ImportError:  # pragma: no cover -- defensive
            scq_version = "unknown"

        cutoff_params = {
            axis: int(getattr(self, axis)) for axis in self._convergence_axes
        }
        # Diagonalization method label: prefer an explicit user setting if
        # provided, otherwise fall back to a generic "default".
        diag_method = getattr(self, "evals_method", None) or "default"
        if not isinstance(diag_method, str):
            diag_method = getattr(diag_method, "__name__", "callable")

        return ImplementationAudit(
            scqubits_version=str(scq_version),
            scqubits_commit=None,
            qubit_class=type(self).__name__,
            basis=getattr(self, "_convergence_basis", "auto"),
            diagonalization_method=str(diag_method),
            cutoff_parameters=cutoff_params,
            fd_stencil_order=None,
            fd_box=None,
            nonpoly_backend=None,
            n_levels_requested=n_levels,
            n_levels_buffer=n_buffer,
            mode=mode,  # type: ignore[arg-type]
            refinement=refinement,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------ mode handlers

    def _convergence_quick(
        self,
        evals_n0: npt.NDArray[np.float64],
        evecs_n0: npt.NDArray[np.float64],
        n_levels: int,
        scope: str,
        target_abs_GHz: float | None,
        target_gap_rel: float,
        g_floor_GHz: float,
    ) -> ConvergenceReport:
        """Quick-mode report: cheap boundary diagnostic only.

        Per the published design, quick mode never returns ``converged``;
        the best it can say is ``likely_converged`` based on a cheap
        boundary-amplitude check.
        """
        # We assume a single axis for PR-1 (Transmon ncut). Multi-axis
        # quick-mode aggregation is straightforward and added with Stage 3.
        axis = self._convergence_axes[0]
        boundary_amplitudes = self._convergence_boundary_diagnostic(
            (evals_n0, evecs_n0), axis
        )
        channel: TruncationChannel = self._convergence_truncation_channel(axis)

        per_level: list[LevelVerdict] = []
        recommendations: list[str] = []
        # Start worst-tracker at the best possible status; accumulate worse
        # as levels are inspected.
        worst_index: int = 0
        worst_status: Status = "converged"

        # Threshold under which a level is "likely_converged" from quick
        # diagnostics alone. Conservative -- the published design only
        # allows quick mode to escape "unverified" in clearly easy regimes.
        QUICK_BOUNDARY_THRESHOLD = 1e-6

        for k in range(n_levels):
            warnings: list[str] = []
            if boundary_amplitudes is None:
                status: Status = "unverified"
                ev: Evidence = "unverified"
                warnings.append("no_boundary_diagnostic_available")
            elif boundary_amplitudes[k] < QUICK_BOUNDARY_THRESHOLD:
                status = "likely_converged"
                ev = "diagnostic"
            else:
                status = "unverified"
                ev = "diagnostic"
                warnings.append("boundary_amplitude_above_threshold")
                recommendations.append(
                    f"level {k}: boundary amplitude {boundary_amplitudes[k]:.2e} "
                    f"exceeds quick-mode threshold {QUICK_BOUNDARY_THRESHOLD:.0e}; "
                    f"re-run with mode='verify' to obtain an empirical estimate"
                )

            per_level.append(
                LevelVerdict(
                    level_index=k,
                    status=status,
                    status_scope=scope,  # type: ignore[arg-type]
                    evidence=ev,
                    abs_err_est_GHz=None,  # quick mode has no error estimate
                    eps_gap_est=None,
                    truncation_channel=channel,
                    estimator_method="boundary_diagnostic",
                    warnings=tuple(warnings),
                )
            )

            # Track worst status for aggregate.
            if _status_rank(status) > _status_rank(worst_status):
                worst_status = status
                worst_index = k

        # If no level escaped "unverified", aggregate is "unverified".
        if worst_index is None:
            worst_index = 0

        return ConvergenceReport(
            per_level=per_level,
            aggregate_status=worst_status,
            worst_level=worst_index,
            channel_breakdown_GHz={},
            clusters=[(k,) for k in range(n_levels)],
            recommendations=list(dict.fromkeys(recommendations)),  # dedupe
            implementation_audit=self._convergence_audit(
                n_levels=n_levels, n_buffer=0, mode="quick", refinement="one_step"
            ),
        )

    def _convergence_refine(
        self,
        evals_n0: npt.NDArray[np.float64],
        evecs_n0: npt.NDArray[np.float64],
        n_levels: int,
        n_buffer: int,
        mode: str,
        scope: str,
        target_abs_GHz: float | None,
        target_gap_rel: float,
        g_floor_GHz: float,
        refinement: str,
    ) -> ConvergenceReport:
        """Verify / strict mode: refine cutoff(s), compare cluster-matched energies."""
        # For PR-1, we handle exactly one axis. Multi-axis aggregation
        # (refine one axis at a time, combine absolute errors by triangle
        # inequality) will be added when Circuit lands in Stage 3.
        if len(self._convergence_axes) != 1:
            raise NotImplementedError(
                "Multi-axis refinement is not yet implemented; only single-axis "
                "qubits (e.g. Transmon) are supported in this version."
            )
        axis = self._convergence_axes[0]
        step = self._convergence_step(axis)
        n_eigs = n_levels + n_buffer

        current_value = getattr(self, axis)
        clone_1 = self._convergence_clone_at({axis: current_value + step})
        evals_n1, _ = clone_1.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]

        evals_n2: npt.NDArray[np.float64] | None = None
        if refinement == "ratio_test":
            clone_2 = self._convergence_clone_at({axis: current_value + 2 * step})
            evals_n2, _ = clone_2.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]

        return self._convergence_build_energy_report(
            evals_n0=evals_n0[:n_levels],
            evals_n1=evals_n1[:n_levels],
            evals_n2=evals_n2[:n_levels] if evals_n2 is not None else None,
            buffer_n0=evals_n0[n_levels:] if n_buffer > 0 else None,
            n_levels=n_levels,
            n_buffer=n_buffer,
            mode=mode,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
            g_floor_GHz=g_floor_GHz,
            refinement=refinement,
            axis=axis,
        )

    def _convergence_build_energy_report(
        self,
        evals_n0: npt.NDArray[np.float64],
        evals_n1: npt.NDArray[np.float64],
        evals_n2: npt.NDArray[np.float64] | None,
        buffer_n0: npt.NDArray[np.float64] | None,
        n_levels: int,
        n_buffer: int,
        mode: str,
        scope: str,
        target_abs_GHz: float | None,
        target_gap_rel: float,
        g_floor_GHz: float,
        refinement: str,
        axis: str,
    ) -> ConvergenceReport:
        """Assemble the LevelVerdicts and the ConvergenceReport from refinement
        eigenvalues. Shared between verify and strict modes."""
        channel: TruncationChannel = self._convergence_truncation_channel(axis)
        cluster_threshold = settings.CONVERGENCE_CLUSTER_RATIO
        safety_factor = settings.CONVERGENCE_SAFETY_FACTOR

        # Cluster-detect on N0 (the reference / user's current spectrum).
        clusters = cutils.detect_clusters(
            evals_n0, gap_ratio_threshold=cluster_threshold
        )
        _, cluster_max_diff_n1 = cutils.cluster_safe_match_energies(
            evals_n0, evals_n1, clusters
        )

        # Per-level: assign the cluster-level estimate to every member.
        per_level_estimate = np.empty(n_levels, dtype=np.float64)
        for cluster_idx, cluster in enumerate(clusters):
            for k in cluster:
                per_level_estimate[k] = cluster_max_diff_n1[cluster_idx]

        # Strict mode: ratio test.
        ratio_per_cluster: npt.NDArray[np.float64] | None = None
        geometric_tail: npt.NDArray[np.float64] | None = None
        asymptotic_flag: npt.NDArray[np.bool_] | None = None
        if refinement == "ratio_test" and evals_n2 is not None:
            _, cluster_max_diff_n2 = cutils.cluster_safe_match_energies(
                evals_n1, evals_n2, clusters
            )
            ratio_per_cluster = np.divide(
                cluster_max_diff_n2,
                cluster_max_diff_n1,
                out=np.full_like(cluster_max_diff_n1, np.inf),
                where=cluster_max_diff_n1 > 0,
            )
            # Geometric-tail estimate: d0 / (1 - R) when R < 1.
            geometric_tail = np.where(
                ratio_per_cluster < 1.0,
                cluster_max_diff_n1 / np.clip(1.0 - ratio_per_cluster, 1e-30, None),
                np.inf,
            )
            asymptotic_flag = ratio_per_cluster < 1.0

        # If strict-mode geometric tail succeeded, use that as the
        # estimate; otherwise stick with safety_factor * d0.
        per_level_abs_err = np.empty(n_levels, dtype=np.float64)
        per_level_evidence: list[Evidence] = []
        per_level_estimator_method: list[str] = []
        per_level_warnings: list[list[str]] = [[] for _ in range(n_levels)]

        for cluster_idx, cluster in enumerate(clusters):
            if refinement == "ratio_test" and geometric_tail is not None:
                if asymptotic_flag is not None and asymptotic_flag[cluster_idx]:
                    est = float(geometric_tail[cluster_idx])
                    ev: Evidence = "calibrated"
                    method = "ratio_test"
                else:
                    est = float(safety_factor * cluster_max_diff_n1[cluster_idx])
                    ev = "verified_empirical"
                    method = "ratio_test_failed_fallback_one_step"
                    for k in cluster:
                        per_level_warnings[k].append("ratio_test_not_asymptotic")
            else:
                est = float(safety_factor * cluster_max_diff_n1[cluster_idx])
                ev = "verified_empirical"
                method = "one_step"

            for k in cluster:
                per_level_abs_err[k] = est
                per_level_evidence.append(ev)
                per_level_estimator_method.append(method)

        # Observed-gap scope: compute local isolation gaps from N0 spectrum.
        eps_gap_est: list[float | None]
        if scope == "observed_gap_scale":
            eps_gap_est = []
            for k in range(n_levels):
                gap = _local_isolation_gap(
                    evals_n0=evals_n0,
                    buffer_n0=buffer_n0,
                    k=k,
                    n_levels=n_levels,
                    g_floor=g_floor_GHz,
                )
                if gap is None:
                    # Topmost level without buffer: cannot compute upper gap.
                    eps_gap_est.append(None)
                    per_level_warnings[k].append("upper_gap_unavailable")
                else:
                    eps_gap_est.append(float(per_level_abs_err[k] / gap))
        else:
            eps_gap_est = [None] * n_levels

        # Assign status per level based on chosen scope and target.
        per_level_status: list[Status] = []
        for k in range(n_levels):
            status = _assign_status(
                abs_err_est=float(per_level_abs_err[k]),
                eps_gap_est=eps_gap_est[k],
                scope=scope,
                target_abs_GHz=target_abs_GHz,
                target_gap_rel=target_gap_rel,
            )
            # Strict mode downgrades any status weaker than verified_empirical.
            if mode == "strict" and status == "converged":
                if not evidence_at_least(per_level_evidence[k], "verified_empirical"):
                    status = "marginal"
                    per_level_warnings[k].append(
                        "strict_mode_downgrade_insufficient_evidence"
                    )
            per_level_status.append(status)

        # Build LevelVerdicts.
        per_level_verdicts: list[LevelVerdict] = []
        for k in range(n_levels):
            per_level_verdicts.append(
                LevelVerdict(
                    level_index=k,
                    status=per_level_status[k],
                    status_scope=scope,  # type: ignore[arg-type]
                    evidence=per_level_evidence[k],
                    abs_err_est_GHz=float(per_level_abs_err[k]),
                    eps_gap_est=eps_gap_est[k],
                    truncation_channel=channel,
                    estimator_method=per_level_estimator_method[k],
                    warnings=tuple(per_level_warnings[k]),
                )
            )

        # Aggregate worst status.
        worst_rank = -1
        worst_idx = 0
        for k, v in enumerate(per_level_verdicts):
            r = _status_rank(v.status)
            if r > worst_rank:
                worst_rank = r
                worst_idx = k
        aggregate_status: Status = per_level_verdicts[worst_idx].status

        # Recommendations.
        recommendations: list[str] = []
        if any(v.status == "underconverged" for v in per_level_verdicts):
            current_value = int(getattr(self, axis))
            step = self._convergence_step(axis)
            recommendations.append(
                f"increase {axis} from {current_value} to at least "
                f"{current_value + step} and re-run; the worst-level "
                f"estimate exceeded the target threshold"
            )
        if (
            scope == "absolute"
            and target_abs_GHz is None
            and any(v.status == "unverified" for v in per_level_verdicts)
        ):
            recommendations.append(
                "supply target_abs_GHz for an automatic status assignment "
                "in absolute scope; report currently exposes abs_err_est_GHz "
                "per level without thresholding"
            )

        return ConvergenceReport(
            per_level=per_level_verdicts,
            aggregate_status=aggregate_status,
            worst_level=worst_idx,
            channel_breakdown_GHz={
                channel: float(np.max(per_level_abs_err)),
            },
            clusters=clusters,
            recommendations=recommendations,
            implementation_audit=self._convergence_audit(
                n_levels=n_levels,
                n_buffer=n_buffer,
                mode=mode,
                refinement=refinement,
            ),
        )


# ------------------------------------------------------------ module helpers


_STATUS_RANK: dict[Status, int] = {
    "converged": 0,
    "likely_converged": 1,
    "marginal": 2,
    "underconverged": 3,
    "unverified": 4,
}


def _status_rank(status: Status) -> int:
    """Rank for aggregating worst per-level status; higher is worse."""
    return _STATUS_RANK[status]


def _local_isolation_gap(
    evals_n0: npt.NDArray[np.float64],
    buffer_n0: npt.NDArray[np.float64] | None,
    k: int,
    n_levels: int,
    g_floor: float,
) -> float | None:
    """Compute the local isolation gap for level ``k``.

    Per the design specification:

    - k == 0: ``E_1 - E_0``
    - 1 <= k <= n_levels - 2: ``min(E_k - E_{k-1}, E_{k+1} - E_k)``
    - k == n_levels - 1: upper gap available only if ``buffer_n0`` is
      supplied; otherwise return ``None``.

    The returned gap is floored at ``g_floor``.
    """
    if k == 0:
        if n_levels < 2:
            return None
        return max(float(evals_n0[1] - evals_n0[0]), g_floor)
    if k < n_levels - 1:
        return max(
            min(
                float(evals_n0[k] - evals_n0[k - 1]),
                float(evals_n0[k + 1] - evals_n0[k]),
            ),
            g_floor,
        )
    # k == n_levels - 1
    if buffer_n0 is None or len(buffer_n0) == 0:
        return None
    return max(
        min(
            float(evals_n0[k] - evals_n0[k - 1]),
            float(buffer_n0[0] - evals_n0[k]),
        ),
        g_floor,
    )


def _assign_status(
    abs_err_est: float,
    eps_gap_est: float | None,
    scope: str,
    target_abs_GHz: float | None,
    target_gap_rel: float,
) -> Status:
    """Assign a Status based on the configured scope and target."""
    if scope == "absolute":
        if target_abs_GHz is None:
            return "unverified"
        if abs_err_est < target_abs_GHz:
            return "converged"
        if abs_err_est < 10.0 * target_abs_GHz:
            return "marginal"
        return "underconverged"
    # observed_gap_scale
    if eps_gap_est is None:
        return "unverified"
    if eps_gap_est < target_gap_rel:
        return "converged"
    if eps_gap_est < 10.0 * target_gap_rel:
        return "marginal"
    return "underconverged"


def estimate_convergence(qubit: Any, **kwargs: Any) -> ConvergenceReport:
    """Convenience top-level shim: forwards to ``qubit.estimate_convergence(...)``.

    Raises ``TypeError`` if the qubit does not subclass
    :class:`ConvergenceCheckable`.
    """
    if not isinstance(qubit, ConvergenceCheckable):
        raise TypeError(
            f"{type(qubit).__name__} does not implement convergence checking. "
            "Subclasses of ConvergenceCheckable in this version: Transmon "
            "(further qubits land in subsequent PRs)."
        )
    return qubit.estimate_convergence(**kwargs)
