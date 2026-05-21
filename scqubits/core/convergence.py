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
import dataclasses

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

import scqubits.core.units as units
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
            If True, additionally assess the requested ``derived_quantities``
            and attach their per-level sub-reports under
            :attr:`ConvergenceReport.derived`. Requires ``mode`` of
            ``"verify"`` or ``"strict"`` -- derived quantities need a
            refinement comparison.
        derived_quantities:
            Subset of ``{"wavefunctions", "matrix_elements", "coherence"}`` to
            assess when ``include_derived`` is set.
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
        requested_derived: tuple[str, ...] = tuple(derived_quantities or ())
        if include_derived:
            if mode == "quick":
                raise ValueError(
                    "include_derived requires mode='verify' or 'strict'; "
                    "derived quantities need a refinement comparison"
                )
            if not requested_derived:
                raise ValueError(
                    "include_derived=True requires derived_quantities, e.g. "
                    "['wavefunctions', 'matrix_elements']"
                )
            unknown = set(requested_derived) - {
                "wavefunctions",
                "matrix_elements",
                "coherence",
            }
            if unknown:
                raise ValueError(
                    f"unknown derived_quantities: {sorted(unknown)}; valid: "
                    "'wavefunctions', 'matrix_elements', 'coherence'"
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
            include_derived=include_derived,
            derived_quantities=requested_derived,
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
        include_derived: bool = False,
        derived_quantities: tuple[str, ...] = (),
    ) -> ConvergenceReport:
        """Verify / strict mode: refine cutoff(s), compare cluster-matched energies.

        When ``include_derived`` is set, the refined eigenvectors (and clones,
        for matrix elements) are reused to build the requested derived per-level
        sub-reports, which are attached under ``ConvergenceReport.derived``.
        """
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
        evals_n1, evecs_n1 = clone_1.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]

        evals_n2: npt.NDArray[np.float64] | None = None
        evecs_n2: npt.NDArray[np.float64] | None = None
        clone_2: ConvergenceCheckable | None = None
        if refinement == "ratio_test":
            clone_2 = self._convergence_clone_at({axis: current_value + 2 * step})
            evals_n2, evecs_n2 = clone_2.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]

        report = self._convergence_build_energy_report(
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

        if not include_derived:
            return report

        return self._attach_derived_reports(
            report=report,
            derived_quantities=derived_quantities,
            evals_n0=evals_n0,
            evals_n1=evals_n1,
            evecs_n0=evecs_n0,
            evecs_n1=evecs_n1,
            evecs_n2=evecs_n2,
            clone_1=clone_1,
            clone_2=clone_2,
            ncut_0=int(current_value),
            step=step,
            n_levels=n_levels,
            n_buffer=n_buffer,
            mode=mode,
            target_gap_rel=target_gap_rel,
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
        safety_factor = settings.CONVERGENCE_SAFETY_FACTOR

        # Cluster-detect on N0 (the reference / user's current spectrum) and
        # measure the one-step movement N0 -> N1 per cluster.
        clusters = cutils.detect_clusters(
            evals_n0, gap_ratio_threshold=settings.CONVERGENCE_CLUSTER_RATIO
        )
        _, cluster_max_diff_n1 = cutils.cluster_safe_match_energies(
            evals_n0, evals_n1, clusters
        )

        # Strict mode: a second refinement enables the geometric ratio test.
        geometric_tail: npt.NDArray[np.float64] | None = None
        asymptotic_flag: npt.NDArray[np.bool_] | None = None
        if refinement == "ratio_test" and evals_n2 is not None:
            _, cluster_max_diff_n2 = cutils.cluster_safe_match_energies(
                evals_n1, evals_n2, clusters
            )
            _, geometric_tail, asymptotic_flag = cutils.geometric_ratio_test(
                cluster_max_diff_n1, cluster_max_diff_n2
            )

        (
            per_level_abs_err,
            per_level_evidence,
            per_level_estimator_method,
            per_level_warnings,
        ) = _per_cluster_energy_estimates(
            clusters=clusters,
            cluster_max_diff_n1=cluster_max_diff_n1,
            refinement=refinement,
            geometric_tail=geometric_tail,
            asymptotic_flag=asymptotic_flag,
            safety_factor=safety_factor,
            n_levels=n_levels,
        )

        eps_gap_est = _observed_gap_eps(
            scope=scope,
            evals_n0=evals_n0,
            buffer_n0=buffer_n0,
            n_levels=n_levels,
            g_floor_GHz=g_floor_GHz,
            per_level_abs_err=per_level_abs_err,
            per_level_warnings=per_level_warnings,
        )

        per_level_status = _assign_level_statuses(
            n_levels=n_levels,
            per_level_abs_err=per_level_abs_err,
            eps_gap_est=eps_gap_est,
            per_level_evidence=per_level_evidence,
            per_level_warnings=per_level_warnings,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
            mode=mode,
        )

        per_level_verdicts = [
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
            for k in range(n_levels)
        ]

        worst_idx, aggregate_status = _aggregate_worst(per_level_verdicts)

        recommendations = self._energy_recommendations(
            verdicts=per_level_verdicts,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            axis=axis,
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

    def _energy_recommendations(
        self,
        verdicts: list[LevelVerdict],
        scope: str,
        target_abs_GHz: float | None,
        axis: str,
    ) -> list[str]:
        """Build the next-step recommendations for an energy convergence report.

        Suggests increasing the truncation axis when any level is
        underconverged, and prompts for a ``target_abs_GHz`` when absolute-scope
        levels are unverified for lack of a target.
        """
        recommendations: list[str] = []
        if any(v.status == "underconverged" for v in verdicts):
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
            and any(v.status == "unverified" for v in verdicts)
        ):
            recommendations.append(
                "supply target_abs_GHz for an automatic status assignment "
                "in absolute scope; report currently exposes abs_err_est_GHz "
                "per level without thresholding"
            )
        return recommendations

    def _attach_derived_reports(
        self,
        report: ConvergenceReport,
        derived_quantities: tuple[str, ...],
        evals_n0: npt.NDArray[np.float64],
        evals_n1: npt.NDArray[np.float64],
        evecs_n0: npt.NDArray[np.float64],
        evecs_n1: npt.NDArray[np.float64],
        evecs_n2: npt.NDArray[np.float64] | None,
        clone_1: "ConvergenceCheckable",
        clone_2: "ConvergenceCheckable | None",
        ncut_0: int,
        step: int,
        n_levels: int,
        n_buffer: int,
        mode: str,
        target_gap_rel: float,
        refinement: str,
        axis: str,
    ) -> ConvergenceReport:
        """Build and attach the requested per-level derived sub-reports.

        Reuses the already-computed reference and refined eigenvectors (and the
        refinement clones, for matrix elements) to assess wavefunction and
        matrix-element convergence on the same per-level footing as energies,
        then returns ``report`` with its ``derived`` mapping populated.
        """
        channel = self._convergence_truncation_channel(axis)
        clusters = cutils.detect_clusters(
            evals_n0[:n_levels],
            gap_ratio_threshold=settings.CONVERGENCE_CLUSTER_RATIO,
        )
        audit = self._convergence_audit(
            n_levels=n_levels, n_buffer=n_buffer, mode=mode, refinement=refinement
        )
        ncut_1 = ncut_0 + step
        ncut_2 = ncut_0 + 2 * step
        derived: dict[str, ConvergenceReport] = {}

        if "wavefunctions" in derived_quantities:
            movement_first = _wavefunction_movement(
                evecs_n0, evecs_n1, ncut_0, ncut_1, clusters, n_levels
            )
            movement_second: npt.NDArray[np.float64] | None = None
            if refinement == "ratio_test" and evecs_n2 is not None:
                movement_second = _wavefunction_movement(
                    evecs_n1, evecs_n2, ncut_1, ncut_2, clusters, n_levels
                )
            derived["wavefunctions"] = _build_metric_report(
                movement_first=movement_first,
                movement_second=movement_second,
                channel=channel,
                estimator_method="wavefunction_overlap",
                clusters=clusters,
                n_levels=n_levels,
                mode=mode,
                target_gap_rel=target_gap_rel,
                refinement=refinement,
                audit=audit,
                axis=axis,
                current_value=ncut_0,
                step=step,
            )

        if "matrix_elements" in derived_quantities:
            me_movement_first, skipped = _matrix_element_movement(
                self, clone_1, evecs_n0, evecs_n1, n_levels
            )
            me_movement_second: npt.NDArray[np.float64] | None = None
            if (
                refinement == "ratio_test"
                and clone_2 is not None
                and evecs_n2 is not None
            ):
                me_movement_second, _ = _matrix_element_movement(
                    clone_1, clone_2, evecs_n1, evecs_n2, n_levels
                )
            derived["matrix_elements"] = _build_metric_report(
                movement_first=me_movement_first,
                movement_second=me_movement_second,
                channel=channel,
                estimator_method="matrix_element_frobenius",
                clusters=clusters,
                n_levels=n_levels,
                mode=mode,
                target_gap_rel=target_gap_rel,
                refinement=refinement,
                audit=audit,
                axis=axis,
                current_value=ncut_0,
                step=step,
                skipped=skipped,
            )

        if "coherence" in derived_quantities:
            channels = list(
                self.effective_noise_channels()  # type: ignore[attr-defined]
            ) + ["t1_effective", "t2_effective"]
            names, rate_movement, floor_flags, skipped_channels = (
                _coherence_rate_movement(
                    self,
                    clone_1,
                    (evals_n0, evecs_n0),
                    (evals_n1, evecs_n1),
                    channels,
                    settings.CONVERGENCE_RATE_FLOOR_HZ,
                )
            )
            derived["coherence"] = _build_coherence_report(
                channel_names=names,
                movement=rate_movement,
                floor_flags=floor_flags,
                channel=channel,
                target_gap_rel=target_gap_rel,
                audit=audit,
                axis=axis,
                current_value=ncut_0,
                step=step,
                skipped=skipped_channels,
            )

        return dataclasses.replace(report, derived=derived)


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


def _per_cluster_energy_estimates(
    clusters: list[tuple[int, ...]],
    cluster_max_diff_n1: npt.NDArray[np.float64],
    refinement: str,
    geometric_tail: npt.NDArray[np.float64] | None,
    asymptotic_flag: npt.NDArray[np.bool_] | None,
    safety_factor: float,
    n_levels: int,
) -> tuple[npt.NDArray[np.float64], list[Evidence], list[str], list[list[str]]]:
    """Derive each level's error estimate, evidence, estimator method, and warnings.

    Verify mode (and the strict-mode fallback) takes ``safety_factor`` times the
    one-step movement, tagged ``verified_empirical``. Strict mode uses the
    geometric-tail estimate tagged ``calibrated`` for clusters confirmed to be in
    the asymptotic regime, and otherwise falls back to the one-step estimate and
    records a ``ratio_test_not_asymptotic`` warning. Every level inherits the
    values of the cluster it belongs to.
    """
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

    return (
        per_level_abs_err,
        per_level_evidence,
        per_level_estimator_method,
        per_level_warnings,
    )


def _observed_gap_eps(
    scope: str,
    evals_n0: npt.NDArray[np.float64],
    buffer_n0: npt.NDArray[np.float64] | None,
    n_levels: int,
    g_floor_GHz: float,
    per_level_abs_err: npt.NDArray[np.float64],
    per_level_warnings: list[list[str]],
) -> list[float | None]:
    """Normalize each level's error by its local isolation gap.

    Returns ``None`` for every level outside ``observed_gap_scale`` scope.
    Within that scope, each entry is the absolute error divided by the level's
    local isolation gap; a level whose upper gap is unavailable (the topmost
    requested level without a buffer) is left as ``None`` and gets an
    ``upper_gap_unavailable`` warning.
    """
    if scope != "observed_gap_scale":
        return [None] * n_levels
    eps_gap_est: list[float | None] = []
    for k in range(n_levels):
        gap = _local_isolation_gap(
            evals_n0=evals_n0,
            buffer_n0=buffer_n0,
            k=k,
            n_levels=n_levels,
            g_floor=g_floor_GHz,
        )
        if gap is None:
            eps_gap_est.append(None)
            per_level_warnings[k].append("upper_gap_unavailable")
        else:
            eps_gap_est.append(float(per_level_abs_err[k] / gap))
    return eps_gap_est


def _assign_level_statuses(
    n_levels: int,
    per_level_abs_err: npt.NDArray[np.float64],
    eps_gap_est: list[float | None],
    per_level_evidence: list[Evidence],
    per_level_warnings: list[list[str]],
    scope: str,
    target_abs_GHz: float | None,
    target_gap_rel: float,
    mode: str,
) -> list[Status]:
    """Assign each level's Status from its error estimate and the active target.

    In strict mode a ``converged`` verdict backed by evidence weaker than
    ``verified_empirical`` is downgraded to ``marginal`` and a
    ``strict_mode_downgrade_insufficient_evidence`` warning is recorded.
    """
    per_level_status: list[Status] = []
    for k in range(n_levels):
        status = _assign_status(
            abs_err_est=float(per_level_abs_err[k]),
            eps_gap_est=eps_gap_est[k],
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
        )
        if mode == "strict" and status == "converged":
            if not evidence_at_least(per_level_evidence[k], "verified_empirical"):
                status = "marginal"
                per_level_warnings[k].append(
                    "strict_mode_downgrade_insufficient_evidence"
                )
        per_level_status.append(status)
    return per_level_status


def _aggregate_worst(verdicts: list[LevelVerdict]) -> tuple[int, Status]:
    """Return the index and Status of the worst-ranked level verdict."""
    worst_rank = -1
    worst_idx = 0
    for k, v in enumerate(verdicts):
        r = _status_rank(v.status)
        if r > worst_rank:
            worst_rank = r
            worst_idx = k
    return worst_idx, verdicts[worst_idx].status


# Reference-norm floor for relative matrix-element comparison, guarding against
# division by a vanishing (selection-rule-zero) reference row/column.
_MATELEM_REF_FLOOR = 1e-12


def _wavefunction_movement(
    evecs_a: npt.NDArray[np.float64],
    evecs_b: npt.NDArray[np.float64],
    ncut_a: int,
    ncut_b: int,
    clusters: list[tuple[int, ...]],
    n_levels: int,
) -> npt.NDArray[np.float64]:
    """Per-level wavefunction movement between two cutoffs.

    Isolated levels use the overlap deficit ``1 - |<a_k | b_k>|``; near-degenerate
    clusters use the subspace angle (assigned to every member), which is robust to
    eigenvector rotations within the block.
    """
    overlaps = cutils.wavefunction_overlap(
        evecs_a[:, :n_levels], evecs_b[:, :n_levels], ncut_a, ncut_b
    )
    movement = 1.0 - np.minimum(1.0, overlaps)
    ncut_max = max(ncut_a, ncut_b)
    a = cutils.pad_charge_basis(evecs_a[:, :n_levels], ncut_a, ncut_max)
    b = cutils.pad_charge_basis(evecs_b[:, :n_levels], ncut_b, ncut_max)
    for cluster in clusters:
        if len(cluster) > 1:
            cols = list(cluster)
            angle = cutils.subspace_angle(a[:, cols], b[:, cols])
            for k in cluster:
                movement[k] = angle
    return movement


def _matrix_element_movement(
    qubit_a: Any,
    qubit_b: Any,
    evecs_a: npt.NDArray[np.float64],
    evecs_b: npt.NDArray[np.float64],
    n_levels: int,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Per-level relative matrix-element movement between two cutoffs.

    For each operator returned by ``get_operator_names`` the ``n_levels`` x
    ``n_levels`` matrix-element table is formed at both cutoffs; level ``k`` is
    assigned the worst relative change of its matrix-element row and column,
    maximized over operators. The relative change normalizes by the refined
    table's row/column norm, floored to guard against selection-rule zeros.
    Operators that raise or return a shape-incompatible table are skipped and
    reported (graceful degradation), not silently dropped.
    """
    movement = np.zeros(n_levels, dtype=np.float64)
    skipped: list[str] = []
    for op_name in qubit_a.get_operator_names():
        try:
            m0 = np.asarray(
                qubit_a.matrixelement_table(op_name, evecs=evecs_a[:, :n_levels])
            )
            m1 = np.asarray(
                qubit_b.matrixelement_table(op_name, evecs=evecs_b[:, :n_levels])
            )
        except Exception:
            skipped.append(op_name)
            continue
        if m0.shape != (n_levels, n_levels) or m1.shape != (n_levels, n_levels):
            skipped.append(op_name)
            continue
        delta = np.abs(m1 - m0)
        for k in range(n_levels):
            row_ref = max(float(np.linalg.norm(m1[k, :])), _MATELEM_REF_FLOOR)
            col_ref = max(float(np.linalg.norm(m1[:, k])), _MATELEM_REF_FLOOR)
            rel_k = max(
                float(np.linalg.norm(delta[k, :])) / row_ref,
                float(np.linalg.norm(delta[:, k])) / col_ref,
            )
            movement[k] = max(float(movement[k]), rel_k)
    return movement, skipped


def _build_metric_report(
    movement_first: npt.NDArray[np.float64],
    movement_second: npt.NDArray[np.float64] | None,
    channel: TruncationChannel,
    estimator_method: str,
    clusters: list[tuple[int, ...]],
    n_levels: int,
    mode: str,
    target_gap_rel: float,
    refinement: str,
    audit: ImplementationAudit,
    axis: str,
    current_value: int,
    step: int,
    skipped: list[str] | None = None,
) -> ConvergenceReport:
    """Assemble a per-level ConvergenceReport for a dimensionless derived metric.

    ``movement_first`` holds each level's change between the base and first
    refinement (overlap deficit, subspace angle, or relative matrix-element
    change). In ratio-test mode ``movement_second`` (the first-to-second
    refinement change) drives a geometric extrapolation; levels confirmed
    asymptotic use the extrapolated tail and are tagged ``calibrated``, the rest
    fall back to the one-step movement with a recorded warning. Verdicts apply
    the observed-gap-scale ladder against ``target_gap_rel``.
    """
    geometric_tail: npt.NDArray[np.float64] | None = None
    asymptotic: npt.NDArray[np.bool_] | None = None
    if refinement == "ratio_test" and movement_second is not None:
        _, geometric_tail, asymptotic = cutils.geometric_ratio_test(
            movement_first, movement_second
        )

    safety_factor = settings.CONVERGENCE_SAFETY_FACTOR
    per_level_eps = np.empty(n_levels, dtype=np.float64)
    per_level_evidence: list[Evidence] = []
    per_level_method: list[str] = []
    per_level_warnings: list[list[str]] = [[] for _ in range(n_levels)]

    for k in range(n_levels):
        if geometric_tail is not None and asymptotic is not None and asymptotic[k]:
            # Geometric tail is already an extrapolated remaining-error estimate.
            per_level_eps[k] = float(geometric_tail[k])
            per_level_evidence.append("calibrated")
            per_level_method.append(f"{estimator_method}_ratio_test")
        else:
            # One-step movement is a lower bound on the remaining error; apply
            # the same safety factor the energy channel uses so a derived
            # "converged" carries the same margin.
            per_level_eps[k] = float(safety_factor * movement_first[k])
            per_level_evidence.append("verified_empirical")
            if refinement == "ratio_test":
                per_level_method.append(f"{estimator_method}_ratio_test_fallback")
                per_level_warnings[k].append("ratio_test_not_asymptotic")
            else:
                per_level_method.append(estimator_method)

    verdicts = [
        LevelVerdict(
            level_index=k,
            status=_assign_status(
                abs_err_est=0.0,
                eps_gap_est=float(per_level_eps[k]),
                scope="observed_gap_scale",
                target_abs_GHz=None,
                target_gap_rel=target_gap_rel,
            ),
            status_scope="observed_gap_scale",
            evidence=per_level_evidence[k],
            abs_err_est_GHz=None,
            eps_gap_est=float(per_level_eps[k]),
            truncation_channel=channel,
            estimator_method=per_level_method[k],
            warnings=tuple(per_level_warnings[k]),
        )
        for k in range(n_levels)
    ]
    worst_idx, aggregate_status = _aggregate_worst(verdicts)

    recommendations: list[str] = []
    if any(v.status == "underconverged" for v in verdicts):
        recommendations.append(
            f"increase {axis} from {current_value} to at least "
            f"{current_value + step} and re-run; a derived-quantity estimate "
            f"exceeded the target threshold"
        )
    if skipped:
        recommendations.append(
            "skipped operators (raised or shape-incompatible): " + ", ".join(skipped)
        )

    return ConvergenceReport(
        per_level=verdicts,
        aggregate_status=aggregate_status,
        worst_level=worst_idx,
        channel_breakdown_GHz={},
        clusters=clusters,
        recommendations=recommendations,
        implementation_audit=audit,
    )


def _coherence_rate_movement(
    qubit_a: Any,
    qubit_b: Any,
    esys_a: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    esys_b: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    channels: list[str],
    rate_floor: float,
) -> tuple[list[str], npt.NDArray[np.float64], list[bool], list[str]]:
    """Per-channel relative change of noise rates between two cutoffs.

    Each channel's rate is evaluated at both cutoffs (``get_rate=True``, reusing
    the supplied eigensystems so no extra diagonalization is needed), converted
    to Hz so that ``rate_floor`` (in Hz) is meaningful regardless of the active
    unit system, and the relative change ``|G_b - G_a| / max(|G_b|, rate_floor)``
    is recorded. A channel whose refined rate falls below ``rate_floor`` is
    flagged -- its rate sits at the noise floor, so the corresponding lifetime is
    not physically meaningful. Channels that raise are skipped and reported, not
    silently dropped.
    """
    to_hz = units.units_scale_factor()
    names: list[str] = []
    movement: list[float] = []
    floor_flags: list[bool] = []
    skipped: list[str] = []
    for channel in channels:
        try:
            rate_a = (
                abs(float(getattr(qubit_a, channel)(get_rate=True, esys=esys_a)))
                * to_hz
            )
            rate_b = (
                abs(float(getattr(qubit_b, channel)(get_rate=True, esys=esys_b)))
                * to_hz
            )
        except Exception:
            skipped.append(channel)
            continue
        names.append(channel)
        movement.append(abs(rate_b - rate_a) / max(rate_b, rate_floor))
        floor_flags.append(rate_b < rate_floor)
    return names, np.asarray(movement, dtype=np.float64), floor_flags, skipped


def _build_coherence_report(
    channel_names: list[str],
    movement: npt.NDArray[np.float64],
    floor_flags: list[bool],
    channel: TruncationChannel,
    target_gap_rel: float,
    audit: ImplementationAudit,
    axis: str,
    current_value: int,
    step: int,
    skipped: list[str],
) -> ConvergenceReport:
    """Assemble the per-noise-channel coherence convergence sub-report.

    Each verdict is one noise channel; ``eps_gap_est`` is the relative change of
    its rate between cutoffs and ``estimator_method`` names the channel. Rates
    are assessed first: a channel whose refined rate is below the floor carries a
    ``noise_floor`` warning rather than a lifetime claim. Verdicts use the
    observed-gap ladder against ``target_gap_rel``.
    """
    verdicts: list[LevelVerdict] = []
    for idx, name in enumerate(channel_names):
        warnings: tuple[str, ...] = ("noise_floor",) if floor_flags[idx] else ()
        verdicts.append(
            LevelVerdict(
                level_index=idx,
                status=_assign_status(
                    abs_err_est=0.0,
                    eps_gap_est=float(movement[idx]),
                    scope="observed_gap_scale",
                    target_abs_GHz=None,
                    target_gap_rel=target_gap_rel,
                ),
                status_scope="observed_gap_scale",
                evidence="verified_empirical",
                abs_err_est_GHz=None,
                eps_gap_est=float(movement[idx]),
                truncation_channel=channel,
                estimator_method=f"{name}_rate",
                warnings=warnings,
            )
        )

    if verdicts:
        worst_idx, aggregate_status = _aggregate_worst(verdicts)
    else:
        worst_idx, aggregate_status = 0, "unverified"

    recommendations: list[str] = []
    if any(v.status == "underconverged" for v in verdicts):
        recommendations.append(
            f"increase {axis} from {current_value} to at least "
            f"{current_value + step} and re-run; a noise-rate estimate exceeded "
            f"the target threshold"
        )
    if not verdicts:
        recommendations.append(
            "no noise-channel rates could be evaluated for this qubit"
        )
    if skipped:
        recommendations.append("skipped noise channels (raised): " + ", ".join(skipped))

    return ConvergenceReport(
        per_level=verdicts,
        aggregate_status=aggregate_status,
        worst_level=worst_idx,
        channel_breakdown_GHz={},
        clusters=[],
        recommendations=recommendations,
        implementation_audit=audit,
    )


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
