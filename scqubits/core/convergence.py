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
"""Verified-refinement convergence diagnostics for qubit classes.

A concrete qubit that adds ``ConvergenceCheckable`` to its MRO and declares
``_convergence_axes`` gains an :meth:`estimate_convergence` method returning a
:class:`~scqubits.core.convergence_report.ConvergenceReport`. The engine assesses
the energy spectrum and, on request, the derived wavefunction, matrix-element,
and coherence channels.

The refinement engine clones the qubit, bumps a cutoff axis by a step,
re-diagonalizes, and compares cluster-matched eigenvalues. Cheap-mode
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
    ParamSweepConvergence,
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

    def _convergence_axis_value(self, axis: str) -> int:
        """Return the current integer size of truncation ``axis``.

        Default reads the integer attribute named ``axis``. Qubits whose axis is
        not a plain integer override this -- e.g. a discretized grid returns its
        point count.
        """
        return int(getattr(self, axis))

    def _convergence_set_axis(
        self, clone: "ConvergenceCheckable", axis: str, value: int
    ) -> None:
        """Set truncation ``axis`` of ``clone`` to integer size ``value``.

        Default assigns the integer attribute named ``axis``. Qubits whose axis
        is not a plain integer override this -- e.g. a discretized grid rebuilds
        its ``Grid1d`` with the new point count over the same window.
        """
        setattr(clone, axis, value)

    def _convergence_step(self, axis: str) -> int:
        """Return the refinement step (axis units) for ``axis``.

        Default: ``max(4, current_value // 4)``. Concrete qubits override
        if a different heuristic better matches their convergence law.
        """
        current = self._convergence_axis_value(axis)
        return max(4, current // 4)

    def _convergence_truncation_channel(self, axis: str) -> TruncationChannel:
        """Return the physical channel label for ``axis``.

        Defaults to ``"charge_tail"``. Concrete qubits override for HO bases
        (``"HO_tail"``), finite-difference grids (``"FD_stencil"`` /
        ``"FD_box"``), etc.
        """
        return "charge_tail"

    def _convergence_richardson_order(self, axis: str) -> int | None:
        """Return the Richardson error order ``p`` for a finite-difference axis.

        An axis whose discretization error scales as ``h**p`` with the spacing
        ``h`` proportional to ``1 / (N - 1)`` (a grid refined at a fixed window)
        returns ``p`` so that strict mode verifies its asymptoticity with
        Richardson extrapolation instead of the geometric ratio test (design
        spec). Returns ``None`` (the default) for axes that converge
        geometrically -- charge and oscillator tails, and finite-box expansion.
        """
        return None

    def _convergence_boundary_diagnostic(
        self, esys: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], axis: str
    ) -> npt.NDArray[np.float64] | None:
        """Return a per-level cheap boundary-amplitude diagnostic.

        For Transmon: ``|c_{-ncut, k}|^2 + |c_{+ncut, k}|^2`` for each kept
        level ``k``. Returns ``None`` if no cheap diagnostic is available
        for this axis; quick mode then falls back to ``unverified``.
        """
        return None

    def _convergence_tail_estimate(
        self, esys: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], axis: str
    ) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]
        | None
    ):
        """Return a cheap perturbative truncation-error estimate for ``axis``.

        A qubit with a tractable dropped-space residual (e.g. a 1D charge tail)
        overrides this to return ``(estimate, perturbative_ok, boundary_prob)``
        per level: a second-order tail error estimate (GHz), a flag that the tail
        is in the perturbative regime, and the boundary probability. Quick mode
        then reports a ``perturbative`` estimate instead of a bare diagnostic.
        Returns ``None`` (the default) when no such estimate is available, in
        which case quick mode falls back to the boundary diagnostic.
        """
        return None

    def _convergence_pad_eigenvectors(
        self,
        evecs: npt.NDArray[np.float64],
        value_from: int,
        value_to: int,
    ) -> npt.NDArray[np.float64]:
        """Embed eigenvectors from the basis at one truncation value into a larger one.

        Used by the wavefunction channel to bring two eigenvector sets, computed
        at different cutoffs, into a common basis before comparison. The
        embedding is basis-specific (charge bases pad symmetrically, a harmonic
        oscillator pads at the high-Fock end), so a qubit must override this to
        support the wavefunction channel; the default signals that the channel is
        unavailable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement eigenvector padding, so "
            "the wavefunction convergence channel is unavailable for it."
        )

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

    def estimate_convergence_vs_paramvals(
        self,
        param_name: str,
        param_vals: npt.NDArray[np.float64],
        sample: int | None = 5,
        **kwargs: Any,
    ) -> ParamSweepConvergence:
        """Assess convergence across a swept parameter, returning the worst case.

        A single :meth:`estimate_convergence` call certifies only the current
        parameter set. A plot such as ``plot_evals_vs_paramvals`` instead sweeps
        a parameter at a fixed cutoff, and truncation convergence can vary across
        that range (e.g. fluxonium near half flux). This runs the per-point check
        at sampled values of ``param_name`` and reports the worst case -- the
        value at which the chosen cutoff is least trustworthy.

        Parameters
        ----------
        param_name:
            Name of the swept qubit parameter (e.g. ``"flux"``), as used by
            ``get_spectrum_vs_paramvals``.
        param_vals:
            The parameter values of the intended sweep.
        sample:
            Number of values to actually check, spread across the range and
            always including both endpoints (the usual worst case). ``None``
            checks every value in ``param_vals``. Sampling keeps the check cheap
            relative to a full per-point sweep.
        **kwargs:
            Forwarded to :meth:`estimate_convergence` (e.g. ``n_levels``,
            ``mode``, ``target_abs_GHz``, ``scope``).

        Returns
        -------
        :class:`~scqubits.core.convergence_report.ParamSweepConvergence`
        """
        values = np.asarray(param_vals, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("param_vals must be a non-empty 1D array")
        if sample is None or sample >= values.size:
            indices = list(range(values.size))
        else:
            if sample < 2:
                raise ValueError("sample must be at least 2 (the two endpoints)")
            indices = sorted(
                set(np.linspace(0, values.size - 1, sample).round().astype(int))
            )
        sampled_vals = [float(values[i]) for i in indices]

        original = getattr(self, param_name)
        reports: list[ConvergenceReport] = []
        try:
            for value in sampled_vals:
                setattr(self, param_name, value)
                reports.append(self.estimate_convergence(**kwargs))
        finally:
            setattr(self, param_name, original)

        worst_index = max(
            range(len(reports)),
            key=lambda i: _status_rank(reports[i].aggregate_status),
        )
        return ParamSweepConvergence(
            param_name=param_name,
            param_vals=sampled_vals,
            reports=reports,
            worst_index=worst_index,
            aggregate_status=reports[worst_index].aggregate_status,
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
            self._convergence_set_axis(clone, axis, value)
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
            axis: self._convergence_axis_value(axis) for axis in self._convergence_axes
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
        """Quick-mode report: cheap perturbative tail estimate or boundary diagnostic.

        If the qubit supplies a perturbative tail estimate (e.g. the charge
        finite-tail), quick mode reports it with ``perturbative`` evidence;
        otherwise it falls back to the bare boundary-amplitude diagnostic. Per the
        published design, quick mode never returns an unqualified ``converged`` --
        the best it can say is ``likely_converged``.
        """
        # Quick mode assesses the first (dominant) axis without refinement.
        axis = self._convergence_axes[0]
        channel: TruncationChannel = self._convergence_truncation_channel(axis)

        tail = self._convergence_tail_estimate((evals_n0, evecs_n0), axis)
        if tail is not None:
            return self._convergence_quick_perturbative(
                tail=tail,
                channel=channel,
                evals_n0=evals_n0,
                n_levels=n_levels,
                scope=scope,
                target_abs_GHz=target_abs_GHz,
                target_gap_rel=target_gap_rel,
                g_floor_GHz=g_floor_GHz,
                axis=axis,
            )

        boundary_amplitudes = self._convergence_boundary_diagnostic(
            (evals_n0, evecs_n0), axis
        )

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

    def _convergence_quick_perturbative(
        self,
        tail: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]
        ],
        channel: TruncationChannel,
        evals_n0: npt.NDArray[np.float64],
        n_levels: int,
        scope: str,
        target_abs_GHz: float | None,
        target_gap_rel: float,
        g_floor_GHz: float,
        axis: str,
    ) -> ConvergenceReport:
        """Quick-mode report from a perturbative tail estimate.

        Reports the per-level tail estimate with ``perturbative`` evidence and
        the usual status ladder, but caps ``converged`` at ``likely_converged``
        (quick mode never makes an unqualified convergence claim). A level whose
        boundary probability is large, or whose tail is not in the perturbative
        regime, is forced to ``underconverged`` / ``unverified`` regardless of the
        estimate (design-spec edge cases).
        """
        estimate, perturbative_ok, boundary_prob = tail
        large_boundary_prob = _BOUNDARY_PROBABILITY_LARGE
        # The perturbative tail estimate is a lower bound (it omits higher-order
        # and far-tail contributions); apply the same safety factor the refinement
        # modes use so a quick verdict carries a comparable margin.
        safety_factor = settings.CONVERGENCE_SAFETY_FACTOR

        per_level: list[LevelVerdict] = []
        for k in range(n_levels):
            warnings: list[str] = []
            est = safety_factor * float(estimate[k])
            gap = (
                _local_isolation_gap(evals_n0, None, k, n_levels, g_floor_GHz)
                if scope == "observed_gap_scale"
                else None
            )
            eps = est / gap if gap is not None else None
            if scope == "observed_gap_scale" and gap is None:
                warnings.append("upper_gap_unavailable")

            if float(boundary_prob[k]) > large_boundary_prob:
                status: Status = "underconverged"
                evidence: Evidence = "perturbative"
                warnings.append("boundary_probability_large")
            elif not bool(perturbative_ok[k]):
                status = "unverified"
                evidence = "unverified"
                warnings.append("tail_not_perturbative")
            elif scope == "absolute" and target_abs_GHz is None:
                # No absolute target: fall back to a target-free gut check -- a
                # small perturbative tail with little boundary support is
                # likely_converged (never an unqualified converged).
                status = "likely_converged"
                evidence = "perturbative"
            else:
                status = _assign_status(est, eps, scope, target_abs_GHz, target_gap_rel)
                evidence = "perturbative"
                if status == "converged":
                    # Quick mode never makes an unqualified convergence claim.
                    status = "likely_converged"

            per_level.append(
                LevelVerdict(
                    level_index=k,
                    status=status,
                    status_scope=scope,  # type: ignore[arg-type]
                    evidence=evidence,
                    abs_err_est_GHz=est,
                    eps_gap_est=eps,
                    truncation_channel=channel,
                    estimator_method="finite_tail_resolvent",
                    warnings=tuple(warnings),
                )
            )

        worst_idx, aggregate_status = _aggregate_worst(per_level)
        recommendations: list[str] = []
        if any(v.status in ("underconverged", "unverified") for v in per_level):
            current = self._convergence_axis_value(axis)
            step = self._convergence_step(axis)
            recommendations.append(
                f"increase {axis} from {current} to at least {current + step}, or "
                f"re-run with mode='verify' for a refinement-verified estimate"
            )

        return ConvergenceReport(
            per_level=per_level,
            aggregate_status=aggregate_status,
            worst_level=worst_idx,
            channel_breakdown_GHz={},
            clusters=[(k,) for k in range(n_levels)],
            recommendations=recommendations,
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
        axes = self._convergence_axes
        n_eigs = n_levels + n_buffer
        safety_factor = settings.CONVERGENCE_SAFETY_FACTOR
        clusters = cutils.detect_clusters(
            evals_n0[:n_levels], gap_ratio_threshold=settings.CONVERGENCE_CLUSTER_RATIO
        )
        n_clusters = len(clusters)

        # Refine each truncation axis independently (bump it, hold the others)
        # and sum the per-axis movements -- a triangle-inequality upper bound on
        # the combined truncation error.
        combined_diff_n1 = np.zeros(n_clusters, dtype=np.float64)
        combined_diff_n2: npt.NDArray[np.float64] | None = (
            np.zeros(n_clusters, dtype=np.float64)
            if refinement == "ratio_test"
            else None
        )
        # When a finite-difference (Richardson) axis is present, the strict-mode
        # estimate must be formed per-axis (Richardson for the FD-stencil channel,
        # geometric elsewhere) and summed -- a single geometric test on the
        # summed movements would misjudge the h**p stencil channel. Non-FD qubits
        # and verify mode keep the summed-movement path unchanged.
        use_richardson = refinement == "ratio_test" and any(
            self._convergence_richardson_order(a) is not None for a in axes
        )
        combined_estimate: npt.NDArray[np.float64] | None = (
            np.zeros(n_clusters, dtype=np.float64) if use_richardson else None
        )
        combined_non_asymptotic: npt.NDArray[np.bool_] | None = (
            np.zeros(n_clusters, dtype=np.bool_) if use_richardson else None
        )
        channel_breakdown: dict[str, float] = {}
        per_axis_weight: dict[str, float] = {}
        # Keep each axis's first-refinement clone/eigensystem so the derived
        # channels can reuse them without re-diagonalizing.
        axis_refinements: dict[str, tuple[Any, ...]] = {}

        for axis in axes:
            step = self._convergence_step(axis)
            current = self._convergence_axis_value(axis)
            clone_1 = self._convergence_clone_at({axis: current + step})
            evals_a1, evecs_a1 = clone_1.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]
            _, diff_n1 = cutils.cluster_safe_match_energies(
                evals_n0[:n_levels], evals_a1[:n_levels], clusters
            )
            combined_diff_n1 += diff_n1

            clone_2: ConvergenceCheckable | None = None
            evecs_a2: npt.NDArray[np.float64] | None = None
            diff_n2: npt.NDArray[np.float64] | None = None
            if refinement == "ratio_test":
                clone_2 = self._convergence_clone_at({axis: current + 2 * step})
                evals_a2, evecs_a2 = clone_2.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]
                _, diff_n2 = cutils.cluster_safe_match_energies(
                    evals_a1[:n_levels], evals_a2[:n_levels], clusters
                )
                assert combined_diff_n2 is not None
                combined_diff_n2 += diff_n2

            axis_channel = self._convergence_truncation_channel(axis)
            if use_richardson and diff_n2 is not None:
                # Per-axis absolute estimate (Richardson for an h**p FD-stencil
                # axis, geometric otherwise), summed via the triangle inequality.
                order = self._convergence_richardson_order(axis)
                if order is not None:
                    est_raw, asymptotic = cutils.richardson_estimate(
                        diff_n1,
                        diff_n2,
                        current,
                        current + step,
                        current + 2 * step,
                        order,
                    )
                else:
                    _, est_raw, asymptotic = cutils.geometric_ratio_test(
                        diff_n1, diff_n2
                    )
                est_axis = np.where(
                    asymptotic, est_raw, safety_factor * diff_n1
                ).astype(np.float64)
                non_asymptotic_axis = (~asymptotic) & (diff_n1 > 0.0)
                assert combined_estimate is not None
                assert combined_non_asymptotic is not None
                combined_estimate += est_axis
                combined_non_asymptotic |= non_asymptotic_axis
                axis_weight = float(np.max(est_axis)) if n_clusters else 0.0
            else:
                axis_weight = (
                    safety_factor * float(np.max(diff_n1)) if n_clusters else 0.0
                )
            channel_breakdown[axis_channel] = (
                channel_breakdown.get(axis_channel, 0.0) + axis_weight
            )
            per_axis_weight[axis] = axis_weight
            axis_refinements[axis] = (
                clone_1,
                evals_a1,
                evecs_a1,
                clone_2,
                evecs_a2,
                current,
                step,
            )

        multi_axis = len(axes) > 1
        per_level_channel: TruncationChannel = (
            "composite_coupling"
            if multi_axis
            else self._convergence_truncation_channel(axes[0])
        )
        dominant_axis = max(per_axis_weight, key=lambda a: per_axis_weight[a])

        # Flag levels whose kept eigenstate reaches a basis boundary so strongly
        # that the dropped tail is non-perturbative -- a reliable underconvergence
        # signal independent of the refinement movement (design spec). Uses each
        # axis's cheap boundary diagnostic on the base eigensystem.
        boundary_large = [False] * n_levels
        for axis in axes:
            diagnostic = self._convergence_boundary_diagnostic(
                (evals_n0, evecs_n0), axis
            )
            if diagnostic is None:
                continue
            for k in range(n_levels):
                if float(diagnostic[k]) > _BOUNDARY_PROBABILITY_LARGE:
                    boundary_large[k] = True

        report = self._convergence_build_energy_report(
            evals_n0=evals_n0[:n_levels],
            buffer_n0=evals_n0[n_levels:] if n_buffer > 0 else None,
            clusters=clusters,
            cluster_max_diff_n1=combined_diff_n1,
            cluster_max_diff_n2=combined_diff_n2,
            channel=per_level_channel,
            channel_breakdown=channel_breakdown,
            n_levels=n_levels,
            n_buffer=n_buffer,
            mode=mode,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
            g_floor_GHz=g_floor_GHz,
            refinement=refinement,
            dominant_axis=dominant_axis,
            precomputed_estimate=combined_estimate,
            precomputed_non_asymptotic=combined_non_asymptotic,
            boundary_large=boundary_large,
        )

        if not include_derived:
            return report

        if multi_axis:
            return self._attach_derived_reports_multiaxis(
                report=report,
                derived_quantities=derived_quantities,
                evals_n0=evals_n0,
                evecs_n0=evecs_n0,
                axis_refinements=axis_refinements,
                clusters=clusters,
                n_levels=n_levels,
                n_buffer=n_buffer,
                mode=mode,
                target_gap_rel=target_gap_rel,
                dominant_axis=dominant_axis,
            )

        axis = axes[0]
        clone_1, evals_a1, evecs_a1, clone_2, evecs_a2, current, step = (
            axis_refinements[axis]
        )
        return self._attach_derived_reports(
            report=report,
            derived_quantities=derived_quantities,
            evals_n0=evals_n0,
            evals_n1=evals_a1,
            evecs_n0=evecs_n0,
            evecs_n1=evecs_a1,
            evecs_n2=evecs_a2,
            clone_1=clone_1,
            clone_2=clone_2,
            ncut_0=int(current),
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
        buffer_n0: npt.NDArray[np.float64] | None,
        clusters: list[tuple[int, ...]],
        cluster_max_diff_n1: npt.NDArray[np.float64],
        cluster_max_diff_n2: npt.NDArray[np.float64] | None,
        channel: TruncationChannel,
        channel_breakdown: dict[str, float],
        n_levels: int,
        n_buffer: int,
        mode: str,
        scope: str,
        target_abs_GHz: float | None,
        target_gap_rel: float,
        g_floor_GHz: float,
        refinement: str,
        dominant_axis: str,
        precomputed_estimate: npt.NDArray[np.float64] | None = None,
        precomputed_non_asymptotic: npt.NDArray[np.bool_] | None = None,
        boundary_large: list[bool] | None = None,
    ) -> ConvergenceReport:
        """Assemble the LevelVerdicts and ConvergenceReport from per-cluster
        movements.

        Shared by single- and multi-axis refinement. The caller supplies the
        cluster partition and the per-cluster movements (already summed over axes
        for the multi-axis case), the per-level ``channel``
        (``"composite_coupling"`` when multiple axes contribute), and the
        per-channel ``channel_breakdown``.

        When ``precomputed_estimate`` is given (a finite-difference axis used the
        Richardson estimator), it is the per-cluster absolute error already summed
        over axes, and ``precomputed_non_asymptotic`` flags clusters whose ratio
        or Richardson test failed; the geometric ratio test on the summed
        movements is then bypassed. ``boundary_large`` flags levels that reach a
        basis boundary so strongly that the dropped tail is non-perturbative; they
        receive a ``boundary_probability_large`` warning.
        """
        safety_factor = settings.CONVERGENCE_SAFETY_FACTOR

        # Strict mode without a Richardson axis: a second refinement enables the
        # geometric ratio test on the summed movements.
        geometric_tail: npt.NDArray[np.float64] | None = None
        asymptotic_flag: npt.NDArray[np.bool_] | None = None
        if (
            refinement == "ratio_test"
            and cluster_max_diff_n2 is not None
            and precomputed_estimate is None
        ):
            _, geometric_tail, asymptotic_flag = cutils.geometric_ratio_test(
                cluster_max_diff_n1, cluster_max_diff_n2
            )

        (
            per_level_abs_err,
            per_level_evidence,
            per_level_estimator_method,
            per_level_warnings,
            per_level_non_asymptotic,
        ) = _per_cluster_energy_estimates(
            clusters=clusters,
            cluster_max_diff_n1=cluster_max_diff_n1,
            refinement=refinement,
            geometric_tail=geometric_tail,
            asymptotic_flag=asymptotic_flag,
            safety_factor=safety_factor,
            n_levels=n_levels,
            precomputed_estimate=precomputed_estimate,
            precomputed_non_asymptotic=precomputed_non_asymptotic,
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
            non_asymptotic=per_level_non_asymptotic,
        )

        # A level reaching a basis boundary has a non-perturbative dropped tail:
        # surface it so the recommendation's reference is concrete.
        if boundary_large is not None:
            for k in range(n_levels):
                if boundary_large[k]:
                    per_level_warnings[k].append("boundary_probability_large")

        # Per-level transition-error estimates: for a transition k -> j the
        # triangle inequality gives the bound errhat_k + errhat_j (design spec).
        per_level_verdicts = [
            LevelVerdict(
                level_index=k,
                status=per_level_status[k],
                status_scope=scope,  # type: ignore[arg-type]
                evidence=per_level_evidence[k],
                abs_err_est_GHz=float(per_level_abs_err[k]),
                eps_gap_est=eps_gap_est[k],
                transition_err_est_GHz={
                    (k, j): float(per_level_abs_err[k] + per_level_abs_err[j])
                    for j in range(n_levels)
                    if j != k
                },
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
            axis=dominant_axis,
        )

        return ConvergenceReport(
            per_level=per_level_verdicts,
            aggregate_status=aggregate_status,
            worst_level=worst_idx,
            channel_breakdown_GHz=channel_breakdown,
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
        """Build channel-specific next-step recommendations for an energy report.

        When a level is underconverged, the advice is tailored to the dominant
        truncation channel (design spec): a charge tail grows ``ncut``, an HO
        tail grows the oscillator cutoff (or changes backend), an FD box must be
        enlarged rather than merely refined, and an FD stencil is refined at a
        fixed box. Levels carrying the ``boundary_probability_large`` warning are
        called out as having a non-perturbative tail. Also flags near-degenerate
        clusters whose labels are ambiguous, and prompts for a ``target_abs_GHz``
        when absolute-scope levels are unverified for lack of a target.
        """
        recommendations: list[str] = []
        if any(v.status == "underconverged" for v in verdicts):
            current_value = self._convergence_axis_value(axis)
            step = self._convergence_step(axis)
            channel = self._convergence_truncation_channel(axis)
            bump = (
                f"increase {axis} from {current_value} to at least "
                f"{current_value + step}"
            )
            if channel == "charge_tail":
                recommendations.append(
                    f"charge-basis tail dominates: {bump} (charge cutoff) and re-run"
                )
            elif channel == "HO_tail":
                recommendations.append(
                    f"oscillator tail dominates: {bump} (oscillator cutoff) and "
                    f"re-run; if the matrix-function backend shows instability, "
                    f"prefer an alternative representation or backend instead"
                )
            elif channel == "FD_box":
                recommendations.append(
                    "finite box dominates: enlarge the coordinate window (a wider "
                    "box at comparable grid spacing); adding grid points at the "
                    "same window cannot fix a box error"
                )
            elif channel == "FD_stencil":
                recommendations.append(
                    f"grid spacing dominates: {bump} (grid points at a fixed "
                    f"window) or raise the stencil order, then re-run"
                )
            else:
                recommendations.append(
                    f"{bump} and re-run; the worst-level estimate exceeded the "
                    f"target threshold"
                )
        flagged = [
            v.level_index
            for v in verdicts
            if "boundary_probability_large" in v.warnings
        ]
        if flagged:
            recommendations.append(
                f"levels {flagged} carry the 'boundary_probability_large' warning: "
                f"the kept state reaches the basis boundary, so the dropped tail is "
                f"non-perturbative -- increase the cutoff aggressively"
            )
        if any(
            v.status == "underconverged" and "cluster_index_ambiguity" in v.warnings
            for v in verdicts
        ):
            recommendations.append(
                "an underconverged level lies in a near-degenerate cluster; "
                "compare clusters as sets (cluster-safe matching) rather than by "
                "individual level labels, and refine the dominant cutoff"
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
            movement_first = self._convergence_wavefunction_movement(
                evecs_n0, evecs_n1, ncut_0, ncut_1, clusters, n_levels
            )
            movement_second: npt.NDArray[np.float64] | None = None
            if refinement == "ratio_test" and evecs_n2 is not None:
                movement_second = self._convergence_wavefunction_movement(
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

    def _convergence_wavefunction_movement(
        self,
        evecs_a: npt.NDArray[np.float64],
        evecs_b: npt.NDArray[np.float64],
        value_a: int,
        value_b: int,
        clusters: list[tuple[int, ...]],
        n_levels: int,
    ) -> npt.NDArray[np.float64]:
        """Per-level wavefunction movement between two cutoffs.

        Both eigenvector sets are embedded into the larger basis via the
        qubit's :meth:`_convergence_pad_eigenvectors` hook. Isolated levels use
        the overlap deficit ``1 - |<a_k | b_k>|`` (invariant to each vector's
        global phase); near-degenerate clusters use the subspace angle, assigned
        to every member, which is robust to eigenvector rotations within the
        block.
        """
        value_max = max(value_a, value_b)
        a = self._convergence_pad_eigenvectors(
            evecs_a[:, :n_levels], value_a, value_max
        )
        b = self._convergence_pad_eigenvectors(
            evecs_b[:, :n_levels], value_b, value_max
        )
        movement = np.empty(n_levels, dtype=np.float64)
        for k in range(n_levels):
            movement[k] = 1.0 - min(1.0, abs(complex(np.vdot(a[:, k], b[:, k]))))
        for cluster in clusters:
            if len(cluster) > 1:
                cols = list(cluster)
                angle = cutils.subspace_angle(a[:, cols], b[:, cols])
                for k in cluster:
                    movement[k] = angle
        return movement

    def _attach_derived_reports_multiaxis(
        self,
        report: ConvergenceReport,
        derived_quantities: tuple[str, ...],
        evals_n0: npt.NDArray[np.float64],
        evecs_n0: npt.NDArray[np.float64],
        axis_refinements: dict[str, tuple[Any, ...]],
        clusters: list[tuple[int, ...]],
        n_levels: int,
        n_buffer: int,
        mode: str,
        target_gap_rel: float,
        dominant_axis: str,
    ) -> ConvergenceReport:
        """Attach derived sub-reports for a multi-axis qubit.

        Each derived metric is measured against every axis's first refinement and
        the per-axis movements are summed (triangle-inequality bound), mirroring
        the energy channel. Matrix elements and coherence are basis-agnostic and
        supported here; wavefunctions require axis-aware tensor padding and are
        not yet available for multi-axis qubits.
        """
        if "wavefunctions" in derived_quantities:
            raise NotImplementedError(
                "wavefunction convergence for multi-axis qubits is not yet "
                "available; request matrix_elements and/or coherence."
            )
        channel: TruncationChannel = "composite_coupling"
        audit = self._convergence_audit(
            n_levels=n_levels, n_buffer=n_buffer, mode=mode, refinement="one_step"
        )
        dom_current = self._convergence_axis_value(dominant_axis)
        dom_step = self._convergence_step(dominant_axis)
        derived: dict[str, ConvergenceReport] = {}

        if "matrix_elements" in derived_quantities:
            combined = np.zeros(n_levels, dtype=np.float64)
            skipped: set[str] = set()
            for refinement_data in axis_refinements.values():
                clone_axis, evecs_axis = refinement_data[0], refinement_data[2]
                movement, axis_skipped = _matrix_element_movement(
                    self, clone_axis, evecs_n0, evecs_axis, n_levels
                )
                combined += movement
                skipped.update(axis_skipped)
            derived["matrix_elements"] = _build_metric_report(
                movement_first=combined,
                movement_second=None,
                channel=channel,
                estimator_method="matrix_element_frobenius",
                clusters=clusters,
                n_levels=n_levels,
                mode=mode,
                target_gap_rel=target_gap_rel,
                refinement="one_step",
                audit=audit,
                axis=dominant_axis,
                current_value=dom_current,
                step=dom_step,
                skipped=sorted(skipped),
            )

        if "coherence" in derived_quantities:
            channels = list(
                self.effective_noise_channels()  # type: ignore[attr-defined]
            ) + ["t1_effective", "t2_effective"]
            combined_by_channel: dict[str, float] = {}
            floor_by_channel: dict[str, bool] = {}
            skipped_channels: set[str] = set()
            for refinement_data in axis_refinements.values():
                clone_axis, evals_axis, evecs_axis = refinement_data[:3]
                names, movement, floor_flags, axis_skipped = _coherence_rate_movement(
                    self,
                    clone_axis,
                    (evals_n0, evecs_n0),
                    (evals_axis, evecs_axis),
                    channels,
                    settings.CONVERGENCE_RATE_FLOOR_HZ,
                )
                skipped_channels.update(axis_skipped)
                for idx, name in enumerate(names):
                    combined_by_channel[name] = combined_by_channel.get(
                        name, 0.0
                    ) + float(movement[idx])
                    floor_by_channel[name] = floor_by_channel.get(name, False) or bool(
                        floor_flags[idx]
                    )
            final_names = list(combined_by_channel)
            final_movement = np.asarray(
                [combined_by_channel[name] for name in final_names], dtype=np.float64
            )
            final_floor = [floor_by_channel[name] for name in final_names]
            derived["coherence"] = _build_coherence_report(
                channel_names=final_names,
                movement=final_movement,
                floor_flags=final_floor,
                channel=channel,
                target_gap_rel=target_gap_rel,
                audit=audit,
                axis=dominant_axis,
                current_value=dom_current,
                step=dom_step,
                skipped=sorted(skipped_channels - set(final_names)),
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
    precomputed_estimate: npt.NDArray[np.float64] | None = None,
    precomputed_non_asymptotic: npt.NDArray[np.bool_] | None = None,
) -> tuple[
    npt.NDArray[np.float64], list[Evidence], list[str], list[list[str]], list[bool]
]:
    """Derive each level's error estimate, evidence, estimator method, and warnings.

    A one-step estimate (verify mode, or the strict-mode fallback) takes
    ``safety_factor`` times the one-step movement; with the spec-recommended
    calibrated safety factor this is ``verified_empirical`` (design spec: a
    single-step comparison becomes ``verified_empirical`` once the safety factor
    is calibrated in the regime). A successful two-step ratio test is also
    ``verified_empirical`` -- it adds the asymptoticity check the spec attaches to
    that label -- and reports the geometric-tail estimate. When the ratio test
    finds ``R >= 1`` (no reliable asymptotic regime), the level is flagged
    non-asymptotic so the status logic can refuse a softened verdict. Every level
    inherits the values of the cluster it belongs to.

    When ``precomputed_estimate`` is supplied (a finite-difference axis used the
    Richardson estimator), it is the per-cluster absolute error already summed
    over axes and ``precomputed_non_asymptotic`` flags clusters whose per-axis
    test failed; these are used directly instead of the geometric path.

    The fifth returned element flags, per level, a failed asymptoticity test on a
    non-zero movement; the status logic forces such levels to ``underconverged``
    rather than ``marginal``.
    """
    per_level_abs_err = np.empty(n_levels, dtype=np.float64)
    per_level_evidence: list[Evidence] = []
    per_level_estimator_method: list[str] = []
    per_level_warnings: list[list[str]] = [[] for _ in range(n_levels)]
    per_level_non_asymptotic: list[bool] = [False] * n_levels

    for cluster_idx, cluster in enumerate(clusters):
        non_asymptotic = False
        if precomputed_estimate is not None:
            # Strict mode with a finite-difference axis: the per-axis estimate
            # (Richardson for the FD-stencil channel, geometric elsewhere) has
            # already been summed over axes.
            est = float(precomputed_estimate[cluster_idx])
            ev: Evidence = "verified_empirical"
            non_asymptotic = bool(
                precomputed_non_asymptotic is not None
                and precomputed_non_asymptotic[cluster_idx]
            )
            if non_asymptotic:
                method = "richardson_composite_not_asymptotic"
                for k in cluster:
                    per_level_warnings[k].append("ratio_test_not_asymptotic")
            else:
                method = "richardson_composite"
        elif refinement == "ratio_test" and geometric_tail is not None:
            d0 = float(cluster_max_diff_n1[cluster_idx])
            if asymptotic_flag is not None and asymptotic_flag[cluster_idx]:
                # Ratio test confirmed the geometric (asymptotic) regime: the
                # asymptoticity check is what makes this verified_empirical.
                est = float(geometric_tail[cluster_idx])
                ev = "verified_empirical"
                method = "ratio_test"
            elif d0 == 0.0:
                # No movement across the first refinement: already stable, so the
                # ratio (0/0) is undefined rather than a failed asymptoticity test.
                est = 0.0
                ev = "verified_empirical"
                method = "ratio_test"
            else:
                # R >= 1: not a reliable asymptotic regime. The one-step movement
                # is only a lower bound and the geometric extrapolation is
                # inapplicable; flag the level so it cannot be reported converged
                # or merely marginal.
                est = float(safety_factor * d0)
                ev = "verified_empirical"
                method = "ratio_test_failed_fallback_one_step"
                non_asymptotic = True
                for k in cluster:
                    per_level_warnings[k].append("ratio_test_not_asymptotic")
        else:
            est = float(safety_factor * cluster_max_diff_n1[cluster_idx])
            ev = "verified_empirical"
            method = "one_step"

        # A multi-level (near-degenerate) cluster is compared as a sorted set,
        # so individual level labels inside it are not reliable. Flag every
        # member (design spec).
        cluster_ambiguous = len(cluster) > 1

        for k in cluster:
            per_level_abs_err[k] = est
            per_level_evidence.append(ev)
            per_level_estimator_method.append(method)
            per_level_non_asymptotic[k] = non_asymptotic
            if cluster_ambiguous:
                per_level_warnings[k].append("cluster_index_ambiguity")

    return (
        per_level_abs_err,
        per_level_evidence,
        per_level_estimator_method,
        per_level_warnings,
        per_level_non_asymptotic,
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
    non_asymptotic: list[bool],
) -> list[Status]:
    """Assign each level's Status from its error estimate and the active target.

    In strict mode a ``converged`` verdict backed by evidence weaker than
    ``verified_empirical`` is downgraded to ``marginal`` and a
    ``strict_mode_downgrade_insufficient_evidence`` warning is recorded.

    A level whose two-step ratio test failed (``R >= 1`` on a non-zero movement,
    flagged in ``non_asymptotic``) cannot be reported as ``marginal``: the design
    spec requires ``underconverged`` or ``unverified``, since the error estimate
    is not trustworthy. Such a level is forced to ``underconverged`` and its
    evidence is downgraded to ``unverified``. A level the ladder already rates
    ``converged`` (estimate below target -- a stable, noise-floor level) is left
    intact, carrying only its ``ratio_test_not_asymptotic`` warning.
    """
    per_level_status: list[Status] = []
    for k in range(n_levels):
        status: Status = _assign_status(
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
        if non_asymptotic[k] and status in ("marginal", "underconverged"):
            status = "underconverged"
            per_level_evidence[k] = "unverified"
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

# Boundary probability above which the kept eigenstate reaches the basis edge so
# strongly that the dropped tail is non-perturbative; the level then carries a
# ``boundary_probability_large`` warning regardless of any perturbative estimate.
_BOUNDARY_PROBABILITY_LARGE = 1e-3


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
    Matrix elements are compared by magnitude, since each eigenvector's global
    phase is a gauge choice that can differ between cutoffs (a signed comparison
    would report spurious movement from phase flips). Operators that raise or
    return a shape-incompatible table are skipped and reported (graceful
    degradation), not silently dropped.
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
        delta = np.abs(np.abs(m1) - np.abs(m0))
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
    asymptotic use the extrapolated tail (tagged ``verified_empirical`` -- the
    asymptoticity check is what earns that label), the rest fall back to the
    one-step movement with a recorded warning. A failed ratio test (``R >= 1`` on
    a non-zero movement) cannot be reported as ``marginal``: the level is forced
    to ``underconverged`` with ``unverified`` evidence, mirroring the energy
    channel. Verdicts apply the observed-gap-scale ladder against
    ``target_gap_rel``.
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
    per_level_non_asymptotic: list[bool] = [False] * n_levels

    for k in range(n_levels):
        if geometric_tail is not None and asymptotic is not None and asymptotic[k]:
            # Geometric tail is already an extrapolated remaining-error estimate;
            # the asymptoticity check makes it verified_empirical.
            per_level_eps[k] = float(geometric_tail[k])
            per_level_evidence.append("verified_empirical")
            per_level_method.append(f"{estimator_method}_ratio_test")
        elif refinement == "ratio_test" and float(movement_first[k]) == 0.0:
            # No movement: already stable, the ratio is undefined rather than a
            # failed asymptoticity test.
            per_level_eps[k] = 0.0
            per_level_evidence.append("verified_empirical")
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
                per_level_non_asymptotic[k] = True
            else:
                per_level_method.append(estimator_method)

    verdicts: list[LevelVerdict] = []
    for k in range(n_levels):
        status: Status = _assign_status(
            abs_err_est=0.0,
            eps_gap_est=float(per_level_eps[k]),
            scope="observed_gap_scale",
            target_abs_GHz=None,
            target_gap_rel=target_gap_rel,
        )
        evidence: Evidence = per_level_evidence[k]
        if per_level_non_asymptotic[k] and status in ("marginal", "underconverged"):
            status = "underconverged"
            evidence = "unverified"
        verdicts.append(
            LevelVerdict(
                level_index=k,
                status=status,
                status_scope="observed_gap_scale",
                evidence=evidence,
                abs_err_est_GHz=None,
                eps_gap_est=float(per_level_eps[k]),
                truncation_channel=channel,
                estimator_method=per_level_method[k],
                warnings=tuple(per_level_warnings[k]),
            )
        )
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
    safety_factor = settings.CONVERGENCE_SAFETY_FACTOR
    verdicts: list[LevelVerdict] = []
    for idx, name in enumerate(channel_names):
        warnings: tuple[str, ...] = ("noise_floor",) if floor_flags[idx] else ()
        # One-step rate change times the safety factor, matching the energy and
        # other derived channels.
        eps = float(safety_factor * movement[idx])
        verdicts.append(
            LevelVerdict(
                level_index=idx,
                status=_assign_status(
                    abs_err_est=0.0,
                    eps_gap_est=eps,
                    scope="observed_gap_scale",
                    target_abs_GHz=None,
                    target_gap_rel=target_gap_rel,
                ),
                status_scope="observed_gap_scale",
                evidence="verified_empirical",
                abs_err_est_GHz=None,
                eps_gap_est=eps,
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


def estimate_convergence_vs_paramvals(
    qubit: Any,
    param_name: str,
    param_vals: npt.NDArray[np.float64],
    **kwargs: Any,
) -> ParamSweepConvergence:
    """Top-level shim: forwards to ``qubit.estimate_convergence_vs_paramvals(...)``.

    Raises ``TypeError`` if the qubit does not subclass
    :class:`ConvergenceCheckable`.
    """
    if not isinstance(qubit, ConvergenceCheckable):
        raise TypeError(
            f"{type(qubit).__name__} does not implement convergence checking."
        )
    return qubit.estimate_convergence_vs_paramvals(param_name, param_vals, **kwargs)
