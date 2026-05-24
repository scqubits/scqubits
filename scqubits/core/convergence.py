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

from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt

import scqubits.core.units as units
import scqubits.utils.convergence_utils as cutils

from scqubits import settings
from scqubits.core.convergence_report import (
    ConvergenceReport,
    ImplementationAudit,
    LevelVerdict,
    ParamSweepConvergence,
    Status,
    TruncationChannel,
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

        A concrete qubit maps each refinement axis to its truncation channel: a
        charge tail (``"charge_tail"``), an oscillator tail (``"HO_tail"``), a
        finite-difference grid (``"FD_stencil"`` / ``"FD_box"``), or a composite
        coupling (``"composite_coupling"``). There is deliberately no default:
        the channel selects channel-specific estimators and recommendations and,
        for the charge basis alone, enables the variational monotonicity check
        (the only exactly-nested truncation), so a permissive default would
        silently mislabel a new qubit's axis and could enable that check on a
        non-nested basis. A subclass mixing in ``ConvergenceCheckable`` must
        implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not map convergence axis '{axis}' to a "
            "truncation channel; override _convergence_truncation_channel."
        )

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

    def _convergence_potential_envelope(self, axis: str) -> (
        tuple[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            float,
            float,
            float,
        ]
        | None
    ):
        """Return ``(V_eff, box_min, box_max, grid_spacing)`` for a box axis.

        ``V_eff(phi)`` is the potential along a discretized coordinate, minimized
        over the other coordinates and on the same energy reference as the
        eigenvalues, valid for any ``phi`` including beyond the current box.  It
        powers the FD_box turning-point completeness check (does the box contain
        the classically allowed region for the computed window?).  The default
        returns ``None``; a qubit with a finite-difference box coordinate
        overrides it for the relevant axis.
        """
        return None

    def _convergence_box_refine_step(
        self,
        axis: str,
        evals_n0: npt.NDArray[np.float64],
        n_levels: int,
    ) -> int:
        """Refinement step (in axis units) for ``axis``.

        For a box axis whose qubit supplies a potential envelope, widen the box
        far enough that the refined grid encloses the full classically allowed
        region for the computed window -- the outermost turning points of
        ``V_eff(phi) = max_k E_k + margin``, plus a buffer -- but never less than
        the ordinary step. Sizing the widening this way makes the refinement
        comparison unable to miss a low-lying well excluded by a too-small box: if
        such a well hosts a state, including it shifts the cluster-matched
        spectrum; if it is quantum-empty, the comparison shows no change and the
        box is accepted. Non-box axes use the ordinary step.
        """
        normal_step = self._convergence_step(axis)
        envelope = self._convergence_potential_envelope(axis)
        if envelope is None:
            return normal_step
        v_eff, box_min, box_max, spacing = envelope
        evals = np.asarray(evals_n0[:n_levels], dtype=np.float64)
        if evals.size == 0 or spacing <= 0.0:
            return normal_step
        e_win = float(np.max(evals)) + _box_completeness_margin(evals)
        turning = cutils.outermost_turning_points(v_eff, e_win, box_min, box_max)
        if turning is None:
            return normal_step
        phi_left, phi_right = turning
        buffer = 3.0 * spacing
        midpoint = 0.5 * (box_min + box_max)
        needed_half = max(
            midpoint - (phi_left - buffer), (phi_right + buffer) - midpoint
        )
        target_points = int(np.ceil(2.0 * needed_half / spacing)) + 1
        current = self._convergence_axis_value(axis)
        return max(target_points - current, normal_step)

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
        mode: str = "moderate",
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
            One of ``"cheap"`` (cheap diagnostics only, no extra
            diagonalizations; verdicts cap at ``unverified``), ``"moderate"``
            (one refinement at a bumped cutoff; default; verdicts cap at
            ``maybe_converged``), or ``"strict"`` (ratio test across two
            successive refinements; verdicts cap at ``likely_converged``).
            A test can only dismiss convergence, never prove it; a more
            rigorous mode wields a sharper dismissal test.
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
            ``"moderate"`` or ``"strict"`` -- derived quantities need a
            refinement comparison.
        derived_quantities:
            Subset of ``{"wavefunctions", "matrix_elements", "coherence"}`` to
            assess when ``include_derived`` is set.
        refinement:
            ``"one_step"`` for moderate mode; ``"ratio_test"`` for strict
            mode. Coerced to match ``mode`` if inconsistent.

        Returns
        -------
        :class:`~scqubits.core.convergence_report.ConvergenceReport`
        """
        # -------- input validation
        if mode not in ("cheap", "moderate", "strict"):
            raise ValueError(
                f"mode must be 'cheap', 'moderate', or 'strict'; got {mode!r}"
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
            if mode == "cheap":
                raise ValueError(
                    "include_derived requires mode='moderate' or 'strict'; "
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

        # In strict mode, ratio_test is the implied refinement; the cheaper
        # modes do at most one refinement.
        if mode == "strict" and refinement == "one_step":
            refinement = "ratio_test"

        # -------- diagonalize at the current cutoff
        # Buffer one extra level so the topmost reported level still has an
        # upper gap available for the observed-gap-scale denominator.
        n_buffer = 1 if scope == "observed_gap_scale" else 0
        n_eigs = n_levels + n_buffer
        evals_n0, evecs_n0 = self.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]

        # -------- dispatch by mode
        if mode == "cheap":
            return self._convergence_cheap(
                evals_n0=evals_n0,
                evecs_n0=evecs_n0,
                n_levels=n_levels,
                scope=scope,
                target_abs_GHz=target_abs_GHz,
                target_gap_rel=target_gap_rel,
                g_floor_GHz=g_floor_GHz,
            )
        # moderate / strict
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
            including both endpoints. The hardest point is often *interior*
            (e.g. fluxonium near half flux, avoided crossings), so the sampling
            spans the whole range rather than just the ends; pass any known hard
            points explicitly via ``param_vals``, or ``sample=None`` to check
            every value. Sampling keeps the check cheap relative to a full
            per-point sweep.
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

    def _convergence_unverified_report(
        self,
        n_levels: int,
        scope: str,
        mode: str,
        recommendations: list[str],
        warning: str,
        channel: TruncationChannel = "composite_coupling",
    ) -> ConvergenceReport:
        """Build an all-levels-``unverified`` report with no truncation estimate.

        Shared by coupled-system classes (``HilbertSpace``, ``FullZeroPi``) for the
        cases where there is no cheap composite estimate (cheap mode) or where a
        refinement cannot be carried out. Inner/layer-1 checks are attached
        separately by the caller.
        """
        audit = self._convergence_audit(n_levels, 0, mode, "one_step")
        per_level = [
            LevelVerdict(
                level_index=k,
                status="unverified",
                status_scope=scope,  # type: ignore[arg-type]
                abs_err_est_GHz=None,
                eps_gap_est=None,
                truncation_channel=channel,
                estimator_method="moderate_recommended",
                warnings=(warning,),
            )
            for k in range(n_levels)
        ]
        return ConvergenceReport(
            per_level=per_level,
            aggregate_status="unverified",
            worst_level=0 if n_levels else None,
            channel_breakdown_GHz={},
            clusters=[],
            recommendations=list(recommendations),
            implementation_audit=audit,
            derived=None,
        )

    # ------------------------------------------------------------ mode handlers

    def _convergence_cheap(
        self,
        evals_n0: npt.NDArray[np.float64],
        evecs_n0: npt.NDArray[np.float64],
        n_levels: int,
        scope: str,
        target_abs_GHz: float | None,
        target_gap_rel: float,
        g_floor_GHz: float,
    ) -> ConvergenceReport:
        """Cheap-mode report: perturbative tail estimate or boundary diagnostic.

        If the qubit supplies a perturbative tail estimate (e.g. the charge
        finite-tail), cheap mode reports it; otherwise it falls back to the bare
        boundary-amplitude diagnostic. Cheap mode never makes a verification
        claim -- the best verdict it returns is ``unverified``.
        """
        # Quick mode assesses the first (dominant) axis without refinement.
        axis = self._convergence_axes[0]
        channel: TruncationChannel = self._convergence_truncation_channel(axis)

        tail = self._convergence_tail_estimate((evals_n0, evecs_n0), axis)
        if tail is not None:
            return self._convergence_cheap_perturbative(
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
        worst_status: Status = "likely_converged"

        # Boundary amplitude under which the cheap diagnostic raises no red flag.
        # Conservative -- a larger amplitude triggers a recommendation to escalate.
        QUICK_BOUNDARY_THRESHOLD = 1e-6

        for k in range(n_levels):
            warnings: list[str] = []
            # Cheap mode makes no verification claim; the best verdict is
            # "unverified" (no dismissal). A boundary amplitude large enough to
            # signal a non-perturbative dropped tail is itself a dismissal
            # (-> distrust); a smaller but non-negligible amplitude only warns.
            status: Status = "unverified"
            if boundary_amplitudes is None:
                warnings.append("no_boundary_diagnostic_available")
            elif boundary_amplitudes[k] > _BOUNDARY_PROBABILITY_LARGE:
                status = "distrust"
                warnings.append("boundary_probability_large")
                recommendations.append(
                    f"level {k}: boundary amplitude {boundary_amplitudes[k]:.2e} "
                    f"exceeds {_BOUNDARY_PROBABILITY_LARGE:.0e}; the kept state reaches "
                    f"the basis boundary -- increase the cutoff (or widen the box)"
                )
            elif boundary_amplitudes[k] >= QUICK_BOUNDARY_THRESHOLD:
                warnings.append("boundary_amplitude_above_threshold")
                recommendations.append(
                    f"level {k}: boundary amplitude {boundary_amplitudes[k]:.2e} "
                    f"exceeds cheap-mode threshold {QUICK_BOUNDARY_THRESHOLD:.0e}; "
                    f"re-run with mode='moderate' to obtain an empirical estimate"
                )

            per_level.append(
                LevelVerdict(
                    level_index=k,
                    status=status,
                    status_scope=scope,  # type: ignore[arg-type]
                    abs_err_est_GHz=None,  # cheap mode has no error estimate
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

        return ConvergenceReport(
            per_level=per_level,
            aggregate_status=worst_status,
            worst_level=worst_index,
            channel_breakdown_GHz={},
            clusters=[(k,) for k in range(n_levels)],
            recommendations=list(dict.fromkeys(recommendations)),  # dedupe
            implementation_audit=self._convergence_audit(
                n_levels=n_levels, n_buffer=0, mode="cheap", refinement="one_step"
            ),
        )

    def _convergence_cheap_perturbative(
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
        """Cheap-mode report from a perturbative tail estimate.

        Reports the per-level tail estimate but caps a passing verdict at
        ``unverified`` (cheap mode never makes a verification claim). A level
        whose boundary probability is large is dismissed to ``distrust``; a level
        whose tail is not in the perturbative regime is ``unverified``.
        """
        estimate, perturbative_ok, boundary_prob = tail
        large_boundary_prob = _BOUNDARY_PROBABILITY_LARGE
        # The perturbative tail estimate omits higher-order and far-tail
        # contributions, so it is not a bound; apply the same safety factor the
        # refinement modes use so a quick verdict carries a comparable margin.
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
                # The kept eigenstate reaches the basis edge: a dismissal signal.
                status: Status = "distrust"
                warnings.append("boundary_probability_large")
            elif not bool(perturbative_ok[k]):
                status = "unverified"
                warnings.append("tail_not_perturbative")
            elif scope == "absolute" and target_abs_GHz is None:
                # No absolute target: a cheap pass earns only "unverified".
                status = "unverified"
            else:
                status = _assign_status(est, eps, scope, target_abs_GHz, target_gap_rel)
                # Cheap mode makes no verification claim: a pass caps at
                # "unverified".
                status = _apply_mode_ceiling(status, "cheap")

            per_level.append(
                LevelVerdict(
                    level_index=k,
                    status=status,
                    status_scope=scope,  # type: ignore[arg-type]
                    abs_err_est_GHz=est,
                    eps_gap_est=eps,
                    truncation_channel=channel,
                    estimator_method="finite_tail_resolvent",
                    warnings=tuple(warnings),
                )
            )

        worst_idx, aggregate_status = _aggregate_worst(per_level)
        recommendations: list[str] = []
        if any(v.status in ("distrust", "marginal") for v in per_level):
            current = self._convergence_axis_value(axis)
            step = self._convergence_step(axis)
            recommendations.append(
                f"increase {axis} from {current} to at least {current + step}, or "
                f"re-run with mode='moderate' for a refinement-verified estimate"
            )

        return ConvergenceReport(
            per_level=per_level,
            aggregate_status=aggregate_status,
            worst_level=worst_idx,
            channel_breakdown_GHz={},
            clusters=[(k,) for k in range(n_levels)],
            recommendations=recommendations,
            implementation_audit=self._convergence_audit(
                n_levels=n_levels, n_buffer=0, mode="cheap", refinement="one_step"
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
        # and sum the per-axis refinement differences -- a triangle-inequality upper bound on
        # the combined truncation error.
        combined_diff_n1 = np.zeros(n_clusters, dtype=np.float64)
        combined_diff_n2: npt.NDArray[np.float64] | None = (
            np.zeros(n_clusters, dtype=np.float64)
            if refinement == "ratio_test"
            else None
        )
        # Per-cluster monotonicity flags: an exact (charge-basis) Galerkin
        # truncation must not raise an ordered eigenvalue when enlarged. A
        # violation dismisses the level in any mode.
        monotonicity_violation = np.zeros(n_clusters, dtype=np.bool_)
        # When a finite-difference (Richardson) axis is present, the strict-mode
        # estimate must be formed per-axis (Richardson for the FD-stencil channel,
        # geometric elsewhere) and summed -- a single geometric test on the
        # summed refinement differences would misjudge the h**p stencil channel. Non-FD qubits
        # and verify mode keep the summed-refinement difference path unchanged.
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
            current = self._convergence_axis_value(axis)
            axis_channel = self._convergence_truncation_channel(axis)
            # Only the charge basis is an exact principal-submatrix (Galerkin)
            # truncation, so enlarging it cannot raise an ordered eigenvalue
            # (min-max). Harmonic-oscillator bases build the potential from
            # matrix functions of truncated quadratures, and finite-difference
            # grids are non-variational, so neither truncation is nested and a
            # rise there is legitimate, not a bug -- both are excluded.
            axis_is_nested_basis = axis_channel == "charge_tail"
            step = self._convergence_box_refine_step(axis, evals_n0, n_levels)
            clone_1 = self._convergence_clone_at({axis: current + step})
            evals_a1, evecs_a1 = clone_1.eigensys(evals_count=n_eigs)  # type: ignore[attr-defined]
            _, diff_n1 = cutils.cluster_safe_match_energies(
                evals_n0[:n_levels], evals_a1[:n_levels], clusters
            )
            combined_diff_n1 += diff_n1
            if axis_is_nested_basis:
                monotonicity_violation |= cutils.nested_basis_monotonicity_violation(
                    evals_n0[:n_levels],
                    evals_a1[:n_levels],
                    clusters,
                    settings.CONVERGENCE_MONOTONICITY_REL_TOL,
                    _REFINEMENT_NOISE_FLOOR_GHz,
                )

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
                if axis_is_nested_basis:
                    monotonicity_violation |= (
                        cutils.nested_basis_monotonicity_violation(
                            evals_a1[:n_levels],
                            evals_a2[:n_levels],
                            clusters,
                            settings.CONVERGENCE_MONOTONICITY_REL_TOL,
                            _REFINEMENT_NOISE_FLOOR_GHz,
                        )
                    )
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
                    # Conservative margin on the extrapolated geometric tail.
                    est_raw = safety_factor * est_raw
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
        dominant_axis = max(per_axis_weight, key=lambda a: per_axis_weight[a])
        # For a single multi-coordinate qubit the per-level channel is the
        # dominant physical channel (the full per-channel split is in
        # channel_breakdown_GHz). ``composite_coupling`` is reserved for
        # coupled-subsystem HilbertSpace truncation, not a multi-axis qubit.
        per_level_channel: TruncationChannel = self._convergence_truncation_channel(
            dominant_axis
        )

        # Flag levels whose kept eigenstate reaches a basis boundary so strongly
        # that the dropped tail is non-perturbative -- a reliable underconvergence
        # signal independent of the refinement difference (design spec). Uses each
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
            monotonicity_violation=monotonicity_violation,
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
        monotonicity_violation: npt.NDArray[np.bool_] | None = None,
    ) -> ConvergenceReport:
        """Assemble the LevelVerdicts and ConvergenceReport from per-cluster
        refinement differences.

        Shared by single- and multi-axis refinement. The caller supplies the
        cluster partition and the per-cluster refinement differences (already summed over axes
        for the multi-axis case), the per-level ``channel`` (the dominant physical
        channel for a multi-coordinate qubit), and the per-channel
        ``channel_breakdown``.

        When ``precomputed_estimate`` is given (a finite-difference axis used the
        Richardson estimator), it is the per-cluster absolute error already summed
        over axes, and ``precomputed_non_asymptotic`` flags clusters whose ratio
        or Richardson test failed; the geometric ratio test on the summed
        refinement differences is then bypassed. ``boundary_large`` flags levels that reach a
        basis boundary so strongly that the dropped tail is non-perturbative; they
        receive a ``boundary_probability_large`` warning.
        """
        safety_factor = settings.CONVERGENCE_SAFETY_FACTOR

        # Strict mode without a Richardson axis: a second refinement enables the
        # geometric ratio test on the summed refinement differences.
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
            cluster_max_diff_n2=cluster_max_diff_n2,
            precomputed_estimate=precomputed_estimate,
            precomputed_non_asymptotic=precomputed_non_asymptotic,
        )

        # Expand the per-cluster monotonicity flags to per-level: a variational
        # basis whose ordered energies rose under refinement is dismissed.
        per_level_monotonicity = [False] * n_levels
        if monotonicity_violation is not None:
            for cluster_idx, cluster in enumerate(clusters):
                if monotonicity_violation[cluster_idx]:
                    for k in cluster:
                        per_level_monotonicity[k] = True
                        per_level_warnings[k].append("monotonicity_violation")

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
            per_level_warnings=per_level_warnings,
            scope=scope,
            target_abs_GHz=target_abs_GHz,
            target_gap_rel=target_gap_rel,
            mode=mode,
            non_asymptotic=per_level_non_asymptotic,
            monotonicity_violation=per_level_monotonicity,
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

        When a level is dismissed (``distrust``), the advice is tailored to the
        dominant truncation channel (design spec): a charge tail grows ``ncut``, an HO
        tail grows the oscillator cutoff (or changes backend), an FD box must be
        enlarged rather than merely refined, and an FD stencil is refined at a
        fixed box. Levels carrying the ``boundary_probability_large`` warning are
        called out as having a non-perturbative tail. Also flags near-degenerate
        clusters whose labels are ambiguous, and prompts for a ``target_abs_GHz``
        when absolute-scope levels are unverified for lack of a target.
        """
        recommendations: list[str] = []
        if any(v.status == "distrust" for v in verdicts):
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
            elif channel == "composite_coupling":
                recommendations.append(
                    f"composite truncation dominates: increase truncated_dim of "
                    f"'{axis}' from {current_value} to at least "
                    f"{current_value + step} and re-run (it sets how many of that "
                    f"subsystem's levels enter the product space)"
                )
            else:
                recommendations.append(
                    f"{bump} and re-run; the worst-level estimate exceeded the "
                    f"target threshold"
                )
        if any("monotonicity_violation" in v.warnings for v in verdicts):
            recommendations.append(
                "energies increased when the basis was enlarged, violating the "
                "variational bound: this indicates a basis-construction, operator, "
                "or eigensolver problem rather than simple undertruncation -- "
                "verify the model setup and backend before trusting any level"
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
            v.status == "distrust" and "cluster_index_ambiguity" in v.warnings
            for v in verdicts
        ):
            recommendations.append(
                "a dismissed level lies in a near-degenerate cluster; "
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
            refinement_diff_first = self._convergence_wavefunction_refinement_diff(
                evecs_n0, evecs_n1, ncut_0, ncut_1, clusters, n_levels
            )
            refinement_diff_second: npt.NDArray[np.float64] | None = None
            if refinement == "ratio_test" and evecs_n2 is not None:
                refinement_diff_second = self._convergence_wavefunction_refinement_diff(
                    evecs_n1, evecs_n2, ncut_1, ncut_2, clusters, n_levels
                )
            derived["wavefunctions"] = _build_metric_report(
                refinement_diff_first=refinement_diff_first,
                refinement_diff_second=refinement_diff_second,
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
            me_refinement_diff_first, skipped = _matrix_element_refinement_diff(
                self, clone_1, evecs_n0, evecs_n1, n_levels
            )
            me_refinement_diff_second: npt.NDArray[np.float64] | None = None
            if (
                refinement == "ratio_test"
                and clone_2 is not None
                and evecs_n2 is not None
            ):
                me_refinement_diff_second, _ = _matrix_element_refinement_diff(
                    clone_1, clone_2, evecs_n1, evecs_n2, n_levels
                )
            derived["matrix_elements"] = _build_metric_report(
                refinement_diff_first=me_refinement_diff_first,
                refinement_diff_second=me_refinement_diff_second,
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
            names, rate_refinement_diff, floor_flags, skipped_channels = (
                _coherence_rate_refinement_diff(
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
                refinement_diff=rate_refinement_diff,
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

    def _convergence_wavefunction_refinement_diff(
        self,
        evecs_a: npt.NDArray[np.float64],
        evecs_b: npt.NDArray[np.float64],
        value_a: int,
        value_b: int,
        clusters: list[tuple[int, ...]],
        n_levels: int,
    ) -> npt.NDArray[np.float64]:
        """Per-level wavefunction refinement difference between two cutoffs.

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
        refinement_diff = np.empty(n_levels, dtype=np.float64)
        for k in range(n_levels):
            refinement_diff[k] = 1.0 - min(1.0, abs(complex(np.vdot(a[:, k], b[:, k]))))
        for cluster in clusters:
            if len(cluster) > 1:
                cols = list(cluster)
                angle = cutils.subspace_angle(a[:, cols], b[:, cols])
                for k in cluster:
                    refinement_diff[k] = angle
        return refinement_diff

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
        the per-axis refinement differences are summed (triangle-inequality bound), mirroring
        the energy channel. Matrix elements and coherence are basis-agnostic and
        supported here; wavefunctions require axis-aware tensor padding and are
        not yet available for multi-axis qubits.
        """
        if "wavefunctions" in derived_quantities:
            raise NotImplementedError(
                "wavefunction convergence for multi-axis qubits is not yet "
                "available; request matrix_elements and/or coherence."
            )
        channel: TruncationChannel = self._convergence_truncation_channel(dominant_axis)
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
                refinement_diff, axis_skipped = _matrix_element_refinement_diff(
                    self, clone_axis, evecs_n0, evecs_axis, n_levels
                )
                combined += refinement_diff
                skipped.update(axis_skipped)
            derived["matrix_elements"] = _build_metric_report(
                refinement_diff_first=combined,
                refinement_diff_second=None,
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
                names, refinement_diff, floor_flags, axis_skipped = (
                    _coherence_rate_refinement_diff(
                        self,
                        clone_axis,
                        (evals_n0, evecs_n0),
                        (evals_axis, evecs_axis),
                        channels,
                        settings.CONVERGENCE_RATE_FLOOR_HZ,
                    )
                )
                skipped_channels.update(axis_skipped)
                for idx, name in enumerate(names):
                    combined_by_channel[name] = combined_by_channel.get(
                        name, 0.0
                    ) + float(refinement_diff[idx])
                    floor_by_channel[name] = floor_by_channel.get(name, False) or bool(
                        floor_flags[idx]
                    )
            final_names = list(combined_by_channel)
            final_refinement_diff = np.asarray(
                [combined_by_channel[name] for name in final_names], dtype=np.float64
            )
            final_floor = [floor_by_channel[name] for name in final_names]
            derived["coherence"] = _build_coherence_report(
                channel_names=final_names,
                refinement_diff=final_refinement_diff,
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
    "likely_converged": 0,
    "maybe_converged": 1,
    "marginal": 2,
    "unverified": 3,
    "distrust": 4,
}

# The best verdict each mode is entitled to return for a passing estimate. A
# test can only dismiss convergence, never prove it, so a "pass" rises only to
# the ceiling set by how rigorous the mode was.
_MODE_CEILING: dict[str, Status] = {
    "cheap": "unverified",
    "moderate": "maybe_converged",
    "strict": "likely_converged",
}

# Refinement differences at or below this absolute size (GHz) are treated as
# eigensolver noise rather than real movement, so a failed ratio test on them is
# not read as a non-asymptotic dismissal.
_REFINEMENT_NOISE_FLOOR_GHz: float = 1e-9


def _status_rank(status: Status) -> int:
    """Rank for aggregating worst per-level status; higher is worse."""
    return _STATUS_RANK[status]


def _apply_mode_ceiling(status: Status, mode: str) -> Status:
    """Cap a passing verdict at the best the mode is entitled to claim.

    A verdict better than the mode's ceiling is lowered to that ceiling; a
    ``marginal`` or ``distrust`` verdict (a dismissal) is never raised.
    """
    ceiling = _MODE_CEILING[mode]
    if _STATUS_RANK[status] < _STATUS_RANK[ceiling]:
        return ceiling
    return status


def _box_completeness_margin(evals: npt.NDArray[np.float64]) -> float:
    """Energy margin above the top requested level for the box-completeness check.

    Uses the mean requested-level spacing, so the box is checked against the
    classically allowed region for roughly one level beyond the requested window.
    """
    if evals.size < 2:
        return 0.0
    return float((float(np.max(evals)) - float(np.min(evals))) / (evals.size - 1))


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
    """Assign the best verdict an estimate supports, before mode capping.

    Returns ``likely_converged`` for an estimate below the target, ``marginal``
    for an estimate within a decade of it, and ``distrust`` for a clearly larger
    estimate. The caller lowers a passing verdict to the mode ceiling via
    :func:`_apply_mode_ceiling`; a missing target or metric yields ``unverified``.
    """
    if scope == "absolute":
        if target_abs_GHz is None:
            return "unverified"
        if abs_err_est < target_abs_GHz:
            return "likely_converged"
        if abs_err_est < 10.0 * target_abs_GHz:
            return "marginal"
        return "distrust"
    # observed_gap_scale
    if eps_gap_est is None:
        return "unverified"
    if eps_gap_est < target_gap_rel:
        return "likely_converged"
    if eps_gap_est < 10.0 * target_gap_rel:
        return "marginal"
    return "distrust"


def _per_cluster_energy_estimates(
    clusters: list[tuple[int, ...]],
    cluster_max_diff_n1: npt.NDArray[np.float64],
    refinement: str,
    geometric_tail: npt.NDArray[np.float64] | None,
    asymptotic_flag: npt.NDArray[np.bool_] | None,
    safety_factor: float,
    n_levels: int,
    cluster_max_diff_n2: npt.NDArray[np.float64] | None = None,
    precomputed_estimate: npt.NDArray[np.float64] | None = None,
    precomputed_non_asymptotic: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], list[str], list[list[str]], list[bool]]:
    """Derive each level's error estimate, estimator method, and warnings.

    A one-step estimate (moderate mode, or the strict-mode fallback) takes
    ``safety_factor`` times the one-step refinement difference. A successful
    two-step ratio test instead reports the geometric-tail estimate. When the
    ratio test finds ``R >= 1`` (no reliable asymptotic regime), or a flat first
    refinement is followed by real movement, the level is flagged non-asymptotic
    so the status logic dismisses it. Refinement differences at or below the
    eigensolver noise floor are treated as no movement, not a failed ratio test.
    Every level inherits the values of the cluster it belongs to.

    When ``precomputed_estimate`` is supplied (a finite-difference axis used the
    Richardson estimator), it is the per-cluster absolute error already summed
    over axes and ``precomputed_non_asymptotic`` flags clusters whose per-axis
    test failed; these are used directly instead of the geometric path.

    The last returned element flags, per level, a failed asymptoticity test; the
    status logic forces such levels to ``distrust``.
    """
    per_level_abs_err = np.empty(n_levels, dtype=np.float64)
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
            d1 = (
                float(cluster_max_diff_n2[cluster_idx])
                if cluster_max_diff_n2 is not None
                else 0.0
            )
            if d0 <= _REFINEMENT_NOISE_FLOOR_GHz and d1 <= _REFINEMENT_NOISE_FLOOR_GHz:
                # Both refinements are flat at the eigensolver noise floor: no
                # real movement, so the undefined ratio is not a dismissal. The
                # safety factor keeps the (negligible) reported estimate a
                # conservative bound on the residual movement.
                est = safety_factor * max(d0, d1)
                method = "ratio_test_noise_floor"
            elif asymptotic_flag is not None and asymptotic_flag[cluster_idx]:
                # Ratio test confirmed the geometric (asymptotic) regime. Apply
                # the safety factor to the extrapolated tail too, so a series that
                # is not perfectly geometric (e.g. algebraic decay) is bounded
                # conservatively rather than slightly under-estimated.
                est = float(safety_factor * geometric_tail[cluster_idx])
                method = "ratio_test"
            else:
                # R >= 1, or a flat first refinement followed by real movement:
                # not a reliable asymptotic regime. Flag the level so the status
                # logic dismisses it; report the larger one-step movement as the
                # (non-trustworthy) estimate.
                est = float(safety_factor * max(d0, d1))
                method = "ratio_test_failed_fallback_one_step"
                non_asymptotic = True
                for k in cluster:
                    per_level_warnings[k].append("ratio_test_not_asymptotic")
        else:
            est = float(safety_factor * cluster_max_diff_n1[cluster_idx])
            method = "one_step"

        # A multi-level (near-degenerate) cluster is compared as a sorted set,
        # so individual level labels inside it are not reliable. Flag every
        # member (design spec).
        cluster_ambiguous = len(cluster) > 1

        for k in cluster:
            per_level_abs_err[k] = est
            per_level_estimator_method.append(method)
            per_level_non_asymptotic[k] = non_asymptotic
            if cluster_ambiguous:
                per_level_warnings[k].append("cluster_index_ambiguity")

    return (
        per_level_abs_err,
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
    per_level_warnings: list[list[str]],
    scope: str,
    target_abs_GHz: float | None,
    target_gap_rel: float,
    mode: str,
    non_asymptotic: list[bool],
    monotonicity_violation: list[bool],
) -> list[Status]:
    """Assign each level's verdict from its estimate, the target, and the mode.

    A passing estimate is capped at the verdict the mode is entitled to make
    (:func:`_apply_mode_ceiling`): a test can dismiss convergence, never prove
    it. A level whose two-step ratio test failed (``R >= 1`` on real movement,
    flagged in ``non_asymptotic``) is dismissed to ``distrust`` -- its error
    estimate is not trustworthy -- regardless of how small the one-step movement
    happened to be. A level flagged in ``monotonicity_violation`` (a variational
    basis whose ordered energies rose under refinement) is likewise dismissed to
    ``distrust`` in every mode.
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
        status = _apply_mode_ceiling(status, mode)
        if non_asymptotic[k] or monotonicity_violation[k]:
            status = "distrust"
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


def _matrix_element_refinement_diff(
    qubit_a: Any,
    qubit_b: Any,
    evecs_a: npt.NDArray[np.float64],
    evecs_b: npt.NDArray[np.float64],
    n_levels: int,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Per-level relative matrix-element refinement difference between two cutoffs.

    For each operator returned by ``get_operator_names`` the ``n_levels`` x
    ``n_levels`` matrix-element table is formed at both cutoffs; level ``k`` is
    assigned the worst relative change of its matrix-element row and column,
    maximized over operators. The relative change normalizes by the refined
    table's row/column norm, floored to guard against selection-rule zeros.
    Matrix elements are compared by magnitude, since each eigenvector's global
    phase is a gauge choice that can differ between cutoffs (a signed comparison
    would report spurious refinement difference from phase flips). Operators that raise or
    return a shape-incompatible table are skipped and reported (graceful
    degradation), not silently dropped.
    """
    refinement_diff = np.zeros(n_levels, dtype=np.float64)
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
            refinement_diff[k] = max(float(refinement_diff[k]), rel_k)
    return refinement_diff, skipped


def _build_metric_report(
    refinement_diff_first: npt.NDArray[np.float64],
    refinement_diff_second: npt.NDArray[np.float64] | None,
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

    ``refinement_diff_first`` holds each level's change between the base and first
    refinement (overlap deficit, subspace angle, or relative matrix-element
    change). In ratio-test mode ``refinement_diff_second`` (the first-to-second
    refinement change) drives a geometric extrapolation; levels confirmed
    asymptotic use the extrapolated tail, the rest fall back to the one-step
    refinement difference with a recorded warning. A failed ratio test
    (``R >= 1``) dismisses the level to ``distrust``, mirroring the energy
    channel. A passing verdict is capped at the mode ceiling (``maybe_converged``
    for a one-step check, ``likely_converged`` for a ratio-tested one). Verdicts
    apply the observed-gap-scale ladder against ``target_gap_rel``.
    """
    geometric_tail: npt.NDArray[np.float64] | None = None
    asymptotic: npt.NDArray[np.bool_] | None = None
    if refinement == "ratio_test" and refinement_diff_second is not None:
        _, geometric_tail, asymptotic = cutils.geometric_ratio_test(
            refinement_diff_first, refinement_diff_second
        )

    safety_factor = settings.CONVERGENCE_SAFETY_FACTOR
    per_level_eps = np.empty(n_levels, dtype=np.float64)
    per_level_method: list[str] = []
    per_level_warnings: list[list[str]] = [[] for _ in range(n_levels)]
    per_level_non_asymptotic: list[bool] = [False] * n_levels

    for k in range(n_levels):
        if geometric_tail is not None and asymptotic is not None and asymptotic[k]:
            # Extrapolated remaining-error estimate; apply the safety factor for a
            # conservative margin (matching the energy channel).
            per_level_eps[k] = float(safety_factor * geometric_tail[k])
            per_level_method.append(f"{estimator_method}_ratio_test")
        elif refinement == "ratio_test" and float(refinement_diff_first[k]) == 0.0:
            # No refinement difference: already stable, the ratio is undefined
            # rather than a failed asymptoticity test.
            per_level_eps[k] = 0.0
            per_level_method.append(f"{estimator_method}_ratio_test")
        else:
            # One-step refinement difference underestimates the remaining change;
            # apply the same safety factor the energy channel uses.
            per_level_eps[k] = float(safety_factor * refinement_diff_first[k])
            if refinement == "ratio_test":
                per_level_method.append(f"{estimator_method}_ratio_test_fallback")
                per_level_warnings[k].append("ratio_test_not_asymptotic")
                per_level_non_asymptotic[k] = True
            else:
                per_level_method.append(estimator_method)

    # A one-step (moderate) derived check caps at maybe_converged; a ratio-tested
    # (strict) one can reach likely_converged.
    derived_mode = "strict" if refinement == "ratio_test" else "moderate"
    verdicts: list[LevelVerdict] = []
    for k in range(n_levels):
        status: Status = _assign_status(
            abs_err_est=0.0,
            eps_gap_est=float(per_level_eps[k]),
            scope="observed_gap_scale",
            target_abs_GHz=None,
            target_gap_rel=target_gap_rel,
        )
        status = _apply_mode_ceiling(status, derived_mode)
        if per_level_non_asymptotic[k]:
            status = "distrust"
        verdicts.append(
            LevelVerdict(
                level_index=k,
                status=status,
                status_scope="observed_gap_scale",
                abs_err_est_GHz=None,
                eps_gap_est=None,
                rel_change_est=float(per_level_eps[k]),
                truncation_channel=channel,
                estimator_method=per_level_method[k],
                warnings=tuple(per_level_warnings[k]),
            )
        )
    worst_idx, aggregate_status = _aggregate_worst(verdicts)

    recommendations: list[str] = []
    if any(v.status == "distrust" for v in verdicts):
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


def _coherence_rate_refinement_diff(
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
    refinement_diff: list[float] = []
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
        refinement_diff.append(abs(rate_b - rate_a) / max(rate_b, rate_floor))
        floor_flags.append(rate_b < rate_floor)
    return names, np.asarray(refinement_diff, dtype=np.float64), floor_flags, skipped


def _build_coherence_report(
    channel_names: list[str],
    refinement_diff: npt.NDArray[np.float64],
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
        eps = float(safety_factor * refinement_diff[idx])
        verdicts.append(
            LevelVerdict(
                level_index=idx,
                status=_apply_mode_ceiling(
                    _assign_status(
                        abs_err_est=0.0,
                        eps_gap_est=eps,
                        scope="observed_gap_scale",
                        target_abs_GHz=None,
                        target_gap_rel=target_gap_rel,
                    ),
                    "moderate",
                ),
                status_scope="observed_gap_scale",
                abs_err_est_GHz=None,
                eps_gap_est=None,
                rel_change_est=eps,
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
    if any(v.status == "distrust" for v in verdicts):
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
            f"{type(qubit).__name__} does not support convergence checking; "
            "it must subclass ConvergenceCheckable."
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
