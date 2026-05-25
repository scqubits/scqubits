# convergence_report.py
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
"""Verdict and report data classes for the convergence-diagnostics framework.

The frozen dataclasses defined here form the return type of
:meth:`scqubits.core.convergence.ConvergenceCheckable.estimate_convergence`.
The schema deliberately separates true (unknown) error from estimated error and
separates absolute from observed-gap-scale error metrics. A convergence test can
only ever *dismiss* convergence (a negative result); a passing verdict means the
applied test failed to dismiss the level, never that convergence is guaranteed.
"""

from __future__ import annotations

import textwrap

from dataclasses import dataclass, field
from typing import Literal

Status = Literal[
    "likely_converged", "maybe_converged", "marginal", "unverified", "distrust"
]
"""Convergence verdict, ordered from best to worst.

A test can only dismiss convergence, never prove it, so the verdict records how
hard we *failed to dismiss* a level -- which depends on how rigorous the mode was:

- ``likely_converged``: passed the ``strict`` ratio/asymptoticity test (the
  strongest statement the framework makes; never an outright guarantee).
- ``maybe_converged``: passed the ``moderate`` one-step refinement check.
- ``marginal``: an estimate close to the requested target (borderline).
- ``unverified``: no dismissal and no verification -- a ``cheap`` pass, or a
  level that could not be assessed.
- ``distrust``: a test actively dismissed convergence (the result is not
  trustworthy at this cutoff).

The best verdict a mode can return is capped by the mode: ``cheap`` ->
``unverified``, ``moderate`` -> ``maybe_converged``, ``strict`` ->
``likely_converged``.
"""

StatusScope = Literal["absolute", "observed_gap_scale"]

Mode = Literal["cheap", "moderate", "strict"]

Refinement = Literal["one_step", "ratio_test"]
"""Whether the convergence threshold was applied to an absolute energy error
(in GHz) or to an error normalized by an observed local spectral gap.

The basis-parameter scale (e.g. fluxonium ``omega_LC``) is intentionally not a
reporting scope -- see the published design specification for the rationale.
"""

TruncationChannel = Literal[
    "charge_tail",
    "HO_tail",
    "FD_box",
    "FD_stencil",
    "composite_coupling",
]
"""Physical truncation channel responsible for a (sub-)report.

Reserved for actual error sources (which basis is being truncated), distinct
from ``estimator_method`` (how the error was estimated) and ``warnings`` (what
caveats accompany the verdict). The three vocabularies are disjoint by design.

Channels follow the design specification:

- ``charge_tail``: truncation of a charge basis ``|n| <= ncut`` (transmon,
  flux qubit, ZeroPi theta).
- ``HO_tail``: truncation of a harmonic-oscillator (Fock) basis (fluxonium).
- ``FD_box``: finite extent of a finite-difference coordinate box ``[-L, L]``.
- ``FD_stencil``: finite finite-difference grid spacing at fixed box.
- ``composite_coupling``: combined estimate over several channels (a
  multi-coordinate qubit's per-axis sum, or a composite Hilbert space).
"""

CheckStatus = Literal["pass", "fail", "not_applicable"]
"""Outcome of one convergence check applied to a level.

- ``pass``: the check ran and did not dismiss the level.
- ``fail``: the check ran and dismissed (or flagged) the level.
- ``not_applicable``: the check does not apply in the chosen mode or for this
  truncation channel.

A ``fail`` is a falsification (the check caught a problem), consistent with the
falsification philosophy: tests dismiss convergence, they never prove it.
"""


@dataclass(frozen=True)
class CheckOutcome:
    """Outcome of a single named convergence check for a level.

    Makes the per-check pass/fail picture first-class, complementing ``status``
    (the overall verdict) and ``estimator_method`` (which estimator produced the
    number). ``detail`` is a short human-readable note, e.g. a measured boundary
    probability or why a check did not apply.
    """

    name: str
    status: CheckStatus
    detail: str = ""


@dataclass(frozen=True)
class LevelVerdict:
    """Per-level convergence verdict.

    All numerical fields suffixed ``_est`` are computed estimates, not the
    unknown true error. ``abs_err_est_GHz`` and ``eps_gap_est`` (an
    isolation-gap-normalized energy error) describe an energy level;
    ``rel_change_est`` is the dimensionless change of a *derived* quantity
    (wavefunction, matrix element, coherence rate) and is set only in derived
    sub-reports. ``transition_err_est_GHz`` keys are pairs ``(i, j)`` for the
    transition ``i -> j``.
    """

    level_index: int
    status: Status
    status_scope: StatusScope
    abs_err_est_GHz: float | None
    eps_gap_est: float | None
    rel_change_est: float | None = None
    transition_err_est_GHz: dict[tuple[int, int], float] = field(default_factory=dict)
    truncation_channel: TruncationChannel = "charge_tail"
    estimator_method: str = "one_step"
    warnings: tuple[str, ...] = ()
    checks: tuple[CheckOutcome, ...] = ()


@dataclass(frozen=True)
class ImplementationAudit:
    """Snapshot of the implementation pathway diagnosed by a report.

    Recorded so that warnings and recommendations remain tied to the backend
    they were derived from; a future change in scqubits' Hamiltonian
    construction does not silently invalidate previous reports.
    """

    scqubits_version: str
    scqubits_commit: str | None
    qubit_class: str
    basis: str
    diagonalization_method: str
    cutoff_parameters: dict[str, int]
    fd_stencil_order: int | None
    fd_box: tuple[float, float] | None
    nonpoly_backend: str | None
    n_levels_requested: int
    n_levels_buffer: int
    mode: Mode
    refinement: Refinement


@dataclass(frozen=True)
class ConvergenceReport:
    """Structured convergence diagnostic for a qubit's lowest levels.

    The ``derived`` field holds sub-reports: one per requested derived quantity
    (wavefunctions, matrix elements, coherence rates) and, for composite
    systems, one per subsystem.
    """

    per_level: list[LevelVerdict]
    aggregate_status: Status
    worst_level: int | None
    channel_breakdown_GHz: dict[str, float]
    clusters: list[tuple[int, ...]]
    recommendations: list[str]
    implementation_audit: ImplementationAudit
    derived: dict[str, "ConvergenceReport"] | None = None

    def level(self, level_index: int) -> LevelVerdict:
        """Return the per-level verdict for the given level index.

        Looks the verdict up by its ``level_index`` (not by list position), so
        ``report.level(report.worst_level)`` reliably retrieves the driving
        level. Raises :class:`KeyError` if that level was not assessed.
        """
        for verdict in self.per_level:
            if verdict.level_index == level_index:
                return verdict
        assessed = [verdict.level_index for verdict in self.per_level]
        raise KeyError(
            f"no verdict for level_index {level_index}; assessed levels: {assessed}"
        )

    def summary(self) -> str:
        """Return a compact, human-readable multi-line summary of the report.

        Shows the aggregate verdict and worst level, then an aligned per-level
        table (level, status, channel, error estimate, estimator); the structured
        per-check record is shown beneath any dismissed or borderline level. The
        per-channel breakdown, recommendations, and any derived sub-reports
        (indented) follow. Columns that do not apply to the report's scope are
        omitted rather than printed as placeholders.

        Returns
        -------
        The formatted summary, the same text produced by ``print(report)``.
        """
        lines = [
            f"aggregate: {self.aggregate_status}   (worst: level {self.worst_level})"
        ]
        if self.per_level:
            lines.append("")
            lines.extend(self._level_table())
        if self.channel_breakdown_GHz:
            breakdown = "  ".join(
                f"{name}={value:.2e}"
                for name, value in self.channel_breakdown_GHz.items()
            )
            lines += ["", f"  error by channel (GHz): {breakdown}"]
        for recommendation in self.recommendations:
            lines.append(f"  -> {recommendation}")
        if self.derived:
            for name, sub_report in self.derived.items():
                lines.append(f"  derived [{name}]:")
                lines.append(textwrap.indent(sub_report.summary(), "    "))
        return "\n".join(lines)

    def _level_table(self) -> list[str]:
        """Render the per-level verdicts as an aligned text table.

        The metric columns are chosen by what the report actually carries: an
        energy report shows ``err (GHz)`` (and ``gap_rel`` in observed-gap scope);
        a derived sub-report shows ``rel_chg``. Warnings trail the row; the
        structured per-check record is shown on an indented line beneath any
        ``distrust`` or ``marginal`` level (the full record is always available in
        ``LevelVerdict.checks``).
        """
        levels = self.per_level

        def fmt(value: float | None) -> str:
            return "" if value is None else f"{value:.2e}"

        # (header, cells, right-align); a metric column is shown only if populated.
        cols: list[tuple[str, list[str], bool]] = [
            ("lvl", [str(v.level_index) for v in levels], True),
            ("status", [v.status for v in levels], False),
            ("channel", [str(v.truncation_channel) for v in levels], False),
        ]
        if any(v.abs_err_est_GHz is not None for v in levels):
            cols.append(("err (GHz)", [fmt(v.abs_err_est_GHz) for v in levels], True))
        if any(v.eps_gap_est is not None for v in levels):
            cols.append(("gap_rel", [fmt(v.eps_gap_est) for v in levels], True))
        if any(v.rel_change_est is not None for v in levels):
            cols.append(("rel_chg", [fmt(v.rel_change_est) for v in levels], True))
        cols.append(("via", [v.estimator_method for v in levels], False))

        widths = [
            max(len(header), max(len(cell) for cell in cells))
            for header, cells, _ in cols
        ]

        def render(row_cells: list[str]) -> str:
            return "  " + "   ".join(
                cell.rjust(width) if right else cell.ljust(width)
                for cell, (_, _, right), width in zip(row_cells, cols, widths)
            )

        shorthand = {"not_applicable": "n/a"}
        out = [render([header for header, _, _ in cols]).rstrip()]
        for index, verdict in enumerate(levels):
            row = render([cells[index] for _, cells, _ in cols])
            if verdict.warnings:
                row += "   [" + ", ".join(verdict.warnings) + "]"
            out.append(row.rstrip())
            if verdict.checks and verdict.status in ("distrust", "marginal"):
                parts = [
                    f"{check.name}={shorthand.get(check.status, check.status)}"
                    + (f"({check.detail})" if check.detail else "")
                    for check in verdict.checks
                ]
                out.append(f"      checks {verdict.level_index}: " + "  ".join(parts))
        return out

    def __str__(self) -> str:
        """Return :meth:`summary` so ``print(report)`` shows the readable form."""
        return self.summary()


@dataclass(frozen=True)
class ParamSweepConvergence:
    """Convergence across a swept parameter (e.g. a ``plot_..._vs_paramvals`` range).

    Holds the per-point :class:`ConvergenceReport` at each sampled parameter
    value, and identifies the worst-case point -- the value at which the fixed
    cutoff is least trustworthy.
    """

    param_name: str
    param_vals: list[float]
    reports: list[ConvergenceReport]
    worst_index: int
    aggregate_status: Status

    def worst_param_val(self) -> float:
        """Return the parameter value with the worst per-point aggregate status."""
        return self.param_vals[self.worst_index]

    def worst_report(self) -> ConvergenceReport:
        """Return the :class:`ConvergenceReport` at the worst-case parameter value."""
        return self.reports[self.worst_index]

    def summary(self) -> str:
        """Return a compact, human-readable summary across the swept values.

        Lists each sampled parameter value with its per-point aggregate status,
        marks the worst-case point, and states the overall worst status. The
        full per-point reports remain available in :attr:`reports`.
        """
        lines = [
            f"convergence vs {self.param_name} ({len(self.param_vals)} points): "
            f"worst = {self.aggregate_status} at "
            f"{self.param_name}={self.worst_param_val():.4g}"
        ]
        for index, (value, report) in enumerate(zip(self.param_vals, self.reports)):
            marker = "  <-- worst" if index == self.worst_index else ""
            lines.append(
                f"  {self.param_name}={value:<10.4g} {report.aggregate_status}{marker}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return :meth:`summary` so ``print(sweep)`` shows the readable form."""
        return self.summary()


@dataclass(frozen=True)
class ParameterSweepConvergence:
    """Worst-case convergence across a :class:`.ParameterSweep` grid.

    Holds the per-point :class:`ConvergenceReport` at each sampled grid point of an
    N-dimensional parameter sweep, and identifies the worst-case point -- the grid
    point at which the fixed cutoffs are least trustworthy. Each point is a mapping
    from swept-parameter name to value.
    """

    param_names: tuple[str, ...]
    param_points: list[dict[str, float]]
    reports: list[ConvergenceReport]
    worst_index: int
    aggregate_status: Status

    def worst_point(self) -> dict[str, float]:
        """Return the parameter point with the worst per-point aggregate status."""
        return self.param_points[self.worst_index]

    def worst_report(self) -> ConvergenceReport:
        """Return the :class:`ConvergenceReport` at the worst-case grid point."""
        return self.reports[self.worst_index]

    def summary(self) -> str:
        """Return a compact, human-readable summary across the sampled grid points.

        Lists each sampled point (its swept-parameter coordinates) with the
        per-point aggregate status, marks the worst-case point, and states the
        overall worst status. The full per-point reports remain available in
        :attr:`reports`.
        """
        names = self.param_names
        worst = self.worst_point()
        worst_coords = ", ".join(f"{name}={worst[name]:.4g}" for name in names)
        lines = [
            f"convergence across sweep of ({', '.join(names)}) "
            f"({len(self.param_points)} points): worst = {self.aggregate_status} "
            f"at {worst_coords}"
        ]
        for index, (point, report) in enumerate(zip(self.param_points, self.reports)):
            coords = ", ".join(f"{name}={point[name]:.4g}" for name in names)
            marker = "  <-- worst" if index == self.worst_index else ""
            lines.append(f"  {coords}: {report.aggregate_status}{marker}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return :meth:`summary` so ``print(...)`` shows the readable form."""
        return self.summary()
