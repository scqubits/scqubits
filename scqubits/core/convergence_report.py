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

The three frozen dataclasses defined here form the return type of
:meth:`scqubits.core.convergence.ConvergenceCheckable.estimate_convergence`.
The schema deliberately separates true (unknown) error from estimated error,
separates absolute from observed-gap-scale error metrics, and attaches an
explicit evidence label to every numerical conclusion.
"""

from __future__ import annotations

import textwrap

from dataclasses import dataclass, field
from typing import Literal

Status = Literal[
    "converged", "likely_converged", "marginal", "underconverged", "unverified"
]
"""Convergence status, in order from best to worst.

``likely_converged`` is reserved for quick-mode reports based only on cheap
diagnostics; an unqualified ``converged`` always implies at least
``verified_empirical`` evidence.
"""

StatusScope = Literal["absolute", "observed_gap_scale"]
"""Whether the convergence threshold was applied to an absolute energy error
(in GHz) or to an error normalized by an observed local spectral gap.

The basis-parameter scale (e.g. fluxonium ``omega_LC``) is intentionally not a
reporting scope -- see the published design specification for the rationale.
"""

Evidence = Literal[
    "certified",
    "verified_empirical",
    "calibrated",
    "perturbative",
    "diagnostic",
    "unverified",
]
"""Evidence label, ordered from strongest to weakest.

Meanings:

- ``certified``: theorem-level bound with all hypotheses checked at runtime.
- ``verified_empirical``: refinement or cross-representation comparison with
  a ratio, asymptoticity, or stability check.
- ``calibrated``: estimator whose mapping to true error has been measured on
  a calibration grid covering the stated regime.
- ``perturbative``: derivation from perturbation theory or a block-resolvent
  approximation with unverified runtime hypotheses.
- ``diagnostic``: useful signal of possible failure, not an estimate of the
  error itself.
- ``unverified``: inputs unavailable, assumptions failed, or estimator
  inapplicable.
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


@dataclass(frozen=True)
class LevelVerdict:
    """Per-level convergence verdict.

    All numerical fields suffixed ``_est`` are computed estimates, not the
    unknown true error. ``transition_err_est_GHz`` keys are pairs
    ``(i, j)`` for the transition ``i -> j``.
    """

    level_index: int
    status: Status
    status_scope: StatusScope
    evidence: Evidence
    abs_err_est_GHz: float | None
    eps_gap_est: float | None
    transition_err_est_GHz: dict[tuple[int, int], float] = field(default_factory=dict)
    truncation_channel: TruncationChannel = "charge_tail"
    estimator_method: str = "one_step"
    warnings: tuple[str, ...] = ()


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
    mode: Literal["quick", "verify", "strict"]
    refinement: Literal["one_step", "ratio_test"]


@dataclass(frozen=True)
class ConvergenceReport:
    """Structured convergence diagnostic for a qubit's lowest levels.

    The ``derived`` field is reserved for sub-reports (currently only used
    in later PRs for matrix-element and coherence sub-channels, and in a
    future HilbertSpace-level extension to compose per-subsystem reports).
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

        Lists the aggregate status and worst level, then per level the status,
        evidence label, truncation channel, error estimate, and any warnings,
        followed by the per-channel breakdown, the recommendations, and any
        derived sub-reports (indented).

        Returns
        -------
        The formatted summary, the same text produced by ``print(report)``.
        """
        lines = [
            f"aggregate: {self.aggregate_status}   (worst level: {self.worst_level})"
        ]
        for verdict in self.per_level:
            abs_err = (
                "  -  "
                if verdict.abs_err_est_GHz is None
                else f"{verdict.abs_err_est_GHz:.2e}"
            )
            eps_gap = (
                "  -  " if verdict.eps_gap_est is None else f"{verdict.eps_gap_est:.2e}"
            )
            warns = (
                "  [" + ", ".join(verdict.warnings) + "]" if verdict.warnings else ""
            )
            lines.append(
                f"  level {verdict.level_index}: {verdict.status:<16} "
                f"evidence={verdict.evidence:<18} "
                f"channel={str(verdict.truncation_channel):<18} "
                f"abs_err={abs_err}  eps_gap={eps_gap}  "
                f"via {verdict.estimator_method}{warns}"
            )
        if self.channel_breakdown_GHz:
            breakdown = {
                name: f"{value:.2e}"
                for name, value in self.channel_breakdown_GHz.items()
            }
            lines.append(f"  channel_breakdown_GHz: {breakdown}")
        for recommendation in self.recommendations:
            lines.append(f"  recommendation: {recommendation}")
        if self.derived:
            for name, sub_report in self.derived.items():
                lines.append(f"  derived [{name}]:")
                lines.append(textwrap.indent(sub_report.summary(), "    "))
        return "\n".join(lines)

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


# Ordered list used by callers wanting to filter on minimum evidence strength.
EVIDENCE_ORDER: tuple[Evidence, ...] = (
    "certified",
    "verified_empirical",
    "calibrated",
    "perturbative",
    "diagnostic",
    "unverified",
)


def evidence_at_least(actual: Evidence, minimum: Evidence) -> bool:
    """Return True if ``actual`` is at least as strong as ``minimum``.

    Used by ``mode='strict'`` to gate which evidence levels may still claim
    ``converged`` status.
    """
    return EVIDENCE_ORDER.index(actual) <= EVIDENCE_ORDER.index(minimum)
