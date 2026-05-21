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
