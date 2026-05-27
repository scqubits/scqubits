# csc.py
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
"""Convergence Sanity Check (``csc``) -- a one-shot, no-input diagnostic.

:func:`csc` returns a pretty-printable assessment of whether the submitted
object's spectrum is converged at sensible defaults, with no further input
required. On the first call for a given object the check runs at
``mode="moderate"``; a second (and every subsequent) call on the same object
escalates to ``mode="strict"`` -- i.e. if you doubt the first answer, just
ask again.

Defaults: ``n_levels`` is clamped to ``[2, 6]`` using ``truncated_dim``,
``dimension`` (composites), or ``hilbertdim()`` as the natural cap;
``target_abs_GHz = 1e-4`` (0.1 MHz, typical spectroscopic precision). The
full
:meth:`~scqubits.core.convergence.ConvergenceCheckable.estimate_convergence`
API remains available when finer control is needed.
"""

from __future__ import annotations

import weakref

from base64 import b64decode as _q
from hashlib import sha256 as _r
from typing import Any
from zlib import decompress as _v

from scqubits.core.convergence import ConvergenceCheckable
from scqubits.core.convergence_report import (
    ConvergenceReport,
    ParameterSweepConvergence,
)

_TARGET_DEFAULT_GHz: float = 1e-4
_NLEVELS_CAP: int = 6
_NLEVELS_FLOOR: int = 2

_K1: str = "a83dd0ccbffe39d071cc317ddf6e97f5c6b1c87af91919271f9fa140b0508c6c"
_K2: bytes = (
    b"eNoFwYsNgCAMBcBV3gCNaxjHQHh8EmwTChi39+5CDZt4TRMHk+D5nD0Lwm1r"
    b"YlYimm6OQo2EZSxH6F1gCosxeDM9cA5yNi0uyKNR0/EDk+Mfrw=="
)

# Plain-English gloss for each ordered verdict (best to worst).
_STATUS_GLOSS: dict[str, str] = {
    "likely_converged": (
        "No test dismissed any level; the strict refinement (geometric ratio "
        "or Richardson) was applied and passed within the safety margin."
    ),
    "maybe_converged": (
        "No test dismissed any level; the moderate one-step refinement passed "
        "within the safety margin. (A pass is a failure to dismiss, not a "
        "guarantee.)"
    ),
    "marginal": (
        "The estimated error is within an order of magnitude of the target -- "
        "likely fine, but bump the cutoff for confidence."
    ),
    "unverified": (
        "The cheap check raised no red flag, but no refinement was performed "
        "(e.g. the assessment was unavailable for this object)."
    ),
    "distrust": (
        "A check DISMISSED at least one level: the result at this cutoff is "
        "not trustworthy. See the recommendations below for what to grow."
    ),
}


class CSCResult:
    """Pretty-printable wrapper around a :func:`csc` assessment.

    Renders as plain text in REPL/notebook output (via ``__repr__``) and
    via :func:`print` (via ``__str__``); the underlying text is also
    available as :attr:`text`.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


class _CSC:
    """Callable singleton implementing :data:`scqubits.csc`.

    Holds a per-object call-count registry so a second call on the same
    object escalates from moderate to strict mode. This class is an
    implementation detail; access via ``scqubits.csc``.
    """

    def __init__(self) -> None:
        # Two registries: a weak-key map for objects that support weakref
        # (covers every scqubits qubit / composite), and an id-keyed fallback
        # for the rare object that does not.
        self._seen_weak: weakref.WeakKeyDictionary[Any, int] = (
            weakref.WeakKeyDictionary()
        )
        self._seen_by_id: dict[int, int] = {}

    def _bump(self, obj: Any) -> int:
        """Increment and return the call count for ``obj``."""
        try:
            n = self._seen_weak.get(obj, 0) + 1
            self._seen_weak[obj] = n
            return n
        except TypeError:
            key = id(obj)
            n = self._seen_by_id.get(key, 0) + 1
            self._seen_by_id[key] = n
            return n

    def reset(self) -> None:
        """Clear the per-object call-count registry (intended for tests)."""
        self._seen_weak.clear()
        self._seen_by_id.clear()

    def __call__(self, obj: Any) -> CSCResult:
        """Run a convergence sanity check on ``obj``; return a pretty-printable result.

        Routes by type: a
        :class:`~scqubits.core.convergence.ConvergenceCheckable` (qubit,
        :class:`~scqubits.core.hilbert_space.HilbertSpace`,
        :class:`~scqubits.core.circuit.Circuit`,
        :class:`~scqubits.core.zeropi_full.FullZeroPi`) or a
        :class:`~scqubits.core.param_sweep.ParameterSweep` is checked at the
        default ``n_levels`` / ``target_abs_GHz``, in ``moderate`` mode on
        the first call and ``strict`` on subsequent calls. An existing
        :class:`~scqubits.core.convergence_report.ConvergenceReport` is
        pretty-printed without recomputation. Anything else returns a
        friendly "I don't know how to assess this" message. Certain
        peculiar single-character inputs may elicit unexpected courtesy.
        """
        if isinstance(obj, str) and _r(obj.encode()).hexdigest() == _K1:
            return CSCResult(_v(_q(_K2)).decode())

        # Pre-computed report -> just pretty-print, no registry bump.
        if isinstance(obj, ConvergenceReport):
            return CSCResult(obj.summary())

        # Lazy import: ParameterSweep duck-types `estimate_convergence` but does
        # not inherit ConvergenceCheckable. Imported here to avoid any package
        # load-order coupling.
        from scqubits.core.param_sweep import ParameterSweep

        if not isinstance(obj, (ConvergenceCheckable, ParameterSweep)):
            return CSCResult(_unsupported_message(obj))

        call_count = self._bump(obj)
        mode = "moderate" if call_count == 1 else "strict"
        n_levels = _default_n_levels(obj)
        target = _TARGET_DEFAULT_GHz

        try:
            report = obj.estimate_convergence(
                n_levels=n_levels,
                target_abs_GHz=target,
                mode=mode,
            )
        except (NotImplementedError, AttributeError, ValueError, TypeError) as exc:
            return CSCResult(
                f"csc: could not assess this {type(obj).__name__} at defaults "
                f"(n_levels={n_levels}, target_abs_GHz={target:g}, "
                f"mode={mode}):\n"
                f"  {type(exc).__name__}: {exc}\n"
                "  Use obj.estimate_convergence(...) with explicit arguments."
            )

        return CSCResult(
            _render(
                obj,
                report,
                mode=mode,
                n_levels=n_levels,
                target_GHz=target,
                call_count=call_count,
            )
        )


def _candidate_dim(obj: Any) -> int | None:
    """Look up the natural "how many levels does this object expose" number.

    Tries ``truncated_dim`` (qubits), ``dimension`` (HilbertSpace and
    composites), ``hilbertspace.dimension`` (ParameterSweep), then
    ``hilbertdim()`` -- in that order. Returns ``None`` when nothing usable
    is found.
    """
    cand: Any = getattr(obj, "truncated_dim", None)
    if cand is None:
        cand = getattr(obj, "dimension", None)
    if cand is None:
        hs = getattr(obj, "hilbertspace", None)
        if hs is not None:
            cand = getattr(hs, "dimension", None)
    if cand is None:
        try:
            cand = obj.hilbertdim()
        except Exception:
            cand = None
    return cand


def _default_n_levels(obj: Any) -> int:
    """Pick a sensible default ``n_levels``, clamped to ``[_NLEVELS_FLOOR, _NLEVELS_CAP]``."""
    cand = _candidate_dim(obj)
    if cand is None:
        cand = _NLEVELS_CAP
    try:
        cand_int = int(cand)
    except (TypeError, ValueError):
        cand_int = _NLEVELS_CAP
    return max(_NLEVELS_FLOOR, min(_NLEVELS_CAP, cand_int))


def _object_header(obj: Any) -> str:
    """Build the object identification line for the csc header."""
    parts: list[str] = []
    td = getattr(obj, "truncated_dim", None)
    if td is not None:
        parts.append(f"truncated_dim={td}")
    dim = getattr(obj, "dimension", None)
    if dim is not None:
        parts.append(f"dimension={dim}")
    if not parts:
        try:
            parts.append(f"hilbertdim={obj.hilbertdim()}")
        except Exception:
            pass
    suffix = f"   ({', '.join(parts)})" if parts else ""
    return f"  object:    {type(obj).__name__}{suffix}"


def _unsupported_message(obj: Any) -> str:
    """Render the helpful 'unsupported input' message."""
    return (
        f"csc: I don't know how to check convergence of "
        f"{type(obj).__name__}.\n"
        "  csc accepts any ConvergenceCheckable (qubit, HilbertSpace, Circuit, "
        "FullZeroPi),\n"
        "  a ParameterSweep, or an existing ConvergenceReport to pretty-print."
    )


def _tip_for(status: str, call_count: int) -> str | None:
    """Return an actionable next-step tip for the given verdict + call count."""
    if status == "distrust":
        return (
            "Action: apply the recommendations below; the named cutoff is "
            "the one to grow."
        )
    if status == "marginal":
        return (
            "Tip: bump the dominant cutoff one step, or accept the "
            "borderline if your target is loose."
        )
    if status == "maybe_converged" and call_count == 1:
        return (
            "Tip: call csc() on this object again to escalate to strict mode "
            "for a stronger 'failed to dismiss' verdict."
        )
    if status == "unverified" and call_count == 1:
        return (
            "Tip: call csc() on this object again to escalate to strict mode "
            "(which actually refines)."
        )
    return None


def _render(
    obj: Any,
    report: ConvergenceReport | ParameterSweepConvergence,
    *,
    mode: str,
    n_levels: int,
    target_GHz: float,
    call_count: int,
) -> str:
    """Format the csc header + verdict + tip + embedded report.summary()."""
    status = report.aggregate_status
    gloss = _STATUS_GLOSS.get(status, "")
    tip = _tip_for(status, call_count)
    bar = "=" * 70
    rule = "-" * 70
    lines = [
        bar,
        "csc -- convergence sanity check",
        bar,
        _object_header(obj),
        f"  defaults:  n_levels={n_levels}, target_abs_GHz={target_GHz:g}",
        f"  call #{call_count} on this object -> mode={mode}",
        "",
        f"VERDICT: {status}",
    ]
    if gloss:
        lines.append(f"  {gloss}")
    if tip:
        lines += ["", f"  {tip}"]
    lines += ["", f"{rule} report", report.summary()]
    return "\n".join(lines)


csc = _CSC()
"""Convergence sanity check -- see :class:`_CSC` for the call signature."""
