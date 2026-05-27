# test_csc.py
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
"""Tests for the :func:`scqubits.csc` convenience (Convergence Sanity Check)."""

from __future__ import annotations

import pytest

import scqubits as scq

from scqubits.utils.csc import CSCResult, csc


@pytest.fixture(autouse=True)
def _isolate_csc_registry():
    csc.reset()
    yield
    csc.reset()


def _well_converged_transmon() -> scq.Transmon:
    return scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=4)


def _underconverged_transmon() -> scq.Transmon:
    # Tight target + very low ncut for the kept levels -> distrust.
    return scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=4, truncated_dim=4)


class TestEasterEgg:
    def test_capital_I_returns_greeting(self):
        out = csc("I")
        assert isinstance(out, CSCResult)
        text = str(out)
        assert "wondered, myself, about the convergence of us all" in text
        assert "Greetings, friend." in text

    def test_lowercase_i_does_not_trigger_egg(self):
        text = str(csc("i"))
        assert "Greetings" not in text
        assert "don't know" in text

    def test_other_string_falls_through_to_unsupported(self):
        text = str(csc("hello"))
        assert "Greetings" not in text
        assert "don't know" in text


class TestUnsupportedInputs:
    def test_int_returns_helpful_message(self):
        text = str(csc(42))
        assert "don't know" in text
        assert "ConvergenceCheckable" in text

    def test_dict_returns_helpful_message(self):
        text = str(csc({"a": 1}))
        assert "don't know" in text


class TestPrettyPrint:
    def test_csc_result_repr_matches_str(self):
        result = csc("I")
        assert repr(result) == str(result)

    def test_csc_result_text_attribute_carries_payload(self):
        result = csc(42)
        assert result.text == str(result)


class TestCheckable:
    def test_first_call_uses_moderate(self):
        q = _well_converged_transmon()
        text = str(csc(q))
        assert "csc -- convergence sanity check" in text
        assert "call #1 on this object -> mode=moderate" in text
        assert "VERDICT:" in text
        assert "Transmon" in text
        assert "truncated_dim=4" in text

    def test_second_call_escalates_to_strict(self):
        q = _well_converged_transmon()
        _ = csc(q)
        text = str(csc(q))
        assert "call #2 on this object -> mode=strict" in text

    def test_subsequent_calls_stay_strict(self):
        q = _well_converged_transmon()
        for _ in range(3):
            _ = csc(q)
        text = str(csc(q))
        assert "call #4 on this object -> mode=strict" in text

    def test_distinct_objects_each_start_at_call_1(self):
        q1 = _well_converged_transmon()
        q2 = _well_converged_transmon()
        _ = csc(q1)
        text2 = str(csc(q2))
        assert "call #1" in text2

    def test_well_converged_transmon_does_not_distrust(self):
        q = _well_converged_transmon()
        text = str(csc(q))
        assert "VERDICT: distrust" not in text
        assert "VERDICT: likely_converged" in text or "VERDICT: maybe_converged" in text

    def test_underconverged_transmon_dismisses(self):
        # Build a transmon whose ncut is so small the kept levels are not
        # converged at the default 1e-4 GHz target.
        q = _underconverged_transmon()
        text = str(csc(q))
        # Either distrust or marginal -- both are non-passes; we just want the
        # csc verdict to flag a problem.
        assert "VERDICT: distrust" in text or "VERDICT: marginal" in text

    def test_reset_clears_registry(self):
        q = _well_converged_transmon()
        _ = csc(q)
        csc.reset()
        text = str(csc(q))
        assert "call #1" in text


class TestReportPassthrough:
    def test_existing_report_pretty_prints_without_header(self):
        q = _well_converged_transmon()
        report = q.estimate_convergence(
            n_levels=3, target_abs_GHz=1e-4, mode="moderate"
        )
        text = str(csc(report))
        # Should be exactly the report's own summary -- no csc framing.
        assert "csc --" not in text
        assert "aggregate:" in text

    def test_report_passthrough_does_not_bump_registry(self):
        q = _well_converged_transmon()
        report = q.estimate_convergence(
            n_levels=3, target_abs_GHz=1e-4, mode="moderate"
        )
        _ = csc(report)
        # The qubit was not the argument to csc, so its registry entry is still
        # at zero; the next call on the qubit should be call #1.
        text = str(csc(q))
        assert "call #1" in text


class TestHilbertSpace:
    def test_hilbertspace_default_uses_dimension(self):
        # HilbertSpace has no truncated_dim/hilbertdim() but exposes .dimension;
        # csc must surface that in the header and use it as the n_levels cap.
        tmon = scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=4)
        osc = scq.Oscillator(E_osc=5.0, truncated_dim=3)
        hs = scq.HilbertSpace([tmon, osc])
        text = str(csc(hs))
        assert "don't know" not in text
        assert "HilbertSpace" in text
        assert "dimension=" in text


class TestParameterSweep:
    def test_parametersweep_is_routed_not_unsupported(self):
        # ParameterSweep does not inherit ConvergenceCheckable but supplies its
        # own estimate_convergence; csc must route it (not the unsupported path).
        import numpy as np

        tmon = scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=31, truncated_dim=4)
        hs = scq.HilbertSpace([tmon])

        def _update(EJ):
            tmon.EJ = EJ

        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name={"EJ": np.array([18.0, 22.0])},
            update_hilbertspace=_update,
            evals_count=3,
        )
        text = str(csc(sweep))
        assert "don't know how to check" not in text
        # Either a real csc header (success) or a wrapped failure -- both prove
        # the routing landed and ParameterSweep is no longer 'unsupported'.
        assert "csc --" in text or "could not assess" in text


class TestFailureWrap:
    def test_estimate_convergence_failure_is_wrapped(self):
        # F3 contract: a narrowed except wraps expected estimate_convergence
        # failures into a friendly message rather than propagating.
        from scqubits.core.convergence import ConvergenceCheckable

        class _FailingCheckable(ConvergenceCheckable):
            def estimate_convergence(self, **kwargs):
                raise ValueError("simulated estimate_convergence failure")

        text = str(csc(_FailingCheckable()))
        assert "could not assess" in text
        assert "simulated estimate_convergence failure" in text
        assert "ValueError" in text
