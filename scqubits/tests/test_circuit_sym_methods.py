# test_circuit_sym_methods.py
# meant to be run with 'pytest'
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

import pytest
import sympy as sm

from scqubits.core.circuit_internals.sym_methods import CircuitSymMethods


class TestContainsTrigonometricTerms:
    def test_constant_is_not_trigonometric(self):
        expr = sm.Float(1.0)
        assert CircuitSymMethods._contains_trigonometric_terms(expr) is False

    def test_polynomial_is_not_trigonometric(self):
        x = sm.symbols("x")
        expr = x**2 + 3 * x + 2
        assert CircuitSymMethods._contains_trigonometric_terms(expr) is False

    def test_cos_term_detected(self):
        x = sm.symbols("x")
        assert CircuitSymMethods._contains_trigonometric_terms(sm.cos(x)) is True

    def test_sin_term_detected(self):
        x = sm.symbols("x")
        assert CircuitSymMethods._contains_trigonometric_terms(sm.sin(x)) is True

    def test_sawtooth_term_detected(self):
        x = sm.symbols("x")
        saw = sm.Function("saw", real=True)
        assert CircuitSymMethods._contains_trigonometric_terms(saw(x)) is True

    def test_mixed_expression_detected(self):
        x, y = sm.symbols("x y")
        expr = x**2 + sm.cos(y)
        assert CircuitSymMethods._contains_trigonometric_terms(expr) is True


class TestIsSymbolPeriodicCharge:
    @pytest.mark.parametrize("name", ["n0", "n1", "n23"])
    def test_periodic_charge_names_recognized(self, name):
        assert CircuitSymMethods._is_symbol_periodic_charge(sm.symbols(name))

    @pytest.mark.parametrize("name", ["Q1", "θ1", "n", "N1", "ncut", "n_1"])
    def test_non_periodic_charge_names_rejected(self, name):
        assert not CircuitSymMethods._is_symbol_periodic_charge(sm.symbols(name))


class TestIsSymbolContinuousCharge:
    @pytest.mark.parametrize("name", ["Q0", "Q1", "Q42"])
    def test_continuous_charge_names_recognized(self, name):
        assert CircuitSymMethods._is_symbol_continuous_charge(sm.symbols(name))

    @pytest.mark.parametrize("name", ["n1", "θ1", "q1", "Q"])
    def test_non_continuous_charge_names_rejected(self, name):
        assert not CircuitSymMethods._is_symbol_continuous_charge(sm.symbols(name))


class TestIsSymbolPhase:
    @pytest.mark.parametrize("name", ["θ0", "θ1", "θ7"])
    def test_phase_names_recognized(self, name):
        assert CircuitSymMethods._is_symbol_phase(sm.symbols(name))

    @pytest.mark.parametrize("name", ["n1", "Q1", "theta1", "θ"])
    def test_non_phase_names_rejected(self, name):
        assert not CircuitSymMethods._is_symbol_phase(sm.symbols(name))


class TestFindAndCategorizeVariableIndices:
    def test_empty_expression(self):
        periodic, extended, phase = (
            CircuitSymMethods._find_and_categorize_variable_indices(sm.Float(1.0))
        )
        assert periodic == set()
        assert extended == set()
        assert phase == set()

    def test_periodic_only(self):
        n1, n2 = sm.symbols("n1 n2")
        periodic, extended, phase = (
            CircuitSymMethods._find_and_categorize_variable_indices(n1 + n2)
        )
        assert periodic == {1, 2}
        assert extended == set()
        assert phase == set()

    def test_continuous_only(self):
        Q1, Q3 = sm.symbols("Q1 Q3")
        periodic, extended, phase = (
            CircuitSymMethods._find_and_categorize_variable_indices(Q1 * Q3)
        )
        assert periodic == set()
        assert extended == {1, 3}
        assert phase == set()

    def test_phase_only(self):
        theta1, theta2 = sm.symbols("θ1 θ2")
        periodic, extended, phase = (
            CircuitSymMethods._find_and_categorize_variable_indices(
                sm.cos(theta1) + theta2
            )
        )
        assert periodic == set()
        assert extended == set()
        assert phase == {1, 2}

    def test_all_three_types_together(self):
        n1, Q2, theta3 = sm.symbols("n1 Q2 θ3")
        periodic, extended, phase = (
            CircuitSymMethods._find_and_categorize_variable_indices(n1 + Q2 + theta3)
        )
        assert periodic == {1}
        assert extended == {2}
        assert phase == {3}

    def test_ignores_non_categorized_symbols(self):
        # External flux Φ, identity I, arbitrary symbol x should not be
        # picked up.
        n1, Phi, x = sm.symbols("n1 Φ1 x")
        periodic, extended, phase = (
            CircuitSymMethods._find_and_categorize_variable_indices(n1 + Phi + x)
        )
        assert periodic == {1}
        assert extended == set()
        assert phase == set()
