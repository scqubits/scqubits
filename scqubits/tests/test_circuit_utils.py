# test_circuit_utils.py
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
import numpy as np
import pytest
import scipy.sparse as sparse
import sympy as sm

from scqubits.core import discretization
from scqubits.core.branch_metadata import (
    _capacitance_variable_for_branch,
    _junction_order,
)
from scqubits.core.charge_basis_operators import (
    _cos_theta,
    _exp_i_theta_operator,
    _exp_i_theta_operator_conjugate,
    _identity_theta,
    _n_theta_operator,
    _sin_theta,
)
from scqubits.core.circuit_utils import (
    _cos_dia,
    _cos_dia_dense,
    _generate_symbols_list,
    _sin_dia,
    _sin_dia_dense,
    example_circuit,
    get_trailing_number,
    is_potential_term,
    matrix_power_sparse,
    round_symbolic_expr,
    sawtooth_operator,
    sawtooth_potential,
    truncation_template,
)
from scqubits.core.discretized_phi_operators import (
    _cos_phi,
    _i_d_dphi_operator,
    _identity_phi,
    _phi_operator,
    _sin_phi,
)


class TestJunctionOrder:
    @pytest.mark.parametrize(
        "branch_type, expected",
        [("JJ", 1), ("JJ2", 2), ("JJ3", 3), ("JJ5", 5), ("JJs", 1)],
    )
    def test_returns_declared_order(self, branch_type, expected):
        assert _junction_order(branch_type) == expected

    @pytest.mark.parametrize("branch_type", ["C", "L", "", "JAB"])
    def test_raises_on_non_jj_branch(self, branch_type):
        with pytest.raises(ValueError, match="not a JJ"):
            _junction_order(branch_type)


class TestCapacitanceVariableForBranch:
    def test_capacitor_branch_returns_EC(self):
        assert _capacitance_variable_for_branch("C") == "EC"

    def test_junction_branch_returns_ECJ(self):
        assert _capacitance_variable_for_branch("JJ") == "ECJ"

    def test_inductor_branch_raises(self):
        with pytest.raises(ValueError, match="not a capacitor or a JJ"):
            _capacitance_variable_for_branch("L")


class TestGetTrailingNumber:
    @pytest.mark.parametrize(
        "input_str, expected",
        [("a23", 23), ("var10", 10), ("θ7", 7), ("123", 123), ("x0", 0)],
    )
    def test_returns_trailing_integer(self, input_str, expected):
        assert get_trailing_number(input_str) == expected

    @pytest.mark.parametrize("input_str", ["foo", "abc", ""])
    def test_raises_when_no_trailing_digits(self, input_str):
        with pytest.raises(ValueError, match="no trailing digits"):
            get_trailing_number(input_str)


class TestIsPotentialTerm:
    def test_theta_symbol_is_potential(self):
        theta1 = sm.symbols("θ1")
        assert is_potential_term(theta1) is True

    def test_phi_symbol_is_potential(self):
        Phi1 = sm.symbols("Φ1")
        assert is_potential_term(Phi1) is True

    def test_charge_symbol_is_not_potential(self):
        Q1 = sm.symbols("Q1")
        assert is_potential_term(Q1) is False

    def test_mixed_term_is_potential(self):
        theta1, Q1 = sm.symbols("θ1 Q1")
        assert is_potential_term(theta1 * Q1) is True


class TestExampleCircuit:
    @pytest.mark.parametrize("name", ["fluxonium", "transmon", "cos2phi", "zero_pi"])
    def test_returns_branches_yaml_for_known_qubit(self, name):
        result = example_circuit(name)
        assert isinstance(result, str)
        assert "branches" in result

    def test_raises_on_unknown_qubit_name(self):
        with pytest.raises(AttributeError, match="not available"):
            example_circuit("not_a_qubit")


class TestDiagonalTrig:
    def test_cos_dia_matches_np_cos(self):
        diag = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        x = sparse.diags(diag).tocsc()
        result = _cos_dia(x)
        assert np.allclose(result.diagonal(), np.cos(diag))

    def test_sin_dia_matches_np_sin(self):
        diag = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        x = sparse.diags(diag).tocsc()
        result = _sin_dia(x)
        assert np.allclose(result.diagonal(), np.sin(diag))

    def test_cos_dia_dense_matches_np_cos(self):
        diag = np.array([0.0, np.pi / 3, np.pi])
        x = np.diag(diag)
        result = _cos_dia_dense(x)
        assert np.allclose(np.diag(result), np.cos(diag))

    def test_sin_dia_dense_matches_np_sin(self):
        diag = np.array([0.0, np.pi / 3, np.pi])
        x = np.diag(diag)
        result = _sin_dia_dense(x)
        assert np.allclose(np.diag(result), np.sin(diag))


class TestMatrixPowerSparse:
    def test_power_3_matches_np(self):
        A = np.array([[1.0, 0.5], [0.25, 2.0]])
        expected = A @ A @ A
        result = matrix_power_sparse(A, 3)
        assert sparse.issparse(result)
        assert np.allclose(result.toarray(), expected)

    def test_power_0_is_identity(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        result = matrix_power_sparse(A, 0)
        assert np.allclose(result.toarray(), np.eye(2))


class TestSawtooth:
    def test_potential_is_2pi_periodic(self):
        phi = np.linspace(-np.pi, np.pi, 21)
        assert np.allclose(sawtooth_potential(phi), sawtooth_potential(phi + 2 * np.pi))

    def test_potential_is_even(self):
        phi = np.linspace(0.01, 2.0, 10)
        assert np.allclose(sawtooth_potential(phi), sawtooth_potential(-phi))

    def test_operator_diagonal_matches_potential(self):
        diag = np.array([0.1, 0.5, 1.0, 2.0])
        x = sparse.diags(diag).tocsc()
        op = sawtooth_operator(x)
        assert sparse.issparse(op)
        assert np.allclose(op.diagonal(), sawtooth_potential(diag))


class TestTruncationTemplate:
    def test_flat_hierarchy(self):
        result = truncation_template([[0], [1], [2]], individual_trunc_dim=6)
        assert result == [6, 6, 6]

    def test_nested_hierarchy(self):
        result = truncation_template(
            [[0], [[1], [2]]], individual_trunc_dim=6, combined_trunc_dim=30
        )
        assert result == [6, [30, [6, 6]]]


class TestGenerateSymbolsList:
    def test_names_concatenated(self):
        result = _generate_symbols_list("x", [0, 1, 2])
        assert [s.name for s in result] == ["x0", "x1", "x2"]
        assert all(isinstance(s, sm.Symbol) for s in result)


class TestRoundSymbolicExpr:
    def test_strips_noise_level_coefficient(self):
        x, y = sm.symbols("x y")
        expr = sm.Float("2.5") * x + sm.Float("1e-15") * y
        rounded = round_symbolic_expr(expr, 12)
        assert rounded.coeff(y) == 0

    def test_keeps_signal_coefficient(self):
        x = sm.symbols("x")
        expr = sm.Float("2.5") * x
        rounded = round_symbolic_expr(expr, 12)
        assert float(rounded.coeff(x)) == pytest.approx(2.5)

    def test_preserves_integer_coefficients(self):
        x = sm.symbols("x")
        expr = 2 * x
        assert round_symbolic_expr(expr, 12) == 2 * x


class TestGridOperatorsPhi:
    @pytest.fixture
    def grid(self):
        return discretization.Grid1d(-2 * np.pi, 2 * np.pi, 20)

    def test_identity_phi_shape_and_diagonal(self, grid):
        op = _identity_phi(grid)
        assert op.shape == (20, 20)
        assert np.allclose(op.diagonal(), np.ones(20))

    def test_phi_operator_diagonal_is_linspace(self, grid):
        op = _phi_operator(grid)
        assert np.allclose(op.diagonal(), grid.make_linspace())

    def test_cos_phi_diagonal(self, grid):
        op = _cos_phi(grid)
        assert np.allclose(op.diagonal(), np.cos(grid.make_linspace()))

    def test_sin_phi_diagonal(self, grid):
        op = _sin_phi(grid)
        assert np.allclose(op.diagonal(), np.sin(grid.make_linspace()))

    def test_i_d_dphi_is_hermitian(self, grid):
        # The finite-difference stencil is chosen so that i*d/dphi is Hermitian.
        op = _i_d_dphi_operator(grid).toarray()
        assert np.allclose(op, op.conj().T)


class TestChargeBasisOperators:
    @pytest.mark.parametrize("ncut", [1, 3, 5])
    def test_identity_theta_shape(self, ncut):
        op = _identity_theta(ncut)
        dim = 2 * ncut + 1
        assert op.shape == (dim, dim)
        assert np.allclose(op.diagonal(), np.ones(dim))

    @pytest.mark.parametrize("ncut", [1, 3, 5])
    def test_n_theta_diagonal(self, ncut):
        op = _n_theta_operator(ncut)
        assert np.allclose(op.diagonal(), np.arange(-ncut, ncut + 1))

    def test_exp_i_theta_operators_are_mutual_adjoints(self):
        op = _exp_i_theta_operator(2, prefactor=1).toarray()
        conj = _exp_i_theta_operator_conjugate(2).toarray()
        assert np.allclose(op, conj.conj().T)

    def test_cos_theta_is_hermitian(self):
        op = _cos_theta(3).toarray()
        assert np.allclose(op, op.conj().T)

    def test_sin_theta_is_hermitian(self):
        op = _sin_theta(3).toarray()
        assert np.allclose(op, op.conj().T)

    def test_cos2_plus_sin2_equals_identity_away_from_truncation(self):
        # cos^2 + sin^2 = I holds only away from the +-ncut truncation edges.
        ncut = 5
        cos_op = _cos_theta(ncut).toarray()
        sin_op = _sin_theta(ncut).toarray()
        product = cos_op @ cos_op + sin_op @ sin_op
        interior = slice(2, 2 * ncut - 1)
        assert np.allclose(
            product[interior, interior], np.eye(2 * ncut + 1)[interior, interior]
        )
