# test_noise.py
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
import scipy as sp

from scipy.sparse import csc_matrix

import scqubits.settings

from scqubits import (
    Cos2PhiQubit,
    Fluxonium,
    FluxQubit,
    Grid1d,
    Transmon,
    TunableTransmon,
    ZeroPi,
)
from scqubits.core.noise import calc_therm_ratio, convert_eV_to_Hz

# WE do not need a warning during testing
scqubits.settings.T1_DEFAULT_WARNING = False

data = {
    "Transmon": np.array(
        [
            1.33428506e07,
            3.77005675e00,
            2.16683864e06,
            5.02320540e02,
            2.16683864e06,
            3.77005240e00,
        ]
    ),
    "TunableTransmon": np.array(
        [
            2.03732888e04,
            1.60438006e06,
            9.42324266e05,
            1.50547341e04,
            np.inf,
            1.86529682e-01,
            1.50547341e04,
            1.19075232e04,
        ]
    ),
    "Fluxonium": np.array(
        [
            4.32535846e06,
            np.inf,
            5.48954061e06,
            8.71690601e02,
            1.09436102e08,
            5.18435915e06,
            np.inf,
            2.60287723e06,
            2.36244752e06,
        ]
    ),
    "FluxQubit": np.array(
        [
            44697811.55376147,
            44697796.19976752,
            1822725.05439264,
            1685277.68169804,
            66760.19238822,
            np.inf,
            61859.237880050845,
        ]
    ),
    "ZeroPi": np.array(
        [3806664.01882053, 41202847.68864658, np.inf, np.inf, np.inf, 3484716.71474053]
    ),
    "Cos2PhiQubit": np.array(
        [
            138385704.8120299,
            np.inf,
            np.inf,
            1252199267.3149421,
            1273137.1296252932,
            422898419.10840535,
            1268030.4864707834,
            2490421.4209534894,
        ]
    ),
}


def calc_coherence(qubit, noise_methods=None):
    if noise_methods is None:
        noise_methods = qubit.supported_noise_channels() + [
            "t1_effective",
            "t2_effective",
        ]

    def cap_coherence(time):
        return np.inf if time > 1e14 else time

    return np.array([cap_coherence(getattr(qubit, m)()) for m in noise_methods])


def compare_coherence_to_reference(qubit, qubit_name):
    noise = calc_coherence(qubit)
    print(
        "comparison:\n", [(noise[i], data[qubit_name][i]) for i, _ in enumerate(noise)]
    )
    return np.allclose(noise, data[qubit_name], equal_nan=True)


class TestNoise:
    def test_Transmon(self):
        qubit = Transmon(EJ=0.5, EC=12.0, ng=0.3, ncut=150)
        assert compare_coherence_to_reference(qubit, "Transmon")

    def test_TunableTransmon(self):
        qubit = TunableTransmon(EJmax=20.0, EC=0.5, d=0.00, flux=0.04, ng=0.3, ncut=150)
        assert compare_coherence_to_reference(qubit, "TunableTransmon")

    def test_Fluxonium(self):
        qubit = Fluxonium(EJ=8.9, EC=2.5, EL=0.5, cutoff=150, flux=0.5)
        assert compare_coherence_to_reference(qubit, "Fluxonium")

    def test_FluxQubit(self):
        RATIO = 60.0
        ALPHA = 0.8
        qubit = FluxQubit(
            EJ1=1.0,
            EJ2=1.0,
            EJ3=ALPHA * 1.0,
            ECJ1=1.0 / RATIO,
            ECJ2=1.0 / RATIO,
            ECJ3=1.0 / ALPHA / RATIO,
            ECg1=50.0 / RATIO,
            ECg2=50.0 / RATIO,
            ng1=0.0,
            ng2=0.0,
            flux=0.4,
            ncut=10,
        )
        assert compare_coherence_to_reference(qubit, "FluxQubit")

    def test_ZeroPi(self):
        phi_grid = Grid1d(-6 * np.pi, 6 * np.pi, 200)
        EJ_CONST = 1 / 3.95  # note that EJ and ECJ are interrelated
        qubit = ZeroPi(
            grid=phi_grid,
            EJ=EJ_CONST,
            EL=10.0 ** (-2),
            ECJ=1 / (8.0 * EJ_CONST),
            EC=None,
            ECS=10.0 ** (-3),
            ng=0.1,
            flux=0.23,
            ncut=30,
        )
        assert compare_coherence_to_reference(qubit, "ZeroPi")

    def test_Cos2PhiQubit(self):
        qubit = Cos2PhiQubit(
            EJ=15.0,
            ECJ=2.0,
            EL=1.0,
            EC=0.04,
            dCJ=0.0,
            dL=0.6,
            dEJ=0.0,
            flux=0.5,
            ng=0.0,
            ncut=7,
            zeta_cut=30,
            phi_cut=7,
        )
        assert compare_coherence_to_reference(qubit, "Cos2PhiQubit")


@pytest.fixture
def transmon():
    return Transmon(EJ=0.5, EC=12.0, ng=0.3, ncut=150)


@pytest.fixture
def tunable_transmon():
    return TunableTransmon(EJmax=20.0, EC=0.5, d=0.00, flux=0.04, ng=0.3, ncut=150)


class TestNoiseHelpers:
    """Unit tests for module-level helper functions in noise.py."""

    def test_calc_therm_ratio_standard_units(self):
        omega = 2 * np.pi * 5e9
        T = 0.020
        expected = sp.constants.hbar * omega / (sp.constants.k * T)
        assert np.isclose(
            calc_therm_ratio(omega, T, omega_in_standard_units=True), expected
        )

    def test_calc_therm_ratio_system_units(self):
        omega_sys = 2 * np.pi * 5.0
        T = 0.020
        direct = calc_therm_ratio(2 * np.pi * 5e9, T, omega_in_standard_units=True)
        assert np.isclose(calc_therm_ratio(omega_sys, T), direct)

    def test_convert_eV_to_Hz(self):
        expected = sp.constants.e / sp.constants.h
        assert np.isclose(convert_eV_to_Hz(1.0), expected)
        assert np.isclose(convert_eV_to_Hz(0.0), 0.0)


class TestNoiseBranches:
    @pytest.mark.parametrize(
        "channel",
        [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ],
    )
    def test_get_rate_is_reciprocal(self, transmon, channel):
        time = getattr(transmon, channel)(get_rate=False)
        rate = getattr(transmon, channel)(get_rate=True)
        assert np.isclose(rate * time, 1.0)

    @pytest.mark.parametrize(
        "channel",
        [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ],
    )
    def test_explicit_esys_matches_default_esys(self, transmon, channel):
        esys = transmon.eigensys(evals_count=4)
        default = getattr(transmon, channel)()
        explicit = getattr(transmon, channel)(esys=esys)
        assert np.isclose(default, explicit)

    def test_unsupported_channel_raises(self, transmon):
        with pytest.raises(RuntimeError, match="not supported"):
            transmon.tphi_1_over_f_flux()

    def test_tphi_1_over_f_rejects_equal_or_negative_indices(self, transmon):
        esys = transmon.eigensys(evals_count=2)
        noise_op = np.eye(transmon.hilbertdim())
        with pytest.raises(ValueError, match="must be different"):
            transmon.tphi_1_over_f(A_noise=1e-6, i=0, j=0, noise_op=noise_op, esys=esys)
        with pytest.raises(ValueError, match="must be different"):
            transmon.tphi_1_over_f(
                A_noise=1e-6, i=-1, j=1, noise_op=noise_op, esys=esys
            )

    def test_t1_effective_equals_sum_of_rates(self, transmon):
        t1_channels = [
            c for c in transmon.effective_noise_channels() if c.startswith("t1")
        ]
        esys = transmon.eigensys(evals_count=3)
        individual_rates = [
            getattr(transmon, c)(esys=esys, get_rate=True) for c in t1_channels
        ]
        expected_rate = sum(individual_rates)
        actual_rate = transmon.t1_effective(esys=esys, get_rate=True)
        assert np.isclose(actual_rate, expected_rate)

    def test_t2_effective_sums_tphi_rates_and_halved_t1_rates(self, transmon):
        channels = transmon.effective_noise_channels()
        esys = transmon.eigensys(evals_count=3)
        expected_rate = 0.0
        for channel in channels:
            rate_k = getattr(transmon, channel)(esys=esys, get_rate=True)
            expected_rate += 0.5 * rate_k if channel.startswith("t1") else rate_k
        actual_rate = transmon.t2_effective(esys=esys, get_rate=True)
        assert np.isclose(actual_rate, expected_rate)

    def test_t1_effective_rejects_tphi_channel(self, transmon):
        with pytest.raises(ValueError, match="Only t1 channels"):
            transmon.t1_effective(noise_channels=["tphi_1_over_f_cc"])

    def test_t1_effective_string_channel_equals_one_element_list(self, transmon):
        esys = transmon.eigensys(evals_count=3)
        from_str = transmon.t1_effective(
            noise_channels="t1_capacitive", esys=esys, get_rate=True
        )
        from_list = transmon.t1_effective(
            noise_channels=["t1_capacitive"], esys=esys, get_rate=True
        )
        assert np.isclose(from_str, from_list)

    def test_effective_rate_rejects_invalid_channel_type(self, transmon):
        esys = transmon.eigensys(evals_count=3)
        with pytest.raises(ValueError, match="noise_channels"):
            transmon._effective_rate(
                noise_channels=[123],
                common_noise_options={},
                esys=esys,
                noise_type="t1",
            )

    def test_tphi_1_over_f_dense_vs_sparse_noise_op(self, transmon):
        esys = transmon.eigensys(evals_count=3)
        dense_op = np.eye(transmon.hilbertdim()) * 0.3
        sparse_op = csc_matrix(dense_op)
        rate_dense = transmon.tphi_1_over_f(
            A_noise=1e-6, i=0, j=1, noise_op=dense_op, esys=esys, get_rate=True
        )
        rate_sparse = transmon.tphi_1_over_f(
            A_noise=1e-6, i=0, j=1, noise_op=sparse_op, esys=esys, get_rate=True
        )
        assert np.isclose(rate_dense, rate_sparse)

    def test_t1_bidirectional_rate_is_at_least_unidirectional_rate(self, transmon):
        # total=True sums the i->j and j->i spectral-density contributions;
        # total=False keeps only one. The total rate must therefore be >= .
        esys = transmon.eigensys(evals_count=3)
        rate_bidirectional = transmon.t1_capacitive(
            i=1, j=0, esys=esys, total=True, get_rate=True
        )
        rate_unidirectional = transmon.t1_capacitive(
            i=1, j=0, esys=esys, total=False, get_rate=True
        )
        assert rate_bidirectional >= rate_unidirectional
        assert rate_bidirectional > 0 and rate_unidirectional > 0
