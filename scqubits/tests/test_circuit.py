# test_circuit.py
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

import scqubits as scq
import numpy as np
import os

TESTDIR, _ = os.path.split(scq.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")
DATADIR = os.path.join(TESTDIR, "data", "")


def test_zero_pi_discertized():
    """
    Test for symmetric zero-pi in discretized phi basis.
    """
    zp_yaml = """# zero-pi circuit
    branches:
    - ["JJ", 1, 2, 10, 20]
    - ["JJ", 3, 4, 10, 20]
    - ["L", 2, 3, 0.008]
    - ["L", 4, 1, 0.008]
    - ["C", 1, 3, 0.02]
    - ["C", 2, 4, 0.02]
    """

    circ_d = scq.Circuit.from_yaml(zp_yaml, from_file=False, ext_basis="discretized")
    circ_d.cutoff_n_1 = 30
    circ_d.cutoff_ext_2 = 30
    circ_d.cutoff_ext_3 = 80
    circ_d.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[30, 20])

    circ_d.cutoff_ext_3 = 200
    sym_zp = circ_d.subsystems[0]
    eigensys = sym_zp.eigensys()
    eigs = eigensys[0]
    eigs_ref = np.array(
        [16.30909571, 16.30950862, 17.10295785, 17.10340158, 17.22768726, 17.23176627]
    )

    assert np.allclose(eigs, eigs_ref)


def test_zero_pi_harmonic():
    """
    Test for symmmetric zero-pi in harmonic oscillator basis.
    """
    zp_yaml = """# zero-pi circuit
    branches:
    - ["JJ", 1, 2, 10, 20]
    - ["JJ", 3, 4, 10, 20]
    - ["L", 2, 3, 0.008]
    - ["L", 4, 1, 0.008]
    - ["C", 1, 3, 0.02]
    - ["C", 2, 4, 0.02]
    """
    circ = scq.Circuit.from_yaml(zp_yaml, from_file=False, ext_basis="harmonic")
    circ.cutoff_n_1 = 30
    circ.cutoff_ext_2 = 30
    circ.cutoff_ext_3 = 80
    circ.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[30, 20])
    circ.cutoff_ext_3 = 200
    sym_zp = circ.subsystems[0]
    eigensys = sym_zp.eigensys()
    eigs = eigensys[0]
    eig_ref = np.array(
        [16.30141756, 16.30738101, 17.09536804, 17.10010527, 17.18795968, 17.18996676]
    )
    assert np.allclose(eigs, eig_ref)


def test_eigenvals_harmonic():
    ref_eigs = np.array(
        [0.0, 0.03559404, 0.05819727, 0.09378676, 4.39927874, 4.43488613]
    )
    DFC = scq.Circuit(
        DATADIR + "circuit_DFC.yaml",
        ext_basis="harmonic",
        initiate_sym_calc=False,
        basis_completion="canonical",
    )

    closure_branches = [DFC.branches[0], DFC.branches[4], DFC.branches[-1]]
    system_hierarchy = [[[1], [3]], [2], [4]]
    subsystem_trunc_dims = [[34, [6, 6]], 6, 6]

    DFC.configure(
        closure_branches=closure_branches,
        system_hierarchy=system_hierarchy,
        subsystem_trunc_dims=subsystem_trunc_dims,
    )

    DFC._Φ1 = 0.5 + 0.01768
    DFC._Φ2 = -0.2662
    DFC._Φ3 = -0.5 + 0.01768

    DFC._cutoff_ext_1 = 110
    DFC._cutoff_ext_2 = 110
    DFC._cutoff_ext_3 = 110
    DFC._cutoff_ext_4 = 110

    DFC.EJ = 4.6
    eigs = DFC.eigenvals()
    generated_eigs = eigs - eigs[0]
    assert np.allclose(generated_eigs, ref_eigs)


def test_eigenvals_discretized():
    ref_eigs = np.array(
        [0.0, 0.03559217, 0.05819503, 0.09378266, 4.39921833, 4.43482385]
    )
    DFC = scq.Circuit(
        DATADIR + "circuit_DFC.yaml",
        ext_basis="discretized",
        initiate_sym_calc=False,
        basis_completion="canonical",
    )

    closure_branches = [DFC.branches[0], DFC.branches[4], DFC.branches[-1]]
    system_hierarchy = [[[1], [3]], [2], [4]]
    subsystem_trunc_dims = [[34, [6, 6]], 6, 6]

    DFC.configure(
        closure_branches=closure_branches,
        system_hierarchy=system_hierarchy,
        subsystem_trunc_dims=subsystem_trunc_dims,
    )

    DFC._Φ1 = 0.5 + 0.01768
    DFC._Φ2 = -0.2662
    DFC._Φ3 = -0.5 + 0.01768

    DFC._cutoff_ext_1 = 110
    DFC._cutoff_ext_2 = 110
    DFC._cutoff_ext_3 = 110
    DFC._cutoff_ext_4 = 110

    DFC.EJ = 4.6
    eigs = DFC.eigenvals()
    generated_eigs = eigs - eigs[0]
    assert np.allclose(generated_eigs, ref_eigs)


def test_harmonic_oscillator():
    lc_yaml = """# LC circuit
branches:
- [L, 0, 1, 1]
- [C, 0, 2, 2]
- [L, 0, 3, 4.56]
- [C, 2, 3, 40]
- [C, 2, 1, EJ=40]
- [C, 4, 1, 10]
- [L, 4, 2, 10]
"""
    circ = scq.Circuit(
        lc_yaml, from_file=False, initiate_sym_calc=True, ext_basis="harmonic"
    )
    circ.EJ = 0.01
    eigs_ref = np.array(
        [
            35.68194846783804,
            39.59478931227575,
            43.50763015671346,
            47.42047100115117,
            51.33331184558888,
            55.24615269002659,
        ]
    )
    eigs_test = circ.eigenvals()
    assert np.allclose(eigs_test, eigs_ref)
