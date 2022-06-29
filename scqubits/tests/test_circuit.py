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
        [-3.69090429, -3.69049138, -2.89704215, -2.89659842, -2.77231275, -2.76823373]
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
        [-3.69858244, -3.69261899, -2.90463196, -2.89989473, -2.81204032, -2.81003324]
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
            56.52525535996244,
            65.14793883999926,
            73.7706223200361,
            82.39330580007291,
            91.01598928010975,
            95.6773203630156,
        ]
    )
    eigs_test = circ.eigenvals()
    assert np.allclose(eigs_test, eigs_ref)
