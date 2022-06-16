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


def test_eigenvals_harmonic():
    ref_eigs = np.array(
        [0.0, 0.03559404, 0.05819727, 0.09378676, 4.39927874, 4.43488613]
    )
    DFC = scq.Circuit.from_yaml(
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
    DFC = scq.Circuit.from_yaml(
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
