# test_parameters.py
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

from scqubits.core.namedslots_array import Parameters

paramvals1 = np.asarray(range(10))
paramvals2 = np.linspace(0.0, 1.0, 100)
paramvals_by_name = {"p1": paramvals1, "p2": paramvals2}


def test_initialize_parameters():
    tst = Parameters(paramvals_by_name)


def test_get_by_name():
    tst = Parameters(paramvals_by_name)
    assert np.allclose(tst["p1"], paramvals1)


def test_get_by_index():
    tst = Parameters(paramvals_by_name)
    assert np.allclose(tst[1], paramvals2)


def test_params_count():
    tst = Parameters(paramvals_by_name)
    assert tst.counts[1] == len(paramvals2)


def test_iterate():
    tst = Parameters(paramvals_by_name)
    lst = [paramvals1, paramvals2]
    for index, paramvals in enumerate(tst):
        assert np.allclose(paramvals, lst[index])


def tst_name():
    tst = Parameters(paramvals_by_name)
    assert tst.names[0] == "p1"


def test_paravals_list():
    tst = Parameters(paramvals_by_name)
    assert tst.paramvals_list == [paramvals1, paramvals2]
