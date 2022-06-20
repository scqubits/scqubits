# test_namedslotsndarray.py
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

from scqubits.core.namedslots_array import NamedSlotsNdarray

paramvals1 = np.asarray(range(10))
paramvals2 = np.linspace(0.0, 1.0, 100)
paramvals_by_name = {"p1": paramvals1, "p2": paramvals2}

data = np.random.rand(len(paramvals1), len(paramvals2), 10)


def test_initialize():
    tst = NamedSlotsNdarray(data, paramvals_by_name)


def test_index_access():
    tst = NamedSlotsNdarray(data, paramvals_by_name)
    assert np.allclose(tst[0], data[0])


def test_name_access():
    tst = NamedSlotsNdarray(data, paramvals_by_name)
    assert np.allclose(tst["p2":1], data[:, 1, :])


def test_value_access():
    tst = NamedSlotsNdarray(data, paramvals_by_name)
    param_val = float(paramvals1[4])
    assert np.allclose(tst["p1":param_val], data[4, :, :])


def test_named_slice():
    tst = NamedSlotsNdarray(data, paramvals_by_name)
    assert np.allclose(tst["p2":2:-1], data[:, 2:-1, :])
    assert np.allclose(tst["p2":2, "p1":0], tst["p1":0, "p2":2])
