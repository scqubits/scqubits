# test_units.py
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

import scqubits as scq

from scqubits import Transmon


class TestUnits:
    def test_get_units(self):
        assert scq.get_units() == "GHz"

    def test_set_units(self):
        scq.set_units("MHz")
        assert scq.get_units() == "MHz"

    def test_units_warning(self):
        scq.set_units("GHz")
        qubit = Transmon(EJ=0.5, EC=12.0, ng=0.3, ncut=150)
        # Expect a warning when changing units since a QuantumSystem is present
        with pytest.warns(UserWarning):
            scq.set_units("MHz")

        # Do not expect warning after deleting the only QuantumSystem
        del qubit
        scq.set_units("kHz")

    def test_units_auxiliary(self):
        scq.get_units_time_label()
        scq.show_supported_units()
        scq.to_standard_units(1.0)
        scq.from_standard_units(1.0)
