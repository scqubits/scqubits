# test_storage.py
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

from scqubits.core.storage import SpectrumData


def test_subtract_ground_for_each_parameter_value():
    spectrum_data = SpectrumData(
        energy_table=np.array(
            [
                [10.0, 11.0, 13.0],
                [20.0, 22.0, 25.0],
            ]
        ),
        system_params={},
    )

    spectrum_data.subtract_ground()

    np.testing.assert_allclose(
        spectrum_data.energy_table,
        [
            [0.0, 1.0, 3.0],
            [0.0, 2.0, 5.0],
        ],
    )
