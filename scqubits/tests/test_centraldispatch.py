# test_centraldispatch.py
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

import logging

import numpy as np

import scqubits as scq
import scqubits.core.central_dispatch as central_dispatch

LOGGER = logging.getLogger(__name__)


class TestCentralDispatch:
    def test_register(self, caplog):
        central_dispatch.LOGGER.setLevel(logging.DEBUG)
        qbt = scq.Transmon.create()
        hs = scq.HilbertSpace([qbt])
        assert "Registering HilbertSpace for QUANTUMSYSTEM_UPDATE" in caplog.text
        central_dispatch.LOGGER.setLevel(logging.WARNING)

    def test_unregister(self, caplog):
        qbt = scq.Transmon.create()
        hs = scq.HilbertSpace([qbt])
        central_dispatch.LOGGER.setLevel(logging.DEBUG)
        del hs
        assert "Unregistering HilbertSpace" in caplog.text
        central_dispatch.LOGGER.setLevel(logging.WARNING)

    def test_qubit_broadcast(self, caplog):
        qbt = scq.Transmon.create()
        central_dispatch.LOGGER.setLevel(logging.DEBUG)
        qbt.EC = 0.67
        assert "Transmon broadcasting QUANTUMSYSTEM_UPDATE" in caplog.text
        central_dispatch.LOGGER.setLevel(logging.WARNING)

    def test_fullchain(self, caplog):
        qbt1 = scq.Transmon.create()
        qbt2 = scq.TunableTransmon.create()
        hs = scq.HilbertSpace([qbt1, qbt2])
        pvals_by_name = {"ng": np.linspace(0.0, 1.0, 3)}

        def update_hilbertspace(ng):
            qbt1.ng = ng

        sweep = scq.ParameterSweep(
            hilbertspace=hs,
            paramvals_by_name=pvals_by_name,
            update_hilbertspace=update_hilbertspace,
            evals_count=5,
        )
        central_dispatch.LOGGER.setLevel(logging.DEBUG)
        qbt1.EC = 0.67
        assert "Transmon broadcasting QUANTUMSYSTEM_UPDATE" in caplog.text
        assert (
            "Central dispatch calling HilbertSpace about QUANTUMSYSTEM_UPDATE"
            in caplog.text
        )
        assert "Client HilbertSpace broadcasting HILBERTSPACE_UPDATE" in caplog.text
        assert (
            "Central dispatch calling ParameterSweep about HILBERTSPACE_UPDATE"
            in caplog.text
        )
        central_dispatch.LOGGER.setLevel(logging.WARNING)
