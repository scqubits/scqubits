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


class _EventCounter(central_dispatch.DispatchClient):
    """Test listener that counts dispatched events of a chosen type.

    The instance must be kept referenced for the lifetime of the check, since
    CentralDispatch holds its callback only weakly.
    """

    def __init__(self, event):
        self._watched_event = event
        self.count = 0
        central_dispatch.CENTRAL_DISPATCH.register(event, self)

    def receive(self, event, sender, **kwargs):
        if event == self._watched_event:
            self.count += 1


class TestWatchedPropertyEquality:
    """A WatchedProperty broadcasts only when its value actually changes (issue #276)."""

    def test_values_equal_handles_scalars_arrays_and_none(self):
        from scqubits.core.descriptors import _values_equal

        assert _values_equal(5.0, 5.0)
        assert not _values_equal(5.0, 6.0)
        assert _values_equal(None, None)
        assert not _values_equal(None, 5.0)
        assert _values_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert not _values_equal(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        # shape mismatch must not raise -- treated as not equal
        assert not _values_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_no_broadcast_during_construction(self):
        counter = _EventCounter("QUANTUMSYSTEM_UPDATE")
        scq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=30)
        assert counter.count == 0

    def test_broadcast_on_value_change(self):
        qbt = scq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=30)
        counter = _EventCounter("QUANTUMSYSTEM_UPDATE")
        qbt.EJ = 18.0
        assert counter.count == 1

    def test_no_broadcast_on_same_value(self):
        qbt = scq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=30)
        counter = _EventCounter("QUANTUMSYSTEM_UPDATE")
        qbt.EJ = qbt.EJ  # reassigning the identical value must not broadcast
        assert counter.count == 0

    def test_none_default_first_assignment_is_initialized(self):
        # Oscillator's l_osc defaults to None; the same-value short-circuit must not
        # skip the first assignment, or reading l_osc later raises KeyError.
        osc = scq.Oscillator(E_osc=5.0, truncated_dim=10)
        assert osc.l_osc is None
