# test_fileio_qutip.py
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

"""Regression tests for QutipEigenstates serialization.

These tests guard against a numpy/qutip compatibility issue: numpy >= 1.24
refuses implicit object-dtype creation from ragged (inhomogeneous) lists.
qutip's ``Qobj.dims`` for a multipartite ket is `[[d_1, ..., d_n], [1]]`
in compact form (newer qutip) or `[[d_1, ..., d_n], [1, 1, ..., 1]]` in
expanded form (older qutip). The compact form trips numpy's check while the
expanded form does not. Local development environments may have either
version, so a regression test must construct the inhomogeneous case
directly to be reliable.
"""

import numpy as np
import qutip as qt

from scqubits.io_utils.fileio_qutip import QutipEigenstates


class TestQutipEigenstatesSerialize:
    def _make_ket_with_dims(self, dims: list[list[int]]) -> qt.Qobj:
        """Construct a normalized ket with explicit ``dims``."""
        total = int(np.prod(dims[0]))
        data = np.zeros((total, 1), dtype=complex)
        data[0] = 1.0
        return qt.Qobj(data, dims=dims)

    def _serialize_one_ket(self, dims: list[list[int]]) -> None:
        ket = self._make_ket_with_dims(dims)
        # QutipEigenstates is a numpy ndarray subclass; build a 1-element
        # object-array view of it (mirrors how callers use it).
        arr = np.empty(1, dtype=object)
        arr[0] = ket
        view = arr.view(QutipEigenstates)
        iodata = view.serialize()
        # Sanity: dims round-trips via .tolist() back to the original
        round_tripped = iodata.ndarrays["qobj_dims"].tolist()
        assert round_tripped == dims, (
            f"dims round-trip mismatch: {round_tripped!r} != {dims!r}"
        )

    def test_serialize_inhomogeneous_dims_compact(self):
        """Compact qutip dims `[[3, 4, 4], [1]]` must serialize cleanly.

        This is the form that triggered the Azure CI failure (numpy >= 1.24).
        Direct construction bypasses qutip-version-dependent dim formatting.
        """
        self._serialize_one_ket([[3, 4, 4], [1]])

    def test_serialize_inhomogeneous_dims_compact_alt(self):
        """Another inhomogeneous case `[[3, 3, 4], [1]]` (different shapes)."""
        self._serialize_one_ket([[3, 3, 4], [1]])

    def test_serialize_homogeneous_dims_expanded(self):
        """Expanded qutip dims `[[3, 4, 4], [1, 1, 1]]` (older qutip form).

        This case happened to pass under numpy >= 1.24 because the lists are
        all length 3; including it ensures both qutip-version forms continue
        to work after the fix.
        """
        self._serialize_one_ket([[3, 4, 4], [1, 1, 1]])

    def test_serialize_single_subsystem(self):
        """Single-subsystem ket `[[5], [1]]` (already homogeneous)."""
        self._serialize_one_ket([[5], [1]])
