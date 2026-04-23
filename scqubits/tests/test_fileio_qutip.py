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

Guards against a numpy/qutip/h5py compatibility issue: qutip >= 5 reports
multipartite-ket dims compactly as ``[[d_1, ..., d_n], [1]]`` (ragged list).
This trips up two layers downstream:

1. NumPy >= 1.24 raises ``ValueError`` on implicit ``np.asarray(...)`` of
   a ragged list. Asking for ``dtype=object`` lets numpy build the array,
   but then...
2. h5py rejects ``object`` dtype with
   ``TypeError: Object dtype dtype('O') has no native HDF5 equivalent``,
   so the file write fails.

The fix in :mod:`scqubits.io_utils.fileio_qutip` pads the compact ``[1]``
row to the expanded ``[1, 1, ..., 1]`` form so the result is a uniform
integer array. qutip auto-normalizes back to the compact form on
deserialize, so the round-trip is semantically lossless.
"""

import numpy as np
import qutip as qt

from scqubits.io_utils.fileio_qutip import QutipEigenstates, _expand_qobj_dims


class TestExpandQobjDims:
    """Direct unit tests for the dims-padding helper (no qutip needed)."""

    def test_compact_ket_padded(self):
        assert _expand_qobj_dims([[3, 4, 4], [1]]) == [[3, 4, 4], [1, 1, 1]]

    def test_compact_bra_padded(self):
        assert _expand_qobj_dims([[1], [3, 4, 4]]) == [[1, 1, 1], [3, 4, 4]]

    def test_already_expanded_unchanged(self):
        dims = [[3, 4, 4], [1, 1, 1]]
        assert _expand_qobj_dims(dims) == dims

    def test_operator_unchanged(self):
        dims = [[3, 4, 4], [3, 4, 4]]
        assert _expand_qobj_dims(dims) == dims

    def test_single_subsystem_unchanged(self):
        dims = [[5], [1]]
        assert _expand_qobj_dims(dims) == dims


class TestQutipEigenstatesSerialize:
    def _make_ket_with_dims(self, dims: list[list[int]]) -> qt.Qobj:
        """Construct a normalized ket with explicit ``dims``."""
        total = int(np.prod(dims[0]))
        data = np.zeros((total, 1), dtype=complex)
        data[0] = 1.0
        return qt.Qobj(data, dims=dims)

    def _serialize_one_ket(self, dims: list[list[int]]):
        ket = self._make_ket_with_dims(dims)
        # QutipEigenstates is a numpy ndarray subclass; build a 1-element
        # object-array view of it (mirrors how callers use it).
        arr = np.empty(1, dtype=object)
        arr[0] = ket
        view = arr.view(QutipEigenstates)
        return view.serialize()

    def test_serialize_inhomogeneous_dims_no_crash(self):
        """``np.asarray`` step must succeed for compact dims ``[[3,4,4],[1]]``.

        Pre-fix this raised ``ValueError`` from numpy on ragged lists.
        """
        self._serialize_one_ket([[3, 4, 4], [1]])

    def test_serialize_dims_h5py_compatible(self):
        """Serialized ``qobj_dims`` must be a uniform numeric array.

        h5py rejects object-dtype arrays with
        ``TypeError: Object dtype dtype('O') has no native HDF5 equivalent``.
        This test catches that downstream failure directly without
        actually opening an HDF5 file.
        """
        iodata = self._serialize_one_ket([[3, 4, 4], [1]])
        qobj_dims = iodata.ndarrays["qobj_dims"]
        assert (
            qobj_dims.dtype != object
        ), f"qobj_dims has dtype {qobj_dims.dtype!r}; h5py cannot store this"
        assert np.issubdtype(
            qobj_dims.dtype, np.integer
        ), f"qobj_dims should be integer dtype, got {qobj_dims.dtype!r}"

    def test_serialize_deserialize_roundtrip_compact(self):
        """End-to-end round-trip preserves the Qobj's shape and per-subsystem
        dimensions.

        We compare ``shape`` and the row of ``dims`` carrying the actual
        per-subsystem sizes, because qutip's normalization of the trailing
        ``[1]`` row varies by version (some versions preserve what you pass,
        others rewrite ``[1, 1, 1]`` -> ``[1]`` or vice-versa). What matters
        for users is that the reconstructed Qobj has the right shape and
        subsystem structure.
        """
        original = self._make_ket_with_dims([[3, 4, 4], [1]])
        arr = np.empty(1, dtype=object)
        arr[0] = original
        view = arr.view(QutipEigenstates)
        iodata = view.serialize()
        reconstructed = QutipEigenstates.deserialize(iodata)
        assert reconstructed[0].shape == original.shape
        # Compare the per-subsystem dim row (the [3, 4, 4] side), which is
        # the substantive content; the trailing [1]-vs-[1,1,1] is qutip's
        # housekeeping.
        assert reconstructed[0].dims[0] == original.dims[0]

    def test_serialize_single_subsystem(self):
        """Single-subsystem ket ``[[5], [1]]`` (already homogeneous)."""
        iodata = self._serialize_one_ket([[5], [1]])
        assert iodata.ndarrays["qobj_dims"].dtype != object
