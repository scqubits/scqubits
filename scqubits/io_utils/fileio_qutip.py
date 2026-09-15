# fileio_qutip.py
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
import qutip as qt

from scqubits.io_utils.fileio import IOData
from scqubits.io_utils.fileio_serializers import Serializable
from scqubits.utils import misc as utils


def _expand_qobj_dims(dims: list) -> list:
    """Pad qutip's compact ``[1]`` row to match the partner row's length.

    qutip >= 5 reports a multipartite ket's ``Qobj.dims`` compactly as
    ``[[d_1, ..., d_n], [1]]`` rather than the expanded
    ``[[d_1, ..., d_n], [1, 1, ..., 1]]`` form. The compact form is a
    ragged list which:

    1. NumPy >= 1.24 rejects from ``np.asarray(...)`` without ``dtype=object``;
    2. ``dtype=object`` arrays cannot be stored by h5py.

    Padding to the expanded form yields a uniform integer array that h5py
    accepts. qutip auto-normalizes back to the compact form when the dims
    are passed to ``Qobj(...)`` on deserialize, so the round-trip is
    semantically lossless.

    Returns ``dims`` unchanged when no padding is needed (operators, or
    already-expanded kets/bras).
    """
    if len(dims) != 2:
        return dims
    n0, n1 = len(dims[0]), len(dims[1])
    if n0 == n1:
        return dims
    if dims[0] == [1]:
        return [[1] * n1, dims[1]]
    if dims[1] == [1]:
        return [dims[0], [1] * n0]
    return dims


class QutipEigenstates(np.ndarray, Serializable):
    """Wrapper class that adds serialization functionality to the numpy ndarray
    class."""

    @classmethod
    def deserialize(cls, io_data: IOData) -> np.ndarray:  # type: ignore
        """Take the given IOData and return an instance of the described class,
        initialized with the data stored in io_data."""
        # Qobj in Qutip>=5 wants this to be a nested list
        qobj_dims = io_data.ndarrays["qobj_dims"].tolist()
        qobj_shape = io_data.ndarrays["qobj_shape"]
        evec_array = io_data.ndarrays["evecs"]

        qt_eigenstates = np.asarray(
            [qt.Qobj(evec, dims=qobj_dims) for evec in evec_array],
            dtype=np.dtype("O"),
        )
        return qt_eigenstates

    def serialize(self) -> IOData:
        """Convert the content of the current class instance into IOData format."""
        import scqubits.io_utils.fileio as io

        typename = type(self).__name__
        evec_count = len(self)
        # Pad qutip's compact dims (e.g. [[3, 4, 4], [1]]) to the expanded
        # form so the result is a uniform integer array compatible with h5py.
        # See :func:`_expand_qobj_dims` for rationale.
        qobj_dims = np.asarray(_expand_qobj_dims(self[0].dims))
        qobj_shape = np.asarray(self[0].shape)
        io_attributes = {"evec_count": evec_count}
        io_ndarrays = {
            "evecs": np.asarray(
                [utils.qt_ket_to_ndarray(qobj_ket) for qobj_ket in self]
            ),
            "qobj_dims": qobj_dims,
            "qobj_shape": qobj_shape,
        }
        return io.IOData(typename, io_attributes, io_ndarrays, objects=None)

    def filewrite(self, filename: str):
        """Convenience method bound to the class.

        Simply accesses the
        `write` function.
        """
        import scqubits.io_utils.fileio as io

        io.write(self, filename)
