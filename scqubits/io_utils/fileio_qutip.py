# fileio_qutip.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
import numpy as np
import qutip as qt

from scqubits.io_utils.fileio_serializers import Serializable
from scqubits.utils import misc as utils


class QutipEigenstates(np.ndarray, Serializable):
    """Wrapper class that adds serialization functionality to the numpy ndarray class."""
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#extra-gotchas-custom-del-methods-and-ndarray-base
    @classmethod
    def deserialize(cls, io_data):
        """
        Take the given IOData and return an instance of the described class, initialized with the data stored in
        io_data.

        Parameters
        ----------
        io_data: IOData

        Returns
        -------
        Serializable
        """
        qobj_dims = io_data.ndarrays['qobj_dims']
        qobj_shape = io_data.ndarrays['qobj_shape']
        evec_array = io_data.ndarrays['evecs']
        qt_eigenstates = np.asarray([qt.Qobj(inpt=evec, dims=qobj_dims, shape=qobj_shape, type='ket')
                                     for evec in evec_array], dtype=np.dtype('O'))
        return qt_eigenstates

    def serialize(self):
        """
        Convert the content of the current class instance into IOData format.

        Returns
        -------
        IOData
        """
        import scqubits.io_utils.fileio as io
        typename = type(self).__name__
        evec_count = len(self)
        qobj_dims = np.asarray(self[0].dims)
        qobj_shape = np.asarray(self[0].shape)
        io_attributes = {'evec_count': evec_count}
        io_ndarrays = {'evecs': np.asarray([utils.qt_ket_to_ndarray(qobj_ket) for qobj_ket in self]),
                       'qobj_dims': qobj_dims,
                       'qobj_shape': qobj_shape}
        return io.IOData(typename, io_attributes, io_ndarrays, objects=None)

    def filewrite(self, filename):
        """Convenience method bound to the class. Simply accesses the `write` function.

        Parameters
        ----------
        filename: str
        """
        import scqubits.io_utils.fileio as io
        io.write(self, filename)
