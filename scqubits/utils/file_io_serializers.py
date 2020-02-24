# file_io_serializers.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
"""
Helper classes for writing data to files.
"""

import inspect
from abc import ABC
from numbers import Number

import numpy as np
import qutip as qt

from scqubits.utils.misc import qt_ket_to_ndarray, remove_nones


class Serializable(ABC):
    """Mix-in class that makes descendant classes serializable."""
    def __new__(cls, *args, **kwargs):
        """Used to set up class attributes that record which __init__ parameters will be stored as attributes, ndarrays,
        and objects."""
        cls._init_params = get_init_params(cls)
        return super().__new__(cls)

    def get_initdata(self):  # used for Serializable serialization
        return {name: getattr(self, name) for name in self._init_params}

    @classmethod
    def deserialize(cls, io_data):
        """
        Take the given IOData and return an instance of the described class, initialized with the data stored in
        io_data.

        Parameters
        ----------
        io_data: scqubits.utils.file_io_base.IOData

        Returns
        -------
        Serializable
        """
        return cls(**io_data.as_kwargs())

    def serialize(self):
        """
        Convert the content of the current class instance into IOData format.

        Returns
        -------
        scqubits.utils.file_io_base.IOData
        """
        initdata = {name: getattr(self, name) for name in self._init_params}
        iodata = dict_serialize(initdata)
        iodata.typename = type(self).__name__
        return iodata

    def filewrite(self, filename):
        """Convenience method bound to the class. Simply accesses the `write` function.

        Parameters
        ----------
        filename: str
        """
        import scqubits.utils.file_io as io
        io.write(self, filename)

    @classmethod
    def create_from_file(cls, filename):
        """Read initdata and spectral data from file, and use those to create a new SpectrumData object.

        Parameters
        ----------
        filename: str

        Returns
        -------
        SpectrumData
            new SpectrumData object, initialized with data read from file
        """
        import scqubits.utils.file_io as io
        return io.read(filename)


class QutipEigenstates(np.ndarray, Serializable):
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
        import scqubits.utils.file_io as io
        typename = type(self).__name__
        evec_count = len(self)
        qobj_dims = np.asarray(self[0].dims)
        qobj_shape = np.asarray(self[0].shape)
        io_attributes = {'evec_count': evec_count}
        io_ndarrays = {'evecs': np.asarray([qt_ket_to_ndarray(qobj_ket) for qobj_ket in self]),
                       'qobj_dims': qobj_dims,
                       'qobj_shape': qobj_shape}
        return io.IOData(typename, io_attributes, io_ndarrays, objects=None)

    def filewrite(self, filename):
        """Convenience method bound to the class. Simply accesses the `write` function.

        Parameters
        ----------
        filename: str
        """
        import scqubits.utils.file_io as io
        io.write(self, filename)


def _add_object(name, obj, attributes, ndarrays, objects):
    objects[name] = obj
    return attributes, ndarrays, objects


def _add_ndarray(name, obj, attributes, ndarrays, objects):
    ndarrays[name] = obj
    return attributes, ndarrays, objects


def _add_attribute(name, obj, attributes, ndarrays, objects):
    attributes[name] = obj
    return attributes, ndarrays, objects


TO_ATTRIBUTE = (str, Number, dict, list, tuple)
TO_NDARRAY = (np.ndarray,)
TO_OBJECT = (Serializable, QutipEigenstates)


def type_dispatch(entity):
    if isinstance(entity, TO_ATTRIBUTE):
        return _add_attribute
    if isinstance(entity, TO_OBJECT):
        return _add_object
    if isinstance(entity, TO_NDARRAY):
        return _add_ndarray
    # no match, try treating as object, though this may fail
    return _add_object


def dict_serialize(dict_instance):
    """
    Create an IOData instance from dictionary data.

    Parameters
    ----------
    dict_instance: dict

    Returns
    -------
    IOData
    """
    import scqubits.utils.file_io as io
    dict_instance = remove_nones(dict_instance)
    attributes = {}
    ndarrays = {}
    objects = {}
    typename = 'dict'

    for name, content in dict_instance.items():
        update_func = type_dispatch(content)
        attributes, ndarrays, objects = update_func(name, content, attributes, ndarrays, objects)
    return io.IOData(typename, attributes, ndarrays, objects)


def list_serialize(list_instance):
    """
    Create an IOData instance from lisy data.

    Parameters
    ----------
    list_instance: list or tuple

    Returns
    -------
    IOData
    """
    import scqubits.utils.file_io as io
    attributes = {}
    ndarrays = {}
    objects = {}
    typename = 'list'
    for index, item in enumerate(list_instance):
        update_func = type_dispatch(item)
        attributes, ndarrays, objects = update_func(str(index), item, attributes, ndarrays, objects)
    return io.IOData(typename, attributes, ndarrays, objects)


tuple_serialize = list_serialize


def dict_deserialize(iodata):
    return dict(**iodata.as_kwargs())


def list_deserialize(iodata):
    return list(iodata.as_kwargs().values())


def tuple_deserialize(iodata):
    return tuple(iodata.as_kwargs().values())


def get_init_params(obj):
    """
    Returns a list of the parameters entering the `__init__` method of the given object `obj`.

    Parameters
    ----------
    obj: Serializable

    Returns
    -------
    list of str
    """
    init_params = list(inspect.signature(obj.__init__).parameters.keys())
    if 'self' in init_params:
        init_params.remove('self')
    if 'kwargs' in init_params:
        init_params.remove('kwargs')
    return init_params


