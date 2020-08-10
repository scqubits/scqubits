# fileio_serializers.py
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

import scqubits.utils.misc as utils

SERIALIZABLE_REGISTRY = {}


class Serializable(ABC):
    """Mix-in class that makes descendant classes serializable."""
    _subclasses = []

    def __new__(cls, *args, **kwargs):
        """Modified `__new__` to set up `cls._init_params`. The latter is used to record which of the `__init__`
         parameters are to be stored/read in file IO."""
        cls._init_params = get_init_params(cls)
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs):
        """Used to register all non-abstract subclasses as a list in `QuantumSystem.subclasses`."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._subclasses.append(cls)
            SERIALIZABLE_REGISTRY[cls.__name__] = cls

    def get_initdata(self):
        """Returns dict appropriate for creating/initializing a new Serializable object.

        Returns
        -------
        dict
        """
        return {name: getattr(self, name) for name in self._init_params}

    @classmethod
    def deserialize(cls, io_data):
        """
        Take the given IOData and return an instance of the described class, initialized with the data stored in
        io_data.

        Parameters
        ----------
        io_data: scqubits.io_utils.file_io_base.IOData

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
        scqubits.io_utils.file_io_base.IOData
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
        import scqubits.io_utils.fileio as io
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
        import scqubits.io_utils.fileio as io
        return io.read(filename)


def _add_object(name, obj, attributes, ndarrays, objects):
    objects[name] = obj
    return attributes, ndarrays, objects


def _add_ndarray(name, obj, attributes, ndarrays, objects):
    ndarrays[name] = obj
    return attributes, ndarrays, objects


def _add_attribute(name, obj, attributes, ndarrays, objects):
    attributes[name] = obj
    return attributes, ndarrays, objects


TO_ATTRIBUTE = (str, Number, dict, list, tuple, bool, np.bool_)
TO_NDARRAY = (np.ndarray,)
TO_OBJECT = (Serializable,)


def type_dispatch(entity):
    """
    Based on the type of the object ``entity``, return the appropriate function that converts the entity into the
    appropriate category of IOData

    Parameters
    ----------
    entity: instance of serializable class

    Returns
    -------
    function
    """
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
    import scqubits.io_utils.fileio as io
    dict_instance = utils.remove_nones(dict_instance)
    attributes = {}
    ndarrays = {}
    objects = {}
    typename = 'dict'

    for name, content in dict_instance.items():
        update_func = type_dispatch(content)
        attributes, ndarrays, objects = update_func(name, content, attributes, ndarrays, objects)
    return io.IOData(typename, attributes, ndarrays, objects)


def listlike_serialize(listlike_instance):
    """
    Create an IOData instance from list data.

    Parameters
    ----------
    listlike_instance: list or tuple

    Returns
    -------
    IOData
    """
    import scqubits.io_utils.fileio as io
    attributes = {}
    ndarrays = {}
    objects = {}
    typename = type(listlike_instance).__name__
    for index, item in enumerate(listlike_instance):
        update_func = type_dispatch(item)
        attributes, ndarrays, objects = update_func(str(index), item, attributes, ndarrays, objects)
    return io.IOData(typename, attributes, ndarrays, objects)


list_serialize = listlike_serialize


tuple_serialize = listlike_serialize


def dict_deserialize(iodata):
    """Turn IOData instance back into a dict"""
    return dict(**iodata.as_kwargs())


def list_deserialize(iodata):
    """Turn IOData instance back into a list"""
    dict_data = iodata.as_kwargs()
    return [dict_data[key] for key in sorted(dict_data, key=int)]


def tuple_deserialize(iodata):
    """Turn IOData instance back into a tuple"""
    return tuple(list_deserialize(iodata))


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
