# io_utils.py
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
Helper routines for writing data to files.
"""

import os

import scqubits.core.constants as const
import scqubits.io_utils.fileio_serializers as io_serializers


def serialize(the_object):
    """
    Turn the given Python object into an IOData object, needed for writing data to file.

    Parameters
    ----------
    the_object: Serializable

    Returns
    -------
    IOData
    """
    if hasattr(the_object, 'serialize'):
        return the_object.serialize()

    typename = type(the_object).__name__
    if hasattr(io_serializers, typename + '_serialize'):
        serializer_method = getattr(io_serializers, typename + '_serialize')
        return serializer_method(the_object)

    raise NotImplementedError("No implementation for writing {} to file".format(typename))


def deserialize(iodata):
    """
    Turn IOData back into a Python object of the appropriate kind.
    An object is deemed deserializable if
    1) it is recorded in SERIALIZABLE_REGISTRY and has a `.deserialize` method
    2) there exists a function `file_io_serializers.<typename>_deserialize`

    Parameters
    ----------
    iodata: IOData

    Returns
    -------
    class instance
    """
    typename = iodata.typename
    if typename in io_serializers.SERIALIZABLE_REGISTRY:
        cls = io_serializers.SERIALIZABLE_REGISTRY[typename]
        return cls.deserialize(iodata)

    if hasattr(io_serializers, typename + '_deserialize'):
        deserialize_method = getattr(io_serializers, typename + '_deserialize')
        return deserialize_method(iodata)

    raise NotImplementedError("No implementation for converting {} data to Python object.".format(typename))


def write(the_object, filename, file_handle=None):
    """
    Write `the_object` to a file with name `filename`. The optional `file_handle` parameter is used as a group name
    in case of h5 files.

    Parameters
    ----------
    the_object: io_serializers_serializers.Serializable
        object to be written
    filename: str
        Name of file to be written.
    file_handle: h5py.Group, optional
        Name of h5 group to be used for writing (only applies to h5 output format)
    """
    iodata = serialize(the_object)
    writer = IO.get_writer(filename, file_handle=file_handle)
    writer.to_file(iodata, file_handle=file_handle)


def read(filename, file_handle=None):
    """
    Read a Serializable object from file.

    Parameters
    ----------
    filename: str
        Name of file to be read.
    file_handle: h5py.Group, optional
        Specify Group inside h5 file if only this subgroup should be read.

    Returns
    -------
    Serializable
        class instance initialized with the data from the file
    """
    reader = IO.get_reader(filename, file_handle=file_handle)
    iodata = reader.from_file(filename, file_handle=file_handle)
    return deserialize(iodata)


class FileIOFactory:
    """Factory method for choosing reader/writer according to given format"""
    def get_writer(self, file_name, file_handle=None):
        """
        Based on the extension of the provided file name, return the appropriate writer engine.

        Parameters
        ----------
        file_name: str
        file_handle: h5py.Group, optional

        Returns
        -------
        IOWriter
        """
        import scqubits.io_utils.fileio_backends as io_backends
        _, suffix = os.path.splitext(file_name)
        if suffix == '.csv':
            return io_backends.CSVWriter(file_name)
        if suffix in ('.h5', '.hdf5'):
            return io_backends.H5Writer(file_name, file_handle=file_handle)
        raise Exception("Extension '{}' of given file name '{}' does not match any supported "
                        "file type: {}".format(suffix, file_name, const.FILE_TYPES))

    def get_reader(self, file_name, file_handle=None, get_external_reader=False):
        """
        Based on the extension of the provided file name, return the appropriate reader engine.

        Parameters
        ----------
        file_name: str
        file_handle: h5py.Group, optional
        get_external_reader: book, optional

        Returns
        -------
        H5Reader or CSVReader
        """
        if get_external_reader:
            return get_external_reader(file_name, file_handle=file_handle)

        import scqubits.io_utils.fileio_backends as io_backends
        _, suffix = os.path.splitext(file_name)
        if suffix == '.csv':
            return io_backends.CSVReader()
        if suffix in ('.h5', '.hdf5'):
            return io_backends.H5Reader(file_name, file_handle=file_handle)
        raise Exception("Extension '{}' of given file name '{}' does not match any supported "
                        "file type: {}".format(suffix, file_name, const.FILE_TYPES))


IO = FileIOFactory()


class IOData:
    """
    typename: str
    attributes: dict of {str: number or str}
    ndarrays: dict of {str: ndarray}
    objects: dict of {str: Serializable}, optional
    """
    def __init__(self, typename, attributes, ndarrays, objects=None):
        self.typename = typename
        self.attributes = attributes or {}
        self.ndarrays = ndarrays or {}
        self.objects = objects or {}

    def as_kwargs(self):
        """Return a joint dictionary of attributes, ndarrays, and objects, as used in __init__ calls"""
        return {**self.attributes, **self.ndarrays, **self.objects}
