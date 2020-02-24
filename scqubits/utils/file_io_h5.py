# file_io_h5.py
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
Helper routines for writing data to h5 files.
"""

try:
    import h5py
except ImportError:
    _HAS_H5PY = False
else:
    _HAS_H5PY = True

import scqubits.utils.file_io as io
from scqubits.utils.file_io_base import IOWriter
from scqubits.utils.misc import Required


class H5Writer(IOWriter):
    def write_attributes(self, h5file_group):
        """
        Attribute data consists of those __init__ parameters that are of type str or numerical, and are directly written
        into h5py.attributes

        Parameters
        ----------
        h5file_group: h5py.Group
        """
        h5file_group.attrs.create("__type", self.io_data.typename)    # Record the type of the current class instance
        attributes = self.io_data.attributes
        for attr_name, attr_value in attributes.items():
            if isinstance(attr_value, dict):  # h5py does not serialize dicts automatically, so have to do it manually
                group_name = "__dicts/" + attr_name
                h5file_group.create_group(group_name)
                io.write(attr_value, self.filename, file_handle=h5file_group[group_name])
            elif isinstance(attr_value, (list, tuple)):
                group_name = "__lists/" + attr_name
                h5file_group.create_group(group_name)
                io.write(attr_value, self.filename, file_handle=h5file_group[group_name])
            else:
                h5file_group.attrs[attr_name] = attr_value

    def write_ndarrays(self, h5file_group):
        for name, array in self.io_data.ndarrays.items():
            h5file_group.create_dataset(name, data=array, dtype=array.dtype, compression="gzip")

    def write_objects(self, h5file_group):
        h5file_group = h5file_group.create_group("__objects")
        for obj_name in self.io_data.objects.keys():
            new_h5group = h5file_group.create_group(obj_name)
            io.write(self.io_data.objects[obj_name], self.filename, file_handle=new_h5group)

    @Required(h5py=_HAS_H5PY)
    def to_file(self, io_data, file_handle=None):
        """
        Takes the serialized IOData and writes it to the given h5py.Group

        Parameters
        ----------
        io_data: IOData
        file_handle: h5py.Group, optional
        """
        self.io_data = io_data
        if file_handle is None:
            h5file_group = h5py.File(self.filename, 'w')
        else:
            h5file_group = file_handle

        self.write_attributes(h5file_group)
        self.write_ndarrays(h5file_group)
        self.write_objects(h5file_group)


class H5Reader:
    def __init__(self, filename, file_handle=None):
        self.filename = filename
        self.io_data = None
        self.file_handle = file_handle

    @staticmethod
    def h5_attrs_to_dict(h5_attrs):
        return {attr_name: attr_value for attr_name, attr_value in h5_attrs.items()}

    def read_attributes(self, h5file_group):
        attributes = self.h5_attrs_to_dict(h5file_group.attrs)
        if '__dicts' in h5file_group:
            for dict_name in h5file_group['__dicts']:
                attributes[dict_name] = io.read(self.filename, h5file_group['__dicts/' + dict_name])
        if '__lists' in h5file_group:
            for list_name in h5file_group['__lists']:
                attributes[list_name] = io.read(self.filename, h5file_group['__lists/' + list_name])
        return attributes

    def read_ndarrays(self, h5file_group):
        ndarrays = {name: array[:] for name, array in h5file_group.items() if isinstance(array, h5py.Dataset)}
        return ndarrays

    def read_objects(self, h5file_group):
        inner_objects = {}
        h5file_group = h5file_group["__objects"]
        for obj_name in h5file_group:
            inner_objects[obj_name] = io.read(self.filename, h5file_group[obj_name])
        return inner_objects

    @Required(h5py=_HAS_H5PY)
    def from_file(self, filename, file_handle=None):
        """
        Parameters
        ----------
        filename: str
        file_handle: h5.Group, optional

        Returns
        -------
        class instance generated from file data
        """
        if file_handle is None:
            h5file_group = h5py.File(filename, 'r')
        else:
            h5file_group = file_handle

        attributes = self.read_attributes(h5file_group)
        typename = attributes['__type']
        del attributes['__type']
        ndarrays = self.read_ndarrays(h5file_group)
        inner_objects = self.read_objects(h5file_group)
        return io.IOData(typename, attributes, ndarrays, inner_objects)
