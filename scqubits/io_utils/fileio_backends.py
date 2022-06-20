# fileio_backends.py
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
"""
Helper routines for writing data to h5 files.
"""

import ast
import csv
import os
import re

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np

from numpy import ndarray

import scqubits.io_utils.fileio as io
import scqubits.utils.misc as utils

try:
    import h5py

    from h5py import AttributeManager, File, Group
except ImportError:
    _HAS_H5PY = False
else:
    _HAS_H5PY = True

if TYPE_CHECKING:
    from h5py import File, Group


class IOWriter(ABC):
    """
    ABC for writing class instance data to file.

    Parameters
    ----------
    filename: str
    file_handle: h5.Group, optional
    """

    def __init__(self, filename: str, file_handle: "Group" = None) -> None:
        self.filename = filename
        self.io_data: io.IOData
        self.file_handle = file_handle

    @abstractmethod
    def to_file(self, io_data: io.IOData, **kwargs):
        pass

    @abstractmethod
    def write_attributes(self, *args, **kwargs):
        pass

    @abstractmethod
    def write_ndarrays(self, *args, **kwargs):
        pass

    @abstractmethod
    def write_objects(self, *args, **kwargs):
        pass


@utils.Required(h5py=_HAS_H5PY)
class H5Writer(IOWriter):
    """Writes IOData to a custom-format h5 file"""

    def write_attributes(self, h5file_group: Union["Group", "File"]) -> None:  # type: ignore
        """
        Attribute data consists of

         1. `__init__` parameters that are of type str or numerical. These are
         directly written into `h5py.Group.attrs` 2. lists are stored under
         `<h5py.Group>/__lists` 3. dicts are stored under `<h5py.Group>/__dicts`
        """
        h5file_group.attrs.create(
            "__type", self.io_data.typename
        )  # Record the type of the current class instance
        attributes = self.io_data.attributes
        for attr_name, attr_value in attributes.items():
            if isinstance(
                attr_value, dict
            ):  # h5py does not serialize dicts automatically, so have to do it manually
                group_name = "__dicts/" + attr_name
                h5file_group.create_group(group_name)
                io.write(
                    attr_value, self.filename, file_handle=h5file_group[group_name]
                )
            elif isinstance(attr_value, (list, tuple)):
                group_name = "__lists/" + attr_name
                h5file_group.create_group(group_name)
                io.write(
                    attr_value, self.filename, file_handle=h5file_group[group_name]
                )
            else:
                h5file_group.attrs[attr_name] = attr_value

    def write_ndarrays(self, h5file_group: Union["Group", "File"]) -> None:  # type: ignore
        """
        Writes ndarray (float or complex) data contained in `self.iodata` to the
        provided `h5py.Group` as a `h5py.Dataset`, using gzip compression.
        """
        data_group = h5file_group.file.require_group("__data")
        for name, array in self.io_data.ndarrays.items():
            array_id = hash(array.tobytes())
            h5file_group.create_dataset(name, data=[array_id], dtype="int64")
            if str(array_id) not in data_group:
                data_group.create_dataset(
                    str(array_id), data=array, dtype=array.dtype, compression="gzip"
                )

    def write_objects(self, h5file_group: Union["Group", "File"]) -> None:  # type: ignore
        """
        Writes data representing a Python object other than ndarray, list and dict,
        contained in `self.iodata` to the provided `h5py.Group`  und
        `<h5py.Group>/__objects`.
        """
        h5file_group = h5file_group.create_group("__objects")
        for obj_name in self.io_data.objects.keys():
            new_h5group = h5file_group.create_group(obj_name)
            io.write(
                self.io_data.objects[obj_name], self.filename, file_handle=new_h5group
            )

    def to_file(self, io_data: io.IOData, file_handle: "Group" = None) -> None:
        """
        Takes the serialized IOData and writes it either to a new h5 file with file
        name given by `self.filename` to to the given h5py.Group of an open h5 file.
        """
        self.io_data = io_data
        if file_handle is None:
            h5file_group = h5py.File(self.filename, "w", rdcc_nbytes=1024**2 * 200)
            _ = h5file_group.create_group("__data")
            close_when_done = True
        else:
            h5file_group = file_handle
            close_when_done = False

        self.write_attributes(h5file_group)
        self.write_ndarrays(h5file_group)
        self.write_objects(h5file_group)
        if close_when_done:
            h5file_group.close()


@utils.Required(h5py=_HAS_H5PY)
class H5Reader:
    """
    Enables reading h5 files generated with scqubits.
    """

    def __init__(self, filename: str, file_handle: "Group" = None) -> None:
        self.filename = filename
        self.io_data = None
        self.file_handle = file_handle

    @staticmethod
    def h5_attrs_to_dict(
        h5_attrs: "AttributeManager",
    ) -> Dict[str, Union[float, str, int]]:
        """
        Converts h5 attribute data to a Python dictionary.

        Parameters
        ----------
        h5_attrs: h5py.AttributeManager
            as obtained by accessing `<h5py.Group>.attrs`
        """
        return {attr_name: attr_value for attr_name, attr_value in h5_attrs.items()}

    def read_attributes(self, h5file_group: Union["Group", "File"]) -> Dict[str, Any]:
        """
        Read data from h5 file group that is stored directly as `<h5py.Group>.attrs`,
        or saved in subgroups titled `<h5py.Group>/__lists` and `<h5py.Group>/__dicts`.
        """
        attributes = self.h5_attrs_to_dict(h5file_group.attrs)
        if "__dicts" in h5file_group:
            for dict_name in h5file_group["__dicts"]:
                attributes[dict_name] = io.read(
                    self.filename, h5file_group["__dicts/" + dict_name]
                )
        if "__lists" in h5file_group:
            for list_name in h5file_group["__lists"]:
                attributes[list_name] = io.read(
                    self.filename, h5file_group["__lists/" + list_name]
                )
        return attributes

    def read_ndarrays(self, h5file_group: Union["Group", "File"]) -> Dict[str, ndarray]:
        """
        Read numpy array data from h5 file group.
        """
        ndarrays = {}

        if "__data" in h5file_group.file:
            datagroup = h5file_group.file.require_group("__data")
            for name, id_dataset in h5file_group.items():
                if isinstance(id_dataset, h5py.Dataset):
                    id_int = id_dataset[:][0]
                    data = datagroup[str(id_int)][:]
                    ndarrays[name] = data
            return ndarrays

        # legacy support
        ndarrays = {
            name: array[:]
            for name, array in h5file_group.items()
            if isinstance(array, h5py.Dataset)
        }
        return ndarrays

    def read_objects(
        self, h5file_group: Union["Group", "File"]
    ) -> Dict[str, io.IOData]:
        """
        Read data from the given h5 file group that represents a Python object other
        than an ndarray, list, or dict.
        """
        inner_objects = {}
        h5file_group = h5file_group["__objects"]
        for obj_name in h5file_group:
            inner_objects[obj_name] = io.read(self.filename, h5file_group[obj_name])
        return inner_objects

    def from_file(self, filename: str, file_handle: "Group" = None) -> io.IOData:
        """
        Either opens a new h5 file for reading or accesses an already opened file via
        the given h5.Group handle. Reads all data from the three categories of
        attributes (incl. lists and dicts), ndarrays, and objects.
        """
        if file_handle is None:
            h5file_group = h5py.File(filename, "r", rdcc_nbytes=1024**2 * 200)
        else:
            h5file_group = file_handle

        attributes = self.read_attributes(h5file_group)
        typename = attributes["__type"]
        assert isinstance(typename, str)
        del attributes["__type"]
        ndarrays = self.read_ndarrays(h5file_group)
        inner_objects = self.read_objects(h5file_group)
        return io.IOData(typename, attributes, ndarrays, inner_objects)


class CSVWriter(IOWriter):
    """
    Given `filename="somename.csv"`, write initdata into somename.csv Then, additional
    csv files are written for each dataset, with filenames: `"somename_" + dataname0 +
    ".csv"` etc.
    """

    def append_ndarray_info(self, attributes):
        """Add data set information to attributes, so that dataset names and
        dimensions are available in attributes CSV file."""
        for index, dataname in enumerate(self.io_data.ndarrays.keys()):
            data = self.io_data.ndarrays[dataname]
            attributes["dataset" + str(index)] = dataname

            if data.ndim == 3:
                slice_count = len(data)
            else:
                slice_count = 1
            attributes["dataset" + str(index) + ".slices"] = slice_count
        return attributes

    def write_attributes(self, filename: str):  # type: ignore
        attributes = self.io_data.attributes
        attributes["__type"] = self.io_data.typename
        attributes = self.append_ndarray_info(attributes)
        with open(filename, mode="w", newline="") as meta_file:
            file_writer = csv.writer(meta_file, delimiter=",")
            file_writer.writerow(attributes.keys())
            file_writer.writerow(attributes.values())

    def write_ndarrays(self, filename: str):  # type: ignore
        filename_stub, _ = os.path.splitext(filename)
        for dataname, dataset in self.io_data.ndarrays.items():
            filename = filename_stub + "_" + dataname + ".csv"
            self.write_data(filename, dataset)

    def write_data(self, filename: str, dataset: ndarray):  # type: ignore
        if dataset.ndim <= 2:
            np.savetxt(filename, dataset)
        elif dataset.ndim == 3:
            np_savetxt_3d(dataset, filename)
        else:
            raise Exception("Dataset has dimensions > 3. Cannot write to CSV file.")

    def write_objects(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError

    def to_file(self, io_data: io.IOData, **kwargs):
        self.io_data = io_data
        self.write_attributes(self.filename)
        self.write_ndarrays(self.filename)
        # no support for write_objects in CSV format


class CSVReader:
    @staticmethod
    def read_attributes(filename):
        with open(filename, mode="r") as meta_file:
            file_reader = csv.reader(meta_file, delimiter=",")
            meta_keys = file_reader.__next__()
            meta_values = file_reader.__next__()
        return dict(zip(meta_keys, meta_values))

    def process_metadict(self, meta_dict: Dict) -> Tuple[Dict, List[str], ndarray]:
        attributes = {
            attr_name: utils.to_expression_or_string(attr_value)
            for attr_name, attr_value in meta_dict.items()
            if not re.match(r"dataset\d+", attr_name)
        }
        data_names = [
            dataname
            for datalabel, dataname in meta_dict.items()
            if re.match(r"dataset\d+$", datalabel)
        ]
        data_slices = np.asarray(
            [
                ast.literal_eval(value)
                for key, value in meta_dict.items()
                if re.match(r"dataset\d+.slices", key)
            ]
        )
        return attributes, data_names, data_slices

    @staticmethod
    def read_data(filename, slices):
        try:
            data_array = np.loadtxt(filename)
        except ValueError:
            data_array = np.loadtxt(filename, dtype=np.complex_)
        if slices > 1:
            nrows, ncols = data_array.shape
            return data_array.reshape((slices, nrows // slices, ncols))
        return data_array

    def from_file(self, filename: str, **kwargs) -> io.IOData:
        """
        Returns
        -------
            class instance generated from file data
        """
        ext_attributes = self.read_attributes(filename)
        typename = ext_attributes["__type"]
        del ext_attributes["__type"]

        attributes, data_names, data_slices = self.process_metadict(ext_attributes)

        filename_stub, _ = os.path.splitext(filename)
        ndarrays = {}
        for index, dataname in enumerate(data_names):
            data_filename = filename_stub + "_" + dataname + ".csv"
            slices = data_slices[index]
            ndarrays[dataname] = self.read_data(data_filename, slices)

        return io.IOData(typename, attributes, ndarrays, objects=None)


def np_savetxt_3d(array3d: ndarray, filename: str):
    """
    Helper function that splits a 3d numpy array into 2d slices for writing as csv
    data to a new file. Slices are separated by a comment row `# New slice`.

    Parameters
    ----------
    array3d:
        ndarray with ndim = 3
    filename:
        name of output file
    """
    with open(filename, mode="w", newline="") as datafile:
        datafile.write("# Array shape: {0}\n".format(array3d.shape))
        for data_slice in array3d:
            np.savetxt(datafile, data_slice)
            datafile.write("# New slice\n")
