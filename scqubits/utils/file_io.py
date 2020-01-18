# file_io.py
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
Helper routines for writing data to CSV and h5 files.
"""

import csv
import os

import h5py
import numpy as np

from scqubits.settings import FileType


class FileIOFactory:
    """Factory method for choosing reader/writer according to given format"""
    def get_writer(self, file_format):
        if file_format is FileType.csv:
            return CsvWriter()
        if file_format is FileType.h5:
            return H5Writer()

    def get_reader(self, file_format):
        if file_format is FileType.csv:
            return CsvReader()
        if file_format is FileType.h5:
            return H5Reader()


factory = FileIOFactory()


class ObjectWriter:
    """Sets up the appropriate writer, calls the object's serializer to obtain data, then writes to file."""
    def filewrite(self, the_object, file_format, filename):
        """
        Parameters
        ----------
        the_object: object
        file_format: FileType
        filename: str

        Returns
        -------
        exit_info
        """
        writer = factory.get_writer(file_format)
        the_object._serialize(writer)
        return writer.do_writing(filename)


class ObjectReader:
    """Sets up the appropriate reader, extracts data from file, the sets parameters of existing object or creates
    a new object initialized to read data."""
    def set_params_from_fileread(self, the_object, file_format, filename):
        """
        Parameters
        ----------
        the_object: object
        file_format: FileType
        filename: str
        """
        reader = factory.get_reader(file_format)
        extracted_data = reader.do_reading(filename)
        the_object.set_from_data(*extracted_data)

    def create_from_fileread(self, class_object, file_format, filename):
        """
        Parameters
        ----------
        class_object: class
        file_format: FileType
        filename: str
        """
        reader = factory.get_reader(file_format)
        extracted_data = reader.do_reading(filename)
        return class_object._init_from_data(*extracted_data)


class BaseWriter:
    def __init__(self):
        self._current_object_meta = None
        self._current_object_data = {}

    def create_meta(self, meta_data):
        """
        Parameters
        ----------
        meta_data: dict
        """
        self._current_object_meta = meta_data

    def add_dataset(self, name, data):
        self._current_object_data[name] = data

    def do_writing(self, filename):
        pass


class CsvWriter(BaseWriter):
    def do_writing(self, filename):
        filename_stub = os.path.splitext(filename)[0]

        with open(filename_stub + '_meta.csv', mode='w') as meta_file:
            file_writer = csv.writer(meta_file, delimiter=',')
            file_writer.writerow(self._current_object_meta.keys())
            file_writer.writerow(self._current_object_meta.values())

        for dataname, dataset in self._current_object_data.items():
            np.savetxt(filename_stub + '_' + dataname + '.csv', dataset)


class H5Writer(BaseWriter):
    def do_writing(self, filename):
        """
        Parameters
        ----------
        filename: str
        """
        filename_stub = os.path.splitext(filename)[0]
        h5file = h5py.File(filename_stub + '.hdf5', 'w')
        h5file_root = h5file.create_group('root')

        h5file_root.attrs.update(self._current_object_meta)

        for dataname, dataset in self._current_object_data.items():
            h5file_root.create_dataset(dataname, data=dataset, dtype=dataset.dtype, compression="gzip")


class CsvReader:
    pass


class H5Reader:
    def do_reading(self, filename):
        """
        Parameters
        ----------
        filename: str

        Returns
        -------
        dict, list, list
            dictionary of metadata, list of dataset names, list of ndarrays with data
        """
        metadata = {}
        data_list = []
        dataname_list = []

        filename_stub = os.path.splitext(filename)[0]
        with h5py.File(filename_stub + '.hdf5', 'r') as h5file:
            for dataname, dataset in h5file['root'].items():
                dataname_list.append(dataname)
                data_list.append(dataset[:])
            for key, value in h5file['root'].attrs.items():
                metadata[key] = value
        return metadata, dataname_list, data_list
