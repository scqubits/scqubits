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
import numpy as np
import re

import scqubits.core.constants as const

try:
    import h5py
except ImportError:
    const._HAS_H5PY = False
else:
    const._HAS_H5PY = True


from scqubits.utils.misc import Required


class FileIOFactory:
    """Factory method for choosing reader/writer according to given format"""
    def get_writer(self, file_name):
        _, suffix = os.path.splitext(file_name)
        if suffix == '.csv':
            return CsvWriter()
        if suffix in ('.h5', '.hdf5'):
            return H5Writer()
        raise Exception("Extension '{}' of given file name '{}' does not match any supported "
                        "file type: {}".format(suffix, file_name, const.FILE_TYPES))

    def get_reader(self, file_name):
        _, suffix = os.path.splitext(file_name)
        if suffix == '.csv':
            return CsvReader()
        if suffix in ('.h5', '.hdf5'):
            return H5Reader()
        raise Exception("Extension '{}' of given file name '{}' does not match any supported "
                        "file type: {}".format(suffix, file_name, const.FILE_TYPES))


factory = FileIOFactory()


class ObjectWriter:
    """Sets up the appropriate writer, calls the object's serializer to obtain data, then writes to file."""
    def filewrite(self, the_object, filename):
        """
        Parameters
        ----------
        the_object: object
        filename: str

        Returns
        -------
        exit_info
        """
        writer = factory.get_writer(filename)
        the_object._serialize(writer)
        return writer.do_writing(filename)


class ObjectReader:
    """Sets up the appropriate reader, extracts data from file, the sets parameters of existing object or creates
    a new object initialized to read data."""
    def set_params_from_file(self, the_object, filename):
        """
        Parameters
        ----------
        the_object: object
        filename: str
        """
        reader = factory.get_reader(filename)
        extracted_data = reader.do_reading(filename)
        the_object.set_from_data(*extracted_data)

    def create_from_file(self, class_object, filename):
        """
        Parameters
        ----------
        class_object: class
        filename: str
        """
        reader = factory.get_reader(filename)
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
        raise NotImplementedError


class CsvWriter(BaseWriter):
    """
    Given filename='somename.csv', write the following metadata into somename.csv
    1. all dict information provided via ._current_object_meta
    2. 'dataset0' -> name of first dataset, 'dataset1', -> name of second dataset,...
    As a result, the first and second row are the only row entries in this csv file and have the form

    paramname1, paramname2, ..., paramname_n, 'dataset0', 'dataset1', ...
    paramval1,  paramval2,  ..., paramval_n,  dataname0,  dataname1, ...

    Then, additional csv files are written for each dataset, with filenames: 'somename_' + dataname0 + '.csv' etc.
    """
    def do_writing(self, filename):
        filename_stub, _ = os.path.splitext(filename)
        metadata = self._current_object_meta
        datasets = self._current_object_data

        for index, dataname in enumerate(datasets.keys()):
            metadata['dataset' + str(index)] = dataname

        with open(filename_stub + '.csv', mode='w', newline='') as meta_file:
            file_writer = csv.writer(meta_file, delimiter=',')
            file_writer.writerow(metadata.keys())
            file_writer.writerow(metadata.values())

        for dataname, dataset in datasets.items():
            np.savetxt(filename_stub + '_' + dataname + '.csv', dataset)


class H5Writer(BaseWriter):
    @Required(h5py=const._HAS_H5PY)
    def do_writing(self, filename):
        """
        Parameters
        ----------
        filename: str
        """
        filename_stub, _ = os.path.splitext(filename)
        h5file = h5py.File(filename_stub + '.hdf5', 'w')
        h5file_root = h5file.create_group('root')

        h5file_root.attrs.update(self._current_object_meta)

        for dataname, dataset in self._current_object_data.items():
            h5file_root.create_dataset(dataname, data=dataset, dtype=dataset.dtype, compression="gzip")


class CsvReader:
    def do_reading(self, filename):
        """See `CsvWriter` for a description of how metadata and multiple datasets are split up among csv files"""
        filename_stub, _ = os.path.splitext(filename)
        with open(filename_stub + '.csv', mode='r') as meta_file:
            file_reader = csv.reader(meta_file, delimiter=',')
            meta_keys = file_reader.__next__()
            meta_values = file_reader.__next__()
        meta_dict = dict(zip(meta_keys, meta_values))

        metadata = {key: value for key, value in meta_dict if not re.match('dataset\d+', key)}
        dataname_list = [value for key, value in meta_dict if re.match('dataset\d+', key)]
        data_list = []

        for dataname in dataname_list:
            data_filename = filename_stub + '-' + dataname + '.csv'
            try:
                data_array = np.loadtxt(data_filename)
            except ValueError:
                data_array = np.loadtxt(data_filename, dtype=np.complex_)
            else:
                raise ValueError("Unable to read '{}' -- does not appear to be "
                                 "a float or complex ndarray.".format(data_filename))

            data_list.append(data_array)
        return metadata, dataname_list, data_list


class H5Reader:
    @Required(h5py=const._HAS_H5PY)
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
