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

import ast
import csv
import os
import re

import numpy as np

import scqubits.core.constants as const

try:
    import h5py
except ImportError:
    const._HAS_H5PY = False
else:
    const._HAS_H5PY = True


from scqubits.utils.misc import Required, to_expression_or_string, remove_nones


class FileIOFactory:
    """Factory method for choosing reader/writer according to given format"""
    def get_writer(self, file_name):
        _, suffix = os.path.splitext(file_name)
        if suffix == '.csv':
            return CSVFileOutput()
        if suffix in ('.h5', '.hdf5'):
            return H5FileOutput()
        raise Exception("Extension '{}' of given file name '{}' does not match any supported "
                        "file type: {}".format(suffix, file_name, const.FILE_TYPES))

    def get_reader(self, file_name):
        _, suffix = os.path.splitext(file_name)
        if suffix == '.csv':
            return CSVFileInput()
        if suffix in ('.h5', '.hdf5'):
            return H5FileInput()
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


class FileOutput:
    def __init__(self):
        self.metadata = None
        self.datasets = {}

    def write_metadata(self, meta_data):
        """
        Parameters
        ----------
        meta_data: dict
        """
        self.metadata = meta_data

    def write_dataset(self, name, data):
        self.datasets[name] = data

    def do_writing(self, filename):
        raise NotImplementedError


class CSVFileOutput(FileOutput):
    """
    Given filename='somename.csv', write metadata into somename.csv
    Then, additional csv files are written for each dataset, with filenames: 'somename_' + dataname0 + '.csv' etc.
    """
    def append_dataset_info(self, metadata):
        """Add data set information to metadata, so that data set names and dimensions are available
        in metadata CSV file."""
        for index, dataname in enumerate(self.datasets.keys()):
            data = self.datasets[dataname]
            metadata['dataset' + str(index)] = dataname

            if data.ndim == 3:
                slice_count = len(data)
            else:
                slice_count = 1
            metadata['dataset' + str(index) + '.slices'] = slice_count
        return metadata

    def write_metadata_file(self, filename):
        metadata = self.metadata
        metadata = self.append_dataset_info(metadata)
        with open(filename, mode='w', newline='') as meta_file:
            file_writer = csv.writer(meta_file, delimiter=',')
            file_writer.writerow(metadata.keys())
            file_writer.writerow(metadata.values())

    def write_data_file(self, filename, dataset):
        if dataset.ndim <= 2:
            np.savetxt(filename, dataset)
        elif dataset.ndim == 3:
            np_savetxt_3d(dataset, filename)
        else:
            raise Exception("Dataset has dimensions > 3. Cannot write to CSV file.")

    def do_writing(self, filename):
        self.write_metadata_file(filename)

        filename_stub, _ = os.path.splitext(filename)
        for dataname, dataset in self.datasets.items():
            filename = filename_stub + '_' + dataname + '.csv'
            self.write_data_file(filename, dataset)


class H5FileOutput(FileOutput):
    @Required(h5py=const._HAS_H5PY)
    def do_writing(self, filename):
        """
        Parameters
        ----------
        filename: str
        """
        h5file = h5py.File(filename, 'w')
        h5file_root = h5file.create_group('root')
        h5file_root.attrs.update(remove_nones(self.metadata))

        for dataname, dataset in self.datasets.items():
            h5file_root.create_dataset(dataname, data=dataset, dtype=dataset.dtype, compression="gzip")


class CSVFileInput:
    @staticmethod
    def read_metadata(filename):
        with open(filename, mode='r') as meta_file:
            file_reader = csv.reader(meta_file, delimiter=',')
            meta_keys = file_reader.__next__()
            meta_values = file_reader.__next__()
        return dict(zip(meta_keys, meta_values))

    def process_metadict(self, meta_dict):
        metadata = {attr_name: to_expression_or_string(attr_value) for attr_name, attr_value in meta_dict.items()
                    if not re.match(r'dataset\d+', attr_name)}
        dataname_list = [dataname for datalabel, dataname in meta_dict.items() if re.match(r'dataset\d+$', datalabel)]
        data_slices_list = [ast.literal_eval(value) for key, value in meta_dict.items()
                            if re.match(r'dataset\d+.slices', key)]
        return metadata, dataname_list, data_slices_list

    @staticmethod
    def read_data(filename, slices):
        try:
            data_array = np.loadtxt(filename)
        except ValueError:
            data_array = np.loadtxt(filename, dtype=np.complex_)
        if slices > 1:
            nrows, ncols = data_array.shape
            return data_array.reshape((slices, nrows//slices, ncols))
        return data_array

    def do_reading(self, filename):
        """See `CSVFileOutput` for a description of how metadata and multiple datasets are split up among csv files"""
        meta_dict = self.read_metadata(filename)
        metadata, dataname_list, data_slices_list = self.process_metadict(meta_dict)

        filename_stub, _ = os.path.splitext(filename)
        data_list = []
        for index, dataname in enumerate(dataname_list):
            data_filename = filename_stub + '_' + dataname + '.csv'
            slices = data_slices_list[index]
            data_list.append(self.read_data(data_filename, slices))

        return metadata, dataname_list, data_list


class H5FileInput:
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


def np_savetxt_3d(array3d, filename):
    with open(filename, mode='w', newline='') as datafile:
        datafile.write('# Array shape: {0}\n'.format(array3d.shape))
        for data_slice in array3d:
            np.savetxt(datafile, data_slice)
            datafile.write('# New slice\n')
