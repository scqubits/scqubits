# file_io_csv.py
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
Helper routines for writing data to CSV.
"""

import ast
import csv
import os
import re

import numpy as np

import scqubits.utils.file_io as io
import scqubits.utils.file_io_base as io_base
from scqubits.utils.misc import to_expression_or_string


class CSVWriter(io_base.IOWriter):
    """
    Given filename='somename.csv', write initdata into somename.csv
    Then, additional csv files are written for each dataset, with filenames: 'somename_' + dataname0 + '.csv' etc.
    """
    def append_ndarray_info(self, attributes):
        """Add data set information to attributes, so that dataset names and dimensions are available
        in attributes CSV file."""
        for index, dataname in enumerate(self.io_data.ndarrays.keys()):
            data = self.io_data.ndarrays[dataname]
            attributes['dataset' + str(index)] = dataname

            if data.ndim == 3:
                slice_count = len(data)
            else:
                slice_count = 1
            attributes['dataset' + str(index) + '.slices'] = slice_count
        return attributes

    def write_attributes(self, filename):
        attributes = self.io_data.attributes
        attributes["__type"] = self.io_data.typename
        attributes = self.append_ndarray_info(attributes)
        with open(filename, mode='w', newline='') as meta_file:
            file_writer = csv.writer(meta_file, delimiter=',')
            file_writer.writerow(attributes.keys())
            file_writer.writerow(attributes.values())

    def write_ndarrays(self, filename):
        filename_stub, _ = os.path.splitext(filename)
        for dataname, dataset in self.io_data.ndarrays.items():
            filename = filename_stub + '_' + dataname + '.csv'
            self.write_data(filename, dataset)

    def write_data(self, filename, dataset):
        if dataset.ndim <= 2:
            np.savetxt(filename, dataset)
        elif dataset.ndim == 3:
            np_savetxt_3d(dataset, filename)
        else:
            raise Exception("Dataset has dimensions > 3. Cannot write to CSV file.")

    def write_objects(self, *args, **kwargs):
        raise NotImplementedError

    def to_file(self, io_data, **kwargs):
        self.io_data = io_data
        self.write_attributes(self.filename)
        self.write_ndarrays(self.filename)
        # no support for write_objects in CSV format


class CSVReader:
    @staticmethod
    def read_attributes(filename):
        with open(filename, mode='r') as meta_file:
            file_reader = csv.reader(meta_file, delimiter=',')
            meta_keys = file_reader.__next__()
            meta_values = file_reader.__next__()
        return dict(zip(meta_keys, meta_values))

    def process_metadict(self, meta_dict):
        attributes = {attr_name: to_expression_or_string(attr_value) for attr_name, attr_value in meta_dict.items()
                      if not re.match(r'dataset\d+', attr_name)}
        data_names = [dataname for datalabel, dataname in meta_dict.items() if re.match(r'dataset\d+$', datalabel)]
        data_slices = [ast.literal_eval(value) for key, value in meta_dict.items()
                       if re.match(r'dataset\d+.slices', key)]
        return attributes, data_names, data_slices

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

    def from_file(self, filename, **kwargs):
        """
        Parameters
        ----------
        filename: str

        Returns
        -------
        class instance generated from file data
        """
        ext_attributes = self.read_attributes(filename)
        typename = ext_attributes['__type']
        del ext_attributes['__type']

        attributes, data_names, data_slices = self.process_metadict(ext_attributes)

        filename_stub, _ = os.path.splitext(filename)
        ndarrays = {}
        for index, dataname in enumerate(data_names):
            data_filename = filename_stub + '_' + dataname + '.csv'
            slices = data_slices[index]
            ndarrays[dataname] = self.read_data(data_filename, slices)

        return io.IOData(typename, attributes, ndarrays, objects=None)


def np_savetxt_3d(array3d, filename):
    with open(filename, mode='w', newline='') as datafile:
        datafile.write('# Array shape: {0}\n'.format(array3d.shape))
        for data_slice in array3d:
            np.savetxt(datafile, data_slice)
            datafile.write('# New slice\n')
