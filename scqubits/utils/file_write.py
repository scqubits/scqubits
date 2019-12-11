# file_write.py
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

import h5py
import numpy as np

def filewrite_csvdata(filename, numpy_array):
    """Writes numpy array to file as CSV data.

    Parameters
    ----------
    filename: str
        path and filename of output file (.csv suffix appended automatically)
        
    numpy_array: ndarray
        data to be written
    """
    np.savetxt(filename + '.csv', numpy_array, delimiter=",")


def filewrite_h5data(filename, numpy_data_list, data_info_strings, param_info_dict):
    """Write given data along with information describing each data set to an h5 data file.

    Parameters
    ----------
    filename: str
        path and filename of output file (.h5 suffix appended automatically)
    numpy_data_list: ndarray
        data to be written
    data_info_strings: list of str
        text describing the data items to be written
    param_info_dict: dict of str
        describes system parameters corresponding to data
    """
    h5file = h5py.File(filename + '.hdf5', 'w')
    h5group = h5file.create_group('root')
    for dataset_index, dataset in enumerate(numpy_data_list):
        if dataset is not None:
            h5dataset = h5group.create_dataset(np.string_('data_' + str(dataset_index)), data=dataset,
                                               compression="gzip")
            h5dataset.attrs['data_info_' + str(dataset_index)] = np.string_(data_info_strings[dataset_index])
    for key, info in param_info_dict.items():
        h5group.attrs[key] = np.string_(info)
    h5file.close()
