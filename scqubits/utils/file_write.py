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


def filewrite_h5data(file_hook, numpy_data_list, data_info_strings):
    """Write given data along with information describing each data set to an h5 data file.

    Parameters
    ----------
    file_hook: str or h5py root group
    numpy_data_list: ndarray
        data to be written
    data_info_strings: list of str
        text describing the data items to be written
    """
    if isinstance(file_hook, str):
        h5file = h5py.File(filename + '.hdf5', 'w')
        h5file_root = h5file.create_group('root')
    else:
        h5file_root = file_hook

    for dataset_index, dataset in enumerate(numpy_data_list):
        if dataset is not None or []:
            h5dataset = h5file_root.create_dataset("data_" + str(dataset_index), data=dataset, dtype=dataset.dtype)
            h5dataset.attrs['data_info_' + str(dataset_index)] = str(data_info_strings[dataset_index])
