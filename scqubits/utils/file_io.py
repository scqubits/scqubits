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

import numpy as np
import h5py


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
    numpy_data_list: list
        list of ndarrays containing datasets to be written
    data_info_strings: list of str
        text describing the data items to be written
    """
    if isinstance(file_hook, str):
        h5file = h5py.File(file_hook + '.hdf5', 'w')
        h5file_root = h5file.create_group('root')
    else:
        h5file_root = file_hook

    for dataset_index, dataset in enumerate(numpy_data_list):
        if dataset is not None or []:
            h5dataset = h5file_root.create_dataset("data_" + str(dataset_index), data=dataset, dtype=dataset.dtype)
            h5dataset.attrs['data_info_' + str(dataset_index)] = str(data_info_strings[dataset_index])


def read_h5(filename):
    """Read scqubit data from an h5 file

    Parameters
    ----------
    filename: str or h5py.File
        Path for file to be opened, or open h5py handle

    Returns
    -------
    h5py.Group: handle to 'root' of h5 file
    list: list of datasets
    """
    h5file = h5py.File(filename, 'r')
    datalist = []

    def func(name, obj):
        if isinstance(obj, h5py.Dataset):
            datalist.append(obj)

    h5file['root'].visititems(func)
    return h5file['root'], datalist
