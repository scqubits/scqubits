# misc.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import ast
import functools

import numpy as np
import qutip as qt

from scqubits.settings import IN_IPYTHON
from scqubits.utils.spectrum_utils import convert_esys_to_ndarray

if IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def process_which(which, max_index):
    """
    Parameters
    ----------
    which: int or tuple or list
        single index or tuple/list of integers indexing the eigenobjects.
        If which is -1, all indices up to the max_index limit are included.
    max_index: int
        maximum index value

    Returns
    -------
    list or iterable of indices
    """
    if isinstance(which, int):
        if which == -1:
            return range(max_index)
        return [which]
    return which


def make_bare_labels(subsystem_count, *args):
    """
    For two given subsystem states, return the full-system bare state label obtained by placing all remaining
    subsystems in their ground states.

    Parameters
    ----------
    subsystem_count: int
        number of subsystems inside Hilbert space
    *args: tuple(int, int)
        each argument is a tuple of the form (subsys_index, label)

    Returns
    -------
    tuple
        Suppose there are 5 subsystems in total. Let (subsys_index1=0, label1=3), (subsys_index2=2, label2=1). Then the
        returned bare-state tuple is: (3,0,1,0,0)
    """
    bare_labels = [0] * subsystem_count
    for subsys_index, label in args:
        bare_labels[subsys_index] = label
    return tuple(bare_labels)


def process_metadata(full_dict):
    """Convert an extended system dictionary, as obtained through __dict__, to a reduced one that can be written to
    a file

    Parameters
    ----------
    full_dict: dict
    """
    reduced_dict = {}
    for key, param_obj in full_dict.items():
        if key[0] == '_':
            continue
        if is_numerical(param_obj):
            reduced_dict[key] = param_obj
        elif key == 'grid':
            grid_dict = param_obj._get_metadata_dict()
            reduced_dict.update(grid_dict)
        else:
            reduced_dict[key] = str(param_obj)
    return reduced_dict


def filter_metadata(full_dict):
    """Filter for entries in the full dictionary that have numerical values"""
    reduced_dict = {}
    for param_name, param_value in full_dict.items():
        if is_numerical(param_value):
            reduced_dict[param_name] = param_value
    return reduced_dict


def is_numerical(entity):
    return isinstance(entity, (int, float, complex, np.number))


def is_array_like(entity):
    return isinstance(entity, (list, np.ndarray))


def key_in_grid1d(key):
    return key in ['min_val', 'max_val', 'pt_count']


def value_not_none(key_value):
    _, value = key_value
    return value is not None


def is_numerical_ndarray(entity):
    return isinstance(entity, np.ndarray) and entity.dtype.kind in set('biufc')


def is_qutip_eigenstates(entity):
    return isinstance(entity, np.ndarray) and isinstance(entity[0], qt.Qobj)


def convert_to_ndarray(entity):
    """Convert the object `entity` to a numpy ndarray of numerical dtype. This is needed in the routines for writing
    content of DataStores and SpectrumData to disk.

    Parameters
    ----------
    entity: array_like
    """
    if is_numerical_ndarray(entity):
        return entity
    if is_qutip_eigenstates(entity):  # entity is output from qt.eigenstates
        return convert_esys_to_ndarray(entity)
    if isinstance(entity, (list, np.ndarray)) and is_qutip_eigenstates(entity[0]):
        # entity is a list of qt.eigenstates
        return np.asarray([convert_esys_to_ndarray(entry) for entry in entity])
    # possibly we have a list of numerical values or a list of ndarrays
    converted_entity = np.asarray(entity)
    if converted_entity.dtype.kind not in set('biufc'):
        raise TypeError('Unable to convert data to numerical numpy array: ', entity)
    return converted_entity


class InfoBar:
    def __init__(self, desc, num_cpus):
        self.desc = desc
        self.num_cpus = num_cpus
        self.infobar = None

    def __enter__(self):
        self.infobar = tqdm(total=0, disable=(self.num_cpus == 1), leave=False, desc=self.desc, bar_format="{desc}")

    def __exit__(self, *args):
        self.infobar.close()


class Required:
    def __init__(self, **requirements):
        self.requirements_bools = list(requirements.values())
        self.requirements_names = list(requirements.keys())

    def __call__(self, func, *args, **kwargs):
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            if all(self.requirements_bools):
                return func(*args, **kwargs)
            else:
                raise Exception("ImportError: need extra package(s) {}".format(self.requirements_names))
        return decorated_func


def to_expression_or_string(string_expr):
    try:
        return ast.literal_eval(string_expr)
    except ValueError:
        return string_expr


def remove_nones(dict_data):
    return {key: value for key, value in dict_data.items() if value is not None}
