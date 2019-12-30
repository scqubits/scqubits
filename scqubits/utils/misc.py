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


import numpy as np


def process_which(which, max_index):
    """

    Parameters
    ----------
    which: int or tuple or list, optional
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


def make_bare_labels(hilbertspace, subsys_index1, label1, subsys_index2, label2):
    bare_labels = [0] * hilbertspace.subsystem_count
    bare_labels[subsys_index1] = label1
    bare_labels[subsys_index2] = label2
    return tuple(bare_labels)


def process_metadata(full_dict):
    reduced_dict = {}
    for key, param_obj in full_dict.items():
        if key[0] == '_':
            continue
        if isinstance(param_obj, (int, float, np.number)):
            reduced_dict[key] = param_obj
        elif key == 'grid':
            grid_dict = param_obj._get_metadata_dict()
            reduced_dict.update(grid_dict)
        else:
            reduced_dict[key] = str(param_obj)
    return reduced_dict
