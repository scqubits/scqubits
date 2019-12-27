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
