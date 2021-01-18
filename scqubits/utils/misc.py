# fitting.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import ast
import functools
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import qutip.qobj as qt

from scqubits.settings import IN_IPYTHON

if IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def process_which(which: Union[int, Iterable[int]], max_index: int) -> List[int]:
    """
    Processes different ways of specifying the selection of wanted eigenvalues/eigenstates.

    Parameters
    ----------
    which:
        single index or tuple/list of integers indexing the eigenobjects.
        If which is -1, all indices up to the max_index limit are included.
    max_index:
        maximum index value

    Returns
    -------
        indices
    """
    if isinstance(which, int):
        if which == -1:
            return list(range(max_index))
        return [which]
    return list(which)


def make_bare_labels(subsystem_count: int, *args) -> Tuple[int, ...]:
    """
    For two given subsystem states, return the full-system bare state label obtained by placing all remaining
    subsys_list in their ground states.

    Parameters
    ----------
    subsystem_count:
        number of subsys_list inside Hilbert space
    *args:
        each argument is a tuple of the form (subsys_index, label)

    Returns
    -------
        Suppose there are 5 subsys_list in total. Let (subsys_index1=0, label1=3), (subsys_index2=2, label2=1). Then the
        returned bare-state tuple is: (3,0,1,0,0)
    """
    bare_labels = [0] * subsystem_count
    for subsys_index, label in args:
        bare_labels[subsys_index] = label
    return tuple(bare_labels)


def drop_private_keys(full_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter for entries in the full dictionary that have numerical values"""
    return {key: value for key, value in full_dict.items() if key[0] != '_'}


class InfoBar:
    """Static "progress" bar used whenever multiprocessing is involved.

    Parameters
    ----------
    desc:
        Description text to be displayed on the static information bar.
    num_cpus:
        Number of CPUS/cores employed in underlying calculation.
    """

    def __init__(self, desc: str, num_cpus: int) -> None:
        self.desc = desc
        self.num_cpus = num_cpus
        self.tqdm_bar: tqdm = None

    def __enter__(self) -> None:
        self.tqdm_bar = tqdm(total=0, disable=(self.num_cpus == 1), leave=False, desc=self.desc, bar_format="{desc}")

    def __exit__(self, *args) -> None:
        self.tqdm_bar.close()


class Required:
    """Decorator class, ensuring that a given requirement or set of requirements is fulfilled.

    Parameters
    ----------
    dict {str: bool}
        All bool conditions have to be True to pass. The provided str keys are used to display information on what
        condition is failing.
    """

    def __init__(self, **requirements) -> None:
        self.requirements_bools = list(requirements.values())
        self.requirements_names = list(requirements.keys())

    def __call__(self, func: Callable, *args, **kwargs) -> Callable:
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            if all(self.requirements_bools):
                return func(*args, **kwargs)
            else:
                raise Exception("ImportError: use of this method requires the optional package(s): {}. If you wish to "
                                "use this functionality, the corresponding package(s) must be installed manually. "
                                "(Installation via `conda install -c conda-forge <packagename>` or "
                                "`pip install <packagename>` is recommended.)".format(self.requirements_names))
        return decorated_func


def to_expression_or_string(string_expr: str) -> Any:
    try:
        return ast.literal_eval(string_expr)
    except ValueError:
        return string_expr


def remove_nones(dict_data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict_data.items() if value is not None}


def qt_ket_to_ndarray(qobj_ket: qt.Qobj) -> np.ndarray:
    # Qutip's `.eigenstates()` returns an object-valued ndarray, each entry of which is a Qobj ket.
    return np.asarray(qobj_ket.data.todense())
