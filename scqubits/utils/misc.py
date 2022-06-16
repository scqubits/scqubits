# misc.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import ast
import functools
import platform
import warnings

from collections.abc import Sequence
from io import StringIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import qutip as qt
import scipy as sp

from scqubits.settings import IN_IPYTHON

if IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def process_which(which: Union[int, Iterable[int]], max_index: int) -> List[int]:
    """Processes different ways of specifying the selection of  wanted
    eigenvalues/eigenstates.

    Parameters
    ----------
    which:
        single index or tuple/list of integers indexing the eigenobjects.
        If 'which' is -1, all indices up to the max_index limit are included.
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
    For two given subsystem states, return the full-system bare state label obtained
    by placing all remaining subsys_list in their ground states.

    Parameters
    ----------
    subsystem_count:
        number of subsys_list inside Hilbert space
    *args:
        each argument is a tuple of the form (subsys_index, label)

    Returns
    -------
        Suppose there are 5 subsys_list in total. Let (subsys_index1=0,
        label1=3), (subsys_index2=2, label2=1). Then the returned bare-state tuple is:
        (3,0,1,0,0)
    """
    bare_labels = [0] * subsystem_count
    for subsys_index, label in args:
        bare_labels[subsys_index] = label
    return tuple(bare_labels)


def drop_private_keys(full_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter for entries in the full dictionary that have numerical values"""
    return {key: value for key, value in full_dict.items() if key[0] != "_"}


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
        self.tqdm_bar: Optional[tqdm] = None

    def __enter__(self) -> None:
        self.tqdm_bar = tqdm(
            total=0,
            disable=(self.num_cpus == 1),
            leave=False,
            desc=self.desc,
            bar_format="{desc}",
        )

    def __exit__(self, *args) -> None:
        self.tqdm_bar.close()


class Required:
    """Decorator class, ensuring that a given requirement or set of requirements is
    fulfilled.

    Parameters
    ----------
    dict {str: bool}
        All bool conditions have to be True to pass. The provided str keys are used to
        display information on what condition is failing.
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
                raise Exception(
                    "ImportError: use of this method requires the optional package(s):"
                    " {}. If you wish to use this functionality, the corresponding"
                    " package(s) must be installed manually. (Installation via `conda"
                    " install -c conda-forge <packagename>` or `pip install"
                    " <packagename>` is recommended.)".format(self.requirements_names)
                )

        return decorated_func


def check_sync_status(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._out_of_sync:
            warnings.warn(
                "[scqubits] Some system parameters have been changed and"
                " generated spectrum data could be outdated, potentially leading to"
                " incorrect results. Spectral data can be refreshed via"
                " <HilbertSpace>.generate_lookup() or <ParameterSweep>.run()",
                Warning,
            )
        return func(self, *args, **kwargs)

    return wrapper


class DeprecationMessage:
    """Decorator class, producing an adjustable warning and info upon usage of the
    decorated function.

    Parameters
    ----------
    warning_message:
        Warnings message to be sent upon decorated (deprecated) routing
    """

    def __init__(self, warning_msg: str) -> None:
        self.warning_msg = warning_msg

    def __call__(self, func: Callable, *args, **kwargs) -> Callable:
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            warnings.warn(self.warning_msg, FutureWarning)
            return func(*args, **kwargs)

        return decorated_func


def to_expression_or_string(string_expr: str) -> Any:
    try:
        return ast.literal_eval(string_expr)
    except ValueError:
        return string_expr


def remove_nones(dict_data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict_data.items() if value is not None}


def qt_ket_to_ndarray(qobj_ket: qt.Qobj) -> np.ndarray:
    # Qutip's `.eigenstates()` returns an object-valued ndarray, each idx_entry of which
    # is a Qobj ket.
    return np.asarray(qobj_ket.data.todense())


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """
    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = "not all lists have the same length"
            raise ValueError(msg)

    shape += (len(lst),)

    # recurse
    shape = get_shape(lst[0], shape)

    return shape


def tuple_to_short_str(the_tuple: tuple) -> str:
    short_str = ""
    for entry in the_tuple:
        short_str += str(entry) + ","
    return short_str[:-1]


def to_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return [obj]


def about(print_info=True):
    """Prints or returns a string with basic information about
    scqubits as well as installed version of various packages
    that scqubits depends on.

    Parameters
    ----------
    print_info: bool
        Flag that determines if string with information should
        be printed (if True) or returned (if False).

    Returns
    -------
    None or str
    """
    from scqubits import __version__

    fs = StringIO()

    fs.write("scqubits: a Python library for simulating superconducting qubits\n")
    fs.write("****************************************************************\n")
    fs.write("Developed by J. Koch, P. Groszkowski\n")
    fs.write("Main Github page: https://github.com/scqubits/scqubits\n")
    fs.write(
        "Online documentation page: https://scqubits.readthedocs.io/en/latest/\n\n"
    )
    fs.write("scqubits version: {}\n".format(__version__))
    fs.write("numpy version:    {}\n".format(np.__version__))
    fs.write("scipy version:    {}\n".format(sp.__version__))
    fs.write("QuTiP version:    {}\n".format(qt.__version__))
    fs.write(
        "Platform:         {} ({})\n".format(platform.system(), platform.machine())
    )

    if print_info:
        print(fs.getvalue())
        return None
    else:
        return fs.getvalue()


def cite(print_info=True):
    """Prints or returns a string with scqubits citation
    information.

    Parameters
    ----------
    print_info: bool
        Flag that determines if string with information should
        be printed (if True) or returned (if False).

    Returns
    -------
    None or str

    """
    fs = StringIO()
    fs.write("Peter Groszkowski and Jens Koch,\n")
    fs.write("'scqubits: a Python package for superconducting qubits'\n")
    fs.write("Quantum 5, 583 (2021).\n")
    fs.write("https://quantum-journal.org/papers/q-2021-11-17-583/\n")

    if print_info:
        print(fs.getvalue())
        return None
    else:
        return fs.getvalue()


def is_float_string(the_string: str) -> bool:
    try:
        float(the_string)
        return True
    except ValueError:
        return False


def list_intersection(list1: list, list2: list) -> list:
    return list(set(list1) & set(list2))


def flatten_list(nested_list):
    """
    Flattens a list of lists once, not recursive.

    Parameters
    ----------

    nested_list:
        A list of lists, which can hold any class instance.

    Returns
    -------
    Flattened list of objects
    """
    return functools.reduce(lambda a, b: a + b, nested_list)


def flatten_list_recursive(S):
    """
    Flattens a list of lists recursively.

    Parameters
    ----------

    nested_list:
        A list of lists, which can hold any class instance.

    Returns
    -------
    Flattened list of objects
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten_list_recursive(S[0]) + flatten_list_recursive(S[1:])
    return S[:1] + flatten_list_recursive(S[1:])
