# misc.py
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
import math

from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import qutip.qobj as qt
from numpy import ndarray

from scqubits.settings import IN_IPYTHON

if IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


Number = Union[int, float, complex]


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
    For two given subsystem states, return the full-system bare state label obtained
    by placing all remaining subsys_list in their ground states.

    Parameters
    ----------
    subsystem_count:
        number of subsys_list inside Hilbert space
    *args:
        each argument is a tuple of the form (subsys_index, label)

    Returns ------- Suppose there are 5 subsys_list in total. Let (subsys_index1=0,
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
        self.tqdm_bar: tqdm = None

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


def to_expression_or_string(string_expr: str) -> Any:
    try:
        return ast.literal_eval(string_expr)
    except ValueError:
        return string_expr


def remove_nones(dict_data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict_data.items() if value is not None}


def qt_ket_to_ndarray(qobj_ket: qt.Qobj) -> np.ndarray:
    # Qutip's `.eigenstates()` returns an object-valued ndarray, each entry of which
    # is a Qobj ket.
    return np.asarray(qobj_ket.data.todense())


class NamedSliceableSlots:
    """
    This mixin class applies to multi-dimensional arrays, for which the leading M
    dimensions are each associated with a slot name and a corresponding array of slot
    values (float or complex or str). All standard slicing of the multi-dimensional
    array with integer-valued indices is supported as usual, e.g.

        some_array[0, 3:-1, -4, ::2]

    Slicing of the multi-dimensional array associated with named sets of values is
    extended in two ways:

    (1) Value-based slicing
    Integer indices other than the `step` index may be
    replaced by a float or a complex number or a str. This prompts a lookup and
    substitution by the integer index representing the location of the closest
    element (as measured by the absolute value of the difference for numbers,
    and an exact match for str) in the set of slot values.

    As an example, consider the situation of two named value sets

        values_by_slotname = {'param1': np.asarray([-4.4, -0.1, 0.3, 10.0]),
                              'param2': np.asarray([0.1*1j, 3.0 - 4.0*1j, 25.0])}

    Then, the following are examples of value-based slicing:

        some_array[0.25, 0:2]                   -->     some_array[2, 0:2]
        some_array[-3.0, 0.0:(2.0 - 4.0*1j)]    -->     some_array[0, 0:1]


    (2) Name-based slicing
    Sometimes, it is convenient to refer to one of the slots
    by its name rather than its position within the multiple sets. As an example, let

        values_by_slotname = {'ng': np.asarray([-0.1, 0.0, 0.1, 0.2]),
                             'flux': np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])}

    If we are interested in the slice of `some_array` obtained by setting 'flux' to a
    value or the value associated with a given index, we can now use:

        some_array['flux':0.5]            -->    some_array[:, 1]
        some_array['flux'::2, 'ng':-1]    -->    some_array[-1, :2]

    Name-based slicing has the format `<name str>:start:stop`  where `start` and
    `stop` may be integers or make use of value-based slicing. Note: the `step`
    option is not available in name-based slicing. Name-based and standard
    position-based slicing cannot be combined: `some_array['name1':3, 2:4]` is not
    supported. For such mixed- mode slicing, use several stages of slicing as in
    `some_array['name1':3][2:4]`.

    A special treatment is reserved for a pure string entry in position 0: this
    string will be directly converted into an index via the corresponding
    values_by_slotindex.
    """

    values_by_slotname: Dict[str, Iterable]
    values_by_slotindex: Dict[int, Iterable]
    slotname_by_slotindex: Dict[str, int]
    slotindex_by_slotname: Dict[int, str]
    data_callback: Union[ndarray, Callable]

    def __getitem__(
        self, multi_index: Union[Number, Tuple[Union[Number, slice], ...]]
    ) -> Any:
        """Overwrites the magic method for element selection and slicing to support
        the extended slicing options."""
        if not isinstance(multi_index, tuple):
            multi_index = (multi_index,)

        if callable(self.data_callback):
            return self.data_callback(self.convert_to_standard_multi_index(multi_index))
        return self.data_callback[self.convert_to_standard_multi_index(multi_index)]

    def convert_slotnames_to_indices(
        self, multi_index: Tuple[Union[Number, slice], ...]
    ) -> Tuple[Union[Number, slice], ...]:
        """Converts a name-based multi-index into a position-based multi-index."""
        converted_multi_index = [slice(None)] * self.slot_count
        for this_slice in multi_index:
            try:
                name = this_slice.start
                index = self.slotindex_by_slotname[name]
                if this_slice.step is None:
                    converted_slice = this_slice.stop
                else:
                    converted_slice = slice(this_slice.stop, this_slice.step)
                converted_multi_index[index] = converted_slice
            except (AttributeError, KeyError):
                raise Exception(
                    "Slicing error: could not convert slot-name based slices to "
                    "ordinary slices."
                )
        return tuple(converted_multi_index)

    def convert_to_standard_multi_index(
        self, multi_index: Tuple[Union[Number, slice], ...]
    ) -> Tuple[Union[int, slice], ...]:
        """Takes an extended-syntax multi-index_entry and converts it to a standard
        position-based multi-index_entry with only integer-valued indices."""

        # inspect first index_entry to determine whether multi-index_entry is name-based
        index_entry = multi_index[0]

        if isinstance(index_entry, slice) and isinstance(index_entry.start, str):
            # Multi-index_entry is name-based (slices with a str as the start attribute)
            # -> convert to position-based multi-index_entry
            multi_index = self.convert_slotnames_to_indices(multi_index)
            processed_multi_index = [slice(None)] * self.slot_count
        else:
            # Multi-index_entry is position based (nothing to do, just convert to
            # list further to be processed)
            processed_multi_index = list(multi_index)

        # Check for value-based indices and slice entries, and convert to standard
        # indices
        for position, index_entry in enumerate(multi_index):
            if isinstance(index_entry, int):
                # all int entries are taken as standard indices
                processed_multi_index[position] = index_entry
            elif isinstance(index_entry, (float, complex)):
                # individual value-based index_entry
                value = index_entry
                index = self.find_index_if_value_exists(position, value)
                if index is None:
                    raise ValueError(
                        "No matching entry for parameter value {} in the array.".format(
                            value
                        )
                    )
                processed_multi_index[position] = index
            elif isinstance(index_entry, str) and position == 0:
                processed_multi_index[position] = np.where(
                    self.values_by_slotindex[0] == index_entry
                )[0]
            elif isinstance(index_entry, slice):
                # slice objects must be checked for internal value-based entries in
                # start and stop attributes
                this_slice = (
                    index_entry
                    if self.is_standard_slice(index_entry)
                    else self.convert_to_standard_slice(index_entry, position)
                )
                processed_multi_index[position] = this_slice
        return tuple(processed_multi_index)

    def is_standard_slice(self, this_slice: slice) -> bool:
        """Checks whether slice is standard, i.e., all entries for start, stop,
        and step are integers."""
        if (
            isinstance(this_slice.start, (int, type(None)))
            and isinstance(this_slice.stop, (int, type(None)))
            and isinstance(this_slice.step, (int, type(None)))
        ):
            return True
        return False

    def convert_to_standard_slice(self, this_slice: slice, range_index: int) -> slice:
        """Takes a single slice object that includes value-based entries and converts
        them into integer indices reflecting the position of the closest element in
        the given value set."""
        start = this_slice.start
        stop = this_slice.stop
        step = this_slice.step
        if not isinstance(start, (int, type(None))):
            value = start
            start = self.get_index_closest_value(value, range_index)
        if not isinstance(stop, (int, type(None))):
            value = stop
            stop = self.get_index_closest_value(value, range_index)
        if not isinstance(step, (int, type(None))):
            raise Exception("Slicing error: non-integer step sizes not supported.")
        return slice(start, stop, step)

    @property
    def slot_count(self) -> int:
        return len(self.values_by_slotname)

    def get_index_closest_value(self, value: Number, index: int) -> int:
        location = np.abs(self.values_by_slotindex[index] - value).argmin()
        return location

    def find_index_if_value_exists(
        self, position: int, value: Number
    ) -> Union[int, None]:
        location = np.abs(self.values_by_slotindex[index] - value).argmin()
        if math.isclose(self.values_by_slotindex[index][location], value):
            return location
        return None


class NamedSlotsNdarray(np.ndarray, NamedSliceableSlots):
    def __new__(cls, input_array: np.ndarray, values_by_name: Dict[str, Iterable]):
        implied_shape = tuple(len(values) for name, values in values_by_name.items())
        if input_array.shape[0 : len(values_by_name)] != implied_shape:
            raise ValueError(
                "Given input array with shape {} not compatible with "
                "provided dict calling for shape {}.".format(
                    input_array.shape, implied_shape
                )
            )

        obj = np.asarray(input_array).view(cls)
        obj.values_by_slotname = values_by_name
        obj.slotindex_by_slotname = {
            name: index for index, name in enumerate(values_by_name.keys())
        }
        obj.slotname_by_slotindex = {
            index: name for name, index in obj.slotindex_by_slotname.items()
        }
        obj.values_by_slotindex = {
            obj.slotindex_by_slotname[name]: values_by_name[name]
            for name in values_by_name.keys()
        }
        obj.data_callback = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.values_by_name = getattr(obj, "values_by_slotname", None)
        self.slotindex_by_slotname = getattr(obj, "slotindex_by_slotname", None)
        self.slotname_by_slotindex = getattr(obj, "slotname_by_slotindex", None)
        self.values_by_index = getattr(obj, "values_by_slotindex", None)
        self.data_callback = getattr(obj, "data_callback", None)

    def __getitem__(
        self, multi_index: Union[Number, Tuple[Union[Number, slice], ...]]
    ) -> Any:
        """Overwrites the magic method for element selection and slicing to support
        the extended slicing options."""
        if not isinstance(multi_index, tuple):
            multi_index = (multi_index,)
        return super().__getitem__(self.convert_to_standard_multi_index(multi_index))
