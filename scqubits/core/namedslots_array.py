# namedslots_array.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import math

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.utils.plotting as plot

from scqubits.io_utils.fileio import IOData
from scqubits.io_utils.fileio_serializers import Serializable

Number = Union[int, float, complex]

NpIndex = Union[int, slice, Tuple[int], List[int]]
NpIndexTuple = Tuple[NpIndex, ...]
NpIndices = Union[NpIndex, NpIndexTuple]
NpSliceEntry = Union[int, None]

GIndex = Union[Number, slice, Tuple[int], List[int]]
GIndexTuple = Tuple[GIndex, ...]
GIndices = Union[GIndex, GIndexTuple]
GSliceEntry = Union[Number, str, None]
GIndexObjectTuple = Tuple["GIndexObject", ...]


def idx_for_value(value: Number, param_vals: ndarray) -> int:
    location = np.abs(param_vals - value).argmin()
    if math.isclose(param_vals[location], value):
        return int(location)
    raise ValueError(
        "No matching idx_entry for parameter value {} in the array.".format(value)
    )


def convert_to_std_npindex(
    index_tuple: GIndexTuple, parameters: "Parameters"
) -> NpIndexTuple:
    gidx_obj_tuple = tuple(
        GIndexObject(entry, parameters, slot=slot_index)
        for slot_index, entry in enumerate(index_tuple)
    )
    np_indices = GIndexTupleObject(gidx_obj_tuple).convert_to_np_index_exp()
    return np_indices


class GIndexObject:
    """Object used for enabling enhanced indexing in NamedSlotNdarray. Handles a
    single idx_entry in multi-index"""

    def __init__(
        self, idx_entry: GIndex, parameters: "Parameters", slot: Optional[int] = None
    ) -> None:
        self.idx_entry = idx_entry
        self.parameters = parameters
        self.slot = slot
        self.name = None
        self.type, self.std_idx_entry = self.convert_to_np_idx_entry(idx_entry)

    def convert_to_np_slice_entry(self, slice_entry: GSliceEntry) -> NpSliceEntry:
        """Handles value-based slices, converting a float or complex value based
        idx_entry into the corresponding position-based idx_entry"""
        if isinstance(slice_entry, int):
            return slice_entry
        if slice_entry is None:
            return None
        if isinstance(slice_entry, (float, complex)):
            return idx_for_value(slice_entry, self.parameters[self.slot])

        raise TypeError("Invalid slice idx_entry: {}".format(slice_entry))

    def convert_to_np_idx_entry(self, idx_entry: GIndex) -> Tuple[str, NpIndex]:
        """Convert a generalized multi-index entry into a valid numpy multi-index entry,
        and returns that along with a str recording the idx_entry type"""
        if isinstance(idx_entry, int):
            return "int", idx_entry

        if idx_entry is Ellipsis:
            return "ellipsis", idx_entry

        if isinstance(idx_entry, (tuple, list)):
            return "tuple", idx_entry

        if isinstance(idx_entry, (float, complex)):
            return "val", idx_for_value(self.idx_entry, self.parameters[self.slot])

        # slice(<str>, ...):  handle str based slices
        if isinstance(idx_entry, slice) and isinstance(idx_entry.start, str):
            self.name = idx_entry.start

            start = self.convert_to_np_slice_entry(idx_entry.stop)
            if isinstance(start, (complex, float)):
                start = idx_for_value(start, self.parameters[self.slot])

            stop = self.convert_to_np_slice_entry(idx_entry.step)
            if isinstance(stop, (complex, float)):
                stop = idx_for_value(stop, self.parameters[self.slot])

            if isinstance(start, int) and (stop is None):
                return "slice.name", start
            return "slice.name", slice(start, stop, None)

        # slice(<Number> or <None>, ...):  handle slices with value-based entries
        if isinstance(idx_entry, slice):
            start = self.convert_to_np_slice_entry(idx_entry.start)
            stop = self.convert_to_np_slice_entry(idx_entry.stop)
            if idx_entry.step is None or isinstance(idx_entry.step, int):
                step = self.convert_to_np_slice_entry(idx_entry.step)
            else:
                raise TypeError(
                    "slice.step can only be int or None. Found {} "
                    "instead.".format(idx_entry.step)
                )
            return "slice", slice(start, stop, step)

        raise TypeError("Invalid index: {}".format(idx_entry))


class GIndexTupleObject:
    def __init__(self, gidx_tuple: Tuple[GIndexObject, ...]):
        self.parameters = gidx_tuple[0].parameters
        self.slot_count = len(self.parameters)
        self.gidx_tuple = gidx_tuple

    def _name_based_to_np_index_exp(self) -> NpIndexTuple:
        """Converts a name-based multi-index into a standard numpy index_exp."""
        converted_multi_index = [slice(None)] * self.slot_count
        for gidx_object in self.gidx_tuple:
            if gidx_object.type != "slice.name":
                raise TypeError("If one index is name-based, all indices must be.")
            slot_index = self.parameters.index_by_name[gidx_object.name]
            converted_multi_index[slot_index] = gidx_object.std_idx_entry

        return tuple(converted_multi_index)

    def convert_to_np_index_exp(self) -> NpIndexTuple:
        """Takes an extended-syntax multi-index idx_entry and converts it to a standard
        position-based multi-index_entry with only integer-valued indices."""
        # inspect first index_entry to determine whether multi-index idx_entry is name-based
        first_gidx = self.gidx_tuple[0]

        if first_gidx.type == "slice.name":  # if one is name based, all must be
            return self._name_based_to_np_index_exp()

        return tuple(gidx.std_idx_entry for gidx in self.gidx_tuple)


class Parameters:
    """Convenience class for maintaining multiple parameter sets: names and values of
    each parameter set, along with an ordering among sets.
    Used in ParameterSweep as `.parameters`. Can access in several ways:
    Parameters[<name str>] = parameter values under this name
    Parameters[<index int>] = parameter values saved as the index-th set
    Parameters[<slice> or tuple(int)] = slice over the list of parameter sets
    Mostly meant for internal use inside ParameterSweep.

    paramvals_by_name:
        dictionary giving names of and values of parameter sets (note problem with
        ordering in python dictionaries
    paramnames_list:
        optional list of same names as in dictionary to set ordering
    """

    def __init__(
        self,
        paramvals_by_name: Dict[str, ndarray],
        paramnames_list: Optional[List[str]] = None,
    ) -> None:
        if paramnames_list is not None:
            self.paramnames_list = paramnames_list
        else:
            self.paramnames_list = list(paramvals_by_name.keys())

        self.names = self.paramnames_list
        self.ordered_dict = OrderedDict(
            [(name, paramvals_by_name[name]) for name in self.names]
        )
        self.paramvals_by_name = self.ordered_dict
        self.index_by_name = {
            name: index for index, name in enumerate(self.paramnames_list)
        }
        self.name_by_index = {
            index: name for index, name in enumerate(self.paramnames_list)
        }
        self.paramvals_by_index = {
            self.index_by_name[name]: param_vals
            for name, param_vals in self.paramvals_by_name.items()
        }

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.paramvals_by_name[key]
        if isinstance(key, int):
            return self.paramvals_by_name[self.paramnames_list[key]]
        if key is Ellipsis:
            key = slice(None, None, None)
        if isinstance(key, slice):
            sliced_paramnames_list = self.paramnames_list[key]
            return [self.paramvals_by_name[name] for name in sliced_paramnames_list]
        if isinstance(key, (tuple, list)):
            key = self._process_ellipsis(key)
            return [
                self.paramvals_by_name[self.paramnames_list[index]][key[index]]
                for index in range(len(self))
            ]

    def __len__(self):
        return len(self.paramnames_list)

    def __iter__(self):
        return iter(self.paramvals_list)

    def _process_ellipsis(self, multi_idx: Union[tuple, list]) -> Union[tuple, list]:
        if isinstance(multi_idx, list):
            return multi_idx
        if Ellipsis not in multi_idx:
            return multi_idx
        new_multi_idx = [slice(None, None, None)] * len(self)
        slot = 0
        while multi_idx[slot] is not Ellipsis:
            new_multi_idx[slot] = multi_idx[slot]
            slot += 1
        slot = -1
        while multi_idx[slot] is not Ellipsis:
            new_multi_idx[slot] = multi_idx[slot]
            slot -= 1
        return tuple(new_multi_idx)

    @property
    def counts_by_name(self) -> Dict[str, int]:
        """Returns a dictionary specifying for each parameter name the number of
        parameter values"""
        return {
            name: len(self.paramvals_by_name[name])
            for name in self.paramvals_by_name.keys()
        }

    @property
    def ranges(self) -> List[Iterable]:
        """Return a list of range objects suitable for looping over each parametr set"""
        return [range(count) for count in self.counts]

    @property
    def paramvals_list(self) -> List[ndarray]:
        """Return list of all parameter values sets"""
        return [self.paramvals_by_name[name] for name in self.paramnames_list]

    def get_index(self, value: float, slotindex: int) -> int:
        """Return the parameter index for a given parameter value of parameter set in
        specified slotindex"""
        location = np.abs(self[slotindex] - value).argmin()
        return int(location)

    @property
    def counts(self) -> Tuple[int]:
        """Returns list of the number of parameter values for each parameter set"""
        return tuple(len(paramvals) for paramvals in self)

    def create_reduced(
        self,
        fixed_parametername_list: List[str],
        fixed_values: Optional[List[float]] = None,
    ) -> "Parameters":
        """
        Creates and returns a reduced Parameters object reflecting the fixing of a
        subset of parameters

        Parameters
        ----------
        fixed_parametername_list:
            names of parameters to be fixed
        fixed_values:
            list of values to which parameters are fixed, optional (default: use the
            0-th element of the array of each fixed parameter)

        Returns
        -------
            Parameters object with all parameters; fixed ones only including one
            value
        """
        if fixed_values is not None:
            # need to reformat as array of single-idx_entry arrays
            fixed_values = [np.asarray(value) for value in fixed_values]
        else:
            fixed_values = [
                np.asarray([self[name][0]]) for name in fixed_parametername_list
            ]

        reduced_paramvals_by_name = {name: self[name] for name in self.paramnames_list}
        for index, name in enumerate(fixed_parametername_list):
            reduced_paramvals_by_name[name] = fixed_values[index]
        return Parameters(reduced_paramvals_by_name)

    def create_sliced(
        self, np_indices: NpIndices, remove_fixed: bool = True
    ) -> "Parameters":
        """
        Create and return a sliced Parameters object according to numpy slicing
        information.

        Parameters
        ----------
        np_indices:
            numpy slicing entries
        remove_fixed:
            if True, do not include fixed parameters in the returned Parameters object

        Returns
        -------
            Parameters object with either fixed parameters removed or including only
            the fixed value
        """
        parameter_array = np.asarray(self.paramvals_list, dtype=object).copy()
        for index, np_index in enumerate(np_indices):
            array_entry = parameter_array[index][np_index]
            if isinstance(array_entry, (float, int, complex)):
                array_entry = np.asarray([array_entry])
            parameter_array[index] = array_entry

        reduced_paramvals_by_name = {}
        for index, name in enumerate(self.paramnames_list):
            paramvals = parameter_array[index]
            if not remove_fixed:
                reduced_paramvals_by_name[name] = paramvals
            elif isinstance(paramvals, (ndarray, list, range)) and len(paramvals) > 1:
                reduced_paramvals_by_name[name] = paramvals

        return Parameters(reduced_paramvals_by_name)


class NamedSlotsNdarray(np.ndarray, Serializable):
    """
    This class implements multi-dimensional arrays, for which the leading M dimensions
    are each associated with a slot name and a corresponding array of slot
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

    A special treatment is reserved for a pure string idx_entry in position 0: this
    string will be directly converted into an index via the corresponding
    values_by_slotindex.
    """

    parameters: Parameters

    def __new__(
        cls, input_array: np.ndarray, values_by_name: Dict[str, Iterable]
    ) -> "NamedSlotsNdarray":
        implied_shape = tuple(len(values) for name, values in values_by_name.items())
        if input_array.shape[0 : len(values_by_name)] != implied_shape:
            raise ValueError(
                "Given input array with shape {} not compatible with "
                "provided dict calling for shape {}. values_by_name: {}".format(
                    input_array.shape, implied_shape, values_by_name
                )
            )

        obj = np.asarray(input_array).view(cls)

        obj.parameters = Parameters(values_by_name)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.parameters = getattr(obj, "parameters", None)

    def __getitem__(self, multi_index: GIndices) -> Any:
        """Overwrites the magic method for element selection and slicing to support
        extended string and value based slicing."""
        multi_index = np.index_exp[multi_index]  # convert to standardized tuple form
        try:
            obj = super().__getitem__(multi_index)
            # This attempt fails if multi-index is string- or value-based
        except (TypeError, IndexError):
            multi_index = convert_to_std_npindex(multi_index, self.parameters)
            obj = super().__getitem__(multi_index)

        # If the resulting obj is a sliced view of the current array, then we must
        # adjust the internal Parameters instance accordingly
        if isinstance(obj, NamedSlotsNdarray):
            # Check whether all parameters are getting fixed; if not, adjust
            # Parameters for the new object
            if not hasattr(self, "parameters"):
                print("it's me", len(self))
            param_count = len(self.parameters)
            dummy_array = np.empty(shape=self.parameters.counts)
            if not isinstance(dummy_array[multi_index[:param_count]], float):
                # have not reduced to one element
                obj.parameters = self.parameters.create_sliced(
                    multi_index[:param_count]
                )
            elif obj.parameters.paramvals_by_name == self.parameters.paramvals_by_name:
                # Have reduced to one element, which is still an array however. If this
                # was a regular ndarray (not NamedSlotsNdarray), the Parameters entry
                # will be the same as in parent array. Need to delete this, i.e., just
                # return ordinary ndarray.
                obj = obj.view(ndarray)
        return obj

    def __reduce__(self):
        # needed for multiprocessing / proper pickling
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self.parameters,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # needed for multiprocessing / proper pickling
        self.parameters = state[-1]
        super().__setstate__(state[0:-1])

    @classmethod
    def deserialize(cls, io_data: IOData) -> "NamedSlotsNdarray":
        """
        Take the given IOData and return an instance of the described class, initialized
        with the data stored in io_data.
        """
        input_array = np.asarray(io_data.objects["input_array"], dtype=object)
        values_by_name = io_data.objects["values_by_name"]
        return NamedSlotsNdarray(input_array, values_by_name)

    def serialize(self) -> IOData:
        """
        Convert the content of the current class instance into IOData format.
        """
        import scqubits.io_utils.fileio as io

        typename = type(self).__name__
        io_attributes = None
        io_ndarrays = None
        objects = {
            "input_array": self.tolist(),
            "values_by_name": self.parameters.paramvals_by_name,
        }
        return io.IOData(typename, io_attributes, io_ndarrays, objects=objects)

    @property
    def slot_count(self) -> int:
        return len(self.parameters.paramvals_by_name)

    def plot(self, **kwargs) -> Tuple[Figure, Axes]:
        if len(self.parameters) != 1:
            raise ValueError(
                "Plotting of NamedSlotNdarray only supported for a "
                "one-dimensional parameter sweep. (Consider slicing.)"
            )
        return plot.data_vs_paramvals(
            xdata=self.parameters.paramvals_list[0], ydata=self, **kwargs
        )

    def recast(self) -> "NamedSlotsNdarray":
        return NamedSlotsNdarray(
            np.asarray(self[:].tolist()), self.parameters.paramvals_by_name
        )

    def toarray(self) -> ndarray:
        return np.asarray(self[:].tolist())
