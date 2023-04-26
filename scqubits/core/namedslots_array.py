# namedslots_array.py
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

import cmath
import numbers
import warnings

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.settings as settings
import scqubits.utils.misc as utils
import scqubits.utils.plotting as plot

from scqubits.io_utils.fileio import IOData
from scqubits.io_utils.fileio_serializers import Serializable

EllipsisType = Any  # unfortunate workaround (see ongoing discussion)

# Standard numpy types valid as a single slot index; with and without Ellipsis
# ExtIndex is a single-slot index that enables custom slicing options, including
# value-based indexing
NpIndexNoEllipsis = Union[
    int,
    np.integer,
    slice,
    Tuple[int],
    List[int],
    Tuple[int, np.integer],
    List[np.integer],
]
NpIndex = Union[NpIndexNoEllipsis, EllipsisType]
ExtIndex = Union[NpIndex, float, complex]

# Tuple of standard numpy or extended indices spans/can span multiple slots; with and
# without Ellipsis
NpIndexTupleNoEllipsis = Tuple[NpIndexNoEllipsis, ...]
NpIndexTuple = Tuple[NpIndex, ...]
ExtIndexTuple = Tuple[ExtIndex, ...]

# Single- or multi-slot numpy or extended index; with and without Ellipsis
NpIndicesNoEllipsis = Union[NpIndexNoEllipsis, NpIndexTupleNoEllipsis]
NpIndices = Union[NpIndex, NpIndexTuple]
ExtIndices = Union[ExtIndex, ExtIndexTuple]

# Numpy: valid slice(a, b, c) entry types
NpSliceEntry = Union[int, np.integer, None]
ExtSliceEntry = Union[NpSliceEntry, float, complex, str]


def idx_for_value(value: Union[int, float, complex], param_vals: ndarray) -> int:
    location = int(np.abs(param_vals - value).argmin())
    selected_value = param_vals[location]

    if cmath.isclose(param_vals[location], value):
        return location

    if not settings.FUZZY_SLICING:
        raise ValueError(
            "No matching entry for parameter value {} in the array.".format(value)
        )

    if not cmath.isclose(selected_value, value) and settings.FUZZY_WARNING:
        warnings.warn_explicit(
            "Using fuzzy value based indexing: selected value is {}".format(
                selected_value
            ),
            UserWarning,
            "",
            location,
        )
    return location


def convert_to_std_npindex(
    index_tuple: ExtIndexTuple, parameters: "Parameters"
) -> NpIndexTuple:
    """
    converts name-based and value-based indexing to standard numpy indexing

    Parameters
    ----------
    index_tuple:
        the indexing object to be converted
    parameters:
        records the parameters associated with the indices

    Returns
    -------
        standard numpy index tuple
    """
    extindex_obj_tuple = tuple(
        ExtIndexObject(entry, parameters, slot=slot_index)
        for slot_index, entry in enumerate(index_tuple)
    )
    np_indices = ExtIndexTupleObject(extindex_obj_tuple).convert_to_np_index_exp()
    return np_indices


def process_ellipsis(
    array: Union["Parameters", np.ndarray], multi_idx: NpIndexTuple
) -> NpIndexTupleNoEllipsis:
    """
    Removes `...` from the multi-index by explicit slicing.

    Parameters
    ----------
    array:
        numpy array
    multi_idx:
        numpy slicing multi-index that MUST contain an instance of `...` (Ellipsis)

    Returns
    -------
        Processed multi-index not containing any `...`
    """
    new_multi_idx: List[NpIndexNoEllipsis] = [
        slice(None, None, None)
    ] * array.ndim  # type:ignore
    # Replace the slice(None, None, None) entries, starting from beginning until
    # Ellipsis is encountered
    slot = 0
    while multi_idx[slot] is not Ellipsis:
        new_multi_idx[slot] = multi_idx[slot]
        slot += 1
    slot = -1
    # Replace the slice(None, None, None) entries, now starting from the end until
    # Ellipsis is encountered
    while multi_idx[slot] is not Ellipsis:
        new_multi_idx[slot] = multi_idx[slot]
        slot -= 1
    return tuple(new_multi_idx)


class ExtIndexObject:
    """Object used for enabling enhanced indexing in NamedSlotsNdarray. Handles a
    single idx_entry in multi-index"""

    def __init__(
        self, idx_entry: ExtIndex, parameters: "Parameters", slot: Optional[int] = None
    ) -> None:
        self.idx_entry = idx_entry
        self._parameters = parameters
        self.slot = slot
        self.name: Optional[str] = None
        self.type, self.std_idx_entry = self.convert_to_np_idx_entry(idx_entry)

    def convert_to_np_slice_entry(self, slice_entry: ExtSliceEntry) -> NpSliceEntry:
        """Handles value-based slices, converting a float or complex value based
        entry into the corresponding position-based entry"""
        if isinstance(slice_entry, (int, np.integer)):
            return slice_entry
        if slice_entry is None:
            return None
        if isinstance(slice_entry, (float, complex)):
            assert isinstance(self.name, str)
            return idx_for_value(
                slice_entry, self._parameters.paramvals_by_name[self.name]
            )

        raise TypeError("Invalid slice entry: {}".format(slice_entry))

    def convert_to_np_idx_entry(self, idx_entry: ExtIndex) -> Tuple[str, NpIndex]:
        """Convert a generalized multi-index entry into a valid numpy multi-index entry,
        and returns that along with a str recording the idx_entry type"""
        if isinstance(idx_entry, (int, np.integer)):
            return "int", idx_entry

        if idx_entry is Ellipsis:
            return "ellipsis", idx_entry

        if isinstance(idx_entry, (tuple, list)):
            return "tuple", idx_entry

        if isinstance(idx_entry, (float, complex)):
            return "val", idx_for_value(idx_entry, self._parameters[self.slot])

        # slice(<str>, ...):  handle str based slices
        if isinstance(idx_entry, slice) and isinstance(idx_entry.start, str):
            self.name = idx_entry.start

            start = self.convert_to_np_slice_entry(idx_entry.stop)
            if isinstance(start, (complex, float)):
                start = idx_for_value(start, self._parameters[self.slot])

            stop = self.convert_to_np_slice_entry(idx_entry.step)
            if isinstance(stop, (complex, float)):
                stop = idx_for_value(stop, self._parameters[self.slot])

            if isinstance(start, (int, np.integer)) and (stop is None):
                return "slice.name", start
            return "slice.name", slice(start, stop, None)

        # slice(<Number> or <None>, ...):  handle slices with value-based entries
        if isinstance(idx_entry, slice):
            start = self.convert_to_np_slice_entry(idx_entry.start)
            stop = self.convert_to_np_slice_entry(idx_entry.stop)
            if idx_entry.step is None or isinstance(idx_entry.step, (int, np.integer)):
                step = self.convert_to_np_slice_entry(idx_entry.step)
            else:
                raise TypeError(
                    "slice.step can only be int or None. Found {} "
                    "instead.".format(idx_entry.step)
                )
            return "slice", slice(start, stop, step)

        raise TypeError("Invalid index: {}".format(idx_entry))


class ExtIndexTupleObject:
    def __init__(self, extindex_tuple: Tuple[ExtIndexObject, ...]):
        self._parameters = extindex_tuple[0]._parameters
        self.slot_count = len(self._parameters)
        self.extindex_tuple = extindex_tuple

    def _name_based_to_np_index_exp(self) -> NpIndexTuple:
        """Converts a name-based multi-index into a standard numpy index_exp."""
        converted_multi_index: List[NpIndex] = [
            slice(None, None, None)
        ] * self.slot_count

        for extindex_object in self.extindex_tuple:
            if extindex_object.type != "slice.name":
                raise TypeError("If one index is name-based, all indices must be.")
            assert (
                extindex_object.name is not None
            ), "Internal error in NamedSlotsNdarray: index missing `name` attribute!"
            slot_index = self._parameters.index_by_name[
                extindex_object.name
            ]  # type:ignore
            assert isinstance(slot_index, int), "Internal NamedSlotsNdarray error"
            converted_multi_index[slot_index] = extindex_object.std_idx_entry
        return tuple(converted_multi_index)

    def convert_to_np_index_exp(self) -> NpIndexTuple:
        """Takes an extended-syntax multi-index entry and converts it to a standard
        position-based multi-index_entry with only integer-valued indices."""
        # inspect first index_entry to determine whether multi-index entry is name-based
        first_extindex = self.extindex_tuple[0]

        if first_extindex.type == "slice.name":  # if one is name based, all must be
            return self._name_based_to_np_index_exp()

        return tuple(extindex.std_idx_entry for extindex in self.extindex_tuple)


class Parameters:
    """Convenience class for maintaining multiple parameter sets: names and values of
    each parameter set, along with an ordering among sets.
    Used in ParameterSweep as `._parameters`. Can access in several ways:
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
        if isinstance(key, (int, np.integer)):
            return self.paramvals_by_name[self.paramnames_list[key]]
        if key is Ellipsis:
            key = slice(None, None, None)
        if isinstance(key, slice):
            sliced_paramnames_list = self.paramnames_list[key]
            return [self.paramvals_by_name[name] for name in sliced_paramnames_list]
        if isinstance(key, (tuple, list)):
            if not isinstance(key, list) and Ellipsis in key:
                key = process_ellipsis(self, key)
            return [
                self.paramvals_by_name[self.paramnames_list[index]][key[index]]
                for index in range(len(self))
            ]

    def __len__(self):
        return len(self.paramnames_list)

    def __iter__(self):
        return iter(self.paramvals_list)

    def ndim(self):
        # Alias to support numpy's ndim
        return len(self.paramnames_list)

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
        """Return a list of range objects suitable for looping over each parameter
        set"""
        return [range(count) for count in self.counts]

    @property
    def paramvals_list(self) -> List[ndarray]:
        """Return list of all parameter values sets"""
        return [self.paramvals_by_name[name] for name in self.paramnames_list]

    @property
    def counts(self) -> Tuple[int, ...]:
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
            # need to reformat as array of single-entry arrays
            fixed_values_list = [np.asarray(value) for value in fixed_values]
        else:
            fixed_values_list = [
                np.asarray([self[name][0]]) for name in fixed_parametername_list
            ]

        reduced_paramvals_by_name = {name: self[name] for name in self.paramnames_list}
        for index, name in enumerate(fixed_parametername_list):
            reduced_paramvals_by_name[name] = fixed_values_list[index]
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
        new_paramvals_list = self.paramvals_list.copy()
        if not isinstance(np_indices, tuple):
            np_indices = (np_indices,)
        for index, np_index in enumerate(np_indices):
            array_entry = new_paramvals_list[index][np_index]
            if isinstance(array_entry, numbers.Number):
                array_entry = np.asarray([array_entry])
            new_paramvals_list[index] = array_entry

        if not remove_fixed:
            paramvals_by_name = {
                name: new_paramvals_list[index]
                for index, name in enumerate(self.paramnames_list)
            }
            return Parameters(paramvals_by_name)

        reduced_paramvals_by_name = {}
        for index, name in enumerate(self.paramnames_list):
            paramvals = new_paramvals_list[index]
            # Keep all parameters intact that still have more than one element.
            if len(paramvals) > 1:
                reduced_paramvals_by_name[name] = paramvals
            # If only one element is left, check whether this reduction was caused
            # by explicit reduction through slicing. If not, then the single-element
            # parameter was there previously, and we will not reduce it in order
            # to support NamedSlotsNdarray's with axes containing only one element.
            elif index >= len(np_indices):
                reduced_paramvals_by_name[name] = paramvals
            elif np_indices[index] == slice(None, None, None):
                reduced_paramvals_by_name[name] = paramvals

        return Parameters(reduced_paramvals_by_name)


class NamedSlotsNdarray(np.ndarray, Serializable):
    """
    This class implements multi-dimensional arrays, for which the leading M dimensions
    are each associated with a slot name and a corresponding array of slot
    values (float or complex or str). All standard slicing of the multi-dimensional
    array with integer-valued indices is supported as usual, e.g.::

        some_array[0, 3:-1, -4, ::2]

    Slicing of the multi-dimensional array associated with named sets of values is
    extended in two ways:

    (1) Value-based slicing
    Integer indices other than the `step` index may be
    replaced by a float or a complex number or a str. This prompts a lookup and
    substitution by the integer index representing the location of the closest
    element (as measured by the absolute value of the difference for numbers,
    and an exact match for str) in the set of slot values.

    As an example, consider the situation of two named value sets::

        values_by_slotname = {'param1': np.asarray([-4.4, -0.1, 0.3, 10.0]),
                              'param2': np.asarray([0.1*1j, 3.0 - 4.0*1j, 25.0])}

    Then, the following are examples of value-based slicing::

        some_array[0.25, 0:2]                   -->     some_array[2, 0:2]
        some_array[-3.0, 0.0:(2.0 - 4.0*1j)]    -->     some_array[0, 0:1]


    (2) Name-based slicing
    Sometimes, it is convenient to refer to one of the slots
    by its name rather than its position within the multiple sets. As an example, let::

        values_by_slotname = {'ng': np.asarray([-0.1, 0.0, 0.1, 0.2]),
                             'flux': np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])}

    If we are interested in the slice of `some_array` obtained by setting 'flux' to a
    value or the value associated with a given index, we can now use::

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

    parameters: Parameters

    def __new__(
        cls, input_array: np.ndarray, values_by_name: Dict[str, ndarray]
    ) -> "NamedSlotsNdarray":
        implied_shape = tuple(len(values) for values in values_by_name.values())
        if input_array.shape[0 : len(values_by_name)] != implied_shape:
            raise ValueError(
                "Given input array {} with shape {} not compatible with "
                "provided dict calling for shape {}. values_by_name: {}".format(
                    input_array, input_array.shape, implied_shape, values_by_name
                )
            )
        obj = np.asarray(input_array).view(cls)
        obj._parameters = Parameters(values_by_name)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._parameters = getattr(obj, "_parameters", None)

    def __getitem__(self, multi_index: ExtIndices) -> Any:
        """Overwrites the magic method for element selection and slicing to support
        extended string and value based slicing."""
        multi_index_std = np.index_exp[multi_index]  # convert to standard tuple form
        try:
            obj = super().__getitem__(multi_index_std)
            # This attempt fails if multi-index is string- or value-based
        except (TypeError, IndexError):
            multi_index_std = convert_to_std_npindex(multi_index_std, self._parameters)
            obj = super().__getitem__(multi_index_std)

        if Ellipsis in multi_index_std:
            multi_index_std = process_ellipsis(self, multi_index_std)
        # If the resulting obj is a sliced view of the current array, then we must
        # adjust the internal Parameters instance accordingly
        if isinstance(obj, NamedSlotsNdarray):
            param_count = len(self._parameters)
            dummy_array = np.empty(shape=self._parameters.counts)
            # Check whether all parameters are getting fixed; if not, adjust
            # Parameters for the new object
            if not isinstance(dummy_array[multi_index_std[:param_count]], float):
                # have not reduced to one element
                obj._parameters = self._parameters.create_sliced(
                    multi_index_std[:param_count]
                )
            elif (
                obj._parameters.paramvals_by_name == self._parameters.paramvals_by_name
            ):
                # Have reduced to one element, which is still an array however. If this
                # was a regular ndarray (not NamedSlotsNdarray), the Parameters entry
                # would be the same as in parent array. Need to delete this, i.e., just
                # return ordinary ndarray.
                obj = obj.view(ndarray)
        return obj

    def __reduce__(self):
        # needed for multiprocessing / proper pickling
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self._parameters,)
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for multiprocessing / proper pickling
        self._parameters = state[-1]
        super().__setstate__(state[0:-1])

    @classmethod
    def deserialize(cls, io_data: IOData) -> "NamedSlotsNdarray":
        """
        Take the given IOData and return an instance of the described class, initialized
        with the data stored in io_data.
        """
        if "input_array" in io_data.ndarrays:
            input_array = io_data.ndarrays["input_array"]
        else:
            list_data = io_data.objects["input_array"]
            nested_list_shape = utils.get_shape(list_data)
            input_array = np.empty(nested_list_shape, dtype=object)
            input_array[:] = list_data
        values_by_name = io_data.objects["values_by_name"]
        return NamedSlotsNdarray(input_array, values_by_name)

    def serialize(self) -> IOData:
        """
        Convert the content of the current class instance into IOData format.
        """
        import scqubits.io_utils.fileio as io

        typename = "NamedSlotsNdarray"
        io_attributes = None
        if self.dtype in [np.float_, np.complex_, np.int_]:
            io_ndarrays: Optional[Dict[str, ndarray]] = {
                "input_array": self.view(np.ndarray)
            }
            objects = {"values_by_name": self._parameters.paramvals_by_name}
        else:
            io_ndarrays = None
            objects = {
                "values_by_name": self._parameters.paramvals_by_name,
                "input_array": self[:].tolist(),
            }
        return io.IOData(typename, io_attributes, io_ndarrays, objects=objects)

    @property
    def slot_count(self) -> int:
        return len(self._parameters.paramvals_by_name)

    @rc_context(settings.matplotlib_settings)
    def plot(self, **kwargs) -> Tuple[Figure, Axes]:
        if len(self._parameters) != 1:
            raise ValueError(
                "Plotting of NamedSlotNdarray only supported for a "
                "one-dimensional parameter sweep. (Consider slicing.)"
            )
        return plot.data_vs_paramvals(
            xdata=self._parameters.paramvals_list[0],
            ydata=self,
            xlabel=self._parameters.names[0],
            **kwargs
        )

    @property
    def param_info(self) -> Dict[str, ndarray]:
        return self._parameters.paramvals_by_name

    def recast(self) -> "NamedSlotsNdarray":
        return NamedSlotsNdarray(
            np.asarray(self[:].tolist()), self._parameters.paramvals_by_name
        )

    def toarray(self) -> ndarray:
        return self.view(ndarray)
