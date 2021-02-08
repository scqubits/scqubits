# param_sweep.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import functools
import itertools

from collections import Mapping, OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from numpy import ndarray
from qutip.qobj import Qobj

import scqubits as scq
import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.hilbert_space as hspace
import scqubits.core.spec_lookup as spec_lookup
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_qutip as qutip_serializer
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils

from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.spec_lookup import SpectrumLookup
from scqubits.core.storage import DataStore, SpectrumData
from scqubits.io_utils.fileio_qutip import QutipEigenstates

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


QuantumSys = Union[QubitBaseClass, Oscillator]
Number = Union[int, float, complex]


def _update_subsys_compute_esys(
    update_func: Callable, subsystem: QuantumSys, paramval_tuple: Tuple[float]
) -> ndarray:
    update_func(*paramval_tuple)
    evals, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
    esys_array = np.empty(shape=(2,), dtype=object)
    esys_array[0] = evals
    esys_array[1] = evecs
    return esys_array


def _update_and_compute_dressed_esys(
    hilbertspace: HilbertSpace,
    evals_count: int,
    update_func: Callable,
    paramval_tuple: Tuple[float],
) -> ndarray:
    update_func(*paramval_tuple)
    evals, evecs = hilbertspace.eigensys(evals_count=evals_count)
    esys_array = np.empty(shape=(2,), dtype=object)
    esys_array[0] = evals
    esys_array[1] = evecs
    return esys_array


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
                processed_multi_index[position] = self.get_index_closest_value(
                    value, position
                )
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


class NamedSlotsNdarray(np.ndarray, NamedSliceableSlots):
    def __new__(cls, input_array: np.ndarray, values_by_name: Dict[str, Iterable]):
        implied_shape = tuple(len(values) for name, values in values_by_name.items())
        if input_array.shape != implied_shape:
            raise ValueError("Given input array not compatible with provided dict.")

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


class Parameters:
    def __init__(
        self,
        paramvals_by_name: Dict[str, ndarray],
        paramnames_list: Optional[List[str]] = None,
    ) -> None:
        # This is the internal storage
        self._paramvals_by_name = paramvals_by_name

        # The following list of parameter names sets the ordering among parameter values
        if paramnames_list is not None:
            self._paramnames_list = paramnames_list
        else:
            self._paramnames_list = list(paramvals_by_name.keys())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._paramvals_by_name[key]
        if isinstance(key, int):
            return self._paramvals_by_name[self._paramnames_list[key]]
        if isinstance(key, slice):
            sliced_paramnames_list = self._paramnames_list[key]
            return [self._paramvals_by_name[name] for name in sliced_paramnames_list]

    def __len__(self):
        return len(self._paramnames_list)

    def __iter__(self):
        return iter(self.paramvals_list)

    @property
    def names(self):
        return self._paramnames_list

    @property
    def counts_by_name(self):
        return {
            name: len(self._paramvals_by_name[name])
            for name in self._paramvals_by_name.keys()
        }

    def index_by_name(self, name: str) -> int:
        return self._paramnames_list.index(name)

    @property
    def paramvals_list(self):
        return [self._paramvals_by_name[name] for name in self._paramnames_list]

    @property
    def counts(self):
        return [len(paramvals) for paramvals in self]

    def reorder(self, ordering: Union[List[str], List[int]]):
        if sorted(ordering) == sorted(self._paramnames_list):
            self._paramnames_list = ordering
        elif sorted(ordering) == list(range(len(self))):
            self._paramnames_list = [self._paramnames_list[index] for index in ordering]
        else:
            raise ValueError("Not a valid ordering for parameters.")

    @property
    def ordered_dict(self) -> Dict[str, Iterable]:
        return OrderedDict([(name, self[name]) for name in self.names])

    def create_reduced(self, fixed_parametername_list, fixed_values=None):
        if fixed_values is not None:
            fixed_values = [np.asarray(value) for value in fixed_values]
        else:
            fixed_values = [
                np.asarray([self[name][0]]) for name in fixed_parametername_list
            ]

        reduced_paramvals_by_name = {name: self[name] for name in self._paramnames_list}
        for index, name in enumerate(fixed_parametername_list):
            reduced_paramvals_by_name[name] = fixed_values[index]

        return Parameters(reduced_paramvals_by_name)


class Sweep(Mapping):
    """
    Sweep acts like a dictionary
    {
      'esys': NamedSlotsNdarray,
      'bare_esys': NamedSlotsNdarray,
      'lookup': NamedSlotsNdArray (?),
      '<observable1>': NamedSlotsNdarray,
      '<observable2>': NamedSlotsNdarray,
      ...
    }

    Parameters
    ----------
    hilbertspace:
    paramvals_by_name:
    sweep_generators:
    update_hilbertspace:
        function that updates the associated ``hilbertspace`` object with a given
        set of parameters
    evals_count:
        number of dressed eigenvalues/eigenstates to keep. (The number of bare
        eigenvalues/eigenstates is determined for each subsystem by `truncated_dim`.)
    subsys_update_info:
        To speed up calculations, the user may provide information that specifies which
        subsystems are being updated for each of the given parameter sweeps. This
        information is specified by a dictionary of the following form:
        {
         '<parameter name 1>': [<subsystem a>],
         '<parameter name 2>': [<subsystem b>, <subsystem c>, ...],
          ...
        }
        This indicates that changes in <parameter name 1> only require updates of
        <subsystem a> while leaving other subsystems unchanged. Similarly, sweeping
        <parameter name 2> affects <subsystem b>, <subsystem c> etc.

    generate_spectrum_lookup:
    autorun:
    num_cpus:
        number of CPUS requested for computing the sweep
        (default value settings.NUM_CPUS)
    """

    def __init__(
        self,
        hilbertspace: HilbertSpace,
        paramvals_by_name: Dict[str, ndarray],
        update_hilbertspace: Callable,
        sweep_generators: Optional[Dict[str, Callable]] = None,
        evals_count: int = 6,
        subsys_update_info: Optional[Dict[str, List[QuantumSys]]] = None,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: int = settings.NUM_CPUS,
    ) -> None:

        self.parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._system_info = hilbertspace.__str__()
        self._sweep_generators = sweep_generators
        self._evals_count = evals_count
        self._update_hilbertspace = update_hilbertspace
        self._subsys_update_info = subsys_update_info
        self._data: Dict[str, Optional[NamedSlotsNdarray]] = {}
        self._num_cpus = num_cpus
        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        if autorun:
            self.generate_sweeps()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        if isinstance(key, tuple):
            return self._data[key[0]][key[1:]]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def generate_sweeps(self) -> None:
        self._data["esys_bare"] = self.bare_spectrum_sweep()
        self._data["esys_dressed"] = self.dressed_spectrum_sweep()
        self._data["lookup"] = self.spectrum_lookup_sweep()

        if self._sweep_generators is None:
            return
        for sweep_name, sweep_generator in self._sweep_generators.items():
            self._data[sweep_name] = self.custom_sweep(sweep_generator)

    def bare_spectrum_sweep(
        self,
    ) -> NamedSlotsNdarray:
        """
        The bare energy spectra are computed according to the following scheme.
        1. Perform a loop over all subsystems to separately obtain the bare energy
            eigenvalues and eigenstates for each subsystem.
        2. If `update_subsystem_info` is given, remove those sweeps that leave the
            subsystem fixed.
        3. If self._num_cpus > 1, parallelize.

        Returns
        -------
            NamedSlotsNdarray["subsystem", <paramname1>, <paramname2>, ..., "esys"]
            where "subsystem": 0, 1, ... enumerates subsystems and
            "esys": 0, 1 yields eigenvalues and eigenvectors, respectively
        """
        bare_spectrum = []
        for subsystem in self._hilbertspace:
            bare_spectrum += [self._subsys_bare_spectrum_sweep(subsystem)]
        bare_spectrum = np.asarray(bare_spectrum, dtype=object)

        slotparamvals_by_name = OrderedDict(
            [("subsys", range(len(self._hilbertspace)))]
        )
        slotparamvals_by_name.update(self.parameters.ordered_dict)
        slotparamvals_by_name.update(OrderedDict([("esys", [0, 1])]))

        return NamedSlotsNdarray(bare_spectrum, OrderedDict(slotparamvals_by_name))

    def paramnames_no_subsys_update(self, subsystem) -> List[str]:
        if self._subsys_update_info is None:
            return []
        updating_parameters = [
            name
            for name in self._subsys_update_info.keys()
            if subsystem in self._subsys_update_info[name]
        ]
        return list(set(self.parameters.names) - set(updating_parameters))

    def _subsys_bare_spectrum_sweep(self, subsystem) -> ndarray:
        """

        Parameters
        ----------
        subsystem:
            subsystem for which the bare spectrum sweep is to be computed

        Returns
        -------
            multidimensional array of the format
            array[p1, p2, p3, ..., pN] = np.asarray[[evals, evecs]]
        """
        fixed_paramnames = self.paramnames_no_subsys_update(subsystem)
        reduced_parameters = self.parameters.create_reduced(fixed_paramnames)

        target_map = cpu_switch.get_map_method(self._num_cpus)
        with utils.InfoBar(
            "Parallel compute bare eigensys [num_cpus={}]".format(self._num_cpus),
            self._num_cpus,
        ):
            bare_eigendata = target_map(
                functools.partial(
                    _update_subsys_compute_esys,
                    self._update_hilbertspace,
                    subsystem,
                ),
                tqdm(
                    itertools.product(*reduced_parameters.paramvals_list),
                    desc="Bare spectra",
                    leave=False,
                    disable=self.tqdm_disabled,
                ),
            )
        bare_eigendata = np.asarray(list(bare_eigendata), dtype=object)
        bare_eigendata = bare_eigendata.reshape((*reduced_parameters.counts, 2))

        # Bare spectral data was only computed once for each parameter that has no
        # update effect on the subsystem. Now extend the array to reflect this
        # for the full parameter array by repeating
        for name in fixed_paramnames:
            index = self.parameters.index_by_name(name)
            param_count = self.parameters.counts[index]
            bare_eigendata = np.repeat(bare_eigendata, param_count, axis=index)

        return bare_eigendata

    def dressed_spectrum_sweep(
        self,
    ) -> NamedSlotsNdarray:
        """

        Returns
        -------
            NamedSlotsNdarray[<paramname1>, <paramname2>, ..., "esys"]
            "esys": 0, 1 yields eigenvalues and eigenvectors, respectively
        """
        target_map = cpu_switch.get_map_method(self._num_cpus)
        with utils.InfoBar(
            "Parallel compute dressed eigensys [num_cpus={}]".format(self._num_cpus),
            self._num_cpus,
        ):
            spectrum_data = target_map(
                functools.partial(
                    _update_and_compute_dressed_esys,
                    self._hilbertspace,
                    self._evals_count,
                    self._update_hilbertspace,
                ),
                tqdm(
                    itertools.product(*self.parameters.paramvals_list),
                    desc="Dressed spectrum",
                    leave=False,
                    disable=self.tqdm_disabled,
                ),
            )
        spectrum_data = np.asarray(list(spectrum_data), dtype=object)
        spectrum_data = spectrum_data.reshape((*self.parameters.counts, 2))
        slotparamvals_by_name = self.parameters.ordered_dict
        slotparamvals_by_name.update(OrderedDict([("esys", [0, 1])]))

        return NamedSlotsNdarray(spectrum_data, OrderedDict(slotparamvals_by_name))

    def spectrum_lookup_sweep(self):
        pass

    def custom_sweep(self, sweep_generator: Callable):
        pass

    def add_sweep(self, sweep_name: str, sweep_generator: Callable) -> None:
        pass
