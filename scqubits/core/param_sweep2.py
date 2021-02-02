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
import weakref
from abc import ABC
from collections import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING, Tuple, Union

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


class NamedSlotsSliceable:
    """
    This mixin class applies to multi-dimensional arrays, for which the leading M dimensions are each associated with
    a slot name and a corresponding array of slot values (float or complex or str). All standard slicing of the
    multi-dimensional array with integer-valued indices is supported as usual, e.g.

        some_array[0, 3:-1, -4, ::2]

    Slicing of the multi-dimensional array associated with named sets of values is extended in two ways:

    (1) Value-based slicing
    Integer indices other than the `step` index may be replaced by a float or a complex number or a str. This prompts a
    lookup and substitution by the integer index representing the location of the closest element (as measured by the
    absolute value of the difference for numbers, and an exact match for str) in the set of slot values.

    As an example, consider the situation of two named value sets

        values_by_slotname = {'param1': np.asarray([-4.4, -0.1, 0.3, 10.0]),
                              'param2': np.asarray([0.1*1j, 3.0 - 4.0*1j, 25.0])}

    Then, the following are examples of value-based slicing:

        some_array[0.25, 0:2]                   -->     some_array[2, 0:2]
        some_array[-3.0, 0.0:(2.0 - 4.0*1j)]    -->     some_array[0, 0:1]


    (2) Name-based slicing
    Sometimes, it is convenient to refer to one of the slots by its name rather than its position within
    the multiple sets. As an example, let

        values_by_slotname = {'ng': np.asarray([-0.1, 0.0, 0.1, 0.2]),
                             'flux': np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])}

    If we are interested in the slice of `some_array` obtained by setting 'flux' to a value or the value associated with
    a given index, we can now use:

        some_array['flux':0.5]            -->    some_array[:, 1]
        some_array['flux'::2, 'ng':-1]    -->    some_array[-1, :2]

    Name-based slicing has the format `<name str>:start:stop`  where `start` and `stop` may be integers or make use of
    value-based slicing. Note: the `step` option is not available in name-based slicing. Name-based and
    standard position-based slicing cannot be combined: `some_array['name1':3, 2:4]` is not supported. For such mixed-
    mode slicing, use several stages of slicing as in `some_array['name1':3][2:4]`.

    A special treatment is reserved for a pure string entry in position 0: this string will be directly converted into
    an index via the corresponding values_by_slotindex.
    """
    values_by_slotname: Dict[str, Iterable]
    values_by_slotindex: Dict[int, Iterable]
    slotname_by_slotindex: Dict[str, int]
    slotindex_by_slotname: Dict[int, str]
    data_callback: Union[ndarray, Callable]

    def __getitem__(self, multi_index: Union[Number, Tuple[Union[Number, slice], ...]]) -> Any:
        """Overwrites the magic method for element selection and slicing to support the extended slicing options."""
        if not isinstance(multi_index, tuple):
            multi_index = (multi_index,)

        if callable(self.data_callback):
            return self.data_callback(self.convert_to_standard_multi_index(multi_index))
        return self.data_callback[self.convert_to_standard_multi_index(multi_index)]

    def convert_slotnames_to_indices(self,
                                     multi_index: Tuple[Union[Number, slice], ...]
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
                raise Exception("Slicing error: could not convert slot-name based slices to ordinary slices.")
        return tuple(converted_multi_index)

    def convert_to_standard_multi_index(self,
                                        multi_index: Tuple[Union[Number, slice], ...]
                                        ) -> Tuple[Union[int, slice], ...]:
        """Takes an extended-syntax multi-index_entry and converts it to a standard position-based multi-index_entry
        with only integer-valued indices."""
        index_entry = multi_index[0]  # inspect first index_entry to determine whether multi-index_entry is name-based
        if isinstance(index_entry, slice) and isinstance(index_entry.start, str):
            # Multi-index_entry is name-based (slices with a str as the start attribute)
            # -> convert to position-based multi-index_entry
            multi_index = self.convert_slotnames_to_indices(multi_index)
            processed_multi_index = [slice(None)] * self.slot_count
        else:
            # Multi-index_entry is position based (nothing to do, just convert to list further to be processed)
            processed_multi_index = list(multi_index)

        # Check for value-based indices and slice entries, and convert to standard indices
        for position, index_entry in enumerate(multi_index):
            if isinstance(index_entry, int):
                # all int entries are taken as standard indices
                processed_multi_index[position] = index_entry
            elif isinstance(index_entry, (float, complex)):
                # individual value-based index_entry
                value = index_entry
                processed_multi_index[position] = self.get_index_closest_value(value,
                                                                               self.values_by_slotindex[position])
            elif isinstance(index_entry, str) and position == 0:
                processed_multi_index[position] = np.where(self.values_by_slotindex[0] == index_entry)[0]
            elif isinstance(index_entry, slice):
                # slice objects must be checked for internal value-based entries in start and stop attributes
                this_slice = (index_entry if self.is_standard_slice(index_entry)
                              else self.convert_to_standard_slice(index_entry, position))
                processed_multi_index[position] = this_slice
        return tuple(processed_multi_index)

    def is_standard_slice(self, this_slice: slice) -> bool:
        """Checks whether slice is standard, i.e., all entries for start, stop, and step are integers."""
        if (isinstance(this_slice.start, (int, type(None))) and
                isinstance(this_slice.stop, (int, type(None))) and
                isinstance(this_slice.step, (int, type(None)))):
            return True
        return False

    def convert_to_standard_slice(self, this_slice: slice, range_index) -> slice:
        """Takes a single slice object that includes value-based entries and converts them into
        integer indices reflecting the position of the closest element in the given value set."""
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

    def get_index_closest_value(self, value: Number, slot_index: int) -> int:
        location = np.abs(self.values_by_slotindex[slot_index] - value).argmin()
        return location


class NamedSlotsArray(np.ndarray, NamedSlotsSliceable):
    def __new__(cls, input_array: np.ndarray, values_by_name: Dict[str, Iterable]):
        obj = np.asarray(input_array).view(cls)
        obj.values_by_slotname = values_by_name
        obj.slotindex_by_slotname = {name: index for index, name in enumerate(values_by_name.keys())}
        obj.slotname_by_slotindex = {index: name for name, index in obj.slotindex_by_slotname.items()}
        obj.values_by_slotindex = {obj.slotindex_by_slotname[name]: values_by_name[name]
                                   for name in values_by_name.keys()}
        obj.data_callback = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.values_by_name = getattr(obj, 'values_by_slotname', None)
        self.slotindex_by_slotname = getattr(obj, 'slotindex_by_slotname', None)
        self.slotname_by_slotindex = getattr(obj, 'slotname_by_slotindex', None)
        self.values_by_index = getattr(obj, 'values_by_slotindex', None)
        self.data_callback = getattr(obj, 'data_callback', None)

    def __getitem__(self, multi_index: Union[Number, Tuple[Union[Number, slice], ...]]) -> Any:
        """Overwrites the magic method for element selection and slicing to support the extended slicing options."""
        if not isinstance(multi_index, tuple):
            multi_index = (multi_index,)
        return super().__getitem__(self.convert_to_standard_multi_index(multi_index))


class Sweep1(Mapping):
    def __init__(self,
                 hilbertspace: HilbertSpace,
                 paramvals_by_name: Dict[str, ndarray],
                 sweep_generators: Dict[str, Callable],
                 evals_count=6,
                 subsys_update_info: Dict[str, List[QuantumSys]] = None,
                 generate_spectrum_lookups=True,
                 autorun=settings.AUTORUN_SWEEP
                 ):
        # self.values_by_slotname = paramvals_by_name
        # self.slotindex_by_slotname = {name: index for index, name in enumerate(self.values_by_slotname.keys())}
        # self.slotname_by_slotindex = {index: name for name, index in self.slotindex_by_slotname.items()}
        # self.values_by_slotindex = {self.slotindex_by_slotname[name]: self.values_by_slotname[name]
        #                             for name in self.values_by_slotname.keys()}

        # loop_order = loop_order or list(paramvals_by_name.keys())
        self._paramvals_by_name = paramvals_by_name   # OrderedDict([(name, paramvals_by_name[name]) for name in loop_order])
        self._paramvals_count_by_name = {name: len(paramvals_by_name[name]) for name in paramvals_by_name.keys()}
        self._hilbertspace = hilbertspace
        self._system_info = hilbertspace.__str__()
        self._sweep_generators = sweep_generators
        self._generator_info = str(list(sweep_generators.keys()))
        self._evals_count = evals_count
        self._subsys_update_info = subsys_update_info
        # self._shape = tuple([len(value) for value in self.values_by_slotindex.values()])
        self._spectrum_lookup = None
        self._data: Dict[str, Optional[NamedSlotsArray]]
        self._data['spec_lookup'] = None

        if autorun:
            self.generate_sweeps()

    def set_loop_order(self):
        performance_cost = {scq.FullZeroPi: 50,
                            scq.ZeroPi: 20,
                            scq.FluxQubit: 20,
                            scq.Fluxonium: 4,
                            scq.TunableTransmon: 2,
                            scq.Transmon: 2,
                            scq.Oscillator: 1}

        def nested_loop_cost(paramnames_ordering):
            outermost_paramname = paramnames_ordering[0]
            outermost_iterations = len(self._paramvals_by_name[outermost_paramname])
            cost_per_iteration = sum([performance_cost[type(subsys)]
                                      for subsys in self._update_subsys_list[outermost_paramname]])
            if len(paramnames_ordering) == 1:
                return outermost_iterations * cost_per_iteration
            return outermost_iterations * (cost_per_iteration + nested_loop_cost(paramnames_ordering[1:]))

        paramname_permutations = itertools.permutations(list(self._paramvals_by_name.keys()))
        loop_costs = [(paramname_list, nested_loop_cost(paramname_list)) for paramname_list in paramname_permutations]
        loop_order = sorted(loop_costs, key=lambda item: item[1])[0][0]
        return loop_order

    def recursive_loops(self,
                        paramvals_list: List[Iterable],
                        sweep_name: str,
                        sweep_func: Callable,
                        param_indices: Tuple[int, ...] = tuple(),
                        param_values: Tuple[Iterable, ...] = tuple()
                        ) -> None:
        loop_count = len(paramvals_list)
        if loop_count == 1:
            for index, value in enumerate(paramvals_list[0]):
                self._data[sweep_name] = sweep_func(param_indices + (index,),
                                                    param_values + (value,),
                                                    self._data['spec_lookup'])
        else:
            for index, value in enumerate(paramvals_list[0]):
                recursive_loops(paramvals_list[1:],
                                sweep_name,
                                sweep_func,
                                param_indices + (index,),
                                param_values + (value,))

    def generate_sweeps(self):
        if self._subsys_update_info is None:
            loop_order = list(self._paramvals_by_name.keys())
        else:
            loop_order = self.set_loop_order()
        self._data['evals'], self._data['evecs'] = self.generate_eigensystem_sweep()
        if spec_lookup:
            self._data['spectrum_lookup'] = self.generate_spectrumlookup_sweep()
        for sweep_name, sweep_generator in self._sweep_generators.items():
            self._data[sweep_name] = self.custom_sweep(sweep_generator)

    def custom_sweep(self, sweep_generator: Callable):
        pass

    def eigensystem_sweep(self):
        pass

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        if isinstance(key, tuple):
            return self._data[key[0]][key[1:]]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class Sweep(NamedSlotsSliceable):
    def __init__(self,
                 paramvals_by_name: Dict[str, Iterable],
                 sweep_generators: Dict[str, Callable],
                 system_info: str,
                 evals_count=6,
                 autorun=True
                 ):
        self.values_by_slotname = {'data': ['evals', 'evecs', 'spec_lookup'] + list(sweep_generators.keys())}
        self.values_by_slotname.update(paramvals_by_name)
        self.slotindex_by_slotname = {name: index for index, name in enumerate(self.values_by_slotname.keys())}
        self.slotname_by_slotindex = {index: name for name, index in self.slotindex_by_slotname.items()}
        self.values_by_slotindex = {self.slotindex_by_slotname[name]: self.values_by_slotname[name]
                                    for name in self.values_by_slotname.keys()}

        self._system_info = system_info
        self._generator_info = list(sweep_generators.keys())
        self._shape = tuple([len(value) for value in self.values_by_slotindex.values()])
        self._data_array = np.empty(shape=self._shape, dtype=object)
        self.data_callback = self.return_sliced_data

    @classmethod
    def create_from_data(cls,
                         paramvals_by_name: Dict[str, Iterable],
                         data_by_name: Dict[str, Iterable],
                         system_info: str
                         ):
        # shape = [len(data_by_name)] + [len(paramvals) for paramvals in paramvals_by_name.values()]
        evals_count = data_by_name['evals'].shape[-1]
        sweep_generators = {name: None for name in data_by_name.keys()}
        new_sweep = Sweep(paramvals_by_name, sweep_generators, system_info, evals_count=evals_count, autorun=False)
        new_sweep._data_array = np.empty(shape=new_sweep._shape, dtype=object)

    def __str__(self):
        info = "Sweep\n" \
               "=========================\n\n" \
               "This sweep was generated as a scan over the following parameter(s): {}\n" \
               "The arrays specifying the parameter-value sets can be accessed via\n" \
               "    <ParameterSweep obj>['<label>']['<parameter name>']\n" \
               "Custom data was computed with the generating function:\n\n" \
               "{}\n\n" \
               "The sweep is based on the following composite quantum system:\n" \
               "{}".format(str(list(self.values_by_slotname.keys())),
                           self.generator_info,
                           self.system_info)
        return info

    def return_sliced_data(self, multi_index: Tuple[Union[int, slice], ...]) -> 'Sweep':
        sliced_paramvals_by_name = {self.slotname_by_slotindex[index]: self.values_by_slotindex[index][this_slice]
                                    for index, this_slice in enumerate(multi_index[:self.slot_count])}
        return Sweep(paramvals_by_name=sliced_paramvals_by_name,
                     eigenenergies=self.eigenenergies[multi_index],
                     eigenstates=self.eigenstates[multi_index],
                     spectrum_lookup_data=self.spectrum_lookup_data[multi_index],
                     custom_data=self.custom_data)
        return sliced_data

    # VISUALIZATION METHODS MAY WANT TO GO HERE


# class SweepInfo:
#     def __init__(self,
#                  label: str,
#                  system_info: str,
#                  evals_count: int,
#                  values_by_slotname: Dict[str, Iterable],
#                  sweep_generator: Union[str, Callable[[Union[float, ndarray]], Dict[str, Union[Number, ndarray]]]],
#                  fixed_subsys_list: List[QuantumSys] = None,
#                  num_cpus: int = settings.NUM_CPUS
#                  ):
#         self.label = label
#         self.system_info = system_info
#         self.evals_count = evals_count
#         self.values_by_slotname = values_by_slotname
#         self.fixed_subsys_list = fixed_subsys_list
#         self.num_cpus = num_cpus
#
#         self.generator_info = sweep_generator if isinstance(sweep_generator, str) else inspect.getsource(sweep_generator)
#
#         if not callable(sweep_generator):
#             self.sweep_generator = self.create_generator(sweep_generator)
#         else:
#             self.sweep_generator = sweep_generator
#


class ParameterSweepBase(ABC):
    """
    The ParameterSweepBase class is an abstract base class for ParameterSweep and StoredSweep
    """
    hilbertspace = descriptors.ReadOnlyProperty()
    _sweeps: Dict[str, Sweep] = {}

    def __getitem__(self, name: str) -> Sweep:
        return self._sweeps[name]

    def __setitem__(self, key: str, item):
        raise NotImplementedError

    def __delitem__(self, name: str) -> None:
        del self._sweeps[name]

    def get_subsys(self, index: int) -> QuantumSys:
        return self._hilbertspace[index]

    def get_subsys_index(self, subsys: QuantumSys) -> int:
        return self._hilbertspace.get_subsys_index(subsys)

    def system_info(self) -> str:
        return self.hilbertspace.__str__()

    @property
    def osc_subsys_list(self) -> List[Tuple[int, Oscillator]]:
        return self._hilbertspace.osc_subsys_list

    @property
    def qbt_subsys_list(self) -> List[Tuple[int, QubitBaseClass]]:
        return self._hilbertspace.qbt_subsys_list

    @property
    def subsystem_count(self) -> int:
        return self._hilbertspace.subsystem_count


class ParameterSweep(ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable):
    """
    The ParameterSweep class helps generate spectral and associated data for a composite quantum system, when one or
    multiple external parameter(s), such as flux, is/are swept over some given interval of values. Upon initialization,
    these data are calculated and stored internally, so that plots can be generated efficiently. (This is of particular
    use for interactive displays used in the Explorer class.)

    Note: the full interacting Hamiltonian is represented in the bare product basis, i.e. the eigenbasis of the non-
    interacting Hamiltonian consisting only of the individual Hamiltonians of each subsystem. Consequently, for a given
    parameter set, construction invariably requires diagonalization of the individual subsystem Hamiltonians. Since this
    diagonalization can be costly itself (think, e.g., of `FullZeroPi`), significant savings in computation time can be
    achieved by identifying subsystems unaffected by a particular scan. it is of significant benefit for

    Note

    Parameters
    ----------
    param_name:
        name of external parameter to be varied
    param_vals:
        array of parameter values
    evals_count:
        number of eigenvalues and eigenstates to be calculated for the composite Hilbert space
    hilbertspace:
        collects all data specifying the Hilbert space of interest
    subsys_update_list:
        list of subsys_list in the Hilbert space which get modified when the external parameter changes
    update_hilbertspace:
        update_hilbertspace(param_val) specifies how a change in the external parameter affects
        the Hilbert space components
    num_cpus:
        number of CPUS requested for computing the sweep (default value settings.NUM_CPUS)
    """

    def __init__(self,
                 label: str,
                 hilbertspace: hspace.HilbertSpace,
                 evals_count: int,
                 paramvals_by_name: Dict[str, Iterable],
                 sweep_generator: Callable[[Union[float, ndarray]], Dict[str, Union[Number, ndarray]]],
                 fixed_subsys_list: Optional[List[QuantumSys]] = None,
                 num_cpus: int = settings.NUM_CPUS
                 ):
        self._hilbertspace = hilbertspace

        # The following quantities reflect the current sweep being generated
        self.label = label
        self.evals_count = evals_count
        self.paramvals_by_name = paramvals_by_name
        self.sweep_generator = sweep_generator
        self.fixed_subsys_list = fixed_subsys_list
        self.num_cpus = num_cpus

        self._lookup: Union[SpectrumLookup, None] = None
        self._bare_hamiltonian_constant: Qobj

        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        dispatch.CENTRAL_DISPATCH.register('PARAMETERSWEEP_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('HILBERTSPACE_UPDATE', self)

        # generate the spectral data sweep
        if settings.AUTORUN_SWEEP:
            self.run()

    def create_generator(self, generator_string: str):
        raise NotImplementedError

    def add_sweep(self,
                  label: str,
                  evals_count: int,
                  paramvals_by_name: Dict[str, Iterable],
                  sweep_generator: Callable[[Union[float, ndarray]], Dict[str, Union[Number, ndarray]]],
                  fixed_subsys_list: List[QuantumSys] = None,
                  num_cpus: int = settings.NUM_CPUS
                  ) -> None:
        self.__init__(label,
                      self.hilbertspace,
                      evals_count,
                      paramvals_by_name,
                      sweep_generator,
                      fixed_subsys_list,
                      num_cpus)

    def run(self) -> None:
        """Top-level method for generating all parameter sweep data"""
        # self.cause_dispatch()   # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        settings.DISPATCH_ENABLED = False
        bare_specdata_list = self._compute_bare_specdata_sweep()
        dressed_specdata = self._compute_dressed_specdata_sweep(bare_specdata_list)
        self._lookup = spec_lookup.SpectrumLookup(self, dressed_specdata, bare_specdata_list)
        settings.DISPATCH_ENABLED = True

    # HilbertSpace: methods for CentralDispatch ----------------------------------------------------
    # def cause_dispatch(self) -> None:
    #     self.update_hilbertspace(self.param_vals[0])
    #
    # def receive(self, event: str, sender: object, **kwargs) -> None:
    #     """Hook to CENTRAL_DISPATCH. This method is accessed by the global CentralDispatch instance whenever an event
    #     occurs that ParameterSweep is registered for. In reaction to update events, the lookup table is marked as out
    #     of sync.
    #
    #     Parameters
    #     ----------
    #     event:
    #         type of event being received
    #     sender:
    #         identity of sender announcing the event
    #     **kwargs
    #     """
    #     if self._lookup is not None:
    #         if event == 'HILBERTSPACE_UPDATE' and sender is self._hilbertspace:
    #             self._lookup._out_of_sync = True
    #             # print('Lookup table now out of sync')
    #         elif event == 'PARAMETERSWEEP_UPDATE' and sender is self:
    #             self._lookup._out_of_sync = True
    #             # print('Lookup table now out of sync')

    # ParameterSweep: file IO methods ---------------------------------------------------------------
    # @classmethod
    # def deserialize(cls, iodata: 'IOData') -> 'StoredSweep':
    #     """
    #     Take the given IOData and return an instance of the described class, initialized with the data stored in
    #     io_data.
    #
    #     Parameters
    #     ----------
    #     iodata: IOData
    #
    #     Returns
    #     -------
    #     StoredSweep
    #     """
    #     data_dict = iodata.as_kwargs()
    #     lookup = data_dict.pop('_lookup')
    #     data_dict['dressed_specdata'] = lookup._dressed_specdata
    #     data_dict['bare_specdata_list'] = lookup._bare_specdata_list
    #     new_storedsweep = StoredSweep(**data_dict)
    #     new_storedsweep._lookup = lookup
    #     return new_storedsweep

    # def serialize(self) -> 'IOData':
    #     """
    #     Convert the content of the current class instance into IOData format.
    #
    #     Returns
    #     -------
    #     IOData
    #     """
    #     if self._lookup is None:
    #         raise ValueError('Nothing to save - no lookup data has been generated yet.')
    #
    #     initdata = {'param_name': self.param_name,
    #                 'param_vals': self.param_vals,
    #                 'evals_count': self.evals_count,
    #                 'hilbertspace': self._hilbertspace,
    #                 '_lookup': self._lookup}
    #     iodata = serializers.dict_serialize(initdata)
    #     iodata.typename = 'StoredSweep'
    #     return iodata

    # ParameterSweep: private methods for generating the sweep -------------------------------------------------
    def _compute_bare_specdata_sweep(self) -> List[SpectrumData]:
        """
        Pre-calculates all bare spectral data needed for the interactive explorer display.
        """
        bare_eigendata_constant = [self._compute_bare_spectrum_constant()] * self.param_count
        target_map = cpu_switch.get_map_method(self.num_cpus)
        with utils.InfoBar("Parallel compute bare eigensys [num_cpus={}]".format(self.num_cpus), self.num_cpus):
            bare_eigendata_varying = list(
                target_map(self._compute_bare_spectrum_varying,
                           tqdm(self.param_vals, desc='Bare spectra', leave=False, disable=self.tqdm_disabled))
            )
        bare_specdata_list = self._recast_bare_eigendata(bare_eigendata_constant, bare_eigendata_varying)
        del bare_eigendata_constant
        del bare_eigendata_varying
        return bare_specdata_list

    def _compute_dressed_specdata_sweep(self, bare_specdata_list: List[SpectrumData]) -> SpectrumData:
        """
        Calculates and returns all dressed spectral data.
        """
        self._bare_hamiltonian_constant = self._compute_bare_hamiltonian_constant(bare_specdata_list)
        param_indices = range(self.param_count)
        func = functools.partial(self._compute_dressed_eigensystem, bare_specdata_list=bare_specdata_list)
        target_map = cpu_switch.get_map_method(self.num_cpus)

        with utils.InfoBar("Parallel compute dressed eigensys [num_cpus={}]".format(self.num_cpus), self.num_cpus):
            dressed_eigendata = list(target_map(func, tqdm(param_indices, desc='Dressed spectrum', leave=False,
                                                           disable=self.tqdm_disabled)))
        dressed_specdata = self._recast_dressed_eigendata(dressed_eigendata)
        del dressed_eigendata
        return dressed_specdata

    def _recast_bare_eigendata(self,
                               static_eigendata: List[List[Tuple[ndarray, ndarray]]],
                               bare_eigendata: List[List[Tuple[ndarray, ndarray]]]
                               ) -> List[SpectrumData]:
        specdata_list = []
        for index, subsys in enumerate(self._hilbertspace):
            if subsys in self.subsys_update_list:
                eigendata = bare_eigendata
            else:
                eigendata = static_eigendata
            evals_count = subsys.truncated_dim
            dim = subsys.hilbertdim()
            esys_dtype = subsys._evec_dtype

            energy_table = np.empty(shape=(self.param_count, evals_count), dtype=np.float_)
            state_table = np.empty(shape=(self.param_count, dim, evals_count), dtype=esys_dtype)
            for j in range(self.param_count):
                energy_table[j] = eigendata[j][index][0]
                state_table[j] = eigendata[j][index][1]
            specdata_list.append(storage.SpectrumData(energy_table,
                                                      system_params={},
                                                      param_name=self.param_name,
                                                      param_vals=self.param_vals,
                                                      state_table=state_table))
        return specdata_list

    def _recast_dressed_eigendata(self,
                                  dressed_eigendata: List[Tuple[ndarray, QutipEigenstates]]
                                  ) -> SpectrumData:
        evals_count = self.evals_count
        energy_table = np.empty(shape=(self.param_count, evals_count), dtype=np.float_)
        state_table = []  # for dressed states, entries are Qobj
        for j in range(self.param_count):
            energy_table[j] = np.real_if_close(dressed_eigendata[j][0])
            state_table.append(dressed_eigendata[j][1])
        specdata = storage.SpectrumData(energy_table,
                                        system_params={},
                                        param_name=self.param_name,
                                        param_vals=self.param_vals,
                                        state_table=state_table)
        return specdata

    def _compute_bare_hamiltonian_constant(self, bare_specdata_list: List[SpectrumData]) -> Qobj:
        """
        Returns
        -------
            composite Hamiltonian composed of bare Hamiltonians of subsys_list independent of the external parameter
        """
        static_hamiltonian = 0
        for index, subsys in enumerate(self._hilbertspace):
            if subsys not in self.subsys_update_list:
                evals = bare_specdata_list[index].energy_table[0]
                static_hamiltonian += self._hilbertspace.diag_hamiltonian(subsys, evals)
        return static_hamiltonian

    def _compute_bare_hamiltonian_varying(self, bare_specdata_list: List[SpectrumData], param_index: int) -> Qobj:
        """
        Parameters
        ----------
        param_index:
            position index of current value of the external parameter

        Returns
        -------
            composite Hamiltonian consisting of all bare Hamiltonians which depend on the external parameter
        """
        hamiltonian = 0
        for index, subsys in enumerate(self._hilbertspace):
            if subsys in self.subsys_update_list:
                evals = bare_specdata_list[index].energy_table[param_index]
                hamiltonian += self._hilbertspace.diag_hamiltonian(subsys, evals)
        return hamiltonian

    def _compute_bare_spectrum_constant(self) -> List[Tuple[ndarray, ndarray]]:
        """
        Returns
        -------
            eigensystem data for each subsystem that is not affected by a change of the external parameter
        """
        eigendata = []
        for subsys in self._hilbertspace:
            if subsys not in self.subsys_update_list:
                evals_count = subsys.truncated_dim
                eigendata.append(subsys.eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)  # type: ignore
        return eigendata

    def _compute_bare_spectrum_varying(self, param_val: float) -> List[Tuple[ndarray, ndarray]]:
        """
        For given external parameter value obtain the bare eigenspectra of each bare subsystem that is affected by
        changes in the external parameter. Formulated to be used with Pool.map()

        Returns
        -------
            (evals, evecs) bare eigendata for each subsystem that is parameter-dependent
        """
        eigendata = []
        self.update_hilbertspace(param_val)
        for subsys in self._hilbertspace:
            if subsys in self.subsys_update_list:
                evals_count = subsys.truncated_dim
                subsys_index = self._hilbertspace.get_subsys_index(subsys)
                eigendata.append(self._hilbertspace[subsys_index].eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)  # type: ignore
        return eigendata

    def _compute_dressed_eigensystem(self,
                                     param_index: int,
                                     bare_specdata_list: List[SpectrumData]
                                     ) -> Tuple[ndarray, QutipEigenstates]:
        hamiltonian = (self._bare_hamiltonian_constant +
                       self._compute_bare_hamiltonian_varying(bare_specdata_list, param_index))

        for interaction_term in self._hilbertspace.interaction_list:
            evecs1 = self._lookup_bare_eigenstates(param_index, interaction_term.subsys1, bare_specdata_list)
            evecs2 = self._lookup_bare_eigenstates(param_index, interaction_term.subsys2, bare_specdata_list)
            hamiltonian += self._hilbertspace.interactionterm_hamiltonian(interaction_term,
                                                                          evecs1=evecs1, evecs2=evecs2)
        evals, evecs = hamiltonian.eigenstates(eigvals=self.evals_count)
        evecs = evecs.view(qutip_serializer.QutipEigenstates)
        return evals, evecs

    def _lookup_bare_eigenstates(self,
                                 param_index: int,
                                 subsys: QuantumSys,
                                 bare_specdata_list: List[SpectrumData]
                                 ) -> ndarray:
        """
        Parameters
        ----------
        param_index:
            position index of parameter value in question
        subsys:
            Hilbert space subsystem for which bare eigendata is to be looked up
        bare_specdata_list:
            may be provided during partial generation of the lookup

        Returns
        -------
            bare eigenvectors for the specified subsystem and the external parameter fixed to the value indicated by
            its index
        """
        subsys_index = self.get_subsys_index(subsys)
        return bare_specdata_list[subsys_index].state_table[param_index]  # type: ignore


class StoredSweep(ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable):
    param_name = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_vals = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    evals_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    lookup = descriptors.ReadOnlyProperty()

    def __init__(self,
                 param_name: str,
                 param_vals: ndarray,
                 evals_count: int,
                 hilbertspace: HilbertSpace,
                 dressed_specdata: SpectrumData,
                 bare_specdata_list: List[SpectrumData]
                 ) -> None:
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self._hilbertspace = hilbertspace
        self._lookup = spec_lookup.SpectrumLookup(hilbertspace, dressed_specdata, bare_specdata_list, auto_run=False)

    # StoredSweep: file IO methods ---------------------------------------------------------------
    @classmethod
    def deserialize(cls, iodata: 'IOData') -> 'StoredSweep':
        """
        Take the given IOData and return an instance of the described class, initialized with the data stored in
        io_data.

        Parameters
        ----------
        iodata: IOData

        Returns
        -------
        StoredSweep
        """
        data_dict = iodata.as_kwargs()
        lookup = data_dict.pop('_lookup')
        data_dict['dressed_specdata'] = lookup._dressed_specdata
        data_dict['bare_specdata_list'] = lookup._bare_specdata_list
        new_storedsweep = StoredSweep(**data_dict)
        new_storedsweep._lookup = lookup
        new_storedsweep._lookup._hilbertspace = weakref.proxy(new_storedsweep._hilbertspace)
        return new_storedsweep

    # StoredSweep: other methods
    def get_hilbertspace(self) -> HilbertSpace:
        return self._hilbertspace

    def new_sweep(self,
                  subsys_update_list: List[QuantumSys],
                  update_hilbertspace: Callable,
                  num_cpus: int = settings.NUM_CPUS
                  ) -> ParameterSweep:
        return ParameterSweep(
            self.param_name,
            self.param_vals,
            self.evals_count,
            self._hilbertspace,
            subsys_update_list,
            update_hilbertspace,
            num_cpus
        )
