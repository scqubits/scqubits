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
import weakref
from abc import ABC
from collections import Mapping, namedtuple
from typing import Any, Callable, Dict, Iterable, List, TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy import ndarray
from qutip.qobj import Qobj

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


class ReadOnlyDict(Mapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class ParameterSweepBase(ABC):
    """
    The ParameterSweepBase class is an abstract base class for ParameterSweep and StoredSweep
    """
    hilbertspace = descriptors.ReadOnlyProperty()
    _sweeps: Dict[str, SweepData]

    def __init__(self,
                 hilbertspace: hspace.HilbertSpace,
                 sweep_name: str,
                 param_ranges: Dict[str, Iterable],
                 sweep_function: Callable[[Union[float, ndarray]], Union[float, ndarray]]):
        self._hilbertspace = hilbertspace
        self.add_sweep(sweep_name, param_ranges, sweep_function)

    def __getitem__(self, name: str) -> SweepData:
        return self._sweeps[name]

    def __setitem__(self, key: str, item):
        raise NotImplementedError

    def __delitem__(self, name: str) -> None:
        del self._sweeps[name]

    def add_sweep(self,
                  sweep_name: str,
                  param_ranges: Dict[str, Iterable],
                  sweep_function: Callable[[Union[float, ndarray]], Union[float, ndarray]]) -> None:
        self._sweeps[sweep_name] = self.generate_sweep(param_ranges, sweep_function)
        return None

    def get_subsys(self, index: int) -> QuantumSys:
        return self._hilbertspace[index]

    def get_subsys_index(self, subsys: QuantumSys) -> int:
        return self._hilbertspace.get_subsys_index(subsys)

    @property
    def osc_subsys_list(self) -> List[Tuple[int, Oscillator]]:
        return self._hilbertspace.osc_subsys_list

    @property
    def qbt_subsys_list(self) -> List[Tuple[int, QubitBaseClass]]:
        return self._hilbertspace.qbt_subsys_list

    @property
    def subsystem_count(self) -> int:
        return self._hilbertspace.subsystem_count


class NamedRangeSlicing:
    def __init__(self,
                 ranges: Dict[str, Iterable],
                 data_callback: Callable):
        self.range_by_name = ranges
        self.index_by_name = {name: index for index, name in enumerate(ranges.keys())}
        self.name_by_index = {index: name for name, index in self.index_by_name.items()}
        self.range_by_index = {self.index_by_name[name]: ranges[name] for name in ranges.keys()}
        self.data_callback = data_callback

    def __getitem__(self, *multi_index: Union[Number, slice]) -> Any:
        # two options: 1. positional slicing
        #              2. all indices are referred to by parameter name, e.g.,
        #                    sweep['flux1':start:stop, 'ng1':start:stop]
        return self.data_callback(self.process_slicing(multi_index))


    def names_to_indices(self,
                         multi_index: Tuple[Union[Number, slice], ...]
                         ) ->  Tuple[Union[Number, slice], ...]:
        converted_multi_index = [slice(None)] * self.param_count
        for this_slice in multi_index:
            try:
                name = this_slice.start
                converted_slice = slice(start=this_slice.stop, stop=this_slice.step)
                index = self.index_by_name[name]
                converted_multi_index[index] = converted_slice
            except (AttributeError, KeyError):
                raise Exception("Slicing error: could not convert parameter-name based slices to ordinary slices.")
        return tuple(converted_multi_index)

    def process_slicing(self, multi_index: Tuple[Union[Number, slice], ...]) -> Tuple[int]:
        index = multi_index[0]
        if isinstance(index, slice) and isinstance(index.start, str):
            multi_index = self.names_to_indices(multi_index)

        processed_multi_index = [slice(None)] * self.param_count
        for position, index in enumerate(multi_index):
            if isinstance(index, int):
                processed_multi_index[position] = index
            elif isinstance(index, (float, complex)):
                value = index
                processed_multi_index[position] = self.get_index_closest_value(value, self.range_by_index[position])
            elif isinstance(index, slice):
                this_slice = index if self.is_regular_slice(index) else self.convert_to_regular_slice(index, position)
                processed_multi_index[position] = this_slice

        return tuple(processed_multi_index)

    def is_regular_slice(self, this_slice: slice) -> bool:
        if (isinstance(this_slice.start, (int, None)) and
                isinstance(this_slice.stop, (int, None)) and
                isinstance(this_slice.step, (int, None))):
            return True
        return False

    def convert_to_regular_slice(self, this_slice: slice, range_index) -> slice:
        if not isinstance(this_slice.start, (int, None)):
            value = this_slice.start
            this_slice.start = self.get_index_closest_value(value, range_index)
        if not isinstance(this_slice.stop, (int, None)):
            value = this_slice.stop
            this_slice.stop = self.get_index_closest_value(value, range_index)
        if not isinstance(this_slice.step, (int, None)):
            raise Exception("Slicing error: non-integer step sizes not supported.")
        return this_slice

    def params_count(self) -> int:
        return len(self.param_range)


SweepData = namedtuple('SweepData', ['eigenenergies', 'eigenstates', 'lookup_data', 'custom_data'])


class Sweep(NamedRangeSlicing):
    eigenenergies = descriptors.ReadOnlyProperty()
    eigenstates = descriptors.ReadOnlyProperty()
    lookup_data = descriptors.ReadOnlyProperty()
    custom_data = descriptors.ReadOnlyProperty()

    def __init__(self,
                 param_range: Dict[str, Iterable],
                 eigenenergies: ndarray,
                 eigenstates: ndarray,
                 spectrum_lookup_data: ndarray,
                 custom_data: ndarray):

        super().__init__(param_range)
        self.param_range = ReadOnlyDict(param_range)
        self.param_index = ReadOnlyDict({param_name: index for index, param_name in param_range.keys()})
        self.param_name = ReadOnlyDict({index: param_name for index, param_name in param_range.keys()})
        self._eigenenergies = eigenenergies
        self._eigenstates = eigenstates
        self._spectrum_lookup_data = spectrum_lookup_data
        self._custom_data = custom_data

    def return_sweepdata(self, multi_index: Tuple[Union[Int, slice],...]) -> SweepData:
        sliced_data = SweepData(eigenenergies=self.eigenenergies[multi_index],
                                eigenstates=self.eigenstates[multi_index],
                                lookup_data=self.lookup_data[multi_index],
                                custom_data=self.custom_data
                                )

    def __getitem__(self, *multi_index1, **multi_index2):
        if multi_index1 and multi_index2:
            raise NotImplementedError


    def multi_index(self, **fixed_params: Union[int, List[float]]):
        multi_index = np.empty(self.params_count(), dtype=np.int)
        for param_name, value_or_index in fixed_params.items():
            if isinstance(value_or_index, int):
                index = value_or_index
                multi_index[self.param_index[param_name]] = index
            elif isinstance(value_or_index, list):
                value = value_or_index[0]
                index = self.get_index_closest_value(value)
            else:
                raise TypeError



    param_vals = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    evals_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    lookup = descriptors.ReadOnlyProperty()
    _hilbertspace: hspace.HilbertSpace


    @property
    def bare_specdata_list(self) -> List[SpectrumData]:
        return self.lookup._bare_specdata_list

    @property
    def dressed_specdata(self) -> SpectrumData:
        return self.lookup._dressed_specdata

    def _lookup_bare_eigenstates(self,
                                 param_index: int,
                                 subsys: QuantumSys,
                                 bare_specdata_list: List[SpectrumData]
                                 ) -> Union[ndarray, List[QutipEigenstates]]:
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

    @property
    def system_params(self) -> Dict[str, Any]:
        return self._hilbertspace.get_initdata()

    def new_datastore(self, **kwargs) -> DataStore:
        """Return DataStore object with system/sweep information obtained from self."""
        return storage.DataStore(self.system_params, self.param_name, self.param_vals, **kwargs)


class ParameterSweep(ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable):
    """
    The ParameterSweep class helps generate spectral and associated data for a composite quantum system, as an externa,
    parameter, such as flux, is swept over some given interval of values. Upon initialization, these data are calculated
    and stored internally, so that plots can be generated efficiently. This is of particular use for interactive
    displays used in the Explorer class.

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
    param_name = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_vals = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    evals_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    subsys_update_list = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    update_hilbertspace = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    lookup = descriptors.ReadOnlyProperty()

    def __init__(self,
                 param_name: str,
                 param_vals: ndarray,
                 evals_count: int,
                 hilbertspace: HilbertSpace,
                 subsys_update_list: List[QuantumSys],
                 update_hilbertspace: Callable,
                 num_cpus: int = settings.NUM_CPUS
                 ) -> None:
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self._hilbertspace = hilbertspace
        self.subsys_update_list = tuple(subsys_update_list)
        self.update_hilbertspace = update_hilbertspace
        self.num_cpus = num_cpus
        self._lookup: Union[SpectrumLookup, None] = None
        self._bare_hamiltonian_constant: Qobj

        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        dispatch.CENTRAL_DISPATCH.register('PARAMETERSWEEP_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('HILBERTSPACE_UPDATE', self)

        # generate the spectral data sweep
        if settings.AUTORUN_SWEEP:
            self.run()

    def run(self) -> None:
        """Top-level method for generating all parameter sweep data"""
        self.cause_dispatch()   # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        settings.DISPATCH_ENABLED = False
        bare_specdata_list = self._compute_bare_specdata_sweep()
        dressed_specdata = self._compute_dressed_specdata_sweep(bare_specdata_list)
        self._lookup = spec_lookup.SpectrumLookup(self, dressed_specdata, bare_specdata_list)
        settings.DISPATCH_ENABLED = True

    # HilbertSpace: methods for CentralDispatch ----------------------------------------------------
    def cause_dispatch(self) -> None:
        self.update_hilbertspace(self.param_vals[0])

    def receive(self, event: str, sender: object, **kwargs) -> None:
        """Hook to CENTRAL_DISPATCH. This method is accessed by the global CentralDispatch instance whenever an event
        occurs that ParameterSweep is registered for. In reaction to update events, the lookup table is marked as out
        of sync.

        Parameters
        ----------
        event:
            type of event being received
        sender:
            identity of sender announcing the event
        **kwargs
        """
        if self._lookup is not None:
            if event == 'HILBERTSPACE_UPDATE' and sender is self._hilbertspace:
                self._lookup._out_of_sync = True
                # print('Lookup table now out of sync')
            elif event == 'PARAMETERSWEEP_UPDATE' and sender is self:
                self._lookup._out_of_sync = True
                # print('Lookup table now out of sync')

    # ParameterSweep: file IO methods ---------------------------------------------------------------
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
        return new_storedsweep

    def serialize(self) -> 'IOData':
        """
        Convert the content of the current class instance into IOData format.

        Returns
        -------
        IOData
        """
        if self._lookup is None:
            raise ValueError('Nothing to save - no lookup data has been generated yet.')

        initdata = {'param_name': self.param_name,
                    'param_vals': self.param_vals,
                    'evals_count': self.evals_count,
                    'hilbertspace': self._hilbertspace,
                    '_lookup': self._lookup}
        iodata = serializers.dict_serialize(initdata)
        iodata.typename = 'StoredSweep'
        return iodata

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
