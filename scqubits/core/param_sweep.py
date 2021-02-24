# param_sweep.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
# This source code is licensed under the BSD-style license found in the LICENSE file
# in the root directory of this source tree.
# ###########################################################################

import functools
import itertools
import warnings

from abc import ABC
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from numpy import ndarray

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils

from scqubits.core._param_sweep import _ParameterSweep
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.namedslots_array import GIndexObject, NamedSlotsNdarray, Parameters
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.spectrum_lookup import SpectrumLookupMixin
from scqubits.core.storage import SpectrumData

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


QuantumSys = Union[QubitBaseClass, Oscillator]
Number = Union[int, float, complex]


class ParameterSweepBase(ABC):
    """
    The_ParameterSweepBase class is an abstract base class for ParameterSweep and
    StoredSweep
    """

    parameters = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    _out_of_sync = False
    _current_param_indices: Union[tuple, slice]

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

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]

        # The following enables the following syntax:
        # <Sweep>[p1, p2, ...].dressed_eigenstates()
        if isinstance(key, (tuple, int)):
            self._current_param_indices = key
        elif isinstance(key, slice):
            self._current_param_indices = (key,)
        return self

    def receive(self, event: str, sender: object, **kwargs) -> None:
        """Hook to CENTRAL_DISPATCH. This method is accessed by the global
        CentralDispatch instance whenever an event occurs that ParameterSweep is
        registered for. In reaction to update events, the lookup table is marked as
        out of sync.

        Parameters
        ----------
        event:
            type of event being received
        sender:
            identity of sender announcing the event
        **kwargs
        """
        if "lookup" in self._data:
            if event == "HILBERTSPACE_UPDATE" and sender is self._hilbertspace:
                self._out_of_sync = True
            elif event == "PARAMETERSWEEP_UPDATE" and sender is self:
                self._out_of_sync = True

    @property
    def bare_specdata_list(self) -> "List[SpectrumData]":
        multi_index = self._current_param_indices
        sweep_param_indices = self.get_sweep_indices(multi_index)
        if len(sweep_param_indices) != 1:
            raise ValueError(
                "All but one parameter must be fixed for `bare_specdata_list`."
            )
        sweep_param_name = self.parameters.name_by_index[sweep_param_indices[0]]
        specdata_list: List[SpectrumData] = []
        for subsys_index, subsystem in enumerate(self._hilbertspace):
            evals_swp = self["bare_esys"]["subsys":subsys_index][multi_index]["esys":0]
            evecs_swp = self["bare_esys"]["subsys":subsys_index][multi_index]["esys":1]
            specdata_list.append(
                SpectrumData(
                    energy_table=evals_swp.toarray(),
                    state_table=evecs_swp.toarray(),
                    system_params=self._hilbertspace.get_initdata(),
                    param_name=sweep_param_name,
                    param_vals=self.parameters[sweep_param_name],
                )
            )
        return specdata_list

    @property
    def dressed_specdata(self) -> "List[SpectrumData]":
        multi_index = self._current_param_indices
        sweep_param_indices = self.get_sweep_indices(multi_index)
        if len(sweep_param_indices) != 1:
            raise ValueError(
                "All but one parameter must be fixed for `dressed_specdata`."
            )
        sweep_param_name = self.parameters.name_by_index[sweep_param_indices[0]]

        specdata = SpectrumData(
            energy_table=self["esys"][multi_index + (0,)].toarray(),
            state_table=self["esys"][multi_index + (1,)].toarray(),
            system_params=self._hilbertspace.get_initdata(),
            param_name=sweep_param_name,
            param_vals=self.parameters[sweep_param_name],
        )
        return specdata

    def get_sweep_indices(self, multi_index):
        gidx_obj_tuple = tuple(
            GIndexObject(entry, self.parameters, slot=slot_index)
            for slot_index, entry in enumerate(multi_index)
        )
        std_multi_index = (self._data["esys"])._to_std_index_tuple(gidx_obj_tuple)
        sweep_indices = [
            index
            for index, index_obj in enumerate(std_multi_index)
            if isinstance(
                self.parameters.paramvals_list[index][index_obj], (list, tuple, ndarray)
            )
        ]
        return sweep_indices

    @property
    def system_params(self) -> Dict[str, Any]:
        return self._hilbertspace.get_initdata()


class ParameterSweep(
    ParameterSweepBase,
    SpectrumLookupMixin,
    dispatch.DispatchClient,
    serializers.Serializable,
):
    """
    Sweep allows dict-like and array-like access. For
     <Sweep>[<str>], return data according to:
    {
      'esys': NamedSlotsNdarray of dressed eigenspectrum,
      'bare_esys': NamedSlotsNdarray of bare eigenspectrum,
      'lookup': NamedSlotsNdAdarray of dressed indices correposponding to bare
      product state labels in canonical order,
      '<observable1>': NamedSlotsNdarray,
      '<observable2>': NamedSlotsNdarray,
      ...
    }

    For array-like access (including named slicing allowed for NamedSlotsNdarray),
    enable lookup functionality such as
    <Sweep>[p1, p2, ...].eigensys()

    Parameters
    ----------
    hilbertspace:
        HilbertSpace object describing the quantum system of interest
    paramvals_by_name:
        Dictionary that, for each set of parameter values, specifies a parameter name
        and the set of values to be used in the sweep.
    sweep_generators:
        Dictionary that contains names of custom sweeps and a function computing the
        quantity of interest
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
    bare_only:
        if set to True, only bare eigendata is calculated; useful when performing a
        sweep for a single quantum system, no interaction (default: False)
    autorun:
        Determines whether to directly run the sweep or delay it until `.run()` is
        called manually. (Default: settings.AUTORUN_SWEEP=True)
    num_cpus:
        number of CPUS requested for computing the sweep
        (default value settings.NUM_CPUS)
    """

    def __new__(cls, *args, **kwargs) -> "Union[ParameterSweep, _ParameterSweep]":
        if args and isinstance(args[0], str) or "param_name" in kwargs:
            # old-style ParameterSweep interface is being used
            warnings.warn(
                "The implementation of the `ParameterSweep` class has changed and this "
                "old-style interface will cease to be supported in the future.",
                FutureWarning,
            )
            return _ParameterSweep(*args, **kwargs)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        hilbertspace: HilbertSpace,
        paramvals_by_name: Dict[str, ndarray],
        update_hilbertspace: Callable,
        sweep_generators: Optional[Dict[str, Callable]] = None,
        evals_count: int = 6,
        subsys_update_info: Optional[Dict[str, List[QuantumSys]]] = None,
        bare_only: bool = False,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: Optional[int] = None,
    ) -> None:
        num_cpus = num_cpus or settings.NUM_CPUS
        self.parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._sweep_generators = sweep_generators
        self._evals_count = evals_count
        self._update_hilbertspace = update_hilbertspace
        self._subsys_update_info = subsys_update_info
        self._data: Dict[str, Optional[NamedSlotsNdarray]] = {}
        self._bare_only = bare_only
        self._num_cpus = num_cpus
        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        self._out_of_sync = False
        self._current_param_indices = tuple()

        if autorun:
            self.run()

    def cause_dispatch(self) -> None:
        initial_parameters = tuple(paramvals[0] for paramvals in self.parameters)
        self._update_hilbertspace(*initial_parameters)

    @classmethod
    def deserialize(cls, iodata: "IOData") -> "StoredSweep":
        pass

    def serialize(self) -> "IOData":
        """
        Convert the content of the current class instance into IOData format.

        Returns
        -------
        IOData
        """
        initdata = {
            "paramvals_by_name": self.parameters.ordered_dict,
            "hilbertspace": self._hilbertspace,
            "evals_count": self._evals_count,
            "_data": self._data,
        }
        iodata = serializers.dict_serialize(initdata)
        iodata.typename = "StoredSweep"
        return iodata

    def run(self) -> None:
        """Create all sweep data: bare spectral data, dressed spectral data, lookup
        data and custom sweep data."""
        # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        self.cause_dispatch()
        settings.DISPATCH_ENABLED = False
        self._data["bare_esys"] = self._bare_spectrum_sweep()
        if not self._bare_only:
            self._data["esys"] = self._dressed_spectrum_sweep()
            self._data["dressed_indices"] = self.generate_lookup()
        if self._sweep_generators is not None:
            for sweep_name, sweep_generator in self._sweep_generators.items():
                self._data[sweep_name] = self._custom_sweeps(sweep_generator)
        settings.DISPATCH_ENABLED = True

    def _bare_spectrum_sweep(self) -> NamedSlotsNdarray:
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
            [
                ("subsys", range(len(self._hilbertspace))),
                *[
                    (name, paramvals)
                    for name, paramvals in self.parameters.ordered_dict.items()
                ],
                ("esys", [0, 1]),
            ]
        )
        return NamedSlotsNdarray(bare_spectrum, slotparamvals_by_name)

    def _update_subsys_compute_esys(
        self, update_func: Callable, subsystem: QuantumSys, paramval_tuple: Tuple[float]
    ) -> ndarray:
        update_func(*paramval_tuple)
        evals, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
        esys_array = np.empty(shape=(2,), dtype=object)
        esys_array[0] = evals
        esys_array[1] = evecs
        return esys_array

    def _paramnames_no_subsys_update(self, subsystem) -> List[str]:
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
        fixed_paramnames = self._paramnames_no_subsys_update(subsystem)
        reduced_parameters = self.parameters.create_reduced(fixed_paramnames)
        total_count = np.prod([len(param_vals) for param_vals in reduced_parameters])

        multi_cpu = self._num_cpus > 1
        target_map = cpu_switch.get_map_method(self._num_cpus)

        with utils.InfoBar(
            "Parallel compute bare eigensys [num_cpus={}]".format(self._num_cpus),
            self._num_cpus,
        ) as p:
            bare_eigendata = list(
                tqdm(
                    target_map(
                        functools.partial(
                            self._update_subsys_compute_esys,
                            self._update_hilbertspace,
                            subsystem,
                        ),
                        itertools.product(*reduced_parameters.paramvals_list),
                    ),
                    total=total_count,
                    desc="Bare spectra",
                    leave=False,
                    disable=multi_cpu,
                )
            )

        bare_eigendata = np.asarray(bare_eigendata, dtype=object)
        bare_eigendata = bare_eigendata.reshape((*reduced_parameters.counts, 2))

        # Bare spectral data was only computed once for each parameter that has no
        # update effect on the subsystem. Now extend the array to reflect this
        # for the full parameter array by repeating
        for name in fixed_paramnames:
            index = self.parameters.index_by_name[name]
            param_count = self.parameters.counts[index]
            bare_eigendata = np.repeat(bare_eigendata, param_count, axis=index)

        return bare_eigendata

    def _update_and_compute_dressed_esys(
        self,
        hilbertspace: HilbertSpace,
        evals_count: int,
        update_func: Callable,
        paramindex_tuple: Tuple[int],
    ) -> ndarray:
        paramval_tuple = self.parameters[paramindex_tuple]
        update_func(*paramval_tuple)

        assert self._data is not None
        bare_esys = {
            subsys_index: self._data["bare_esys"][(subsys_index,) + paramindex_tuple]
            for subsys_index, _ in enumerate(self._hilbertspace)
        }
        evals, evecs = hilbertspace.eigensys(
            evals_count=evals_count, bare_esys=bare_esys
        )
        esys_array = np.empty(shape=(2,), dtype=object)
        esys_array[0] = evals
        esys_array[1] = evecs
        return esys_array

    def _dressed_spectrum_sweep(
        self,
    ) -> NamedSlotsNdarray:
        """

        Returns
        -------
            NamedSlotsNdarray[<paramname1>, <paramname2>, ..., "esys"]
            "esys": 0, 1 yields eigenvalues and eigenvectors, respectively
        """
        if len(self._hilbertspace) == 1 and self._hilbertspace.interaction_list == []:
            return self._data["bare_esys"]["subsys":0]

        multi_cpu = self._num_cpus > 1
        target_map = cpu_switch.get_map_method(self._num_cpus)
        total_count = np.prod(self.parameters.counts)

        with utils.InfoBar(
            "Parallel compute dressed eigensys [num_cpus={}]".format(self._num_cpus),
            self._num_cpus,
        ) as p:
            spectrum_data = list(
                tqdm(
                    target_map(
                        functools.partial(
                            self._update_and_compute_dressed_esys,
                            self._hilbertspace,
                            self._evals_count,
                            self._update_hilbertspace,
                        ),
                        itertools.product(*self.parameters.ranges),
                    ),
                    total=total_count,
                    desc="Dressed spectrum",
                    leave=False,
                    disable=multi_cpu,
                )
            )

        spectrum_data = np.asarray(spectrum_data, dtype=object)
        spectrum_data = spectrum_data.reshape((*self.parameters.counts, 2))
        slotparamvals_by_name = self.parameters.ordered_dict.copy()
        slotparamvals_by_name.update(OrderedDict([("esys", [0, 1])]))

        return NamedSlotsNdarray(spectrum_data, OrderedDict(slotparamvals_by_name))

    def _custom_sweeps(
        self,
        sweep_generators: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
    ) -> None:
        for name, generator_obj in sweep_generators.items():
            if callable(generator_obj):
                generator = generator_obj
                self._data[name] = generator(self)
            else:
                generator = generator_obj[0]
                kwargs = generator_obj[1]
                self._data[name] = generator(self, **kwargs)

    def add_sweep(
        self,
        sweep_name: str,
        sweep_generator: Union[Callable, Tuple[Callable, Dict[str, Any]]],
    ) -> None:
        self._custom_sweeps({sweep_name: sweep_generator})


class StoredSweep(
    ParameterSweepBase,
    SpectrumLookupMixin,
    dispatch.DispatchClient,
    serializers.Serializable,
):
    parameters = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    def __init__(self, paramvals_by_name, hilbertspace, evals_count, _data) -> None:
        self.parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._evals_count = evals_count
        self._data = _data

        self._out_of_sync = False
        self._current_param_indices = tuple()

    @classmethod
    def deserialize(cls, iodata: "IOData") -> "StoredSweep":
        """
        Take the given IOData and return an instance of the described class, initialized
        with the data stored in io_data.

        Parameters
        ----------
        iodata: IOData

        Returns
        -------
        StoredSweep
        """
        return StoredSweep(**iodata.as_kwargs())

    def serialize(self) -> "IOData":
        pass

    # StoredSweep: other methods
    def get_hilbertspace(self) -> HilbertSpace:
        return self._hilbertspace

    def new_sweep(
        self,
        paramvals_by_name: Dict[str, ndarray],
        update_hilbertspace: Callable,
        sweep_generators: Optional[Dict[str, Callable]] = None,
        evals_count: int = 6,
        subsys_update_info: Optional[Dict[str, List[QuantumSys]]] = None,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: Optional[int] = None,
    ) -> ParameterSweep:
        return ParameterSweep(
            self._hilbertspace,
            paramvals_by_name,
            update_hilbertspace,
            sweep_generators=sweep_generators,
            evals_count=evals_count,
            subsys_update_info=subsys_update_info,
            autorun=autorun,
            num_cpus=num_cpus,
        )
