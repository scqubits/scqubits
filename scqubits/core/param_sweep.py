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

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.sweeps as sweeps
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils

from scqubits.core._param_sweep import _ParameterSweep
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.namedslots_array import (
    NamedSlotsNdarray,
    Parameters,
    convert_to_std_npindex,
)
from scqubits.core.oscillator import Oscillator
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
GIndex = Union[Number, slice, Tuple[int], List[int]]
GIndexTuple = Tuple[GIndex, ...]
NpIndex = Union[int, slice, Tuple[int], List[int]]
NpIndexTuple = Tuple[NpIndex, ...]
NpIndices = Union[NpIndex, NpIndexTuple]


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

        # The following enables the pre-slicing syntax:
        # <Sweep>[p1, p2, ...].dressed_eigenstates()
        if isinstance(key, tuple):
            self._current_param_indices = convert_to_std_npindex(key, self.parameters)
        elif isinstance(key, (int, slice)):
            self._current_param_indices = convert_to_std_npindex(
                (key,), self.parameters
            )
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
    def bare_specdata_list(self) -> List[SpectrumData]:
        """
        Wrap bare eigensystem data into a SpectrumData object. To be used with
        pre-slicing, e.g. `<ParameterSweep>[0, :].bare_specdata_list`

        Returns
        -------
            List of `SpectrumData` objects with bare eigensystem data, one per subsystem
        """
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
    def dressed_specdata(self) -> "SpectrumData":
        """
        Wrap dressed eigensystem data into a SpectrumData object. To be used with
        pre-slicing, e.g. `<ParameterSweep>[0, :].dressed_specdata`

        Returns
        -------
            `SpectrumData` object with bare eigensystem data
        """
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

    def get_sweep_indices(self, multi_index: GIndexTuple) -> List[int]:
        """
        For given generalized multi-index, return a list of the indices that are being
        swept.
        """
        std_multi_index = convert_to_std_npindex(multi_index, self.parameters)

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

    def _final_states_subsys(
        self, subsystem: QuantumSys, initial_tuple: Tuple[int, ...]
    ) -> List[Tuple[int, ...]]:
        """For given initial statet of the composite quantum system, return the final
        states possible to reach by changing the energy level of the given
        `subsystem`"""
        subsys_index = self._hilbertspace.get_subsys_index(subsystem)
        final_tuples_list = []

        for level in range(subsystem.truncated_dim):
            final_state = list(initial_tuple)
            final_state[subsys_index] = level
            final_tuples_list.append(tuple(final_state))
        return final_tuples_list

    def _get_final_states(
        self,
        initial_state: List[int],
        subsys_list: List[QuantumSys],
        final: Union[int, Tuple[int, ...], None],
        sidebands: bool,
    ):
        """Construct and return the possible final states as a list, based on the
        provided initial state, a list of active subsystems and flag for whether to
        include sideband transitions."""
        if final:
            if isinstance(final, int):
                final = (final,)
            final_state_list = [self._complete_state(final)]
            return final_state_list

        if not sidebands:
            final_state_list = []
            for subsys in subsys_list:
                final_state_list += self._final_states_subsys(subsys, initial_state)
            return final_state_list

        # if sidebands:
        #     final_state_list = list(itertools.product(*range_list))
        #     return final_state_list
        range_list = [range(dim) for dim in self._hilbertspace.subsystem_dims]
        for subsys_index, subsys in enumerate(self._hilbertspace):
            if subsys not in subsys_list:
                range_list[subsys_index] = [initial_state[subsys_index]]
        final_state_list = list(itertools.product(*range_list))
        return final_state_list

    def _complete_state(
        self,
        partial_state: Union[List[int], Tuple[int]],
        subsys_list: List[QuantumSys],
    ) -> List[int]:
        """A partial state only includes entries for active subsystems. Complete this
        state by inserting 0 entries for all inactive subsystems."""
        state_full = [0] * len(self._hilbertspace)
        for entry, subsys in zip(partial_state, subsys_list):
            subsys_index = self.get_subsys_index(subsys)
            state_full[subsys_index] = entry
        return state_full

    def transitions(
        self,
        subsystem: Optional[Union[QuantumSys, List[QuantumSys]]] = None,
        initial: Optional[Union[int, Tuple[int, ...]]] = None,
        final: Optional[Union[int, Tuple[int, ...]]] = None,
        sidebands: bool = False,
        make_positive: bool = False,
        as_specdata: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> Union[Tuple[List[Tuple[int, ...]], List[NamedSlotsNdarray]], SpectrumData]:
        """
        Use dressed eigenenergy data and lookup based on bare product state labels to
        extract transition energy data. Usage is based on preslicing to select all or
        a subset of parameters to be involved in the sweep, e.g.,

        <ParameterSweep>[0, :, 2].transitions()

        produces all eigenenergy differences for transitions starting in the ground
        state (default when no initial state is specified) as a function of the middle
        parameter while parameters 1 and 3 are fixed by the indices 0 and 2.

        Parameters
        ----------
        subsystem:
            single subsystem or list of subsystems considered as "active" for the
            transitions to be generated; if omitted as a parameter, all subsystems
            are considered as actively participating in the transitions
        initial:
            initial state from which transitions originate, specified as a bare product
            state of either all subsystems the subset of active subsystems
            (default: ground state of the system)
        final:
            concrete final state for which the transition energy should be generated; if
            not provided, a list of allowed final states is generated
        sidebands:
            if set to true, sideband transitions with multiple subsystems changing
            excitation levels are included (default: False)
        make_positive:
            boolean option relevant if the initial state is an excited state;
            downwards transition energies would regularly be negative, but are
            converted to positive if this flag is set to True
        as_specdata:
            whether data is handed back in raw array form or wrapped into a SpectrumData
            object (default: False)
        param_indices:
            usually to be omitted, as param_indices will be set via pre-slicing

        Returns
        -------
            A tuple consisting of a list of all the transitions and a corresponding
            list of difference energies, e.g.
            ((0,0,0), (0,0,1)),    <energy array for transition 0,0,0 -> 0,0,1>.
            If as_specdata is set to True, a SpectrumData object is returned instead,
            saving transition label info in an attribute named `labels`.
        """
        param_indices = param_indices or self._current_param_indices

        if subsystem is None:
            subsys_list = self._hilbertspace.subsys_list
        elif isinstance(subsystem, (QubitBaseClass, Oscillator)):
            subsys_list = [subsystem]
        else:
            subsys_list = subsystem

        if initial is None:
            initial_state = (0,) * len(self._hilbertspace)
        elif isinstance(initial, int):
            initial_state = (initial,)
        else:
            initial_state = initial

        if len(initial_state) not in [len(self._hilbertspace), len(subsys_list)]:
            raise ValueError(
                "Initial state information provided is not compatible "
                "with the specified subsystem(s) provided."
            )
        elif len(initial_state) < len(self._hilbertspace):
            initial_state = self._complete_initial_state(initial_state, subsys_list)

        final_states_list = self._get_final_states(
            initial_state, subsys_list, final, sidebands
        )

        transitions = [
            (initial_state, final_state) for final_state in final_states_list
        ]
        transitions = []
        transition_energies = []
        for final_state in final_states_list:
            initial_energies = self[param_indices].energy_by_bare_index(initial_state)
            final_energies = self[param_indices].energy_by_bare_index(final_state)
            diff_energies = (final_energies - initial_energies).astype(float)
            if make_positive:
                diff_energies = np.abs(diff_energies)
            if not np.isnan(diff_energies).all():
                transitions.append((initial_state, final_state))
                transition_energies.append(diff_energies)

        if not as_specdata:
            return transitions, transition_energies

        reduced_parameters = self.parameters.create_sliced(param_indices)
        if len(reduced_parameters) == 1:
            name = reduced_parameters.names[0]
            vals = reduced_parameters[name]
            return SpectrumData(
                energy_table=np.asarray(transition_energies).T,
                system_params=self.system_params,
                param_name=name,
                param_vals=vals,
                labels=list(map(str, transitions)),
            )
        return SpectrumData(
            energy_table=np.asarray(transition_energies),
            system_params=self.system_params,
            label=list(map(str, transitions)),
        )

    def plot_transitions(
        self,
        subsystem: Optional[Union[QuantumSys, List[QuantumSys]]] = None,
        initial: Optional[Union[int, Tuple[int, ...]]] = None,
        final: Optional[Union[int, Tuple[int, ...]]] = None,
        sidebands: bool = False,
        make_positive: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> Tuple[Figure, Axes]:
        """
        Plot transition energies as a function of one external parameter. Usage is based
        on preslicing of the ParameterSweep object to select a single parameter to be
        involved in the sweep. E.g.,

        <ParameterSweep>[0, :, 2].plot_transitions()

        plots all eigenenergy differences for transitions starting in the ground
        state (default when no initial state is specified) as a function of the middle
        parameter while parameters 1 and 3 are fixed by the indices 0 and 2.

        Parameters
        ----------
        subsystem:
            single subsystem or list of subsystems considered as "active" for the
            transitions to be generated; if omitted as a parameter, all subsystems
            are considered as actively participating in the transitions
        initial:
            initial state from which transitions originate, specified as a bare product
            state of either all subsystems the subset of active subsystems
            (default: ground state of the system)
        final:
            concrete final state for which the transition energy should be generated; if
            not provided, a list of allowed final states is generated
        sidebands:
            if set to true, sideband transitions with multiple subsystems changing
            excitation levels are included (default: False)
        make_positive:
            boolean option relevant if the initial state is an excited state;
            downwards transition energies would regularly be negative, but are
            converted to positive if this flag is set to True
        param_indices:
            usually to be omitted, as param_indices will be set via pre-slicing

        Returns
        -------
            A tuple consisting of a list of all the transitions and a corresponding
            list of difference energies, e.g.
            ((0,0,0), (0,0,1)),    <energy array for transition 0,0,0 -> 0,0,1>.
            If as_specdata is set to True, a SpectrumData object is returned instead,
            saving transition label info in an attribute named `labels`.
        """

        param_indices = param_indices or self._current_param_indices
        if len(self.parameters.create_sliced(param_indices)) > 1:
            raise ValueError(
                "Transition plots are only supported for a sweep over a "
                "single parameter. You can reduce a multidimensional "
                "sweep by pre-slicing, e.g.,  <ParameterSweep>[0, :, "
                "0].plot_transitions(...)"
            )
        specdata = self.transitions(
            subsystem,
            initial,
            final,
            sidebands,
            make_positive,
            as_specdata=True,
            param_indices=param_indices,
        )
        return specdata.plot_evals_vs_paramvals(label_list=specdata.labels)

    def add_sweep(
        self,
        sweep_function: Union[str, Callable],
        sweep_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Add a new sweep to the ParameterSweep object. The generated data is
        subsequently accessible through <ParameterSweep>[<sweep_function>] or
        <ParameterSweep>[<sweep_name>]

        Parameters
        ----------
        sweep_function:
            name of a sweep function in scq.sweeps as str, or custom function (
            callable) provided by the user
        sweep_name:
            if given, the generated data is stored in <ParameterSweep>[<sweep_name>]
            rather than [<sweep_name>]
        kwargs:
            keyword arguments handed over to the sweep function

        Returns
        -------
            None
        """
        if callable(sweep_function):
            sweep_name = sweep_name or sweep_function.__name__
            func = sweep_function
            self._data[sweep_name] = sweeps.generator(self, func, **kwargs)
        else:
            sweep_name = sweep_name or sweep_function
            func = getattr(sweeps, sweep_function)
            self._data[sweep_name] = func(**kwargs)


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
    update_hilbertspace:
        function that updates the associated ``hilbertspace`` object with a given
        set of parameters
    evals_count:
        number of dressed eigenvalues/eigenstates to keep. (The number of bare
        eigenvalues/eigenstates is determined for each subsystem by `truncated_dim`.)
        [default: 20]
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
        evals_count: int = 20,
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
            NamedSlotsNdarray[<paramname1>, <paramname2>, ..., "subsystem", "esys"]
            where "subsystem": 0, 1, ... enumerates subsystems and
            "esys": 0, 1 yields eigenvalues and eigenvectors, respectively
        """
        bare_spectrum = []
        for subsystem in self._hilbertspace:
            bare_spectrum += [self._subsys_bare_spectrum_sweep(subsystem)]
        bare_spectrum = np.asarray(bare_spectrum, dtype=object)
        bare_spectrum = np.moveaxis(bare_spectrum, 0, -2)

        slotparamvals_by_name = OrderedDict(
            [
                *[
                    (name, paramvals)
                    for name, paramvals in self.parameters.ordered_dict.items()
                ],
                ("subsys", np.arange(len(self._hilbertspace))),
                ("esys", np.asarray([0, 1])),
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
            subsys_index: self._data["bare_esys"][paramindex_tuple + (subsys_index,)]
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
        slotparamvals_by_name.update(OrderedDict([("esys", np.asarray([0, 1]))]))

        return NamedSlotsNdarray(spectrum_data, OrderedDict(slotparamvals_by_name))


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
