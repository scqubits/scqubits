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
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scqubits.core.central_dispatch as dispatch
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils

from numpy import ndarray
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.spec_lookup2 import SpectrumLookupMixin

if TYPE_CHECKING:
    pass

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
        return tuple(len(paramvals) for paramvals in self)

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


class Sweep(SpectrumLookupMixin, dispatch.DispatchClient, serializers.Serializable):
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
        self._data: Dict[str, Optional[utils.NamedSlotsNdarray]] = {}
        self._num_cpus = num_cpus
        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        self._out_of_sync = False
        self._current_param_indices = None

        if autorun:
            self.generate_sweeps()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        # The following enables the following syntax:
        # <Sweep>[p1, p2, ...].dressed_eigenstates()
        if isinstance(key, tuple):
            self._current_param_indices = key
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

    def cause_dispatch(self) -> None:
        initial_parameters = tuple(paramvals[0] for paramvals in self.parameters)
        self._update_hilbertspace(*initial_parameters)

    def generate_sweeps(self) -> None:
        self.cause_dispatch()  # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        settings.DISPATCH_ENABLED = False
        self._data["esys_bare"] = self.bare_spectrum_sweep()
        self._data["esys"] = self.dressed_spectrum_sweep()
        self._data["lookup"] = self.spectrum_lookup_sweep()
        if self._sweep_generators is not None:
            for sweep_name, sweep_generator in self._sweep_generators.items():
                self._data[sweep_name] = self.custom_sweep(sweep_generator)
        settings.DISPATCH_ENABLED = True

    def bare_spectrum_sweep(
        self,
    ) -> utils.NamedSlotsNdarray:
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

        return utils.NamedSlotsNdarray(
            bare_spectrum, OrderedDict(slotparamvals_by_name)
        )

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
    ) -> utils.NamedSlotsNdarray:
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

        return utils.NamedSlotsNdarray(
            spectrum_data, OrderedDict(slotparamvals_by_name)
        )

    def custom_sweep(self, sweep_generator: Callable):
        pass

    def add_sweep(self, sweep_name: str, sweep_generator: Callable) -> None:
        pass
