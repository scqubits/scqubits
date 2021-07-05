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

import copy
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
from qutip import Qobj

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.sweeps as sweeps
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils
import scqubits.utils.plotting as plot

from scqubits import settings as settings
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.namedslots_array import (
    NamedSlotsNdarray,
    Parameters,
    convert_to_std_npindex,
)
from scqubits.core.oscillator import Oscillator
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.spec_lookup import SpectrumLookupMixin
from scqubits.core.storage import SpectrumData

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData
    from scqubits.legacy._param_sweep import _ParameterSweep

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


QuantumSys = Union[QubitBaseClass, Oscillator]
GIndex = Union[int, float, complex, slice, Tuple[int], List[int]]
GIndexTuple = Tuple[GIndex, ...]
NpIndex = Union[int, slice, Tuple[int], List[int]]
NpIndexTuple = Tuple[NpIndex, ...]
NpIndices = Union[NpIndex, NpIndexTuple]


class ParameterSweepBase(ABC):
    """
    The_ParameterSweepBase class is an abstract base class for ParameterSweep and
    StoredSweep
    """

    _parameters = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    _out_of_sync = False
    _current_param_indices: NpIndices

    def get_subsys(self, index: int) -> QuantumSys:
        return self._hilbertspace[index]

    def get_subsys_index(self, subsys: QuantumSys) -> int:
        return self._hilbertspace.get_subsys_index(subsys)

    @property
    def osc_subsys_list(self) -> List[Oscillator]:
        return self._hilbertspace.osc_subsys_list

    @property
    def qbt_subsys_list(self) -> List[QubitBaseClass]:
        return self._hilbertspace.qbt_subsys_list

    @property
    def subsystem_count(self) -> int:
        return self._hilbertspace.subsystem_count

    @utils.check_sync_status
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]

        # The following enables the pre-slicing syntax:
        # <Sweep>[p1, p2, ...].dressed_eigenstates()
        if isinstance(key, tuple):
            self._current_param_indices = convert_to_std_npindex(key, self._parameters)
        elif isinstance(key, slice):
            if key == slice(None) or key == slice(None, None, None):
                key = (key,) * len(self._parameters)
            else:
                key = (key,)
            self._current_param_indices = convert_to_std_npindex(key, self._parameters)
        elif isinstance(key, np.integer):
            key = (key,)
            self._current_param_indices = convert_to_std_npindex(key, self._parameters)
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
        if self._data:
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
        sweep_param_name = self._parameters.name_by_index[sweep_param_indices[0]]
        specdata_list: List[SpectrumData] = []
        for subsys_index, subsystem in enumerate(self._hilbertspace):
            evals_swp = self["bare_evals"][subsys_index][multi_index]
            evecs_swp = self["bare_evecs"][subsys_index][multi_index]
            specdata_list.append(
                SpectrumData(
                    energy_table=evals_swp.toarray(),
                    state_table=evecs_swp.toarray(),
                    system_params=self._hilbertspace.get_initdata(),
                    param_name=sweep_param_name,
                    param_vals=self._parameters[sweep_param_name],
                )
            )
        self._current_param_indices = slice(None, None, None)
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
        sweep_param_name = self._parameters.name_by_index[sweep_param_indices[0]]

        specdata = SpectrumData(
            energy_table=self["evals"][multi_index].toarray(),
            state_table=self["evecs"][multi_index].toarray(),
            system_params=self._hilbertspace.get_initdata(),
            param_name=sweep_param_name,
            param_vals=self._parameters[sweep_param_name],
        )
        self._current_param_indices = slice(None, None, None)
        return specdata

    def get_sweep_indices(self, multi_index: GIndexTuple) -> List[int]:
        """
        For given generalized multi-index, return a list of the indices that are being
        swept.
        """
        std_multi_index = convert_to_std_npindex(multi_index, self._parameters)

        sweep_indices = [
            index
            for index, index_obj in enumerate(std_multi_index)
            if isinstance(
                self._parameters.paramvals_list[index][index_obj],
                (list, tuple, ndarray),
            )
        ]
        self._current_param_indices = slice(None, None, None)
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
        final_tuples_list.remove(initial_tuple)
        return final_tuples_list

    def _get_final_states(
        self,
        initial_state: Tuple[int],
        subsys_list: List[QuantumSys],
        final: Union[Tuple[int, ...], None],
        sidebands: bool,
    ) -> List[Tuple[int, ...]]:
        """Construct and return the possible final states as a list, based on the
        provided initial state, a list of active subsystems and flag for whether to
        include sideband transitions."""
        if final:
            return [final]

        if not sidebands:
            final_state_list = []
            for subsys in subsys_list:
                final_state_list += self._final_states_subsys(subsys, initial_state)
            return final_state_list

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
        subsystems: Optional[Union[QuantumSys, List[QuantumSys]]] = None,
        initial: Optional[Union[int, Tuple[int, ...]]] = None,
        final: Optional[Tuple[int, ...]] = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        as_specdata: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> Union[Tuple[List[Tuple[int, ...]], List[NamedSlotsNdarray]], SpectrumData]:
        """
        Use dressed eigenenergy data and lookup based on bare product state labels to
        extract transition energy data. Usage is based on preslicing to select all or
        a subset of parameters to be involved in the sweep, e.g.,

        `<ParameterSweep>[0, :, 2].transitions()`

        produces all eigenenergy differences for transitions starting in the ground
        state (default when no initial state is specified) as a function of the middle
        parameter while parameters 1 and 3 are fixed by the indices 0 and 2.

        Parameters
        ----------
        subsystems:
            single subsystems or list of subsystems considered as "active" for the
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
        photon_number:
            number of photons involved in transition; transition frequencies are divided
            by this number (default: photon_number=1, i.e., single-photon transitions)
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

        if subsystems is None:
            subsys_list = self._hilbertspace.subsys_list
        elif isinstance(subsystems, (QubitBaseClass, Oscillator)):
            subsys_list = [subsystems]
        else:
            subsys_list = subsystems

        if initial is None:
            initial_state = (0,) * len(self._hilbertspace)
        elif isinstance(initial, int):
            initial_state = (initial,)
        else:
            initial_state = initial

        if len(initial_state) not in [len(self._hilbertspace), len(subsys_list)]:
            raise ValueError(
                "Initial state information provided is not compatible "
                "with the specified subsystems(s) provided."
            )

        if len(initial_state) < len(self._hilbertspace):
            initial_state = self._complete_initial_state(initial_state, subsys_list)

        final_states_list = self._get_final_states(
            initial_state, subsys_list, final, sidebands
        )

        transitions = []
        transition_energies = []

        if sum(initial_state) == 0:
            # Identify the (0,0,...,0) state as ground state. Even if it is strongly
            # hybridized, we can still subtract the true ground state energy. This
            # addresses issue 103.
            initial_energies = self[param_indices].energy_by_dressed_index(0)
        else:
            initial_energies = self[param_indices].energy_by_bare_index(initial_state)

        for final_state in final_states_list:
            final_energies = self[param_indices].energy_by_bare_index(final_state)
            diff_energies = (final_energies - initial_energies).astype(float)
            diff_energies /= photon_number
            if make_positive:
                diff_energies = np.abs(diff_energies)
            if not np.isnan(diff_energies.toarray()).all():
                transitions.append((initial_state, final_state))
                transition_energies.append(diff_energies)

        self._current_param_indices = slice(None, None, None)

        if not as_specdata:
            return transitions, transition_energies

        label_list = [
            r"{}$\to${}".format(
                utils.tuple_to_short_str(elem[0]), utils.tuple_to_short_str(elem[1])
            )
            for elem in transitions
        ]

        reduced_parameters = self._parameters.create_sliced(param_indices)
        if len(reduced_parameters) == 1:
            name = reduced_parameters.names[0]
            vals = reduced_parameters[name]
            return SpectrumData(
                energy_table=np.asarray(transition_energies).T,
                system_params=self.system_params,
                param_name=name,
                param_vals=vals,
                labels=label_list,
                subtract=np.asarray(
                    [initial_energies] * self._evals_count, dtype=float
                ).T,
            )

        return SpectrumData(
            energy_table=np.asarray(transition_energies),
            system_params=self.system_params,
            labels=label_list,
        )

    def plot_transitions(
        self,
        subsystems: Optional[Union[QuantumSys, List[QuantumSys]]] = None,
        initial: Optional[Union[int, Tuple[int, ...]]] = None,
        final: Optional[Union[int, Tuple[int, ...]]] = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = True,
        coloring: Union[str, ndarray] = "transition",
        param_indices: Optional[NpIndices] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot transition energies as a function of one external parameter. Usage is based
        on preslicing of the ParameterSweep object to select a single parameter to be
        involved in the sweep. E.g.,

        `<ParameterSweep>[0, :, 2].plot_transitions()`

        plots all eigenenergy differences for transitions starting in the ground
        state (default when no initial state is specified) as a function of the middle
        parameter while parameters 1 and 3 are fixed by the indices 0 and 2.

        Parameters
        ----------
        subsystems:
            single subsystems or list of subsystems considered as "active" for the
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
        photon_number:
            number of photons involved in transition; transition frequencies are divided
            by this number (default: photon_number=1, i.e., single-photon transitions)
        make_positive:
            boolean option relevant if the initial state is an excited state;
            downwards transition energies would regularly be negative, but are
            converted to positive if this flag is set to True (default: True)
        coloring:
            For `"transition"` (default), transitions are colored by their
            dispersive nature; "`plain`", curves are colored naively
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
        if len(self._parameters.create_sliced(param_indices)) > 1:
            raise ValueError(
                "Transition plots are only supported for a sweep over a "
                "single parameter. You can reduce a multidimensional "
                "sweep by pre-slicing, e.g.,  <ParameterSweep>[0, :, "
                "0].plot_transitions(...)"
            )
        specdata = self.transitions(
            subsystems,
            initial,
            final,
            sidebands,
            photon_number,
            make_positive,
            as_specdata=True,
            param_indices=param_indices,
        )
        specdata_all = copy.deepcopy(self[param_indices].dressed_specdata)
        specdata_all.energy_table -= specdata.subtract
        specdata_all.energy_table /= photon_number
        if make_positive:
            specdata_all.energy_table = np.abs(specdata_all.energy_table)
        self._current_param_indices = slice(None, None, None)  # reset from pre-slicing

        if coloring == "plain":
            return specdata_all.plot_evals_vs_paramvals()

        if "fig_ax" in kwargs:
            fig_ax = kwargs.pop("fig_ax")
        else:
            fig_ax = None
        fig_ax = specdata_all.plot_evals_vs_paramvals(
            color="gainsboro", linewidth=0.75, fig_ax=fig_ax
        )

        labellines_status = plot._LABELLINES_ENABLED
        plot._LABELLINES_ENABLED = False
        fig, axes = specdata.plot_evals_vs_paramvals(
            label_list=specdata.labels, fig_ax=fig_ax, **kwargs
        )
        plot._LABELLINES_ENABLED = labellines_status
        return fig, axes

    def keys(self):
        return self._data.keys()

    def add_sweep(
        self,
        sweep_function: Union[str, Callable],
        sweep_name: Optional[str] = None,
        **kwargs,
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
            if not hasattr(sweep_function, "__name__") and not sweep_name:
                raise ValueError(
                    "Sweep function name cannot be accessed. Provide an "
                    "explicit `sweep_name` instead."
                )
            sweep_name = sweep_name or sweep_function.__name__
            func = sweep_function
            self._data[sweep_name] = generator(self, func, **kwargs)
        else:
            sweep_name = sweep_name or sweep_function
            func = getattr(sweeps, sweep_function)
            self._data[sweep_name] = func(**kwargs)

    def add_matelem_sweep(
        self,
        operator: Union[str, Qobj],
        sweep_name: str,
        subsystem: "QuantumSys" = None,
    ) -> None:
        """Generate data for matrix elements with respect to a given operator, as a
        function of the sweep parameter(s)

        Parameters
        ----------
        operator:
            name of the operator in question (str), or full operator in Qobj form
        sweep_name:
            The sweep data will be accessible as <ParameterSweep>[<sweep_name>]
        subsystem:
            subsystems for which to compute matrix elements, required if operator is
            given in str form

        Returns
        -------
            None; results are saved as <ParameterSweep>[<sweep_name>]
        """
        if isinstance(operator, str):
            operator_func = functools.partial(
                sweeps.bare_matrixelement, operator_name=operator, subsystem=subsystem,
            )
        elif isinstance(operator, Qobj):
            operator_func = functools.partial(
                sweeps.dressed_matrixelement, operator=operator,
            )
        else:
            raise TypeError(
                "Unrecognized type of operator for matrix elements; "
                "expected: str or Qobj."
            )

        matrix_element_data = generator(self, operator_func,)
        self._data[sweep_name] = matrix_element_data


class ParameterSweep(
    ParameterSweepBase,
    SpectrumLookupMixin,
    dispatch.DispatchClient,
    serializers.Serializable,
):
    """
    `ParameterSweep` supports array-like access ("pre-slicing") and dict-like access.
    With dict-like access via string-keywords `<ParameterSweep>[<str>]`,
    the following data is returned:

    `"evals"` and `"evecs"`
        dressed eigenenergies and eigenstates as
        `NamedSlotsNdarray`; eigenstates are decomposed in the bare product-state basis
        of the non-interacting subsystems' eigenbases
    `"bare_evals"` and `"bare_evecs"`
        bare eigenenergies and eigenstates as `NamedSlotsNdarray`
    `"lamb"`, `"chi"`, and `"kerr"`
        dispersive energy coefficients
    `"<custom sweep>"`
        NamedSlotsNdarray for custom data generated with `add_sweep`.

    Array-like access is responsible for "pre-slicing",
    enable lookup functionality such as
    `<Sweep>[p1, p2, ...].eigensys()`

    Parameters
    ----------
    hilbertspace:
        HilbertSpace object describing the quantum system of interest
    paramvals_by_name:
        Dictionary that, for each set of parameter values, specifies a parameter name
        and the set of values to be used in the sweep.
    update_hilbertspace:
        function that updates the associated `hilbertspace` object with a given
        set of parameters
    evals_count:
        number of dressed eigenvalues/eigenstates to keep. (The number of bare
        eigenvalues/eigenstates is determined for each subsystems by `truncated_dim`.)
        [default: 20]
    subsys_update_info:
        To speed up calculations, the user may provide information that specifies which
        subsystems are being updated for each of the given parameter sweeps. This
        information is specified by a dictionary of the following form::

            {
                "<parameter name 1>": [<subsystem a>],
                "<parameter name 2>": [<subsystem b>, <subsystem c>, ...],
                ...
            }

        This indicates that changes in `<parameter name 1>` only require updates of
        `<subsystem a>` while leaving other subsystems unchanged. Similarly, sweeping
        `<parameter name 2>` affects `<subsystem b>`, `<subsystem c>` etc.
    bare_only:
        if set to True, only bare eigendata is calculated; useful when performing a
        sweep for a single quantum system, no interaction (default: False)
    autorun:
        Determines whether to directly run the sweep or delay it until `.run()` is
        called manually. (Default: `settings.AUTORUN_SWEEP=True`)
    num_cpus:
        number of CPU cores requested for computing the sweep
        (default value `settings.NUM_CPUS`)

    """

    def __new__(cls, *args, **kwargs) -> "Union[ParameterSweep, _ParameterSweep]":
        if args and isinstance(args[0], str) or "param_name" in kwargs:
            # old-style ParameterSweep interface is being used
            warnings.warn(
                "The implementation of the `ParameterSweep` class has changed and this "
                "old-style interface will cease to be supported in the future.",
                FutureWarning,
            )
            from scqubits.legacy._param_sweep import _ParameterSweep

            return _ParameterSweep(*args, **kwargs)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        hilbertspace: HilbertSpace,
        paramvals_by_name: Dict[str, ndarray],
        update_hilbertspace: Callable,
        evals_count: int = 20,
        subsys_update_info: Optional[Dict[str, List[QuantumSys]]] = None,
        bare_only: bool = False,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: Optional[int] = None,
    ) -> None:
        num_cpus = num_cpus or settings.NUM_CPUS
        self._parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._evals_count = evals_count
        self._update_hilbertspace = update_hilbertspace
        self._subsys_update_info = subsys_update_info
        self._data: Dict[str, Optional[NamedSlotsNdarray]] = {}
        self._bare_only = bare_only
        self._num_cpus = num_cpus
        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        self._out_of_sync = False
        self._current_param_indices = slice(None, None, None)

        dispatch.CENTRAL_DISPATCH.register("PARAMETERSWEEP_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("HILBERTSPACE_UPDATE", self)

        if autorun:
            self.run()

    def cause_dispatch(self) -> None:
        initial_parameters = tuple(paramvals[0] for paramvals in self._parameters)
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
            "paramvals_by_name": self._parameters.ordered_dict,
            "hilbertspace": self._hilbertspace,
            "evals_count": self._evals_count,
            "_data": self._data,
        }
        iodata = serializers.dict_serialize(initdata)
        iodata.typename = "StoredSweep"
        return iodata

    @property
    def param_info(self) -> Dict[str, ndarray]:
        """Return a dictionary of the parameter names and values used in this sweep."""
        return self._parameters.paramvals_by_name

    def run(self) -> None:
        """Create all sweep data: bare spectral data, dressed spectral data, lookup
        data and custom sweep data."""
        # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        self.cause_dispatch()
        settings.DISPATCH_ENABLED = False
        self._data["bare_evals"], self._data["bare_evecs"] = self._bare_spectrum_sweep()
        if not self._bare_only:
            self._data["evals"], self._data["evecs"] = self._dressed_spectrum_sweep()
            self._data["dressed_indices"] = self.generate_lookup()
            (
                self._data["lamb"],
                self._data["chi"],
                self._data["kerr"],
            ) = self._dispersive_coefficients()
        settings.DISPATCH_ENABLED = True

    def _bare_spectrum_sweep(self) -> Tuple[NamedSlotsNdarray, NamedSlotsNdarray]:
        """
        The bare energy spectra are computed according to the following scheme.
        1. Perform a loop over all subsystems to separately obtain the bare energy
            eigenvalues and eigenstates for each subsystems.
        2. If `update_subsystem_info` is given, remove those sweeps that leave the
            subsystems fixed.
        3. If self._num_cpus > 1, parallelize.

        Returns
        -------
            NamedSlotsNdarray[<paramname1>, <paramname2>, ..., "subsys"] for evals,
            likewise for evecs;
            here, "subsys": 0, 1, ... enumerates subsystems and
        """
        bare_evals = np.empty((self.subsystem_count,), dtype=object)
        bare_evecs = np.empty((self.subsystem_count,), dtype=object)

        for subsys_index, subsystem in enumerate(self._hilbertspace):
            bare_esys = self._subsys_bare_spectrum_sweep(subsystem)
            bare_evals[subsys_index] = NamedSlotsNdarray(
                np.asarray(bare_esys[..., 0].tolist()),
                self._parameters.paramvals_by_name,
            )
            bare_evecs[subsys_index] = NamedSlotsNdarray(
                np.asarray(bare_esys[..., 1].tolist()),
                self._parameters.paramvals_by_name,
            )

        return (
            NamedSlotsNdarray(bare_evals, {"subsys": np.arange(self.subsystem_count)}),
            NamedSlotsNdarray(bare_evecs, {"subsys": np.arange(self.subsystem_count)}),
        )

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
        return list(set(self._parameters.names) - set(updating_parameters))

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
        reduced_parameters = self._parameters.create_reduced(fixed_paramnames)
        total_count = np.prod([len(param_vals) for param_vals in reduced_parameters])

        multi_cpu = self._num_cpus > 1
        target_map = cpu_switch.get_map_method(self._num_cpus)

        with utils.InfoBar(
            "Parallel compute bare eigensys [num_cpus={}]".format(self._num_cpus),
            self._num_cpus,
        ) as p:
            bare_eigendata = tqdm(
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

        bare_eigendata = np.asarray(list(bare_eigendata), dtype=object)
        bare_eigendata = bare_eigendata.reshape((*reduced_parameters.counts, 2))

        # Bare spectral data was only computed once for each parameter that has no
        # update effect on the subsystems. Now extend the array to reflect this
        # for the full parameter array by repeating
        for name in fixed_paramnames:
            index = self._parameters.index_by_name[name]
            param_count = self._parameters.counts[index]
            bare_eigendata = np.repeat(bare_eigendata, param_count, axis=index)

        return bare_eigendata

    def _update_and_compute_dressed_esys(
        self,
        hilbertspace: HilbertSpace,
        evals_count: int,
        update_func: Callable,
        paramindex_tuple: Tuple[int],
    ) -> ndarray:
        paramval_tuple = self._parameters[paramindex_tuple]
        update_func(*paramval_tuple)

        assert self._data is not None
        bare_esys = {
            subsys_index: [
                self._data["bare_evals"][subsys_index][paramindex_tuple],
                self._data["bare_evecs"][subsys_index][paramindex_tuple],
            ]
            for subsys_index, _ in enumerate(self._hilbertspace)
        }

        evals, evecs = hilbertspace.eigensys(
            evals_count=evals_count, bare_esys=bare_esys
        )
        esys_array = np.empty(shape=(2,), dtype=object)
        esys_array[0] = evals
        esys_array[1] = evecs
        return esys_array

    def _dressed_spectrum_sweep(self,) -> Tuple[NamedSlotsNdarray, NamedSlotsNdarray]:
        """

        Returns
        -------
            NamedSlotsNdarray[<paramname1>, <paramname2>, ...] of eigenvalues,
            likewise for eigenvectors
        """
        multi_cpu = self._num_cpus > 1
        target_map = cpu_switch.get_map_method(self._num_cpus)
        total_count = np.prod(self._parameters.counts)

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
                        itertools.product(*self._parameters.ranges),
                    ),
                    total=total_count,
                    desc="Dressed spectrum",
                    leave=False,
                    disable=multi_cpu,
                )
            )

        spectrum_data = np.asarray(spectrum_data, dtype=object)
        spectrum_data = spectrum_data.reshape((*self._parameters.counts, 2))
        slotparamvals_by_name = OrderedDict(self._parameters.ordered_dict.copy())

        evals = np.asarray(spectrum_data[..., 0].tolist())
        evecs = spectrum_data[..., 1]

        return (
            NamedSlotsNdarray(evals, slotparamvals_by_name),
            NamedSlotsNdarray(evecs, slotparamvals_by_name),
        )

    def _energies_1(self, subsys):
        bare_label = np.zeros(len(self._hilbertspace))
        bare_label[self.get_subsys_index(subsys)] = 1

        energies_all_l = np.empty(self._parameters.counts + (subsys.truncated_dim,))
        for l in range(subsys.truncated_dim):
            energies_all_l[..., l] = self[:].energy_by_bare_index(tuple(l * bare_label))
        return energies_all_l

    def _energies_2(self, subsys1, subsys2):
        bare_label1 = np.zeros(len(self._hilbertspace))
        bare_label1[self.get_subsys_index(subsys1)] = 1
        bare_label2 = np.zeros(len(self._hilbertspace))
        bare_label2[self.get_subsys_index(subsys2)] = 1

        energies_all_l1_l2 = np.empty(
            self._parameters.counts
            + (subsys1.truncated_dim,)
            + (subsys2.truncated_dim,)
        )
        for l1 in range(subsys1.truncated_dim):
            for l2 in range(subsys2.truncated_dim):
                energies_all_l1_l2[..., l1, l2] = self[:].energy_by_bare_index(
                    tuple(l1 * bare_label1 + l2 * bare_label2)
                )
        return energies_all_l1_l2

    def _dispersive_coefficients(
        self,
    ) -> Tuple[NamedSlotsNdarray, NamedSlotsNdarray, NamedSlotsNdarray]:
        energy_0 = self[:].energy_by_dressed_index(0).toarray()

        lamb_data = np.empty(self.subsystem_count, dtype=object)
        kerr_data = np.empty((self.subsystem_count, self.subsystem_count), dtype=object)

        for subsys_index1, subsys1 in enumerate(self._hilbertspace):
            energy_subsys1_all_l1 = self._energies_1(subsys1)
            bare_energy_subsys1_all_l1 = self["bare_evals"][subsys_index1].toarray()
            lamb_subsys1_all_l1 = (
                energy_subsys1_all_l1
                - energy_0[..., None]
                - bare_energy_subsys1_all_l1
                + bare_energy_subsys1_all_l1[..., 0][..., None]
            )
            lamb_data[subsys_index1] = NamedSlotsNdarray(
                lamb_subsys1_all_l1, self._parameters.paramvals_by_name
            )

            for subsys_index2, subsys2 in enumerate(self._hilbertspace):
                energy_subsys2_all_l2 = self._energies_1(subsys2)
                energy_subsys1_subsys2_all_l1_l2 = self._energies_2(subsys1, subsys2)
                kerr_subsys1_subsys2_all_l1_l2 = (
                    energy_subsys1_subsys2_all_l1_l2
                    + energy_0[..., None, None]
                    - energy_subsys1_all_l1[..., :, None]
                    - energy_subsys2_all_l2[..., None, :]
                )
                if subsys1 is subsys2:
                    kerr_subsys1_subsys2_all_l1_l2 /= 2.0  # self-Kerr needs factor 1/2

                kerr_data[subsys_index1, subsys_index2] = NamedSlotsNdarray(
                    kerr_subsys1_subsys2_all_l1_l2, self._parameters.paramvals_by_name
                )

        sys_indices = np.arange(self.subsystem_count)
        lamb_data = NamedSlotsNdarray(lamb_data, {"subsys": sys_indices})
        kerr_data = NamedSlotsNdarray(
            kerr_data, {"subsys1": sys_indices, "subsys2": sys_indices}
        )
        chi_data = kerr_data.copy()

        for subsys_index1, subsys1 in enumerate(self._hilbertspace):
            for subsys_index2, subsys2 in enumerate(self._hilbertspace):
                if subsys1 in self.osc_subsys_list:
                    if subsys2 in self.qbt_subsys_list:
                        chi_data[subsys_index1, subsys_index2] = chi_data[
                            subsys_index1, subsys_index2
                        ][..., 1, :]
                        kerr_data[subsys_index1, subsys_index2] = np.asarray([])
                    else:
                        chi_data[subsys_index1, subsys_index2] = np.asarray([])
                elif subsys1 in self.qbt_subsys_list:
                    if subsys2 in self.osc_subsys_list:
                        chi_data[subsys_index1, subsys_index2] = chi_data[
                            subsys_index1, subsys_index2
                        ][..., :, 1]
                        kerr_data[subsys_index1, subsys_index2] = np.asarray([])
                    else:
                        chi_data[subsys_index1, subsys_index2] = np.asarray([])

        return lamb_data, chi_data, kerr_data


class StoredSweep(
    ParameterSweepBase,
    SpectrumLookupMixin,
    dispatch.DispatchClient,
    serializers.Serializable,
):
    _parameters = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty("PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    def __init__(self, paramvals_by_name, hilbertspace, evals_count, _data) -> None:
        self._parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._evals_count = evals_count
        self._data = _data

        self._out_of_sync = False
        self._current_param_indices = slice(None, None, None)

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
        evals_count: int = 6,
        subsys_update_info: Optional[Dict[str, List[QuantumSys]]] = None,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: Optional[int] = None,
    ) -> ParameterSweep:
        return ParameterSweep(
            self._hilbertspace,
            paramvals_by_name,
            update_hilbertspace,
            evals_count=evals_count,
            subsys_update_info=subsys_update_info,
            autorun=autorun,
            num_cpus=num_cpus,
        )


def generator(sweep: "ParameterSweep", func: callable, **kwargs) -> np.ndarray:
    """Method for computing custom data as a function of the external parameter,
    calculated via the function `func`.

    Parameters
    ----------
    sweep:
        ParameterSweep object containing HilbertSpace and spectral information
    func:
        signature: `func(parametersweep, paramindex_tuple, paramvals_tuple,
        **kwargs)`, specifies how to calculate the data for a single choice of
        parameter(s)
    **kwargs:
        keyword arguments to be included in func

    Returns
    -------
        array of custom data
    """
    # obtain reduced parameters from pre-slicing info
    reduced_parameters = sweep._parameters.create_sliced(
        sweep._current_param_indices, remove_fixed=False
    )
    total_count = np.prod(reduced_parameters.counts)

    def func_effective(paramindex_tuple: Tuple[int], params, **kw) -> Any:
        paramvals_tuple = params[paramindex_tuple]
        return func(
            sweep,
            paramindex_tuple=paramindex_tuple,
            paramvals_tuple=paramvals_tuple,
            **kw,
        )

    if hasattr(func, "__name__"):
        func_name = func.__name__
    else:
        func_name = ""

    data_array = list(
        tqdm(
            map(
                functools.partial(func_effective, params=reduced_parameters, **kwargs,),
                itertools.product(*reduced_parameters.ranges),
            ),
            total=total_count,
            desc="sweeping " + func_name,
            leave=False,
            disable=settings.PROGRESSBAR_DISABLED,
        )
    )
    element_shape = tuple()
    if isinstance(data_array[0], np.ndarray):
        element_shape = data_array[0].shape

    data_array = np.asarray(data_array)
    return NamedSlotsNdarray(
        data_array.reshape(reduced_parameters.counts + element_shape),
        reduced_parameters.paramvals_by_name,
    )
