# param_sweep.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
# This source code is licensed under the BSD-style license found in the LICENSE file
# in the root directory of this source tree.
# ###########################################################################

import copy
import functools
import inspect
import itertools
import warnings

from abc import ABC
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from qutip import Qobj
from typing_extensions import Literal

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
from scqubits.core.qubit_base import QuantumSystem, QubitBaseClass
from scqubits.core.spec_lookup import SpectrumLookupMixin
from scqubits.core.storage import SpectrumData

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from scqubits.utils.typedefs import GIndexTuple, NpIndices, QubitList

BareLabel = Tuple[int, ...]
DressedLabel = int
StateLabel = Union[DressedLabel, BareLabel]


class ParameterSlice:
    """
    Stores information about a 1d slice of a (possibly) multi-dimensional parameter
    sweep.

    Parameters
    ----------
    param_name:
        name of the single parameter which is being swept
    param_val:
        single selected value of the parameter (as used, e.g., in the Explorer)
    fixed_params:
        dictionary giving the names of the fixed parameters and their corresponding
        values
    params_ordered:
        list of all parameter names, giving their ordering
    """

    def __init__(
        self,
        param_name: str,
        param_val: float,
        fixed_params: Dict[str, float],
        params_ordered: List[str],
    ):
        self.param_name = param_name
        self.param_val = param_val
        self.fixed_dict = fixed_params
        self.all_dict = {param_name: param_val, **fixed_params}
        self.fixed = tuple(
            slice(name, value) for name, value in self.fixed_dict.items()
        )
        self.all = tuple(slice(name, value) for name, value in self.all_dict.items())
        self.all_values = [self.all_dict[name] for name in params_ordered]


class ParameterSweepBase(ABC, SpectrumLookupMixin):
    """
    The_ParameterSweepBase class is an abstract base class for ParameterSweep and
    StoredSweep
    """

    _lookup_exists = False
    _parameters = descriptors.WatchedProperty(Parameters, "PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty(int, "PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty(Dict[str, ndarray], "PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    _out_of_sync = False
    _current_param_indices: NpIndices

    @property
    def hilbertspace(self) -> HilbertSpace:
        return self._hilbertspace

    @property
    def parameters(self) -> Parameters:
        """Return the Parameter object (access parameter values/indexing)"""
        return self._parameters

    @property
    def param_info(self) -> Dict[str, ndarray]:
        """Return a dictionary of the parameter names and values used in this sweep."""
        return self._parameters.paramvals_by_name

    def get_subsys(self, index: int) -> QuantumSystem:
        return self.hilbertspace[index]

    def subsys_by_id_str(self, id_str: str) -> QuantumSystem:
        return self.hilbertspace.subsys_by_id_str(id_str)

    def subsys_evals_count(self, subsys_index: int) -> int:
        return self["bare_evals"]["subsys":subsys_index].shape[-1]

    def dressed_evals_count(self) -> int:
        """Returns number of dressed eigenvalues included in sweep."""
        return self._evals_count

    def get_subsys_index(self, subsys: QuantumSystem) -> int:
        return self.hilbertspace.get_subsys_index(subsys)

    @property
    def osc_subsys_list(self) -> List[Oscillator]:
        return self.hilbertspace.osc_subsys_list

    @property
    def qbt_subsys_list(self) -> List[QubitBaseClass]:
        return self.hilbertspace.qbt_subsys_list

    @property
    def subsystem_count(self) -> int:
        return self.hilbertspace.subsystem_count

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
            if event == "HILBERTSPACE_UPDATE" and sender is self.hilbertspace:
                self._out_of_sync = True
            elif event == "PARAMETERSWEEP_UPDATE" and sender is self:
                self._out_of_sync = True

    def set_update_func(self, update_hilbertspace: Callable) -> Callable:
        """Account for the two possible signatures of the `update_hilbertspace`
        function. Inspect whether a `self` argument is given. If not, return a
        function that accepts `self` as a dummy argument."""
        arguments = inspect.signature(update_hilbertspace)
        if len(arguments.parameters) == len(self._parameters) + 1:
            # update_hilbertspace function already includes self argument
            return update_hilbertspace

        # function is missing self argument; create function with self dummy variable
        def full_update_func(sweep: "ParameterSweep", *args):
            return update_hilbertspace(*args)

        return full_update_func

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
        sweep_param_indices = self.get_sweep_indices(multi_index)  # type:ignore
        if len(sweep_param_indices) != 1:
            raise ValueError(
                "All but one parameter must be fixed for `bare_specdata_list`."
            )
        sweep_param_name = self._parameters.name_by_index[sweep_param_indices[0]]
        specdata_list: List[SpectrumData] = []
        for subsys_index, subsystem in enumerate(self.hilbertspace):
            evals_swp = self["bare_evals"][subsys_index][multi_index]
            evecs_swp = self["bare_evecs"][subsys_index][multi_index]
            specdata_list.append(
                SpectrumData(
                    energy_table=evals_swp.toarray(),
                    state_table=evecs_swp.toarray(),
                    system_params=self.hilbertspace.get_initdata(),
                    param_name=sweep_param_name,
                    param_vals=self._parameters[sweep_param_name],
                )
            )
        self._preslicing_reset()
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
        sweep_param_indices = self.get_sweep_indices(multi_index)  # type:ignore
        if len(sweep_param_indices) != 1:
            raise ValueError(
                "All but one parameter must be fixed for `dressed_specdata`."
            )
        sweep_param_name = self._parameters.name_by_index[sweep_param_indices[0]]

        specdata = SpectrumData(
            energy_table=self["evals"][multi_index].toarray(),
            state_table=self["evecs"][multi_index].toarray(),
            system_params=self.hilbertspace.get_initdata(),
            param_name=sweep_param_name,
            param_vals=self._parameters[sweep_param_name],
        )
        self._preslicing_reset()
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
        self._preslicing_reset()
        return sweep_indices

    @property
    def system_params(self) -> Dict[str, Any]:
        return self.hilbertspace.get_initdata()

    def _preslicing_reset(self) -> None:
        self._current_param_indices = slice(None, None, None)

    def _slice_is_1d_sweep(self, param_indices: Optional[NpIndices]) -> bool:
        param_indices = param_indices or self._current_param_indices
        reduced_parameters = self._parameters.create_sliced(param_indices)
        return len(reduced_parameters) == 1

    def _final_states_for_subsys_transition(
        self, subsystem: QuantumSystem, initial_tuple: BareLabel
    ) -> List[BareLabel]:
        """For given initial state of the composite quantum system, return the final
        states possible to reach by changing the energy level of the given
        `subsystem`"""
        subsys_index = self.hilbertspace.get_subsys_index(subsystem)
        final_tuples_list = []

        for level in range(subsystem.truncated_dim):
            final_state = list(initial_tuple)
            final_state[subsys_index] = level
            final_tuples_list.append(tuple(final_state))
        final_tuples_list.remove(initial_tuple)
        return final_tuples_list

    def _get_final_states_list(
        self,
        initial_state: Union[BareLabel, DressedLabel],
        subsys_list: List[QuantumSystem],
        sidebands: bool,
    ) -> List[BareLabel]:
        """Construct and return the possible final states as a list, based on the
        provided initial state, a list of active subsystems and flag for whether to
        include sideband transitions."""
        if isinstance(initial_state, DressedLabel):
            return self._bare_product_states_labels

        if not sidebands:
            final_state_list = []
            for subsys in subsys_list:
                final_state_list += self._final_states_for_subsys_transition(
                    subsys, initial_state
                )
            return final_state_list

        range_list = [range(dim) for dim in self.hilbertspace.subsystem_dims]
        for subsys_index, subsys in enumerate(self.hilbertspace):
            if subsys not in subsys_list:
                range_list[subsys_index] = [initial_state[subsys_index]]  # type:ignore
        final_state_list = list(itertools.product(*range_list))
        return final_state_list

    def _complete_state(
        self,
        partial_state: BareLabel,
        subsys_list: List[QuantumSystem],
    ) -> BareLabel:
        """A partial state only includes entries for active subsystems. Complete this
        state by inserting 0 entries for all inactive subsystems."""
        state_full = [0] * len(self.hilbertspace)
        for entry, subsys in zip(partial_state, subsys_list):
            subsys_index = self.get_subsys_index(subsys)
            state_full[subsys_index] = entry
        return tuple(state_full)

    def _process_subsystems_option(
        self, subsystems: Optional[Union[QuantumSystem, List[QuantumSystem]]]
    ) -> List[QuantumSystem]:
        if subsystems is None:
            return self.hilbertspace.subsystem_list
        if isinstance(subsystems, list):
            return subsystems
        if isinstance(subsystems, QuantumSystem):
            return [subsystems]
        raise TypeError("Argument `subsystems` has invalid type.")

    def _process_initial_option(
        self,
        initial: Union[None, StateLabel],
        subsys_list: List[QuantumSystem],
    ) -> Tuple[bool, Callable, StateLabel]:
        if isinstance(initial, DressedLabel):
            initial_dressed = True
            return initial_dressed, self.energy_by_dressed_index, initial

        if initial is None:
            initial_dressed = False
            initial = (0,) * len(self.hilbertspace)
            return initial_dressed, self.energy_by_bare_index, initial

        initial_dressed = False
        if len(initial) not in [len(self.hilbertspace), len(subsys_list)]:
            raise ValueError(
                "State information provided is not compatible "
                "with the set of subsystems(s)."
            )
        if len(initial) < len(self.hilbertspace):
            initial = tuple(self._complete_state(initial, subsys_list))
        return initial_dressed, self.energy_by_bare_index, initial

    def _process_final_option(
        self,
        final: Union[None, StateLabel],
        initial: StateLabel,
        subsys_list: List[QuantumSystem],
        sidebands: bool,
    ) -> Tuple[bool, Callable, Union[List[DressedLabel], List[BareLabel]]]:
        if final is None:
            final_dressed = False
            final_states_list = self._get_final_states_list(
                initial, subsys_list, sidebands
            )
            return final_dressed, self.energy_by_bare_index, final_states_list
        if final == -1:
            final_dressed = True
            return (
                final_dressed,
                self.energy_by_dressed_index,
                list(range(0, self.dressed_evals_count())),
            )

        if isinstance(final, DressedLabel):
            final_dressed = True
            return final_dressed, self.energy_by_dressed_index, [final]

        if isinstance(final, tuple):
            final_dressed = False
            return final_dressed, self.energy_by_bare_index, [final]

        raise TypeError("Invalid type for final state.")

    def _validate_bare_initial(
        self,
        initial: BareLabel,
        initial_energies: NamedSlotsNdarray,
        param_indices: NpIndices,
    ) -> None:
        if np.isnan(initial_energies.toarray().astype(np.float_)).any():
            warnings.warn(
                "The initial state undergoes significant hybridization. "
                "Identification with a bare product state was not (fully) "
                "successful. Consider running ParameterSweep with "
                "`ignore_low_overlap=True` or specify `initial` as a dressed-state "
                "index (integer) instead of a bare product state.\n",
                UserWarning,
            )
        elif sum(initial) == 0 and not np.all(
            initial_energies == self["evals"][param_indices][..., 0]
        ):
            warnings.warn(
                "The state (0,0, ...,0) may not be dispersively connected "
                "to the true ground state. Specifying `initial=0` (dressed-state "
                "index) may be preferable.\n",
                UserWarning,
            )

    def _generate_transition_labels(
        self,
        initial_dressed: bool,
        final_dressed: bool,
        transitions: List[Tuple[StateLabel, StateLabel]],
    ) -> List[str]:
        identity_map = lambda x: x
        initial_label_func = (
            identity_map if initial_dressed else utils.tuple_to_short_str
        )
        final_label_func = identity_map if final_dressed else utils.tuple_to_short_str
        return [
            r"{}$\to${}".format(initial_label_func(initial), final_label_func(final))
            for initial, final in transitions
        ]

    @overload
    def transitions(
        self,
        as_specdata: Literal[True] = True,
        subsystems: Optional[Union[QuantumSystem, List[QuantumSystem]]] = None,
        initial: Optional[StateLabel] = None,
        final: Optional[StateLabel] = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> SpectrumData:
        ...

    @overload
    def transitions(
        self,
        as_specdata: Literal[False],
        subsystems: Optional[Union[QuantumSystem, List[QuantumSystem]]] = None,
        initial: Optional[StateLabel] = None,
        final: Optional[StateLabel] = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> Tuple[List[Tuple[StateLabel, StateLabel]], List[NamedSlotsNdarray]]:
        ...

    def transitions(
        self,
        as_specdata: bool = False,
        subsystems: Optional[Union[QuantumSystem, List[QuantumSystem]]] = None,
        initial: Optional[StateLabel] = None,
        final: Optional[StateLabel] = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> Union[
        Tuple[List[Tuple[StateLabel, StateLabel]], List[NamedSlotsNdarray]],
        SpectrumData,
    ]:
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
            initial state from which transitions originate, specified either (1) as a
            bare product state (tuple of excitation numbers of all subsystems or of the
            active ones given in `subsystems`); or (2) as a dressed-state index in
            the form of an integer >= 0.
            (default: (0,0,...,0) which is usually closest to the ground state)
        final:
            concrete final state for which the transition energy should be generated,
            given either as a bare product state (tuple of excitation numbers),
            or as a dressed state (non-negative integer). If `final` is omitted
            a list of final states is generated for dispersive transitions within each
            (active) subsystem. Sidebands can be switched on with the subsequent
            keyword option. `final=-1` can be chosen for a final state list to all
            other dressed states (helpful when the dispersive limit breaks down).
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
            If `as_specdata` is set to True, a SpectrumData object is returned instead,
            saving transition label info in an attribute named `labels`.
        """
        subsys_list = self._process_subsystems_option(subsystems)

        (
            initial_dressed,
            initial_energy_lookup_func,
            initial_state,
        ) = self._process_initial_option(initial, subsys_list)
        (
            final_dressed,
            final_energy_lookup_func,
            final_states_list,
        ) = self._process_final_option(final, initial_state, subsys_list, sidebands)

        transitions: List[Tuple[StateLabel, StateLabel]] = []
        transition_energies: List[NamedSlotsNdarray] = []

        param_indices = param_indices or self._current_param_indices
        _ = self[param_indices]  # trigger pre-slicing

        initial_energies = initial_energy_lookup_func(initial_state)
        if not initial_dressed:
            self._validate_bare_initial(initial_state, initial_energies, param_indices)

        for final_state in final_states_list:
            final_energies = final_energy_lookup_func(final_state)
            diff_energies = (final_energies - initial_energies).astype(float)
            diff_energies /= photon_number
            if make_positive:
                diff_energies = np.abs(diff_energies)
            if not np.isnan(diff_energies).all():  # omit transitions with all nans
                transitions.append((initial_state, final_state))
                transition_energies.append(diff_energies)

        self.reset_preslicing()

        if not as_specdata:
            return transitions, transition_energies

        label_list = self._generate_transition_labels(
            initial_dressed, final_dressed, transitions
        )

        if self._slice_is_1d_sweep(param_indices):
            reduced_parameters = self._parameters.create_sliced(param_indices)
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
        subsystems: Optional[Union[QuantumSystem, List[QuantumSystem]]] = None,
        initial: Optional[StateLabel] = None,
        final: Optional[StateLabel] = None,
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
            initial state from which transitions originate: the initial state
            can either be specified as a tuple referring to a bare product state,
            or as an integer representing the dressed state index. For bare product
            states, the required tuple has as many entries as the underlying
            `HilbertSpace` object has subsystems. (If `subsystems` is given, then the
            tuple may be reduced to entries for just these subsystems; other subsystems
            are given a "0" entry automatically.) The dressed state corresponding to the
            given bare product state is determined by considerations of overlaps.
            Note: for an initial dressed state, the `sidebands` option is ignored.
        final:
            concrete final state for which the transition energy should be generated,
            given either as a bare product state (tuple of excitation numbers),
            or as a dressed state (non-negative integer). If `final` is omitted
            a list of final states is generated for dispersive transitions within each
            (active) subsystem. Sidebands can be switched on with the subsequent
            keyword option. `final=-1` can be chosen for a final state list to all
            other dressed states (helpful when the dispersive limit breaks down).
        sidebands:
            if set to true, sideband transitions with multiple subsystems changing
            excitation levels are included (default: False). This option is ignored
            if `initial` is given as an integer dressed state index.
        photon_number:
            number of photons involved in transition; transition frequencies are divided
            by this number (default: photon_number=1, i.e., single-photon transitions)
        make_positive:
            boolean option relevant if the initial state is an excited state;
            downwards transition energies would regularly be negative, but are
            converted to positive if this flag is set to True (default: True)
        coloring:
            For `"transition"` (default), transitions are colored by their
            dispersive nature; for "`plain`" no selective highlighting is attempted.
        param_indices:
            usually to be omitted, as param_indices will be set via pre-slicing

        Returns
        -------
            Plot Figure and Axes objects
        """
        param_indices = param_indices or self._current_param_indices
        if not self._slice_is_1d_sweep(param_indices):
            raise ValueError(
                "Transition plots are only supported for a sweep over a "
                "single parameter. You can reduce a multi-dimensional "
                "sweep by pre-slicing, e.g.,  <ParameterSweep>[0, :, "
                "0].plot_transitions(...)"
            )

        specdata_for_highlighting = self.transitions(
            subsystems=subsystems,
            initial=initial,
            final=final,
            sidebands=sidebands,
            photon_number=photon_number,
            make_positive=make_positive,
            as_specdata=True,
            param_indices=param_indices,
        )

        specdata_all = copy.deepcopy(self[param_indices].dressed_specdata)
        specdata_all.energy_table -= specdata_for_highlighting.subtract  # type:ignore
        specdata_all.energy_table /= photon_number
        if make_positive:
            specdata_all.energy_table = np.abs(specdata_all.energy_table)

        self.reset_preslicing()

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
        fig, axes = specdata_for_highlighting.plot_evals_vs_paramvals(
            label_list=specdata_for_highlighting.labels,
            fig_ax=fig_ax,
            **kwargs,  # type:ignore
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
        subsystem: QuantumSystem = None,
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
                sweeps.bare_matrixelement,
                operator_name=operator,
                subsystem=subsystem,
            )
        elif isinstance(operator, Qobj):
            operator_func = functools.partial(
                sweeps.dressed_matrixelement,
                operator=operator,
            )
        else:
            raise TypeError(
                "Unrecognized type of operator for matrix elements; "
                "expected: str or Qobj."
            )

        matrix_element_data = generator(
            self,
            operator_func,
        )
        self._data[sweep_name] = matrix_element_data


class ParameterSweep(  # type:ignore
    ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable
):
    """
    Create multi-dimensional parameter sweeps for a quantum system described by a
    `HilbertSpace` object.

    Parameters
    ----------
    hilbertspace:
        `HilbertSpace` object describing the quantum system of interest
    paramvals_by_name:
        Dictionary which specifies a parameter name for each set of parameter values,
        and the set of values to be used in the sweep.
    update_hilbertspace:
        function that updates the associated `hilbertspace` object with a given
        set of parameters; signature is either
        `update_hilbertspace(paramval1, paramval2, ...)`
        or
        `update_hilbertspace(self, paramval1, paramval2, ...)`
        where `self` makes the `ParameterSweep` instance available, and thereby
        dict-like access to subsystems and interaction terms
    evals_count:
        number of dressed eigenvalues/eigenstates to keep. (The number of bare
        eigenvalues/eigenstates is determined for each subsystem by `truncated_dim`.)
        (default: 20)
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
    ignore_low_overlap:
        if set to False (default), bare product states and dressed eigenstates are
        identified if `|<psi_bare|psi_dressed>|^2 > 0.5`; if True,
        then identification will always take place based on which bare product state
        has the maximum overlap
    autorun:
        Determines whether to directly run the sweep or delay it until `.run()` is
        called manually. (Default: `settings.AUTORUN_SWEEP=True`)
    deepcopy:
        if set to True, the parameter sweep is run with an exact copy of the Hilbert
        space; this ensures that all parameters after the sweep are identical to
        parameters before the sweep. Note: changing global HilbertSpace or
        QuantumSystem attributes will have no effect with this option; all updates
        must be made via `<ParameterSweep>.hilbertspace[<id_str>] = ...` If
        set to False (default), updates to global instances have the expected effect.
        The HilbertSpace object and all its constituent parts are left in the state
        reached by the very final parameter update.
    num_cpus:
        number of CPU cores requested for computing the sweep
        (default value `settings.NUM_CPUS`)


    Notes
    -----
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
    """

    def __init__(
        self,
        hilbertspace: HilbertSpace,
        paramvals_by_name: Dict[str, ndarray],
        update_hilbertspace: Callable,
        evals_count: int = 20,
        subsys_update_info: Optional[Dict[str, List[QuantumSystem]]] = None,
        bare_only: bool = False,
        ignore_low_overlap: bool = False,
        autorun: bool = settings.AUTORUN_SWEEP,
        deepcopy: bool = False,
        num_cpus: Optional[int] = None,
    ) -> None:
        num_cpus = num_cpus or settings.NUM_CPUS
        self._parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._evals_count = evals_count
        self._update_hilbertspace = self.set_update_func(update_hilbertspace)
        self._subsys_update_info = subsys_update_info
        self._data: Dict[str, Any] = {}
        self._bare_only = bare_only
        self._ignore_low_overlap = ignore_low_overlap
        self._deepcopy = deepcopy
        self._num_cpus = num_cpus
        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        self._out_of_sync = False
        self.reset_preslicing()

        dispatch.CENTRAL_DISPATCH.register("PARAMETERSWEEP_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("HILBERTSPACE_UPDATE", self)

        if autorun:
            self.run()

    def cause_dispatch(self) -> None:
        initial_parameters = tuple(paramvals[0] for paramvals in self._parameters)
        self._update_hilbertspace(self, *initial_parameters)

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

    def run(self) -> None:
        """Create all sweep data: bare spectral data, dressed spectral data, lookup
        data and custom sweep data."""
        # generate one dispatch before temporarily disabling CENTRAL_DISPATCH

        self._lookup_exists = True
        if self._deepcopy:
            stored_hilbertspace = copy.deepcopy(self.hilbertspace)
            self._hilbertspace = copy.deepcopy(self.hilbertspace)
        else:
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
        if self._deepcopy:
            self._hilbertspace = stored_hilbertspace  # restore original state
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

        for subsys_index, subsystem in enumerate(self.hilbertspace):
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
        self,
        update_func: Callable,
        subsystem: QuantumSystem,
        paramval_tuple: Tuple[float],
    ) -> ndarray:
        update_func(self, *paramval_tuple)
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
            "Parallel compute bare eigensys for subsystem {} [num_cpus={}]".format(
                subsystem.id_str, self._num_cpus
            ),
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
        update_func(self, *paramval_tuple)

        assert self._data is not None
        bare_esys: Dict[int, List[ndarray]] = {
            subsys_index: [
                self._data["bare_evals"][subsys_index][paramindex_tuple],
                self._data["bare_evecs"][subsys_index][paramindex_tuple],
            ]
            for subsys_index, _ in enumerate(self.hilbertspace)
        }

        evals, evecs = hilbertspace.eigensys(
            evals_count=evals_count, bare_esys=bare_esys  # type:ignore
        )
        esys_array = np.empty(shape=(2,), dtype=object)
        esys_array[0] = evals
        esys_array[1] = evecs
        return esys_array

    def _dressed_spectrum_sweep(
        self,
    ) -> Tuple[NamedSlotsNdarray, NamedSlotsNdarray]:
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

        spectrum_data_ndarray = np.asarray(spectrum_data, dtype=object)
        spectrum_data_ndarray = spectrum_data_ndarray.reshape(
            (*self._parameters.counts, 2)
        )
        slotparamvals_by_name = OrderedDict(self._parameters.ordered_dict.copy())

        evals = np.asarray(spectrum_data_ndarray[..., 0].tolist())
        evecs = spectrum_data_ndarray[..., 1]

        return (
            NamedSlotsNdarray(evals, slotparamvals_by_name),
            NamedSlotsNdarray(evecs, slotparamvals_by_name),
        )

    def _energies_1(self, subsys):
        bare_label = np.zeros(len(self.hilbertspace))
        bare_label[self.get_subsys_index(subsys)] = 1

        energies_all_l = np.empty(self._parameters.counts + (subsys.truncated_dim,))
        for l in range(subsys.truncated_dim):
            energies_all_l[..., l] = self[:].energy_by_bare_index(tuple(l * bare_label))
        return energies_all_l

    def _energies_2(self, subsys1, subsys2):
        bare_label1 = np.zeros(len(self.hilbertspace))
        bare_label1[self.get_subsys_index(subsys1)] = 1
        bare_label2 = np.zeros(len(self.hilbertspace))
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
        chi_data = np.empty((self.subsystem_count, self.subsystem_count), dtype=object)

        # Lamb shifts
        for subsys_index1, subsys1 in enumerate(self.hilbertspace):
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

        # Kerr and ac Stark
        for subsys_index1, subsys1 in enumerate(self.hilbertspace):
            energy_subsys1_all_l1 = self._energies_1(subsys1)
            for subsys_index2, subsys2 in enumerate(self.hilbertspace):
                energy_subsys2_all_l2 = self._energies_1(subsys2)
                energy_subsys1_subsys2_all_l1_l2 = self._energies_2(subsys1, subsys2)
                kerr_subsys1_subsys2_all_l1_l2 = (
                    energy_subsys1_subsys2_all_l1_l2
                    + energy_0[..., None, None]
                    - energy_subsys1_all_l1[..., :, None]
                    - energy_subsys2_all_l2[..., None, :]
                )

                # self-Kerr and cross-Kerr: oscillator modes
                if subsys1 in self.osc_subsys_list and subsys2 in self.osc_subsys_list:
                    if subsys1 is subsys2:
                        # oscillator self-Kerr
                        kerr_subsys1_subsys2_all_l1_l2 /= 2.0  # osc self-Kerr: 1/2
                    kerr_data[subsys_index1, subsys_index2] = NamedSlotsNdarray(
                        kerr_subsys1_subsys2_all_l1_l2[..., 1, 1],
                        self._parameters.paramvals_by_name,
                    )
                    chi_data[subsys_index1, subsys_index2] = np.asarray([])
                # self-Kerr and cross-Kerr: qubit modes
                elif (
                    subsys1 in self.qbt_subsys_list and subsys2 in self.qbt_subsys_list
                ):
                    kerr_data[subsys_index1, subsys_index2] = NamedSlotsNdarray(
                        kerr_subsys1_subsys2_all_l1_l2,
                        self._parameters.paramvals_by_name,
                    )
                    chi_data[subsys_index1, subsys_index2] = np.asarray([])
                # ac Stark shifts
                else:
                    if subsys1 in self.qbt_subsys_list:
                        chi_data[subsys_index1, subsys_index2] = NamedSlotsNdarray(
                            kerr_subsys1_subsys2_all_l1_l2[..., 1, :],
                            self._parameters.paramvals_by_name,
                        )
                    else:
                        chi_data[subsys_index1, subsys_index2] = NamedSlotsNdarray(
                            kerr_subsys1_subsys2_all_l1_l2[..., :, 1],
                            self._parameters.paramvals_by_name,
                        )
                    kerr_data[subsys_index1, subsys_index2] = np.asarray([])

        sys_indices = np.arange(self.subsystem_count)
        lamb_data = NamedSlotsNdarray(lamb_data, {"subsys": sys_indices})
        kerr_data = NamedSlotsNdarray(
            kerr_data, {"subsys1": sys_indices, "subsys2": sys_indices}
        )
        chi_data = NamedSlotsNdarray(
            chi_data, {"subsys1": sys_indices, "subsys2": sys_indices}
        )

        return lamb_data, chi_data, kerr_data


class StoredSweep(
    ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable
):
    _parameters = descriptors.WatchedProperty(Parameters, "PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty(int, "PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty(Dict[str, ndarray], "PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    def __init__(
        self,
        paramvals_by_name: Dict[str, ndarray],
        hilbertspace: HilbertSpace,
        evals_count: int,
        _data,
    ) -> None:
        self._lookup_exists = True
        self._parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._evals_count = evals_count
        self._data = _data

        self._out_of_sync = False
        self._current_param_indices: NpIndices = slice(None, None, None)

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

    def get_hilbertspace(self) -> HilbertSpace:
        return self.hilbertspace

    def new_sweep(
        self,
        paramvals_by_name: Dict[str, ndarray],
        update_hilbertspace: Callable,
        evals_count: int = 6,
        subsys_update_info: Optional[Dict[str, List[QuantumSystem]]] = None,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: Optional[int] = None,
    ) -> ParameterSweep:
        return ParameterSweep(
            self.hilbertspace,
            paramvals_by_name,
            update_hilbertspace,
            evals_count=evals_count,
            subsys_update_info=subsys_update_info,
            autorun=autorun,
            num_cpus=num_cpus,
        )


def generator(sweep: "ParameterSweepBase", func: Callable, **kwargs) -> np.ndarray:
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
                functools.partial(
                    func_effective,
                    params=reduced_parameters,
                    **kwargs,
                ),
                itertools.product(*reduced_parameters.ranges),
            ),
            total=total_count,
            desc=f"sweeping {func_name}",
            leave=False,
            disable=settings.PROGRESSBAR_DISABLED,
        )
    )
    element_shape: Tuple[int, ...] = tuple()
    if isinstance(data_array[0], np.ndarray):
        element_shape = data_array[0].shape

    data_ndarray = np.asarray(data_array)
    return NamedSlotsNdarray(
        data_ndarray.reshape(reduced_parameters.counts + element_shape),
        reduced_parameters.paramvals_by_name,
    )
