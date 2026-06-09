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

from __future__ import annotations

import copy
import functools
import inspect
import itertools
import warnings

from abc import ABC
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import dill
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from qutip import Qobj
from scipy.sparse import csc_matrix

import scqubits as scq
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

from tqdm.auto import tqdm

from scqubits.utils.typedefs import GIndexTuple, NpIndices

BareLabel = tuple[int, ...]
DressedLabel = int
StateLabel = DressedLabel | BareLabel

_faulty_interactionterm_warning_issued = False  # flag to ensure single-time warning


class ParameterSlice:
    """Store information about a 1d slice of a multi-dimensional parameter sweep.

    Parameters
    ----------
    param_name:
        name of the single parameter being swept
    param_val:
        selected value of the parameter (as used, e.g., in the Explorer)
    fixed_params:
        dictionary mapping fixed parameter names to their values
    params_ordered:
        list of all parameter names in their canonical order
    """

    def __init__(
        self,
        param_name: str,
        param_val: float,
        fixed_params: dict[str, float],
        params_ordered: list[str],
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
    """Abstract base class for :class:`ParameterSweep` and :class:`StoredSweep`."""

    _lookup_exists = False
    _parameters = descriptors.WatchedProperty(Parameters, "PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty(int, "PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty(dict[str, ndarray], "PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    _out_of_sync = False
    _out_of_sync_warning_issued = False
    _current_param_indices: NpIndices

    @property
    def hilbertspace(self) -> HilbertSpace:
        """Return the underlying :class:`HilbertSpace` instance."""
        return self._hilbertspace

    @property
    def parameters(self) -> Parameters:
        """Return the :class:`.Parameters` object (access parameter values/indexing)."""
        return self._parameters

    @property
    def param_info(self) -> dict[str, ndarray]:
        """Return a dictionary of the parameter names and values used in this sweep."""
        return self._parameters.paramvals_by_name

    def get_subsys(self, index: int) -> QuantumSystem:
        """Return the subsystem of the underlying :class:`HilbertSpace` at ``index``.

        Parameters
        ----------
        index:
            position of the subsystem in the :class:`HilbertSpace` subsystem list
        """
        return self.hilbertspace[index]

    def subsys_by_id_str(self, id_str: str) -> QuantumSystem:
        """Return the subsystem identified by its ``id_str`` attribute.

        Parameters
        ----------
        id_str:
            identifier string of the subsystem to be retrieved
        """
        return self.hilbertspace.subsys_by_id_str(id_str)

    def subsys_evals_count(self, subsys_index: int) -> int:
        """Return the number of bare eigenvalues stored for the given subsystem.

        Parameters
        ----------
        subsys_index:
            index of the subsystem within the :class:`HilbertSpace` subsystem list
        """
        return self["bare_evals"]["subsys":subsys_index].shape[-1]  # type: ignore[misc]

    def dressed_evals_count(self) -> int:
        """Return the number of dressed eigenvalues included in the sweep."""
        return self._evals_count

    def get_subsys_index(self, subsys: QuantumSystem) -> int:
        """Return the index of ``subsys`` in the underlying :class:`HilbertSpace`.

        Parameters
        ----------
        subsys:
            subsystem whose index is requested
        """
        return self.hilbertspace.get_subsys_index(subsys)  # type: ignore[arg-type]

    @property
    def osc_subsys_list(self) -> list[Oscillator]:
        """Return the list of oscillator subsystems of the :class:`HilbertSpace`."""
        return self.hilbertspace.osc_subsys_list

    @property
    def qbt_subsys_list(self) -> list[QubitBaseClass]:
        """Return the list of qubit subsystems of the :class:`HilbertSpace`."""
        return self.hilbertspace.qbt_subsys_list  # type: ignore[return-value]

    @property
    def subsystem_count(self) -> int:
        """Return the number of subsystems of the underlying :class:`HilbertSpace`."""
        return self.hilbertspace.subsystem_count

    @utils.check_sync_status
    def __getitem__(self, key):
        """Return stored data by string key, or set up pre-slicing for index keys.

        If ``key`` is a string, return the corresponding entry of the internal
        data dictionary (e.g., ``"evals"``, ``"evecs"``, ``"bare_evals"``).
        Otherwise, ``key`` is interpreted as a numpy-style index and stored as
        the current parameter pre-slicing; ``self`` is then returned to support
        chained access such as ``<Sweep>[p1, p2, ...].dressed_eigenstates()``.

        Note: this method mutates ``self._current_param_indices`` as a side
        effect of pre-slicing.

        Parameters
        ----------
        key:
            string label, or numpy-style index (tuple, slice, integer) used for
            pre-slicing
        """
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
        """Hook for ``CENTRAL_DISPATCH``.

        Called by the global ``CentralDispatch`` instance whenever an event
        occurs that :class:`ParameterSweep` is registered for. In response to
        update events, the lookup table is marked as out of sync by setting
        ``self._out_of_sync = True``.

        Parameters
        ----------
        event:
            type of event being received
        sender:
            sender announcing the event
        **kwargs:
            additional keyword arguments forwarded by the dispatcher (unused)
        """
        if self._data:
            if event == "HILBERTSPACE_UPDATE" and sender is self.hilbertspace:
                self._out_of_sync = True
            elif event == "PARAMETERSWEEP_UPDATE" and sender is self:
                self._out_of_sync = True

    def set_update_func(self, update_hilbertspace: Callable) -> Callable:
        """Account for the two possible signatures of ``update_hilbertspace``.

        Inspect whether a ``self`` argument is given; if not, return a wrapper
        that accepts ``self`` as a dummy argument.

        Parameters
        ----------
        update_hilbertspace:
            user-provided callable used to update the underlying
            :class:`HilbertSpace` for a given set of parameter values; signature
            is either ``update_hilbertspace(paramval1, ...)`` or
            ``update_hilbertspace(self, paramval1, ...)``
        """
        arguments = inspect.signature(update_hilbertspace)
        if len(arguments.parameters) == len(self._parameters) + 1:
            # update_hilbertspace function already includes self argument
            return update_hilbertspace

        # function is missing self argument; create function with self dummy variable
        def full_update_func(sweep: "ParameterSweep", *args):
            return update_hilbertspace(*args)

        return full_update_func

    @property
    def bare_specdata_list(self) -> list[SpectrumData]:
        """Wrap bare eigensystem data into :class:`SpectrumData` objects.

        To be used with pre-slicing, e.g.
        ``<ParameterSweep>[0, :].bare_specdata_list``.

        Returns
        -------
        List of :class:`SpectrumData` objects with bare eigensystem data, one
        per subsystem.
        """
        multi_index = self._current_param_indices
        sweep_param_indices = self.get_sweep_indices(multi_index)  # type: ignore[arg-type]
        if len(sweep_param_indices) != 1:
            raise ValueError(
                "All but one parameter must be fixed for `bare_specdata_list`."
            )
        sweep_param_name = self._parameters.name_by_index[sweep_param_indices[0]]
        specdata_list: list[SpectrumData] = []
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
        self.reset_preslicing()
        return specdata_list

    @property
    def dressed_specdata(self) -> "SpectrumData":
        """Wrap dressed eigensystem data into a :class:`SpectrumData` object.

        To be used with pre-slicing, e.g.
        ``<ParameterSweep>[0, :].dressed_specdata``.

        Returns
        -------
        :class:`SpectrumData` object with dressed eigensystem data.
        """
        multi_index = self._current_param_indices
        sweep_param_indices = self.get_sweep_indices(multi_index)  # type: ignore[arg-type]
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
        self.reset_preslicing()
        return specdata

    def get_sweep_indices(self, multi_index: GIndexTuple) -> list[int]:
        """Return the parameter indices being swept for the given multi-index.

        For a given generalized multi-index, return a list of the indices that
        are being swept (i.e., for which the selection corresponds to more than
        a single value).

        Parameters
        ----------
        multi_index:
            generalized numpy-style multi-index over the parameter grid
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
        self.reset_preslicing()
        return sweep_indices

    @property
    def system_params(self) -> dict[str, Any]:
        """Return the initialization data of the underlying :class:`HilbertSpace`."""
        return self.hilbertspace.get_initdata()

    def _slice_is_1d_sweep(self, param_indices: NpIndices | None) -> bool:
        """Return whether the given pre-slicing reduces the sweep to one parameter.

        Parameters
        ----------
        param_indices:
            numpy-style index over the parameter grid; if ``None``, the current
            pre-slicing stored on the instance is used
        """
        param_indices = param_indices or self._current_param_indices
        reduced_parameters = self._parameters.create_sliced(param_indices)
        return len(reduced_parameters) == 1

    def _final_states_for_subsys_transition(
        self, subsystem: QuantumSystem, initial_tuple: BareLabel
    ) -> list[BareLabel]:
        """Return the final states reachable by changing one subsystem's level.

        For the given initial state of the composite quantum system, return the
        final states possible to reach by changing the energy level of the
        given ``subsystem``.

        Parameters
        ----------
        subsystem:
            subsystem whose excitation level is varied
        initial_tuple:
            bare product-state label of the initial state
        """
        subsys_index = self.hilbertspace.get_subsys_index(subsystem)  # type: ignore[arg-type]
        final_tuples_list = []

        for level in range(subsystem.truncated_dim):
            final_state = list(initial_tuple)
            final_state[subsys_index] = level
            final_tuples_list.append(tuple(final_state))
        final_tuples_list.remove(initial_tuple)
        return final_tuples_list

    def _get_final_states_list(
        self,
        initial_state: BareLabel | DressedLabel,
        subsys_list: list[QuantumSystem],
        sidebands: bool,
    ) -> list[BareLabel]:
        """Construct and return the possible final states as a list.

        The list is built from the provided initial state, the list of active
        subsystems, and a flag for whether to include sideband transitions.

        Parameters
        ----------
        initial_state:
            initial state, given either as a bare product-state label or as a
            dressed-state index
        subsys_list:
            list of subsystems considered active for the transitions
        sidebands:
            if ``True``, include sideband transitions in which multiple
            subsystems change excitation levels
        """
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
                range_list[subsys_index] = [initial_state[subsys_index]]  # type: ignore[call-overload]
        final_state_list = list(itertools.product(*range_list))
        return final_state_list

    def _complete_state(
        self,
        partial_state: BareLabel,
        subsys_list: list[QuantumSystem],
    ) -> BareLabel:
        """Complete a partial bare state by inserting zeros for inactive subsystems.

        A partial state only includes entries for active subsystems. This
        method inserts ``0`` entries for all inactive subsystems.

        Parameters
        ----------
        partial_state:
            tuple of excitation numbers for the active subsystems only
        subsys_list:
            list of active subsystems matching the entries of ``partial_state``
        """
        state_full = [0] * len(self.hilbertspace)
        for entry, subsys in zip(partial_state, subsys_list):
            subsys_index = self.get_subsys_index(subsys)
            state_full[subsys_index] = entry
        return tuple(state_full)

    def _process_subsystems_option(
        self, subsystems: QuantumSystem | list[QuantumSystem] | None
    ) -> list[QuantumSystem]:
        """Normalize the ``subsystems`` argument to a list of subsystems.

        Parameters
        ----------
        subsystems:
            a single subsystem, a list of subsystems, or ``None``; ``None`` is
            interpreted as "all subsystems of the underlying HilbertSpace"
        """
        if subsystems is None:
            return self.hilbertspace.subsystem_list  # type: ignore[return-value]
        if isinstance(subsystems, list):
            return subsystems
        if isinstance(subsystems, QuantumSystem):
            return [subsystems]
        raise TypeError("Argument `subsystems` has invalid type.")

    def _process_initial_option(
        self,
        initial: StateLabel | list[tuple[int]] | None,
        subsys_list: list[QuantumSystem],
    ) -> tuple[bool, Callable, StateLabel]:
        """Normalize the ``initial`` state and return a (dressed?, lookup, state) triple.

        Parameters
        ----------
        initial:
            initial state given as a bare product-state label, a dressed-state
            index, or ``None`` (defaulting to the all-zeros bare state)
        subsys_list:
            list of active subsystems used to complete a partial bare state

        Returns
        -------
        Tuple ``(initial_dressed, energy_lookup_func, initial_state)`` where
        ``initial_dressed`` flags whether the initial state is given as a
        dressed-state index, ``energy_lookup_func`` is the matching energy
        lookup, and ``initial_state`` is the (possibly completed) state label.
        """
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
            initial = tuple(self._complete_state(initial, subsys_list))  # type: ignore[arg-type]
        return initial_dressed, self.energy_by_bare_index, initial  # type: ignore[return-value]

    def _process_final_option(
        self,
        final: StateLabel | list[tuple[int]] | None,
        initial: StateLabel,
        subsys_list: list[QuantumSystem],
        sidebands: bool,
    ) -> tuple[bool, Callable, list[DressedLabel] | list[BareLabel]]:
        """Normalize the ``final`` option and return a (dressed?, lookup, states) triple.

        Parameters
        ----------
        final:
            specifies the final state(s); may be a bare product-state label, a
            dressed-state index, ``-1`` (all dressed states), or ``None``
            (auto-generate dispersive transitions)
        initial:
            initial state, used when auto-generating final states
        subsys_list:
            list of active subsystems used when auto-generating final states
        sidebands:
            if ``True``, include sideband transitions when auto-generating

        Returns
        -------
        Tuple ``(final_dressed, energy_lookup_func, final_states_list)`` where
        ``final_dressed`` flags whether final states are dressed-state indices,
        ``energy_lookup_func`` is the matching energy lookup, and
        ``final_states_list`` is the list of final state labels.
        """
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
        """Warn if a bare initial state cannot be cleanly identified with a dressed one.

        The energy array is cast to ``float64`` to test for NaNs (the cast
        drops any imaginary part, but eigenenergies are stored as real
        values).

        Parameters
        ----------
        initial:
            bare product-state label of the initial state
        initial_energies:
            sweep of energies associated with ``initial``
        param_indices:
            current pre-slicing into the parameter grid
        """
        if np.isnan(initial_energies.toarray().astype(np.float64)).any():
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
        transitions: list[tuple[StateLabel, StateLabel]],
    ) -> list[str]:
        """Build human-readable arrow-style labels for each transition.

        Parameters
        ----------
        initial_dressed:
            whether the initial states are given as dressed-state indices
        final_dressed:
            whether the final states are given as dressed-state indices
        transitions:
            list of ``(initial, final)`` state-label pairs to be labeled
        """
        identity_map = lambda x: x
        initial_label_func = (
            identity_map if initial_dressed else utils.tuple_to_short_str
        )
        final_label_func = identity_map if final_dressed else utils.tuple_to_short_str
        return [
            r"{}$\to${}".format(initial_label_func(initial), final_label_func(final))  # type: ignore[arg-type]
            for initial, final in transitions
        ]

    @overload
    def transitions(
        self,
        as_specdata: Literal[True] = True,
        subsystems: QuantumSystem | list[QuantumSystem] | None = None,
        initial: StateLabel | None = None,
        final: StateLabel | None = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        param_indices: NpIndices | None = None,
    ) -> SpectrumData:
        """Overload: when ``as_specdata=True``, return a :class:`SpectrumData`.

        Parameters
        ----------
        as_specdata:
            literal ``True`` selecting this overload
        subsystems:
            see :meth:`transitions` for full description
        initial:
            see :meth:`transitions` for full description
        final:
            see :meth:`transitions` for full description
        sidebands:
            see :meth:`transitions` for full description
        photon_number:
            see :meth:`transitions` for full description
        make_positive:
            see :meth:`transitions` for full description
        param_indices:
            see :meth:`transitions` for full description
        """
        ...

    @overload
    def transitions(
        self,
        as_specdata: Literal[False],
        subsystems: QuantumSystem | list[QuantumSystem] | None = None,
        initial: StateLabel | list[tuple[int]] | None = None,
        final: StateLabel | list[tuple[int]] | None = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        param_indices: NpIndices | None = None,
    ) -> tuple[list[tuple[StateLabel, StateLabel]], list[NamedSlotsNdarray]]:
        """Overload: when ``as_specdata=False``, return ``(transitions, energies)``.

        Parameters
        ----------
        as_specdata:
            literal ``False`` selecting this overload
        subsystems:
            see :meth:`transitions` for full description
        initial:
            see :meth:`transitions` for full description
        final:
            see :meth:`transitions` for full description
        sidebands:
            see :meth:`transitions` for full description
        photon_number:
            see :meth:`transitions` for full description
        make_positive:
            see :meth:`transitions` for full description
        param_indices:
            see :meth:`transitions` for full description
        """
        ...

    def transitions(
        self,
        as_specdata: bool = False,
        subsystems: QuantumSystem | list[QuantumSystem] | None = None,
        initial: StateLabel | list[tuple[int]] | None = None,
        final: StateLabel | list[tuple[int]] | None = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = False,
        param_indices: NpIndices | None = None,
    ) -> (
        tuple[list[tuple[StateLabel, StateLabel]], list[NamedSlotsNdarray]]
        | SpectrumData
    ):
        """Extract transition energies from dressed eigenenergies using bare-state lookup.

        Usage is based on pre-slicing to select all or a subset of parameters
        to be involved in the sweep, e.g.,

        ``<ParameterSweep>[0, :, 2].transitions()``

        produces all eigenenergy differences for transitions starting in the
        ground state (the default when no initial state is specified) as a
        function of the middle parameter, with parameters 1 and 3 fixed by the
        indices 0 and 2.

        Note: difference energies are cast to real-valued floats; any imaginary
        residual is discarded.

        Parameters
        ----------
        subsystems:
            single subsystem or list of subsystems considered "active" for the
            transitions to be generated; if omitted, all subsystems are treated
            as actively participating
        initial:
            initial state from which transitions originate, specified either
            (1) as a bare product state (tuple of excitation numbers of all
            subsystems, or of the active ones given in ``subsystems``), or
            (2) as a dressed-state index (integer ``>= 0``).
            (default: ``(0, 0, ..., 0)``, usually closest to the ground state)
        final:
            concrete final state for which the transition energy should be
            generated, given either as a bare product state (tuple of
            excitation numbers) or as a dressed state (non-negative integer).
            If ``final`` is omitted, a list of final states is generated for
            dispersive transitions within each (active) subsystem; sidebands
            can be switched on with the ``sidebands`` keyword. ``final=-1``
            selects all other dressed states as final states (helpful when the
            dispersive limit breaks down).
        sidebands:
            if ``True``, include sideband transitions in which multiple
            subsystems change excitation levels (default: ``False``)
        photon_number:
            number of photons involved in the transition; transition
            frequencies are divided by this number (default: ``1``, i.e.,
            single-photon transitions)
        make_positive:
            relevant when the initial state is an excited state; downward
            transition energies are normally negative but are converted to
            positive values when this flag is ``True``
        as_specdata:
            if ``True``, return a :class:`SpectrumData` object; otherwise
            return raw arrays (default: ``False``)
        param_indices:
            usually omitted, as ``param_indices`` is set via pre-slicing

        Returns
        -------
        A tuple consisting of a list of transitions and a corresponding list
        of difference energies, e.g.
        ``((0,0,0), (0,0,1)), <energy array for transition 0,0,0 -> 0,0,1>``.
        If ``as_specdata`` is ``True``, a :class:`SpectrumData` object is
        returned instead, with transition label info stored in an attribute
        named ``labels``.
        """
        subsys_list = self._process_subsystems_option(subsystems)
        initial_states = initial if isinstance(initial, list) else [initial]  # type: ignore[list-item]

        final_states = final if isinstance(final, list) else [final]  # type: ignore[list-item]

        transitions: list[tuple[StateLabel, StateLabel]] = []
        transition_energies: list[NamedSlotsNdarray] = []
        for initial in initial_states:
            initial_dressed, initial_energy_lookup_func, initial_state = (
                self._process_initial_option(initial, subsys_list)
            )
            for final in final_states:
                final_dressed, final_energy_lookup_func, final_states_list = (
                    self._process_final_option(
                        final, initial_state, subsys_list, sidebands
                    )
                )

                param_indices = param_indices or self._current_param_indices
                _ = self[param_indices]  # trigger pre-slicing

                initial_energies = initial_energy_lookup_func(initial_state)
                if not initial_dressed:
                    self._validate_bare_initial(
                        initial_state, initial_energies, param_indices  # type: ignore[arg-type]
                    )

                for final_state in final_states_list:
                    final_energies = final_energy_lookup_func(final_state)
                    diff_energies = (final_energies - initial_energies).astype(float)
                    diff_energies /= photon_number
                    if make_positive:
                        diff_energies = np.abs(diff_energies)
                    if not np.isnan(
                        diff_energies
                    ).all():  # omit transitions with all nans
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

    def _validate_states(
        self,
        initial: StateLabel | list[tuple[int, ...]] | None = None,
        final: StateLabel | list[tuple[int, ...]] | None = None,
    ) -> None:
        """Validate initial and final state tuples against the HilbertSpace dimensions.

        Ensure that each state tuple, initial or final, is correctly structured
        and within the valid range for the quantum system's dimensions. State
        arguments that are not already lists are wrapped into lists for
        validation.

        Parameters
        ----------
        initial:
            initial state(s) to be validated; either a single state or a list
            of states. Each state is either a ``StateLabel`` or a tuple of
            subsystem excitation numbers.
        final:
            final state(s) to be validated, with the same structure as
            ``initial``.

        Raises
        ------
        ValueError
            If any tuple length does not match the number of subsystems, or if
            any tuple value exceeds the maximum allowed dimension of the
            corresponding subsystem.
        """
        initial = initial if isinstance(initial, list) else [initial]  # type: ignore[list-item]
        final = final if isinstance(final, list) else [final]  # type: ignore[list-item]
        # Dressed-state indices (``int`` form of ``StateLabel``) bypass the
        # tuple-shape checks; their bounds are enforced downstream when
        # ``_process_initial_option`` / ``_process_final_option`` consult
        # the dressed-state lookup tables.  The tuple checks below apply
        # only to bare-product-state labels.
        for initial_state, final_state in itertools.product(initial, final):
            if isinstance(initial_state, tuple):
                if len(initial_state) != len(self.hilbertspace.subsystem_dims):
                    raise ValueError(
                        "Initial state tuple does not match the number of subsystems."
                    )
                if max(initial_state) >= max(self.hilbertspace.subsystem_dims):
                    raise ValueError(
                        "Initial state tuple exceeds subsystem dimensions."
                    )
            if isinstance(final_state, tuple):
                if len(final_state) != len(self.hilbertspace.subsystem_dims):
                    raise ValueError(
                        "Final state tuple does not match the number of subsystems."
                    )
                if max(final_state) >= max(self.hilbertspace.subsystem_dims):
                    raise ValueError("Final state tuple exceeds subsystem dimensions.")

    def plot_transitions(
        self,
        subsystems: QuantumSystem | list[QuantumSystem] | None = None,
        initial: StateLabel | list[tuple[int, ...]] | None = None,
        final: StateLabel | list[tuple[int, ...]] | None = None,
        sidebands: bool = False,
        photon_number: int = 1,
        make_positive: bool = True,
        coloring: str | ndarray = "transition",
        param_indices: NpIndices | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot transition energies as a function of one external parameter.

        Usage is based on pre-slicing of the :class:`ParameterSweep` object to
        select a single parameter to be involved in the sweep, e.g.,

        ``<ParameterSweep>[0, :, 2].plot_transitions()``

        plots all eigenenergy differences for transitions starting in the
        ground state (the default when no initial state is specified) as a
        function of the middle parameter, with parameters 1 and 3 fixed by
        the indices 0 and 2.

        Parameters
        ----------
        subsystems:
            single subsystem or list of subsystems considered "active" for the
            transitions to be generated; if omitted, all subsystems are
            treated as actively participating
        initial:
            initial state from which transitions originate, given either as a
            tuple referring to a bare product state, or as an integer dressed
            state index. For a bare product state, the tuple has as many
            entries as the underlying :class:`HilbertSpace` has subsystems.
            (If ``subsystems`` is given, the tuple may be reduced to entries
            for just those subsystems; other subsystems are filled with ``0``
            automatically.) The dressed state corresponding to a given bare
            product state is determined by overlap.
            Note: for an initial dressed state, the ``sidebands`` option is
            ignored.
        final:
            concrete final state for which the transition energy should be
            generated, given either as a bare product state (tuple of
            excitation numbers) or as a dressed state (non-negative integer).
            If ``final`` is omitted, a list of final states is generated for
            dispersive transitions within each (active) subsystem; sidebands
            can be switched on with the ``sidebands`` keyword. ``final=-1``
            selects all other dressed states as final states (helpful when the
            dispersive limit breaks down).
        sidebands:
            if ``True``, include sideband transitions in which multiple
            subsystems change excitation levels (default: ``False``); ignored
            when ``initial`` is given as an integer dressed-state index
        photon_number:
            number of photons involved in the transition; transition
            frequencies are divided by this number (default: ``1``, i.e.,
            single-photon transitions)
        make_positive:
            relevant when the initial state is an excited state; downward
            transition energies are normally negative but are converted to
            positive values when this flag is ``True`` (default: ``True``)
        coloring:
            ``"transition"`` (default) colors transitions by their dispersive
            character; ``"plain"`` applies no selective highlighting
        param_indices:
            usually omitted, as ``param_indices`` is set via pre-slicing

        Returns
        -------
        Matplotlib :class:`~matplotlib.figure.Figure` and
        :class:`~matplotlib.axes.Axes` objects.
        """
        self._validate_states(initial, final)

        param_indices = param_indices or self._current_param_indices
        if not self._slice_is_1d_sweep(param_indices):
            raise ValueError(
                "Transition plots are only supported for a sweep over a "
                "single parameter. You can reduce a multi-dimensional "
                "sweep by pre-slicing, e.g.,  <ParameterSweep>[0, :, "
                "0].plot_transitions(...)"
            )
        specdata_for_highlighting = self.transitions(  # type: ignore[call-overload,misc]
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
        specdata_all.energy_table -= specdata_for_highlighting.subtract
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
            **kwargs,
        )
        plot._LABELLINES_ENABLED = labellines_status
        return fig, axes

    def keys(self):
        """Return the dict-like keys of the sweep data store."""
        return self._data.keys()

    def add_sweep(
        self,
        sweep_function: str | Callable,
        sweep_name: str | None = None,
        **kwargs,
    ) -> None:
        """Add a new sweep to the :class:`ParameterSweep` object.

        The generated data is subsequently accessible through
        ``<ParameterSweep>[<sweep_function>]`` or
        ``<ParameterSweep>[<sweep_name>]``.

        Parameters
        ----------
        sweep_function:
            name of a sweep function in :mod:`scqubits.core.sweeps` (as a
            string), or a custom callable provided by the user
        sweep_name:
            if given, store the generated data under
            ``<ParameterSweep>[<sweep_name>]`` rather than
            ``<ParameterSweep>[<sweep_function>]``
        **kwargs:
            keyword arguments forwarded to the sweep function
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
        operator: str | Qobj,
        sweep_name: str,
        subsystem: QuantumSystem | None = None,
    ) -> None:
        """Generate matrix-element data for a given operator as a function of the sweep.

        The resulting data is accessible as ``<ParameterSweep>[<sweep_name>]``.

        Parameters
        ----------
        operator:
            name of the operator (``str``), or full operator as a
            :class:`qutip.Qobj`
        sweep_name:
            key under which the generated data is stored
        subsystem:
            subsystem for which to compute matrix elements; required when
            ``operator`` is given as a string
        """
        if isinstance(operator, str):
            operator_func = functools.partial(
                sweeps.bare_matrixelement,
                operator_name=operator,
                subsystem=subsystem,  # type: ignore[arg-type]
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


class ParameterSweep(
    ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable
):
    """Create multi-dimensional parameter sweeps for a quantum system.

    The quantum system is described by a :class:`HilbertSpace` object.

    Parameters
    ----------
    hilbertspace:
        :class:`HilbertSpace` object describing the quantum system of interest
    paramvals_by_name:
        dictionary mapping each parameter name to the array of values to be
        used in the sweep
    update_hilbertspace:
        function that updates the associated :class:`HilbertSpace` for a given
        set of parameter values; signature is either
        ``update_hilbertspace(paramval1, paramval2, ...)`` or
        ``update_hilbertspace(self, paramval1, paramval2, ...)``,
        where ``self`` exposes the :class:`ParameterSweep` instance (and
        thereby dict-like access to subsystems and interaction terms)
    evals_count:
        number of dressed eigenvalues/eigenstates to keep. (The number of bare
        eigenvalues/eigenstates is determined for each subsystem by
        :attr:`truncated_dim`.) (default: 20)
    subsys_update_info:
        optional dictionary specifying which subsystems are updated by each
        sweep parameter, used to speed up calculations. The expected form is::

            {
                "<parameter name 1>": [<subsystem a>],
                "<parameter name 2>": [<subsystem b>, <subsystem c>, ...],
                ...
            }

        i.e., changes in ``<parameter name 1>`` only require updates of
        ``<subsystem a>``, leaving other subsystems unchanged; similarly,
        sweeping ``<parameter name 2>`` affects ``<subsystem b>``,
        ``<subsystem c>``, etc.
    bare_only:
        if ``True``, only bare eigendata is calculated; useful for sweeping a
        single quantum system without interactions (default: ``False``)
    labeling_scheme:
        scheme used to generate the dressed-state labeling in the lookup
        table:

        - ``"DE"`` (Dressed Energy): traverse the eigenstates in order of
          dressed energy, and find the corresponding bare-state label by
          overlap (default)
        - ``"LX"`` (Lexical ordering): traverse the bare states in `lexical
          order`_ and perform the branch analysis generalized from Dumas et
          al. (2024)
        - ``"BE"`` (Bare Energy): traverse the bare states in order of their
          energy before coupling and perform label assignment. Particularly
          useful when the Hilbert space is too large for all eigenstates to be
          labeled.
    labeling_subsys_priority:
        a permutation of the subsystem indices and bare labels. If provided,
        lexical ordering is performed on the permuted labels. A "branch" is
        defined as a series of eigenstates formed by putting excitations into
        the last subsystem in the list.
    labeling_BEs_count:
        number of dressed states to be labeled (``"BE"`` scheme only)
    ignore_low_overlap:
        if ``False`` (default), bare product states and dressed eigenstates
        are identified only if
        :math:`|\\langle\\psi_\\text{bare}|\\psi_\\text{dressed}\\rangle|^2 > 0.5`;
        if ``True``, identification is always made based on which bare product
        state has the maximum overlap
    autorun:
        if ``True``, run the sweep immediately; if ``False``, delay until
        :meth:`run` is called manually (default: ``settings.AUTORUN_SWEEP``)
    deepcopy:
        if ``True``, the parameter sweep is run with an exact copy of the
        :class:`HilbertSpace`; this ensures that all parameters after the
        sweep match those before the sweep. Note: changes to global
        :class:`HilbertSpace` or :class:`QuantumSystem` attributes will have
        no effect under this option; all updates must be made via
        ``<ParameterSweep>.hilbertspace[<id_str>] = ...``. If ``False``
        (default), updates to global instances have the expected effect, and
        the :class:`HilbertSpace` object and its constituents are left in the
        state reached by the final parameter update.
    num_cpus:
        number of CPU cores requested for computing the sweep
        (default: ``settings.NUM_CPUS``)

    Notes
    -----
    :class:`ParameterSweep` supports array-like access ("pre-slicing") and
    dict-like access. With dict-like access via string keys
    ``<ParameterSweep>[<str>]``, the following data is returned:

    ``"evals"`` and ``"evecs"``
        dressed eigenenergies and eigenstates as
        :obj:`.NamedSlotsNdarray`; eigenstates are decomposed in the bare
        product-state basis of the non-interacting subsystems' eigenbases
    ``"bare_evals"`` and ``"bare_evecs"``
        bare eigenenergies and eigenstates as :obj:`.NamedSlotsNdarray`
    ``"lamb"``, ``"chi"``, and ``"kerr"``
        dispersive energy coefficients
    ``"<custom sweep>"``
        :obj:`.NamedSlotsNdarray` of custom data generated with
        :meth:`add_sweep`

    Array-like access drives "pre-slicing" and enables lookup functionality
    such as ``<Sweep>[p1, p2, ...].eigensys()``.

    .. _lexical order: https://en.wikipedia.org/wiki/Lexicographic_order#Cartesian_products/
    """

    def __init__(
        self,
        hilbertspace: HilbertSpace,
        paramvals_by_name: dict[str, ndarray],
        update_hilbertspace: Callable,
        evals_count: int = 20,
        subsys_update_info: dict[str, list[QuantumSystem]] | None = None,
        bare_only: bool = False,
        labeling_scheme: Literal["DE", "LX", "BE"] = "DE",
        labeling_subsys_priority: list[int] | None = None,
        labeling_BEs_count: int | None = None,
        ignore_low_overlap: bool = False,
        autorun: bool = settings.AUTORUN_SWEEP,
        deepcopy: bool = False,
        num_cpus: int | str | None = None,
    ) -> None:
        if num_cpus == "auto" or (
            num_cpus is None and getattr(settings, "AUTO_PARALLEL", False)
        ):
            # decide before the autorun, from the constructor arguments
            from scqubits.utils.parallel_tuning import _auto_config

            counts = [len(vals) for vals in paramvals_by_name.values()]
            total_points = int(np.prod(counts)) if counts else 1
            auto = _auto_config(hilbertspace.dimension, total_points, evals_count)
            num_cpus = auto.num_cpus
            blas_threads = auto.blas_threads
        else:
            num_cpus = cpu_switch._resolve_explicit_num_cpus(num_cpus)
            blas_threads = None
        self._parameters: Parameters = Parameters(paramvals_by_name)
        self._hilbertspace = hilbertspace
        self._evals_count: int = evals_count
        self._update_hilbertspace = self.set_update_func(update_hilbertspace)
        self._subsys_update_info = subsys_update_info
        self._data: dict[str, Any] = {}
        self._bare_only = bare_only
        self._labeling_scheme = labeling_scheme
        self._labeling_subsys_priority = labeling_subsys_priority
        self._labeling_BEs_count = labeling_BEs_count
        self._ignore_low_overlap = ignore_low_overlap
        self._deepcopy = deepcopy
        self._num_cpus = num_cpus
        self._blas_threads = blas_threads

        self._out_of_sync = False
        self.reset_preslicing()

        self._check_subsys_id_strs()
        self._check_subsys_update_info()

        dispatch.CENTRAL_DISPATCH.register("PARAMETERSWEEP_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("HILBERTSPACE_UPDATE", self)

        global _faulty_interactionterm_warning_issued
        if (
            self.faulty_interactionterm_suspected()
            and not _faulty_interactionterm_warning_issued
        ):
            warnings.warn(
                "The interactions specified for this HilbertSpace object involve coupling operators stored as fixed "
                "matrices. This may be unintended, as the operators of quantum systems (specifically, their "
                "representation with respect to some basis) may change as a function of sweep parameters. \nFor that "
                "reason, it is recommended to use coupling operators specified as callable functions.\n",
                UserWarning,
            )
            _faulty_interactionterm_warning_issued = True

        if autorun:
            self.run()

    def _check_subsys_id_strs(self) -> None:
        """Ensure subsystem ``id_str`` values are unique within the sweep.

        Repeated ``id_str`` values are not allowed: the lookup of subsystems
        listed in ``subsys_update_info`` relies on ``id_str``.
        """
        id_strs = [subsystem.id_str for subsystem in self.hilbertspace.subsystem_list]
        if len(id_strs) != len(set(id_strs)):
            raise ValueError("Repeated id_str are not allowed in ParameterSweep.")

    def _check_subsys_update_info(self) -> None:
        """Validate the user-provided ``subsys_update_info`` dictionary.

        ``subsys_update_info`` is a dictionary with parameter names as keys and
        corresponding subsystem lists as values. This method checks that all
        referenced parameter names and subsystems are known to the sweep.
        """
        if self._subsys_update_info is None:
            return

        param_names = self._parameters.names
        id_strs = [subsystem.id_str for subsystem in self.hilbertspace.subsystem_list]

        for parameter_name, subsystems in self._subsys_update_info.items():
            if not all(subsystem.id_str in id_strs for subsystem in subsystems):
                raise ValueError(
                    f"Subsystems specified in "
                    f"subsys_update_info['{parameter_name}'] are not "
                    "found in the provided HilbertSpace object."
                )
            if not parameter_name in param_names:
                raise ValueError(
                    f"Parameter name '{parameter_name}' in "
                    "subsys_update_info is not found in the "
                    "parameter list."
                )

    @property
    def tqdm_disabled(self) -> bool:
        """Return whether the tqdm progress bar should be disabled for this sweep."""
        return settings.PROGRESSBAR_DISABLED

    def _consume_with_bar(
        self, mapped: Any, total: int, desc: str, progress_bars: list
    ) -> list:
        """Collect the mapped results, driving a tqdm progress bar.

        Under IPython (a Jupyter notebook or a terminal IPython shell, i.e.
        ``settings.IN_IPYTHON``) the finished bar is kept on screen and appended to
        ``progress_bars`` (a list owned by :meth:`run`), so the bare-spectrum bar(s)
        stay visible above the dressed-spectrum bar; :meth:`run` clears them together
        once the dressed sweep finishes. Otherwise (a plain Python interpreter or
        script) the bar is closed as soon as its phase finishes, preserving the
        previous sequential display.

        The bars live in a caller-owned list, never on the instance: a ``tqdm`` bar is
        not picklable, and the worker tasks pickle ``self``, so a bar stored on
        ``self`` would break parallel dispatch.

        Parameters
        ----------
        mapped:
            iterator of per-grid-point results (the output of the pool ``imap`` or the
            built-in ``map``), yielded in grid order.
        total:
            number of grid points, used for the bar length.
        desc:
            progress-bar label for this phase.
        progress_bars:
            caller-owned list collecting bars to clear together at the end of the run.
        """
        bar = tqdm(total=total, desc=desc, leave=False, disable=self.tqdm_disabled)
        results = []
        try:
            for item in mapped:
                results.append(item)
                bar.update(1)
        except BaseException:
            # the mapped iterator raised mid-sweep: close this bar so it does not leak
            # (it was never registered in progress_bars, which only happens on success)
            bar.close()
            raise
        if settings.IN_IPYTHON:
            bar.refresh()
            progress_bars.append(bar)
        else:
            bar.close()
        return results

    def faulty_interactionterm_suspected(self) -> bool:
        """Check if any interaction terms are specified as fixed matrices."""
        for interactionterm in self._hilbertspace.interaction_list:
            if isinstance(interactionterm, (ndarray, Qobj, csc_matrix)):
                return True
            for idx_operator in interactionterm.operator_list:
                if isinstance(idx_operator[1], (ndarray, Qobj, csc_matrix)):
                    return True
        return False

    def cause_dispatch(self) -> None:
        """Trigger a single ``HILBERTSPACE_UPDATE`` dispatch via ``update_hilbertspace``.

        This is used at the start of :meth:`run` to initialize subsystem state
        with the first parameter point of the sweep before bulk computation.
        """
        initial_parameters = tuple(paramvals[0] for paramvals in self._parameters)
        self._update_hilbertspace(self, *initial_parameters)

    @classmethod
    def deserialize(cls, iodata: "IOData") -> "StoredSweep":  # type: ignore[override,empty-body]
        """Return a :class:`StoredSweep` reconstructed from the given :class:`IOData`.

        Parameters
        ----------
        iodata:
            serialized representation as produced by :meth:`serialize`
        """
        pass

    def serialize(self) -> "IOData":
        """Convert this :class:`ParameterSweep` into :class:`IOData` form."""
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
        """Create all sweep data.

        Computes bare spectral data, dressed spectral data, lookup data, and
        any custom sweep data. As a side effect, this method mutates instance
        flags such as ``_out_of_sync``, ``_out_of_sync_warning_issued``, and
        ``_lookup_exists``, and (when ``deepcopy=True``) temporarily replaces
        ``_hilbertspace`` with a deep copy.
        """
        # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        self._out_of_sync = False
        self._out_of_sync_warning_issued = False

        self._lookup_exists = True
        if self._deepcopy:
            instance_str = dill.dumps(self.hilbertspace)
            stored_hilbertspace = self._hilbertspace
            self._hilbertspace = dill.loads(instance_str)
        else:
            self.cause_dispatch()
        settings.DISPATCH_ENABLED = False

        # The outer try restores global state (DISPATCH_ENABLED, the deepcopied
        # hilbertspace) even if a sweep raises. The inner try closes the phase progress
        # bars right when the dressed sweep finishes -- the bare bar(s) stay stacked
        # above the dressed bar until then; see _consume_with_bar. progress_bars is a
        # local (not on self) because tqdm bars are not picklable and self is shipped
        # to workers during parallel dispatch.
        progress_bars: list = []
        try:
            try:
                (
                    self._data["bare_evals"],
                    self._data["bare_evecs"],
                    self._data["circuit_esys"],
                ) = self._bare_spectrum_sweep(progress_bars)
                if not self._bare_only:
                    self._data["evals"], self._data["evecs"] = (
                        self._dressed_spectrum_sweep(progress_bars)
                    )
            finally:
                for bar in progress_bars:
                    bar.close()
            if not self._bare_only:
                self._data["dressed_indices"] = self.generate_lookup(
                    ordering=self._labeling_scheme,
                    subsys_priority=self._labeling_subsys_priority,
                    BEs_count=self._labeling_BEs_count,
                )
                (
                    self._data["lamb"],
                    self._data["chi"],
                    self._data["kerr"],
                ) = self._dispersive_coefficients()
        finally:
            if self._deepcopy:
                self._hilbertspace = stored_hilbertspace  # restore original state
            settings.DISPATCH_ENABLED = True

    def _bare_spectrum_sweep(
        self, progress_bars: list
    ) -> tuple[NamedSlotsNdarray, NamedSlotsNdarray, NamedSlotsNdarray]:
        """Compute bare eigenvalues, eigenvectors, and circuit ``esys`` arrays.

        The bare energy spectra are computed according to the following scheme:

        1. Loop over all subsystems to separately obtain the bare eigenvalues
           and eigenstates for each subsystem.
        2. If ``subsys_update_info`` is given, drop those sweeps that leave the
           subsystem fixed.
        3. If ``self._num_cpus > 1``, parallelize.

        Returns
        -------
        Three :class:`NamedSlotsNdarray` instances with axes
        ``[<paramname1>, <paramname2>, ..., "subsys"]`` containing,
        respectively, bare eigenvalues, bare eigenvectors, and the
        per-subsystem ``circuit_esys`` data used by hierarchical
        diagonalization. Here, ``"subsys": 0, 1, ...`` enumerates subsystems.
        """
        bare_evals = np.empty((self.subsystem_count,), dtype=object)
        bare_evecs = np.empty((self.subsystem_count,), dtype=object)
        # creating data arrays for subsystems, to store the esys for all subsystems when HD is used
        circuit_esys = np.empty((self.subsystem_count,), dtype=object)

        for subsys_index, subsystem in enumerate(self.hilbertspace):
            bare_esys = self._subsys_bare_spectrum_sweep(subsystem, progress_bars)
            if (
                hasattr(subsystem, "hierarchical_diagonalization")
                and subsystem.hierarchical_diagonalization
            ):
                evals = np.empty_like(bare_esys[..., 0])
                evecs = np.empty_like(bare_esys[..., 0])
                for array_index, esys in np.ndenumerate(bare_esys[..., 0]):
                    evals[array_index] = esys[0]
                    evecs[array_index] = esys[1]
            else:
                evals = bare_esys[..., 0]
                evecs = bare_esys[..., 1]
            bare_evals[subsys_index] = NamedSlotsNdarray(
                np.asarray(evals.tolist()),
                self._parameters.paramvals_by_name,
            )
            bare_evecs[subsys_index] = NamedSlotsNdarray(
                np.asarray(evecs.tolist()),
                self._parameters.paramvals_by_name,
            )
            circuit_esys[subsys_index] = (
                bare_esys  # when param =(p0, p1, p2, ...), subsys i esys is circuit_esys[i][p0, p1, p2, ...]
            )

        return (
            NamedSlotsNdarray(bare_evals, {"subsys": np.arange(self.subsystem_count)}),
            NamedSlotsNdarray(bare_evecs, {"subsys": np.arange(self.subsystem_count)}),
            NamedSlotsNdarray(
                circuit_esys, {"subsys": np.arange(self.subsystem_count)}
            ),
        )

    def _update_subsys_compute_esys(
        self,
        update_func: Callable,
        subsystem: QuantumSystem,
        paramval_tuple: tuple[float],
    ) -> ndarray:
        """Update parameters via ``update_func`` and return ``subsystem``'s eigensystem.

        For circuit subsystems with hierarchical diagonalization, the full
        bare eigensystem is returned by :meth:`generate_bare_eigensys`.
        Otherwise, the subsystem's standard :meth:`eigensys` is called and the
        result is packed into an object array of shape ``(2,)``.

        Parameters
        ----------
        update_func:
            callable used to apply the current parameter values to the
            :class:`HilbertSpace` and its subsystems
        subsystem:
            subsystem whose bare eigensystem is computed
        paramval_tuple:
            tuple of parameter values for the current grid point
        """
        update_func(self, *paramval_tuple)
        # use the Circuit method to return esys for all the subsystems when HD is used
        if isinstance(subsystem, (scq.Circuit, scq.core.circuit.Subsystem)):
            return subsystem.generate_bare_eigensys()
        evals, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)  # type: ignore[attr-defined]
        esys_array = np.empty(shape=(2,), dtype=object)
        esys_array[0] = evals
        esys_array[1] = evecs
        return esys_array

    def _paramnames_no_subsys_update(self, subsystem: QuantumSystem) -> list[str]:
        """Return the parameter names that do not require updating ``subsystem``.

        Uses the user-provided ``subsys_update_info`` to determine which
        parameters can be skipped when sweeping the bare spectrum of the
        given ``subsystem``.

        Parameters
        ----------
        subsystem:
            subsystem whose update-irrelevant parameters are requested
        """
        if self._subsys_update_info is None:
            return []
        updating_parameters = [
            name
            for name in self._subsys_update_info.keys()
            if subsystem.id_str
            in [subsys.id_str for subsys in self._subsys_update_info[name]]
        ]
        return list(set(self._parameters.names) - set(updating_parameters))

    def _subsys_bare_spectrum_sweep(
        self, subsystem: QuantumSystem, progress_bars: list
    ) -> ndarray:
        """Compute the bare eigensystem of ``subsystem`` over the full parameter grid.

        Parameters
        ----------
        subsystem:
            subsystem for which the bare spectrum sweep is to be computed

        Returns
        -------
        Multidimensional object array of the format
        ``array[p1, p2, ..., pN] = np.asarray([evals, evecs])``.
        """
        fixed_paramnames = self._paramnames_no_subsys_update(subsystem)
        reduced_parameters = self._parameters.create_reduced(fixed_paramnames)
        total_count = int(
            np.prod([len(param_vals) for param_vals in reduced_parameters])
        )

        target_map = cpu_switch.get_map_method(
            self._num_cpus, self._blas_threads, total=total_count
        )

        bare_eigendata = np.asarray(
            self._consume_with_bar(
                target_map(
                    functools.partial(
                        self._update_subsys_compute_esys,
                        self._update_hilbertspace,
                        subsystem,
                    ),
                    itertools.product(*reduced_parameters.paramvals_list),
                ),
                total_count,
                "Bare spectra [{}]".format(subsystem.id_str),
                progress_bars,
            ),
            dtype=object,
        )
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
        work_item: tuple[tuple[int, ...], dict[int, list[ndarray]], dict[int, Any]],
    ) -> ndarray:
        """Update the HilbertSpace and compute dressed eigensystem at one grid point.

        Updates the parent :class:`HilbertSpace`, propagates the cached bare
        eigensystem of each subsystem (handling hierarchical-diagonalization
        children and parent updates), and returns the dressed eigensystem.

        The per-point bare eigensystem is supplied through ``work_item`` rather
        than read from ``self._data`` so that mapping this method over the grid
        does not pickle the full bare-spectrum arrays of every grid point into
        each worker task (see :meth:`_dressed_spectrum_sweep`).

        Parameters
        ----------
        hilbertspace:
            :class:`HilbertSpace` instance whose dressed eigensystem is computed
        evals_count:
            number of dressed eigenvalues/eigenstates to compute
        update_func:
            callable used to update parameter-dependent attributes for the
            current grid point
        work_item:
            ``(paramindex_tuple, bare_esys, circuit_esys_point)`` for the grid
            point: the integer index tuple, the per-subsystem ``[evals, evecs]``
            bare eigensystem, and the per-subsystem circuit eigensystem used by
            hierarchically diagonalized subsystems.

        Returns
        -------
        Object array of shape ``(2,)`` whose entries are the dressed
        eigenvalues array and the corresponding eigenvectors array.
        """
        paramindex_tuple, bare_esys, circuit_esys_point = work_item
        paramval_tuple = self._parameters[paramindex_tuple]
        update_func(self, *paramval_tuple)
        # update the lookuptables for subsystems using hierarchical diagonalization
        for subsys_index, subsys in enumerate(hilbertspace.subsystem_list):
            if (
                hasattr(subsys, "hierarchical_diagonalization")
                and subsys.hierarchical_diagonalization
            ):
                subsys.set_bare_eigensys(  # type: ignore[union-attr]
                    circuit_esys_point[subsys_index]
                )

            # if the subsys has a parent Circuit/Subsystem module, then update its hilbert_space
            if hasattr(
                subsys, "parent"
            ):  # update necessary interactions and attributes
                parent_subsys_idx = subsys.parent.get_subsystem_index(
                    subsys.dynamic_var_indices[0]  # type: ignore[union-attr]
                )
                # update parents HilbertSpace database
                subsys.parent.hilbert_space._data["bare_evals"][parent_subsys_idx] = (
                    np.array([bare_esys[subsys_index][0]])
                )
                subsys.parent.hilbert_space._data["bare_evecs"][parent_subsys_idx] = (
                    np.array([bare_esys[subsys_index][1]])
                )
                # remove the subsystem from affected_subsystem_indices
                if parent_subsys_idx in subsys.parent.affected_subsystem_indices:
                    subsys.parent.affected_subsystem_indices.remove(parent_subsys_idx)

        evals, evecs = hilbertspace.eigensys(
            evals_count=evals_count, bare_esys=bare_esys  # type: ignore[arg-type]
        )
        esys_array = np.empty(shape=(2,), dtype=object)
        esys_array[0] = evals
        esys_array[1] = evecs
        return esys_array

    def _dressed_spectrum_sweep(
        self, progress_bars: list
    ) -> tuple[NamedSlotsNdarray, NamedSlotsNdarray]:
        """Compute dressed eigenvalues and eigenvectors over the full parameter grid.

        Returns
        -------
        Pair of :class:`NamedSlotsNdarray` instances with axes
        ``[<paramname1>, <paramname2>, ...]`` containing dressed eigenvalues
        and dressed eigenvectors, respectively.

        Notes
        -----
        While the grid is being mapped, ``self._data["bare_evals"]``,
        ``self._data["bare_evecs"]`` and ``self._data["circuit_esys"]`` are
        temporarily set to ``None`` (restored in a ``finally`` block) so that the
        bound worker callable does not pickle the whole grid's bare spectrum into
        every task. A user-supplied ``update_hilbertspace`` callback must therefore
        not read these ``self._data`` entries: it receives the parameter values it
        needs as arguments, and the per-point bare eigensystem is delivered to the
        worker through the work item instead.
        """
        total_count = int(np.prod(self._parameters.counts))
        target_map = cpu_switch.get_map_method(
            self._num_cpus, self._blas_threads, total=total_count
        )

        # Ship only the current grid point's bare eigensystem to each worker.
        # Reading these slices from ``self._data`` inside the worker would pickle
        # the whole grid's bare-spectrum arrays into every task (O(N^2) IPC), so
        # detach them here, pass per-point slices through the work items, and
        # restore ``self._data`` afterwards. While detached, these three entries
        # are ``None``; a user ``update_hilbertspace`` callback must not read them
        # (see the Notes in this method's docstring).
        bare_evals = self._data["bare_evals"]
        bare_evecs = self._data["bare_evecs"]
        circuit_esys = self._data["circuit_esys"]
        subsys_count = len(self.hilbertspace)

        def _work_items() -> Any:
            for paramindex_tuple in itertools.product(*self._parameters.ranges):
                bare_esys = {
                    subsys_index: [
                        bare_evals[subsys_index][paramindex_tuple],
                        bare_evecs[subsys_index][paramindex_tuple],
                    ]
                    for subsys_index in range(subsys_count)
                }
                circuit_esys_point = {
                    subsys_index: circuit_esys[subsys_index][paramindex_tuple]
                    for subsys_index in range(subsys_count)
                }
                yield (paramindex_tuple, bare_esys, circuit_esys_point)

        self._data["bare_evals"] = None
        self._data["bare_evecs"] = None
        self._data["circuit_esys"] = None
        try:
            spectrum_data = self._consume_with_bar(
                target_map(
                    functools.partial(
                        self._update_and_compute_dressed_esys,
                        self._hilbertspace,
                        self._evals_count,
                        self._update_hilbertspace,
                    ),
                    _work_items(),
                ),
                total_count,
                "Dressed spectrum",
                progress_bars,
            )
        finally:
            self._data["bare_evals"] = bare_evals
            self._data["bare_evecs"] = bare_evecs
            self._data["circuit_esys"] = circuit_esys

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

    def _energies_1(self, subsys: QuantumSystem) -> ndarray:
        """Return dressed energies for all single-subsystem excitations of ``subsys``.

        For each level ``l`` of ``subsys.truncated_dim``, looks up the dressed
        energy associated with the bare product state in which only ``subsys``
        is excited to level ``l``.

        Parameters
        ----------
        subsys:
            subsystem whose single-mode excited-state energies are collected
        """
        bare_label = np.zeros(len(self.hilbertspace))
        bare_label[self.get_subsys_index(subsys)] = 1

        energies_all_l = np.empty(self._parameters.counts + (subsys.truncated_dim,))
        for l in range(subsys.truncated_dim):
            energies_all_l[..., l] = self[:].energy_by_bare_index(tuple(l * bare_label))
        return energies_all_l

    def _energies_2(self, subsys1: QuantumSystem, subsys2: QuantumSystem) -> ndarray:
        """Return dressed energies for joint excitations of ``subsys1`` and ``subsys2``.

        For each pair of levels ``(l1, l2)`` within the truncated dimensions
        of the two subsystems, looks up the dressed energy associated with the
        bare product state in which only ``subsys1`` and ``subsys2`` are
        excited to levels ``l1`` and ``l2``, respectively.

        Parameters
        ----------
        subsys1:
            first subsystem (excited to ``l1``)
        subsys2:
            second subsystem (excited to ``l2``)
        """
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
    ) -> tuple[NamedSlotsNdarray, NamedSlotsNdarray, NamedSlotsNdarray]:
        """Compute dispersive Lamb shifts, ac Stark shifts, and Kerr coefficients.

        For each subsystem and each pair of subsystems, the relevant dressed
        energy combinations are evaluated to extract Lamb shifts (single-mode
        shifts of the excited-state energies relative to bare values), ac
        Stark / cross-Kerr couplings between qubit and oscillator (or
        qubit-qubit / oscillator-oscillator) modes.

        Returns
        -------
        Tuple ``(lamb_data, chi_data, kerr_data)`` of :class:`NamedSlotsNdarray`
        instances. ``lamb_data`` is indexed by ``"subsys"``; ``chi_data`` and
        ``kerr_data`` are indexed by ``("subsys1", "subsys2")``.
        """
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
    """Container for a previously computed :class:`ParameterSweep` loaded from disk.

    A :class:`StoredSweep` exposes the :class:`ParameterSweepBase` interface
    (pre-slicing, lookup, transition extraction, plotting) on top of pre-computed
    sweep data without recomputing eigensystems.

    Parameters
    ----------
    paramvals_by_name:
        dictionary mapping each parameter name to its array of swept values
    hilbertspace:
        :class:`HilbertSpace` instance that produced the stored data
    evals_count:
        number of dressed eigenvalues/eigenstates kept in the stored sweep
    _data:
        dictionary of pre-computed sweep arrays (e.g., ``"evals"``,
        ``"evecs"``, ``"bare_evals"``, ``"bare_evecs"``, custom sweeps)
    """

    _parameters = descriptors.WatchedProperty(Parameters, "PARAMETERSWEEP_UPDATE")
    _evals_count = descriptors.WatchedProperty(int, "PARAMETERSWEEP_UPDATE")
    _data = descriptors.WatchedProperty(dict[str, ndarray], "PARAMETERSWEEP_UPDATE")
    _hilbertspace: HilbertSpace

    def __init__(
        self,
        paramvals_by_name: dict[str, ndarray],
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
        """Reconstruct a :class:`StoredSweep` from the given :class:`IOData`.

        Parameters
        ----------
        iodata:
            serialized representation as produced by :meth:`serialize`

        Returns
        -------
        StoredSweep
        """
        return StoredSweep(**iodata.as_kwargs())  # type: ignore[abstract]

    def serialize(self) -> "IOData":  # type: ignore[empty-body]
        """Convert this :class:`StoredSweep` into :class:`IOData` form."""
        pass

    def get_hilbertspace(self) -> HilbertSpace:
        """Return the underlying :class:`HilbertSpace` instance."""
        return self.hilbertspace

    def new_sweep(
        self,
        paramvals_by_name: dict[str, ndarray],
        update_hilbertspace: Callable,
        evals_count: int = 6,
        subsys_update_info: dict[str, list[QuantumSystem]] | None = None,
        autorun: bool = settings.AUTORUN_SWEEP,
        num_cpus: int | str | None = None,
    ) -> ParameterSweep:
        """Create a new :class:`ParameterSweep` reusing this sweep's :class:`HilbertSpace`.

        Parameters
        ----------
        paramvals_by_name:
            dictionary specifying the parameter names and the values to sweep
        update_hilbertspace:
            callable that updates the :class:`HilbertSpace` for given
            parameter values; same conventions as in :class:`ParameterSweep`
        evals_count:
            number of dressed eigenvalues/eigenstates to keep (default: 6)
        subsys_update_info:
            optional dictionary mapping parameter names to the list of
            subsystems they affect; see :class:`ParameterSweep`
        autorun:
            if ``True``, immediately run the sweep upon construction
            (default: ``settings.AUTORUN_SWEEP``)
        num_cpus:
            number of CPU cores to use, or ``"auto"`` to let scqubits choose
            from the workload (see :func:`~scqubits.recommend_parallelization`);
            defaults to ``settings.NUM_CPUS``.
        """
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
    """Compute custom sweep data as a function of the external parameter(s).

    The data is computed via the user-provided callable ``func``.

    Parameters
    ----------
    sweep:
        :class:`ParameterSweep` object containing :class:`HilbertSpace` and
        spectral information
    func:
        callable with signature
        ``func(parametersweep, paramindex_tuple, paramvals_tuple, **kwargs)``
        that returns the data for a single choice of parameter(s)
    **kwargs:
        keyword arguments forwarded to ``func``

    Returns
    -------
    Array of custom data.
    """
    # obtain reduced parameters from pre-slicing info
    reduced_parameters = sweep._parameters.create_sliced(
        sweep._current_param_indices, remove_fixed=False
    )
    total_count = int(np.prod(reduced_parameters.counts))

    def func_effective(paramindex_tuple: tuple[int], params, **kw) -> Any:
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
    element_shape: tuple[int, ...] = tuple()
    if isinstance(data_array[0], np.ndarray):
        element_shape = data_array[0].shape

    data_ndarray = np.asarray(data_array)
    return NamedSlotsNdarray(
        data_ndarray.reshape(reduced_parameters.counts + element_shape),
        reduced_parameters.paramvals_by_name,
    )
