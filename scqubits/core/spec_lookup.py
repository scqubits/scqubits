# spec_lookup.py
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

import itertools

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj
from typing_extensions import Protocol

import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils
import scqubits.settings as settings

from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.utils.typedefs import NpIndexTuple, NpIndices

if TYPE_CHECKING:
    from typing_extensions import Protocol

    from scqubits import HilbertSpace
    from scqubits.core.descriptors import WatchedProperty
    from scqubits.core.param_sweep import Parameters
    from scqubits.core.qubit_base import QuantumSystem
    from scqubits.io_utils.fileio_qutip import QutipEigenstates
    from scqubits.utils.typedefs import QuantumSys


class MixinCompatible(Protocol):
    _parameters: "WatchedProperty[Parameters]"
    _evals_count: "WatchedProperty[int]"
    _current_param_indices: NpIndices
    _ignore_low_overlap: bool
    _data: Dict[str, Any]
    _out_of_sync: bool
    hilbertspace: "HilbertSpace"

    def __getitem__(self, key: Any) -> Any:
        ...

    @property
    def hilbertspace(self) -> "HilbertSpace":
        ...


class SpectrumLookupAdapter:
    """Bridges earlier functionality provided by the `SpectrumLookup` class to
    universal use of the `SpectrumLookupMixin` class (originally developed for
    `ParameterSweep` and then extended to `HilbertSpace`)."""

    def __init__(self, hilbertspace):
        self.hilbertspace = hilbertspace

    @property
    def _out_of_sync(self):
        return self.hilbertspace._out_of_sync

    @_out_of_sync.setter
    def _out_of_sync(self, value: bool):
        self.hilbertspace._out_of_sync = value

    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.run is deprecated. "
        "Use <HilbertSpace>.generate_lookup instead."
    )
    def run(self) -> None:
        self.hilbertspace.generate_lookup()

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.dressed_index is "
        "deprecated. Use <HilbertSpace>.dressed_index instead."
    )
    def dressed_index(self, bare_labels: Tuple[int, ...]) -> Union[int, None]:
        """
        For given bare product state return the corresponding dressed-state index.

        Parameters
        ----------
        bare_labels:
            bare_labels = (index, index2, ...)

        Returns
        -------
            dressed state index closest to the specified bare state
        """
        return self.hilbertspace.dressed_index(bare_labels)

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.bare_index is "
        "deprecated. Use <HilbertSpace>.bare_index instead."
    )
    def bare_index(self, dressed_index: int) -> Union[Tuple[int, ...], None]:
        """
        For given dressed index, look up the corresponding bare index.

        Returns
        -------
            Bare state specification in tuple form. Example: (1,0,3) means subsystem 1
            is in bare state 1, subsystem 2 in bare state 0, and subsystem 3 in bare
            state 3.
        """
        return self.hilbertspace.bare_index(dressed_index)

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.dressed_eigenstates is "
        "deprecated. Use <HilbertSpace>['evecs'] instead."
    )
    def dressed_eigenstates(self) -> List["QutipEigenstates"]:
        """
        Return the list of dressed eigenvectors

        Returns
        -------
            dressed eigenvectors for the external parameter fixed to the value
            indicated by the provided index
        """
        return self.hilbertspace["evecs"]

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.dressed_eigenenergies is "
        "deprecated. Use <HilbertSpace>['evals'] instead."
    )
    def dressed_eigenenergies(self) -> ndarray:
        """
        Return the array of dressed eigenenergies


        Returns
        -------
            dressed eigenenergies for the external parameter fixed to the value
            indicated by the provided index
        """
        return self.hilbertspace["evals"]

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.energy_bare_index is "
        "deprecated. Use "
        "<HilbertSpace>.energy_by_bare_index instead."
    )
    def energy_bare_index(self, bare_tuple: Tuple[int, ...]) -> Union[float, None]:
        """
        Look up dressed energy most closely corresponding to the given bare-state labels

        Parameters
        ----------
        bare_tuple:
            bare state indices

        Returns
        -------
            dressed energy, if lookup successful
        """
        return self.hilbertspace.energy_by_bare_index(bare_tuple)

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.energy_dressed_index is "
        "deprecated. Use "
        "<HilbertSpace>.energy_by_dressed_index instead."
    )
    def energy_dressed_index(self, dressed_index: int) -> float:
        """
        Look up the dressed eigenenergy belonging to the given dressed index.

        Parameters
        ----------
        dressed_index:
            index of dressed state of interest

        Returns
        -------
            dressed energy
        """
        return self.hilbertspace.energy_by_dressed_index(dressed_index)

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.bare_eigenstates is deprecated. "
        "Use <HilbertSpace>.bare_eigenstates instead."
    )
    def bare_eigenstates(self, subsys: "QuantumSystem") -> ndarray:
        """
        Return ndarray of bare eigenstates for given subsystem.
        Eigenstates are expressed in the basis internal to the subsystem.
        """
        return self.hilbertspace.bare_eigenstates(subsys)

    @utils.check_sync_status
    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.bare_eigenenergies is deprecated. "
        "Use <HilbertSpace>.bare_eigenvals instead."
    )
    def bare_eigenenergies(self, subsys: "QuantumSystem") -> ndarray:
        """
        Return list of bare eigenenergies for given subsystem.

        Parameters
        ----------
        subsys:
            Hilbert space subsystem for which bare eigendata is to be looked up

        Returns
        -------
            bare eigenenergies for the specified subsystem
        """
        return self.hilbertspace.bare_eigenvals(subsys)

    @utils.DeprecationMessage(
        "<HilbertSpace>.lookup.bare_productstate is deprecated. "
        "Use <HilbertSpace>.bare_productstate instead."
    )
    def bare_productstate(self, bare_index: Tuple[int, ...]) -> Qobj:
        """
        Return the bare product state specified by `bare_index`.

        Parameters
        ----------
        bare_index:

        Returns
        -------
            ket in full Hilbert space
        """
        return self.hilbertspace.bare_productstate(bare_index)


class SpectrumLookupMixin(MixinCompatible):
    """
    SpectrumLookupMixin is used as a mix-in class by `ParameterSweep`. It makes various
    spectrum and spectrum lookup related methods directly available at the
    `ParameterSweep` level.
    """

    @property
    def _bare_product_states_labels(self) -> List[Tuple[int, ...]]:
        """
        Generates the list of bare-state labels in canonical order. For example,
         for a Hilbert space composed of two subsystems sys1 and sys2, each label is
         of the type (3,0) meaning sys1 is in bare eigenstate 3, sys2 in bare
         eigenstate 0. The full list then reads
         [(0,0), (0,1), (0,2), ..., (0,max_2),
         (1,0), (1,1), (1,2), ..., (1,max_2),
         ...
         (max_1,0), (max_1,1), (max_1,2), ..., (max_1,max_2)]
        """
        return list(np.ndindex(*self.hilbertspace.subsystem_dims))

    def generate_lookup(self) -> NamedSlotsNdarray:
        """
        For each parameter value of the parameter sweep, generate the map between
        bare states and
        dressed states.

        Returns
        -------
            each list item is a list of dressed indices whose order corresponds to the
            ordering of bare indices (as stored in .canonical_bare_labels,
            thus establishing the mapping)
        """
        dressed_indices = np.empty(shape=self._parameters.counts, dtype=object)

        param_indices = itertools.product(*map(range, self._parameters.counts))
        for index in param_indices:
            dressed_indices[index] = self._generate_single_mapping(index)
        dressed_indices = np.asarray(dressed_indices[:].tolist())

        parameter_dict = self._parameters.ordered_dict.copy()
        return NamedSlotsNdarray(dressed_indices, parameter_dict)

    def _generate_single_mapping(
        self,
        param_indices: Tuple[int, ...],
    ) -> ndarray:
        """
        For a single set of parameter values, specified by a tuple of indices
        ``param_indices``, create an array of the dressed-state indices in an order
        that corresponds one to one to the bare product states with largest overlap
        (whenever possible).

        Parameters
        ----------
        param_indices:
            indices of the parameter values

        Returns
        -------
            dressed-state indices
        """
        overlap_matrix = spec_utils.convert_evecs_to_ndarray(
            self._data["evecs"][param_indices]
        )

        dim = self.hilbertspace.dimension
        dressed_indices: List[Union[int, None]] = [None] * dim
        for dressed_index in range(self._evals_count):
            max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
            max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
            if self._ignore_low_overlap or (
                max_overlap**2 > settings.OVERLAP_THRESHOLD
            ):
                overlap_matrix[:, max_position] = 0
                dressed_indices[int(max_position)] = dressed_index

        return np.asarray(dressed_indices)

    def set_npindextuple(
        self, param_indices: Optional[NpIndices] = None
    ) -> NpIndexTuple:
        param_indices = param_indices or self._current_param_indices
        if not isinstance(param_indices, tuple):
            param_indices = (param_indices,)
        return param_indices

    @utils.check_sync_status
    def dressed_index(
        self,
        bare_labels: Tuple[int, ...],
        param_indices: Optional[NpIndices] = None,
    ) -> Union[ndarray, int, None]:
        """
        For given bare product state return the corresponding dressed-state index.

        Parameters
        ----------
        bare_labels:
            bare_labels = (index, index2, ...)
        param_indices:
            indices of parameter values of interest

        Returns
        -------
            dressed state index closest to the specified bare state
        """
        param_indices = self.set_npindextuple(param_indices)
        try:
            lookup_position = self._bare_product_states_labels.index(bare_labels)
        except ValueError:
            return None
        return self._data["dressed_indices"][param_indices + (lookup_position,)]

    @utils.check_sync_status
    def bare_index(
        self,
        dressed_index: int,
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> Union[Tuple[int, ...], None]:
        """
        For given dressed index, look up the corresponding bare index.

        Returns
        -------
            Bare state specification in tuple form. Example: (1,0,3) means subsystem 1
            is in bare state 1, subsystem 2 in bare state 0,
            and subsystem 3 in bare state 3.
        """
        param_index_tuple = self.set_npindextuple(param_indices)
        if not self.all_params_fixed(param_index_tuple):
            raise ValueError(
                "All parameters must be fixed to concrete values for "
                "the use of `.bare_index`."
            )
        try:
            lookup_position = np.where(
                self._data["dressed_indices"][param_index_tuple] == dressed_index
            )[0][0]
        except IndexError:
            raise ValueError(
                "Could not identify a bare index for the given dressed "
                "index {}.".format(dressed_index)
            )
        basis_labels = self._bare_product_states_labels[lookup_position]
        return basis_labels

    @utils.check_sync_status
    def eigensys(
        self,
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> ndarray:
        """
        Return the list of dressed eigenvectors

        Parameters
        ----------
        param_indices:
            position indices of parameter values in question

        Returns
        -------
            dressed eigensystem for the external parameter fixed to the value indicated
            by the provided index
        """
        param_index_tuple = self.set_npindextuple(param_indices)
        return self._data["evecs"][param_index_tuple]

    @utils.check_sync_status
    def eigenvals(
        self,
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> ndarray:
        """
        Return the array of dressed eigenenergies - primarily for running the sweep

        Parameters
        ----------
            position indices of parameter values in question

        Returns
        -------
            dressed eigenenergies for the external parameters fixed to the values
            indicated by the provided indices
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        return self._data["evals"][param_indices_tuple]

    @utils.check_sync_status
    def energy_by_bare_index(
        self,
        bare_tuple: Tuple[int, ...],
        subtract_ground: bool = False,
        param_indices: Optional[NpIndices] = None,
    ) -> NamedSlotsNdarray:  # the return value may also be np.nan
        """
        Look up dressed energy most closely corresponding to the given bare-state labels

        Parameters
        ----------
        bare_tuple:
            bare state indices
        subtract_ground:
            whether to subtract the ground state energy
        param_indices:
            indices specifying the set of parameters

        Returns
        -------
            dressed energies, if lookup successful, otherwise nan;
        """
        param_indices = self.set_npindextuple(param_indices)
        dressed_index = self.dressed_index(bare_tuple, param_indices)

        if dressed_index is None:
            return np.nan  # type:ignore
        if isinstance(dressed_index, int):
            energy = self["evals"][param_indices + (dressed_index,)]
            if subtract_ground:
                energy -= self["evals"][param_indices + (0,)]
            return energy

        dressed_index = np.asarray(dressed_index)
        energies = np.empty_like(dressed_index)
        it = np.nditer(dressed_index, flags=["multi_index", "refs_ok"])
        sliced_energies = self["evals"][param_indices]

        for location in it:
            location = location.tolist()
            if location is None:
                energies[it.multi_index] = np.nan
            else:
                energies[it.multi_index] = sliced_energies[it.multi_index][location]
                if subtract_ground:
                    energies[it.multi_index] -= sliced_energies[it.multi_index][0]
        return NamedSlotsNdarray(
            energies, sliced_energies._parameters.paramvals_by_name
        )

    @utils.check_sync_status
    def energy_by_dressed_index(
        self,
        dressed_index: int,
        subtract_ground: bool = False,
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> float:
        """
        Look up the dressed eigenenergy belonging to the given dressed index,
        usually to be used with pre-slicing

        Parameters
        ----------
        dressed_index:
            index of dressed state of interest
        subtract_ground:
            whether to subtract the ground state energy
        param_indices:
            specifies the desired choice of parameter values

        Returns
        -------
            dressed energy
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        energies = self["evals"][param_indices_tuple + (dressed_index,)]
        if subtract_ground:
            energies -= self["evals"][param_indices_tuple + (0,)]
        return energies

    @utils.check_sync_status
    def bare_eigenstates(
        self,
        subsys: "QuantumSys",
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> NamedSlotsNdarray:
        """
        Return ndarray of bare eigenstates for given subsystems and parameter index.
        Eigenstates are expressed in the basis internal to the subsystems. Usually to be
        with pre-slicing.
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        subsys_index = self.hilbertspace.get_subsys_index(subsys)
        self._current_param_indices = slice(None, None, None)
        return self["bare_evecs"][subsys_index][param_indices_tuple]

    @utils.check_sync_status
    def bare_eigenvals(
        self,
        subsys: "QuantumSys",
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> NamedSlotsNdarray:
        """
        Return `NamedSlotsNdarray` of bare eigenenergies for given subsystem, usually
        to be used with preslicing.

        Parameters
        ----------
        subsys:
            Hilbert space subsystem for which bare eigendata is to be looked up
        param_indices:
            position indices of parameter values in question

        Returns
        -------
            bare eigenenergies for the specified subsystem and the external parameter
            fixed to the value indicated by its index
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        subsys_index = self.hilbertspace.get_subsys_index(subsys)
        self._current_param_indices = slice(None, None, None)
        return self["bare_evals"][subsys_index][param_indices_tuple]

    def bare_productstate(
        self,
        bare_index: Tuple[int, ...],
    ) -> Qobj:
        """
        Return the bare product state specified by `bare_index`. Note: no parameter
        dependence here, since the Hamiltonian is always represented in the bare
        product eigenbasis.

        Parameters
        ----------
        bare_index:

        Returns
        -------
            ket in full Hilbert space
        """
        subsys_dims = self.hilbertspace.subsystem_dims
        product_state_list = []
        for subsys_index, state_index in enumerate(bare_index):
            dim = subsys_dims[subsys_index]
            product_state_list.append(qt.basis(dim, state_index))
        return qt.tensor(*product_state_list)

    def all_params_fixed(self, param_indices: Union[slice, tuple]) -> bool:
        """
        Checks whether the indices provided fix all the parameters.

        Parameters
        ----------
        param_indices:
            Tuple or slice fixing all or a subset of the parameters.

        Returns
        -------
            True if all parameters are being fixed by `param_indices`.

        """
        if isinstance(param_indices, slice):
            param_indices = (param_indices,)
        return len(self._parameters) == len(param_indices)
