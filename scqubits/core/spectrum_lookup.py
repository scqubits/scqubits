# spectrum_lookup.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import itertools
import weakref

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj

import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.namedslots_array import NamedSlotsNdarray

if TYPE_CHECKING:
    from scqubits import HilbertSpace, ParameterSweep
    from scqubits.core.qubit_base import QuantumSystem


NpIndex = Union[int, slice, Tuple[int], List[int]]
NpIndexTuple = Tuple[NpIndex, ...]
NpIndices = Union[NpIndex, NpIndexTuple]


class SpectrumLookupMixin:
    """
    Spectrum lookup is an integral building block of the `HilbertSpace` and
    `ParameterSweep` classes. In both cases it provides a convenient way to translate
    back and forth between labelling of eigenstates and eigenenergies via the indices
    of the dressed spectrum j = 0, 1, 2, ... on one hand, and the bare product-state
    labels of the form (0,0,0), (0,0,1), (2,1,3),... (here for the example of three
    subsys_list). The lookup table should be `.generate_lookup()` in the case of a
    `HilbertSpace` object. For `Sweep` objects, the lookup table is
    generated automatically upon init, or manually via `<Sweep>.run()`.
    """

    def __init__(self, hilbertspace: "HilbertSpace"):
        if not hasattr(self, "_hilbertspace"):
            self._hilbertspace = weakref.ref(hilbertspace)

    @property
    def _bare_product_states_labels(self: ParameterSweep) -> List[Tuple[int, ...]]:
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
        return list(itertools.product(*map(range, self._hilbertspace.subsystem_dims)))

    def generate_lookup(self: ParameterSweep) -> NamedSlotsNdarray:
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
        dressed_indices = np.empty(shape=self.parameters.counts, dtype=object)

        param_indices = itertools.product(*map(range, self.parameters.counts))
        for index in param_indices:
            dressed_indices[index] = self._generate_single_mapping(index)
        dressed_indices = np.asarray(dressed_indices[:].tolist())

        parameter_dict = self.parameters.ordered_dict.copy()
        return NamedSlotsNdarray(dressed_indices, parameter_dict)

    def _generate_single_mapping(
        self: ParameterSweep, param_indices: Tuple[int, ...]
    ) -> ndarray:
        """
        For a single set of parameter values, specified by with a tuple of indices
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

        dim = self._hilbertspace.dimension
        dressed_indices: List[Union[int, None]] = [None] * dim
        for dressed_index in range(self._evals_count):
            max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
            max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
            if max_overlap ** 2 > 0.5:
                overlap_matrix[:, max_position] = 0
                dressed_indices[int(max_position)] = dressed_index

        return np.asarray(dressed_indices)

    @utils.check_sync_status
    def dressed_index(
        self: ParameterSweep,
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
        param_indices = param_indices or self._current_param_indices
        try:
            lookup_position = self._bare_product_states_labels.index(bare_labels)
        except ValueError:
            return None
        return self._data["dressed_indices"][param_indices + (lookup_position,)]

    @utils.check_sync_status
    def bare_index(
        self: ParameterSweep,
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
        param_indices = param_indices or self._current_param_indices
        if not self.all_params_fixed(param_indices):
            raise ValueError(
                "All parameters must be fixed to concrete values for "
                "the use of `.bare_index`."
            )
        try:
            lookup_position = np.where(
                self._data["dressed_indices"][param_indices] == dressed_index
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
        self: ParameterSweep, param_indices: Optional[Tuple[int, ...]] = None
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
        param_indices = param_indices or self._current_param_indices
        return self._data["evecs"][param_indices]

    @utils.check_sync_status
    def eigenvals(
        self: ParameterSweep, param_indices: Optional[Tuple[int, ...]] = None
    ) -> ndarray:
        """
        Return the array of dressed eigenenergies

        Parameters
        ----------
            position indices of parameter values in question

        Returns
        -------
            dressed eigenenergies for the external parameters fixed to the values
            indicated by the provided indices
        """
        param_indices = param_indices or self._current_param_indices
        return self._data["evals"][param_indices]

    @utils.check_sync_status
    def energy_by_bare_index(
        self: ParameterSweep,
        bare_tuple: Tuple[int, ...],
        param_indices: Optional[NpIndices] = None,
    ) -> NamedSlotsNdarray:
        """
        Look up dressed energy most closely corresponding to the given bare-state labels

        Parameters
        ----------
        bare_tuple:
            bare state indices
        param_indices:
            indices specifying the set of parameters

        Returns
        -------
            dressed energies, if lookup successful, otherwise nan;
        """
        param_indices = param_indices or self._current_param_indices
        dressed_index = self.dressed_index(bare_tuple, param_indices)

        if dressed_index is None:
            return np.nan
        if isinstance(dressed_index, int):
            return self["evals"][param_indices + (dressed_index,)]

        dressed_index = np.asarray(dressed_index)
        select_energies = np.empty_like(dressed_index)
        it = np.nditer(dressed_index, flags=["multi_index", "refs_ok"])
        sliced_eigenenergies = self["evals"][param_indices]

        for location in it:
            location = location.tolist()
            if location is None:
                select_energies[it.multi_index] = np.nan
            else:
                select_energies[it.multi_index] = sliced_eigenenergies[it.multi_index][
                    location
                ]
        return NamedSlotsNdarray(
            select_energies, sliced_eigenenergies.parameters.paramvals_by_name
        )

    @utils.check_sync_status
    def energy_by_dressed_index(
        self: ParameterSweep,
        dressed_index: int,
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> float:
        """
        Look up the dressed eigenenergy belonging to the given dressed index,
        usually to be used with pre-slicing

        Parameters
        ----------
        dressed_index:
            index of dressed state of interest
        param_indices:
            specifies the desired choice of parameter values

        Returns
        -------
            dressed energy
        """
        param_indices = param_indices or self._current_param_indices
        self._current_param_indices = None
        return self["evals"][param_indices + (dressed_index,)]

    @utils.check_sync_status
    def bare_eigenstates(
        self: ParameterSweep,
        subsys: "QuantumSystem",
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> NamedSlotsNdarray:
        """
        Return ndarray of bare eigenstates for given subsystems and parameter index.
        Eigenstates are expressed in the basis internal to the subsystems. Usually to be
        with pre-slicing.
        """
        param_indices = param_indices or self._current_param_indices
        subsys_index = self._hilbertspace.get_subsys_index(subsys)
        self._current_param_indices = None
        return self["bare_evecs"][subsys_index][param_indices]

    @utils.check_sync_status
    def bare_eigenvals(
        self: ParameterSweep,
        subsys: "QuantumSystem",
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> NamedSlotsNdarray:
        """
        Return list of bare eigenenergies for given subsystem, usually to be used
        with preslicing.

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
        param_indices = param_indices or self._current_param_indices
        subsys_index = self._hilbertspace.get_subsys_index(subsys)
        self._current_param_indices = None
        return self["bare_evals"][subsys_index][param_indices]

    def bare_productstate(self: ParameterSweep, bare_index: Tuple[int, ...]) -> Qobj:
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
        subsys_dims = self._hilbertspace.subsystem_dims
        product_state_list = []
        for subsys_index, state_index in enumerate(bare_index):
            dim = subsys_dims[subsys_index]
            product_state_list.append(qt.basis(dim, state_index))
        return qt.tensor(*product_state_list)

    def all_params_fixed(self: ParameterSweep, param_indices) -> bool:
        if isinstance(param_indices, slice):
            param_indices = (param_indices,)
        return len(self.parameters) == len(param_indices)
