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
import numbers
from copy import copy
from warnings import warn

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj
from typing_extensions import Protocol

import scqubits.settings as settings
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.utils.typedefs import NpIndexTuple, NpIndices
from scqubits.utils.spectrum_utils import identity_wrap

if TYPE_CHECKING:
    from typing_extensions import Protocol

    from scqubits import HilbertSpace
    from scqubits.core.descriptors import WatchedProperty
    from scqubits.core.param_sweep import Parameters
    from scqubits.utils.typedefs import QuantumSys


class MixinCompatible(Protocol):
    _parameters: "WatchedProperty[Parameters]"
    _evals_count: "WatchedProperty[int]"
    _current_param_indices: NpIndices
    _ignore_low_overlap: bool
    _data: Dict[str, Any]
    _out_of_sync: bool
    hilbertspace: "HilbertSpace"

    def __getitem__(self, key: Any) -> Any: ...

    @property
    def hilbertspace(self) -> "HilbertSpace": ...


class SpectrumLookupMixin(MixinCompatible):
    """
    SpectrumLookupMixin is used as a mix-in class by `ParameterSweep`. It makes various
    spectrum and spectrum lookup related methods directly available at the
    `ParameterSweep` level.
    """

    _inside_hilbertspace = False

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__ == "HilbertSpace":
            cls._inside_hilbertspace = True
        else:
            cls._inside_hilbertspace = False

    def reset_preslicing(self):
        if self._inside_hilbertspace:
            self._current_param_indices = 0
        else:
            self._current_param_indices = (slice(None, None, None),) * self._parameters.ndim()

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
        that corresponds one-to-one to the bare product states with largest overlap
        (whenever possible).

        Parameters
        ----------
        param_indices:
            indices of the parameter values
            Length of tuple must match the number of parameters in the `ParameterSweep` object inheriting from
            `SpectrumLookupMixin`.

        Returns
        -------
            1d array of dressed-state indices
            Dimensions: (`self.hilbertspace.dimension`,)

            Array which contains the dressed-state indices in an order that corresponds to the canonically ordered bare
            product state basis, i.e. (0,0,0), (0,0,1), (0,0,2), ..., (0,1,0), (0,1,1), (0,1,2), ... etc.
            For example, for two subsystems with two states each, the array [0, 2, 1, 3] would mean:
            (0,0) corresponds to the dressed state 0,
            (0,1) corresponds to the dressed state 2,
            (1,0) corresponds to the dressed state 1,
            (1,1) corresponds to the dressed state 3.
        """
        # Overlaps between dressed energy eigenstates and bare product states, <e1, e2, ...| E_j>
        # Since the Hamiltonian matrix is explicitly constructed in the bare product states basis, this is just the same
        # as the matrix of eigenvectors handed back when diagonalizing the Hamiltonian matrix.
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
        """
        Convert the NpIndices parameter indices to a tuple of NpIndices.
        """
        param_indices = param_indices or self._current_param_indices
        if not isinstance(param_indices, tuple):
            param_indices = (param_indices,)
        return param_indices

    @utils.check_lookup_exists
    @utils.check_sync_status
    def dressed_index(
        self,
        bare_labels: Tuple[int, ...],
        param_npindices: Optional[NpIndices] = None,
    ) -> Union[ndarray, int, None]:
        """
        For given bare product state return the corresponding dressed-state index.

        Parameters
        ----------
        bare_labels:
            bare_labels = (index, index2, ...)
            Dimension: (`self.hilbertspace.subsystem_count`,)
        param_npindices:
            indices of parameter values of interest
            Depending on the nature of the slice, this can be a single parameter point or multiple ones.

        Returns
        -------
            dressed state index closest to the specified bare state with excitation numbers given by `bare_labels`.
            If `param_npindices` spans multiple parameter points, then this returns a corresponding 1d array of
            length dictated by the number of parameter points.
        """
        param_npindices = self.set_npindextuple(param_npindices)
        try:
            lookup_position = self._bare_product_states_labels.index(bare_labels)
        except ValueError:
            return None

        return self._data["dressed_indices"][param_npindices + (lookup_position,)]

    @utils.check_lookup_exists
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

    @utils.check_lookup_exists
    @utils.check_sync_status
    def energy_by_bare_index(
        self,
        bare_tuple: Tuple[int, ...],
        subtract_ground: bool = False,
        param_npindices: Optional[NpIndices] = None,
    ) -> Union[float, NamedSlotsNdarray]:  # the return value may also be np.nan
        """
        Look up dressed energy most closely corresponding to the given bare-state labels

        Parameters
        ----------
        bare_tuple:
            bare state indices
        subtract_ground:
            whether to subtract the ground state energy
        param_npindices:
            indices specifying the set of parameters

        Returns
        -------
            dressed energies, if lookup successful, otherwise nan;
        """
        param_npindices = self.set_npindextuple(param_npindices)
        dressed_index = self.dressed_index(bare_tuple, param_npindices)

        if dressed_index is None:
            return np.nan  # type:ignore
        if isinstance(dressed_index, numbers.Number):
            energy = self["evals"][param_npindices + (dressed_index,)]
            if subtract_ground:
                energy -= self["evals"][param_npindices + (0,)]
            return energy

        dressed_index = np.asarray(dressed_index)
        energies = np.empty_like(dressed_index, dtype=np.float64)
        it = np.nditer(dressed_index, flags=["multi_index", "refs_ok"])
        sliced_energies = self["evals"][param_npindices]

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

    @utils.check_lookup_exists
    @utils.check_sync_status
    def energy_by_dressed_index(
        self,
        dressed_index: int,
        subtract_ground: bool = False,
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> Union[float, NamedSlotsNdarray]:
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

    @utils.check_lookup_exists
    @utils.check_sync_status
    def bare_eigenstates(
        self,
        subsys: "QuantumSys",
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> NamedSlotsNdarray:
        """
        Return ndarray of bare eigenstates for given subsystems and parameter index.
        Eigenstates are expressed in the basis internal to the subsystems. Usually to be
        used with pre-slicing when part of `ParameterSweep`.
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        subsys_index = self.hilbertspace.get_subsys_index(subsys)
        self.reset_preslicing()
        return self["bare_evecs"][subsys_index][param_indices_tuple]

    @utils.check_lookup_exists
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
        self.reset_preslicing()
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
            # Convert a single slice to a tuple containing just that slice
            param_indices = (param_indices,)

        # Check if each element in param_indices is an integer or a non-range slice
        # A non-range slice would be something like slice(1, 2) which effectively selects a single index
        fixed = []
        for dim, idx in enumerate(param_indices):
            
            if len(self._parameters[dim]) == 1:
                # if the parameter has only one value, then it is already fixed
                fixed.append(True)
            elif isinstance(idx, int):
                # if the parameter is a single index, then it is fixed
                fixed.append(True)
            elif (
                isinstance(idx, slice) 
                and idx.start is not None 
                and isinstance(idx.start, str)
            ):
                # if the parameter is a name based slice and the only allowed
                # name based slice is a single value
                fixed.append(True)
            elif (
                isinstance(idx, slice) 
                and idx.start is not None 
                and idx.stop is not None 
                and idx.stop - idx.start == 1
            ):
                # if the parameter is a single value slice, then it is fixed
                fixed.append(True)
            else:
                fixed.append(False)

        return all(fixed)

    @utils.check_lookup_exists
    @utils.check_sync_status
    def dressed_state_component(
        self, 
        state_label: Union[Tuple[int, ...], List[int], int],
        num_components: Union[int, None] = None,
        param_npindices: Optional[NpIndices] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        A dressed state is a superposition of bare states. This function returns 
        a dressed state's bare conponents and the 
        corresponding occupation probability. 
        They are sorted by probability in descending order.

        Parameters
        ----------
        state_label:
            The bare label of the dressed state of interest. Could be 
                - a tuple/list of bare labels (int)
                - a single dressed label (int)
                
        num_components:
            The number of components to be returned. If None, all components 
            will be returned.
            
        param_npindices:
            The parameter indices to be used. If None, the current parameter 
            indices will be used.
        
        Returns
        -------
        A tuple of two lists: 
            - the first list contains the bare labels of the components
            - the second list contains the occupation probabilities of the components
        """
        param_npindices = self.set_npindextuple(param_npindices)

        if not self.all_params_fixed(param_npindices):
            raise ValueError(
                "All parameters must be fixed to concrete values for "
                "the use of `.dressed_state_component`."
            )

        # I don't know why I use this zero_idx for slicing before.
        # Now I think it is totally wrong.
        # zero_idx = (0,) * len(self._parameters)
        
        evecs = self["evecs"][param_npindices]
            
        # find the desired state vector
        if isinstance(state_label, tuple | list): 
            raveled_label = np.ravel_multi_index(state_label, self.hilbertspace.subsystem_dims)
            drs_idx = self["dressed_indices"][param_npindices][raveled_label]
            if drs_idx is None:
                raise IndexError(f"no dressed state found for bare label {state_label}")
        elif isinstance(state_label, int | np.int_):
            drs_idx = state_label
        evec_1 = evecs[drs_idx]

        ordered_label = np.argsort(np.abs(evec_1.full()[:, 0]))[::-1]
        bare_label_list = []
        prob_list = []
        for idx in range(evec_1.shape[0]):
            raveled_label = int(ordered_label[idx])
            bare_label = np.unravel_index(raveled_label, self.hilbertspace.subsystem_dims)
            prob = (np.abs(evec_1.full()[:, 0])**2)[raveled_label]

            bare_label_list.append(bare_label)
            prob_list.append(prob)

        if num_components is not None:
            bare_label_list = bare_label_list[:num_components]
            prob_list = prob_list[:num_components]

        return bare_label_list, prob_list

    def _branch_analysis_excite_op(
        self,
        mode: "Union[int, QuantumSys]",
    ) -> Qobj:
        """
        Branch analysis requires a step by step excitation of a chosen state, 
        which help to cover the entire Hilbert space and complete the 
        assignment of dressed indices.
        This function returns the excitation operator for a given mode.
        
        For the moment, it returns the creation operator for linear modes,
        and Sum_i |i+1><i| operator for other modes.
        
        Parameters
        ----------
        mode:
            The mode to be excited.
        
        Returns
        -------
        The excitation operator for the given mode, tensor producted with 
        the identity operators of the other subsystems.
        """
        hilbertspace = self.hilbertspace
        if isinstance(mode, int):
            mode_idx = mode
            mode = hilbertspace.subsystem_list[mode]
        else:
            mode_idx = hilbertspace.subsystem_list.index(mode)
            
        if mode in hilbertspace.osc_subsys_list:
            # annhilation operator
            return hilbertspace.annihilate(mode).dag()
        else:
            # a sum of |j+1><j|
            dims = hilbertspace.subsystem_dims
            op = qt.qdiags(
                np.ones(dims[mode_idx] - 1),
                -1,
            )
            return identity_wrap(op, mode, hilbertspace.subsystem_list)
            
    def _branch_analysis_DF_step(
        self,
        mode_priority: List[int],
        recusion_depth: int,
        init_drs_idx: int, 
        init_state: qt.Qobj, 
        remaining_drs_indices: List[int], 
        remaining_evecs: List[qt.Qobj], 
    ) -> Tuple[List, List]:
        """
        Perform a single branch analysis according to Dumas et al. (2024). This 
        is a core function to be run recursively, which realized a depth-first
        search in the tree - its leaves can be labeled by bare labels.

        In a nutshell, the function will:
        1. Start from the "ground" state / starting point the branch, find
        all of the branch states
        2. Remove the found states from the remaining candidates
        3. [If at the end of the depth-first search] Return the branch states
        4. [If not at the end] For each branch state, use it as an init state to 
        start such search again, which will return a (nested) list of branch 
        states. Combine the list of branch states and return a nested list of
        those states

        In such way, the function will recursively go through this multi-dimensional
        Hilbert space and assign the eigenstates to their labels.

        Parameters
        ----------
        self:
            SpectrumLookupMixin object, could be a `ParameterSweep` object or 
            `HilbertSpace` object.
        mode_priority:
            A permutation of the mode indices. 
            It represents the depth of the mode labels to be traversed. The later
            the mode appears in the list, the deeper it is in the recursion.
            For the last mode in the list, its states will be organized in a 
            single branch - the innermost part of the nested list. 
        recusion_depth:
            The current depth of the recursion. It should be 0 at the beginning.
        init_drs_idx:
            The dressed index of the initial state of this branch.
        init_state:
            The initial state of this branch.
        remaining_drs_indices:
            The list of the remaining dressed indices to be assigned.
        remaining_evecs:
            The list of the remaining eigenstates to be assigned.
        
        Returns
        -------
        branch_drs_indices, branch_states
            The (nested) list of the branch states and their dressed indices.
        """

        hspace = self.hilbertspace
        mode_index = mode_priority[recusion_depth]
        mode = hspace.subsystem_list[mode_index]
        terminate_branch_length = hspace.subsystem_dims[mode_index]

        # photon addition operator
        excite_op = self._branch_analysis_excite_op(mode)

        # loop over and find all states that matches the excited initial state
        current_state = init_state
        current_drs_idx = init_drs_idx
        branch_drs_indices = []
        branch_states = []
        while True:
            if recusion_depth == len(mode_priority) - 1:
                # we are at the end of the depth-first search:
                # just add the state to the branch
                branch_drs_indices.append(current_drs_idx)
                branch_states.append(current_state)
            else:
                # continue the depth-first search:
                # recursively call the function and append all the branch states
                (
                    _branch_drs_indices, _branch_states
                ) = self._branch_analysis_DF_step(
                    mode_priority, 
                    recusion_depth + 1,
                    current_drs_idx,
                    current_state, 
                    remaining_drs_indices,
                    remaining_evecs, 
                )
                branch_drs_indices.append(_branch_drs_indices)
                branch_states.append(_branch_states)

            # if the branch is long enough, terminate the loop
            if len(branch_states) == terminate_branch_length:
                break

            # find the closest state to the excited current state
            if len(remaining_evecs) == 0:
                raise ValueError(
                    "No more states to assign. It's likely that the eignestates "
                    "are not complete. Please try to obtain a complete set of "
                    "eigenstates by increasing `evals_count`."
                )

            excited_state = (excite_op * current_state).unit()
            overlaps = [np.abs(excited_state.overlap(evec)) for evec in remaining_evecs]
            max_overlap_index = np.argmax(overlaps)

            current_state = remaining_evecs[max_overlap_index]
            current_drs_idx = remaining_drs_indices[max_overlap_index]

            # remove the state from the remaining states
            remaining_evecs.pop(max_overlap_index)
            remaining_drs_indices.pop(max_overlap_index)

        return branch_drs_indices, branch_states

    def branch_analysis_DF(
        self,
        param_indices: Tuple[int, ...],
        mode_priority: Optional[List[int]] = None,
        transpose: bool = False,
    ) -> np.ndarray:
        """
        Perform a full branch analysis according to Dumas et al. (2024) for 
        a single parameter point using depth-first traversal. It will start 
        a recursive search using method `_branch_analysis_DF_step`.

        Since the eigenstates-picking is "first-come-first-served", the 
        ordering of such search will play an important role, which is specified
        by `mode_priority`. It represents the depth of the mode labels to 
        be traversed. The later the mode appears in the list, the deeper it is 
        in the recursion. For the last mode in the list, its states will be 
        organized in a single branch - the innermost part of the nested list.
        
        At the end, this function will organize the eigenstates into a 
        multi-dimensional array according to the mode_priority. 

        Parameters
        ----------
        self:
            SpectrumLookupMixin object, could be a `ParameterSweep` object or 
            `HilbertSpace` object.
        param_indices:
            The indices of the parameter sweep to be analyzed.
        mode_priority:
            A permutation of the mode indices. 
            It represents the depth of the mode labels to be traversed. The later
            the mode appears in the list, the deeper it is in the recursion.
            For the last mode in the list, its states will be organized in a 
            single branch - the innermost part of the nested list.
        transpose:
            If True, the returned array will be transposed according to the
            mode_priority. Otherwise, the array will be in the shape of 
            the subsystem dimensions in the original order.

        Returns
        -------
        branch_drs_indices
            The multi-dimensional array of the dressed indices organized by 
            the mode_priority. If the dimensions of the subsystems are
            D0, D1 and D2, the returned array will have the shape (D0, D1, D2).
            If transposed is True, the array will be transposed according to
            the mode_priority.
        """
        if mode_priority is None:
            mode_priority = list(range(self.hilbertspace.subsystem_count))
        
        # we assume that the ground state always has bare label (0, 0, ...)
        evecs = self._data["evecs"][param_indices]
        init_state = evecs[0]
        remaining_evecs = list(evecs[1:])
        remaining_drs_indices = list(range(1, self.hilbertspace.dimension))

        branch_drs_indices, _ = self._branch_analysis_DF_step(
            mode_priority, 
            0, 
            0, init_state,
            remaining_drs_indices, remaining_evecs
        )
        branch_drs_indices = np.array(branch_drs_indices)

        if not transpose:
            reversed_permutation = np.argsort(mode_priority)
            return np.transpose(
                branch_drs_indices, reversed_permutation
            )

        return branch_drs_indices

    def branch_analysis_EF(
        self,
        param_indices: Tuple[int, ...],
        truncate: int | None = None,
    ) -> np.ndarray:
        """
        Perform a full branch analysis according to Dumas et al. (2024) for 
        a single parameter point for a few eigenstates with the lowest bare 
        energies. It is particularly useful when the Hilbert space is too large 
        and not all the eigenstates need to be labeled.
        
        Parameters
        ----------
        param_indices:
            The indices of the parameter sweep to be analyzed.
        truncate:
            The number of states to be assigned. If None, all states will be 
            assigned.
        """
        hspace = self.hilbertspace
        dims = hspace.subsystem_dims
        
        if truncate is None:
            truncate = len(self._data["evecs"][param_indices])
        elif len(self._data["evecs"][param_indices]) < truncate:
            truncate = len(self._data["evecs"][param_indices])
            warn(
                "evals_count is less than truncate, truncate is set to "
                f"{len(self._data['evecs'][param_indices])}."
            )
        
        # get the associated excitation operators
        excite_op_list = [self._branch_analysis_excite_op(mode) for mode in hspace.subsystem_list]
        
        # generate a list of their bare energies
        bare_evals_by_sys = self._data["bare_evals"]
        bare_evals = np.zeros(dims)
        for idx in np.ndindex(tuple(dims)):
            subsys_eval = [
                bare_evals_by_sys[subsys_idx][param_indices][level_idx]
                for subsys_idx, level_idx in enumerate(idx)
            ]
            bare_evals[idx] = np.sum(subsys_eval)
        bare_evals = bare_evals.ravel()
        
        # sort the bare energies
        # which will be the order of state assignment
        sorted_indices = np.argsort(bare_evals)[:truncate]
            
        # mode assignment
        branch_drs_indices = np.ndarray(dims, dtype=object)
        branch_drs_indices.fill(None)
        evecs = self._data["evecs"][param_indices]
        remaining_evecs = list(evecs)
        remaining_drs_indices = list(range(0, self.hilbertspace.dimension))
        
        for raveled_bare_idx in sorted_indices:
            # assign the dressed index for bare_idx
            bare_idx = list(np.unravel_index(raveled_bare_idx, dims))
            
            if raveled_bare_idx == 0:
                # the (0, 0, ...) is always assigned the dressed index 0
                branch_drs_indices[tuple(bare_idx)] = 0
                remaining_drs_indices.pop(0)
                remaining_evecs.pop(0)
                continue
            
            # get previously assigned states (one less excitation) 
            # By comparing the excited states with the dressed states,
            # we can find the dressed index of the current state
            prev_bare_indices = []
            potential_drs_indices = []
            for subsys_idx in range(len(dims)):
                
                # obtain the a bare index with one less excitation
                prev_idx = copy(bare_idx)
                if prev_idx[subsys_idx] == 0:
                    continue
                prev_idx[subsys_idx] -= 1
                prev_drs_idx = branch_drs_indices[tuple(prev_idx)]
                
                prev_bare_indices.append(prev_idx)
                
                # state vector
                prev_state = evecs[prev_drs_idx]
                excited_state = excite_op_list[subsys_idx] * prev_state
                excited_state = excited_state.unit()
                
                # find the dressed index
                overlaps = [np.abs(excited_state.overlap(evec)) for evec in remaining_evecs]
                max_overlap_index = np.argmax(overlaps)
                
                potential_drs_indices.append(remaining_drs_indices[max_overlap_index])
                
            # do a majority vote, if equal, chose the first one
            unique_votes, counts = np.unique(potential_drs_indices, return_counts=True)
            vote_result = np.argmax(counts)
            drs_idx = unique_votes[vote_result]
            idx_in_remaining_list = remaining_drs_indices.index(drs_idx)
            
            # remove the state from the remaining states
            remaining_evecs.pop(idx_in_remaining_list)
            remaining_drs_indices.pop(idx_in_remaining_list)
            
            branch_drs_indices[tuple(bare_idx)] = drs_idx
            
        return branch_drs_indices

    def branch_analysis(
        self,
        mode: Literal["DF", "EF"] = "EF",
        mode_priority: Optional[List[int]] = None,
        transpose: bool = False,
        truncate: int | None = None,
    ) -> NamedSlotsNdarray:
        """
        Perform a full branch analysis for all parameter points, according to 
        Dumas et al. (2024). We provide two orderings methods for the labeling:
        - "DF": depth-first traversal the tree formed by the bare state labels
        - "EF": traversal ordered by the bare energy, which is particularly 
            useful when the Hilbert space is too large and not all the eigenstates
            need to be labeled.
        
        Parameters
        ----------
        mode: Literal["DF", "EF"]
            The ordering method for the labeling
            - "DF": depth-first traversal the tree formed by the bare state labels
            - "EF": traversal ordered by the bare energy  
            
        mode_priority: List[int] | None
            A permutation of the mode indices. 
            Since the eigenstates-bare-state-paring is based on the 
            "first-come-first-served" principle, the ordering of such traversal will 
            play an important role, which is specified by `mode_priority`. It 
            represents the depth of the mode labels to be traversed. The later 
            the mode appears in the list, the deeper it is in the recursion. For 
            the last mode in the list, its states will be organized in a single 
            branch - the innermost part of the nested list.
        
        transpose: bool
            For "DF" mode only. If True, the returned array will be transposed 
            according to the mode_priority. Otherwise, the array will be in the 
            shape of the subsystem dimensions in the original order.
            
        truncate: int | None
            For "EF" mode only. The number of eigenstates to be assigned.
            
        Returns
        -------
        branch_drs_indices: NamedSlotsNdarray
            A NamedSlotsNdarray object containing the branch analysis results
            organized by the parameter indices.
            Each element is a multi-dimensional array of the 
            dressed indices organized by the mode_priority. If the dimensions 
            of the subsystems are D0, D1 and D2, the returned array will have 
            the shape (D0, D1, D2). If transposed is True, the array will be 
            transposed according to the mode_priority (for "DF" mode only).
        """
        dressed_indices = np.empty(shape=self._parameters.counts, dtype=object)

        param_indices = itertools.product(*map(range, self._parameters.counts))
        
        for index in param_indices:
            if mode == "DF":
                dressed_indices[index] = self.branch_analysis_DF(
                    index, mode_priority, transpose,
                )
            elif mode == "EF":
                dressed_indices[index] = self.branch_analysis_EF(
                    index, truncate,
                )
            else:
                raise ValueError(f"Mode {mode} is not supported.")
            
        dressed_indices = np.asarray(dressed_indices[:].tolist())

        parameter_dict = self._parameters.ordered_dict.copy()
        shape = self._parameters.counts
        return NamedSlotsNdarray(
            dressed_indices.reshape(shape + (-1,)), 
            parameter_dict,
        )

