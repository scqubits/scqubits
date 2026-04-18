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

from __future__ import annotations

import itertools
import numbers
from copy import copy
from warnings import warn

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj
from typing_extensions import Protocol

import scqubits.settings as settings
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.namedslots_array import NamedSlotsNdarray, convert_to_std_npindex
from scqubits.utils.typedefs import NpIndexTuple, NpIndices
from scqubits.utils.spectrum_utils import identity_wrap

if TYPE_CHECKING:
    from typing_extensions import Protocol

    from scqubits import HilbertSpace
    from scqubits.core.descriptors import WatchedProperty
    from scqubits.core.param_sweep import Parameters
    from scqubits.utils.typedefs import QuantumSys


class MixinCompatible(Protocol):
    """Structural protocol describing attributes required by :class:`SpectrumLookupMixin`.

    Classes mixing in :class:`SpectrumLookupMixin` (currently
    :class:`HilbertSpace` and :class:`ParameterSweep`) must provide the
    attributes and methods declared here.
    """

    # WatchedProperty is a descriptor; at access time the underlying value type is
    # what subclasses see. Annotate as the underlying type so attribute access
    # patterns (.counts, .paramvals_by_name, etc.) resolve.
    _parameters: "Parameters"
    _evals_count: int
    _current_param_indices: NpIndices
    _ignore_low_overlap: bool
    _data: dict[str, Any]
    _out_of_sync: bool

    def __getitem__(self, key: Any) -> Any:
        """Return the stored entry associated with ``key``.

        Parameters
        ----------
        key:
            indexing object (string label or numpy-style index) selecting an
            entry from the underlying spectrum data.
        """
        ...

    @property
    def hilbertspace(self) -> "HilbertSpace":
        """Return the underlying :class:`HilbertSpace` instance."""
        ...


class SpectrumLookupMixin(MixinCompatible):
    """SpectrumLookupMixin is used as a mix-in class by :class:`ParameterSweep`.

    It makes various
    spectrum and spectrum lookup related methods directly available at the
    :class:`ParameterSweep` level.
    """

    _inside_hilbertspace = False

    def __init_subclass__(cls):
        """Record whether the subclass is :class:`HilbertSpace` itself.

        Sets the class-level flag :attr:`_inside_hilbertspace` which controls
        how preslicing is initialized for instances of the subclass.
        """
        super().__init_subclass__()
        if cls.__name__ == "HilbertSpace":
            cls._inside_hilbertspace = True
        else:
            cls._inside_hilbertspace = False

    def reset_preslicing(self):
        """Reset the current parameter preslicing to select all parameter points.

        When used inside a :class:`HilbertSpace`, preslicing is set to the scalar
        ``0``; for :class:`ParameterSweep` it is set to a tuple of full-slice
        selectors matching the parameter dimensionality.
        """
        if self._inside_hilbertspace:
            self._current_param_indices = 0
        else:
            self._current_param_indices = (
                slice(None, None, None),
            ) * self._parameters.ndim()

    @property
    def _bare_product_states_labels(self) -> list[tuple[int, ...]]:
        """Generates the list of bare-state labels in canonical order.

        For example,
        for a Hilbert space composed of two subsystems sys1 and sys2, each label is
        of the type (3,0) meaning sys1 is in bare eigenstate 3, sys2 in bare
        eigenstate 0. The full list then reads
        [(0,0), (0,1), (0,2), ..., (0,max_2),
        (1,0), (1,1), (1,2), ..., (1,max_2),
        ...
        (max_1,0), (max_1,1), (max_1,2), ..., (max_1,max_2)]
        """
        return list(np.ndindex(*self.hilbertspace.subsystem_dims))

    def generate_lookup(
        self,
        ordering: Literal["DE", "LX", "BE"] = "DE",
        subsys_priority: list[int] | None = None,
        BEs_count: int | None = None,
    ) -> NamedSlotsNdarray:
        """Label the dressed states by bare labels and generate the lookup table.

        The table is built by one of the following methods:

        - Dressed Energy (``ordering="DE"``): traverse the eigenstates
          in the order of their dressed energy, and find the corresponding bare
          state label by overlaps (default).
        - Lexical (``ordering="LX"``): traverse the bare states in `lexical order`_,
          and perform the branch analysis generalized from Dumas et al. (2024).
        - Bare Energy (``ordering="BE"``): traverse the bare states in the order of
          their energy before coupling and perform label assignment. This is
          particularly useful when the Hilbert space is too large and not all the
          eigenstates need to be labeled.

        Parameters
        ----------
        ordering:
            the ordering method for the dressed state labeling
            - "DE": Dressed Energy (default)
            - "LX": Lexical ordering
            - "BE": Bare Energy
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If it is
            provided, lexical ordering is performed on the permuted labels. A
            "branch" is defined as a series of eigenstates formed by putting
            excitations into the last subsystem in the list.
        BEs_count:
            the number of eigenstates to be assigned, for "BE" scheme only. If
            ``None``, all available eigenstates will be labeled.

        Returns
        -------
        A :class:`.NamedSlotsNdarray` object containing the branch analysis
        results organized by the parameter indices. For each parameter point, a
        flattened multi-dimensional array is stored, representing the dressed
        indices organized by the bare indices. E.g. if the dimensions of the
        subsystems are D0, D1 and D2, the returned array will be ravelled from
        the shape ``(D0, D1, D2)``.

        .. _lexical order: https://en.wikipedia.org/wiki/Lexicographic_order#Cartesian_products/
        """
        if ordering == "LX" or ordering == "BE":
            return self._branch_analysis(
                ordering=ordering,
                subsys_priority=subsys_priority,
                transpose=False,
                BEs_count=BEs_count,
            )
        elif ordering == "DE":
            if BEs_count is not None:
                warn(
                    "BEs_count is not supported for DE ordering, " "it will be ignored."
                )
            if subsys_priority is not None:
                warn(
                    "subsys_priority is not supported for DE ordering, "
                    "it will be ignored."
                )
            return self._generate_lookup_by_overlap()
        else:
            raise ValueError(f"Invalid ordering method: {ordering}")

    def _generate_lookup_by_overlap(self) -> NamedSlotsNdarray:
        """Generate the bare-to-dressed state map based on the overlap criterion.

        For each parameter value of the parameter sweep, the mapping between bare
        states and dressed states is established by identifying, for every
        dressed eigenstate, the bare product state with largest overlap.

        Returns
        -------
        A :class:`.NamedSlotsNdarray` whose entries at each parameter point are
        lists of dressed indices whose order corresponds to the ordering of bare
        indices (as stored in ``.canonical_bare_labels``), thus establishing the
        mapping.
        """
        dressed_indices = np.empty(shape=self._parameters.counts, dtype=object)

        param_indices = itertools.product(*map(range, self._parameters.counts))
        for index in param_indices:
            dressed_indices[index] = self._generate_single_mapping_by_overlap(index)
        dressed_indices = np.asarray(dressed_indices[:].tolist())

        parameter_dict = self._parameters.ordered_dict.copy()
        return NamedSlotsNdarray(dressed_indices, parameter_dict)

    def _generate_single_mapping_by_overlap(
        self,
        param_indices: tuple[int, ...],
    ) -> ndarray:
        """Create the per-parameter-point dressed-state index array by overlap.

        For a single set of parameter values, specified by a tuple of indices
        ``param_indices``, build an array of the dressed-state indices in an
        order that corresponds one-to-one to the bare product states with
        largest overlap (whenever possible).

        Parameters
        ----------
        param_indices:
            indices of the parameter values. Length of tuple must match the
            number of parameters in the :class:`.ParameterSweep` object
            inheriting from :class:`SpectrumLookupMixin`.

        Returns
        -------
        1d array of dressed-state indices with dimensions
        ``(self.hilbertspace.dimension,)``, containing the dressed-state
        indices in an order that corresponds to the canonically ordered bare
        product state basis, i.e. ``(0,0,0), (0,0,1), (0,0,2), ..., (0,1,0),
        (0,1,1), (0,1,2), ...`` etc. For example, for two subsystems with two
        states each, the array ``[0, 2, 1, 3]`` would mean: ``(0,0)``
        corresponds to the dressed state 0, ``(0,1)`` to the dressed state 2,
        ``(1,0)`` to the dressed state 1, and ``(1,1)`` to the dressed state 3.
        """
        # Overlaps between dressed energy eigenstates and bare product states, <e1, e2, ...| E_j>
        # Since the Hamiltonian matrix is explicitly constructed in the bare product states basis, this is just the same
        # as the matrix of eigenvectors handed back when diagonalizing the Hamiltonian matrix.
        overlap_matrix = spec_utils.convert_evecs_to_ndarray(
            self._data["evecs"][param_indices]
        )

        dim = self.hilbertspace.dimension
        dressed_indices: list[int | None] = [None] * dim
        for dressed_index in range(self._evals_count):
            max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
            max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
            if self._ignore_low_overlap or (
                max_overlap**2 > settings.OVERLAP_THRESHOLD
            ):
                overlap_matrix[:, max_position] = 0
                dressed_indices[int(max_position)] = dressed_index

        return np.asarray(dressed_indices)

    def set_npindextuple(self, param_indices: NpIndices | None = None) -> NpIndexTuple:
        """Convert ``NpIndices`` parameter indices to a tuple of ``NpIndices``.

        Parameters
        ----------
        param_indices:
            parameter indices to normalize. If ``None``, the currently active
            preslicing stored in ``self._current_param_indices`` is used.

        Returns
        -------
        A tuple of ``NpIndices`` suitable for indexing the stored lookup arrays.
        """
        param_indices = param_indices or self._current_param_indices
        if not isinstance(param_indices, tuple):
            param_indices = (param_indices,)
        return param_indices

    @utils.check_lookup_exists
    @utils.check_sync_status
    def dressed_index(
        self,
        bare_labels: tuple[int, ...],
        param_npindices: NpIndices | None = None,
    ) -> ndarray | int | None:
        """For given bare product state return the corresponding dressed-state index.

        Parameters
        ----------
        bare_labels:
            bare-state labels ``(index, index2, ...)`` of length
            ``self.hilbertspace.subsystem_count``.
        param_npindices:
            indices of parameter values of interest. Depending on the nature of
            the slice, this can be a single parameter point or multiple ones.

        Returns
        -------
        Dressed-state index closest to the specified bare state with excitation
        numbers given by ``bare_labels``. If ``param_npindices`` spans multiple
        parameter points, a corresponding 1d array of length dictated by the
        number of parameter points is returned.
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
        param_indices: tuple[int, ...] | None = None,
    ) -> tuple[int, ...] | None:
        """For given dressed index, look up the corresponding bare index.

        Parameters
        ----------
        dressed_index:
            dressed-state index whose bare label is requested.
        param_indices:
            parameter-index tuple selecting a single parameter point. All
            parameters must be fully fixed; otherwise a :class:`ValueError` is
            raised.

        Returns
        -------
        Bare state specification in tuple form. Example: ``(1,0,3)`` means
        subsystem 1 is in bare state 1, subsystem 2 in bare state 0, and
        subsystem 3 in bare state 3.
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
        param_indices: tuple[int, ...] | None = None,
    ) -> ndarray:
        """Return the list of dressed eigenvectors.

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
        param_indices: tuple[int, ...] | None = None,
    ) -> ndarray:
        """Return the array of dressed eigenenergies.

        Primarily used for running the sweep.

        Parameters
        ----------
        param_indices:
            position indices of parameter values in question.

        Returns
        -------
        Dressed eigenenergies for the external parameters fixed to the values
        indicated by the provided indices.
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        return self._data["evals"][param_indices_tuple]

    @utils.check_lookup_exists
    @utils.check_sync_status
    def energy_by_bare_index(
        self,
        bare_tuple: tuple[int, ...],
        subtract_ground: bool = False,
        param_npindices: NpIndices | None = None,
    ) -> float | NamedSlotsNdarray:  # the return value may also be np.nan
        """Look up the dressed energy most closely matching the given bare labels.

        Parameters
        ----------
        bare_tuple:
            bare state indices.
        subtract_ground:
            whether to subtract the ground state energy.
        param_npindices:
            indices specifying the set of parameters.

        Returns
        -------
        Dressed energies if lookup is successful, otherwise ``np.nan``.
        """
        param_npindices = self.set_npindextuple(param_npindices)
        dressed_index = self.dressed_index(bare_tuple, param_npindices)

        if dressed_index is None:
            return np.nan
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
            location = location.tolist()  # type: ignore[attr-defined]
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
        param_indices: tuple[int, ...] | None = None,
    ) -> float | NamedSlotsNdarray:
        """Look up the dressed eigenenergy belonging to the given dressed index.

        Usually to be used with pre-slicing.

        Parameters
        ----------
        dressed_index:
            index of dressed state of interest.
        subtract_ground:
            whether to subtract the ground state energy.
        param_indices:
            specifies the desired choice of parameter values.

        Returns
        -------
        Dressed energy.
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
        param_indices: tuple[int, ...] | None = None,
    ) -> NamedSlotsNdarray:
        """Return ndarray of bare eigenstates for given subsystems and parameter index.

        Eigenstates are expressed in the basis internal to the subsystems.
        Usually to be used with pre-slicing when part of
        :class:`.ParameterSweep`.

        Parameters
        ----------
        subsys:
            Hilbert-space subsystem for which bare eigenstates are looked up.
        param_indices:
            position indices of parameter values in question.

        Returns
        -------
        :class:`.NamedSlotsNdarray` of bare eigenstates indexed by the
        parameter values selected through ``param_indices``.
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
        param_indices: tuple[int, ...] | None = None,
    ) -> NamedSlotsNdarray:
        """Return :obj:`.NamedSlotsNdarray` of bare eigenenergies for a subsystem.

        Usually to be used with preslicing.

        Parameters
        ----------
        subsys:
            Hilbert space subsystem for which bare eigendata is to be looked up.
        param_indices:
            position indices of parameter values in question.

        Returns
        -------
        Bare eigenenergies for the specified subsystem and the external
        parameter fixed to the value indicated by its index.
        """
        param_indices_tuple = self.set_npindextuple(param_indices)
        subsys_index = self.hilbertspace.get_subsys_index(subsys)
        self.reset_preslicing()
        return self["bare_evals"][subsys_index][param_indices_tuple]

    def bare_productstate(
        self,
        bare_index: tuple[int, ...],
    ) -> Qobj:
        """Return the bare product state specified by ``bare_index``.

        Note: no parameter dependence here, since the Hamiltonian is always
        represented in the bare product eigenbasis.

        Parameters
        ----------
        bare_index:
            tuple of subsystem level indices specifying the product state.

        Returns
        -------
        Ket in the full Hilbert space.
        """
        subsys_dims = self.hilbertspace.subsystem_dims
        product_state_list = []
        for subsys_index, state_index in enumerate(bare_index):
            dim = subsys_dims[subsys_index]
            product_state_list.append(qt.basis(dim, state_index))
        return qt.tensor(*product_state_list)

    def all_params_fixed(self, param_indices: slice | tuple) -> bool:
        """Checks whether the indices provided fix all the parameters.

        Parameters
        ----------
        param_indices:
            tuple or slice fixing all or a subset of the parameters.

        Returns
        -------
        True if all parameters are being fixed by `param_indices`.
        """
        param_indices_std = convert_to_std_npindex(
            np.index_exp[param_indices], self._parameters
        )

        # Check if each dimension is being fixed to a single value or a length-1 array
        fixed = []
        for params, idx in zip(
            self._parameters.paramvals_by_name.values(), param_indices_std
        ):
            fixed.append(np.size(params[idx]) == 1)

        return all(fixed)

    @utils.check_lookup_exists
    @utils.check_sync_status
    def dressed_state_components(
        self,
        state_label: tuple[int, ...] | list[int] | int,
        components_count: int | None = None,
        return_probability: bool = True,
        param_npindices: NpIndices | None = None,
    ) -> dict[tuple[int, ...], float]:
        """Return the bare-state components and probabilities of a dressed state.

        A dressed state is a superposition of bare states. This function returns
        a dressed state's bare components and the associated occupation
        probabilities. They are sorted by probability in descending order.

        Parameters
        ----------
        state_label:
            The bare label of the dressed state of interest. Could be
                - a tuple/list of bare labels (int)
                - a single dressed label (int)

        components_count:
            The number of components to be returned. If None, all components
            will be returned.

        return_probability:
            Whether to return the occupation probabilities. If not, return
            the probability amplitudes.

        param_npindices:
            This method only allows for a HilbertSpace object or a single
            parameter ParameterSweep. If it's a multi-dimensional sweep,
            param_npindices should be provided to specify a point in the
            parameter space. If None, the current parameter preslicing will
            be used.

        Returns
        -------
        A dictionary of the bare labels and their associated probability
        (or probability amplitude if specified).
        """
        param_npindices = self.set_npindextuple(param_npindices)

        if not self.all_params_fixed(param_npindices):
            raise ValueError(
                "All parameters must be fixed to concrete values for "
                "the use of `.dressed_state_component`."
            )

        evecs = self["evecs"][param_npindices]

        # find the desired state vector
        if isinstance(state_label, tuple | list):
            raveled_label = np.ravel_multi_index(
                state_label, self.hilbertspace.subsystem_dims
            )
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
            raveled_label = int(ordered_label[idx])  # type: ignore[assignment]
            bare_label = np.unravel_index(
                raveled_label, self.hilbertspace.subsystem_dims
            )
            prob_amp = evec_1.full()[raveled_label, 0]

            bare_label_list.append(bare_label)

            if return_probability:
                prob = np.abs(prob_amp) ** 2
                prob_list.append(prob)
            else:
                prob_list.append(prob_amp)

        if components_count is not None:
            bare_label_list = bare_label_list[:components_count]
            prob_list = prob_list[:components_count]

        return dict(zip(bare_label_list, prob_list))  # type: ignore[arg-type]

    def _branch_analysis_excite_op(
        self,
        mode: "int | QuantumSys",
    ) -> Qobj:
        """Return the excitation operator used by branch analysis for ``mode``.

        Branch analysis requires a step-by-step excitation of a chosen state,
        which helps cover the entire Hilbert space and complete the assignment
        of dressed indices. For the moment, this returns the creation operator
        for linear modes, and the :math:`\\sum_i |i+1\\rangle\\langle i|`
        operator for other modes.

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
            # sum_j |j+1><j|
            dims = hilbertspace.subsystem_dims
            op = qt.qdiags(
                np.ones(dims[mode_idx] - 1),
                -1,
            )
            return identity_wrap(
                op, mode, hilbertspace.subsystem_list, op_in_eigenbasis=True
            )

    def _branch_analysis_LX_step(
        self,
        subsys_priority: list[int],
        recusion_depth: int,
        init_drs_idx: int,
        init_state: qt.Qobj,
        remaining_drs_indices: list[int],
        remaining_evecs: list[qt.Qobj],
    ) -> tuple[list, list]:
        """Perform one recursive step of branch analysis (Dumas et al. 2024).

        This is a core function to be run recursively, realizing a depth-first
        search in the tree whose leaves can be labeled by bare labels.

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
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If it is
            provided, lexical ordering is performed on the permuted labels.
            It also represents the depth of the subsystem labels to be traversed. The later
            the subsystem appears in the list, the deeper it is in the recursion.
            A "branch" is defined as a series of eigenstates formed by
            putting excitations into the last subsystem in the list.
        recusion_depth:
            the current depth of the recursion. It should be 0 at the beginning.
        init_drs_idx:
            the dressed index of the initial state of this branch.
        init_state:
            the initial state of this branch.
        remaining_drs_indices:
            the list of the remaining dressed indices to be assigned.
        remaining_evecs:
            The list of the remaining eigenstates to be assigned.

        Returns
        -------
        branch_drs_indices, branch_states
            The (nested) list of the branch states and their dressed indices.
        """

        hspace = self.hilbertspace
        mode_index = subsys_priority[recusion_depth]
        mode = hspace.subsystem_list[mode_index]
        terminate_branch_length = hspace.subsystem_dims[mode_index]

        # photon addition operator
        excite_op = self._branch_analysis_excite_op(mode)

        # loop over and find all states that matches the excited initial state
        current_state = init_state
        current_drs_idx = init_drs_idx
        branch_drs_indices: list[Any] = []
        branch_states: list[Any] = []
        while True:
            if recusion_depth == len(subsys_priority) - 1:
                # we are at the end of the depth-first search:
                # just add the state to the branch
                branch_drs_indices.append(current_drs_idx)
                branch_states.append(current_state)
            else:
                # continue the depth-first search:
                # recursively call the function and append all the branch states
                (_branch_drs_indices, _branch_states) = self._branch_analysis_LX_step(
                    subsys_priority,
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
                    "No enough eigenstates to be assigned with a label. "
                    "It's likely that the eignestates are not complete. "
                    "Please try to obtain a complete set of eigenstates by "
                    "increasing `evals_count` before running the branch analysis."
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

    def _branch_analysis_LX(
        self,
        param_indices: tuple[int, ...],
        subsys_priority: list[int] | None = None,
        transpose: bool = False,
    ) -> np.ndarray:
        """Perform full branch analysis at a single parameter point (lexical order).

        Following Dumas et al. (2024), running through all bare labels in the
        lexical order is equivalent to a depth-first traversal in a tree
        structure. The method starts a recursive labeling using
        :meth:`_branch_analysis_LX_step`.

        The eigenstates-bare-state-paring is based on the
        "first-come-first-served" principle, the ordering of such traversal will
        permute the bare labels and change the traversal order based on the
        lexical order. For the last mode in the list, its states will be labelled
        sequentially and organized in a single branch.

        At the end, this function will organize the eigenstates into a
        multi-dimensional array according to the mode_priority.

        Parameters
        ----------
        param_indices:
            the indices of the parameter sweep to be analyzed.

        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            it is provided, lexical ordering is performed on the permuted labels.
            A "branch" is defined as a series of eigenstates formed by putting
            excitations into the last subsystem in the list.

        transpose:
            if True, the returned array will be transposed, according to the
            mode_priority. Otherwise, the array will be in the
            shape of the subsystem dimensions in the original order. Now
            it is a purely internal knob for testing.

        Returns
        -------
        branch_drs_indices
            the multi-dimensional array of the dressed indices organized by
            the mode_priority. If the dimensions of the subsystems are
            D0, D1 and D2, the returned array will have the shape (D0, D1, D2).
            If transposed is True, the array will be transposed according to
            the mode_priority.
        """
        if subsys_priority is None:
            subsys_priority = list(range(self.hilbertspace.subsystem_count))
        else:
            # check if the subsys_priority is a valid permutation of
            # the subsystem indices: length and unique
            if len(subsys_priority) != self.hilbertspace.subsystem_count:
                raise ValueError(
                    "The length of subsys_priority does not match "
                    "the number of subsystems."
                )
            if len(subsys_priority) != len(set(subsys_priority)):
                raise ValueError(
                    "subsys_priority contains duplicate values, "
                    "which is supposed to be a permutation."
                )

        # we assume that the ground state always has bare label (0, 0, ...)
        evecs = self._data["evecs"][param_indices]
        init_state = evecs[0]
        remaining_evecs = list(evecs[1:])
        remaining_drs_indices = list(range(1, self.hilbertspace.dimension))

        branch_drs_indices, _ = self._branch_analysis_LX_step(
            subsys_priority, 0, 0, init_state, remaining_drs_indices, remaining_evecs
        )
        branch_drs_indices = np.array(branch_drs_indices)  # type: ignore[assignment]

        if not transpose:
            reversed_permutation = np.argsort(subsys_priority)
            return np.transpose(branch_drs_indices, reversed_permutation)

        return branch_drs_indices  # type: ignore[return-value]

    def _branch_analysis_BE(
        self,
        param_indices: tuple[int, ...],
        subsys_priority: list[int] | None = None,
        BEs_count: int | None = None,
        source_maj_vote: bool = False,
    ) -> np.ndarray:
        """Perform full branch analysis at one parameter point in bare-energy order.

        Following Dumas et al. (2024), this labels a few eigenstates with the
        lowest bare energies. It is particularly useful when the Hilbert space
        is too large and not all the eigenstates need to be labeled.

        In the bare energy ordering for branch analysis, the way to obtain the
        excited dressed states
        is ambiguous, e.g. |21> can be excited from |11> or |20>. So we need the
        user to input `subsys_priority` to specify the path / branch to be taken.
        It specifies the order of the subsystems to be excited, the last subsystem
        in the list will be excited if possible.

        Parameters
        ----------
        param_indices:
            the indices of the parameter sweep to be analyzed.
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            it is provided, lexical ordering is performed on the permuted labels.
            A "branch" is defined as a series of eigenstates formed by putting
            excitations into the last subsystem in the list.
        BEs_count:
            the number of states to be assigned. If None, all available eigenstates
            will be assigned.
        source_maj_vote:
            if True, the branch will be determined by majority vote of the
            potential candidates. It is purely an internal knob to test the
            behavior of the branch analysis. It overrides mode_priority.

        Returns
        -------
        the multi-dimensional array of the dressed indices
        """
        hspace = self.hilbertspace
        dims = hspace.subsystem_dims

        if subsys_priority is None:
            subsys_priority = list(range(hspace.subsystem_count))

        if BEs_count is None:
            BEs_count = len(self._data["evecs"][param_indices])
        elif len(self._data["evecs"][param_indices]) < BEs_count:
            BEs_count = len(self._data["evecs"][param_indices])
            warn(
                "evals_count is less than BEs_count, BEs_count is set to "
                f"{len(self._data['evecs'][param_indices])}."
            )

        # get the associated excitation operators
        excite_op_list = [
            self._branch_analysis_excite_op(mode) for mode in hspace.subsystem_list
        ]

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
        sorted_indices = np.argsort(bare_evals)[:BEs_count]

        # mode assignment
        branch_drs_indices: np.ndarray = np.ndarray(dims, dtype=object)
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
            for subsys_idx in subsys_priority[::-1]:

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
                overlaps = [
                    np.abs(excited_state.overlap(evec)) for evec in remaining_evecs
                ]
                max_overlap_index = np.argmax(overlaps)

                potential_drs_indices.append(remaining_drs_indices[max_overlap_index])

                if not source_maj_vote:
                    # we only need one path, which is the last one in the mode_priority
                    break
                else:
                    # we need to check all the paths
                    continue

            # do a majority vote, if equal, chose the first one
            # this also works for source_maj_vote = False, when all lists are length 1
            unique_votes, counts = np.unique(potential_drs_indices, return_counts=True)
            vote_result = np.argmax(counts)
            drs_idx = unique_votes[vote_result]
            idx_in_remaining_list = remaining_drs_indices.index(drs_idx)

            # remove the state from the remaining states
            remaining_evecs.pop(idx_in_remaining_list)
            remaining_drs_indices.pop(idx_in_remaining_list)

            branch_drs_indices[tuple(bare_idx)] = drs_idx

        return branch_drs_indices

    def _branch_analysis(
        self,
        ordering: Literal["LX", "BE"] = "BE",
        subsys_priority: list[int] | None = None,
        transpose: bool = False,
        BEs_count: int | None = None,
    ) -> NamedSlotsNdarray:
        """Perform full branch analysis for all parameter points (Dumas et al. 2024).

        Two ordering methods for the labeling are provided:

        - Lexical (``ordering="LX"``): traverse the bare states in
          `lexical order`_, and perform the branch analysis generalized from
          Dumas et al. (2024).
        - Bare Energy (``ordering="BE"``): traverse the bare states in the order
          of their energy before coupling and perform label assignment. This is
          particularly useful when the Hilbert space is too large and not all
          the eigenstates need to be labeled.

        Parameters
        ----------
        ordering:
            the ordering method for the labeling
            - "LX": Lexical ordering
            - "BE": Bare Energy
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If it is
            provided, lexical ordering is performed on the permuted labels. A
            "branch" is defined as a series of eigenstates formed by putting
            excitations into the last subsystem in the list.
        transpose:
            if ``True``, the array returned by the lexical-ordering pass is
            transposed according to ``subsys_priority``; otherwise it is
            organized in the original subsystem order. Internal knob for
            testing.
        BEs_count:
            the number of eigenstates to be labeled, for "BE" scheme only. If
            ``None``, all available eigenstates will be labeled.

        Returns
        -------
        A :class:`.NamedSlotsNdarray` object containing the branch analysis
        results organized by the parameter indices. For each parameter point, a
        flattened multi-dimensional array is stored, representing the dressed
        indices organized by the bare indices. E.g. if the dimensions of the
        subsystems are D0, D1 and D2, the returned array will be ravelled from
        the shape ``(D0, D1, D2)``.

        .. _lexical order: https://en.wikipedia.org/wiki/Lexicographic_order#Cartesian_products/
        """
        dressed_indices = np.empty(shape=self._parameters.counts, dtype=object)

        param_indices = itertools.product(*map(range, self._parameters.counts))

        for index in param_indices:
            if ordering == "LX":
                if BEs_count is not None:
                    warn(
                        "BEs_count is not supported for lexical ordering, "
                        "it will be ignored."
                    )
                dressed_indices[index] = self._branch_analysis_LX(
                    index,
                    subsys_priority,
                    transpose,
                )
            elif ordering == "BE":
                dressed_indices[index] = self._branch_analysis_BE(
                    index,
                    subsys_priority,
                    BEs_count,
                )
            else:
                raise ValueError(f"Ordering {ordering} is not supported.")

        dressed_indices = np.asarray(dressed_indices[:].tolist())

        parameter_dict = self._parameters.ordered_dict.copy()
        shape = self._parameters.counts
        return NamedSlotsNdarray(
            dressed_indices.reshape(shape + (-1,)),
            parameter_dict,
        )
