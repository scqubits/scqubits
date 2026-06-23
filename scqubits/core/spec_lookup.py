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
from typing import TYPE_CHECKING, Any, Literal, Protocol
from warnings import warn

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import scqubits.settings as settings
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.namedslots_array import NamedSlotsNdarray, convert_to_std_npindex
import scqubits.core.units as units
from scqubits.utils.spectrum_utils import identity_wrap
import scqubits.utils.plotting as plot
from scqubits.utils.typedefs import NpIndexTuple, NpIndices

if TYPE_CHECKING:
    from scqubits import HilbertSpace
    from scqubits.core.descriptors import WatchedProperty
    from scqubits.core.param_sweep import Parameters
    from scqubits.utils.typedefs import QuantumSys


def _normalize_ba_branch_index(
    branch: int | tuple[int, ...] | list[int] | list[tuple[int, ...]],
    non_primary_count: int,
) -> list[int | tuple[int, ...]]:
    """Normalize ``branch`` into a list of branch indices.

    For a Hilbert space with two subsystems (``non_primary_count == 1``):

    - One branch: an ``int``, a one-element ``tuple[int, ...]``, or a
      one-element ``list[int]`` (e.g. ``0``, ``(0,)``, or ``[0]``).
    - Multiple branches: a ``list[int]`` with more than one entry (e.g.
      ``[0, 1, 2]``).

    For a Hilbert space with three or more subsystems
    (``non_primary_count > 1``):

    - One branch: a ``tuple[int, ...]``, or a ``list[int]`` whose length
      equals ``non_primary_count`` (e.g. ``(0, 1)`` or ``[0, 1]``).
    - Multiple branches: a ``list[tuple[int, ...]]`` (e.g.
      ``[(0, 0), (0, 1)]``).
    """
    if isinstance(branch, int):
        return [branch]
    if isinstance(branch, tuple):
        return [branch]
    if not branch:
        raise ValueError("branch must not be empty.")
    if all(isinstance(entry, tuple) for entry in branch):
        return list(branch)
    if all(isinstance(entry, int) for entry in branch):
        int_branch = [entry for entry in branch if isinstance(entry, int)]
        if len(int_branch) == non_primary_count:
            return [tuple(int_branch)]
        branches_out: list[int | tuple[int, ...]] = []
        for entry in int_branch:
            branches_out.append(entry)
        return branches_out
    raise ValueError("branch list must contain either all ints or all tuples.")


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
    """Mix-in class providing spectrum-lookup methods.

    Used by :class:`HilbertSpace` and :class:`ParameterSweep` to expose
    spectrum and spectrum-lookup methods directly on those classes.
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
        """Return the list of bare-state labels in canonical order.

        For example, for a Hilbert space composed of two subsystems ``sys1``
        and ``sys2``, each label has the form ``(3, 0)``, meaning ``sys1`` is
        in bare eigenstate 3 and ``sys2`` in bare eigenstate 0. The full list
        reads ``[(0,0), (0,1), ..., (0,max_2), (1,0), (1,1), ..., (1,max_2),
        ..., (max_1,0), ..., (max_1,max_2)]``.
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
        - Lexical (``ordering="LX"``): traverse the bare states in
          `lexical order <https://en.wikipedia.org/wiki/Lexicographic_order#Cartesian_products>`__,
          and perform the branch analysis generalized from Dumas et al. (2024).
        - Bare Energy (``ordering="BE"``): traverse the bare states in order
          of their energy before coupling and perform label assignment.
          Particularly useful when the Hilbert space is too large for every
          eigenstate to be labeled.

        Parameters
        ----------
        ordering:
            the ordering method for the dressed state labeling
            - "DE": Dressed Energy (default)
            - "LX": Lexical ordering
            - "BE": Bare Energy
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            provided, lexical ordering is performed on the permuted labels. A
            "branch" is a series of eigenstates formed by putting excitations
            into the last subsystem in the list.
        BEs_count:
            number of eigenstates to be assigned (``"BE"`` scheme only). If
            ``None``, all available eigenstates will be labeled.

        Returns
        -------
        A :class:`.NamedSlotsNdarray` object containing the branch analysis
        results organized by the parameter indices. For each parameter point, a
        flattened multi-dimensional array is stored, representing the dressed
        indices organized by the bare indices. E.g. if the dimensions of the
        subsystems are D0, D1 and D2, the returned array will be ravelled from
        the shape ``(D0, D1, D2)``.
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
        """Return the dressed-state index for a given bare product state.

        Parameters
        ----------
        bare_labels:
            bare-state labels ``(index, index2, ...)`` of length
            ``self.hilbertspace.subsystem_count``.
        param_npindices:
            indices of parameter values of interest. Depending on the nature of
            the slice, this may be a single parameter point or multiple ones.

        Returns
        -------
        Dressed-state index closest to the specified bare state with excitation
        numbers given by ``bare_labels``. If ``param_npindices`` spans multiple
        parameter points, a 1d array of corresponding length is returned.
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
        """Return the bare index corresponding to a given dressed index.

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
        Bare state specification as a tuple. Example: ``(1, 0, 3)`` means
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
            position indices of parameter values in question.

        Returns
        -------
        dressed eigensystem for the external parameter fixed to the value
        indicated by the provided index.
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
        """Return an ndarray of bare eigenstates for the given subsystem and parameter.

        Eigenstates are expressed in the basis internal to the subsystem.
        Typically used with preslicing when part of :class:`.ParameterSweep`.

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

        Typically used with preslicing.

        Parameters
        ----------
        subsys:
            Hilbert-space subsystem for which bare eigenenergies are looked up.
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

    def all_params_fixed(self, param_indices: int | slice | tuple) -> bool:
        """Return whether the provided indices fix every parameter.

        Parameters
        ----------
        param_indices:
            tuple or slice fixing all or a subset of the parameters.

        Returns
        -------
        ``True`` if all parameters are fixed by ``param_indices``.
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

        A dressed state is a superposition of bare states. This method returns
        a dressed state's bare components and the associated occupation
        probabilities, sorted by probability in descending order.

        Parameters
        ----------
        state_label:
            label identifying the dressed state of interest. Either:

            - a tuple/list of bare labels (int), or
            - a single dressed label (int).
        components_count:
            number of components to return. If ``None``, all components are
            returned.
        return_probability:
            if ``True``, return occupation probabilities; if ``False``, return
            probability amplitudes.
        param_npindices:
            indices specifying a point in parameter space. Required for
            multi-dimensional sweeps; otherwise the current parameter
            preslicing is used. Only a :class:`HilbertSpace` or a
            single-parameter :class:`ParameterSweep` may omit this.

        Returns
        -------
        Dictionary mapping bare labels to their associated probability (or
        probability amplitude if ``return_probability=False``).
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

        Branch analysis requires step-by-step excitation of a chosen state to
        cover the entire Hilbert space and complete the assignment of dressed
        indices. Currently returns the creation operator for linear modes, and
        the :math:`\\sum_i |i+1\\rangle\\langle i|` operator for other modes.

        Parameters
        ----------
        mode:
            the mode to be excited.

        Returns
        -------
        The excitation operator for the given mode, tensored with identity
        operators on the other subsystems.
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

        Core routine, run recursively, realizing a depth-first search in the
        tree whose leaves can be labeled by bare labels.

        In brief, the function:

        1. Starts from the "ground" state / starting point of the branch and
           finds all branch states.
        2. Removes the found states from the remaining candidates.
        3. [If at the end of the depth-first search] returns the branch states.
        4. [Otherwise] for each branch state, uses it as an initial state to
           start such a search again, returning a (nested) list of branch
           states. Combines the lists and returns a nested list of states.

        This way, the function recursively traverses the multi-dimensional
        Hilbert space and assigns each eigenstate to its label.

        Parameters
        ----------
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            provided, lexical ordering is performed on the permuted labels.
            Also represents the depth of the subsystem labels to be traversed:
            the later a subsystem appears in the list, the deeper it lies in
            the recursion. A "branch" is a series of eigenstates formed by
            putting excitations into the last subsystem in the list.
        recusion_depth:
            current depth of the recursion. Should be 0 at the start.
        init_drs_idx:
            dressed index of the initial state of this branch.
        init_state:
            initial state of this branch.
        remaining_drs_indices:
            list of remaining dressed indices to be assigned.
        remaining_evecs:
            list of remaining eigenstates to be assigned.

        Returns
        -------
        branch_drs_indices, branch_states
            (nested) lists of branch states and their dressed indices.
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
                _branch_drs_indices, _branch_states = self._branch_analysis_LX_step(
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

        Following Dumas et al. (2024), running through all bare labels in
        lexical order is equivalent to a depth-first traversal of a tree
        structure. The method starts a recursive labeling via
        :meth:`_branch_analysis_LX_step`.

        Eigenstate-bare-state pairing follows the "first-come-first-served"
        principle; the traversal order permutes the bare labels according to
        ``subsys_priority``. The last mode in the list has its states labelled
        sequentially and organized in a single branch.

        Finally, this method organizes the eigenstates into a multi-dimensional
        array according to ``subsys_priority``.

        Parameters
        ----------
        param_indices:
            indices of the parameter sweep to be analyzed.
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            provided, lexical ordering is performed on the permuted labels. A
            "branch" is a series of eigenstates formed by putting excitations
            into the last subsystem in the list.
        transpose:
            if ``True``, the returned array is transposed according to
            ``subsys_priority``; otherwise it has the shape of the subsystem
            dimensions in the original order. Internal knob for testing.

        Returns
        -------
        branch_drs_indices
            multi-dimensional array of dressed indices organized by
            ``subsys_priority``. If the subsystem dimensions are ``D0, D1, D2``,
            the returned array has shape ``(D0, D1, D2)``. If ``transpose`` is
            ``True``, the array is transposed according to ``subsys_priority``.
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
        lowest bare energies. Particularly useful when the Hilbert space is
        too large for every eigenstate to be labeled.

        In bare-energy ordering, the way to obtain excited dressed states is
        ambiguous: e.g., ``|21>`` may be excited from ``|11>`` or ``|20>``. The
        user must therefore supply ``subsys_priority`` to specify the
        path/branch taken. It gives the order in which subsystems are excited;
        the last subsystem in the list is excited if possible.

        Parameters
        ----------
        param_indices:
            indices of the parameter sweep to be analyzed.
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            provided, lexical ordering is performed on the permuted labels. A
            "branch" is a series of eigenstates formed by putting excitations
            into the last subsystem in the list.
        BEs_count:
            number of states to be assigned. If ``None``, all available
            eigenstates are assigned.
        source_maj_vote:
            if ``True``, the branch is determined by majority vote of the
            candidates. Internal knob for testing branch-analysis behavior;
            overrides ``subsys_priority``.

        Returns
        -------
        Multi-dimensional array of dressed indices.
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
          `lexical order <https://en.wikipedia.org/wiki/Lexicographic_order#Cartesian_products>`__
          and perform the branch analysis generalized from
          Dumas et al. (2024).
        - Bare Energy (``ordering="BE"``): traverse the bare states in order
          of their energy before coupling and perform label assignment.
          Particularly useful when the Hilbert space is too large for every
          eigenstate to be labeled.

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

    def _branch_analysis_parse_mode(
        self, mode: "int | QuantumSys", name: str
    ) -> tuple[int, "QuantumSys"]:
        """Resolve a subsystem index or :class:`QuantumSys` instance to ``(index, subsys)``."""
        if isinstance(mode, int):
            if not (0 <= mode < self.hilbertspace.subsystem_count):
                raise ValueError(f"{name} mode index is out of range.")
            mode_idx = mode
            mode_subsys = self.hilbertspace.subsystem_list[mode_idx]
        else:
            if mode not in self.hilbertspace.subsystem_list:
                raise ValueError(f"{name} mode is not found in the HilbertSpace.")
            mode_subsys = mode
            mode_idx = self.hilbertspace.subsystem_list.index(mode_subsys)
        return mode_idx, mode_subsys

    def branch_analysis_exp_vals(
        self,
        primary_mode: "int | QuantumSys",
        secondary_mode: "int | QuantumSys | None" = None,
        observable: Literal["N", "EM"] = "N",
        param_npindices: int | slice | tuple[int, ...] | tuple[slice, ...] = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute observable expectation values for doing branch analysis.

        Returns two arrays where each element corresponds to
        an eigenstate of the system.

        The first array is the expectation value of the primary-mode number
        operator for each eigenstate.
        The second array is controlled by ``observable``:
        - ``"N"``: the occupation number of ``secondary_mode``
        - ``"EM"``: eigenenergy modulo the bare energy of the ``primary_mode``

        Parameters
        ----------
        primary_mode:
            The subsystem (index or instance) whose excitations form branches,
            typically the resonator.
        secondary_mode:
            Subsystem whose ⟨N⟩ is returned when ``observable="N"``.
            If ``observable="N"`` and the Hilbert space has more than two
            subsystems, ``secondary_mode`` must be given.
        observable:
            Choice of observable, either ``"N"`` or ``"EM"``.
        param_npindices:
            Parameter sweep indices; all parameters must be fixed.

        Returns
        -------
        x_val, y_val:
            ``⟨N_prim⟩`` and the selected ``observable`` quantity, shaped like
            ``hilbertspace.subsystem_dims``.
        """
        if not self.all_params_fixed(param_npindices):
            raise ValueError("Not all parameters are fixed.")

        if self.hilbertspace.subsystem_count <= 1:
            raise ValueError(
                "For evaluating branch analysis observables, "
                "the HilbertSpace must have at least 2 subsystems."
            )

        if observable not in ("N", "EM"):
            raise ValueError(f"observable must be 'N' or 'EM', got {observable!r}")

        dims = self.hilbertspace.subsystem_dims
        branch_indices = self["dressed_indices"][param_npindices].reshape(dims)

        primary_mode_idx, primary_subsys = self._branch_analysis_parse_mode(
            primary_mode, "Primary"
        )

        if observable == "N":
            if secondary_mode is None:
                if self.hilbertspace.subsystem_count > 2:
                    raise ValueError(
                        'secondary_mode is required when observable="N" and the '
                        "HilbertSpace has more than 2 subsystems."
                    )
                secondary_mode = 1 - primary_mode_idx
            secondary_mode_idx, secondary_subsys = self._branch_analysis_parse_mode(
                secondary_mode, "Secondary"
            )
            if secondary_mode_idx == primary_mode_idx:
                raise ValueError("Primary and secondary modes cannot be the same.")

        x_N_op = identity_wrap(
            qt.num(primary_subsys.truncated_dim),
            primary_subsys,
            self.hilbertspace.subsystem_list,
            op_in_eigenbasis=True,
        )

        evecs = self["evecs"][param_npindices]
        x_val_arr = np.zeros_like(branch_indices, dtype=float)
        y_val_arr = np.zeros_like(branch_indices, dtype=float)

        if observable == "N":
            y_N_op = identity_wrap(
                qt.num(secondary_subsys.truncated_dim),
                secondary_subsys,
                self.hilbertspace.subsystem_list,
                op_in_eigenbasis=True,
            )
            for idx, drs_idx in np.ndenumerate(branch_indices):
                if drs_idx is None:
                    x_val_arr[idx] = np.nan
                    y_val_arr[idx] = np.nan
                    continue
                ket = evecs[drs_idx]
                x_val_arr[idx] = qt.expect(x_N_op, ket)
                y_val_arr[idx] = qt.expect(y_N_op, ket)
        else:
            bare_primary = self["bare_evals"][primary_mode_idx][param_npindices]
            y_E_mod = bare_primary[1] - bare_primary[0]
            evals = self["evals"][param_npindices]
            for idx, drs_idx in np.ndenumerate(branch_indices):
                if drs_idx is None:
                    x_val_arr[idx] = np.nan
                    y_val_arr[idx] = np.nan
                    continue
                ket = evecs[drs_idx]
                x_val_arr[idx] = qt.expect(x_N_op, ket)
                y_val_arr[idx] = evals[drs_idx] % y_E_mod

        return x_val_arr, y_val_arr

    def _evaluate_BA_n_crit(
        self,
        N_matrix: np.ndarray,
        branch: int | tuple[int, ...] | list[int] | list[tuple[int, ...]],
        primary_mode_idx: int,
        secondary_mode_idx: int,
        occupation_threshold: float = 2,
    ) -> int | None:
        """
        Helper for branch analysis: critical primary-mode occupation for a branch.

        ``N_matrix`` holds ⟨N_secondary⟩ per bare eigenstate label (aligned with
        :meth:`branch_analysis_exp_vals` with ``observable="N"``).

        Parameters
        ----------
        N_matrix:
            ⟨N_secondary⟩ on the bare product grid (subsystem dimensions).
        branch:
            State indices for all modes other than ``primary_mode_idx``.
            A single branch index may be given as an ``int`` (2 subsystems), a
            ``tuple[int, ...]``, or a ``list[int]`` of length
            ``subsystem_count - 1`` (3+ subsystems). Multiple branch indices
            may be given as a ``list[int]`` (2 subsystems), or as a
            ``list[tuple[int, ...]]`` (3+ subsystems). When multiple branch
            indices are given, return the smallest ``n_crit`` found across all
            branches.
        primary_mode_idx:
            Subsystem index of the primary mode.
        secondary_mode_idx:
            Subsystem index of the mode whose number operator expectation is in
            ``N_matrix`` when ``observable="N"``.
        occupation_threshold:
            Offset added to the secondary mode's bare index in the inequality.

        Returns
        -------
        n_crit
            The critical primary-mode occupation index, or None if not reached.
        """
        # grab a column of the N_matrix (branch)
        non_primary_count = len(self.hilbertspace.subsystem_list) - 1
        branches = _normalize_ba_branch_index(branch, non_primary_count)

        n_crit_list: list[int | None] = []
        for br in branches:
            br_indices: list[int] = [br] if isinstance(br, int) else list(br)
            if len(br_indices) != len(self.hilbertspace.subsystem_list) - 1:
                raise ValueError(
                    "Branch must specify one bare index per non-primary subsystem "
                    f"(expected {len(self.hilbertspace.subsystem_list) - 1} "
                    f"entries, got {len(br_indices)})."
                )
            slice_list: list[int | slice] = list(br_indices)
            slice_list.insert(primary_mode_idx, slice(None))
            branch_slice = tuple(slice_list)

            N_branch = N_matrix[branch_slice]

            secondary_branch_idx = sum(
                1 for i in range(secondary_mode_idx) if i != primary_mode_idx
            )

            # find the critical photon number for the primary mode
            N_threshold = br_indices[secondary_branch_idx] + occupation_threshold
            true_indices = np.where(N_branch > N_threshold)[0]
            if len(true_indices) == 0:
                n_crit_list.append(None)  # no critical point found
            else:
                n_crit_list.append(int(true_indices[0]))

        # Filter out None values and return min if there is any, else return None
        n_crit_filtered = [val for val in n_crit_list if val is not None]
        if not n_crit_filtered:
            return None
        return min(n_crit_filtered)

    def branch_analysis_n_crit(
        self,
        primary_mode: "int | QuantumSys",
        branch: int | tuple[int, ...] | list[int] | list[tuple[int, ...]],
        secondary_mode: "int | QuantumSys | None" = None,
        param_npindices: int | slice | tuple[int, ...] | tuple[slice, ...] = 0,
        occupation_threshold: float = 2,
    ) -> int | None:
        """
        Determine the critical occupation number for a given branch from the
        branch analysis results.

        Definition
        ----------
        Eigenstates of the full system are labeled by the bare indices
        (i, j, ..., k, n), where n is the occupation number of the primary mode.
        Let N_sec be the number operator for ``secondary_mode``. For the branch
        given by bare indices of all non-primary modes
        ``(i, j, ..., k)``, the critical primary occupation ``n_crit`` is the
        smallest n such that

        ``⟨ i, j, ..., k, n | N_sec | i, j, ..., k, n ⟩``
        ``> bare_index(sec) + occupation_threshold``

        where ``bare_index(sec)`` is the component of `(i,j,...,k)` that labels
        ``secondary_mode``.

        Requires LX branch labeling so ``dressed_indices`` assigns a dressed
        eigenstate index to each bare product label—either from
        ``HilbertSpace.generate_lookup(ordering='LX')`` or
        ``ParameterSweep(..., labeling_scheme='LX')``.

        Parameters
        ----------
        primary_mode:
            The subsystem (index or instance) whose excitations form branches,
            typically the resonator.
        branch:
            Branch indices for the non-primary modes. A single branch may be
            given as an ``int`` (two subsystems), a ``tuple[int, ...]``, or a
            ``list[int]`` of length ``subsystem_count - 1``. Multiple branches
            may be given as a ``list[int]`` when there is only one non-primary
            subsystem, or as a ``list[tuple[int, ...]]`` otherwise. When
            multiple branches are provided, the smallest critical occupation
            number across all branches is returned.
        secondary_mode:
            The subsystem (index or instance) whose ⟨N⟩ enters the threshold
            comparison (same as the second return value of
            :meth:`branch_analysis_exp_vals` with ``observable="N"``).
            Required when the Hilbert space has more than two subsystems.
        param_npindices:
            Parameter sweep indices to select for the analysis.
        occupation_threshold:
            The threshold for the occupation number that determines the critical
            point.

        Returns
        -------
        n_crit
            Critical occupation number for the specified branch(es). When
            no critical number is found up to the truncation dimension, None
            is returned.
        """
        _, N_matrix = self.branch_analysis_exp_vals(
            primary_mode,
            secondary_mode=secondary_mode,
            observable="N",
            param_npindices=param_npindices,
        )

        primary_mode_idx, _ = self._branch_analysis_parse_mode(primary_mode, "Primary")
        if secondary_mode is None:
            secondary_mode_idx = 1 - primary_mode_idx
        else:
            secondary_mode_idx, _ = self._branch_analysis_parse_mode(
                secondary_mode, "Secondary"
            )

        return self._evaluate_BA_n_crit(
            N_matrix,
            branch,
            primary_mode_idx,
            secondary_mode_idx,
            occupation_threshold=occupation_threshold,
        )

    def plot_branch_analysis(
        self,
        primary_mode: "int | QuantumSys",
        secondary_mode: "int | QuantumSys | None" = None,
        y_axis: Literal["N", "EM"] = "N",
        param_npindices: int | slice | tuple[int, ...] | tuple[slice, ...] = 0,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot branch analysis results where each point corresponds to a system
        eigenstate. The x-axis represents the occupation number of the primary
        mode (typically the resonator). The y-axis can be set to one of two options:
        - ``"N"``: expectation value of the ``secondary_mode`` number operator
        - ``"EM"``: eigenenergy modulo the bare energy of the primary mode

        Requires LX branch labeling so ``dressed_indices`` assigns a dressed
        eigenstate index to each bare product label—either from
        ``HilbertSpace.generate_lookup(ordering='LX')`` or
        ``ParameterSweep(..., labeling_scheme='LX')``.

        Parameters
        ----------
        primary_mode:
            The subsystem (index or instance) whose excitations form branches,
            typically the resonator.
        secondary_mode:
            The subsystem (index or instance) whose occupation number is plotted
            on the y-axis when ``y_axis="N"``.
            Required when y_axis="N" and the Hilbert space has more than two subsystems.
        y_axis:
            Choice of y-axis for the plot, either ``"N"`` or ``"EM"``.
        param_npindices:
            Parameter sweep indices to select for the plot.
        **kwargs:
            Additional keyword arguments passed to ``plot.data_vs_paramvals``.

        Returns
        -------
        fig, ax:
            Matplotlib figure and axes objects containing the plot.
        """
        x_val, y_val = self.branch_analysis_exp_vals(
            primary_mode,
            secondary_mode=secondary_mode,
            observable=y_axis,
            param_npindices=param_npindices,
        )

        dims = self.hilbertspace.subsystem_dims
        primary_mode_idx, primary_subsys = self._branch_analysis_parse_mode(
            primary_mode, "Primary"
        )
        primary_mode = primary_subsys
        primary_mode_dim = dims[primary_mode_idx]
        x_val = np.moveaxis(x_val, primary_mode_idx, 0)
        y_val = np.moveaxis(y_val, primary_mode_idx, 0)

        # Labels
        new_shape = x_val.shape
        label_list = [
            ", ".join(f"{idx}" for idx in indices)
            for indices in np.ndindex(*new_shape[1:])
        ]

        # reshape
        x_val = x_val.reshape(primary_mode_dim, -1)
        y_val = y_val.reshape(primary_mode_dim, -1)

        kwargs_default = {
            "marker": "o",
            "markersize": 2,
            "linewidth": 1,
            "xlabel": rf"$\langle N_\text{{{primary_mode.id_str}}} \rangle$",
        }
        if y_axis == "N":
            if secondary_mode is None:
                _, secondary_for_label = self._branch_analysis_parse_mode(
                    1 - primary_mode_idx, "Secondary"
                )
            else:
                _, secondary_for_label = self._branch_analysis_parse_mode(
                    secondary_mode, "Secondary"
                )
            kwargs_default["ylabel"] = (
                rf"$\langle N_\text{{{secondary_for_label.id_str}}} \rangle$"
            )
        else:
            kwargs_default["ylabel"] = (
                rf"$(E \ \text{{mod}} \ E_\text{{{primary_mode.id_str}}}) / h$  "
                f"[{units.get_units()}]"
            )
        kwargs_default.update(kwargs)

        return plot.data_vs_paramvals(x_val, y_val, label_list, **kwargs_default)
