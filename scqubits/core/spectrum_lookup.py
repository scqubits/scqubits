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
import warnings
import weakref

from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj

import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.namedslots_array import NamedSlotsNdarray

if TYPE_CHECKING:
    from scqubits import HilbertSpace
    from scqubits.core.qubit_base import QuantumSystem
    from scqubits.io_utils.fileio_qutip import QutipEigenstates


def check_sync_status(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._out_of_sync:
            warnings.warn(
                "SCQUBITS\nSpectrum lookup data is out of sync with systems originally"
                " involved in generating it. This will generally lead to incorrect"
                " results. Consider regenerating the lookup data using"
                " <HilbertSpace>.generate_lookup() or <ParameterSweep>.run()",
                Warning,
            )
        return func(self, *args, **kwargs)

    return wrapper


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
        return list(itertools.product(*map(range, self._hilbertspace.subsystem_dims)))

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
        dressed_indices = np.empty(shape=self.parameters.counts, dtype=object)

        param_indices = itertools.product(*map(range, self.parameters.counts))
        for index in param_indices:
            dressed_indices[index] = self._generate_single_mapping(index)

        return NamedSlotsNdarray(dressed_indices, self.parameters.ordered_dict)

    def _generate_single_mapping(self, param_indices: Tuple[int, ...]) -> ndarray:
        """
        For a single set of parameter values, specified by with a tuple of indices
        ``param_indices`` (or a single integer if one one parameter is involved)
        create an array of the dressed-state indices in an order that corresponds one to
        one to the bare product states with largest overlap (whenever possible).

        Parameters
        ----------
        param_indices:
            indices of the parameter values

        Returns
        -------
            dressed-state indices
        """
        overlap_matrix = spec_utils.convert_esys_to_ndarray(
            self._data["esys"][param_indices][1]
        )

        dressed_indices: List[Union[int, None]] = []
        # for given bare basis index, find dressed index
        for bare_basis_index in range(self._hilbertspace.dimension):
            max_position = (np.abs(overlap_matrix[:, bare_basis_index])).argmax()
            max_overlap = np.abs(overlap_matrix[max_position, bare_basis_index])
            if max_overlap < 0.5:  # overlap too low, make no assignment
                dressed_indices.append(None)
            else:
                dressed_indices.append(max_position)
        return np.asarray(dressed_indices, dtype=object)

    @check_sync_status
    def dressed_index(
        self,
        bare_labels: Tuple[int, ...],
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> Union[int, None]:
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
        return self._data["dressed_indices"][param_indices][lookup_position]

    @check_sync_status
    def bare_index(
        self, dressed_index: int, param_indices: Optional[Tuple[int, ...]] = None
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
        try:
            lookup_position = np.where(
                self._data["dressed_indices"][param_indices] == dressed_index
            )[0][0]
        except ValueError:
            return None
        basis_labels = self._bare_product_states_labels[lookup_position]
        return basis_labels

    @check_sync_status
    def eigensys(self, param_indices: Optional[Tuple[int, ...]] = None) -> ndarray:
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
        return self._data["esys"][param_indices]

    @check_sync_status
    def eigenvals(self, param_indices: Optional[Tuple[int, ...]] = None) -> ndarray:
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
        return self._data["esys"][param_indices][0]

    @check_sync_status
    def energy_by_bare_index(
        self,
        bare_tuple: Tuple[int, ...],
        param_indices: Optional[Tuple[int, ...]] = None,
    ) -> Union[float, None]:
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
            dressed energy, if lookup successful
        """
        param_indices = param_indices or self._current_param_indices
        dressed_index = self.dressed_index(bare_tuple, param_indices)
        if dressed_index is None:
            return None
        return self["esys"][param_indices][0][dressed_index]

    @check_sync_status
    def energy_by_dressed_index(
        self, dressed_index: int, param_indices: Optional[Tuple[int, ...]] = None
    ) -> float:
        """
        Look up the dressed eigenenergy belonging to the given dressed index.

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
        return self["esys"][param_indices][0][dressed_index]

    @check_sync_status
    def bare_eigensys(
        self, subsys: "QuantumSystem", param_indices: Optional[Tuple[int, ...]] = None
    ) -> "Dict[QuantumSystem, ndarray]":
        """
        Return ndarray of bare eigenstates for given subsystem and parameter index.
        Eigenstates are expressed in the basis internal to the subsystem.
        """
        param_indices = param_indices or self._current_param_indices
        subsys_index = self._hilbertspace.get_subsys_index(subsys)
        return self["bare_esys"][subsys_index][param_indices]

    @check_sync_status
    def bare_eigenvals(
        self, subsys: "QuantumSystem", param_indices: Optional[Tuple[int, ...]] = None
    ) -> ndarray:
        """
        Return list of bare eigenenergies for given subsystem.

        Parameters
        ----------
        subsys:
            Hilbert space subsystem for which bare eigendata is to be looked up
        param_indices:
            position indices of parameter values in question

        Returns
        -------
            bare eigenenergies for the specified subsystem and the external parameter
            fixed to the value indicated by
            its index
        """
        param_indices = param_indices or self._current_param_indices
        subsys_index = self._hilbertspace.get_subsys_index(subsys)
        return self["bare_esys"][subsys_index][param_indices][0]

    def bare_productstate(self, bare_index: Tuple[int, ...]) -> Qobj:
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
