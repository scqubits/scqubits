# spec_lookup.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import itertools

import numpy as np

import scqubits
from scqubits.utils.spectrum_utils import convert_esys_to_ndarray


def shared_lookup_bare_eigenstates(self, param_index, subsys):
    """
    Parameters
    ----------
    self: ParameterSweep
    param_index: int
        position index of parameter value in question
    subsys: QuantumSystem
        Hilbert space subsystem for which bare eigendata is to be looked up

    Returns
    -------
    ndarray
        bare eigenvectors for the specified subsystem and the external parameter fixed to the value indicated by
        its index
    """
    subsys_index = self.hilbertspace.index(subsys)
    if subsys in self.subsys_update_list:
        return self.bare_specdata_list[subsys_index].state_table[param_index]
    return self.bare_specdata_list[subsys_index].state_table


class SpectrumLookup:
    """
    Generate and store a lookup table facilitating a map between bare state labels and dressed-state indices. The class
    provides such a lookup table for either a HilbertSpace or a ParameterSweep object.
    `_dressed_indices` and `bare_labels` are lists of as many elements as there are parameter values. (For HilbertSpace
    objects these are single-element lists.) Each element is a list of dressed indices (int) or bare labels (tuples).


    Parameters
    ----------
    framework: HilbertSpace or ParameterSweep
    spectrumdata: SpectrumData
        must be provided when generating for a HilbertSpace object, as HilbertSpace does not store specdata internally
    """
    def __init__(self, framework, spectrumdata=None):
        if isinstance(framework, scqubits.ParameterSweep):
            self.sweep = framework
            self.hilbertspace = self.sweep.hilbertspace
            self.dressed_specdata = self.sweep.dressed_specdata
            self.bare_specdata_list = self.sweep.bare_specdata_list
        elif isinstance(framework, scqubits.HilbertSpace):
            self.sweep = None
            self.hilbertspace = framework
            self.dressed_specdata = spectrumdata
            self.bare_specdata_list = None
        else:
            raise TypeError

        self._canonical_bare_labels = self.generate_bare_labels()
        self._dressed_indices = self.generate_lookup()      # lists of as many elements as there are parameter values.
                                                            # For HilbertSpace objects this is a single-element list.

    def generate_bare_labels(self):
        dim_list = self.hilbertspace.subsystem_dims
        basis_label_ranges = [list(range(dim_list[subsys_index])) for subsys_index
                              in range(self.hilbertspace.subsystem_count)]
        basis_labels_list = list(itertools.product(*basis_label_ranges))   # generate list of bare basis states (tuples)
        return basis_labels_list

    def generate_lookup(self):
        param_indices = range(self.dressed_specdata.param_count())
        dressed_indices_list = []
        for index in param_indices:
            dressed_indices = self.generate_single_lookup(index)
            dressed_indices_list.append(dressed_indices)
        return dressed_indices_list

    def generate_single_lookup(self, param_index):
        """
        For a single parameter value with given index, create a list of the dressed-state indices corresponding to the
        canonical bare-state product states (whenever possible).

        Parameters
        ----------
        param_index: int

        Returns
        -------
        list of int
            dressed-state indices
        """
        overlap_matrix = convert_esys_to_ndarray(self.dressed_specdata.state_table[param_index])  # overlap amplitudes

        dressed_indices = []
        for bare_basis_index in range(self.hilbertspace.dimension):   # for given bare basis index, find dressed index
            max_position = (np.abs(overlap_matrix[:, bare_basis_index])).argmax()
            max_overlap = np.abs(overlap_matrix[max_position, bare_basis_index])
            if max_overlap < 0.5:     # overlap too low, make no assignment
                dressed_indices.append(None)
            else:
                dressed_indices.append(max_position)
        return dressed_indices

    def dressed_index(self, bare_labels, param_index=0):
        """
        Parameters
        ----------
        bare_labels: tuple of ints
            bare_labels = (index, index2, ...)
        param_index: int
            index of parameter value of interest

        Returns
        -------
        int
            dressed state index closest to the specified bare state
        """
        try:
            lookup_position = self._canonical_bare_labels.index(bare_labels)
        except ValueError:
            return None
        return self._dressed_indices[param_index][lookup_position]

    def bare_index(self, dressed_index, param_index=0):
        """
        For given dressed index, look up the corresponding bare index from the state_lookup_table.

        Parameters
        ----------
        dressed_index: int
        param_index: int

        Returns
        -------
        tuple(int)
        """
        try:
            lookup_position = self._dressed_indices[param_index].index(dressed_index)
        except ValueError:
            return None
        basis_labels = self._canonical_bare_labels[lookup_position]
        return basis_labels

    def dressed_eigenstates(self, param_index):
        """
        Parameters
        ----------
        param_index: int
            position index of parameter value in question

        Returns
        -------
        list of qutip.qobj eigenvectors
            dressed eigenvectors for the external parameter fixed to the value indicated by the provided index
        """
        return self.dressed_specdata.state_table[param_index]

    def dressed_eigenenergies(self, param_index):
        """
        Parameters
        ----------
        param_index: int
            position index of parameter value in question

        Returns
        -------
        ndarray
            dressed eigenenergies for the external parameter fixed to the value indicated by the provided index
        """
        return self.dressed_specdata.energy_table[param_index]

    def energy_bare_index(self, bare_tuples, param_index=0):
        """
        Look up dressed energy most closely corresponding to the given bare-state labels

        Parameters
        ----------
        bare_tuples: tuple(int)
            bare state indices
        param_index: int
            index specifying the position in the self.param_vals array

        Returns
        -------
        dressed energy: float
        """
        dressed_index = self.dressed_index(bare_tuples, param_index)
        if dressed_index is not None:
            return self.dressed_specdata.energy_table[param_index][dressed_index]
        return None

    def energy_dressed_index(self, dressed_index, param_index=0):
        """
        Look up the dressed eigenenergy belonging to the given dressed index.

        Parameters
        ----------
        dressed_index: int
        param_index: int
            relevant if used in the context of a ParameterSweep

        Returns
        -------
        dressed energy: float
        """
        return self.dressed_specdata.energy_table[param_index][dressed_index]

    def bare_eigenstates(self, param_index, subsys):
        return shared_lookup_bare_eigenstates(self.sweep, param_index, subsys)

    def bare_eigenenergies(self, param_index, subsys):
        """
        Parameters
        ----------
        param_index: int
            position index of parameter value in question
        subsys: QuantumSystem
            Hilbert space subsystem for which bare eigendata is to be looked up

        Returns
        -------
        ndarray
            bare eigenenergies for the specified subsystem and the external parameter fixed to the value indicated by
            its index
        """
        subsys_index = self.hilbertspace.index(subsys)
        if subsys in self.sweep.subsys_update_list:
            return self.bare_specdata_list[subsys_index].energy_table[param_index]
        return self.bare_specdata_list[subsys_index].energy_table
