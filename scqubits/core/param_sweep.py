# param_sweep.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np
from tqdm.notebook import tqdm

from scqubits.core.spec_lookup import SpectrumLookup, shared_lookup_bare_eigenstates
from scqubits.core.spectrum import SpectrumData
from scqubits.settings import TQDM_KWARGS


class ParameterSweep:
    """
    The ParameterSweep class helps generate spectral and associated data for a composite quantum system, as an externa,
    parameter, such as flux, is swept over some given interval of values. Upon initialization, these data are calculated
    and stored internally, so that plots can be generated efficiently. This is of particular use for interactive
    displays used in the Explorer class.

    Parameters
    ----------
    param_name: str
        name of external parameter to be varied
    param_vals: ndarray
        array of parameter values
    evals_count: int
        number of eigenvalues and eigenstates to be calculated for the composite Hilbert space
    hilbertspace: HilbertSpace
        collects all data specifying the Hilbert space of interest
    subsys_update_list: list or iterable
        list of subsystems in the Hilbert space which get modified when the external parameter changes
    update_hilbertspace: function
        update_hilbertspace(param_val) specifies how a change in the external parameter affects
        the Hilbert space components
    """
    def __init__(self, param_name, param_vals, evals_count, hilbertspace, subsys_update_list, update_hilbertspace):
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self.hilbertspace = hilbertspace
        self.subsys_update_list = subsys_update_list
        self.update_hilbertspace = update_hilbertspace

        self.lookup = None
        self.bare_specdata_list = None
        self.dressed_specdata = None
        self._bare_hamiltonian_constant = None

        self.sweep_data = {}

        # generate the spectral data sweep
        self.generate_parameter_sweep()

    def generate_parameter_sweep(self):
        """Top-level method for generating all parameter sweep data"""
        self.bare_specdata_list = self._compute_bare_specdata_sweep()
        self.dressed_specdata = self._compute_dressed_specdata_sweep()
        self.lookup = SpectrumLookup(self)

    def _compute_bare_specdata_sweep(self):
        """
        Pre-calculates all bare spectral data needed for the interactive explorer display.
        """
        bare_eigendata_constant = self._compute_bare_spectrum_constant()
        bare_eigendata_varying = [self._compute_bare_spectrum_varying(param_val)
                                  for param_val in tqdm(self.param_vals, desc='Bare spectra', **TQDM_KWARGS)]
        bare_specdata_list = self._recast_bare_eigendata(bare_eigendata_constant, bare_eigendata_varying)
        del bare_eigendata_constant
        del bare_eigendata_varying
        return bare_specdata_list

    def _compute_dressed_specdata_sweep(self):
        """
        Pre-calculates and all dressed spectral data.

        Returns
        -------
        list(SpectrumData)
        """
        self._bare_hamiltonian_constant = self._compute_bare_hamiltonian_constant()

        dressed_eigendata = [self._compute_dressed_eigensystem(j)
                             for j in tqdm(range(self.param_count), desc='Dressed spectrum', **TQDM_KWARGS)]
        dressed_specdata = self._recast_dressed_eigendata(dressed_eigendata)
        del dressed_eigendata
        return dressed_specdata

    def _recast_bare_eigendata(self, static_eigendata, bare_eigendata):
        """
        Parameters
        ----------
        static_eigendata: list of eigensystem tuples
        bare_eigendata: list of eigensystem tuples

        Returns
        -------
        list of SpectrumData
        """
        specdata_list = []
        for index, subsys in enumerate(self.hilbertspace):
            if subsys in self.subsys_update_list:
                evals_count = subsys.truncated_dim
                dim = subsys.hilbertdim()

                esys_dtype = subsys._evec_dtype
                energy_table = np.empty(shape=(self.param_count, evals_count), dtype=np.float_)
                state_table = np.empty(shape=(self.param_count, dim, evals_count), dtype=esys_dtype)
                for j in range(self.param_count):
                    energy_table[j] = bare_eigendata[j][index][0]
                    state_table[j] = bare_eigendata[j][index][1]
                specdata_list.append(SpectrumData(self.param_name, self.param_vals,
                                                  energy_table, subsys.__dict__, state_table))
            else:
                specdata_list.append(SpectrumData(self.param_name, self.param_vals,
                                                  energy_table=static_eigendata[index][0],
                                                  system_params=subsys.__dict__,
                                                  state_table=static_eigendata[index][1]))
        return specdata_list

    def _recast_dressed_eigendata(self, dressed_eigendata):
        """
        Parameters
        ----------
        dressed_eigendata: list of tuple(evals, qutip evecs)

        Returns
        -------
        SpectrumData
        """
        evals_count = self.evals_count
        energy_table = np.empty(shape=(self.param_count, evals_count), dtype=np.float_)
        state_table = []  # for dressed states, entries are Qobj
        for j in range(self.param_count):
            energy_table[j] = dressed_eigendata[j][0]
            state_table.append(dressed_eigendata[j][1])
        specdata = SpectrumData(self.param_name, self.param_vals, energy_table, state_table=state_table,
                                system_params=self.hilbertspace._get_metadata_dict())
        return specdata

    def _compute_bare_hamiltonian_constant(self):
        """
        Returns
        -------
        qutip.Qobj operator
            composite Hamiltonian composed of bare Hamiltonians of subsystems independent of the external parameter
        """
        static_hamiltonian = 0
        for index, subsys in enumerate(self.hilbertspace):
            if subsys not in self.subsys_update_list:
                evals = self.bare_specdata_list[index].energy_table
                static_hamiltonian += self.hilbertspace.diag_hamiltonian(subsys, evals)
        return static_hamiltonian

    def _compute_bare_hamiltonian_varying(self, param_index):
        """
        Parameters
        ----------
        param_index: int
            position index of current value of the external parameter

        Returns
        -------
        qutip.Qobj operator
            composite Hamiltonian consisting of all bare Hamiltonians which depend on the external parameter
        """
        hamiltonian = 0
        for index, subsys in enumerate(self.hilbertspace):
            if subsys in self.subsys_update_list:
                evals = self.bare_specdata_list[index].energy_table[param_index]
                hamiltonian += self.hilbertspace.diag_hamiltonian(subsys, evals)
        return hamiltonian

    def _compute_bare_spectrum_constant(self):
        """
        Returns
        -------
        list of (ndarray, ndarray)
            eigensystem data for each subsystem that is not affected by a change of the external parameter
        """
        eigendata = []
        for subsys in self.hilbertspace:
            if subsys not in self.subsys_update_list:
                evals_count = subsys.truncated_dim
                eigendata.append(subsys.eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)
        return eigendata

    def _compute_bare_spectrum_varying(self, param_val):
        """
        For given external parameter value obtain the bare eigenspectra of each bare subsystem that is affected by
        changes in the external parameter. Formulated to be used with Pool.map()

        Parameters
        ----------
        param_val: float

        Returns
        -------
        list of tuples(ndarray, ndarray)
            (evals, evecs) bare eigendata for each subsystem that is parameter-dependent
        """
        eigendata = []
        self.update_hilbertspace(param_val)
        for subsys in self.hilbertspace:
            evals_count = subsys.truncated_dim

            if subsys in self.subsys_update_list:
                subsys_index = self.hilbertspace.index(subsys)
                eigendata.append(self.hilbertspace[subsys_index].eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)
        return eigendata

    def _compute_dressed_eigensystem(self, param_index):
        hamiltonian = self._bare_hamiltonian_constant + self._compute_bare_hamiltonian_varying(param_index)

        for interaction_term in self.hilbertspace.interaction_list:
            evecs1 = self.lookup_bare_eigenstates(param_index, interaction_term.subsys1)
            evecs2 = self.lookup_bare_eigenstates(param_index, interaction_term.subsys2)
            hamiltonian += interaction_term.hamiltonian(evecs1=evecs1, evecs2=evecs2)
        return hamiltonian.eigenstates(eigvals=self.evals_count)

    def compute_custom_data_sweep(self, data_name, func, **kwargs):
        """Method for computing custom data as a function of the external parameter, calculated via the function `func`.

        Parameters
        ----------
        data_name: str
        func: function
            signature: `func(parametersweep, param_value, **kwargs)`, specifies how to calculate the data
        **kwargs: optional
            other parameters to be included in func
        """
        data = [func(self, param_index, **kwargs) for param_index in range(self.param_count)]
        self.sweep_data[data_name] = np.asarray(data)

    lookup_bare_eigenstates = shared_lookup_bare_eigenstates
