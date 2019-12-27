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

import copy

import h5py
import numpy as np
from tqdm.notebook import tqdm

from scqubits.core.data_containers import SpectrumData
from scqubits.settings import TQDM_KWARGS


class HilbertSpaceSweep:
    """
    This class allows interactive exploration of coupled quantum systems. The generate() method pre-calculates spectral
    data as a function of a given parameter, which can then be displayed and modified by sliders (when inside jupyter
    notebook or jupyter lab).

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
        update_hilbertspace(param_val, hilbertspace) specifies how a change in the external parameter affects
        the hilbert space components
    interaction_list: list or iterable of InteractionTerm
        specifies the interaction Hamiltonian
    """
    def __init__(self, param_name, param_vals, evals_count, hilbertspace, subsys_update_list, update_hilbertspace,
                 interaction_list):
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self.hilbertspace = hilbertspace
        self.subsys_update_list = subsys_update_list
        self.update_hilbertspace = update_hilbertspace
        self.interaction_list = interaction_list
        self.bare_specdata_list = self.generate_sweep_bare_specdata()
        self.dressed_specdata = self.generate_sweep_dressed_specdata()
        self.bare_hamiltonian_constant = self.get_bare_hamiltonian_constant()
        self.dressed_hamiltonian_constant = None
        self.hilbertspace.state_lookup_table = \
            [hilbertspace.generate_state_lookup_table(self.dressed_specdata, param_index)
             for param_index in range(self.param_count)]

    def bare_evecs_lookup(self, param_index, subsys):
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
            bare eigenvectors for the specified subsystem and the external parameter fixed to the value indicated by
            its index
        """
        subsys_index = self.hilbertspace.index(subsys)
        if subsys in self.subsys_update_list:
            return self.bare_specdata_list[subsys_index].state_table[param_index]
        return self.bare_specdata_list[subsys_index].state_table

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
        dressed_eigendata: list of qutip.eigenstates() tuples

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
                                system_params='')
        return specdata

    def get_bare_hamiltonian_constant(self):
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

    def get_bare_hamiltonian_varying(self, param_index):
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

    def get_dressed_eigensystem(self, param_index):
        hamiltonian = self.bare_hamiltonian_constant + self.get_bare_hamiltonian_varying(param_index)

        for interaction_term in self.interaction_list:
            evecs1 = self.bare_evecs_lookup(param_index, interaction_term.subsys1)
            evecs2 = self.bare_evecs_lookup(param_index, interaction_term.subsys2)
            hamiltonian += interaction_term.hamiltonian(evecs1=evecs1, evecs2=evecs2)
        return hamiltonian.eigenstates(eigvals=self.evals_count)

    def get_bare_spectrum_constant(self):
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

    def get_bare_spectrum_varying(self, param_val):
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
        new_hilbertspace = copy.deepcopy(self.hilbertspace)

        eigendata = []
        for subsys in self.hilbertspace:
            self.update_hilbertspace(param_val, new_hilbertspace)
            evals_count = subsys.truncated_dim

            if subsys in self.subsys_update_list:
                subsys_index = self.hilbertspace.index(subsys)
                eigendata.append(new_hilbertspace[subsys_index].eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)
        return eigendata

    def generate_sweep_bare_specdata(self):
        """
        Pre-calculates all bare spectral data needed for the interactive explorer display.
        """
        bare_eigendata_constant = self.get_bare_spectrum_constant()
        bare_eigendata_varying = [self.get_bare_spectrum_varying(param_val)
                                  for param_val in tqdm(self.param_vals, desc='Bare spectra', **TQDM_KWARGS)]
        bare_specdata_list = self._recast_bare_eigendata(bare_eigendata_constant, bare_eigendata_varying)
        del bare_eigendata_constant
        del bare_eigendata_varying
        return bare_specdata_list

    def generate_sweep_dressed_specdata(self):
        """
        Pre-calculates all dressed spectral data needed for the interactive explorer display.
        """
        self.bare_hamiltonian_constant = self.get_bare_hamiltonian_constant()
        dressed_eigendata = [self.get_dressed_eigensystem(j)
                             for j in tqdm(range(self.param_count), desc='Dressed spectrum', **TQDM_KWARGS)]
        dressed_specdata = self._recast_dressed_eigendata(dressed_eigendata)
        del(dressed_eigendata)
        return dressed_specdata

    def get_energy_bare_index(self, bare_tuples, param_index=0):
        dressed_index = self.hilbertspace.get_dressed_index(bare_tuples, param_index)
        return self.dressed_specdata.energy_table[param_index][dressed_index]

    def get_energy_dressed_index(self, dressed_index, param_index=0):
        return self.dressed_specdata.energy_table[param_index][dressed_index]

    def plot_bare_spectrum(self, subsys, which=-1, title=None, fig_ax=None):
        """
        Plots energy spectrum of bare system `subsys`

        Parameters
        ----------
        subsys: QuantumSystem
        which: None or int or list(int)
            default: -1, signals to plot all wavefunctions within the truncated Hilbert space;
            int>0: plot wavefunctions 0..int-1; list(int) plot specific wavefunctions
        title: str, optional
            plot title
        fig_ax: Figure, Axes

        Returns
        -------
        fig, axes
        """
        subsys_index = self.hilbertspace.index(subsys)
        specdata = self.bare_specdata_list[subsys_index]
        if which is None:
            which = subsys.truncated_dim
        return specdata.plot_evals_vs_paramvals(which=which, title=title, fig_ax=fig_ax)

    def plot_dressed_spectrum(self, title=None, fig_ax=None):
        """
        Plots energy spectrum of dressed system

        Returns
        -------
        fig, axes
        """
        ymax = np.max(self.dressed_specdata.energy_table) - np.min(self.dressed_specdata.energy_table)
        return self.dressed_specdata.plot_evals_vs_paramvals(subtract_ground=True, ymax=min(15, ymax),
                                                             title=title, fig_ax=fig_ax)

    def get_difference_spectrum(self, initial_state_ind=0):
        """Takes spectral data of energy eigenvalues and subtracts the energy of a select state, given by its state
        index.

        Parameters
        ----------
        initial_state_ind: int or (i1, i2, ...)
            index of the initial state whose energy is supposed to be subtracted from the spectral data

        Returns
        -------
        SpectrumData object
        """
        param_count = self.param_count
        evals_count = self.evals_count
        diff_eigenenergy_table = np.empty(shape=(param_count, evals_count))

        for param_index in tqdm(range(param_count), **TQDM_KWARGS):
            eigenenergies = self.dressed_specdata.energy_table[param_index]
            if isinstance(initial_state_ind, int):
                eigenenergy_index = initial_state_ind
            else:
                eigenenergy_index = self.hilbertspace.get_dressed_index(initial_state_ind, param_index)
            diff_eigenenergies = eigenenergies - eigenenergies[eigenenergy_index]
            diff_eigenenergy_table[param_index] = diff_eigenenergies
        return SpectrumData(self.param_name, self.param_vals, diff_eigenenergy_table, self.hilbertspace.__dict__)

    def plot_difference_spectrum(self, initial_state_ind=0):
        return self.get_difference_spectrum(initial_state_ind).plot_evals_vs_paramvals()

    def get_n_photon_qubit_spectrum(self, photonnumber, osc_subsys, initial_state_ind=0):
        """
        Assumes that there is only one oscillator subsystem `osc_subsys` whose photon number remains invariant under
        transitions.

        Parameters
        ----------
        photonnumber: int
            number of photons used in transition
        osc_subsys: QuantumSystem
            specifies the oscillator subsystem, assumed to be only one
        initial_state_ind: int or tuple (i1, i2, ...)
            index of the initial state whose energy is supposed to be subtracted from the spectral data

        Returns
        -------
        SpectrumData object
        """
        param_count = self.param_count
        osc_subsys_index = self.hilbertspace.index(osc_subsys)
        diff_eigenenergy_table = [[] for _ in range(param_count)]

        for param_index in range(param_count):
            if isinstance(initial_state_ind, int):
                eigenenergy_index = initial_state_ind
            else:
                eigenenergy_index = self.hilbertspace.get_dressed_index(initial_state_ind, param_index)
                if eigenenergy_index is None:
                    raise Exception("Cannot identify bare initial state with a dressed state!")
            initial_energy = self.dressed_specdata.energy_table[param_index][eigenenergy_index]
            initial_labels = self.hilbertspace.get_bare_index(eigenenergy_index, param_index)
            eigenenergies = self.dressed_specdata.energy_table[param_index]
            for index, _ in enumerate(eigenenergies):
                target_labels = self.hilbertspace.get_bare_index(index, param_index)
                if target_labels is not None:
                    photonnumber_diff = target_labels[osc_subsys_index] - initial_labels[osc_subsys_index]
                    energy_difference = (eigenenergies[index] - initial_energy) / photonnumber
                    if (photonnumber_diff == 0) and (energy_difference > 0):
                        diff_eigenenergy_table[param_index].append(energy_difference)
                    else:
                        diff_eigenenergy_table[param_index].append(np.NaN)
                else:
                    diff_eigenenergy_table[param_index].append(np.NaN)
        diff_eigenenergy_table = np.asarray(diff_eigenenergy_table)
        return SpectrumData(self.param_name, self.param_vals, diff_eigenenergy_table, self.hilbertspace.__dict__)

    def plot_n_photon_qubit_spectrum(self, photonnumber, osc_subsys, initial_state_ind=0, title=None, fig_ax=None):
        n_photon_specdata = self.get_n_photon_qubit_spectrum(photonnumber, osc_subsys, initial_state_ind)
        return n_photon_specdata.plot_evals_vs_paramvals(title=title, fig_ax=fig_ax, c='black')

    def filewrite_h5(self, file_hook):
        """Write spectrum data to h5 file

            Parameters
            ----------
            file_hook: str or h5py root group
                path for file to be openend, or h5py.Group handle to root in open h5 file
            """
        if isinstance(file_hook, str):
            h5file = h5py.File(file_hook + '.hdf5', 'w')
            h5file_root = h5file.create_group('root')
        else:
            h5file_root = file_hook

        for index, subsys in enumerate(self.hilbertspace):
            h5file_subgroup = h5file_root.create_group('subsys_' + str(index))
            h5file_subgroup.attrs['type'] = type(subsys)
            subsys.filewrite_params_h5(h5file_subgroup)
