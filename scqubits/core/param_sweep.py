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

from scqubits.core.data_containers import SpectrumData
from scqubits.core.harmonic_osc import Oscillator
from scqubits.settings import TQDM_KWARGS
from scqubits.utils.misc import make_bare_labels


class ParameterSweep:
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
        update_hilbertspace(param_val) specifies how a change in the external parameter affects
        the Hilbert space components
    interaction_list: list or iterable of InteractionTerm
        specifies the interaction Hamiltonian
    """
    def __init__(self, param_name, param_vals, evals_count, hilbertspace, subsys_update_list, update_hilbertspace,
                 interaction_list, generate_chi=False):
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self.hilbertspace = hilbertspace
        self.subsys_update_list = subsys_update_list
        self.update_hilbertspace = update_hilbertspace
        self.interaction_list = interaction_list

        self.bare_specdata_list = None
        self.dressed_specdata = None
        self._bare_hamiltonian_constant = None
        self.hilbertspace._state_lookup_table = None

        self.sweep_data = {}
        self._generate_chi = generate_chi

        # generate the spectral data sweep
        self.generate_parameter_sweep()

    def generate_parameter_sweep(self):
        self.bare_specdata_list = self._compute_bare_specdata_sweep()
        self.dressed_specdata = self._compute_dressed_specdata_sweep()

        self.hilbertspace.state_lookup_table = \
            [self.hilbertspace.generate_state_lookup_table(self.dressed_specdata, param_index)
             for param_index in range(self.param_count)]
        if self._generate_chi:
            self._generate_chi_sweep()

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
        for subsys in self.hilbertspace:
            self.update_hilbertspace(param_val)
            evals_count = subsys.truncated_dim

            if subsys in self.subsys_update_list:
                subsys_index = self.hilbertspace.index(subsys)
                eigendata.append(self.hilbertspace[subsys_index].eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)
        return eigendata

    def _compute_dressed_eigensystem(self, param_index):
        hamiltonian = self._bare_hamiltonian_constant + self._compute_bare_hamiltonian_varying(param_index)

        for interaction_term in self.interaction_list:
            evecs1 = self.lookup_bare_eigenstates(param_index, interaction_term.subsys1)
            evecs2 = self.lookup_bare_eigenstates(param_index, interaction_term.subsys2)
            hamiltonian += interaction_term.hamiltonian(evecs1=evecs1, evecs2=evecs2)

        return hamiltonian.eigenstates(eigvals=self.evals_count)

    def _generate_chi_sweep(self):
        osc_subsys_list = self.hilbertspace.osc_subsys_list
        qbt_subsys_list = self.hilbertspace.qbt_subsys_list

        for osc_index, osc_subsys in osc_subsys_list:
            for qbt_index, qubit_subsys in qbt_subsys_list:
                self.compute_custom_data_sweep('chi_osc{}_qbt{}'.format(osc_index, qbt_index), dispersive_chi_01,
                                               qubit_subsys=qubit_subsys, osc_subsys=osc_subsys)

    def compute_custom_data_sweep(self, data_name, func, **kwargs):
        data = [func(self, param_index, **kwargs) for param_index in range(self.param_count)]
        self.sweep_data[data_name] = np.asarray(data)

    def lookup_bare_eigenstates(self, param_index, subsys):
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

    def lookup_dressed_eigenstates(self, param_index):
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

    def lookup_bare_eigenenergies(self, param_index, subsys):
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
        if subsys in self.subsys_update_list:
            return self.bare_specdata_list[subsys_index].energy_table[param_index]
        return self.bare_specdata_list[subsys_index].energy_table

    def lookup_dressed_eigenenergies(self, param_index):
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

    def lookup_energy_bare_index(self, bare_tuples, param_index=0):
        dressed_index = self.hilbertspace.get_dressed_index(bare_tuples, param_index)
        if dressed_index is not None:
            return self.dressed_specdata.energy_table[param_index][dressed_index]
        return None

    def lookup_energy_dressed_index(self, dressed_index, param_index=0):
        return self.dressed_specdata.energy_table[param_index][dressed_index]

    @staticmethod
    def _process_diff_eigendata(labels_table, energies_table):
        """
        Helper routine to process difference spectrum energy data. The `energies_table` produced by
        `get_n_photon_qubit_spectrum` includes NaNs and has more columns than there are transitions. Recombine
        data, and order according to target labels.

        Parameters
        ----------
        labels_table: list
            python matrix of str, giving the transition target labels corresponding to the entries in the
            `energies_table`. NaN energies are marked by '' entries
        energies_table: list
            python matrix with transition energies and NaNs

        Returns
        -------
        list(str), ndarray
            Unique transition targets, cleaned up transition energy matrix
        """
        energies_table = np.asarray(energies_table)
        labels_table = np.asarray(labels_table)
        unique_labels = np.unique(labels_table)[1:]
        nrow, _ = energies_table.shape
        ncol = unique_labels.size
        diff_array = np.full(shape=(nrow, ncol), fill_value=np.NaN)
        for col_idx, label in enumerate(unique_labels):
            for row_idx, row in enumerate(labels_table):
                position = np.where(row == label)
                if position[0].size > 0:
                    diff_array[row_idx, col_idx] = energies_table[row_idx, position[0][0]]
        return unique_labels, diff_array

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

    def get_n_photon_qubit_spectrum(self, photonnumber, osc_subsys_list, initial_state_ind=0):
        """
        Extracts energies for transitions among qubit states only, while all oscillator subsystems maintain their
        excitation level.

        Parameters
        ----------
        photonnumber: int
            number of photons used in transition
        osc_subsys_list: list of (index, QuantumSystem)
            specifies the oscillator subsystem, assumed to be only one
        initial_state_ind: int or tuple (i1, i2, ...)
            index of the initial state whose energy is supposed to be subtracted from the spectral data

        Returns
        -------
        SpectrumData object
        """
        param_count = self.param_count
        osc_index_list = [index for (index, subsys) in osc_subsys_list]
        diff_eigenenergy_table = [[] for _ in range(param_count)]
        target_labels_table = [[] for _ in range(param_count)]

        eigenenergy_index = initial_state_ind
        for param_index in range(param_count):
            if not isinstance(initial_state_ind, int):
                eigenenergy_index = self.hilbertspace.get_dressed_index(initial_state_ind, param_index)

            initial_energy = self.dressed_specdata.energy_table[param_index][eigenenergy_index]
            initial_labels = self.hilbertspace.get_bare_index(eigenenergy_index, param_index)
            eigenenergies = self.dressed_specdata.energy_table[param_index]

            for index, _ in enumerate(eigenenergies):
                target_labels = self.hilbertspace.get_bare_index(index, param_index)
                if target_labels is not None and initial_labels is not None:
                    oscillator_change = sum(abs(target_labels[osc_index] - initial_labels[osc_index])
                                            for osc_index in osc_index_list)
                    energy_difference = (eigenenergies[index] - initial_energy) / photonnumber
                    if (oscillator_change == 0) and (energy_difference > 0):
                        diff_eigenenergy_table[param_index].append(energy_difference)
                        target_labels_table[param_index].append(str(target_labels))
                    else:
                        diff_eigenenergy_table[param_index].append(np.NaN)
                        target_labels_table[param_index].append('')
                else:
                    diff_eigenenergy_table[param_index].append(np.NaN)
                    target_labels_table[param_index].append('')

        target_label_list, diff_eigenenergy_table = self._process_diff_eigendata(target_labels_table,
                                                                                 diff_eigenenergy_table)
        return target_label_list, SpectrumData(self.param_name, self.param_vals,
                                               diff_eigenenergy_table, self.hilbertspace.__dict__)

# sweep_data generators --------------------------------------------------------------------------------


def dispersive_chis(sweep, param_index, qubit_subsys=None, osc_subsys=None):
    """
    For a given HilbertSpaceSweep, calculate dispersive shift data for one value of the external parameter.

    Parameters
    ----------
    sweep: ParameterSweep
    param_index: int
    qubit_subsys: QuantumSystem
    osc_subsys: Oscillator

    Returns
    -------
    ndarray
        dispersive shifts chi_0, chi_1, ...
    """
    qubitsys_index = sweep.hilbertspace.get_subsys_index(qubit_subsys)
    oscsys_index = sweep.hilbertspace.get_subsys_index(osc_subsys)
    qubit_dim = qubit_subsys.truncated_dim
    omega = osc_subsys.omega

    chi_values = np.empty(qubit_dim, dtype=np.float_)
    # chi_j = E_1j - E_0j - omega
    for j in range(qubit_dim):
        bare_0j = make_bare_labels(sweep.hilbertspace, qubitsys_index, j, oscsys_index, 0)
        bare_1j = make_bare_labels(sweep.hilbertspace, qubitsys_index, j, oscsys_index, 1)
        energy_0j = sweep.lookup_energy_bare_index(bare_0j, param_index)
        energy_1j = sweep.lookup_energy_bare_index(bare_1j, param_index)

        if energy_0j and energy_1j:
            chi_values[j] = energy_1j - energy_0j - omega
        else:
            chi_values[j] = np.NaN
    return chi_values


def dispersive_chi_01(sweep, param_index, qubit_subsys=None, osc_subsys=None):
    """
    For a given HilbertSpaceSweep, calculate the dispersive shift difference chi_01 for one value of the
    external parameter.

    Parameters
    ----------
    sweep: ParameterSweep
    param_index: int
    qubit_subsys: QuantumSystem
    osc_subsys: Oscillator

    Returns
    -------
    float
        dispersive shift chi_01
    """
    qubitsys_index = sweep.hilbertspace.get_subsys_index(qubit_subsys)
    oscsys_index = sweep.hilbertspace.get_subsys_index(osc_subsys)
    omega = osc_subsys.omega

    chi_values = np.empty(2, dtype=np.float_)
    # chi_j = E_1j - E_0j - omega
    for j in range(2):
        bare_0j = make_bare_labels(sweep.hilbertspace, qubitsys_index, j, oscsys_index, 0)
        bare_1j = make_bare_labels(sweep.hilbertspace, qubitsys_index, j, oscsys_index, 1)
        energy_0j = sweep.lookup_energy_bare_index(bare_0j, param_index)
        energy_1j = sweep.lookup_energy_bare_index(bare_1j, param_index)

        if energy_0j and energy_1j:
            chi_values[j] = energy_1j - energy_0j - omega
        else:
            chi_values[j] = np.NaN
    return chi_values[1] - chi_values[0]
