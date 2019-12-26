# explorer.py
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
import numpy as np
from tqdm import tqdm

from scqubits.core.data_containers import SpectrumData
from scqubits.settings import TQDM_KWARGS


class InteractionTerm:
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert space, and constructing
    the Hamiltonian in qutip.Qobj format. The expected form of the interaction term is g A B, where g is the
    interaction strength, A an operator in subsystem 1 and B and operator in subsystem 2.

    Parameters
    ----------
    g_strength: float
        coefficient parametrizing the interaction strength
    hilbertspace: HilbertSpace
        specifies the Hilbert space components
    subsys1, subsys2: QuantumSystem
        the two subsystems involved in the interaction
    op1, op2: str or ndarray
        names of operators in the two subsystems
    evecs1, evecs2: ndarray
        bare eigenvectors allowing the calculation of op1, op2 in the two bare eigenbases
    """
    def __init__(self, g_strength, hilbertspace, subsys1, op1, subsys2, op2, evecs1=None, evecs2=None):
        self.g_strength = g_strength
        self.hilbertspace = hilbertspace
        self.subsys1 = subsys1
        self.op1 = op1
        self.subsys2 = subsys2
        self.op2 = op2
        self.evecs1 = evecs1
        self.evecs2 = evecs2

    def hamiltonian(self, evecs1=None, evecs2=None):
        """
        Parameters
        ----------
        evecs1, evecs2: None or ndarray
            subsystem eigenvectors used to calculated interaction Hamiltonian; calculated on the fly if not given

        Returns
        -------
        qutip.Qobj operator
            interaction Hamiltonian
        """
        return self.g_strength * self.hilbertspace.identity_wrap(self.op1, self.subsys1, evecs=evecs1)\
               * self.hilbertspace.identity_wrap(self.op2, self.subsys2, evecs=evecs2)


class Explorer:
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
    static_hamiltonian: qutip.Qobj, optional
        the part of the composite-system Hamiltonian that is parameter-independent
    bare_specdata_list: list of SpectrumData objects, optional
        stores all pre-calculated spectral data of the bare subsystems
    dressed_specdata: SpectrumData, optional
        stores all pre-calculated spectral data of the dressed (composite) system
    """
    def __init__(self, param_name, param_vals, evals_count, hilbertspace, subsys_update_list, update_hilbertspace,
                 interaction_list, static_hamiltonian=None, bare_specdata_list=None, dressed_specdata=None):
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self.hilbertspace = hilbertspace
        self.subsys_update_list = subsys_update_list
        self.update_hilbertspace = update_hilbertspace
        self.static_hamiltonian = static_hamiltonian
        self.interaction_list = interaction_list
        self.bare_specdata_list = bare_specdata_list
        self.dressed_specdata = dressed_specdata

    def _evecs_lookup(self, param_index, subsys):
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

    def _get_bare_spectrum_fixed(self):
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
        energy_table = []
        state_table = []
        for j in range(self.param_count):
            energy_table.append(dressed_eigendata[j][0])
            state_table.append(dressed_eigendata[j][1])
        specdata = SpectrumData(self.param_name, self.param_vals, energy_table, state_table=state_table,
                                system_params='')
        return specdata

    def get_hamiltonian_static(self):
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

    def get_hamiltonian_vary(self, param_index):
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

    def generate(self, pool):
        """
        Pre-calculates all spectral data needed for the interactive explorer display.

        Parameters
        ----------
        pool: pathos.multiprocessing.ProcessPool
            pool for parallelizattion, should be started outside and protected by `if __name__ ==  '__main__':` for
            Windows compatibility
        """
        def _get_dressed_spectrum(param_index):
            """
            Worker function for pool.map()

            Parameters
            ----------
            param_index: int
                position index of current parameter value

            Returns
            -------
            eigvals: ndarray
                Array of eigenvalues for operator.
            eigvecs: ndarray of qutip Qobj
                Array of ccorresponding eigenkets of the Hamiltonian.
            """
            # print(self.static_hamiltonian)
            hamiltonian = self.static_hamiltonian + self.get_hamiltonian_vary(param_index)
            # print(hamiltonian)
            for interaction_term in self.interaction_list:
                evecs1 = self._evecs_lookup(param_index, interaction_term.subsys1)
                evecs2 = self._evecs_lookup(param_index, interaction_term.subsys2)
                hamiltonian += interaction_term.hamiltonian(evecs1=evecs1, evecs2=evecs2)
                # print(hamiltonian)
            return hamiltonian.eigenstates(eigvals=self.evals_count)
        # ENDDEF

        def _get_bare_spectrum_vary(param_val):
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
        # ENDDEF

        # print("Compute spectral data for bare subsystems...", end='')
        static_eigendata = self._get_bare_spectrum_fixed()

        # bare_eigendata = pool.map(_get_bare_spectrum_vary, self.param_vals)
        # tqdm.set_postfix_str('|{elapsed_s}{postfix}')
        bare_eigendata = [_get_bare_spectrum_vary(param_val) for param_val in tqdm(self.param_vals,
                                                                                   desc='Bare spectra    ',
                                                                                   **TQDM_KWARGS)]
        self.bare_specdata_list = self._recast_bare_eigendata(static_eigendata, bare_eigendata)
        del(bare_eigendata)
        del(static_eigendata)
        # print('DONE')

        # print("Compute static Hamiltonian...", end='')
        self.static_hamiltonian = self.get_hamiltonian_static()
        # print('DONE')

        #print("Compute dressed spectral data...", end='')
        # kwargs = {'explorer': self}
        dressed_eigendata = [_get_dressed_spectrum(j) for j in tqdm(range(self.param_count),
                                                                    desc='Dressed spectrum', **TQDM_KWARGS)]
        # dressed_eigendata = pool.map(_get_dressed_spectrum, range(self.param_count))
        # print('DONE')
        # print("Recast dressed spectral data...", end='')
        self.dressed_specdata = self._recast_dressed_eigendata(dressed_eigendata)
        # print('DONE')

    def plot_bare_spectrum(self, subsys):
        """
        Plots energy spectrum of bare system `subsys`

        Parameters
        ----------
        subsys: QuantumSystem

        Returns
        -------
        fig, axes
        """
        subsys_index = self.hilbertspace.index(subsys)
        specdata = self.bare_specdata_list[subsys_index]
        return specdata.plot_evals_vs_paramvals()

    def plot_bare_wavefunction(self, param_val, subsys, which=-1, phi_count=None):
        """
        Plot bare wavefunctions for given parameter value and subsystem.

        Parameters
        ----------
        param_val: float
            value of the external parameter
        subsys: QuantumSystem
        which: int or list
        phi_count: None or int

        Returns
        -------
        fig, axes
        """
        if which == -1:
            which = range(subsys.truncated_dim)

        subsys_index = self.hilbertspace.index(subsys)
        new_hilbertspace = copy.deepcopy(self.hilbertspace)
        self.update_hilbertspace(param_val, new_hilbertspace)
        subsys = new_hilbertspace[subsys_index]

        param_index = np.searchsorted(self.param_vals, param_val)

        evals = self.bare_specdata_list[subsys_index].energy_table[param_index]
        evecs = self.bare_specdata_list[subsys_index].state_table[param_index]
        return subsys.plot_wavefunction(esys=(evals, evecs), which=-1, phi_count=phi_count)
