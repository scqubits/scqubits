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


import functools
from abc import ABC, abstractmethod

import numpy as np

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.hilbert_space as hspace
import scqubits.core.spec_lookup as spec_lookup
import scqubits.core.storage as storage
import scqubits.io_utils.fileio as io
import scqubits.io_utils.fileio_qutip as qutip_serializer
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ParameterSweepBase(ABC):
    """
    The ParameterSweepBase class is an abstract base class for ParameterSweep and StoredSweep
    """
    param_name: str
    param_vals: np.ndarray
    param_count: int
    evals_count: int
    lookup: spec_lookup.SpectrumLookup
    _hilbertspace: hspace.HilbertSpace

    @abstractmethod
    def lookup(self):
        pass

    def get_subsys(self, index):
        return self._hilbertspace[index]

    def get_subsys_index(self, subsys):
        return self._hilbertspace.get_subsys_index(subsys)

    @property
    def osc_subsys_list(self):
        return self._hilbertspace.osc_subsys_list

    @property
    def qbt_subsys_list(self):
        return self._hilbertspace.qbt_subsys_list

    @property
    def subsystem_count(self):
        return self._hilbertspace.subsystem_count

    @property
    def bare_specdata_list(self):
        return self.lookup._bare_specdata_list

    @property
    def dressed_specdata(self):
        return self.lookup._dressed_specdata

    def _lookup_bare_eigenstates(self, param_index, subsys, bare_specdata_list):
        """
        Parameters
        ----------
        self: ParameterSweep or HilbertSpace
        param_index: int
            position index of parameter value in question
        subsys: QuantumSystem
            Hilbert space subsystem for which bare eigendata is to be looked up
        bare_specdata_list: list of SpectrumData
            may be provided during partial generation of the lookup

        Returns
        -------
        ndarray
            bare eigenvectors for the specified subsystem and the external parameter fixed to the value indicated by
            its index
        """
        subsys_index = self.get_subsys_index(subsys)
        return bare_specdata_list[subsys_index].state_table[param_index]

    @property
    def system_params(self):
        return self._hilbertspace.get_initdata()

    def new_datastore(self, **kwargs):
        """Return DataStore object with system/sweep information obtained from self."""
        return storage.DataStore(self.system_params, self.param_name, self.param_vals, **kwargs)


class ParameterSweep(ParameterSweepBase, dispatch.DispatchClient, serializers.Serializable):
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
        list of subsys_list in the Hilbert space which get modified when the external parameter changes
    update_hilbertspace: function
        update_hilbertspace(param_val) specifies how a change in the external parameter affects
        the Hilbert space components
    num_cpus: int, optional
        number of CPUS requested for computing the sweep (default value settings.NUM_CPUS)
    """
    param_name = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_vals = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    param_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    evals_count = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    subsys_update_list = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    update_hilbertspace = descriptors.WatchedProperty('PARAMETERSWEEP_UPDATE')
    lookup = descriptors.ReadOnlyProperty()

    def __init__(self, param_name, param_vals, evals_count, hilbertspace, subsys_update_list, update_hilbertspace,
                 num_cpus=settings.NUM_CPUS):
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self._hilbertspace = hilbertspace
        self.subsys_update_list = tuple(subsys_update_list)
        self.update_hilbertspace = update_hilbertspace
        self.num_cpus = num_cpus

        self.tqdm_disabled = settings.PROGRESSBAR_DISABLED or (num_cpus > 1)

        self._lookup = None
        self._bare_hamiltonian_constant = None

        # setup for file Serializable

        dispatch.CENTRAL_DISPATCH.register('PARAMETERSWEEP_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('HILBERTSPACE_UPDATE', self)

        # generate the spectral data sweep
        if settings.AUTORUN_SWEEP:
            self.run()

    def run(self):
        """Top-level method for generating all parameter sweep data"""
        self.cause_dispatch()   # generate one dispatch before temporarily disabling CENTRAL_DISPATCH
        settings.DISPATCH_ENABLED = False
        bare_specdata_list = self._compute_bare_specdata_sweep()
        dressed_specdata = self._compute_dressed_specdata_sweep(bare_specdata_list)
        self._lookup = spec_lookup.SpectrumLookup(self, dressed_specdata, bare_specdata_list)
        settings.DISPATCH_ENABLED = True

    def cause_dispatch(self):
        self.update_hilbertspace(self.param_vals[0])

    def receive(self, event, sender, **kwargs):
        """Hook to CENTRAL_DISPATCH. This method is accessed by the global CentralDispatch instance whenever an event
        occurs that ParameterSweep is registered for. In reaction to update events, the lookup table is marked as out
        of sync.

        Parameters
        ----------
        event: str
            type of event being received
        sender: object
            identity of sender announcing the event
        **kwargs
        """
        if self.lookup is not None:
            if event == 'HILBERTSPACE_UPDATE' and sender is self._hilbertspace:
                self._lookup._out_of_sync = True
                # print('Lookup table now out of sync')
            elif event == 'PARAMETERSWEEP_UPDATE' and sender is self:
                self._lookup._out_of_sync = True
                # print('Lookup table now out of sync')

    def _compute_bare_specdata_sweep(self):
        """
        Pre-calculates all bare spectral data needed for the interactive explorer display.
        """
        bare_eigendata_constant = [self._compute_bare_spectrum_constant()] * self.param_count
        target_map = cpu_switch.get_map_method(self.num_cpus)
        with utils.InfoBar("Parallel compute bare eigensys [num_cpus={}]".format(self.num_cpus), self.num_cpus):
            bare_eigendata_varying = list(
                target_map(self._compute_bare_spectrum_varying,
                           tqdm(self.param_vals, desc='Bare spectra', leave=False, disable=self.tqdm_disabled))
            )
        bare_specdata_list = self._recast_bare_eigendata(bare_eigendata_constant, bare_eigendata_varying)
        del bare_eigendata_constant
        del bare_eigendata_varying
        return bare_specdata_list

    def _compute_dressed_specdata_sweep(self, bare_specdata_list):
        """
        Calculates and returns all dressed spectral data.

        Returns
        -------
        SpectrumData
        """
        self._bare_hamiltonian_constant = self._compute_bare_hamiltonian_constant(bare_specdata_list)
        param_indices = range(self.param_count)
        func = functools.partial(self._compute_dressed_eigensystem, bare_specdata_list=bare_specdata_list)
        target_map = cpu_switch.get_map_method(self.num_cpus)

        with utils.InfoBar("Parallel compute dressed eigensys [num_cpus={}]".format(self.num_cpus), self.num_cpus):
            dressed_eigendata = list(target_map(func, tqdm(param_indices, desc='Dressed spectrum', leave=False,
                                                           disable=self.tqdm_disabled)))
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
        for index, subsys in enumerate(self._hilbertspace):
            if subsys in self.subsys_update_list:
                eigendata = bare_eigendata
            else:
                eigendata = static_eigendata
            evals_count = subsys.truncated_dim
            dim = subsys.hilbertdim()
            esys_dtype = subsys._evec_dtype

            energy_table = np.empty(shape=(self.param_count, evals_count), dtype=np.float_)
            state_table = np.empty(shape=(self.param_count, dim, evals_count), dtype=esys_dtype)
            for j in range(self.param_count):
                energy_table[j] = eigendata[j][index][0]
                state_table[j] = eigendata[j][index][1]
            specdata_list.append(storage.SpectrumData(energy_table, system_params={}, param_name=self.param_name,
                                                      param_vals=self.param_vals, state_table=state_table))
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
            energy_table[j] = np.real_if_close(dressed_eigendata[j][0])
            state_table.append(dressed_eigendata[j][1])
        specdata = storage.SpectrumData(energy_table, system_params={}, param_name=self.param_name,
                                        param_vals=self.param_vals, state_table=state_table)
        return specdata

    def _compute_bare_hamiltonian_constant(self, bare_specdata_list):
        """
        Returns
        -------
        qutip.Qobj operator
            composite Hamiltonian composed of bare Hamiltonians of subsys_list independent of the external parameter
        """
        static_hamiltonian = 0
        for index, subsys in enumerate(self._hilbertspace):
            if subsys not in self.subsys_update_list:
                evals = bare_specdata_list[index].energy_table[0]
                static_hamiltonian += self._hilbertspace.diag_hamiltonian(subsys, evals)
        return static_hamiltonian

    def _compute_bare_hamiltonian_varying(self, bare_specdata_list, param_index):
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
        for index, subsys in enumerate(self._hilbertspace):
            if subsys in self.subsys_update_list:
                evals = bare_specdata_list[index].energy_table[param_index]
                hamiltonian += self._hilbertspace.diag_hamiltonian(subsys, evals)
        return hamiltonian

    def _compute_bare_spectrum_constant(self):
        """
        Returns
        -------
        list of (ndarray, ndarray)
            eigensystem data for each subsystem that is not affected by a change of the external parameter
        """
        eigendata = []
        for subsys in self._hilbertspace:
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
        for subsys in self._hilbertspace:
            if subsys in self.subsys_update_list:
                evals_count = subsys.truncated_dim
                subsys_index = self._hilbertspace.index(subsys)
                eigendata.append(self._hilbertspace[subsys_index].eigensys(evals_count=evals_count))
            else:
                eigendata.append(None)
        return eigendata

    def _compute_dressed_eigensystem(self, param_index, bare_specdata_list):
        hamiltonian = (self._bare_hamiltonian_constant +
                       self._compute_bare_hamiltonian_varying(bare_specdata_list, param_index))

        for interaction_term in self._hilbertspace.interaction_list:
            evecs1 = self._lookup_bare_eigenstates(param_index, interaction_term.subsys1, bare_specdata_list)
            evecs2 = self._lookup_bare_eigenstates(param_index, interaction_term.subsys2, bare_specdata_list)
            hamiltonian += self._hilbertspace.interactionterm_hamiltonian(interaction_term,
                                                                          evecs1=evecs1, evecs2=evecs2)
        evals, evecs = hamiltonian.eigenstates(eigvals=self.evals_count)
        evecs = evecs.view(qutip_serializer.QutipEigenstates)
        return evals, evecs

    def _lookup_bare_eigenstates(self, param_index, subsys, bare_specdata_list):
        """
        Parameters
        ----------
        self: ParameterSweep or HilbertSpace
        param_index: int
            position index of parameter value in question
        subsys: QuantumSystem
            Hilbert space subsystem for which bare eigendata is to be looked up
        bare_specdata_list: list of SpectrumData
            may be provided during partial generation of the lookup

        Returns
        -------
        ndarray
            bare eigenvectors for the specified subsystem and the external parameter fixed to the value indicated by
            its index
        """
        subsys_index = self.get_subsys_index(subsys)
        return bare_specdata_list[subsys_index].state_table[param_index]

    @classmethod
    def deserialize(cls, iodata):
        """
        Take the given IOData and return an instance of the described class, initialized with the data stored in
        io_data.

        Parameters
        ----------
        iodata: IOData

        Returns
        -------
        StoredSweep
        """
        return cls(**iodata.as_kwargs())

    def serialize(self):
        """
        Convert the content of the current class instance into IOData format.

        Returns
        -------
        IOData
        """
        initdata = {'param_name': self.param_name,
                    'param_vals': self.param_vals,
                    'evals_count': self.evals_count,
                    'hilbertspace': self._hilbertspace,
                    'dressed_specdata': self._lookup._dressed_specdata,
                    'bare_specdata_list': self._lookup._bare_specdata_list}
        iodata = serializers.dict_serialize(initdata)
        iodata.typename = 'StoredSweep'
        return iodata

    def filewrite(self, filename):
        """Convenience method bound to the class. Simply accesses the `write` function.

        Parameters
        ----------
        filename: str
        """
        io.write(self, filename)


class StoredSweep(ParameterSweepBase, serializers.Serializable):
    def __init__(self, param_name, param_vals, evals_count, hilbertspace, dressed_specdata, bare_specdata_list):
        self.param_name = param_name
        self.param_vals = param_vals
        self.param_count = len(param_vals)
        self.evals_count = evals_count
        self._hilbertspace = hilbertspace
        self._lookup = spec_lookup.SpectrumLookup(hilbertspace, dressed_specdata, bare_specdata_list)

    @property
    def lookup(self):
        return self._lookup

    def get_hilbertspace(self):
        return self._hilbertspace

    def new_sweep(self, subsys_update_list, update_hilbertspace, num_cpus=settings.NUM_CPUS):
        return ParameterSweep(
            self.param_name,
            self.param_vals,
            self.evals_count,
            self._hilbertspace,
            subsys_update_list,
            update_hilbertspace,
            num_cpus
        )
