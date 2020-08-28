# hilbert_space.py
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
import warnings

import numpy as np
import qutip as qt

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.harmonic_osc as osc
import scqubits.core.spec_lookup as spec_lookup
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_qutip
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.ui.hspace_widget
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class InteractionTerm(dispatch.DispatchClient, serializers.Serializable):
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert space, and constructing
    the Hamiltonian in qutip.Qobj format. The expected form of the interaction term is of two possible types:
    1. V = g A B, where A, B are Hermitean operators in two specified subsys_list,
    2. V = g A B + h.c., where A, B may be non-Hermitean
    
    Parameters
    ----------
    g_strength: float
        coefficient parametrizing the interaction strength
    hilbertspace: HilbertSpace
        specifies the Hilbert space components
    subsys1, subsys2: QuantumSystem
        the two subsys_list involved in the interaction
    op1, op2: str or ndarray
        names of operators in the two subsys_list
    add_hc: bool, optional (default=False)
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitean conjugate is added.
    """
    g_strength = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    subsys1 = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    subsys2 = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    op1 = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    op2 = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')

    def __init__(self, g_strength, subsys1, op1, subsys2, op2, add_hc=False, hilbertspace=None):
        if hilbertspace:
            warnings.warn("`hilbertspace` is no longer a parameter for initializing an InteractionTerm object.",
                          FutureWarning)
        self.g_strength = g_strength
        self.subsys1 = subsys1
        self.op1 = op1
        self.subsys2 = subsys2
        self.op2 = op2
        self.add_hc = add_hc

        self._init_params.remove('hilbertspace')

    def __repr__(self):
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f'(**{init_dict!r})'

    def __str__(self):
        output = type(self).__name__.upper() + '\n ———— PARAMETERS ————'
        for param_name in self._init_params:
            output += '\n' + str(param_name) + '\t: ' + str(getattr(self, param_name))
        return output + '\n'


class HilbertSpace(dispatch.DispatchClient, serializers.Serializable):
    """Class holding information about the full Hilbert space, usually composed of multiple subsys_list.
    The class provides methods to turn subsystem operators into operators acting on the full Hilbert space, and
    establishes the interface to qutip. Returned operators are of the `qutip.Qobj` type. The class also provides methods
    for obtaining eigenvalues, absorption and emission spectra as a function of an external parameter.
    """
    osc_subsys_list = descriptors.ReadOnlyProperty()
    qbt_subsys_list = descriptors.ReadOnlyProperty()
    lookup = descriptors.ReadOnlyProperty()
    interaction_list = descriptors.WatchedProperty('INTERACTIONLIST_UPDATE')

    def __init__(self, subsystem_list, interaction_list=None):

        # Make sure all the given subsystems have required parameters set up. 
        self._subsystems_check(subsystem_list)

        self._subsystems = tuple(subsystem_list)
        if interaction_list:
            self.interaction_list = tuple(interaction_list)
        else:
            self.interaction_list = []

        self._lookup = None
        self._osc_subsys_list = [(index, subsys) for (index, subsys) in enumerate(self)
                                 if isinstance(subsys, osc.Oscillator)]
        self._qbt_subsys_list = [(index, subsys) for (index, subsys) in enumerate(self)
                                 if not isinstance(subsys, osc.Oscillator)]

        dispatch.CENTRAL_DISPATCH.register('QUANTUMSYSTEM_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('INTERACTIONTERM_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('INTERACTIONLIST_UPDATE', self)

    @classmethod
    def create(cls):
        hilbertspace = cls([])
        scqubits.ui.hspace_widget.create_hilbertspace_widget(hilbertspace.__init__)
        return hilbertspace

    def __getitem__(self, index):
        return self._subsystems[index]

    def __repr__(self):
        init_dict = self.get_initdata()
        return type(self).__name__ + f'(**{init_dict!r})'

    def __str__(self):
        output = '====== HilbertSpace object ======\n'
        for subsystem in self:
            output += '\n' + str(subsystem) + '\n'
        if self.interaction_list:
            for interaction_term in self.interaction_list:
                output += '\n' + str(interaction_term) + '\n'
        return output
    
    def _subsystems_check(self, subsystems):
        """Check if all the subsystems have truncated_dim set, which 
        is required for HilbertSpace to work correctly.
        Raise an exception if not.

        Parameters
        ----------
        subsystems: list of QuantumSystems

        """
        bad_indices = [i for i, sub_sys in enumerate(subsystems) if sub_sys.truncated_dim is None]
        
        if bad_indices:
            msg = "Subsystems with indices '{}' do".format(", ".join([str(i) for i in bad_indices]))  \
                if len(bad_indices) > 1 else "Subsystem with index '{:d}' does".format(bad_indices[0])

            raise RuntimeError("""{} not have `truncated_dim` set,
            which is required for `HilbertSpace` to operate correctly. This parameter can be
            set at object creation time, e.g.

            tmon = scqubits.Transmon(EJ=30.02, EC=1.2, ng=0.3, ncut=31, truncated_dim=4)

            or after the fact via tmon.truncated_dim = 4.
            """.format(msg))

    def index(self, item):
        return self._subsystems.index(item)

    def get_initdata(self):
        """Returns dict appropriate for creating/initializing a new HilbertSpace object.

        Returns
        -------
        dict
        """
        return {'subsystem_list': self._subsystems, 'interaction_list': self.interaction_list}

    def receive(self, event, sender, **kwargs):
        if self.lookup is not None:
            if event == 'QUANTUMSYSTEM_UPDATE' and sender in self:
                self.broadcast('HILBERTSPACE_UPDATE')
                self._lookup._out_of_sync = True
            elif event == 'INTERACTIONTERM_UPDATE' and sender in self.interaction_list:
                self.broadcast('HILBERTSPACE_UPDATE')
                self._lookup._out_of_sync = True
            elif event == 'INTERACTIONLIST_UPDATE' and sender is self:
                self.broadcast('HILBERTSPACE_UPDATE')
                self._lookup._out_of_sync = True

    @property
    def subsystem_list(self):
        return self._subsystems

    @property
    def subsystem_dims(self):
        """Returns list of the Hilbert space dimensions of each subsystem

        Returns
        -------
        list of int"""
        return [subsystem.truncated_dim for subsystem in self]

    @property
    def dimension(self):
        """Returns total dimension of joint Hilbert space

        Returns
        -------
        int"""
        return np.prod(np.asarray(self.subsystem_dims))

    @property
    def subsystem_count(self):
        """Returns number of subsys_list composing the joint Hilbert space

        Returns
        -------
        int"""
        return len(self._subsystems)

    def generate_lookup(self):
        bare_specdata_list = []
        for index, subsys in enumerate(self):
            evals, evecs = subsys.eigensys(evals_count=subsys.truncated_dim)
            bare_specdata_list.append(storage.SpectrumData(energy_table=[evals], state_table=[evecs],
                                                           system_params=subsys.get_initdata()))

        evals, evecs = self.eigensys(evals_count=self.dimension)
        dressed_specdata = storage.SpectrumData(energy_table=[evals], state_table=[evecs],
                                                system_params=self.get_initdata())
        self._lookup = spec_lookup.SpectrumLookup(self, bare_specdata_list=bare_specdata_list,
                                                  dressed_specdata=dressed_specdata)

    def eigenvals(self, evals_count=6):
        """Calculates eigenvalues of the full Hamiltonian using `qutip.Qob.eigenenergies()`.

        Parameters
        ----------
        evals_count: int, optional
            number of desired eigenvalues/eigenstates

        Returns
        -------
        eigenvalues: ndarray of float
        """
        hamiltonian_mat = self.hamiltonian()
        return hamiltonian_mat.eigenenergies(eigvals=evals_count)

    def eigensys(self, evals_count):
        """Calculates eigenvalues and eigenvectore of the full Hamiltonian using `qutip.Qob.eigenstates()`.

        Parameters
        ----------
        evals_count: int, optional
            number of desired eigenvalues/eigenstates

        Returns
        -------
        evals: ndarray of float
        evecs: ndarray of Qobj kets
        """
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = hamiltonian_mat.eigenstates(eigvals=evals_count)
        evecs = evecs.view(scqubits.io_utils.fileio_qutip.QutipEigenstates)
        return evals, evecs

    def diag_operator(self, diag_elements, subsystem):
        """For given diagonal elements of a diagonal operator in `subsystem`, return the `Qobj` operator for the
        full Hilbert space (perform wrapping in identities for other subsys_list).

        Parameters
        ----------
        diag_elements: ndarray of floats
            diagonal elements of subsystem diagonal operator
        subsystem: object derived from QuantumSystem
            subsystem where diagonal operator is defined

        Returns
        -------
        qutip.Qobj operator

        """
        dim = subsystem.truncated_dim
        index = range(dim)
        diag_matrix = np.zeros((dim, dim), dtype=np.float_)
        diag_matrix[index, index] = diag_elements
        return self.identity_wrap(diag_matrix, subsystem)

    def diag_hamiltonian(self, subsystem, evals=None):
        """Returns a `qutip.Qobj` which has the eigenenergies of the object `subsystem` on the diagonal.

        Parameters
        ----------
        subsystem: object derived from `QuantumSystem`
            Subsystem for which the Hamiltonian is to be provided.
        evals: ndarray, optional
            Eigenenergies can be provided as `evals`; otherwise, they are calculated.

        Returns
        -------
        qutip.Qobj operator
        """
        evals_count = subsystem.truncated_dim
        if evals is None:
            evals = subsystem.eigenvals(evals_count=evals_count)
        diag_qt_op = qt.Qobj(inpt=np.diagflat(evals[0:evals_count]))
        return self.identity_wrap(diag_qt_op, subsystem)

    def identity_wrap(self, operator, subsystem, op_in_eigenbasis=False, evecs=None):
        """Wrap given operator in subspace `subsystem` in identity operators to form full Hilbert-space operator.

        Parameters
        ----------
        operator: ndarray or qutip.Qobj or str
            operator acting in Hilbert space of `subsystem`; if str, then this should be an operator name in
            the subsystem, typically not in eigenbasis
        subsystem: object derived from QuantumSystem
            subsystem where diagonal operator is defined
        op_in_eigenbasis: bool
            whether `operator` is given in the `subsystem` eigenbasis; otherwise, the internal QuantumSystem basis is
            assumed
        evecs: ndarray, optional
            internal QuantumSystem eigenstates, used to convert `operator` into eigenbasis

        Returns
        -------
        qutip.Qobj operator
        """
        subsys_operator = spec_utils.convert_operator_to_qobj(operator, subsystem, op_in_eigenbasis, evecs)
        operator_identitywrap_list = [qt.operators.qeye(the_subsys.truncated_dim) for the_subsys in self]
        subsystem_index = self.get_subsys_index(subsystem)
        operator_identitywrap_list[subsystem_index] = subsys_operator
        return qt.tensor(operator_identitywrap_list)

    def hubbard_operator(self, j, k, subsystem):
        """Hubbard operator :math:`|j\\rangle\\langle k|` for system `subsystem`

        Parameters
        ----------
        j,k: int
            eigenstate indices for Hubbard operator
        subsystem: instance derived from QuantumSystem class
            subsystem in which Hubbard operator acts

        Returns
        -------
        qutip.Qobj operator
        """
        dim = subsystem.truncated_dim
        operator = (qt.states.basis(dim, j) * qt.states.basis(dim, k).dag())
        return self.identity_wrap(operator, subsystem)

    def annihilate(self, subsystem):
        """Annihilation operator a for `subsystem`

        Parameters
        ----------
        subsystem: object derived from QuantumSystem
            specifies subsystem in which annihilation operator acts

        Returns
        -------
        qutip.Qobj operator
        """
        dim = subsystem.truncated_dim
        operator = (qt.destroy(dim))
        return self.identity_wrap(operator, subsystem)

    def get_subsys_index(self, subsys):
        """
        Return the index of the given subsystem in the HilbertSpace.

        Parameters
        ----------
        subsys: QuantumSystem

        Returns
        -------
        int
        """
        return self.index(subsys)

    def bare_hamiltonian(self):
        """
        Returns
        -------
        qutip.Qobj operator
            composite Hamiltonian composed of bare Hamiltonians of subsys_list independent of the external parameter
        """
        bare_hamiltonian = 0
        for subsys in self:
            evals = subsys.eigenvals(evals_count=subsys.truncated_dim)
            bare_hamiltonian += self.diag_hamiltonian(subsys, evals)
        return bare_hamiltonian

    def get_bare_hamiltonian(self):
        """Deprecated, use `bare_hamiltonian()` instead."""
        warnings.warn('bare_hamiltonian() is deprecated, use bare_hamiltonian() instead', FutureWarning)
        return self.bare_hamiltonian()

    def hamiltonian(self):
        """

        Returns
        -------
        qutip.qobj
            Hamiltonian of the composite system, including the interaction between components
        """
        return self.bare_hamiltonian() + self.interaction_hamiltonian()

    def get_hamiltonian(self):
        """Deprecated, use `hamiltonian()` instead."""
        return self.hamiltonian()

    def interaction_hamiltonian(self):
        """
        Returns
        -------
        qutip.Qobj operator
            interaction Hamiltonian
        """
        if not self.interaction_list:
            return 0

        hamiltonian = [self.interactionterm_hamiltonian(term) for term in self.interaction_list]
        return sum(hamiltonian)

    def interactionterm_hamiltonian(self, interactionterm, evecs1=None, evecs2=None):
        interaction_op1 = self.identity_wrap(interactionterm.op1, interactionterm.subsys1, evecs=evecs1)
        interaction_op2 = self.identity_wrap(interactionterm.op2, interactionterm.subsys2, evecs=evecs2)
        hamiltonian = interactionterm.g_strength * interaction_op1 * interaction_op2
        if interactionterm.add_hc:
            return hamiltonian + hamiltonian.dag()
        return hamiltonian

    def _esys_for_paramval(self, paramval, update_hilbertspace, evals_count):
        update_hilbertspace(paramval)
        return self.eigensys(evals_count)

    def _evals_for_paramval(self, paramval, update_hilbertspace, evals_count):
        update_hilbertspace(paramval)
        return self.eigenvals(evals_count)

    def get_spectrum_vs_paramvals(self, param_vals, update_hilbertspace, evals_count=10, get_eigenstates=False,
                                  param_name="external_parameter", num_cpus=settings.NUM_CPUS):
        """Return eigenvalues (and optionally eigenstates) of the full Hamiltonian as a function of a parameter.
        Parameter values are specified as a list or array in `param_vals`. The Hamiltonian `hamiltonian_func`
        must be a function of that particular parameter, and is expected to internally set subsystem parameters.
        If a `filename` string is provided, then eigenvalue data is written to that file.

        Parameters
        ----------
        param_vals: ndarray of floats
            array of parameter values
        update_hilbertspace: function
            update_hilbertspace(param_val) specifies how a change in the external parameter affects
            the Hilbert space components
        evals_count: int, optional
            number of desired energy levels (default value = 10)
        get_eigenstates: bool, optional
            set to true if eigenstates should be returned as well (default value = False)
        param_name: str, optional
            name for the parameter that is varied in `param_vals` (default value = "external_parameter")
        num_cpus: int, optional
            number of cores to be used for computation (default value: settings.NUM_CPUS)

        Returns
        -------
        SpectrumData object
        """
        target_map = cpu_switch.get_map_method(num_cpus)
        if get_eigenstates:
            func = functools.partial(self._esys_for_paramval, update_hilbertspace=update_hilbertspace,
                                     evals_count=evals_count)
            with utils.InfoBar("Parallel computation of eigenvalues [num_cpus={}]".format(num_cpus), num_cpus):
                eigensystem_mapdata = list(target_map(func, tqdm(param_vals, desc='Spectral data', leave=False,
                                                                 disable=(num_cpus > 1))))
            eigenvalue_table, eigenstate_table = spec_utils.recast_esys_mapdata(eigensystem_mapdata)
        else:
            func = functools.partial(self._evals_for_paramval,  update_hilbertspace=update_hilbertspace,
                                     evals_count=evals_count)
            with utils.InfoBar("Parallel computation of eigensystems [num_cpus={}]".format(num_cpus), num_cpus):
                eigenvalue_table = list(target_map(func, tqdm(param_vals, desc='Spectral data', leave=False,
                                                              disable=(num_cpus > 1))))
            eigenvalue_table = np.asarray(eigenvalue_table)
            eigenstate_table = None

        return storage.SpectrumData(eigenvalue_table, self.get_initdata(), param_name, param_vals,
                                    state_table=eigenstate_table)
