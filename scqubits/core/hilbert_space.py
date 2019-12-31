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

import itertools

import numpy as np
import qutip as qt

from scqubits.core.data_containers import SpectrumData
from scqubits.core.harmonic_osc import Oscillator
from scqubits.settings import in_ipython, TQDM_KWARGS
from scqubits.utils.spectrum_utils import get_matrixelement_table

if in_ipython:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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
        interaction_op1 = self.hilbertspace.identity_wrap(self.op1, self.subsys1, op_in_eigenbasis=False, evecs=evecs1)
        interaction_op2 = self.hilbertspace.identity_wrap(self.op2, self.subsys2, op_in_eigenbasis=False, evecs=evecs2)
        return self.g_strength * interaction_op1 * interaction_op2


class HilbertSpace(list):
    """Class holding information about the full Hilbert space, usually composed of multiple subsystems.
    The class provides methods to turn subsystem operators into operators acting on the full Hilbert space, and
    establishes the interface to qutip. Returned operators are of the `qutip.Qobj` type. The class also provides methods
    for obtaining eigenvalues, absorption and emission spectra as a function of an external parameter.
    """

    def __init__(self, subsystem_list, interaction_list=None):
        list.__init__(self, subsystem_list)
        self.interaction_list = interaction_list
        self.state_lookup_table = None
        self.osc_subsys_list = [(index, subsys) for (index, subsys) in enumerate(self)
                                if isinstance(subsys, Oscillator)]
        self.qbt_subsys_list = [(index, subsys) for (index, subsys) in enumerate(self)
                                if not isinstance(subsys, Oscillator)]

    def __str__(self):
        output = '====== HilbertSpace object ======\n'
        for subsystem in self:
            output += '\n' + str(subsystem) + '\n'
        return output

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
        """Returns number of subsystems composing the joint Hilbert space

        Returns
        -------
        int"""
        return len(self)

    def diag_operator(self, diag_elements, subsystem):
        """For given diagonal elements of a diagonal operator in `subsystem`, return the `Qobj` operator for the
        full Hilbert space (perform wrapping in identities for other subsystems).

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
            Eigenenergies can be provided as `evals`; otherwise, they are calculated. (Default value = None)

        Returns
        -------
        qutip.Qobj operator
        """
        evals_count = subsystem.truncated_dim
        if evals is None:
            evals = subsystem.eigenvals(evals_count=evals_count)
        diag_qt_op = qt.Qobj(inpt=np.diagflat(evals[0:evals_count]))
        return self.identity_wrap(diag_qt_op, subsystem)

    def identity_wrap(self, operator, subsystem, op_in_eigenbasis=None, evecs=None):
        """Wrap given operator in subspace `subsystem` in identity operators to form full Hilbert-space operator.

        Parameters
        ----------
        operator: ndarray or qutip.Qobj or str
            operator acting in Hilbert space of `subsystem`; if str, then this should be an operator name in
            the subsystem, typically not in eigenbasis
        subsystem: object derived from QuantumSystem
            subsystem where diagonal operator is defined
        op_in_eigenbasis: None or bool
            whether `operator` is given in the `subsystem` eigenbasis; otherwise, the internal QuantumSystem basis is
            assumed
        evecs: None or ndarray
            internal QuantumSystem eigenstates, used to convert `operator` into eigenbasis

        Returns
        -------
        qutip.Qobj operator
        """
        dim = subsystem.truncated_dim

        if isinstance(operator, np.ndarray):
            if op_in_eigenbasis is False:
                if evecs is None:
                    _, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
                operator_matrixelements = get_matrixelement_table(operator, evecs)
                subsys_operator = qt.Qobj(inpt=operator_matrixelements)
            else:
                subsys_operator = qt.Qobj(inpt=operator[:dim, :dim])
        elif isinstance(operator, str):
            if evecs is None:
                _, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
            operator_matrixelements = subsystem.matrixelement_table(operator, evecs=evecs)
            subsys_operator = qt.Qobj(inpt=operator_matrixelements)
        elif isinstance(operator, qt.Qobj):
            subsys_operator = operator
        else:
            raise TypeError('Unsupported operator type: ', type(operator))

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

    def get_bare_hamiltonian(self):
        """
        Returns
        -------
        qutip.Qobj operator
            composite Hamiltonian composed of bare Hamiltonians of subsystems independent of the external parameter
        """
        bare_hamiltonian = 0
        for index, subsys in enumerate(self):
            evals = subsys.eigenvals(evals_count=subsys.truncated_dim)
            bare_hamiltonian += self.diag_hamiltonian(subsys, evals)
        return bare_hamiltonian

    def get_hamiltonian(self):
        """

        Returns
        -------
        qutip.qobj
            Hamiltonian of the composite system, including the interaction between components
        """
        hamiltonian = self.get_bare_hamiltonian()
        for interaction_term in self.interaction_list:
            hamiltonian += interaction_term.hamiltonian()
        return hamiltonian

    def get_spectrum_vs_paramvals(self, hamiltonian_func, param_vals, evals_count=10, get_eigenstates=False,
                                  param_name="external_parameter", filename=None):
        """Return eigenvalues (and optionally eigenstates) of the full Hamiltonian as a function of a parameter.
        Parameter values are specified as a list or array in `param_vals`. The Hamiltonian `hamiltonian_func`
        must be a function of that particular parameter, and is expected to internally set subsystem parameters.
        If a `filename` string is provided, then eigenvalue data is written to that file.

        Parameters
        ----------
        hamiltonian_func: function of one parameter
            function returning the Hamiltonian in `qutip.Qobj` format
        param_vals: ndarray of floats
            array of parameter values
        evals_count: int, optional
            number of desired energy levels (Default value = 10)
        get_eigenstates: bool, optional
            set to true if eigenstates should be returned as well (Default value = False)
        param_name: str, optional
            name for the parameter that is varied in `param_vals` (Default value = "external_parameter")
        filename: str, optional
            write data to file if path/filename is provided (Default value = None)

        Returns
        -------
        SpectrumData object
        """
        paramvals_count = len(param_vals)

        eigenenergy_table = np.empty((paramvals_count, evals_count))
        if get_eigenstates:
            eigenstatesQobj_table = [0] * paramvals_count
        else:
            eigenstatesQobj_table = None

        for param_index, paramval in tqdm(enumerate(param_vals), total=len(param_vals), **TQDM_KWARGS):
            paramval = param_vals[param_index]
            hamiltonian = hamiltonian_func(paramval)

            if get_eigenstates:
                eigenenergies, eigenstates_Qobj = hamiltonian.eigenstates(eigvals=evals_count)
                eigenenergy_table[param_index] = eigenenergies
                eigenstatesQobj_table[param_index] = eigenstates_Qobj
            else:
                eigenenergies = hamiltonian.eigenstates(eigvals=evals_count)
                eigenenergy_table[param_index] = eigenenergies

        spectrumdata = SpectrumData(param_name, param_vals, eigenenergy_table, self.__dict__,
                                    state_table=eigenstatesQobj_table)
        if filename:
            spectrumdata.filewrite(filename)

        return spectrumdata

    def generate_state_lookup_table(self, spectrum_data, param_index=0):
        """
        Create a lookup table that associates each dressed-state index with the corresponding bare-state product state
        index (whenever possible). Usually to be saved as self.state_lookup_table.

        Parameters
        ----------
        spectrum_data: SpectrumData
        param_index: int
            indices > 0 become relevant when using ParameterSweep

        Returns
        -------
        list(int), list(tuple)
            dressed indices, corresponding bare indices
        """
        dims = self.subsystem_dims
        product_dim = np.prod(dims)
        unit_vecs = np.eye(product_dim)
        bare_basis = [qt.Qobj(dims=[dims, [1]*self.subsystem_count], inpt=unit_vecs[j]) for j in range(product_dim)]

        overlap_matrix = np.asarray([[eigenstate.overlap(bare_basis_state) for bare_basis_state in bare_basis]
                                    for eigenstate in spectrum_data.state_table[param_index]])

        dressed_indices = []
        for bare_basis_index in range(product_dim):
            max_overlap = np.max(np.abs(overlap_matrix[:, bare_basis_index]))
            if max_overlap < 0.5:
                dressed_indices.append(None)
            else:
                dressed_indices.append((np.abs(overlap_matrix[:, bare_basis_index])).argmax())
        basis_label_ranges = [list(range(dims[subsys_index])) for subsys_index in range(self.subsystem_count)]
        basis_labels_list = [elem for elem in itertools.product(*basis_label_ranges)]
        return [dressed_indices, basis_labels_list]

    def lookup_dressed_index(self, bare_labels, param_index=0):
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
            lookup_position = self.state_lookup_table[param_index][1].index(bare_labels)
        except ValueError:
            return None
        return self.state_lookup_table[param_index][0][lookup_position]

    def lookup_bare_index(self, dressed_index, param_index=0):
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
            lookup_position = self.state_lookup_table[param_index][0].index(dressed_index)
        except ValueError:
            return None
        basis_labels = self.state_lookup_table[param_index][1][lookup_position]
        return basis_labels

    def _get_metadata_dict(self):
        meta_dict = {}
        for index, subsystem in enumerate(self):
            subsys_meta = subsystem._get_metadata_dict()
            renamed_subsys_meta = {}
            for key in subsys_meta.keys():
                renamed_subsys_meta[type(subsystem).__name__ + str(index) + '_' + key] = subsys_meta[key]
            meta_dict.update(renamed_subsys_meta)
        return meta_dict
