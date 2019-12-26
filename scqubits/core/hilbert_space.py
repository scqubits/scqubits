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

import h5py
import numpy as np
import qutip as qt
from tqdm import tqdm

from scqubits.settings import TQDM_KWARGS
from scqubits.core.data_containers import SpectrumData
from scqubits.utils.spectrum_utils import get_eigenstate_index_maxoverlap


class HilbertSpace(list):
    """Class holding information about the full Hilbert space, usually composed of multiple subsystems.
    The class provides methods to turn subsystem operators into operators acting on the full Hilbert space, and
    establishes the interface to qutip. Returned operators are of the `qutip.Qobj` type. The class also provides methods
    for obtaining eigenvalues, absorption and emission spectra as a function of an external parameter.
    """

    def __init__(self, subsystem_list):
        list.__init__(self, subsystem_list)

    def __repr__(self):
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

    def identity_wrap(self, operator, subsystem, evecs=None):
        """Wrap given operator in subspace `subsystem` in identity operators to form full Hilbert-space operator.

        Parameters
        ----------
        operator: ndarray or list or qutip.Qobj or str
            operator acting in Hilbert space of `subsystem`
        subsystem: object derived from QuantumSystem
            subsystem where diagonal operator is defined
        evecs: None or ndarray
            if operator is given in string form, it must be computed in the subsystem's eigenbasis; if given,
            evecs are used for that purpose

        Returns
        -------
        qutip.Qobj operator
        """
        if isinstance(operator, (list, np.ndarray)):
            dim = subsystem.truncated_dim
            subsys_operator = qt.Qobj(inpt=operator[:dim, :dim])
        elif isinstance(operator, qt.Qobj):
            subsys_operator = operator
        elif isinstance(operator, str):
            if evecs is None:
                _, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
            operator_matrixelements = subsystem.matrixelement_table(operator, evecs=evecs)
            subsys_operator = qt.Qobj(inpt=operator_matrixelements)
        else:
            raise TypeError

        operator_identitywrap_list = [qt.operators.qeye(the_subsys.truncated_dim) for the_subsys in self]
        subsystem_index = self.index(subsystem)
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
                eigenenergy_table[param_index] = hamiltonian.eigenenergies(eigvals=evals_count)

        spectrumdata = SpectrumData(param_name, param_vals, eigenenergy_table, self.__dict__,
                                    state_table=eigenstatesQobj_table)
        if filename:
            spectrumdata.filewrite(filename)

        return spectrumdata

    def difference_spectrum(self, spectrum_data, initial_state_ind, initial_as_bare=False):
        """Takes spectral data of energy eigenvalues and subtracts the energy of a select state, given by its state
        index.

        Parameters
        ----------
        spectrum_data: SpectrumData object
            spectral data composed of eigenenergies
        initial_state_ind: int or tuple ((subsys1, i1), (subsys2, i2), ...)
            index of the initial state whose energy is supposed to be subtracted from the spectral data
        initial_as_bare: bool, optional
            if `True`, then the index is a tuple labeling a bare eigenstate; if `False`, label refers to a state from
            `spectrum_data` (Default value = False)

        Returns
        -------
        SpectrumData object
        """
        paramvals_count = len(spectrum_data.param_vals)
        evals_count = len(spectrum_data.energy_table[0])
        diff_eigenenergy_table = np.empty((paramvals_count, evals_count))

        for param_index in tqdm(range(paramvals_count), **TQDM_KWARGS):
            eigenenergies = spectrum_data.energy_table[param_index]
            if initial_as_bare:
                basis_list = [None] * self.subsystem_count
                for (subsys, state_index) in initial_state_ind:
                    subsys_index = self.index(subsys)
                    basis_list[subsys_index] = qt.basis(subsys.truncated_dim, state_index)
                bare_state = qt.tensor(basis_list)
                eigenenergy_index = get_eigenstate_index_maxoverlap(spectrum_data.state_table[param_index],
                                                                    bare_state)
            else:
                eigenenergy_index = initial_state_ind

            diff_eigenenergies = eigenenergies - eigenenergies[eigenenergy_index]
            diff_eigenenergy_table[param_index] = diff_eigenenergies
        return SpectrumData(spectrum_data.param_name, spectrum_data.param_vals, diff_eigenenergy_table,
                            self.__dict__, state_table=None)

    def absorption_spectrum(self, spectrum_data, initial_state_ind, initial_as_bare=False):
        """Takes spectral data of energy eigenvalues and returns the absorption spectrum relative to a state
        of given index. Calculated by subtracting from eigenenergies the energy of the select state. Resulting negative
        frequencies, if the reference state is not the ground state, are omitted.

        Parameters
        ----------
        spectrum_data: SpectrumData object
            spectral data composed of eigenenergies
        initial_state_ind: int or tuple ((subsys1, i1), (subsys2, i2), ...)
            index of the initial state whose energy is supposed to be subtracted from the spectral data
        initial_as_bare: bool, optional
            if `True`, then the index is a tuple labeling a bare eigenstate; if `False`, label refers to a state from
            `spectrum_data` (Default value = False)

        Returns
        -------
        SpectrumData object
        """
        spectrum_data = self.difference_spectrum(spectrum_data, initial_state_ind, initial_as_bare)
        spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)
        return spectrum_data

    def emission_spectrum(self, spectrum_data, initial_state_ind, initial_as_bare=False):
        """Takes spectral data of energy eigenvalues and returns the emission spectrum relative to a state
        of given index. The resulting "upwards" transition frequencies are calculated by subtracting from eigenenergies
        the energy of the select state, and multiplying the result by -1. Resulting negative
        frequencies, corresponding to absorption instead, are omitted.

        Parameters
        ----------
        spectrum_data: SpectrumData object
            spectral data composed of eigenenergies
        initial_state_ind: int or tuple ((subsys1, i1), (subsys2, i2), ...)
            index of the initial state whose energy is supposed to be subtracted from the spectral data
        initial_as_bare: bool, optional
            if `True`, then the index is a tuple labeling a bare eigenstate; if `False`, label refers to a state from
            `spectrum_data` (Default value = False)

        Returns
        -------
        SpectrumData object
        """
        spectrum_data = self.difference_spectrum(spectrum_data, initial_state_ind, initial_as_bare)
        spectrum_data.energy_table *= -1.0
        spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)
        return spectrum_data

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

        for index, subsys in enumerate(self):
            h5file_subgroup = h5file_root.create_group('subsys_' + str(index))
            h5file_subgroup.attrs['type'] = type(subsys)
            subsys.filewrite_params_h5(h5file_subgroup)
