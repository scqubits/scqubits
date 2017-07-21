# sc_qubits.py
"""
The 'sc_qubits' module provides routines for the basic description of common superconducting qubits such as
the Cooper pair box/transmon, fluxonium etc. Each qubit is realized as a class, providing relevant
methods such as calculating eigenvalues and eigenvectors, or plotting the energy spectrum vs. a select
external parameter.
"""

from __future__ import division
from __future__ import print_function
from builtins import *

import cmath
import copy
import itertools
import math
import sys

import h5py
import matplotlib.backends.backend_pdf as mplpdf
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

import config as globals
import operators as op
import plotting as plot



# routine for displaying a progress bar
def update_progress_bar(progress_in_percent):
    """Displays simple, text-based progress bar. The bar length is given by 'progress_in_percent'.
    @param progress_in_percent: (float) self-explanatory
    @return (None) only display output
    """
    bar_max_length = 20   # Modify this to change the length of the progress bar
    status_string = ""

    progress_in_percent = float(progress_in_percent)
    if progress_in_percent < 0.0:
        progress_in_percent = 0.0
        status_string = "Halt...\r\n"
    if progress_in_percent >= 1.0:
        progress_in_percent = 1.0
        status_string = "Done.\r\n"
    bar_length = int(round(bar_max_length * progress_in_percent))
    progress_display_string = "\r[{0}] {1}% {2}".format("=" * bar_length + "." * (bar_max_length - bar_length),
                                                        round(progress_in_percent * 100), status_string)
    sys.stdout.write(progress_display_string)
    sys.stdout.flush()
    return None

def initialize_progress_bar():
    """Set up use of text-based progress bar.
    @return (None)
    """
    print("")
    update_progress_bar(0)
    return None

# ---Auxiliary routines  ---------------------------------------------------------


def order_eigensystem(evals, evecs):
    """
    Takes eigenvalues and corresponding eigenvectors and orders them according to the eigenvalues (from smallest
    to largest; real valued eigenvalues are assumed).
    @param evals: (array, real-valued) array of eigenvalues
    @param evecs: (array) array containing eigenvectors; evecs[:, 0] is the first eigenvector etc.
    @return None (evals and evecs are reordered in place!)
    """
    ordered_evals_indices = evals.argsort()  # eigsh does not guarantee consistent ordering within result?! http://stackoverflow.com/questions/22806398
    evals = evals[ordered_evals_indices]
    evecs = evecs[:, ordered_evals_indices]
    return None


def extract_phase(complex_array):
    intermediate_index = int(len(complex_array) / 3)    # intermediate position for extracting phase (dangerous in tail or midpoint)
    return cmath.phase(complex_array[-intermediate_index])


def filewrite_csvdata(filename, numpy_array):
    np.savetxt(filename + '.csv', numpy_array, delimiter=",")
    return None

def filewrite_h5data(filename, numpy_data_list, data_info_strings, param_info_dict):
    """Write given data (numpy_data_list) along with information for each set (data_info_strings)
    to the h5 data file with path and name (filename)+'.hdf5'. Additional information about
    the chosen parameter (param_info_dict) set is added as well.
    @param filename: (str) file path and name (not including suffix)
    @param numpy_data_list: ([np_array1, np_array2, ...]) data sets to be written
    @param data_info_strings: ([info_str1, info_str2, ...]) strings labeling/describing the individual datasets
    @param param_info_dict: (dict) record of the parameter info used to generate the data
    @return (None) only file output
    """
    h5file = h5py.File(filename + '.hdf5', 'w')
    h5group = h5file.create_group('root')
    for dataset_index, dataset in enumerate(numpy_data_list):
        h5dataset = h5group.create_dataset(np.string_('data_' + str(dataset_index)), data=dataset, compression="gzip")
        h5dataset.attrs['data_info_' + str(dataset_index)] = np.string_(data_info_strings[dataset_index])
    for key, info in param_info_dict.items():
        h5group.attrs[key] = np.string_(info)
    h5file.close()
    return None

# ---Matrix elements and operators (outside qutip) ---------------------------------------------------------

#TODO matrix element handling is confusing. some inside, some outside classes. how to know what basis is being used
#TODO by operators?

def matrix_element(state1, operator, state2):
    """Calculate the matrix element <state1|operator|state2>.
    @param state1, state2: (numpy arrays|qt.Qobj) state vectors/kets
    @param operator: (qt.Qobj|numpy array|numpy sparse object) representation of an operator
    @return (np.complex_) matrix element <state1|operator|state2>
    """
    if isinstance(operator, qt.Qobj):
        op_matrix = operator.data
    else:
        op_matrix = operator

    if isinstance(state1, qt.Qobj):
        vec1 = state1.data.toarray()
        vec2 = state2.data.toarray()
    else:
        vec1 = state1
        vec2 = state2

    if isinstance(op_matrix, np.ndarray):    # Is operator given in dense form?
        return (np.vdot(vec1, np.dot(operator, vec2)))  # Yes - use numpy's 'vdot' and 'dot'.
    else:
        return (np.vdot(vec1, op_matrix.dot(vec2)))      # No, operator is sparse. Must use its own 'dot' method.


def matrixelem_table(operator, state_table, real_valued=False):
    """Calculate a table of matrix elements based on
    operator: numpy array or sparse matrix object
    state_table:    list (or array) of numpy arrays representing the states |v0>, |v1>, ...

    Returns a numpy array corresponding to the matrix element table
    <v0|operator|v0>   <v0|operator|v1>   ...
    <v1|operator|v0>   <v1|operator|v1>   ...
          ...                 ...

    Note: state_list expected to be in scipy's eigsh transposed form
    """
    if isinstance(operator, qt.Qobj):
        state_list = state_table
    else:
        state_list = state_table.T

    if real_valued:
        the_dtype = np.float_
    else:
        the_dtype = np.complex_

    tablesize = len(state_list)
    mtable = np.empty(shape=[tablesize, tablesize], dtype=the_dtype)
    for n in range(tablesize):
        for m in range(n + 1):
            mtable[n, m] = matrix_element(state_list[n], operator, state_list[m])
            if real_valued:
                mtable[m, n] = mtable[n, m]
            else:
                mtable[m, n] = np.conj(mtable[n, m])
    return mtable


def harm_osc_wavefunction(n, x, losc):
    """For given quantum number n=0,1,2,... this returns the value of the harmonic oscillator
    harmonic oscillator wave function \\psi_n(x) = N H_n(x/losc) exp(-x^2/2losc), N being the
    proper normalization factor.
    @param n: (int) index of wave function, n=0 is ground state
    @param x: (float) coordinate where wave function is evaluated
    @param losc: (float) oscillator length, defined via <0|x^2|0> = losc^2/2
    @return (float) value of harmonic oscillator wave function
    """
    return ((2.0**n * sp.special.gamma(n+1.0) * losc)**(-0.5) * np.pi**(-0.25) *
            sp.special.eval_hermite(n, x / losc) * np.exp(-(x * x) / (2 * losc * losc)))


def closest_dressed_energy(bare_energy, dressed_energy_vals):
    index = (np.abs(dressed_energy_vals - bare_energy)).argmin()
    return dressed_energy_vals[index]


def get_eigenstate_index_maxoverlap(eigenstates_Qobj, reference_state_Qobj, return_overlap=False):
    """For given qutip eigenstates object, find index of the eigenstate that has largest
    overlap with the qutip ket reference_state_Qobj
    @param eigenstates_Qobj: (array of qutip.Qobj) as obtained from qutip .eigenstates()
    @param return_overlap: (bool) set to true if the value of largest overlap should be also returned
    @param reference_state_Qobj: (qutip.Qobj ket) specific reference state
    @return (int) index of eigenstates_Qobj state with largest overlap if return_overlap set to False, 
            otherwise, tuple((int), (float)) tuple with the index as well as the corresponding overlap value.
    """
    overlaps = np.asarray([eigenstates_Qobj[j].overlap(reference_state_Qobj) for j in range(len(eigenstates_Qobj))])
    index = (np.abs(overlaps)).argmax()
    if return_overlap:
        return (index, np.abs(overlaps[index]))
    else:
        return index


class HilbertSpace(list):
    """Class holding information about the full Hilbert space, as composed of multiple subsystems.
    - Provides methods to turn subsystem operators in operators acting on the full Hilbert space, and
      establishes the interface to qutip. Returned operators are of the qutip.Qobj type.
    - Provides methods for obtaining eigenvalues, absorption and emission spectra as a function of an
      external paramater."""
    def __init__(self, subsystem_list, subsys_names=None):
        list.__init__(self, subsystem_list)

        if subsys_names is not None:
            for index, subsysname in enumerate(subsys_names):
                setattr(self, subsysname, subsystem_list[index])

    def __repr__(self):
        output = '====== HILBERT SPACE OBJECT ======'
        for parameter_name in self.__dict__.keys():
            parameter_val = self.__dict__[parameter_name]
            output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val) + '\n'
        return output

    @property
    def subsystem_dims(self):
        return [subsystem.truncated_dim for subsystem in self]

    @property
    def dimension(self):
        return np.prod(np.asarray(self.subsystem_dims))

    @property
    def subsystem_count(self):
        return len(self)

    def dict_reformat(self):
        """Returns HilbertSpace.__dict__ in reformatted form (all strings); needed for .h5 output.
        @return (dict {str: str,...}) reformatted self.__dict__
        """
        dict_reformatted = copy.deepcopy(self.__dict__)
        for key, value in dict_reformatted.items():
            dict_reformatted[key] = str(value)
        return dict_reformatted

    def filewrite_parameters(self, filename):
        """Writes system parameters as obtained from .__repr__() to file specified by 'filename'. File name suffix
        is appended.
        @param filename: (str) path and name for parameter file to be created
        @return (None)
        """
        with open(filename + globals.PARAMETER_FILESUFFIX, 'w') as target_file:
            target_file.write(self.__repr__())
        return None

    def diag_operator(self, diag_elements, subsystem):
        """For given diagonal elements of a diagonal operator in 'subsystem', return the Qobj operator for the
        full Hilbert space (perform wrapping in identities for other subsystems).
        @param diag_elements: (array of floats) diagonal elements of subsystem diagonal operator
        @param subsystem: (object derived from GenericQSys) subsystem where diagonal operator is defined
        @return (Qobj operator) full Hilbert space operator
        """
        dim = subsystem.truncated_dim
        index = range(dim)
        diag_matrix = np.zeros((dim, dim), dtype = np.float_)
        diag_matrix[index, index] = diag_elements
        return self.identity_wrap(diag_matrix, subsystem)

    def diag_hamiltonian(self, subsystem, evals=None):
        """Returns a qt.Qobj which has the eigenenergies of the object 'subsystem' on the diagonal."""
        evals_count = subsystem.truncated_dim
        if evals is None:
            evals = subsystem.eigenvals(evals_count=evals_count)
        diag_qt_op = qt.Qobj(inpt=np.diagflat(evals[0:evals_count]))
        return self.identity_wrap(diag_qt_op, subsystem)

    def identity_wrap(self, operator, subsystem):
        """Wrap given operator in subspace 'subsystem' in identity operators to form full Hilbert space operator.
        @param operator: (array|list|qt.Qobj) operator acting in Hilbert space of `subsystem`
        @param subsystem: (object derived from GenericQSys) subsystem where diagonal operator is defined
        @return (Qobj operator) full Hilbert space operator
        """
        if type(operator) in [list, np.ndarray]:
            dim = subsystem.truncated_dim
            subsys_operator = qt.Qobj(inpt=operator[:dim, :dim])
        else:
            subsys_operator = operator
        operator_identitywrap_list = [qt.operators.qeye(sys.truncated_dim) for sys in self]
        subsystem_index = self.index(subsystem)
        operator_identitywrap_list[subsystem_index] = subsys_operator
        return qt.tensor(operator_identitywrap_list)

    def hubbard_operator(self, j, k, subsystem):
        """Hubbard operator |j><k| for system 'subsystem'
        @param j, k: (int) eigenstate indices for Hubbard operator
        @param subsystem: (instance derived from GenericQSys class) subsystem in which Hubbard operator acts
        @return (qutip.Qobj of operator type) Hubbard operator in full Hilbert space
        """
        dim = subsystem.truncated_dim
        operator = (qt.states.basis(dim, j) * qt.states.basis(dim, k).dag())
        return self.identity_wrap(operator, subsystem)

    def annihilate(self, subsystem):
        """Annihilation operator a for 'subsystem'
        @param subsystem: (instance of class derived from GenericQSys) specifies subsystem in which annihilation operator acts
        @return (qt.Qobj of operator type) annihilation operator for subsystem, full Hilbert space
        """
        dim = subsystem.truncated_dim
        operator = (qt.destroy(dim))
        return self.identity_wrap(operator, subsystem)

    # def matrix_element(self, qt_operator, qt_states):
    #     dim = len(qt_states)
    #     matelem_table = np.empty((dim, dim), dtype=np.complex_)
    #     for j1 in range(dim):
    #         for j2 in range(j1 + 1):
    #             matelem_table[j1][j2] = qt_operator.matrix_element(qt_states[j1].dag(), qt_states[j2])
    #             matelem_table[j2][j1] = np.conj(matelem_table[j1][j2])
    #     return matelem_table
    #
    # def matrixelement_table(self, qt_operator, qt_states):
    #     dim = len(qt_states)
    #     matelem_table = np.empty((dim, dim), dtype=np.complex_)
    #     for j1 in range(dim):
    #         for j2 in range(j1 + 1):
    #             matelem_table[j1][j2] = qt_operator.matrix_element(qt_states[j1].dag(), qt_states[j2])
    #             matelem_table[j2][j1] = np.conj(matelem_table[j1][j2])
    #    return matelem_table

    def get_spectrum_vs_paramvals(self, hamiltonian_func, param_vals, evals_count=10, get_eigenstates=False,
                                  param_name="external_parameter", filename=None):
        """Return eigenvalues (and optionally eigenstates) of the full Hamiltonian as a function of a parameter.
        Parameter values are specified as a list or array in `param_vals`. The Hamiltonian `hamiltonian_func`
        must be a function of that particular parameter, and is expected to internally set subsystem parameters.
        If a `filename` string is provided, then eigenvalue data is written to that file.
        :param hamiltonian_func: function of one parameter, returning the hamiltonian in qt.Qobj format
        @param hamiltonian_func: (reference to function) The function hamiltonian_func takes one argument (an element
                                  of param_vals) and returns the Hamiltonian in qt.Qobj format
        @param param_vals: (array of floats) array of parameter values
        @param evals_count: (int) number of desired energy levels
        @param get_eigenstates: (bool) set to true if eigenstates should be returned as well
        @param param_name: (str) name for the parameter that is varied in `param_vals`
        @param filename: (None|str) write data to file if path/filename is provided
        @return (SpectrumData object) object containing parameter name, values, spectrum data, and system parameters
        """
        paramvals_count = len(param_vals)
        subsys_count = self.subsystem_count

        eigenenergy_table = np.empty((paramvals_count, evals_count))
        if get_eigenstates:
            eigenstatesQobj_table = [0] * paramvals_count
        else:
            eigenstatesQobj_table = None

        initialize_progress_bar()
        for param_index, paramval in enumerate(param_vals):
            hamiltonian = hamiltonian_func(paramval)

            if get_eigenstates:
                eigenenergies, eigenstates_Qobj = hamiltonian.eigenstates(eigvals=evals_count)
                eigenenergy_table[param_index] = eigenenergies
                eigenstatesQobj_table[param_index] = eigenstates_Qobj
            else:
                eigenenergy_table[param_index] = hamiltonian.eigenenergies(eigvals=evals_count)
            progress_in_percent = (param_index + 1) / paramvals_count
            update_progress_bar(progress_in_percent)

        if filename:
            if globals.FILE_FORMAT == 'csv':
                filewrite_csvdata(filename + '_' + 'param', param_vals)
                filewrite_csvdata(filename + '_specdata', eigenenergy_table)
                self.filewrite_parameters(filename)
            elif globals.FILE_FORMAT == 'h5':
                filewrite_h5data(filename, [param_vals, eigenenergy_table], ["external parameter", "eigenenergies"],
                                 self.dict_reformat())
        return SpectrumData(param_name, param_vals, eigenenergy_table, self.dict_reformat(),
                            state_table=eigenstatesQobj_table)

    def difference_spectrum(self, spectrum_data, initial_state_ind, initial_as_bare=False):
        paramvals_count = len(spectrum_data.param_vals)
        evals_count = len(spectrum_data.energy_table[0])
        diff_eigenenergy_table = np.empty((paramvals_count, evals_count))

        initialize_progress_bar()
        for param_index in range(paramvals_count):
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

            progress_in_percent = (param_index + 1) / paramvals_count
            update_progress_bar(progress_in_percent)
        return SpectrumData(spectrum_data.param_name, spectrum_data.param_vals, diff_eigenenergy_table,
                            self.dict_reformat(), state_table=None)

    def absorption_spectrum(self, spectrum_data, initial_state_ind, initial_as_bare=False):
        spectrum_data = self.difference_spectrum(spectrum_data, initial_state_ind, initial_as_bare)
        spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)
        return spectrum_data

    def emission_spectrum(self, spectrum_data, initial_state_ind, initial_as_bare=False):
        spectrum_data = self.difference_spectrum(spectrum_data, initial_state_ind, initial_as_bare)
        spectrum_data.energy_table *= -1.0
        spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)
        return spectrum_data


class WaveFunction(object):
    """Container for wave function amplitudes defined for a specific basis. Optionally, a corresponding
    energy is saved as well."""

    def __init__(self, basis_labels, amplitudes, energy=None):
        self.basis_labels = basis_labels
        self.amplitudes = amplitudes
        self.energy = energy


class Grid(object):
    """Data structure and methods for setting up discretized coordinate grid, generating corresponding derivative
    matrices."""

    def __init__(self, minmaxpts_array):
        self.min_vals = minmaxpts_array[:, 0]
        self.max_vals = minmaxpts_array[:, 1]
        self.var_count = len(self.min_vals)
        self.pt_counts = minmaxpts_array[:, 2].astype(np.int) #these are used as indices; need to be whole numbers.  

    def __repr__(self):
        output = '    Grid ......'
        for parameter_name in sorted(self.__dict__.keys()):
            parameter_val = self.__dict__[parameter_name]
            output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        return output

    def unwrap(self):
        """Auxiliary routine that yields a tuple of the parameters specifying the grid.
        @return (float array, float array, int array, int) tuple of grid parameters
        """
        return self.min_vals, self.max_vals, self.pt_counts, self.var_count

    def first_derivative_matrix(self, drvtv_var_index, prefactor=1.0, periodic=None):
        if periodic is not None:
            periodic_var_indices = (drvtv_var_index,)
        else:
            periodic_var_indices = None
        return self.multi_first_derivatives_matrix([drvtv_var_index], prefactor, periodic_var_indices)

    def multi_first_derivatives_matrix(self, deriv_var_list, prefactor=1.0, periodic_var_indices=None):
        """Generate sparse derivative matrices of the form \\partial_{x_1} \\partial_{x_2} ...,
        i.e., a product of first order derivatives (with respect to different variables).
        Uses f'(x) ~= [f(x+h) - f(x-h)]/2h, delta=2h
        Note: deriv_var_list is expected to be ordered!
        @param deriv_var_list: (list of ints) ordered list of variable indices, w.r.t. which derivatives are taken
        @param prefactor: (float|complex) optional prefactor of the derivative matrix
        @param periodic_var_indices: (list of ints) ordered sublist of deriv_var_list specifying periodic variables
        @return (sp.sparse.dia_matrix) sparse first derivative matrix"""
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_

        min_vals, max_vals, pt_counts, var_count = self.unwrap()

        deriv_order = len(deriv_var_list)  # total order of derivative

        offdiag_elements = [0.0] * deriv_order
        drvtv_mat = [0.0] * deriv_order

        # Loop over the elements of var_list and generate the derivative matrices
        for d_var_index, d_var in enumerate(deriv_var_list):
            d_var_range = (max_vals[d_var] - min_vals[d_var])
            offdiag_elements[d_var_index] = pt_counts[d_var] / (2.0 * d_var_range)
            if d_var_index == 0:
                offdiag_elements[d_var_index] *= prefactor  # first variable has prefactor absorbed into 1/delta
            drvtv_mat[d_var_index] = sp.sparse.dia_matrix((pt_counts[d_var], pt_counts[d_var]), dtype=dtp)
            drvtv_mat[d_var_index].setdiag(offdiag_elements[d_var_index], k=1)  # occupy first off-diagonal to the right
            drvtv_mat[d_var_index].setdiag(-offdiag_elements[d_var_index], k=-1)  # and left

            if d_var in periodic_var_indices:
                drvtv_mat[d_var_index].setdiag(-offdiag_elements[d_var_index], k=pt_counts[d_var] - 1)
                drvtv_mat[d_var_index].setdiag(offdiag_elements[d_var_index], k=-pt_counts[d_var] + 1)

        # Procedure to generate full matrix as follows. Example: derivatives w.r.t. 2, 4, and 5
        # 0  1  d2   3  d4  d5  6  7  8
        # (a) Set current index to first derivative index (ex: 2)
        # (b) Fill in identities to the left (Kronecker products, ex: 1, 0)
        # (c) Fill in identities to the right up to next derivative index or end (ex: 3)
        # (d) Insert next derivative.
        # (e) Repeat (c) and (d) until all variables finished.

        full_mat = drvtv_mat[0]  # (a) First derivative
        for var_index in range(deriv_var_list[0] - 1, -1, -1):  # (b) fill in identities to left of first variable
            full_mat = sp.sparse.kron(sp.sparse.identity(pt_counts[var_index], format='dia'), full_mat)

        for d_var_index, d_var in enumerate(deriv_var_list[:-1]):  # loop over remaining derivatives up to very last one:
            for var_index in range(d_var + 1, deriv_var_list[d_var_index + 1]):  # (c) fill in identities to the right
                full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[var_index], format='dia'))
            full_mat = sp.sparse.kron(full_mat, drvtv_mat[d_var_index + 1])  # (d) Insert next derivative

        for var_index in range(deriv_var_list[-1] + 1, var_count):  # Fill in remaining identities to right
            full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[var_index], format='dia'))

        return full_mat


    def second_derivative_matrix(self, drvtv_var_index, prefactor=1.0, periodic=None):
        min_vals, max_vals, pt_counts, var_count = self.unwrap()

        offdiag_element_val = prefactor * ((max_vals[drvtv_var_index] - min_vals[drvtv_var_index]) /
                                           pt_counts[drvtv_var_index]) ** (-2)

        drvtv_mat = sp.sparse.dia_matrix((pt_counts[drvtv_var_index], pt_counts[drvtv_var_index]), dtype=np.float_)
        drvtv_mat.setdiag(-2.0 * offdiag_element_val, k=0)
        drvtv_mat.setdiag(offdiag_element_val, k=1)
        drvtv_mat.setdiag(offdiag_element_val, k=-1)

        if periodic:
            drvtv_mat.setdiag(offdiag_element_val, k=pt_counts[drvtv_var_index] - 1)
            drvtv_mat.setdiag(offdiag_element_val, k=-pt_counts[drvtv_var_index] + 1)

        full_mat = drvtv_mat
        # Now fill in identity matrices to the left of var_ind, with variable indices
        # smaller than var_ind. Note: range(3,0,-1) -> [3,2,1]
        for j in range(drvtv_var_index - 1, -1, -1):
            full_mat = sp.sparse.kron(sp.sparse.identity(pt_counts[j], format='dia'), full_mat)
        # Next, fill in identity matrices with larger variable indices to the right.
        for j in range(drvtv_var_index + 1, var_count):
            full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[j], format='dia'))

        return full_mat


class WaveFunctionOnGrid(object):
    """Container for wave function amplitudes defined on a coordinate grid. Optionally, a corresponding
    energy is saved as well."""

    def __init__(self, grid, amplitudes, energy=None):
        self.grid = grid
        self.amplitudes = amplitudes
        self.energy = energy


class SpectrumData(object):
    """Container holding energy and state data as a function of a particular parameter that is varied.
    Also stores all other system parameters used for generating the set, and provides method for writing
    data to file."""

    def __init__(self, param_name, param_vals, energy_table, system_params, state_table=None):
        self.param_name = param_name
        self.param_vals = param_vals
        self.energy_table = energy_table
        self.state_table = state_table
        self.system_params = system_params

    def plot(self, axes=None, ylim=None,xlim=None, title=None, *args, **kw):
        """TODO: double check if ylim, xlim can be passed to axes.plot() directly - maybe not!?
        If we're given an axes object, update it with a plot, and return it for continence. 
        Otherwise, just show the plot.
        """

        if axes is None:
            axes_provided=False
            fig=plt.figure()
            axes = fig.add_subplot(1, 1, 1) 
        else:
            axes_provided=True

        axes.plot(self.param_vals, self.energy_table, *args, **kw)
        axes.set_xlabel(self.param_name)
        if xlim:
            axes.set_xlim(*xlim)
        if ylim:
            axes.set_ylim(*ylim)
        if title:
            axes.set_title(title)

        if axes_provided:
            return axes
        else:
            fig.tight_layout()
            plt.show()
            return None


    def filewrite(self, filename, write_states=False):
        if globals.FILE_FORMAT == 'csv':
            filewrite_csvdata(filename + '_' + self.param_name, self.paramval_list)
            filewrite_csvdata(filename + '_energies', self.energy_table)
            if write_states:
                filewrite_csvdata(filename + '_states', self.state_table)
            with open(filename + globals.PARAMETER_FILESUFFIX, 'w') as target_file:
                target_file.write(self.system_params)
        elif globals.FILE_FORMAT == 'h5':
            if write_states:
                filewrite_h5data(filename, [self.param_vals, self.energy_table, self.state_table],
                                 [self.param_name, "spectrum energies", "states"],
                                 self.system_params)
            else:
                filewrite_h5data(filename, [self.param_vals, self.energy_table], [self.param_name, "spectrum energies"],
                                 self.system_params)


# ---Generic quantum system container and Qubit base class--------------------------------------------------------------


class GenericQSys(object):

    """Generic quantum system class, blank except for holding the truncation parameter 'dim'.
    Defines methods for checking initialization parameters according to the _EXPECTED_PARAMS_DICT
    and the _OPTIONAL_PARAMS_DICT.
    """
    _EXPECTED_PARAMS_DICT = {}
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def print_expected_params_message(self):
        print('Expected parameters are:')
        for k, v in self._EXPECTED_PARAMS_DICT.items():
            print("{:<5} {:<40} ".format(k, v))
        return None

    def are_parameters_valid(self, given_params_dict):
        """Checks whether the parameter dictionary provided (given_params_dict) match the
        expected parameter entries given in _EXPECTED_PARAMS_DICT for a certain type of qubit class.
        Returns True when the two match exactly (no missing, no superfluous arguments).
        :param given_params_dict:
        :returns: True if given and expected parameter entries match for initialization.
        :rtype: bool
        """
        for expected_key in self._EXPECTED_PARAMS_DICT:
            if expected_key not in given_params_dict:
                print('>>Error<<: one or multiple parameter(s) have not been assigned values.')
                self.print_expected_params_message()
                return False
        for given_key in given_params_dict:
            if given_key not in self._EXPECTED_PARAMS_DICT and given_key not in self._OPTIONAL_PARAMS_DICT:
                print('>>Error<<: one or multiple of the specified parameters is/are unknown.')
                self.print_expected_params_message()
                return False
        return True

    def __init__(self, **parameter_args):
        if not self.are_parameters_valid(parameter_args):
            raise UserWarning('Parameter mismatch')
        else:
            self._sys_type = 'GenericQSys - used mainly as interface to qutip (e.g., resonator subsystem)'
            self.init_parameters(**parameter_args)

    def init_parameters(self, **parameter_args):
        for parameter_name, parameter_val in parameter_args.items():
            setattr(self, parameter_name, parameter_val)
        return None

    def __repr__(self):
        output = self.__dict__['_sys_type'] + ' -- PARAMETERS -------'
        for parameter_name in self.__dict__.keys():
            if parameter_name[0] is not '_':
                parameter_val = self.__dict__[parameter_name]
                output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        return output


class Oscillator(GenericQSys):
    """General class for mode of an oscillator/resonator."""

    _EXPECTED_PARAMS_DICT = {'omega': 'oscillator frequency'}
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    _eigenvals_stored = None

    def __init__(self, **parameter_args):
        super(Oscillator, self).__init__(**parameter_args)
        if not self.are_parameters_valid(parameter_args):
            raise UserWarning('Parameter mismatch')
        else:
            self._sys_type = 'Oscillator'
            self.init_parameters(**parameter_args)

    def __repr__(self):
        output = self.__dict__['_sys_type'] + ' -- PARAMETERS -------'
        for parameter_name in self.__dict__.keys():
            if parameter_name[0] is not '_':
                parameter_val = self.__dict__[parameter_name]
                output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        return output

    def eigenvals(self, evals_count=6, from_stored=False):
        if from_stored:
            evals = self._eigenvals_stored
        else:
            evals = [self.omega * n for n in range(evals_count)]
            self._eigenvals_stored = evals
        return np.asarray(evals)


class BaseClass(GenericQSys):
    """Base class for superconducting qubit objects. Provide general mechanisms and routines for
    checking validity of initialization parameters, writing data to files, and plotting.
    """
    _EXPECTED_PARAMS_DICT = {}
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    _eigenvals_stored = None

    def __init__(self, **parameter_args):
        super(BaseClass, self).__init__(**parameter_args)
        if not self.are_parameters_valid(parameter_args):
            raise UserWarning('Parameter mismatch')
        else:
            self._sys_type = 'BaseClass - mainly used as class template'
            self.init_parameters(**parameter_args)

    def __repr__(self):
        output = self.__dict__['_sys_type'] + ' -- PARAMETERS -------'
        for parameter_name in sorted(self.__dict__.keys()):
            if parameter_name[0] is not '_':
                parameter_val = self.__dict__[parameter_name]
                output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        output += '\nHilbert space dimension\t: ' + str(self.hilbertdim())
        return output

    def dict_reformat(self):
        dict_reformatted = copy.deepcopy(self.__dict__)
        for key, value in dict_reformatted.items():
            dict_reformatted[key] = str(value)
        return dict_reformatted

    def filewrite_parameters(self, filename):
        with open(filename + globals.PARAMETER_FILESUFFIX, 'w') as target_file:
            target_file.write(self.__repr__())

    @staticmethod
    def filewrite_evals(filename, evals):
        filewrite_csvdata(filename + globals.EVALS_FILESUFFIX, evals)

    @staticmethod
    def filewrite_evecs(filename, evecs):
        filewrite_csvdata(filename + globals.EVECS_FILESUFFIX, evecs)

    def hilbertdim(self):
        """Must be implemented in child classes"""
        pass

    def hamiltonian(self):
        """Must be implemented in child classes"""
        pass

    def _evals_calc(self, evals_count):
        """Employs scipy.linalg.eigh to obtain eigenvalues of Hamiltonian matrix (Hermitian)
        @param evals_count: (int) desired number of eigenvalues
        @return (array of floats) eigenvalues
        """
        hamiltonian_mat = self.hamiltonian()
        return sp.linalg.eigh(hamiltonian_mat, eigvals_only=True, eigvals=(0, evals_count - 1))

    def _esys_calc(self, evals_count):
        """Employs scipy.linalg.eigh to obtain eigenvalues and eigenstates of Hamiltonian matrix (Hermitian)
        @param evals_count: (int) desired number of eigenvalues
        @return (array of floats, 2d array of complex) eigenvalues, eigenstates
        """
        hamiltonian_mat = self.hamiltonian()
        return sp.linalg.eigh(hamiltonian_mat, eigvals_only=False, eigvals=(0, evals_count - 1))

    def eigenvals(self, evals_count=6, from_stored=False, filename=None):
        """Calculates eigenvalues via _evals_calc(), returns numpy array of eigenvalues.
        @param evals_count: (int) number of desired eigenvalues/eigenstates
        @param filename: (None|str) path and filename without suffix, if file output desired
        @param from_stored: (bool) retrieve eigenvalues from storage (last calculation). NO INTERNAL CHECKING FOR PARAMETER CHANGES!
        @return (array) eigenvalues, ordered by increasing eigenvalues
        """
        if from_stored:
            evals = self._eigenvals_stored
        else:
            evals = np.sort(self._evals_calc(evals_count))
            self._eigenvals_stored = evals
        if filename:
            self.filewrite_evals(filename, evals)
            self.filewrite_parameters(filename)
        return evals

    def eigensys(self, evals_count=6, filename=None):
        """Calculates eigenvalues and corresponding eigenvectors via _esys_calc()). Returns
        two numpy arrays containing the eigenvalues and eigenvectors, respectively.
        evals_count:   number of desired eigenvalues (sorted from smallest to largest)
        filename: write data to file if path and filename are specified
        @param evals_count: (int) number of desired eigenvalues/eigenstates
        @param filename: (None|str) path and filename without suffix, if file output desired
        @return (array, array) eigenvalues, and eigenstate matrix (ordered by increasing eigenvalues)

        """
        evals, evecs = self._esys_calc(evals_count)
        order_eigensystem(evals, evecs)
        if filename:
            self.filewrite_evals(filename, evals)
            self.filewrite_evecs(filename, evecs)
            self.filewrite_parameters(filename)
        return evals, evecs

    def matrixelement_table(self, operator, esys=None, evals_count=6):
        """Returns table of matrix elements for 'operator' with respect to the eigenstates of the qubit.
        The operator is given as a string matching a class method returning an operator matrix.
        E.g., for an instance 'trm' of Transmon,  the matrix element table for the charge operator is given by
        `trm.op_matrixelement_table('n_operator')`.
        When 'esys' is set to None, the eigensystem is calculated on-the-fly.
        @param operator: (str) name of class method in string form, returning operator matrix in qubit-internal basis.
        @param esys: (None|(array,array)) eigensystem data; if set to `None`, eigensystem is calculated on-the-fly
        @param evals_count: (int) number of desired matrix elements, starting with ground state
        @return (array) matrix elements <j|operator|j'>
        """
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        operator_matrix = getattr(self, operator)()
        return matrixelem_table(operator_matrix, evecs)

    def plot_matrixelements(self, operator, esys=None, evals_count=6, mode='abs', xlabel='', ylabel='', zlabel=''):
        """Plots matrix elements for 'operator', given as a string referring to a class method
        that returns an operator matrix. E.g., for instance 'trm' of Transmon, the matrix element plot
        for the charge operator 'n' is obtained by 'trm.plot_matrixelements('n').
        When 'esys' is set to None, the eigensystem with 'evals_count' eigenvectors is calculated.
        @param operator: (str) name of class method in string form, returning operator matrix
        @param esys: (None|(array,array)) eigensystem data of evals, evecs; calculates eigensystem if set to None
        @param evals_count: (int) number of desired matrix elements, starting with ground state
        @param mode: (str) entry from MODE_FUNC_DICTIONARY, e.g., 'abs' for absolute value
        @param xlabel, ylabel, zlabel: (str) labels for the three plot axes
        @return (None) graphics output
        """
        matrixelem_array = self.matrixelement_table(operator, esys, evals_count)
        plot.matrixelements(matrixelem_array, mode, xlabel, ylabel, zlabel)
        return None

    def get_spectrum_vs_paramvals(self, parameter_name, paramval_list, evals_count=6, subtract_ground=False,
                                  get_eigenstates=False, filename=None):
        """Calculates eigenvalues for varying system parameter 'param', where the values for 'param' are elements of
        paramval_list. Returns a SpectrumData object with energy_data[n] containing eigenvalues calculated for
        parameter value paramval_list[n].
        @param parameter_name: (str) name of parameter to be varied
        @param paramval_list:  (array) parameter values to be plugged in for param
        @param evals_count: (int) number of desired eigenvalues (sorted from smallest to largest)
        @param subtract_ground: (bool)  if True, eigenvalues are returned relative to the ground state eigenvalue
        @param get_eigenstates: (bool) return eigenstates along with eigenvalues
        @param filename: (None|str) write data to file if path and filename are specified
        @return (SpectrumData object) object containing parameter name, parameter values, eigenenergies,
                                        eigenstates, if desired, and system parameters
        """
        previous_paramval = getattr(self, parameter_name)

        paramvals_count = len(paramval_list)
        eigenvalue_table = np.zeros((paramvals_count, evals_count), dtype=np.float_)

        if get_eigenstates:
            eigenstate_table = np.empty(shape=(paramvals_count, self.hilbertdim(), evals_count), dtype=np.float_)
        else:
            eigenstate_table = None
        
        initialize_progress_bar()
        for index, paramval in enumerate(paramval_list):
            setattr(self, parameter_name, paramval)

            if get_eigenstates:
                evals, evecs = self.eigensys(evals_count)
                eigenstate_table[index] = evecs
            else:
                evals = self.eigenvals(evals_count)

            eigenvalue_table[index] = evals

            if subtract_ground:
                eigenvalue_table[index] -= evals[0]
            progress_in_percent = (index + 1) / paramvals_count
            update_progress_bar(progress_in_percent)
        setattr(self, parameter_name, previous_paramval)

        spectrumdata = SpectrumData(parameter_name, paramval_list, eigenvalue_table, self.dict_reformat(),
                                    state_table=eigenstate_table)

        if filename:
            spectrumdata.filewrite(filename, write_states=get_eigenstates)

        return spectrumdata

    def plot_evals_vs_paramvals(self, parameter_name, paramval_list, evals_count=6,
                                yrange=False, subtract_ground=False, shift=0, filename=None):
        """Generates a simple plot of a set of eigenvalues as a function of parameter 'param'.
        The individual points correspond to the parameter values listed in paramval_list.

        param:           string, gives name of parameter to be varied
        paramval_list:     list of parameter values to be plugged in for param
        subtract_ground: if True, then eigenvalues are returned relative to the ground state eigenvalues
                         (useful if transition energies from ground state are the relevant quantity)
        evals_count:           number of desired eigenvalues (sorted from smallest to largest)
        yrange:          [ymin, ymax] -- custom y-range for the plot
        shift:           apply a shift of this size to all eigenvalues
        filename:         write graphics and parameter set to file if path and filename are specified
        """
        specdata = self.get_spectrum_vs_paramvals(parameter_name, paramval_list, evals_count, subtract_ground)

        x = paramval_list
        y = specdata.energy_table
        if yrange:
            plt.axis([np.amin(x), np.amax(x), yrange[0], yrange[1]])
        else:
            plt.axis([np.amin(x), np.amax(x), np.amin(y + shift), np.amax(y + shift)])
        plt.xlabel(parameter_name)
        plt.ylabel('energy')
        plt.plot(x, y + shift)
        if filename:
            out_file = mplpdf.PdfPages(filename + '.pdf')
            out_file.savefig()
            out_file.close()
            self.filewrite_parameters(filename)
        plt.show()
        return None


# ---Cooper pair box / transmon-----------------------------------------------------------


class Transmon(BaseClass):

    """Class for the Cooper pair box / transmon qubit. Hamiltonian is represented in dense form. Expected parameters:
        EJ:   Josephson energy
        EC:   charging energy
        ng:   offset charge
        ncut: charge basis cutoff, n = -ncut, ..., ncut'

    Initialize with, e.g.
    >>> qubit = Transmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EC': 'charging energy',
        'ng': 'offset charge',
        'ncut': 'charge basis cutoff, n = -ncut, ..., ncut'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(Transmon, self).__init__(**parameter_args)
        self._sys_type = 'Cooper pair box'

    def hamiltonian(self):
        dimension = 2*self.ncut + 1
        hamiltonian_mat = np.zeros((dimension, dimension), dtype=np.float_)
        for i in range(dimension):
            hamiltonian_mat[i][i] = 4.0 * self.EC * (i - self.ncut - self.ng)**2
        for i in range(dimension - 1):
            hamiltonian_mat[i][i + 1] = -self.EJ / 2.0
            hamiltonian_mat[i + 1][i] = -self.EJ / 2.0
        return hamiltonian_mat

    def hilbertdim(self):
        return (2*self.ncut + 1)

    def n_operator(self):
        """Charge operator in charge basis for the transmon or CPB qubit."""
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1)
        return np.diagflat(diag_elements)

    def plot_n_wavefunction(self, esys, mode, which=0, nrange=(-5, 6)):
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        modefunction = globals.MODE_FUNC_DICT[mode]
        n_wavefunc.amplitudes = modefunction(n_wavefunc.amplitudes)
        plot.wavefunction1d_discrete(n_wavefunc, nrange)
        return None

    def plot_phi_wavefunction(self, esys, which=0, phi_points=251, mode='abs_sqr'):
        modefunction = globals.MODE_FUNC_DICT[mode]
        if isinstance(which, int):
            index_tuple = (which,)
        else:
            index_tuple = which
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for wavefunc_index in index_tuple:
            phi_wavefunc = self.phasebasis_wavefunction(esys, which=wavefunc_index, phi_points=phi_points)
            phase = extract_phase(phi_wavefunc.amplitudes)
            phi_wavefunc.amplitudes *= cmath.exp(-1j * phase)
            if np.sum(phi_wavefunc.amplitudes) < 0:
                phi_wavefunc.amplitudes *= -1.0

            phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)
            potential_vals = -self.EJ * np.cos(phi_wavefunc.basis_labels)
            plot.wavefunction1d(phi_wavefunc, potential_vals,
                                offset=phi_wavefunc.energy, scaling=0.3*self.EJ, xlabel='phi', axes = ax)
        return None

    def numberbasis_wavefunction(self, esys, which=0):
        """Return the transmon wave function in number basis. The specific index of the wave function is: 'which'.
        'esys' can be provided, but if set to 'None' then it is calculated frst.
        @param esys: (None | array, array) eigenvalue and eigenvector arrays as obtained from .eigensystem()
        @param which: (int) eigenfunction index
        @return (WaveFunction object) object containing charge values, corresponding wave function amplitudes, and eigenvalue
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, evecs = esys

        n_vals = np.arange(-self.ncut, self.ncut + 1)
        return WaveFunction(n_vals, evecs[:, which], evals[which])

    def phasebasis_wavefunction(self, esys, which=0, phi_points=251):
        """Return the transmon wave function in phase basis. The specific index of the wavefunction is: 'which'.
        'esys' can be provided, but if set to 'None' then it is calculated first.
        @param esys: (None | array, array) eigenvalue and eigenvector arrays as obtained from .eigensystem()
        @param which: (int) eigenfunction index
        @return (WaveFunction object) object containing phi values, corresponding wave function amplitudes, and eigenvalue
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, evecs = esys
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        phi_basis_labels = np.linspace(-np.pi, np.pi, phi_points)
        phi_wavefunc_amplitudes = np.empty(phi_points, dtype=np.complex_)
        for k in range(phi_points):
            phi_wavefunc_amplitudes[k] = ((1.0 / math.sqrt(2 * np.pi)) *
                                          np.sum(n_wavefunc.amplitudes * np.exp(1j * phi_basis_labels[k] * n_wavefunc.basis_labels)))
        return WaveFunction(phi_basis_labels, phi_wavefunc_amplitudes, energy=evals[which])


# ---Fluxonium qubit ------------------------------------------------------------------------


class Fluxonium(BaseClass):
    """Class for the fluxonium qubit. Hamiltonian is represented in dense form. The employed
    basis is the EC-EL harmonic oscillator basis. The cosine term in the potential is handled
    via matrix exponentiation.
    Expected parameters:
        EJ:   Josephson energy
        EC:   charging energy
        EL:   inductive energy
        flux: external magnetic flux (angular units, 2pi corresponds to one flux quantum)
        cutoff: number of harm. osc. basis states used in diagonalization

    Initialize with, e.g.
    >>> qubit = Fluxonium(EJ=1.0, EC=2.0, EL=0.3, flux=0.2, cutoff=120)
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EC': 'charging energy',
        'EL': 'inductive energy',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'cutoff': 'number of harm. osc. basis states used in diagonalization',
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(Fluxonium, self).__init__(**parameter_args)
        self._sys_type = 'fluxonium'

    def phi_osc(self):
        """Returns oscillator length for the LC oscillator by fluxonium inductance and capacitance.
        @return (float) value of oscillator length
        """
        return (8.0 * self.EC / self.EL)**(0.25)        # LC oscillator length

    def omega_p(self):
        return math.sqrt(8.0 * self.EL * self.EC)        # LC plasma oscillation frequency

    def phi_operator(self):
        """Returns the phi operator in the LC harmonic oscillator basis"""
        dimension = self.hilbertdim()
        return (op.creation(dimension) + op.annihilation(dimension)) * self.phi_osc() / math.sqrt(2)

    def n_operator(self):
        """Returns the n = - i d/dphi operator in the LC harmonic oscillator basis"""
        dimension = self.hilbertdim()
        return 1j * (op.creation(dimension) - op.annihilation(dimension)) / (self.phi_osc() * math.sqrt(2))

    def hamiltonian(self):           # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harm. osc. basis and return as sparse.dia_matrix"""
        dimension = self.hilbertdim()
        diag_elements = [i * self.omega_p() for i in range(dimension)]
        lc_osc_matrix = np.diagflat(diag_elements)

        exponent = 1j * self.phi_operator()
        exp_matrix = 0.5 * sp.linalg.expm(exponent) * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = exp_matrix + np.conj(exp_matrix.T)

        hamiltonian_mat = lc_osc_matrix - self.EJ*cos_matrix
        return np.real(hamiltonian_mat)     # use np.real to remove rounding errors from matrix exponential

    def hilbertdim(self):
        """
        @return (int) Hilbert space dimension
        """
        return self.cutoff

    def potential(self, phi):
        """
        Fluxonium potential evaluated at 'phi'.
        @param phi: (float) fluxonium phase variable
        @return (float) potential value at 'phi'
        """
        return 0.5 * self.EL * phi * phi - self.EJ * np.cos(phi + 2.0 * np.pi * self.flux)

    def wavefunction(self, esys, which=0, phi_range=(-6*np.pi, 6*np.pi), phi_points=251):
        evals_count = max(which + 1, 3)
        if esys is None:
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys

        dim = self.hilbertdim()
        phi_basis_labels = np.linspace(phi_range[0], phi_range[1], phi_points)
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros(phi_points, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * harm_osc_wavefunction(n, phi_basis_labels, phi_osc)
        return WaveFunction(phi_basis_labels, phi_wavefunc_amplitudes, energy=evals[which])

    def plot_wavefunction(self, esys, which=(0,), phi_range=(-6*np.pi, 6*np.pi), mode='abs_sqr', yrange=None, phi_points=251):
        """Different modes:
        'abs_sqr': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        modefunction = globals.MODE_FUNC_DICT[mode]
        if isinstance(which, int):
            index_tuple = (which,)
        else:
            index_tuple = which
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for wavefunc_index in index_tuple:
            phi_wavefunc = self.wavefunction(esys, wavefunc_index, phi_range, phi_points)
            phase = extract_phase(phi_wavefunc.amplitudes)
            phi_wavefunc.amplitudes *= cmath.exp(-1j * phase)
            if np.sum(phi_wavefunc.amplitudes) < 0:
                phi_wavefunc.amplitudes *= -1.0

            phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)
            plot.wavefunction1d(phi_wavefunc, self.potential(phi_wavefunc.basis_labels), offset=phi_wavefunc.energy,
                                scaling=5*self.EJ, xlabel='phi', yrange=yrange, axes = ax)
        return None


# ---Fluxonium qubit with SQUID loop----------------------------------------------------------------------


class FluxoniumSQUID(Fluxonium):

    """Class for the fluxonium qubit with two Josephson elements. Hamiltonian is represented in sparse form. The employed
    basis is the EC-EL harmonic oscillator basis. The cosine term in the potential is handled
    via matrix exponentiation.
    Expected parameters:
        EJ1:   Josephson energy 1
        EJ2:   Josephson energy 2
        EC:   charging energy
        EL:   inductive energy
        flux: external magnetic flux through primary loop in units of flux quanta (h/2e)
        fluxsquid: external magnetic flux through the SQUID loop in units of flux quanta (h/2e)
        cutoff: number of harm. osc. basis states used in diagonalization
    Initialize with, e.g.
    >>> qubit = FluxoniumSQUID(EJ1=1.0, EJ2=1.0, EC=2.0, EL=0.3, flux=0.2, fluxsquid=0.1, cutoff=120)
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ1': 'Josephson energy 1',
        'EJ2': 'Josephson energy 2',
        'EC': 'charging energy',
        'EL': 'inductive energy',
        'flux': 'external magnetic flux through primary loop in units of flux quanta (h/2e)',
        'fluxsquid': 'external magnetic flux through the SQUID loop in units of flux quanta (h/2e)',
        'cutoff': 'number of harm. osc. basis states used in diagonalization'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(FluxoniumSQUID, self).__init__(**parameter_args)
        self._sys_type = 'Fluxonium with small-junction SQUID loop'

    def hamiltonian(self):
        """Construct Hamiltonian matrix in harm. osc. basis and return as sparse.dia_matrix"""
        EJ1 = self.EJ1
        EJ2 = self.EJ2
        flux = self.flux
        fluxsquid = self.fluxsquid
        dim = self.hilbertdim()

        omega_p = self.omega_p()         # plasma osc. frequency
        d = (EJ1 - EJ2) / (EJ1 + EJ2)
        chi = math.atan(d * math.tan(-1 * np.pi * fluxsquid))        # just a term in the phase argument
        prefactor = math.cos(np.pi * fluxsquid) * math.sqrt(1.0 + (d * math.tan(np.pi * fluxsquid))**(2))  # just a prefactor in the transformed EJcos term

        diag_elements = [i * omega_p for i in range(dim)]
        lc_osc_matrix = np.diagflat(diag_elements)

        exponential = 1j * (op.creation(dim) + op.annihilation(dim)) * self.phi_osc() / math.sqrt(2)
        exp_matrix = 0.5 * sp.linalg.expm(exponential) * cmath.exp(1j * (2.0 * np.pi * flux - np.pi * fluxsquid + chi))
        cos_matrix = exp_matrix + np.conj(exp_matrix.T)

        hamiltonian_mat = lc_osc_matrix - (EJ1 + EJ2) * prefactor * cos_matrix
        return hamiltonian_mat

    def potential(self, phi):
        return (0.5 * self.EL * (phi)**(2) - self.EJ1 * np.cos(phi + 2 * np.pi * self.flux) -
                self.EJ2 * np.cos(phi - 2 * np.pi * self.fluxsquid + 2 * np.pi * self.flux))

    def param_sweep_plot(self, param1_name, paramval_list, param2_name, minimum, maximum, step, evals_count=6):
        """Plots evals against param1_name in range paramval_list.
        Plots for values of param2+name from minimum to maximum with separation step."""
        previous_param2val = getattr(self, param2_name)
        self.plot_evals_vs_paramvals(param1_name, paramval_list, evals_count)
        for i in range(int((maximum - minimum) / step)):
            setattr(self, param2_name, (minimum + step))
            minimum = minimum + step
            self.plot_evals_vs_paramvals(param1_name, paramval_list, evals_count)
        setattr(self, param2_name, previous_param2val)


# ---Symmetric 0-pi qubit--------------------------------------------------------------------


class SymZeroPi(BaseClass):

    """Symmetric Zero-Pi Qubit
    [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    The symmetric model, Eq. (8) in [2], assumes pair-wise identical circuit elements and describes the
    phi and theta degrees of freedom (chi decoupled). Formulation of the Hamiltonian matrix proceeds
    by discretization of the phi-theta space into a simple square/rectangular lattice.
    Expected parameters are:

    EJ:   Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'flux': 'external magnetic flux in angular units (2pi corresponds to one flux quantum)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(SymZeroPi, self).__init__(**parameter_args)
        self._sys_type = 'symmetric 0-Pi qubit (zero offset charge)'

    def hilbertdim(self):
        pt_counts = self.grid.pt_counts
        return int(np.prod(pt_counts))

    def potential(self, phi, theta):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) + self.EL * phi**2 + 2.0 * self.EJ)

    def sparse_kineticmat(self):
        kmat = self.grid.second_derivative_matrix(globals.PHI_INDEX, prefactor=-2.0*self.ECJ)    # -2E_{CJ}\\partial_\\phi^2
        kmat += self.grid.second_derivative_matrix(globals.THETA_INDEX, prefactor=-2.0*self.ECS, periodic=True)  # -2E_{C\\Sigma}\\partial_\\theta^2
        return kmat

    def sparse_potentialmat(self, potential=None):
        """Returns the potential energy matrix for the potential in sparse (dia_matrix) form.
        The potential energy can be passed in as a parameter. This may be useful, when we want to calculate
        sparse representations of (for example) derivatives of U - see d_hamiltonian_flux()
        """
        if potential is None:
            potential=self.potential

        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()
        hilbertspace_dim = int(np.prod(pt_counts))
        var_count = len(min_vals)
    
        # Xvals = [np.linspace(min_vals[j], max_vals[j], pt_counts[j]) for j in range(var_count)]  # list of coordinate arrays
        #We have to account fhe fact that \theta is periodic, hence the points at the end of the interval should be the same as at the beginning
        # Xvals = [np.linspace(min_vals[globals.PHI_INDEX], max_vals[globals.PHI_INDEX], pt_counts[globals.PHI_INDEX]), 
                # np.linspace(min_vals[globals.THETA_INDEX], max_vals[globals.THETA_INDEX] - 2.0*np.pi/pt_counts[globals.THETA_INDEX], pt_counts[globals.THETA_INDEX])]

        Xvals=[]
        for j in range(var_count):
            #We have to account fhe fact that \theta is periodic, hence the points at the end of the interval should be the same as at the beginning
            if j==globals.THETA_INDEX: 
                Xvals.append(np.linspace(min_vals[j], max_vals[j] - 2.0*np.pi/pt_counts[j], pt_counts[j]))
            else:
                Xvals.append(np.linspace(min_vals[j], max_vals[j], pt_counts[j]))

        diag_elements = np.empty([1, hilbertspace_dim], dtype=np.float_)

        for j, coord_tuple in enumerate(itertools.product(*Xvals)):
            # diag_elements[0][j] = self.potential(*coord_tuple)   # diagonal matrix elements
            diag_elements[0][j] = potential(*coord_tuple)   # diagonal matrix elements
        return sp.sparse.dia_matrix((diag_elements, [0]), shape=(hilbertspace_dim, hilbertspace_dim))

    def hamiltonian(self):
        return (self.sparse_kineticmat() + self.sparse_potentialmat())
    
    def d_potential_flux(self, phi, theta):
        """Returns a derivative of U w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial U}{ \partial \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0.
        """
        return  -(2.0 * np.pi * self.EJ * np.cos(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0)  )

    def d2_potential_flux(self, phi, theta):
        """Returns a derivative of U w.r.t flux, at the "current" value of flux, as stored in the object.  The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial U}{ \partial \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0.
        """
        return  (2.0 * np.pi**2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)  )

    def d_hamiltonian_flux(self):
        """Returns a derivative of the H w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial H}{ \partial \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0.
        """
        return self.sparse_potentialmat(potential=self.d_potential_flux)

    def d2_hamiltonian_flux(self):
        """Returns a second derivative of the H w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial^2 H}{ \partial^2 \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0^2.
        """
        return self.sparse_potentialmat(potential=self.d2_potential_flux)

    def d_potential_EJ(self, phi, theta):
        """Returns a derivative of the H w.r.t EJ. 
        This can be used for calculating critical current noise, which requires a derivative of d_H/d_I_c. 
        
        NOTE: We disregard the constant part of the potential energy ~ 2.0*self.EJ
        """
        return (-2.0 * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) ) 

    def d_hamiltonian_EJ(self):
        """Returns a derivative of the H w.r.t EJ.
        This can be used for calculating critical current noise, which requires a derivative of d_H/d_I_c. 
        Here, we differentiate w.r.t EJ in order not to impose units on H. 
        """
        return self.sparse_potentialmat(potential=self.d_potential_EJ)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return evals

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        return evals, evecs

    def i_d_dphi_operator(self):
        """Return the operator i \\partial_\\phi in sparse.dia_matrix form"""
        return self.grid.first_derivative_matrix(globals.PHI_INDEX, prefactor=1j, periodic=False)

    def d_dphi_operator(self):
        """Return the operator \\partial_\\phi in sparse.dia_matrix form
        """
        return self.grid.first_derivative_matrix(globals.PHI_INDEX, periodic=False)

    def i_d_dtheta_operator(self):
        """Return the operator i \\partial_\\theta (periodic variable) in sparse.dia_matrix form"""
        return self.grid.first_derivative_matrix(globals.THETA_INDEX, prefactor=1j, periodic=True)

    def d_dtheta_operator(self):
        """Return the operator \\partial_\\theta (periodic variable) in sparse.dia_matrix form
        """
        return self.grid.first_derivative_matrix(globals.THETA_INDEX, periodic=True)

    # return the operator \\phi
    def phi_operator(self):
        min_vals, max_vals, pt_counts, var_count = self.grid.unwrap()
        phi_matrix = sp.sparse.dia_matrix((pt_counts[globals.PHI_INDEX], pt_counts[globals.PHI_INDEX]), dtype=np.float_)
        diag_elements = np.linspace(min_vals[globals.PHI_INDEX], max_vals[globals.PHI_INDEX], pt_counts[globals.PHI_INDEX])
        phi_matrix.setdiag(diag_elements)
        for j in range(1, var_count):
            phi_matrix = sp.sparse.kron(phi_matrix, sp.sparse.identity(pt_counts[j], format='dia'))
        return phi_matrix

    def plot_potential(self, contour_vals=None, aspect_ratio=None, filename=None):
        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()
        x_vals = np.linspace(min_vals[globals.PHI_INDEX], max_vals[globals.PHI_INDEX], pt_counts[globals.PHI_INDEX])
        y_vals = np.linspace(min_vals[globals.THETA_INDEX], max_vals[globals.THETA_INDEX], pt_counts[globals.THETA_INDEX])
        plot.contours(x_vals, y_vals, self.potential, contour_vals, aspect_ratio, filename)
        return None

    def wavefunction(self, esys, which=0):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        pt_counts = self.grid.pt_counts
        wavefunc_amplitudes = evecs[:, which].reshape(pt_counts[globals.PHI_INDEX], pt_counts[globals.THETA_INDEX]).T
        return WaveFunctionOnGrid(self.grid, wavefunc_amplitudes)

    def plot_wavefunction(self, esys, which=0, mode='abs', figsize=(20, 10), aspect_ratio=3, zero_calibrate=False, axes=None):
        """Different modes:
        'abs_sqr': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        modefunction = globals.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, which)
        wavefunc.amplitudes = modefunction(wavefunc.amplitudes)
        return plot.wavefunction2d(wavefunc, figsize, aspect_ratio, zero_calibrate, axes=axes)

# ----------------------------------------------------------------------------------------

class SymZeroPiNg(SymZeroPi):

    """
    TODO: We should get rid of this class and add a charge offset directly to SymZeroPi.
    
    Symmetric Zero-Pi Qubit taking into account offset charge ng
    [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    The symmetric model, Eq. (8) in [2], assumes pair-wise identical circuit elements and describes the
    phi and theta degrees of freedom (chi decoupled). Including the offset charge leads to the substitution
    T = ... + CS \\dot{theta}^2  ==>    T = ... + CS (\\dot{theta} + ng)^2
    [This is not described in the two references above.]

    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta space into a simple
    square/rectangular lattice.
    Expected parameters are:

    EJ:   Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    ng:   offset charge
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'ng': 'offset charge',
        'flux': 'external magnetic flux units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(SymZeroPiNg, self).__init__(**parameter_args)
        self._sys_type = 'symmetric 0-Pi qubit with offset charge ng'

    def sparse_kineticmat(self):
        pt_counts = self.grid.pt_counts
        return (self.grid.second_derivative_matrix(globals.PHI_INDEX, prefactor=-2.0 * self.ECJ) +  # -2E_{CJ}\\partial_\\phi^2
                self.grid.second_derivative_matrix(globals.THETA_INDEX, prefactor=-2.0 * self.ECS, periodic=True) +   # 2E_{C\\Sigma}(i\\partial_\\theta + n_g)^2
                self.grid.first_derivative_matrix(globals.THETA_INDEX, prefactor=4.0 * 1j * self.ECS * self.ng, periodic=True) +
                sp.sparse.kron(sp.sparse.identity(pt_counts[globals.PHI_INDEX], format='dia'),
                               sp.sparse.identity(pt_counts[globals.THETA_INDEX], format='dia') * 2.0 * self.ECS * (self.ng)**2))

    # def sparse_kineticmat_org(self):
        # pt_counts = self.grid.pt_counts
        # return (self.grid.second_derivative_matrix(globals.PHI_INDEX, prefactor=-2.0 * self.ECJ) +  # -2E_{CJ}\\partial_\\phi^2
                # self.grid.second_derivative_matrix(globals.THETA_INDEX, prefactor=-2.0 * self.ECS, periodic=True) +   # 2E_{C\\Sigma}(i\\partial_\\theta + n_g)^2
                # self.grid.first_derivative_matrix(globals.THETA_INDEX, prefactor=4.0 * 1j * self.ECS * self.ng, periodic=True) +
                # sp.sparse.kron(sp.sparse.identity(pt_counts[globals.PHI_INDEX], format='dia'),
                               # sp.sparse.identity(pt_counts[globals.THETA_INDEX], format='dia') * 2.0 * self.ECS * (self.ng)**2))

# ----------------------------------------------------------------------------------------


class DisZeroPi(SymZeroPi):

    """Zero-Pi Qubit with disorder in EJ and EC. This disorder type still leaves chi decoupled,
    see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta space
    into a simple square/rectangular lattice.
    Expected parameters are:

    EJ:   mean Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    dEJ:  relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dCJ:  relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined
    as relative ones.

    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(SymZeroPi, self).__init__(**parameter_args)
        self._sys_type = '0-Pi qubit with EJ and CJ disorder, no coupling to chi mode (zero offset charge)'

    def potential(self, phi, theta):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) + self.EL * phi**2 + 2.0 * self.EJ +
                 self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def sparse_kineticmat(self):
        dphi2 = self.grid.second_derivative_matrix(globals.PHI_INDEX, prefactor=-2.0 * self.ECJ)                   # -2E_{CJ}\\partial_\\phi^2
        dth2 = self.grid.second_derivative_matrix(globals.THETA_INDEX, prefactor=-2.0 * self.ECS, periodic=True)     # -2E_{C\\Sigma}\\partial_\\theta^2
        dphidtheta = self.grid.multi_first_derivatives_matrix([globals.PHI_INDEX, globals.THETA_INDEX],
                                                              prefactor=2.0 * self.ECS * self.dCJ, periodic_var_indices=(globals.THETA_INDEX, ))
        return (dphi2 + dth2 + dphidtheta)


    def d_potential_flux(self, phi, theta):
        """Returns a derivative of U w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial U}{ \partial \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0.
        """
        return  - (2.0 * np.pi * self.EJ * np.cos(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))  \
                - (np.pi * self.EJ * self.dEJ * np.sin(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0))

    def d2_potential_flux(self, phi, theta):
        """Returns a second derivative of U w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial^2 U}{ \partial^2 \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0^2.
        """
        return  (2.0 * np.pi**2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0)  ) \
                - (np.pi**2.0 * self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def d_potential_EJ(self, phi, theta):
        """Returns a derivative of the H w.r.t EJ. 
        This can be used for calculating critical current noise, which requires a derivative of d_H/d_I_c. 
        
        NOTE: We disregard the constant part of the potential energy ~ 2.0*self.EJ
        """
        return (-2.0 * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) + 
                self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))



# ----------------------------------------------------------------------------------------

class DisZeroPiNg(DisZeroPi):

    """Zero-Pi Qubit with disorder in EJ and EC. This disorder type still leaves chi decoupled,
    see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta space
    into a simple square/rectangular lattice.
    Expected parameters are:

    EJ:   mean Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    dEJ:  relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dCJ:  relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    ng:   offset charge along theta
    grid: Grid object specifying the range and spacing of the discretization lattice

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined
    as relative ones.

    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'ng': 'offset charge along theta',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(DisZeroPiNg, self).__init__(**parameter_args)
        self._sys_type = '0-Pi qubit with EJ and CJ disorder, and offset charge, no coupling to chi mode'

    def sparse_kineticmat(self):
        dphi2 = self.grid.second_derivative_matrix(globals.PHI_INDEX, prefactor=-2.0 * self.ECJ)                   # -2E_{CJ}\\partial_\\phi^2
        dth2 = self.grid.second_derivative_matrix(globals.THETA_INDEX, prefactor=-2.0 * self.ECS, periodic=True)     # -2E_{C\\Sigma}\\partial_\\theta^2
        dphidtheta = self.grid.multi_first_derivatives_matrix([globals.PHI_INDEX, globals.THETA_INDEX],
                                                              prefactor=2.0 * self.ECS * self.dCJ, periodic_var_indices=(globals.THETA_INDEX, ))
        ngdtheta=0.0
        if self.ng !=0:
            # pt_counts = self.grid.pt_counts
            #2E_{C\\Sigma}( 2 i\\partial_\\theta + n_g^2 )
            ngdtheta = self.grid.first_derivative_matrix(globals.THETA_INDEX, prefactor=4.0 * 1j * self.ECS * self.ng, periodic=True) # + \
                       # sp.sparse.kron(sp.sparse.identity(pt_counts[globals.PHI_INDEX], format='dia'), 
                                      # sp.sparse.identity(pt_counts[globals.THETA_INDEX], format='dia') * 2.0 * self.ECS * (self.ng)**2) #needed?

        return (dphi2 + dth2 + dphidtheta + ngdtheta)

    def d_hamiltonian_ng(self):
        """Returns a derivative of the H w.r.t ng.
        This can be used for calculating charge noise.
        """
        # pt_counts = self.grid.pt_counts
        # \partial/\partial n_g  ( 2E_{C\\Sigma}(i\\partial_\\theta + n_g)^2 )
        return self.grid.first_derivative_matrix(globals.THETA_INDEX, prefactor=4.0 * 1j * self.ECS, periodic=True) # + \
              # sp.sparse.kron(sp.sparse.identity(pt_counts[globals.PHI_INDEX], format='dia'), 
                             # sp.sparse.identity(pt_counts[globals.THETA_INDEX], format='dia') * 4.0 * self.ECS * (self.ng)) #needed?


# ----------------------------------------------------------------------------------------



class FullZeroPi(SymZeroPi):
    """
    TODO: should add charge offset directly to this class 
    
    Full Zero-Pi Qubit, with all disorder types in circuit element parameters included. This couples
    the chi degree     of freedom, see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta-chi space
    into a simple cubic lattice.
    Expected parameters are:

    EJ:   mean Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    EC:   charging energy associated with chi degree of freedom
    dEJ:  relative disorder in EJ, i.e., (EJ1-EJ2)/EJ(mean)
    dEL:  relative disorder in EL, i.e., (EL1-EL2)/EL(mean)
    dCJ:  relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/C(mean)
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined as
    relative ones.

    TODO:
    - has to get updated to support charge offset. 
    - double check that factor of 1/2 consistent with disorder definition.


    """

    VARNAME_TO_INDEX = {'phi': globals.PHI_INDEX, 'theta': globals.THETA_INDEX, 'chi': globals.CHI_INDEX}

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'EC': 'charging energy associated with chi degree of freedom',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'dC': 'relative deviation between the two shunt capacitances',
        'dEL': 'relative deviation between the two inductances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(FullZeroPi, self).__init__(**parameter_args)
        self._sys_type = 'full 0-Pi circuit (phi, theta, chi), no offset charge'

    def sparse_kineticmat(self):
        return (
            self.grid.second_derivative_matrix(globals.PHI_INDEX, prefactor=-2.0 * self.ECJ) +                  # -2E_{CJ}\\partial_\\phi^2
            self.grid.second_derivative_matrix(globals.THETA_INDEX, prefactor=-2.0 * self.ECS, periodic=True) +   # -2E_{C\\Sigma}\\partial_\\theta^2
            self.grid.second_derivative_matrix(globals.CHI_INDEX, prefactor=-2.0 * self.EC) +                   # -2E_{C}\\partial_\\chi^2
            self.grid.multi_first_derivatives_matrix([globals.PHI_INDEX, globals.THETA_INDEX], prefactor=2.0 * self.ECS * self.dCJ,
                                                     periodic_var_indices=(globals.THETA_INDEX,)) +  # 4E_{C\\Sigma}(\\delta C_J/C_J)\\partial_\\phi \\partial_\\theta
            self.grid.multi_first_derivatives_matrix([globals.THETA_INDEX, globals.CHI_INDEX], prefactor=2.0 * self.ECS * self.dC,
                                                     periodic_var_indices=(globals.THETA_INDEX,))     # 4E_{C\\Sigma}(\\delta C/C)\\partial_\\theta \\partial_\\chi
            )

    def potential(self, phi, theta, chi):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2) + self.EL * phi**2 + 2 * self.EJ +   # symmetric 0-pi contributions
                self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2) + self.EL * chi**2 +       # correction terms in presence of disorder
                 self.EL * self.dEL * phi * chi + self.EJ * self.dEJ)                                                  # correction terms in presence of disorder

    def plot_potential(self, fixedvar_name, fixedvar_val, contour_vals=None, aspect_ratio=None, filename=None):
        fixedvar_index = self.VARNAME_TO_INDEX[fixedvar_name]

        othervar_indices = list({globals.PHI_INDEX, globals.THETA_INDEX, globals.CHI_INDEX} - {fixedvar_index})

        def reduced_potential(x, y):    # not very elegant, suspect there is a better way of coding this?
            func_arguments = [fixedvar_val] * 3
            func_arguments[othervar_indices[0]] = x
            func_arguments[othervar_indices[1]] = y
            return self.potential(*func_arguments)

        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()
        x_vals = np.linspace(min_vals[othervar_indices[0]], max_vals[othervar_indices[0]], pt_counts[othervar_indices[0]])
        y_vals = np.linspace(min_vals[othervar_indices[1]], max_vals[othervar_indices[1]], pt_counts[othervar_indices[1]])
        plot.contours(x_vals, y_vals, reduced_potential, contour_vals, aspect_ratio, filename)
        return None

    def d_potential_flux(self, phi, theta, chi):
        """Returns a derivative of U w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial U}{ \partial \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0.
        """
        return  (2.0 * np.pi * self.EJ * np.cos(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))  \
                - (np.pi * self.EJ * self.dEJ * np.sin(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0))
  
    def d_potential_EJ(self, phi, theta, chi):
        """Returns a derivative of the H w.r.t EJ. 
        This can be used for calculating critical current noise, which requires a derivative of d_H/d_I_c. 
        
        NOTE: We disregard the constant part of the potential energy ~ 2.0*self.EJ
        """
        return (-2.0 * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) + 
                self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def wavefunction(self, esys, which=0):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        return evecs[:, which]

    def plot_wavefunction(self, esys, fixedvar_name, fixedvar_val, which=0, mode='abs', figsize=(20, 10), aspect_ratio=3, axes=None):
        """Different modes:
        'abs_sqr': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """

        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()

        wavefunc = self.wavefunction(esys, which)
        modefunction = globals.MODE_FUNC_DICT[mode]
        wavefunc = modefunction(wavefunc)
        wavefunc = wavefunc.reshape(pt_counts[0], pt_counts[1], pt_counts[2])

        fixedvar_index = self.VARNAME_TO_INDEX[fixedvar_name]

        slice_index = int(pt_counts[fixedvar_index] * (fixedvar_val - min_vals[fixedvar_index]) /
                          (max_vals[fixedvar_index] - min_vals[fixedvar_index]))

        slice_coordinates3d = [slice(None), slice(None), slice(None)]
        slice_coordinates3d[fixedvar_index] = slice_index
        wavefunc = wavefunc[tuple(slice_coordinates3d)].T
        return plot.wavefunction2d(wavefunc, figsize, aspect_ratio, axes=axes)


# ----------------------------------------------------------------------------------------


class FullZeroPi_ProductBasis(BaseClass):

    """Full Zero-Pi Qubit, with all disorder types in circuit element parameters included. This couples
    the chi degree     of freedom, see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds in the product basis of the disordered (dEJ, dCJ)
    Zero-Pi qubit on one hand and the chi LC oscillator on the other hand.

    Expected parameters are:

    EJ:    mean Josephson energy of the two junctions
    EL:    inductive energy of the two (super-)inductors
    ECJ:   charging energy associated with the two junctions
    ECS:   charging energy including the large shunting capacitances
    EC:    charging energy associated with chi degree of freedom
    dEJ:   relative disorder in EJ, i.e., (EJ1-EJ2)/EJ(mean)
    dEL:   relative disorder in EL, i.e., (EL1-EL2)/EL(mean)
    dCJ:   relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/C(mean)
    flux:  magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    ng:    offset charge along theta
    zeropi_cutoff: cutoff in the number of states of the disordered zero-pi qubit
    chi_cut: cutoff in the chi oscillator basis (Fock state basis)
    grid: Grid object specifying the range and spacing of the discretization lattice


    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined as
    relative ones.
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'EC': 'charging energy associated with chi degree of freedom',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'dC': 'relative deviation between the two shunt capacitances',
        'dEL': 'relative deviation between the two inductances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'ng':    'offset charge along theta',
        'zeropi_cutoff': 'cutoff in the number of states of the disordered zero-pi qubit',
        'chi_cutoff': 'cutoff in the chi oscillator basis (Fock state basis)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    __initialized = False

    def init_parameters(self, **parameter_args):
        for parameter_name, parameter_val in parameter_args.items():
            setattr(self, parameter_name, parameter_val)

    def __init__(self, **parameter_args):
        """
        TODO (peterg): maybe we should get rid of one of the interdependent vars (ex. ECS) and calculate it 
        based on the others (ex: ECJ, EC)
        """
        super(FullZeroPi_ProductBasis, self).__init__(**parameter_args)
        self._sys_type = 'full 0-Pi circuit (phi, theta, chi) in 0pi - chi product basis'
        self._zeropi = DisZeroPiNg(
            EJ = self.EJ,
            EL = self.EL,
            ECJ = self.ECJ,
            ECS = self.ECS,
            dEJ = self.dEJ,
            dCJ = self.dCJ,
            flux = self.flux,
            ng = self.ng,
            grid = self.grid,
            truncated_dim = self.zeropi_cutoff
        )
        self.__initialized = True

    def __setattr__(self, parameter_name, parameter_val):
        super(FullZeroPi_ProductBasis, self).__setattr__(parameter_name, parameter_val)
        if self.__initialized and parameter_name in self._zeropi._EXPECTED_PARAMS_DICT.keys():
                self._zeropi.__setattr__(parameter_name, parameter_val)

    def omega_chi(self):
        return (8.0 * self.EL * self.EC)**0.5

    def hamiltonian(self, return_parts=False):
        """
        @param return_parts: Determines if we should also return intermediate components such as
                             the zeropi esys as well as the coupling matrix. 
        """
        zeropi_dim = self.zeropi_cutoff
        zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)
        zeropi_diag_hamiltonian = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.float_)
        zeropi_diag_hamiltonian.setdiag(zeropi_evals)

        chi_dim = self.chi_cutoff
        prefactor = self.omega_chi()
        chi_diag_hamiltonian = op.number_sparse(chi_dim, prefactor)

        hamiltonian_mat = sp.sparse.kron(zeropi_diag_hamiltonian, sp.sparse.identity(chi_dim, format='dia', dtype=np.float_))
        hamiltonian_mat += sp.sparse.kron(sp.sparse.identity(zeropi_dim, format='dia', dtype=np.float_), chi_diag_hamiltonian)

        gmat = self.g_coupling_matrix(zeropi_evecs)
        zeropi_coupling = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.float_)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                zeropi_coupling += gmat[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)
        hamiltonian_mat += sp.sparse.kron(zeropi_coupling, op.annihilation_sparse(chi_dim) + op.creation_sparse(chi_dim))

        if return_parts:
            return [hamiltonian_mat, zeropi_evals, zeropi_evecs, gmat]
        else:
            return hamiltonian_mat

    def _zeropi_operator_in_prodcuct_basis(self, zeropi_operator, zeropi_evecs=None):
        """
        Helper method that converts a zeropi operator into one in the product basis'

        TODO: Could update d_hamiltonian_EJ(),  d_hamiltonian_ng(),  d_hamiltonian_flux() to use this. 
        """
        zeropi_dim = self.zeropi_cutoff
        chi_dim = self.chi_cutoff

        if zeropi_evecs is None:
            zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        op_eigen_basis = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.complex_) #is this guaranteed to be zero?

        op_zeropi = matrixelem_table(zeropi_operator, zeropi_evecs, real_valued=False)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                op_eigen_basis += op_zeropi[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)

        return sp.sparse.kron(op_eigen_basis, sp.sparse.identity(chi_dim, format='dia', dtype=np.complex_))

    def i_d_dphi_operator(self, zeropi_evecs=None):
        """Return the operator i \\partial_\\phi"""
        return self._zeropi_operator_in_prodcuct_basis(self._zeropi.i_d_dphi_operator(), zeropi_evecs=zeropi_evecs)

    def d_dphi_operator(self, zeropi_evecs=None):
        """Return the operator \\partial_\\phi"""
        return self._zeropi_operator_in_prodcuct_basis(self._zeropi.d_dphi_operator(), zeropi_evecs=zeropi_evecs)

    def i_d_dtheta_operator(self, zeropi_evecs=None):
        """Return the operator i \\partial_\\theta (periodic variable)"""
        return self._zeropi_operator_in_prodcuct_basis(self._zeropi.i_d_dtheta_operator(), zeropi_evecs=zeropi_evecs)

    def d_dtheta_operator(self, zeropi_evecs=None):
        """Return the operator  \\partial_\\theta (periodic variable)"""
        return self._zeropi_operator_in_prodcuct_basis(self._zeropi.d_dtheta_operator(), zeropi_evecs=zeropi_evecs)

    def phi_operator(self, zeropi_evecs=None):
        """Return \phi operator"""
        return self._zeropi_operator_in_prodcuct_basis(self._zeropi.phi_operator(), zeropi_evecs=zeropi_evecs)

    def d2_hamiltonian_flux(self, zeropi_evecs=None):
        """Returns a second derivative of the H w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial^2 H}{ \partial^2 \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0^2.
        """
        return self._zeropi_operator_in_prodcuct_basis(self._zeropi.d2_hamiltonian_flux(), zeropi_evecs=zeropi_evecs)

    def d_hamiltonian_flux(self, zeropi_evecs=None):
        """Returns a derivative of the H w.r.t flux, at the "current" value of flux, as stored in the object. 
        The flux is assumed to be given in the units of the ratio \Phi_{ext}/\Phi_0. So if one needs a \frac{\partial H}{ \partial \Phi_{\rm ext}}, 
        the expression returned by this function, needs to be multiplied by 1/\Phi_0.

        TODO: update to _zeropi_operator_in_prodcuct_basis()
        """
        zeropi_dim = self.zeropi_cutoff
        chi_dim = self.chi_cutoff

        if zeropi_evecs is None:
            zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        d_h_flux_zeropi_eigen_basis = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.complex_) #is this guaranteed to be zero?

        d_h_flux_zeropi = matrixelem_table(self._zeropi.d_hamiltonian_flux(), zeropi_evecs, real_valued=False)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                d_h_flux_zeropi_eigen_basis += d_h_flux_zeropi[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)

        return sp.sparse.kron(d_h_flux_zeropi_eigen_basis, sp.sparse.identity(chi_dim, format='dia', dtype=np.complex_))

    def d_hamiltonian_ng(self, zeropi_evecs=None):
        """Returns a derivative of the H w.r.t ng

        TODO: update to _zeropi_operator_in_prodcuct_basis()
        """
        zeropi_dim = self.zeropi_cutoff
        chi_dim = self.chi_cutoff

        if zeropi_evecs is None:
            zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        d_h_ng_zeropi_eigen_basis = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.complex_) #is this guaranteed to be zero?

        d_h_ng_zeropi = matrixelem_table(self._zeropi.d_hamiltonian_ng(), zeropi_evecs, real_valued=False)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                d_h_ng_zeropi_eigen_basis += d_h_ng_zeropi[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)

        return sp.sparse.kron(d_h_ng_zeropi_eigen_basis, sp.sparse.identity(chi_dim, format='dia', dtype=np.complex_))

    def d_hamiltonian_EJ(self, zeropi_evecs=None):
        """Returns a derivative of the H w.r.t EJ

        TODO: update to _zeropi_operator_in_prodcuct_basis()
        """
        zeropi_dim = self.zeropi_cutoff
        chi_dim = self.chi_cutoff

        if zeropi_evecs is None:
            zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        d_h_EJ_zeropi_eigen_basis = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.complex_) #is this guaranteed to be zero?

        d_h_EJ_zeropi = matrixelem_table(self._zeropi.d_hamiltonian_EJ(), zeropi_evecs, real_valued=False)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                d_h_EJ_zeropi_eigen_basis += d_h_EJ_zeropi[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)

        return sp.sparse.kron(d_h_EJ_zeropi_eigen_basis, sp.sparse.identity(chi_dim, format='dia', dtype=np.complex_))


    def hilbertdim(self):
        return (self.zeropi_cutoff * self.chi_cutoff)

    def _evals_calc(self, evals_count, hamiltonian_mat=None):
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return evals

    def _esys_calc(self, evals_count, hamiltonian_mat=None):
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        return evals, evecs

    def g_phi_coupling_matrix(self, zeropi_states):
        """Returns a matrix of coupling strengths g^\\phi_{ll'} [cmp. Dempster et al., Eq. (18)], using the states
        from the list 'zeropi_states'. Most commonly, 'zeropi_states' will contain eigenvectors of the
        DisZeroPi type, so 'transpose' is enabled by default.
        """
        # prefactor = self.EL * self.dEL * (8.0 * self.EC / self.EL)**0.25
        prefactor = self.EL * (self.dEL / 2.0) * (8.0 * self.EC / self.EL)**0.25
        return (prefactor * matrixelem_table(self._zeropi.phi_operator(), zeropi_states, real_valued=True))

    def g_theta_coupling_matrix(self, zeropi_states):
        """Returns a matrix of coupling strengths i*g^\\theta_{ll'} [cmp. Dempster et al., Eq. (17)], using the states
        from the list 'zeropi_states'. Most commonly, 'zeropi_states' will contain eigenvectors, so 'transpose' is enabled by
        default.
        """
        # prefactor = - self.ECS * self.dC * (32.0 * self.EL / self.EC)**0.25
        prefactor = - self.ECS * (self.dC / 2.0) * (32.0 * self.EL / self.EC)**0.25
        return (prefactor * matrixelem_table(self._zeropi.d_dtheta_operator(), zeropi_states, real_valued=True))

    def g_coupling_matrix(self, zeropi_states=None, evals_count=None):
        """Returns a matrix of coupling strengths g_{ll'} [cmp. Dempster et al., text above Eq. (17)], using the states
        from 'state_list'.  Most commonly, 'zeropi_states' will contain eigenvectors of the
        DisZeroPi type, so 'transpose' is enabled by default.
        If zeropi_states==None, then a set of self.zeropi eigenstates is calculated. Only in that case is evals_count
        used for the eigenstate number (and hence the coupling matrix size).

        """
        if evals_count is None:
            evals_count = self._zeropi.truncated_dim
        if zeropi_states is None:
            _, zeropi_states = self._zeropi.eigensys(evals_count=evals_count)
        return (self.g_phi_coupling_matrix(zeropi_states) + self.g_theta_coupling_matrix(zeropi_states))

    def get_spectrum_vs_paramvals(self, parameter_name, paramval_list, evals_count=6, subtract_ground=False,
                                  get_eigenstates=False, filename=None):
        """We need to be careful here; some of the parameters depend on one another... so if we try to vary just one
        without adjusting the others, we'd get inconsistent results.

        TODO: Add a checks like this to other ZeroPi classes 
        """
        inter_dep_params=["ECS", "ECJ", "EC"] 
        if parameter_name in inter_dep_params: 
            raise ValueError("Currently can't vary any of {}, as they are inter-dependent.".format(", ".join(inter_dep_params)))

        return super(FullZeroPi_ProductBasis, self).get_spectrum_vs_paramvals(parameter_name, paramval_list, evals_count=evals_count, subtract_ground=subtract_ground,
                                  get_eigenstates=get_eigenstates, filename=filename)

    

