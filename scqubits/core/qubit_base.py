# qubit_base.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
"""
Provides the base classes for qubits
"""

import abc
import copy
import numpy as np
import scipy as sp

import scqubits.settings as config
import scqubits.utils.constants as constants
import scqubits.utils.plotting as plot
import scqubits.utils.progressbar as progressbar

from scqubits.utils.constants import FileType
from scqubits.core.data_containers import SpectrumData
from scqubits.utils.file_write import filewrite_csvdata, filewrite_h5data
from scqubits.utils.spectrum_utils import order_eigensystem, get_matrixelement_table

# —Generic quantum system container and Qubit base class————————————————————————————————————————————————————————————————

class QuantumSystem(object):
    """Generic quantum system class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._sys_type = 'generic quantum system - only used as class template'

    def __repr__(self):
        output = self._sys_type + '\n ———— PARAMETERS ————'
        for param_name in self.__dict__.keys():
            if param_name[0] is not '_':
                paramval = self.__dict__[param_name]
                output += '\n' + str(param_name) + '\t: ' + str(paramval)
        output += '\nHilbert space dimension\t: ' + str(self.hilbertdim())
        return output

    def dict_reformat(self):
        """Provides `__dict__` in reformatted dictionary form (all values as strings); needed for .h5 output.

        Returns
        -------
        dict(str)
        """
        dict_reformatted = copy.deepcopy(self.__dict__)
        for key, value in dict_reformatted.items():
            dict_reformatted[key] = str(value)
        return dict_reformatted

    @abc.abstractmethod
    def hilbertdim(self):
        """Returns dimension of Hilbert space"""
        pass




# —QubitBaseClass———————————————————————————————————————————————————————————————————————————————————————————————————————

class QubitBaseClass(QuantumSystem):
    """Base class for superconducting qubit objects. Provide general mechanisms and routines for
    checking validity of initialization parameters, writing data to files, and plotting.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._sys_type = 'QubitBaseClass - mainly used as class template'

    def filewrite_evals(self, filename, evals):
        """Write eigenvalue data to file.

        Parameters
        ----------
        filename: str
            path and name of output file (suffix appended automatically)
        evals: ndarray
            eigenvalue data
        """
        if config.file_format is FileType.csv:
            filewrite_csvdata(filename + constants.EVALS_FILESUFFIX, evals)
            with open(filename + constants.PARAMETER_FILESUFFIX, 'w') as target_file:
                target_file.write(self.__repr__())
        elif config.file_format is FileType.h5:
            filewrite_h5data(filename, [evals], ["eigenenergies"], self.dict_reformat())

    def filewrite_esys(self, filename, evals, evecs):
        """Write eigenvector data to file.

        Parameters
        ----------
        filename: str
            path and name of output file (suffix appended automatically)
        evals: ndarray
            eigenvalue data
        evecs: ndarray
            eigenvector data
        """
        if config.file_format is FileType.csv:
            filewrite_csvdata(filename + constants.EVALS_FILESUFFIX, evals)
            filewrite_csvdata(filename + constants.EVECS_FILESUFFIX, evecs)
            with open(filename + constants.PARAMETER_FILESUFFIX, 'w') as target_file:
                target_file.write(self.__repr__())
        elif config.file_format is FileType.h5:
            filewrite_h5data(filename, [evals, evecs], ["eigenenergies", "eigenvectors"], self.dict_reformat())

    @abc.abstractmethod
    def hamiltonian(self):
        """Returns the Hamiltonian"""
        pass

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(hamiltonian_mat, eigvals_only=True, eigvals=(0, evals_count - 1))
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(hamiltonian_mat, eigvals_only=False, eigvals=(0, evals_count - 1))
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    def eigenvals(self, evals_count=6, filename=None):
        """Calculates eigenvalues using `scipy.linalg.eigh`, returns numpy array of eigenvalues.

        Parameters
        ----------
        evals_count: int
            number of desired eigenvalues/eigenstates (Default value = 6)
        filename: str, optional
            path and filename without suffix, if file output desired (Default value = None)

        Returns
        -------
        ndarray
        """
        evals = self._evals_calc(evals_count)
        if filename:
            self.filewrite_evals(filename, evals)
        return evals

    def eigensys(self, evals_count=6, filename=None):
        """Calculates eigenvalues and corresponding eigenvectors using `scipy.linalg.eigh`. Returns
        two numpy arrays containing the eigenvalues and eigenvectors, respectively.

        Parameters
        ----------
        evals_count: int, optional
            number of desired eigenvalues/eigenstates (Default value = 6)
        filename: str, optional
            path and filename without suffix, if file output desired (Default value = None)

        Returns
        -------
        ndarray, ndarray
            eigenvalues, eigenvectors
        """
        evals, evecs = self._esys_calc(evals_count)
        if filename:
            self.filewrite_esys(filename, evals, evecs)
        return evals, evecs

    def matrixelement_table(self, operator, esys=None, evals_count=6):
        """Returns table of matrix elements for `operator` with respect to the eigenstates of the qubit.
        The operator is given as a string matching a class method returning an operator matrix.
        E.g., for an instance `trm` of Transmon,  the matrix element table for the charge operator is given by
        `trm.op_matrixelement_table('n_operator')`.
        When `esys` is set to `None`, the eigensystem is calculated on-the-fly.

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix in qubit-internal basis.
        esys: tuple(ndarray, ndarray), optional
            if set, matrix elements are calculated based on the provided eigensystem data (Default value = None)
        evals_count: int
            number of desired matrix elements, starting with ground state (Default value = 6)

        Returns
        -------
        ndarray
        """
        if esys is None:
            evals, evecs = self.eigensys(evals_count=evals_count)
        else:
            _, evecs = esys
        operator_matrix = getattr(self, operator)()
        return get_matrixelement_table(operator_matrix, evecs)

    def plot_matrixelements(self, operator, esys=None, evals_count=6, mode='abs', xlabel='', ylabel='', zlabel='',
                            fig_ax=None):
        """Plots matrix elements for `operator`, given as a string referring to a class method
        that returns an operator matrix. E.g., for instance `trm` of Transmon, the matrix element plot
        for the charge operator `n` is obtained by `trm.plot_matrixelements('n')`.
        When `esys` is set to None, the eigensystem with `evals_count` eigenvectors is calculated.

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix
        esys: tuple(ndarray,ndarray), optional
            eigensystem data of evals, evecs; calculates eigensystem if set to None (Default value = None)
        evals_count: int, optional
            number of desired matrix elements, starting with ground state (Default value = 6)
        mode: str, optional
            entry from MODE_FUNC_DICTIONARY, e.g., `'abs'` for absolute value (default)
        xlabel, ylabel, zlabel: str, optional
            labels for the three plot axes
        fig_ax: tuple(Figure, Axes), optional
            fig and ax objects for matplotlib figure addition (Default value = None)
        xlabel :
             (Default value = '')
        ylabel :
             (Default value = '')
        zlabel :
             (Default value = '')

        Returns
        -------
        Figure, Axes
        """
        matrixelem_array = self.matrixelement_table(operator, esys, evals_count)
        return plot.matrix(matrixelem_array, mode, xlabel, ylabel, zlabel, fig_ax=fig_ax)

    def get_spectrum_vs_paramvals(self, param_name, param_vals, evals_count=6, subtract_ground=False,
                                  get_eigenstates=False, filename=None):
        """Calculates eigenvalues for a varying system parameter, given an array of parameter values. Returns a
        `SpectrumData` object with `energy_data[n]` containing eigenvalues calculated for
        parameter value `param_vals[n]`.

        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        evals_count: int, optional
            number of desired eigenvalues (sorted from smallest to largest) (Default value = 6)
        subtract_ground: bool, optional
            if True, eigenvalues are returned relative to the ground state eigenvalue (Default value = False)
        get_eigenstates: bool, optional
            return eigenstates along with eigenvalues (Default value = False)
        filename: str, optional
            write data to file if path and filename are specified (Default value = None)

        Returns
        -------
        SpectrumData object
        """
        previous_paramval = getattr(self, param_name)
        paramvals_count = len(param_vals)
        eigenvalue_table = np.zeros((paramvals_count, evals_count), dtype=np.float_)

        if get_eigenstates:
            eigenstate_table = np.empty(shape=(paramvals_count, self.hilbertdim(), evals_count), dtype=np.float_)
        else:
            eigenstate_table = None

        progressbar.initialize()
        for index, paramval in enumerate(param_vals):
            setattr(self, param_name, paramval)

            if get_eigenstates:
                evals, evecs = self.eigensys(evals_count)
                eigenstate_table[index] = evecs
            else:
                evals = self.eigenvals(evals_count)
            eigenvalue_table[index] = evals

            if subtract_ground:
                eigenvalue_table[index] -= evals[0]

            progress_in_percent = (index + 1) / paramvals_count
            progressbar.update(progress_in_percent)
        setattr(self, param_name, previous_paramval)

        spectrumdata = SpectrumData(param_name, param_vals, eigenvalue_table, self.dict_reformat(),
                                    state_table=eigenstate_table)
        if filename:
            spectrumdata.filewrite(filename)

        return spectrumdata

    def get_matelements_vs_paramvals(self, operator, param_name, param_vals, evals_count=6, filename=None):
        """Calculates matrix elements for a varying system parameter, given an array of parameter values. Returns a
        `SpectrumData` object containing matrix element data, eigenvalue data, and eigenstate data..

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        evals_count: int, optional
            number of desired eigenvalues (sorted from smallest to largest) (Default value = 6)
        filename: str, optional
            write data to file if path and filename are specified (Default value = None)

        Returns
        -------
        SpectrumData object
        """
        previous_paramval = getattr(self, param_name)
        paramvals_count = len(param_vals)
        eigenvalue_table = np.zeros((paramvals_count, evals_count), dtype=np.float_)

        eigenstate_table = np.empty(shape=(paramvals_count, self.hilbertdim(), evals_count), dtype=np.complex_)
        matelem_table = np.empty(shape=(paramvals_count, evals_count, evals_count), dtype=np.complex_)

        progressbar.initialize()
        for index, paramval in enumerate(param_vals):
            setattr(self, param_name, paramval)
            evals, evecs = self.eigensys(evals_count)
            eigenstate_table[index] = evecs
            eigenvalue_table[index] = evals

            matelem_table[index] = self.matrixelement_table(operator, evals_count=evals_count)

            progress_in_percent = (index + 1) / paramvals_count
            progressbar.update(progress_in_percent)

        setattr(self, param_name, previous_paramval)

        spectrumdata = SpectrumData(param_name, param_vals, eigenvalue_table, self.dict_reformat(),
                                    state_table=eigenstate_table, matrixelem_table=matelem_table)
        if filename:
            spectrumdata.filewrite(filename)

        return spectrumdata

    def plot_evals_vs_paramvals(self, param_name, param_vals, evals_count=6, subtract_ground=False,
                                x_range=False, y_range=False, filename=None, fig_ax=None):
        """Generates a simple plot of a set of eigenvalues as a function of one parameter.
        The individual points correspond to the a provided array of parameter values.

        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        evals_count: int, optional
            number of desired eigenvalues (sorted from smallest to largest) (Default value = 6)
        subtract_ground: bool, optional
            whether to subtract ground state energy from all eigenvalues (Default value = False)
        x_range: (float, float), optional
            custom x-range for the plot (Default value = False)
        y_range: (float, float), optional
            custom y-range for the plot (Default value = False)
        filename: str, optional
            write graphics and parameter set to file if path and filename are specified (Default value = None)
        fig_ax: tuple(Figure, Axes), optional
            fig and ax objects for matplotlib figure addition (Default value = None)

        Returns
        -------
        Figure, Axes
        """
        specdata = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count, subtract_ground)
        return plot.evals_vs_paramvals(specdata, evals_count=evals_count, xlim=x_range, ylim=y_range, filename=filename,
                                       fig_ax=fig_ax)

    def plot_matelem_vs_paramvals(self, operator, param_name, param_vals, select_elems=4, mode='abs',
                                  x_range=False, y_range=False, filename=None, fig_ax=None):
        """Generates a simple plot of a set of eigenvalues as a function of one parameter.
        The individual points correspond to the a provided array of parameter values.

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        select_elems: int or list, optional
            either maximum index of desired matrix elements, or list [(i1, i2), (i3, i4), ...] of index tuples
            for specific desired matrix elements (Default value = 4)
        mode: str, optional
            entry from MODE_FUNC_DICTIONARY, e.g., `'abs'` for absolute value (Default value = 'abs')
        x_range: (float, float), optional
            custom x-range for the plot (Default value = False)
        y_range: (float, float), optional
            custom y-range for the plot (Default value = False)
        filename: str, optional
            write graphics and parameter set to file if path and filename are specified (Default value = None)
        fig_ax: tuple(Figure, Axes), optional
            fig and ax objects for matplotlib figure addition (Default value = None)

        Returns
        -------
        Figure, Axes
        """
        if isinstance(select_elems, int):
            evals_count = select_elems
        else:
            flattened_list = [index for tupl in select_elems for index in tupl]
            evals_count = max(flattened_list) + 1

        specdata = self.get_matelements_vs_paramvals(operator, param_name, param_vals, evals_count=evals_count,
                                                     filename=None)
        return plot.matelem_vs_paramvals(specdata, select_elems=select_elems, mode=mode, xlim=x_range, ylim=y_range,
                                         filename=filename, fig_ax=fig_ax)
