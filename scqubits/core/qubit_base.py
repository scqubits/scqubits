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

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import scqubits as qubits
import scqubits.utils.constants as constants
import scqubits.utils.plotting as plot
from scqubits.core.data_containers import SpectrumData
from scqubits.settings import in_ipython, TQDM_KWARGS
from scqubits.utils.misc import process_which, process_metadata
from scqubits.utils.spectrum_utils import order_eigensystem, get_matrixelement_table

if in_ipython:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# —Generic quantum system container and Qubit base class————————————————————————————————————————————————————————————————

class QuantumSystem:
    """Generic quantum system class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._sys_type = 'quantum_system'

    def __str__(self):
        output = self._sys_type.upper() + '\n ———— PARAMETERS ————'
        for param_name, param_val in self.__dict__.items():
            if param_name[0] != '_':
                output += '\n' + str(param_name) + '\t: ' + str(param_val)
        output += '\nHilbert space dimension\t: ' + str(self.hilbertdim())
        return output

    @abc.abstractmethod
    def hilbertdim(self):
        """Returns dimension of Hilbert space"""

    def _get_metadata_dict(self):
        return process_metadata(self.__dict__)


# —QubitBaseClass———————————————————————————————————————————————————————————————————————————————————————————————————————

class QubitBaseClass(QuantumSystem):
    """Base class for superconducting qubit objects. Provide general mechanisms and routines
    for plotting spectra, matrix elements, and writing data to files
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, truncated_dim=None):
        super().__init__()
        self._default_var_range = None
        self._default_var_count = None
        self._evec_dtype = None

    @abc.abstractmethod
    def hamiltonian(self):
        """Returns the Hamiltonian"""

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
            specdata = SpectrumData('const_parameters', param_vals=np.empty(0), energy_table=evals,
                                    system_params=self._get_metadata_dict())
            specdata.filewrite(filename)
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
            specdata = SpectrumData('const_parameters', param_vals=np.empty(0), energy_table=evals,
                                    system_params=self._get_metadata_dict(), state_table=evecs)
            specdata.filewrite(filename)
        return evals, evecs

    def try_defaults(self, var_range, var_count):
        """
        Parameters
        ----------
        var_range: None or tuple(float, float)
        var_count: None or int

        Returns
        -------
        If any of the arguments is None, return default values.
        """
        if var_range is None:
            var_range = self._default_var_range
        if var_count is None:
            var_count = self._default_var_count
        return var_range, var_count

    def matrixelement_table(self, operator, evecs=None, evals_count=6, filename=None):
        """Returns table of matrix elements for `operator` with respect to the eigenstates of the qubit.
        The operator is given as a string matching a class method returning an operator matrix.
        E.g., for an instance `trm` of Transmon,  the matrix element table for the charge operator is given by
        `trm.op_matrixelement_table('n_operator')`.
        When `esys` is set to `None`, the eigensystem is calculated on-the-fly.

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix in qubit-internal basis.
        evecs: ndarray, optional
            if not provided, then the necesssary eigenstates are calculated on the fly
        evals_count: int
            number of desired matrix elements, starting with ground state (Default value = 6)
        filename: str, optional
            output file name

        Returns
        -------
        ndarray
        """
        if evecs is None:
            _, evecs = self.eigensys(evals_count=evals_count)
        operator_matrix = getattr(self, operator)()
        table = get_matrixelement_table(operator_matrix, evecs)
        if filename:
            specdata = SpectrumData('const_parameters', param_vals=np.empty(0), energy_table=np.empty(0),
                                    system_params=self._get_metadata_dict(), matrixelem_table=table)
            specdata.filewrite(filename)
        return table

    def plot_matrixelements(self, operator, evecs=None, evals_count=6, mode='abs', xlabel='', ylabel='', zlabel='',
                            fig_ax=None):
        """Plots matrix elements for `operator`, given as a string referring to a class method
        that returns an operator matrix. E.g., for instance `trm` of Transmon, the matrix element plot
        for the charge operator `n` is obtained by `trm.plot_matrixelements('n')`.
        When `esys` is set to None, the eigensystem with `which` eigenvectors is calculated.

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix
        evecs: None or ndarray, optional
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
        matrixelem_array = self.matrixelement_table(operator, evecs, evals_count)
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
            eigenstate_table = np.empty(shape=(paramvals_count, self.hilbertdim(), evals_count), dtype=self._evec_dtype)
        else:
            eigenstate_table = None

        for index, paramval in tqdm(enumerate(param_vals), total=len(param_vals), **TQDM_KWARGS):
            setattr(self, param_name, paramval)

            if get_eigenstates:
                evals, evecs = self.eigensys(evals_count)
                eigenstate_table[index] = evecs
            else:
                evals = self.eigenvals(evals_count)
            eigenvalue_table[index] = np.real(evals)   # for complex-hermitean H, eigenvalues have type np.complex_

            if subtract_ground:
                eigenvalue_table[index] -= evals[0]
        setattr(self, param_name, previous_paramval)

        spectrumdata = SpectrumData(param_name, param_vals, eigenvalue_table, self._get_metadata_dict(),
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

        for index, paramval in tqdm(enumerate(param_vals), total=len(param_vals), **TQDM_KWARGS):
            setattr(self, param_name, paramval)
            evals, evecs = self.eigensys(evals_count)
            eigenstate_table[index] = evecs
            eigenvalue_table[index] = evals
            matelem_table[index] = self.matrixelement_table(operator, evals_count=evals_count)
        setattr(self, param_name, previous_paramval)

        spectrumdata = SpectrumData(param_name, param_vals, eigenvalue_table, self._get_metadata_dict(),
                                    state_table=eigenstate_table, matrixelem_table=matelem_table)
        if filename:
            spectrumdata.filewrite(filename)
        return spectrumdata

    def plot_evals_vs_paramvals(self, param_name, param_vals, evals_count=6, subtract_ground=None,
                                x_range=None, ymax=None, filename=None, fig_ax=None):
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
            custom x-range for the plot
        ymax: float, optional
            custom upper y bound for the plot
        filename: str, optional
            write graphics and parameter set to file if path and filename are specified (Default value = None)
        fig_ax: tuple(Figure, Axes), optional
            fig and ax objects for matplotlib figure addition (Default value = None)

        Returns
        -------
        Figure, Axes
        """
        specdata = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count, subtract_ground)
        return plot.evals_vs_paramvals(specdata, which=range(evals_count), x_range=x_range, ymax=ymax,
                                       filename=filename, fig_ax=fig_ax)

    def plot_matelem_vs_paramvals(self, operator, param_name, param_vals, select_elems=4, mode='abs',
                                  x_range=None, y_range=None, filename=None, fig_ax=None):
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
        return plot.matelem_vs_paramvals(specdata, select_elems=select_elems, mode=mode, x_range=x_range,
                                         y_range=y_range, filename=filename, fig_ax=fig_ax)

    def set_params_from_dict(self, meta_dict):
        """Set object parameters by given metadata dictionary

        Parameters
        ----------
        meta_dict: dict
        """
        for param_name in meta_dict.keys():
            param_value = meta_dict[param_name]
            if isinstance(param_value, (int, float, np.number)):
                setattr(self, param_name, param_value)


# —QubitBaseClass1d———————————————————————————————————————————————————————————————————————————————————————————————————————

class QubitBaseClass1d(QubitBaseClass):
    """Base class for superconducting qubit objects with one degree of freedom. Provide general mechanisms and routines
    for plotting spectra, matrix elements, and writing data to files.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, truncated_dim=None):
        self._sys_type = 'qubit system'
        self._evec_dtype = np.float_
        self.truncated_dim = truncated_dim
        self._default_var_range = None
        self._default_var_count = None

    @abc.abstractmethod
    def wavefunction(self, esys, which=0, phi_range=None, phi_count=None):
        pass

    @abc.abstractmethod
    def potential(self, phi):
        pass

    def plot_wavefunction(self, esys, which=0, phi_range=None, phi_count=None, mode='real', scaling=None,
                          xlabel=r'$\varphi$', ylabel=r'$\psi_j(\varphi),\, V(\varphi)$', y_range=None, title=None,
                          filename=None, fig_ax=None):
        """Plot 1d phase-basis wave function(s). Must be overwritten by higher-dimensional qubits like FluxQubits and
        ZeroPi.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int or tuple or list, optional
            single index or tuple/list of integers indexing the wave function(s) to be plotted.
            If which is -1, all wavefunctions up to the truncation limit are plotted.
        phi_range: None or tuple(float, float)
            used for setting a custom plot range for phi
        phi_count: int, optional
            number of points on the x-axis (resolution) (Default value = 251)
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (Default value = 'abs_sqr')
        scaling: float or None, optional
            custom scaling of wave function amplitude/modulus
        xlabel, ylabel: str, optional
            axes labels
        y_range: tuple(float, float), optional
            used to set custom y range for plot
        title: str, optional
            plot title
        filename: str, optional
            file path and name (not including suffix) for output
        fig_ax: Figure, Axes

        Returns
        -------
        Figure, Axes
        """
        modefunction = constants.MODE_FUNC_DICT[mode]

        index_list = process_which(which, self.truncated_dim)

        if fig_ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = fig_ax

        phi_wavefunc = self.wavefunction(esys, which=index_list[-1], phi_range=phi_range, phi_count=phi_count)
        potential_vals = self.potential(phi_wavefunc.basis_labels)

        if scaling is None:
            if isinstance(self, qubits.Transmon):
                scale = 0.2 * self.EJ
            elif isinstance(self, qubits.Fluxonium):
                scale = 0.125 * (np.max(potential_vals) - np.min(potential_vals))
        else:
            scale = scaling

        for wavefunc_index in index_list:
            phi_wavefunc = self.wavefunction(esys, which=wavefunc_index, phi_range=phi_range, phi_count=phi_count)
            if np.sum(phi_wavefunc.amplitudes) < 0:
                phi_wavefunc.amplitudes *= -1.0

            phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)

            plot.wavefunction1d(phi_wavefunc, potential_vals=potential_vals, offset=phi_wavefunc.energy,
                                scaling=scale, xlabel=xlabel, ylabel=ylabel, y_range=y_range, title=title,
                                fig_ax=(fig, ax), filename=filename)
        return fig, ax
