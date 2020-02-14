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
import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import scqubits.core.constants as constants
import scqubits.settings as settings
import scqubits.utils.plotting as plot
from scqubits.core.central_dispatch import DispatchClient
from scqubits.core.discretization import Grid1d
from scqubits.core.storage import SpectrumData
from scqubits.settings import IN_IPYTHON, TQDM_KWARGS
from scqubits.utils.misc import process_which, process_metadata, filter_metadata, InfoBar
from scqubits.utils.plot_defaults import set_scaling
from scqubits.utils.processing_switch import get_map_method
from scqubits.utils.spectrum_utils import (order_eigensystem, get_matrixelement_table, standardize_sign,
                                           recast_esys_mapdata)

if IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# —Generic quantum system container and Qubit base class————————————————————————————————————————————————————————————————

class QuantumSystem(DispatchClient):
    """Generic quantum system class"""
    __metaclass__ = abc.ABCMeta

    # see PEP 526 https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations
    truncated_dim: int
    _evec_dtype: type
    _sys_type: str

    def __str__(self):
        output = self._sys_type.upper() + '\n ———— PARAMETERS ————'
        for param_name, param_val in self.__dict__.items():
            if param_name[0] != '_':
                output += '\n' + str(param_name) + '\t: ' + str(param_val)
        output += '\nHilbert space dimension\t: ' + str(self.hilbertdim())
        return output

    def _get_metadata_dict(self):
        return process_metadata(self.__dict__)

    @abc.abstractmethod
    def hilbertdim(self):
        """Returns dimension of Hilbert space"""
        raise NotImplementedError


# —QubitBaseClass———————————————————————————————————————————————————————————————————————————————————————————————————————

class QubitBaseClass(QuantumSystem):
    """Base class for superconducting qubit objects. Provide general mechanisms and routines
    for plotting spectra, matrix elements, and writing data to files
    """
    __metaclass__ = abc.ABCMeta

    # see PEP 526 https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations
    truncated_dim: int
    _default_grid: Grid1d
    _evec_dtype: type
    _sys_type: str

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

    @classmethod
    def create_from_dict(cls, meta_dict):
        """Return a qubit object initialized by data from metadata dictionary

        Parameters
        ----------
        meta_dict: dict

        Returns
        -------
        object
        """
        filtered_dict = filter_metadata(meta_dict)
        return cls(**filtered_dict)

    def set_params_from_dict(self, meta_dict):
        """Set object parameters by given metadata dictionary

        Parameters
        ----------
        meta_dict: dict
        """
        filtered_dict = filter_metadata(meta_dict)
        for param_name, param_value in filtered_dict.items():
            setattr(self, param_name, param_value)

    def eigenvals(self, evals_count=6, filename=None):
        """Calculates eigenvalues using `scipy.linalg.eigh`, returns numpy array of eigenvalues.

        Parameters
        ----------
        evals_count: int
            number of desired eigenvalues/eigenstates (default value = 6)
        filename: str, optional
            path and filename without suffix, if file output desired (default value = None)

        Returns
        -------
        ndarray
        """
        evals = self._evals_calc(evals_count)
        if filename:
            specdata = SpectrumData(energy_table=evals, system_params=self._get_metadata_dict())
            specdata.filewrite(filename)
        return evals

    def eigensys(self, evals_count=6, filename=None):
        """Calculates eigenvalues and corresponding eigenvectors using `scipy.linalg.eigh`. Returns
        two numpy arrays containing the eigenvalues and eigenvectors, respectively.

        Parameters
        ----------
        evals_count: int, optional
            number of desired eigenvalues/eigenstates (default value = 6)
        filename: str, optional
            path and filename without suffix, if file output desired (default value = None)

        Returns
        -------
        ndarray, ndarray
            eigenvalues, eigenvectors
        """
        evals, evecs = self._esys_calc(evals_count)
        if filename:
            specdata = SpectrumData(energy_table=evals, system_params=self._get_metadata_dict(), state_table=evecs)
            specdata.filewrite(filename)
        return evals, evecs

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
        evals_count: int, optional
            number of desired matrix elements, starting with ground state (default value = 6)
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
            specdata = SpectrumData(energy_table=np.empty(0), system_params=self._get_metadata_dict(),
                                    param_name='const_parameters', param_vals=np.empty(0), matrixelem_table=table)
            specdata.filewrite(filename)
        return table

    def _esys_for_paramval(self, paramval, param_name, evals_count):
        setattr(self, param_name, paramval)
        return self.eigensys(evals_count)

    def _evals_for_paramval(self, paramval, param_name, evals_count):
        setattr(self, param_name, paramval)
        return self.eigenvals(evals_count)

    def get_spectrum_vs_paramvals(self, param_name, param_vals, evals_count=6, subtract_ground=False,
                                  get_eigenstates=False, filename=None, num_cpus=settings.NUM_CPUS):
        """Calculates eigenvalues/eigenstates for a varying system parameter, given an array of parameter values.
        Returns a `SpectrumData` object with `energy_data[n]` containing eigenvalues calculated for
        parameter value `param_vals[n]`.

        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        evals_count: int, optional
            number of desired eigenvalues (sorted from smallest to largest) (default value = 6)
        subtract_ground: bool, optional
            if True, eigenvalues are returned relative to the ground state eigenvalue (default value = False)
        get_eigenstates: bool, optional
            return eigenstates along with eigenvalues (default value = False)
        filename: str, optional
            file name if direct output to disk is wanted
        num_cpus: int, optional
            number of cores to be used for computation (default value: settings.NUM_CPUS)

        Returns
        -------
        SpectrumData object
        """
        previous_paramval = getattr(self, param_name)

        target_map = get_map_method(num_cpus)
        if get_eigenstates:
            func = functools.partial(self._esys_for_paramval, param_name=param_name, evals_count=evals_count)
            with InfoBar("Parallel computation of eigenvalues [num_cpus={}]".format(num_cpus), num_cpus):
                eigensystem_mapdata = list(target_map(func, tqdm(param_vals, desc='Spectral data', leave=False,
                                                                 disable=(num_cpus > 1))))
            eigenvalue_table, eigenstate_table = recast_esys_mapdata(eigensystem_mapdata)
        else:
            func = functools.partial(self._evals_for_paramval, param_name=param_name, evals_count=evals_count)
            with InfoBar("Parallel computation of eigensystems [num_cpus={}]".format(num_cpus), num_cpus):
                eigenvalue_table = list(target_map(func, tqdm(param_vals, desc='Spectral data', leave=False,
                                                              disable=(num_cpus > 1))))
            eigenvalue_table = np.asarray(eigenvalue_table)
            eigenstate_table = None

        if subtract_ground:
            for param_index, _ in enumerate(param_vals):
                eigenvalue_table[param_index] -= eigenvalue_table[param_index, 0]

        setattr(self, param_name, previous_paramval)
        specdata = SpectrumData(eigenvalue_table, self._get_metadata_dict(), param_name, param_vals,
                                state_table=eigenstate_table)
        if filename:
            specdata.filewrite(filename)

        return SpectrumData(eigenvalue_table, self._get_metadata_dict(), param_name, param_vals,
                            state_table=eigenstate_table)

    def get_matelements_vs_paramvals(self, operator, param_name, param_vals, evals_count=6, num_cpus=settings.NUM_CPUS):
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
            number of desired eigenvalues (sorted from smallest to largest) (default value = 6)
        num_cpus: int, optional
            number of cores to be used for computation (default value: settings.NUM_CPUS)

        Returns
        -------
        SpectrumData object
        """
        spectrumdata = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=evals_count,
                                                      get_eigenstates=True, num_cpus=num_cpus)
        paramvals_count = len(param_vals)
        matelem_table = np.empty(shape=(paramvals_count, evals_count, evals_count), dtype=np.complex_)

        for index, paramval in tqdm(enumerate(param_vals), total=len(param_vals), **TQDM_KWARGS):
            evecs = spectrumdata.state_table[index]
            matelem_table[index] = self.matrixelement_table(operator, evecs=evecs, evals_count=evals_count)

        spectrumdata.matrixelem_table = matelem_table
        return spectrumdata

    def plot_evals_vs_paramvals(self, param_name, param_vals,
                                evals_count=6, subtract_ground=None, num_cpus=settings.NUM_CPUS, **kwargs):
        """Generates a simple plot of a set of eigenvalues as a function of one parameter.
        The individual points correspond to the a provided array of parameter values.

        Parameters
        ----------
        param_name: str
            name of parameter to be varied
        param_vals: ndarray
            parameter values to be plugged in
        evals_count: int, optional
            number of desired eigenvalues (sorted from smallest to largest) (default value = 6)
        subtract_ground: bool, optional
            whether to subtract ground state energy from all eigenvalues (default value = False)
        num_cpus: int, optional
            number of cores to be used for computation (default value: settings.NUM_CPUS)
        **kwargs: dict
            standard plotting option (see separate documentation)

        Returns
        -------
        Figure, Axes
        """
        specdata = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=evals_count,
                                                  subtract_ground=subtract_ground, num_cpus=num_cpus)
        return plot.evals_vs_paramvals(specdata, which=range(evals_count), **kwargs)

    def plot_matrixelements(self, operator, evecs=None, evals_count=6, mode='abs', **kwargs):
        """Plots matrix elements for `operator`, given as a string referring to a class method
        that returns an operator matrix. E.g., for instance `trm` of Transmon, the matrix element plot
        for the charge operator `n` is obtained by `trm.plot_matrixelements('n')`.
        When `esys` is set to None, the eigensystem with `which` eigenvectors is calculated.

        Parameters
        ----------
        operator: str
            name of class method in string form, returning operator matrix
        evecs: ndarray, optional
            eigensystem data of evals, evecs; eigensystem will be calculated if set to None (default value = None)
        evals_count: int, optional
            number of desired matrix elements, starting with ground state (default value = 6)
        mode: str, optional
            entry from MODE_FUNC_DICTIONARY, e.g., `'abs'` for absolute value (default)
        **kwargs: dict
            standard plotting option (see separate documentation)

        Returns
        -------
        Figure, Axes
        """
        matrixelem_array = self.matrixelement_table(operator, evecs, evals_count)
        return plot.matrix(matrixelem_array, mode, **kwargs)

    def plot_matelem_vs_paramvals(self, operator, param_name, param_vals,
                                  select_elems=4, mode='abs', num_cpus=settings.NUM_CPUS, **kwargs):
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
            for specific desired matrix elements (default value = 4)
        mode: str, optional
            entry from MODE_FUNC_DICTIONARY, e.g., `'abs'` for absolute value (default value = 'abs')
        num_cpus: int, optional
            number of cores to be used for computation (default value = 1)
        **kwargs: dict
            standard plotting option (see separate documentation)

        Returns
        -------
        Figure, Axes
        """
        if isinstance(select_elems, int):
            evals_count = select_elems
        else:
            flattened_list = [index for tupl in select_elems for index in tupl]
            evals_count = max(flattened_list) + 1

        specdata = self.get_matelements_vs_paramvals(operator, param_name, param_vals,
                                                     evals_count=evals_count, num_cpus=num_cpus)
        return plot.matelem_vs_paramvals(specdata, select_elems=select_elems, mode=mode, **kwargs)


# —QubitBaseClass1d—————————————————————————————————————————————————————————————————————————————————————————————————————

class QubitBaseClass1d(QubitBaseClass):
    """Base class for superconducting qubit objects with one degree of freedom. Provide general mechanisms and routines
    for plotting spectra, matrix elements, and writing data to files.
    """
    __metaclass__ = abc.ABCMeta

    _evec_dtype = np.float_
    _default_grid: Grid1d

    @abc.abstractmethod
    def potential(self, phi):
        raise NotImplementedError

    @abc.abstractmethod
    def wavefunction(self, esys, which=0, phi_grid=None):
        raise NotImplementedError

    def wavefunction1d_defaults(self, mode, evals, wavefunc_count):
        raise NotImplementedError

    def plot_wavefunction(self, which=0,  mode='real', esys=None, phi_grid=None, scaling=None, **kwargs):
        """Plot 1d phase-basis wave function(s). Must be overwritten by higher-dimensional qubits like FluxQubits and
        ZeroPi.

        Parameters
        ----------
        esys: (ndarray, ndarray), optional
            eigenvalues, eigenvectors
        which: int or tuple or list, optional
            single index or tuple/list of integers indexing the wave function(s) to be plotted.
            If which is -1, all wavefunctions up to the truncation limit are plotted.
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        scaling: float or None, optional
            custom scaling of wave function amplitude/modulus
        **kwargs: dict
            standard plotting option (see separate documentation)

        Returns
        -------
        Figure, Axes
        """
        fig_ax = kwargs.get('fig_ax') or plt.subplots()
        kwargs['fig_ax'] = fig_ax

        index_list = process_which(which, self.truncated_dim)
        if esys is None:
            evals_count = max(index_list) + 2
            esys = self.eigensys(evals_count)
        evals, _ = esys

        phi_grid = phi_grid or self._default_grid
        potential_vals = self.potential(phi_grid.make_linspace())

        evals_count = len(index_list)
        if evals_count == 1:
            scale = set_scaling(self, scaling, potential_vals)
        else:
            scale = 0.75 * (evals[-1] - evals[0]) / evals_count

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        kwargs = {**self.wavefunction1d_defaults(mode, evals, wavefunc_count=len(index_list)), **kwargs}
        # in merging the dictionaries in the previous line: if any duplicates, later ones survive
        for wavefunc_index in index_list:
            phi_wavefunc = self.wavefunction(esys, which=wavefunc_index, phi_grid=phi_grid)
            phi_wavefunc.amplitudes = standardize_sign(phi_wavefunc.amplitudes)
            phi_wavefunc.amplitudes = amplitude_modifier(phi_wavefunc.amplitudes)
            plot.wavefunction1d(phi_wavefunc, potential_vals=potential_vals, offset=phi_wavefunc.energy,
                                scaling=scale, **kwargs)
        return fig_ax
