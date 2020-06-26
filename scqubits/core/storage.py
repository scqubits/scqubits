# storage.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot


# —WaveFunction class———————————————————————————————————————————————————————————————————————————————————————————————————

class WaveFunction:
    """Container for wave function amplitudes defined for a specific basis. Optionally, a corresponding
    energy is saved as well.

    Parameters
    ----------
    basis_labels: ndarray
        labels of basis states; for example, in position basis: values of position variable
    amplitudes: ndarray
        wave function amplitudes for each basis label value
    energy: float, optional
        energy of the wave function
    """
    def __init__(self, basis_labels, amplitudes, energy=None):
        self.basis_labels = basis_labels
        self.amplitudes = amplitudes
        self.energy = energy


# —WaveFunctionOnGrid class—————————————————————————————————————————————————————————————————————————————————————————————

class WaveFunctionOnGrid:
    """Container for wave function amplitudes defined on a coordinate grid (arbitrary dimensions).
    Optionally, a corresponding eigenenergy is saved as well.

    Parameters
    ----------
    gridspec: GridSpec object
        grid specifications for the stored wave function
    amplitudes: ndarray
        wave function amplitudes on each grid point
    energy: float, optional
        energy corresponding to the wave function
    """
    def __init__(self, gridspec, amplitudes, energy=None):
        self.gridspec = gridspec
        self.amplitudes = amplitudes
        self.energy = energy


# —BaseData class———————————————————————————————————————————————————————————————————————————————————————————————————


class DataStore(serializers.Serializable):
    """Base class for storing and processing spectral data and custom data from parameter sweeps.

    Parameters
    ----------
    param_name: str
        name of parameter being varies
    param_vals: ndarray
        parameter values for which spectrum data are stored
    system_params: dict
        info about system parameters

    **kwargs:
        keyword arguments for data to be stored: ``dataname=data``, where data should be an array-like object
    """
    def __init__(self, system_params, param_name='', param_vals=None, **kwargs):
        self.system_params = system_params
        self.param_name = param_name
        self.param_vals = param_vals
        if param_vals is not None:
            self.param_count = len(self.param_vals)
        else:
            self.param_count = 1   # just one value if there is no parameter sweep

        self._datanames = []  # stores names of additional datasets
        for dataname, data in kwargs.items():
            setattr(self, dataname, data)
            self._datanames.append(dataname)
            self._init_params.append(dataname)  # register additional dataset for file IO

    def add_data(self, **kwargs):
        """
        Adds one or several data sets to the DataStorage object.

        Parameters
        ----------
        **kwargs:
            ``dataname=data`` with ``data`` an array-like object. The data set will be accessible through
            ``<DataStorage>.dataname``.
        """
        for dataname, data in kwargs.items():
            setattr(self, dataname, data)
            self._datanames.append(dataname)
            self._init_params.append(dataname)   # register additional dataset for file IO


# —SpectrumData class———————————————————————————————————————————————————————————————————————————————————————————————————

class SpectrumData(DataStore):
    """Container holding energy and state data as a function of a particular parameter that is varied.
    Also stores all other system parameters used for generating the set, and provides method for writing
    data to file.

    Parameters
    ----------
    param_name: str
        name of parameter being varies
    param_vals: ndarray
        parameter values for which spectrum data are stored
    energy_table: ndarray
        energy eigenvalues stored for each `param_vals` point
    system_params: dict
        info about system parameters
    state_table: ndarray or list, optional
        eigenstate data stored for each `param_vals` point, either as pure ndarray or list of qutip.qobj
    matrixelem_table: ndarray, optional
        matrix element data stored for each `param_vals` point
    """
    # mark for file serializers purposes:
    def __init__(self, energy_table, system_params, param_name=None, param_vals=None, state_table=None,
                 matrixelem_table=None, **kwargs):
        super().__init__(system_params=system_params, param_name=param_name, param_vals=param_vals,
                         energy_table=energy_table, state_table=state_table, matrixelem_table=matrixelem_table,
                         **kwargs)

    def subtract_ground(self):
        """Subtract ground state energies from spectrum"""
        self.energy_table -= self.energy_table[:, 0]

    def plot_evals_vs_paramvals(self, which=-1, subtract_ground=False, label_list=None, **kwargs):
        """Plots eigenvalues of as a function of one parameter, as stored in SpectrumData object.

        Parameters
        ----------
        which: int or list(int)
            default: -1, signals to plot all eigenvalues; int>0: plot eigenvalues 0..int-1; list(int) plot the specific
            eigenvalues (indices listed)
        subtract_ground: bool, optional
            whether to subtract the ground state energy, default: False
        label_list: list(str), optional
            list of labels associated with the individual curves to be plotted
        **kwargs: dict
            standard plotting option (see separate documentation)

        Returns
        -------
        Figure, Axes
        """
        return plot.evals_vs_paramvals(self, which=which, subtract_ground=subtract_ground,
                                       label_list=label_list, **kwargs)
