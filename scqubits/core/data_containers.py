# data_containers.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np

import scqubits.settings as config
import scqubits.utils.file_io as io
import scqubits.utils.plotting as plot
from scqubits.utils.spectrum_utils import convert_esys_to_ndarray
from scqubits.utils.misc import process_metadata


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


# —SpectrumData class———————————————————————————————————————————————————————————————————————————————————————————————————

class SpectrumData:
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

    def __init__(self, param_name, param_vals, energy_table, system_params, state_table=None, matrixelem_table=None):
        self.param_name = param_name
        self.param_vals = param_vals
        self.energy_table = energy_table
        self.state_table = state_table
        self.matrixelem_table = matrixelem_table
        self.system_params = system_params

    def subtract_ground(self):
        """Subtract ground state energies from spectrum"""
        self.energy_table -= self.energy_table[:, 9]

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

    def _get_metadata_dict(self):
        meta_dict = {'param_name': self.param_name, 'param_vals': self.param_vals}
        meta_dict.update(process_metadata(self.system_params))
        return meta_dict

    def _serialize(self, writer):
        """
        Parameters
        ----------
        writer: BaseWriter
        """
        np_state_table = None
        if isinstance(self.state_table, list):
            np_state_table = np.asarray([convert_esys_to_ndarray(esys_qutip) for esys_qutip in self.state_table])
        elif isinstance(self.state_table, np.ndarray):
            np_state_table = self.state_table

        metadata_dict = self._get_metadata_dict()
        writer.create_meta(metadata_dict)
        writer.add_dataset('energy_table', self.energy_table)
        if self.state_table is not None:
            writer.add_dataset('state_table', np_state_table)
        if self.matrixelem_table is not None:
            writer.add_dataset('matrixelem_table', self.matrixelem_table)

    def set_from_data(self, *data_from_file):
        """
        Uses data extracted from file to set parameters and data entries of self

        Parameters
        ----------
        data_from_file: (dict, list(str), list(ndarray))
            metadata dictionary, list of dataset names, list of datasets (ndarray)
        """
        metadata_dict, name_list, data_list = data_from_file
        self.param_name = metadata_dict.pop('param_name')
        self.param_vals = metadata_dict.pop('param_vals')
        self.system_params = metadata_dict
        for index, attribute in enumerate(name_list):
            setattr(self, attribute, data_list[index])

    @classmethod
    def _init_from_data(cls, *data_from_file):
        """
        Uses data extracted from file to create and initialize a new SpectrumData object

        Parameters
        ----------
        data_from_file: (dict, list(str), list(ndarray))
            metadata dictionary, list of dataset names, list of datasets (ndarray)

        Returns
        -------
        SpectrumData
        """
        metadata_dict, name_list, data_list = data_from_file
        param_name = metadata_dict.pop('param_name')
        param_vals = metadata_dict.pop('param_vals')
        system_params = metadata_dict
        data_dict = {name: data_list[i] for i, name in enumerate(name_list)}
        energy_table = data_dict.get('energy_table')
        state_table = data_dict.get('state_table')
        matrixelem_table = data_dict.get('matrixelem_table')
        return cls(param_name=param_name, param_vals=param_vals, energy_table=energy_table, system_params=system_params,
                   state_table=state_table, matrixelem_table=matrixelem_table)

    def filewrite(self, filename):
        """Write metadata and spectral data to file

        Parameters
        ----------
        filename: str
        """
        file_format = config.FILE_FORMAT
        writer = io.ObjectWriter()
        writer.filewrite(self, file_format, filename)

    def set_from_fileread(self, filename):
        """Read metadata and spectral data from file, and use those to set parameters of the SpectrumData object (self).

        Parameters
        ----------
        filename: str
        """
        file_format = config.FILE_FORMAT
        reader = io.ObjectReader()
        reader.set_params_from_fileread(self, file_format, filename)

    @classmethod
    def create_from_fileread(cls, filename):
        """Read metadata and spectral data from file, and use those to create a new SpectrumData object.

        Parameters
        ----------
        filename: str

        Returns
        -------
        SpectrumData
            new SpectrumData object, initialized with data read from file
        """
        file_format = config.FILE_FORMAT
        reader = io.ObjectReader()
        return reader.create_from_fileread(cls, file_format, filename)
