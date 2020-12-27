# storage.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import Any, Dict, List, TYPE_CHECKING, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
from scqubits.io_utils.fileio_qutip import QutipEigenstates

if TYPE_CHECKING:
    from scqubits.core.discretization import GridSpec


# —WaveFunction class———————————————————————————————————————————————————————————————————————————————————————————————————

class WaveFunction:
    """Container for wave function amplitudes defined for a specific basis. Optionally, a corresponding
    energy is saved as well.

    Parameters
    ----------
    basis_labels:
        labels of basis states; for example, in position basis: values of position variable
    amplitudes:
        wave function amplitudes for each basis label value
    energy:
        energy of the wave function
    """
    def __init__(self,
                 basis_labels: ndarray,
                 amplitudes: ndarray,
                 energy: float = None
                 ) -> None:
        self.basis_labels = basis_labels
        self.amplitudes = amplitudes
        self.energy = energy


# —WaveFunctionOnGrid class—————————————————————————————————————————————————————————————————————————————————————————————

class WaveFunctionOnGrid:
    """Container for wave function amplitudes defined on a coordinate grid (arbitrary dimensions).
    Optionally, a corresponding eigenenergy is saved as well.

    Parameters
    ----------
    gridspec: GridSpec
        grid specifications for the stored wave function
    amplitudes:
        wave function amplitudes on each grid point
    energy:
        energy corresponding to the wave function
    """
    def __init__(self,
                 gridspec: 'GridSpec',
                 amplitudes: ndarray,
                 energy: float = None
                 ) -> None:
        self.gridspec = gridspec
        self.amplitudes = amplitudes
        self.energy = energy


# —BaseData class———————————————————————————————————————————————————————————————————————————————————————————————————


class DataStore(serializers.Serializable):
    """Base class for storing and processing spectral data and custom data from parameter sweeps.

    Parameters
    ----------
    system_params:
        info about system parameters
    param_name:
        name of parameter being varies
    param_vals:
        parameter values for which spectrum data are stored
    **kwargs:
        keyword arguments for data to be stored: ``dataname=data``, where data should be an array-like object
    """
    def __init__(self,
                 system_params: Dict[str, Any],
                 param_name: str = None,
                 param_vals: ndarray = None,
                 **kwargs
                 ) -> None:
        self.system_params = system_params
        self.param_name = param_name
        self.param_vals = param_vals
        if isinstance(param_vals, ndarray):
            self.param_count = len(self.param_vals)  # type: ignore
        else:
            self.param_count = 1   # just one value if there is no parameter sweep

        self._datanames = []  # stores names of additional datasets
        for dataname, data in kwargs.items():
            setattr(self, dataname, data)
            self._datanames.append(dataname)
            self._init_params.append(dataname)  # register additional dataset for file IO

    def add_data(self, **kwargs) -> None:
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
    energy_table:
        energy eigenvalues stored for each `param_vals` point,
        [[evals for first param_val], [evals for second param_val], ...]
    system_params:
        info about system parameters
    param_name:
        name of parameter being varies
    param_vals:
        parameter values for which spectrum data are stored
    state_table: Union[List[QutipEigenstates], ndarray, List[ndarray]]
        eigenstate data stored for each `param_vals` point, either as pure ndarray or list of qutip.qobj
    matrixelem_table:
        matrix element data stored for each `param_vals` point
    """
    # mark for file serializers purposes:
    def __init__(self,
                 energy_table: ndarray,
                 system_params: Dict[str, Any],
                 param_name: str = None,
                 param_vals: ndarray = None,
                 state_table: Union[List[QutipEigenstates], ndarray, List[ndarray]] = None,
                 matrixelem_table: ndarray = None,
                 **kwargs
                 ) -> None:
        self.system_params = system_params
        self.param_name = param_name
        self.param_vals = param_vals
        self.energy_table = energy_table
        self.state_table = state_table
        self.matrixelem_table: ndarray = matrixelem_table
        super().__init__(system_params=system_params,
                         param_name=param_name,
                         param_vals=param_vals,
                         energy_table=energy_table,
                         state_table=state_table,
                         matrixelem_table=matrixelem_table,
                         **kwargs)

    def subtract_ground(self) -> None:
        """Subtract ground state energies from spectrum"""
        self.energy_table -= self.energy_table[:, 0]

    def plot_evals_vs_paramvals(self,
                                which: Union[int, List[int]] = -1,
                                subtract_ground: bool = False,
                                label_list: List[str] = None,
                                **kwargs
                                ) -> 'Tuple[Figure, Axes]':
        """Plots eigenvalues of as a function of one parameter, as stored in SpectrumData object.

        Parameters
        ----------
        which:
            default: -1, signals to plot all eigenvalues; int>0: plot eigenvalues 0..int-1; list(int) plot the specific
            eigenvalues (indices listed)
        subtract_ground:
            whether to subtract the ground state energy, default: False
        label_list:
            list of labels associated with the individual curves to be plotted
        **kwargs:
            standard plotting option (see separate documentation)

        Returns
        -------
            Figure and Axes objects for further processing
        """
        return plot.evals_vs_paramvals(self, which=which, subtract_ground=subtract_ground,
                                       label_list=label_list, **kwargs)
