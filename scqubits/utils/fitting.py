# fitting.py
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

import numpy as np
import matplotlib.pyplot as plt

import scqubits.utils.misc as utils

try:
    import lmfit
except ImportError:
    _HAS_LMFIT = False
else:
    _HAS_LMFIT = True


import scqubits
import scqubits.core.qubit_base as qubit_base
import scqubits.io.file_io_serializers as serializers
import scqubits.core.param_sweep as param_sweep
import scqubits.settings as settings
import scqubits.utils.sweep_plotting as splot


class CalibrationModel(serializers.Serializable):
    def __init__(self, rawVec1=None, rawVec2=None, mapVec1=None, mapVec2=None):
        """
        Store calibration data for x and y axes, and provide methods to transform between uncalibrated and calibrated
        data.

        Parameters
        ----------
        rawVec1, rawVec2, mapVec1, mapVec2: ndarray
            Each of these is a two component vector (x,y) marking a point. The calibration maps raw_vec1 -> map_vec1,
            raw_vec2 -> map_vec2 with an affine-linear transformation:   mapVecN = alpha_mat . rawVecN + b_vec.
        """
        self.raw_vec1 = rawVec1
        self.raw_vec2 = rawVec2
        self.map_vec1 = mapVec1
        self.map_vec2 = mapVec2
        self.b_vec = None
        self.alpha_mat = None

        if rawVec1 and rawVec2 and mapVec1 and mapVec2:
            self.set_calibration(rawVec1, rawVec2, mapVec1, mapVec2)
        else:
            self.set_calibration((1., 0.), (0., 1.), (1., 0.), (0., 1.))

        self.apply_calibration = False

    def set_calibration(self, rvec1, rvec2, mvec1, mvec2):
        x1, y1 = rvec1
        x2, y2 = rvec2
        x1p, y1p = mvec1
        x2p, y2p = mvec2

        alphaX = (x1p - x2p) / (x1 - x2)
        alphaY = (y1p - y2p) / (y1 - y2)

        self.b_vec = np.asarray([x1p - alphaX * x1, y1p - alphaY * y1])
        self.alpha_mat = np.asarray([[alphaX, 0.], [0., alphaY]])
        self.raw_vec1, self.raw_vec2, self.map_vec1, self.map_vec2 = rvec1, rvec2, mvec1, mvec2

    def calibrate_dataset(self, array):
        return np.apply_along_axis(self.calibrate_datapoint, axis=0, arr=array)

    def calibrate_datapoint(self, rawvec):
        if isinstance(rawvec, list):
            rawvec = np.asarray(rawvec)
        mvec = np.matmul(self.alpha_mat, rawvec) + self.b_vec
        return mvec


scqubits.io.file_io_serializers.SERIALIZABLE_REGISTRY['CalibrationModel'] = CalibrationModel


class FitData(serializers.Serializable):
    def __init__(self, datanames, datalist,
                 z_data=None, x_data=None, y_data=None, image_data=None, calibration_data=None, fit_results=None):
        """
        Class for fitting experimental spectroscopy data to the Hamiltonian model of the qubit / coupled quantum system.

        Parameters
        ----------
        datanames: list of str
        datalist: list of ndarray
            Each ndarray has float entries and is of the form array([[x1,y1], [x2,y2], ...]). Each such set corresponds
            to data extracted from experimental spectroscopy data. Each corresponds to one particular transition among
            dressed-level eigenstates.
        z_data: ndarray
        x_data: ndarray
        y_data: ndarray
        image_data: ndarray
            as obtained with matplotlib.image.imread
        calibration_data: datapyc.CalibrationModel
        """
        self.datanames = datanames
        self.datalist = datalist
        self.x_data = x_data
        self.y_data = y_data
        self.z_data = z_data
        self.image_data = image_data
        self.calibration_data = calibration_data
        self.fit_results = fit_results

        self.system = None
        self.subsys_names = None
        self.sweep_name = None
        self.sweep_vals = None
        self.sweep_update_func = None
        self.subsys_update_list = None
        self.params = None if not _HAS_LMFIT else lmfit.Parameters()
        self.fit_params = {}
        self.evals_count = None
        self.sweep = None
        self.transitions = None

    def setup(self, system, sweep_name, update_func, transitions):
        """
        While the list of datasets and their names are provided at initialization (or when reading from a
        datapyc file), additional initialization is carried out through the `setup` method. Use of this method records
        * the `HilbertSpace` object underlying the fit data,
        * the sweep parameter,
        * the function that specifies the sweep by updating parameters in the HilbertSpace components
        * the transition belonging to each fit dataset

        Parameters
        ----------
        system: HilbertSpace or QuantumSystem
            The `systsm` sets the model underlying the experimental data provided as `datalist`. The parameter entries
            provided with HilbertSpace act as initial guess values in the fitting procedure. If only a single qubit
            instance is provided, it will be wrapped into a HilbertSpace object.
        sweep_name: str
            Name of the sweep parameter
        update_func: function
            Function specifying how a change in the sweep parameter translates into a change of parameters entering the
            HilbertSpace object.
        transitions: list of tuples of int
            Each list entry can have one of three forms.
            1. int: an integer entry is interpreted as denoting a single-
                photon transition from the ground state the the excited state specified by the integer.
            2. tuple(int, int):
                a tuple of two integers i1, i2 denotes a single photon-transition from i1 to i2.
            3. tuple(int, int, int):
                The three integers i1, i2, n specify an n-photon transition among levels i1 and i2.

        """
        self.transitions = process_t(transitions)
        self.evals_count = self.max_level() + 1
        self.sweep_name = sweep_name
        self.sweep_update_func = update_func

        self._set_system(system)
        self._register_subsys_names()
        self._register_fit_params()
        self._setup_sweepvals()

    def _set_system(self, sys_object):
        """Set the `.system` attribute to record the given `sys_object`. If not provided as a HilbertSpace object, then
        it is wraped into a HilbertSpace instance

        Parameters
        -----------
        sys_object: HilbertSpace or QuantumSystem
        """
        if isinstance(sys_object, scqubits.HilbertSpace):
            self.system = sys_object
        elif isinstance(sys_object, qubit_base.QubitBaseClass):
            sys_object.truncated_dim = sys_object.truncated_dim or self.evals_count
            self.system = scqubits.HilbertSpace([sys_object])
            self.system.interaction_list = []
            self.subsys_update_list = [sys_object]

    def _register_subsys_names(self):
        """Record the type names of the individual subsystems."""
        name_list = [type(subsys).__name__ for subsys in self.system]
        self.subsys_names = []
        for index, name in enumerate(name_list):
            name_count = name_list[0:index].count(name)
            self.subsys_names.append(name + str(name_count))

    def _register_fit_params(self):
        """Call the `.fit_params()` method for each HilbertSpace subsystem and the InteractionTerms to obtain the list
        of possible fit parameters. (Any of then can be frozen via FitData.params.add(<name>, vary=False.)"""
        if self.system is None:
            raise Exception('FitData Error: system must be specified before setting fit parameters!')
        for index, subsys in enumerate(self.system):
            fitparams = subsys.fit_params()
            for name in fitparams:
                long_name = name + '_' + str(index)
                self.fit_params[long_name] = (index, name)
                self.params.add(long_name, value=getattr(self.system[index], name), min=0.0)
        for index, interaction in self.system.interaction_list:
            name = 'g_strength'
            long_name = name + '_' + str(index)
            self.fit_params[long_name] = (index, name)
            self.params.add(long_name, value=getattr(self.system.interaction_list[index], name))

    def _setup_sweepvals(self):
        """Compose the array of sweep values from the union of all x values of extracted data points."""
        sweep_values = [dataset.transpose()[0] for dataset in self.datalist]
        sweep_values = functools.reduce(np.union1d, sweep_values)
        self.sweep_vals = sweep_values

    def max_level(self):
        """Determine the maximum energy level required according to the specified transitions."""
        return np.max(np.union1d(self.transitions[:, 0], self.transitions[:, 1]))

    def change_param_values(self, params):
        """Change the parameter values of the HilbertSpace object.

        Parameters
        ----------
        params: dict (str: Number)
        """
        for long_name, value in params.items():
            index, name = self.fit_params[long_name]
            if name == 'g_strength':
                setattr(self.system.interaction_list[index], name, value)
            else:
                setattr(self.system[index], name, value)

    def new_sweep(self, num_cpus=settings.NUM_CPUS):
        settings.PROGRESSBAR_DISABLED = True
        self.sweep = param_sweep.ParameterSweep(
            param_name=self.sweep_name,
            param_vals=self.sweep_vals,
            evals_count=self.evals_count,
            hilbertspace=self.system,
            subsys_update_list=self.system,
            update_hilbertspace=self.sweep_update_func,
            num_cpus=num_cpus
        )
        settings.PROGRESSBAR_DISABLED = False

    @utils.Required(lmfit=_HAS_LMFIT)
    def fit(self, num_cpus=settings.NUM_CPUS, **kwargs):
        minim = lmfit.Minimizer(residuals, self.params, fcn_args=(self, num_cpus))
        self.fit_results = minim.minimize(**kwargs)
        return self.fit_results

    def plot(self, **kwargs):
        self.change_param_values(self.fit_results.params.valuesdict())

        fig, (axes1, axes2) = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.set_figwidth(10)

        res_xvals = np.concatenate([dataset.transpose()[0] for dataset in self.datalist])
        width = (np.max(res_xvals) - np.min(res_xvals)) / 70.0
        axes2.bar(res_xvals, self.fit_results.residual, width=width)
        axes2.set_ylabel('residuals')

        for dataset in self.datalist:
            xvals, yvals = dataset.transpose()
            axes1.scatter(xvals, yvals, c='k', marker='x')

        if (self.image_data is not None) or (self.x_data is None) or (self.y_data is None):
            xmin = 0
            ymin = 0
            ymax = len(self.image_data)
            xmax = len(self.image_data[0])
        else:
            xmin = self.x_data[0]
            xmax = self.x_data[-1]
            ymin = self.y_data[0]
            ymax = self.y_data[-1]

        calibrated_xmin, calibrated_ymin = self.calibration_data.calibrate_datapoint([xmin, ymin])
        calibrated_xmax, calibrated_ymax = self.calibration_data.calibrate_datapoint([xmax, ymax])

        if self.image_data is not None:
            axes1.imshow(self.z_data, aspect='auto',
                         extent=[calibrated_xmin, calibrated_xmax, calibrated_ymax, calibrated_ymin], **kwargs)
        else:
            axes1.imshow(self.z_data, aspect='auto', origin='lower',
                         extent=[calibrated_xmin, calibrated_xmax, calibrated_ymin, calibrated_ymax], **kwargs)

        splot.difference_spectrum(self.sweep, fig_ax=(fig, axes1), ylim=(calibrated_ymin, calibrated_ymax))

        return fig, (axes1, axes2)


def residuals(params, fitdata, num_cpus=settings.NUM_CPUS):
    fitdata.change_param_values(params)
    fitdata.new_sweep(num_cpus=num_cpus)
    resids = []
    for data, transition in zip(fitdata.datalist, fitdata.transitions):
        for point in data:
            initial = transition[0]
            final = transition[1]
            nphoton = transition[2]
            index = np.where(np.isclose(fitdata.sweep_vals, point[0]))[0][0]
            transition_energy = (fitdata.sweep.lookup.energy_dressed_index(dressed_index=final, param_index=index)
                                 - fitdata.sweep.lookup.energy_dressed_index(dressed_index=initial, param_index=index)
                                 ) / nphoton
            resids.append(transition_energy - point[1])
    return resids


def process_t(transitions):
    new = []
    for item in transitions:
        if isinstance(item, int):
            new.append((0, item, 1))
        elif len(item) == 2:
            new.append(item + (1,))
        else:
            new.append(item)
    return np.asarray(new)
