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
import warnings

import matplotlib.pyplot as plt
import numpy as np

from scqubits.utils.spectrum_utils import closest_dressed_energy

try:
    import lmfit
except ImportError:
    _HAS_LMFIT = False
else:
    _HAS_LMFIT = True

import scqubits
import scqubits.core.qubit_base as qubit_base
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.core.param_sweep as param_sweep
import scqubits.settings as settings
import scqubits.utils.sweep_plotting as splot
import scqubits.utils.misc as utils

if settings.IN_IPYTHON:
    from IPython.core.display import clear_output, display, HTML


# tagging types (to facilitate file io: do not use Enum)
NO_TAG = 'NO_TAG'
DISPERSIVE_DRESSED = 'DISPERSIVE_DRESSED'
DISPERSIVE_BARE = 'DISPERSIVE_BARE'
CROSSING = 'CROSSING'
CROSSING_DRESSED = 'CROSSING_DRESSED'


class Tag(serializers.Serializable):
    """
    Store a single dataset tag. The tag can be of different types:
    - NO_TAG: user did not tag data
    - DISPERSIVE_DRESSED: transition between two states in the dispersive regime, tagged by dressed-states indices
    - DISPERSIVE_BARE: : transition between two states in the dispersive regime, tagged by bare-states indices
    - CROSSING: avoided crossing, left untagged (fitting should use closest-energy states)
    - CROSSING_DRESSED: avoided crossing, tagged by dressed-states indices

    Parameters
    ----------
    tagType: str
        one of the tag types listed above
    initial, final: int, or tuple of int, or None
        - For NO_TAG and CROSSING, no initial and final state are specified.
        - For DISPERSIVE_DRESSED and CROSSING_DRESSED, initial and final state are specified by an int dressed index.
        - FOR DISPERSIVE_BARE, initial and final state are specified by a tuple of ints (exc. levels of each subsys)
    photons: int or None
        - For NO_TAG, no photon number is specified.
        - For all other tag types, this int specifies the photon number rank of the transition.
    subsysList: list of str
        list of subsystem names
    """
    def __init__(self, tagType=NO_TAG, initial=None, final=None, photons=None, subsysList=None):
        self.tag_type = tagType  # PySide2/datapyc use camelCase... revert here to align with PEP8
        self.initial = initial
        self.final = final
        self.photons = photons
        self.subsys_list = subsysList


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

    def calibrate_fitdataset(self, array):
        array = array.transpose()
        return np.apply_along_axis(self.calibrate_datapoint, axis=0, arr=array).transpose()

    def calibrate_datapoint(self, rawvec):
        if isinstance(rawvec, list):
            rawvec = np.asarray(rawvec)
        mvec = np.matmul(self.alpha_mat, rawvec) + self.b_vec
        return mvec


scqubits.io_utils.fileio_serializers.SERIALIZABLE_REGISTRY['CalibrationModel'] = CalibrationModel


class FitData(serializers.Serializable):
    def __init__(self, datanames, datalist,
                 z_data=None, x_data=None, y_data=None, image_data=None,
                 calibration_data=None, tag_data=None, fit_results=None):
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
            measurement z data (required unless measurement data given as `image_data`
        x_data: ndarray, optional
            array specifying x axis values
        y_data: ndarray, optional
            array specifying y axis values
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
        self.transition_tags = tag_data
        self.fit_results = fit_results

        self.calibrated_datalist = [self.calibration_data.calibrate_fitdataset(dataset) for dataset in self.datalist]

        self.system = None
        self.subsys_names = None
        self.sweep_name = None
        self.sweep_vals = None
        self.sweep_update_func = None
        self.subsys_update_list = None
        self.lmfit_params = None if not _HAS_LMFIT else lmfit.Parameters()
        self.fit_params = {}
        self.evals_count = None
        self.extra_params = None
        self.sweep = None

    def setup(self, system, sweep_name, update_func, evals_count, **extra_params):
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
        evals_count: int
            number of (dressed) eigenenergies to be calculated
        """
        self.sweep_name = sweep_name
        self.sweep_update_func = update_func
        self.evals_count = evals_count
        self.extra_params = extra_params
        self.lmfit_params = lmfit.Parameters()

        self._set_system(system)
        self._register_subsys_names()
        self._register_fit_params()
        self._setup_sweepvals()

    def _set_system(self, sys_object):
        """Set the `.system` attribute to record the given `sys_object`. If not provided as a HilbertSpace object, then
        it is wrapped into a HilbertSpace instance

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
        for tag in self.transition_tags:
            if tag.tag_type == DISPERSIVE_BARE:
                self.subsys_names = tag.subsys_list
                break
        if self.subsys_names is not None:
            return

        name_list = [type(subsys).__name__ for subsys in self.system]
        self.subsys_names = []
        for index, name in enumerate(name_list):
            name_count = name_list[0:index].count(name)
            self.subsys_names.append(name + str(name_count))

    def _register_fit_params(self):
        """Call the `.fit_params()` method for each HilbertSpace subsystem and InteractionTerm to obtain the list
        of possible fit parameters. (Any of then can be frozen via FitData.lmfit_params.add(<name>, vary=False.)"""
        if self.system is None:
            raise Exception('FitData Error: system must be specified with `<FitData>.setup(..)` before setting fit '
                            'parameters!')
        for index, subsys in enumerate(self.system):
            fitparams = subsys.fit_params()
            for name in fitparams:
                long_name = name + '_' + str(index)
                self.fit_params[long_name] = (index, name)
                self.lmfit_params.add(long_name, value=getattr(self.system[index], name), min=0.0)
        for index, interaction in enumerate(self.system.interaction_list):
            name = 'g_strength'
            long_name = name + '_' + str(index)
            self.fit_params[long_name] = (index, name)
            self.lmfit_params.add(long_name, value=getattr(self.system.interaction_list[index], name))
        for name, value in self.extra_params.items():
            self.fit_params[name] = (None, name)
            self.lmfit_params.add(name, value=value)

    def _setup_sweepvals(self):
        """Compose the array of sweep values from the union of all x values of extracted data points."""
        sweep_values = [dataset.transpose()[0] for dataset in self.calibrated_datalist]
        sweep_values = functools.reduce(np.union1d, sweep_values)
        self.sweep_vals = sweep_values

    def change_param_values(self, params):
        """Change the parameter values of the HilbertSpace object.

        Parameters
        ----------
        params: dict (str: Number)
            Dictionary of fit parameter values, using "long names" as generated by `_register_fit_params`
        """
        for long_name, value in params.items():
            index, name = self.fit_params[long_name]
            if name == 'g_strength':
                setattr(self.system.interaction_list[index], name, value)
            elif index is None:
                self.extra_params[name] = value
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
    def fit(self, num_cpus=settings.NUM_CPUS, dispersive_only=False, verbose=False, **kwargs):
        minim = lmfit.Minimizer(residuals,
                                self.lmfit_params,
                                fcn_args=(self,),
                                fcn_kws={'dispersive_only': dispersive_only, 'num_cpus': num_cpus},
                                iter_cb=self.iteration_info if verbose else None,
                                nan_policy='raise')
        self.fit_results = minim.minimize(**kwargs)
        return self.fit_results

    def plot(self, progress_info=False, **kwargs):
        fig, (axes1, axes2) = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.set_figwidth(10)

        for dataset in self.calibrated_datalist:
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

        calibrated_x0, calibrated_y0 = self.calibration_data.calibrate_datapoint([xmin, ymin])
        calibrated_x1, calibrated_y1 = self.calibration_data.calibrate_datapoint([xmax, ymax])
        # take into account that calibration may exchange min <--> max
        calibrated_xmin = min(calibrated_x0, calibrated_x1)
        calibrated_xmax = max(calibrated_x0, calibrated_x1)
        calibrated_ymin = min(calibrated_y0, calibrated_y1)
        calibrated_ymax = max(calibrated_y0, calibrated_y1)

        if self.image_data is not None:
            axes1.imshow(self.image_data, aspect='auto',
                         extent=[calibrated_xmin, calibrated_xmax, calibrated_ymin, calibrated_ymax], **kwargs)
        else:
            axes1.imshow(self.z_data, aspect='auto', origin='lower',
                         extent=[calibrated_xmin, calibrated_xmax, calibrated_ymin, calibrated_ymax], **kwargs)

        if self.fit_results is not None:
            self.change_param_values(self.fit_results.params.valuesdict())
            res_xvals = np.concatenate([dataset.transpose()[0] for dataset in self.calibrated_datalist])
            width = (np.max(res_xvals) - np.min(res_xvals)) / 70.0
            axes2.bar(res_xvals, self.fit_results.residual, width=width)
            axes2.set_ylabel('residuals')

            param_vals = np.linspace(-0.5, 0.5, 200)
            self.sweep.param_vals = param_vals
            self.sweep.param_count = len(param_vals)
            self.sweep.run()
            splot.difference_spectrum(self.sweep, fig_ax=(fig, axes1), ylim=(calibrated_ymin, calibrated_ymax))
        else:
            axes2.text(0.5, 0.5, 'residuals: N/A  (run fit)',
                       horizontalalignment='center', verticalalignment='center', transform=axes2.transAxes)

        return fig, (axes1, axes2)

    @utils.Required(IPython=settings.IN_IPYTHON)
    def iteration_info(self, params, iteration, resid, *args, **kwargs):
        """Function called at each fit iteration.
        See https://lmfit.github.io/lmfit-py/fitting.html#using-the-minimizer-class

        Parameters
        ----------
        params: lmfit.Parameters
            current parameter values
        iteration: int
            current iteration number
        resid: array
            the current residual array
        *args, **kwargs:
            as passed to the objective function.
        """
        if iteration % 5 == 0:
            clear_output(wait=True)
            display(params)
            display(HTML("iteration: {}, squared deviations: {}".format(iteration, np.dot(resid, resid))))


def residuals(params, fitdata, dispersive_only=False, num_cpus=settings.NUM_CPUS):
    fitdata.change_param_values(params)
    fitdata.new_sweep(num_cpus=num_cpus)
    residuals_array = []
    for dataset, transition_tag in zip(fitdata.calibrated_datalist, fitdata.transition_tags):
        if dispersive_only and (transition_tag.tag_type in [CROSSING, CROSSING_DRESSED, NO_TAG]):
            continue

        for point in dataset:
            initial = transition_tag.initial
            final = transition_tag.final
            nphoton = transition_tag.photons
            index = np.where(np.isclose(fitdata.sweep_vals, point[0]))[0][0]

            if transition_tag.tag_type in [CROSSING, NO_TAG]:
                # Get residuals as differences with respect to the closest transition energy E_j - E_0
                # i.e., only 1-photon transitions starting in ground state
                energies = fitdata.sweep.lookup.dressed_eigenenergies(param_index=index)
                transition_energy_list = (energies - energies[0])[1:]
                min_index = (np.abs(transition_energy_list - point[1])).argmin()
                transition_energy = transition_energy_list[min_index]
            elif transition_tag.tag_type in [DISPERSIVE_DRESSED, CROSSING_DRESSED]:
                transition_energy = (
                    fitdata.sweep.lookup.energy_dressed_index(dressed_index=final, param_index=index)
                    - fitdata.sweep.lookup.energy_dressed_index(dressed_index=initial, param_index=index)
                ) / nphoton
            elif transition_tag.tag_type == DISPERSIVE_BARE:
                energy_initial = fitdata.sweep.lookup.energy_bare_index(bare_tuple=initial, param_index=index)
                energy_final = fitdata.sweep.lookup.energy_bare_index(bare_tuple=final, param_index=index)
                if energy_initial is None or energy_final is None:
                    offending_state = initial if energy_initial is None else final
                    warnings.warn("No match for state with bare label {}. Possible causes: state is above "
                                  "the cutoff specified by evals_count; or, state is strongly hybridized, rendering "
                                  "identification through bare labels inappropriate. "
                                  "Skipping...".format(offending_state))
                    residuals_array.append(0.0)
                    continue
                transition_energy = (
                    fitdata.sweep.lookup.energy_bare_index(bare_tuple=final, param_index=index)
                    - fitdata.sweep.lookup.energy_bare_index(bare_tuple=initial, param_index=index)
                ) / nphoton
            residuals_array.append(transition_energy - point[1])
    return residuals_array
