# explorer.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import copy
import numpy as np
import matplotlib.pyplot as plt

try:
    import ipywidgets
except ImportError:
    raise Exception("ImportError: failed to import ipywidgets. For use of scqubits.explorer,"
                    "ipywidgets must be installed")


class Explorer:
    """
    This class allows interactive exploration of coupled quantum systems. The generate() method pre-calculates spectral
    data as a function of a given parameter, which can then be displayed and modified by sliders (when inside jupyter
    notebook or jupyter lab).

    Parameters
    ----------
    sweep: HilbertSpaceSweep
    evals_count: int
    figsize: tuple(int,int), optional
    """
    def __init__(self, sweep, evals_count, figsize=(10, 7)):
        self.param_name = sweep.param_name
        self.param_vals = sweep.param_vals
        self.param_count = sweep.param_count
        self.sweep = sweep
        self.evals_count = evals_count
        self.figsize = figsize

    def plot_bare_wavefunction(self, param_val, subsys, which=-1, phi_count=None, title=None, fig_ax=None):
        """
        Plot bare wavefunctions for given parameter value and subsystem.

        Parameters
        ----------
        param_val: float
            value of the external parameter
        subsys: QuantumSystem
        which: int or list(int)
            default: -1, signals to plot all wavefunctions; int>0: plot wavefunctions 0..int-1; list(int) plot specific
            wavefunctions
        phi_count: None or int
        title: str, optional
        fig_ax: Figure, Axes

        Returns
        -------
        fig, axes
        """
        subsys_index = self.sweep.hilbertspace.index(subsys)
        new_hilbertspace = copy.deepcopy(self.sweep.hilbertspace)
        self.sweep.update_hilbertspace(param_val, new_hilbertspace)
        subsys = new_hilbertspace[subsys_index]

        param_index = np.searchsorted(self.param_vals, param_val)

        evals = self.sweep.bare_specdata_list[subsys_index].energy_table[param_index]
        evecs = self.sweep.bare_specdata_list[subsys_index].state_table[param_index]
        return subsys.plot_wavefunction(esys=(evals, evecs), which=which, mode='real', phi_count=phi_count,
                                        title=title, fig_ax=fig_ax)

    def transition_explorer_plot(self, param_val, photonnumber, initial_index, final_index):
        param_index = np.searchsorted(self.param_vals, param_val)
        param_val = self.param_vals[param_index]
        bare_initial = self.sweep.hilbertspace.get_bare_index(initial_index, param_index)
        bare_final = self.sweep.hilbertspace.get_bare_index(final_index, param_index)
        energy_ground = self.sweep.get_energy_dressed_index(0, param_index)
        energy_initial = self.sweep.get_energy_dressed_index(initial_index, param_index) - energy_ground
        energy_final = self.sweep.get_energy_dressed_index(final_index, param_index) - energy_ground
        energy_difference = energy_final - energy_initial

        dynamic_subsys_count = len(self.sweep.subsys_update_list)
        ncols = 2
        nrows = dynamic_subsys_count + 1

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=self.figsize)

        axes_list_flattened = [elem for sublist in axs for elem in sublist]
        index = 0
        for subsys in self.sweep.subsys_update_list:
            title = 'bare spectrum: subsystem {} ({})'.format(self.sweep.hilbertspace.index(subsys), subsys._sys_type)
            __ = self.sweep.plot_bare_spectrum(subsys, title=title, fig_ax=(fig, axes_list_flattened[index]))
            axes_list_flattened[index].axvline(param_val, color='gray', linestyle=':')
            index += 1

            title = 'wavefunctions: subsystem {} ({})'.format(self.sweep.hilbertspace.index(subsys), subsys._sys_type)
            __ = self.plot_bare_wavefunction(param_val, subsys, title=title, fig_ax=(fig, axes_list_flattened[index]))
            index += 1

        title = r'{} $\rightarrow$ {}: {:.4f} GHz'.format(bare_initial, bare_final, energy_difference)
        __ = self.sweep.plot_dressed_spectrum(title=title, fig_ax=(fig, axes_list_flattened[index]))
        axes_list_flattened[index].axvline(param_val,  color='gray', linestyle=':')
        axes_list_flattened[index].scatter([param_val] * 2, [energy_initial, energy_final], s=40, c='black')
        index += 1

        title = '{}-photon qubit transitions'.format(photonnumber)
        __ = self.sweep.plot_n_photon_qubit_spectrum(photonnumber, self.sweep.hilbertspace[1],
                                                     initial_state_ind=initial_index,
                                                     title=title,
                                                     fig_ax=(fig, axes_list_flattened[index]))
        axes_list_flattened[index].axvline(param_val, color='gray', linestyle=':')
        axes_list_flattened[index].scatter([param_val], [(energy_final - energy_initial)/photonnumber], s=40, c='black')
        fig.tight_layout()
        return fig, axs

    def explore_transitions(self):
        param_min = self.param_vals[0]
        param_max = self.param_vals[-1]
        param_step = self.param_vals[1] - self.param_vals[0]

        param_slider = ipywidgets.FloatSlider(min=param_min, max=param_max, step=param_step,
                                              description=self.param_name, continuous_update=False)
        interactive_plot = ipywidgets.interactive(self.transition_explorer_plot,
                                                  param_val=param_slider,
                                                  photonnumber=[1, 2],
                                                  initial_index=range(self.evals_count),
                                                  final_index=range(self.evals_count))
        return interactive_plot
