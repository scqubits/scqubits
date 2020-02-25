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

import matplotlib.pyplot as plt
import numpy as np

try:
    import ipywidgets
except ImportError:
    _HAS_IPYWIDGETS = False
else:
    _HAS_IPYWIDGETS = True

try:
    from IPython.display import display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

import scqubits.core.sweep_generators as swp
import scqubits.utils.explorer_panels as panels
import scqubits.utils.misc as utils


class Explorer:
    """
    This class allows interactive exploration of coupled quantum systems. The generate() method pre-calculates spectral
    data as a function of a given parameter, which can then be displayed and modified by sliders (when inside jupyter
    notebook or jupyter lab).

    Parameters
    ----------
    sweep: ParameterSweep
    evals_count: int
    figsize: tuple(int,int), optional
    """
    def __init__(self, sweep, evals_count, figsize=(10, 8)):
        self.param_name = sweep.param_name
        self.param_vals = sweep.param_vals
        self.param_count = sweep.param_count
        self.sweep = sweep
        self.evals_count = evals_count
        self.figsize = figsize

        self.chi_data = swp.generate_chi_sweep(sweep)
        self.charge_matelem_data = swp.generate_charge_matrixelem_sweep(sweep)

    def plot_explorer_panels(self, param_val, photonnumber, initial_index, final_index, qbt_index, osc_index):
        """
        Create a panel of plots (bare spectra, bare wavefunctions, dressed spectrum, n-photon qubit transitions, chi).

        Parameters
        ----------
        param_val: float
            current value of the external parameter
        photonnumber: int
            photon number n used for display of n-photon qubit transition
        initial_index: int
            dressed-state index of the initial state used in transition
        final_index: int
            dressed-state index of the final state used in transition (in dressed spectrum display)
        qbt_index: int
            index of qubit subsystem for which matrix elements and chi's are displayed
        osc_index: int
            index of oscillator subsystem for which chi's are calculated

        Returns
        -------
        Figure, Axes: matplotlib.Figure, matplotlib.Axes
        """
        def fig_ax(index):
            return fig, axes_list_flattened[index]

        param_index = np.searchsorted(self.param_vals, param_val)
        param_val = self.param_vals[param_index]

        initial_bare = self.sweep.lookup.bare_index(initial_index, param_index)
        final_bare = self.sweep.lookup.bare_index(final_index, param_index)
        energy_ground = self.sweep.lookup.energy_dressed_index(0, param_index)
        energy_initial = self.sweep.lookup.energy_dressed_index(initial_index, param_index) - energy_ground
        energy_final = self.sweep.lookup.energy_dressed_index(final_index, param_index) - energy_ground
        qbt_subsys = self.sweep.get_subsys(qbt_index)

        row_count = 3
        column_count = 2
        fig, axs = plt.subplots(ncols=column_count, nrows=row_count, figsize=self.figsize)
        axes_list_flattened = [elem for sublist in axs for elem in sublist]

        # Panel 1 ----------------------------------
        panels.display_bare_spectrum(self.sweep, qbt_subsys, param_val, fig_ax(0))

        # Panels 2 and 6----------------------------
        if type(qbt_subsys).__name__ in ['Transmon', 'Fluxonium']:   # do not plot wavefunctions if multi-dimensional
            panels.display_bare_wavefunctions(self.sweep, qbt_subsys, param_val, fig_ax(1))
            panels.display_charge_matrixelems(self.charge_matelem_data, initial_bare, (qbt_index, qbt_subsys),
                                              param_val, fig_ax(5))

        # Panel 3 ----------------------------------
        panels.display_dressed_spectrum(self.sweep, initial_bare, final_bare, energy_initial, energy_final, param_val,
                                        fig_ax(2))

        # Panel 4 ----------------------------------
        panels.display_n_photon_qubit_transitions(self.sweep, photonnumber, initial_bare, param_val, fig_ax(3))

        # Panel 5 ----------------------------------
        panels.display_chi_01(self.chi_data, qbt_index, osc_index, param_index, fig_ax(4))

        fig.tight_layout()
        return fig, axs

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def interact(self):
        """Drives the interactive display of the plot explorer panels"""
        param_min = self.param_vals[0]
        param_max = self.param_vals[-1]
        param_step = self.param_vals[1] - self.param_vals[0]

        qbt_indices = [index for (index, subsystem) in self.sweep.qbt_subsys_list]
        osc_indices = [index for (index, subsystem) in self.sweep.osc_subsys_list]

        param_slider = ipywidgets.FloatSlider(min=param_min, max=param_max, step=param_step,
                                              description=self.param_name, continuous_update=False)
        photon_slider = ipywidgets.IntSlider(value=1, min=1, max=4, description='photon number')
        initial_slider = ipywidgets.IntSlider(value=0, min=0, max=self.evals_count, description='initial state index')
        final_slider = ipywidgets.IntSlider(value=1, min=1, max=self.evals_count, description='final state index')

        qbt_dropdown = ipywidgets.Dropdown(options=qbt_indices, description='qubit subsys')
        osc_dropdown = ipywidgets.Dropdown(options=osc_indices, description='oscillator subsys')

        def update_min_final_index(*args):
            final_slider.min = initial_slider.value + 1

        initial_slider.observe(update_min_final_index, 'value')

        out = ipywidgets.interactive_output(self.plot_explorer_panels,
                                            {'param_val': param_slider,
                                             'photonnumber': photon_slider,
                                             'initial_index': initial_slider,
                                             'final_index': final_slider,
                                             'qbt_index': qbt_dropdown,
                                             'osc_index': osc_dropdown
                                             })

        left_box = ipywidgets.VBox([param_slider])
        mid_box = ipywidgets.VBox([initial_slider, final_slider, photon_slider])
        right_box = ipywidgets.VBox([qbt_dropdown, osc_dropdown])

        user_interface = ipywidgets.HBox([left_box, mid_box, right_box])
        display(user_interface, out)
