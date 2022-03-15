# explorer.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.explorer.explorer_panels as panels
import scqubits.utils.misc as utils

from scqubits.core.qubit_base import QubitBaseClass1d

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep


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


class Explorer:
    """
    This class allows interactive exploration of coupled quantum systems. The
    Explorer is currently compatible with systems composed of `Transmon`,
    `Fluxoniuam` and `Oscillator` subsystems. The Explorer displays pre-calculated
    spectral data and enables changes of a given parameter by sliders (when inside
    jupyter notebook or jupyter lab).

    Parameters
    ----------
    sweep:
        `ParameterSweep` object, must correspond to a 1d sweep
    evals_count:
        number of levels to include in energy spectra
    figsize:
        custom size of individual panels in plot (optional)
    """

    def __init__(
        self,
        sweep: "ParameterSweep",
        evals_count: int,
        figsize: Tuple[float, float] = (10.5, 8),
    ) -> None:
        if len(sweep.param_info) > 1:
            raise ValueError(
                "ParameterSweep provided for Explorer must be 1-dimensional."
            )
        for name, vals in sweep.param_info.items():
            self.param_name = name
            self.param_vals = vals
        self.param_count = len(self.param_vals)

        self.sweep = sweep
        self.evals_count = evals_count
        self.figsize = figsize

        self.chi_data = sweep["chi"]

        for index, subsys in enumerate(self.sweep.qbt_subsys_list):
            self.sweep.add_matelem_sweep(
                operator="n_operator",
                sweep_name="n_operator qubit " + str(index),
                subsystem=subsys,
            )

    def plot_explorer_panels(
        self,
        param_val: float,
        photonnumber: int,
        initial_index: int,
        final_index: int,
        qbt_index: int,
        osc_index: int,
    ) -> Tuple[Figure, Axes]:
        """
        Create a panel of plots (bare spectra, bare wavefunctions, dressed spectrum,
        n-photon qubit transitions, chi).

        Parameters
        ----------
        param_val:
            current value of the external parameter
        photonnumber:
            photon number n used for display of n-photon qubit transition
        initial_index:
            dressed-state index of the initial state used in transition
        final_index:
            dressed-state index of the final state used in transition (in dressed
            spectrum display)
        qbt_index:
            index of qubit subsystem for which matrix elements and chi's are displayed
        osc_index:
            index of oscillator subsystem for which chi's are calculated

        Returns
        -------
            tuple of matplotlib Figure and Axes objects
        """

        def fig_ax(index):
            return fig, axes_array_flattened[index]

        param_index = np.searchsorted(self.param_vals, param_val)
        param_val = self.param_vals[param_index]

        initial_bare = self.sweep[param_index].bare_index(initial_index)
        final_bare = self.sweep[param_index].bare_index(final_index)
        energy_ground = self.sweep[param_index].energy_by_dressed_index(0)
        energy_initial = (
            self.sweep[param_index].energy_by_dressed_index(initial_index)
            - energy_ground
        )
        energy_final = (
            self.sweep[param_index].energy_by_dressed_index(final_index) - energy_ground
        )
        qbt_subsys = self.sweep.get_subsys(qbt_index)
        assert isinstance(qbt_subsys, QubitBaseClass1d), (
            "Unsupported qubit. " "Explorer currently only " "accepts 1d qubits."
        )

        row_count = 3
        column_count = 2
        fig, axes_table = plt.subplots(
            ncols=column_count, nrows=row_count, figsize=self.figsize
        )
        axes_array_flattened = np.asarray(axes_table).flatten()

        # Panel 1 ----------------------------------
        panels.display_bare_spectrum(self.sweep, qbt_subsys, param_val, fig_ax(0))

        # Panels 2 and 6----------------------------
        if type(qbt_subsys).__name__ in [
            "Transmon",
            "Fluxonium",
        ]:  # do not plot wavefunctions if multi-dimensional
            panels.display_bare_wavefunctions(
                self.sweep, qbt_subsys, param_val, fig_ax(1)
            )
            panels.display_charge_matrixelems(
                self.sweep,
                initial_bare,
                qbt_index,
                param_val,
                fig_ax(5),
            )

        # Panel 3 ----------------------------------
        panels.display_dressed_spectrum(
            self.sweep,
            initial_bare,
            final_bare,
            energy_initial,
            energy_final,
            param_val,
            fig_ax(2),
        )

        # Panel 4 ----------------------------------
        panels.display_n_photon_qubit_transitions(
            self.sweep, photonnumber, qbt_subsys, initial_bare, param_val, fig_ax(3)
        )

        # Panel 5 ----------------------------------
        panels.display_chi_01(self.sweep, qbt_index, osc_index, param_val, fig_ax(4))

        fig.tight_layout()
        return fig, axes_table

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def interact(self):
        """Drives the interactive display of the plot explorer panels"""
        param_min = self.param_vals[0]
        param_max = self.param_vals[-1]
        param_step = self.param_vals[1] - self.param_vals[0]

        qbt_indices = [
            self.sweep.get_subsys_index(subsystem)
            for subsystem in self.sweep.qbt_subsys_list
        ]
        osc_indices = [
            self.sweep.get_subsys_index(subsystem)
            for subsystem in self.sweep.osc_subsys_list
        ]

        param_slider = ipywidgets.FloatSlider(
            min=param_min,
            max=param_max,
            step=param_step,
            description=self.param_name,
            continuous_update=False,
        )
        photon_slider = ipywidgets.IntSlider(
            value=1, min=1, max=4, description="photon number"
        )
        initial_slider = ipywidgets.IntSlider(
            value=0, min=0, max=self.evals_count, description="initial state index"
        )
        final_slider = ipywidgets.IntSlider(
            value=1, min=1, max=self.evals_count, description="final state index"
        )

        qbt_dropdown = ipywidgets.Dropdown(
            options=qbt_indices, description="qubit subsys"
        )
        osc_dropdown = ipywidgets.Dropdown(
            options=osc_indices, description="oscillator subsys"
        )

        def update_min_final_index(*args):
            final_slider.min = initial_slider.value + 1

        initial_slider.observe(update_min_final_index, "value")

        out = ipywidgets.interactive_output(
            self.plot_explorer_panels,
            {
                "param_val": param_slider,
                "photonnumber": photon_slider,
                "initial_index": initial_slider,
                "final_index": final_slider,
                "qbt_index": qbt_dropdown,
                "osc_index": osc_dropdown,
            },
        )

        left_box = ipywidgets.VBox([param_slider])
        mid_box = ipywidgets.VBox([initial_slider, final_slider, photon_slider])
        right_box = ipywidgets.VBox([qbt_dropdown, osc_dropdown])

        user_interface = ipywidgets.HBox([left_box, mid_box, right_box])
        display(user_interface, out)
