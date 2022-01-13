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
from scqubits.core.param_sweep import ParameterSlice


if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep
    from ui.explorer_widget import ExplorerSetup


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
    `Fluxonium` and `Oscillator` subsystems. The Explorer displays pre-calculated
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
        primary_subsys_index: int,
        secondary_subsys_index: int,
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
            initial state index for the bare primary subsystem
        primary_subsys_index:
            index of subsystem for which single-system plots are displayed
        secondary_subsys_index:
            index of subsystem for which chi or Kerr is computed in conjunction with
            primary system


        Returns
        -------
            tuple of matplotlib Figure and Axes objects
        """

        def fig_ax(index):
            return fig, axes_array_flattened[index]

        param_index = np.searchsorted(self.param_vals, param_val)
        param_val = self.param_vals[param_index]

        initial_bare_list = [0] * len(self.sweep.hilbertspace)
        initial_bare_list[primary_subsys_index] = initial_index
        initial_bare = tuple(initial_bare_list)

        energy_ground = self.sweep[param_index].energy_by_dressed_index(0)
        energy_initial = (
            self.sweep[param_index].energy_by_bare_index(initial_bare) - energy_ground
        )

        qbt_subsys = self.sweep.get_subsys(primary_subsys_index)
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
                primary_subsys_index,
                param_val,
                fig_ax(5),
            )

        # Panel 3 ----------------------------------
        # panels.display_dressed_spectrum(
        #     self.sweep,
        #     initial_bare,
        #     final_bare,
        #     energy_initial,
        #     energy_final,
        #     param_val,
        #     fig_ax(2),
        # )
        panels.display_anharmonicity(self.sweep, qbt_subsys, param_val, fig_ax(2))

        # Panel 4 ----------------------------------
        # initial_dressed_index = self.sweep[param_index].dressed_index(initial_bare)
        panels.display_n_photon_qubit_transitions(
            self.sweep, photonnumber, qbt_subsys, initial_bare, param_val, fig_ax(3)
        )

        # Panel 5 ----------------------------------
        panels.display_kerrlike(
            self.sweep,
            primary_subsys_index,
            secondary_subsys_index,
            param_val,
            fig_ax(4),
        )

        fig.tight_layout()
        plt.show()
        return fig, axes_table

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def interact(self):
        """Drives the interactive display of the plot explorer panels"""
        param_min = self.param_vals[0]
        param_max = self.param_vals[-1]
        param_step = self.param_vals[1] - self.param_vals[0]

        subsys_id_to_index = [
            (subsystem.id_str, self.sweep.get_subsys_index(subsystem))
            for subsystem in self.sweep.hilbertspace
        ]
        subsys_id_to_index.reverse()

        qbt_id_to_index = [
            (subsystem.id_str, self.sweep.get_subsys_index(subsystem))
            for subsystem in self.sweep.qbt_subsys_list
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

        primary_subsys_dropdown = ipywidgets.Dropdown(
            options=qbt_id_to_index, description="primary subsys"
        )
        secondary_subsys_dropdown = ipywidgets.Dropdown(
            options=subsys_id_to_index, description="secondary subsys"
        )

        out = ipywidgets.interactive_output(
            self.plot_explorer_panels,
            {
                "param_val": param_slider,
                "photonnumber": photon_slider,
                "initial_index": initial_slider,
                "primary_subsys_index": primary_subsys_dropdown,
                "secondary_subsys_index": secondary_subsys_dropdown,
            },
        )

        left_box = ipywidgets.VBox([param_slider])
        mid_box = ipywidgets.VBox([initial_slider, photon_slider])
        right_box = ipywidgets.VBox(
            [primary_subsys_dropdown, secondary_subsys_dropdown]
        )

        user_interface = ipywidgets.HBox([left_box, mid_box, right_box])
        display(user_interface, out)


class ExplorerMixin:
    """
    This class allows interactive exploration of coupled quantum systems. The
    Explorer is currently compatible with systems composed of `Transmon`,
    `Fluxonium` and `Oscillator` subsystems. The Explorer displays pre-calculated
    spectral data and enables changes of a given parameter by sliders (when inside
    jupyter notebook or jupyter lab).

    Parameters
    ----------
    sweep:
        `ParameterSweep` object, must correspond to a 1d sweep
    panels:
        list of Panel objects specifying the panels to be displayed
    options:
        options in keyword arguments form
        - figsize: tuple(float, float)
    """

    def update_panels(self: "ExplorerSetup") -> None:
        out = ipywidgets.interactive_output(self.update_explorer, controls={})

    def plot_explorer_panels(
        self: "ExplorerSetup",
    ) -> Tuple[Figure, Axes]:
        """
        Returns
        -------
            tuple of matplotlib Figure and Axes objects
        """
        panels = self.get_panels_list()

        ncols = 2
        nrows = len(panels)

        fig, axes_table = plt.subplots(ncols=ncols, nrows=nrows, figsize=self.figsize)

        def axes_by_index(index):
            row_index = index // nrows
            col_index = index % ncols
            return axes_table[row_index, col_index]

        param_slice = ParameterSlice(
            self.ui_sweep_param_dropdown.value,
            self.ui_sweep_value_slider.value,
            self.fixed_params,
            list(self.sweep.param_info.keys()),
        )

        for index, panel in enumerate(panels):
            panel.plot_func(fig_ax=axes_by_index(index))

        # initial_bare_list = [0] * len(self.sweep.hilbertspace)
        # initial_bare_list[primary_subsys_index] = initial_index
        # initial_bare = tuple(initial_bare_list)

        # energy_ground = self.sweep[param_slice.all].energy_by_dressed_index(0)
        # energy_initial = (
        #     self.sweep[param_slice.fixed].energy_by_bare_index(initial_bare)
        #     - energy_ground
        # )

        # qbt_subsys = self.sweep.get_subsys(primary_subsys_index)
        # assert isinstance(qbt_subsys, QubitBaseClass1d), (
        #     "Unsupported qubit. Explorer currently only accepts 1d qubits."
        # )
        #
        # row_count = 3
        # column_count = 2
        # fig, axes_table = plt.subplots(
        #     ncols=column_count, nrows=row_count, figsize=self.figsize
        # )
        # axes_array_flattened = np.asarray(axes_table).flatten()
        #
        # Panel 1 ----------------------------------
        # panels.display_bare_spectrum(self.sweep, qbt_subsys, param_slice, fig_ax(0))
        # #
        # # # Panels 2 and 6----------------------------
        # panels.display_bare_wavefunctions(
        #     self.sweep, qbt_subsys, param_slice, fig_ax(1)
        # )
        # #     panels.display_charge_matrixelems(
        # #         self.sweep,
        # #         initial_bare,
        # #         primary_subsys_index,
        # #         param_val,
        # #         fig_ax(5),
        # #     )
        # #
        # # # Panel 3 ----------------------------------
        #
        # panels.display_anharmonicity(self.sweep, qbt_subsys, param_slice, fig_ax(2))
        #
        # panels.display_n_photon_qubit_transitions(
        #     self.sweep, photonnumber, qbt_subsys, initial_bare, param_slice, fig_ax(3)
        # )
        #
        # # # Panel 5 ----------------------------------
        # # panels.display_kerrlike(
        # #     self.sweep,
        # #     primary_subsys_index,
        # #     secondary_subsys_index,
        # #     param_val,
        # #     fig_ax(4),
        # # )

        fig.tight_layout()
        plt.show()
        return fig, axes_table

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
    def interact(self):
        """Drives the interactive display of the plot explorer panels"""

        self.out = ipywidgets.interactive_output(
            self.plot_explorer_panels,
            {
                "param_val": self.param_slider,
                "photonnumber": self.photon_slider,
                "initial_index": self.initial_slider,
                "primary_subsys_index": self.primary_subsys_dropdown,
                "secondary_subsys_index": self.secondary_subsys_dropdown,
            },
        )

        left_box = ipywidgets.VBox([self.param_slider])
        mid_box = ipywidgets.VBox([self.initial_slider, self.photon_slider])
        right_box = ipywidgets.VBox(
            [self.primary_subsys_dropdown, self.secondary_subsys_dropdown]
        )

        self.user_interface = ipywidgets.HBox([left_box, mid_box, right_box])
        display(self.user_interface, self.out)
