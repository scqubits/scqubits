"""SELF_KERR panel: self-Kerr coefficient vs the swept parameter.

Dispatches at runtime between the oscillator-style plot (single curve,
no settings) and the qubit-style plot (level-pair selector + multi
curve).  The ``isinstance(subsys, Oscillator)`` check is the
authoritative discriminator; the legacy code keyed off the truthiness
of ``self.settings[plot_id]`` (empty list for oscillators, populated
for qubits), which left the type assumption implicit.
"""

from __future__ import annotations

import itertools

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits
import scqubits.core.units as units
import scqubits.ui.gui_custom_widgets as ui

from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.oscillator import Oscillator
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QuantumSystem
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID


@rc_context(matplotlib_settings)
def display_qubit_self_kerr(
    sweep: "ParameterSweep",
    subsys: "QuantumSystem",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
    which: list[tuple[int, int]] | None = None,
) -> tuple[Figure, Axes]:
    """Plot qubit-mode self-Kerr ``Lambda^{ll}`` curves for the selected level pairs."""
    subsys_index = sweep.get_subsys_index(subsys)
    title = f"Self-Kerr: {subsys.id_str}"
    ylabel = (
        r"Kerr coefficient $\Lambda^{" + f"{subsys_index},{subsys_index}" + r"}_{ll}$"
    )

    kerr_data = sweep["kerr"][subsys_index, subsys_index]
    if param_slice.fixed != tuple():
        kerr_data = kerr_data[param_slice.fixed]

    if not which:
        level_pairs = list(itertools.combinations(list(range(subsys.truncated_dim)), 2))
    else:
        level_pairs = which

    kerr_datasets = []
    for pair in level_pairs:
        kerr_datasets.append(kerr_data[..., pair[0], pair[1]])
    kerr_datasets_arr = np.asarray(kerr_datasets).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets_arr, kerr_data.param_info)

    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=level_pairs,
        ylabel=f"{ylabel}[{units.get_units()}]",
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


@rc_context(matplotlib_settings)
def display_self_kerr(
    sweep: "ParameterSweep",
    subsys: "scqubits.Oscillator | scqubits.KerrOscillator",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]:
    """Plot oscillator self-Kerr ``K`` (a single curve) vs the swept parameter."""
    subsys_index = sweep.get_subsys_index(subsys)
    title = f"Self-Kerr: {subsys.id_str}"
    ylabel = f"Kerr coefficient $K_{{{subsys_index}}}$"
    kerr_data = sweep["kerr"][subsys_index, subsys_index]
    if param_slice.fixed != tuple():
        kerr_data = kerr_data[param_slice.fixed]

    fig, axes = kerr_data.plot(
        title=title,
        ylabel=f"{ylabel}[{units.get_units()}]",
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


class SelfKerrPanelBuilder:
    """Builder for the SELF_KERR panel (oscillator and qubit variants)."""

    plot_type: ClassVar[PlotType] = PlotType.SELF_KERR
    slider_invariant: ClassVar[bool] = True

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        subsys = plot_id.subsystems[0]
        if isinstance(subsys, Oscillator):
            return display_self_kerr(
                sweep=explorer.sweep,
                subsys=subsys,
                param_slice=param_slice,
                fig_ax=fig_ax,
            )
        ui_state_selection = explorer.settings[plot_id][0]
        return display_qubit_self_kerr(
            sweep=explorer.sweep,
            subsys=subsys,
            param_slice=param_slice,
            fig_ax=fig_ax,
            which=ui_state_selection.v_model,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        subsys = plot_id.subsystems[0]
        if isinstance(subsys, Oscillator):
            return []
        ui_kerr_selector = ui.InitializedSelect(
            label="Selected pair of levels",
            items=list(
                itertools.combinations_with_replacement(
                    list(range(subsys.truncated_dim)), 2
                )
            ),
            v_model=[[1, 1], [1, 2], [2, 2]],
            multiple=True,
        )
        ui_kerr_selector.observe(settings.explorer.update_plots, names="v_model")
        return [ui_kerr_selector]
