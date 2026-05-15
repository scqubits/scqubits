"""CROSS_KERR / AC_STARK panel: pairwise Kerr / Stark coefficients vs the swept parameter.

``display_cross_kerr`` is shared by both ``CrossKerrPanelBuilder`` (this
file) and ``AcStarkPanelBuilder`` (``ac_stark.py``); the latter imports
it from here.  Internally the function branches on the number of
:class:`Oscillator` subsystems among its two inputs to choose between
AC-Stark, oscillator-oscillator cross-Kerr, and qubit-qubit cross-Kerr
formulas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scqubits.core.units as units

from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.oscillator import Oscillator
from scqubits.settings import matplotlib_settings
from scqubits.ui.gui_defaults import PlotType
from scqubits.utils.misc import tuple_to_short_str

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSlice, ParameterSweep
    from scqubits.core.qubit_base import QuantumSystem
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID


@rc_context(matplotlib_settings)
def display_cross_kerr(
    sweep: "ParameterSweep",
    subsys1: "QuantumSystem",
    subsys2: "QuantumSystem",
    param_slice: "ParameterSlice",
    fig_ax: tuple[Figure, Axes],
    which: int | tuple[int, int] | None = None,
) -> tuple[Figure, Axes]:
    """Plot pairwise Kerr / Stark coefficients between two subsystems."""
    subsys1_index = sweep.get_subsys_index(subsys1)
    subsys2_index = sweep.get_subsys_index(subsys2)
    type_list = [type(sys) for sys in [subsys1, subsys2]]
    label_list: list[int | tuple[int, int]] | list[str]
    if type_list.count(Oscillator) == 1:
        title = f"AC Stark: {subsys1.id_str} + {subsys2.id_str}"
        ylabel = rf"AC Stark Shift $\chi^{{{subsys1_index},{subsys2_index}}}_\ell$"
        level = which or 1
        kerr_data = sweep["chi"][subsys1_index, subsys2_index]
        if param_slice.fixed != tuple():
            kerr_data = kerr_data[param_slice.fixed]
        label_list = [level]
        kerr_datasets = [kerr_data[..., level]]
    elif type_list.count(Oscillator) == 2:
        title = f"Cross-Kerr: {subsys1.id_str} - {subsys2.id_str}"
        ylabel = rf"Kerr Coefficient $K_{{{subsys1_index},{subsys2_index}}}$"
        level_pairs = [(1, 1)]
        kerr_data = sweep["kerr"][subsys1_index, subsys2_index]
        if param_slice.fixed != tuple():
            kerr_data = kerr_data[param_slice.fixed]
        label_list = []
        kerr_datasets = [
            kerr_data[..., level1, level2] for level1, level2 in level_pairs
        ]
    else:
        title = f"Cross-Kerr: {subsys1.id_str} ↔ {subsys2.id_str}"
        ylabel = (
            r"Kerr Coefficient $\Lambda^{"
            + f"{subsys1_index},{subsys2_index}"
            + r"}_{ll'}$"
        )
        level_pairs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
        kerr_data = sweep["kerr"][subsys1_index, subsys2_index]
        if param_slice.fixed != tuple():
            kerr_data = kerr_data[param_slice.fixed]
        label_list = [tuple_to_short_str(pair) for pair in level_pairs]
        kerr_datasets = [
            kerr_data[..., level1, level2] for level1, level2 in level_pairs
        ]

    kerr_datasets_arr = np.asarray(kerr_datasets).T
    kerr_namedarray = NamedSlotsNdarray(kerr_datasets_arr, kerr_data.param_info)
    fig, axes = kerr_namedarray.plot(
        title=title,
        label_list=label_list if label_list else None,
        ylabel=f"{ylabel}[{units.get_units()}]",
        fig_ax=fig_ax,
    )
    axes.axvline(param_slice.param_val, color="gray", linestyle=":")
    return fig, axes


class CrossKerrPanelBuilder:
    """Builder for the CROSS_KERR panel (composite, no per-plot settings)."""

    plot_type: ClassVar[PlotType] = PlotType.CROSS_KERR
    slider_invariant: ClassVar[bool] = True

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: tuple[Figure, Axes],
    ) -> tuple[Figure, Axes]:
        return display_cross_kerr(
            sweep=explorer.sweep,
            subsys1=plot_id.subsystems[0],
            subsys2=plot_id.subsystems[1],
            param_slice=param_slice,
            fig_ax=fig_ax,
        )

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        return []
