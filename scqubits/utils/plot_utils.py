# plot_utils.py
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
import functools
import operator
import os

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

from scqubits import settings as settings
from scqubits.settings import matplotlib_settings
from scqubits.utils import plot_defaults as defaults

if TYPE_CHECKING:
    from scqubits.core.storage import WaveFunction


# A dictionary of plotting options that are directly passed to specific matplotlib's
# plot commands.
_direct_plot_options = {
    "plot": (
        "alpha",
        "color",
        "linestyle",
        "linewidth",
        "marker",
        "markersize",
        "label",
    ),
    "imshow": ("interpolation",),
    "contourf": tuple(),  # empty for now
}


@mpl.rc_context(matplotlib_settings)
def _extract_kwargs_options(
    kwargs: Dict[str, Any],
    plot_type: str,
    direct_plot_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Select options from kwargs for a given plot_type and return them in a dictionary.

    Parameters
    ----------
    kwargs:
        dictionary with options that can be passed to different plotting commands
    plot_type:
        a type of plot for which the options should be selected
    direct_plot_options:
        a lookup dictionary with supported options for a given plot_type

    Returns
    ----------
        dictionary with key/value pairs corresponding to selected options from kwargs

    """
    direct_plot_options = direct_plot_options or _direct_plot_options
    if plot_type not in direct_plot_options:
        return {}

    selected_options = {}

    for key in kwargs:
        if key in direct_plot_options[plot_type]:
            selected_options[key] = kwargs[key]
    return selected_options


@mpl.rc_context(matplotlib_settings)
def _process_options(
    figure: Figure, axes: Axes, opts: Optional[Dict[str, Any]] = None, **kwargs
) -> None:
    """
    Processes plotting options.

    Parameters
    ----------
    figure:
    axes:
    opts:
        keyword dictionary with custom options
    **kwargs:
        standard plotting option (see separate documentation)
    """
    opts = opts or {}

    # Only process items in kwargs that would not have been
    # processed through _extract_kwargs_options()
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in functools.reduce(
            operator.concat, _direct_plot_options.values()  # type:ignore
        )
    }

    option_dict = {**opts, **filtered_kwargs}

    for key, value in option_dict.items():
        if key in defaults.SPECIAL_PLOT_OPTIONS:
            _process_special_option(figure, axes, key, value)
        else:
            set_method = getattr(axes, f"set_{key}")
            set_method(value)

    filename = kwargs.get("filename")
    if filename:
        figure.savefig(os.path.splitext(filename)[0] + ".pdf")

    if settings.DESPINE and not axes.name == "3d":
        despine_axes(axes)


@mpl.rc_context(matplotlib_settings)
def _process_special_option(figure: Figure, axes: Axes, key: str, value: Any) -> None:
    """Processes a single 'special' option, i.e., one internal to scqubits and not to be
    handed further down to matplotlib.
    """
    if key == "ymax":
        ymax = value
        ymin, _ = axes.get_ylim()
        ymin = ymin - (ymax - ymin) * 0.05
        axes.set_ylim(ymin, ymax)
    elif key == "figsize":
        figure.set_size_inches(value)
    elif key == "grid":
        if isinstance(value, dict):
            axes.grid(**value)
        else:
            axes.grid(value)


@mpl.rc_context(matplotlib_settings)
def despine_axes(axes: Axes) -> None:
    # Hide the right and top spines
    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    axes.yaxis.set_ticks_position("left")
    axes.xaxis.set_ticks_position("bottom")


@mpl.rc_context(matplotlib_settings)
def scale_wavefunctions(
    wavefunc_list: List["WaveFunction"],
    potential_vals: np.ndarray,
    scaling: Optional[float],
) -> List["WaveFunction"]:
    scale_factors = np.array(
        [wavefunc.amplitude_scale_factor(potential_vals) for wavefunc in wavefunc_list]
    )
    for wavefunc in wavefunc_list:
        wavefunc.rescale(np.max(scale_factors))
    adaptive_scalefactor = scaling or defaults.set_wavefunction_scaling(
        wavefunc_list, potential_vals
    )
    for wavefunc in wavefunc_list:
        wavefunc.rescale(adaptive_scalefactor)
    return wavefunc_list


@mpl.rc_context(matplotlib_settings)
def plot_wavefunction_to_axes(
    axes: Axes, wavefunction: "WaveFunction", energy_offset: float, **kwargs
) -> None:
    x_vals = wavefunction.basis_labels
    y_vals = energy_offset + wavefunction.amplitudes
    offset_vals = [energy_offset] * len(x_vals)

    axes.plot(x_vals, y_vals, **_extract_kwargs_options(kwargs, "plot"))
    axes.fill_between(
        x_vals, y_vals, offset_vals, where=(y_vals != offset_vals), interpolate=True
    )


@mpl.rc_context(matplotlib_settings)
def plot_potential_to_axes(
    axes: Axes,
    x_vals: ndarray,
    potential_vals: Union[ndarray, List[float]],
    offset_list: Union[ndarray, List[float]],
    **kwargs,
) -> None:
    y_min = np.min(potential_vals)
    y_max = np.max(offset_list)
    y_range = y_max - y_min

    y_max += 0.3 * y_range
    y_min = np.min(potential_vals) - 0.1 * y_range
    axes.set_ylim([y_min, y_max])

    axes.plot(
        x_vals, potential_vals, color="gray", **_extract_kwargs_options(kwargs, "plot")
    )


@mpl.rc_context(matplotlib_settings)
def add_numbers_to_axes(
    axes: Axes, matrix: ndarray, modefunc: Callable, fontsize: int = 8
) -> None:
    for y_index in range(matrix.shape[0]):
        for x_index in range(matrix.shape[1]):
            axes.text(
                x_index,
                y_index,
                "{:.03f}".format(modefunc(matrix[y_index, x_index])),
                va="center",
                ha="center",
                fontsize=fontsize,
                rotation=45,
                color="white",
            )


@mpl.rc_context(matplotlib_settings)
def color_normalize(vals, mode: str) -> Tuple[float, float, mpl.colors.Normalize]:
    minval = min(vals)
    maxval = max(vals)
    if mode in ["abs", "abs_sqr"]:
        minval = 0

    nrm = mpl.colors.Normalize(minval, maxval)
    return minval, maxval, nrm
