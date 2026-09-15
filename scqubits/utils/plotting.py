# plotting.py
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

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import scqubits.core.constants as constants
import scqubits.utils.misc as utils
import scqubits.utils.plot_defaults as defaults

from scqubits.settings import matplotlib_settings
from scqubits.utils.plot_utils import (
    _extract_kwargs_options,
    _process_options,
    add_numbers_to_axes,
    color_normalize,
    plot_potential_to_axes,
    plot_wavefunction_to_axes,
    scale_wavefunctions,
)

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData, WaveFunction, WaveFunctionOnGrid

try:
    from labellines import labelLines

    _LABELLINES_ENABLED = True
except ImportError:
    _LABELLINES_ENABLED = False


@mpl.rc_context(matplotlib_settings)
def wavefunction1d(
    wavefuncs: "WaveFunction" | "list[WaveFunction]",
    potential_vals: np.ndarray,
    offset: float | Iterable[float] = 0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plots the amplitude of a single real-valued 1d wave function, along with the
    potential energy if provided.

    Parameters
    ----------
    wavefuncs:
        basis and amplitude data of wave function to be plotted
    potential_vals:
        potential energies, array length must match basis array of `wavefunc`
    offset:
        y-offset for the wave function (e.g., shift by eigenenergy)
    scaling:
        scaling factor for wave function amplitudes
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    offset_list = utils.to_list(offset)
    wavefunc_list: list[WaveFunction] = utils.to_list(wavefuncs)
    wavefunc_list = scale_wavefunctions(wavefunc_list, potential_vals, scaling)

    for wavefunction, energy_offset in zip(wavefunc_list, offset_list):
        plot_wavefunction_to_axes(axes, wavefunction, energy_offset, **kwargs)

    x_vals = wavefunc_list[0].basis_labels
    plot_potential_to_axes(axes, x_vals, potential_vals, offset_list, **kwargs)

    _process_options(fig, axes, **kwargs)
    return fig, axes


@mpl.rc_context(matplotlib_settings)
def wavefunction1d_nopotential(
    wavefuncs: "WaveFunction" | "list[WaveFunction]",
    offset: float | Iterable[float] = 0,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plots the amplitude of a single real-valued 1d wave function, along with the
    potential energy if provided.

    Parameters
    ----------
    wavefuncs:
        basis and amplitude data of wave function to be plotted
    potential_vals:
        potential energies, array length must match basis array of `wavefunc`
    offset:
        y-offset for the wave function (e.g., shift by eigenenergy)
    scaling:
        scaling factor for wave function amplitudes
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    offset_list = utils.to_list(offset)
    wavefunc_list: list[WaveFunction] = utils.to_list(wavefuncs)

    for wavefunction, energy_offset in zip(wavefunc_list, offset_list):
        plot_wavefunction_to_axes(axes, wavefunction, energy_offset, **kwargs)

    _process_options(fig, axes, **kwargs)
    return fig, axes


@mpl.rc_context(matplotlib_settings)
def wavefunction1d_discrete(wavefunc: "WaveFunction", **kwargs) -> tuple[Figure, Axes]:
    """Plots the amplitude of a real-valued 1d wave function in a discrete basis.
    (Example: transmon in the charge basis.)

    Parameters
    ----------
    wavefunc:
        basis and amplitude data of wave function to be plotted
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    x_vals = wavefunc.basis_labels
    width = 0.75
    axes.bar(x_vals, wavefunc.amplitudes, width=width)

    axes.set_xticks(x_vals)
    axes.set_xticklabels(x_vals)
    _process_options(fig, axes, defaults.wavefunction1d_discrete(), **kwargs)

    return fig, axes


@mpl.rc_context(matplotlib_settings)
def wavefunction2d(
    wavefunc: "WaveFunctionOnGrid", zero_calibrate: bool = False, **kwargs
) -> tuple[Figure, Axes]:
    """Creates a density plot of the amplitude of a real-valued wave function in 2
    "spatial" dimensions.

    Parameters
    ----------
    wavefunc:
        basis and amplitude data of wave function to be plotted
    zero_calibrate:
        whether to calibrate plot to zero amplitude
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    min_vals = wavefunc.gridspec.min_vals
    max_vals = wavefunc.gridspec.max_vals

    if zero_calibrate:
        absmax = np.amax(np.abs(wavefunc.amplitudes))
        imshow_minval = -absmax
        imshow_maxval = absmax
        cmap = plt.get_cmap("PRGn")
    else:
        imshow_minval = np.min(wavefunc.amplitudes)
        imshow_maxval = np.max(wavefunc.amplitudes)
        cmap = plt.get_cmap("viridis")

    im = axes.imshow(
        wavefunc.amplitudes,
        extent=(min_vals[0], max_vals[0], min_vals[1], max_vals[1]),
        cmap=cmap,
        vmin=imshow_minval,
        vmax=imshow_maxval,
        origin="lower",
        aspect="auto",
        **_extract_kwargs_options(kwargs, "imshow"),
    )
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax)

    _process_options(fig, axes, defaults.wavefunction2d(), **kwargs)
    return fig, axes


@mpl.rc_context(matplotlib_settings)
def contours(
    x_vals: list[float] | np.ndarray,
    y_vals: list[float] | np.ndarray,
    func: Callable,
    contour_vals: list[float] | np.ndarray | None = None,
    show_colorbar: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Contour plot of a 2d function `func(x,y)`.

    Parameters
    ----------
    x_vals:
        x values for the x-y evaluation grid
    y_vals:
        y values for the x-y evaluation grid
    func:
        function f(x,y) for which contours are to be plotted
    contour_vals:
        contour values can be specified if so desired
    show_colorbar:
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_array = func(x_grid, y_grid)

    im = axes.contourf(
        x_grid,
        y_grid,
        z_array,
        levels=contour_vals,
        cmap=plt.get_cmap("viridis"),
        origin="lower",
        **_extract_kwargs_options(kwargs, "contourf"),
    )

    if show_colorbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        fig.colorbar(im, cax=cax)

    _process_options(fig, axes, opts=defaults.contours(x_vals, y_vals), **kwargs)
    return fig, axes


@mpl.rc_context(matplotlib_settings)
def matrix(
    data_matrix: np.ndarray, mode: str = "abs", show_numbers: bool = False, **kwargs
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create a "skyscraper" plot and a 2d color-coded plot of a matrix.

    Parameters
    ----------
    data_matrix:
        2d matrix data
    mode:
        choice from `constants.MODE_FUNC_DICT` for processing function to be applied to
        data
    show_numbers:
        determines whether matrix element values are printed on top of the plot
        (default: False)
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        figure and axes objects for further editing
    """
    if "fig_ax" in kwargs:
        fig, (ax1, ax2) = kwargs["fig_ax"]
        del kwargs["fig_ax"]
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = plt.subplot(1, 2, 2)

    fig, ax2 = matrix2d(
        data_matrix, mode=mode, show_numbers=show_numbers, fig_ax=(fig, ax2), **kwargs
    )
    fig, ax1 = matrix_skyscraper(data_matrix, mode=mode, fig_ax=(fig, ax1), **kwargs)
    return fig, (ax1, ax2)


@mpl.rc_context(matplotlib_settings)
def matrix_skyscraper(
    matrix: np.ndarray, mode: str = "abs", **kwargs
) -> tuple[Figure, Axes]:
    """Display a 3d skyscraper plot of the matrix.

    Parameters
    ----------
    matrix:
        2d matrix data
    mode:
        choice from `constants.MODE_FUNC_DICT` for processing function to be applied to
        data
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        figure and axes objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots(projection="3d")

    y_count, x_count = matrix.shape  # We label the columns as "x", while rows as "y"
    element_count = x_count * y_count  # total num. of elements to plot

    xgrid, ygrid = np.meshgrid(range(x_count), range(y_count))
    xgrid = xgrid.flatten()
    ygrid = ygrid.flatten()

    zbottom = np.zeros(element_count)  # all bars start at z=0
    dx, dy = 0.75, 0.75  # width of bars in x and y directions

    modefunction = cast(Callable, constants.MODE_FUNC_DICT[mode])
    zheight = modefunction(matrix).flatten()  # height of bars from matrix elements

    min_zheight, max_zheight, nrm = color_normalize(zheight, mode)
    colors = plt.get_cmap("viridis")(nrm(zheight))  # list of colors for each bar

    # skyscraper plot (axes has projection="3d")
    axes = cast(Axes3D, axes)
    axes.view_init(azim=210, elev=23)
    axes.bar3d(xgrid, ygrid, zbottom, dx, dy, zheight, color=colors)

    if mode in ["real", "imag"]:
        min_zheight = 0 if min_zheight > 0 else min_zheight
        max_zheight = 0 if max_zheight < 0 else max_zheight

    if min_zheight == max_zheight:
        # pad with small values so we don't get warnings
        max_zheight += 0.0000001

    axes.set_zlim3d([min_zheight, max_zheight])

    for axis, locs in [
        (axes.xaxis, np.arange(x_count)),
        (axes.yaxis, np.arange(y_count)),
    ]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs + 0.5, ticklabels=locs)

    axes.tick_params(axis="x", pad=-5)
    axes.tick_params(axis="y", pad=-5)
    axes.tick_params(axis="z", pad=-2)

    _process_options(fig, axes, opts=defaults.matrix(), **kwargs)

    return fig, axes


@mpl.rc_context(matplotlib_settings)
def matrix2d(
    matrix: np.ndarray,
    mode: str = "abs",
    show_numbers: bool = True,
    show_colorbar: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Display a matrix as a color-coded 2d plot, optionally printing the numerical
    values of the matrix elements.

    Parameters
    ----------
    matrix:
        2d matrix data
    mode:
        choice from `constants.MODE_FUNC_DICT` for processing function to be applied to
        data
    show_numbers:
        determines whether matrix element values are printed on top of the plot
        (default: True)
    show_colorbar:
        switch whether to draw a color legend (default: True)
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        figure and axes objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    modefunction = cast(Callable, constants.MODE_FUNC_DICT[mode])
    zheight = modefunction(matrix).flatten()  # height of bars from matrix elements

    min_zheight, max_zheight, nrm = color_normalize(zheight, mode)

    if show_colorbar:
        # add colorbar with normalized range
        if hasattr(axes, "colorbar"):  # update existing colorbar
            axes.colorbar.update_normal(
                mpl.cm.ScalarMappable(norm=nrm, cmap=plt.get_cmap("viridis"))
            )
        else:  # create new colorbar
            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=nrm, cmap=plt.get_cmap("viridis")),
                ax=axes,
                fraction=0.046,
                pad=0.04,
            )
            axes.colorbar = cb  # type: ignore[union-attr]
    cax = axes.matshow(
        modefunction(matrix), cmap=plt.get_cmap("viridis"), interpolation=None
    )

    if show_numbers:
        fig_width, fig_height = fig.get_size_inches()
        box_width_inches = fig_width / matrix.shape[1]
        box_height_inches = fig_height / matrix.shape[0]
        font_size = min(box_width_inches, box_height_inches) * 11
        add_numbers_to_axes(axes, matrix, modefunction, fontsize=font_size)

    # shift the grid
    for axis, locs in [
        (axes.xaxis, np.arange(matrix.shape[1])),
        (axes.yaxis, np.arange(matrix.shape[0])),
    ]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs, ticklabels=locs)
    axes.grid(False)

    _process_options(fig, axes, **kwargs)
    axes.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    return fig, axes


def _repeat_curve_columns(
    columns: np.ndarray | list[np.ndarray],
    n_curves: int,
) -> np.ndarray:
    """Repeat a single curve column for multi-curve broadcasting."""
    if isinstance(columns, np.ndarray):
        return np.repeat(columns, n_curves, axis=0)
    return np.array(columns * n_curves)


@mpl.rc_context(matplotlib_settings)
def data_vs_paramvals(
    xdata: np.ndarray,
    ydata: np.ndarray,
    label_list: list[str] | list[int] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot of a set of ydata vs xdata. The individual points correspond to the a
    provided array of parameter values.

    Parameters
    ----------
    xdata, ydata:
        must have compatible shapes for matplotlib.pyplot.plot
    label_list:
        list of labels associated with the individual curves to be plotted
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Normalise to 2-dim column arrays so the loop below is uniform.
    # A 1-dim array is treated as a single dataset (one column).
    x_cols = xdata.T if xdata.ndim == 2 else [xdata]
    y_cols = ydata.T if ydata.ndim == 2 else [ydata]

    # Broadcast: if one side has a single column, reuse it for every column of the other.
    n_curves = max(len(x_cols), len(y_cols))
    if len(x_cols) == 1:
        x_cols = _repeat_curve_columns(x_cols, n_curves)
    if len(y_cols) == 1:
        y_cols = _repeat_curve_columns(y_cols, n_curves)

    if label_list is None:
        for xdataset, ydataset in zip(x_cols, y_cols):
            axes.plot(xdataset, ydataset, **_extract_kwargs_options(kwargs, "plot"))
        _process_options(fig, axes, **kwargs)
        return fig, axes

    for idx, (xdataset, ydataset) in enumerate(zip(x_cols, y_cols)):
        axes.plot(
            xdataset,
            ydataset,
            label=label_list[idx],
            **_extract_kwargs_options(kwargs, "plot"),
        )
    if _LABELLINES_ENABLED:
        # Drop curves whose y values are all NaN (transition energies near
        # level crossings can produce such curves). labelLines raises and
        # warns on them, leaving every other curve unlabelled too.
        visible_lines = [
            line for line in axes.get_lines() if not np.all(np.isnan(line.get_ydata()))
        ]
        if visible_lines:
            try:
                labelLines(visible_lines, zorder=2.0)
            except Exception:
                pass
    else:
        axes.legend(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            frameon=False,
        )
    _process_options(fig, axes, **kwargs)

    # Ensures that np.nan entries (as present in transition energy plots)
    # cannot reduce the intended x range. Use the full x span so multi-curve
    # plots (e.g. branch analysis) are not clipped to the first column only.
    x_min = np.nanmin(xdata)
    x_max = np.nanmax(xdata)
    axes.update_datalim(np.array([[x_min, 0.0], [x_max, 0.0]]), updatey=False)
    axes.autoscale()

    return fig, axes


@mpl.rc_context(matplotlib_settings)
def evals_vs_paramvals(
    specdata: "SpectrumData",
    which: int | Iterable[int] = -1,
    subtract_ground: bool = False,
    label_list: list[str] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Generates a simple plot of a set of eigenvalues as a function of one parameter.
    The individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    specdata:
        object includes parameter name, values, and resulting eigenenergies
    which:
        number of desired eigenvalues (sorted from smallest to largest); default: -1,
        signals all eigenvalues
        or: list of specific eigenvalues to include
    subtract_ground:
        whether to subtract the ground state energy
    label_list:
        list of labels associated with the individual curves to be plotted
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        matplotlib objects for further editing
    """
    index_list = utils.process_which(which, specdata.energy_table[0].size)

    assert specdata.param_vals is not None, "SpectrumData is missing parameter values!"
    xdata = specdata.param_vals
    assert isinstance(specdata.energy_table, np.ndarray)
    ydata = specdata.energy_table[:, index_list]
    if subtract_ground:
        ydata = (ydata.T - ydata[:, 0]).T

    return data_vs_paramvals(
        xdata,
        ydata,
        label_list=label_list,
        **defaults.evals_vs_paramvals(specdata, **kwargs),
    )


@mpl.rc_context(matplotlib_settings)
def matelem_vs_paramvals(
    specdata: "SpectrumData",
    select_elems: int | list[tuple[int, int]] = 4,
    mode: str = "abs",
    **kwargs,
) -> tuple[Figure, Axes]:
    """Generates a simple plot of matrix elements as a function of one parameter. The
    individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    specdata:
        object includes parameter name, values, and matrix elements
    select_elems:
        either maximum index of desired matrix elements,
        or list [(i1, i2), (i3, i4), ...] of index tuples
        for specific desired matrix elements
    mode:
        choice of processing function to be applied to data (default value = 'abs')
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
    matplotlib objects for further editing
    """
    assert (
        specdata.matrixelem_table is not None
    ), "SpectrumData is missing matrix element data!"
    fig, axes = kwargs.get("fig_ax") or plt.subplots()
    x_vals = cast(np.ndarray, specdata.param_vals)
    modefunction = cast(Callable, constants.MODE_FUNC_DICT[mode])

    if isinstance(select_elems, int):
        index_pairs = [
            (row, col) for row in range(select_elems) for col in range(row + 1)
        ]
    else:
        index_pairs = select_elems

    for row, col in index_pairs:
        y_vals = modefunction(specdata.matrixelem_table[:, row, col])
        axes.plot(
            x_vals,
            y_vals,
            label=f"{row},{col}",
            **_extract_kwargs_options(kwargs, "plot"),
        )

    if _LABELLINES_ENABLED:
        visible_lines = [
            line for line in axes.get_lines() if not np.all(np.isnan(line.get_ydata()))
        ]
        if visible_lines:
            labelLines(visible_lines, zorder=1.5)
    else:
        axes.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    _process_options(fig, axes, opts=defaults.matelem_vs_paramvals(specdata), **kwargs)
    return fig, axes
