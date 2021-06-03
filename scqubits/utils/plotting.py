# plotting.py
#
# This file is part of scqubits.
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
import warnings

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import eval_hermite, gamma

import scqubits.core.constants as constants
import scqubits.settings as settings
import scqubits.utils.misc as utils
import scqubits.utils.plot_defaults as defaults

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData, WaveFunction, WaveFunctionOnGrid

try:
    from labellines import labelLines

    _LABELLINES_ENABLED = True
except ImportError:
    _LABELLINES_ENABLED = False


# A dictionary of plotting options that are directly passed to specific matplotlib's
# plot commands.
_direct_plot_options = {
    "plot": ("alpha", "color", "linestyle", "linewidth", "marker", "markersize"),
    "imshow": ("interpolation",),
    "contourf": tuple(),  # empty for now
}


def _extract_kwargs_options(
    kwargs: Dict[str, Any], plot_type: str, direct_plot_options: Dict[str, Any] = None
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
    d = {}
    if plot_type in direct_plot_options:
        for key in kwargs:
            if key in direct_plot_options[plot_type]:
                d[key] = kwargs[key]
    return d


def _process_options(
    figure: Figure, axes: Axes, opts: Dict[str, Any] = None, **kwargs
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
        if key not in functools.reduce(operator.concat, _direct_plot_options.values())
    }  # type: ignore

    option_dict = {**opts, **filtered_kwargs}

    for key, value in option_dict.items():
        if key in defaults.SPECIAL_PLOT_OPTIONS:
            _process_special_option(figure, axes, key, value)
        else:
            set_method = getattr(axes, "set_" + key)
            set_method(value)

    filename = kwargs.get("filename")
    if filename:
        figure.savefig(os.path.splitext(filename)[0] + ".pdf")

    if settings.DESPINE and not axes.name == "3d":
        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")


def _process_special_option(figure: Figure, axes: Axes, key: str, value: Any) -> None:
    """Processes a single 'special' option, i.e., one internal to scqubits and not to be handed further down to
    matplotlib.
    """
    if key == "ymax":
        ymax = value
        ymin, _ = axes.get_ylim()
        ymin = ymin - (ymax - ymin) * 0.05
        axes.set_ylim(ymin, ymax)
    elif key == "figsize":
        figure.set_size_inches(value)
    elif key == "grid":
        axes.grid(**value) if isinstance(value, dict) else axes.grid(value)


def wavefunction1d(
    wavefuncs: Union["WaveFunction", "List[WaveFunction]"],
    potential_vals: np.ndarray = None,
    offset: Union[float, Iterable[float]] = 0,
    scaling: Optional[float] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plots the amplitude of a single real-valued 1d wave function, along with the potential energy if provided.

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

    offset_list = [offset] if not isinstance(offset, (list, np.ndarray)) else offset
    wavefunc_list = [wavefuncs] if not isinstance(wavefuncs, list) else wavefuncs

    scale_constant = renormalization_factor(wavefunc_list[0], potential_vals)
    for wavefunc in wavefunc_list:
        wavefunc.amplitudes *= scale_constant

    scale_factor = scaling or defaults.set_wavefunction_scaling(
        wavefunc_list, potential_vals
    )

    for wavefunction, energy_offset in zip(wavefunc_list, offset_list):
        x_vals = wavefunction.basis_labels
        y_vals = energy_offset + scale_factor * wavefunction.amplitudes
        offset_vals = [energy_offset] * len(x_vals)

        axes.plot(x_vals, y_vals, **_extract_kwargs_options(kwargs, "plot"))
    #        axes.fill_between(
    #            x_vals, y_vals, offset_vals, where=(y_vals != offset_vals), interpolate=True
    #        )

    if potential_vals is not None:
        y_min = np.min(potential_vals)
        y_max = np.max(offset_list)
        y_range = y_max - y_min

        y_max += 0.3 * y_range
        y_min = np.min(potential_vals) - 0.1 * y_range
        axes.set_ylim([y_min, y_max])

        axes.plot(
            x_vals,
            potential_vals,
            color="gray",
            **_extract_kwargs_options(kwargs, "plot")
        )

    _process_options(fig, axes, **kwargs)
    return fig, axes


def renormalization_factor(
    wavefunc: "WaveFunction", potential_vals: np.ndarray
) -> float:
    """
    Takes the amplitudes of one wavefunction and the potential values to scale the
    dimensionless amplitude to a (pseudo-)energy that allows us to plot wavefunctions
    and energies in the same plot.

    Parameters
    ----------
    wavefunc:
        ndarray of wavefunction amplitudes
    potential_vals:
        array of potential energy values (that determine the energy range on the y axis

    Returns
    -------
    renormalization factor that converts the wavefunction amplitudes into energy units
    """
    FILL_FACTOR = 0.1
    energy_range = np.max(potential_vals) - np.min(potential_vals)
    amplitude_range = np.max(wavefunc.amplitudes) - np.min(wavefunc.amplitudes)
    if amplitude_range < 1.0e-10:
        return 0.0
    return FILL_FACTOR * energy_range / amplitude_range


def wavefunction1d_discrete(wavefunc: "WaveFunction", **kwargs) -> Tuple[Figure, Axes]:
    """
    Plots the amplitude of a real-valued 1d wave function in a discrete basis.
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


def wavefunction2d(
    wavefunc: "WaveFunctionOnGrid", zero_calibrate: bool = False, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Creates a density plot of the amplitude of a real-valued wave function in 2
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
        cmap = plt.cm.viridis

    im = axes.imshow(
        wavefunc.amplitudes,
        extent=[min_vals[0], max_vals[0], min_vals[1], max_vals[1]],
        cmap=cmap,
        vmin=imshow_minval,
        vmax=imshow_maxval,
        origin="lower",
        aspect="auto",
        **_extract_kwargs_options(kwargs, "imshow")
    )
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax)

    _process_options(fig, axes, defaults.wavefunction2d(), **kwargs)
    return fig, axes


def contours(
    x_vals: Iterable[float],
    y_vals: Iterable[float],
    func: Callable,
    contour_vals: Iterable[float] = None,
    show_colorbar: bool = True,
    **kwargs
) -> Tuple[Figure, Axes]:
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
        cmap=plt.cm.viridis,
        origin="lower",
        **_extract_kwargs_options(kwargs, "contourf")
    )

    if show_colorbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        fig.colorbar(im, cax=cax)

    _process_options(fig, axes, opts=defaults.contours(x_vals, y_vals), **kwargs)
    return fig, axes


def matrix(
    data_matrix: np.ndarray, mode: str = "abs", show_numbers: bool = False, **kwargs
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Create a "skyscraper" plot and a 2d color-coded plot of a matrix.

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
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = plt.subplot(1, 2, 2)

    fig, ax2 = matrix2d(
        data_matrix, mode=mode, show_numbers=show_numbers, fig_ax=(fig, ax2), **kwargs
    )
    fig, ax1 = matrix_skyscraper(data_matrix, mode=mode, fig_ax=(fig, ax1), **kwargs)
    return fig, (ax1, ax2)


def matrix_skyscraper(
    matrix: np.ndarray, mode: str = "abs", **kwargs
) -> Tuple[Figure, Axes]:
    """Display a 3d skyscraper plot of the matrix

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

    modefunction = constants.MODE_FUNC_DICT[mode]
    zheight = modefunction(matrix).flatten()  # height of bars from matrix elements

    min_zheight, max_zheight = min(zheight), max(zheight)

    if mode == "abs" or mode == "abs_sqr":
        nrm = mpl.colors.Normalize(
            0, max_zheight
        )  # normalize colors between 0 and max. data
    else:
        nrm = mpl.colors.Normalize(
            min_zheight, max_zheight
        )  # normalize colors between min. and max. of data

    colors = plt.cm.viridis(nrm(zheight))  # list of colors for each bar

    # skyscraper plot
    axes.view_init(azim=210, elev=23)
    axes.bar3d(xgrid, ygrid, zbottom, dx, dy, zheight, color=colors)

    if mode == "abs" or mode == "abs_sqr":
        min_z, max_z = 0, max_zheight
    else:  # mode is "real" or "imag"
        min_z = 0 if min_zheight > 0 else min_zheight
        max_z = 0 if max_zheight < 0 else max_zheight

    if min_z == max_z:
        # pad with small values so we don't get warnings
        max_z += 0.0000001

    axes.set_zlim3d([min_z, max_z])

    for axis, locs in [
        (axes.xaxis, np.arange(x_count)),
        (axes.yaxis, np.arange(y_count)),
    ]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs + 0.5, ticklabels=locs)

    _process_options(fig, axes, opts=defaults.matrix(), **kwargs)

    return fig, axes


def matrix2d(
    matrix: np.ndarray, mode: str = "abs", show_numbers: bool = True, **kwargs
) -> Tuple[Figure, Axes]:
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
    **kwargs:
        standard plotting option (see separate documentation)

    Returns
    -------
        figure and axes objects for further editing
    """
    fig, axes = kwargs.get("fig_ax") or plt.subplots()

    modefunction = constants.MODE_FUNC_DICT[mode]
    zheight = modefunction(matrix).flatten()  # height of bars from matrix elements

    if mode == "abs" or mode == "abs_sqr":
        nrm = mpl.colors.Normalize(
            0, max(zheight)
        )  # normalize colors between 0 and max. data
    else:
        nrm = mpl.colors.Normalize(
            min(zheight), max(zheight)
        )  # normalize colors between min. and max. of data

    axes.matshow(modefunction(matrix), cmap=plt.cm.viridis, interpolation=None)
    cax, _ = mpl.colorbar.make_axes(
        axes, shrink=0.75, pad=0.02
    )  # add colorbar with normalized range
    mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.viridis, norm=nrm)

    if show_numbers:
        for y_index in range(matrix.shape[0]):
            for x_index in range(matrix.shape[1]):
                axes.text(
                    x_index,
                    y_index,
                    "{:.02f}".format(modefunction(matrix[y_index, x_index])),
                    va="center",
                    ha="center",
                    fontsize=8,
                    rotation=45,
                    color="white",
                )
    # shift the grid
    for axis, locs in [
        (axes.xaxis, np.arange(matrix.shape[1])),
        (axes.yaxis, np.arange(matrix.shape[0])),
    ]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs, ticklabels=locs)
    axes.grid(False)

    _process_options(fig, axes, **kwargs)
    axes.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)

    return fig, axes


def data_vs_paramvals(
    xdata: np.ndarray,
    ydata: np.ndarray,
    label_list: Union[List[str], List[int]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Plot of a set of yadata vs xdata.
    The individual points correspond to the a provided array of parameter values.

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

    if label_list is None:
        axes.plot(xdata, ydata, **_extract_kwargs_options(kwargs, "plot"))
    else:
        for idx, ydataset in enumerate(ydata.T):
            axes.plot(
                xdata,
                ydataset,
                label=label_list[idx],
                **_extract_kwargs_options(kwargs, "plot")
            )
        if _LABELLINES_ENABLED:
            try:
                labelLines(axes.get_lines(), zorder=2.0)
            except Exception:
                pass
        else:
            axes.legend(
                bbox_to_anchor=(1.04, 0.5),
                loc="center left",
                borderaxespad=0,
                frameon=False,
            )
            # legend(loc="center left", bbox_to_anchor=(1, 0.5))
    _process_options(fig, axes, **kwargs)
    return fig, axes


def evals_vs_paramvals(
    specdata: "SpectrumData",
    which: Union[int, Iterable[int]] = -1,
    subtract_ground: bool = False,
    label_list: List[str] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
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

    xdata = specdata.param_vals
    ydata = specdata.energy_table[:, index_list]
    if subtract_ground:
        ydata = (ydata.T - ydata[:, 0]).T

    return data_vs_paramvals(
        xdata,
        ydata,
        label_list=label_list,
        **defaults.evals_vs_paramvals(specdata, **kwargs)
    )


def matelem_vs_paramvals(
    specdata: "SpectrumData",
    select_elems: Union[int, List[Tuple[int, int]]] = 4,
    mode: str = "abs",
    **kwargs
) -> Tuple[Figure, Axes]:
    """Generates a simple plot of matrix elements as a function of one parameter.
    The individual points correspond to the a provided array of parameter values.

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
    fig, axes = kwargs.get("fig_ax") or plt.subplots()
    x = specdata.param_vals
    modefunction = constants.MODE_FUNC_DICT[mode]

    if isinstance(select_elems, int):
        index_pairs = [
            (row, col) for row in range(select_elems) for col in range(row + 1)
        ]
    else:
        index_pairs = select_elems

    for (row, col) in index_pairs:
        y = modefunction(specdata.matrixelem_table[:, row, col])
        axes.plot(
            x,
            y,
            label=str(row) + "," + str(col),
            **_extract_kwargs_options(kwargs, "plot")
        )

    if _LABELLINES_ENABLED:
        labelLines(axes.get_lines(), zorder=1.5)
    else:
        axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    _process_options(fig, axes, opts=defaults.matelem_vs_paramvals(specdata), **kwargs)
    return fig, axes


def harm_osc_wavefunction(n, x):
    """For given quantum number n=0,1,2,... return the value of the harmonic oscillator wave function
    :math:`\\psi_n(x) = N H_n(x) \\exp(-x^2/2)`, N being the proper normalization factor. It is assumed
    that the harmonic length has already been accounted for. Therefore that portion of the normalization
    factor must be accounted for outside the function.
    Parameters
    ----------
    n: int
        index of wave function, n=0 is ground state
    x: float or ndarray
        coordinate(s) where wave function is evaluated
    Returns
    -------
    float or ndarray
        value(s) of harmonic oscillator wave function
    """
    return (
        (2.0 ** n * gamma(n + 1.0)) ** (-0.5)
        * np.pi ** (-0.25)
        * eval_hermite(n, x)
        * np.exp(-(x ** 2) / 2.0)
    )


def multiply_two_harm_osc_functions(n1, n2, x1, x2):
    """Useful for plotting 2D wavefunctions using harmonic oscillator states. Assumes x1 and x2
    are arrays with the same dimensionality"""
    return np.multiply(harm_osc_wavefunction(n1, x1), harm_osc_wavefunction(n2, x2))
