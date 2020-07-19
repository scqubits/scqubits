# plotting.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import functools
import operator

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scqubits.core.constants as constants
import scqubits.utils.misc as utils
import scqubits.utils.plot_defaults as defaults
import scqubits.settings as settings

try:
    from labellines import labelLines
    _LABELLINES_ENABLED = True
except ImportError:
    _LABELLINES_ENABLED = False


# A dictionary of plotting options that are directly passed to specific matplotlib's
# plot commands.
_direct_plot_options = {
        'plot': ('alpha', 'color', 'linestyle', 'linewidth', 'marker', 'markersize'),
        'imshow': ('interpolation',),
        'contourf': tuple()  # empty for now
    }


def _extract_kwargs_options(kwargs, plot_type, direct_plot_options=None):
    """
    Select options from kwargs for a given plot_type and return them in a dictionary.
    
    Parameters
    ----------
    kwargs: dict
        dictionary with options that can be passed to different plotting commands
    plot_type: str
        a type of plot for which the options should be selected
    direct_plot_options: dict
        a lookup dictionary with supported options for a given plot_type
        
    Returns
    ----------
    dict
        dictionary with key/value pairs corresponding to selected options from kwargs

    """
    direct_plot_options = direct_plot_options or _direct_plot_options
    d = {}
    if plot_type in direct_plot_options:
        for key in kwargs:  
            if key in direct_plot_options[plot_type]:
                d[key] = kwargs[key]
    return d
    

def _process_options(figure, axes, opts=None, **kwargs):
    """
    Processes plotting options.

    Parameters
    ----------
    figure: matplotlib.Figure
    axes: matplotlib.Axes
    opts: dict
        keyword dictionary with custom options
    **kwargs: dict
        standard plotting option (see separate documentation)
    """
    opts = opts or {}

    # Only process items in kwargs that would not have been
    # processed through _extract_kwargs_options()
    filtered_kwargs = {key: value for key, value in kwargs.items()
                       if key not in functools.reduce(operator.concat, _direct_plot_options.values())}

    option_dict = {**opts, **filtered_kwargs}

    for key, value in option_dict.items():
        if key in defaults.SPECIAL_PLOT_OPTIONS:
            _process_special_option(figure, axes, key, value)
        else:
            set_method = getattr(axes, 'set_' + key)
            set_method(value)
 
    filename = kwargs.get('filename')
    if filename:
        figure.savefig(os.path.splitext(filename)[0] + '.pdf')

    if settings.DESPINE and not axes.name == '3d':
        # Hide the right and top spines
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position('left')
        axes.xaxis.set_ticks_position('bottom')


def _process_special_option(figure, axes, key, value):
    """Processes a single 'special' option, i.e., one internal to scqubits and not to be handed further down to
    matplotlib.

    Parameters
    ----------
    figure: matplotlib.Figure
    axes: matplotlib.Axes
    key: str
    value: anything
    """
    if key == 'x_range':
        warnings.warn('x_range is deprecated, use xlim instead', FutureWarning)
        axes.set_xlim(value)
    elif key == 'y_range':
        warnings.warn('y_range is deprecated, use ylim instead', FutureWarning)
        axes.set_ylim(value)
    elif key == 'ymax':
        ymax = value
        ymin, _ = axes.get_ylim()
        ymin = ymin - (ymax - ymin) * 0.05
        axes.set_ylim(ymin, ymax)
    elif key == 'figsize':
        figure.set_size_inches(value)
    elif key == 'grid':
        axes.grid(**value) if isinstance(value, dict) else axes.grid(value)


def wavefunction1d(wavefunc, potential_vals=None, offset=0, scaling=1, **kwargs):
    """
    Plots the amplitude of a single real-valued 1d wave function, along with the potential energy if provided.

    Parameters
    ----------
    wavefunc: WaveFunction object
        basis and amplitude data of wave function to be plotted
    potential_vals: array of float
        potential energies, array length must match basis array of `wavefunc`
    offset: float
        y-offset for the wave function (e.g., shift by eigenenergy)
    scaling: float, optional
        scaling factor for wave function amplitudes
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots()

    x_vals = wavefunc.basis_labels
    y_vals = offset + scaling * wavefunc.amplitudes
    offset_vals = [offset] * len(x_vals)

    if potential_vals is not None:
        axes.plot(x_vals, potential_vals, color='gray', **_extract_kwargs_options(kwargs, 'plot'))

    axes.plot(x_vals, y_vals, **_extract_kwargs_options(kwargs, 'plot'))
    axes.fill_between(x_vals, y_vals, offset_vals, where=(y_vals != offset_vals), interpolate=True)
    _process_options(fig, axes, **kwargs)
    return fig, axes


def wavefunction1d_discrete(wavefunc, **kwargs):
    """
    Plots the amplitude of a real-valued 1d wave function in a discrete basis. (Example: transmon in the charge basis.)

    Parameters
    ----------
    wavefunc: WaveFunction object
        basis and amplitude data of wave function to be plotted
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots()

    x_vals = wavefunc.basis_labels
    width = .75
    axes.bar(x_vals, wavefunc.amplitudes, width=width)

    axes.set_xticks(x_vals)
    axes.set_xticklabels(x_vals)
    _process_options(fig, axes, defaults.wavefunction1d_discrete(), **kwargs)

    return fig, axes


def wavefunction2d(wavefunc, zero_calibrate=False, **kwargs):
    """
    Creates a density plot of the amplitude of a real-valued wave function in 2 "spatial" dimensions.

    Parameters
    ----------
    wavefunc: WaveFunctionOnGrid object
        basis and amplitude data of wave function to be plotted
    zero_calibrate: bool, optional
        whether to calibrate plot to zero amplitude
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots()

    min_vals = wavefunc.gridspec.min_vals
    max_vals = wavefunc.gridspec.max_vals

    if zero_calibrate:
        absmax = np.amax(np.abs(wavefunc.amplitudes))
        imshow_minval = -absmax
        imshow_maxval = absmax
        cmap = plt.get_cmap('PRGn')
    else:
        imshow_minval = np.min(wavefunc.amplitudes)
        imshow_maxval = np.max(wavefunc.amplitudes)
        cmap = plt.cm.viridis

    im = axes.imshow(wavefunc.amplitudes, extent=[min_vals[0], max_vals[0], min_vals[1], max_vals[1]],
                     cmap=cmap, vmin=imshow_minval, vmax=imshow_maxval, origin='lower', aspect='auto',
                     **_extract_kwargs_options(kwargs, 'imshow'))
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax)

    _process_options(fig, axes, defaults.wavefunction2d(), **kwargs)
    return fig, axes


def contours(x_vals, y_vals, func, contour_vals=None, show_colorbar=True, **kwargs):
    """Contour plot of a 2d function `func(x,y)`.

    Parameters
    ----------
    x_vals: (ordered) list
        x values for the x-y evaluation grid
    y_vals: (ordered) list
        y values for the x-y evaluation grid
    func: function f(x,y)
        function for which contours are to be plotted
    contour_vals: list of float, optional
        contour values can be specified if so desired
    show_colorbar: bool, optional
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots()

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_array = func(x_grid, y_grid)

    im = axes.contourf(x_grid, y_grid, z_array, levels=contour_vals, cmap=plt.cm.viridis, origin="lower",
                       **_extract_kwargs_options(kwargs, 'contourf'))

    if show_colorbar:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        fig.colorbar(im, cax=cax)

    _process_options(fig, axes, opts=defaults.contours(x_vals, y_vals), **kwargs)
    return fig, axes


def matrix(data_matrix, mode='abs', show_numbers=False, **kwargs):
    """
    Create a "skyscraper" plot and a 2d color-coded plot of a matrix.

    Parameters
    ----------
    data_matrix: ndarray of float or complex
        2d matrix data
    mode: str from `constants.MODE_FUNC_DICT`
        choice of processing function to be applied to data
    show_numbers: bool, optional
        determines whether matrix element values are printed on top of the plot (default: False)
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, (Axes1, Axes2)
        figure and axes objects for further editing
    """
    if 'fig_ax' in kwargs:
        fig, (ax1, ax2) = kwargs['fig_ax']
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = plt.subplot(1, 2, 2)

    fig, ax2 = matrix2d(data_matrix, mode=mode, show_numbers=show_numbers, fig_ax=(fig, ax2), **kwargs)
    fig, ax1 = matrix_skyscraper(data_matrix, mode=mode, fig_ax=(fig, ax1), **kwargs)
    return fig, (ax1, ax2)


def matrix_skyscraper(matrix, mode='abs', **kwargs):
    """Display a 3d skyscraper plot of the matrix

    Parameters
    ----------
    matrix: ndarray of float or complex
        2d matrix data
    mode: str from `constants.MODE_FUNC_DICT`
        choice of processing function to be applied to data
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
        figure and axes objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots(projection='3d')

    matsize = len(matrix)
    element_count = matsize ** 2  # num. of elements to plot

    xgrid, ygrid = np.meshgrid(range(matsize), range(matsize))
    xgrid = xgrid.T.flatten() - 0.5  # center bars on integer value of x-axis
    ygrid = ygrid.T.flatten() - 0.5  # center bars on integer value of y-axis

    zbottom = np.zeros(element_count)  # all bars start at z=0
    dx = 0.75 * np.ones(element_count)  # width of bars in x-direction
    dy = dx  # width of bars in y-direction (same as x-direction)

    modefunction = constants.MODE_FUNC_DICT[mode]
    zheight = modefunction(matrix).flatten()  # height of bars from matrix elements
    nrm = mpl.colors.Normalize(0, max(zheight))  # <-- normalize colors to max. data
    colors = plt.cm.viridis(nrm(zheight))  # list of colors for each bar

    # skyscraper plot
    axes.view_init(azim=210, elev=23)
    axes.bar3d(xgrid, ygrid, zbottom, dx, dy, zheight, color=colors)
    axes.axes.xaxis.set_major_locator(plt.IndexLocator(1, -0.5))  # set x-ticks to integers
    axes.axes.yaxis.set_major_locator(plt.IndexLocator(1, -0.5))  # set y-ticks to integers
    axes.set_zlim3d([0, max(zheight)])

    _process_options(fig, axes, opts=defaults.matrix(), **kwargs)
    return fig, axes


def matrix2d(matrix, mode='abs', show_numbers=True, **kwargs):
    """Display a matrix as a color-coded 2d plot, optionally printing the numerical values of the matrix elements.

    Parameters
    ----------
    matrix: ndarray of float or complex
        2d matrix data
    mode: str from `constants.MODE_FUNC_DICT`
        choice of processing function to be applied to data
    show_numbers: bool, optional
        determines whether matrix element values are printed on top of the plot (default: True)
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    Figure, Axes
        figure and axes objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots()

    modefunction = constants.MODE_FUNC_DICT[mode]
    zheight = modefunction(matrix).flatten()  # height of bars from matrix elements
    nrm = mpl.colors.Normalize(0, max(zheight))  # <-- normalize colors to max. data

    axes.matshow(modefunction(matrix), cmap=plt.cm.viridis, interpolation=None)
    cax, _ = mpl.colorbar.make_axes(axes, shrink=.75, pad=.02)  # add colorbar with normalized range
    mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.viridis, norm=nrm)

    if show_numbers:
        for y_index in range(matrix.shape[0]):
            for x_index in range(matrix.shape[1]):
                axes.text(x_index, y_index, "{:.03f}".format(matrix[y_index, x_index]),
                          va='center', ha='center', fontsize=8, rotation=45, color='white')
    # shift the grid
    for axis, locs in [(axes.xaxis, np.arange(matrix.shape[1])), (axes.yaxis, np.arange(matrix.shape[0]))]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs, ticklabels=locs)
    axes.grid(True, which='minor', linewidth=0)
    axes.grid(False, which='major', linewidth=0)

    _process_options(fig, axes, **kwargs)
    return fig, axes


print_matrix = matrix2d  # legacv, support of name now deprecated


def data_vs_paramvals(xdata, ydata, label_list=None, **kwargs):
    """Plot of a set of yadata vs xdata.
    The individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    xdata, ydata: ndarray
        must have compatible shapes for matplotlib.pyplot.plot
    label_list: list(str), optional
        list of labels associated with the individual curves to be plotted
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    fig, axes = kwargs.get('fig_ax') or plt.subplots()
 
    if label_list is None: 
        axes.plot(xdata, ydata, **_extract_kwargs_options(kwargs, 'plot'))
    else:
        for idx, ydataset in enumerate(ydata.T):
            axes.plot(xdata, ydataset, label=label_list[idx], **_extract_kwargs_options(kwargs, 'plot'))
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _process_options(fig, axes, **kwargs)
    return fig, axes


def evals_vs_paramvals(specdata, which=-1, subtract_ground=False, label_list=None, **kwargs):
    """Generates a simple plot of a set of eigenvalues as a function of one parameter.
    The individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    specdata: SpectrumData
        object includes parameter name, values, and resulting eigenenergies
    which: int or list(int)
        number of desired eigenvalues (sorted from smallest to largest); default: -1, signals all eigenvalues
        or: list of specific eigenvalues to include
    subtract_ground: bool
        whether to subtract the ground state energy
    label_list: list(str), optional
        list of labels associated with the individual curves to be plotted
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    index_list = utils.process_which(which, specdata.energy_table[0].size)

    xdata = specdata.param_vals
    ydata = specdata.energy_table[:, index_list]
    if subtract_ground:
        ydata = (ydata.T - ydata[:, 0]).T
    return data_vs_paramvals(xdata, ydata, label_list=label_list,
                             **defaults.evals_vs_paramvals(specdata, **kwargs))


def matelem_vs_paramvals(specdata, select_elems=4, mode='abs', **kwargs):
    """Generates a simple plot of matrix elements as a function of one parameter.
    The individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    specdata: SpectrumData
        object includes parameter name, values, and matrix elements
    select_elems: int or list
        either maximum index of desired matrix elements, or list [(i1, i2), (i3, i4), ...] of index tuples
        for specific desired matrix elements
    mode: str from `constants.MODE_FUNC_DICT`, optional
        choice of processing function to be applied to data (default value = 'abs')
    **kwargs: dict
        standard plotting option (see separate documentation)

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    def request_range(sel_elems):
        return isinstance(sel_elems, int)

    fig, axes = kwargs.get('fig_ax') or plt.subplots()
    x = specdata.param_vals
    modefunction = constants.MODE_FUNC_DICT[mode]

    if request_range(select_elems):
        index_pairs = [(row, col) for row in range(select_elems) for col in range(row + 1)]
    else:
        index_pairs = select_elems

    for (row, col) in index_pairs:
        y = modefunction(specdata.matrixelem_table[:, row, col])
        axes.plot(x, y, label=str(row) + ',' + str(col), **_extract_kwargs_options(kwargs, 'plot'))

    if _LABELLINES_ENABLED:
        labelLines(axes.get_lines(), zorder=1.5)
    else:
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _process_options(fig, axes, opts=defaults.matelem_vs_paramvals(specdata), **kwargs)
    return fig, axes
