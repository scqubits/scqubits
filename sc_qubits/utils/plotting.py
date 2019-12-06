# plotting.py
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.backends.backend_pdf as mplpdf
import matplotlib.pyplot as plt

try:
    from labellines import labelLine, labelLines
    _labellines_enabled = True
except ImportError:
    _labellines_enabled = False

import sc_qubits.utils.constants as constants


mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['figure.dpi']= 150


def wavefunction1d(wavefunc, potential_vals=None, offset=0, scaling=1, ylabel='wavefunction', xlabel='x',
                   yrange=None, fig_ax=None, filename=None, **kwargs):
    """
    Plots the amplitude of a real-valued 1d wave function, along with the potential energy if provided.

    Parameters
    ----------
    wavefunc: WaveFunction object
        basis and amplitude data of wave function to be plotted
    potential_vals: array of float
        potential energies, array length must match basis array of `wavefunc`
    offset: float
        y-offset for the wave function (e.g., shift by eigenenergy)
    scaling: float
        scaling factor for wave function amplitudes
    ylabel: str
        y-axis label
    xlabel: str
        x-axis label
    yrange: (float, float)
        plot range for y-axis
    fig_ax: None or tuple(Figure, Axes)
        fig and ax objects for matplotlib figure addition
    **kwargs:
        keyword arguments passed on to axes.plot()

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    if fig_ax is None:
        fig, axes = plt.subplots()
    else:
        fig, axes = fig_ax

    x_vals = wavefunc.basis_labels

    axes.plot(x_vals, offset + scaling * wavefunc.amplitudes, **kwargs)
    if potential_vals is not None:
        axes.plot(x_vals, potential_vals)
        axes.plot(x_vals, [offset] * len(x_vals), 'b--')

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.set_xlim(left=x_vals[0], right=x_vals[-1])
    if yrange is not None:
        axes.set_ylim(*yrange)

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()

    return fig, axes


def wavefunction1d_discrete(wavefunc, xrange, xlabel='x', ylabel='wavefunction', filename=None, fig_ax=None, **kwargs):
    """
    Plots the amplitude of a real-valued 1d wave function in a discrete basis. (Example: transmon in the charge basis.)

    Parameters
    ----------
    wavefunc: WaveFunction object
        basis and amplitude data of wave function to be plotted
    xrange: tupel(int, int)
        lower and upper bound for values on the x axis
    xlabel: str
        x-axis label
    ylabel: str
        y-axis label
    filename: None or str
        file path and name (not including suffix)
    fig_ax: None or tuple(Figure, Axes)
        fig and ax objects for matplotlib figure addition
    **kwargs:
        keyword arguments passed on to axes.plot()


    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    if fig_ax is None:
        fig, axes = plt.subplots()
    else:
        fig, axes = fig_ax

    x_vals = wavefunc.basis_labels
    width = .75

    axes.bar(x_vals, wavefunc.amplitudes, width=width, **kwargs)
    axes.set_xticks(x_vals + width / 2)
    axes.set_xticklabels(x_vals)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xrange)

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()

    return fig, axes


def wavefunction2d(wavefunc, figsize=(10, 5), aspect_ratio=3, zero_calibrate=False, filename=None, fig_ax=None):
    """
    Creates a density plot of the amplitude of a real-valued wave function in 2 "spatial" dimensions.

    Parameters
    ----------
    wavefunc: WaveFunctionOnGrid object
        basis and amplitude data of wave function to be plotted
    figsize: tuple(float, float)
        width, height in inches
    aspect_ratio: float
        aspect ratio
    zero_calibrate: bool
        whether to calibrate plot to zero amplitude
    filename: None or str
        file path and name (not including suffix)
    fig_ax: None or tuple(Figure, Axes)
        fig and ax objects for matplotlib figure addition

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    if fig_ax is None:
        fig, axes = plt.subplots(figsize=figsize)
    else:
        fig, axes = fig_ax

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
    
    m = axes.imshow(wavefunc.amplitudes, extent=[min_vals[0], max_vals[0], min_vals[1], max_vals[1]],
                    aspect=aspect_ratio, cmap=cmap, vmin=imshow_minval, vmax=imshow_maxval, origin='lower')
    cbar = fig.colorbar(m, ax=axes)

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()

    return fig, axes


def contours(x_vals, y_vals, func, contour_vals=None, aspect_ratio=None, show_colorbar=True, filename=None,
             fig_ax=None):
    """Contour plot of a 2d function `func(x,y)`.

    Parameters
    ----------
    x_vals: (ordered) list
        x values for the x-y evaluation grid
    y_vals: (ordered) list
        y values for the x-y evaluation grid
    func: function f(x,y)
        function for which contours are to be plotted
    contour_vals: list
        contour values can be specified if so desired
    aspect_ratio: float
    show_colorbar: bool
    filename: None or str
        file path and name (not including suffix)
    fig_ax: None or tuple(Figure, Axes)
        fig and ax objects for matplotlib figure addition

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_array = func(x_grid, y_grid)

    if fig_ax is None:
        if aspect_ratio is None:
            aspect_ratio = (y_vals[-1] - y_vals[0])/(x_vals[-1] - x_vals[0])
        w, h = plt.figaspect(aspect_ratio)
        fig, axes = plt.subplots(figsize=(w, h))
    else:
        fig, axes = fig_ax

    im = axes.contourf(x_grid, y_grid, z_array, levels=contour_vals, cmap=plt.cm.viridis)

    if show_colorbar:
        fig.colorbar(im, ax=axes)

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()

    return fig, axes


def matrix(data_matrix, mode='abs', xlabel='', ylabel='', zlabel='', filename=None, fig_ax=None):
    """
    Create a "skyscraper" plot and a 2d color-coded plot of a matrix.

    Parameters
    ----------
    data_matrix: ndarray of float or complex
        2d matrix data
    mode: str from `constants.MODE_FUNC_DICT`
        choice of processing function to be applied to data
    xlabel: str
    ylabel: str
    zlabel: str
    filename: None or str
        file path and name (not including suffix)
    fig_ax: None or tuple(Figure, (Axes, Axes))
        fig and ax objects for matplotlib figure addition

    Returns
    -------
    Figure, (Axes1, Axes2)
        figure and axes objects for further editing
    """
    if fig_ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = plt.subplot(1, 2, 2)
    else:
        fig, (ax1, ax2) = fig_ax

    matsize = len(data_matrix)
    element_count = matsize ** 2  # num. of elements to plot

    xgrid, ygrid = np.meshgrid(range(matsize), range(matsize))
    xgrid = xgrid.T.flatten() - 0.5  # center bars on integer value of x-axis
    ygrid = ygrid.T.flatten() - 0.5  # center bars on integer value of y-axis

    zbottom = np.zeros(element_count)  # all bars start at z=0
    dx = 0.75 * np.ones(element_count)  # width of bars in x-direction
    dy = dx  # width of bars in y-direction (same as x-direction)

    modefunction = constants.MODE_FUNC_DICT[mode]
    zheight = modefunction(data_matrix).flatten()  # height of bars from matrix elements
    nrm = mpl.colors.Normalize(0, max(zheight))  # <-- normalize colors to max. data
    colors = plt.cm.viridis(nrm(zheight))  # list of colors for each bar

    # skyscraper plot

    ax1.view_init(azim=210, elev=23)
    ax1.bar3d(xgrid, ygrid, zbottom, dx, dy, zheight, color=colors)
    ax1.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))  # set x-ticks to integers
    ax1.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))  # set y-ticks to integers
    ax1.set_zlim3d([0, max(zheight)])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_zlabel(zlabel)

    # 2d plot

    ax2.matshow(modefunction(data_matrix), cmap=plt.cm.viridis)

    cax, kw = mpl.colorbar.make_axes(ax2, shrink=.75, pad=.02)  # add colorbar with normalized range
    _ = mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.viridis, norm=nrm)

    plt.show()

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()

    return fig, (ax1, ax2)


def evals_vs_paramvals(specdata, evals_count=-1, xlim=False, ylim=False, shift=0, filename=None,
                       fig_ax=None, **kwargs):
    """Generates a simple plot of a set of eigenvalues as a function of one parameter.
    The individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    specdata: SpectrumData
        object includes parameter name, values, and resulting eigenenergies
    evals_count: int
        number of desired eigenvalues (sorted from smallest to largest)
    xlim: (float, float)
        custom x-range for the plot
    ylim: (float, float)
        custom y-range for the plot
    shift: float
        apply a shift of this size to all eigenvalues
    filename: str
        write graphics and parameter set to file if path and filename are specified
    fig_ax: None or tuple(Figure, Axes)
        fig and ax objects for matplotlib figure addition
    **kwargs:
        keyword arguments passed on to axes.plot()

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    if fig_ax is None:
        fig, axes = plt.subplots()
    else:
        fig, axes = fig_ax

    x = specdata.param_vals
    y = specdata.energy_table[:, 0:evals_count]
    if xlim:
        axes.set_xlim(*xlim)
    else:
        axes.set_xlim(np.amin(x), np.amax(x))

    if ylim:
        axes.set_ylim(*ylim)
    else:
        axes.set_ylim(np.amin(y + shift), np.amax(y + shift))

    axes.set_xlabel(specdata.param_name)
    axes.set_ylabel('energy')
    axes.plot(x, y + shift, **kwargs)

    if filename:
        out_file = mplpdf.PdfPages(filename + '.pdf')
        out_file.savefig()
        out_file.close()
    plt.show()

    return fig, axes


def matelem_vs_paramvals(specdata, select_elems=4, mode='abs', xlim=False, ylim=False, filename=None,
                         fig_ax=None, **kwargs):
    """Generates a simple plot of matrix elements as a function of one parameter.
    The individual points correspond to the a provided array of parameter values.

    Parameters
    ----------
    specdata: SpectrumData
        object includes parameter name, values, and matrix elements
    select_elems: int or list
        either maximum index of desired matrix elements, or list [(i1, i2), (i3, i4), ...] of index tuples
        for specific desired matrix elements
    mode: str from `constants.MODE_FUNC_DICT`
        choice of processing function to be applied to data
    xlim: (float, float)
        custom x-range for the plot
    ylim: (float, float)
        custom y-range for the plot
    shift: float
        apply a shift of this size to all eigenvalues
    filename: str
        write graphics and parameter set to file if path and filename are specified
    fig_ax: None or tuple(Figure, Axes)
        fig and ax objects for matplotlib figure addition
    **kwargs:
        keyword arguments passed on to axes.plot()

    Returns
    -------
    tuple(Figure, Axes)
        matplotlib objects for further editing
    """
    if fig_ax is None:
        fig, axes = plt.subplots()
    else:
        fig, axes = fig_ax

    if xlim:
        axes.set_xlim(*xlim)
    if ylim:
        axes.set_ylim(*ylim)
    axes.set_xlabel(specdata.param_name)
    axes.set_ylabel('matrix element')

    modefunction = constants.MODE_FUNC_DICT[mode]
    x = specdata.param_vals

    if isinstance(select_elems, int):
        for row in range(select_elems):
            for col in range(row + 1):
                y = modefunction(specdata.matrixelem_table[:, row, col])
                axes.plot(x, y, label=str(row)+','+str(col), **kwargs)
    else:
        for index_pair in select_elems:
            y = modefunction(specdata.matrixelem_table[:, index_pair[0], index_pair[1]])
            axes.plot(x, y, label=str(index_pair[0]) + ',' + str(index_pair[1]), **kwargs)

    if _labellines_enabled:
        labelLines(axes.get_lines(), zorder=2.5)
    else:
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if filename:
        out_file = mplpdf.PdfPages(filename + '.pdf')
        out_file.savefig()
        out_file.close()

    plt.show()

    return fig, axes


def print_matrix(matrix, show_numbers=True, fig_ax=None, **kwargs):
    """Pretty print a matrix, optionally printing the numerical values of the data.   
    """
    if fig_ax is None:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, axes = fig_ax

    m = axes.matshow(matrix, cmap=plt.cm.viridis, interpolation='none', **kwargs)
    cbar = fig.colorbar(m, ax=axes)
    
    if show_numbers:
        for y_index in range(matrix.shape[0]):
            for x_index in range(matrix.shape[1]):
                axes.text(x_index, y_index, "{:.03f}".format(matrix[y_index, x_index]),
                          va='center', ha='center', fontsize=8, rotation=45, color='white')
    # shift the grid
    for axis, locs in [(axes.xaxis, np.arange(matrix.shape[1])), (axes.yaxis, np.arange(matrix.shape[0]))]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs, ticklabels=locs)
    axes.grid(True, which='minor')
    axes.grid(False, which='major')

    return fig, axes


def spectrum_with_matrixelement(spectrum_data, matrixelement_table, param_name='external parameter',
                                energy_name='energy', matrixelement_name='matrix element', norm_range=None,
                                x_range=None, y_range=None, colormap='jet', figsize=(15, 10), line_width=2):
    """Takes a list of x-values,
    a list of lists with each element containing the y-values corresponding to a particular curve,
    a list of lists with each element containing the external parameter value (t-value)
    that determines the color of each curve at each y-value,
    and a normalization interval for the t-values."""
    fig = plt.figure(figsize=figsize)

    if norm_range is None:
        norm_range = (np.min(matrixelement_table), np.max(matrixelement_table))

    for i in range(len(spectrum_data.energy_table[0])):
        pts = np.array([spectrum_data.param_vals, spectrum_data.energy_table[:, i]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        line_collection = mpl.collections.LineCollection(segs, cmap=plt.get_cmap(colormap),
                                                         norm=plt.Normalize(*norm_range))
        line_collection.set_array(matrixelement_table[:, i])
        line_collection.set_linewidth(line_width)
        plt.gca().add_collection(line_collection)

    plt.xlabel(param_name)
    plt.ylabel(energy_name)
    if not x_range:
        x_range = [np.amin(spectrum_data.param_vals), np.amax(spectrum_data.param_vals)]
    if not y_range:
        y_range = [np.amin(spectrum_data.energy_table), np.max(spectrum_data.energy_table)]

    plt.xlim(*x_range)
    plt.ylim(*y_range)

    axcb = fig.colorbar(line_collection)
    axcb.set_label(matrixelement_name)
    plt.show()
