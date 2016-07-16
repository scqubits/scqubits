# sc_qubits.py
"""
The 'sc_qubits' module provides routines for the basic description of common superconducting qubits such as
the Cooper pair box/transmon, fluxonium etc. Each qubit is realized as a class, providing relevant
methods such as calculating eigenvalues and eigenvectors, or plotting the energy spectrum vs. a select
external parameter.
"""

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf as mplpdf
import mpl_toolkits.mplot3d as mpl3d
import numpy as np
import scipy as sp
from scipy import sparse, linalg
from scipy.sparse import linalg
import math
import cmath
import itertools
import qutip as qt
import sys

import operators as op

EVECS_FILESUFFIX = '_evecs'
EVALS_FILESUFFIX = '_evals'
PARAMETER_FILESUFFIX = '.prm'

MODE_FUNC_DICT = {'abs2': (lambda x: np.abs(x)**2),
                  'abs': (lambda x: np.abs(x)),
                  'real': (lambda x: np.real(x)),
                  'imag': (lambda x: np.imag(x))}

PHI_INDEX = 0
THETA_INDEX = 1
CHI_INDEX = 2


class HilbertSpace(object):

    def __init__(self, subsystem_list):
        self.subsystem_list = subsystem_list
        self.subsystem_count = len(subsystem_list)
        self.subsystem_dims = [subsystem.truncated_dim for subsystem in subsystem_list]
        self.dimension = np.sum(np.asarray(self.subsystem_dims))

    def __repr__(self):
        output = '====== HILBERT SPACE OBJECT ======'
        for parameter_name in self.__dict__.keys():
            parameter_val = self.__dict__[parameter_name]
            output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val) + '\n'
        return output

    def diag_hamiltonian(self, subsystem):
        """Returns a qt.Qobj which has the eigenenergies of the object 'subsystem' on the diagonal."""
        evals_count = subsystem.truncated_dim
        evals = subsystem.eigenvals(evals_count=evals_count)
        diag_qt_op = qt.Qobj(inpt=np.diagflat(evals))
        return self.identity_wrap(diag_qt_op, subsystem)

    def identity_wrap(self, operator, subsystem):
        if type(operator) in [list, np.ndarray]:
            dim = subsystem.truncated_dim
            subsys_operator = qt.Qobj(inpt=operator[:dim, :dim])
        else:
            subsys_operator = operator
        operator_identitywrap_list = [qt.operators.qeye(sys.truncated_dim) for sys in self.subsystem_list]
        subsystem_index = self.subsystem_list.index(subsystem)
        operator_identitywrap_list[subsystem_index] = subsys_operator
        return qt.tensor(operator_identitywrap_list)

    def hubbard_operator(self, j, k, subsystem):
        """Hubbard operator |j><k| for system 'subsystem'"""
        dim = subsystem.truncated_dim
        operator = (qt.states.basis(dim, j) * qt.states.basis(dim, k).dag())
        return self.identity_wrap(operator, subsystem)


class WaveFunction(object):

    def __init__(self, basis_vals, amplitudes, eigenval=None):
        self.basis_vals = basis_vals
        self.amplitudes = amplitudes
        self.eigenval = eigenval


class GridSpecifications(object):

    def __init__(self, minmaxpts_array):
        self.min_vals = minmaxpts_array[:, 0]
        self.max_vals = minmaxpts_array[:, 1]
        self.pt_counts = minmaxpts_array[:, 2]
        self.var_count = len(self.min_vals)

    def __repr__(self):
        output = '    Grid ......'
        for parameter_name in sorted(self.__dict__.keys()):
            parameter_val = self.__dict__[parameter_name]
            output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        return output

    def unwrap(self):
        return self.min_vals, self.max_vals, self.pt_counts, self.var_count


class WaveFunctionOnGrid(object):

    def __init__(self, grid, amplitudes, eigenval=None):
        self.grid = grid
        self.amplitudes = amplitudes
        self.eigenval = eigenval


# ---Auxiliary routines  ---------------------------------------------------------


def order_eigensystem(evals, evecs):
    ordered_evals_indices = evals.argsort()  # eigsh does not guarantee consistent ordering within result?! http://stackoverflow.com/questions/22806398
    evals = evals[ordered_evals_indices]
    evecs = evecs[:, ordered_evals_indices]
    return None


def subtract_groundenergy(energy_vals):
    """Takes a list of energies (such as obtained by diagonalization) and returns them after
    subtracting the ground state energy."""
    return (energy_vals - energy_vals[0])


def extract_phase(complex_array):
    intermediate_index = int(len(complex_array) / 3)    # intermediate position for extracting phase (dangerous in tail or midpoint)
    return cmath.phase(complex_array[intermediate_index])


def filewrite_csvdata(filename, numpy_array):
    np.savetxt(filename + '.csv', numpy_array, delimiter=",")
    return None


# routine for displaying a progress bar
def display_progress_bar(progress_in_percent):
    bar_max_length = 20   # Modify this to change the length of the progress bar
    status_string = ""

    progress_in_percent = float(progress_in_percent)
    if progress_in_percent < 0.0:
        progress_in_percent = 0.0
        status_string = "Halt...\r\n"
    if progress_in_percent >= 1.0:
        progress_in_percent = 1.0
        status_string = "Done.\r\n"
    bar_length = int(round(bar_max_length * progress_in_percent))
    progress_display_string = "\r[{0}] {1}% {2}".format("=" * bar_length + "." * (bar_max_length - bar_length),
                                                        round(progress_in_percent * 100), status_string)
    sys.stdout.write(progress_display_string)
    sys.stdout.flush()
    return None


# ---Matrix elements and operators (outside qutip) ---------------------------------------------------------


def matrix_element(state1, operator, state2):
    """Calculate the matrix element <state1|operator|state2>.
    state1, state2: numpy arrays
    operator:       numpy array or sparse matrix object
    """
    if isinstance(operator, np.ndarray):    # Is operator given in dense form?
        return (np.vdot(state1, np.dot(operator, state2)))  # Yes - use numpy's 'vdot' and 'dot'.
    else:
        return (np.vdot(state1, operator.dot(state2)))      # No, operator is sparse. Must use its own 'dot' method.


def matrixelem_table(operator, vector_list, real_valued=False):
    """Calculate a table of matrix elements based on
    operator: numpy array or sparse matrix object
    vlist:    list (or array) of numpy arrays representing the states |v0>, |v1>, ...

    Returns a numpy array corresponding to the matrix element table
    <v0|operator|v0>   <v0|operator|v1>   ...
    <v1|operator|v0>   <v1|operator|v1>   ...
          ...                 ...

    Note: vector_list expected to be in scipy's eigsh transposed form
    """
    vec_list = vector_list.T

    if real_valued:
        the_dtype = np.float_
    else:
        the_dtype = np.complex_

    tablesize = len(vec_list)
    mtable = np.empty(shape=[tablesize, tablesize], dtype=the_dtype)
    for n in range(tablesize):
        for m in range(n + 1):
            mtable[n, m] = matrix_element(vec_list[n], operator, vec_list[m])
            if real_valued:
                mtable[m, n] = mtable[n, m]
            else:
                mtable[m, n] = np.conj(mtable[n, m])
    return mtable


# ---Harmonic oscillator--------------------------------------------------------------------


def harm_osc_wavefunction(n, x, losc):
    """For given quantum number n=0,1,2,... this returns the value of the harmonic oscillator
    harmonic oscillator wave function \psi_n(x) = N H_n(x/losc) exp(-x^2/2losc), N being the
    proper normalization factor.
    """
    return ((2**n * math.factorial(n) * losc)**(-0.5) * np.pi**(-0.25) *
            sp.special.eval_hermite(n, x / losc) * np.exp(-(x * x) / (2 * losc * losc)))


# ---Plotting-------------------------------------------------------------------------------


def contourplot(x_vals, y_vals, func, contour_vals=None, aspect_ratio=None, filename=False):
    """Contour plot of a 2d function 'func(x,y)'.
    x_vals: (ordered) list of x values for the x-y evaluation grid
    y_vals: (ordered) list of y values for the x-y evaluation grid
    func: function f(x,y) for which contours are to be plotted
    contour_values: contour values can be specified if so desired

    """
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_array = func(x_grid, y_grid)
    # print(z_array)
    if aspect_ratio is None:
        plt.figure(figsize=(x_vals[-1] - x_vals[0], y_vals[-1] - y_vals[0]))
    else:
        w, h = plt.figaspect(aspect_ratio)
        plt.figure(figsize=(w, h))

    if contour_vals is None:
        plt.contourf(x_grid, y_grid, z_array, cmap=plt.cm.viridis)
    else:
        plt.contourf(x_grid, y_grid, z_array, levels=contour_vals, cmap=plt.cm.viridis)

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()
    return None


def plot_matrixelements(mtable, mode='abs', xlabel='', ylabel='', zlabel='', filename=False):
    """Create a "skyscraper" and a color-coded plot of the matrix element table given as 'mtable'"""
    modefunction = MODE_FUNC_DICT[mode]
    matsize = len(mtable)
    element_count = matsize**2   # num. of elements to plot
    xgrid, ygrid = np.meshgrid(range(matsize), range(matsize))
    xgrid = xgrid.T.flatten() - 0.5  # center bars on integer value of x-axis
    ygrid = ygrid.T.flatten() - 0.5  # center bars on integer value of y-axis
    zvals = np.zeros(element_count)       # all bars start at z=0
    dx = 0.75 * np.ones(element_count)      # width of bars in x-direction
    dy = dx.copy()      # width of bars in y-direction (same as x-dir here)
    dz = modefunction(mtable).flatten()  # height of bars from density matrix elements (should use 'real()' if complex)

    nrm = mpl.colors.Normalize(0, max(dz))   # <-- normalize colors to max. data
    colors = plt.cm.viridis(nrm(dz))  # list of colors for each bar

    # plot figure

    fig = plt.figure()
    ax = mpl3d.Axes3D(fig, azim=215, elev=45)
    ax.bar3d(xgrid, ygrid, zvals, dx, dy, dz, color=colors)
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))  # set x-ticks to integers
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))  # set y-ticks to integers
    ax.set_zlim3d([0, max(dz)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.02)  # add colorbar with normalized range
    mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.viridis, norm=nrm)

    plt.matshow(modefunction(mtable), cmap=plt.cm.viridis)
    plt.show()

    if filename:
        out_file = mplpdf.PdfPages(filename)
        out_file.savefig()
        out_file.close()
    return None


def spectrum_vs_param_colored_plot(x_param_name, x_vals, y_param_name, y_vals_matrix, color_param_name, color_vals_matrix, norm_range=[0, 1],
                                   x_range=False, y_range=False, colormap='jet', figsize=(15, 10), line_width=2):
    """Takes a list of x-values,
    a list of lists with each element containing the y-values corresponding to a particular curve,
    a list of lists with each element containing the external parameter value (t-value)
    that determines the color of each curve at each y-value,
    and a normalization interval for the t-values."""
    fig = plt.figure(figsize=figsize)
    for i in range(len(y_vals_matrix)):
        pts = np.asarray([x_vals, y_vals_matrix[i]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        line_collection = mpl.collections.LineCollection(segs, cmap=plt.get_cmap(colormap), norm=plt.Normalize(*norm_range))
        line_collection.set_array(color_vals_matrix[i])
        line_collection.set_linewidth(line_width)
        plt.gca().add_collection(line_collection)

    plt.xlabel(x_param_name)

    if not x_range:
        x_range = [np.amin(x_vals), np.amax(x_vals)]
    if not y_range:
        y_range = [np.amin(y_vals_matrix), np.max(y_vals_matrix)]

    plt.xlim(*x_range)
    plt.ylim(*y_range)

    axcb = fig.colorbar(line_collection)
    axcb.set_label(color_param_name)
    plt.show()


# ---Generic quantum system container and Qubit base class---------------------------------------------------------------------


class GenericQSys(object):

    """Generic quantum system class, blank except for holding the truncation parameter 'dim'.
    The main purpose is as a wrapper for interfacing with qutip. E.g., a resonator could be
    resonator = GenericQSys(dim=4)
    In this case, photon states n=0,1,2,3 would be retained.
    """
    _EXPECTED_PARAMS_DICT = {}
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def print_expected_params_message(self):
        print('Expected parameters are:')
        for k, v in self._EXPECTED_PARAMS_DICT.items():
            print("{:<5} {:<40} ".format(k, v))
        return None

    def are_parameters_valid(self, given_params_dict):
        """Checks whether the keyword argumments provided (given_params_dict) match the
        keyword arguments expected for a certain type of qubit.
        Returns True when the two match exactly (no missing, no superfluous arguments).
        """
        for expected_key in self._EXPECTED_PARAMS_DICT:
            if expected_key not in given_params_dict:
                print('>>Error<<: one or multiple parameter(s) have not been assigned values.')
                self.print_expected_params_message()
                return False
        for given_key in given_params_dict:
            if given_key not in self._EXPECTED_PARAMS_DICT and given_key not in self._OPTIONAL_PARAMS_DICT:
                print('>>Error<<: one or multiple of the specified parameters is/are unknown.')
                self.print_expected_params_message()
                return False
        return True

    def __init__(self, **parameter_args):
        if not self.are_parameters_valid(parameter_args):
            raise UserWarning('Parameter mismatch')
        else:
            self._qubit_type = 'GenericQSys - used mainly as interface to qutip (e.g., resonator subsystem)'
            self.init_parameters(**parameter_args)

    def init_parameters(self, **parameter_args):
        for parameter_name, parameter_val in parameter_args.items():
            setattr(self, parameter_name, parameter_val)

    def __repr__(self):
        output = self.__dict__['_qubit_type'] + ' -- PARAMETERS -------'
        for parameter_name in self.__dict__.keys():
            if parameter_name[0] is not '_':
                parameter_val = self.__dict__[parameter_name]
                output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        return output


class BaseClass(GenericQSys):

    """Base class for superconducting qubit objects. Provide general mechanisms and routines for
    checking validity of initialization parameters, writing data to files, and plotting.
    """

    _EXPECTED_PARAMS_DICT = {}
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        if not self.are_parameters_valid(parameter_args):
            raise UserWarning('Parameter mismatch')
        else:
            self._qubit_type = 'BaseClass - mainly used as class template'
            self.init_parameters(**parameter_args)

    def __repr__(self):
        output = self.__dict__['_qubit_type'] + ' -- PARAMETERS -------'
        for parameter_name in sorted(self.__dict__.keys()):
            if parameter_name[0] is not '_':
                parameter_val = self.__dict__[parameter_name]
                output += '\n' + str(parameter_name) + '\t: ' + str(parameter_val)
        output += '\nHilbert space dimension\t: ' + str(self.hilbertdim())
        return output

    def filewrite_parameters(self, filename):
        with open(filename + PARAMETER_FILESUFFIX, 'w') as target_file:
            target_file.write(self.__repr__())

    @staticmethod
    def filewrite_evals(filename, evals):
        filewrite_csvdata(filename + EVALS_FILESUFFIX, evals)

    @staticmethod
    def filewrite_evecs(filename, evecs):
        filewrite_csvdata(filename + EVECS_FILESUFFIX, evecs)

    def hilbertdim(self):
        """Must be implemented in child classes"""
        pass

    def hamiltonian(self):
        """Must be implemented in child classes"""
        pass

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        return sp.linalg.eigh(hamiltonian_mat, eigvals_only=True, eigvals=(0, evals_count - 1))

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        return sp.linalg.eigh(hamiltonian_mat, eigvals_only=False, eigvals=(0, evals_count - 1))

    def eigenvals(self, evals_count=6, filename=None):
        """Calculates eigenvalues (via qubit-specific _evals_calc()), and returns a numpy array of eigenvalues.

        evals_count:   number of desired eigenvalues (sorted from smallest to largest)
        filename: write data to file if path and filename are specified
        """
        evals = np.sort(self._evals_calc(evals_count))
        if filename:
            self.filewrite_evals(filename, evals)
            self.filewrite_paramaters(filename)

        return evals

    def eigensys(self, evals_count=6, filename=None):
        """Calculates eigenvalues and corresponding eigenvectores (via qubit-specific _esys_calc()), and returns
        two numpy arrays containing the eigenvalues and eigenvectors, respectively.

        evals_count:   number of desired eigenvalues (sorted from smallest to largest)
        filename: write data to file if path and filename are specified
        """
        evals, evecs = self._esys_calc(evals_count)
        order_eigensystem(evals, evecs)

        if filename:
            self.filewrite_evals(filename, evals)
            self.filewrite_evecs(filename, evecs)
            self.filewrite_paramaters(filename)

        return evals, evecs

    def matrixelements(self, operator, esys, evals_count):
        """Returns a table of matrix elements for 'operator', given as a string referring to a class method
        that returns an operator matrix. E.g., for 'transmon = Transmon(...)', the matrix element table
        for the charge operator n can be accessed via 'transmon.op_matrixelement_table('n', esys=None, evals_count=6).
        When 'esys' is set to None, the eigensystem with 'evals_count' eigenvectors is calculated.
        """
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        operator_matrix = getattr(self, operator)()
        return matrixelem_table(operator_matrix, evecs)

    def get_evals_vs_paramvals(self, parameter_name, paramval_list, evals_count=6, subtract_ground=False, filename=None):
        """Calculates a set of eigenvalues as a function of the parameter 'param', where the discrete values
        for 'param' are contained in the list prmval_list. Returns a numpy array where specdata[n] is set
        of eigenvalues calculated for parameter value prmval_list[n]

        param_name:           string, gives name of parameter to be varied
        prmval_list:          list of parameter values to be plugged in for param
        subtract_ground:      if True, then eigenvalues are returned relative to the ground state eigenvalues
                              (useful if transition energies from ground state are the relevant quantity)
        evals_count:           number of desired eigenvalues (sorted from smallest to largest)
        filename:         write data to file if path and filename are specified
        """
        previous_paramval = getattr(self, parameter_name)

        paramvals_count = len(paramval_list)
        spectrumdata = np.empty((paramvals_count, evals_count))
        print("")
        display_progress_bar(0)
        for index, paramval in enumerate(paramval_list):
            setattr(self, parameter_name, paramval)
            evals = self.eigenvals(evals_count)
            spectrumdata[index] = evals
            if subtract_ground:
                spectrumdata[index] -= evals[0]
            progress_in_percent = (index + 1) / paramvals_count
            display_progress_bar(progress_in_percent)
        setattr(self, parameter_name, previous_paramval)
        if filename:
            filewrite_csvdata(filename + '_' + parameter_name, paramval_list)
            filewrite_csvdata(filename + '_specdata', spectrumdata)
            self.filewrite_parameters(filename)

        if subtract_ground:
            return spectrumdata[:, 1:]
        else:
            return spectrumdata

    def plot_evals_vs_paramvals(self, parameter_name, paramval_list, evals_count=6,
                                yrange=False, subtract_ground=False, shift=0, filename=None):
        """Generates a simple plot of a set of eigenvalues as a function of parameter 'param'.
        The individual points correspond to the parameter values listed in paramval_list.

        param:           string, gives name of parameter to be varied
        paramval_list:     list of parameter values to be plugged in for param
        subtract_ground: if True, then eigenvalues are returned relative to the ground state eigenvalues
                         (useful if transition energies from ground state are the relevant quantity)
        evals_count:           number of desired eigenvalues (sorted from smallest to largest)
        yrange:          [ymin, ymax] -- custom y-range for the plot
        shift:           apply a shift of this size to all eigenvalues
        filename:         write graphics and parameter set to file if path and filename are specified
        """
        x = paramval_list
        y = self.get_evals_vs_paramvals(parameter_name, paramval_list, evals_count, subtract_ground)
        if yrange:
            plt.axis([np.amin(x), np.amax(x), yrange[0], yrange[1]])
        else:
            plt.axis([np.amin(x), np.amax(x), np.amin(y + shift), np.amax(y + shift)])
        plt.xlabel(parameter_name)
        plt.ylabel('energy')
        plt.plot(x, y + shift)
        if filename:
            out_file = mplpdf.PdfPages(filename + '.pdf')
            out_file.savefig()
            out_file.close()
            self.filewrite_parameters(filename)
        plt.show()
        return None

    @staticmethod
    def _plot_wavefunction1d(wavefunc, potential_vals, offset=0, scaling=1,
                             ylabel='wavefunction', xlabel='x'):
        x_vals = wavefunc.basis_vals
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vals, offset + scaling * wavefunc.amplitudes)
        if potential_vals is not None:
            ax.plot(x_vals, potential_vals)
            ax.plot(x_vals, [offset] * len(x_vals), 'b--')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xmin=x_vals[0], xmax=x_vals[-1])
        plt.show()
        return None

    @staticmethod
    def _plot_wavefunction1d_discrete(wavefunc, nrange, ylabel='wavefunction', xlabel='x'):
        x_vals = wavefunc.basis_vals
        width = .75
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(x_vals, wavefunc.amplitudes, width=width)
        ax.set_xticks(x_vals + width / 2)
        ax.set_xticklabels(x_vals)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(nrange)
        plt.show()
        return None

    @staticmethod
    def _plot_wavefunction2d(wavefunc, figsize, aspect_ratio, zero_calibrate=False):
        plt.figure(figsize=figsize)

        if zero_calibrate:
            absmax = np.amax(np.abs(wavefunc.amplitudes))
            imshow_minval = -absmax
            imshow_maxval = absmax
            cmap = plt.get_cmap('PRGn')
        else:
            imshow_minval = np.min(wavefunc.amplitudes)
            imshow_maxval = np.max(wavefunc.amplitudes)
            cmap = plt.cm.viridis

        min_vals, max_vals, _, _ = wavefunc.grid.unwrap()
        plt.imshow(wavefunc.amplitudes, extent=[min_vals[0], max_vals[0], min_vals[1], max_vals[1]],
                   aspect=aspect_ratio, cmap=cmap, vmin=imshow_minval, vmax=imshow_maxval)
        plt.colorbar(fraction=0.017, pad=0.04)
        plt.show()
        return None


# ---Cooper pair box / transmon-----------------------------------------------------------


class Transmon(BaseClass):

    """Class for the Cooper pair box / transmon qubit. Hamiltonian is represented in dense form. Expected parameters:
        EJ:   Josephson energy
        EC:   charging energy
        ng:   offset charge
        ncut: charge basis cutoff, n = -ncut, ..., ncut'

    Initialize with, e.g.
    >>> qubit = Transmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EC': 'charging energy',
        'ng': 'offset charge',
        'ncut': 'charge basis cutoff, n = -ncut, ..., ncut'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(Transmon, self).__init__(**parameter_args)
        self._qubit_type = 'Cooper pair box'

    def hamiltonian(self):
        dimension = 2 * self.ncut + 1
        hamiltonian_mat = np.zeros((dimension, dimension), dtype=np.float_)
        for i in range(dimension):
            hamiltonian_mat[i][i] = 4.0 * self.EC * (i - self.ncut - self.ng)**2
        for i in range(dimension - 1):
            hamiltonian_mat[i][i + 1] = -self.EJ / 2.0
            hamiltonian_mat[i + 1][i] = -self.EJ / 2.0
        return hamiltonian_mat

    def hilbertdim(self):
        return (2 * self.ncut + 1)

    def n_operator(self):
        """Charge operator in charge basis for the transmon or CPB qubit."""
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1)
        return np.diagflat(diag_elements)

    def plot_n_wavefunction(self, esys, which, mode, nrange=[-5, 6]):
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        modefunction = MODE_FUNC_DICT[mode]
        n_wavefunc.amplitudes = modefunction(n_wavefunc.amplitudes)
        self._plot_wavefunction1d_discrete(n_wavefunc, nrange)
        return None

    def plot_phi_wavefunction(self, esys, which=0, phi_points=251, mode='abs2'):
        phi_wavefunc = self.phasebasis_wavefunction(esys, which=which, phi_points=phi_points)
        phase = extract_phase(phi_wavefunc.amplitudes)
        phi_wavefunc.amplitudes *= cmath.exp(-1j * phase)

        modefunction = MODE_FUNC_DICT[mode]
        phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)
        potential_vals = -self.EJ * np.cos(phi_wavefunc.basis_vals)
        self._plot_wavefunction1d(phi_wavefunc, potential_vals,
                                  offset=phi_wavefunc.eigenval, scaling=0.3 * self.EJ, xlabel='phi')
        return None

    def numberbasis_wavefunction(self, esys, which=0):
        """Return the transmon wave function in number basis. The specific index of the wavefunction is: 'which'.
        'esys' can be provided, but if set to 'None' then it is calculated frst.
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, evecs = esys

        n_vals = np.arange(-self.ncut, self.ncut + 1)
        return WaveFunction(n_vals, evecs[:, which])

    def phasebasis_wavefunction(self, esys, which=0, phi_points=251):
        """Return the transmon wave function in phase basis. The specific index of the wavefunction is: 'which'.
        'esys' can be provided, but if set to 'None' then it is calculated frst.
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count)
        evals, evecs = esys
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        phi_basis_vals = np.linspace(-np.pi, np.pi, phi_points)
        phi_wavefunc_amplitudes = np.empty(phi_points, dtype=np.complex_)
        for k in range(phi_points):
            phi_wavefunc_amplitudes[k] = ((1.0 / math.sqrt(2 * np.pi)) *
                                          np.sum(n_wavefunc.amplitudes * np.exp(1j * phi_basis_vals[k] * n_wavefunc.basis_vals)))
        return WaveFunction(phi_basis_vals, phi_wavefunc_amplitudes, eigenval=evals[which])


# ---Fluxonium qubit ------------------------------------------------------------------------


class Fluxonium(BaseClass):

    """Class for the fluxonium qubit. Hamiltonian is represented in dense form. The employed
    basis is the EC-EL harmonic oscillator basis. The cosine term in the potential is handled
    via matrix exponentiation.
    Expected parameters:
        EJ:   Josephson energy
        EC:   charging energy
        EL:   inductive energy
        flux: external magnetic flux (angular units, 2pi corresponds to one flux quantum)
        cutoff: number of harm. osc. basis states used in diagonalization

    Initialize with, e.g.
    >>> qubit = Fluxonium(EJ=1.0, EC=2.0, EL=0.3, flux=0.2, cutoff=120)
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EC': 'charging energy',
        'EL': 'inductive energy',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'cutoff': 'number of harm. osc. basis states used in diagonalization',
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(Fluxonium, self).__init__(**parameter_args)
        self._qubit_type = 'fluxonium'

    def phi_osc(self):
        return (8.0 * self.EC / self.EL)**(0.25)        # LC oscillator length

    def omega_p(self):
        return math.sqrt(8.0 * self.EL * self.EC)        # LC plasma oscillation frequency

    def phi_operator(self):
        """Returns the phi operator in the LC harmonic oscillator basis"""
        dimension = self.hilbertdim()
        return (op.creation(dimension) + op.annihilation(dimension)) * self.phi_osc() / math.sqrt(2)

    def n_operator(self):
        """Returns the n = - i d/dphi operator in the LC harmonic oscillator basis"""
        dimension = self.hilbertdim()
        return 1j * (op.creation(dimension) - op.annihilation(dimension)) / (self.phi_osc() * math.sqrt(2))

    def hamiltonian(self):           # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harm. osc. basis and return as sparse.dia_matrix"""
        dimension = self.hilbertdim()
        diag_elements = [i * self.omega_p() for i in range(dimension)]
        LC_osc_matrix = np.diagflat(diag_elements)

        exponent = 1j * self.phi_operator()
        exp_matrix = 0.5 * sp.linalg.expm(exponent) * cmath.exp(1j * 2 * np.pi * self.flux)
        cos_matrix = exp_matrix + np.conj(exp_matrix.T)

        hamiltonian_mat = LC_osc_matrix - self.EJ * cos_matrix
        return hamiltonian_mat

    def hilbertdim(self):
        return (self.cutoff)

    def potential(self, phi):
        return (0.5 * self.EL * phi * phi - self.EJ * np.cos(phi + 2.0 * np.pi * self.flux))

    def wavefunction(self, esys, which=0, phi_range=[-6 * np.pi, 6 * np.pi], phi_points=251):
        evals_count = max(which + 1, 3)
        if esys is None:
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys

        dim = self.hilbertdim()
        phi_basis_vals = np.linspace(phi_range[0], phi_range[1], phi_points)
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros(phi_points, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * harm_osc_wavefunction(n, phi_basis_vals, phi_osc)
        return WaveFunction(phi_basis_vals, phi_wavefunc_amplitudes, evals[which])

    def plot_wavefunction(self, esys, which=0, phi_range=[-6 * np.pi, 6 * np.pi], mode='abs2', phi_points=251):
        """Different modes:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        modefunction = MODE_FUNC_DICT[mode]
        phi_wavefunc = self.wavefunction(esys, which, phi_range, phi_points)
        phase = extract_phase(phi_wavefunc.amplitudes)
        phi_wavefunc.amplitudes *= cmath.exp(-1j * phase)

        modefunction = MODE_FUNC_DICT[mode]
        phi_wavefunc.amplitudes = modefunction(phi_wavefunc.amplitudes)
        self._plot_wavefunction1d(phi_wavefunc, self.potential(phi_wavefunc.basis_vals),
                                  offset=phi_wavefunc.eigenval, scaling=5 * self.EJ, xlabel='phi')
        return None

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction2d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction2'")


# ---Fluxonium qubit with SQUID loop----------------------------------------------------------------------


class FluxoniumSQUID(Fluxonium):

    """Class for the fluxonium qubit with two Josephson elements. Hamiltonian is represented in sparse form. The employed
    basis is the EC-EL harmonic oscillator basis. The cosine term in the potential is handled
    via matrix exponentiation.
    Expected parameters:
        EJ1:   Josephson energy 1
        EJ2:   Josephson energy 2
        EC:   charging energy
        EL:   inductive energy
        flux: external magnetic flux through primary loop in units of flux quanta (h/2e)
        fluxsquid: external magnetic flux through the SQUID loop in units of flux quanta (h/2e)
        cutoff: number of harm. osc. basis states used in diagonalization
    Initialize with, e.g.
    >>> qubit = FluxoniumSQUID(EJ1=1.0, EJ2=1.0, EC=2.0, EL=0.3, flux=0.2, fluxsquid=0.1, cutoff=120)
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ1': 'Josephson energy 1',
        'EJ2': 'Josephson energy 2',
        'EC': 'charging energy',
        'EL': 'inductive energy',
        'flux': 'external magnetic flux through primary loop in units of flux quanta (h/2e)',
        'fluxsquid': 'external magnetic flux through the SQUID loop in units of flux quanta (h/2e)',
        'cutoff': 'number of harm. osc. basis states used in diagonalization'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(FluxoniumSQUID, self).__init__(**parameter_args)
        self._qubit_type = 'Fluxonium with small-junction SQUID loop'

    def hamiltonian(self):
        """Construct Hamiltonian matrix in harm. osc. basis and return as sparse.dia_matrix"""
        EJ1 = self.EJ1
        EJ2 = self.EJ2
        flux = self.flux
        fluxsquid = self.fluxsquid
        dim = self.hilbertdim()

        omega_p = self.omega_p()         # plasma osc. frequency
        d = (EJ1 - EJ2) / (EJ1 + EJ2)
        chi = math.atan(d * math.tan(-1 * np.pi * fluxsquid))        # just a term in the phase argument
        prefactor = math.cos(np.pi * fluxsquid) * math.sqrt(1.0 + (d * math.tan(np.pi * fluxsquid))**(2))  # just a prefactor in the transformed EJcos term

        diag_elements = [i * omega_p for i in range(dim)]
        LC_osc_matrix = np.diagflat(diag_elements)

        exponential = 1j * (op.creation(dim) + op.annihilation(dim)) * self.phi_osc() / math.sqrt(2)
        exp_matrix = 0.5 * sp.linalg.expm(exponential) * cmath.exp(1j * (2.0 * np.pi * flux - np.pi * fluxsquid + chi))
        cos_matrix = exp_matrix + np.conj(exp_matrix.T)

        hamiltonian_mat = LC_osc_matrix - (EJ1 + EJ2) * prefactor * cos_matrix
        return hamiltonian_mat

    def potential(self, phi):
        return (0.5 * self.EL * (phi)**(2) - self.EJ1 * np.cos(phi + 2 * np.pi * self.flux) -
                self.EJ2 * np.cos(phi - 2 * np.pi * self.fluxsquid + 2 * np.pi * self.flux))

    def param_sweep_plot(self, param1_name, paramval_list, param2_name, minimum, maximum, step, evals_count=6):
        """Plots evals against param1_name in range paramval_list.
        Plots for values of param2+name from minimum to maximum with separation step."""
        previous_param2val = getattr(self, param2_name)
        self.plot_evals_vs_paramvals(param1_name, paramval_list, evals_count)
        for i in range(int((maximum - minimum) / step)):
            setattr(self, param2_name, (minimum + step))
            minimum = minimum + step
            self.plot_evals_vs_paramvals(param1_name, paramval_list, evals_count)
        setattr(self, param2_name, previous_param2val)


# ---Routines for translating 1st and 2nd derivatives by discretization into sparse matrix form-------


def grid_first_derivative(var_ind, grid, prefactor=1, periodic=False):

    if isinstance(prefactor, complex):
        the_dtype = np.complex_
    else:
        the_dtype = np.float_

    min_vals, max_vals, pt_counts, var_count = grid.unwrap()

    delta_inv = prefactor * pt_counts[var_ind] / (2 * (max_vals[var_ind] - min_vals[var_ind]))
    drvtv_mat = sp.sparse.dia_matrix((pt_counts[var_ind], pt_counts[var_ind]), dtype=the_dtype)
    drvtv_mat.setdiag(delta_inv, k=1)
    drvtv_mat.setdiag(-delta_inv, k=-1)

    if periodic:
        drvtv_mat.setdiag(-delta_inv, k=pt_counts[var_ind] - 1)
        drvtv_mat.setdiag(delta_inv, k=-pt_counts[var_ind] + 1)

    full_mat = drvtv_mat

    # Now fill in identity matrices to the left of var_ind, with variable indices
    # smaller than var_ind. Note: range(3,0,-1) -> [3,2,1]
    for j in range(var_ind - 1, -1, -1):
        full_mat = sp.sparse.kron(sp.sparse.identity(pt_counts[j], format='dia'), full_mat)
    # Next, fill in identity matrices with larger variable indices to the right.
    for j in range(var_ind + 1, var_count):
        full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[j], format='dia'))

    return full_mat


def grid_second_derivative(var_ind, grid, prefactor=1, periodic=False):
    min_vals, max_vals, pt_counts, var_count = grid.unwrap()

    delta_inv_sqr = prefactor * ((max_vals[var_ind] - min_vals[var_ind]) / pt_counts[var_ind])**(-2)

    drvtv_mat = sp.sparse.dia_matrix((pt_counts[var_ind], pt_counts[var_ind]), dtype=np.float_)
    drvtv_mat.setdiag(-2.0 * delta_inv_sqr, k=0)
    drvtv_mat.setdiag(delta_inv_sqr, k=1)
    drvtv_mat.setdiag(delta_inv_sqr, k=-1)

    if periodic:
        drvtv_mat.setdiag(delta_inv_sqr, k=pt_counts[var_ind] - 1)
        drvtv_mat.setdiag(delta_inv_sqr, k=-pt_counts[var_ind] + 1)

    full_mat = drvtv_mat
    # Now fill in identity matrices to the left of var_ind, with variable indices
    # smaller than var_ind. Note: range(3,0,-1) -> [3,2,1]
    for j in range(var_ind - 1, -1, -1):
        full_mat = sp.sparse.kron(sp.sparse.identity(pt_counts[j], format='dia'), full_mat)
    # Next, fill in identity matrices with larger variable indices to the right.
    for j in range(var_ind + 1, var_count):
        full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[j], format='dia'))
    return full_mat


def grid_multiple_first_derivatives(deriv_var_list, grid, prefactor=1, periodic_list=False):
    """Generate sparse derivative matrices of the form \partial_{x_1} \partial_{x_2} ...,
    i.e., a product of first order derivatives (with respect to different variables).
    Note: var_list is expected to be ordered!
    """
    if isinstance(prefactor, complex):
        dtp = np.complex_
    else:
        dtp = np.float_

    min_vals, max_vals, pt_counts, var_count = grid.unwrap()

    deriv_order = len(deriv_var_list)   # total order of derivative (number of variables wrt which first derivatives are taken)

    delta_inv = [0] * deriv_order   # this is 1/delta from f'(x) ~= [f(x+h) - f(x-h)]/2h, delta=2h
    drvtv_mat = [0] * deriv_order

    # Loop over the elements of var_list and generate the derivative matrices
    for j in range(deriv_order):
        delta_inv[j] = pt_counts[deriv_var_list[j]] / (2.0 * (max_vals[deriv_var_list[j]] - min_vals[deriv_var_list[j]]))
        if j == 0:
            delta_inv[j] *= prefactor   # first variable has prefactor absorbed into 1/delta
        drvtv_mat[j] = sp.sparse.dia_matrix((pt_counts[deriv_var_list[j]], pt_counts[deriv_var_list[j]]), dtype=dtp)
        drvtv_mat[j].setdiag(delta_inv[j], k=1)      # occupy the first off-diagonal to the right
        drvtv_mat[j].setdiag(-delta_inv[j], k=-1)    # and left

        if deriv_var_list[j] in periodic_list:
            drvtv_mat[j].setdiag(-delta_inv[j], k=pt_counts[deriv_var_list[j]] - 1)
            drvtv_mat[j].setdiag(delta_inv[j], k=-pt_counts[deriv_var_list[j]] + 1)

    # Procedure to generate full matrix as follows. Example: derivatives w.r.t. 2, 4, and 5
    # 0  1  d2   3  d4  d5  6  7  8
    # (a) Set current index to first derivative index (ex: 2)
    # (b) Fill in identities to the left (Kronecker products, ex: 1, 0)
    # (c) Fill in identities to the right up to next derivative index or end (ex: 3)
    # (d) Insert next derivative.
    # (e) Repeat (c) and (d) until all variables finished.
    #
    # (a) First derivative
    full_mat = drvtv_mat[0]
    # (b) Now fill in identity matrices to the left of first variable index
    for j in range(deriv_var_list[0] - 1, -1, -1):
        full_mat = sp.sparse.kron(sp.sparse.identity(pt_counts[j], format='dia'), full_mat)
    # Loop over remaining derivatives up to very last one:
    for index, deriv_index in enumerate(deriv_var_list[:-1]):
        # (c) Fill in identities to the right
        for j in range(deriv_index + 1, deriv_var_list[index + 1]):
            full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[j], format='dia'))
        # (d) Insert next derivative
        full_mat = sp.sparse.kron(full_mat, drvtv_mat[index + 1])
    # Fill in all remaining identity matrices to the right until last variable covered
    for j in range(deriv_var_list[-1] + 1, var_count):
        full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(pt_counts[j], format='dia'))

    return full_mat


# ---Symmetric 0-pi qubit--------------------------------------------------------------------


class SymZeroPi(BaseClass):

    """Symmetric Zero-Pi Qubit
    [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    The symmetric model, Eq. (8) in [2], assumes pair-wise identical circuit elements and describes the
    phi and theta degrees of freedom (chi decoupled). Formulation of the Hamiltonian matrix proceeds
    by discretization of the phi-theta space into a simple square/rectangular lattice.
    Expected parameters are:

    EJ:   Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'flux': 'external magnetic flux in angular units (2pi corresponds to one flux quantum)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(SymZeroPi, self).__init__(**parameter_args)
        self._qubit_type = 'symmetric 0-Pi qubit (zero offset charge)'

    def hilbertdim(self):
        pt_counts = self.grid.pt_counts
        return int(np.prod(pt_counts))

    def potential(self, phi, theta):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) + self.EL * phi**2 + 2.0 * self.EJ)

    def sparse_kineticmat(self):
        kmat = grid_second_derivative(PHI_INDEX, self.grid, prefactor=-2.0 * self.ECJ)    # -2E_{CJ}\partial_\phi^2
        kmat += grid_second_derivative(THETA_INDEX, self.grid, prefactor=-2.0 * self.ECS, periodic=True)  # -2E_{C\Sigma}\partial_\theta^2
        return kmat

    def sparse_potentialmat(self):
        """Returns the potential energy matrix for the potential in sparse (dia_matrix) form."""
        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()
        hilbertspace_dim = int(np.prod(pt_counts))
        var_count = len(min_vals)
        Xvals = [np.linspace(min_vals[j], max_vals[j], pt_counts[j]) for j in range(var_count)]  # list of coordinate arrays
        diag_elements = np.empty([1, hilbertspace_dim], dtype=np.float_)

        for j, coord_tuple in enumerate(itertools.product(*Xvals)):
            diag_elements[0][j] = self.potential(*coord_tuple)   # diagonal matrix elements
        return sp.sparse.dia_matrix((diag_elements, [0]), shape=(hilbertspace_dim, hilbertspace_dim))

    def hamiltonian(self):
        return (self.sparse_kineticmat() + self.sparse_potentialmat())

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return evals

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        return evals, evecs

    def i_d_dphi_operator(self):
        """Return the operator i \partial_\phi in sparse.dia_matrix form"""
        return grid_first_derivative(PHI_INDEX, self.grid, prefactor=1j, periodic=False)

    def i_d_dtheta_operator(self):
        """Return the operator i \partial_\theta (periodic variable) in sparse.dia_matrix form"""
        return grid_first_derivative(THETA_INDEX, self.grid, prefactor=1j, periodic=True)

    def d_dtheta_operator(self):
        """Return the operator i \partial_\theta (periodic variable) in sparse.dia_matrix form"""
        return grid_first_derivative(THETA_INDEX, self.grid, periodic=True)

    # return the operator \phi
    def phi_operator(self):
        min_vals, max_vals, pt_counts, var_count = self.grid.unwrap()
        phi_matrix = sp.sparse.dia_matrix((pt_counts[PHI_INDEX], pt_counts[PHI_INDEX]), dtype=np.float_)
        diag_elements = np.linspace(min_vals[PHI_INDEX], max_vals[PHI_INDEX], pt_counts[PHI_INDEX])
        phi_matrix.setdiag(diag_elements)
        for j in range(1, var_count):
            phi_matrix = sp.sparse.kron(phi_matrix, sp.sparse.identity(pt_counts[j], format='dia'))
        return phi_matrix

    def plot_potential(self, contour_vals=None, aspect_ratio=None, filename=None):
        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()
        x_vals = np.linspace(min_vals[PHI_INDEX], max_vals[PHI_INDEX], pt_counts[PHI_INDEX])
        y_vals = np.linspace(min_vals[THETA_INDEX], max_vals[THETA_INDEX], pt_counts[THETA_INDEX])
        contourplot(x_vals, y_vals, self.potential, contour_vals, aspect_ratio, filename)
        return None

    def wavefunction(self, esys, which=0):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        pt_counts = self.grid.pt_counts
        wavefunc_amplitudes = evecs[:, which].reshape(pt_counts[PHI_INDEX], pt_counts[THETA_INDEX]).T
        return WaveFunctionOnGrid(self.grid, wavefunc_amplitudes)

    def plot_wavefunction(self, esys, which=0, mode='abs', figsize=(20, 10), aspect_ratio=3, zero_calibrate=False):
        """Different modes:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        modefunction = MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, which)
        wavefunc.amplitudes = modefunction(wavefunc.amplitudes)
        self._plot_wavefunction2d(wavefunc, figsize, aspect_ratio, zero_calibrate)
        return None

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction1d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d'")


# ----------------------------------------------------------------------------------------


class DisZeroPi(SymZeroPi):

    """Zero-Pi Qubit with disorder in EJ and EC. This disorder type still leaves chi decoupled,
    see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta space
    into a simple square/rectangular lattice.
    Expected parameters are:

    EJ:   mean Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    dEJ:  relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dCJ:  relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined
    as relative ones.

    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(SymZeroPi, self).__init__(**parameter_args)
        self._qubit_type = '0-Pi qubit with EJ and CJ disorder, no coupling to chi mode (zero offset charge)'

    def potential(self, phi, theta):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2.0) + self.EL * phi**2 + 2.0 * self.EJ +
                2.0 * self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2.0))

    def sparse_kineticmat(self):
        dphi2 = grid_second_derivative(PHI_INDEX, self.grid, prefactor=-2.0 * self.ECJ)                   # -2E_{CJ}\partial_\phi^2
        dth2 = grid_second_derivative(THETA_INDEX, self.grid, prefactor=-2.0 * self.ECS, periodic=True)     # -2E_{C\Sigma}\partial_\theta^2
        dphidtheta = grid_multiple_first_derivatives([PHI_INDEX, THETA_INDEX], self.grid,
                                                     prefactor=4.0 * self.ECS * self.dCJ, periodic_list=[THETA_INDEX])
        return (dphi2 + dth2 + dphidtheta)


# ----------------------------------------------------------------------------------------


class SymZeroPiNg(SymZeroPi):

    """Symmetric Zero-Pi Qubit taking into account offset charge ng
    [1] Brooks et al., Physical Review A, 87(5), 052306 (2013). http://doi.org/10.1103/PhysRevA.87.052306
    [2] Dempster et al., Phys. Rev. B, 90, 094518 (2014). http://doi.org/10.1103/PhysRevB.90.094518
    The symmetric model, Eq. (8) in [2], assumes pair-wise identical circuit elements and describes the
    phi and theta degrees of freedom (chi decoupled). Including the offset charge leads to the substitution
    T = ... + CS \dot{theta}^2  ==>    T = ... + CS (\dot{theta} + ng)^2
    [This is not described in the two references above.]

    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta space into a simple
    square/rectangular lattice.
    Expected parameters are:

    EJ:   Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    ng:   offset charge
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'ng': 'offset charge',
        'flux': 'external magnetic flux units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(SymZeroPiNg, self).__init__(**parameter_args)
        self._qubit_type = 'symmetric 0-Pi qubit with offset charge ng'

    def sparse_kineticmat(self):
        pt_counts = self.grid.pt_counts
        return (grid_second_derivative(PHI_INDEX, self.grid, prefactor=-2.0 * self.ECJ) +  # -2E_{CJ}\partial_\phi^2
                grid_second_derivative(THETA_INDEX, self.grid, prefactor=-2.0 * self.ECS, periodic=True) +   # 2E_{C\Sigma}(i\partial_\theta + n_g)^2
                grid_first_derivative(THETA_INDEX, self.grid, prefactor=4.0 * 1j * self.ECS * self.ng, periodic=True) +
                sp.sparse.kron(sp.sparse.identity(pt_counts[PHI_INDEX], format='dia'),
                               sp.sparse.identity(pt_counts[THETA_INDEX], format='dia') * 2.0 * self.ECS * (self.ng)**2))


# ----------------------------------------------------------------------------------------


class FullZeroPi(SymZeroPi):

    """Full Zero-Pi Qubit, with all disorder types in circuit element parameters included. This couples
    the chi degree     of freedom, see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds by discretization of the phi-theta-chi space
    into a simple cubic lattice.
    Expected parameters are:

    EJ:   mean Josephson energy of the two junctions
    EL:   inductive energy of the two (super-)inductors
    ECJ:  charging energy associated with the two junctions
    ECS:  charging energy including the large shunting capacitances
    EC:   charging energy associated with chi degree of freedom
    dEJ:  relative disorder in EJ, i.e., (EJ1-EJ2)/EJ(mean)
    dEL:  relative disorder in EL, i.e., (EL1-EL2)/EL(mean)
    dCJ:  relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/C(mean)
    flux: magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid: Grid object specifying the range and spacing of the discretization lattice

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined as
    relative ones.
    """

    VARNAME_TO_INDEX = {'phi': PHI_INDEX, 'theta': THETA_INDEX, 'chi': CHI_INDEX}

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'EC': 'charging energy associated with chi degree of freedom',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'dC': 'relative deviation between the two shunt capacitances',
        'dEL': 'relative deviation between the two inductances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(FullZeroPi, self).__init__(**parameter_args)
        self._qubit_type = 'full 0-Pi circuit (phi, theta, chi), no offset charge'

    def sparse_kineticmat(self):
        return (grid_second_derivative(PHI_INDEX, self.grid, prefactor=-2.0 * self.ECJ) +                  # -2E_{CJ}\partial_\phi^2
                grid_second_derivative(THETA_INDEX, self.grid, prefactor=-2.0 * self.ECS, periodic=True) +   # -2E_{C\Sigma}\partial_\theta^2
                grid_second_derivative(CHI_INDEX, self.grid, prefactor=-2.0 * self.EC) +                   # -2E_{C}\partial_\chi^2
                grid_multiple_first_derivatives([PHI_INDEX, THETA_INDEX], self.grid, prefactor=4.0 * self.ECS * self.dCJ, periodic_list=[THETA_INDEX]) +  # 4E_{C\Sigma}(\delta C_J/C_J)\partial_\phi \partial_\theta
                grid_multiple_first_derivatives([THETA_INDEX, CHI_INDEX], self.grid, prefactor=4.0 * self.ECS * self.dC, periodic_list=[THETA_INDEX]))    # 4E_{C\Sigma}(\delta C/C)\partial_\theta \partial_\chi

    def potential(self, phi, theta, chi):
        return (-2.0 * self.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * self.flux / 2) + self.EL * phi**2 + 2 * self.EJ +   # symmetric 0-pi contributions
                2.0 * self.EJ * self.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * self.flux / 2) + self.EL * chi**2 +       # correction terms in presence of disorder
                2.0 * self.EL * self.dEL * phi * chi + 2 * self.EJ * self.dEJ)                                                  # correction terms in presence of disorder

    def plot_potential(self, fixedvar_name, fixedvar_val, contour_vals=None, aspect_ratio=None, filename=None):
        fixedvar_index = self.VARNAME_TO_INDEX[fixedvar_name]

        othervar_indices = list(set([PHI_INDEX, THETA_INDEX, CHI_INDEX]) - set([fixedvar_index]))

        def reduced_potential(x, y):    # not very elegant, suspect there is a better way of coding this?
            func_arguments = [fixedvar_val] * 3
            func_arguments[othervar_indices[0]] = x
            func_arguments[othervar_indices[1]] = y
            return self.potential(*func_arguments)

        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()
        x_vals = np.linspace(min_vals[othervar_indices[0]], max_vals[othervar_indices[0]], pt_counts[othervar_indices[0]])
        y_vals = np.linspace(min_vals[othervar_indices[1]], max_vals[othervar_indices[1]], pt_counts[othervar_indices[1]])
        contourplot(x_vals, y_vals, reduced_potential, contour_vals, aspect_ratio, filename)
        return None

    def wavefunction(self, esys, which=0):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        return evecs[:, which]

    def plot_wavefunction(self, esys, fixedvar_name, fixedvar_val, which=0, mode='abs', figsize=(20, 10), aspect_ratio=3):
        """Different modes:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """

        min_vals, max_vals, pt_counts, _ = self.grid.unwrap()

        wavefunc = self.wavefunction(esys, which)
        modefunction = MODE_FUNC_DICT[mode]
        wavefunc = modefunction(wavefunc)
        wavefunc = wavefunc.reshape(pt_counts[0], pt_counts[1], pt_counts[2])

        fixedvar_index = self.VARNAME_TO_INDEX[fixedvar_name]

        slice_index = int(pt_counts[fixedvar_index] * (fixedvar_val - min_vals[fixedvar_index]) /
                          (max_vals[fixedvar_index] - min_vals[fixedvar_index]))

        slice_coordinates3d = [slice(None), slice(None), slice(None)]
        slice_coordinates3d[fixedvar_index] = slice_index
        wavefunc = wavefunc[tuple(slice_coordinates3d)].T
        self._plot_wavefunction2d(wavefunc, figsize, aspect_ratio)
        return None

    def _plot_wavefunction2d(self, wavefunc, figsize, aspect_ratio):
        plt.figure(figsize=figsize)
        plt.imshow(wavefunc, cmap=plt.cm.viridis, aspect=aspect_ratio)
        plt.colorbar(fraction=0.017, pad=0.04)
        plt.show()

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction1d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d'")


# ----------------------------------------------------------------------------------------


class FullZeroPi_ProductBasis(BaseClass):

    """Full Zero-Pi Qubit, with all disorder types in circuit element parameters included. This couples
    the chi degree     of freedom, see Eq. (15) in Dempster et al., Phys. Rev. B, 90, 094518 (2014).
    Formulation of the Hamiltonian matrix proceeds in the product basis of the disordered (dEJ, dCJ)
    Zero-Pi qubit on one hand and the chi LC oscillator on the other hand.

    Expected parameters are:

    EJ:    mean Josephson energy of the two junctions
    EL:    inductive energy of the two (super-)inductors
    ECJ:   charging energy associated with the two junctions
    ECS:   charging energy including the large shunting capacitances
    EC:    charging energy associated with chi degree of freedom
    dEJ:   relative disorder in EJ, i.e., (EJ1-EJ2)/EJ(mean)
    dEL:   relative disorder in EL, i.e., (EL1-EL2)/EL(mean)
    dCJ:   relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/C(mean)
    flux:  magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    zeropi_cutoff: cutoff in the number of states of the disordered zero-pi qubit
    chi_cut: cutoff in the chi oscillator basis (Fock state basis)
    grid: Grid object specifying the range and spacing of the discretization lattice


    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined as
    relative ones.
    """

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'EC': 'charging energy associated with chi degree of freedom',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'dC': 'relative deviation between the two shunt capacitances',
        'dEL': 'relative deviation between the two inductances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'zeropi_cutoff': 'cutoff in the number of states of the disordered zero-pi qubit',
        'chi_cutoff': 'cutoff in the chi oscillator basis (Fock state basis)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    __initialized = False

    def init_parameters(self, **parameter_args):
        for parameter_name, parameter_val in parameter_args.items():
            setattr(self, parameter_name, parameter_val)

    def __init__(self, **parameter_args):
        super(FullZeroPi_ProductBasis, self).__init__(**parameter_args)
        self._qubit_type = 'full 0-Pi circuit (phi, theta, chi) in 0pi - chi product basis'
        self._zeropi = DisZeroPi(
            EJ=self.EJ,
            EL=self.EL,
            ECJ=self.ECJ,
            ECS=self.ECS,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
            flux=self.flux,
            grid=self.grid
        )
        self.__initialized = True

    def __setattr__(self, parameter_name, parameter_val):
        super(FullZeroPi_ProductBasis, self).__setattr__(parameter_name, parameter_val)
        if self.__initialized and parameter_name in self._zeropi._EXPECTED_PARAMS_DICT.keys():
                self._zeropi.__setattr__(parameter_name, parameter_val)

    def hamiltonian(self):
        zeropi_dim = self.zeropi_cutoff
        zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)
        zeropi_diag_hamiltonian = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.float_)
        zeropi_diag_hamiltonian.setdiag(zeropi_evals)

        chi_dim = self.chi_cutoff
        prefactor = (8.0 * self.EL * self.EC)**0.5
        chi_diag_hamiltonian = op.number_sparse(chi_dim, prefactor)

        hamiltonian_mat = sp.sparse.kron(zeropi_diag_hamiltonian, sp.sparse.identity(chi_dim, format='dia', dtype=np.float_))
        hamiltonian_mat += sp.sparse.kron(sp.sparse.identity(zeropi_dim, format='dia', dtype=np.float_), chi_diag_hamiltonian)

        gmat = self.g_coupling_matrix(zeropi_evecs)
        zeropi_coupling = sp.sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.float_)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                zeropi_coupling += gmat[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)
        hamiltonian_mat += sp.sparse.kron(zeropi_coupling, op.annihilation_sparse(chi_dim) + op.creation_sparse(chi_dim))
        return hamiltonian_mat

    def hilbertdim(self):
        return (self.zeropi_cutoff * self.chi_cutoff)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return evals

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.sparse.linalg.eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, which='SA')
        return evals, evecs

    def g_phi_coupling_matrix(self, zeropi_states):
        """Returns a matrix of coupling strengths g^\phi_{ll'} [cmp. Dempster et al., Eq. (18)], using the states
        from the list 'zeropi_states'. Most commonly, 'zeropi_states' will contain eigenvectors of the
        DisZeroPi type, so 'transpose' is enabled by default.
        """
        prefactor = self.EL * self.dEL * (8.0 * self.EC / self.EL)**0.25
        return (prefactor * matrixelem_table(self._zeropi.phi_operator(), zeropi_states, real_valued=True))

    def g_theta_coupling_matrix(self, zeropi_states):
        """Returns a matrix of coupling strengths i*g^\theta_{ll'} [cmp. Dempster et al., Eq. (17)], using the states
        from the list 'zeropi_states'. Most commonly, 'zeropi_states' will contain eigenvectors, so 'transpose' is enabled by
        default.
        """
        prefactor = - self.ECS * self.dC * (32.0 * self.EL / self.EC)**0.25
        return (prefactor * matrixelem_table(self._zeropi.d_dtheta_operator(), zeropi_states, real_valued=True))

    def g_coupling_matrix(self, zeropi_states, evals_count=6):
        """Returns a matrix of coupling strengths g_{ll'} [cmp. Dempster et al., text above Eq. (17)], using the states
        from 'state_list'.  Most commonly, 'zeropi_states' will contain eigenvectors of the
        DisZeroPi type, so 'transpose' is enabled by default.
        If zeropi_states==None, then a set of self.zeropi eigenstates is calculated. Only in that case is evals_count
        used for the eigenstate number (and hence the coupling matrix size).

        """
        if zeropi_states is None:
            _, zeropi_states = self._zeropi.eigensys(evals_count=evals_count)
        return (self.g_phi_coupling_matrix(zeropi_states) + self.g_theta_coupling_matrix(zeropi_states))

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction1d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d'")


# ----------------------------------------------------------------------------------------


class ModZeroPi(SymZeroPi):

    """[still experimental] modified version of the symmetric 0-pi qubit based on replacing inductors
    by Josephson junctions"""

    _EXPECTED_PARAMS_DICT = {
        'EJ': 'Josephson energy',
        'EJp': 'Josephson energy of the junction substituting the inductor',
        'ECphi': 'charging energy for phi term',
        'ECth': 'charging energy for theta term',
        'ECchi': 'charging energy for chi term',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'grid': 'Grid object specifying the range and spacing of the discretization lattice'
    }
    _OPTIONAL_PARAMS_DICT = {'truncated_dim': 'dimension parameter for truncated system (used in interface to qutip)'}

    def __init__(self, **parameter_args):
        super(ModZeroPi, self).__init__(**parameter_args)
        self.parameter._qubit_type = 'modified symmetric 0-Pi qubit (EL->EJp)'
        print("Implementation of this 0-pi device variation is still experimental & not complete")

    def potential(self, phi, theta, chi):
        return -2.0 * np.cos(phi) * (self.EJ * np.cos(theta + 2.0 * np.pi * self.flux / 2) + self.EJp * np.cos(chi)) + 2 * (self.EJ + self.EJp)

    def sparse_kineticmat(self):
        dphi2 = grid_second_derivative(0, self.grid, prefactor=-2.0 * self.ECphi, periodic=True)   # CJ + CJp
        dth2 = grid_second_derivative(1, self.grid, prefactor=-2.0 * self.ECth, periodic=True)     # C + CJ
        dchi2 = grid_second_derivative(2, self.grid, prefactor=-2.0 * self.ECchi, periodic=True)     # C + CJp
        return (dphi2 + dth2 + dchi2)
