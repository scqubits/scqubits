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
import progress_bar  # for displaying progress bars during calculations


# ---Auxiliary routings and qutip wrappers ---------------------------------------------------------


def diag_hmltn(qubit, **kwargs):
    """Returns a qt.Qobj which has the eigenenergies of the object 'qubit' on the diagonal."""
    for pmname, pmval in kwargs.items():
        setattr(qubit.pm, pmname, pmval)
    evnum = qubit.dim
    evals = qubit.eigenvals(evnum=evnum)
    return qt.Qobj(inpt=np.diagflat(evals))


def op(opr, sys, subspaces):
    """Wrapper for qutip. The operator opr may be a list, numpy array, or qt.Qobj. In the former
    two cases, opr is first converted into qt.Qobj. In all cases, tensor_op is classed to concatenate
    the operator with the right number and dimensions of identity matrices.
    """
    if type(opr) in [list, np.ndarray]:
        qt_op = qt.Qobj(inpt=opr[:sys.dim, :sys.dim])
    else:
        qt_op = opr
    return tensor_op(qt_op, sys, subspaces)


def tensor_op(qtp_op, sys, subspaces):
    """'subspaces' is a record of the Hilbert space structure, in the form [obj1, obj2, ...]
    where each object may be an object from sc_qubits or a GenerQSys object from the qutipwrapper module.

    """
    op_list = [qt.operators.qeye(system.dim)
               for _, system in enumerate(subspaces)]
    sys_index = subspaces.index(sys)
    op_list[sys_index] = qtp_op
    return qt.tensor(op_list)


def hbd(j, k, sys, subspaces=None):
    """Hubbard operator |j><k| for system 'sys'"""
    dim = sys.dim
    op = (qt.states.basis(dim, j) * qt.states.basis(dim, k).dag())
    if subspaces is None:
        return op
    else:
        return tensor_op(op, sys, subspaces)


def subtr_gnd(energies):
    """Takes a list of energies (such as obtained by diagonalization) and returns them after
    subtracting the ground state energy."""
    return (energies - energies[0])

def subtr_energy(energies, e0):
    """Takes a list of energies (such as obtained by diagonalization) and returns them after
    subtracting the energy e0."""
    return (energies - e0)

# ---Matrix elements and operators---------------------------------------------------------


def matrix_element(state1, operator, state2):
    """Calculate the matrix element <state1|operator|state2>.
    state1, state2: numpy arrays
    operator:       numpy array or sparse matrix object
    """
    if isinstance(operator, np.ndarray):    # Is operator given in dense form?
        return (np.vdot(state1, np.dot(operator, state2)))  # Yes - use numpy's 'vdot' and 'dot'.
    else:
        return (np.vdot(state1, operator.dot(state2)))      # No, operator is sparse. Must use its own 'dot' method.


def matrixelem_table(operator, vlist, transpose=True, real_valued=False):
    """Calculate a table of matrix elements based on
    operator: numpy array or sparse matrix object
    vlist:    list (or array) of numpy arrays representing the states |v0>, |v1>, ...

    Returns a numpy array corresponding to the matrix element table
    <v0|operator|v0>   <v0|operator|v1>   ...
    <v1|operator|v0>   <v1|operator|v1>   ...
          ...                 ...

    If transpose=True, then vlist is transposed (scipy's eigsh yields eigenvector array in transposed form)
    """
    if transpose:
        vec_list = vlist.T
    else:
        vec_list = vlist

    if real_valued:
        dtp = np.float_
    else:
        dtp = np.complex_

    tablesize = len(vec_list)
    mtable = np.empty(shape=[tablesize, tablesize], dtype=dtp)
    for n in range(tablesize):
        for m in range(n + 1):
            mtable[n, m] = matrix_element(vec_list[n], operator, vec_list[m])
            if real_valued:
                mtable[m, n] = mtable[n, m]
            else:
                mtable[m, n] = np.conj(mtable[n, m])
    return mtable


def annihilate(dimension):
    """Returns a dense matrix of size dimension x dimension representing the annihilation operator."""
    offd_elements = [math.sqrt(i) for i in range(1, dimension)]
    return np.diagflat(offd_elements, 1)


def create(dimension):
    """Returns a dense matrix of size dimension x dimension representing the creation operator."""
    return annihilate(dimension).T


def number(dimension, prefac=None):
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor can be directly
    included in the generation of the matrix by supplying 'prefac'.
    """
    diag = np.arange(dimension)
    if prefac:
        diag = diag * prefac
    return np.diagflat(diag)


def sparse_annihilate(dimension):
    """Returns a matrix of size dimension x dimension representing the annihilation operator
    in the format of a scipy sparse.csc.
    """
    offd_elements = [math.sqrt(i) for i in range(dimension)]
    return sp.sparse.dia_matrix((offd_elements, [1]), shape=(dimension, dimension)).tocsc()


def sparse_create(dimension):
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc.
    """
    return sparse_annihilate(dimension).transpose().tocsc()


def sparse_number(dimension, prefac=None):
    """Number operator matrix of size dimension x dimension in sparse matrix representation. An additional prefactor can be directly
    included in the generation of the matrix by supplying 'prefac'.
    """
    diag = np.arange(dimension)
    if prefac:
        diag = diag * prefac
    return sp.sparse.dia_matrix((diag, [0]), shape=(dimension, dimension), dtype=np.float_)


def sparse_potentialmat(min_max_pts, potential_func):
    """Returns the potential energy matrix for the potential given by 'potential_func', in sparse
    (dia_matrix) form.
    """
    varmin = min_max_pts[:, 0]
    varmax = min_max_pts[:, 1]
    varpts = min_max_pts[:, 2]
    hilbertdim = int(np.prod(varpts))
    nr_of_variables = len(varmin)
    Xvals = [np.linspace(varmin[j], varmax[j], varpts[j]) for j in range(nr_of_variables)]  # list of coordinate arrays
    diag = np.empty([1, hilbertdim], dtype=np.float_)

    for j, coord_tuple in enumerate(itertools.product(*Xvals)):
        diag[0][j] = potential_func(*coord_tuple)   # diagonal matrix elements
    return sp.sparse.dia_matrix((diag, [0]), shape=(hilbertdim, hilbertdim))


def sparse_hubbardmat(dimension, j1, j2):
    """The Hubbard operator |j1><j2| is returned as a matrix of linear size 'dimension'."""
    hubbardmat = sp.sparse.dok_matrix((dimension, dimension), dtype=np.float_)
    hubbardmat[j1, j2] = 1.0
    return hubbardmat.asformat('dia')

# ---Harmonic oscillator--------------------------------------------------------------------


def harm_osc_wavefunction(n, x, losc):
    """For given quantum number n=0,1,2,... this returns the value of the harmonic oscillator
    harmonic oscillator wave function \psi_n(x) = N H_n(x/losc) exp(-x^2/2losc), N being the
    proper normalization factor.
    """
    return ((2**n * math.factorial(n) * losc)**(-0.5) * np.pi**(-0.25) *
            sp.special.eval_hermite(n, x / losc) * np.exp(-(x * x) / (2 * losc * losc)))


# ---Plotting-------------------------------------------------------------------------------


def contourplot(x_list, y_list, func, levls=None, aspect_ratio=None, to_file=False):
    """Contour plot of a 2d function 'func(x,y)'.
    x_list: (ordered) list of x values for the x-y evaluation grid
    y_list: (ordered) list of y values for the x-y evaluation grid
    func: function f(x,y) for which contours are to be plotted
    levls: contour levels can be specified if so desired

    """
    x_grid, y_grid = np.meshgrid(x_list, y_list)
    z_array = func(x_grid, y_grid)
    # print(z_array)
    if aspect_ratio is None:
        plt.figure(figsize=(x_list[-1] - x_list[0], y_list[-1] - y_list[0]))
    else:
        w, h = plt.figaspect(aspect_ratio)
        plt.figure(figsize=(w, h))

    if levls is None:
        plt.contourf(x_grid, y_grid, z_array, cmap=plt.cm.viridis)
    else:
        plt.contourf(x_grid, y_grid, z_array, levels=levls, cmap=plt.cm.viridis)

    if to_file:
        out_file = mplpdf.PdfPages(to_file)
        out_file.savefig()
        out_file.close()
    return None


def plot_matrixelements(mtable, mode='abs', xlabel='', ylabel='', zlabel='', to_file=False):
    """Create a "skyscraper" and a color-coded plot of the matrix element table given as 'mtable'"""
    mode_dict = {'abs2': (lambda x: np.abs(x)**2),
                 'abs': (lambda x: np.abs(x)),
                 'real': (lambda x: np.real(x)),
                 'imag': (lambda x: np.imag(x))}
    matsize = len(mtable)
    num_elem = matsize**2   # num. of elements to plot
    xgrid, ygrid = np.meshgrid(range(matsize), range(matsize))
    xgrid = xgrid.T.flatten() - 0.5  # center bars on integer value of x-axis
    ygrid = ygrid.T.flatten() - 0.5  # center bars on integer value of y-axis
    zvals = np.zeros(num_elem)       # all bars start at z=0
    dx = 0.75 * np.ones(num_elem)      # width of bars in x-direction
    dy = dx.copy()      # width of bars in y-direction (same as x-dir here)
    dz = mode_dict[mode](mtable).flatten()  # height of bars from density matrix elements (should use 'real()' if complex)

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
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.viridis, norm=nrm)

    plt.matshow(mode_dict[mode](mtable), cmap=plt.cm.viridis)
    plt.show()

    if to_file:
        out_file = mplpdf.PdfPages(to_file)
        out_file.savefig()
        out_file.close()
    return None


def cmap_plot(x_param, x_list, y_param, y_vals, t_param, t_vals, norm_min, norm_max, x_min=False, x_max=False, y_min=False, y_max=False, colormap='jet', x_figsize=15, y_figsize=10, line_width=2):
    """Takes a list of x-values, 
    a list of lists with each element containing the y-values corresponding to a particular curve, 
    a list of lists with each element containing the external parameter value (t-value) 
    that determines the color of each curve at each y-value,
    and a normalization interval for the t-values."""
    y = []
    for i in range(len(y_vals)):
        for p in range(len(y_vals[i])):
            y.append(y_vals[i][p])
    fig = plt.figure(figsize=(x_figsize,y_figsize))
    for i in range(len(y_vals)):
        pts = np.array([x_list,y_vals[i]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
        lc = LineCollection(segs, cmap=plt.get_cmap(colormap),norm=plt.Normalize(norm_min, norm_max))
        lc.set_array(np.array(t_vals[i]))
        lc.set_linewidth(line_width)
        plt.gca().add_collection(lc)
    
    plt.xlabel(x_param)
    if x_min and x_max:
        plt.xlim(x_min, x_max)
    elif x_min:
        plt.xlim(x_min, max(x_list))
    elif x_max:
        plt.xlim(min(x_list), x_max)
    else:
        plt.xlim(min(x_list), max(x_list))
        
    plt.ylabel(y_param)
    if y_min and y_max:
        plt.ylim(y_min, y_max)
    elif y_min:
        plt.ylim(y_min, max(y))
    elif y_max:
        plt.ylim(min(y), y_max)
    else:
        plt.ylim(min(y), max(y))
    
    axcb = fig.colorbar(lc)
    axcb.set_label(t_param)
    plt.show()


# ---Parameter checking and storing----------------------------------------------------------


def valid_parameters(expected_params_dict, optional_params_dict, given_params_dict):
    """Checks whether the keyword argumments provided (given_params_dict) match the
    keyword arguments expected for a certain type of qubit (expected_params_dict).
    Returns True when the two match exactly (no missing, no superfluous arguments).
    """
    if 'force_valid' in given_params_dict:      # this is a hack: may need to allow class instatiation without parameters (see ZeroPiFull_ProductBasis)
        if given_params_dict['force_valid']:
            return True
    for expected_key in expected_params_dict:
        if expected_key not in given_params_dict:
            print('>>Error<<: one or multiple parameter(s) have not been assigned values.')
            print('Expected parameters are:')
            for k, v in expected_params_dict.items():
                print("{:<5} {:<40} ".format(k, v))
            return False

    for given_key in given_params_dict:
        if given_key not in expected_params_dict and given_key not in optional_params_dict:
            print('>>Error<<: one or multiple of the specified parameters is/are unknown.')
            print('Expected parameters are:')
            for k, v in expected_params_dict.items():
                print("{:<5} {:<40} ".format(k, v))
            return False
    return True


class Parameters(object):

    """Simple class to hold all physical parameters, cutoffs, etc. relevant to a specific qubit.
    Initialize with the appropriate keyword arguments, e.g.,
    >>> params = Parameters(EJ=2.0, EC=0.1, ng=0.0, ncut=20)
    If the specific qubit class works with parameter discretization, then min_max_pts=[[var1min, var1max, var1pts], [var2min,...]]
    gives the lower and upper bounds for each variable and the number of points within each such interval.
    """

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
                setattr(self, key, val)

    def __repr__(self):
        output = 'QUBIT PARAMETERS -------'
        for key in sorted(self.__dict__.keys()):
            if str(key) == '_min_max_pts':
                output += '\nvariables min, max, pts\t: ' + str(self.__dict__[key])
            else:
                output += '\n' + str(key) + '\t: ' + str(self.__dict__[key])
        return output


# ---Generic quantum system contained and Qubit base class---------------------------------------------------------------------


class GenericQSys(object):

    """Generic quantum system class, blank exccept for holding the truncation parameter 'dim'.
    The main purpose is as a wrapper for interfacing with qutip. E.g., a resonator could be
    resonator = GenericQSys(dim=4)
    In this case, photon states n=0,1,2,3 would be retained.
    """
    def __init__(self, dim):
        self.pm = Parameters(dim=dim)

    @property
    def dim(self):
        return self.pm.dim


class QubitBaseClass(object):

    """Base class for superconducting qubit objects. Provide general mechanisms and routines for
    checking validity of initialization parameters, writing data to files, and plotting.
    """

    _expected_parameters = {}
    _optional_parameters = {'dim': 'dimension parameter for truncated system (used in qubitwrapper interface to qutip)'}

    def __init__(self, **kwargs):
        if not valid_parameters(self._expected_parameters, self._optional_parameters, kwargs):
            raise UserWarning('Parameter mismatch')
        else:
            self.pm = Parameters(**kwargs)
            self.pm._qubit_type = 'generic - no use other than serving as class template'

    def __repr__(self):
        """Printable representation of the object. Redirects to the parameter object pm to output
        the qubit parameters.
        """
        output = self.pm.__repr__()
        output += '\nHilbert space dimension\t: ' + str(self.hilbertdim())
        return output

    @property
    def dim(self):            # helper function for qubitwrapper
        return self.pm.dim

    def hilbertdim(self):
        """Must be implemented in child classes"""
        pass

    def name(self):
        """Returns the 'pm._qubit_type'."""
        return self.pm._qubit_type

    def _evals_calc(self, evnum):
        """Must be implemented in child classes"""
        pass

    def _esys_calc(self, evnum):
        """Must be implemented in child classes"""
        pass

    def hamiltonian(self):
        """Must be implemented in child classes"""
        pass

    def plot_wavefunction(self):
        """Must be implemented in child classes"""
        pass

    def eigenvals(self, evnum=6, to_file=None):
        """Calculates eigenvalues (via qubit-specific _evals_calc()), and returns a numpy array of eigenvalues.

        evnum:   number of desired eigenvalues (sorted from smallest to largest)
        to_file: write data to file if path and filename are specified
        """
        evals = np.sort(self._evals_calc(evnum))
        if to_file:
            print("Writing eigenvals data and parameters to {} and {}.".format(to_file + "evals.csv", to_file + ".prm"))
            np.savetxt(to_file + "evals.csv", evals, delimiter=",")
            with open(to_file + ".prm", "w") as text_file:
                text_file.write(self.pm.__repr__())
        return evals

    def eigensys(self, evnum=6, to_file=None):
        """Calculates eigenvalues and corresponding eigenvectores (via qubit-specific _esys_calc()), and returns
        two numpy arrays containing the eigenvalues and eigenvectors, respectively.

        evnum:   number of desired eigenvalues (sorted from smallest to largest)
        to_file: write data to file if path and filename are specified
        """
        evals, evecs = self._esys_calc(evnum)
        sorted_indices = evals.argsort()  # eigsh does not guarantee consistent ordering within result?! http://stackoverflow.com/questions/22806398
        evals = evals[sorted_indices]
        evecs = evecs[:, sorted_indices]
        if to_file:
            print("Writing eigensys data and parameters to {}, {}, and {}.".format(to_file + "evals.csv", to_file + "evecs.csv", to_file + ".prm"))
            np.savetxt(to_file + "evals.csv", evals, delimiter=",")
            np.savetxt(to_file + "evecs.csv", evecs.T, delimiter=",")
            with open(to_file + ".prm", "w") as text_file:
                text_file.write(self.pm.__repr__())
        return evals, evecs

    def matrixelements(self, operator, esys, evnum):
        """Returns a table of matrix elements for 'operator', given as a string referring to a class method
        that returns an operator matrix. E.g., for 'transmon = QubitTransmon(...)', the matrix element table
        for the charge operator n can be accessed via 'transmon.op_matrixelement_table('n', esys=None, evnum=6).
        When 'esys' is set to None, the eigensystem with 'evnum' eigenvectors is calculated.
        """
        if esys is None:
            _, evecs = self.eigensys(evnum)
        else:
            _, evecs = esys
        opmatrix = getattr(self, operator)()
        return matrixelem_table(opmatrix, evecs)

    def get_evals_vs_paramvals(self, param, paramval_list, evnum=6, subtract_ground=False, to_file=None):
        """Calculates a set of eigenvalues as a function of the parameter 'param', where the discrete values
        for 'param' are contained in the list prmval_list. Returns a numpy array where specdata[n] is set
        of eigenvalues calculated for parameter value prmval_list[n]

        param:           string, gives name of parameter to be varied
        prmval_list:     list of parameter values to be plugged in for param
        subtract_ground: if True, then eigenvalues are returned relative to the ground state eigenvalues
                         (useful if transition energies from ground state are the relevant quantity)
        evnum:           number of desired eigenvalues (sorted from smallest to largest)
        to_file:         write data to file if path and filename are specified
        """
        saved_prmval = getattr(self.pm, param)
        nr_paramvals = len(paramval_list)
        specdata = np.empty((nr_paramvals, evnum))
        print("")
        progress_bar.update_progress(0)
        for ind, paramval in enumerate(paramval_list):
            setattr(self.pm, param, paramval)
            evals = self.eigenvals(evnum)
            specdata[ind] = evals
            if subtract_ground:
                specdata[ind] = evals - evals[0]
            else:
                specdata[ind] = evals
            progress_bar.update_progress((ind + 1) / nr_paramvals)
        setattr(self.pm, param, saved_prmval)
        if to_file:
            print("Writing eigenval data and parameters to {}, {}, and {}.".format(to_file + '_' + param + '.csv',
                  to_file + "_evals.csv", to_file + ".prm"))
            np.savetxt(to_file + '_' + param + '.csv', paramval_list, delimiter=",")
            np.savetxt(to_file + "_evals.csv", specdata, delimiter=",")
            with open(to_file + ".prm", "w") as text_file:
                text_file.write(self.pm.__repr__())
        if subtract_ground:
            return specdata[:, 1:]
        else:
            return specdata

    def plot_evals_vs_paramvals(self, param, paramval_list, evnum=6,
                                yrange=False, subtract_ground=False, shift=0, to_file=None):
        """Generates a simple plot of a set of eigenvalues as a function of parameter 'param'.
        The individual points correspond to the parameter values listed in paramval_list.

        param:           string, gives name of parameter to be varied
        paramval_list:     list of parameter values to be plugged in for param
        subtract_ground: if True, then eigenvalues are returned relative to the ground state eigenvalues
                         (useful if transition energies from ground state are the relevant quantity)
        evnum:           number of desired eigenvalues (sorted from smallest to largest)
        yrange:          [ymin, ymax] -- custom y-range for the plot
        shift:           apply a shift of this size to all eigenvalues
        to_file:         write graphics and parameter set to file if path and filename are specified
        """
        x = paramval_list
        y = self.get_evals_vs_paramvals(param, paramval_list, evnum, subtract_ground)
        if not yrange:
            plt.axis([np.amin(x), np.amax(x), np.amin(y + shift), np.amax(y + shift)])
        else:
            plt.axis([np.amin(x), np.amax(x), yrange[0], yrange[1]])
        plt.xlabel(param)
        plt.ylabel("energy")
        plt.plot(x, y + shift)
        if to_file:
            print("Writing graphics and parameters to {} and {}.".format(to_file + ".pdf", to_file + ".prm"))
            out_file = mplpdf.PdfPages(to_file + ".pdf")
            out_file.savefig()
            out_file.close()
            with open(to_file + ".prm", "w") as text_file:
                text_file.write(self.pm.__repr__())
        plt.show()
        return None

    @staticmethod
    def _plot_wavefunction1d(wavefunc_values, potential_values, x_values, offset=0, scaling=1,
                             ylabel='wavefunction', xlabel='x'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_values, offset + scaling * wavefunc_values)
        if potential_values is not None:
            ax.plot(x_values, potential_values)
            ax.plot(x_values, [offset] * len(x_values), 'b--')

        ax.set_xlim(xmin=x_values[0], xmax=x_values[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
        return None

    @staticmethod
    def _plot_wavefunction1d_discrete(wavefunc_values, x_values, ylabel='wavefunction', xlabel='x'):
        width = .75
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(x_values, wavefunc_values, width=width)
        ax.set_xlim(xmin=x_values[0], xmax=x_values[-1])
        ax.set_xticks(x_values + width / 2)
        ax.set_xticklabels(x_values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
        return None

    @staticmethod
    def _plot_wavefunction2d(wavefunc, min_max_pts, figsize, aspect_ratio, zero_calibrate=False):
        plt.figure(figsize=figsize)

        if zero_calibrate:
            absmax = np.max(np.abs(wavefunc))
            minval = -absmax
            maxval = absmax
            cmap = plt.get_cmap('PRGn')
        else:
            minval = np.min(wavefunc)
            maxval = np.max(wavefunc)
            cmap = plt.cm.viridis

        plt.imshow(wavefunc, extent=[min_max_pts[0][0], min_max_pts[0][1], min_max_pts[1][0],
                                     min_max_pts[1][1]], aspect=aspect_ratio, cmap=cmap,
                   vmin=minval, vmax=maxval)
        plt.colorbar(fraction=0.017, pad=0.04)
        plt.show()
        return None


# ---Cooper pair box / transmon-----------------------------------------------------------


class QubitTransmon(QubitBaseClass):

    """Class for the Cooper pair box / transmon qubit. Hamiltonian is represented in dense form. Expected parameters:
        EJ:   Josephson energy
        EC:   charging energy
        ng:   offset charge
        ncut: charge basis cutoff, n = -ncut, ..., ncut'

    Initialize with, e.g.
    >>> qubit = QubitTransmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)
    """

    _expected_parameters = {
        'EJ': 'Josephson energy',
        'EC': 'charging energy',
        'ng': 'offset charge',
        'ncut': 'charge basis cutoff, n = -ncut, ..., ncut'
    }

    def __init__(self, **kwargs):
        super(QubitTransmon, self).__init__(**kwargs)
        self.pm._qubit_type = 'Cooper pair box'

    def hamiltonian(self):
        EJ = self.pm.EJ
        EC = self.pm.EC
        ng = self.pm.ng
        ncut = self.pm.ncut
        dimension = 2 * ncut + 1
        hmat = np.zeros((dimension, dimension), dtype=np.float_)
        for i in range(dimension):
            hmat[i][i] = 4.0 * EC * (i - ncut - ng)**2
        for i in range(dimension - 1):
            hmat[i][i + 1] = -EJ / 2.0
            hmat[i + 1][i] = -EJ / 2.0
        return hmat

    def hilbertdim(self):
        return (2 * self.pm.ncut + 1)

    def n(self):
        """Charge operator in charge basis for the transmon or CPB qubit."""
        ncut = self.pm.ncut
        diag = np.arange(-ncut, ncut + 1, 1)
        return np.diagflat(diag)

    def _evals_calc(self, evnum):
        hmat = self.hamiltonian()
        return sp.linalg.eigh(hmat, eigvals_only=True, eigvals=(0, evnum - 1))

    def _esys_calc(self, evnum):
        hmat = self.hamiltonian()
        return sp.linalg.eigh(hmat, eigvals_only=False, eigvals=(0, evnum - 1))

    def plot_wavefunction(self, esys, basis='number', which=0, nrange=[-10, 10],
                          phirange=[-np.pi, np.pi], phipts=151, mode='abs2'):
        """Plots the wave function with index 'which', within mode:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        For the plot, the basis can be chosen as basis='number' or 'phase'.
        """
        mode_dict = {'abs2': (lambda x: np.abs(x)**2),
                     'abs': (lambda x: np.abs(x)),
                     'real': (lambda x: np.real(x)),
                     'imag': (lambda x: np.imag(x))}

        if basis == 'number':
            n_values, wavefunc, _ = self.wavefunction(esys, basis, which, phirange=phirange, phipts=phipts)
            wavefunc = mode_dict[mode](wavefunc)
            self._plot_wavefunction1d_discrete(wavefunc, n_values)

        elif basis == 'phase':
            phi_values, wavefunc, evalue = self.wavefunction(esys, basis, which, phirange=phirange, phipts=phipts)
            intermediate_index = int(len(wavefunc) / 3)    # intermediate position for extracting phase (dangerous in tail or midpoint)
            wavefunc = wavefunc * cmath.exp(-1j * cmath.phase(wavefunc[intermediate_index]))
            wavefunc = mode_dict[mode](wavefunc)
            self._plot_wavefunction1d(wavefunc, -self.pm.EJ * np.cos(phi_values), phi_values,
                                      offset=evalue, scaling=0.3 * self.pm.EJ, xlabel='phi')

    def wavefunction(self, esys, basis='number', which=0, nrange=[-10, 10],
                     phirange=[-np.pi, np.pi], phipts=251):
        """Return the transmon wave function in basis='number' or 'phase' basis. The specific index of the wavefunction is: 'which'.
        'esys' can be provided, but if set to 'None' then it is calculated frst.
        """
        evnum = max(which + 1, 3)
        if esys is None:
            evals, evecs = self.eigensys(evnum)
        else:
            evals, evecs = esys

        nmax = (len(evecs[:, which]) - 1) // 2
        if basis == 'number':
            n_values = np.arange(nrange[0], nrange[1] + 1)
            psi_n_values = evecs[(nmax + nrange[0]):(nmax + nrange[1] + 1), which]
            return n_values, psi_n_values, evals[which]
        elif basis == 'phase':
            n_values = np.arange(-nmax, nmax + 1)
            phi_values = np.linspace(phirange[0], phirange[1], phipts)
            psi_n_values = evecs[:, which]
            wavefunc = np.empty(phipts, dtype=np.complex_)
            for k in range(phipts):
                wavefunc[k] = ((1.0 / math.sqrt(2 * np.pi)) *
                               np.sum(psi_n_values * np.exp(1j * phi_values[k] * n_values)))
            return phi_values, wavefunc, evals[which]


# ---Fluxonium qubit ------------------------------------------------------------------------


class QubitFluxonium(QubitBaseClass):

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
    >>> qubit = QubitFluxonium(EJ=1.0, EC=2.0, EL=0.3, flux=0.2, cutoff=120)
    """

    _expected_parameters = {
        'EJ': 'Josephson energy',
        'EC': 'charging energy',
        'EL': 'inductive energy',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'cutoff': 'number of harm. osc. basis states used in diagonalization',
    }

    def __init__(self, **kwargs):
        super(QubitFluxonium, self).__init__(**kwargs)
        self.pm._qubit_type = 'fluxonium'

    def phi(self):
        """Returns the phi operator in the LC harmonic oscillator basis"""
        EC = self.pm.EC
        EL = self.pm.EL
        dimension = self.pm.cutoff
        phi0 = (8.0 * EC / EL)**(0.25)        # LC oscillator length
        return (create(dimension) + annihilate(dimension)) * phi0 / math.sqrt(2)

    def n(self):
        """Returns the n = - i d/dphi operator in the LC harmonic oscillator basis"""
        EC = self.pm.EC
        EL = self.pm.EL
        dimension = self.pm.cutoff
        phi0 = (8.0 * EC / EL)**(0.25)        # LC oscillator length
        return 1j * (create(dimension) - annihilate(dimension)) / (phi0 * math.sqrt(2))

    def hamiltonian(self):           # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harm. osc. basis and return as sparse.dia_matrix"""
        EJ = self.pm.EJ
        EC = self.pm.EC
        EL = self.pm.EL
        flux = self.pm.flux
        dimension = self.pm.cutoff

        om = math.sqrt(8.0 * EL * EC)         # plasma osc. frequency

        diag = [i * om for i in range(dimension)]
        LCmat = np.diagflat(diag)

        exp_arg = 1j * self.phi()
        exp_mat = 0.5 * sp.linalg.expm(exp_arg) * cmath.exp(1j * 2 * np.pi * flux)
        cos_mat = exp_mat + np.conj(exp_mat.T)
        hmat = LCmat - EJ * cos_mat
        return hmat
    
    def hilbertdim(self):
        return (self.pm.cutoff)   

    def _evals_calc(self, evnum):
        hmat = self.hamiltonian()
        return sp.linalg.eigh(hmat, eigvals_only=True, eigvals=(0, evnum - 1))

    def _esys_calc(self, evnum):
        hmat = self.hamiltonian()
        return sp.linalg.eigh(hmat, eigvals_only=False, eigvals=(0, evnum - 1))

    def potential(self, phi):
        p = self.pm
        return (0.5 * p.EL * phi * phi - p.EJ * np.cos(phi + 2.0 * np.pi * p.flux))

    def wavefunction(self, esys, which=0, phirange=[-6 * np.pi, 6 * np.pi], phipts=251):
        evnum = max(which + 1, 3)
        if esys is None:
            evals, evecs = self.eigensys(evnum)
        else:
            evals, evecs = esys

        ncut = len(evecs[:, which])
        phi_values = np.linspace(phirange[0], phirange[1], phipts)
        psi_n_values = evecs[:, which]
        wavefunc = np.zeros(phipts, dtype=np.complex_)
        harmonic_osc_values = np.empty(phipts, dtype=np.float_)
        phi_losc = (8 * self.pm.EC / self.pm.EL)**0.25
        for n in range(ncut):
            harmonic_osc_values = harm_osc_wavefunction(n, phi_values, phi_losc)  # fixed order of hermite polynomial
            wavefunc = wavefunc + psi_n_values[n] * harmonic_osc_values
        return phi_values, wavefunc, evals[which]

    def plot_wavefunction(self, esys, which=0, phirange=[-6 * np.pi, 6 * np.pi], mode='abs2', phipts=251):
        """Different modes:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        mode_dict = {'abs2': (lambda x: np.abs(x)**2),
                     'abs': (lambda x: np.abs(x)),
                     'real': (lambda x: np.real(x)),
                     'imag': (lambda x: np.imag(x))}

        phi_values, wavefunc, evalue = self.wavefunction(esys, which, phirange, phipts)
        intermediate_index = int(len(wavefunc) / 3)    # intermediate position for extracting phase (dangerous in tail or midpoint)
        wavefunc = wavefunc * cmath.exp(-1j * cmath.phase(wavefunc[intermediate_index]))
        wavefunc = mode_dict[mode](wavefunc)
        self._plot_wavefunction1d(wavefunc, self.potential(phi_values), phi_values,
                                  offset=evalue, scaling=5 * self.pm.EJ, xlabel='phi')
        return None

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction2d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction2'")


# ---Fluxonium qubit with SQUID loop----------------------------------------------------------------------


class QubitFluxoniumSQUID(QubitFluxonium):

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
    >>> qubit = QubitFluxoniumSQUID(EJ1=1.0, EJ2=1.0, EC=2.0, EL=0.3, flux=0.2, fluxsquid=0.1, cutoff=120)
    """

    _expected_parameters = {
        'EJ1': 'Josephson energy 1',
        'EJ2': 'Josephson energy 2',
        'EC': 'charging energy',
        'EL': 'inductive energy',
        'flux': 'external magnetic flux through primary loop in units of flux quanta (h/2e)',
        'fluxsquid': 'external magnetic flux through the SQUID loop in units of flux quanta (h/2e)',
        'cutoff': 'number of harm. osc. basis states used in diagonalization'
    }

    def __init__(self, **kwargs):
        super(QubitFluxoniumSQUID, self).__init__(**kwargs)
        self.pm._qubit_type = 'Fluxonium SQUID'

    def hamiltonian(self):
        """Construct Hamiltonian matrix in harm. osc. basis and return as sparse.dia_matrix"""
        EJ1 = self.pm.EJ1
        EJ2 = self.pm.EJ2
        EC = self.pm.EC
        EL = self.pm.EL
        flux = self.pm.flux
        fluxsquid = self.pm.fluxsquid
        dim = self.pm.cutoff

        phi0 = (8.0 * EC / EL)**(0.25)        # LC oscillator length
        om = math.sqrt(8.0 * EL * EC)         # plasma osc. frequency
        d = (EJ1 - EJ2) / (EJ1 + EJ2)
        chi = math.atan(d * math.tan(-1 * np.pi * fluxsquid))        # just a term in the phase argument
        pre = math.cos(np.pi * fluxsquid) * math.sqrt(1.0 + (d * math.tan(np.pi * fluxsquid))**(2))  # just a prefactor in the transformed EJcos term

        diag = [i * om for i in range(dim)]
        LCmat = np.diagflat(diag)

        exp_arg = 1j * (create(dim) + annihilate(dim)) * phi0 / math.sqrt(2)
        exp_mat = 0.5 * sp.linalg.expm(exp_arg) * cmath.exp(1j * (2.0 * np.pi * flux - np.pi * fluxsquid + chi))
        cos_mat = exp_mat + np.conj(exp_mat.T)

        hmat = LCmat - (EJ1 + EJ2) * pre * cos_mat
        return hmat

    def potential(self, phi):
        p = self.pm
        return (0.5 * p.EL * (phi)**(2) - p.EJ1 * np.cos(phi + 2 * np.pi * p.flux) - p.EJ2 * np.cos(phi - 2 * np.pi * p.fluxsquid + 2 * np.pi * p.flux))

    def param_sweep_plot(self, param1, paramval_list, param2, minimum, maximum, step, evnum=6):
        """Plots evals against param1 in range paramval_list.
        Plots for values of param2 from minimum to maximum with separation step."""
        param2val = getattr(self.pm, param2)
        self.plot_evals_vs_paramvals(param1, paramval_list, evnum)
        for i in range(int((maximum-minimum)/step)):
            setattr(self.pm, param2, (minimum + step))
            minimum = minimum + step
            self.plot_evals_vs_paramvals(param1, paramval_list, evnum)
        setattr(self.pm, param2, param2val)


# ---Routines for translating 1st and 2nd derivatives by discretization into sparse matrix form-------


def derivative_1st(var_ind, min_max_pts, prefac=1, periodic=False):

    if isinstance(prefac, complex):
        dtp = np.complex_
    else:
        dtp = np.float_

    varmin = min_max_pts[:, 0]
    varmax = min_max_pts[:, 1]
    varpts = min_max_pts[:, 2]
    nr_of_variables = len(varmin)

    delta_inv = prefac * varpts[var_ind] / (2 * (varmax[var_ind] - varmin[var_ind]))
    drvtv_mat = sp.sparse.dia_matrix((varpts[var_ind], varpts[var_ind]), dtype=dtp)
    drvtv_mat.setdiag(delta_inv, k=1)
    drvtv_mat.setdiag(-delta_inv, k=-1)

    if periodic:
        drvtv_mat.setdiag(-delta_inv, k=varpts[var_ind] - 1)
        drvtv_mat.setdiag(delta_inv, k=-varpts[var_ind] + 1)

    full_mat = drvtv_mat

    # Now fill in identity matrices to the left of var_ind, with variable indices
    # smaller than var_ind. Note: range(3,0,-1) -> [3,2,1]
    for j in range(var_ind - 1, -1, -1):
        full_mat = sp.sparse.kron(sp.sparse.identity(varpts[j], format='dia'), full_mat)
    # Next, fill in identity matrices with larger variable indices to the right.
    for j in range(var_ind + 1, nr_of_variables):
        full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(varpts[j], format='dia'))

    return full_mat


def derivative_2nd(var_ind, min_max_pts, prefac=1, periodic=False):
    varmin = min_max_pts[:, 0]
    varmax = min_max_pts[:, 1]
    varpts = min_max_pts[:, 2]
    nr_of_variables = len(varmin)

    delta_inv_sqr = prefac * ((varmax[var_ind] - varmin[var_ind]) / varpts[var_ind])**(-2)

    drvtv_mat = sp.sparse.dia_matrix((varpts[var_ind], varpts[var_ind]), dtype=np.float_)
    drvtv_mat.setdiag(-2.0 * delta_inv_sqr, k=0)
    drvtv_mat.setdiag(delta_inv_sqr, k=1)
    drvtv_mat.setdiag(delta_inv_sqr, k=-1)

    if periodic:
        drvtv_mat.setdiag(delta_inv_sqr, k=varpts[var_ind] - 1)
        drvtv_mat.setdiag(delta_inv_sqr, k=-varpts[var_ind] + 1)

    full_mat = drvtv_mat
    # Now fill in identity matrices to the left of var_ind, with variable indices
    # smaller than var_ind. Note: range(3,0,-1) -> [3,2,1]
    for j in range(var_ind - 1, -1, -1):
        full_mat = sp.sparse.kron(sp.sparse.identity(varpts[j], format='dia'), full_mat)
    # Next, fill in identity matrices with larger variable indices to the right.
    for j in range(var_ind + 1, nr_of_variables):
        full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(varpts[j], format='dia'))
    return full_mat


def derivative_mixed_1sts(deriv_var_list, min_max_pts, prefac=1, periodic_list=False):
    """Generate sparse derivative matrices of the form \partial_{x_1} \partial_{x_2} ...,
    i.e., a product of first order derivatives (with respect to different variables).
    Note: var_list is expected to be ordered!
    """
    if isinstance(prefac, complex):
        dtp = np.complex_
    else:
        dtp = np.float_

    varmin = min_max_pts[:, 0]
    varmax = min_max_pts[:, 1]
    varpts = min_max_pts[:, 2]
    nr_of_variables = len(varmin)

    deriv_order = len(deriv_var_list)   # total order of derivative (number of variables wrt which first derivatives are taken)

    delta_inv = [0] * deriv_order   # this is 1/delta from f'(x) ~= [f(x+h) - f(x-h)]/2h, delta=2h
    drvtv_mat = [0] * deriv_order

    # Loop over the elements of var_list and generate the derivative matrices
    for j in range(deriv_order):
        delta_inv[j] = varpts[deriv_var_list[j]] / (2.0 * (varmax[deriv_var_list[j]] - varmin[deriv_var_list[j]]))
        if j == 0:
            delta_inv[j] *= prefac   # first variable has prefactor absorbed into 1/delta
        drvtv_mat[j] = sp.sparse.dia_matrix((varpts[deriv_var_list[j]], varpts[deriv_var_list[j]]), dtype=dtp)
        drvtv_mat[j].setdiag(delta_inv[j], k=1)      # occupy the first off-diagonal to the right
        drvtv_mat[j].setdiag(-delta_inv[j], k=-1)    # and left

        if deriv_var_list[j] in periodic_list:
            drvtv_mat[j].setdiag(-delta_inv[j], k=varpts[deriv_var_list[j]] - 1)
            drvtv_mat[j].setdiag(delta_inv[j], k=-varpts[deriv_var_list[j]] + 1)

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
        full_mat = sp.sparse.kron(sp.sparse.identity(varpts[j], format='dia'), full_mat)
    # Loop over remaining derivatives up to very last one:
    for index, deriv_index in enumerate(deriv_var_list[:-1]):
        # (c) Fill in identities to the right
        for j in range(deriv_index + 1, deriv_var_list[index + 1]):
            full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(varpts[j], format='dia'))
        # (d) Insert next derivative
        full_mat = sp.sparse.kron(full_mat, drvtv_mat[index + 1])
    # Fill in all remaining identity matrices to the right until last variable covered
    for j in range(deriv_var_list[-1] + 1, nr_of_variables):
        full_mat = sp.sparse.kron(full_mat, sp.sparse.identity(varpts[j], format='dia'))

    return full_mat


# ---Symmetric 0-pi qubit--------------------------------------------------------------------


class QubitSymZeroPi(QubitBaseClass):

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
    min_max_pts:  numpy array specifying the range and spacing of the discretization lattice
                  np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])
    """

    _expected_parameters = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'flux': 'external magnetic flux in angular units (2pi corresponds to one flux quantum)',
        'min_max_pts': """discretization range and number of points for phi and theta:
                          np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])"""
    }

    def __init__(self, **kwargs):
        super(QubitSymZeroPi, self).__init__(**kwargs)
        self.pm._qubit_type = 'symmetric 0-Pi qubit (zero offset charge)'

    def hilbertdim(self):
        return int(np.prod(np.asarray(self.pm.min_max_pts)[:, 2]))

    def potential(self, phi, theta):
        p = self.pm
        return (-2.0 * p.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * p.flux / 2.0) + p.EL * phi**2 + 2.0 * p.EJ)

    def sparse_kineticmat(self):
        p = self.pm
        kmat = derivative_2nd(0, p.min_max_pts, prefac=-2.0 * p.ECJ)    # -2E_{CJ}\partial_\phi^2
        kmat += derivative_2nd(1, p.min_max_pts, prefac=-2.0 * p.ECS, periodic=True)  # -2E_{C\Sigma}\partial_\theta^2
        return kmat

    def hamiltonian(self):
        return (self.sparse_kineticmat() + sparse_potentialmat(self.pm.min_max_pts, self.potential))

    def _evals_calc(self, evnum):
        hmat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(hmat, k=evnum, return_eigenvectors=False, which='SA')
        return evals

    def _esys_calc(self, evnum):
        hmat = self.hamiltonian()
        evals, evecs = sp.sparse.linalg.eigsh(hmat, k=evnum, return_eigenvectors=True, which='SA')
        return evals, evecs

    def i_d_dphi(self):
        """Return the operator i \partial_\phi in sparse.dia_matrix form"""
        return derivative_1st(0, self.pm.min_max_pts, prefac=1j, periodic=False)

    def i_d_dtheta(self):
        """Return the operator i \partial_\theta (periodic variable) in sparse.dia_matrix form"""
        return derivative_1st(1, self.pm.min_max_pts, prefac=1j, periodic=True)

    def d_dtheta(self):
        """Return the operator i \partial_\theta (periodic variable) in sparse.dia_matrix form"""
        return derivative_1st(1, self.pm.min_max_pts, periodic=True)

    # return the operator \phi
    def phi(self):
        varmin = self.pm.min_max_pts[:, 0]
        varmax = self.pm.min_max_pts[:, 1]
        varpts = self.pm.min_max_pts[:, 2]
        nr_of_variables = len(varmin)
        phi_matrix = sp.sparse.dia_matrix((varpts[0], varpts[0]), dtype=np.float_)
        diag = np.linspace(varmin[0], varmax[0], varpts[0])
        phi_matrix.setdiag(diag)
        for j in range(1, nr_of_variables):
            phi_matrix = sp.sparse.kron(phi_matrix, sp.sparse.identity(varpts[j], format='dia'))
        return phi_matrix

    def plot_potential(self, levls=None, aspect_ratio=None, to_file=None):
        varmin = self.pm.min_max_pts[:, 0]
        varmax = self.pm.min_max_pts[:, 1]
        varpts = self.pm.min_max_pts[:, 2]
        x_list = np.linspace(varmin[0], varmax[0], varpts[0])
        y_list = np.linspace(varmin[1], varmax[1], varpts[1])
        contourplot(x_list, y_list, self.potential, levls, aspect_ratio, to_file)
        return None

    def wavefunction(self, esys, which=0):
        evnum = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evnum)
        else:
            _, evecs = esys
        varpts = self.pm.min_max_pts[:, 2]
        return evecs[:, which].reshape(varpts[0], varpts[1]).T

    def plot_wavefunction(self, esys, which=0, mode='abs', figsize=(20, 10), aspect_ratio=3, zero_calibrate=False):
        """Different modes:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        mode_dict = {'abs2': (lambda x: np.abs(x)**2),
                     'abs': (lambda x: np.abs(x)),
                     'real': (lambda x: np.real(x)),
                     'imag': (lambda x: np.imag(x))}
        wavefunc = self.wavefunction(esys, which)
        wavefunc = mode_dict[mode](wavefunc)
        self._plot_wavefunction2d(wavefunc, self.pm.min_max_pts, figsize, aspect_ratio, zero_calibrate)
        return None

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction1d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d'")


# ----------------------------------------------------------------------------------------


class QubitDisZeroPi(QubitSymZeroPi):

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
    min_max_pts: numpy array specifying the range and spacing of the discretization lattice
                np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined
    as relative ones.


    """

    _expected_parameters = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'dEJ': 'relative deviation between the two EJs',
        'dCJ': 'relative deviation between the two junction capacitances',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'min_max_pts': """'discretization range and number of points for phi and theta:
                           np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])"""
    }

    def __init__(self, **kwargs):
        super(QubitSymZeroPi, self).__init__(**kwargs)
        self.pm._qubit_type = '0-Pi qubit with EJ and CJ disorder, no coupling to chi mode (zero offset charge)'

    def potential(self, phi, theta):
        p = self.pm
        # 
        # 07/01/16 changed '-2.0 * p.EJ * p.dEJ' term into '+2.0 * ...' (cmp. Dempster eq. 15)
        #
        return (-2.0 * p.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * p.flux / 2.0) + p.EL * phi**2 + 2.0 * p.EJ +
                2.0 * p.EJ * p.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * p.flux / 2.0))

    def sparse_kineticmat(self):
        dphi2 = derivative_2nd(0, self.pm.min_max_pts, prefac=-2.0 * self.pm.ECJ)                   # -2E_{CJ}\partial_\phi^2
        dth2 = derivative_2nd(1, self.pm.min_max_pts, prefac=-2.0 * self.pm.ECS, periodic=True)     # -2E_{C\Sigma}\partial_\theta^2
        dphidtheta = derivative_mixed_1sts([0, 1], self.pm.min_max_pts, prefac=4.0 * self.pm.ECS * self.pm.dCJ, periodic_list=[1])
        return (dphi2 + dth2 + dphidtheta)


# ----------------------------------------------------------------------------------------


class QubitSymZeroPiNg(QubitSymZeroPi):

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
    min_max_pts: numpy array specifying the range and spacing of the discretization lattice, e.g.
                 np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])
    """

    _expected_parameters = {
        'EJ': 'Josephson energy',
        'EL': 'inductive energy',
        'ECJ': 'junction charging energy',
        'ECS': 'total charging energy including C',
        'ng': 'offset charge',
        'flux': 'external magnetic flux units of flux quanta (h/2e)',
        'min_max_pts': """discretization range and number of points for phi and theta:
                          np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])"""
    }

    def __init__(self, **kwargs):
        super(QubitSymZeroPiNg, self).__init__(**kwargs)
        self.pm._qubit_type = 'symmetric 0-Pi qubit with offset charge ng'

    def sparse_kineticmat(self):
        p = self.pm
        varpts = p.min_max_pts[:, 2]

        return (derivative_2nd(0, p.min_max_pts, prefac=-2.0 * p.ECJ) +  # -2E_{CJ}\partial_\phi^2
                derivative_2nd(1, p.min_max_pts, prefac=-2.0 * p.ECS, periodic=True) +   # 2E_{C\Sigma}(i\partial_\theta + n_g)^2
                derivative_1st(1, p.min_max_pts, prefac=4.0 * 1j * p.ECS * self.pm.ng, periodic=True) +
                sp.sparse.kron(sp.sparse.identity(varpts[0], format='dia'),
                               sp.sparse.identity(varpts[1], format='dia') * 2.0 * p.ECS * (p.ng)**2))


# ----------------------------------------------------------------------------------------


class QubitFullZeroPi(QubitSymZeroPi):

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
    min_max_pts: numpy array specifying the range and spacing of the discretization lattice, e.g.
                 np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined as
    relative ones.
    """

    _expected_parameters = {
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
        'min_max_pts': """discretization range and number of points for phi, theta, and chi:
                       np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts], [chimin, chimax, chipts]])"""
    }

    def __init__(self, **kwargs):
        super(QubitFullZeroPi, self).__init__(**kwargs)
        self.pm._qubit_type = 'full 0-Pi circuit (phi, theta, chi), no offset charge'

    def sparse_kineticmat(self):
        p = self.pm
        return (derivative_2nd(0, p.min_max_pts, prefac=-2.0 * p.ECJ) +                  # -2E_{CJ}\partial_\phi^2
                derivative_2nd(1, p.min_max_pts, prefac=-2.0 * p.ECS, periodic=True) +   # -2E_{C\Sigma}\partial_\theta^2
                derivative_2nd(2, p.min_max_pts, prefac=-2.0 * p.EC) +                   # -2E_{C}\partial_\chi^2
                #
                # NOTE: 06/30/16 change 'periodic_list=[2]' to 'periodic_list=[1]''
                #
                derivative_mixed_1sts([0, 1], p.min_max_pts, prefac=4.0 * p.ECS * p.dCJ, periodic_list=[1]) +  # 4E_{C\Sigma}(\delta C_J/C_J)\partial_\phi \partial_\theta
                derivative_mixed_1sts([1, 2], p.min_max_pts, prefac=4.0 * p.ECS * p.dC, periodic_list=[1]))    # 4E_{C\Sigma}(\delta C/C)\partial_\theta \partial_\chi

    def potential(self, phi, theta, chi):
        p = self.pm
        return (-2.0 * p.EJ * np.cos(theta) * np.cos(phi - 2.0 * np.pi * p.flux / 2) + p.EL * phi**2 + 2 * p.EJ +   # symmetric 0-pi contributions
                2.0 * p.EJ * p.dEJ * np.sin(theta) * np.sin(phi - 2.0 * np.pi * p.flux / 2) + p.EL * chi**2 +       # correction terms in presence of disorder
                2.0 * p.EL * p.dEL * phi * chi + 2 * p.EJ * p.dEJ)                                                  # correction terms in presence of disorder

    def plot_potential(self, fixed_arg_key, arg_const, levls=None, aspect_ratio=None, to_file=None):
        arg_dict = {'phi': 0, 'theta': 1, 'chi': 2}
        fixed_arg = arg_dict[fixed_arg_key]

        varying_args = list(set([0, 1, 2]) - set([fixed_arg]))

        def reduced_potential(x, y):    # not very elegant, suspect there is a better way of coding this?
            args = [arg_const] * 3
            args[varying_args[0]] = x
            args[varying_args[1]] = y
            return self.potential(*args)

        varmin = self.pm.min_max_pts[:, 0]
        varmax = self.pm.min_max_pts[:, 1]
        varpts = self.pm.min_max_pts[:, 2]
        x_list = np.linspace(varmin[varying_args[0]], varmax[varying_args[0]], varpts[varying_args[0]])
        y_list = np.linspace(varmin[varying_args[1]], varmax[varying_args[1]], varpts[varying_args[1]])
        contourplot(x_list, y_list, reduced_potential, levls, aspect_ratio, to_file)
        return None

    def wavefunction(self, esys, which=0):
        evnum = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evnum)
        else:
            _, evecs = esys
        return evecs[:, which]

    def plot_wavefunction(self, esys, fixed_arg_key, arg_const, which=0, mode='abs', figsize=(20, 10), aspect_ratio=3):
        """Different modes:
        'abs2': |psi|^2
        'abs':  |psi|
        'real': Re(psi)
        'imag': Im(psi)
        """
        mode_dict = {'abs2': (lambda x: np.abs(x)**2),
                     'abs': (lambda x: np.abs(x)),
                     'real': (lambda x: np.real(x)),
                     'imag': (lambda x: np.imag(x))}
        varmin = self.pm.min_max_pts[:, 0]
        varmax = self.pm.min_max_pts[:, 1]
        varpts = self.pm.min_max_pts[:, 2]
        wavefunc = self.wavefunction(esys, which)
        wavefunc = mode_dict[mode](wavefunc)
        wavefunc = wavefunc.reshape(varpts[0], varpts[1], varpts[2])

        arg_dict = {'phi': 0, 'theta': 1, 'chi': 2}
        fixed_arg = arg_dict[fixed_arg_key]

        argslice = [slice(None), slice(None), slice(None)]

        arg_const_index = int(varpts[fixed_arg] * (arg_const - varmin[fixed_arg]) /
                              (varmax[fixed_arg] - varmin[fixed_arg]))
        argslice[fixed_arg] = arg_const_index
        wavefunc = wavefunc[tuple(argslice)].T
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


class QubitFullZeroPi_ProductBasis(QubitBaseClass):

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
    l_cut: cutoff in the number of states of the disordered zero-pi qubit
    min_max_pts:  array specifying the range and spacing of the discretization lattice for phi and theta
                [[phimin, phimax, phipts], [thetamin, thetamax, thetapts]]
    a_cut: cutoff in the chi oscillator basis (Fock state basis)

    Caveat: different from Eq. (15) in the reference above, all disorder quantities are defined as
    relative ones.
    """

    _expected_parameters = {
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
        'l_cut': 'cutoff in the number of states of the disordered zero-pi qubit',
        'min_max_pts': """numpy array specifying the range and spacing of the discretization lattice for phi and theta
                [[phimin, phimax, phipts], [thetamin, thetamax, thetapts]]""",
        'a_cut': 'cutoff in the chi oscillator basis (Fock state basis)'
    }

    def __init__(self, **kwargs):
        super(QubitFullZeroPi_ProductBasis, self).__init__(**kwargs)
        self.pm._qubit_type = 'full 0-Pi circuit (phi, theta, chi) in 0pi - chi product basis'
        self._zeropi = QubitDisZeroPi(force_valid=True)
        self._zeropi.pm = self.pm

    def hamiltonian(self):
        lcut = self.pm.l_cut
        zpi_evals, zpi_evecs = self._zeropi.eigensys(evnum=lcut)
        zpi_diag_hamiltonian = sp.sparse.dia_matrix((lcut, lcut), dtype=np.float_)
        zpi_diag_hamiltonian.setdiag(zpi_evals)

        acut = self.pm.a_cut
        #
        # 07/01/16 fixing prefactor for chi degree of freedom: 'ECS' -> 'EC'
        prefactor = (8.0 * self.pm.EL * self.pm.EC)**0.5
        #
        #
        chi_diag_hamiltonian = sparse_number(acut, prefactor)

        hamiltonian = sp.sparse.kron(zpi_diag_hamiltonian, sp.sparse.identity(acut, format='dia', dtype=np.float_))
        hamiltonian += sp.sparse.kron(sp.sparse.identity(lcut, format='dia', dtype=np.float_), chi_diag_hamiltonian)

        gmat = self.g_coupling_matrix(zpi_evecs)
        zpi_coupling = sp.sparse.dia_matrix((lcut, lcut), dtype=np.float_)
        for l1 in range(lcut):
            for l2 in range(lcut):
                zpi_coupling += gmat[l1, l2] * sparse_hubbardmat(lcut, l1, l2)
        hamiltonian += sp.sparse.kron(zpi_coupling, sparse_annihilate(acut) + sparse_create(acut))
        return hamiltonian

    def hilbertdim(self):
        return (self.pm.l_cut * self.pm.a_cut)

    def _evals_calc(self, evnum):
        hmat = self.hamiltonian()
        evals = sp.sparse.linalg.eigsh(hmat, k=evnum, return_eigenvectors=False, which='SA')
        return evals

    def _esys_calc(self, evnum):
        hmat = self.hamiltonian()
        evals, evecs = sp.sparse.linalg.eigsh(hmat, k=evnum, return_eigenvectors=True, which='SA')
        return evals, evecs

    def g_phi_coupling_matrix(self, zeropi_states, transpose=True):
        """Returns a matrix of coupling strengths g^\phi_{ll'} [cmp. Dempster et al., Eq. (18)], using the states
        from the list 'zeropi_states'. Most commonly, 'zeropi_states' will contain eigenvectors of the
        QubitDisZeroPi type, so 'transpose' is enabled by default.
        """
        p = self.pm
        prefactor = p.EL * p.dEL * (8.0 * p.EC / p.EL)**0.25
        return (prefactor * matrixelem_table(self._zeropi.phi(), zeropi_states, transpose, real_valued=True))

    def g_theta_coupling_matrix(self, zeropi_states, transpose=True):
        """Returns a matrix of coupling strengths i*g^\theta_{ll'} [cmp. Dempster et al., Eq. (17)], using the states
        from the list 'zeropi_states'. Most commonly, 'zeropi_states' will contain eigenvectors, so 'transpose' is enabled by
        default.
        """
        p = self.pm
        prefactor = - p.ECS * p.dC * (32.0 * p.EL / p.EC)**0.25
        return (prefactor * matrixelem_table(self._zeropi.d_dtheta(), zeropi_states, transpose, real_valued=True))

    def g_coupling_matrix(self, zeropi_states, transpose=True, evnum=6):
        """Returns a matrix of coupling strengths g_{ll'} [cmp. Dempster et al., text above Eq. (17)], using the states
        from 'state_list'.  Most commonly, 'zeropi_states' will contain eigenvectors of the
        QubitDisZeroPi type, so 'transpose' is enabled by default.
        If zeropi_states==None, then a set of self.zeropi eigenstates is calculated. Only in that case is evnum
        used for the eigenstate number (and hence the coupling matrix size).

        """
        if zeropi_states is None:
            _, zeropi_states = self._zeropi.eigensys(evnum=evnum)
        return (self.g_phi_coupling_matrix(zeropi_states, transpose) + self.g_theta_coupling_matrix(zeropi_states, transpose))

    def _plot_wavefunction1d_discrete(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d_discrete'")

    def _plot_wavefunction1d(self):
        raise AttributeError("Qubit object has no attribute '_plot_wavefunction1d'")


# ----------------------------------------------------------------------------------------


class QubitModZeroPi(QubitSymZeroPi):

    """[still experimental] modified version of the symmetric 0-pi qubit based on replacing inductors
    by Josephson junctions"""

    _expected_parameters = {
        'EJ': 'Josephson energy',
        'EJp': 'Josephson energy of the junction substituting the inductor',
        'ECphi': 'charging energy for phi term',
        'ECth': 'charging energy for theta term',
        'ECchi': 'charging energy for chi term',
        'flux': 'external magnetic flux in units of flux quanta (h/2e)',
        'min_max_pts': """discretization range and number of points for phi and theta:
                       np.asarray([[phimin, phimax, phipts], [thetamin, thetamax, thetapts]])"""
    }

    def __init__(self, **kwargs):
        super(QubitModZeroPi, self).__init__(**kwargs)
        self.pm._qubit_type = 'modified symmetric 0-Pi qubit (EL->EJp)'
        print("Implementation of this 0-pi device variation is still experimental & not complete")

    def potential(self, phi, theta, chi):
        p = self.pm
        return -2.0 * np.cos(phi) * (p.EJ * np.cos(theta + 2.0 * np.pi * p.flux / 2) + p.EJp * np.cos(chi)) + 2 * (p.EJ + p.EJp)

    def sparse_kineticmat(self):
        p = self.pm
        dphi2 = derivative_2nd(0, p.min_max_pts, prefac=-2.0 * p.ECphi, periodic=True)   # CJ + CJp
        dth2 = derivative_2nd(1, p.min_max_pts, prefac=-2.0 * p.ECth, periodic=True)     # C + CJ
        dchi2 = derivative_2nd(2, p.min_max_pts, prefac=-2.0 * p.ECchi, periodic=True)     # C + CJp
        return (dphi2 + dth2 + dchi2)
