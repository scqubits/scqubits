import os

import numpy as np
import scipy as sp
import itertools
from scipy.optimize import minimize
from scipy.special import eval_hermite, gamma

import scqubits.core.constants as constants
import scqubits.utils.plotting as plot
import scqubits.core.discretization as discretization
from scqubits.core import descriptors
from scqubits.core.vchos import VCHOS
import scqubits.core.storage as storage
from scqubits.utils.spectrum_utils import standardize_phases


# Flux Qubit using VCHOS

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
    return (2.0 ** n * gamma(n + 1.0)) ** (-0.5) * np.pi ** (-0.25) * eval_hermite(n, x) * np.exp(-x ** 2 / 2.)


class FluxQubitVCHOS(VCHOS):
    ECJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ECg = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EJlist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    alpha = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    nglist = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, ECJ, ECg, EJlist, alpha, nglist, flux, kmax, num_exc, truncated_dim=None):
        VCHOS.__init__(self, EJlist, nglist, flux, kmax, num_exc)
        self.ECJ = ECJ
        self.ECg = ECg
        self.alpha = alpha
        # final term in potential is cos[(+1)\phi_1+(-1)\phi_2-2pi f]
        self.boundary_coeffs = np.array([+1, -1])
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.complex_
        self._default_grid = discretization.Grid1d(-6.5 * np.pi, 6.5 * np.pi, 651)  # for plotting in phi_j basis
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qubit_pngs/fluxqubitvchos.png')

    @staticmethod
    def default_params():
        return {
            'ECJ': 1.0 / 10.0,
            'ECg': 5.0,
            'EJlist': np.array([1.0, 1.0, 0.8]),
            'alpha': 0.8,
            'nglist': np.array(2 * [0.0]),
            'flux': 0.46,
            'kmax': 1,
            'num_exc': 4,
            'truncated_dim': 6
        }

    @staticmethod
    def nonfit_params():
        return ['alpha', 'nglist', 'kmax', 'num_exc', 'squeezing', 'truncated_dim']

    def build_capacitance_matrix(self):
        """Return the capacitance matrix"""
        Cmat = np.zeros((self.number_degrees_freedom(), self.number_degrees_freedom()))

        CJ = self.e ** 2 / (2. * self.ECJ)
        Cg = self.e ** 2 / (2. * self.ECg)

        Cmat[0, 0] = CJ + self.alpha * CJ + Cg
        Cmat[1, 1] = CJ + self.alpha * CJ + Cg
        Cmat[0, 1] = -self.alpha * CJ
        Cmat[1, 0] = -self.alpha * CJ

        return Cmat

    def build_EC_matrix(self):
        """Return the charging energy matrix"""
        Cmat = self.build_capacitance_matrix()
        return 0.5 * self.e ** 2 * sp.linalg.inv(Cmat)

    def number_degrees_freedom(self):
        return 2

    def _ramp(self, k, minima_holder):
        """
        Helper function for find_minima, performing the ramp that
        is described in Sec. III E of [0]
        
        [0] PRB ...
        """
        guess = np.array([1.15 * 2.0 * np.pi * k / 3.0, 2.0 * np.pi * k / 3.0])
        result = minimize(self.potential, guess)
        new_minima = self._check_if_new_minima(result.x, minima_holder)
        if new_minima:
            minima_holder.append(np.array([np.mod(elem, 2 * np.pi) for elem in result.x]))
        return minima_holder, new_minima

    def find_minima(self):
        """
        Index all minima in the variable space of phi1 and phi2
        """
        minima_holder = []
        if self.flux == 0.5:
            guess = np.array([0.15, 0.1])
        else:
            guess = np.array([0.0, 0.0])
        result = minimize(self.potential, guess)
        minima_holder.append(np.array([np.mod(elem, 2 * np.pi) for elem in result.x]))
        for k in range(1, 4):
            (minima_holder, new_minima_positive) = self._ramp(k, minima_holder)
            (minima_holder, new_minima_negative) = self._ramp(-k, minima_holder)
            if not (new_minima_positive and new_minima_negative):
                break
        return minima_holder

    def sorted_minima(self):
        """Sort the minima based on the value of the potential at the minima """
        minima_holder = self.find_minima()
        value_of_potential = np.array([self.potential(minima_holder[x])
                                       for x in range(len(minima_holder))])
        sorted_minima_holder = np.array([x for _, x in
                                         sorted(zip(value_of_potential, minima_holder))])
        return sorted_minima_holder

    def wavefunction(self, esys=None, which=0, phi_grid=None):
        """
        Return a flux qubit wave function in phi1, phi2 basis. Note that this implementation
        does not include the effects of squeezing.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors
        which: int, optional
            index of desired wave function (default value = 0)
        phi_grid: Grid1D object, optional
            used for setting a custom plot range for phi

        Returns
        -------
        WaveFunctionOnGrid object
        """
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys
        phi_grid = phi_grid or self._default_grid
        phi_vec = phi_grid.make_linspace()

        minima_list = self.sorted_minima()
        num_minima = len(minima_list)
        total_num_states = int(self.hilbertdim() / num_minima)

        Xi = self.Xi_matrix()
        Xi_inv = sp.linalg.inv(Xi)
        norm = np.sqrt(np.abs(np.linalg.det(Xi))) ** (-1)

        wavefunc_amplitudes = np.zeros_like(np.outer(phi_vec, phi_vec))

        for i, minimum in enumerate(minima_list):
            klist = itertools.product(np.arange(-self.kmax, self.kmax + 1), repeat=2)
            jkvals = next(klist, -1)
            while jkvals != -1:
                #TODO offset charge not taken into account here. Must fix
                phik = 2.0 * np.pi * np.array([jkvals[0], jkvals[1]])
                phi1_s1_arg = Xi_inv[0, 0] * (phik - minimum)[0]
                phi2_s1_arg = Xi_inv[0, 1] * (phik - minimum)[1]
                phi1_s2_arg = Xi_inv[1, 0] * (phik - minimum)[0]
                phi2_s2_arg = Xi_inv[1, 1] * (phik - minimum)[1]
                state_amplitudes = np.real(np.reshape(evecs[i * total_num_states: (i + 1) * total_num_states, which],
                                                      (self.num_exc + 1, self.num_exc + 1)))
                wavefunc_amplitudes += np.sum([state_amplitudes[s1, s2] * norm
                                               * np.multiply(
                    harm_osc_wavefunction(s1, np.add.outer(Xi_inv[0, 0] * phi_vec + phi1_s1_arg,
                                                           Xi_inv[0, 1] * phi_vec + phi2_s1_arg)),
                    harm_osc_wavefunction(s2, np.add.outer(Xi_inv[1, 0] * phi_vec + phi1_s2_arg,
                                                           Xi_inv[1, 1] * phi_vec + phi2_s2_arg)))
                                               for s2 in range(self.num_exc + 1)
                                               for s1 in range(self.num_exc + 1)], axis=0).T  # FIX .T NOT CORRECT
                jkvals = next(klist, -1)

        grid2d = discretization.GridSpec(np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                                                     [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))

        wavefunc_amplitudes = standardize_phases(wavefunc_amplitudes)

        return storage.WaveFunctionOnGrid(grid2d, wavefunc_amplitudes)

    def plot_wavefunction(self, esys=None, which=0, phi_grid=None, mode='abs', zero_calibrate=True, **kwargs):
        """Plots 2d phase-basis wave function.

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

        Returns
        -------
        Figure, Axes
        """
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, which=which)
        wavefunc.amplitudes = amplitude_modifier(wavefunc.amplitudes)
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (5, 5)
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)
