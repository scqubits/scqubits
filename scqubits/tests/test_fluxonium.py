# test_fluxonium.py
# meant to be run with 'pytest'
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import numpy as np
import pytest

from scqubits.utils.tst_baseclass import BaseTest
from scqubits.utils.spectrum_utils import standardize_phases
from scqubits import Fluxonium


class TestFluxQubit(BaseTest):

    @pytest.fixture(autouse=True)
    def init_qbt(self):
        self.qbt = Fluxonium(
            EJ=8.9,
            EC=2.5,
            EL=0.5,
            flux=0.33,
            cutoff=110
        )

    _EVALS_REFERENCE = np.asarray([-3.30851586, -0.23733983, 6.9133453, 10.55323546, 11.76215604, 16.12300682])
    _EVECS_REFERENCE = np.asarray([-2.25641769e-01, -9.07823403e-02, -2.01681071e-01,  8.47894074e-01,
        2.65759934e-01,  5.00881806e-02, -2.68714199e-01,  4.60768041e-02,
       -8.01048670e-02,  5.08834430e-02,  1.23273137e-01, -3.84743436e-02,
       -7.10132669e-02, -6.34881799e-03,  3.89404389e-02,  2.21908834e-02,
       -2.32943127e-02, -1.50331760e-02,  1.04908421e-02,  4.20852948e-03,
       -4.55237690e-04,  1.92102825e-03, -4.44947862e-03, -3.66377532e-03,
        4.91765067e-03,  3.46252679e-03, -3.32174179e-03, -2.86608297e-03,
        1.57273296e-03,  2.32311443e-03, -4.57344914e-04, -1.81060177e-03,
       -1.92094509e-05,  1.27855869e-03,  1.22207703e-04, -7.63148225e-04,
       -8.66727352e-05,  3.36643601e-04,  3.49469014e-05, -4.53863382e-05,
       -2.96209234e-06, -1.10354785e-04, -1.23156644e-05,  1.63032693e-04,
        2.11847164e-05, -1.54922008e-04, -2.90934726e-05,  1.21443287e-04,
        3.61491733e-05, -8.47714448e-05, -4.03459696e-05,  5.49479311e-05,
        4.02414983e-05, -3.41213213e-05, -3.59317688e-05,  2.07520577e-05,
        2.87510686e-05, -1.24165902e-05, -2.04914071e-05,  7.12854606e-06,
        1.27180280e-05, -3.65435612e-06, -6.41481035e-06,  1.35501525e-06,
        1.94474759e-06,  8.10704481e-08,  7.98514303e-07, -8.41829022e-07,
       -2.17845583e-06,  1.09320949e-06,  2.63077004e-06, -1.00383472e-06,
       -2.54362617e-06,  7.32203731e-07,  2.20400642e-06, -4.06782497e-07,
       -1.79221219e-06,  1.14265623e-07,  1.40243284e-06,  1.00634040e-07,
       -1.07141350e-06, -2.26911652e-07,  8.04247022e-07,  2.75386617e-07,
       -5.92693293e-07, -2.67276881e-07,  4.25725119e-07,  2.25542835e-07,
       -2.94103590e-07, -1.69654213e-07,  1.91327922e-07,  1.13370856e-07,
       -1.12976164e-07, -6.46737165e-08,  5.56971691e-08,  2.68724975e-08,
       -1.64272102e-08, -1.00356998e-10, -7.94517430e-09, -1.72584903e-08,
        2.05681840e-08,  2.74806641e-08, -2.44690005e-08, -3.28106088e-08,
        2.23433236e-08,  3.50935130e-08, -1.64402276e-08, -3.56602334e-08,
        8.53480822e-09,  3.53378058e-08])
    _EVALS_REFERENCE2 = None
    _MATELEM_REFERENCE =  None


    def test_eigenvals(self):
        return self.eigenvals(self._EVALS_REFERENCE)

    def test_eigenvecs(self):
        return self.eigenvecs(self._EVECS_REFERENCE)

    def test_plot_evals_vs_paramvals_ng(self):
        flux_list = np.linspace(-0.5, 0.5, 90)
        return self.plot_evals_vs_paramvals('flux', flux_list)

#    def test_get_spectrum_vs_paramvals(self):
#        flux_list = np.linspace(.46, .54, 40)
#        return self.get_spectrum_vs_paramvals('flux', flux_list, self._EVALS_REFERENCE2)

    def test_plot_evals_vs_paramvals_EJ(self):
        ej_vals = self.qbt.EJ * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 40))
        self.plot_evals_vs_paramvals('EJ', ej_vals)

    def test_qubit_plot_wavefunction(self):
        self.qbt.plot_wavefunction(esys=None, which=(0,1,5), mode='real')

#    def test_matrixelement_table(self):
#        calculated_matrix = self.qbt.matrixelement_table('n_operator', esys=None, evals_count=16)
#        assert np.allclose(np.abs(self._MATELEM_REFERENCE), np.abs(calculated_matrix))

    def test_plot_matrixelements(self):
        self.plot_matrixelements('n_operator', evals_count=10)

    def test_print_matrixelements(self):
        self.print_matrixelements('n_operator')

    def test_plot_matelem_vs_paramvals(self):
        flux_list = np.linspace(.49, .51, 40)
        self.plot_matelem_vs_paramvals('phi_operator', 'flux', flux_list, select_elems=[(0, 0), (1, 4), (1, 0)])
