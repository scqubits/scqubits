"""
scqubits: superconducting qubits in Python
===========================================

[J. Koch](https://github.com/jkochNU), [P. Groszkowski](https://github.com/petergthatsme)

scqubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within scqubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.
"""
# settings.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
#######################################################################################################################


import numpy as np
import pytest

import scqubits.utils.plotting as plot



class BaseTest():
    qbt = None

    @pytest.fixture(autouse=True)
    def set_tmpdir(self, request):
        setattr(self, 'tmpdir', request.getfixturevalue('tmpdir'))

    def eigenvals(self, evals_reference):
        assert np.allclose(evals_reference, self.qbt.eigenvals(filename=self.tmpdir.dirpath() + 'test'))

    def eigenvecs(self, evecs_reference):    # compare 3rd eigenvector
        _, evecs_tst = self.qbt.eigensys(filename=self.tmpdir.dirpath() + 'test')
        evecs_calculated = evecs_tst.T[3]
        print(evecs_calculated)
        assert np.allclose(evecs_reference, evecs_calculated)

    def plot_evals_vs_paramvals(self, param_name, param_list):
        self.qbt.plot_evals_vs_paramvals(param_name, param_list, evals_count=5, subtract_ground=True)

    def get_spectrum_vs_paramvals(self, param_name, param_list, reference_evals):
        calculated_evals = self.qbt.get_spectrum_vs_paramvals(param_name, param_list, evals_count=4,
                                                              subtract_ground=False, get_eigenstates=True,
                                                              filename=self.tmpdir.dirpath() + 'test')
        assert np.allclose(reference_evals, calculated_evals.energy_table)
        # TODO change to checking both evals and evecs

    def matrixelement_table(self, op, matelem_reference, evals_count=10):
        calculated_matrix = self.qbt.matrixelement_table(op, esys=None, evals_count=evals_count)
        assert np.allclose(matelem_reference, calculated_matrix)

    def plot_matrixelements(self, op, evals_count=10):
        self.qbt.plot_matrixelements(op, esys=None, evals_count=evals_count)

    def print_matrixelements(self, op):
        mat_data = self.qbt.matrixelement_table(op)
        plot.print_matrix(abs(mat_data))

    def plot_matelem_vs_paramvals(self, op, param_name, param_list, select_elems):
        fig, ax = self.qbt.plot_matelem_vs_paramvals(op, param_name, param_list, select_elems=select_elems,
                                                     filename=self.tmpdir + 'test')
