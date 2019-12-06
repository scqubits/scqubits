# test_fluxonium.py
# meant to be run with 'nose'
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################

from __future__ import division
from __future__ import print_function

import numpy as np

import sc_qubits as qubit


def fluxonium_initialize():
    fluxonium = qubit.Fluxonium(
        EJ=8.9,
        EC=2.5,
        EL=0.5,
        flux=0.33,
        cutoff=110
    )
    return fluxonium


def test_Fluxonium_init():
    fluxonium = fluxonium_initialize()


def test_Fluxonium_eigenvals():
    print("Fluxonium_eigenvals()")
    fluxonium = fluxonium_initialize()
    evals_reference = np.asarray([-3.30851586, -0.23733983, 6.9133453, 10.55323546, 11.76215604, 16.12300682])
    assert np.allclose(evals_reference, fluxonium.eigenvals())


def test_Fluxonium_eigenvecs():
    print("Fluxonium_eigenvecs()")
    evecs_reference = np.asarray([-2.38363330e-01 + 0.00000000e+00j,
                                  3.20442405e-02 + 2.12358326e-17j,
                                  -1.20810144e-01 - 6.94724735e-18j,
                                  2.43511103e-01 - 2.22081236e-17j,
                                  -5.33736599e-01 - 8.16472432e-17j,
                                  -6.41746715e-01 - 5.57995514e-17j,
                                  3.73801267e-01 - 8.48983863e-17j,
                                  1.95507133e-02 + 3.54189937e-17j,
                                  3.64412246e-02 + 6.94614516e-17j,
                                  -9.37239709e-02 - 9.86168820e-19j,
                                  -2.49439240e-02 - 1.48055473e-17j,
                                  -2.40320700e-02 - 7.24145879e-18j,
                                  -4.40005619e-02 - 1.39442892e-17j,
                                  9.15447995e-02 - 1.45377466e-17j,
                                  3.36070686e-02 + 6.02277858e-17j,
                                  -7.90008863e-02 + 1.38422304e-18j,
                                  -9.82499289e-03 - 7.21745413e-17j,
                                  4.29790987e-02 - 4.37174562e-18j,
                                  5.02340919e-03 + 2.77125495e-17j,
                                  -1.69072583e-02 + 8.22829425e-18j,
                                  -7.47855900e-03 - 8.30489777e-18j,
                                  5.14627755e-03 - 7.07257661e-18j,
                                  6.31589761e-03 + 3.53006276e-18j,
                                  -1.63773843e-03 + 3.65858940e-18j,
                                  -1.78893677e-03 - 8.22761654e-19j,
                                  7.86514528e-04 - 5.45650780e-19j,
                                  -2.34641271e-03 + 8.49987266e-19j,
                                  -1.64190378e-04 - 1.53769933e-18j,
                                  4.20520055e-03 + 4.98341336e-19j,
                                  -4.94155735e-04 + 3.09908437e-18j,
                                  -4.09698008e-03 - 9.89730901e-19j,
                                  8.53478022e-04 - 1.32909833e-18j,
                                  3.08864466e-03 + 8.33956971e-19j,
                                  -7.93617226e-04 + 1.34060537e-18j,
                                  -2.00265671e-03 - 7.53971558e-19j,
                                  4.80192585e-04 - 1.12248119e-18j,
                                  1.18792898e-03 + 3.65923429e-19j,
                                  -1.50392473e-04 + 7.96299586e-19j,
                                  -6.75558017e-04 - 7.62463010e-20j,
                                  -5.18954042e-05 - 5.43957378e-19j,
                                  3.77450313e-04 - 4.35225590e-20j,
                                  1.06410891e-04 + 2.93405913e-19j,
                                  -2.01906770e-04 + 8.45306996e-20j,
                                  -6.59440591e-05 - 1.84663625e-19j,
                                  9.07422076e-05 - 6.09742126e-20j,
                                  -4.13271229e-06 + 7.03154551e-20j,
                                  -1.64670974e-05 - 4.20795028e-20j,
                                  6.02841937e-05 + 1.76061207e-20j,
                                  -3.13081503e-05 + 7.72755882e-20j,
                                  -8.71127835e-05 - 5.03474206e-20j,
                                  5.70144435e-05 - 1.15019871e-19j,
                                  8.83151315e-05 + 7.70399860e-20j,
                                  -6.47863704e-05 + 8.45305434e-20j,
                                  -7.48488378e-05 - 7.68623425e-20j,
                                  5.98829870e-05 - 7.10660252e-20j,
                                  5.68587268e-05 + 6.51151924e-20j,
                                  -4.79891669e-05 + 2.00533230e-20j,
                                  -4.04922202e-05 - 4.11767962e-20j,
                                  3.40414937e-05 - 2.37191733e-20j,
                                  2.80361101e-05 + 3.04652294e-20j,
                                  -2.14432561e-05 + 2.78940465e-20j,
                                  -1.93840260e-05 - 1.82660275e-20j,
                                  1.18972594e-05 - 1.00573125e-20j,
                                  1.34973104e-05 + 7.40579229e-21j,
                                  -5.71829598e-06 + 6.49363847e-21j,
                                  -9.32153137e-06 - 4.03633810e-21j,
                                  2.36264207e-06 - 6.87490019e-21j,
                                  6.14936276e-06 + 1.41174684e-21j,
                                  -9.44047430e-07 + 5.84360395e-21j,
                                  -3.63247615e-06 - 9.93721596e-22j,
                                  6.04011871e-07 - 3.54594177e-21j,
                                  1.65179056e-06 + 5.89537826e-22j,
                                  -7.02741231e-07 + 1.63137176e-21j,
                                  -1.80385205e-07 - 6.73051333e-22j,
                                  8.62753885e-07 - 2.01870580e-23j,
                                  -8.05110828e-07 + 8.97145728e-22j,
                                  -9.23576589e-07 - 3.70125333e-22j,
                                  1.35986744e-06 - 6.77117007e-22j,
                                  8.63075586e-07 + 8.77830722e-22j,
                                  -1.57014495e-06 + 6.59852252e-22j,
                                  -7.23347908e-07 - 5.75212922e-22j,
                                  1.53658824e-06 - 3.12572026e-22j,
                                  5.59071858e-07 + 1.43932846e-21j,
                                  -1.35587735e-06 + 2.87122900e-22j,
                                  -4.10917984e-07 - 9.77323938e-22j,
                                  1.10787893e-06 - 1.51794202e-22j,
                                  2.98238383e-07 + 7.14243936e-22j,
                                  -8.49996518e-07 + 1.80906527e-23j,
                                  -2.22773464e-07 - 5.75460842e-22j,
                                  6.17423687e-07 - 4.41791722e-23j,
                                  1.76277085e-07 + 4.93180984e-22j,
                                  -4.26981769e-07 + 2.40061779e-23j,
                                  -1.47580079e-07 - 3.19944381e-22j,
                                  2.82390553e-07 - 3.91496868e-23j,
                                  1.27157950e-07 + 2.07451443e-22j,
                                  -1.79446817e-07 + 7.32976594e-23j,
                                  -1.09016440e-07 - 1.52115266e-22j,
                                  1.10230599e-07 - 4.96316487e-23j,
                                  9.06247788e-08 + 1.10145617e-22j,
                                  -6.60378531e-08 + 5.72959714e-23j,
                                  -7.18547754e-08 - 6.61766418e-23j,
                                  3.91076558e-08 - 4.81668258e-23j,
                                  5.36897680e-08 + 4.54130059e-23j,
                                  -2.33709155e-08 + 1.87927971e-23j,
                                  -3.71927308e-08 - 2.66303441e-23j,
                                  1.45530792e-08 + 5.58129156e-24j,
                                  2.29147463e-08 + 1.65631458e-23j,
                                  -9.91703073e-09 - 9.76901988e-24j,
                                  -1.07328309e-08 - 9.89062670e-24j, 7.88622810e-09 + 2.78778104e-24j])
    fluxonium = fluxonium_initialize()
    _, evecs = fluxonium.eigensys(evals_count=6)
    evecs_calculated = evecs.T[5]
    assert np.allclose(evecs_reference, evecs_calculated)


def test_Fluxonium_plot_evals_vs_paramvals():
    print("Fluxonium_plot_evals_vs_paramvals()")
    fluxonium = fluxonium_initialize()
    fluxonium.cutoff = 50
    flux_list = np.linspace(-0.5, 0.5, 99)
    fluxonium.plot_evals_vs_paramvals('flux', flux_list, evals_count=5, subtract_ground=True)


def test_Fluxonium_plot_wavefunction():
    print("Fluxonium_plot_wavefunction()")
    fluxonium = fluxonium_initialize()
    fluxonium.plot_wavefunction(esys=None, which=(0,1,5), mode='real')


def test_Fluxonium_plot_matrixelements():
    print("Fluxonium_plot_matrixelements()")
    fluxonium = fluxonium_initialize()
    fluxonium.plot_matrixelements('phi_operator', esys=None, evals_count=10)

