# test_zeropi.py
# meant to be run with 'nose'
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

import scqubits as qubit


def ZeroPi_initialize():
    # parameters for the 0-\pi circuit

    phi_grid = qubit.Grid1d(-6 * np.pi, 6 * np.pi, 200)

    EJ_CONST = 1 / 3.95  # note that EJ and ECJ are interrelated

    zero_pi = qubit.ZeroPi(
        grid=phi_grid,
        EJ=EJ_CONST,
        EL=10.0 ** (-2),
        ECJ=1 / (8.0 * EJ_CONST),
        EC=None,
        ECS=10.0 ** (-3),
        ng=0.1,
        flux=0.23,
        ncut=30
    )
    return zero_pi


def test_ZeroPi_init():
    print("ZeroPi_init()")
    zero_pi = ZeroPi_initialize()


def test_ZeroPi_plot_potential():
    print("ZeroPi_plot_potential()")
    zero_pi = ZeroPi_initialize()
    zero_pi.plot_potential(contour_vals=np.linspace(0, 3, 25), aspect_ratio=0.12)


def test_ZeroPi_eigenvals():
    print("ZeroPi_eigenvals()")
    evals_reference = np.asarray([0.46013402, 0.49073909, 0.5024677 , 0.51986416, 0.53153435, 0.54722838])
    zero_pi = ZeroPi_initialize()
    evals_calculated = zero_pi.eigenvals()
    assert np.allclose(evals_reference, evals_calculated)


def test_ZeroPi_eigenvecs():
    evecs_reference = np.array([ 2.10824880e-03,  1.13272159e-03,  5.25246878e-04,  2.13125620e-04,
        7.64894545e-05,  2.44938000e-05,  7.05000086e-06,  1.83544343e-06,
        4.34618253e-07,  9.40615103e-08,  1.86875810e-08,  3.42173605e-09,
        5.79498393e-10,  9.10743460e-11,  1.33228479e-11,  1.81930361e-12,
        2.32528017e-13,  2.79041486e-14,  3.16067576e-15,  3.76320445e-16,
       -5.12376442e-17,  1.26170880e-17,  2.08003005e-17,  3.13911334e-17,
        5.61833943e-17,  5.59843019e-17,  1.04217327e-16,  4.75945541e-18,
       -5.82473355e-17, -3.66517857e-17,  2.99580889e-17, -1.06683919e-16,
       -6.10250983e-16, -5.85494834e-15, -5.12846424e-14, -4.20187837e-13,
       -3.23452742e-12, -2.32879619e-11, -1.56443325e-10, -9.77737386e-10,
       -5.66754925e-09, -3.03688158e-08, -1.49876424e-07, -6.78525757e-07,
       -2.80536329e-06, -1.05398589e-05, -3.57807079e-05, -1.09047912e-04,
       -2.96107308e-04, -7.09919048e-04, -1.48600539e-03, -2.67634620e-03,
       -4.06294449e-03, -5.03054452e-03, -4.75566175e-03, -2.79832848e-03,
        3.30961518e-04,  3.31531714e-03,  4.94253970e-03,  4.91095652e-03,
        3.79392643e-03,  2.41011703e-03,  1.29673407e-03,  6.02281974e-04,
        2.44833871e-04,  8.80483534e-05,  2.82577633e-05,  8.15276538e-06,
        2.12793185e-06,  5.05228068e-07,  1.09650769e-07,  2.18487480e-08,
        4.01276476e-09,  6.81742065e-10,  1.07492655e-10,  1.57773853e-11,
        2.16187964e-12,  2.77300980e-13,  3.33879523e-14,  3.78104062e-15,
        4.58159704e-16, -4.88679244e-17,  2.02922276e-17,  1.34293866e-17,
        2.48121800e-17,  5.89140064e-17,  6.56339107e-17,  1.05864759e-16,
       -1.52198367e-18, -7.37416810e-17, -3.80849312e-17,  3.54508998e-17,
       -1.04940758e-16, -7.11048438e-16, -6.82938312e-15, -5.94881201e-14,
       -4.86166205e-13, -3.73341070e-12, -2.68186397e-11, -1.79767217e-10])
    zero_pi = ZeroPi_initialize()
    _, evecs = zero_pi.eigensys(evals_count=2)
    evecs_calculated = evecs.T[1][4000:4100]
    assert np.allclose(evecs_calculated / evecs_calculated[0], evecs_reference / evecs_reference[0])


def test_ZeroPi_plot_evals_vs_paramvals():
    zero_pi = ZeroPi_initialize()
    ng_list = np.linspace(0, 1, 12)
    zero_pi.plot_evals_vs_paramvals('ng', ng_list, subtract_ground=True)


def test_ZeroPi_plot_wavefunction():
    zero_pi = ZeroPi_initialize()
    zero_pi.plot_wavefunction(esys=None, which=4, mode='real')
