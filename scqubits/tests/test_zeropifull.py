# test_zeropifull.py
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


def FullZeroPi_initialize():
    phi_grid = qubit.Grid1d(-8 * np.pi, 8 * np.pi, 360)

    # parameters for the modified 0-\pi circuit
    EJ_CONST = 1 / 3.95
    ECJ_CONST = 1 / (8.0 * EJ_CONST)
    ECS_CONST = 10.0 ** (-3)

    zpifull = qubit.FullZeroPi(
        zeropi_cutoff=10,
        zeta_cutoff=40,
        grid=phi_grid,
        ncut=30,
        EJ=EJ_CONST,
        dEJ=0.05,
        EL=10.0 ** (-3),
        dEL=0.05,
        ECJ=ECJ_CONST,
        dCJ=0.05,
        EC=None,
        ECS=ECS_CONST,
        dC=0.08,
        ng=0.3,
        flux=0.2
    )
    return zpifull


def test_FullZeroPi_init():
    print("FullZeroPi_init()")
    zero_pi = FullZeroPi_initialize()

def test_FullZeroPi_eigenvals():
    print("FullZeroPi_eigenvals()")
    evals_reference = np.asarray([0.42764008, 0.42841829, 0.43047294, 0.43125082, 0.4333058 ,
       0.43408335, 0.43613866, 0.43691588, 0.43897152, 0.43974841,
       0.44180438, 0.44258094, 0.44463723, 0.44541347, 0.44747009,
       0.448246  , 0.45030295, 0.45107853, 0.45313581, 0.45391106,
       0.45596866, 0.45609586, 0.45674358, 0.45675526, 0.45880152,
       0.45892944, 0.45957611, 0.45958781, 0.46163438, 0.46176302,
       0.46240864, 0.46242037, 0.46446724, 0.4645966 , 0.46524117,
       0.46525293, 0.46730009, 0.46743018, 0.4680737 , 0.46808549])
    zero_pi = FullZeroPi_initialize()
    evals_calculated = zero_pi.eigenvals(evals_count=40)
    print(evals_reference - evals_calculated)
    assert np.allclose(evals_reference, evals_calculated)
