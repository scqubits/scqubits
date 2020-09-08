# test_transmon.py
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

from scqubits import Fluxonium, FluxQubit, Grid1d, Transmon, TunableTransmon, ZeroPi

data = {}


data['Transmon']=np.array([1.33428506e+07, 3.77005675e+00, 2.16683864e+06, 5.02320540e+02,
       2.16683864e+06, 3.77005240e+00])
data['TunableTransmon']=np.array([2.03732888e+04, 1.60438006e+06, 9.42324266e+05, 1.50547341e+04,
                  np.inf, 1.86529682e-01, 1.50547341e+04, 1.19075232e+04])
data['Fluxonium']=np.array([4.32535846e+06,            np.inf, 5.48954061e+06, 8.71690601e+02,
       1.09436102e+08, 5.18435915e+06,            np.inf, 8.71398774e+02,
       1.74209561e+03])
data['FluxQubit']=np.array([44697811.55376147, 44697796.19977181,  1822725.05439266,
        1685277.68169807,               np.inf,   842638.84084903])
data['ZeroPi']=np.array([ 3803362.64526863, 40606502.43975895,               np.inf,
                     np.inf,               np.inf,  3477633.94999513])



def calc_coherence(qubit, noise_methods=None):
    if noise_methods is None:
        noise_methods = qubit.supported_noise_channels()+['t1_effective', 't2_effective']

    def cap_coherence(time): 
        return np.inf if time>1e14 else time

    return np.array([cap_coherence(getattr(qubit, m)()) for m in noise_methods])

def compare_coherence_to_reference(qubit, qubit_name):
    noise = calc_coherence(qubit)
    print("comparison:\n", [(noise[i], data[qubit_name][i]) for i, _ in enumerate(noise)])
    return np.allclose(noise, data[qubit_name], equal_nan=True)

class TestNoise:

    def test_Transmon(self):
        qubit = Transmon(EJ=0.5, EC=12.0, ng=0.3, ncut=150)
        assert compare_coherence_to_reference(qubit, 'Transmon')

    def test_TunableTransmon(self):
        qubit = TunableTransmon(EJmax=20.0, EC=0.5, d=0.00, flux=0.04, ng=0.3, ncut=150)
        assert compare_coherence_to_reference(qubit, 'TunableTransmon')

    def test_Fluxonium(self):
        qubit = Fluxonium(EJ=8.9, EC=2.5, EL=0.5, cutoff=150, flux=0.5)
        assert compare_coherence_to_reference(qubit, 'Fluxonium')

    def test_FluxQubit(self):
        RATIO = 60.0
        ALPHA = 0.8
        qubit = FluxQubit(
            EJ1=1.0,
            EJ2=1.0,
            EJ3=ALPHA*1.0,
            ECJ1=1.0/RATIO,
            ECJ2=1.0/RATIO,
            ECJ3=1.0/ALPHA/RATIO,
            ECg1=50.0/RATIO,
            ECg2=50.0/RATIO,
            ng1=0.0,
            ng2=0.0,
            flux=0.4,
            ncut=10,
        )
        assert compare_coherence_to_reference(qubit, 'FluxQubit')

    def test_ZeroPi(self):
        phi_grid = Grid1d(-6*np.pi, 6*np.pi, 200)
        EJ_CONST = 1/3.95  # note that EJ and ECJ are interrelated
        qubit = ZeroPi(
            grid=phi_grid,
            EJ=EJ_CONST,
            EL=10.0**(-2),
            ECJ=1/(8.0*EJ_CONST),
            EC=None,
            ECS=10.0**(-3),
            ng=0.1,
            flux=0.23,
            ncut=30
        )
        assert compare_coherence_to_reference(qubit, 'ZeroPi')


