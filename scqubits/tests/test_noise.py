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

data={}

data['Transmon']=np.array([1.33428506e+07, 3.77005675e+00, 2.16683864e+06, 5.02320540e+02])
data['TunableTransmon']=np.array([2.03732888e+04, 1.60438006e+06, 9.42324266e+05, 1.50547341e+04,
       4.39536861e+17, 1.86529682e-01])
data['Fluxonium']=np.array([4.32535846e+06, 1.50856659e+17, 1.09436102e+08, 5.48954061e+06,
       5.09223418e+06, 8.71690601e+02])
data['FluxQubit']=np.array([44697800.15221689, 44697800.1522218 ,  1822725.06489296,
        1685277.680085  ])
data['ZeroPi']=np.array([3.80336265e+06, 4.06065024e+07, 1.49416281e+37, 1.17269763e+33])

class TestNoise:

    def _calc_noise(self, qubit, noise_channel_methods):
        return np.array([getattr(qubit, m)() for m in noise_channel_methods])

    def test_Transmon(self):
        qubit=Transmon(EJ=0.5, EC=12.0, ng=0.3, ncut=150)
        noise=self._calc_noise(qubit, qubit.supported_noise_channels())
        assert np.all(np.isclose(noise, data['Transmon']))

    def test_TunableTransmon(self):
        qubit=TunableTransmon(EJmax=20.0, EC=0.5, d=0.00, flux=0.04, ng=0.3, ncut=150)
        noise=self._calc_noise(qubit, qubit.supported_noise_channels())
        assert np.all(np.isclose(noise, data['TunableTransmon']))

    def test_Fluxonium(self):
        qubit=Fluxonium(EJ=8.9, EC=2.5, EL=0.5, cutoff = 150, flux = 0.5)
        noise=self._calc_noise(qubit, qubit.supported_noise_channels())
        assert np.all(np.isclose(noise, data['Fluxonium']))

    def test_FluxQubit(self):
        RATIO = 60.0
        ALPHA = 0.8
        qubit = FluxQubit(
            EJ1 = 1.0, 
            EJ2 = 1.0, 
            EJ3 = ALPHA*1.0, 
            ECJ1 = 1.0/RATIO, 
            ECJ2 = 1.0/RATIO, 
            ECJ3 = 1.0/ALPHA/RATIO, 
            ECg1 = 50.0/RATIO, 
            ECg2 = 50.0/RATIO, 
            ng1 = 0.0, 
            ng2 = 0.0, 
            flux = 0.4, 
            ncut = 10,
        )
        noise=self._calc_noise(qubit, qubit.supported_noise_channels())
        assert np.all(np.isclose(noise, data['FluxQubit']))

    def test_ZeroPi(self):
        phi_grid = Grid1d(-6*np.pi, 6*np.pi, 200)
        EJ_CONST = 1/3.95  # note that EJ and ECJ are interrelated
        qubit = ZeroPi(
            grid = phi_grid,
            EJ   = EJ_CONST,
            EL   = 10.0**(-2),
            ECJ  = 1/(8.0*EJ_CONST),
            EC = None,
            ECS  = 10.0**(-3),
            ng   = 0.1,
            flux = 0.23,
            ncut = 30
        )
        noise=self._calc_noise(qubit, qubit.supported_noise_channels())
        assert np.all(np.isclose(noise, data['ZeroPi']))
