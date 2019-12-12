# test_hilbertspace.py
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

from __future__ import division
from __future__ import print_function

import numpy as np

import scqubits as qubit
from scqubits.utils.spectrum_utils import get_matrixelement_table
from scqubits.utils.constants import TEMPDIR


def hilbertspace_initialize():
    transmon1 = qubit.Transmon(
        EJ=40.0,
        EC=0.2,
        ng=0.0,
        ncut=40,
        truncated_dim=3  # after diagonalization, we will keep 3 levels
    )

    transmon2 = qubit.Transmon(
        EJ=3.0,
        EC=1.0,
        ng=0.0,
        ncut=10,
        truncated_dim=4
    )

    resonator = qubit.Oscillator(
        omega=6.0,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )

    # Form a list of all components making up the Hilbert space.
    return qubit.HilbertSpace([transmon1, transmon2, resonator])


def test_HilbertSpace_init():
    _ = hilbertspace_initialize()


def test_HilbertSpace_diag_hamiltonian():
    transmon1 = qubit.Transmon(
        EJ=40.0,
        EC=0.2,
        ng=0.0,
        ncut=40,
        truncated_dim=3  # after diagonalization, we will keep 3 levels
    )

    transmon2 = qubit.Transmon(
        EJ=3.0,
        EC=1.0,
        ng=0.0,
        ncut=10,
        truncated_dim=4
    )

    resonator = qubit.Oscillator(
        omega=6.0,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )

    # Form a list of all components making up the Hilbert space.
    hilbertspc = qubit.HilbertSpace([transmon1, transmon2, resonator])

    # Get resonator Hamiltonian (full product Hilbert space)
    hres = hilbertspc.diag_hamiltonian(resonator)

    # Get diagonalized transmon1 Hamiltonian as full-system operator via tensor product with identities.
    h1 = hilbertspc.diag_hamiltonian(transmon1)

    # Get diagonalized transmon2 Hamiltonian as full-system operator via tensor product with identities.
    h2 = hilbertspc.diag_hamiltonian(transmon2)


def test_HilbertSpace_build_hamiltonian():
    transmon1 = qubit.Transmon(
        EJ=40.0,
        EC=0.2,
        ng=0.0,
        ncut=40,
        truncated_dim=3  # after diagonalization, we will keep 3 levels
    )
    transmon2 = qubit.Transmon(
        EJ=3.0,
        EC=1.0,
        ng=0.0,
        ncut=10,
        truncated_dim=4
    )
    resonator = qubit.Oscillator(
        omega=6.0,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )
    # Form a list of all components making up the Hilbert space.
    hilbertspc =  qubit.HilbertSpace([transmon1, transmon2, resonator])
    # Get resonator Hamiltonian (full product Hilbert space)
    hres = hilbertspc.diag_hamiltonian(resonator)
    # Get diagonalized transmon1 Hamiltonian as full-system operator via tensor product with identities.
    h1 = hilbertspc.diag_hamiltonian(transmon1)
    # Get diagonalized transmon2 Hamiltonian as full-system operator via tensor product with identities.
    h2 = hilbertspc.diag_hamiltonian(transmon2)
    g1 = 0.1  # coupling resonator-transmon1 (without charge matrix elements)
    g2 = 0.2  # coupling resonator-transmon2 (without charge matrix elements)
    dim1 = transmon1.truncated_dim
    dim2 = transmon2.truncated_dim
    _, evecs1 = transmon1.eigensys(dim1)
    _, evecs2 = transmon2.eigensys(dim2)
    gmat1 = g1 * get_matrixelement_table(transmon1.n_operator(), evecs1)  # coupling constants for transmon1
    gmat2 = g2 * get_matrixelement_table(transmon2.n_operator(), evecs2)  # and for transmon2
    hbd = hilbertspc.hubbard_operator
    a = hilbertspc.annihilate(resonator)
    hamiltonian0 = h1 + h2 + hres
    vcpb1 = sum([gmat1[j][k] * hbd(j, k, transmon1) for j in range(dim1) for k in range(dim1)])
    vcpb2 = sum([gmat2[j][k] * hbd(j, k, transmon2) for j in range(dim2) for k in range(dim2)])
    hamiltonian1 = (vcpb1 + vcpb2) * (a + a.dag())
    hamiltonian = hamiltonian0 + hamiltonian1


def test_HilbertSpace_diagonalize_hamiltonian():
    transmon1 = qubit.Transmon(
        EJ=40.0,
        EC=0.2,
        ng=0.0,
        ncut=40,
        truncated_dim=3  # after diagonalization, we will keep 3 levels
    )
    transmon2 = qubit.Transmon(
        EJ=3.0,
        EC=1.0,
        ng=0.0,
        ncut=10,
        truncated_dim=4
    )
    resonator = qubit.Oscillator(
        omega=6.0,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )
    # Form a list of all components making up the Hilbert space.
    hilbertspc =  qubit.HilbertSpace([transmon1, transmon2, resonator])
    # Get resonator Hamiltonian (full product Hilbert space)
    hres = hilbertspc.diag_hamiltonian(resonator)
    # Get diagonalized transmon1 Hamiltonian as full-system operator via tensor product with identities.
    h1 = hilbertspc.diag_hamiltonian(transmon1)
    # Get diagonalized transmon2 Hamiltonian as full-system operator via tensor product with identities.
    h2 = hilbertspc.diag_hamiltonian(transmon2)
    g1 = 0.1  # coupling resonator-transmon1 (without charge matrix elements)
    g2 = 0.2  # coupling resonator-transmon2 (without charge matrix elements)
    dim1 = transmon1.truncated_dim
    dim2 = transmon2.truncated_dim
    _, evecs1 = transmon1.eigensys(dim1)
    _, evecs2 = transmon2.eigensys(dim2)
    gmat1 = g1 * get_matrixelement_table(transmon1.n_operator(), evecs1)  # coupling constants for transmon1
    gmat2 = g2 * get_matrixelement_table(transmon2.n_operator(), evecs2)  # and for transmon2
    hbd = hilbertspc.hubbard_operator
    a = hilbertspc.annihilate(resonator)
    hamiltonian0 = h1 + h2 + hres
    vcpb1 = sum([gmat1[j][k] * hbd(j, k, transmon1) for j in range(dim1) for k in range(dim1)])
    vcpb2 = sum([gmat2[j][k] * hbd(j, k, transmon2) for j in range(dim2) for k in range(dim2)])
    hamiltonian1 = (vcpb1 + vcpb2) * (a + a.dag())
    hamiltonian = hamiltonian0 + hamiltonian1
    evals_reference = np.asarray([-36.9898613 , -32.2485069 , -31.31250908, -31.00035225,
       -29.18345776, -26.26664068, -25.32975243, -25.01086732,
       -24.44211916, -23.50612209, -23.19649424, -21.58197308,
       -20.28449459, -19.9790977 , -19.34686735, -19.01220621,
       -18.46278662, -17.52590027, -17.2084294 , -16.84047711,
       -15.90462096, -15.54530262, -14.25509299, -13.99415794,
       -13.33019265, -12.48208655, -12.1727023 , -11.54418665,
       -11.25656601, -10.81121745,  -9.87458635,  -9.51009429,
        -8.00925198,  -6.50020557,  -6.19030846,  -5.57523232,
        -4.78354995,  -4.57123207,  -3.84547113,  -3.58389199,
        -2.01787739,  -0.20685665,   1.17306434,   1.46098501,
         2.09778458,   5.73747149,   7.49164636,  13.4096702 ])
    evals_calculated = hamiltonian.eigenenergies()
    assert np.allclose(evals_calculated, evals_reference)


def test_HilbertSpace_get_spectrum_vs_paramvals():
    transmon1 = qubit.Transmon(
        EJ=40.0,
        EC=0.2,
        ng=0.3,
        ncut=40,
        truncated_dim=3  # after diagonalization, we will keep 3 levels
    )
    transmon2 = qubit.Transmon(
        EJ=3.0,
        EC=1.0,
        ng=0.0,
        ncut=10,
        truncated_dim=4
    )
    resonator = qubit.Oscillator(
        omega=6.0,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )
    # Form a list of all components making up the Hilbert space.
    hilbertspc =  qubit.HilbertSpace([transmon1, transmon2, resonator])
    # Get resonator Hamiltonian (full product Hilbert space)
    hres = hilbertspc.diag_hamiltonian(resonator)
    # Get diagonalized transmon1 Hamiltonian as full-system operator via tensor product with identities.
    h1 = hilbertspc.diag_hamiltonian(transmon1)
    # Get diagonalized transmon2 Hamiltonian as full-system operator via tensor product with identities.
    h2 = hilbertspc.diag_hamiltonian(transmon2)
    g1 = 0.1  # coupling resonator-transmon1 (without charge matrix elements)
    g2 = 0.2  # coupling resonator-transmon2 (without charge matrix elements)
    dim1 = transmon1.truncated_dim
    dim2 = transmon2.truncated_dim
    _, evecs1 = transmon1.eigensys(dim1)
    _, evecs2 = transmon2.eigensys(dim2)
    gmat1 = g1 * get_matrixelement_table(transmon1.n_operator(), evecs1)  # coupling constants for transmon1
    gmat2 = g2 * get_matrixelement_table(transmon2.n_operator(), evecs2)  # and for transmon2
    hbd = hilbertspc.hubbard_operator
    a = hilbertspc.annihilate(resonator)

    vcpb1 = sum([gmat1[j][k] * hbd(j, k, transmon1) for j in range(dim1) for k in range(dim1)])
    vcpb2 = sum([gmat2[j][k] * hbd(j, k, transmon2) for j in range(dim2) for k in range(dim2)])

    def hamiltonian(flux):
        transmon1.EJ = 40.0 * np.cos(np.pi * flux)
        h1 = hilbertspc.diag_hamiltonian(transmon1)
        vcpb1 = sum([gmat1[j][k] * hbd(j, k, transmon1) for j in range(dim1) for k in range(dim1)])

        return h1 + h2 + hres + (vcpb1 + vcpb2) * (a + a.dag())

    flux_list = np.linspace(-0.1, 0.6, 100)
    specdata = hilbertspc.get_spectrum_vs_paramvals(hamiltonian, flux_list, evals_count=15, get_eigenstates=True,
                                                    filename=TEMPDIR + 'test')

    reference_evals = np.array([-35.61671109, -30.87536252, -29.93935539, -29.62839549,
       -27.95521996, -24.89469034, -23.95779031, -23.64010506,
       -23.21389138, -22.27788515, -21.97003287, -20.49827277,
       -18.91372364, -18.6059474 , -17.97609201])
    calculated_evals = specdata.energy_table[2]

    assert np.allclose(reference_evals, calculated_evals)


def test_HilbertSpace_absorption_spectrum():
    transmon1 = qubit.Transmon(
        EJ=40.0,
        EC=0.2,
        ng=0.3,
        ncut=40,
        truncated_dim=3  # after diagonalization, we will keep 3 levels
    )
    transmon2 = qubit.Transmon(
        EJ=3.0,
        EC=1.0,
        ng=0.0,
        ncut=10,
        truncated_dim=4
    )
    resonator = qubit.Oscillator(
        omega=6.0,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )
    # Form a list of all components making up the Hilbert space.
    hilbertspc =  qubit.HilbertSpace([transmon1, transmon2, resonator])
    # Get resonator Hamiltonian (full product Hilbert space)
    hres = hilbertspc.diag_hamiltonian(resonator)
    # Get diagonalized transmon1 Hamiltonian as full-system operator via tensor product with identities.
    h1 = hilbertspc.diag_hamiltonian(transmon1)
    # Get diagonalized transmon2 Hamiltonian as full-system operator via tensor product with identities.
    h2 = hilbertspc.diag_hamiltonian(transmon2)
    g1 = 0.1  # coupling resonator-transmon1 (without charge matrix elements)
    g2 = 0.2  # coupling resonator-transmon2 (without charge matrix elements)
    dim1 = transmon1.truncated_dim
    dim2 = transmon2.truncated_dim
    _, evecs1 = transmon1.eigensys(dim1)
    _, evecs2 = transmon2.eigensys(dim2)
    gmat1 = g1 * get_matrixelement_table(transmon1.n_operator(), evecs1)  # coupling constants for transmon1
    gmat2 = g2 * get_matrixelement_table(transmon2.n_operator(), evecs2)  # and for transmon2
    hbd = hilbertspc.hubbard_operator
    a = hilbertspc.annihilate(resonator)

    vcpb1 = sum([gmat1[j][k] * hbd(j, k, transmon1) for j in range(dim1) for k in range(dim1)])
    vcpb2 = sum([gmat2[j][k] * hbd(j, k, transmon2) for j in range(dim2) for k in range(dim2)])

    def hamiltonian(flux):
        transmon1.EJ = 40.0 * np.cos(np.pi * flux)
        h1 = hilbertspc.diag_hamiltonian(transmon1)
        vcpb1 = sum([gmat1[j][k] * hbd(j, k, transmon1) for j in range(dim1) for k in range(dim1)])

        return h1 + h2 + hres + (vcpb1 + vcpb2) * (a + a.dag())

    flux_list = np.linspace(-0.1, 0.6, 100)
    specdata = hilbertspc.get_spectrum_vs_paramvals(hamiltonian, flux_list, evals_count=15, get_eigenstates=True)
    absorptiondata = hilbertspc.absorption_spectrum(specdata, ((transmon1, 0), (transmon2, 0), (resonator, 0)),
                                                    initial_as_bare=True)

    reference_energies = np.array([  0.        ,   4.74135306,   5.67735204,   5.9888575 ,
         7.72433769,  10.72256729,  11.6594581 ,  11.97768947,
        12.46567274,  13.40167088,  13.71033926,  15.24394963,
        16.70406082,  17.01076357,  17.64169146])


    calculated_energies = absorptiondata.energy_table[5]

    assert np.allclose(reference_energies, calculated_energies)
