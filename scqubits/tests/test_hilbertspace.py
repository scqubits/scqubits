# test_hilbertspace.py
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

import scqubits as qubit
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.sweep_generators import generate_diffspec_sweep
from scqubits.utils.spectrum_utils import absorption_spectrum, get_matrixelement_table


@pytest.mark.usefixtures("num_cpus")
class TestHilbertSpace:
    @pytest.fixture(autouse=True)
    def set_tmpdir(self, request):
        setattr(self, 'tmpdir', request.getfixturevalue('tmpdir'))

    @staticmethod
    def hilbertspace_initialize():
        CPB1 = qubit.Transmon(
            EJ=40.0,
            EC=0.2,
            ng=0.0,
            ncut=40,
            truncated_dim=3  # after diagonalization, we will keep 3 levels
        )

        CPB2 = qubit.Transmon(
            EJ=3.0,
            EC=1.0,
            ng=0.0,
            ncut=10,
            truncated_dim=4
        )

        resonator = qubit.Oscillator(
            E_osc=6.0,
            truncated_dim=4  # up to 3 photons (0,1,2,3)
        )

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([CPB1, CPB2, resonator])

        g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)
        g2 = 0.2  # coupling resonator-CPB2 (without charge matrix elements)

        interaction1 = InteractionTerm(
            g_strength=g1,
            op1=CPB1.n_operator(),
            subsys1=CPB1,
            op2=resonator.creation_operator() + resonator.annihilation_operator(),
            subsys2=resonator
        )

        interaction2 = InteractionTerm(
            g_strength=g2,
            op1=CPB2.n_operator(),
            subsys1=CPB2,
            op2=resonator.creation_operator() + resonator.annihilation_operator(),
            subsys2=resonator
        )

        interaction_list = [interaction1, interaction2]
        hilbertspace.interaction_list = interaction_list

        return hilbertspace

    def test_HilbertSpace_init(self):
        _ = self.hilbertspace_initialize()

    def build_hamiltonian(self):
        hilbertspc = self.hilbertspace_initialize()
        [transmon1, transmon2, resonator] = hilbertspc
        hres = hilbertspc.diag_hamiltonian(resonator)
        h1 = hilbertspc.diag_hamiltonian(transmon1)
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
        return hamiltonian0 + hamiltonian1

    def hamiltonian(self, flux):
        hilbertspc = self.hilbertspace_initialize()
        [transmon1, transmon2, resonator] = hilbertspc

        hres = hilbertspc.diag_hamiltonian(resonator)
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

        transmon1.EJ = 40.0 * np.cos(np.pi * flux)
        h1 = hilbertspc.diag_hamiltonian(transmon1)
        h2 = hilbertspc.diag_hamiltonian(transmon2)

        return h1 + h2 + hres + (vcpb1 + vcpb2) * (a + a.dag())

    def hamiltonian_use_addhc(self):
        res1 = qubit.Oscillator(
            E_osc=6.0,
            truncated_dim=4  # up to 3 photons (0,1,2,3)
        )

        res2 = qubit.Oscillator(
            E_osc=5.5,
            truncated_dim=7
        )

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([res1, res2])

        g1 = 0.29

        interaction1 = InteractionTerm(
            g_strength=g1,
            op1=res1.annihilation_operator(),
            subsys1=res1,
            op2=res2.creation_operator(),
            subsys2=res2,
            add_hc=True
        )

        interaction_list = [interaction1]
        hilbertspace.interaction_list = interaction_list
        return hilbertspace.hamiltonian()

    def test_HilbertSpace_hamiltonian_is_hermitean(self):
        hamiltonian = self.build_hamiltonian()
        assert np.isclose(np.max(np.abs(hamiltonian - hamiltonian.dag())), 0.0)
        hamiltonian = self.hamiltonian(flux=0.23)
        assert np.isclose(np.max(np.abs(hamiltonian - hamiltonian.dag())), 0.0)
        hamiltonian = self.hamiltonian_use_addhc()
        assert np.isclose(np.max(np.abs(hamiltonian - hamiltonian.dag())), 0.0)

    def test_HilbertSpace_diagonalize_hamiltonian(self):
        hamiltonian = self.build_hamiltonian()

        evals_reference = np.asarray([-36.9898613, -32.2485069, -31.31250908, -31.00035225,
                                      -29.18345776, -26.26664068, -25.32975243, -25.01086732,
                                      -24.44211916, -23.50612209, -23.19649424, -21.58197308,
                                      -20.28449459, -19.9790977, -19.34686735, -19.01220621,
                                      -18.46278662, -17.52590027, -17.2084294, -16.84047711,
                                      -15.90462096, -15.54530262, -14.25509299, -13.99415794,
                                      -13.33019265, -12.48208655, -12.1727023, -11.54418665,
                                      -11.25656601, -10.81121745, -9.87458635, -9.51009429,
                                      -8.00925198, -6.50020557, -6.19030846, -5.57523232,
                                      -4.78354995, -4.57123207, -3.84547113, -3.58389199,
                                      -2.01787739, -0.20685665, 1.17306434, 1.46098501,
                                      2.09778458, 5.73747149, 7.49164636, 13.4096702])

        evals_calculated = hamiltonian.eigenenergies()
        assert np.allclose(evals_calculated, evals_reference)

    def test_HilbertSpace_get_spectrum_vs_paramvals(self, num_cpus):
        qubit.settings.MULTIPROC = 'pathos'
        hilbertspc = self.hilbertspace_initialize()
        [transmon1, transmon2, resonator] = hilbertspc

        def update_func(flux):
            transmon1.EJ = 40.0 * np.cos(np.pi * flux)

        flux_list = np.linspace(-0.1, 0.6, 100)
        specdata = hilbertspc.get_spectrum_vs_paramvals(flux_list, update_func, evals_count=15,
                                                        get_eigenstates=True, num_cpus=num_cpus)
        specdata.filewrite(filename=self.tmpdir + 'test.hdf5')

        reference_evals = np.array([-35.61652712, -30.87517395, -29.93917493, -29.62790643, -27.95527403, -24.89419514,
                                    -23.95730396, -23.63931249, -23.21394042, -22.27794233, -21.96970863, -20.49874123,
                                    -18.91294047, -18.60576359, -17.97530778])
        calculated_evals = specdata.energy_table[2]

        assert np.allclose(reference_evals, calculated_evals)

    def test_HilbertSpace_eigenenergies(self):
        hilbertspace = self.hilbertspace_initialize()
        evals, _ = hilbertspace.hamiltonian().eigenstates()

        evals_reference = np.asarray([-36.9898613, -32.2485069, -31.31250908, -31.00035225,
                                      -29.18345776, -26.26664068, -25.32975243, -25.01086732,
                                      -24.44211916, -23.50612209, -23.19649424, -21.58197308,
                                      -20.28449459, -19.9790977, -19.34686735, -19.01220621,
                                      -18.46278662, -17.52590027, -17.2084294, -16.84047711,
                                      -15.90462096, -15.54530262, -14.25509299, -13.99415794,
                                      -13.33019265, -12.48208655, -12.1727023, -11.54418665,
                                      -11.25656601, -10.81121745, -9.87458635, -9.51009429,
                                      -8.00925198, -6.50020557, -6.19030846, -5.57523232,
                                      -4.78354995, -4.57123207, -3.84547113, -3.58389199,
                                      -2.01787739, -0.20685665, 1.17306434, 1.46098501,
                                      2.09778458, 5.73747149, 7.49164636, 13.4096702])
        assert np.allclose(evals, evals_reference)


@pytest.mark.usefixtures("num_cpus")
class TestParameterSweep:
    def initialize(self, num_cpus):
        # Set up the components / subspaces of our Hilbert space
        qubit.settings.MULTIPROC = 'pathos'

        CPB1 = qubit.Transmon(
            EJ=40.0,
            EC=0.2,
            ng=0.0,
            ncut=40,
            truncated_dim=3  # after diagonalization, we will keep 3 levels
        )

        CPB2 = qubit.Transmon(
            EJ=3.0,
            EC=1.0,
            ng=0.0,
            ncut=10,
            truncated_dim=4
        )

        resonator = qubit.Oscillator(
            E_osc=6.0,
            truncated_dim=4  # up to 3 photons (0,1,2,3)
        )

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([CPB1, CPB2, resonator])

        g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)
        g2 = 0.2  # coupling resonator-CPB2 (without charge matrix elements)

        interaction1 = InteractionTerm(
            g_strength=g1,
            op1=CPB1.n_operator(),
            subsys1=CPB1,
            op2=resonator.creation_operator() + resonator.annihilation_operator(),
            subsys2=resonator
        )

        interaction2 = InteractionTerm(
            g_strength=g2,
            op1=CPB2.n_operator(),
            subsys1=CPB2,
            op2=resonator.creation_operator() + resonator.annihilation_operator(),
            subsys2=resonator
        )

        interaction_list = [interaction1, interaction2]
        hilbertspace.interaction_list = interaction_list

        param_name = 'flux'  # name of varying external parameter
        param_vals = np.linspace(-0.1, 0.6, 100)  # parameter values

        subsys_update_list = [CPB1]  # list of HilbertSpace subsys_list which are affected by parameter changes

        def update_hilbertspace(param_val):  # function that shows how Hilbert space
            # components are updated
            CPB1.EJ = 40.0 * np.cos(np.pi * param_val)

        sweep = ParameterSweep(
            param_name=param_name,
            param_vals=param_vals,
            evals_count=15,
            hilbertspace=hilbertspace,
            subsys_update_list=subsys_update_list,
            update_hilbertspace=update_hilbertspace,
            num_cpus=num_cpus
        )
        return sweep

    def test_ParameterSweep(self, num_cpus):
        sweep = self.initialize(num_cpus)

        specdata = absorption_spectrum(generate_diffspec_sweep(sweep, initial_state_ind=0))
        calculated_energies = specdata.energy_table[5]

        reference_energies = np.array([0., 4.74135372, 5.6773522, 5.98902462, 7.72420838, 10.72273595, 11.65962582,
                                       11.97802377, 12.46554431, 13.40154194, 13.71041554, 15.24359501, 16.70439594,
                                       17.01076356, 17.64202619])
        assert np.allclose(reference_energies, calculated_energies)
