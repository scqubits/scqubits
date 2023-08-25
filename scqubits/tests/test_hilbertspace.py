# test_hilbertspace.py
# meant to be run with 'pytest'
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import numpy as np
import pytest
import qutip as qt

import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.utils.spectrum_utils import get_matrixelement_table


@pytest.mark.usefixtures("num_cpus")
class TestHilbertSpace:
    @pytest.fixture(autouse=True)
    def set_tmpdir(self, request):
        setattr(self, "tmpdir", request.getfixturevalue("tmpdir"))

    @staticmethod
    def hilbertspace_initialize():
        fluxonium = scq.Fluxonium.create()
        fluxonium.truncated_dim = 3
        zpifull = scq.FullZeroPi.create()
        zpifull.truncated_dim = 2
        zpifull.grid.pt_count = 20

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([fluxonium, zpifull])

        g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)

        hilbertspace.add_interaction(
            g=g1, op1=fluxonium.n_operator, op2=zpifull.n_theta_operator
        )
        return hilbertspace

    @staticmethod
    def hilbertspace_initialize2():
        CPB1 = scq.Transmon(EJ=40.0, EC=0.2, ng=0.0, ncut=40, truncated_dim=3)
        CPB2 = scq.Transmon(EJ=3.0, EC=1.0, ng=0.0, ncut=10, truncated_dim=4)

        resonator = scq.Oscillator(E_osc=6.0, truncated_dim=4)
        # up to 3 photons (0,1,2,3)

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([CPB1, CPB2, resonator])

        g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)
        g2 = 0.2  # coupling resonator-CPB2 (without charge matrix elements)

        hilbertspace.add_interaction(
            g=g1, op1=CPB1.n_operator, op2=resonator.creation_operator, add_hc=True
        )

        hilbertspace.add_interaction(
            g=g2,
            op1=(CPB2.n_operator(), CPB2),
            op2=(
                resonator.creation_operator() + resonator.annihilation_operator(),
                resonator,
            ),
        )
        return hilbertspace

    def test_HilbertSpace_init(self):
        _ = self.hilbertspace_initialize()

    def manual_hamiltonian(self, esys1=None, esys2=None):
        hilbertspc = self.hilbertspace_initialize()
        [fluxonium, zpifull] = hilbertspc

        g1 = 0.1
        dim1 = fluxonium.truncated_dim
        if esys1 is None:
            evals1, evecs1 = fluxonium.eigensys(dim1)
        else:
            evals1, evecs1 = esys1
        dim2 = zpifull.truncated_dim
        if esys2 is None:
            evals2, evecs2 = zpifull.eigensys(dim1)
        else:
            evals2, evecs2 = esys2
        h1 = hilbertspc.diag_hamiltonian(fluxonium, evals=evals1)
        h2 = hilbertspc.diag_hamiltonian(zpifull, evals=evals2)
        nmat1 = get_matrixelement_table(
            fluxonium.n_operator(), evecs1
        )  # coupling constants for fluxonium
        nmat2 = get_matrixelement_table(
            zpifull.n_theta_operator(), evecs2
        )  # coupling constants for the  zeropi
        hbd = hilbertspc.hubbard_operator
        vfl = sum(
            [
                nmat1[j][k] * hbd(j, k, fluxonium)
                for j in range(dim1)
                for k in range(dim1)
            ]
        )
        vzp = sum(
            [nmat2[j][k] * hbd(j, k, zpifull) for j in range(dim2) for k in range(dim2)]
        )
        bare_hamiltonian = h1 + h2
        interaction_hamiltonian = g1 * vfl * vzp
        return bare_hamiltonian, interaction_hamiltonian

    def test_hamiltonian_InteractionTerm(self):
        hilbertspc = self.hilbertspace_initialize()
        [fluxonium, zpifull] = hilbertspc

        esys1 = fluxonium.eigensys(evals_count=fluxonium.truncated_dim)
        esys2 = zpifull.eigensys(evals_count=zpifull.truncated_dim)
        bare_esys = {0: esys1, 1: esys2}

        (
            bare_hamiltonian_manual,
            interaction_hamiltonian_manual,
        ) = self.manual_hamiltonian(esys1=esys1, esys2=esys2)

        bare_hamiltonian = hilbertspc.bare_hamiltonian()
        interaction_hamiltonian = hilbertspc.interaction_hamiltonian(
            bare_esys=bare_esys
        )
        hamiltonian = hilbertspc.hamiltonian(bare_esys=bare_esys)

        assert bare_hamiltonian_manual == bare_hamiltonian
        assert interaction_hamiltonian_manual == interaction_hamiltonian
        assert bare_hamiltonian + interaction_hamiltonian == hamiltonian

    def test_hamiltonian_InteractionTermStr(self):
        hilbertspc = self.hilbertspace_initialize()
        [fluxonium, zpifull] = hilbertspc

        g1 = 0.1
        hilbertspc.interaction_list = []
        hilbertspc.add_interaction(
            expr="g1 * n * n_theta",
            op1=("n", fluxonium.n_operator),
            op2=("n_theta", zpifull.n_theta_operator(), zpifull),
            const={"g1": g1},
        )

        esys1 = fluxonium.eigensys(evals_count=fluxonium.truncated_dim)
        esys2 = zpifull.eigensys(evals_count=zpifull.truncated_dim)
        bare_esys = {0: esys1, 1: esys2}

        (
            bare_hamiltonian_manual,
            interaction_hamiltonian_manual,
        ) = self.manual_hamiltonian(esys1=esys1, esys2=esys2)

        bare_hamiltonian = hilbertspc.bare_hamiltonian()
        interaction_hamiltonian = hilbertspc.interaction_hamiltonian(
            bare_esys=bare_esys
        )
        hamiltonian = hilbertspc.hamiltonian(bare_esys=bare_esys)

        assert bare_hamiltonian_manual == bare_hamiltonian
        assert interaction_hamiltonian_manual == interaction_hamiltonian
        assert bare_hamiltonian + interaction_hamiltonian == hamiltonian

    def test_hamiltonian_Qobj(self):
        hilbertspc = self.hilbertspace_initialize()
        [fluxonium, zpifull] = hilbertspc

        esys1 = fluxonium.eigensys(evals_count=fluxonium.truncated_dim)
        esys2 = zpifull.eigensys(evals_count=zpifull.truncated_dim)
        bare_esys = {0: esys1, 1: esys2}

        (
            bare_hamiltonian_manual,
            interaction_hamiltonian_manual,
        ) = self.manual_hamiltonian(esys1=esys1, esys2=esys2)

        hilbertspc.interaction_list = []
        hilbertspc.add_interaction(qobj=interaction_hamiltonian_manual)

        bare_hamiltonian = hilbertspc.bare_hamiltonian()
        interaction_hamiltonian = hilbertspc.interaction_hamiltonian(
            bare_esys=bare_esys
        )
        hamiltonian = hilbertspc.hamiltonian(bare_esys=bare_esys)

        assert bare_hamiltonian_manual == bare_hamiltonian
        assert interaction_hamiltonian_manual == interaction_hamiltonian
        assert bare_hamiltonian + interaction_hamiltonian == hamiltonian

    def hamiltonian_use_addhc(self):
        res1 = scq.Oscillator(E_osc=6.0, truncated_dim=4)
        res2 = scq.Oscillator(E_osc=5.5, truncated_dim=7)

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([res1, res2])

        g1 = 0.29

        hilbertspace.add_interaction(
            g=g1,
            op1=res1.annihilation_operator,
            op2=res2.creation_operator,
            add_hc=True,
        )
        return hilbertspace.hamiltonian()

    def test_HilbertSpace_hamiltonian_is_hermitian(self):
        hamiltonian = self.hilbertspace_initialize().hamiltonian()
        assert hamiltonian.isherm
        hamiltonian = self.hamiltonian_use_addhc()
        assert hamiltonian.isherm

    def test_HilbertSpace_diagonalize_hamiltonian(self):
        evals_reference = np.asarray(
            [
                -36.9898613,
                -32.2485069,
                -31.31250908,
                -31.00035225,
                -29.18345776,
                -26.26664068,
                -25.32975243,
                -25.01086732,
                -24.44211916,
                -23.50612209,
                -23.19649424,
                -21.58197308,
                -20.28449459,
                -19.9790977,
                -19.34686735,
                -19.01220621,
                -18.46278662,
                -17.52590027,
                -17.2084294,
                -16.84047711,
                -15.90462096,
                -15.54530262,
                -14.25509299,
                -13.99415794,
                -13.33019265,
                -12.48208655,
                -12.1727023,
                -11.54418665,
                -11.25656601,
                -10.81121745,
                -9.87458635,
                -9.51009429,
                -8.00925198,
                -6.50020557,
                -6.19030846,
                -5.57523232,
                -4.78354995,
                -4.57123207,
                -3.84547113,
                -3.58389199,
                -2.01787739,
                -0.20685665,
                1.17306434,
                1.46098501,
                2.09778458,
                5.73747149,
                7.49164636,
                13.4096702,
            ]
        )
        hilbertspc = self.hilbertspace_initialize2()
        hamiltonian = hilbertspc.hamiltonian()
        evals_calculated = hamiltonian.eigenenergies()
        assert np.allclose(evals_calculated, evals_reference)

    def test_HilbertSpace_get_spectrum_vs_paramvals(self, num_cpus):
        scq.settings.MULTIPROC = "pathos"
        hilbertspc = self.hilbertspace_initialize2()
        [transmon1, transmon2, resonator] = hilbertspc

        def update_func(flux):
            transmon1.EJ = 40.0 * np.cos(np.pi * flux)

        flux_list = np.linspace(-0.1, 0.6, 100)
        specdata = hilbertspc.get_spectrum_vs_paramvals(
            flux_list,
            update_func,
            evals_count=15,
            get_eigenstates=True,
            num_cpus=num_cpus,
        )
        specdata.filewrite(filename=self.tmpdir + "test.hdf5")

        reference_evals = np.array(
            [
                -35.61652712,
                -30.87517395,
                -29.93917493,
                -29.62790643,
                -27.95527403,
                -24.89419514,
                -23.95730396,
                -23.63931249,
                -23.21394042,
                -22.27794233,
                -21.96970863,
                -20.49874123,
                -18.91294047,
                -18.60576359,
                -17.97530778,
            ]
        )
        calculated_evals = specdata.energy_table[2]

        assert np.allclose(reference_evals, calculated_evals)

    def test_HilbertSpace_eigenenergies(self):
        hilbertspace = self.hilbertspace_initialize2()
        evals, _ = hilbertspace.hamiltonian().eigenstates()

        evals_reference = np.asarray(
            [
                -36.9898613,
                -32.2485069,
                -31.31250908,
                -31.00035225,
                -29.18345776,
                -26.26664068,
                -25.32975243,
                -25.01086732,
                -24.44211916,
                -23.50612209,
                -23.19649424,
                -21.58197308,
                -20.28449459,
                -19.9790977,
                -19.34686735,
                -19.01220621,
                -18.46278662,
                -17.52590027,
                -17.2084294,
                -16.84047711,
                -15.90462096,
                -15.54530262,
                -14.25509299,
                -13.99415794,
                -13.33019265,
                -12.48208655,
                -12.1727023,
                -11.54418665,
                -11.25656601,
                -10.81121745,
                -9.87458635,
                -9.51009429,
                -8.00925198,
                -6.50020557,
                -6.19030846,
                -5.57523232,
                -4.78354995,
                -4.57123207,
                -3.84547113,
                -3.58389199,
                -2.01787739,
                -0.20685665,
                1.17306434,
                1.46098501,
                2.09778458,
                5.73747149,
                7.49164636,
                13.4096702,
            ]
        )
        assert np.allclose(evals, evals_reference)

    def test_HilbertSpace_fileIO(self):
        hilbertspc = self.hilbertspace_initialize2()
        hilbertspc.generate_lookup()
        hilbertspc.filewrite(self.tmpdir + "test.h5")
        hilbertspc_copy = scq.read(self.tmpdir + "test.h5")

    @staticmethod
    def test_HilbertSpace_GUI():
        hilbertspace_new = scq.HilbertSpace.create()

    def test_HilbertSpace_op_in_dressed_basis(self):
        E_osc_a = 4.0
        E_osc_b = 3.2
        g = 0.01
        Delta = E_osc_a - E_osc_b
        truncated_dim = 4
        theta = 0.5 * np.arctan(2 * g / Delta)
        osc_a = scq.Oscillator(E_osc=E_osc_a, truncated_dim=truncated_dim)
        osc_b = scq.Oscillator(E_osc=E_osc_b, truncated_dim=truncated_dim)
        hilbert_space = scq.HilbertSpace([osc_a, osc_b])
        hilbert_space.add_interaction(
            g=g,
            op1=osc_a.creation_operator,
            op2=osc_b.annihilation_operator,
            add_hc=True,
        )
        hilbert_space.generate_lookup()
        hilbert_space.standardize_eigenvector_phases()
        # analytic answer for the dressed operator based on a Bogoliubov transformation
        a_id_wrap = scq.utils.spectrum_utils.identity_wrap(
            osc_a.annihilation_operator(), osc_a, hilbert_space.subsystem_list
        )
        b_id_wrap = scq.utils.spectrum_utils.identity_wrap(
            osc_b.annihilation_operator(), osc_b, hilbert_space.subsystem_list
        )
        analytic_op = np.cos(theta) * a_id_wrap - np.sin(theta) * b_id_wrap
        # need to order this operator according to the dressed indices for later
        # comparison with operators expressed in the dressed basis
        ordered_bare_indices = [
            hilbert_space.bare_index(idx) for idx in range(truncated_dim**2)
        ]
        ordered_basis_states = [
            qt.tensor(qt.basis(truncated_dim, idx_a), qt.basis(truncated_dim, idx_b))
            for (idx_a, idx_b) in ordered_bare_indices
        ]
        # consider only matrix elements unaffected by the truncation level
        analytic_op_ordered = qt.Qobj(
            analytic_op.transform(ordered_basis_states)[0:10, 0:10]
        )
        op1 = qt.Qobj(
            hilbert_space.op_in_dressed_eigenbasis(op=osc_a.annihilation_operator)[
                0:10, 0:10
            ]
        )
        op2 = qt.Qobj(
            hilbert_space.op_in_dressed_eigenbasis(
                op=(osc_a.annihilation_operator(), osc_a)
            )[0:10, 0:10]
        )
        op3 = qt.Qobj(
            hilbert_space.op_in_dressed_eigenbasis(
                op=(osc_a.annihilation_operator(), osc_a), op_in_bare_eigenbasis=True
            )[0:10, 0:10]
        )
        op4 = qt.Qobj(
            hilbert_space.op_in_dressed_eigenbasis(
                op=(osc_a.annihilation_operator(), osc_a), op_in_bare_eigenbasis=False
            )[0:10, 0:10]
        )
        assert analytic_op_ordered == op1
        assert analytic_op_ordered == op2
        assert analytic_op_ordered == op3
        assert analytic_op_ordered == op4

    def test_HilbertSpace_op_in_dressed_basis_native_vs_bare_basis(self):
        E_osc = 4.0
        g = 0.01
        truncated_dim = 4
        tmon = scq.Transmon(
            EJ=10.0, EC=0.2, ng=0.0, ncut=15, truncated_dim=truncated_dim
        )
        osc = scq.Oscillator(E_osc=E_osc, truncated_dim=truncated_dim)
        hilbert_space = scq.HilbertSpace([tmon, osc])
        hilbert_space.add_interaction(
            g=g,
            op1=tmon.n_operator,
            op2=osc.annihilation_operator,
            add_hc=True,
        )
        hilbert_space.generate_lookup()
        op1 = hilbert_space.op_in_dressed_eigenbasis(op=tmon.n_operator)
        n_op_bare_eigenbasis_v2 = tmon.n_operator(energy_esys=True)
        op2 = hilbert_space.op_in_dressed_eigenbasis(
            op=(n_op_bare_eigenbasis_v2, tmon), op_in_bare_eigenbasis=True
        )
        assert op1 == op2
