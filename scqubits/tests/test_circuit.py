# test_circuit.py
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

import os
import numpy as np
import pytest

import scqubits as scq

TESTDIR, _ = os.path.split(scq.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")
DATADIR = os.path.join(TESTDIR, "data", "")


@pytest.mark.usefixtures("num_cpus")
class TestCircuit:
    @staticmethod
    def test_sym_lagrangian():
        zp_yaml = """
        # zero-pi circuit
        branches:
        - ["JJ", 1, 2, EJ=10, 20] 
        - ["JJ", 3, 4, EJ, 20]
        - ["L", 2, 3, 0.01]
        - ["L", 4, 1, 0.008]
        - ["C", 1, 3, 0.02]
        - ["C", 2, 4, 0.02]
        """
        REFERENCE = "<bound method Printable.__str__ of EJ*cos(θ1 + θ3) + EJ*cos((2πΦ_{1}) + θ1 - 1.0*θ3) + 6.25625*\\dot{θ_1}**2 + 25.0*\\dot{θ_2}**2 + 0.00625*\\dot{θ_3}**2 - 0.036*θ2**2 - 0.004*θ2*θ3 - 0.009*θ3**2>"

        zero_pi = scq.Circuit(zp_yaml, from_file=False, ext_basis="discretized")
        expression_str = str(
            zero_pi.sym_lagrangian(vars_type="new", return_expr=True).doit().__repr__
        )
        assert expression_str == REFERENCE

    @staticmethod
    def test_zero_pi_discretized():
        """
        Test for symmetric zero-pi in discretized phi basis.
        """
        zp_yaml = """# zero-pi circuit
        branches:
        - ["JJ", 1, 2, 10, 20]
        - ["JJ", 3, 4, 10, 20]
        - ["L", 2, 3, 0.008]
        - ["L", 4, 1, 0.008]
        - ["C", 1, 3, 0.02]
        - ["C", 2, 4, 0.02]
        """

        circ_d = scq.Circuit(zp_yaml, from_file=False, ext_basis="discretized")
        circ_d.cutoff_n_1 = 30
        circ_d.cutoff_ext_2 = 30
        circ_d.cutoff_ext_3 = 80
        circ_d.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[30, 20])

        circ_d.cutoff_ext_3 = 200
        sym_zp = circ_d.subsystems[0]
        eigensys = sym_zp.eigensys()
        eigs = eigensys[0]
        eigs_ref = np.array(
            [
                -3.69090429,
                -3.69049138,
                -2.89704215,
                -2.89659842,
                -2.77231275,
                -2.76823373,
            ]
        )

        assert np.allclose(eigs, eigs_ref)

    @staticmethod
    def test_circuit_with_symbolic_hamiltonian():
        """
        Test for initiating Circuit module with symbolic Hamiltonian.
        """
        import sympy as sm

        sym_hamiltonian = sm.parse_expr(
            "0.25*θ3**2 + 2.0*Q3**2 + 0.790697674419*Q2**2 + 0.45*θ2**2 + 7.674418604651*n1**2 + 7.674418604651*ng1**2 - 1.0*cos(θ1) + 0.5*θ2*θ3 + 1.395348837209*Q2*n1 + 1.395348837209*Q2*ng1 + 15.348837209302*n1*ng1"
        )
        circ = scq.Circuit(
            input_string=None,
            symbolic_hamiltonian=sym_hamiltonian,
            symbolic_param_dict={"ng1": 0},
            ext_basis="harmonic",
        )
        circ.configure(
            transformation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        )
        circ.cutoff_n_1 = 20
        circ.cutoff_ext_2 = 20
        circ.cutoff_ext_3 = 20
        circ.configure(
            system_hierarchy=[[1], [[2], [3]]],
            subsystem_trunc_dims=[20, [50, [10, 10]]],
        )
        # new_circ.configure(system_hierarchy=[[1], [2, 3]], subsystem_trunc_dims=[20, 30])
        circ.ng1 = 0.5
        eigs = circ.eigenvals()
        eigs_ref = np.array(
            [2.51547879, 3.00329327, 3.5556228, 3.57568727, 4.13233136, 4.29671029]
        )
        assert np.allclose(eigs, eigs_ref)

    @staticmethod
    def test_eigenvals_harmonic():
        ref_eigs = np.array(
            [0.0, 0.03559404, 0.05819727, 0.09378676, 4.39927874, 4.43488613]
        )
        inp_yaml = """
        branches:
        - [JJ, 0, 1, 1, 15]
        - [C, 1, 2, 2]
        - [L, 2, 0, 0.4]
        - [C, 2, 0, 0.2]
        - [C, 2, 3, 0.5]
        - [L, 3, 0, 0.5]
        # - [JJ, 3, 0, EJ=0, 1e5]
        """
        circ = scq.Circuit(inp_yaml, from_file=False, ext_basis="discretized")
        circ.configure(
            transformation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        )
        circ.cutoff_n_1 = 20
        circ.cutoff_ext_2 = 10
        circ.cutoff_ext_3 = 10
        circ.configure(system_hierarchy=[[1], [2, 3]], subsystem_trunc_dims=[20, 30])
        circ.ng1 = 0.5
        eigs = circ.eigenvals()
        generated_eigs = eigs - eigs[0]
        ref_eigs = np.array(
            [0.0, 0.48790869, 1.04058606, 1.06037218, 1.61763356, 1.78158506]
        )
        assert np.allclose(generated_eigs, ref_eigs)

    @staticmethod
    def test_eigenvals_discretized():
        ref_eigs = np.array(
            [0.0, 0.03559217, 0.05819503, 0.09378266, 4.39921833, 4.43482385]
        )
        DFC = scq.Circuit(
            DATADIR + "circuit_DFC.yaml",
            ext_basis="discretized",
            initiate_sym_calc=False,
            basis_completion="canonical",
        )

        closure_branches = [DFC.branches[0], DFC.branches[4], DFC.branches[-1]]
        system_hierarchy = [[[1], [3]], [2], [4]]
        subsystem_trunc_dims = [[34, [6, 6]], 6, 6]

        DFC.configure(
            closure_branches=closure_branches,
            system_hierarchy=system_hierarchy,
            subsystem_trunc_dims=subsystem_trunc_dims,
        )

        DFC.Φ1 = 0.5 + 0.01768
        DFC.Φ2 = -0.2662
        DFC.Φ3 = -0.5 + 0.01768

        DFC.cutoff_ext_1 = 110
        DFC.cutoff_ext_2 = 110
        DFC.cutoff_ext_3 = 110
        DFC.cutoff_ext_4 = 110
        DFC.update()

        eigs = DFC.eigenvals()
        generated_eigs = eigs - eigs[0]
        assert np.allclose(generated_eigs, ref_eigs)

    @staticmethod
    def test_harmonic_oscillator():
        lc_yaml = """# LC circuit
    branches:
    - [L, 0, 1, 1]
    - [C, 0, 2, 2]
    - [L, 0, 3, 4.56]
    - [C, 2, 3, 40]
    - [C, 2, 1, EJ=40]
    - [C, 4, 1, 10]
    - [L, 4, 2, 10]
    """
        circ = scq.Circuit(
            lc_yaml, from_file=False, initiate_sym_calc=True, ext_basis="harmonic"
        )
        circ.EJ = 0.01
        circ.update()
        eigs_ref = np.array(
            [
                35.681948467838,
                39.5947893122758,
                43.5076301567135,
                47.4204710011512,
                51.3333118455889,
                55.2461526900266,
            ]
        )
        eigs_test = circ.eigenvals()
        assert np.allclose(eigs_test, eigs_ref)

    @staticmethod
    def test_noisy_circuit(num_cpus):
        yaml_inp = f"""branches:
        - [JJ, 1, 2, EJ=6.8, 1]
        - [L, 1, 2, 0.2]
        """
        circ = scq.Circuit(
            yaml_inp,
            from_file=False,
            ext_basis="harmonic",
            use_dynamic_flux_grouping=True,
        )
        circ.cutoff_ext_1 = 300
        circ.Φ1 = 0.5
        circ.configure(generate_noise_methods=True)
        coherence_times = np.array([circ.t1_effective(), circ.t2_effective()])
        ref = np.array([3319890.8160632304, 5385675.324726781])
        assert np.allclose(coherence_times, ref)

    @staticmethod
    def test_get_spectrum_vs_paramvals(num_cpus):
        DFC = scq.Circuit(
            DATADIR + "circuit_DFC.yaml",
            ext_basis="discretized",
            initiate_sym_calc=False,
            basis_completion="canonical",
        )

        closure_branches = [DFC.branches[0], DFC.branches[4], DFC.branches[-1]]
        system_hierarchy = [[[1], [3]], [2], [4]]
        subsystem_trunc_dims = [[34, [6, 6]], 6, 6]

        DFC.configure(
            closure_branches=closure_branches,
            system_hierarchy=system_hierarchy,
            subsystem_trunc_dims=subsystem_trunc_dims,
        )

        DFC.Φ1 = 0.5 + 0.01768
        DFC.Φ2 = -0.2662
        DFC.Φ3 = -0.5 + 0.01768

        DFC.cutoff_ext_1 = 110
        DFC.cutoff_ext_2 = 110
        DFC.cutoff_ext_3 = 110
        DFC.cutoff_ext_4 = 110
        DFC.update()
        DFC.get_spectrum_vs_paramvals("Φ1", np.linspace(0, 1, 11), num_cpus=num_cpus)

        paramvals_by_name = {
            "Φ1": np.linspace(0.4, 0.5, 6),
            "Φ2": np.linspace(0.4, 0.5, 3),
        }

        # update Hilbert space function
        def update_hilbertspace(Φ1, Φ2):
            DFC.Φ1 = Φ1
            DFC.Φ2 = Φ2
            DFC.update()

        ps = scq.ParameterSweep(
            hilbertspace=DFC.hilbert_space,
            paramvals_by_name=paramvals_by_name,
            update_hilbertspace=update_hilbertspace,
            evals_count=6,
            num_cpus=num_cpus,
        )
