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
import qutip as qt

import scqubits as scq

from scqubits.io_utils.fileio import read

TESTDIR, _ = os.path.split(scq.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")
DATADIR = os.path.join(TESTDIR, "data", "")

# Tests below exercise the legacy ``from_file`` flag and ``Circuit.from_yaml``.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


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
        REFERENCE = "<bound method Printable.__str__ of EJ*cos(θ1 + θ3) + EJ*cos(1.0*(2πΦ_{1}) + θ1 - 1.0*θ3) + 6.25625*\\dot{θ_1}**2 + 25.0*\\dot{θ_2}**2 + 0.00625*\\dot{θ_3}**2 - 0.036*θ2**2 - 0.004*θ2*θ3 - 0.009*θ3**2>"

        zero_pi = scq.Circuit(zp_yaml, from_file=False, ext_basis="discretized")
        expression_str = str(
            zero_pi.sym_lagrangian(vars_type="new", return_expr=True).doit().__repr__
        )
        assert expression_str == REFERENCE

    @staticmethod
    def test_zero_pi_discretized():
        """Test for symmetric zero-pi in discretized phi basis."""
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
        circ_d.cutoff_ext_3 = 200
        circ_d.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[30, 20])

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
        """Test for initiating Circuit module with symbolic Hamiltonian."""
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
    def test_symbolic_hamiltonian_rejects_transformation_matrix():
        """``transformation_matrix=`` is unsupported on the symbolic-Hamiltonian path.

        Regression test for the prior silent-ignore bug: ``Circuit.configure``
        previously accepted ``transformation_matrix=`` without raising and
        without applying it. The fix raises ``ValueError`` instead.
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
        with pytest.raises(ValueError, match="transformation_matrix"):
            circ.configure(
                transformation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
            )

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
    def test_circuit_parametersweep(num_cpus):
        DFC = scq.Circuit(
            DATADIR + "circuit_DFC.yaml",
            ext_basis="discretized",
            initiate_sym_calc=False,
            basis_completion="canonical",
        )

        closure_branches = [DFC.branches[0], DFC.branches[4], DFC.branches[-1]]
        system_hierarchy = [[[1], [3]], [2], [4]]
        subsystem_trunc_dims = [[6, [6, 6]], 6, 6]

        DFC.configure(
            closure_branches=closure_branches,
            system_hierarchy=system_hierarchy,
            subsystem_trunc_dims=subsystem_trunc_dims,
        )

        DFC.Φ1 = 0.5 + 0.01768
        DFC.Φ2 = -0.2662
        DFC.Φ3 = -0.5 + 0.01768

        DFC.cutoff_ext_1 = 40
        DFC.cutoff_ext_2 = 40
        DFC.cutoff_ext_3 = 40
        DFC.cutoff_ext_4 = 40
        DFC.get_spectrum_vs_paramvals("Φ1", np.linspace(0, 1, 3), num_cpus=num_cpus)

        paramvals_by_name = {
            "Φ1": np.linspace(0.4, 0.5, 6),
            "Φ2": np.linspace(0.4, 0.5, 3),
        }

        # update Hilbert space function
        def update_hilbertspace(Φ1, Φ2):
            DFC.Φ1 = Φ1
            DFC.Φ2 = Φ2

        ps = scq.ParameterSweep(
            hilbertspace=DFC.hilbert_space,
            paramvals_by_name=paramvals_by_name,
            update_hilbertspace=update_hilbertspace,
            evals_count=6,
            num_cpus=num_cpus,
        )
        ps.run()

    @staticmethod
    def test_qutip_dynamics(num_cpus):
        # Let's start by defining a fluxonium
        inp_yaml = """
        branches:
        - [JJ, 1, 2, 4, 0.5]
        - [L, 1, 2, 1.3]
        - [C, 1, 2, 2]
        """
        circ = scq.Circuit(
            inp_yaml,
            from_file=False,
            use_dynamic_flux_grouping=True,
            ext_basis="discretized",
        )
        circ.cutoff_ext_1 = 100
        circ.Φ1 = 0.5

        # defining Hierarchical diagonalization to limit to the lowest two states
        circ.configure(system_hierarchy=[[1]], subsystem_trunc_dims=[10])

        # Define time dependent functions for the parameters
        def flux(t, args):
            freq = args["freq"]
            return 0.001 * np.sin(2 * np.pi * freq * t) + 0.5

        # to charge drive the fluxonium, we need an extra parameter ng1. This can be added using extra_terms
        def charge(t, args):
            freq = args["freq"]
            return 0.02 * np.sin(2 * np.pi * freq * t + np.pi / 2)

        # Generating necessary operators and time dependent coefficients
        H_mesolve, *H_sym_ref = circ.hamiltonian_for_qutip_dynamics(
            free_var_func_dict={"Φ1": flux, "ng1": charge},
            extra_terms="Q1*ng1",
            prefactor=np.pi * 2,
        )
        # H_mesolve can be used to evolve the system using qutip functions like mesolve

        # ground state as initial state
        eigs, evecs = circ.eigensys(evals_count=5)
        wf0 = qt.Qobj(evecs[:, 0])

        initial_state_proj = wf0 * wf0.dag()  # to see the overlap
        tf = 100  # final time in nanoseconds
        freq = eigs[1] - eigs[0]  # transition frequency between the first two states

        # time evolve the system
        result = qt.mesolve(
            H_mesolve,
            wf0,
            np.linspace(0, tf, 500),
            args={"freq": freq},
            e_ops=[initial_state_proj],
            options=dict(atol=1e-12),
        )
        expectation_vals = result.expect[0]
        ref_expectation_vals = np.empty_like(expectation_vals)
        ref_expectation_vals[:] = read(DATADIR + "/circuit_qutip_evolution_data.hdf5")[
            :
        ]
        assert np.allclose(
            expectation_vals,
            ref_expectation_vals,
        )


class TestConfigureError:
    """Pin behavior of ``Circuit.configure`` on invalid input.

    On failure, the prior configuration is restored and a
    :class:`~scqubits.core.circuit.ConfigureError` is raised with the
    triggering exception preserved as ``__cause__``.
    """

    @staticmethod
    def _make_zero_pi():
        zp_yaml = """branches:
        - ["JJ", 1, 2, 10, 20]
        - ["JJ", 3, 4, 10, 20]
        - ["L", 2, 3, 0.008]
        - ["L", 4, 1, 0.008]
        - ["C", 1, 3, 0.02]
        - ["C", 2, 4, 0.02]
        """
        return scq.Circuit(zp_yaml, from_file=False, ext_basis="discretized")

    def test_invalid_hierarchy_raises_configure_error(self):
        from scqubits.core.circuit import ConfigureError

        circ = self._make_zero_pi()
        # system_hierarchy without subsystem_trunc_dims triggers the
        # internal Exception in _configure.
        with pytest.raises(ConfigureError) as excinfo:
            circ.configure(system_hierarchy=[[1, 3], [2]])
        assert excinfo.value.__cause__ is not None

    def test_prior_configuration_is_restored_on_failure(self):
        from scqubits.core.circuit import ConfigureError

        circ = self._make_zero_pi()
        prior_hierarchy = circ.system_hierarchy
        prior_trunc_dims = circ.subsystem_trunc_dims
        prior_transformation_matrix = circ.transformation_matrix.copy()
        prior_closure_branches = list(circ.closure_branches)
        prior_ext_basis = circ.ext_basis
        with pytest.raises(ConfigureError):
            circ.configure(system_hierarchy=[[1, 3], [2]])
        assert circ.system_hierarchy == prior_hierarchy
        assert circ.subsystem_trunc_dims == prior_trunc_dims
        assert np.array_equal(circ.transformation_matrix, prior_transformation_matrix)
        assert list(circ.closure_branches) == prior_closure_branches
        assert circ.ext_basis == prior_ext_basis


class TestClearUnnecessaryAttribs:
    """``_clear_unnecessary_attribs`` consults a registry, not name patterns.

    The previous implementation matched attribute names by string
    substring (``"cutoff_n_" in attrib`` etc.); a future per-variable
    property kind whose name didn't match those patterns would have
    silently leaked across reconfigurations. The registry-driven
    implementation drops every name that ``_install_var_properties``
    has installed but the current configuration no longer needs.
    """

    @staticmethod
    def _make_zero_pi():
        zp_yaml = """branches:
        - ["JJ", 1, 2, 10, 20]
        - ["JJ", 3, 4, 10, 20]
        - ["L", 2, 3, 0.008]
        - ["L", 4, 1, 0.008]
        - ["C", 1, 3, 0.02]
        - ["C", 2, 4, 0.02]
        """
        return scq.Circuit(zp_yaml, from_file=False, ext_basis="discretized")

    def test_registry_clears_arbitrary_dynamic_attribute(self):
        """A name in ``_dynamic_var_attribs`` but not in the necessary set is cleared."""
        circ = self._make_zero_pi()
        # Inject a name that does not match any of the legacy string
        # patterns (``"cutoff_n_"``, ``"Φ"``, ``"cutoff_ext_"``,
        # ``[1:3] == "ng"``). The legacy implementation would not have
        # cleared this; the registry approach must.
        object.__setattr__(circ, "_custom_kind_99", 42)
        circ._dynamic_var_attribs.add("custom_kind_99")

        circ._clear_unnecessary_attribs()

        assert not hasattr(circ, "_custom_kind_99")
        assert "custom_kind_99" not in circ._dynamic_var_attribs

    def test_registry_preserves_currently_active_names(self):
        """Names listed in ``cutoff_names`` survive a clearing pass."""
        circ = self._make_zero_pi()
        active_cutoffs = list(circ.cutoff_names)
        # Sanity check the precondition: the registry contains every
        # active cutoff name.
        for name in active_cutoffs:
            assert name in circ._dynamic_var_attribs

        circ._clear_unnecessary_attribs()

        for name in active_cutoffs:
            assert hasattr(circ, f"_{name}"), (
                f"active cutoff {name!r} was wrongly cleared"
            )
            assert name in circ._dynamic_var_attribs


class TestRecomputationContract:
    """``SymbolicCircuit._STAGE2_ATTRIBUTES`` is the single source of truth
    for which attributes flow from stage 1 (symbolic) to stage 2 (numerical).
    A fresh ``Circuit`` constructed from YAML must end up with every name in
    that tuple set on the instance.

    Regression net for the §17.3 maintenance pitfall: a contributor adding a
    new attribute to ``SymbolicCircuit`` but forgetting to register it in
    ``_STAGE2_ATTRIBUTES`` would either leave ``Circuit`` blind to it, or
    rely on it via the wrong helper. This test fails loudly on any name in
    the tuple that doesn't make it onto a fresh instance.
    """

    TRANSMON_YAML = (
        "branches:\n"
        "- [JJ, 1, 2, EJ=10, ECJ=20]\n"
        "- [C, 1, 2, EC=2]\n"
    )
    FLUXONIUM_YAML = (
        "branches:\n"
        "- [JJ, 1, 2, EJ=5.7, ECJ=20]\n"
        "- [L, 1, 2, EL=0.39]\n"
        "- [C, 1, 2, EC=10]\n"
    )
    ZERO_PI_YAML = (
        "branches:\n"
        '- ["JJ", 1, 2, 10, 20]\n'
        '- ["JJ", 3, 4, 10, 20]\n'
        '- ["L", 2, 3, 0.008]\n'
        '- ["L", 4, 1, 0.008]\n'
        '- ["C", 1, 3, 0.02]\n'
        '- ["C", 2, 4, 0.02]\n'
    )

    @pytest.mark.parametrize(
        "yaml,name",
        [
            (TRANSMON_YAML, "transmon"),
            (FLUXONIUM_YAML, "fluxonium"),
            (ZERO_PI_YAML, "zero_pi"),
        ],
    )
    def test_every_stage2_attr_is_set_on_fresh_circuit(self, yaml, name):
        circ = scq.Circuit.from_yaml_string(yaml, ext_basis="discretized")
        for attr in circ.symbolic_circuit._STAGE2_ATTRIBUTES:
            assert hasattr(circ, attr), (
                f"{name}: _STAGE2_ATTRIBUTES name {attr!r} was not propagated "
                f"to the numerical Circuit instance"
            )

    @pytest.mark.parametrize(
        "yaml,name",
        [
            (TRANSMON_YAML, "transmon"),
            (FLUXONIUM_YAML, "fluxonium"),
            (ZERO_PI_YAML, "zero_pi"),
        ],
    )
    def test_topology_stage2_attrs_match_after_configure(self, yaml, name):
        """For attributes that ``_configure`` does not further process, the
        value on a fresh ``Circuit`` should remain reference-identical to the
        value on ``self.symbolic_circuit``. (Symbolic expressions like
        ``hamiltonian_symbolic`` / ``potential_symbolic`` / ``lagrangian_*``
        are excluded because ``_configure`` legitimately rewrites them.)"""
        topology_attrs = (
            "branches",
            "closure_branches",
            "ground_node",
            "input_string",
            "is_grounded",
            "is_purely_harmonic",
            "nodes",
            "external_fluxes",
            "offset_charges",
            "free_charges",
            "var_categories",
        )
        circ = scq.Circuit.from_yaml_string(yaml, ext_basis="discretized")
        for attr in topology_attrs:
            assert attr in circ.symbolic_circuit._STAGE2_ATTRIBUTES, (
                f"{attr!r} should be in _STAGE2_ATTRIBUTES"
            )
            circ_val = getattr(circ, attr)
            sym_val = getattr(circ.symbolic_circuit, attr)
            assert circ_val is sym_val or circ_val == sym_val, (
                f"{name}: {attr} value differs between Circuit and symbolic_circuit"
            )


class TestTypedOperatorAccessor:
    """``Circuit.operator(name, *, energy_esys=...)`` is the typed dispatcher
    for the dynamic ``<name>_operator`` methods that ``_set_operators`` binds
    at ``_configure`` time. The dynamic methods are invisible to ``mypy`` /
    IDE / sphinx-autodoc; the typed accessor restores tooling visibility.
    """

    YAML = (
        "branches:\n"
        "- [JJ, 0, 1, EJ=10, ECJ=20]\n"
        "- [C, 0, 1, EC=2]\n"
    )

    def _make_transmon(self):
        circ = scq.Circuit.from_yaml_string(self.YAML, ext_basis="discretized")
        circ.cutoff_n_1 = 10
        return circ

    def test_operator_routes_to_dynamic_method(self):
        circ = self._make_transmon()
        # Both call paths must produce an identical native operator.
        via_typed = circ.operator("n1")
        via_dynamic = circ.n1_operator()
        # Sparse / Qobj equality: compare the dense forms.
        if hasattr(via_typed, "todense"):
            np.testing.assert_array_equal(
                np.asarray(via_typed.todense()),
                np.asarray(via_dynamic.todense()),
            )
        else:
            np.testing.assert_array_equal(np.asarray(via_typed), np.asarray(via_dynamic))

    def test_operator_unknown_name_lists_available(self):
        circ = self._make_transmon()
        with pytest.raises(AttributeError, match="available operators"):
            circ.operator("does_not_exist")

    def test_operator_energy_esys_kwarg_is_passed_through(self):
        """Passing energy_esys=True must rotate the returned operator into
        the energy eigenbasis. Compare to the dynamic method called with the
        same argument to verify the kwarg flows correctly."""
        circ = self._make_transmon()
        via_typed = circ.operator("n1", energy_esys=True)
        via_dynamic = circ.n1_operator(energy_esys=True)
        np.testing.assert_allclose(
            np.asarray(via_typed), np.asarray(via_dynamic), rtol=1e-12
        )


class TestNamedConstructors:
    """`Circuit.from_yaml_file` / `Circuit.from_yaml_string` are named
    alternatives to the legacy ``Circuit(input_string, from_file=...)`` form.
    """

    YAML = """branches:
    - ["JJ", 1, 2, 10, 20]
    - ["JJ", 3, 4, 10, 20]
    - ["L", 2, 3, 0.008]
    - ["L", 4, 1, 0.008]
    - ["C", 1, 3, 0.02]
    - ["C", 2, 4, 0.02]
    """

    def test_from_yaml_string_matches_legacy_form(self):
        legacy = scq.Circuit(self.YAML, from_file=False, ext_basis="discretized")
        new = scq.Circuit.from_yaml_string(self.YAML, ext_basis="discretized")
        assert isinstance(new, scq.Circuit)
        assert new.ext_basis == legacy.ext_basis
        assert new.is_purely_harmonic == legacy.is_purely_harmonic
        assert new.var_categories == legacy.var_categories

    def test_from_yaml_file_matches_legacy_form(self, tmp_path):
        path = tmp_path / "zp.yaml"
        path.write_text(self.YAML)
        legacy = scq.Circuit(str(path), from_file=True, ext_basis="discretized")
        new = scq.Circuit.from_yaml_file(str(path), ext_basis="discretized")
        assert isinstance(new, scq.Circuit)
        assert new.ext_basis == legacy.ext_basis
        assert new.var_categories == legacy.var_categories


class TestMakeBranchNodeIndexOffset:
    """``make_branch`` accepts an explicit ``node_index_offset`` kwarg.

    The legacy code inferred 0-vs-1-based indexing from ``any(n.is_ground()
    for n in nodes_list)``.  Explicit passing decouples ``make_branch`` from
    that whole-list inference; the inference path is preserved as the
    ``None`` default for backward compatibility.
    """

    YAML_GROUNDED = (
        "branches:\n"
        "- [JJ, 0, 1, 10, 20]\n"
        "- [L, 0, 1, 0.01]\n"
        "- [C, 0, 1, 0.02]\n"
    )
    YAML_UNGROUNDED = (
        "branches:\n"
        "- [JJ, 1, 2, 10, 20]\n"
        "- [L, 1, 2, 0.01]\n"
        "- [C, 1, 2, 0.02]\n"
    )

    def test_yaml_grounded_round_trip(self):
        """Branches in a 0-indexed circuit connect the declared node IDs."""
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        circ = SymbolicCircuit.from_yaml(self.YAML_GROUNDED, from_file=False)
        assert {n.index for n in circ.nodes} == {0, 1}
        for branch in circ.branches:
            assert {n.index for n in branch.nodes} == {0, 1}

    def test_yaml_ungrounded_round_trip(self):
        """Branches in a 1-indexed circuit connect the declared node IDs."""
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        circ = SymbolicCircuit.from_yaml(self.YAML_UNGROUNDED, from_file=False)
        assert {n.index for n in circ.nodes} == {1, 2}
        for branch in circ.branches:
            assert {n.index for n in branch.nodes} == {1, 2}


class TestVariableTransformationMatrixClassification:
    """``variable_transformation_matrix`` partitions basis rows into
    sigma / free / periodic / frozen / rest with precedence
    ``sigma > free > periodic > frozen > rest``.  The single-pass
    dict-lookup must produce the same partition as the legacy
    chain-of-exclusions for any basis layout."""

    @staticmethod
    def _legacy_partition(
        new_basis, Σ, free_modes, periodic_modes, frozen_modes, is_grounded
    ):
        """Reproduces the pre-refactor 5-comprehension cascade for cross-check."""
        if not is_grounded:
            pos_Σ = [i for i in range(len(new_basis)) if new_basis[i].tolist() == Σ]
        else:
            pos_Σ = []
        pos_free = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if new_basis[i].tolist() in free_modes
        ]
        pos_periodic = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_free
            if new_basis[i].tolist() in periodic_modes
        ]
        pos_frozen = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_free
            if i not in pos_periodic
            if new_basis[i].tolist() in frozen_modes
        ]
        pos_rest = [
            i
            for i in range(len(new_basis))
            if i not in pos_Σ
            if i not in pos_free
            if i not in pos_periodic
            if i not in pos_frozen
        ]
        return pos_Σ, pos_free, pos_periodic, pos_frozen, pos_rest

    @staticmethod
    def _new_partition(
        new_basis, Σ, free_modes, periodic_modes, frozen_modes, is_grounded
    ):
        """The new single-pass implementation, isolated for testing."""
        mode_to_label: dict[tuple, str] = {}
        for m in frozen_modes:
            mode_to_label[tuple(m)] = "frozen"
        for m in periodic_modes:
            mode_to_label[tuple(m)] = "periodic"
        for m in free_modes:
            mode_to_label[tuple(m)] = "free"
        if not is_grounded:
            mode_to_label[tuple(Σ)] = "sigma"
        buckets = {"sigma": [], "free": [], "periodic": [], "frozen": [], "rest": []}
        for i, row in enumerate(new_basis):
            buckets[mode_to_label.get(tuple(row.tolist()), "rest")].append(i)
        return (
            buckets["sigma"],
            buckets["free"],
            buckets["periodic"],
            buckets["frozen"],
            buckets["rest"],
        )

    @pytest.mark.parametrize("is_grounded", [True, False])
    def test_overlapping_modes_use_correct_precedence(self, is_grounded):
        """A row in both free and frozen lists must classify as free, not frozen.
        A row equal to Σ (when not grounded) must classify as sigma even if it
        also appears in free / periodic / frozen."""
        new_basis = np.array(
            [
                [1, 0, 0, 0],  # only in frozen — frozen
                [0, 1, 0, 0],  # in free AND frozen — should be free (free > frozen)
                [0, 0, 1, 0],  # only in periodic — periodic
                [1, 1, 1, 1],  # equals Σ when not grounded; also in free
                [0, 0, 0, 1],  # not in any — rest
            ]
        )
        Σ = [1, 1, 1, 1]
        free_modes = [[0, 1, 0, 0], [1, 1, 1, 1]]
        periodic_modes = [[0, 0, 1, 0]]
        frozen_modes = [[1, 0, 0, 0], [0, 1, 0, 0]]
        legacy = self._legacy_partition(
            new_basis, Σ, free_modes, periodic_modes, frozen_modes, is_grounded
        )
        new = self._new_partition(
            new_basis, Σ, free_modes, periodic_modes, frozen_modes, is_grounded
        )
        assert legacy == new

    def test_matches_legacy_on_random_layouts(self):
        """Across many random row layouts, the new partition must equal the legacy one."""
        import random

        rng = random.Random(42)
        for _ in range(50):
            n_cols = rng.choice([3, 4, 5, 6])
            n_rows = n_cols  # square basis
            rows = []
            for _ in range(n_rows):
                rows.append([rng.choice([-1, 0, 1]) for _ in range(n_cols)])
            new_basis = np.array(rows)
            Σ = [1] * n_cols
            # Build random mode lists by sampling rows
            free_modes = [
                rows[i] for i in rng.sample(range(n_rows), k=rng.randint(0, 2))
            ]
            periodic_modes = [
                rows[i] for i in rng.sample(range(n_rows), k=rng.randint(0, 2))
            ]
            frozen_modes = [
                rows[i] for i in rng.sample(range(n_rows), k=rng.randint(0, 2))
            ]
            for is_grounded in (True, False):
                legacy = self._legacy_partition(
                    new_basis, Σ, free_modes, periodic_modes, frozen_modes, is_grounded
                )
                new = self._new_partition(
                    new_basis, Σ, free_modes, periodic_modes, frozen_modes, is_grounded
                )
                assert legacy == new, (
                    f"mismatch: rows={rows}, free={free_modes}, "
                    f"periodic={periodic_modes}, frozen={frozen_modes}, "
                    f"is_grounded={is_grounded}"
                )


class TestFindPathToRootDFSRewrite:
    """``_find_path_to_root`` is now O(depth) DFS via ``_AdjacencyIndex``,
    not O(tree_size!) brute-force permutation enumeration.  The
    spanning-tree path is unique (a tree has no cycles), so the DFS
    answer is identical to what the legacy permutation search would
    have returned."""

    YAML = (
        "branches:\n"
        "- [JJ, 1, 2, 10, 20]\n"
        "- [JJ, 3, 4, 10, 20]\n"
        "- [L, 2, 3, 0.008]\n"
        "- [L, 4, 1, 0.008]\n"
        "- [C, 1, 3, 0.02]\n"
        "- [C, 2, 4, 0.02]\n"
    )

    def test_path_to_root_for_root_returns_empty(self):
        """The root node is generation 0 with empty ancestor / branch lists."""
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        circ = SymbolicCircuit.from_yaml(self.YAML, from_file=False)
        node_sets = circ.spanning_tree_dict["node_sets_for_trees"]
        root = node_sets[0][0][0]
        gen, ancestors, branches, tree_idx = circ._find_path_to_root(
            root, circ.spanning_tree_dict
        )
        assert gen == 0
        assert ancestors == []
        assert branches == []
        assert tree_idx == 0

    def test_path_to_root_for_leaf_includes_all_intermediate(self):
        """For a leaf node, the path includes every ancestor up to the root,
        in root-to-leaf order."""
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        circ = SymbolicCircuit.from_yaml(self.YAML, from_file=False)
        # zero-pi has 4 nodes; pick the one in the deepest layer
        node_sets = circ.spanning_tree_dict["node_sets_for_trees"][0]
        deepest = node_sets[-1][0]
        gen, ancestors, branches, _ = circ._find_path_to_root(
            deepest, circ.spanning_tree_dict
        )
        assert gen == len(node_sets) - 1
        # ancestor list ends just before the leaf, length = generation
        assert len(ancestors) == gen
        assert len(branches) == gen

    def test_path_to_root_raises_for_unknown_node(self):
        """A node not in any spanning tree raises ValueError (the legacy
        permutation code silently returned stale data)."""
        from scqubits.core.symbolic_circuit import SymbolicCircuit
        from scqubits.core.symbolic_circuit_graph import Node

        circ = SymbolicCircuit.from_yaml(self.YAML, from_file=False)
        bogus = Node(99)
        with pytest.raises(ValueError, match="not present in any spanning tree"):
            circ._find_path_to_root(bogus, circ.spanning_tree_dict)


class TestAdjacencyIndexCaching:
    """``_adjacency_for_spanning_tree_dict`` reuses a built ``_AdjacencyIndex``
    across repeated calls with the same spanning-tree dict — important when
    ``_find_loop`` is invoked many times (e.g. once per closure branch in
    ``_time_dependent_flux_distribution``)."""

    YAML = (
        "branches:\n"
        "- [JJ, 1, 2, 10, 20]\n"
        "- [JJ, 3, 4, 10, 20]\n"
        "- [L, 2, 3, 0.008]\n"
        "- [L, 4, 1, 0.008]\n"
        "- [C, 1, 3, 0.02]\n"
        "- [C, 2, 4, 0.02]\n"
    )

    def test_cache_reused_for_same_dict(self):
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        circ = SymbolicCircuit.from_yaml(self.YAML, from_file=False)
        idx_1 = circ._adjacency_for_spanning_tree_dict(circ.spanning_tree_dict)
        idx_2 = circ._adjacency_for_spanning_tree_dict(circ.spanning_tree_dict)
        assert idx_1 is idx_2, "cache returned a freshly-built index"

    def test_cache_rebuilds_for_different_dict(self):
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        circ = SymbolicCircuit.from_yaml(self.YAML, from_file=False)
        idx_1 = circ._adjacency_for_spanning_tree_dict(circ.spanning_tree_dict)
        # build a fresh dict (same content, different object identity)
        fresh = circ._spanning_tree()
        idx_2 = circ._adjacency_for_spanning_tree_dict(fresh)
        assert idx_1 is not idx_2, "cache failed to rebuild for new dict"


class TestHeuristicBasisCompletionPerformance:
    """The heuristic basis-completion in
    ``_complete_basis_with_standard_vectors`` previously rank-tested every
    permutation produced by ``itertools.permutations(vector_ref, n)``,
    most of which were duplicates of vectors already rejected.  For a
    10-node circuit this meant ~564 000 SVDs and ~26 s of construction.
    The deduped variant rank-tests each distinct candidate at most
    once."""

    @staticmethod
    def _ten_node_chain_yaml() -> str:
        """Linear chain of 10 nodes connected by capacitors and a JJ at one end.

        Constructed to be small enough to parse instantly but to land in
        the ``basis_completion="heuristic"`` path with a non-trivial
        ``vector_ref`` (8 ones, 2 zeros).
        """
        lines = ["branches:", "- [JJ, 0, 1, 10, 20]"]
        for i in range(1, 10):
            lines.append(f"- [C, {i}, {i + 1}, 0.02]")
        return "\n".join(lines) + "\n"

    def test_construction_completes_quickly(self):
        """A 10-node circuit must build in well under 5 s.

        Pre-fix: > 20 s on the user's machine (~564k SVDs).
        Post-fix: well under 1 s (~45 SVDs in the heuristic loop)."""
        import time

        from scqubits.core.symbolic_circuit import SymbolicCircuit

        yaml = self._ten_node_chain_yaml()
        start = time.perf_counter()
        SymbolicCircuit.from_yaml(yaml, from_file=False)
        elapsed = time.perf_counter() - start
        # Generous 5 s budget — the post-fix code on a slow CI runner
        # should still come in well under 1 s.
        assert elapsed < 5.0, (
            f"SymbolicCircuit construction for the 10-node chain took "
            f"{elapsed:.2f} s; expected < 1 s post-fix.  Pre-fix this "
            f"was ~26 s due to O(n!) permutation rank-tests."
        )

    def test_dedupe_does_not_change_basis_for_simple_circuit(self):
        """Belt-and-braces: two SymbolicCircuit instances built from the
        same YAML must agree element-wise on
        ``transformation_matrix``.  Different runs of the deduped loop
        on the same input must be deterministic."""
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        yaml = self._ten_node_chain_yaml()
        circ_a = SymbolicCircuit.from_yaml(yaml, from_file=False)
        circ_b = SymbolicCircuit.from_yaml(yaml, from_file=False)
        np.testing.assert_array_equal(
            circ_a.transformation_matrix, circ_b.transformation_matrix
        )


class TestSymbolicCircuitFromYamlResourceHandling:
    """``SymbolicCircuit.from_yaml(file_path, from_file=True)`` must release
    the file handle even if a parse error fires below the read.  The
    pre-refactor code used bare ``open()``/``close()`` and could leak."""

    BAD_YAML = (
        "branches:\n" "- [JJ, 1, 2, 10, 20]\n" "- [BAD_BRANCH_TYPE, 1, 2, 0.01]\n"
    )

    def test_file_handle_released_on_parse_error(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(self.BAD_YAML)
        from scqubits.core.symbolic_circuit import SymbolicCircuit

        with pytest.raises(Exception):
            SymbolicCircuit.from_yaml(str(path), from_file=True)
        # If the handle leaks (Windows-specific), unlink will fail with
        # PermissionError. The ``with`` block in from_yaml prevents that.
        path.unlink()
        assert not path.exists()
