"""Numerical and plotting checks for physical wavefunction sections."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sm

import scqubits as scq

MODES = ("abs-sqr", "abs", "real", "imag")

ZERO_PI = """branches:
- [JJ, 1, 2, EJ=10, 20]
- [JJ, 3, 4, EJ, 20]
- [L, 2, 3, 0.008]
- [L, 4, 1, 0.008]
- [C, 1, 3, 0.02]
- [C, 2, 4, 0.02]
"""


def _mode_values(amplitudes, mode):
    return {
        "abs-sqr": np.abs(amplitudes) ** 2,
        "abs": np.abs(amplitudes),
        "real": np.real(amplitudes),
        "imag": np.imag(amplitudes),
    }[mode]


@pytest.fixture(scope="module")
def charge_state():
    hamiltonian = sm.sympify("4*n1**2 + 5*n2**2 - cos(θ1) - 0.8*cos(θ2) + 0.2*n1*n2")
    circuit = scq.Circuit(symbolic_hamiltonian=hamiltonian)
    circuit.cutoff_n_1 = 1
    circuit.cutoff_n_2 = 2
    # The two normalized, orthogonal states have a nontrivial relative phase.
    amplitudes = np.zeros((3, 5), dtype=complex)
    amplitudes[0, 1] = 1 / np.sqrt(2)
    amplitudes[2, 3] = 1j / np.sqrt(2)
    states = np.column_stack((amplitudes.ravel(), amplitudes.conj().ravel()))
    return circuit, (np.array([0.0, 1.0]), states), amplitudes


@pytest.fixture(params=["harmonic", "discretized"])
def zero_pi(request):
    circuit = scq.Circuit.from_yaml_string(ZERO_PI, ext_basis=request.param)
    circuit.cutoff_n_1 = 2
    circuit.cutoff_ext_2 = 7
    circuit.cutoff_ext_3 = 9
    circuit.set_discretized_phi_range(var_indices=(2, 3), phi_range=(-2, 2))
    return circuit


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("charge_value,column", [(-1.2, 1), (0, 2), (0.8, 3)])
def test_charge_section_preserves_amplitudes(charge_state, mode, charge_value, column):
    circuit, esys, amplitudes = charge_state
    actual = circuit.generate_wf_plot_data(
        var_indices=(1,),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        section={2: charge_value},
        mode=mode,
    )
    np.testing.assert_allclose(actual, _mode_values(amplitudes[:, column], mode))


def test_section_preserves_probability_weight_and_zero_sections(charge_state):
    circuit, esys, _ = charge_state
    selected = circuit.generate_wf_plot_data(
        var_indices=(1,),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        section={2: -1},
    )
    assert selected.sum() == pytest.approx(0.5)
    zero = circuit.generate_wf_plot_data(
        var_indices=(1,),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        section={2: 0},
    )
    np.testing.assert_array_equal(zero, np.zeros(3))


def test_section_selects_requested_state(charge_state):
    circuit, esys, amplitudes = charge_state
    actual = circuit.generate_wf_plot_data(
        which=1,
        mode="imag",
        var_indices=(1,),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        section={2: 1},
    )
    np.testing.assert_allclose(actual, amplitudes.conj()[:, 3].imag)


@pytest.mark.parametrize("mode", MODES)
def test_flux_section_matches_analytic_fourier_transform(charge_state, mode):
    circuit, esys, _ = charge_state
    grid = scq.Grid1d(-np.pi, np.pi, 9)
    theta = grid.make_linspace()
    # 1.7 selects pi/2 on the fixed axis's five-point grid.
    fixed_theta = np.pi / 2
    expected = (
        np.exp(-1j * (theta + fixed_theta)) + 1j * np.exp(1j * (theta + fixed_theta))
    ) / np.sqrt(2)
    actual = circuit.generate_wf_plot_data(
        mode=mode,
        var_indices=(1,),
        eigensys=esys,
        section={2: 1.7},
        grids_dict={1: grid, 2: scq.Grid1d(-np.pi, np.pi, 5)},
    )
    np.testing.assert_allclose(actual, _mode_values(expected, mode), atol=1e-14)


def test_partial_custom_grids_and_numpy_scalars(charge_state):
    circuit, esys, _ = charge_state
    grid = scq.Grid1d(-1, 1, 7)
    custom = {1: grid}
    actual = circuit.generate_wf_plot_data(
        var_indices=(np.int64(1),),
        eigensys=esys,
        section={np.int64(2): np.float64(0)},
        grids_dict=custom,
    )
    theta = grid.make_linspace()
    fixed_grid = circuit.discretized_grids_dict_for_vars()[2].make_linspace()
    fixed_theta = fixed_grid[np.argmin(np.abs(fixed_grid))]
    expected = (
        np.abs(
            (
                np.exp(-1j * (theta + fixed_theta))
                + 1j * np.exp(1j * (theta + fixed_theta))
            )
            / np.sqrt(2)
        )
        ** 2
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert custom == {1: grid}


def test_empty_section_returns_all_axes_in_requested_order(charge_state):
    circuit, esys, amplitudes = charge_state
    actual = circuit.generate_wf_plot_data(
        var_indices=(2, 1),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        section={},
    )
    np.testing.assert_allclose(actual, np.abs(amplitudes.T) ** 2)


def test_default_still_returns_marginal_distribution(charge_state):
    circuit, esys, amplitudes = charge_state
    kwargs = dict(var_indices=(1,), eigensys=esys, change_discrete_charge_to_phi=False)
    expected = np.sum(np.abs(amplitudes) ** 2, axis=1)
    np.testing.assert_allclose(circuit.generate_wf_plot_data(**kwargs), expected)
    np.testing.assert_allclose(
        circuit.generate_wf_plot_data(**kwargs, section=None), expected
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("hierarchical", [False, True])
def test_mixed_basis_section_matches_full_wavefunction(zero_pi, mode, hierarchical):
    circuit = zero_pi
    if hierarchical:
        circuit.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[4, 3])
    esys = circuit.eigensys(evals_count=2)
    grids = circuit.discretized_grids_dict_for_vars()
    grids[1] = scq.Grid1d(-np.pi, np.pi, 11)
    full = circuit.generate_wf_plot_data(
        var_indices=(1, 2, 3), eigensys=esys, grids_dict=grids, mode=mode
    )
    # The explicit fixture hierarchy orders its tensor axes as (1, 3, 2).
    if hierarchical:
        full = full.transpose(0, 2, 1)
    sample = np.argmin(np.abs(grids[1].make_linspace() - 0.6))
    actual = circuit.generate_wf_plot_data(
        var_indices=(2, 3),
        eigensys=esys,
        grids_dict=grids,
        mode=mode,
        section={1: 0.6},
    )
    assert actual.shape == (7, 9)
    np.testing.assert_allclose(actual, full[sample], atol=1e-12)


def test_discretized_section_uses_diagonalization_grid(zero_pi):
    if zero_pi.ext_basis != "discretized":
        pytest.skip("Only a discretized basis has a fixed diagonalization grid")
    circuit = zero_pi
    esys = circuit.eigensys(evals_count=2)
    grids = circuit.discretized_grids_dict_for_vars()
    grids[1] = scq.Grid1d(-np.pi, np.pi, 11)
    full = circuit.generate_wf_plot_data(
        var_indices=(1, 2, 3), eigensys=esys, grids_dict=grids
    )
    sample = np.argmin(np.abs(grids[2].make_linspace() - 0.7))
    incompatible_grid = scq.Grid1d(50, 60, 4)
    custom = {**grids, 2: incompatible_grid}
    actual = circuit.generate_wf_plot_data(
        var_indices=(1, 3),
        eigensys=esys,
        grids_dict=custom,
        section={2: 0.7},
    )
    np.testing.assert_allclose(actual, full[:, sample, :])
    assert custom[2] is incompatible_grid


@pytest.mark.parametrize("mode", MODES)
def test_charge_section_plot_data_and_label(charge_state, mode):
    circuit, esys, amplitudes = charge_state
    fig, axes = circuit.plot_wavefunction(
        var_indices=(1,),
        esys=esys,
        mode=mode,
        change_discrete_charge_to_phi=False,
        section={2: 1},
    )
    try:
        centers = [bar.get_x() + bar.get_width() / 2 for bar in axes.patches]
        heights = [bar.get_height() for bar in axes.patches]
        np.testing.assert_array_equal(centers, [-1, 0, 1])
        np.testing.assert_allclose(heights, _mode_values(amplitudes[:, 3], mode))
        label = axes.get_ylabel()
        assert label != r"$\psi_j(n)$"
        if mode == "abs-sqr":
            assert "^2" in label
        else:
            assert "^2" not in label
    finally:
        plt.close(fig)


def test_2d_section_plot_keeps_axis_labels_and_image_order(zero_pi):
    circuit = zero_pi
    circuit.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[4, 3])
    esys = circuit.eigensys(evals_count=2)
    grids = circuit.discretized_grids_dict_for_vars()
    grids[1] = scq.Grid1d(-np.pi, np.pi, 11)
    expected = circuit.generate_wf_plot_data(
        var_indices=(2, 3),
        eigensys=esys,
        grids_dict=grids,
        section={1: 0},
    )
    fig, axes = circuit.plot_wavefunction(
        var_indices=(3, 2), esys=esys, grids_dict=grids, section={1: 0}
    )
    try:
        np.testing.assert_allclose(axes.images[0].get_array(), expected)
        assert axes.get_xlabel() == r"$\theta_{3}$"
        assert axes.get_ylabel() == r"$\theta_{2}$"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "axes,section",
    [
        ((1,), {}),
        ((1,), {1: 0}),
        ((1,), {2: 0, 3: 0}),
        ((1,), {2: np.nan}),
        ((1,), {2: np.inf}),
        ((1,), {2: -np.inf}),
        ((1,), {2: 3}),
        ((1,), {2: -3}),
        ((1,), {2: "0"}),
        ((1,), {2: 1j}),
        ((1,), {2: [0]}),
        ((1,), {2: True}),
        ((2,), {True: 0}),
        ((1,), {"2": 0}),
        ((1,), [0]),
        ((1,), 0),
        ((), {1: 0, 2: 0}),
        ((1, 1), {2: 0}),
        ((3,), {1: 0, 2: 0}),
        ((True,), {2: 0}),
        ((1, 2, 3), {}),
    ],
)
def test_invalid_section_fails_before_diagonalization(
    charge_state, monkeypatch, axes, section
):
    circuit, _, _ = charge_state

    def unexpected_eigensys(*args, **kwargs):
        pytest.fail("Invalid section must be rejected before diagonalization")

    monkeypatch.setattr(type(circuit), "eigensys", unexpected_eigensys)
    with pytest.raises(ValueError):
        circuit.generate_wf_plot_data(
            var_indices=axes,
            change_discrete_charge_to_phi=False,
            section=section,
        )


def test_partial_section_does_not_silently_marginalize(zero_pi):
    with pytest.raises(ValueError):
        zero_pi.generate_wf_plot_data(var_indices=(1,), section={2: 0})


def test_nested_hierarchy_matches_explicit_tensor_basis_change():
    hamiltonian = sm.sympify(
        "4*n1**2 + 5*n2**2 + 6*n3**2 + 7*n4**2"
        " - cos(θ1) - 0.8*cos(θ2) - 0.7*cos(θ3) - 0.6*cos(θ4)"
        " + 0.1*n1*n2 + 0.2*n2*n3 + 0.15*n3*n4"
    )
    circuit = scq.Circuit(symbolic_hamiltonian=hamiltonian)
    for var, cutoff in {1: 1, 2: 2, 3: 2, 4: 2}.items():
        setattr(circuit, f"cutoff_n_{var}", cutoff)
    circuit.configure(
        system_hierarchy=[[[1, 2], [3]], [4]],
        subsystem_trunc_dims=[[5, [4, 2]], 2],
    )
    esys = circuit.eigensys(evals_count=2)
    left, right = circuit.subsystems
    first, second = left.subsystems
    u_first = first.eigensys(evals_count=first.truncated_dim)[1]
    u_second = second.eigensys(evals_count=second.truncated_dim)[1]
    u_left = left.eigensys(evals_count=left.truncated_dim)[1]
    u_right = right.eigensys(evals_count=right.truncated_dim)[1]
    # Explicit Kronecker products are independent of the recursive plotting code.
    transform = np.kron(np.kron(u_first, u_second) @ u_left, u_right)
    full = (transform @ esys[1][:, 0]).reshape(3, 5, 5, 5)
    actual = circuit.generate_wf_plot_data(
        var_indices=(4, 1),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        section={2: 0, 3: -1},
    )
    np.testing.assert_allclose(actual, np.abs(full[:, 2, 1, :].T) ** 2, atol=1e-12)


def test_fixed_flux_axes_do_not_expand_to_full_grids(monkeypatch):
    hamiltonian = sm.sympify(
        " + ".join(f"{var + 3}*n{var}**2 - cos(θ{var})" for var in range(1, 5))
    )
    circuit = scq.Circuit(symbolic_hamiltonian=hamiltonian)
    for var in range(1, 5):
        setattr(circuit, f"cutoff_n_{var}", 1)
    # A single charge-basis product state has an analytic constant flux density.
    amplitudes = np.zeros((3, 3, 3, 3), dtype=complex)
    amplitudes[2, 1, 0, 2] = 1
    esys = (np.array([0.0]), amplitudes.reshape(-1, 1))
    grids = {var: scq.Grid1d(-np.pi, np.pi, 101) for var in range(1, 5)}
    original = type(circuit)._basis_change_n_to_phi
    intermediate_sizes = []

    def bounded_transform(self, wf, axis, var, grid):
        size = wf.size // wf.shape[axis] * grid.pt_count
        # Fail before allocation if fixed axes start producing a large tensor.
        assert size <= 101 * amplitudes.size
        result = original(self, wf, axis, var, grid)
        intermediate_sizes.append(result.size)
        return result

    monkeypatch.setattr(type(circuit), "_basis_change_n_to_phi", bounded_transform)
    actual = circuit.generate_wf_plot_data(
        var_indices=(1,),
        eigensys=esys,
        grids_dict=grids,
        section={2: 0, 3: 0.4, 4: -0.7},
    )
    np.testing.assert_allclose(actual, np.ones(101), atol=1e-14)
    assert max(intermediate_sizes) <= max(amplitudes.size, actual.size)


@pytest.mark.parametrize("use_leaf", [False, True])
def test_reordered_hierarchy_leaf_uses_native_axis_order(use_leaf):
    from scqubits.core.oscillator import harm_osc_wavefunction

    hamiltonian = sm.sympify("4*n1**2 - cos(θ1) + 2*Q2**2 + 0.5*θ2**2")
    circuit = scq.Circuit(symbolic_hamiltonian=hamiltonian, ext_basis="harmonic")
    circuit.cutoff_n_1 = 1
    circuit.cutoff_ext_2 = 4
    circuit.configure(system_hierarchy=[[2, 1]], subsystem_trunc_dims=[3])
    leaf = circuit.subsystems[0]
    leaf_eigenvalues, leaf_eigenvectors = leaf.eigensys(evals_count=3)
    grid = scq.Grid1d(-1, 1, 5)
    selected_phi = grid.make_linspace()[3]
    if use_leaf:
        target = leaf
        esys = (leaf_eigenvalues, leaf_eigenvectors)
        native = leaf_eigenvectors[:, 0].reshape(4, 3)
    else:
        target = circuit
        esys = circuit.eigensys(evals_count=1)
        native = (leaf_eigenvectors @ esys[1][:, 0]).reshape(4, 3)
    basis_values = np.array(
        [
            harm_osc_wavefunction(
                n, np.array([selected_phi]), abs(leaf.get_osc_param(2, "length"))
            )[0]
            for n in range(4)
        ]
    )
    expected = np.abs(basis_values @ native) ** 2
    actual = target.generate_wf_plot_data(
        var_indices=(1,),
        eigensys=esys,
        change_discrete_charge_to_phi=False,
        grids_dict={2: grid},
        section={2: 0.6},
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
