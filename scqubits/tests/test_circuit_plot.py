from scqubits.core.circuit import example_circuit, truncation_template
import scqubits as scq
import numpy as np

import os

TESTDIR, _ = os.path.split(scq.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")
DATADIR = os.path.join(TESTDIR, "data", "")


circ = scq.Circuit.from_yaml(
    DATADIR + "circuit_zeropi.yaml", ext_basis="harmonic", initiate_sym_calc=False
)
system_hierarchy = [[1, 3], [2]]

circ.configure(
    transformation_matrix=None,
    system_hierarchy=system_hierarchy,
    subsystem_trunc_dims=[100, 30],
)

circ.EJ = 10
circ.Φ1 = 0.0
circ.ng1 = 0.6
# circ.subsystem_truncated_dims["sys_2"] = 10
circ.cutoff_n_1 = 10
circ.cutoff_ext_2 = 40
circ.cutoff_ext_3 = 50


def test_plot_wf():
    circ.plot_wavefunction(which=0, var_indices=(2, 3))


def test_plot_potential():
    circ.plot_potential(
        θ1=circ._default_grid_phi.make_linspace(),
        θ2=circ._default_grid_phi.make_linspace(),
        θ3=0,
    )
