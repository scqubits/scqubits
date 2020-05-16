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

import scqubits as qubit
import scqubits.core.sweep_generators as swp
from scqubits import Explorer, InteractionTerm, ParameterSweep


def test_explorer():
    qbt = qubit.Fluxonium(
        EJ=2.55,
        EC=0.72,
        EL=0.12,
        flux=0.0,
        cutoff=110,
        truncated_dim=9
    )

    osc = qubit.Oscillator(
        E_osc=4.0,
        truncated_dim=5
    )

    hilbertspace = qubit.HilbertSpace([qbt, osc])

    interaction = InteractionTerm(
        g_strength=0.2,
        op1=qbt.n_operator(),
        subsys1=qbt,
        op2=osc.creation_operator() + osc.annihilation_operator(),
        subsys2=osc
    )

    interaction_list = [interaction]
    hilbertspace.interaction_list = interaction_list

    param_name = r'$\Phi_{ext}/\Phi_0$'
    param_vals = np.linspace(-0.5, 0.5, 100)

    subsys_update_list = [qbt]

    def update_hilbertspace(param_val):
        qbt.flux = param_val

    sweep = ParameterSweep(
        param_name=param_name,
        param_vals=param_vals,
        evals_count=10,
        hilbertspace=hilbertspace,
        subsys_update_list=subsys_update_list,
        update_hilbertspace=update_hilbertspace,
    )
    swp.generate_chi_sweep(sweep)
    swp.generate_charge_matrixelem_sweep(sweep)

    explorer = Explorer(
        sweep=sweep,
        evals_count=10
    )

    explorer.interact()
