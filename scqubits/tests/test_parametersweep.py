# test_parametersweep.py
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

import scqubits as scq

from scqubits.core.param_sweep import ParameterSweep


@pytest.mark.usefixtures("num_cpus")
class TestParameterSweep:
    @pytest.fixture(autouse=True)
    def set_tmpdir(self, request):
        setattr(self, "tmpdir", request.getfixturevalue("tmpdir"))

    def initialize(self, num_cpus):
        # Set up the components / subspaces of our Hilbert space
        scq.settings.MULTIPROC = "pathos"

        tmon1 = scq.TunableTransmon(
            EJmax=40.0,
            EC=0.2,
            d=0.1,
            flux=0.0,
            ng=0.3,
            ncut=40,
            truncated_dim=3,  # after diagonalization, we will keep 3 levels
        )

        tmon2 = scq.TunableTransmon(
            EJmax=15.0, EC=0.15, d=0.2, flux=0.0, ng=0.0, ncut=30, truncated_dim=3
        )

        resonator = scq.Oscillator(
            E_osc=4.5, truncated_dim=4  # up to 3 photons (0,1,2,3)
        )

        hilbertspace = scq.HilbertSpace([tmon1, tmon2, resonator])

        g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)
        g2 = 0.2  # coupling resonator-CPB2 (without charge matrix elements)

        hilbertspace.add_interaction(
            g_strength=g1,
            op1=tmon1.n_operator,
            op2=resonator.creation_operator,
            add_hc=True,
        )

        hilbertspace.add_interaction(
            g_strength=g2,
            op1=tmon2.n_operator,
            op2=resonator.creation_operator,
            add_hc=True,
        )

        # Set up parameter name and values
        pname1 = "flux"
        flux_vals = np.linspace(0.0, 2.0, 11)
        pname2 = "ng"
        ng_vals = np.linspace(-0.5, 0.5, 3)

        # combine into a dictionary
        paramvals_by_name = {pname1: flux_vals, pname2: ng_vals}

        area_ratio = 1.2

        def update_hilbertspace(
            flux, ng
        ):  # function that defines how Hilbert space components are updated
            tmon1.flux = flux
            tmon2.flux = area_ratio * flux
            tmon2.ng = ng

        # dictionary with information on which subsystems are affected by changing parameters
        subsys_update_info = {pname1: [tmon1, tmon2], pname2: [tmon2]}

        # create the ParameterSweep
        sweep = ParameterSweep(
            hilbertspace=hilbertspace,
            paramvals_by_name=paramvals_by_name,
            update_hilbertspace=update_hilbertspace,
            evals_count=20,
            subsys_update_info=subsys_update_info,
            num_cpus=num_cpus,
        )
        return sweep

    def test_ParameterSweep(self, num_cpus):
        sweep = self.initialize(num_cpus)

        calculated_evals = sweep["evals"][1, 1]

        reference_evals = np.array(
            [
                -38.24067872,
                -34.80636811,
                -33.70287551,
                -31.50027076,
                -31.23467726,
                -30.28547786,
                -29.16544659,
                -27.80039104,
                -27.08317259,
                -26.69778263,
                -25.76389721,
                -24.59864846,
                -24.49435409,
                -24.43639473,
                -23.2803766,
                -22.63621727,
                -22.16084701,
                -21.00224623,
                -21.00053542,
                -20.07811164,
            ]
        )
        assert np.allclose(reference_evals, calculated_evals)

    def test_ParameterSweep_fileIO(self, num_cpus):
        sweep = self.initialize(num_cpus)
        sweep.filewrite(self.tmpdir + "test.h5")
        sweep_copy = scq.read(self.tmpdir + "test.h5")
