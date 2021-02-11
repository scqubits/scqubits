# test_hilbertspace.py
# meant to be run with 'pytest'
#
# This file is part of scqubits.
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

from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.sweep_generators import generate_diffspec_sweep
from scqubits.utils.spectrum_utils import absorption_spectrum, get_matrixelement_table


@pytest.mark.usefixtures("num_cpus")
class TestParameterSweep:
    @pytest.fixture(autouse=True)
    def set_tmpdir(self, request):
        setattr(self, "tmpdir", request.getfixturevalue("tmpdir"))

    def initialize(self, num_cpus):
        # Set up the components / subspaces of our Hilbert space
        scq.settings.MULTIPROC = "pathos"

        CPB1 = scq.Transmon(
            EJ=40.0,
            EC=0.2,
            ng=0.0,
            ncut=40,
            truncated_dim=3,  # after diagonalization, we will keep 3 levels
        )

        CPB2 = scq.Transmon(EJ=3.0, EC=1.0, ng=0.0, ncut=10, truncated_dim=4)

        resonator = scq.Oscillator(
            E_osc=6.0, truncated_dim=4
        )  # up to 3 photons (0,1,2,3)

        # Form a list of all components making up the Hilbert space.
        hilbertspace = HilbertSpace([CPB1, CPB2, resonator])

        g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)
        g2 = 0.2  # coupling resonator-CPB2 (without charge matrix elements)

        interaction1 = InteractionTerm(
            g_strength=g1,
            op1=CPB1.n_operator(),
            subsys1=CPB1,
            op2=resonator.creation_operator() + resonator.annihilation_operator(),
            subsys2=resonator,
        )

        interaction2 = InteractionTerm(
            g_strength=g2,
            op1=CPB2.n_operator(),
            subsys1=CPB2,
            op2=resonator.creation_operator() + resonator.annihilation_operator(),
            subsys2=resonator,
        )

        interaction_list = [interaction1, interaction2]
        hilbertspace.interaction_list = interaction_list

        param_name = "flux"  # name of varying external parameter
        param_vals = np.linspace(-0.1, 0.6, 100)  # parameter values

        subsys_update_list = [
            CPB1
        ]  # list of HilbertSpace subsys_list which are affected by parameter changes

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
            num_cpus=num_cpus,
        )
        return sweep

    def test_ParameterSweep(self, num_cpus):
        sweep = self.initialize(num_cpus)

        specdata = absorption_spectrum(
            generate_diffspec_sweep(sweep, initial_state_ind=0)
        )
        calculated_energies = specdata.energy_table[5]

        reference_energies = np.array(
            [
                0.0,
                4.74135372,
                5.6773522,
                5.98902462,
                7.72420838,
                10.72273595,
                11.65962582,
                11.97802377,
                12.46554431,
                13.40154194,
                13.71041554,
                15.24359501,
                16.70439594,
                17.01076356,
                17.64202619,
            ]
        )
        assert np.allclose(reference_energies, calculated_energies)

    def test_ParameterSweep_fileIO(self, num_cpus):
        sweep = self.initialize(num_cpus)
        sweep.filewrite(self.tmpdir + "test.h5")
        sweep_copy = scq.read(self.tmpdir + "test.h5")
