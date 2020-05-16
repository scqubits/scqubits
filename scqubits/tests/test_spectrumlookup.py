# test_spectrumlookup.py
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
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.core.param_sweep import ParameterSweep


class TestSpectrumLookup:

    def initialize_hilbertspace(self):
        CPB1 = qubit.Transmon(
            EJ=40.0,
            EC=0.2,
            ng=0.3,
            ncut=40,
            truncated_dim=3  # after diagonalization, we will keep 3 levels
        )

        CPB2 = qubit.Transmon(
            EJ=30.0,
            EC=0.15,
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

    def test_hilbertspace_generate_lookup(self):
        hilbertspace = self.initialize_hilbertspace()
        hilbertspace.generate_lookup()

    def test_hilbertspace_lookup_bare_eigenenergies(self):
        hilbertspace = self.initialize_hilbertspace()
        hilbertspace.generate_lookup()

        CPB = hilbertspace[0]
        reference = np.asarray([-36.05064983, -28.25601136, -20.67410141])
        assert np.allclose(hilbertspace.lookup.bare_eigenenergies(CPB), reference)

    def test_hilbertspace_lookup_bare_index(self):
        hilbertspace = self.initialize_hilbertspace()
        hilbertspace.generate_lookup()
        reference = (1, 0, 1)
        assert hilbertspace.lookup.bare_index(8) == reference

    def test_hilbertspace_lookup_dressed_index(self):
        hilbertspace = self.initialize_hilbertspace()
        hilbertspace.generate_lookup()
        reference = 21
        assert hilbertspace.lookup.dressed_index((1, 2, 1)) == reference

    def test_hilbertspace_lookup_bare_eigenstates(self):
        hilbertspace = self.initialize_hilbertspace()
        hilbertspace.generate_lookup()
        CPB = hilbertspace[0]
        reference = np.asarray(
            [[9.81355277e-48, 9.30854381e-47, 6.24667247e-46],
             [6.43691609e-46, 6.16876506e-45, 4.12289788e-44],
             [4.08432918e-44, 3.89716449e-43, 2.58910351e-42],
             [2.46946241e-42, 2.34112552e-41, 1.54552340e-40],
             [1.41840000e-40, 1.33555952e-39, 8.75825742e-39],
             [7.72924773e-39, 7.22576051e-38, 4.70525632e-37],
             [3.99043912e-37, 3.70232919e-36, 2.39303178e-35],
             [1.94904073e-35, 1.79388526e-34, 1.15041755e-33],
             [8.99241757e-34, 8.20662359e-33, 5.21927229e-32],
             [3.91282194e-32, 3.53890467e-31, 2.23088730e-30],
             [1.60297165e-30, 1.43598881e-29, 8.96770059e-29],
             [6.17171701e-29, 5.47280554e-28, 3.38373118e-27],
             [2.22898717e-27, 1.95522337e-26, 1.19604078e-25],
             [7.53630460e-26, 6.53443664e-25, 3.95184002e-24],
             [2.38030833e-24, 2.03838345e-23, 1.21776107e-22],
             [7.00726389e-23, 5.92116628e-22, 3.49118136e-21],
             [1.91803972e-21, 1.59764762e-20, 9.28738868e-20],
             [4.86905872e-20, 3.99337435e-19, 2.28615708e-18],
             [1.14319364e-18, 9.22005967e-18, 5.19156110e-17],
             [2.47519037e-17, 1.96028507e-16, 1.08406384e-15],
             [4.92660565e-16, 3.82517606e-15, 2.07418437e-14],
             [8.98410446e-15, 6.82608546e-14, 3.62257139e-13],
             [1.49561042e-13, 1.10967323e-12, 5.75121167e-12],
             [2.26406443e-12, 1.63642248e-11, 8.26237305e-11],
             [3.10359624e-11, 2.17915666e-10, 1.06879561e-09],
             [3.83517061e-10, 2.60742769e-09, 1.23809279e-08],
             [4.25136635e-09, 2.78807695e-08, 1.27659489e-07],
             [4.20541851e-08, 2.64836082e-07, 1.16378489e-06],
             [3.69111261e-07, 2.22015398e-06, 9.30982723e-06],
             [2.85699456e-06, 1.63072825e-05, 6.47991812e-05],
             [1.93730997e-05, 1.04109860e-04, 3.88641713e-04],
             [1.14275333e-04, 5.72579791e-04, 1.98618170e-03],
             [5.81958843e-04, 2.68572824e-03, 8.53587859e-03],
             [2.53837018e-03, 1.06226113e-02, 3.03588663e-02],
             [9.40432575e-03, 3.49650614e-02, 8.75591950e-02],
             [2.93435401e-02, 9.42865786e-02, 1.99160495e-01],
             [7.64587603e-02, 2.04183470e-01, 3.42090752e-01],
             [1.65024759e-01, 3.45198039e-01, 4.07470769e-01],
             [2.92888514e-01, 4.33880781e-01, 2.56608116e-01],
             [4.24891514e-01, 3.59598148e-01, -8.79153816e-02],
             [5.01714912e-01, 9.84685217e-02, -3.53429771e-01],
             [4.81272090e-01, -2.20127278e-01, -2.78699112e-01],
             [3.75226600e-01, -4.13778959e-01, 5.98745832e-02],
             [2.38462243e-01, -4.12292719e-01, 3.47513274e-01],
             [1.24144932e-01, -2.88932985e-01, 4.00686521e-01],
             [5.32947942e-02, -1.54131669e-01, 2.86094353e-01],
             [1.90119468e-02, -6.50150668e-02, 1.47843633e-01],
             [5.68278365e-03, -2.22152354e-02, 5.88699461e-02],
             [1.43546171e-03, -6.26030710e-03, 1.87174037e-02],
             [3.09023712e-04, -1.47627434e-03, 4.86852349e-03],
             [5.71637638e-05, -2.94942330e-04, 1.05515547e-03],
             [9.15737091e-06, -5.04652995e-05, 1.93379175e-04],
             [1.27979057e-06, -7.46595985e-06, 3.03408392e-05],
             [1.57114391e-07, -9.63222539e-07, 4.11860306e-06],
             [1.70524328e-08, -1.09208038e-07, 4.88161208e-07],
             [1.64601782e-09, -1.09569023e-08, 5.09307414e-08],
             [1.42087441e-10, -9.78961139e-10, 4.71111386e-09],
             [1.10247361e-11, -7.83397639e-11, 3.88859043e-10],
             [7.72550576e-13, -5.64439383e-12, 2.88076527e-11],
             [4.91062053e-14, -3.67924669e-13, 1.92555426e-12],
             [2.84291230e-15, -2.17934446e-14, 1.16685222e-13],
             [1.50469805e-16, -1.17783922e-15, 6.43856605e-15],
             [7.30668786e-18, -5.83004745e-17, 3.24802569e-16],
             [3.26587050e-19, -2.65213639e-18, 1.50352885e-17],
             [1.34775547e-20, -1.11239813e-19, 6.40837496e-19],
             [5.14987234e-22, -4.31489395e-21, 2.52289937e-20],
             [1.82690101e-23, -1.55217860e-22, 9.20113815e-22],
             [6.03189844e-25, -5.19171482e-24, 3.11715376e-23],
             [1.85795057e-26, -1.61860341e-25, 9.83457910e-25],
             [5.35075470e-28, -4.71440666e-27, 2.89646327e-26],
             [1.44377736e-29, -1.28560173e-28, 7.98111283e-28],
             [3.65714082e-31, -3.28896361e-30, 2.06182159e-29],
             [8.71252159e-33, -7.90887032e-32, 5.00363611e-31],
             [1.95554312e-34, -1.79083458e-33, 1.14280608e-32],
             [4.14221247e-36, -3.82491690e-35, 2.46077129e-34],
             [8.29318581e-38, -7.71816934e-37, 5.00379776e-36],
             [1.57174582e-39, -1.47365562e-38, 9.62359716e-38],
             [2.82377905e-41, -2.66622536e-40, 1.75318677e-39],
             [4.81562064e-43, -4.57736079e-42, 3.02958416e-41],
             [7.80686036e-45, -7.46654748e-44, 4.97258814e-43],
             [1.20604294e-46, -1.15822587e-45, 7.76029780e-45]]
        )
        assert np.allclose(hilbertspace.lookup.bare_eigenstates(CPB), reference)


class TestParameterSweep:

    def initialize(self):
        # Set up the components / subspaces of our Hilbert space
        # Set up the components / subspaces of our Hilbert space

        CPB1 = qubit.Transmon(
            EJ=40.0,
            EC=0.2,
            ng=0.3,
            ncut=40,
            truncated_dim=3  # after diagonalization, we will keep 3 levels
        )

        CPB2 = qubit.Transmon(
            EJ=30.0,
            EC=0.15,
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
        param_vals = np.linspace(0., 2.0, 300)  # parameter values

        subsys_update_list = [CPB1,
                              CPB2]  # list of HilbertSpace subsys_list which are affected by parameter changes

        def update_hilbertspace(param_val):  # function that shows how Hilbert space components are updated
            CPB1.EJ = 20 * np.abs(np.cos(np.pi * param_val))
            CPB2.EJ = 15 * np.abs(np.cos(np.pi * param_val * 0.65))

        sweep = ParameterSweep(
            param_name=param_name,
            param_vals=param_vals,
            evals_count=20,
            hilbertspace=hilbertspace,
            subsys_update_list=subsys_update_list,
            update_hilbertspace=update_hilbertspace
        )
        return sweep

    def test_sweep_bare_eigenenergies(self):
        sweep = self.initialize()
        reference = np.asarray([-12.6254519,  -8.58335482,  -4.70576686,  -1.00508497])
        CPB2 = sweep.get_subsys(1)
        calculated = sweep.lookup.bare_eigenenergies(CPB2, 15)
        print(calculated)
        assert np.allclose(reference, calculated)

    def test_sweep_bare_eigenstates(self):
        sweep = self.initialize()
        reference = np.asarray(
            [[-4.36541328e-50, -2.75880308e-54, 8.97850818e-52],
             [-3.11868192e-49, -2.79006815e-54, -8.74802339e-53],
             [-8.58246250e-50, -1.00876179e-53, -1.01005272e-51],
             [1.07036276e-49, -7.49520810e-52, 3.40138146e-51],
             [2.22076735e-50, -9.29825414e-50, 5.29476367e-49],
             [9.31798507e-49, -1.09455894e-47, 6.20160456e-47],
             [1.48748935e-46, -1.21908616e-45, 6.87331138e-45],
             [1.57335293e-44, -1.28270423e-43, 7.19436330e-43],
             [1.57038899e-42, -1.27291721e-41, 7.10007771e-41],
             [1.47623385e-40, -1.18931710e-39, 6.59489623e-39],
             [1.30464570e-38, -1.04427630e-37, 5.75451766e-37],
             [1.08188824e-36, -8.60002157e-36, 4.70755913e-35],
             [8.40106916e-35, -6.62889364e-34, 3.60280729e-33],
             [6.09538520e-33, -4.77167090e-32, 2.57367705e-31],
             [4.12259568e-31, -3.20001276e-30, 1.71188450e-29],
             [2.59275754e-29, -1.99422862e-28, 1.05746143e-27],
             [1.51222344e-27, -1.15172482e-26, 6.04923527e-26],
             [8.15621522e-26, -6.14597428e-25, 3.19493711e-24],
             [4.05546577e-24, -3.02076696e-23, 1.55282385e-22],
             [1.85279307e-22, -1.36279552e-21, 6.92039152e-21],
             [7.74967547e-21, -5.62219212e-20, 2.81708905e-19],
             [2.95606461e-19, -2.11237965e-18, 1.04300880e-17],
             [1.02393371e-17, -7.19606307e-17, 3.49598322e-16],
             [3.20583976e-16, -2.21183282e-15, 1.05539288e-14],
             [9.02643915e-15, -6.10107117e-14, 2.85332763e-13],
             [2.27284550e-13, -1.50130691e-12, 6.86483062e-12],
             [5.08658910e-12, -3.27390598e-11, 1.45936793e-10],
             [1.00490495e-10, -6.28031136e-10, 2.71942263e-09],
             [1.73929913e-09, -1.05099253e-08, 4.40157221e-08],
             [2.61520163e-08, -1.51995670e-07, 6.12372338e-07],
             [3.38385801e-07, -1.87937831e-06, 7.23484440e-06],
             [3.72810682e-06, -1.96242274e-05, 7.15569990e-05],
             [3.45584871e-05, -1.70590444e-04, 5.82474864e-04],
             [2.65942237e-04, -1.21399020e-03, 3.82167032e-03],
             [1.67360896e-03, -6.93272504e-03, 1.96865734e-02],
             [8.46996689e-03, -3.10088949e-02, 7.69019540e-02],
             [3.38450374e-02, -1.05369628e-01, 2.16719161e-01],
             [1.04711605e-01, -2.61101801e-01, 4.05243750e-01],
             [2.45912280e-01, -4.42899710e-01, 4.12353303e-01],
             [4.30449532e-01, -4.49470571e-01, 3.05581613e-02],
             [5.53739884e-01, -1.35052186e-01, -3.89790634e-01],
             [5.19852531e-01, 2.94931877e-01, -2.63181317e-01],
             [3.56804110e-01, 4.82977168e-01, 2.23412560e-01],
             [1.80869825e-01, 3.77382768e-01, 4.51858334e-01],
             [6.88151152e-02, 1.89260831e-01, 3.34098107e-01],
             [2.00308382e-02, 6.69188913e-02, 1.49539620e-01],
             [4.55033333e-03, 1.75694853e-02, 4.62204064e-02],
             [8.22318568e-04, 3.55076664e-03, 1.05407454e-02],
             [1.20336099e-04, 5.67882562e-04, 1.85083059e-03],
             [1.44906152e-05, 7.34992558e-05, 2.58056443e-04],
             [1.45665042e-06, 7.84270617e-06, 2.92596095e-05],
             [1.23810941e-07, 7.00861842e-07, 2.75014737e-06],
             [8.99993323e-09, 5.31654232e-08, 2.17699883e-07],
             [5.65189610e-10, 3.46352201e-09, 1.47086607e-08],
             [3.09419265e-11, 1.95757580e-10, 8.57946263e-10],
             [1.48871045e-12, 9.68547769e-12, 4.36327789e-11],
             [6.34068704e-14, 4.22836088e-13, 1.95158208e-12],
             [2.40640133e-15, 1.64038032e-14, 7.73561260e-14],
             [8.18610028e-17, 5.69106676e-16, 2.73577781e-15],
             [2.50956428e-18, 1.77583368e-17, 8.68518693e-17],
             [6.96721961e-20, 5.00977686e-19, 2.48862809e-18],
             [1.75956716e-21, 1.28376682e-20, 6.46796092e-20],
             [4.05902398e-23, 3.00104047e-22, 1.53161646e-21],
             [8.58515967e-25, 6.42518490e-24, 3.31807834e-23],
             [1.67070903e-26, 1.26444488e-25, 6.60096121e-25],
             [3.00109957e-28, 2.29490687e-27, 1.21007053e-26],
             [4.99099559e-30, 3.85321228e-29, 2.05059703e-28],
             [7.70601966e-32, 6.00230825e-31, 3.22178302e-30],
             [1.10748033e-33, 8.69778971e-33, 4.70594474e-32],
             [1.48510624e-35, 1.17536887e-34, 6.40674933e-34],
             [1.86242981e-37, 1.48464566e-36, 8.14893098e-36],
             [2.18890702e-39, 1.75670712e-38, 9.70510353e-38],
             [2.41583947e-41, 1.95114772e-40, 1.08452672e-39],
             [2.50853833e-43, 2.03811379e-42, 1.13938276e-41],
             [2.45571628e-45, 2.00585220e-44, 1.12742273e-43],
             [2.30038269e-47, 1.86312496e-46, 1.05255328e-45],
             [2.97703610e-49, 1.63590361e-48, 9.28557377e-48],
             [-2.98788454e-49, 1.35968736e-50, 7.75238654e-50],
             [-2.85901234e-50, 1.01001731e-52, 6.08803345e-52],
             [-3.46227453e-49, 2.24305288e-55, 3.05453863e-52],
             [-5.44193943e-50, -2.99920038e-54, -5.73539237e-53]]
        )
        CPB1 = sweep.get_subsys(0)
        assert np.allclose(reference, sweep.lookup.bare_eigenstates(CPB1, 21))
