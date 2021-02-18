from itertools import product

import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor
from scipy.linalg import inv
from scipy.optimize import root
from sympy import Matrix, S, diff, hessian, simplify, solve, symbols

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.fluxonium import Fluxonium, FluxoniumFluxVariableAllocation
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.utils.spectrum_utils import get_matrixelement_table, standardize_phases


class FluxoniumTunableCouplerFloating(serializers.Serializable):
    def __init__(
        self,
        EJa,
        EJb,
        ECg_top,
        ECg_bottom,
        ECg_sides,
        ECq1,
        ECq2,
        ELa,
        ELb,
        flux_a,
        flux_b,
        flux_c,
        fluxonium_cutoff,
        fluxonium_truncated_dim,
        ECc,
        ECm,
        EL1,
        EL2,
        EJC,
        fluxonium_minus_truncated_dim=6,
        h_o_truncated_dim=3,
    ):
        self.EJa = EJa
        self.EJb = EJb
        self.ECg_top = ECg_top
        self.ECg_bottom = ECg_bottom
        self.ECg_sides = ECg_sides
        self.ECq1 = ECq1
        self.ECq2 = ECq2
        self.ELa = ELa
        self.ELb = ELb
        self.flux_a = flux_a
        self.flux_b = flux_b
        self.flux_c = flux_c
        self.fluxonium_cutoff = fluxonium_cutoff
        self.fluxonium_truncated_dim = fluxonium_truncated_dim
        self.fluxonium_minus_truncated_dim = fluxonium_minus_truncated_dim
        self.h_o_truncated_dim = h_o_truncated_dim
        self.ECc = ECc
        self.ECm = ECm
        self.EL1 = EL1
        self.EL2 = EL2
        self.EJC = EJC
        self._sys_type = type(self).__name__

    @staticmethod
    def _U_matrix():
        return Matrix(
            [
                [1, -1, 0, 0, 0],
                [0, 1, 0, 0, -1],
                [0, 2, -1, -1, 0],
                [0, 0, -1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )

    def capacitance_matrix(self):
        U = self._U_matrix()
        U_inv = U ** -1
        phi1, phi2, phi3, phi4, phi5 = symbols("phi1 phi2 phi3 phi4 phi5")
        phi_vector = Matrix([phi1, phi2, phi3, phi4, phi5])
        Cc = 1 / S(2.0 * self.ECc)
        Cg_top = 1 / S(2.0 * self.ECg_top)
        Cg_bottom = 1 / S(2.0 * self.ECg_bottom)
        Cg_sides = 1 / S(2.0 * self.ECg_sides)
        Cm = 1 / S(2.0 * self.ECm)
        Cq1 = 1 / S(2.0 * self.ECq1)
        Cq2 = 1 / S(2.0 * self.ECq2)
        T = 0.5 * (
            Cc * (phi3 - phi4) ** 2
            + Cg_sides * (phi1 ** 2 + phi5 ** 2)
            + Cg_bottom * phi2 ** 2
            + Cg_top * (phi3 ** 2 + phi4 ** 2)
            + Cm * ((phi2 - phi3) ** 2 + (phi2 - phi4) ** 2)
            + Cq1 * (phi1 - phi2) ** 2
            + Cq2 * (phi2 - phi5) ** 2
        )
        varphia, varphib, varphi1, varphi2, varphisum = symbols(
            "varphia varphib varphi1 varphi2 varphisum"
        )
        varphi_list = Matrix([varphia, varphib, varphi1, varphi2, varphisum])
        phi_subs = U_inv * varphi_list
        T = T.subs([(phival, phi_subs[j]) for j, phival in enumerate(phi_vector)])
        T = simplify(T.subs(varphisum, solve(diff(T, varphisum), varphisum)[0]))
        cap_mat = hessian(T, varphi_list)
        return np.array(cap_mat, dtype=np.float_)[:-1, :-1]

    def _find_ECq(self, target_ECq1, target_ECq2, ECq):
        self.ECq1 = ECq[0]
        self.ECq2 = ECq[1]
        EC_matrix = self.EC_matrix()
        return [EC_matrix[0, 0] - target_ECq1, EC_matrix[1, 1] - target_ECq2]

    def _find_ECq1(self, ECq1, target_ECq1):
        self.ECq1 = ECq1[0]
        return self.EC_matrix()[0, 0] - target_ECq1

    def _find_ECq2(self, ECq2, target_ECq2):
        self.ECq2 = ECq2[0]
        return self.EC_matrix()[1, 1] - target_ECq2

    def find_ECq_given_target(self, given_ECq1, given_ECq2):
        result_ECq1 = root(self._find_ECq1, self.ECq1, given_ECq1)
        result_ECq2 = root(self._find_ECq2, self.ECq2, given_ECq2)
        if not result_ECq1.success:
            self.ECq1 = np.inf
        if not result_ECq2.success:
            self.ECq2 = np.inf
        self.ECq1 = result_ECq1.x[0]
        self.ECq2 = result_ECq2.x[0]

    def EC_matrix(self):
        return 0.5 * inv(self.capacitance_matrix())

    def qubit_a_charging_energy(self):
        return self.EC_matrix()[0, 0]

    def qubit_b_charging_energy(self):
        return self.EC_matrix()[1, 1]

    def off_diagonal_charging(self):
        return self.EC_matrix()[0, 1]

    def generate_coupled_system(self):
        """Returns a HilbertSpace object of the full system of two fluxonium qubits interacting via
        a tunable coupler, which takes the form of a harmonic oscillator degree of freedom and
        fluxonium degree of freedom which are themselves decoupled

        Returns
        -------
        HilbertSpace
        """
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        h_o_plus = self.h_o_plus()
        hilbert_space = HilbertSpace(
            [fluxonium_a, fluxonium_b, fluxonium_minus, h_o_plus]
        )
        phi_minus = fluxonium_minus.phi_operator()
        phi_a = fluxonium_a.phi_operator()
        id_a = np.eye(fluxonium_a.hilbertdim())
        phi_b = fluxonium_b.phi_operator()
        id_b = np.eye(fluxonium_b.hilbertdim())
        n_a = fluxonium_a.n_operator()
        n_b = fluxonium_b.n_operator()
        interaction_term_1 = InteractionTerm(
            g_strength=-0.5 * self.ELa,
            subsys1=fluxonium_a,
            op1=phi_a + 2.0 * np.pi * self.flux_a * id_a,
            subsys2=h_o_plus,
            op2=self.phi_plus(),
        )
        interaction_term_2 = InteractionTerm(
            g_strength=-0.5 * self.ELb,
            subsys1=fluxonium_b,
            op1=phi_b + 2.0 * np.pi * self.flux_b * id_b,
            subsys2=h_o_plus,
            op2=self.phi_plus(),
        )
        interaction_term_3 = InteractionTerm(
            g_strength=-0.5 * self.ELa,
            subsys1=fluxonium_a,
            op1=phi_a + 2.0 * np.pi * self.flux_a * id_a,
            subsys2=fluxonium_minus,
            op2=phi_minus,
        )
        interaction_term_4 = InteractionTerm(
            g_strength=0.5 * self.ELb,
            subsys1=fluxonium_b,
            op1=phi_b + 2.0 * np.pi * self.flux_b * id_b,
            subsys2=fluxonium_minus,
            op2=phi_minus,
        )
        interaction_term_5 = InteractionTerm(
            g_strength=-8.0 * self.off_diagonal_charging(),
            subsys1=fluxonium_a,
            op1=n_a,
            subsys2=fluxonium_b,
            op2=n_b,
        )
        interaction_term_6 = InteractionTerm(
            g_strength=(self.ELa - self.ELb + self.EL1 - self.EL2) / 2.0,
            subsys1=h_o_plus,
            op1=self.phi_plus(),
            subsys2=fluxonium_minus,
            op2=phi_minus,
        )
        hilbert_space.interaction_list = [
            interaction_term_1,
            interaction_term_2,
            interaction_term_3,
            interaction_term_4,
            interaction_term_5,
            interaction_term_6,
        ]
        return hilbert_space

    def find_flux_shift(self):
        fluxonium_minus = self.fluxonium_minus()
        evals_minus, evecs_minus = fluxonium_minus.eigensys(
            evals_count=fluxonium_minus.truncated_dim
        )
        phi_minus_mat = get_matrixelement_table(
            fluxonium_minus.phi_operator(), evecs_minus
        )
        groundstate_expect = np.real(phi_minus_mat[0, 0])
        chi_m = sum(
            abs(phi_minus_mat[0, m]) ** 2 / (evals_minus[m] - evals_minus[0])
            for m in range(1, fluxonium_minus.truncated_dim)
        )
        EL_tilda = self.EL_tilda()

        def flux_shift_qubit(EL_qubit):
            return (
                EL_qubit
                * groundstate_expect
                / (
                    4
                    * 0.5
                    * (EL_qubit - (EL_qubit ** 2 * (0.5 * chi_m + 1.0 / EL_tilda)))
                )
            )

        return (
            flux_shift_qubit(self.ELa) / (2.0 * np.pi),
            flux_shift_qubit(self.ELb) / (2.0 * np.pi),
        )

    def chi_minus(self):
        fluxonium_minus = self.fluxonium_minus()
        evals_minus, evecs_minus = fluxonium_minus.eigensys(
            evals_count=fluxonium_minus.truncated_dim
        )
        phi_minus_mat = get_matrixelement_table(
            fluxonium_minus.phi_operator(), evecs_minus
        )
        return sum(
            abs(phi_minus_mat[0, m]) ** 2 / (evals_minus[m] - evals_minus[0])
            for m in range(1, fluxonium_minus.truncated_dim)
        )

    def fluxonium_minus_gs_expect(self):
        fluxonium_minus = self.fluxonium_minus()
        evals_minus, evecs_minus = fluxonium_minus.eigensys(evals_count=1)
        phi_minus_mat = get_matrixelement_table(
            fluxonium_minus.phi_operator(), evecs_minus
        )
        return np.real(phi_minus_mat[0, 0])

    def _setup_effective_calculation(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        evals_minus, evecs_minus = fluxonium_minus.eigensys(
            evals_count=fluxonium_minus.truncated_dim
        )
        phi_minus_mat = get_matrixelement_table(
            fluxonium_minus.phi_operator(), evecs_minus
        )
        chi_m = sum(
            abs(phi_minus_mat[0, m]) ** 2 / (evals_minus[m] - evals_minus[0])
            for m in range(1, fluxonium_minus.truncated_dim)
        )
        E_La_shift = self.ELa ** 2 * (0.5 * chi_m + 1.0 / self.EL_tilda())
        fluxonium_a.EL = self.ELa - E_La_shift
        E_Lb_shift = self.ELb ** 2 * (0.5 * chi_m + 1.0 / self.EL_tilda())
        fluxonium_b.EL = self.ELb - E_Lb_shift
        J = self.ELa * self.ELb * (0.5 * chi_m - 1.0 / (self.EL_tilda()))
        return fluxonium_a, fluxonium_b, J, E_La_shift, E_Lb_shift

    def schrieffer_wolff_born_oppenheimer_effective_hamiltonian(self):
        (
            fluxonium_a,
            fluxonium_b,
            J,
            E_La_shift,
            E_Lb_shift,
        ) = self._setup_effective_calculation()
        evals_a, evecs_a_uns = fluxonium_a.eigensys(
            evals_count=fluxonium_a.truncated_dim
        )
        evals_b, evecs_b_uns = fluxonium_b.eigensys(
            evals_count=fluxonium_b.truncated_dim
        )
        # Had issues with signs flipping: standardizing overall phase of eigenvectors
        evecs_a = np.zeros_like(evecs_a_uns)
        evecs_b = np.zeros_like(evecs_b_uns)
        evecs_a_uns = evecs_a_uns.T
        evecs_b_uns = evecs_b_uns.T
        for k, evec in enumerate(evecs_a_uns):
            evecs_a[:, k] = standardize_phases(evec)
        for k, evec in enumerate(evecs_b_uns):
            evecs_b[:, k] = standardize_phases(evec)

        # Generate matrix elements
        evals_a = evals_a - evals_a[0]
        evals_b = evals_b - evals_b[0]
        phi_a_mat = get_matrixelement_table(fluxonium_a.phi_operator(), evecs_a)
        phi_b_mat = get_matrixelement_table(fluxonium_b.phi_operator(), evecs_b)
        n_a_mat = get_matrixelement_table(fluxonium_a.n_operator(), evecs_a)
        n_b_mat = get_matrixelement_table(fluxonium_b.n_operator(), evecs_b)
        dim_a = fluxonium_a.truncated_dim
        dim_b = fluxonium_b.truncated_dim

        # For ease of using hubbard_operator, define a spin fluxonium with truncated_dim = 2
        fluxonium_a_spin = self.fluxonium_a()
        fluxonium_b_spin = self.fluxonium_b()
        fluxonium_a_spin.EL = self.ELa - E_La_shift
        fluxonium_b_spin.EL = self.ELb - E_Lb_shift
        fluxonium_a_spin.truncated_dim = 2
        fluxonium_b_spin.truncated_dim = 2
        hilbert_space = HilbertSpace([fluxonium_a_spin, fluxonium_b_spin])
        dim_low_energy_a = fluxonium_a_spin.truncated_dim
        dim_low_energy_b = fluxonium_b_spin.truncated_dim

        off_diag = self.off_diagonal_charging()

        def V_op(init_a, fin_a, init_b, fin_b):
            return (
                J * phi_a_mat[init_a, fin_a] * phi_b_mat[init_b, fin_b]
                - 8.0 * off_diag * n_a_mat[init_a, fin_a] * n_b_mat[init_b, fin_b]
            )

        H_0, H_1, H_2 = 0.0, 0.0, 0.0

        H_0_a = sum(
            evals_a[j] * hilbert_space.hubbard_operator(j, j, fluxonium_a_spin)
            for j in range(dim_low_energy_a)
        )
        H_0_b = sum(
            evals_b[j] * hilbert_space.hubbard_operator(j, j, fluxonium_b_spin)
            for j in range(dim_low_energy_b)
        )
        H_0 = H_0_a + H_0_b

        H_1 = sum(
            (V_op(init_a, fin_a, init_b, fin_b))
            * hilbert_space.hubbard_operator(init_a, fin_a, fluxonium_a_spin)
            * hilbert_space.hubbard_operator(init_b, fin_b, fluxonium_b_spin)
            for init_a in range(dim_low_energy_a)
            for fin_a in range(dim_low_energy_a)
            for init_b in range(dim_low_energy_b)
            for fin_b in range(dim_low_energy_b)
        )

        # virtual_int_states = list(product(np.arange(0, dim_a), np.arange(0, dim_b)))
        # virtual_int_states.remove((0, 0))
        # virtual_int_states.remove((0, 1))
        # virtual_int_states.remove((1, 0))
        # virtual_int_states.remove((1, 1))
        #
        # H_2 = sum(V_op(init_a, inter_a, init_b, inter_b) * V_op(inter_a, fin_a, inter_b, fin_b)
        #           * 0.5 * ((evals_a[init_a] + evals_b[init_b] - (evals_a[inter_a] + evals_b[inter_b])) ** (-1)
        #                    + (evals_a[fin_a] + evals_b[fin_b] - (evals_a[inter_a] + evals_b[inter_b])) ** (-1))
        #           * hilbert_space.hubbard_operator(init_a, fin_a, fluxonium_a_spin)
        #           * hilbert_space.hubbard_operator(init_b, fin_b, fluxonium_b_spin)
        #           for init_a in range(dim_low_energy_a) for fin_a in range(dim_low_energy_a)
        #           for init_b in range(dim_low_energy_b) for fin_b in range(dim_low_energy_b)
        #           for inter_a, inter_b in virtual_int_states)

        return H_0, H_1, H_2

    @staticmethod
    def decompose_matrix_into_specific_paulis(sigmai, sigmaj, matrix):
        sigmaij = tensor(sigmai, sigmaj)
        return 0.5 * np.trace((sigmaij * matrix).data.toarray())

    @staticmethod
    def decompose_matrix_into_paulis(matrix):
        pauli_mats = [qeye(2), sigmax(), sigmay(), sigmaz()]
        pauli_name = ["I", "X", "Y", "Z"]
        pauli_list = []
        for j, pauli_a in enumerate(pauli_mats):
            for k, pauli_b in enumerate(pauli_mats):
                paulia_a_b = tensor(pauli_a, pauli_b)
                coeff = 0.5 * np.trace((paulia_a_b * matrix).data.toarray())
                pauli_list.append((pauli_name[j] + pauli_name[k], coeff))
        return pauli_list

    def born_oppenheimer_effective_hamiltonian(self):
        (
            fluxonium_a,
            fluxonium_b,
            J,
            E_La_shift,
            E_Lb_shift,
        ) = self._setup_effective_calculation()
        g_s_expect = self.fluxonium_minus_gs_expect()
        EL_bar = fluxonium_a.EL
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_a.truncated_dim, fluxonium_b.truncated_dim = 2, 2
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        H_0 = hilbert_space.hamiltonian()
        phi_a_ops, n_a_ops = self._single_hamiltonian_effective(
            fluxonium_a, hilbert_space
        )
        phi_b_ops, n_b_ops = self._single_hamiltonian_effective(
            fluxonium_b, hilbert_space
        )
        H_a = phi_a_ops * (
            2.0 * np.pi * (0.5 - self.flux_a) * EL_bar
            - 2.0 * np.pi * (0.5 - self.flux_b) * J
            - 0.5 * self.ELa * g_s_expect
        )
        H_b = phi_b_ops * (
            2.0 * np.pi * (0.5 - self.flux_b) * EL_bar
            - 2.0 * np.pi * (0.5 - self.flux_a) * J
            + 0.5 * self.ELb * g_s_expect
        )
        H_ab = (
            J * (phi_a_ops * phi_b_ops)
            - 8.0 * self.off_diagonal_charging() * n_a_ops * n_b_ops
        )
        return H_0 + H_a + H_b + H_ab

    @staticmethod
    def _single_hamiltonian_effective(fluxonium_instance, hilbert_space):
        dim = fluxonium_instance.truncated_dim
        evals, evecs = fluxonium_instance.eigensys(evals_count=dim)
        phi_mat = get_matrixelement_table(fluxonium_instance.phi_operator(), evecs)
        n_mat = get_matrixelement_table(fluxonium_instance.n_operator(), evecs)
        phi_ops = sum(
            [
                phi_mat[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_instance)
                for j in range(dim)
                for k in range(dim)
            ]
        )
        n_ops = sum(
            [
                n_mat[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_instance)
                for j in range(dim)
                for k in range(dim)
            ]
        )
        return phi_ops, n_ops

    def fluxonium_a(self):
        return FluxoniumFluxVariableAllocation(
            self.EJa,
            self.qubit_a_charging_energy(),
            self.ELa,
            self.flux_a,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_truncated_dim,
            flux_fraction_with_inductor=1.0
        )

    def fluxonium_b(self):
        return FluxoniumFluxVariableAllocation(
            self.EJb,
            self.qubit_b_charging_energy(),
            self.ELb,
            self.flux_b,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_truncated_dim,
            flux_fraction_with_inductor=1.0
        )

    def EL_tilda(self):
        return self.EL1 + self.EL2 + self.ELa + self.ELb

    def fluxonium_minus(self):
        return FluxoniumFluxVariableAllocation(
            self.EJC,
            self.fluxonium_minus_charging_energy(),
            self.EL_tilda() / 4.0,
            self.flux_c,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_minus_truncated_dim,
            flux_fraction_with_inductor=0.5
        )

    def h_o_plus_charging_energy(self):
        assert np.allclose(self.EC_matrix()[2, 2], 2.0 * self.ECm)
        return self.EC_matrix()[2, 2]

    def fluxonium_minus_charging_energy(self):
        assert np.allclose(
            self.EC_matrix()[3, 3],
            0.5 * (1.0 / (4.0 * self.ECm) + 1.0 / (2.0 * self.ECc)) ** (-1),
        )
        return self.EC_matrix()[3, 3]

    def h_o_plus(self):
        return Oscillator(
            E_osc=np.sqrt(
                8.0 * self.h_o_plus_charging_energy() * self.EL_tilda() / 4.0
            ),
            truncated_dim=self.h_o_truncated_dim,
        )

    def phi_plus(self):
        h_o_plus = self.h_o_plus()
        return (
            (1.0 / np.sqrt(2.0))
            * (16.0 * self.ECm / (self.EL_tilda() / 4.0)) ** (1 / 4)
            * (h_o_plus.annihilation_operator() + h_o_plus.creation_operator())
        )


class FluxoniumTunableCouplerGrounded(
    FluxoniumTunableCouplerFloating, serializers.Serializable, base.QubitBaseClass
):
    def __init__(
        self,
        EJa,
        EJb,
        EC_twoqubit,
        ECq1,
        ECq2,
        ELa,
        ELb,
        flux_a,
        flux_b,
        flux_c,
        fluxonium_cutoff,
        fluxonium_truncated_dim,
        ECc,
        ECm,
        EL1,
        EL2,
        EJC,
        fluxonium_minus_truncated_dim=6,
        h_o_truncated_dim=3,
    ):
        FluxoniumTunableCouplerFloating.__init__(
            self,
            EJa,
            EJb,
            np.inf,
            0.0,
            0.0,
            ECq1,
            ECq2,
            ELa,
            ELb,
            flux_a,
            flux_b,
            flux_c,
            fluxonium_cutoff,
            fluxonium_truncated_dim,
            ECc,
            ECm,
            EL1,
            EL2,
            EJC,
            fluxonium_minus_truncated_dim=fluxonium_minus_truncated_dim,
            h_o_truncated_dim=h_o_truncated_dim,
        )
        self.EC_twoqubit = EC_twoqubit
        self._sys_type = type(self).__name__

    def capacitance_matrix(self):
        C_matrix = np.zeros((4, 4))
        C_matrix[0, 0] = 1.0 / (2.0 * self.ECq1) + 1.0 / (2.0 * self.EC_twoqubit)
        C_matrix[1, 1] = 1.0 / (2.0 * self.ECq2) + 1.0 / (2.0 * self.EC_twoqubit)
        C_matrix[1, 0] = C_matrix[0, 1] = -1.0 / (2.0 * self.EC_twoqubit)
        C_matrix[2, 2] = 1.0 / (2.0 * self.ECm) / 2.0
        C_matrix[3, 3] = 1.0 / (2.0 * self.ECm) / 2.0 + 1.0 / (2.0 * self.ECc)
        return C_matrix

    def hilbertdim(self) -> int:
        pass

    def hamiltonian(self):
        pass

    def default_params(self):
        pass
