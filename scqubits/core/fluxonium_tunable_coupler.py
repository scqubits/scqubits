import numpy as np
from itertools import product

from qutip.states import basis
from qutip import tensor

import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.utils.spectrum_utils import get_matrixelement_table
from scqubits.io_utils.fileio_qutip import QutipEigenstates


class FluxoniumTunableCouplerFloating(serializers.Serializable):
    def __init__(self, EJa, EJb, ECg, ELa, ELb, flux_a, flux_b, flux_c,
                 fluxonium_cutoff, fluxonium_truncated_dim, ECc, ECm, EL1, EL2, EJC,
                 fluxonium_minus_truncated_dim=6, h_o_truncated_dim=3):
        self.EJa = EJa
        self.EJb = EJb
        self.ECg = ECg
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
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b, fluxonium_minus, h_o_plus])
        phi_minus = fluxonium_minus.phi_operator()
        phi_a = fluxonium_a.phi_operator()
        phi_b = fluxonium_b.phi_operator()
        n_a = fluxonium_a.n_operator()
        n_b = fluxonium_b.n_operator()
        interaction_term_1 = InteractionTerm(g_strength=0.5*self.ELa, subsys1=fluxonium_a, op1=phi_a,
                                             subsys2=h_o_plus, op2=self.phi_plus())
        interaction_term_2 = InteractionTerm(g_strength=-0.5*self.ELb, subsys1=fluxonium_b, op1=phi_b,
                                             subsys2=h_o_plus, op2=self.phi_plus())
        interaction_term_3 = InteractionTerm(g_strength=0.5 * self.ELa, subsys1=fluxonium_a, op1=phi_a,
                                             subsys2=fluxonium_minus, op2=phi_minus)
        interaction_term_4 = InteractionTerm(g_strength=0.5 * self.ELb, subsys1=fluxonium_b, op1=phi_b,
                                             subsys2=fluxonium_minus, op2=phi_minus)
        interaction_term_5 = InteractionTerm(g_strength=-8.0 * self.ECg, subsys1=fluxonium_a, op1=n_a,
                                             subsys2=fluxonium_b, op2=n_b)
        hilbert_space.interaction_list = [interaction_term_1, interaction_term_2,
                                          interaction_term_3, interaction_term_4, interaction_term_5]
        return hilbert_space

    def setup_effective_calculation(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        evals_a, evecs_a = fluxonium_a.eigensys(evals_count=fluxonium_a.truncated_dim)
        evals_b, evecs_b = fluxonium_b.eigensys(evals_count=fluxonium_b.truncated_dim)
        evals_minus, evecs_minus = fluxonium_minus.eigensys(evals_count=fluxonium_minus.truncated_dim)
        phi_a_mat = get_matrixelement_table(fluxonium_a.phi_operator(), evecs_a)
        phi_b_mat = get_matrixelement_table(fluxonium_b.phi_operator(), evecs_b)
        n_a_mat = get_matrixelement_table(fluxonium_a.n_operator(), evecs_a)
        n_b_mat = get_matrixelement_table(fluxonium_b.n_operator(), evecs_b)
        phi_minus_mat = get_matrixelement_table(fluxonium_minus.phi_operator(), evecs_minus)
        return evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, n_a_mat, n_b_mat, phi_minus_mat

    def brian_effective_hamiltonian(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        dim_a, dim_b = fluxonium_a.truncated_dim, fluxonium_b.truncated_dim
        (evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat,
         n_a_mat, n_b_mat, phi_minus_mat) = self.setup_effective_calculation()
        groundstate_expect = np.real(phi_minus_mat[0, 0])
        chi_m = sum(abs(phi_minus_mat[0, m]) ** 2 / (evals_minus[m] - evals_minus[0])
                    for m in range(1, fluxonium_minus.truncated_dim))

        E_La_shift = self.ELa ** 2 * (0.5*chi_m + 1.0/self.EL_tilda())
        fluxonium_a.EL = self.ELa - E_La_shift
        E_Lb_shift = self.ELb ** 2 * (0.5*chi_m + 1.0/self.EL_tilda())
        fluxonium_b.EL = self.ELb - E_Lb_shift
        beta = 0.5 * fluxonium_a.EL
        flux_shift = self.ELa*groundstate_expect/(4*beta)

        J = self.ELa * self.ELb * (1.0/(self.EL_tilda()) - 0.5*chi_m)

        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        hamiltonian_a = hilbert_space.diag_hamiltonian(fluxonium_a)
        phi_a_ops = sum([phi_a_mat[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_a)
                        for j in range(dim_a) for k in range(dim_a)])
        n_a_ops = sum([n_a_mat[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_a)
                       for j in range(dim_a) for k in range(dim_a)])

        hamiltonian_b = hilbert_space.diag_hamiltonian(fluxonium_b)
        phi_b_ops = sum([phi_b_mat[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_b)
                         for j in range(dim_b) for k in range(dim_b)])
        n_b_ops = sum([n_b_mat[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_b)
                       for j in range(dim_b) for k in range(dim_b)])

        hamiltonian_ab = (J * (phi_a_ops * phi_b_ops)
                          + 0.5 * groundstate_expect * (self.ELa * phi_a_ops + self.ELb * phi_b_ops)
                          - 8.0 * self.ECg * n_a_ops * n_b_ops)

        return hamiltonian_a + hamiltonian_b + hamiltonian_ab, J*phi_a_mat[0, 1]*phi_b_mat[0, 1], flux_shift

    def fluxonium_a(self):
        return Fluxonium(self.EJa, 2*self.ECg, self.ELa, self.flux_a, cutoff=self.fluxonium_cutoff,
                         truncated_dim=self.fluxonium_truncated_dim)

    def fluxonium_b(self):
        return Fluxonium(self.EJb, 2*self.ECg, self.ELb, self.flux_b, cutoff=self.fluxonium_cutoff,
                         truncated_dim=self.fluxonium_truncated_dim)

    def EL_tilda(self):
        return self.EL1 + self.EL2 + self.ELa + self.ELb

    def fluxonium_minus(self):
        Cm = 1. / self.ECm
        Cc = 1. / self.ECc
        ECminus = 2. / (Cm + 2. * Cc)
        return Fluxonium(self.EJC, ECminus, self.EL_tilda() / 4.0, self.flux_c,
                         cutoff=self.fluxonium_cutoff, truncated_dim=self.fluxonium_minus_truncated_dim)

    def h_o_plus(self):
        return Oscillator(E_osc=np.sqrt(4.0 * self.ECm * self.EL_tilda()), truncated_dim=self.h_o_truncated_dim)

    def phi_plus(self):
        h_o_plus = self.h_o_plus()
        return (16. * self.ECm / self.EL_tilda()) ** (1 / 4) * (h_o_plus.annihilation_operator()
                                                                + h_o_plus.creation_operator())

    def find_off_position(self):
        flux_list = np.linspace(0.0, 0.5, 150)
        off_flux_positions = []
        coupling_strengths = []
        for flux in flux_list:
            self.flux_c = flux
            hilbert_space = self.generate_coupled_system()
#            hilbert_space.generate_lookup()
            dressed_evals = hilbert_space.eigenvals(evals_count=10)
            dressed_evals = dressed_evals - dressed_evals[0]
            coupling_strength = np.abs(dressed_evals[2]-dressed_evals[1])/2
            coupling_strengths.append(coupling_strength)
            if coupling_strength < 0.001:
                off_flux_positions.append(flux)
        return np.array(off_flux_positions), np.array(coupling_strengths)


class FluxoniumTunableCoupler(serializers.Serializable):
    def __init__(self, EJa, EJb, ECa, ECb, ELa, ELb, flux_a, flux_b, flux_c,
                 fluxonium_cutoff, fluxonium_truncated_dim, ECJ, EC, EL1, EL2, EJC,
                 fluxonium_minus_truncated_dim=6, h_o_truncated_dim=3):
        self.EJa = EJa
        self.EJb = EJb
        self.ECa = ECa
        self.ECb = ECb
        self.ELa = ELa
        self.ELb = ELb
        self.flux_a = flux_a
        self.flux_b = flux_b
        self.flux_c = flux_c
        self.fluxonium_cutoff = fluxonium_cutoff
        self.fluxonium_truncated_dim = fluxonium_truncated_dim
        self.fluxonium_minus_truncated_dim = fluxonium_minus_truncated_dim
        self.h_o_truncated_dim = h_o_truncated_dim
        self.ECJ = ECJ
        self.EC = EC
        self.EL1 = EL1
        self.EL2 = EL2
        self.EJC = EJC

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
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b, fluxonium_minus, h_o_plus])
        phi_plus = (self.EC / self.EL_tilda()) ** (1 / 4) * (h_o_plus.annihilation_operator()
                                                             + h_o_plus.creation_operator())
        phi_minus = fluxonium_minus.phi_operator()
        phi_a = fluxonium_a.phi_operator()
        phi_b = fluxonium_b.phi_operator()
        interaction_term_1 = InteractionTerm(g_strength=-self.ELa, subsys1=fluxonium_a, op1=phi_a,
                                             subsys2=h_o_plus, op2=phi_plus)
        interaction_term_2 = InteractionTerm(g_strength=-self.ELb, subsys1=fluxonium_b, op1=phi_b,
                                             subsys2=h_o_plus, op2=phi_plus)
        interaction_term_3 = InteractionTerm(g_strength=-0.5 * self.ELa, subsys1=fluxonium_a, op1=phi_a,
                                             subsys2=fluxonium_minus, op2=phi_minus)
        interaction_term_4 = InteractionTerm(g_strength=0.5 * self.ELb, subsys1=fluxonium_b, op1=phi_b,
                                             subsys2=fluxonium_minus, op2=phi_minus)
        hilbert_space.interaction_list = [interaction_term_1, interaction_term_2,
                                          interaction_term_3, interaction_term_4]
        return hilbert_space

    def fluxonium_a(self):
        return Fluxonium(self.EJa, self.ECa, self.ELa, self.flux_a, cutoff=self.fluxonium_cutoff,
                         truncated_dim=self.fluxonium_truncated_dim)

    def fluxonium_b(self):
        return Fluxonium(self.EJb, self.ECb, self.ELb, self.flux_b, cutoff=self.fluxonium_cutoff,
                         truncated_dim=self.fluxonium_truncated_dim)

    def EC_tilda(self):
        return self.ECJ * self.EC / (2 * self.EC + self.ECJ)

    def EL_tilda(self):
        return self.EL1 + self.EL2 + self.ELa + self.ELb

    def fluxonium_minus(self):
        return Fluxonium(self.EJC, self.EC_tilda() / 8.0, self.EL_tilda() / 4.0, self.flux_c,
                         cutoff=self.fluxonium_cutoff, truncated_dim=self.fluxonium_minus_truncated_dim)

    def h_o_plus(self):
        return Oscillator(E_osc=np.sqrt(4.0 * self.EC * self.EL_tilda()), truncated_dim=self.h_o_truncated_dim)

    def phi_plus(self):
        h_o_plus = self.h_o_plus()
        return (self.EC / self.EL_tilda()) ** (1 / 4) * (h_o_plus.annihilation_operator()
                                                         + h_o_plus.creation_operator())

    def setup_effective_calculation(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        evals_a, evecs_a = fluxonium_a.eigensys(evals_count=fluxonium_a.truncated_dim)
        evals_b, evecs_b = fluxonium_b.eigensys(evals_count=fluxonium_b.truncated_dim)
        evals_minus, evecs_minus = fluxonium_minus.eigensys(evals_count=fluxonium_minus.truncated_dim)
        phi_a_mat = get_matrixelement_table(fluxonium_a.phi_operator(), evecs_a)
        phi_b_mat = get_matrixelement_table(fluxonium_b.phi_operator(), evecs_b)
        phi_minus_mat = get_matrixelement_table(fluxonium_minus.phi_operator(), evecs_minus)
        return evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat

    def get_coefficient_term_in_effective_hamiltonian(self, ell, m, ell_prime, m_prime, effective_hamiltonian=None):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        if not effective_hamiltonian:
            effective_hamiltonian = self.effective_hamiltonian()
        bra_ell_m = tensor(basis(fluxonium_a.truncated_dim, ell).dag(), basis(fluxonium_b.truncated_dim, m).dag())
        ket_ell_prime_m_prime = tensor(basis(fluxonium_a.truncated_dim, ell_prime),
                                       basis(fluxonium_b.truncated_dim, m_prime))
        return bra_ell_m * effective_hamiltonian * ket_ell_prime_m_prime

    @staticmethod
    def kron_delta(ell, ell_prime):
        if ell == ell_prime:
            return 1.0
        else:
            return 0.0

    def brian_effective_hamiltonian(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        h_o_plus = self.h_o_plus()
        dim_a, dim_b = fluxonium_a.truncated_dim, fluxonium_b.truncated_dim
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = self.setup_effective_calculation()
        groundstate_expect = (np.real(phi_minus_mat[0, 0]) / (2 * np.pi))
        osc_len = (4.0 * self.EC / self.EL_tilda()) ** (1 / 4)
        chi_m = sum(abs(phi_minus_mat[0, m]) ** 2 / (evals_minus[m] - evals_minus[0])
                    for m in range(1, fluxonium_minus.truncated_dim))
        chi_p = 0.5 * (osc_len ** 2 / h_o_plus.E_osc)

        E_La_shift = self.ELa ** 2 * (0.5 * chi_m + 2 * chi_p)
        fluxonium_a.EL = self.ELa - E_La_shift
        E_Lb_shift = self.ELb ** 2 * (0.5 * chi_m + 2 * chi_p)
        fluxonium_b.EL = self.ELb - E_Lb_shift

        J = self.ELa * self.ELb * (0.5 * chi_m - 2 * chi_p)
        flux_shift_a = 0.5 * (self.ELa / fluxonium_a.EL) * groundstate_expect
#        print(flux_shift_a)
#        fluxonium_a.flux = self.flux_a + flux_shift_a
        fluxonium_a.flux = 0.5

        flux_shift_b = 0.5 * (self.ELb / fluxonium_b.EL) * groundstate_expect
#        print(flux_shift_b)
#        fluxonium_b.flux = self.flux_b - flux_shift_b
        fluxonium_b.flux = 0.5

        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        hamiltonian_a = hilbert_space.diag_hamiltonian(fluxonium_a)
        _, evecs_a = fluxonium_a.eigensys(dim_a)
        mat_a = get_matrixelement_table(fluxonium_a.phi_operator(), evecs_a)
        va = sum([mat_a[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_a)
                  for j in range(dim_a) for k in range(dim_a)])

        hamiltonian_b = hilbert_space.diag_hamiltonian(fluxonium_b)
        _, evecs_b = fluxonium_b.eigensys(dim_b)
        mat_a = get_matrixelement_table(fluxonium_b.phi_operator(), evecs_b)
        vb = sum([mat_a[j][k] * hilbert_space.hubbard_operator(j, k, fluxonium_b)
                  for j in range(dim_b) for k in range(dim_b)])

        hamiltonian_ab = J * (va * vb) + J * 2 * np.pi * (flux_shift_a * vb - flux_shift_b * va)
        return hamiltonian_a + hamiltonian_b + hamiltonian_ab, flux_shift_a, flux_shift_b

    def get_matrix_element_of_perturbation(self, ell, m, n, p, ell_prime, m_prime, n_prime, p_prime,
                                           effective_quantities):
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = effective_quantities
        phi_plus = self.phi_plus()
        return (- self.ELa * (phi_a_mat[ell, ell_prime] * phi_plus[p, p_prime]
                              * self.kron_delta(n, n_prime) * self.kron_delta(m, m_prime))
                - 0.5 * self.ELa * (phi_a_mat[ell, ell_prime] * phi_minus_mat[n, n_prime]
                                    * self.kron_delta(p, p_prime) * self.kron_delta(m, m_prime))
                - self.ELb * (phi_b_mat[m, m_prime] * phi_plus[p, p_prime]
                              * self.kron_delta(n, n_prime) * self.kron_delta(ell, ell_prime))
                + 0.5 * self.ELb * (phi_b_mat[m, m_prime] * phi_minus_mat[n, n_prime]
                                    * self.kron_delta(ell, ell_prime) * self.kron_delta(p, p_prime))
                )

    def _second_order_matrix_element_for_path(self, ell, m, int_state, ell_prime, m_prime, effective_quantities):
        return (self.get_matrix_element_of_perturbation(ell, m, 0, 0, *int_state, effective_quantities)
                * self.get_matrix_element_of_perturbation(*int_state, ell_prime, m_prime, 0, 0, effective_quantities))

    def _third_order_matrix_element_for_path(self, ell, m, int_state_1, int_state_2, ell_prime, m_prime,
                                             effective_quantities):
        return (self.get_matrix_element_of_perturbation(ell, m, 0, 0, *int_state_1, effective_quantities)
                * self.get_matrix_element_of_perturbation(*int_state_1, *int_state_2, effective_quantities)
                * self.get_matrix_element_of_perturbation(*int_state_2, ell_prime, m_prime, 0, 0, effective_quantities))

    def filter_helper(self, int_state):
        ell, m, n, p = int_state
        return n != 0 or p != 0

    def third_order_sum_up_down_over(self, ell, ell_prime, m, m_prime, effective_quantities):
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = effective_quantities
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        fluxonium_minus = self.fluxonium_minus()
        h_o_plus = self.h_o_plus()
        E_initial = evals_a[ell] + evals_b[m] + evals_minus[0]
        E_final = evals_a[ell_prime] + evals_b[m_prime] + evals_minus[0]
        # This virtual intermediate state explores the high energy subspace
        virtual_int_states_1 = product(np.arange(0, fluxonium_a.truncated_dim), np.arange(0, fluxonium_b.truncated_dim),
                                       np.arange(0, fluxonium_minus.truncated_dim),
                                       np.arange(0, h_o_plus.truncated_dim))
        virtual_int_states_1 = filter(self.filter_helper, virtual_int_states_1)
        # This virtual state is back down in the low energy subspace
        virtual_int_states_2 = product(np.arange(0, fluxonium_a.truncated_dim), np.arange(0, fluxonium_b.truncated_dim),
                                       [0], [0])
        effective_hamiltonian = 0.0
        for int_state_1 in virtual_int_states_1:
            for int_state_2 in virtual_int_states_2:
                E_int_state_1 = (evals_a[int_state_1[0]] + evals_b[int_state_1[1]]
                                 + evals_minus[int_state_1[2]] + h_o_plus.E_osc * int_state_1[3])
                E_int_state_2 = evals_a[int_state_2[0]] + evals_b[int_state_2[1]]
                coefficient_1 = -0.5*(self._third_order_matrix_element_for_path(ell, m, int_state_1, int_state_2,
                                                                                ell_prime, m_prime,
                                                                                effective_quantities)
                                      / ((E_final - E_int_state_1) * (E_int_state_2 - E_int_state_1)))
                coefficient_2 = -0.5*(self._third_order_matrix_element_for_path(ell, m, int_state_2, int_state_1,
                                                                                ell_prime, m_prime,
                                                                                effective_quantities)
                                      / ((E_initial - E_int_state_1) * (E_int_state_2 - E_int_state_1)))
                coefficient = coefficient_1 + coefficient_2
                effective_hamiltonian += coefficient * (hilbert_space.hubbard_operator(ell, ell_prime, fluxonium_a)
                                                        * hilbert_space.hubbard_operator(m, m_prime, fluxonium_b))
        return effective_hamiltonian

    def third_order_sum_up_over_down(self, ell, ell_prime, m, m_prime, effective_quantities):
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = effective_quantities
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        fluxonium_minus = self.fluxonium_minus()
        h_o_plus = self.h_o_plus()
        E_initial = evals_a[ell] + evals_b[m] + evals_minus[0]
        E_final = evals_a[ell_prime] + evals_b[m_prime] + evals_minus[0]
        # This virtual intermediate state explores the high energy subspace
        virtual_int_states_1 = product(np.arange(0, fluxonium_a.truncated_dim), np.arange(0, fluxonium_b.truncated_dim),
                                       np.arange(0, fluxonium_minus.truncated_dim),
                                       np.arange(0, h_o_plus.truncated_dim))
        virtual_int_states_1 = filter(self.filter_helper, virtual_int_states_1)
        # As does this one
        virtual_int_states_2 = product(np.arange(0, fluxonium_a.truncated_dim), np.arange(0, fluxonium_b.truncated_dim),
                                       np.arange(0, fluxonium_minus.truncated_dim),
                                       np.arange(0, h_o_plus.truncated_dim))
        virtual_int_states_2 = filter(self.filter_helper, virtual_int_states_2)
        effective_hamiltonian = 0.0
        for int_state_1 in virtual_int_states_1:
            for int_state_2 in virtual_int_states_2:
                E_int_state_1 = (evals_a[int_state_1[0]] + evals_b[int_state_1[1]]
                                 + evals_minus[int_state_1[2]] + h_o_plus.E_osc * int_state_1[3])
                E_int_state_2 = (evals_a[int_state_2[0]] + evals_b[int_state_2[1]]
                                 + evals_minus[int_state_2[2]] + h_o_plus.E_osc * int_state_2[3])
                coefficient = 0.5*self._third_order_matrix_element_for_path(ell, m, int_state_1, int_state_2,
                                                                            ell_prime, m_prime, effective_quantities)
                E_denom_1 = (E_initial - E_int_state_1) * (E_initial - E_int_state_2)
                E_denom_2 = (E_final - E_int_state_1) * (E_final - E_int_state_2)
                coefficient = coefficient * ((1./E_denom_1) + (1./E_denom_2))
                effective_hamiltonian += coefficient * (hilbert_space.hubbard_operator(ell, ell_prime, fluxonium_a)
                                                        * hilbert_space.hubbard_operator(m, m_prime, fluxonium_b))
        return effective_hamiltonian

    def zeroth_order_effective_hamiltonian(self):
        """
        Returns
        -------
        Qobj
            effective hamiltonian at zeroth order
        """
        return HilbertSpace([self.fluxonium_a(), self.fluxonium_b()]).hamiltonian()

    def first_order_effective_hamiltonian(self):
        """
        Returns
        -------
        Qobj
            effective hamiltonian at first order
        """
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        effective_quantities = self.setup_effective_calculation()
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = effective_quantities
        effective_hamiltonian = -0.5 * self.ELa * (sum([phi_a_mat[ell, ell_prime] * phi_minus_mat[0, 0]
                                                        * hilbert_space.hubbard_operator(ell, ell_prime, fluxonium_a)
                                                        for ell in range(fluxonium_a.truncated_dim)
                                                        for ell_prime in range(fluxonium_a.truncated_dim)]))
        effective_hamiltonian += 0.5 * self.ELb * (sum([phi_b_mat[m, m_prime] * phi_minus_mat[0, 0]
                                                   * hilbert_space.hubbard_operator(m, m_prime, fluxonium_b)
                                                   for m in range(fluxonium_b.truncated_dim)
                                                   for m_prime in range(fluxonium_b.truncated_dim)]))
        return effective_hamiltonian

    def second_order_sum(self, ell, ell_prime, m, m_prime, effective_quantities):
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = effective_quantities
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        fluxonium_minus = self.fluxonium_minus()
        h_o_plus = self.h_o_plus()
        E_initial = evals_a[ell] + evals_b[m] + evals_minus[0]
        E_final = evals_a[ell_prime] + evals_b[m_prime] + evals_minus[0]
        # This virtual intermediate state explores the high energy subspace
        virtual_int_states = product(np.arange(0, fluxonium_a.truncated_dim), np.arange(0, fluxonium_b.truncated_dim),
                                     np.arange(0, fluxonium_minus.truncated_dim), np.arange(0, h_o_plus.truncated_dim))
        virtual_int_states = filter(self.filter_helper, virtual_int_states)
        effective_hamiltonian = 0.0
        for int_state in virtual_int_states:
            E_int_state = (evals_a[int_state[0]] + evals_b[int_state[1]]
                           + evals_minus[int_state[2]] + h_o_plus.E_osc * int_state[3])
            coefficient = 0.5 * self._second_order_matrix_element_for_path(ell, m, int_state, ell_prime, m_prime,
                                                                           effective_quantities)
            E_denom_1 = (E_initial - E_int_state)
            E_denom_2 = (E_final - E_int_state)
            coefficient = coefficient * ((1. / E_denom_1) + (1. / E_denom_2))
            effective_hamiltonian += coefficient * (hilbert_space.hubbard_operator(ell, ell_prime, fluxonium_a)
                                                    * hilbert_space.hubbard_operator(m, m_prime, fluxonium_b))
        return effective_hamiltonian

    def second_order_effective_hamiltonian(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        dim_a = 2
        dim_b = 2
        effective_quantities = self.setup_effective_calculation()
        return sum([self.second_order_sum(ell, ell_prime, m, m_prime, effective_quantities)
                    for ell in range(dim_a) for ell_prime in range(dim_a)
                    for m in range(dim_b) for m_prime in range(dim_b)])

    def third_order_effective_hamiltonian(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        dim_a = fluxonium_a.truncated_dim
        dim_b = fluxonium_b.truncated_dim
        effective_quantities = self.setup_effective_calculation()
        return sum([self.third_order_sum_up_down_over(ell, ell_prime, m, m_prime, effective_quantities)
                    + self.third_order_sum_up_over_down(ell, ell_prime, m, m_prime, effective_quantities)
                    for ell in range(dim_a) for ell_prime in range(dim_a)
                    for m in range(dim_b) for m_prime in range(dim_b)])

    def effective_hamiltonian(self):
        return (self.zeroth_order_effective_hamiltonian()
                + self.first_order_effective_hamiltonian()
                + self.second_order_effective_hamiltonian()
#                + self.third_order_effective_hamiltonian(qubit_a_dim, qubit_b_dim)
                )

    def effective_eigenvals(self, evals_count=6):
        hamiltonian_mat = self.effective_hamiltonian()
        return hamiltonian_mat.eigenenergies(eigvals=evals_count)

    def effective_eigensys(self, evals_count=6):
        hamiltonian_mat = self.effective_hamiltonian()
        evals, evecs = hamiltonian_mat.eigenstates(eigvals=evals_count)
        evecs = evecs.view(QutipEigenstates)
        return evals, evecs

    def find_off_position(self):
        flux_list = np.linspace(0.0, 0.5, 150)
        off_flux_positions = []
        coupling_strengths = []
        for flux in flux_list:
            self.flux_c = flux
            hilbert_space = self.generate_coupled_system()
#            hilbert_space.generate_lookup()
            dressed_evals = hilbert_space.eigenvals(evals_count=10)
            dressed_evals = dressed_evals - dressed_evals[0]
            coupling_strength = np.abs(dressed_evals[2]-dressed_evals[1])/2
            coupling_strengths.append(coupling_strength)
            if coupling_strength < 0.001:
                off_flux_positions.append(flux)
        return np.array(off_flux_positions), np.array(coupling_strengths)
