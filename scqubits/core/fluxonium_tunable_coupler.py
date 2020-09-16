import numpy as np

from qutip.states import basis
from qutip import tensor

import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm
from scqubits.utils.spectrum_utils import get_matrixelement_table
from scqubits.io_utils.fileio_qutip import QutipEigenstates


class FluxoniumTunableCoupler(serializers.Serializable):
    def __init__(self, EJa, EJb, ECa, ECb, ELa, ELb, flux_a, flux_b, flux_c,
                 fluxonium_cutoff, fluxonium_truncated_dim, ECJ, EC, EL1, EL2, EJC):
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
        EL_tilda = self.EL1 + self.EL2 + self.ELa + self.ELb
        phi_plus = (self.EC / EL_tilda) ** (1 / 4) * (h_o_plus.annihilation_operator() + h_o_plus.creation_operator())
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

    def fluxonium_minus(self):
        EC_tilda = self.ECJ * self.EC / (2 * self.EC + self.ECJ)
        EL_tilda = self.EL1 + self.EL2 + self.ELa + self.ELb
        return Fluxonium(self.EJC, EC_tilda / 8.0, EL_tilda / 4.0, self.flux_c, cutoff=self.fluxonium_cutoff,
                         truncated_dim=self.fluxonium_truncated_dim)

    def h_o_plus(self):
        EL_tilda = self.EL1 + self.EL2 + self.ELa + self.ELb
        return Oscillator(E_osc=np.sqrt(4.0 * self.EC * EL_tilda), truncated_dim=3)

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

    def get_coefficient_term_in_effective_hamiltonian(self, ell, m, ell_prime, m_prime):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        effective_hamiltonian = self.effective_hamiltonian()
        bra_ell_m = tensor(basis(fluxonium_a.truncated_dim, ell).dag(), basis(fluxonium_b.truncated_dim, m).dag())
        ket_ell_prime_m_prime = tensor(basis(fluxonium_a.truncated_dim, ell_prime),
                                       basis(fluxonium_b.truncated_dim, m_prime))
        return bra_ell_m * effective_hamiltonian * ket_ell_prime_m_prime

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
        effective_hamiltonian = (sum([phi_a_mat[ell, ell_prime]
                                      * hilbert_space.hubbard_operator(ell, ell_prime, fluxonium_a)
                                      for ell in range(fluxonium_a.truncated_dim)
                                      for ell_prime in range(fluxonium_a.truncated_dim)])
                                 * (-0.5) * self.ELa * phi_minus_mat[0, 0])
        effective_hamiltonian += (sum([phi_b_mat[m, m_prime]
                                       * hilbert_space.hubbard_operator(m, m_prime, fluxonium_b)
                                       for m in range(fluxonium_b.truncated_dim)
                                       for m_prime in range(fluxonium_b.truncated_dim)])
                                  * 0.5 * self.ELb * phi_minus_mat[0, 0])
        return effective_hamiltonian

    def second_order_effective_hamiltonian(self):
        """
        Returns
        -------
        Qobj
            effective hamiltonian at second order
        """
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        effective_quantities = self.setup_effective_calculation()
        evals_a, evals_b, evals_minus, phi_a_mat, phi_b_mat, phi_minus_mat = effective_quantities
        effective_hamiltonian = (self._second_order_sum_over_states_plus_single_qubit(evals_a, phi_a_mat,
                                                                                      fluxonium_a, hilbert_space)
                                 * 0.25 * self.ELa**2)
        effective_hamiltonian += (self._second_order_sum_over_states_plus_single_qubit(evals_b, phi_b_mat,
                                                                                       fluxonium_b, hilbert_space)
                                  * 0.25 * self.ELb**2)
        effective_hamiltonian += (self._second_order_sum_over_states_minus_single_qubit(evals_a, evals_minus, phi_a_mat,
                                                                                        phi_minus_mat, fluxonium_a,
                                                                                        fluxonium_minus, hilbert_space)
                                  * 0.25 * self.ELa**2)
        effective_hamiltonian += (self._second_order_sum_over_states_minus_single_qubit(evals_b, evals_minus, phi_b_mat,
                                                                                        phi_minus_mat, fluxonium_b,
                                                                                        fluxonium_minus, hilbert_space)
                                  * 0.25 * self.ELb ** 2)
        effective_hamiltonian += (self._second_order_sum_over_states_minus_two_qubit(evals_a, evals_b, evals_minus,
                                                                                     phi_a_mat, phi_b_mat,
                                                                                     phi_minus_mat, fluxonium_a,
                                                                                     fluxonium_b, hilbert_space)
                                  * (-0.25) * self.ELa * self.ELb)
        effective_hamiltonian += (self._second_order_sum_over_states_plus_two_qubit(evals_a, evals_b, phi_a_mat,
                                                                                    phi_b_mat, fluxonium_a, fluxonium_b,
                                                                                    hilbert_space)
                                  * 0.5 * self.ELa * self.ELb)
        return effective_hamiltonian

    def effective_hamiltonian(self):
        return (self.zeroth_order_effective_hamiltonian()
                + self.first_order_effective_hamiltonian()
                + self.second_order_effective_hamiltonian())

    def effective_eigenvals(self, evals_count=6):
        hamiltonian_mat = self.effective_hamiltonian()
        return hamiltonian_mat.eigenenergies(eigvals=evals_count)

    def effective_eigensys(self, evals_count=6):
        hamiltonian_mat = self.effective_hamiltonian()
        evals, evecs = hamiltonian_mat.eigenstates(eigvals=evals_count)
        evecs = evecs.view(QutipEigenstates)
        return evals, evecs

    def _second_order_sum_over_states_plus_single_qubit(self, evals, phi_mat, subsys, hilbert_space):
        return sum([self._second_order_plus_single_qubit(ell, ell_prime, ell_prime_prime, evals, phi_mat)
                    * hilbert_space.hubbard_operator(ell, ell_prime_prime, subsys)
                    for ell in range(subsys.truncated_dim) for ell_prime_prime in range(subsys.truncated_dim)
                    for ell_prime in range(subsys.truncated_dim)])

    def _second_order_sum_over_states_minus_single_qubit(self, evals_subsys, evals_minus, phi_mat, phi_minus_mat,
                                                         subsys, fluxonium_minus, hilbert_space):
        return sum([self._second_order_minus_single_qubit(ell, ell_prime, ell_prime_prime, n_prime, evals_subsys,
                                                          evals_minus, phi_mat, phi_minus_mat)
                    * hilbert_space.hubbard_operator(ell, ell_prime_prime, subsys)
                    for ell in range(subsys.truncated_dim) for ell_prime_prime in range(subsys.truncated_dim)
                    for ell_prime in range(subsys.truncated_dim) for n_prime in range(fluxonium_minus.truncated_dim)])

    def _second_order_sum_over_states_minus_two_qubit(self, evals_subsys_a, evals_subsys_b, evals_minus,
                                                      phi_a_mat, phi_b_mat, phi_minus_mat, subsys_a,
                                                      subsys_b, hilbert_space):
        fluxonium_minus = self.fluxonium_minus()
        return sum([self._second_order_minus_two_qubit(ell, ell_prime_prime, m, m_prime_prime, n_prime,
                                                       evals_subsys_a, evals_subsys_b, evals_minus,
                                                       phi_a_mat, phi_b_mat, phi_minus_mat)
                    * hilbert_space.hubbard_operator(ell, ell_prime_prime, subsys_a)
                    * hilbert_space.hubbard_operator(m, m_prime_prime, subsys_b)
                    for ell in range(subsys_a.truncated_dim) for ell_prime_prime in range(subsys_a.truncated_dim)
                    for m in range(subsys_b.truncated_dim) for m_prime_prime in range(subsys_b.truncated_dim)
                    for n_prime in range(fluxonium_minus.truncated_dim)])

    def _second_order_sum_over_states_plus_two_qubit(self, evals_subsys_a, evals_subsys_b,
                                                     phi_a_mat, phi_b_mat, subsys_a,
                                                     subsys_b, hilbert_space):
        return sum([self._second_order_plus_two_qubit(ell, ell_prime_prime, m, m_prime_prime, evals_subsys_a,
                                                      evals_subsys_b, phi_a_mat, phi_b_mat)
                    * hilbert_space.hubbard_operator(ell, ell_prime_prime, subsys_a)
                    * hilbert_space.hubbard_operator(m, m_prime_prime, subsys_b)
                    for ell in range(subsys_a.truncated_dim) for ell_prime_prime in range(subsys_a.truncated_dim)
                    for m in range(subsys_b.truncated_dim) for m_prime_prime in range(subsys_b.truncated_dim)])

    @staticmethod
    def _second_order_minus_two_qubit(ell, ell_prime_prime, m, m_prime_prime, n_prime,
                                      evals_subsys_a, evals_subsys_b, evals_minus, phi_a_mat, phi_b_mat,
                                      phi_minus_mat):
        E_denom_inverse_minus_a = 0.5 * ((evals_subsys_a[ell] - evals_subsys_a[ell_prime_prime]
                                          - evals_minus[n_prime]) ** (-1)
                                         + (evals_subsys_a[ell_prime_prime] - evals_subsys_a[ell]
                                            - evals_minus[n_prime]) ** (-1))
        E_denom_inverse_minus_b = 0.5 * ((evals_subsys_b[m] - evals_subsys_b[m_prime_prime]
                                          - evals_minus[n_prime]) ** (-1)
                                         + (evals_subsys_b[m_prime_prime] - evals_subsys_a[m]
                                            - evals_minus[n_prime]) ** (-1))
        return ((E_denom_inverse_minus_a + E_denom_inverse_minus_b) * np.abs(phi_minus_mat[0, n_prime])**2
                * phi_a_mat[ell, ell_prime_prime] * phi_b_mat[m, m_prime_prime])

    def _second_order_plus_two_qubit(self, ell, ell_prime_prime, m, m_prime_prime,
                                     evals_subsys_a, evals_subsys_b, phi_a_mat, phi_b_mat):
        h_o_plus = self.h_o_plus()
        omega_plus = h_o_plus.E_osc
        EL_tilda = self.EL1 + self.EL2 + self.ELa + self.ELb
        E_denom_inverse_plus_a = 0.5*((evals_subsys_a[ell]-evals_subsys_a[ell_prime_prime]-omega_plus)**(-1)
                                      + (evals_subsys_a[ell_prime_prime]-evals_subsys_a[ell]-omega_plus)**(-1))
        E_denom_inverse_plus_b = 0.5*((evals_subsys_b[m]-evals_subsys_b[m_prime_prime]-omega_plus)**(-1)
                                      + (evals_subsys_b[m_prime_prime]-evals_subsys_a[m]-omega_plus)**(-1))
        return ((E_denom_inverse_plus_a + E_denom_inverse_plus_b) * np.sqrt(4 * self.EC / EL_tilda)
                * phi_a_mat[ell, ell_prime_prime] * phi_b_mat[m, m_prime_prime])

    def _second_order_plus_single_qubit(self, ell, ell_prime, ell_prime_prime, evals, phi_mat):
        h_o_plus = self.h_o_plus()
        omega_plus = h_o_plus.E_osc
        EL_tilda = self.EL1 + self.EL2 + self.ELa + self.ELb
        E_denom_inverse_plus = ((evals[ell] - evals[ell_prime] - omega_plus) ** (-1)
                                + (evals[ell_prime_prime] - evals[ell_prime] - omega_plus) ** (-1))
        coefficient_plus = (phi_mat[ell, ell_prime] * phi_mat[ell_prime, ell_prime_prime]
                            * np.sqrt(4 * self.EC / EL_tilda))
        return coefficient_plus * E_denom_inverse_plus

    @staticmethod
    def _second_order_minus_single_qubit(ell, ell_prime, ell_prime_prime, n_prime, evals_subsys, evals_minus,
                                         phi_mat, phi_minus_mat):
        E_denom_inverse_minus = ((evals_subsys[ell] - evals_subsys[ell_prime]
                                  - evals_minus[n_prime]) ** (-1)
                                 + (evals_subsys[ell_prime_prime] - evals_subsys[ell_prime]
                                    - evals_minus[n_prime])**(-1))
        coefficient_minus = 0.5*(phi_mat[ell, ell_prime] * phi_mat[ell_prime, ell_prime_prime]
                                 * np.abs(phi_minus_mat[0, n_prime])**2)
        return coefficient_minus * E_denom_inverse_minus

    def find_off_position(self):
        flux_list = np.linspace(0.0, 0.5, 250)
        off_flux_positions = []
        coupling_strengths = []
        for flux in flux_list:
            self.flux_c = flux
            hilbert_space = self.generate_coupled_system()
            hilbert_space.generate_lookup()
            dressed_evals = hilbert_space.eigenvals(evals_count=10)
            dressed_evals = dressed_evals - dressed_evals[0]
            coupling_strength = np.abs(dressed_evals[2]-dressed_evals[1])/2
            coupling_strengths.append(coupling_strength)
            if coupling_strength < 0.001:
                off_flux_positions.append(flux)
        return off_flux_positions, flux_list, coupling_strengths
