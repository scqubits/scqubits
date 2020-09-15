import numpy as np

import matplotlib.pyplot as plt

import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm


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
        """

        Returns
        -------

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

    def find_off_position(self):
        flux_list = np.linspace(0.0, 0.5, 250)
        off_flux_positions = []
        coupling_strengths = []
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        evals_a = fluxonium_a.eigenvals()
        evals_a = evals_a-evals_a[0]
        evals_b = fluxonium_b.eigenvals()
        evals_b = evals_b-evals_b[0]
        for flux in flux_list:
            self.flux_c = flux
            hilbert_space = self.generate_coupled_system()
            hilbert_space.generate_lookup()
            dressed_evals = hilbert_space.eigenvals(evals_count=10)
            dressed_evals = dressed_evals - dressed_evals[0]
#            dressed_index_a = hilbert_space.lookup.dressed_index((1, 0, 0, 0))
#            dressed_index_b = hilbert_space.lookup.dressed_index((0, 1, 0, 0))
            coupling_strength = np.abs(dressed_evals[2]-dressed_evals[1])/2
            coupling_strengths.append(coupling_strength)
#            difference_a = np.abs(evals_a[1]-dressed_evals[dressed_index_a])
#            difference_b = np.abs(evals_b[1]-dressed_evals[dressed_index_b])
            if coupling_strength < 0.001:
                off_flux_positions.append(flux)
        return off_flux_positions, flux_list, coupling_strengths


FTC = FluxoniumTunableCoupler(EJa=5.5, EJb=5.5, ECa=1.2, ECb=1.2, ELa=0.1, ELb=0.1, flux_a=0.5, flux_b=0.5, flux_c=0.0,
                              fluxonium_cutoff=110, fluxonium_truncated_dim=8, ECJ=10.0,
                              EC=5.0, EJC=3.0, EL1=4.0, EL2=4.0)
FTC.flux_c = 0.5
hilbert_space = FTC.generate_coupled_system()
hilbert_space.generate_lookup()
#bare_0 = hilbert_space.lookup.bare_index(0)
bare_1 = hilbert_space.lookup.bare_index(1)
#bare_2 = hilbert_space.lookup.bare_index(2)
#bare_3 = hilbert_space.lookup.bare_index(3)
#bare_4 = hilbert_space.lookup.bare_index(4)

#off_positions, flux_list, coupling_strengths = FTC.find_off_position()
#plt.plot(flux_list, coupling_strengths)
#plt.show()
#print(coupling_strengths)
print(0)

