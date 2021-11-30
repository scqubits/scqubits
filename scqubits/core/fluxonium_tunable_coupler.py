from typing import Optional

import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, Qobj
from scipy.linalg import inv
from scipy.optimize import root, minimize
from sympy import Matrix, S, diff, hessian, simplify, solve, symbols

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.oscillator import Oscillator, convert_to_E_osc, convert_to_l_osc
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.utils.spectrum_utils import get_matrixelement_table, standardize_sign


class FluxoniumTunableCouplerFloating(base.QubitBaseClass, serializers.Serializable):
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
        id_str: Optional[str] = None,
    ):
        base.QuantumSystem.__init__(self, id_str=id_str)
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

    def default_params(self):
        pass

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

    @staticmethod
    def signed_evals_evecs_phimat_qubit_instance(qubit_instance):
        evals, evecs_uns = qubit_instance.eigensys(
            evals_count=qubit_instance.truncated_dim
        )
        evecs = np.zeros_like(evecs_uns).T
        for k, evec in enumerate(evecs_uns.T):
            evecs[k, :] = standardize_sign(evec)
        phi_mat = get_matrixelement_table(qubit_instance.phi_operator(), evecs.T)
        return evals, evecs.T, phi_mat

    def schrieffer_wolff_real_flux(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        flux_a, flux_b = fluxonium_a.flux, fluxonium_b.flux
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_minus = self.fluxonium_minus()
        evals_a, evecs_a, phi_a_mat = self.signed_evals_evecs_phimat_qubit_instance(fluxonium_a)
        evals_b, evecs_b, phi_b_mat = self.signed_evals_evecs_phimat_qubit_instance(fluxonium_b)
        evals_m, evecs_m, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_minus
        )
        H_0_a = np.diag(evals_a - evals_a[0])[0:2, 0:2]
        H_0_b = np.diag(evals_b - evals_b[0])[0:2, 0:2]
        H_0 = tensor(Qobj(H_0_a), qeye(2)) + tensor(qeye(2), Qobj(H_0_b))

        # first-order contribution yields sigmax
        H_1_a = (-0.5 * self.ELa * phi_minus_mat[0, 0] * phi_a_mat)[0:2, 0:2]
        # canceled by flux_offset
        H_1_a += - self.ELa * 2.0 * np.pi * (flux_a - 0.5) * phi_a_mat[0:2, 0:2]
        H_1_b = (0.5 * self.ELb * phi_minus_mat[0, 0] * phi_b_mat)[0:2, 0:2]
        H_1_b += - self.ELb * 2.0 * np.pi * (flux_b - 0.5) * phi_b_mat[0:2, 0:2]
        H_1 = tensor(Qobj(H_1_a), qeye(2)) + tensor(qeye(2), Qobj(H_1_b))

        # second order calculation
        H_2_a = self._H2_self_correction_real_flux_coupler(
            evals_a, evals_m, phi_a_mat, phi_minus_mat, fluxonium_a.EL
        )
        H_2_a += self._H2_self_correction_real_flux_highfluxonium(
            evals_a, phi_a_mat, phi_minus_mat, fluxonium_a.EL
        )
        H_2_b = self._H2_self_correction_real_flux_coupler(
            evals_b, evals_m, phi_b_mat, phi_minus_mat, fluxonium_b.EL
        )
        H_2_b += self._H2_self_correction_real_flux_highfluxonium(
            evals_b, phi_b_mat, phi_minus_mat, fluxonium_b.EL
        )
        H_2 = tensor(Qobj(H_2_a[0:2, 0:2]), qeye(2)) + tensor(
            qeye(2), Qobj(H_2_b[0:2, 0:2])
        )
        H_2_ab = self._H2_qubit_coupling_real_flux(
            evals_a,
            evals_b,
            evals_m,
            phi_a_mat,
            phi_b_mat,
            phi_minus_mat,
            fluxonium_a.EL,
            fluxonium_b.EL,
        )
        H_2 += Qobj(H_2_ab, dims=[[2, 2], [2, 2]])
        H_eff = H_0 + H_1 + H_2
        return H_eff

    def _H2_self_correction_real_flux_inductive_disorder(
        self, evals_q, phi_q_mat, phi_minus_mat, EL
    ):
        H_2_ = sum(
            -0.5
            * self._g_plus_minus(phi_minus_mat)
            * self._g_plus(ell, ellp, phi_q_mat, EL)
            * (
                    1.0 / (evals_q[ell] - evals_q[ellp] - self.h_o_plus().E_osc)
                    + 1.0 / (evals_q[ellp] - evals_q[ell] - self.h_o_plus().E_osc)
            )
            * basis(2, ell)
            * basis(2, ellp).dag()
            for ell in range(2)
            for ellp in range(2)
        )
        return H_2_

    def _H2_self_correction_real_flux_highfluxonium(
        self, evals_q, phi_q_mat, phi_minus_mat, EL
    ):
        H_2_ = sum(
            0.5
            * self._g_minus(ell, ellpp, 0, phi_q_mat, phi_minus_mat, EL)
            * self._g_minus(ellpp, ellp, 0, phi_q_mat, phi_minus_mat, EL)
            * (
                1.0 / (evals_q[ell] - evals_q[ellpp])
                + 1.0 / (evals_q[ellp] - evals_q[ellpp])
            )
            * basis(2, ell)
            * basis(2, ellp).dag()
            for ell in range(2)
            for ellp in range(2)
            for ellpp in range(2, self.fluxonium_truncated_dim)
        )
        return H_2_

    def _H2_self_correction_real_flux_coupler(
        self, evals_q, evals_m, phi_q_mat, phi_minus_mat, EL
    ):
        H_2_ = sum(
            0.5
            * self._g_minus(ell, ellpp, n, phi_q_mat, phi_minus_mat, EL)
            * self._g_minus(ellpp, ellp, n, phi_q_mat, phi_minus_mat, EL)
            * (
                1.0 / (evals_q[ell] + evals_m[0] - evals_q[ellpp] - evals_m[n])
                + 1.0 / (evals_q[ellp] + evals_m[0] - evals_q[ellpp] - evals_m[n])
            )
            * basis(2, ell)
            * basis(2, ellp).dag()
            for ell in range(2)
            for ellp in range(2)
            for ellpp in range(self.fluxonium_truncated_dim)
            for n in range(1, self.fluxonium_minus_truncated_dim)
        )
        H_2_ += sum(
            0.5
            * self._g_plus(ell, ellpp, phi_q_mat, EL)
            * self._g_plus(ellpp, ellp, phi_q_mat, EL)
            * (
                1.0 / (evals_q[ell] - evals_q[ellpp] - self.h_o_plus().E_osc)
                + 1.0 / (evals_q[ellp] - evals_q[ellpp] - self.h_o_plus().E_osc)
            )
            * basis(2, ell)
            * basis(2, ellp).dag()
            for ell in range(2)
            for ellp in range(2)
            for ellpp in range(self.fluxonium_truncated_dim)
        )
        return H_2_

    def _H2_qubit_coupling_real_flux(
        self, evals_a, evals_b, evals_m, phi_a_mat, phi_b_mat, phi_m_mat, ELa, ELb
    ):
        H_2_ = 0.5 * sum(
            -self._g_minus(ell, ellp, n, phi_a_mat, phi_m_mat, ELa)
            * self._g_minus(m, mp, n, phi_b_mat, phi_m_mat, ELb)
            * (
                1.0 / (evals_a[ell] + evals_m[0] - evals_a[ellp] - evals_m[n])
                + 1.0 / (evals_b[m] + evals_m[0] - evals_b[mp] - evals_m[n])
                + 1.0 / (evals_a[ellp] + evals_m[0] - evals_a[ell] - evals_m[n])
                + 1.0 / (evals_b[mp] + evals_m[0] - evals_b[m] - evals_m[n])
            )
            * tensor(basis(2, ell), basis(2, m))
            * tensor(basis(2, ellp), basis(2, mp)).dag()
            for ell in range(2)
            for m in range(2)
            for ellp in range(2)
            for mp in range(2)
            for n in range(1, self.fluxonium_minus_truncated_dim)
        )
        H_2_ += 0.5 * sum(
            self._g_plus(ell, ellp, phi_a_mat, ELa)
            * self._g_plus(m, mp, phi_b_mat, ELb)
            * (
                1.0 / (evals_a[ell] - evals_a[ellp] - self.h_o_plus().E_osc)
                + 1.0 / (evals_b[m] - evals_b[mp] - self.h_o_plus().E_osc)
                + 1.0 / (evals_a[ellp] - evals_a[ell] - self.h_o_plus().E_osc)
                + 1.0 / (evals_b[mp] - evals_b[m] - self.h_o_plus().E_osc)
            )
            * tensor(basis(2, ell), basis(2, m))
            * tensor(basis(2, ellp), basis(2, mp)).dag()
            for ell in range(2)
            for m in range(2)
            for ellp in range(2)
            for mp in range(2)
        )
        return H_2_

    def _high_high_self_matelem(
        self, evals_minus, evals_q, phi_q_mat, phi_minus_mat, ELq, ell, ellp, nprime
    ):
        fadim = self.fluxonium_truncated_dim
        fmdim = self.fluxonium_minus_truncated_dim
        matelem = sum(
            self._g_minus_2(ell, ellpp, 0, nprimeprime, phi_q_mat, phi_minus_mat, ELq)
            * self._g_minus_2(
                ellpp, ellp, nprimeprime, nprime, phi_q_mat, phi_minus_mat, ELq
            )
            / (
                (
                    evals_q[ell]
                    + evals_minus[0]
                    - evals_q[ellpp]
                    - evals_minus[nprimeprime]
                )
                * (evals_q[ell] + evals_minus[0] - evals_q[ellp] - evals_minus[nprime])
            )
            for ellpp in range(fadim)
            for nprimeprime in range(1, fmdim)
        )
        return matelem

    def _high_high_cross_matelem(
        self,
        evals_minus,
        evals_a,
        evals_b,
        phi_a_mat,
        phi_b_mat,
        phi_minus_mat,
        ELa,
        ELb,
        ell,
        m,
        ellp,
        mp,
        nprime,
    ):
        fmdim = self.fluxonium_minus_truncated_dim
        matelem = sum(
            self._g_minus_2(ell, ellp, 0, nprimeprime, phi_a_mat, phi_minus_mat, ELa)
            * self._g_minus_2(m, mp, nprimeprime, nprime, phi_b_mat, phi_minus_mat, ELb)
            / (
                (
                    evals_a[ell]
                    + evals_minus[0]
                    - evals_a[ellp]
                    - evals_minus[nprimeprime]
                )
                * (
                    evals_a[ell]
                    + evals_b[m]
                    + evals_minus[0]
                    - evals_a[ellp]
                    - evals_b[mp]
                    - evals_minus[nprime]
                )
            )
            for nprimeprime in range(1, fmdim)
        )
        return matelem

    def _low_high_cross_matelem(
        self,
        evals_minus,
        evals_a,
        evals_b,
        phi_a_mat,
        phi_b_mat,
        phi_minus_mat,
        ELa,
        ELb,
        ell,
        m,
        ellp,
        mp,
        n,
    ):
        matelem = (
            self._g_minus(ell, ellp, 0, phi_a_mat, phi_minus_mat, ELa)
            * self._g_minus(m, mp, n, phi_b_mat, phi_minus_mat, ELb)
            / (
                (
                    evals_a[ell]
                    + evals_b[m]
                    + evals_minus[0]
                    - evals_a[ellp]
                    - evals_b[mp]
                    - evals_minus[n]
                )
                * (evals_b[m] + evals_minus[0] - evals_b[mp] - evals_minus[n])
            )
        )
        matelem += (
            self._g_minus(ell, ellp, n, phi_a_mat, phi_minus_mat, ELa)
            * self._g_minus(m, mp, 0, phi_b_mat, phi_minus_mat, ELb)
            / (
                (
                    evals_a[ell]
                    + evals_b[m]
                    + evals_minus[0]
                    - evals_a[ellp]
                    - evals_b[mp]
                    - evals_minus[n]
                )
                * (evals_a[ell] + evals_minus[0] - evals_a[ellp] - evals_minus[n])
            )
        )
        return matelem

    def _low_high_self_matelem(
        self, evals_minus, evals_q, phi_q_mat, phi_minus_mat, ELq, ell, ellp, n
    ):
        fadim = self.fluxonium_truncated_dim
        matelem = sum(
            self._g_minus(ell, ellpp, 0, phi_q_mat, phi_minus_mat, ELq)
            * self._g_minus(ellpp, ellp, n, phi_q_mat, phi_minus_mat, ELq)
            / (
                (evals_q[ell] + evals_minus[0] - evals_q[ellp] - evals_minus[n])
                * (evals_q[ellpp] + evals_minus[0] - evals_q[ellp] - evals_minus[n])
            )
            for ellpp in range(fadim)
        )
        return matelem

    def _low_high_matelem(
        self,
        evals_minus,
        evals_a,
        evals_b,
        phi_a_mat,
        phi_b_mat,
        phi_minus_mat,
        ELa,
        ELb,
        ell,
        m,
        ellp,
        mp,
        n,
    ):
        matelem = self._low_high_self_matelem(
            evals_minus, evals_a, phi_a_mat, phi_minus_mat, ELa, ell, ellp, n
        )
        matelem += self._low_high_self_matelem(
            evals_minus, evals_b, phi_b_mat, phi_minus_mat, ELb, m, mp, n
        )
        matelem += -self._low_high_cross_matelem(
            evals_minus,
            evals_a,
            evals_b,
            phi_a_mat,
            phi_b_mat,
            phi_minus_mat,
            ELa,
            ELb,
            ell,
            m,
            ellp,
            mp,
            n,
        )
        return matelem

    def _high_high_matelem(
        self,
        evals_minus,
        evals_a,
        evals_b,
        phi_a_mat,
        phi_b_mat,
        phi_minus_mat,
        ELa,
        ELb,
        ell,
        m,
        ellp,
        mp,
        nprime,
    ):
        matelem = self._high_high_self_matelem(
            evals_minus, evals_a, phi_a_mat, phi_minus_mat, ELa, ell, ellp, nprime
        )
        matelem += self._high_high_self_matelem(
            evals_minus, evals_b, phi_b_mat, phi_minus_mat, ELb, m, mp, nprime
        )
        matelem += -self._high_high_cross_matelem(
            evals_minus,
            evals_a,
            evals_b,
            phi_a_mat,
            phi_b_mat,
            phi_minus_mat,
            ELa,
            ELb,
            ell,
            m,
            ellp,
            mp,
            nprime,
        )
        return matelem

    def _eps_ab_2(
        self,
        ell,
        m,
        ellprime,
        mprime,
        n,
        evals_minus,
        evals_a,
        evals_b,
        phi_a_mat,
        phi_b_mat,
        phi_minus_mat,
    ):
        assert ell != ellprime and m != mprime
        low_high = self._low_high_cross_matelem(
            evals_minus,
            evals_a,
            evals_b,
            phi_a_mat,
            phi_b_mat,
            phi_minus_mat,
            self.ELa,
            self.ELb,
            ell,
            m,
            ellprime,
            mprime,
            n,
        )
        high_high = self._high_high_cross_matelem(
            evals_minus,
            evals_a,
            evals_b,
            phi_a_mat,
            phi_b_mat,
            phi_minus_mat,
            self.ELa,
            self.ELb,
            ell,
            m,
            ellprime,
            mprime,
            n,
        )
        return low_high - high_high

    def _generate_fluxonia_evals_phi_for_SW(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        # TODO this is a hack!
        fluxonium_a.flux = 0.5
        fluxonium_b.flux = 0.5
        fluxonium_minus = self.fluxonium_minus()
        evals_a, _, phi_a_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_a
        )
        evals_b, _, phi_b_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_b
        )
        evals_minus, _, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_minus
        )
        return evals_a, phi_a_mat, evals_b, phi_b_mat, evals_minus, phi_minus_mat

    def second_order_generator(self):
        fadim = fbdim = self.fluxonium_truncated_dim
        fmdim = self.fluxonium_minus_truncated_dim
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        low_high = sum(
            self._low_high_matelem(
                evals_minus,
                evals_a,
                evals_b,
                phi_a_mat,
                phi_b_mat,
                phi_minus_mat,
                self.ELa,
                self.ELb,
                ell,
                m,
                ellp,
                mp,
                n,
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ellp, mp, n, 0).dag()
            for ell in range(2)
            for m in range(2)
            for ellp in range(fadim)
            for mp in range(fbdim)
            for n in range(1, fmdim)
        )
        S2 = -low_high + low_high.dag()
        high_high = sum(
            self._high_high_matelem(
                evals_minus,
                evals_a,
                evals_b,
                phi_a_mat,
                phi_b_mat,
                phi_minus_mat,
                self.ELa,
                self.ELb,
                ell,
                m,
                ellp,
                mp,
                nprime,
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ellp, mp, nprime, 0).dag()
            for ell in range(2)
            for m in range(2)
            for ellp in range(fadim)
            for mp in range(fbdim)
            for nprime in range(1, fmdim)
        )
        S2 += high_high - high_high.dag()
        return S2

    def _single_qubit_first_order(
        self, evals_minus, evals_i, phi_i_mat, phi_minus_mat, ELi, i, j
    ):
        fmdim = self.fluxonium_minus_truncated_dim
        return sum(
            phi_minus_mat[0, n]
            * (
                self._eps_1(
                    evals_minus, evals_i, phi_i_mat, phi_minus_mat, ELi, i, j, n
                )
                + self._eps_1(
                    evals_minus, evals_i, phi_i_mat, phi_minus_mat, self.ELa, j, i, n
                )
            )
            for n in range(1, fmdim)
        )

    def XI_matrix_element(self, ell):
        (
            evals_a,
            phi_a_mat,
            _,
            _,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        ellprime = (ell + 1) % 2
        zeroth_order = 0.5 * self.ELa * phi_a_mat[0, 1]
        first_order = (
            -0.25
            * self.EL_tilda()
            * self._single_qubit_first_order(
                evals_minus, evals_a, phi_a_mat, phi_minus_mat, self.ELa, ell, ellprime
            )
        )
        return zeroth_order + first_order

    def IX_matrix_element(self, m):
        (
            _,
            _,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        mprime = (m + 1) % 2
        zeroth_order = -0.5 * self.ELa * phi_b_mat[0, 1]
        first_order = (
            0.25
            * self.EL_tilda()
            * self._single_qubit_first_order(
                evals_minus, evals_b, phi_b_mat, phi_minus_mat, self.ELa, m, mprime
            )
        )
        return zeroth_order + first_order

    def XX_matrix_element(self, ell, m):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        fmdim = self.fluxonium_minus_truncated_dim
        ellprime = (ell + 1) % 2
        mprime = (m + 1) % 2
        S1_m_contr = sum(
            (
                self._eps_1(
                    evals_minus,
                    evals_a,
                    phi_a_mat,
                    phi_minus_mat,
                    self.ELa,
                    i=ell,
                    j=ellprime,
                    n=n,
                )
                * self._eps_1(
                    evals_minus,
                    evals_b,
                    phi_b_mat,
                    phi_minus_mat,
                    self.ELb,
                    i=mprime,
                    j=m,
                    n=nprime,
                )
                + self._eps_1(
                    evals_minus,
                    evals_a,
                    phi_a_mat,
                    phi_minus_mat,
                    self.ELa,
                    i=ellprime,
                    j=ell,
                    n=n,
                )
                * self._eps_1(
                    evals_minus,
                    evals_b,
                    phi_b_mat,
                    phi_minus_mat,
                    self.ELb,
                    i=m,
                    j=mprime,
                    n=nprime,
                )
            )
            * phi_minus_mat[n, nprime]
            for n in range(1, fmdim)
            for nprime in range(1, fmdim)
        )
        S1_0_contr = sum(self._eps_1(evals_minus, evals_a, phi_a_mat, phi_minus_mat, self.ELa, i=ell,
                                     j=ellprime, n=n)
                         * self._eps_1(evals_minus, evals_b, phi_b_mat, phi_minus_mat, self.ELb, i=mprime,
                                       j=m, n=n)
                         + self._eps_1(evals_minus, evals_a, phi_a_mat, phi_minus_mat, self.ELa, i=ellprime,
                                       j=ell, n=n)
                         * self._eps_1(evals_minus, evals_b, phi_b_mat, phi_minus_mat, self.ELb, i=m,
                                       j=mprime, n=n)
                         for n in range(1, fmdim)
                         ) * phi_minus_mat[0, 0]
        S2_contr = sum(
            (
                self._eps_ab_2(
                    ell,
                    m,
                    ellprime,
                    mprime,
                    n,
                    evals_minus,
                    evals_a,
                    evals_b,
                    phi_a_mat,
                    phi_b_mat,
                    phi_minus_mat,
                )
                + self._eps_ab_2(
                    ellprime,
                    mprime,
                    ell,
                    m,
                    n,
                    evals_minus,
                    evals_a,
                    evals_b,
                    phi_a_mat,
                    phi_b_mat,
                    phi_minus_mat,
                )
            )
            * phi_minus_mat[0, n]
            for n in range(1, fmdim)
        )
        return 0.25 * self.EL_tilda() * (S1_m_contr - S2_contr - S1_0_contr)

    @staticmethod
    def _avg_and_rel_dev(A, B):
        avg = 0.5 * (A + B)
        rel_dev = (A - B) / avg
        return avg, rel_dev

    def _g_plus_minus(self, phi_minus_mat):
        ELq, dELq = self._avg_and_rel_dev(self.ELa, self.ELb)
        ELc, dELc = self._avg_and_rel_dev(self.EL1, self.EL2)
        return ((ELq * dELq + ELc * dELc) * self.h_o_plus().l_osc
                * phi_minus_mat[0, 0]) / (4 * np.sqrt(2))

    def _g_minus(self, ell, ellp, n, phi_q_mat, phi_minus_mat, EL):
        return 0.5 * EL * phi_q_mat[ell, ellp] * phi_minus_mat[0, n]

    def _g_minus_2(self, ell, ellp, n, nprime, phi_q_mat, phi_minus_mat, EL):
        """relevant for second order generator"""
        return 0.5 * EL * phi_q_mat[ell, ellp] * phi_minus_mat[n, nprime]

    def _g_plus(self, ell, ellp, phi_q_mat, EL):
        return 0.5 * EL * phi_q_mat[ell, ellp] * self.h_o_plus().l_osc / np.sqrt(2)

    def _eps_1(self, evals_minus, evals_i, phi_i_mat, phi_minus_mat, EL, i=0, j=1, n=1):
        """works for both qubits, need to feed in the right energies, phi_mat and EL"""
        return (
            0.5
            * EL
            * phi_minus_mat[0, n]
            * phi_i_mat[i, j]
            / (evals_minus[n] + evals_i[j] - evals_minus[0] - evals_i[i])
        )

    def _eps_1_plus(self, evals_i, phi_i_mat, EL, i=0, j=1):
        return (
            0.5
            * EL
            * self.h_o_plus().l_osc
            * phi_i_mat[i, j]
            / (self.h_o_plus().E_osc + evals_i[j] - evals_i[i])
        ) / np.sqrt(2)

    def H_c_qobj(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_minus = self.fluxonium_minus()
        ELa = fluxonium_a.EL
        ELb = fluxonium_b.EL
        ELt = self.EL_tilda()
        fmdim = fluxonium_minus.truncated_dim
        fadim = fluxonium_a.truncated_dim
        fbdim = fluxonium_b.truncated_dim
        evals_a, _, phi_a_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_a
        )
        evals_b, _, phi_b_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_b
        )
        evals_minus, _, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_minus
        )
        return (
            -0.25
            * ELt
            * tensor(
                qeye(fadim),
                qeye(fbdim),
                Qobj(phi_minus_mat),
                qeye(self.h_o_truncated_dim),
            )
            + 0.5
            * ELa
            * tensor(
                Qobj(phi_a_mat), qeye(fbdim), qeye(fmdim), qeye(self.h_o_truncated_dim)
            )
            - 0.5
            * ELb
            * tensor(
                qeye(fadim), Qobj(phi_b_mat), qeye(fmdim), qeye(self.h_o_truncated_dim)
            )
        )

    def _bare_product_state_all(self, ell, m, n, p):
        return tensor(
            basis(self.fluxonium_truncated_dim, ell),
            basis(self.fluxonium_truncated_dim, m),
            basis(self.fluxonium_minus_truncated_dim, n),
            basis(self.h_o_truncated_dim, p),
        )

    def first_order_generator(self):
        fadim = fbdim = self.fluxonium_truncated_dim
        fmdim = self.fluxonium_minus_truncated_dim
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        minus_a = sum(
            self._eps_1(
                evals_minus,
                evals_a,
                phi_a_mat,
                phi_minus_mat,
                self.ELa,
                i=ell,
                j=ellp,
                n=n,
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ellp, m, n, 0).dag()
            for ell in range(2)
            for m in range(2)
            for n in range(1, fmdim)
            for ellp in range(fadim)
        )
        # excite higher fluxonium levels
        minus_a += sum(
            self._eps_1(
                evals_minus,
                evals_a,
                phi_a_mat,
                phi_minus_mat,
                self.ELa,
                i=ell,
                j=ellp,
                n=0,
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ellp, m, 0, 0).dag()
            for ell in range(2)
            for m in range(2)
            for ellp in range(2, fadim)
        )
        minus_b = -sum(
            self._eps_1(
                evals_minus, evals_b, phi_b_mat, phi_minus_mat, self.ELb, i=m, j=mp, n=n
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ell, mp, n, 0).dag()
            for ell in range(2)
            for m in range(2)
            for n in range(1, fmdim)
            for mp in range(fbdim)
        )
        minus_b += -sum(
            self._eps_1(
                evals_minus, evals_b, phi_b_mat, phi_minus_mat, self.ELb, i=m, j=mp, n=0
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ell, mp, 0, 0).dag()
            for ell in range(2)
            for m in range(2)
            for mp in range(2, fbdim)
        )
        minus = minus_a + minus_b - (minus_a + minus_b).dag()
        plus_a = sum(
            self._eps_1_plus(evals_a, phi_a_mat, self.ELa, i=ell, j=ellp)
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ellp, m, 0, 1).dag()
            for ell in range(2)
            for m in range(2)
            for ellp in range(fadim)
        )
        plus_b = sum(
            self._eps_1_plus(evals_b, phi_b_mat, self.ELb, i=m, j=mp)
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ell, mp, 0, 1).dag()
            for ell in range(2)
            for m in range(2)
            for mp in range(fbdim)
        )
        plus = plus_a + plus_b - (plus_a + plus_b).dag()
        return plus + minus

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
        phi_a = fluxonium_a.phi_operator
        phi_b = fluxonium_b.phi_operator
        phi_minus = fluxonium_minus.phi_operator
        phi_plus = h_o_plus.phi_operator
        n_a = fluxonium_a.n_operator
        n_b = fluxonium_b.n_operator
        hilbert_space.add_interaction(
            g_strength=-0.5 * self.ELa, op1=phi_a, op2=phi_plus
        )
        hilbert_space.add_interaction(
            g_strength=-0.5 * self.ELb, op1=phi_b, op2=phi_plus
        )
        hilbert_space.add_interaction(
            g_strength=-0.5 * self.ELa, op1=phi_a, op2=phi_minus
        )
        hilbert_space.add_interaction(
            g_strength=0.5 * self.ELb, op1=phi_b, op2=phi_minus
        )
        hilbert_space.add_interaction(
            g_strength=-8.0 * self.off_diagonal_charging(), op1=n_a, op2=n_b
        )
        hilbert_space.add_interaction(
            g_strength=(self.ELa - self.ELb + self.EL1 - self.EL2) / 2.0,
            op1=phi_plus,
            op2=phi_minus,
        )
        return hilbert_space

    def generate_coupled_system_sweetspot(self):
        hilbert_space = self.generate_coupled_system()
        [
            fluxonium_a,
            fluxonium_b,
            fluxonium_minus,
            h_o_plus,
        ] = hilbert_space.subsystem_list
        phi_a = fluxonium_a.phi_operator
        phi_b = fluxonium_b.phi_operator
        phi_minus = fluxonium_minus.phi_operator
        phi_plus = h_o_plus.phi_operator
        offset_flux_a = 2.0 * np.pi * (fluxonium_a.flux - 0.5)
        offset_flux_b = 2.0 * np.pi * (fluxonium_b.flux - 0.5)
        fluxonium_a.flux = 0.5
        fluxonium_b.flux = 0.5
        hilbert_space.add_interaction(g=-self.ELa * offset_flux_a, op1=phi_a)
        hilbert_space.add_interaction(g=0.5 * self.ELa * offset_flux_a, op1=phi_plus)
        hilbert_space.add_interaction(g=0.5 * self.ELa * offset_flux_a, op1=phi_minus)
        hilbert_space.add_interaction(g=-self.ELb * offset_flux_b, op1=phi_b)
        hilbert_space.add_interaction(g=0.5 * self.ELb * offset_flux_b, op1=phi_plus)
        hilbert_space.add_interaction(g=-0.5 * self.ELb * offset_flux_b, op1=phi_minus)
        return hilbert_space

    def hamiltonian(self):
        hilbert_space = self.generate_coupled_system()
        return hilbert_space.hamiltonian().full()

    def hilbertdim(self) -> int:
        return (
            self.fluxonium_truncated_dim ** 2
            * self.fluxonium_minus_truncated_dim
            * self.h_o_truncated_dim
        )

    def find_flux_shift(self):
        # TODO sign problem? seems backwards?
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_minus = self.fluxonium_minus()
        _, _, phi_a_mat = self.signed_evals_evecs_phimat_qubit_instance(fluxonium_a)
        _, _, phi_b_mat = self.signed_evals_evecs_phimat_qubit_instance(fluxonium_b)
        _, _, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(fluxonium_minus)
        flux_a_ind = (self._g_plus_minus(phi_minus_mat)
                      * self._g_plus(0, 1, phi_a_mat, self.ELa) / self.h_o_plus().E_osc)
        flux_b_ind = (self._g_plus_minus(phi_minus_mat)
                      * self._g_plus(0, 1, phi_b_mat, self.ELb) / self.h_o_plus().E_osc)
        flux_shift_a = -phi_minus_mat[0, 0] / 2 + flux_a_ind
        flux_shift_b = phi_minus_mat[0, 0] / 2 + flux_b_ind
        return flux_shift_a / (2.0 * np.pi), flux_shift_b / (2.0 * np.pi)

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
        J = self.ELa * self.ELb * (0.5 * chi_m - 1.0 / self.EL_tilda())
        return fluxonium_a, fluxonium_b, J

    @staticmethod
    def _get_phi_01(fluxonium_instance):
        evals, evecs_uns = fluxonium_instance.eigensys(evals_count=2)
        evecs = np.zeros_like(evecs_uns)
        evecs_uns = evecs_uns.T
        for k, evec in enumerate(evecs_uns):
            evecs[:, k] = standardize_sign(evec)
        phi_mat = get_matrixelement_table(fluxonium_instance.phi_operator(), evecs)
        return phi_mat[0, 1]

    def J_eff_total(self):
        fluxonium_a, fluxonium_b, J = self._setup_effective_calculation()
        phi_a_01 = self._get_phi_01(fluxonium_a)
        phi_b_01 = self._get_phi_01(fluxonium_b)
        return J * phi_a_01 * phi_b_01

    def off_location_coupler_flux(self):
        def _find_J(flux_c):
            self.flux_c = flux_c
            return self.J_eff_total()

        result = root(_find_J, x0=np.array([0.28]))
        assert result.success
        return result.x[0]

    def _evals_zeroed(self):
        evals, _ = self.generate_coupled_system().hamiltonian().eigenstates(eigvals=3)
        return evals - evals[0]

    def _eigenvals_for_flux(self, fluxes):
        """Vary one qubit's flux near the off position to find the exact sweet spot locations"""
        flux_a, flux_b = fluxes
        self.flux_a = flux_a
        self.flux_b = flux_b
        evals = self._evals_zeroed()
        return np.sqrt(evals[1]**2 + evals[2]**2)

    def find_flux_shift_exact(self):
        """near the off position, we want to find the exact qubit fluxes necessary to
        put the qubits at their sweet spots. To do this we acknowledge that the qubits
        are (nearly) uncoupled, therefore each excited state is nearly a product state.
        Thus if we vary the qubit fluxes and minimize the excitation energies, we
        should be able to place both qubits at their sweet spots independently"""
        flux_shift_a_seed, flux_shift_b_seed = self.find_flux_shift()

        result = minimize(
            self._eigenvals_for_flux, x0=np.array([0.5 + flux_shift_a_seed, 0.5 + flux_shift_b_seed]),
            bounds=((0.4, 0.6), (0.4, 0.6)), tol=1e-4
        )
        assert result.success
        return result.x[0]-0.5, result.x[1]-0.5

    def off_location_effective_sweet_spot_fluxes(self):
        flux_c = self.off_location_coupler_flux()
        self.flux_c = flux_c
        flux_shift_a, flux_shift_b = self.find_flux_shift()
        return flux_c, 0.50 + flux_shift_a, 0.50 + flux_shift_b

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

    @staticmethod
    def decompose_matrix_into_paulis_single_qubit(matrix):
        pauli_mats = [qeye(2), sigmax(), sigmay(), sigmaz()]
        pauli_name = ["I", "X", "Y", "Z"]
        pauli_list = []
        for j, pauli in enumerate(pauli_mats):
            coeff = np.trace((pauli * matrix).data.toarray())
            pauli_list.append((pauli_name[j], coeff))
        return pauli_list

    def born_oppenheimer_effective_hamiltonian_static(self):
        (fluxonium_a, fluxonium_b, J) = self._setup_effective_calculation()
        fluxonium_a.truncated_dim = self.fluxonium_truncated_dim
        fluxonium_b.truncated_dim = self.fluxonium_truncated_dim
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        hilbert_space.add_interaction(
            g_strength=J, op1=fluxonium_a.phi_operator, op2=fluxonium_b.phi_operator
        )
        hilbert_space.add_interaction(
            g_strength=-8.0 * self.off_diagonal_charging(),
            op1=fluxonium_a.n_operator,
            op2=fluxonium_b.n_operator,
        )
        return hilbert_space.hamiltonian()

    def born_oppenheimer_effective_hamiltonian(self):
        (fluxonium_a, fluxonium_b, J) = self._setup_effective_calculation()
        g_s_expect = self.fluxonium_minus_gs_expect()
        EL_bar_a = fluxonium_a.EL
        EL_bar_b = fluxonium_b.EL
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_a.truncated_dim, fluxonium_b.truncated_dim = 2, 2
        hilbert_space = HilbertSpace([fluxonium_a, fluxonium_b])
        phi_a_coeff = -(
            EL_bar_a * 2.0 * np.pi * (self.flux_a - 0.5)
            + 0.5 * self.ELa * g_s_expect
            + J * 2.0 * np.pi * (self.flux_b - 0.5)
        )
        phi_b_coeff = -(
            EL_bar_b * 2.0 * np.pi * (self.flux_b - 0.5)
            - 0.5 * self.ELb * g_s_expect
            + J * 2.0 * np.pi * (self.flux_a - 0.5)
        )
        hilbert_space.add_interaction(
            g_strength=phi_a_coeff, op1=fluxonium_a.phi_operator
        )
        hilbert_space.add_interaction(
            g_strength=phi_b_coeff, op1=fluxonium_b.phi_operator
        )
        hilbert_space.add_interaction(
            g_strength=J, op1=fluxonium_a.phi_operator, op2=fluxonium_b.phi_operator
        )
        hilbert_space.add_interaction(
            g_strength=-8.0 * self.off_diagonal_charging(),
            op1=fluxonium_a.n_operator,
            op2=fluxonium_b.n_operator,
        )
        return hilbert_space.hamiltonian()

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
        return Fluxonium(
            self.EJa,
            self.qubit_a_charging_energy(),
            self.ELa,
            self.flux_a,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_truncated_dim,
        )

    def fluxonium_b(self):
        return Fluxonium(
            self.EJb,
            self.qubit_b_charging_energy(),
            self.ELb,
            self.flux_b,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_truncated_dim,
        )

    def fluxonium_minus(self):
        return Fluxonium(
            self.EJC,
            self.fluxonium_minus_charging_energy(),
            self.EL_tilda() / 4.0,
            self.flux_c,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_minus_truncated_dim,
            #            flux_fraction_with_inductor=0.0,
            #            flux_junction_sign=-1,
        )

    def EL_tilda(self):
        return self.EL1 + self.EL2 + self.ELa + self.ELb

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
        E_osc = convert_to_E_osc(
            8.0 * self.h_o_plus_charging_energy(), self.EL_tilda() / 4.0  # 16 EC_{m}
        )
        l_osc = convert_to_l_osc(
            8.0 * self.h_o_plus_charging_energy(), self.EL_tilda() / 4.0
        )
        return Oscillator(
            E_osc=E_osc, l_osc=l_osc, truncated_dim=self.h_o_truncated_dim,
        )


class FluxoniumTunableCouplerGrounded(FluxoniumTunableCouplerFloating):
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
