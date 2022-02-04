import cmath
from itertools import product
from typing import Optional, Callable

import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, Qobj, propagator, sesolve, Options
from scipy.interpolate import interp1d
from scipy.special import jn_zeros
from scipy.linalg import inv
from scipy.optimize import root, minimize
from sympy import Matrix, S, diff, hessian, simplify, solve, symbols

import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.fluxonium import Fluxonium
from scqubits.core.oscillator import Oscillator, convert_to_E_osc, convert_to_l_osc
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.utils.spectrum_utils import (
    get_matrixelement_table,
    standardize_sign,
    convert_evecs_to_ndarray,
    identity_wrap,
)


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

    def schrieffer_wolff_approx(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_minus = self.fluxonium_minus()
        evals_a, evecs_a, phi_a_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_a
        )
        evals_b, evecs_b, phi_b_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_b
        )
        evals_m, evecs_m, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_minus
        )
        chi_m = sum(
            abs(phi_minus_mat[0, m]) ** 2 / (evals_m[m] - evals_m[0])
            for m in range(1, fluxonium_minus.truncated_dim)
        )
        J = (
            self.ELa
            * self.ELb
            * phi_a_mat[0, 1]
            * phi_b_mat[0, 1]
            * (0.5 * chi_m - 1.0 / self.EL_tilda())
        )
        return (
            -0.5 * (evals_a[1] - evals_a[0]) * tensor(sigmaz(), qeye(2))
            - 0.5 * (evals_b[1] - evals_b[0]) * tensor(qeye(2), sigmaz())
            + J * tensor(sigmax(), sigmax())
        )

    def _delta_mu_j(self, j, evals_mu, phi_mu_mat, evals_minus, phi_minus_mat, ELmu):
        ECp = self.h_o_plus_charging_energy()
        ELc = self.EL_tilda() / 4
        omega_p = self.h_o_plus().E_osc
        coupler_minus_sum = -sum(
            (ELmu / 2) ** 2
            * phi_mu_mat[j, jprime] ** 2
            * (
                phi_minus_mat[0, n] ** 2
                / (evals_mu[jprime] + evals_minus[n] - evals_mu[j] - evals_minus[0])
            )
            for n in range(1, self.fluxonium_minus_truncated_dim)
            for jprime in range(0, self.fluxonium_truncated_dim)
        )
        coupler_plus_sum = -sum(
            (ELmu / 2) ** 2
            * phi_mu_mat[j, jprime] ** 2
            * np.sqrt(2 * ECp / ELc)
            / (evals_mu[jprime] + omega_p - evals_mu[j])
            for jprime in range(0, self.fluxonium_truncated_dim)
        )
        high_fluxonium_sum = -sum(
            (ELmu / 2) ** 2
            * phi_mu_mat[j, jprime] ** 2
            * (phi_minus_mat[0, 0] ** 2 / (evals_mu[jprime] - evals_mu[j]))
            for jprime in range(2, self.fluxonium_truncated_dim)
        )
        return coupler_minus_sum + coupler_plus_sum + high_fluxonium_sum

    def _J(self, evals_a, phi_a_mat, evals_b, phi_b_mat, evals_minus, phi_minus_mat):
        coupler_minus_sum = sum(
            phi_minus_mat[0, n] ** 2
            * (
                1.0 / (evals_a[0] + evals_minus[n] - evals_a[1] - evals_minus[0])
                + 1.0 / (evals_b[0] + evals_minus[n] - evals_b[1] - evals_minus[0])
                + 1.0 / (evals_a[1] + evals_minus[n] - evals_a[0] - evals_minus[0])
                + 1.0 / (evals_b[1] + evals_minus[n] - evals_b[0] - evals_minus[0])
            )
            for n in range(1, self.fluxonium_minus_truncated_dim)
        )
        omega_p = self.h_o_plus().E_osc
        ECp = self.h_o_plus_charging_energy()
        ELc = self.EL_tilda() / 4
        coupler_plus = -np.sqrt(2.0 * ECp / ELc) * (
            1.0 / (evals_a[0] + omega_p - evals_a[1])
            + 1.0 / (evals_b[0] + omega_p - evals_b[1])
            + 1.0 / (evals_a[1] + omega_p - evals_a[0])
            + 1.0 / (evals_b[1] + omega_p - evals_b[0])
        )
        return (
            0.5
            * (self.ELa / 2)
            * (self.ELb / 2)
            * phi_a_mat[0, 1]
            * phi_b_mat[0, 1]
            * (coupler_plus + coupler_minus_sum)
        )

    def schrieffer_wolff(self):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        omega_a = evals_a[1] - evals_a[0]
        omega_b = evals_b[1] - evals_b[0]
        delta_a_1 = self._delta_mu_j(
            1, evals_a, phi_a_mat, evals_minus, phi_minus_mat, self.ELa
        )
        delta_a_0 = self._delta_mu_j(
            0, evals_a, phi_a_mat, evals_minus, phi_minus_mat, self.ELa
        )
        delta_a = delta_a_0 - delta_a_1
        delta_b_1 = self._delta_mu_j(
            1, evals_b, phi_b_mat, evals_minus, phi_minus_mat, self.ELb
        )
        delta_b_0 = self._delta_mu_j(
            0, evals_b, phi_b_mat, evals_minus, phi_minus_mat, self.ELb
        )
        delta_b = delta_b_0 - delta_b_1
        J = self._J(evals_a, phi_a_mat, evals_b, phi_b_mat, evals_minus, phi_minus_mat)
        H = (
            -0.5 * (omega_a - delta_a) * tensor(sigmaz(), qeye(2))
            - 0.5 * (omega_b - delta_b) * tensor(qeye(2), sigmaz())
            + J * tensor(sigmax(), sigmax())
        )
        return H

    def op_reduced(self, op, inds_to_remove):
        """this function takes an operator and eliminates states that are
        not relevant as specified by inds_to_remove"""
        if isinstance(op, Qobj):
            op = op.data.toarray()
        new_op = np.delete(op, inds_to_remove, axis=0)
        new_op = np.delete(new_op, inds_to_remove, axis=1)
        return Qobj(new_op)

    def schrieffer_wolff_real_flux(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        flux_a, flux_b = fluxonium_a.flux, fluxonium_b.flux
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_minus = self.fluxonium_minus()
        evals_a, evecs_a, phi_a_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_a
        )
        evals_b, evecs_b, phi_b_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_b
        )
        evals_m, evecs_m, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_minus
        )
        H_0_a = np.diag(evals_a - evals_a[0])[0:2, 0:2]
        H_0_b = np.diag(evals_b - evals_b[0])[0:2, 0:2]
        H_0 = tensor(Qobj(H_0_a), qeye(2)) + tensor(qeye(2), Qobj(H_0_b))

        # first-order contribution yields sigmax
        H_1_a = (-0.5 * self.ELa * phi_minus_mat[0, 0] * phi_a_mat)[0:2, 0:2]
        # canceled by flux_offset
        H_1_a += -self.ELa * 2.0 * np.pi * (flux_a - 0.5) * phi_a_mat[0:2, 0:2]
        H_1_b = (0.5 * self.ELb * phi_minus_mat[0, 0] * phi_b_mat)[0:2, 0:2]
        H_1_b += -self.ELb * 2.0 * np.pi * (flux_b - 0.5) * phi_b_mat[0:2, 0:2]
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

    def H_c_diag(self, ell, m):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        zeroth_order = -0.25 * self.EL_tilda() * phi_minus_mat[0, 0]
        first_order_a = self.ELa * sum(
            phi_a_mat[ell, ellprime]
            * self._eps_1(
                evals_minus,
                evals_a,
                phi_a_mat,
                phi_minus_mat,
                self.ELa,
                i=ell,
                j=ellprime,
                n=0,
            )
            for ellprime in range(2, self.fluxonium_truncated_dim)
        )
        first_order_b = self.ELb * sum(
            phi_b_mat[m, mprime]
            * self._eps_1(
                evals_minus,
                evals_b,
                phi_b_mat,
                phi_minus_mat,
                self.ELb,
                i=m,
                j=mprime,
                n=0,
            )
            for mprime in range(2, self.fluxonium_truncated_dim)
        )
        return zeroth_order + first_order_a + first_order_b

    def H_c_XI(self, ell):
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

    def H_c_IX(self, m):
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

    def H_c_XX(self, ell, m):
        return -0.25 * self.EL_tilda() * self.theta_minus_XX(ell, m)

    def theta_minus_XX(self, ell, m):
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
        S1_0_contr = (
            sum(
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
                    n=n,
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
                    n=n,
                )
                for n in range(1, fmdim)
            )
            * phi_minus_mat[0, 0]
        )
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
        return -S1_m_contr + S2_contr + S1_0_contr

    def H_q_diag(self, ell, which="a"):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        if which is "a":
            evals_q, phi_q_mat, ELq, pref = evals_a, phi_a_mat, self.ELa, 1.0
        else:
            evals_q, phi_q_mat, ELq, pref = evals_b, phi_b_mat, self.ELb, -1.0
        zeroth_order = 0.5 * ELq * pref * phi_minus_mat[0, 0]
        first_order = (
            -ELq
            * 2.0
            * sum(
                phi_q_mat[ell, ellprime]
                * self._eps_1(
                    evals_minus,
                    evals_q,
                    phi_q_mat,
                    phi_minus_mat,
                    ELq,
                    i=ell,
                    j=ellprime,
                    n=0,
                )
                for ellprime in range(2, self.fluxonium_truncated_dim)
            )
        )
        return pref * (zeroth_order - first_order)

    def H_q_correct_X(self, ell, which="a"):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        if which is "a":
            evals_q, phi_q_mat, ELq = evals_a, phi_a_mat, self.ELa
        else:
            evals_q, phi_q_mat, ELq = evals_b, phi_b_mat, self.ELb
        ellp1 = (ell + 1) % 2
        ell_osc = self.h_o_plus().l_osc
        zeroth_order = -ELq * phi_q_mat[0, 1]
        first_order_minus = (
            0.5
            * ELq
            * sum(
                phi_minus_mat[0, n]
                * (
                    self._eps_1(
                        evals_minus,
                        evals_q,
                        phi_q_mat,
                        phi_minus_mat,
                        ELq,
                        i=ell,
                        j=ellp1,
                        n=n,
                    )
                    + self._eps_1(
                        evals_minus,
                        evals_q,
                        phi_q_mat,
                        phi_minus_mat,
                        ELq,
                        i=ellp1,
                        j=ell,
                        n=n,
                    )
                )
                for n in range(1, self.fluxonium_minus_truncated_dim)
            )
        )
        first_order_plus = (
            0.5
            * ELq
            * (ell_osc / np.sqrt(2))
            * (
                self._eps_1_plus(evals_q, phi_q_mat, ELq, i=ell, j=ellp1)
                + self._eps_1_plus(evals_q, phi_q_mat, ELq, i=ellp1, j=ell)
            )
        )
        return zeroth_order + first_order_plus + first_order_minus

    def H_q_wrong_X(self, m, which="a"):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        # note the switch below
        if which is "a":
            evals_q, phi_q_mat, ELq = evals_b, phi_b_mat, self.ELb
            EL_outside = self.ELa
        else:
            evals_q, phi_q_mat, ELq = evals_a, phi_a_mat, self.ELa
            EL_outside = self.ELb
        mp1 = (m + 1) % 2
        ell_osc = self.h_o_plus().l_osc
        first_order_minus = (
            -0.5
            * EL_outside
            * sum(
                phi_minus_mat[0, n]
                * (
                    self._eps_1(
                        evals_minus,
                        evals_q,
                        phi_q_mat,
                        phi_minus_mat,
                        ELq,
                        i=m,
                        j=mp1,
                        n=n,
                    )
                    + self._eps_1(
                        evals_minus,
                        evals_q,
                        phi_q_mat,
                        phi_minus_mat,
                        ELq,
                        i=mp1,
                        j=m,
                        n=n,
                    )
                )
                for n in range(1, self.fluxonium_minus_truncated_dim)
            )
        )
        first_order_plus = (
            0.5
            * EL_outside
            * (ell_osc / np.sqrt(2))
            * (
                self._eps_1_plus(evals_q, phi_q_mat, ELq, i=m, j=mp1)
                + self._eps_1_plus(evals_q, phi_q_mat, ELq, i=mp1, j=m)
            )
        )
        return first_order_plus + first_order_minus

    def H_q_XX(self, ell, m, which="a"):
        if which is "a":
            ELq, pref = self.ELa, 1.0
        else:
            ELq, pref = self.ELb, -1.0
        return 0.5 * ELq * pref * self.theta_minus_XX(ell, m)

    def construct_H_q_eff(self, which="a"):
        qubit_idx_range = [0, 1]
        two_qubit_idxs = list(product(qubit_idx_range, qubit_idx_range))
        H_q = np.zeros((4, 4))
        for ell, m in two_qubit_idxs:
            for ellprime, mprime in two_qubit_idxs:
                if which == "a":
                    idx_comp_row = ell
                    idx_comp_col = ellprime
                else:
                    idx_comp_row = m
                    idx_comp_col = mprime
                row_idx = 2 * ell + m
                col_idx = 2 * ellprime + mprime
                if ell == ellprime and m == mprime:
                    H_q[row_idx, col_idx] = self.H_q_diag(ell, which=which)
                elif ell == (ellprime + 1) % 2 and m == (mprime + 1) % 2:
                    H_q[row_idx, col_idx] = self.H_q_XX(ell, m)
                elif idx_comp_row == (idx_comp_col + 1) % 2:
                    H_q[row_idx, col_idx] = self.H_q_correct_X(
                        idx_comp_row, which=which
                    )
                else:
                    # m entry below doesn't matter
                    H_q[row_idx, col_idx] = self.H_q_wrong_X(0, which=which)
        return H_q

    def construct_H_c_eff(self):
        qubit_idx_range = [0, 1]
        two_qubit_idxs = list(product(qubit_idx_range, qubit_idx_range))
        H_c = np.zeros((4, 4))
        for ell, m in two_qubit_idxs:
            for ellprime, mprime in two_qubit_idxs:
                row_idx = 2 * ell + m
                col_idx = 2 * ellprime + mprime
                if ell == ellprime and m == mprime:
                    H_c[row_idx, col_idx] = self.H_c_diag(ell, m)
                elif ell == (ellprime + 1) % 2 and m == (mprime + 1) % 2:
                    H_c[row_idx, col_idx] = self.H_c_XX(ell, m)
                elif ell == (ellprime + 1) % 2:
                    H_c[row_idx, col_idx] = self.H_c_XI(ell)
                else:
                    H_c[row_idx, col_idx] = self.H_c_IX(m)
        return H_c

    @staticmethod
    def _avg_and_rel_dev(A, B):
        avg = 0.5 * (A + B)
        rel_dev = (A - B) / avg
        return avg, rel_dev

    def _g_plus_minus(self, phi_minus_mat):
        ELq, dELq = self._avg_and_rel_dev(self.ELa, self.ELb)
        ELc, dELc = self._avg_and_rel_dev(self.EL1, self.EL2)
        return (
            (ELq * dELq + ELc * dELc) * self.h_o_plus().l_osc * phi_minus_mat[0, 0]
        ) / (4 * np.sqrt(2))

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
        _, _, phi_minus_mat = self.signed_evals_evecs_phimat_qubit_instance(
            fluxonium_minus
        )
        flux_a_ind = (
            self._g_plus_minus(phi_minus_mat)
            * self._g_plus(0, 1, phi_a_mat, self.ELa)
            / self.h_o_plus().E_osc
        )
        flux_b_ind = (
            self._g_plus_minus(phi_minus_mat)
            * self._g_plus(0, 1, phi_b_mat, self.ELb)
            / self.h_o_plus().E_osc
        )
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

    def hilbert_space_at_sweetspot(self, flux_shift_a=None, flux_shift_b=None):
        if flux_shift_a is None and flux_shift_b is None:
            flux_shift_a, flux_shift_b = self.find_flux_shift_exact()
        self.flux_a = 0.5 + flux_shift_a
        self.flux_b = 0.5 + flux_shift_b
        return self.generate_coupled_system()

    def basis_change(self, op, evecs, hilbert_space, subsystem):
        op_id_wrap = identity_wrap(op, subsystem, hilbert_space.subsys_list)
        op_new_basis = np.real(evecs.T @ op_id_wrap.data @ evecs)
        return Qobj(op_new_basis)

    def operators_at_sweetspot(
        self, num_states=4, flux_shift_a=None, flux_shift_b=None, remove_unnecessary_states=False
    ):
        hilbert_space = self.hilbert_space_at_sweetspot(flux_shift_a, flux_shift_b)
        hilbert_space.generate_lookup()
        evals_qobj = hilbert_space.lookup.dressed_eigenenergies()[0:num_states]
        evecs_qobj = hilbert_space.lookup.dressed_eigenstates()[0:num_states]
        evals_zeroed = evals_qobj - evals_qobj[0]
        H_0 = Qobj(2.0 * np.pi * np.diag(evals_zeroed[0:num_states]))
        evecs_ = convert_evecs_to_ndarray(evecs_qobj).T
        evecs = np.zeros_like(evecs_)
        for i, evec in enumerate(evecs_.T):
            evecs[:, i] = standardize_sign(evec)
        fluxonium_a = hilbert_space["fluxonium_a"]
        fluxonium_b = hilbert_space["fluxonium_b"]
        fluxonium_minus = hilbert_space["fluxonium_minus"]
        h_o_plus = hilbert_space["h_o_plus"]
        phi_a = self.basis_change(
            fluxonium_a.phi_operator(), evecs, hilbert_space, fluxonium_a
        )
        phi_b = self.basis_change(
            fluxonium_b.phi_operator(), evecs, hilbert_space, fluxonium_b
        )
        phi_minus = self.basis_change(
            fluxonium_minus.phi_operator(), evecs, hilbert_space, fluxonium_minus
        )
        phi_plus = self.basis_change(
            h_o_plus.phi_operator(), evecs, hilbert_space, h_o_plus
        )
        H_a = -2.0 * np.pi * self.ELa * (phi_a - 0.5 * phi_plus - 0.5 * phi_minus)
        H_b = -2.0 * np.pi * self.ELb * (phi_b - 0.5 * phi_plus + 0.5 * phi_minus)
        H_c = (
            -2.0
            * np.pi
            * (
                0.25 * self.EL_tilda() * phi_minus
                - 0.5 * self.ELa * phi_a
                + 0.5 * self.ELb * phi_b
            )
        )
        if remove_unnecessary_states:
            bad_bare_labels = [(i, j, 1, 0) for i in range(2) for j in range(2)]
            bad_dressed_indices = []
            for bare_label in bad_bare_labels:
                bad_dressed_index = hilbert_space.lookup.dressed_index(bare_label)
                bad_dressed_indices.append(bad_dressed_index)
            states_to_keep = np.concatenate((np.arange(4), np.array(bad_dressed_indices)))
            all_dressed_indices = np.arange(num_states)
            states_to_remove = np.delete(all_dressed_indices, states_to_keep)
            H_0 = self.op_reduced(H_0, states_to_remove)
            H_a = self.op_reduced(H_a, states_to_remove)
            H_b = self.op_reduced(H_b, states_to_remove)
            H_c = self.op_reduced(H_c, states_to_remove)
        return H_0, H_a, H_b, H_c

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
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        J = self._J(evals_a, phi_a_mat, evals_b, phi_b_mat, evals_minus, phi_minus_mat)
        return J * phi_a_mat[0, 1] * phi_b_mat[0, 1]

    def off_location_coupler_flux(self, epsilon=1e-2):
        def _find_J(flux_c):
            self.flux_c = flux_c
            return self.J_eff_total()

        result = root(_find_J, x0=np.array([0.28]), tol=epsilon)
        assert result.success
        return result.x[0]

    def _evals_zeroed(self):
        evals, _ = self.generate_coupled_system().hamiltonian().eigenstates(eigvals=4)
        return evals - evals[0]

    def _eigenvals_for_flux_shift(self, flux_a):
        """Vary the qubit fluxes to find the exact sweet spot locations"""
        self.flux_a = flux_a
        self.flux_b = 1.0 - flux_a
        evals = self._evals_zeroed()
        return evals[3]

    def _cost_function_off_and_shift_positions(self, fluxes):
        """For efficiency we make the approximation that the flux shifts are equivalent"""
        flux_c, flux_a = fluxes
        flux_shift_a = flux_a - 0.5
        flux_b = 0.5 - flux_shift_a
        self.flux_c = flux_c
        _, H_a, H_b, H_c = self.operators_at_sweetspot(
            flux_shift_a=flux_shift_a, flux_shift_b=-flux_shift_a
        )
        return (
            np.abs(H_c[0, 1])
            + np.abs(H_c[0, 2])
            + np.abs(H_c[1, 3])
            + np.abs(H_c[2, 3])
        )

    def find_flux_shift_exact(self, epsilon=1e-4):
        """near the off position, we want to find the exact qubit fluxes necessary to
        put the qubits at their sweet spots. To do this we acknowledge that the qubits
        are (nearly) uncoupled, therefore each excited state is nearly a product state.
        Thus if we vary the qubit fluxes and minimize the excitation energies, we
        should be able to place both qubits at their sweet spots independently"""
        flux_shift_a_seed, _ = self.find_flux_shift()

        result = minimize(
            self._eigenvals_for_flux_shift,
            x0=np.array([0.5 + flux_shift_a_seed]),
            bounds=((0.5, 0.6),),
            tol=epsilon,
        )
        assert result.success
        return result.x[0] - 0.5

    def find_off_position_and_flux_shift_exact(self, epsilon=1e-4):
        (
            flux_c_seed,
            flux_shift_a_seed,
            _,
        ) = self.off_location_effective_sweet_spot_fluxes()

        result = minimize(
            self._cost_function_off_and_shift_positions,
            x0=np.array([flux_c_seed, 0.5 + flux_shift_a_seed]),
            bounds=((0.0, 0.5), (0.5, 1.0)),
            tol=epsilon,
        )
        assert result.success
        return result.x[0], result.x[1] - 0.5, -(result.x[1] - 0.5)

    def off_location_effective_sweet_spot_fluxes(self):
        flux_c = self.off_location_coupler_flux()
        self.flux_c = flux_c
        flux_shift_a, flux_shift_b = self.find_flux_shift()
        return flux_c, flux_shift_a, flux_shift_b

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
            id_str="fluxonium_a",
        )

    def fluxonium_b(self):
        return Fluxonium(
            self.EJb,
            self.qubit_b_charging_energy(),
            self.ELb,
            self.flux_b,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_truncated_dim,
            id_str="fluxonium_b",
        )

    def fluxonium_minus(self):
        return Fluxonium(
            self.EJC,
            self.fluxonium_minus_charging_energy(),
            self.EL_tilda() / 4.0,
            self.flux_c,
            cutoff=self.fluxonium_cutoff,
            truncated_dim=self.fluxonium_minus_truncated_dim,
            id_str="fluxonium_minus"
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
            8.0 * self.h_o_plus_charging_energy(), self.EL_tilda() / 4.0
        )  # 16 EC_{m}
        l_osc = convert_to_l_osc(
            8.0 * self.h_o_plus_charging_energy(), self.EL_tilda() / 4.0
        )
        return Oscillator(
            E_osc=E_osc,
            l_osc=l_osc,
            truncated_dim=self.h_o_truncated_dim,
            id_str="h_o_plus",
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


class ConstructFullPulse(serializers.Serializable):
    def __init__(self, H_0, H_a, H_b, H_c, control_dt_slow=2.0, control_dt_fast=0.01,
                 max_freq=0.255, min_freq=0.125):
        self.H_0 = H_0
        self.H_a = H_a
        self.H_b = H_b
        self.H_c = H_c
        self.dim = H_0.shape[0]
        self.control_dt_slow = control_dt_slow
        self.control_dt_fast = control_dt_fast
        self.max_freq = max_freq
        self.min_freq = min_freq

    @staticmethod
    def ZA():
        return tensor(sigmaz(), qeye(2))

    @staticmethod
    def ZB():
        return tensor(qeye(2), sigmaz())

    @staticmethod
    def sqrtiSWAP():
        sqrt2 = np.sqrt(2.0)
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1/sqrt2, -1j/sqrt2, 0.0],
                         [0.0, -1j/sqrt2, 1/sqrt2, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])

    def normalized_operators(self):
        norm_a = self.H_a[0, 2]
        norm_b = self.H_b[0, 1]
        norm_c = self.H_c[0, 3]
        XI = self.H_a / norm_a
        IX = self.H_b / norm_b
        XX = self.H_c / norm_c
        return (norm_a, norm_b, norm_c), (XI, IX, XX)

    def RZ(self, theta, which='a'):
        if which is 'a':
            Z = self.ZA()
        else:
            Z = self.ZB()
        return Qobj((-1j * theta * Z / 2.0).expm().data, dims=[[4], [4]])

    @staticmethod
    def fix_w_single_q_gates(gate_):
        alpha = cmath.phase(gate_[0, 0])
        beta = cmath.phase(gate_[1, 1])
        gamma = cmath.phase(gate_[1, 2])
        return np.array([alpha + beta, alpha - gamma - np.pi / 2, -beta + gamma + np.pi / 2])

    def multiply_with_single_q_gates(self, gate):
        (t1, t2, t3) = self.fix_w_single_q_gates(gate)
        gate_ = Qobj(gate[0:4, 0:4])
        return self.RZ(t1, which='a') * self.RZ(t2, which='b') * gate_ * self.RZ(0, which='a') * self.RZ(t3, which='b')

    @staticmethod
    def calc_fidel_4(prop, gate):
        prop = Qobj(prop[0:4, 0:4])
        return (np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2) / 20

    @staticmethod
    def calc_fidel_2(prop, gate):
        prop = Qobj(prop[0:2, 0:2])
        return (np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2) / 6

    @staticmethod
    def get_controls_only_sine(freq, amp, control_dt=0.01):
        sintime = 1.0 / freq
        sin_eval_times = np.linspace(0.0, sintime, int(sintime / control_dt) + 1)
        sin_pulse = amp * np.sin(2.0 * np.pi * freq * sin_eval_times)
        return sin_pulse, sin_eval_times

    @staticmethod
    def amp_from_freq_id(freq):
        bessel_val = jn_zeros(0, 1)
        return 2.0 * np.pi * freq * bessel_val / 2.0

    @staticmethod
    def amp_from_freq_sqrtiswap(omega, omega_d, n=1):
        return 0.125 * np.pi * (omega_d ** 2 - omega ** 2) / (omega_d * np.sin(n * np.pi * omega / omega_d))

    def optimize_amp_id_fidel(self, amp, freq, which_qubit='a'):
        times = np.linspace(0.0, 1. / freq, int(1. / freq / 0.02) + 1)
        omega_a = np.real(self.H_0[2, 2])
        omega_b = np.real(self.H_0[1, 1])
        _, (XI, IX, XX) = self.normalized_operators()
        red_dim = 4

        def control_func(t, args=None):
            return amp * np.sin(2.0 * np.pi * freq * t)
        if which_qubit == 'a':  # driving qubit a
            drive_H = Qobj(XI[0:red_dim, 0:red_dim])
            ideal_prop = self.RZ(-times[-1] * omega_b, which='b')
        else:
            drive_H = Qobj(IX[0:red_dim, 0:red_dim])
            ideal_prop = self.RZ(-times[-1] * omega_a, which='a')
        H = [Qobj(self.H_0[0:red_dim, 0:red_dim]), [drive_H, control_func]]
        prop = propagator(H, times)
        return 1 - self.calc_fidel_4(prop[-1], ideal_prop)

    def synchronize(self, ta, tb):
        if ta <= tb:
            return self._synchronize(ta, tb)
        else:
            output, _ = self._synchronize(tb, ta)
            flipped_output = np.flip(output, axis=1)
            return flipped_output, (ta, tb)

    def _synchronize(self, ta, tb):
        """Assume ta <= tb"""
        tmax = 1. / self.max_freq
        tmin = 1. / self.min_freq
        if ta == tb == 0.0:
            return np.array([(None, None)]), (ta, tb)
        elif tmax <= (tb - ta) <= tmin:
            return np.array([(1. / (tb - ta), None)]), (ta, tb)
        elif (tb - ta) < tmax:
            new_freq = (tb - ta + tmax) ** (-1)
            return np.array([(new_freq, self.max_freq)]), (ta, tb)
        else:  # (tb - ta) > 1. / min_freq
            trial_time, n, r = self._remainder_search(tb - ta)
            return np.array(int(n) * ((1. / trial_time, None),) + ((1. / r, None),)), (ta, tb)

    def _remainder_search(self, tdiff):
        max_time = 1. / self.min_freq
        min_time = 1. / self.max_freq
        time_linspace = np.linspace(max_time, min_time, 101)
        for trial_time in time_linspace:
            n, r = divmod(tdiff, trial_time)
            if min_time <= r <= max_time:
                return trial_time, n, r
        raise(RuntimeError('no sequence of identity pulses found to synchronize the two qubits'))

    def _concatenate_for_qubit(self, freq, total_pulse, total_times, which_qubit='a'):
        amp_0 = self.amp_from_freq_id(freq)
        optimized_amp = minimize(self.optimize_amp_id_fidel, x0=np.array([amp_0]),
                                 args=(freq, which_qubit))
        assert optimized_amp.success
        amp = optimized_amp.x[0]
        controls, times = self.get_controls_only_sine(freq, amp, self.control_dt_fast)
        total_pulse = self.concatenate_times_or_controls((total_pulse, controls),
                                                         self.concatenate_two_controls)
        total_times = self.concatenate_times_or_controls((total_times, times),
                                                         self.concatenate_two_times)
        return total_pulse, total_times

    def parse_synchronize(self, synchronize_output):
        """This function takes the output of synchronize and yields
        the pulses along with the times that synchronize specified"""
        total_pulse_a = np.array([])
        total_pulse_b = np.array([])
        total_times_a = np.array([])
        total_times_b = np.array([])
        output, (t_a, t_b) = synchronize_output
        control_dt = self.control_dt_fast
        for (freq_a_, freq_b_) in output:
            if freq_a_ is not None:
                total_pulse_a, total_times_a = self._concatenate_for_qubit(freq_a_, total_pulse_a,
                                                                           total_times_a, which_qubit='a')
            if freq_b_ is not None:
                total_pulse_b, total_times_b = self._concatenate_for_qubit(freq_b_, total_pulse_b,
                                                                           total_times_b, which_qubit='b')
        # here we add the delay part that actually gives us Z rotations
        delay_time_a = np.linspace(0.0, t_a, int(t_a / control_dt) + 1)
        delay_time_b = np.linspace(0.0, t_b, int(t_b / control_dt) + 1)
        total_times_a = self.concatenate_times_or_controls((total_times_a, delay_time_a),
                                                           self.concatenate_two_times)
        total_times_b = self.concatenate_times_or_controls((total_times_b, delay_time_b),
                                                           self.concatenate_two_times)
        total_pulse_a = self.concatenate_times_or_controls((total_pulse_a, np.zeros_like(delay_time_a)),
                                                           self.concatenate_two_controls)
        total_pulse_b = self.concatenate_times_or_controls((total_pulse_b, np.zeros_like(delay_time_b)),
                                                           self.concatenate_two_controls)
        return total_pulse_a, total_times_a, total_pulse_b, total_times_b

    def concatenate_times_or_controls(self, t_c_tuple: tuple, concatenator: Callable):
        if len(t_c_tuple) == 1:
            return t_c_tuple[0]
        concat_first_two = concatenator(t_c_tuple[0], t_c_tuple[1])
        if len(t_c_tuple) == 2:
            return concat_first_two
        return self.concatenate_times_or_controls((concat_first_two,) + t_c_tuple[2:], concatenator)

    @staticmethod
    def concatenate_two_times(times_1, times_2):
        if times_1.size == 0:
            return times_2
        if times_2.size == 0:
            return times_1
        return np.concatenate((times_1, times_1[-1] + times_2[1:]))

    @staticmethod
    def concatenate_two_controls(controls_1, controls_2):
        if controls_1.size == 0:
            return controls_2
        if controls_2.size == 0:
            return controls_1
        assert np.allclose(controls_1[-1], 0.0) and np.allclose(controls_2[-1], 0.0)
        return np.concatenate((controls_1, controls_2[1:]))

    def propagator_for_coupler_segment(self, amp: float, omega_d: float, num_periods=2, red_dim=4):
        """
        Parameters
        ----------
        amp
            amplitude of the pulse in angular frequency units
        omega_d
            frequency of the pulse in angular frequency units
        num_periods
            number of periods of driving
        red_dim
            dimension of the propagator. the time evolution will
            be carried out includingthe full Hilbert space, but only
            for initial states up to that specified by red_dim. default
            is only for the qubit states

        Returns
        -------
            propagator in the qubit subspace

        """
        def control_func_c(t, args=None):
            return amp * np.sin(omega_d * t)
        control_dt = self.control_dt_slow
        total_time = num_periods * 2.0 * np.pi / omega_d
        _, (_, _, XX) = self.normalized_operators()
        H = [self.H_0, [XX, control_func_c]]
        twoqcontrol_eval_times = np.linspace(0.0, total_time, int(total_time / control_dt) + 1)
        my_prop = np.zeros((red_dim, red_dim), dtype=complex)
        for i in range(red_dim):
            result = sesolve(H, basis(self.dim, i), twoqcontrol_eval_times, options=Options(store_final_state=True))
            my_prop[:, i] = result.final_state.data.toarray()[0:red_dim, 0]
        return Qobj(my_prop)

    def propagator_for_qubit_flux_segment(self, parse_synchronize_output, red_dim=4):
        pulse_a, times_a, pulse_b, times_b = parse_synchronize_output
        _, (XI, IX, _) = self.normalized_operators()
        spline_a = interp1d(times_a, pulse_a, fill_value='extrapolate')
        spline_b = interp1d(times_b, pulse_b, fill_value='extrapolate')

        def control_func_a(t, args=None):
            return spline_a(t)

        def control_func_b(t, args=None):
            return spline_b(t)

        H = [Qobj(self.H_0[0:red_dim, 0:red_dim]),
             [Qobj(XI[0:red_dim, 0:red_dim]), control_func_a],
             [Qobj(IX[0:red_dim, 0:red_dim]), control_func_b]]
        return propagator(H, times_a)[-1]

    def propagator_for_full_pulse(self, amp, omega_d, num_periods, red_dim=4):
        twoqprop = self.propagator_for_coupler_segment(amp, omega_d, num_periods)
        global_phase = cmath.phase(twoqprop[0, 0])
        zeroed_prop = twoqprop * np.exp(-1j * global_phase)
        angles = self.fix_w_single_q_gates(zeroed_prop)
        neg_angles = -angles % (2.0 * np.pi)
        omega_a = np.real(self.H_0[2, 2])
        omega_b = np.real(self.H_0[1, 1])
        times = neg_angles / np.array([omega_a, omega_b, omega_b])
        parse_output_after = self.parse_synchronize(self.synchronize(times[0], times[1]))
        parse_output_before = self.parse_synchronize(self.synchronize(0.0, times[2]))
        before_prop = self.propagator_for_qubit_flux_segment(parse_output_before, red_dim)
        after_prop = self.propagator_for_qubit_flux_segment(parse_output_after, red_dim)
        return after_prop * Qobj(zeroed_prop[0:red_dim, 0:red_dim]) * before_prop
