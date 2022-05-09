import cmath
import warnings
from itertools import product
from typing import Optional, Callable, Tuple

import numpy as np
from numpy import ndarray
from qutip import (
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    tensor,
    basis,
    Qobj,
    propagator,
    sesolve,
    Options, mesolve, spre, spost,
)
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
from scqubits.utils import cpu_switch
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
        var_phi_a, var_phi_b, var_phi_1, var_phi_2, var_phi_sum = symbols(
            "var_phi_a var_phi_b var_phi_1 var_phi_2 var_phi_sum"
        )
        var_phi_list = Matrix([var_phi_a, var_phi_b, var_phi_1, var_phi_2, var_phi_sum])
        phi_subs = U_inv * var_phi_list
        T = T.subs([(phi_val, phi_subs[j]) for j, phi_val in enumerate(phi_vector)])
        T = simplify(T.subs(var_phi_sum, solve(diff(T, var_phi_sum), var_phi_sum)[0]))
        cap_mat = hessian(T, var_phi_list)
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
    def signed_evals_evecs_phi_mat_qubit_instance(qubit_instance):
        evals, evecs_uns = qubit_instance.eigensys(
            evals_count=qubit_instance.truncated_dim
        )
        evecs = np.zeros_like(evecs_uns).T
        for k, evec in enumerate(evecs_uns.T):
            evecs[k, :] = standardize_sign(evec)
        phi_mat = get_matrixelement_table(qubit_instance.phi_operator(), evecs.T)
        return evals, evecs.T, phi_mat

    def _delta_mu_j(self, j, evals_mu, phi_mu_mat, evals_minus, phi_minus_mat, ELmu):
        ECp = self.h_o_plus_charging_energy()
        ELc = self.EL_tilda() / 4
        omega_p = self.h_o_plus().E_osc
        coupler_minus_sum = -sum(
            (ELmu / 2) ** 2
            * phi_mu_mat[j, j_prime] ** 2
            * (
                phi_minus_mat[0, n] ** 2
                / (evals_mu[j_prime] + evals_minus[n] - evals_mu[j] - evals_minus[0])
            )
            for n in range(1, self.fluxonium_minus_truncated_dim)
            for j_prime in range(0, self.fluxonium_truncated_dim)
        )
        coupler_plus_sum = -sum(
            (ELmu / 2) ** 2
            * phi_mu_mat[j, j_prime] ** 2
            * np.sqrt(2 * ECp / ELc)
            / (evals_mu[j_prime] + omega_p - evals_mu[j])
            for j_prime in range(0, self.fluxonium_truncated_dim)
        )
        high_fluxonium_sum = -sum(
            (ELmu / 2) ** 2
            * phi_mu_mat[j, j_prime] ** 2
            * (phi_minus_mat[0, 0] ** 2 / (evals_mu[j_prime] - evals_mu[j]))
            for j_prime in range(2, self.fluxonium_truncated_dim)
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

    def op_reduced(self, op, indices_to_remove):
        """this function takes an operator and eliminates states that are
        not relevant as specified by indices_to_remove"""
        if isinstance(op, Qobj):
            op = op.data.toarray()
        new_op = np.delete(op, indices_to_remove, axis=0)
        new_op = np.delete(new_op, indices_to_remove, axis=1)
        return Qobj(new_op)

    def schrieffer_wolff_real_flux(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        flux_a, flux_b = fluxonium_a.flux, fluxonium_b.flux
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        H_0_a = np.diag(evals_a - evals_a[0])[0:2, 0:2]
        H_0_b = np.diag(evals_b - evals_b[0])[0:2, 0:2]
        H_0 = tensor(Qobj(H_0_a), qeye(2)) + tensor(qeye(2), Qobj(H_0_b))

        # first-order contribution yields sigma_x
        H_1_a = (-0.5 * self.ELa * phi_minus_mat[0, 0] * phi_a_mat)[0:2, 0:2]
        # canceled by flux_offset
        H_1_a += -self.ELa * 2.0 * np.pi * (flux_a - 0.5) * phi_a_mat[0:2, 0:2]
        H_1_b = (0.5 * self.ELb * phi_minus_mat[0, 0] * phi_b_mat)[0:2, 0:2]
        H_1_b += -self.ELb * 2.0 * np.pi * (flux_b - 0.5) * phi_b_mat[0:2, 0:2]
        H_1 = tensor(Qobj(H_1_a), qeye(2)) + tensor(qeye(2), Qobj(H_1_b))

        # second order calculation
        H_2_a = self._H2_self_correction_real_flux_coupler(
            evals_a, evals_minus, phi_a_mat, phi_minus_mat, fluxonium_a.EL
        )
        H_2_a += self._H2_self_correction_real_flux_high_fluxonium(
            evals_a, phi_a_mat, phi_minus_mat, fluxonium_a.EL
        )
        H_2_b = self._H2_self_correction_real_flux_coupler(
            evals_b, evals_minus, phi_b_mat, phi_minus_mat, fluxonium_b.EL
        )
        H_2_b += self._H2_self_correction_real_flux_high_fluxonium(
            evals_b, phi_b_mat, phi_minus_mat, fluxonium_b.EL
        )
        H_2 = tensor(Qobj(H_2_a[0:2, 0:2]), qeye(2)) + tensor(
            qeye(2), Qobj(H_2_b[0:2, 0:2])
        )
        H_2_ab = self._H2_qubit_coupling_real_flux(
            evals_a,
            evals_b,
            evals_minus,
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
            * self._g_plus(ell, ell_prime, phi_q_mat, EL)
            * (
                1.0 / (evals_q[ell] - evals_q[ell_prime] - self.h_o_plus().E_osc)
                + 1.0 / (evals_q[ell_prime] - evals_q[ell] - self.h_o_plus().E_osc)
            )
            * basis(2, ell)
            * basis(2, ell_prime).dag()
            for ell in range(2)
            for ell_prime in range(2)
        )
        return H_2_

    def _H2_self_correction_real_flux_high_fluxonium(
        self, evals_q, phi_q_mat, phi_minus_mat, EL
    ):
        H_2_ = sum(
            0.5
            * self._g_minus(ell, ell_double_prime, 0, phi_q_mat, phi_minus_mat, EL)
            * self._g_minus(
                ell_double_prime, ell_prime, 0, phi_q_mat, phi_minus_mat, EL
            )
            * (
                1.0 / (evals_q[ell] - evals_q[ell_double_prime])
                + 1.0 / (evals_q[ell_prime] - evals_q[ell_double_prime])
            )
            * basis(2, ell)
            * basis(2, ell_prime).dag()
            for ell in range(2)
            for ell_prime in range(2)
            for ell_double_prime in range(2, self.fluxonium_truncated_dim)
        )
        return H_2_

    def _H2_self_correction_real_flux_coupler(
        self, evals_q, evals_m, phi_q_mat, phi_minus_mat, EL
    ):
        H_2_ = sum(
            0.5
            * self._g_minus(ell, ell_double_prime, n, phi_q_mat, phi_minus_mat, EL)
            * self._g_minus(
                ell_double_prime, ell_prime, n, phi_q_mat, phi_minus_mat, EL
            )
            * (
                1.0
                / (evals_q[ell] + evals_m[0] - evals_q[ell_double_prime] - evals_m[n])
                + 1.0
                / (
                    evals_q[ell_prime]
                    + evals_m[0]
                    - evals_q[ell_double_prime]
                    - evals_m[n]
                )
            )
            * basis(2, ell)
            * basis(2, ell_prime).dag()
            for ell in range(2)
            for ell_prime in range(2)
            for ell_double_prime in range(self.fluxonium_truncated_dim)
            for n in range(1, self.fluxonium_minus_truncated_dim)
        )
        H_2_ += sum(
            0.5
            * self._g_plus(ell, ell_double_prime, phi_q_mat, EL)
            * self._g_plus(ell_double_prime, ell_prime, phi_q_mat, EL)
            * (
                1.0 / (evals_q[ell] - evals_q[ell_double_prime] - self.h_o_plus().E_osc)
                + 1.0
                / (
                    evals_q[ell_prime]
                    - evals_q[ell_double_prime]
                    - self.h_o_plus().E_osc
                )
            )
            * basis(2, ell)
            * basis(2, ell_prime).dag()
            for ell in range(2)
            for ell_prime in range(2)
            for ell_double_prime in range(self.fluxonium_truncated_dim)
        )
        return H_2_

    def _H2_qubit_coupling_real_flux(
        self, evals_a, evals_b, evals_m, phi_a_mat, phi_b_mat, phi_m_mat, ELa, ELb
    ):
        H_2_ = 0.5 * sum(
            -self._g_minus(ell, ell_prime, n, phi_a_mat, phi_m_mat, ELa)
            * self._g_minus(m, m_prime, n, phi_b_mat, phi_m_mat, ELb)
            * (
                1.0 / (evals_a[ell] + evals_m[0] - evals_a[ell_prime] - evals_m[n])
                + 1.0 / (evals_b[m] + evals_m[0] - evals_b[m_prime] - evals_m[n])
                + 1.0 / (evals_a[ell_prime] + evals_m[0] - evals_a[ell] - evals_m[n])
                + 1.0 / (evals_b[m_prime] + evals_m[0] - evals_b[m] - evals_m[n])
            )
            * tensor(basis(2, ell), basis(2, m))
            * tensor(basis(2, ell_prime), basis(2, m_prime)).dag()
            for ell in range(2)
            for m in range(2)
            for ell_prime in range(2)
            for m_prime in range(2)
            for n in range(1, self.fluxonium_minus_truncated_dim)
        )
        H_2_ += 0.5 * sum(
            self._g_plus(ell, ell_prime, phi_a_mat, ELa)
            * self._g_plus(m, m_prime, phi_b_mat, ELb)
            * (
                1.0 / (evals_a[ell] - evals_a[ell_prime] - self.h_o_plus().E_osc)
                + 1.0 / (evals_b[m] - evals_b[m_prime] - self.h_o_plus().E_osc)
                + 1.0 / (evals_a[ell_prime] - evals_a[ell] - self.h_o_plus().E_osc)
                + 1.0 / (evals_b[m_prime] - evals_b[m] - self.h_o_plus().E_osc)
            )
            * tensor(basis(2, ell), basis(2, m))
            * tensor(basis(2, ell_prime), basis(2, m_prime)).dag()
            for ell in range(2)
            for m in range(2)
            for ell_prime in range(2)
            for m_prime in range(2)
        )
        return H_2_

    def _high_high_self_mat_elem(
        self,
        evals_minus,
        evals_q,
        phi_q_mat,
        phi_minus_mat,
        ELq,
        ell,
        ell_prime,
        n_prime,
    ):
        f_a_dim = self.fluxonium_truncated_dim
        f_m_dim = self.fluxonium_minus_truncated_dim
        mat_elem = sum(
            self._g_minus_2(
                ell, ell_double_prime, 0, n_double_prime, phi_q_mat, phi_minus_mat, ELq
            )
            * self._g_minus_2(
                ell_double_prime,
                ell_prime,
                n_double_prime,
                n_prime,
                phi_q_mat,
                phi_minus_mat,
                ELq,
            )
            / (
                (
                    evals_q[ell]
                    + evals_minus[0]
                    - evals_q[ell_double_prime]
                    - evals_minus[n_double_prime]
                )
                * (
                    evals_q[ell]
                    + evals_minus[0]
                    - evals_q[ell_prime]
                    - evals_minus[n_prime]
                )
            )
            for ell_double_prime in range(f_a_dim)
            for n_double_prime in range(1, f_m_dim)
        )
        return mat_elem

    def _high_high_cross_mat_elem(
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
        ell_prime,
        m_prime,
        n_prime,
    ):
        f_m_dim = self.fluxonium_minus_truncated_dim
        mat_elem = sum(
            self._g_minus_2(
                ell, ell_prime, 0, n_double_prime, phi_a_mat, phi_minus_mat, ELa
            )
            * self._g_minus_2(
                m, m_prime, n_double_prime, n_prime, phi_b_mat, phi_minus_mat, ELb
            )
            / (
                (
                    evals_a[ell]
                    + evals_minus[0]
                    - evals_a[ell_prime]
                    - evals_minus[n_double_prime]
                )
                * (
                    evals_a[ell]
                    + evals_b[m]
                    + evals_minus[0]
                    - evals_a[ell_prime]
                    - evals_b[m_prime]
                    - evals_minus[n_prime]
                )
            )
            for n_double_prime in range(1, f_m_dim)
        )
        return mat_elem

    def _low_high_cross_mat_elem(
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
        ell_prime,
        m_prime,
        n,
    ):
        mat_elem = (
            self._g_minus(ell, ell_prime, 0, phi_a_mat, phi_minus_mat, ELa)
            * self._g_minus(m, m_prime, n, phi_b_mat, phi_minus_mat, ELb)
            / (
                (
                    evals_a[ell]
                    + evals_b[m]
                    + evals_minus[0]
                    - evals_a[ell_prime]
                    - evals_b[m_prime]
                    - evals_minus[n]
                )
                * (evals_b[m] + evals_minus[0] - evals_b[m_prime] - evals_minus[n])
            )
        )
        mat_elem += (
            self._g_minus(ell, ell_prime, n, phi_a_mat, phi_minus_mat, ELa)
            * self._g_minus(m, m_prime, 0, phi_b_mat, phi_minus_mat, ELb)
            / (
                (
                    evals_a[ell]
                    + evals_b[m]
                    + evals_minus[0]
                    - evals_a[ell_prime]
                    - evals_b[m_prime]
                    - evals_minus[n]
                )
                * (evals_a[ell] + evals_minus[0] - evals_a[ell_prime] - evals_minus[n])
            )
        )
        return mat_elem

    def _low_high_self_mat_elem(
        self, evals_minus, evals_q, phi_q_mat, phi_minus_mat, ELq, ell, ell_prime, n
    ):
        f_a_dim = self.fluxonium_truncated_dim
        mat_elem = sum(
            self._g_minus(ell, ell_double_prime, 0, phi_q_mat, phi_minus_mat, ELq)
            * self._g_minus(
                ell_double_prime, ell_prime, n, phi_q_mat, phi_minus_mat, ELq
            )
            / (
                (evals_q[ell] + evals_minus[0] - evals_q[ell_prime] - evals_minus[n])
                * (
                    evals_q[ell_double_prime]
                    + evals_minus[0]
                    - evals_q[ell_prime]
                    - evals_minus[n]
                )
            )
            for ell_double_prime in range(f_a_dim)
        )
        return mat_elem

    def _low_high_mat_elem(
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
        ell_prime,
        m_prime,
        n,
    ):
        mat_elem = self._low_high_self_mat_elem(
            evals_minus, evals_a, phi_a_mat, phi_minus_mat, ELa, ell, ell_prime, n
        )
        mat_elem += self._low_high_self_mat_elem(
            evals_minus, evals_b, phi_b_mat, phi_minus_mat, ELb, m, m_prime, n
        )
        mat_elem += -self._low_high_cross_mat_elem(
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
            ell_prime,
            m_prime,
            n,
        )
        return mat_elem

    def _high_high_mat_elem(
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
        ell_prime,
        m_prime,
        n_prime,
    ):
        mat_elem = self._high_high_self_mat_elem(
            evals_minus, evals_a, phi_a_mat, phi_minus_mat, ELa, ell, ell_prime, n_prime
        )
        mat_elem += self._high_high_self_mat_elem(
            evals_minus, evals_b, phi_b_mat, phi_minus_mat, ELb, m, m_prime, n_prime
        )
        mat_elem += -self._high_high_cross_mat_elem(
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
            ell_prime,
            m_prime,
            n_prime,
        )
        return mat_elem

    def _eps_ab_2(
        self,
        ell,
        m,
        ell_prime,
        m_prime,
        n,
        evals_minus,
        evals_a,
        evals_b,
        phi_a_mat,
        phi_b_mat,
        phi_minus_mat,
    ):
        assert ell != ell_prime and m != m_prime
        low_high = self._low_high_cross_mat_elem(
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
            ell_prime,
            m_prime,
            n,
        )
        high_high = self._high_high_cross_mat_elem(
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
            ell_prime,
            m_prime,
            n,
        )
        return low_high - high_high

    def _generate_fluxonia_evals_phi_for_SW(self):
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_a.flux = 0.5
        fluxonium_b.flux = 0.5
        fluxonium_minus = self.fluxonium_minus()
        evals_a, _, phi_a_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
            fluxonium_a
        )
        evals_b, _, phi_b_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
            fluxonium_b
        )
        evals_minus, _, phi_minus_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
            fluxonium_minus
        )
        return evals_a, phi_a_mat, evals_b, phi_b_mat, evals_minus, phi_minus_mat

    def second_order_generator(self):
        f_a_dim = f_b_dim = self.fluxonium_truncated_dim
        f_m_dim = self.fluxonium_minus_truncated_dim
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        low_high = sum(
            self._low_high_mat_elem(
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
                ell_prime,
                m_prime,
                n,
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ell_prime, m_prime, n, 0).dag()
            for ell in range(2)
            for m in range(2)
            for ell_prime in range(f_a_dim)
            for m_prime in range(f_b_dim)
            for n in range(1, f_m_dim)
        )
        S2 = -low_high + low_high.dag()
        high_high = sum(
            self._high_high_mat_elem(
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
                ell_prime,
                m_prime,
                n_prime,
            )
            * self._bare_product_state_all(ell, m, 0, 0)
            * self._bare_product_state_all(ell_prime, m_prime, n_prime, 0).dag()
            for ell in range(2)
            for m in range(2)
            for ell_prime in range(f_a_dim)
            for m_prime in range(f_b_dim)
            for n_prime in range(1, f_m_dim)
        )
        S2 += high_high - high_high.dag()
        return S2

    def _single_qubit_first_order(
        self, evals_minus, evals_i, phi_i_mat, phi_minus_mat, ELi, i, j
    ):
        f_m_dim = self.fluxonium_minus_truncated_dim
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
            for n in range(1, f_m_dim)
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
            phi_a_mat[ell, ell_prime]
            * self._eps_1(
                evals_minus,
                evals_a,
                phi_a_mat,
                phi_minus_mat,
                self.ELa,
                i=ell,
                j=ell_prime,
                n=0,
            )
            for ell_prime in range(2, self.fluxonium_truncated_dim)
        )
        first_order_b = self.ELb * sum(
            phi_b_mat[m, m_prime]
            * self._eps_1(
                evals_minus,
                evals_b,
                phi_b_mat,
                phi_minus_mat,
                self.ELb,
                i=m,
                j=m_prime,
                n=0,
            )
            for m_prime in range(2, self.fluxonium_truncated_dim)
        )
        return zeroth_order + first_order_a + first_order_b

    def _single_qubit_H_c(
        self, ell, evals_q, phi_q_mat, evals_minus, phi_minus_mat, ELq, which=0
    ):
        ell_prime = (ell + 1) % 2
        zeroth_order = 0.5 * ELq * phi_q_mat[0, 1]
        first_order = (
            -0.25
            * self.EL_tilda()
            * self._single_qubit_first_order(
                evals_minus, evals_q, phi_q_mat, phi_minus_mat, ELq, ell, ell_prime
            )
        )
        return (-1) ** which * (zeroth_order + first_order)

    def H_c_XI(self, ell):
        (
            evals_a,
            phi_a_mat,
            _,
            _,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        return self._single_qubit_H_c(
            ell, evals_a, phi_a_mat, evals_minus, phi_minus_mat, self.ELa, which=0
        )

    def H_c_IX(self, m):
        (
            _,
            _,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        return self._single_qubit_H_c(
            m, evals_b, phi_b_mat, evals_minus, phi_minus_mat, self.ELb, which=1
        )

    def H_c_XX_onlyS1(self, ell, m):
        return -0.25 * self.EL_tilda() * self.theta_minus_XX_onlyS1(ell, m)

    def H_c_XX(self, ell, m):
        return -0.25 * self.EL_tilda() * self.theta_minus_XX(ell, m)

    def _first_order_generator_squared(
        self,
        evals_a,
        phi_a_mat,
        evals_b,
        phi_b_mat,
        evals_minus,
        phi_minus_mat,
        ell,
        ell_prime,
        m,
        m_prime,
        n,
        n_prime,
    ):
        return self._eps_1(
            evals_minus,
            evals_a,
            phi_a_mat,
            phi_minus_mat,
            self.ELa,
            i=ell,
            j=ell_prime,
            n=n,
        ) * self._eps_1(
            evals_minus,
            evals_b,
            phi_b_mat,
            phi_minus_mat,
            self.ELb,
            i=m_prime,
            j=m,
            n=n_prime,
        ) + self._eps_1(
            evals_minus,
            evals_a,
            phi_a_mat,
            phi_minus_mat,
            self.ELa,
            i=ell_prime,
            j=ell,
            n=n,
        ) * self._eps_1(
            evals_minus,
            evals_b,
            phi_b_mat,
            phi_minus_mat,
            self.ELb,
            i=m,
            j=m_prime,
            n=n_prime,
        )

    def theta_minus_XX(self, ell, m):
        return self.theta_minus_XX_onlyS1(ell, m) + self.theta_minus_XX_onlyS2(ell, m)

    def theta_minus_XX_onlyS2(self, ell, m):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        f_m_dim = self.fluxonium_minus_truncated_dim
        ell_prime = (ell + 1) % 2
        m_prime = (m + 1) % 2
        S2_contr = sum(
            (
                self._eps_ab_2(
                    ell,
                    m,
                    ell_prime,
                    m_prime,
                    n,
                    evals_minus,
                    evals_a,
                    evals_b,
                    phi_a_mat,
                    phi_b_mat,
                    phi_minus_mat,
                )
                + self._eps_ab_2(
                    ell_prime,
                    m_prime,
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
            for n in range(1, f_m_dim)
        )
        return S2_contr

    def theta_minus_XX_onlyS1(self, ell, m):
        (
            evals_a,
            phi_a_mat,
            evals_b,
            phi_b_mat,
            evals_minus,
            phi_minus_mat,
        ) = self._generate_fluxonia_evals_phi_for_SW()
        f_m_dim = self.fluxonium_minus_truncated_dim
        ell_prime = (ell + 1) % 2
        m_prime = (m + 1) % 2
        S1_m_contr = sum(
            self._first_order_generator_squared(
                evals_a,
                phi_a_mat,
                evals_b,
                phi_b_mat,
                evals_minus,
                phi_minus_mat,
                ell,
                ell_prime,
                m,
                m_prime,
                n,
                n_prime,
            )
            * phi_minus_mat[n, n_prime]
            for n in range(1, f_m_dim)
            for n_prime in range(1, f_m_dim)
        )
        S1_0_contr = (
            sum(
                self._first_order_generator_squared(
                    evals_a,
                    phi_a_mat,
                    evals_b,
                    phi_b_mat,
                    evals_minus,
                    phi_minus_mat,
                    ell,
                    ell_prime,
                    m,
                    m_prime,
                    n,
                    n,
                )
                for n in range(1, f_m_dim)
            )
            * phi_minus_mat[0, 0]
        )
        return -S1_m_contr + S1_0_contr

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
                phi_q_mat[ell, ell_prime]
                * self._eps_1(
                    evals_minus,
                    evals_q,
                    phi_q_mat,
                    phi_minus_mat,
                    ELq,
                    i=ell,
                    j=ell_prime,
                    n=0,
                )
                for ell_prime in range(2, self.fluxonium_truncated_dim)
            )
        )
        return pref * (zeroth_order - first_order)

    def _H_q_eps_prod(
        self, evals_q, phi_q_mat, evals_minus, phi_minus_mat, ELq, ell, ell_prime, n
    ):
        return self._eps_1(
            evals_minus,
            evals_q,
            phi_q_mat,
            phi_minus_mat,
            ELq,
            i=ell,
            j=ell_prime,
            n=n,
        ) + self._eps_1(
            evals_minus,
            evals_q,
            phi_q_mat,
            phi_minus_mat,
            ELq,
            i=ell_prime,
            j=ell,
            n=n,
        )

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
        ell_prime = (ell + 1) % 2
        ell_osc = self.h_o_plus().l_osc
        zeroth_order = -ELq * phi_q_mat[0, 1]
        first_order_minus = (
            0.5
            * ELq
            * sum(
                phi_minus_mat[0, n]
                * self._H_q_eps_prod(
                    evals_q,
                    phi_q_mat,
                    evals_minus,
                    phi_minus_mat,
                    ELq,
                    ell,
                    ell_prime,
                    n,
                )
                for n in range(1, self.fluxonium_minus_truncated_dim)
            )
        )
        first_order_plus = (
            0.5
            * ELq
            * (ell_osc / np.sqrt(2))
            * (
                self._eps_1_plus(evals_q, phi_q_mat, ELq, i=ell, j=ell_prime)
                + self._eps_1_plus(evals_q, phi_q_mat, ELq, i=ell_prime, j=ell)
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
        m_prime = (m + 1) % 2
        ell_osc = self.h_o_plus().l_osc
        first_order_minus = (
            -0.5
            * EL_outside
            * sum(
                phi_minus_mat[0, n]
                * (
                    self._H_q_eps_prod(
                        evals_q,
                        phi_q_mat,
                        evals_minus,
                        phi_minus_mat,
                        ELq,
                        m,
                        m_prime,
                        n,
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
                self._eps_1_plus(evals_q, phi_q_mat, ELq, i=m, j=m_prime)
                + self._eps_1_plus(evals_q, phi_q_mat, ELq, i=m_prime, j=m)
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
            for ell_prime, m_prime in two_qubit_idxs:
                if which == "a":
                    idx_comp_row = ell
                    idx_comp_col = ell_prime
                else:
                    idx_comp_row = m
                    idx_comp_col = m_prime
                row_idx = 2 * ell + m
                col_idx = 2 * ell_prime + m_prime
                if ell == ell_prime and m == m_prime:
                    H_q[row_idx, col_idx] = self.H_q_diag(ell, which=which)
                elif ell == (ell_prime + 1) % 2 and m == (m_prime + 1) % 2:
                    H_q[row_idx, col_idx] = self.H_q_XX(ell, m)
                elif idx_comp_row == (idx_comp_col + 1) % 2:
                    H_q[row_idx, col_idx] = self.H_q_correct_X(
                        idx_comp_row, which=which
                    )
                else:
                    # m entry below doesn't matter
                    H_q[row_idx, col_idx] = self.H_q_wrong_X(0, which=which)
        return H_q

    def construct_H_c_eff(self, only_S1=False):
        qubit_idx_range = [0, 1]
        two_qubit_idxs = list(product(qubit_idx_range, qubit_idx_range))
        H_c = np.zeros((4, 4))
        for ell, m in two_qubit_idxs:
            for ell_prime, m_prime in two_qubit_idxs:
                row_idx = 2 * ell + m
                col_idx = 2 * ell_prime + m_prime
                if ell == ell_prime and m == m_prime:
                    H_c[row_idx, col_idx] = self.H_c_diag(ell, m)
                elif ell == (ell_prime + 1) % 2 and m == (m_prime + 1) % 2:
                    if not only_S1:
                        H_c[row_idx, col_idx] = self.H_c_XX(ell, m)
                    else:
                        H_c[row_idx, col_idx] = self.H_c_XX_onlyS1(ell, m)
                elif ell == (ell_prime + 1) % 2:
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

    def _g_minus(self, ell, ell_prime, n, phi_q_mat, phi_minus_mat, EL):
        return 0.5 * EL * phi_q_mat[ell, ell_prime] * phi_minus_mat[0, n]

    def _g_minus_2(self, ell, ell_prime, n, n_prime, phi_q_mat, phi_minus_mat, EL):
        """relevant for second order generator"""
        return 0.5 * EL * phi_q_mat[ell, ell_prime] * phi_minus_mat[n, n_prime]

    def _g_plus(self, ell, ell_prime, phi_q_mat, EL):
        return 0.5 * EL * phi_q_mat[ell, ell_prime] * self.h_o_plus().l_osc / np.sqrt(2)

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
        f_m_dim = fluxonium_minus.truncated_dim
        f_a_dim = fluxonium_a.truncated_dim
        f_b_dim = fluxonium_b.truncated_dim
        evals_a, _, phi_a_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
            fluxonium_a
        )
        evals_b, _, phi_b_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
            fluxonium_b
        )
        evals_minus, _, phi_minus_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
            fluxonium_minus
        )
        return (
            -0.25
            * ELt
            * tensor(
                qeye(f_a_dim),
                qeye(f_b_dim),
                Qobj(phi_minus_mat),
                qeye(self.h_o_truncated_dim),
            )
            + 0.5
            * ELa
            * tensor(
                Qobj(phi_a_mat),
                qeye(f_b_dim),
                qeye(f_m_dim),
                qeye(self.h_o_truncated_dim),
            )
            - 0.5
            * ELb
            * tensor(
                qeye(f_a_dim),
                Qobj(phi_b_mat),
                qeye(f_m_dim),
                qeye(self.h_o_truncated_dim),
            )
        )

    def _bare_product_state_all(self, ell, m, n, p):
        return tensor(
            basis(self.fluxonium_truncated_dim, ell),
            basis(self.fluxonium_truncated_dim, m),
            basis(self.fluxonium_minus_truncated_dim, n),
            basis(self.h_o_truncated_dim, p),
        )

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
            g_strength=-0.5 * self.ELa, op1=phi_a, op2=phi_plus, check_validity=False
        )
        hilbert_space.add_interaction(
            g_strength=-0.5 * self.ELb, op1=phi_b, op2=phi_plus, check_validity=False
        )
        hilbert_space.add_interaction(
            g_strength=-0.5 * self.ELa, op1=phi_a, op2=phi_minus, check_validity=False
        )
        hilbert_space.add_interaction(
            g_strength=0.5 * self.ELb, op1=phi_b, op2=phi_minus, check_validity=False
        )
        hilbert_space.add_interaction(
            g_strength=-8.0 * self.off_diagonal_charging(), op1=n_a, op2=n_b, check_validity=False
        )
        hilbert_space.add_interaction(
            g_strength=(self.ELa - self.ELb + self.EL1 - self.EL2) / 2.0,
            op1=phi_plus,
            op2=phi_minus,
            check_validity=False,
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
        fluxonium_a = self.fluxonium_a()
        fluxonium_b = self.fluxonium_b()
        fluxonium_a.flux, fluxonium_b.flux = 0.5, 0.5
        fluxonium_minus = self.fluxonium_minus()
        _, _, phi_a_mat = self.signed_evals_evecs_phi_mat_qubit_instance(fluxonium_a)
        _, _, phi_b_mat = self.signed_evals_evecs_phi_mat_qubit_instance(fluxonium_b)
        _, _, phi_minus_mat = self.signed_evals_evecs_phi_mat_qubit_instance(
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
            flux_shift_a = self.find_flux_shift_exact()
        self.flux_a = 0.5 + flux_shift_a
        self.flux_b = 0.5 - flux_shift_a
        return self.generate_coupled_system()

    @staticmethod
    def basis_change(op, evecs, hilbert_space, subsystem):
        evecs_bare = hilbert_space.lookup.bare_eigenstates(subsystem)
        op_id_wrap = identity_wrap(op, subsystem, hilbert_space.subsys_list, evecs=evecs_bare)
        op_new_basis = np.real(evecs.T @ op_id_wrap.data @ evecs)
        return Qobj(op_new_basis)

    def operators_at_sweetspot(
        self,
        num_states=4,
        flux_shift_a=None,
        flux_shift_b=None,
        remove_unnecessary_states=False,
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
            states_to_keep = np.concatenate(
                (np.arange(4), np.array(bad_dressed_indices))
            )
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
        return self._J(
            evals_a, phi_a_mat, evals_b, phi_b_mat, evals_minus, phi_minus_mat
        )

    def off_location_coupler_flux(self, epsilon=1e-3):
        def _find_J(flux_c):
            self.flux_c = flux_c
            return np.abs(self.J_eff_total())
        result = minimize(_find_J, x0=np.array([0.28]), bounds=((0.25, 0.5),),
                          tol=epsilon)
        assert result.success
        if not (np.abs(result.fun) < 0.001):
            print(
                warnings.warn(
                    f"off value of J is {result.fun}", Warning
                )
            )
        return result.x[0]

    def _evals_zeroed(self):
        evals, _ = self.generate_coupled_system().hamiltonian().eigenstates(eigvals=4)
        return evals - evals[0]

    def _cost_function_off_and_shift_positions(self, fluxes):
        """For efficiency we make the approximation that the flux shifts are equivalent"""
        flux_c, flux_a = fluxes
        flux_shift_a = flux_a - 0.5
        flux_b = 0.5 - flux_shift_a
        self.flux_a = flux_a
        self.flux_b = flux_b
        self.flux_c = flux_c
        return self._evals_zeroed()[3]

    def _cost_function_just_shift_positions(self, flux_a):
        """For efficiency we make the approximation that the flux shifts are equivalent"""
        flux_shift_a = flux_a - 0.5
        flux_b = 0.5 - flux_shift_a
        self.flux_a = flux_a
        self.flux_b = flux_b
        return self._evals_zeroed()[3]

    def find_flux_shift_exact(self, epsilon=1e-4):
        """near the off position, we want to find the exact qubit fluxes necessary to
        put the qubits at their sweet spots. To do this we acknowledge that the qubits
        are (nearly) uncoupled, therefore each excited state is nearly a product state.
        Thus if we vary the qubit fluxes and minimize the excitation energies, we
        should be able to place both qubits at their sweet spots independently"""
        flux_shift_a_seed, _ = self.find_flux_shift()

        result = minimize(
            self._cost_function_just_shift_positions,
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
            bounds=((0.25, 0.5), (0.5, 0.6)),
            tol=epsilon,
        )
        assert result.success
        return result.x[0], result.x[1] - 0.5

    def off_location_effective_sweet_spot_fluxes(self):
        flux_c = self.off_location_coupler_flux()
        self.flux_c = flux_c
        flux_shift_a, flux_shift_b = self.find_flux_shift()
        return flux_c, flux_shift_a, flux_shift_b

    @staticmethod
    def decompose_matrix_into_specific_paulis(sigmai, sigmaj, matrix):
        sigmaij = tensor(sigmai, sigmaj)
        return 0.25 * np.trace((sigmaij * matrix).data.toarray())

    @staticmethod
    def decompose_matrix_into_paulis(matrix):
        pauli_mats = [qeye(2), sigmax(), sigmay(), sigmaz()]
        pauli_name = ["I", "X", "Y", "Z"]
        pauli_dict = {}
        for j, pauli_a in enumerate(pauli_mats):
            for k, pauli_b in enumerate(pauli_mats):
                paulia_a_b = tensor(pauli_a, pauli_b)
                coeff = 0.25 * np.trace((paulia_a_b * matrix).data.toarray())
                pauli_dict[pauli_name[j] + pauli_name[k]] = coeff
        return pauli_dict

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
    def __init__(
        self,
        H_0,
        H_a,
        H_b,
        H_c,
        control_dt_slow=2.0,
        control_dt_fast=0.01,
        max_freq=0.255,
        min_freq=0.125,
    ):
        self.H_0 = H_0
        self.H_a = H_a
        self.H_b = H_b
        self.H_c = H_c
        self.omega_a = np.real(self.H_0[2, 2])
        self.omega_b = np.real(self.H_0[1, 1])
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
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1 / sqrt2, -1j / sqrt2, 0.0],
                [0.0, -1j / sqrt2, 1 / sqrt2, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def sqrtdSWAP(down=0):
        sqrt2 = np.sqrt(2.0)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1 / sqrt2, (-1)**(down+1) / sqrt2, 0.0],
                [0.0, (-1)**down / sqrt2, 1 / sqrt2, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def normalized_operators(self):
        norm_a = self.H_a[0, 2]
        norm_b = self.H_b[0, 1]
        norm_c = self.H_c[0, 3]
        XI = self.H_a / norm_a
        IX = self.H_b / norm_b
        XX = self.H_c / norm_c
        return (norm_a, norm_b, norm_c), (XI, IX, XX)

    def RZ(self, theta, which="a"):
        if which is "a":
            Z = self.ZA()
        else:
            Z = self.ZB()
        return Qobj((-1j * theta * Z / 2.0).expm().data, dims=[[4], [4]])

    @staticmethod
    def fix_w_single_q_gates(gate_, which_Z_exclude=2, const_angle_val=0.0, which_gate='sqrtiswap'):
        Z_matrix = 0.5 * np.array([[-1, -1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]])
        const_angle_col = Z_matrix[:, which_Z_exclude]
        new_Z_matrix = np.delete(Z_matrix, np.array(which_Z_exclude), axis=1)
        inv_Z_matrix = inv(new_Z_matrix)
        alpha = cmath.phase(gate_[0, 0])
        beta = cmath.phase(gate_[1, 1])
        gamma = cmath.phase((-1)*gate_[1, 2])
        if which_gate == 'sqrtiswap':
            angles_to_correct = (
                np.array([-alpha, -beta, gamma + np.pi / 2])
                - const_angle_val * const_angle_col
            )
        elif which_gate == 'sqrtdswap':
            angles_to_correct = (
                np.array([-alpha, -beta, -gamma])
                - const_angle_val * const_angle_col
            )
        elif which_gate == 'sqrtdswap_down':
            angles_to_correct = (
                np.array([-alpha, -beta, gamma])
                - const_angle_val * const_angle_col
            )
        else:
            raise RuntimeError('not a recognized gate')
        Z_rotation_angles = inv_Z_matrix @ angles_to_correct
        Z_rotation_angles = np.insert(
            Z_rotation_angles, which_Z_exclude, const_angle_val
        )
        return Z_rotation_angles

    def return_single_q_gate_unitaries(self, gate, which_gate='sqrtiswap'):
        (t1, t2, t3, t4) = self.fix_w_single_q_gates(gate, which_gate=which_gate)
        return self.RZ(t1, which='a'), self.RZ(t2, which='b'), self.RZ(t3, which='a'), self.RZ(t4, which='b')

    def return_single_q_gate_superops(self, gate, which_gate='sqrtiswap'):
        superops_list = []
        gates = self.return_single_q_gate_unitaries(gate, which_gate=which_gate)
        for gate in gates:
            superops_list.append(spre(gate) * spost(gate.dag()))
        return superops_list

    def multiply_with_single_q_superops(self, superop_gate, coherent_gate, which_gate='sqrtiswap'):
        (RZa1, RZb1, RZa2, RZb2) = self.return_single_q_gate_superops(coherent_gate, which_gate=which_gate)
        return RZa1 * RZb1 * superop_gate * RZa2 * RZb2

    def multiply_with_single_q_gates(self, gate, which_gate='sqrtiswap'):
        (t1, t2, t3, t4) = self.fix_w_single_q_gates(gate, which_gate=which_gate)
        gate_ = Qobj(gate[0:4, 0:4])
        return (
            self.RZ(t1, which="a")
            * self.RZ(t2, which="b")
            * gate_
            * self.RZ(t3, which="a")
            * self.RZ(t4, which="b")
        )

    @staticmethod
    def calc_fidel_4(prop, gate):
        prop = Qobj(prop[0:4, 0:4])
        return (
            np.abs(
                np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2
            )
            / 20
        )

    @staticmethod
    def calc_fidel_2(prop, gate):
        prop = Qobj(prop[0:2, 0:2])
        return (
            np.abs(
                np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2
            )
            / 6
        )

    @staticmethod
    def get_controls_only_sine(freq, amp, control_dt=0.01):
        sin_time = 1.0 / freq
        sin_eval_times = np.linspace(0.0, sin_time, int(sin_time / control_dt) + 1)
        sin_pulse = amp * np.sin(2.0 * np.pi * freq * sin_eval_times)
        return sin_pulse, sin_eval_times

    @staticmethod
    def amp_from_freq_id(freq):
        bessel_val = jn_zeros(0, 1)
        return 2.0 * np.pi * freq * bessel_val / 2.0

    @staticmethod
    def amp_from_freq_sqrtiswap(omega, omega_d, n=1):
        return np.abs(
            0.125
            * np.pi
            * (omega_d ** 2 - omega ** 2)
            / (omega_d * np.sin(n * np.pi * omega / omega_d))
        )

    def omega_plus(self):
        return np.abs(self.omega_b + self.omega_a)

    def omega_minus(self):
        return np.abs(self.omega_b - self.omega_a)

    def drive_freq_sqrtiswap(self, m=4):
        omega_a = np.real(self.H_0[2, 2])
        omega_b = np.real(self.H_0[1, 1])
        return (omega_a + omega_b) / m

    def optimize_amp_id_fidel(self, amp, freq, which_qubit="a"):
        times = np.linspace(0.0, 1.0 / freq, int(1.0 / freq / self.control_dt_fast) + 1)
        omega_a = np.real(self.H_0[2, 2])
        omega_b = np.real(self.H_0[1, 1])
        _, (XI, IX, XX) = self.normalized_operators()
        red_dim = 4

        def control_func(t, args=None):
            return amp * np.sin(2.0 * np.pi * freq * t)

        if which_qubit == "a":  # driving qubit a
            drive_H = Qobj(XI[0:red_dim, 0:red_dim])
            ideal_prop = self.RZ(-times[-1] * omega_b, which="b")
        else:
            drive_H = Qobj(IX[0:red_dim, 0:red_dim])
            ideal_prop = self.RZ(-times[-1] * omega_a, which="a")
        H = [Qobj(self.H_0[0:red_dim, 0:red_dim]), [drive_H, control_func]]
        prop = propagator(H, times)
        return 1 - self.calc_fidel_4(prop[-1], ideal_prop)

    def synchronize(self, ta, tb):
        if ta == tb == 0.0:
            return None
        if ta <= tb:
            return self._synchronize(ta, tb)
        else:
            output, _ = self._synchronize(tb, ta)
            flipped_output = np.flip(output, axis=1)
            return flipped_output, (ta, tb)

    def _synchronize(self, ta, tb):
        """Assume ta <= tb"""
        tmax = 1.0 / self.max_freq
        tmin = 1.0 / self.min_freq
        if ta == tb == 0.0:
            return np.array([(None, None)]), (ta, tb)
        elif tmax <= (tb - ta) <= tmin:
            return np.array([(1.0 / (tb - ta), None)]), (ta, tb)
        elif (tb - ta) < tmax:
            new_freq = (tb - ta + tmax) ** (-1)
            return np.array([(new_freq, self.max_freq)]), (ta, tb)
        else:  # (tb - ta) > 1. / min_freq
            trial_time, n, r = self._remainder_search(tb - ta)
            return (
                np.array(int(n) * ((1.0 / trial_time, None),) + ((1.0 / r, None),)),
                (ta, tb),
            )

    def _remainder_search(self, tdiff):
        max_time = 1.0 / self.min_freq
        min_time = 1.0 / self.max_freq
        time_linspace = np.linspace(max_time, min_time, 101)
        for trial_time in time_linspace:
            n, r = divmod(tdiff, trial_time)
            if min_time <= r <= max_time:
                return trial_time, n, r
        raise (
            RuntimeError(
                "no sequence of identity pulses found to synchronize the two qubits"
            )
        )

    def _concatenate_for_qubit(self, freq, total_pulse, total_times, which_qubit="a"):
        amp_0 = self.amp_from_freq_id(freq)
        optimized_amp = minimize(
            self.optimize_amp_id_fidel, x0=np.array([amp_0]), args=(freq, which_qubit)
        )
        if not optimized_amp.success:
            print(optimized_amp)
            print(
                warnings.warn(
                    "optimization of the qubit pulses did not succeed", Warning
                )
            )
            amp = amp_0
        else:
            print(
                f"optimized qubit {which_qubit} id pulse with time {1./freq} and"
                f" with infidelity 1-F={optimized_amp.fun}"
            )
            amp = optimized_amp.x[0]
        controls, times = self.get_controls_only_sine(freq, amp, self.control_dt_fast)
        total_pulse = self.concatenate_times_or_controls(
            (total_pulse, controls), self.concatenate_two_controls
        )
        total_times = self.concatenate_times_or_controls(
            (total_times, times), self.concatenate_two_times
        )
        return total_pulse, total_times

    def parse_synchronize(
        self, synchronize_output
    ) -> Optional[Tuple[ndarray, ndarray, ndarray, ndarray]]:
        """This function takes the output of synchronize and yields
        the pulses along with the times that synchronize specified"""
        total_pulse_a = np.array([])
        total_pulse_b = np.array([])
        total_times_a = np.array([])
        total_times_b = np.array([])
        if synchronize_output is None:
            return None
        output, (t_a, t_b) = synchronize_output
        control_dt = self.control_dt_fast
        for (freq_a_, freq_b_) in output:
            if freq_a_ is not None:
                total_pulse_a, total_times_a = self._concatenate_for_qubit(
                    freq_a_, total_pulse_a, total_times_a, which_qubit="a"
                )
            if freq_b_ is not None:
                total_pulse_b, total_times_b = self._concatenate_for_qubit(
                    freq_b_, total_pulse_b, total_times_b, which_qubit="b"
                )
        # here we add the delay part that actually gives us Z rotations
        delay_time_a = np.linspace(0.0, t_a, int(t_a / control_dt) + 1)
        delay_time_b = np.linspace(0.0, t_b, int(t_b / control_dt) + 1)
        total_times_a = self.concatenate_times_or_controls(
            (total_times_a, delay_time_a), self.concatenate_two_times
        )
        total_times_b = self.concatenate_times_or_controls(
            (total_times_b, delay_time_b), self.concatenate_two_times
        )
        total_pulse_a = self.concatenate_times_or_controls(
            (total_pulse_a, np.zeros_like(delay_time_a)), self.concatenate_two_controls
        )
        total_pulse_b = self.concatenate_times_or_controls(
            (total_pulse_b, np.zeros_like(delay_time_b)), self.concatenate_two_controls
        )
        return total_pulse_a, total_times_a, total_pulse_b, total_times_b

    def concatenate_times_or_controls(self, t_c_tuple: tuple, concatenator: Callable):
        if len(t_c_tuple) == 1:
            return t_c_tuple[0]
        concat_first_two = concatenator(t_c_tuple[0], t_c_tuple[1])
        if len(t_c_tuple) == 2:
            return concat_first_two
        return self.concatenate_times_or_controls(
            (concat_first_two,) + t_c_tuple[2:], concatenator
        )

    @staticmethod
    def concatenate_two_times(times_1: ndarray, times_2: ndarray) -> ndarray:
        if times_1.size == 0:
            return times_2
        if times_2.size == 0:
            return times_1
        return np.concatenate((times_1, times_1[-1] + times_2[1:]))

    @staticmethod
    def concatenate_two_controls(controls_1: ndarray, controls_2: ndarray) -> ndarray:
        if controls_1.size == 0:
            return controls_2
        if controls_2.size == 0:
            return controls_1
        assert np.allclose(controls_1[-1], 0.0) and np.allclose(controls_2[-1], 0.0)
        return np.concatenate((controls_1, controls_2[1:]))

    def propagator_for_coupler_segment(
        self, amp: float, omega_d: float, num_periods: int = 2, num_cpus: int = 1, c_ops=None
    ) -> Qobj:
        """
        Parameters
        ----------
        amp
            amplitude of the pulse in angular frequency units
        omega_d
            frequency of the pulse in angular frequency units
        num_periods
            number of periods of driving
        num_cpus
            number cpus
        c_ops
            collapse operators. If specified, the resulting propagator
            is a superoperator of dimension dim**2 x dim**2

        Returns
        -------
            when c_ops is none, returns the propagator in the qubit subspace.
            If c_ops is not None, the propagator is a superoperator
            that has not yet been truncated to the qubit subspace

        """
        total_time = num_periods * 2.0 * np.pi / omega_d
        _, (_, _, XX) = self.normalized_operators()
        H = [self.H_0, [XX, lambda t, a: amp * np.sin(omega_d * t)]]
        twoqcontrol_eval_times = np.linspace(
            0.0, total_time, int(total_time / self.control_dt_slow) + 1
        )
        if c_ops is None:
            return self.my_propagator(H, self.dim, twoqcontrol_eval_times, num_cpus)
        else:
            return propagator(H, twoqcontrol_eval_times, c_ops)[-1]

    @staticmethod
    def my_propagator(H, dim, times, num_cpus=1):
        target_map = cpu_switch.get_map_method(num_cpus)
        my_prop = np.zeros((4, 4), dtype=complex)

        def _run_sesolve(initial_state):
            return sesolve(
                H,
                basis(dim, initial_state),
                times,
                options=Options(store_final_state=True),
            ).final_state.data.toarray()[0:4, 0]

        initial_states = range(4)
        result = list(target_map(_run_sesolve, initial_states))
        for i in range(4):
            my_prop[:, i] = result[i]
        return Qobj(my_prop)

    def propagator_for_qubit_flux_segment(
        self, parse_synchronize_output, red_dim=4, num_cpus=1, c_ops=None
    ):
        if parse_synchronize_output is None:
            return qeye(red_dim)
        pulse_a, times_a, pulse_b, times_b = parse_synchronize_output
        _, (XI, IX, _) = self.normalized_operators()
        spline_a = interp1d(times_a, pulse_a, fill_value="extrapolate")
        spline_b = interp1d(times_b, pulse_b, fill_value="extrapolate")
        H = [
            Qobj(self.H_0[0:red_dim, 0:red_dim]),
            [Qobj(XI[0:red_dim, 0:red_dim]), lambda t, a: spline_a(t)],
            [Qobj(IX[0:red_dim, 0:red_dim]), lambda t, a: spline_b(t)],
        ]
        if c_ops is None:
            return self.my_propagator(H, red_dim, times_a, num_cpus), times_a[-1]
        else:
            return propagator(H, times_a, c_ops)[-1], times_a[-1]

    def times_to_correct_prop(
        self, prop, which_Z_exclude: int = 2, const_angle_val: float = 0.0, which_gate: str = 'sqrtiswap'
    ):
        angles = self.fix_w_single_q_gates(
            prop, which_Z_exclude=which_Z_exclude, const_angle_val=const_angle_val, which_gate=which_gate
        )
        neg_angles = -angles % (2.0 * np.pi)
        omega_a = np.real(self.H_0[2, 2])
        omega_b = np.real(self.H_0[1, 1])
        times = neg_angles / np.array([omega_a, omega_b, omega_a, omega_b])
        return times

    def time_evolution_initial_state_full_pulse(
        self,
        initial_state: int,
        amp: float,
        omega_d: float,
        num_periods: int = 2,
        num_cpus: int = 1,
        c_ops=None,
        which_Z_exclude: int = 2,
        const_angle_val: float = 0.0,
        which_gate='sqrtiswap'
    ):
        (
            spline_a,
            spline_b,
            spline_c,
            total_times_a,
            total_times_b,
            total_times_c,
        ) = self.all_control_functions(
            amp,
            omega_d,
            num_periods,
            num_cpus=num_cpus,
            which_Z_exclude=which_Z_exclude,
            const_angle_val=const_angle_val,
            which_gate=which_gate
        )
        (norm_a, norm_b, norm_c), (XI, IX, XX) = self.normalized_operators()
        H = [
            self.H_0,
            [XI, lambda t, a: 2.0 * np.pi * norm_a * spline_a(t)],
            [IX, lambda t, a: 2.0 * np.pi * norm_b * spline_b(t)],
            [XX, lambda t, a: 2.0 * np.pi * norm_c * spline_c(t)],
        ]
        e_ops = [basis(self.dim, i) * basis(self.dim, i).dag() for i in range(4)]
        if c_ops is None:
            result = sesolve(
                H,
                basis(self.dim, initial_state),
                total_times_a,
                e_ops,
                options=Options(store_final_state=True),
            )
            return result
        else:
            rho0 = basis(self.dim, initial_state) * basis(self.dim, initial_state).dag()
            return mesolve(H, rho0, total_times_a, c_ops, e_ops,
                           options=Options(store_final_state=True))

    @staticmethod
    def global_phase(prop):
        """most useful when accounting for phases due to higher levels"""
        return (cmath.phase(prop[0, 0]) + cmath.phase(prop[3, 3])) / 2

    def propagator_for_full_pulse(
        self,
        amp: float,
        omega_d: float,
        num_periods: int = 2,
        red_dim: int = 4,
        num_cpus: int = 1,
        c_ops=None,
        which_Z_exclude: int = 2,
        const_angle_val: float = 0.0,
        which_gate: str = 'sqrtiswap'
    ):
        """

        Parameters
        ----------
        amp
            amplitude of the coupler pulse
        omega_d
            frequency of the coupler pulse
        num_periods
            number of periods of the coupler pulse
        red_dim
            how many states to restrict time evolution to for
            qubit flux segment
        num_cpus
            number cpus

        Returns
        -------
        propagator of the full pulse, along with quantities of interest: the
        time necessary to idle for each qubit to achieve the required dynamical
        phase factors, along with the associated outputs from calling parse_synchronize

        """
        twoqprop = self.propagator_for_coupler_segment(
            amp, omega_d, num_periods=num_periods, num_cpus=num_cpus, c_ops=c_ops
        )
        zeroed_prop = twoqprop * np.exp(-1j * self.global_phase(twoqprop))
        times = self.times_to_correct_prop(
            zeroed_prop,
            which_Z_exclude=which_Z_exclude,
            const_angle_val=const_angle_val,
            which_gate=which_gate
        )
        before_prop, after_prop, single_q_time = self.construct_qubit_propagators(
            times, red_dim=red_dim, num_cpus=num_cpus, c_ops=c_ops
        )
        return (
            after_prop * zeroed_prop * before_prop,
            times,
            num_periods * 2.0 * np.pi / omega_d + single_q_time
        )

    def construct_qubit_propagators(self, times, red_dim=4, num_cpus=1, c_ops=None):
        parse_output_before = self.parse_synchronize(
            self.synchronize(times[2], times[3])
        )
        parse_output_after = self.parse_synchronize(
            self.synchronize(times[0], times[1])
        )
        before_prop, before_time = self.propagator_for_qubit_flux_segment(
            parse_output_before, red_dim=red_dim, num_cpus=num_cpus, c_ops=c_ops
        )
        after_prop, after_time = self.propagator_for_qubit_flux_segment(
            parse_output_after, red_dim=red_dim, num_cpus=num_cpus, c_ops=c_ops
        )
        return before_prop, after_prop, before_time + after_time

    def propagator_full_pulse_optimize_qubit_fluxes(
        self,
        twoq_prop,
        const_angle_array: ndarray = np.linspace(
            0.0, 2.0 * np.pi, num=10, endpoint=False
        ),
        which_gate: str = 'sqrtiswap',
        red_dim: int = 4,
        num_cpus: int = 1,
        c_ops=None,
    ):
        zeroed_prop = twoq_prop * np.exp(-1j * self.global_phase(twoq_prop))
        # find that minimize doesn't work well here to optimize over the initial angle
        # since we aren't necessarily close to a minimum and changes in infidelity are
        # so small. Cheaper just to brute-force scan
        max_fidel = 0.0
        if which_gate == 'sqrtiswap':
            ideal_gate = self.sqrtiSWAP()
        elif which_gate == 'sqrtdswap':
            ideal_gate = self.sqrtdSWAP()
        else:
            raise RuntimeError('target gate must be sqrtiswap or sqrtdswap')
        for const_angle_val in const_angle_array:
            times = self.times_to_correct_prop(
                zeroed_prop, which_Z_exclude=0, const_angle_val=const_angle_val, which_gate=which_gate
            )
            before_prop, after_prop, single_q_time = self.construct_qubit_propagators(
                times, red_dim=red_dim, num_cpus=num_cpus, c_ops=c_ops
            )
            full_prop = after_prop * zeroed_prop * before_prop
            fidel = self.calc_fidel_4(full_prop, ideal_gate)
            if fidel > max_fidel:
                max_const_angle = const_angle_val
                max_prop = full_prop
                max_fidel = fidel
                max_times = times
        return max_const_angle, max_prop, max_fidel, max_times

    @staticmethod
    def truncate_superoperator(superop, keep_dim, trunc_dim):
        """
        Parameters
        ----------
        superop
            superoperator to truncate. We consider the situation where high-lying
            states are relevant for predicting time evolution, however the gate under consideration
            does not care about the time evolution of those high-lying states themselves. We
            assume that the states are ordered such that all of the relevant states come first,
            and all of the states we'd like to truncate away come afterwards
        keep_dim
            dimension of the relevant subspace
        trunc_dim
            dimension of the subspace to truncate away
        Returns
        -------

        """
        truncated_mat = np.zeros((keep_dim ** 2, keep_dim ** 2), dtype=complex)
        total_dim = keep_dim + trunc_dim
        for i in range(keep_dim):  # sum over groups of columns
            for j in range(keep_dim):  # sum over individual columns
                full_column = superop.data.toarray()[:, total_dim * i + j]
                reduced_column = np.array([full_column[total_dim * k: total_dim * k + keep_dim]
                                           for k in range(keep_dim)]).flatten()
                truncated_mat[:, keep_dim*i+j] = reduced_column
        return Qobj(truncated_mat, type='super', dims=[[[keep_dim], [keep_dim]],
                                                       [[keep_dim], [keep_dim]]])

    def all_control_functions(
        self,
        amp: float,
        omega_d: float,
        num_periods: int = 2,
        num_cpus: int = 1,
        c_ops=None,
        which_Z_exclude: int = 2,
        const_angle_val: float = 0.0,
        which_gate: str = 'sqrtiswap',
        parsed_outputs=None,
    ):
        """

        Parameters
        ----------
        amp
            amplitude of the coupler pulse
        omega_d
            frequency of the coupler pulse
        num_periods
            number of periods of the coupler pulse
        num_cpus
            number cpus
        which_Z_exclude
            index of which Z rotation to exclude.
        parsed_outputs
            outputs of calling parse_synchronize for before and after the
            coupler pulse. If this is not given, it is calculated on the fly.
            The option to pass it in here reflects that it is costly to calculate,
            as it requires calculating the propagator for the coupler segment.

        Returns
        -------
        functions representing the flux modulation required to effect the pulse
        for all three fluxes

        """
        if parsed_outputs is None:
            twoqprop = self.propagator_for_coupler_segment(
                amp, omega_d, num_periods=num_periods, num_cpus=num_cpus, c_ops=c_ops
            )
            zeroed_prop = twoqprop * np.exp(-1j * self.global_phase(twoqprop))
            times = self.times_to_correct_prop(
                zeroed_prop,
                which_Z_exclude=which_Z_exclude,
                const_angle_val=const_angle_val,
                which_gate=which_gate
            )
            parse_output_before = self.parse_synchronize(
                self.synchronize(times[2], times[3])
            )
            parse_output_after = self.parse_synchronize(
                self.synchronize(times[0], times[1])
            )
        else:
            parse_output_before, parse_output_after = parsed_outputs
        twoq_time = num_periods * 2.0 * np.pi / omega_d
        twoqcontrol_eval_times = np.linspace(
            0.0, twoq_time, int(twoq_time / self.control_dt_slow) + 1
        )
        controls_2q = amp * np.sin(omega_d * twoqcontrol_eval_times)

        after_pulse_a, after_times_a, after_pulse_b, after_times_b = parse_output_after
        (
            before_pulse_a,
            before_times_a,
            before_pulse_b,
            before_times_b,
        ) = parse_output_before
        total_pulse_a = self.concatenate_times_or_controls(
            (before_pulse_a, np.zeros_like(controls_2q), after_pulse_a),
            self.concatenate_two_controls,
        )
        total_pulse_b = self.concatenate_times_or_controls(
            (before_pulse_b, np.zeros_like(controls_2q), after_pulse_b),
            self.concatenate_two_controls,
        )
        total_pulse_c = self.concatenate_times_or_controls(
            (np.zeros_like(before_pulse_a), controls_2q, np.zeros_like(after_pulse_a)),
            self.concatenate_two_controls,
        )
        total_times_a = self.concatenate_times_or_controls(
            (before_times_a, twoqcontrol_eval_times, after_times_a),
            self.concatenate_two_times,
        )
        total_times_b = self.concatenate_times_or_controls(
            (before_times_b, twoqcontrol_eval_times, after_times_b),
            self.concatenate_two_times,
        )
        total_times_c = self.concatenate_times_or_controls(
            (before_times_a, twoqcontrol_eval_times, after_times_a),
            self.concatenate_two_times,
        )
        (norm_a, norm_b, norm_c), _ = self.normalized_operators()
        spline_a = interp1d(
            total_times_a,
            total_pulse_a / norm_a / 2.0 / np.pi,
            fill_value="extrapolate",
        )
        spline_b = interp1d(
            total_times_b,
            total_pulse_b / norm_b / 2.0 / np.pi,
            fill_value="extrapolate",
        )
        spline_c = interp1d(
            total_times_c,
            total_pulse_c / norm_c / 2.0 / np.pi,
            fill_value="extrapolate",
        )
        return spline_a, spline_b, spline_c, total_times_a, total_times_b, total_times_c
