# circuit_noise.py
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

from abc import ABC
import operator as builtin_op
import functools
from numpy import ndarray
import numpy as np
import qutip as qt
import copy

from scqubits.core.noise import NOISE_PARAMS, NoisySystem
from scqubits.core.circuit_utils import get_trailing_number

from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy as sm

from scqubits.core.symbolic_circuit import Branch


class NoisyCircuit(NoisySystem, ABC):
    def __getattribute__(self, __name: str) -> Any:
        try:
            return self.__dict__["_data"][__name]
        except:
            return super().__getattribute__(__name)

    def _evaluate_symbolic_expr(self, sym_expr):
        expr_dict = sym_expr.as_coefficients_dict()
        terms = list(expr_dict.keys())

        eval_matrix_list = []

        for idx, term in enumerate(terms):
            coefficient_sympy = expr_dict[term]
            if any([arg.has(sm.cos) or arg.has(sm.sin) for arg in (1.0 * term).args]):
                eval_matrix_list.append(
                    float(coefficient_sympy) * self._evaluate_matrix_cosine_terms(term)
                )
            else:
                product_matrix_list = []
                for free_sym in term.free_symbols:
                    product_matrix_list.append(self.get_operator_by_name(free_sym.name))
                eval_matrix_list.append(
                    float(coefficient_sympy)
                    * functools.reduce(builtin_op.mul, product_matrix_list)
                )
        return sum(eval_matrix_list)

    def _transform_expr_to_new_variables(
        self, expr_node_vars: sm.Expr, substitute_symbol: Optional[str] = None
    ):
        transformation_mat = self.transformation_matrix
        expr_node_vars = expr_node_vars.expand()
        num_vars = len(self.symbolic_circuit._node_list_without_ground)
        new_vars = [sm.symbols(f"θ{index}") for index in range(1, 1 + num_vars)]
        old_vars = [sm.symbols(f"φ{index}") for index in range(1, 1 + num_vars)]
        transformed_expr = transformation_mat.dot(new_vars)
        for idx, var in enumerate(old_vars):
            expr_node_vars = expr_node_vars.subs(var, transformed_expr[idx])

        if substitute_symbol:
            for var in expr_node_vars.free_symbols:
                expr_node_vars = expr_node_vars.subs(
                    var,
                    sm.symbols(f"{substitute_symbol}{get_trailing_number(var.name)}"),
                )
        return expr_node_vars

    def generate_methods_d_hamiltonian_d(self):
        """
        Generate methods which return the derivative of the Hamiltonian with respect to
        offset charges, external fluxes and junction energies.
        """
        self._frozen = False
        self._data = {}
        hamiltonian, _ = self.generate_hamiltonian_sym_for_numerics(
            hamiltonian=self.hamiltonian_symbolic,
            shift_potential_to_origin=False,
            return_exprs=True,
        )
        hamiltonian = hamiltonian.subs("I", 1)
        ext_flux_1_over_f_methods = {}
        ng_1_over_f_methods = {}
        cc_1_over_f_methods = {}
        all_sym_parameters = (
            list(self.symbolic_params.keys())
            + self.external_fluxes
            + self.offset_charges
        )
        for param_sym in self.external_fluxes + self.offset_charges:
            diff_sym_expr = hamiltonian.diff(param_sym)

            def param_derivative(
                self=self,
                diff_sym_expr=diff_sym_expr,
                all_sym_parameters=all_sym_parameters,
            ):
                # substitute all symbolic params
                for param in all_sym_parameters:
                    diff_sym_expr = diff_sym_expr.subs(param, getattr(self, param.name))
                # evaluate the expression
                return self._evaluate_symbolic_expr(diff_sym_expr)

            if param_sym in self.external_fluxes:
                ext_flux_1_over_f_methods[
                    f"d_hamiltonian_d_flux{get_trailing_number(param_sym.name)}"
                ] = param_derivative
            elif param_sym in self.offset_charges:
                ng_1_over_f_methods[
                    f"d_hamiltonian_d_ng{get_trailing_number(param_sym.name)}"
                ] = param_derivative
        ## cc noise methods
        junction_branches = [branch for branch in self.branches if branch.type == "JJ"]
        for idx, branch in enumerate(junction_branches):

            def param_derivative(self=self, branch=branch):
                return self.junction_related_evaluation(branch, calc="dhdEJ")

            cc_1_over_f_methods[f"d_hamiltonian_d_EJ{branch.id_str}"] = param_derivative
        self._data.update(ext_flux_1_over_f_methods)
        self._data.update(ng_1_over_f_methods)
        self._data.update(cc_1_over_f_methods)
        self._frozen = True

    def junction_related_evaluation(self, branch_junction: Branch, calc="dhdEJ"):
        hamiltonian, _ = self.generate_hamiltonian_sym_for_numerics(
            hamiltonian=self.hamiltonian_symbolic,
            shift_potential_to_origin=False,
            return_exprs=True,
        )
        for sym in self.offset_charges + list(self.symbolic_params.keys()):
            hamiltonian = hamiltonian.subs(sym, getattr(self, sym.name))
        hamiltonian = hamiltonian.subs("I", 1)
        derivative_node_expr = sm.cos(
            sm.symbols(f"φ{branch_junction.nodes[0].index}")
            - sm.symbols(f"φ{branch_junction.nodes[1].index}")
        )
        derivarive_expr = self._transform_expr_to_new_variables(derivative_node_expr)
        expr_dict = hamiltonian.as_coefficients_dict()
        for term, coefficient in expr_dict.items():
            term_without_ext_flux = copy.copy(term)
            for flux in self.external_fluxes:
                term_without_ext_flux = term_without_ext_flux.subs(flux, 0)
            if term_without_ext_flux == derivarive_expr:
                break
        # substitute external flux
        for flux in self.external_fluxes:
            term = term.subs(flux, getattr(self, flux.name))
        if calc == "sin_phi_qp":
            term = term.subs(sm.cos, sm.sin)
            term = term.subs(term.args[0], term.args[0] / 2)
        return self._evaluate_symbolic_expr(term)

    def generate_tphi_1_over_f_methods(self):
        """Generate methods tphi_1_over_f_{noise_type}{index} methods for
        noise_type=['cc', 'ng', 'flux']; individual noise sources differentiated
        using index."""
        # calculating the rates from each of the flux sources
        junction_branches = [branch for branch in self.branches if branch.type == "JJ"]
        methods_noise_rates_from_flux = {}
        methods_noise_rates_from_ng = {}
        methods_noise_rates_from_cc = {}
        for param_sym in self.external_fluxes + self.offset_charges + junction_branches:
            if param_sym in self.external_fluxes:
                diff_func_name = "d_hamiltonian_d_flux"
                noise_type = "flux"
            elif param_sym in self.offset_charges:
                diff_func_name = "d_hamiltonian_d_ng"
                noise_type = "ng"
            if param_sym in junction_branches:
                diff_func_name = "d_hamiltonian_d_EJ"
                noise_type = "cc"
            if isinstance(param_sym, sm.Expr):
                trailing_number = get_trailing_number(param_sym.name)
                noise_op_func = getattr(self, f"{diff_func_name}{trailing_number}")
            elif param_sym in junction_branches:
                trailing_number = param_sym.id_str
                noise_op_func = getattr(self, f"{diff_func_name}{trailing_number}")

            def tphi_1_over_f_func(
                self=self,
                A_noise: float = NOISE_PARAMS[f"A_{noise_type}"],
                i: int = 0,
                j: int = 1,
                esys: Tuple[ndarray, ndarray] = None,
                get_rate: bool = False,
                noise_op_func=noise_op_func,
                **kwargs,
            ) -> float:
                r"""
                Calculate the 1/f dephasing time (or rate) due to critical current noise of junction associated with
                Josephson energy :math:`EJ3`.

                Parameters
                ----------
                A_noise:
                    noise strength
                i:
                    state index that along with j defines a qubit
                j:
                    state index that along with i defines a qubit
                esys:
                    evals, evecs tuple
                get_rate:
                    get rate or time

                Returns
                -------
                    decoherence time in units of :math:`2\pi ({\rm system\,\,units})`, or rate in inverse units.
                """
                noise_op = noise_op_func()
                if isinstance(noise_op, qt.Qobj):
                    noise_op = noise_op.data.tocsc()

                return self.tphi_1_over_f(
                    A_noise=A_noise,
                    i=i,
                    j=j,
                    noise_op=noise_op,
                    esys=esys,
                    get_rate=get_rate,
                    **kwargs,
                )

            if param_sym in self.external_fluxes:
                methods_noise_rates_from_flux[
                    f"tphi_1_over_f_flux{trailing_number}"
                ] = tphi_1_over_f_func
            elif param_sym in self.offset_charges:
                methods_noise_rates_from_ng[
                    f"tphi_1_over_f_ng{trailing_number}"
                ] = tphi_1_over_f_func
            elif param_sym in junction_branches:
                methods_noise_rates_from_cc[
                    f"tphi_1_over_f_cc{trailing_number}"
                ] = tphi_1_over_f_func

        self._data.update(methods_noise_rates_from_flux)
        self._data.update(methods_noise_rates_from_ng)
        self._data.update(methods_noise_rates_from_cc)

    def generate_overall_tphi_cc(self):
        def tphi_1_over_f_cc(
            self=self,
            A_noise: float = NOISE_PARAMS["A_cc"],
            i: int = 0,
            j: int = 1,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
            **kwargs,
        ) -> float:
            tphi_times = []
            for branch in [brnch for brnch in self.branches if brnch.type == "JJ"]:
                tphi_times.append(
                    getattr(self, f"tphi_1_over_f_cc{branch.id_str}")(
                        A_noise=A_noise, i=i, j=j, esys=esys, **kwargs
                    )
                )
            total_rate = sum([1 / tphi for tphi in tphi_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["tphi_1_over_f_cc"] = tphi_1_over_f_cc

    def generate_overall_tphi_flux(self):
        def tphi_1_over_f_flux(
            self=self,
            A_noise: float = NOISE_PARAMS["A_flux"],
            i: int = 0,
            j: int = 1,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
            **kwargs,
        ) -> float:
            tphi_times = []
            for flux_sym in self.external_fluxes:
                tphi_times.append(
                    getattr(
                        self, f"tphi_1_over_f_flux{get_trailing_number(flux_sym.name)}"
                    )(
                        A_noise=A_noise,
                        i=i,
                        j=j,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / tphi for tphi in tphi_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["tphi_1_over_f_flux"] = tphi_1_over_f_flux

    def generate_overall_tphi_ng(self):
        def tphi_1_over_f_ng(
            self=self,
            A_noise: float = NOISE_PARAMS["A_ng"],
            i: int = 0,
            j: int = 1,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
            **kwargs,
        ) -> float:
            tphi_times = []
            for flux_sym in self.offset_charges:
                tphi_times.append(
                    getattr(
                        self, f"tphi_1_over_f_ng{get_trailing_number(flux_sym.name)}"
                    )(
                        A_noise=A_noise,
                        i=i,
                        j=j,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / tphi for tphi in tphi_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["tphi_1_over_f_ng"] = tphi_1_over_f_ng

    def generate_t1_flux_bias_line_methods(self):
        """
        Generate methods for flux bian line t1 coherence times.
        """
        flux_bias_line_methods = {}
        for flux_sym in self.external_fluxes:
            trailing_number = get_trailing_number(flux_sym.name)
            noise_op_method = getattr(self, f"d_hamiltonian_d_flux{trailing_number}")

            def flux_bias_noise(
                self=self,
                i: int = 1,
                j: int = 0,
                M: float = NOISE_PARAMS["M"],
                Z: Union[complex, float, Callable] = NOISE_PARAMS["R_0"],
                T: float = NOISE_PARAMS["T"],
                total: bool = True,
                esys: Tuple[ndarray, ndarray] = None,
                get_rate: bool = False,
                noise_op_method=noise_op_method,
            ):

                return NoisySystem.t1_flux_bias_line(
                    self=self,
                    i=i,
                    j=j,
                    M=M,
                    Z=Z,
                    T=T,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    noise_op_method=noise_op_method,
                )

            flux_bias_line_methods[
                f"t1_flux_bias_line{trailing_number}"
            ] = flux_bias_noise
        self._data.update(flux_bias_line_methods)

    def generate_t1_methods(self):
        t1_capacitive_methods = {}
        t1_inductive_methods = {}
        t1_charge_impedance_methods = {}
        t1_quasiparticle_tunneling_methods = {}

        for branch in self.branches:
            if branch.type == "L":
                var_str = "θ"
            else:
                var_str = "Q"

            branch_var_expr_node = sm.symbols(f"φ{branch.nodes[0].index}") - sm.symbols(
                f"φ{branch.nodes[1].index}"
            )
            branch_var_expr = self._transform_expr_to_new_variables(
                branch_var_expr_node, substitute_symbol=var_str
            )

            if branch.type != "L":
                branch_param = (
                    branch.parameters["EC"]
                    if branch.type == "C"
                    else branch.parameters["ECJ"]
                )
            else:
                branch_param = branch.parameters["EL"]
            if isinstance(branch_param, sm.Expr):
                branch_param = getattr(self, branch_param.name)

            if branch.type == "L":
                t1_inductive_methods[
                    f"t1_inductive{branch.id_str}"
                ] = self.wrapper_t1_inductive_capacitive(
                    "inductive", branch_var_expr, branch_param
                )
            else:
                t1_capacitive_methods[
                    f"t1_capacitive{branch.id_str}"
                ] = self.wrapper_t1_inductive_capacitive(
                    "capacitive", branch_var_expr, branch_param
                )
                t1_charge_impedance_methods[
                    f"t1_charge_impedance{branch.id_str}"
                ] = self.wrapper_t1_charge_impedance(branch_var_expr)
            # quasiparticle noise
            if branch.type == "JJ":
                t1_quasiparticle_tunneling_methods[
                    f"t1_quasiparticle_tunneling{branch.id_str}"
                ] = self.wrapper_t1_quasipartice_tunneling(branch)

        self._data.update(t1_capacitive_methods)
        self._data.update(t1_inductive_methods)
        self._data.update(t1_charge_impedance_methods)
        self._data.update(t1_quasiparticle_tunneling_methods)

    def wrapper_t1_quasipartice_tunneling(self, branch: Branch):
        def t1_quasiparticle_tunneling(
            self=self,
            i: int = 1,
            j: int = 0,
            Y_qp: Union[float, Callable] = None,
            x_qp: float = NOISE_PARAMS["x_qp"],
            T: float = NOISE_PARAMS["T"],
            Delta: float = NOISE_PARAMS["Delta"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:

            return NoisySystem.t1_quasiparticle_tunneling(
                self=self,
                i=i,
                j=j,
                Y_qp=Y_qp,
                x_qp=x_qp,
                T=T,
                Delta=Delta,
                total=total,
                esys=esys,
                get_rate=get_rate,
                noise_op=self.junction_related_evaluation(branch, calc="sin_phi_qp"),
            )

        return t1_quasiparticle_tunneling

    def wrapper_t1_charge_impedance(self, branch_var_expr: sm.Expr):
        def t1_charge_impedance(
            self=self,
            i: int = 1,
            j: int = 0,
            Z: Union[float, Callable] = NOISE_PARAMS["R_0"],
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:

            return NoisySystem.t1_charge_impedance(
                self=self,
                i=i,
                j=j,
                Z=Z,
                T=T,
                total=total,
                esys=esys,
                get_rate=get_rate,
                noise_op=self._evaluate_symbolic_expr(branch_var_expr),
            )

        return t1_charge_impedance

    def wrapper_t1_inductive_capacitive(
        self, noise_type: str, branch_var_expr: sm.Expr, branch_param: Union[int, float]
    ):
        if noise_type == "capacitive":

            def t1_method(
                self=self,
                i: int = 1,
                j: int = 0,
                Q_cap: Union[float, Callable] = None,
                T: float = NOISE_PARAMS["T"],
                total: bool = True,
                esys: Tuple[ndarray, ndarray] = None,
                get_rate: bool = False,
            ) -> float:
                return NoisySystem.t1_capacitive(
                    self=self,
                    i=i,
                    j=j,
                    Q_cap=Q_cap,
                    T=T,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    noise_op=self._evaluate_symbolic_expr(branch_var_expr),
                    branch_params=branch_param,
                )

        elif noise_type == "inductive":

            def t1_method(
                self=self,
                i: int = 1,
                j: int = 0,
                Q_ind: Union[float, Callable] = None,
                T: float = NOISE_PARAMS["T"],
                total: bool = True,
                esys: Tuple[ndarray, ndarray] = None,
                get_rate: bool = False,
            ) -> float:
                return NoisySystem.t1_inductive(
                    self=self,
                    i=i,
                    j=j,
                    Q_ind=Q_ind,
                    T=T,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    noise_op=self._evaluate_symbolic_expr(branch_var_expr),
                    branch_params=branch_param,
                )

        return t1_method

    def generate_overall_t1_quasipartice_tunneling(self):
        if self.is_purely_harmonic:
            return None

        def t1_quasiparticle_tunneling(
            self=self,
            i: int = 1,
            j: int = 0,
            Y_qp: Union[float, Callable] = None,
            x_qp: float = NOISE_PARAMS["x_qp"],
            T: float = NOISE_PARAMS["T"],
            Delta: float = NOISE_PARAMS["Delta"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            for branch in [b for b in self.branches if b.type == "JJ"]:
                t1_times.append(
                    getattr(self, f"t1_quasiparticle_tunneling{branch.id_str}")(
                        i=i,
                        j=j,
                        Y_qp=Y_qp,
                        x_qp=x_qp,
                        T=T,
                        Delta=Delta,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["t1_quasiparticle_tunneling"] = t1_quasiparticle_tunneling

    def generate_overall_t1_inductive(self):
        def t1_method(
            self=self,
            i: int = 1,
            j: int = 0,
            Q_ind: Union[float, Callable] = None,
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            for branch in [b for b in self.branches if b.type == "L"]:
                t1_times.append(
                    getattr(self, f"t1_inductive{branch.id_str}")(
                        i=i,
                        j=j,
                        Q_ind=Q_ind,
                        T=T,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["t1_inductive"] = t1_method

    def generate_overall_t1_capacitive(self):
        def t1_method(
            self=self,
            i: int = 1,
            j: int = 0,
            Q_cap: Union[float, Callable] = None,
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            for branch in [b for b in self.branches if b.type != "L"]:
                t1_times.append(
                    getattr(self, f"t1_capacitive{branch.id_str}")(
                        i=i,
                        j=j,
                        Q_cap=Q_cap,
                        T=T,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["t1_capacitive"] = t1_method

    def generate_overall_t1_charge_impedance(self):
        def t1_method(
            self=self,
            i: int = 1,
            j: int = 0,
            Z: Union[float, Callable] = NOISE_PARAMS["R_0"],
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            for branch in [b for b in self.branches if b.type != "L"]:
                t1_times.append(
                    getattr(self, f"t1_charge_impedance{branch.id_str}")(
                        i=i,
                        j=j,
                        Z=Z,
                        T=T,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["t1_charge_impedance"] = t1_method

    def generate_overall_t1_flux_bias_line(self):
        def t1_flux_bias_line(
            self=self,
            i: int = 1,
            j: int = 0,
            M: float = NOISE_PARAMS["M"],
            Z: Union[complex, float, Callable] = NOISE_PARAMS["R_0"],
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            for external_flux_sym in self.external_fluxes:
                t1_times.append(
                    getattr(
                        self,
                        f"t1_flux_bias_line{get_trailing_number(external_flux_sym.name)}",
                    )(
                        i=i,
                        j=j,
                        M=M,
                        Z=Z,
                        T=T,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        self._data["t1_flux_bias_line"] = t1_flux_bias_line

    def generate_all_noise_methods(self):

        self.generate_methods_d_hamiltonian_d()
        self.generate_tphi_1_over_f_methods()
        self.generate_t1_flux_bias_line_methods()
        self.generate_t1_methods()

        self.generate_overall_tphi_cc()
        self.generate_overall_tphi_flux()
        self.generate_overall_tphi_ng()
        self.generate_overall_t1_capacitive()
        self.generate_overall_t1_charge_impedance()
        self.generate_overall_t1_inductive()
        self.generate_overall_t1_flux_bias_line()
        self.generate_overall_t1_quasipartice_tunneling()
