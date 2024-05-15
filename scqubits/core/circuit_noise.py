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
import scipy as sp
import re
import copy

from scqubits.core.noise import NOISE_PARAMS, NoisySystem, calc_therm_ratio
from scqubits.core.circuit_utils import get_trailing_number, keep_terms_for_subsystem
from scqubits.utils.misc import is_string_float, Qobj_to_scipy_csc_matrix
import scqubits.core.units as units

from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy as sm

from scqubits.core.symbolic_circuit import Branch


class NoisyCircuit(NoisySystem, ABC):
    @staticmethod
    def Q_from_branch(branch):
        key = "Q_" + ("ind" if branch.type == "L" else "cap")
        if key in branch.aux_params.keys():
            Q_str = branch.aux_params[key]
            if not is_string_float(Q_str):

                def Q_func(omega, T):
                    return eval(Q_str)

                return Q_func
            else:
                return float(Q_str)
        return None

    def generate_methods_d_hamiltonian_d(self):
        """
        Generate methods which return the derivative of the Hamiltonian with respect to
        offset charges, external fluxes and junction energies.
        """

        ext_flux_1_over_f_methods = {}
        ng_1_over_f_methods = {}
        cc_1_over_f_methods = {}

        for param_sym in self.external_fluxes + self.offset_charges:

            def param_derivative(self, param_sym=param_sym):
                parent_instance = self.return_parent_circuit()
                hamiltonian = parent_instance.fetch_symbolic_hamiltonian()
                hamiltonian = parent_instance._hamiltonian_sym_for_numerics
                hamiltonian = hamiltonian.subs("I", 1)
                all_sym_parameters = (
                    list(parent_instance.symbolic_params.keys())
                    + parent_instance.external_fluxes
                    + parent_instance.offset_charges
                )
                diff_sym_expr = hamiltonian.diff(param_sym)
                # substitute all symbolic params
                for param in all_sym_parameters:
                    diff_sym_expr = diff_sym_expr.subs(
                        param, getattr(parent_instance, param.name)
                    )
                # evaluate the expression
                return parent_instance._evaluate_symbolic_expr(diff_sym_expr)

            if param_sym in self.external_fluxes:
                ext_flux_1_over_f_methods[
                    f"d_hamiltonian_d_flux{get_trailing_number(param_sym.name)}"
                ] = param_derivative
            elif param_sym in self.offset_charges:
                ng_1_over_f_methods[
                    f"d_hamiltonian_d_ng{get_trailing_number(param_sym.name)}"
                ] = param_derivative
        ## cc noise methods
        junction_branches = [branch for branch in self.branches if "JJ" in branch.type]
        for idx, branch in enumerate(junction_branches):

            def param_derivative(self, branch=branch):
                return -self.junction_related_evaluation(branch, calc="dhdEJ")

            cc_1_over_f_methods[f"d_hamiltonian_d_EJ{branch.id_str}"] = param_derivative
        noise_helper_methods = {
            **ext_flux_1_over_f_methods,
            **ng_1_over_f_methods,
            **cc_1_over_f_methods,
        }
        self.noise_helper_methods = noise_helper_methods
        for method_name in noise_helper_methods:
            setattr(
                self, method_name, MethodType(noise_helper_methods[method_name], self)
            )

    def _transform_expr_to_new_variables(
        self, expr_node_vars: sm.Expr, substitute_symbol: Optional[str] = None
    ):
        transformation_mat = self.transformation_matrix
        expr_node_vars = expr_node_vars.expand()
        num_vars = len(self.symbolic_circuit.nodes) - (1 if self.is_grounded else 0)
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

    def junction_related_evaluation(self, branch_junction: Branch, calc="dhdEJ"):
        parent_instance = self.return_parent_circuit()
        hamiltonian = parent_instance.fetch_symbolic_hamiltonian()
        hamiltonian = parent_instance._hamiltonian_sym_for_numerics

        for sym in parent_instance.offset_charges + list(
            parent_instance.symbolic_params.keys()
        ):
            hamiltonian = hamiltonian.subs(sym, getattr(parent_instance, sym.name))
        hamiltonian = hamiltonian.subs("I", 1)
        branch_cos_node_expr = sm.cos(
            sm.symbols(f"φ{branch_junction.nodes[0].index}")
            - sm.symbols(f"φ{branch_junction.nodes[1].index}")
        )
        branch_cos_node_expr = branch_cos_node_expr.subs(
            "φ0", 0
        )  # setting ground node to zero.
        branch_cos_expr = parent_instance._transform_expr_to_new_variables(
            branch_cos_node_expr
        )
        expr_dict = hamiltonian.as_coefficients_dict()
        for term, coefficient in expr_dict.items():
            term_without_ext_flux = copy.copy(term)
            for flux in parent_instance.external_fluxes:
                term_without_ext_flux = term_without_ext_flux.subs(flux, 0)
            if term_without_ext_flux == branch_cos_expr:
                break
        # substitute external flux
        for flux in parent_instance.external_fluxes:
            term = term.subs(flux, getattr(parent_instance, flux.name))
        if calc == "sin_phi_qp":
            term = term.subs(sm.cos, sm.sin)
            term = term.subs(term.args[0], term.args[0] / 2)

        # evaluate the expression
        return parent_instance._evaluate_symbolic_expr(term)

    def generate_tphi_1_over_f_methods(self):
        """Generate methods tphi_1_over_f_{noise_type}{index} methods for
        noise_type=['cc', 'ng', 'flux']; individual noise sources differentiated
        using index."""
        # calculating the rates from each of the flux sources
        junction_branches = [branch for branch in self.branches if "JJ" in branch.type]
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
                    noise_op = Qobj_to_scipy_csc_matrix(noise_op)

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
                methods_noise_rates_from_ng[f"tphi_1_over_f_ng{trailing_number}"] = (
                    tphi_1_over_f_func
                )
            elif param_sym in junction_branches:
                methods_noise_rates_from_cc[f"tphi_1_over_f_cc{trailing_number}"] = (
                    tphi_1_over_f_func
                )

        noise_methods = {
            **methods_noise_rates_from_flux,
            **methods_noise_rates_from_ng,
            **methods_noise_rates_from_cc,
        }
        for method_name in noise_methods:
            setattr(self, method_name, MethodType(noise_methods[method_name], self))

    def generate_overall_tphi_cc(self):
        if not any([re.match(r"tphi_1_over_f_cc\d+$", method) for method in dir(self)]):
            return None

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
            for branch in [brnch for brnch in self.branches if "JJ" in brnch.type]:
                tphi_times.append(
                    getattr(self, f"tphi_1_over_f_cc{branch.id_str}")(
                        A_noise=A_noise, i=i, j=j, esys=esys, **kwargs
                    )
                )
            total_rate = sum([1 / tphi for tphi in tphi_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        setattr(self, "tphi_1_over_f_cc", MethodType(tphi_1_over_f_cc, self))

    def generate_overall_tphi_flux(self):
        if not any(
            [re.match(r"tphi_1_over_f_flux\d+$", method) for method in dir(self)]
        ):
            return None

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

        setattr(self, "tphi_1_over_f_flux", MethodType(tphi_1_over_f_flux, self))

    def generate_overall_tphi_ng(self):
        if not any([re.match(r"tphi_1_over_f_ng\d+$", method) for method in dir(self)]):
            return None

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

        setattr(self, "tphi_1_over_f_ng", MethodType(tphi_1_over_f_ng, self))

    def generate_t1_flux_bias_line_methods(self):
        """
        Generate methods for flux bias line t1 coherence times.
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

            flux_bias_line_methods[f"t1_flux_bias_line{trailing_number}"] = (
                flux_bias_noise
            )

        for method_name in flux_bias_line_methods:
            setattr(
                self, method_name, MethodType(flux_bias_line_methods[method_name], self)
            )

    def generate_t1_methods(self):
        t1_capacitive_methods = {}
        t1_inductive_methods = {}
        t1_charge_impedance_methods = {}
        t1_quasiparticle_tunneling_methods = {}

        for branch in self.branches:
            if branch.type == "L":
                t1_inductive_methods[f"t1_inductive{branch.id_str}"] = (
                    self.wrapper_t1_inductive_capacitive(branch)
                )
            else:
                t1_capacitive_methods[f"t1_capacitive{branch.id_str}"] = (
                    self.wrapper_t1_inductive_capacitive(branch)
                )
            # # quasiparticle noise
            # if "JJ" in branch.type:
            #     t1_quasiparticle_tunneling_methods[
            #         f"t1_quasiparticle_tunneling{branch.id_str}"
            #     ] = self.wrapper_t1_quasiparticle_tunneling(branch)
            # quasiparticle noise methods are not included yet
        noise_methods = {
            **t1_capacitive_methods,
            **t1_inductive_methods,
            **t1_charge_impedance_methods,
        }
        for method_name in noise_methods:
            setattr(self, method_name, MethodType(noise_methods[method_name], self))
        # self._data.update(t1_quasiparticle_tunneling_methods)

    def wrapper_t1_quasiparticle_tunneling(self, branch: Branch):
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

    def wrapper_t1_charge_impedance(self, branch: Branch):
        def t1_charge_impedance(
            self,
            i: int = 1,
            j: int = 0,
            Z: Union[float, Callable] = NOISE_PARAMS["R_0"],
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
            branch=branch,
        ) -> float:
            parent_circuit = self.return_parent_circuit()
            branch_var_expr = parent_circuit.symbolic_circuit._branch_sym_expr(
                branch, return_charge=False if branch.type == "L" else True
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
                branch_param = getattr(parent_circuit, branch_param.name)

            return NoisySystem.t1_charge_impedance(
                self=self,
                i=i,
                j=j,
                Z=Z,
                T=T,
                total=total,
                esys=esys,
                get_rate=get_rate,
                noise_op=parent_circuit._evaluate_symbolic_expr(branch_var_expr),
            )

        return t1_charge_impedance

    def wrapper_t1_inductive_capacitive(
        self,
        branch: Branch,
    ):
        if branch.type != "L":

            def t1_method(
                self,
                i: int = 1,
                j: int = 0,
                Q_cap: Union[float, Callable] = None,
                T: float = NOISE_PARAMS["T"],
                total: bool = True,
                esys: Tuple[ndarray, ndarray] = None,
                get_rate: bool = False,
                branch: Branch = branch,
            ) -> float:
                parent_circuit = self.return_parent_circuit()
                branch_charge_expr = parent_circuit.symbolic_circuit._branch_sym_expr(
                    branch, return_charge=True
                )

                branch_param = (
                    branch.parameters["EC"]
                    if branch.type == "C"
                    else branch.parameters["ECJ"]
                )
                if isinstance(branch_param, sm.Expr):
                    branch_param = getattr(parent_circuit, branch_param.name)

                return NoisySystem.t1_capacitive(
                    self=self,
                    i=i,
                    j=j,
                    Q_cap=Q_cap or self.Q_from_branch(branch),
                    T=T,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    noise_op=parent_circuit._evaluate_symbolic_expr(branch_charge_expr),
                    branch_params=branch_param,
                )

        else:

            def t1_method(
                self,
                i: int = 1,
                j: int = 0,
                Q_ind: Union[float, Callable] = None,
                T: float = NOISE_PARAMS["T"],
                total: bool = True,
                esys: Tuple[ndarray, ndarray] = None,
                get_rate: bool = False,
                branch: Branch = branch,
            ) -> float:
                parent_circuit = self.return_parent_circuit()
                branch_var_expr = parent_circuit.symbolic_circuit._branch_sym_expr(
                    branch
                )

                branch_param = branch.parameters["EL"]

                if isinstance(branch_param, sm.Expr):
                    branch_param = getattr(parent_circuit, branch_param.name)

                return NoisySystem.t1_inductive(
                    self=self,
                    i=i,
                    j=j,
                    Q_ind=Q_ind or self.Q_from_branch(branch),
                    T=T,
                    total=total,
                    esys=esys,
                    get_rate=get_rate,
                    noise_op=parent_circuit._evaluate_symbolic_expr(branch_var_expr),
                    branch_params=branch_param,
                )

        return t1_method

    def generate_overall_t1_quasiparticle_tunneling(self):
        if not any(
            [
                re.match(r"t1_quasiparticle_tunneling\d+$", method)
                for method in dir(self)
            ]
        ):
            return None
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
            for branch in [b for b in self.branches if "JJ" in b.type]:
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

        setattr(
            self,
            "t1_quasiparticle_tunneling",
            MethodType(t1_quasiparticle_tunneling, self),
        )

    def generate_overall_t1_inductive(self):
        if not any([re.match(r"t1_inductive\d+$", method) for method in dir(self)]):
            return None

        def t1_method(
            self,
            i: int = 1,
            j: int = 0,
            Q_ind: Union[float, Callable] = None,
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            parent_circuit = self.return_parent_circuit()
            for branch in [b for b in parent_circuit.branches if b.type == "L"]:
                t1_times.append(
                    getattr(parent_circuit, f"t1_inductive{branch.id_str}")(
                        i=i,
                        j=j,
                        Q_ind=Q_ind or self.Q_from_branch(branch),
                        T=T,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        setattr(self, "t1_inductive", MethodType(t1_method, self))

    def generate_overall_t1_capacitive(self):
        if not any([re.match(r"t1_capacitive\d+$", method) for method in dir(self)]):
            return None

        def t1_method(
            self,
            i: int = 1,
            j: int = 0,
            Q_cap: Union[float, Callable] = None,
            T: float = NOISE_PARAMS["T"],
            total: bool = True,
            esys: Tuple[ndarray, ndarray] = None,
            get_rate: bool = False,
        ) -> float:
            t1_times = []
            parent_circuit = self.return_parent_circuit()
            for branch in [b for b in parent_circuit.branches if b.type != "L"]:
                t1_times.append(
                    getattr(parent_circuit, f"t1_capacitive{branch.id_str}")(
                        i=i,
                        j=j,
                        Q_cap=Q_cap or self.Q_from_branch(branch),
                        T=T,
                        total=total,
                        esys=esys,
                    )
                )
            total_rate = sum([1 / t1 for t1 in t1_times])
            if get_rate:
                return total_rate
            return 1 / total_rate if total_rate != 0 else np.inf

        setattr(self, "t1_capacitive", MethodType(t1_method, self))

    def generate_overall_t1_charge_impedance(self):
        if not any(
            [re.match(r"t1_charge_impedance\d+$", method) for method in dir(self)]
        ):
            return None

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
            parent_circuit = self.return_parent_circuit()
            for branch in [b for b in parent_circuit.branches if b.type != "L"]:
                t1_times.append(
                    getattr(parent_circuit, f"t1_charge_impedance{branch.id_str}")(
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

        setattr(self, "t1_charge_impedance", MethodType(t1_method, self))

    def generate_overall_t1_flux_bias_line(self):
        if not any(
            [re.match(r"t1_flux_bias_line\d+$", method) for method in dir(self)]
        ):
            return None

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

        setattr(self, "t1_flux_bias_line", MethodType(t1_flux_bias_line, self))

    def generate_noise_methods(self):
        self._frozen = False
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
        self.generate_overall_t1_quasiparticle_tunneling()
        self._noise_methods_generated = True
        self._frozen = True
