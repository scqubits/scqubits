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

from abc import ABC, abstractmethod
import operator as builtin_op
import functools
from numpy import ndarray
import numpy as np
import qutip as qt
import copy

from scqubits.core.noise import NOISE_PARAMS, NoisySystem
from scqubits.core.circuit_utils import get_trailing_number

# from scqubits.core.circuit import Circuit, Subsystem

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
                    functools.reduce(builtin_op.mul, product_matrix_list)
                )
        return sum(eval_matrix_list)

    def _transform_expr_to_new_variables(self, expr_node_vars: sm.Expr):
        transformation_mat = self.transformation_matrix
        expr_node_vars = expr_node_vars.expand()
        num_vars = len(self.symbolic_circuit._node_list_without_ground)
        new_vars = [sm.symbols(f"θ{index}") for index in range(1, 1 + num_vars)]
        old_vars = [sm.symbols(f"φ{index}") for index in range(1, 1 + num_vars)]
        transformed_expr = transformation_mat.dot(new_vars)
        for idx, var in enumerate(old_vars):
            expr_node_vars = expr_node_vars.subs(var, transformed_expr[idx])

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

            def param_derivative(self=self, diff_sym_expr=diff_sym_expr):
                # substitute all symbolic params
                for param in all_sym_parameters:
                    diff_sym_expr = diff_sym_expr.subs(param, getattr(self, param.name))
                diff_sym_expr = diff_sym_expr.subs("I", 1)
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
                return self.d_hamiltonian_d_EJ(branch)

            cc_1_over_f_methods[f"d_hamiltonian_d_EJ{idx + 1}"] = param_derivative
        self._data.update(ext_flux_1_over_f_methods)
        self._data.update(ng_1_over_f_methods)
        self._data.update(cc_1_over_f_methods)
        self._frozen = True

    def d_hamiltonian_d_EJ(self, branch_junction: Branch):
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
        return self._evaluate_symbolic_expr(term)

    def generate_tphi_1_over_f_methods(self):
        """Generate methods tphi_1_over_f_{noise_type}{index} methods for
        noise_type=['cc', 'ng', 'flux'] for individual noise sources differentiated
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
            else:
                trailing_number = junction_branches.index(param_sym) + 1
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
                if "tphi_1_over_f_flux" not in self.supported_noise_channels():
                    raise RuntimeError(
                        "Critical current noise channel 'tphi_1_over_f_cc3' is not supported in"
                        " this system."
                    )
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

    def generate_overall_t_methods(self):
        """Generate methods tphi_1_over_f_{noise_type} and t_1_{noise_type} methods for
        noise_type=['cc', 'ng', 'flux'] and ["capacitive", "charge_impedance",
        "flux_bias_line", "inductive", "quasiparticle_tunneling"]"""
        noise_types_tphi = ["cc", "ng", "flux"]
        noise_types_t1 = ["flux_bias_line"]
        # generating the total flux and charge noise methods
        for noise_type in noise_types_tphi + noise_types_t1:
            if noise_type in noise_types_tphi:
                func_name = f"tphi_1_over_f_{noise_type}"
            elif noise_type in noise_types_t1:
                func_name = f"t1_{noise_type}"

            def total_coherence_time_from_noise_type(
                self=self, noise_type=noise_type, func_name=func_name
            ):
                tphi_1_over_f_times = []
                if noise_type == "flux" or "flux_bias_line":
                    num_noise_sources = len(self.external_fluxes)
                elif noise_type == "ng":
                    num_noise_sources = len(self.offset_charges)
                elif noise_type == "cc":
                    num_noise_sources = len(
                        [branch for branch in self.branches if branch.type == "JJ"]
                    )
                for trailing_number in range(1, num_noise_sources + 1):

                    tphi_1_over_f_times.append(
                        getattr(self, f"{func_name}{trailing_number}")()
                    )
                total_tphi = sum([1 / tphi for tphi in tphi_1_over_f_times])
                return 1 / total_tphi

            self._data[func_name] = total_coherence_time_from_noise_type

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
        noise_types = ["capacitive", "inductive"]
        t1_capacitive_methods = {}
        t1_inductive_methods = {}
        all_variable_indices = (
            self.var_categories["periodic"] + self.var_categories["extended"]
        )
        number_of_ungrounded_nodes = len(
            self.symbolic_circuit._node_list_without_ground
        )

        for noise_type in noise_types:
            if noise_type in ["capacitive"]:
                relevant_brnchs = [
                    branch
                    for branch in self.branches
                    if branch.type
                    in [
                        "C",
                        "JJ",
                    ]
                ]
                var_str = "Q"
            else:
                relevant_brnchs = [
                    branch for branch in self.branches if branch.type == "L"
                ]
                var_str = "θ"
            node_var_transformation = self.transformation_matrix @ np.array(
                [
                    sm.symbols(f"{var_str}{idx}")
                    for idx in range(1, 1 + number_of_ungrounded_nodes)
                ]
            )
            node_var_transformation = [0] + list(node_var_transformation)
            for idx, branch in enumerate(relevant_brnchs):
                branch_flux = (
                    node_var_transformation[branch.node_ids()[0]]
                    - node_var_transformation[branch.node_ids()[1]]
                )


    def wrapper_t1_inductive(self, noise_op_method, ):
                def t1_cap_ind(
                        self,
                        i: int = 1,
                        j: int = 0,
                        Q_cap: Union[float, Callable] = None,
                        T: float = NOISE_PARAMS["T"],
                        total: bool = True,
                        esys: Tuple[ndarray, ndarray] = None,
                        get_rate: bool = False,
                        noise_op_method: Optional[Callable] = None,
                    ) -> float:


    # def d_hamiltonian_d_param_function_factory(self, params: List):
    #     hamiltonian = self.circuit.hamiltonian_symbolic
