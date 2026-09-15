# subsystem_tree.py
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
"""Build / refresh the hierarchical-diagonalization subsystem tree.

Holds the methods that decompose a parent ``Circuit``'s symbolic
Hamiltonian into per-subsystem and interaction terms, instantiate the
child :class:`~scqubits.core.circuit.Subsystem` objects, validate
truncation indices, and rebuild the resulting
:class:`~scqubits.HilbertSpace` interactions.

These responsibilities form a coherent block within
:class:`~scqubits.core.circuit_internals.routines.CircuitRoutines`:
they all run during the "split this Hamiltonian into subsystems and
wire them together" phase of ``_configure`` /
``_configure_sym_hamiltonian``.  Keeping them in a dedicated mixin
lets the rest of ``routines.py`` focus on operator construction and
parameter sync.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import sympy as sm

import scqubits.core.circuit as circuit

from scqubits import HilbertSpace
from scqubits.core.circuit_internals._protocols import CircuitProtocol
from scqubits.utils.misc import flatten_list_recursive

__all__ = [
    "SubsystemTreeMixin",
]


class SubsystemTreeMixin(ABC, CircuitProtocol):
    """Mixin: subsystem-tree construction for hierarchical diagonalization.

    The methods here mutate the parent's ``self.subsystems``,
    ``self.subsystem_interactions``, ``self.subsystem_hamiltonians``,
    and ``self.hilbert_space``.  They assume the surrounding mixin
    chain (``CircuitSymMethods``, ``CircuitRoutines``) has already
    populated the symbolic Hamiltonian and the helper callables
    declared on
    :class:`~scqubits.core.circuit_internals._protocols.CircuitProtocol`.
    """

    def _check_truncation_indices(self):
        """Validate that subsystem truncation indices fit within their Hilbert dim."""
        if not self.hierarchical_diagonalization:
            return

        for subsystem_idx, subsystem in enumerate(self.subsystems):
            if subsystem.truncated_dim >= subsystem.hilbertdim() - 1:
                # find the correct position of the subsystem where the truncation
                # index  is too big
                subsystem_position = f"subsystem {subsystem_idx} "
                parent = subsystem.parent
                while parent.is_child:
                    grandparent = parent.parent
                    # find the subsystem position of the parent system
                    subsystem_position += f"of subsystem {grandparent.get_subsystem_index(parent.dynamic_var_indices[0])} "
                    parent = grandparent
                raise ValueError(
                    f"The truncation index for {subsystem_position} exceeds the maximum"
                    f" size of {subsystem.hilbertdim() - 1}."
                )
            elif not (
                isinstance(subsystem.truncated_dim, int)
                and (subsystem.truncated_dim > 0)
            ):
                raise ValueError(
                    "Invalid value encountered in subsystem_trunc_dims. "
                    "Truncated dimension must be a positive integer."
                )

    def _generate_subsystems(
        self,
        only_update_subsystems: bool = False,
        subsys_dict: dict[str, Any] | None = None,
    ):
        """Generate (or refresh) child subsystems following :attr:`system_hierarchy`.

        Parameters
        ----------
        only_update_subsystems:
            if ``True``, update existing subsystems in place instead of
            constructing new ones.
        subsys_dict:
            optional pre-computed ``{"systems_sym": ..., "interaction_sym": ...}``
            decomposition; if ``None``, it is recomputed from the current
            symbolic Hamiltonian.
        """
        hamiltonian = self.hamiltonian_symbolic
        systems_sym, interaction_sym = self._get_systems_and_interactions(
            hamiltonian, subsys_dict
        )

        if only_update_subsystems:
            self._update_existing_subsystems(systems_sym, interaction_sym)
        else:
            self._create_new_subsystems(systems_sym, interaction_sym)

    def _get_systems_and_interactions(
        self, hamiltonian: sm.Expr, subsys_dict: dict[str, Any] | None
    ) -> tuple[list[sm.Expr], list[sm.Expr]]:
        """Split ``hamiltonian`` into per-subsystem and interaction symbolic terms.

        Parameters
        ----------
        hamiltonian:
            full symbolic Hamiltonian to decompose.
        subsys_dict:
            optional pre-computed decomposition; if provided, returned as-is.
        """
        non_operator_symbols = (
            self.offset_charges
            + self.free_charges
            + self.external_fluxes
            + list(self.symbolic_params.keys())
            + [sm.symbols("I")]
        )
        if subsys_dict:
            return subsys_dict["systems_sym"], subsys_dict["interaction_sym"]
        return self._sym_subsystem_hamiltonian_and_interactions(
            hamiltonian, self.system_hierarchy, non_operator_symbols
        )

    def _update_existing_subsystems(
        self, systems_sym: list[sm.Expr], interaction_sym: list[sm.Expr]
    ):
        """Refresh the symbolic Hamiltonians of existing child subsystems.

        Parameters
        ----------
        systems_sym:
            list of per-subsystem symbolic Hamiltonians, one per entry of
            :attr:`system_hierarchy`.
        interaction_sym:
            list of symbolic interaction terms, one per entry of
            :attr:`system_hierarchy`.
        """
        for subsys_index, subsys in enumerate(self.subsystems):
            subsys.hamiltonian_symbolic = systems_sym[subsys_index]
            subsys._frozen = False
            subsys._find_and_set_sym_attrs()
            self.subsystem_interactions[subsys_index] = interaction_sym[subsys_index]

    def _create_new_subsystems(
        self, systems_sym: list[sm.Expr], interaction_sym: list[sm.Expr]
    ):
        """Construct fresh :class:`Subsystem` instances and a :class:`HilbertSpace`.

        Parameters
        ----------
        systems_sym:
            list of per-subsystem symbolic Hamiltonians, one per entry of
            :attr:`system_hierarchy`.
        interaction_sym:
            list of symbolic interaction terms, one per entry of
            :attr:`system_hierarchy`.
        """
        self.subsystem_hamiltonians = dict(
            zip(range(len(self.system_hierarchy)), systems_sym)
        )
        self.subsystem_interactions = dict(
            zip(range(len(self.system_hierarchy)), interaction_sym)
        )
        self.subsystems = []
        for index in range(len(self.system_hierarchy)):
            is_purely_harmonic = self._is_expression_purely_harmonic(systems_sym[index])
            if isinstance(self.ext_basis, list):
                ext_basis = self.ext_basis[index]
            else:
                ext_basis = "harmonic" if is_purely_harmonic else self.ext_basis
            self.subsystems.append(
                circuit.Subsystem(
                    self,  # type: ignore[arg-type]
                    systems_sym[index],
                    system_hierarchy=self.system_hierarchy[index],
                    truncated_dim=(
                        self.subsystem_trunc_dims[index][0]
                        if isinstance(self.subsystem_trunc_dims[index], list)
                        else self.subsystem_trunc_dims[index]
                    ),
                    ext_basis=ext_basis,
                    subsystem_trunc_dims=(
                        self.subsystem_trunc_dims[index][1]
                        if isinstance(self.subsystem_trunc_dims[index], list)
                        else None
                    ),
                    evals_method=(
                        self.evals_method if not is_purely_harmonic else None
                    ),
                    evals_method_options=(
                        self.evals_method_options if not is_purely_harmonic else None
                    ),
                    esys_method=(self.esys_method if not is_purely_harmonic else None),
                    esys_method_options=(
                        self.esys_method_options if not is_purely_harmonic else None
                    ),
                )
            )
        self.hilbert_space = HilbertSpace(self.subsystems)

    def get_subsystem_index(self, var_index: int) -> int:
        """Return the subsystem index that owns the given variable index.

        Parameters
        ----------
        var_index:
            variable index in integer starting from 1.

        Returns
        -------
        subsystem index which can be used to identify the subsystem index in the
        list self.subsystems.
        """
        for index, system_hierarchy in enumerate(self.system_hierarchy):
            if var_index in flatten_list_recursive(system_hierarchy):
                return index
        raise ValueError(
            f"The var_index={var_index} could not be identified with any subsystem."
        )

    def _update_interactions(self, recursive: bool = False) -> None:
        """Rebuild HilbertSpace interactions when hierarchical diagonalization is on.

        Parameters
        ----------
        recursive:
            if ``True``, also recurse into subsystems that themselves use
            hierarchical diagonalization.
        """
        self.hilbert_space.interaction_list = []

        # Adding interactions using the symbolic interaction term
        for sys_index in range(len(self.system_hierarchy)):
            interaction = self.subsystem_interactions[sys_index].expand()
            if interaction == 0:  # if the interaction term is zero
                continue

            interaction = interaction.subs("I", 1)
            # adding a factor of 2pi for external flux
            for sym_param in interaction.free_symbols:
                if sym_param in self.external_fluxes:
                    interaction = interaction.subs(sym_param, 2 * np.pi * sym_param)

            expr_dict = interaction.as_coefficients_dict()
            interaction_terms = list(expr_dict.keys())

            for idx, term in enumerate(interaction_terms):
                coefficient_sympy = expr_dict[term]

                branch_sym_params = [
                    symbol
                    for symbol in term.free_symbols
                    if symbol in list(self.symbolic_params.keys())
                ]
                operator_expr, param_expr = term.as_independent(
                    *branch_sym_params, as_Mul=True
                )

                param_expr = coefficient_sympy * param_expr
                for param in list(self.symbolic_params.keys()):
                    param_expr = param_expr.subs(
                        param, sm.symbols("self." + param.name)
                    )
                param_expr_str = str(param_expr)
                self.hilbert_space.add_interaction(
                    expr=param_expr_str + "*operator_expr",
                    const={"self": self},
                    op1=(
                        "operator_expr",
                        self._operator_from_sym_expr_wrapper(operator_expr),
                    ),
                    check_validity=False,
                )
        if recursive:
            for subsys in self.subsystems:
                if subsys.hierarchical_diagonalization:
                    subsys._update_interactions(recursive=recursive)
