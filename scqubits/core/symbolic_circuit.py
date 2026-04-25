# symbolic_circuit.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from __future__ import annotations

import copy
import itertools
import warnings

from itertools import chain
from typing import Any

import numpy as np
import scipy as sp
import sympy
import sympy as sm

from numpy import ndarray
from sympy import Symbol, symbols

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings

from scqubits.core.circuit_internals.branch_metadata import (
    _capacitance_variable_for_branch,
    _junction_order,
)
from scqubits.core.circuit_internals.input import (
    parse_code_line,
    process_param,
    remove_branchline,
    remove_comments,
    strip_empty_lines,
)
from scqubits.core.circuit_internals.utils import get_trailing_number
from scqubits.core.symbolic_circuit_graph import (
    Branch,
    Coupler,
    Node,
    SymbolicCircuitGraph,
)
from scqubits.core.circuit_internals.sympy_helpers import round_symbolic_expr
from scqubits.utils.misc import (
    flatten_list_recursive,
    unique_elements_in_list,
)


class SymbolicCircuit(serializers.Serializable, SymbolicCircuitGraph):
    r"""Describes a circuit consisting of nodes and branches.

    Examples
    --------
    For a transmon qubit, the input file reads::

        # file_name: transmon_num.inp
        nodes: 2
        branches:
        C       1,2     1
        JJ      1,2     1       10

    The :class:`Circuit` object can be initiated using::
        `Circuit.from_input_file("transmon_num.inp")`

    Parameters
    ----------
    nodes_list:
        list of :class:`Node` instances in the circuit
    branches_list:
        list of :class:`Branch` instances connecting the nodes
    couplers_list:
        list of :class:`Coupler` instances connecting the branches
    branch_var_dict:
        mapping from symbolic branch-parameter symbols to their numeric
        default values
    basis_completion:
        selects the type of basis used to complete the transformation matrix;
        choices are ``"heuristic"`` (default) or ``"canonical"``
    use_dynamic_flux_grouping:
        when ``True``, the flux allocation is performed assuming time-dependent
        flux, which disables the option to change the closure branches
        (default: ``False``)
    initiate_sym_calc:
        when ``True`` (default), the object attributes are initialized by
        calling :meth:`configure`
    input_string:
        the raw input string from which the circuit was parsed (empty by
        default); used for bookkeeping and serialization
    """

    def __init__(
        self,
        nodes_list: list[Node],
        branches_list: list[Branch],
        couplers_list: list[Coupler],
        branch_var_dict: dict[Any | Symbol, Any | float],
        basis_completion: str = "heuristic",
        use_dynamic_flux_grouping: bool = False,
        initiate_sym_calc: bool = True,
        input_string: str = "",
    ):
        self.branches: list[Branch] = branches_list
        self.nodes: list[Node] = nodes_list
        self.couplers: list[Coupler] = couplers_list
        self.input_string: str = input_string

        self._sys_type = type(self).__name__  # for object description

        # attributes set by methods
        self.transformation_matrix: ndarray

        self.var_categories: dict[str, list[int]] = {}
        self.external_fluxes: list[Symbol] = []
        self.closure_branches: list[Branch | dict[Branch, float]] = []

        self.symbolic_params: dict[Symbol, float] = branch_var_dict

        self.hamiltonian_symbolic: sympy.Expr
        # to store the internally used lagrangian
        self._lagrangian_symbolic: sympy.Expr
        self.lagrangian_symbolic: sympy.Expr
        # symbolic lagrangian in terms of untransformed generalized flux variables
        self.lagrangian_node_vars: sympy.Expr
        # symbolic expression for potential energy
        self.potential_symbolic: sympy.Expr
        self.potential_node_vars: sympy.Expr

        # parameters for grounding the circuit
        self.is_grounded = False
        self.ground_node: Node | None = None
        for node in self.nodes:
            if node.is_ground():
                self.ground_node = node
                self.is_grounded = True

        # switch to control the dynamic flux allocation in the loops
        self.use_dynamic_flux_grouping = use_dynamic_flux_grouping

        # parameter for choosing matrix used for basis completion in the variable
        # transformation matrix
        self.basis_completion = (
            basis_completion  # default, the other choice is standard
        )

        self.initiate_sym_calc = initiate_sym_calc

        # Calling the function to initiate the class variables
        if initiate_sym_calc:
            self.configure()

    def _is_any_branch_parameter_symbolic(self) -> bool:
        """Return ``True`` if any branch parameter is symbolic."""
        return True if len(self.symbolic_params) > 0 else False

    @staticmethod
    def _gram_schmidt(initial_vecs: ndarray, metric: ndarray) -> ndarray:
        r"""Return the Gram-Schmidt orthogonalization of `initial_vecs`.

        Orthogonalization is performed with respect to the inner product
        :math:`\langle u, v\rangle = u^\top M v` defined by the given
        `metric` matrix :math:`M`.

        Parameters
        ----------
        initial_vecs:
            row-stacked input vectors to be orthogonalized
        metric:
            square symmetric matrix defining the inner product
        """

        def inner_product(u: ndarray, v: ndarray, metric: ndarray) -> ndarray:
            """Return the metric-weighted inner product ``u @ metric @ v``."""
            return u @ metric @ v

        def projection(u: ndarray, v: ndarray, metric: ndarray) -> ndarray:
            """Return the metric-weighted projection of ``u`` onto ``v``.

            Parameters
            ----------
            u:
                vector being projected
            v:
                vector defining the projection direction
            metric:
                square symmetric matrix defining the inner product
            """
            return v * inner_product(v, u, metric) / inner_product(v, v, metric)

        orthogonal_vecs = [initial_vecs[0]]
        for i in range(1, len(initial_vecs)):
            vec = initial_vecs[i]
            projection_on_orthovecs = sum(
                [projection(vec, ortho_vec, metric) for ortho_vec in orthogonal_vecs]
            )
            orthogonal_vecs.append(vec - projection_on_orthovecs)
        return np.array(orthogonal_vecs).T

    def _orthogonalize_degenerate_eigen_vecs(
        self,
        evecs: ndarray,
        eigs: ndarray,
        relevant_eig_indices: list[int] | range | ndarray,
        cap_matrix: ndarray,
    ) -> ndarray:
        """Re-orthogonalize eigenvector columns belonging to degenerate eigenvalues.

        Within each group of (nearly) coincident eigenvalues, the corresponding
        columns of `evecs` are replaced by a Gram-Schmidt orthogonalization
        with respect to the metric `cap_matrix`. Non-degenerate columns are
        left unchanged.

        Parameters
        ----------
        evecs:
            eigenvector matrix with eigenvectors stored as columns
        eigs:
            eigenvalues corresponding to the columns of `evecs`
        relevant_eig_indices:
            indices into `eigs` that are considered when identifying
            degenerate sets
        cap_matrix:
            capacitance matrix used as the Gram-Schmidt metric
        """
        relevant_eigs = eigs[relevant_eig_indices]
        unique_eigs = np.unique(np.round(relevant_eigs, 10))
        close_eigs = [
            list(np.where(np.abs(eigs - eig) < 1e-10)[0]) for eig in unique_eigs
        ]
        degenerate_indices_list = [
            indices for indices in close_eigs if len(indices) > 1
        ]

        orthogonal_evecs = evecs.copy()

        for degenerate_set in degenerate_indices_list:
            orthogonal_evecs[:, degenerate_set] = self._gram_schmidt(
                evecs[:, degenerate_set].T, metric=cap_matrix
            )

        return orthogonal_evecs

    def _purely_harmonic_transformation(self) -> tuple[ndarray, ndarray]:
        r"""Return normal-mode frequencies and transformation for a harmonic circuit.

        Diagonalizes the generalized eigenvalue problem
        :math:`L^{-1} v = \omega^2 C\, v` using
        ``scipy.linalg.eig``, which for a general (non-Hermitian) pair can
        return complex eigenvalues; any residual imaginary part is discarded
        via ``.real`` after taking the square root of the finite, non-zero
        eigenvalues.

        Returns
        -------
        Tuple ``(normal_mode_freqs, trans_mat_new)``. ``normal_mode_freqs`` is
        an array of the positive normal-mode frequencies (square roots of the
        finite, non-zero generalized eigenvalues). ``trans_mat_new`` is the
        variable transformation matrix whose first columns span the normal
        modes orthogonalized with respect to the capacitance metric.
        """
        trans_mat, _ = self.variable_transformation_matrix()
        c_mat = (
            trans_mat.T @ self._capacitance_matrix(substitute_params=True) @ trans_mat
        )
        l_inv_mat = (
            trans_mat.T
            @ self._inductance_inverse_matrix(substitute_params=True)
            @ trans_mat
        )
        if not self.is_grounded:
            c_mat = c_mat[:-1, :-1]
            l_inv_mat = l_inv_mat[:-1, :-1]
        normal_mode_freqs, normal_mode_vecs = sp.linalg.eig(l_inv_mat, c_mat)
        normal_mode_freqs = normal_mode_freqs.round(10)
        # rounding to the tenth digit to remove numerical errors in eig calculation
        # rearranging the vectors
        idx = normal_mode_freqs.argsort()
        normal_freq_ids = [
            id
            for id in idx
            if normal_mode_freqs[id] != 0 and not np.isinf(normal_mode_freqs[id])
        ]
        zero_freq_ids = [id for id in idx if normal_mode_freqs[id] == 0]
        inf_freq_ids = [id for id in idx if np.isinf(normal_mode_freqs[id])]
        idx = normal_freq_ids + zero_freq_ids + inf_freq_ids
        # sorting so that all the zero frequencies show up at the end

        normal_mode_freqs = normal_mode_freqs[idx]
        normal_mode_vecs = normal_mode_vecs[:, idx]

        orthogonalized_normal_mode_vecs = self._orthogonalize_degenerate_eigen_vecs(
            normal_mode_vecs, normal_mode_freqs, range(len(normal_freq_ids)), c_mat
        )

        # constructing the new transformation
        trans_mat_new = trans_mat.copy()
        trans_mat_new[:, : len(c_mat)] = (
            trans_mat[:, : len(c_mat)] @ orthogonalized_normal_mode_vecs
        )

        return (
            np.real(
                np.sqrt(
                    [
                        freq
                        for freq in normal_mode_freqs
                        if not np.isinf(freq) and freq != 0
                    ]
                )
            ),
            trans_mat_new,
        )

    def configure(
        self,
        transformation_matrix: ndarray | None = None,
        closure_branches: list[Branch | dict[Branch, float]] = [],
        use_dynamic_flux_grouping: bool | None = None,
    ):
        """Initialize attributes derived from the circuit topology and parameters.

        Parameters
        ----------
        transformation_matrix:
            array used to set a transformation matrix other than the one
            generated by :meth:`variable_transformation_matrix`
        closure_branches:
            each element corresponds to one external flux variable; if the
            element is a :class:`Branch`, the external flux is associated with
            that branch, while if the element is a mapping, the external flux
            is distributed across its branches with the factors given as
            values
        use_dynamic_flux_grouping:
            optional override for :attr:`use_dynamic_flux_grouping`; when
            truthy, the instance's flag is updated before (re)building the
            symbolic circuit
        """
        if transformation_matrix is None and hasattr(self, "transformation_matrix"):
            transformation_matrix = self.transformation_matrix
        closure_branches = closure_branches or self.closure_branches
        if use_dynamic_flux_grouping:
            self.use_dynamic_flux_grouping = use_dynamic_flux_grouping

        # if the circuit is purely harmonic, then store the eigenfrequencies
        branch_type_list = [branch.type for branch in self.branches]
        self.is_purely_harmonic = "JJ" not in "".join(branch_type_list)

        if self.is_purely_harmonic:
            (
                self.normal_mode_freqs,
                transformation_matrix_normal_mode,
            ) = self._purely_harmonic_transformation()
            if transformation_matrix is None:
                transformation_matrix = transformation_matrix_normal_mode

        # if the user provides a transformation matrix
        if transformation_matrix is not None:
            self.var_categories = self.check_transformation_matrix(
                transformation_matrix, enable_warnings=not self.is_purely_harmonic
            )
            self.transformation_matrix = transformation_matrix
        # calculate the transformation matrix and identify the boundary conditions if
        # the user does not provide a custom transformation matrix
        else:
            (
                self.transformation_matrix,
                self.var_categories,
            ) = self.variable_transformation_matrix()

        # find the closure branches in the circuit
        default_spanning_tree_dict = self._spanning_tree(
            consider_capacitive_loops=self.use_dynamic_flux_grouping,
            use_closure_branches=False,
        )
        if closure_branches:
            if len(closure_branches) != len(
                flatten_list_recursive(
                    default_spanning_tree_dict["closure_branches_for_trees"]
                )
            ):
                raise ValueError(
                    """The number of closure branches should be equal to the number of loops present in the circuit. Please check the attribute spanning_tree_dict of the SymbolicCircuit instance."""
                )
            self.closure_branches = closure_branches
            self.spanning_tree_dict = self._spanning_tree(
                consider_capacitive_loops=self.use_dynamic_flux_grouping
            )
        else:
            self.spanning_tree_dict = default_spanning_tree_dict
            self.closure_branches = self._closure_branches()

        # setting external flux and offset charge variables
        self._set_external_fluxes(closure_branches=self.closure_branches)
        self.offset_charges = [
            symbols(f"ng{index}") for index in self.var_categories["periodic"]
        ]
        self.free_charges = [
            symbols(f"Qf{index}") for index in self.var_categories["free"]
        ]
        # store the flux allocation for each branches
        self.branch_flux_allocations = self._generate_branch_flux_allocations()
        # calculating the Hamiltonian directly when the number of nodes is less than 3
        substitute_params = False
        if (
            len(self.nodes) >= settings.SYM_INVERSION_MAX_NODES
            or len(self.var_categories["frozen"]) > 0
        ):  # only calculate the symbolic hamiltonian when the number of nodes is less
            # than 3. Else, the calculation will be skipped to the end when numerical
            # Hamiltonian of the circuit is requested.
            substitute_params = True
            self.frozen_var_exprs: dict[int | None, sm.Expr] = {}

        # Calculate the Lagrangian
        (
            self._lagrangian_symbolic,
            self.potential_symbolic,
            self.lagrangian_node_vars,
            self.potential_node_vars,
        ) = self.generate_symbolic_lagrangian(substitute_params=substitute_params)

        # replacing energies with capacitances in the kinetic energy of the Lagrangian
        (
            self.lagrangian_symbolic,
            self.lagrangian_node_vars,
        ) = self._replace_energies_with_capacitances_L()

        self.hamiltonian_symbolic = self.generate_symbolic_hamiltonian(
            substitute_params=substitute_params
        )

    def _replace_energies_with_capacitances_L(self) -> tuple[sympy.Expr, sympy.Expr]:
        """Replace symbolic energies in the Lagrangian with capacitance symbols.

        Symbolic capacitive energies ``ECj`` are substituted by
        ``1 / (8 * Cj)`` so that the Lagrangian reads in dimensionally
        conventional form.
        """
        # Replacing energies with capacitances if any branch parameters are symbolic
        L = self._lagrangian_symbolic.expand()
        L_old = self.lagrangian_node_vars
        if self._is_any_branch_parameter_symbolic():
            # finding the unique capacitances
            uniq_capacitances = []
            for c, b in enumerate(
                [t for t in self.branches if t.type == "C" or "JJ" in t.type]
            ):
                if len(set(b.nodes)) > 1:  # check to see if branch is shorted
                    if (
                        b.parameters[_capacitance_variable_for_branch(b.type)]
                        not in uniq_capacitances
                    ):
                        uniq_capacitances.append(
                            b.parameters[_capacitance_variable_for_branch(b.type)]
                        )

            for index, var in enumerate(uniq_capacitances):
                L = L.subs(var, 1 / (8 * symbols(f"C{index + 1}")))
                L_old = L_old.subs(var, 1 / (8 * symbols(f"C{index + 1}")))
        return L, L_old

    # Serialize will not currently work for the Circuit class.
    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return an empty default-parameter dict (serialization stub)."""
        # return {"EJ": 15.0, "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}
        return {}

    def update_param_init_val(self, param_name: str, value: float) -> None:
        """Update the initial value associated with a symbolic branch parameter.

        Parameters
        ----------
        param_name:
            name of the symbolic parameter as a string
        value:
            new numeric value to associate with the symbol
        """
        self.symbolic_params[sm.symbols(param_name)] = value
        self.configure()

    def _junction_terms(self) -> sm.Expr | int:
        r"""Return the sum of cosine Josephson terms for all standard JJ branches.

        For each Josephson branch (excluding sawtooth ``JJs`` branches) and each
        junction harmonic order, the contribution
        :math:`-E_{Jk}\cos[k(\varphi_a - \varphi_b + \varphi_\text{ext})]`
        is added, with appropriate handling of the ground node.
        """
        terms = 0
        # looping over all the junction terms
        junction_branches = [
            branch
            for branch in self.branches
            if "JJ" in branch.type and "JJs" not in branch.type
        ]
        junction_branch_order = [
            _junction_order(branch.type) for branch in junction_branches
        ]

        for branch_idx, jj_branch in enumerate(junction_branches):
            # adding external flux
            phi_ext = self.branch_flux_allocations[jj_branch.index]

            # if loop to check for the presence of ground node
            for order in range(1, junction_branch_order[branch_idx] + 1):
                junction_param = "EJ" if order == 1 else f"EJ{order}"
                if jj_branch.nodes[1].index == 0:
                    terms += -jj_branch.parameters[junction_param] * sympy.cos(
                        (order)
                        * (-sympy.symbols(f"φ{jj_branch.nodes[0].index}") + phi_ext)
                    )
                elif jj_branch.nodes[0].index == 0:
                    terms += -jj_branch.parameters[junction_param] * sympy.cos(
                        (order)
                        * (sympy.symbols(f"φ{jj_branch.nodes[1].index}") + phi_ext)
                    )
                else:
                    terms += -jj_branch.parameters[junction_param] * sympy.cos(
                        (order)
                        * (
                            (
                                sympy.symbols(f"φ{jj_branch.nodes[1].index}")
                                - sympy.symbols(f"φ{jj_branch.nodes[0].index}")
                            )
                            + phi_ext
                        )
                    )
        return terms

    def _JJs_terms(self):
        """To add terms for the sawtooth josephson junction."""
        terms = 0
        # looping over all the junction terms
        junction_branches = [branch for branch in self.branches if "JJs" in branch.type]

        # defining a function for sawtooth
        saw = sympy.Function("saw", real=True)

        for branch_idx, jj_branch in enumerate(junction_branches):
            # adding external flux
            phi_ext = self.branch_flux_allocations[jj_branch.index]

            # if loop to check for the presence of ground node
            junction_param = "EJ"
            if jj_branch.nodes[1].index == 0:
                terms += jj_branch.parameters[junction_param] * saw(
                    (-sympy.symbols(f"φ{jj_branch.nodes[0].index}") + phi_ext)
                )
            elif jj_branch.nodes[0].index == 0:
                terms += jj_branch.parameters[junction_param] * saw(
                    (sympy.symbols(f"φ{jj_branch.nodes[1].index}") + phi_ext)
                )
            else:
                terms += jj_branch.parameters[junction_param] * saw(
                    (
                        (
                            sympy.symbols(f"φ{jj_branch.nodes[1].index}")
                            - sympy.symbols(f"φ{jj_branch.nodes[0].index}")
                        )
                        + phi_ext
                    )
                )
        return terms

    def _inductance_inverse_matrix(self, substitute_params: bool = False):
        """Generate a inductance matrix for the circuit.

        Parameters
        ----------
        substitute_params:
            when set to True all the symbolic branch parameters are substituted with
            their corresponding attributes in float, by default False

        Returns
        -------
        _type_
            _description_
        """
        branches_with_inductance = [
            branch for branch in self.branches if branch.type == "L"
        ]

        param_init_vals_dict = self.symbolic_params

        # filling the non-diagonal entries
        if not self.is_grounded:
            num_nodes = len(self.nodes) - self.is_grounded
        else:
            num_nodes = len(self.nodes) - self.is_grounded + 1
        if not self._is_any_branch_parameter_symbolic() or substitute_params:
            L_mat = np.zeros([num_nodes, num_nodes])
        else:
            L_mat = sympy.zeros(num_nodes)

        for branch in branches_with_inductance:
            if len(set(branch.nodes)) > 1:  # branch if shorted is not considered
                inductance = branch.parameters["EL"]
                if not isinstance(inductance, float) and substitute_params:
                    inductance = param_init_vals_dict[inductance]
                if self.is_grounded:
                    L_mat[branch.nodes[0].index, branch.nodes[1].index] += -inductance
                else:
                    L_mat[
                        branch.nodes[0].index - 1, branch.nodes[1].index - 1
                    ] += -inductance

        if not self._is_any_branch_parameter_symbolic() or substitute_params:
            L_mat = L_mat + L_mat.T - np.diag(L_mat.diagonal())
        else:
            L_mat = L_mat + L_mat.T - sympy.diag(*L_mat.diagonal())

        for i in range(L_mat.shape[0]):  # filling the diagonal entries
            L_mat[i, i] = -np.sum(L_mat[i, :])

        if self.is_grounded:  # if grounded remove the 0th column and row from L_mat
            L_mat = L_mat[1:, 1:]
        return L_mat

    def _capacitance_matrix(self, substitute_params: bool = False):
        """Generate a capacitance matrix for the circuit.

        Parameters
        ----------
        substitute_params:
            when set to True all the symbolic branch parameters are substituted with
            their corresponding attributes in float, by default False

        Returns
        -------
        _type_
            _description_
        """
        branches_with_capacitance = [
            branch
            for branch in self.branches
            if ("C" == branch.type or "JJ" in branch.type)
        ]

        param_init_vals_dict = self.symbolic_params

        # filling the non-diagonal entries
        if not self.is_grounded:
            num_nodes = len(self.nodes) - self.is_grounded
        else:
            num_nodes = len(self.nodes) - self.is_grounded + 1
        if not self._is_any_branch_parameter_symbolic() or substitute_params:
            C_mat = np.zeros([num_nodes, num_nodes])
        else:
            C_mat = sympy.zeros(num_nodes)

        for branch in branches_with_capacitance:
            if len(set(branch.nodes)) > 1:  # branch if shorted is not considered
                capacitance = branch.parameters[
                    _capacitance_variable_for_branch(branch.type)
                ]
                if not isinstance(capacitance, float) and substitute_params:
                    capacitance = param_init_vals_dict[capacitance]
                if self.is_grounded:
                    C_mat[branch.nodes[0].index, branch.nodes[1].index] += -1 / (
                        capacitance * 8
                    )
                else:
                    C_mat[
                        branch.nodes[0].index - 1, branch.nodes[1].index - 1
                    ] += -1 / (capacitance * 8)

        if not self._is_any_branch_parameter_symbolic() or substitute_params:
            C_mat = C_mat + C_mat.T - np.diag(C_mat.diagonal())
        else:
            C_mat = C_mat + C_mat.T - sympy.diag(*C_mat.diagonal())

        for i in range(C_mat.shape[0]):  # filling the diagonal entries
            C_mat[i, i] = -np.sum(C_mat[i, :])

        if self.is_grounded:  # if grounded remove the 0th column and row from C_mat
            C_mat = C_mat[1:, 1:]
        return C_mat

    def _EC_matrix(self, substitute_params: bool = False) -> ndarray | sm.Matrix:
        """Return the charging energy matrix for the circuit.

        Parameters
        ----------
        substitute_params:
            when ``True``, all symbolic branch parameters are replaced by their
            numeric default values (default: ``False``)
        """
        transformation_matrix = self.transformation_matrix

        frozen_indices = [
            i - 1 for i in self.var_categories["frozen"] + self.var_categories["sigma"]
        ]
        # generating the C_mat_θ by inverting the capacitance matrix
        if self._is_any_branch_parameter_symbolic() and not substitute_params:
            C_mat_θ = (
                transformation_matrix.T
                * self._capacitance_matrix()
                * transformation_matrix
            )
            relevant_indices = [
                i for i in range(C_mat_θ.shape[0]) if i not in frozen_indices
            ]
            C_mat_θ = C_mat_θ[relevant_indices, relevant_indices]
            EC_mat_θ = C_mat_θ.inv()
        else:
            C_mat_θ = (
                transformation_matrix.T
                @ self._capacitance_matrix(substitute_params=substitute_params)
                @ transformation_matrix
            )
            C_mat_θ = np.delete(C_mat_θ, frozen_indices, 0)
            C_mat_θ = np.delete(C_mat_θ, frozen_indices, 1)
            EC_mat_θ = np.linalg.inv(C_mat_θ)
        return EC_mat_θ

    def _capacitor_terms(self) -> sm.Expr | int:
        r"""Return the kinetic-energy contribution from all capacitive branches.

        For each capacitive (or junction) branch, the term
        :math:`(v_{\varphi_a} - v_{\varphi_b})^2 / (16 E_C)` is accumulated,
        with the ground-node case handled separately.
        """
        terms = 0
        branches_with_capacitance = [
            branch
            for branch in self.branches
            if branch.type == "C" or "JJ" in branch.type
        ]
        for c_branch in branches_with_capacitance:
            if c_branch.nodes[1].index == 0:
                terms += (
                    1
                    / (
                        16
                        * c_branch.parameters[
                            _capacitance_variable_for_branch(c_branch.type)
                        ]
                    )
                    * (symbols(f"vφ{c_branch.nodes[0].index}")) ** 2
                )
            elif c_branch.nodes[0].index == 0:
                terms += (
                    1
                    / (
                        16
                        * c_branch.parameters[
                            _capacitance_variable_for_branch(c_branch.type)
                        ]
                    )
                    * (-symbols(f"vφ{c_branch.nodes[1].index}")) ** 2
                )
            else:
                terms += (
                    1
                    / (
                        16
                        * c_branch.parameters[
                            _capacitance_variable_for_branch(c_branch.type)
                        ]
                    )
                    * (
                        symbols(f"vφ{c_branch.nodes[1].index}")
                        - symbols(f"vφ{c_branch.nodes[0].index}")
                    )
                    ** 2
                )
        return terms

    def _inductance_matrix_branch_vars(
        self, substitute_params: bool = False, return_inverse: bool = False
    ):
        """Generate the branch-variable inductance matrix, including mutuals.

        Parameters
        ----------
        substitute_params:
            when ``True``, all symbolic branch parameters are replaced by their
            numeric default values (default: ``False``)
        return_inverse:
            when ``True``, the inverse of the inductance matrix is returned
            (default: ``False``)

        Returns
        -------
        Inductance matrix (or its inverse) restricted to the inductive
        branches. The matrix is returned as a ``numpy.ndarray`` whenever the
        branch parameters are numeric (or `substitute_params` is ``True``);
        otherwise as a ``sympy.Matrix``.
        """
        num_branches = len(self.branches)
        if not self._is_any_branch_parameter_symbolic() or substitute_params:
            L_mat = np.zeros([num_branches, num_branches])
        else:
            L_mat = sympy.zeros(num_branches)
        # filling the diagonal entries
        for branch in [b for b in self.branches if b.type == "L"]:
            EL = branch.parameters["EL"]
            if not isinstance(EL, float) and substitute_params:
                EL = self.symbolic_params[EL]
            L_mat[branch.index, branch.index] = 1 / EL
        # filling the non-diagonal entries
        for idx, coupler in enumerate([c for c in self.couplers if c.type == "ML"]):
            EML = coupler.parameters["EML"]
            if not isinstance(EML, float) and substitute_params:
                EML = self.symbolic_params[EML]
            L_mat[coupler.branches[0].index, coupler.branches[1].index] = 1 / EML
            L_mat[coupler.branches[1].index, coupler.branches[0].index] = 1 / EML
        # remove all zero rows and columns
        irrelevant_indices = [
            idx for idx in range(num_branches) if self.branches[idx].type != "L"
        ]
        L_mat = np.delete(L_mat, irrelevant_indices, axis=0)
        L_mat = np.delete(L_mat, irrelevant_indices, axis=1)
        if return_inverse:
            if substitute_params or not self._is_any_branch_parameter_symbolic():
                return np.linalg.inv(L_mat)
            else:
                # return sympy matrix inverse
                return sympy.Matrix(L_mat).inv()

        return L_mat

    def _generate_branch_flux_allocations(self):
        """Return an array of the flux allocation for each branch in the circuit."""
        if self.use_dynamic_flux_grouping:
            return self._time_dependent_flux_distribution()

        if not self.closure_branches or len(self.closure_branches) == 0:
            return np.zeros(len(self.branches))
        flux_allocation_array = np.zeros(
            (len(self.branches), len(self.closure_branches))
        )
        if not self.closure_branches:
            return flux_allocation_array
        for flux_idx, element in enumerate(self.closure_branches):
            if isinstance(element, Branch):
                flux_allocation_array[element.index, flux_idx] = 1
                continue
            for branch in element:
                flux_allocation_array[branch.index, flux_idx] = element[branch]
        return np.dot(flux_allocation_array, self.external_fluxes)

    def _inductor_terms(self, substitute_params: bool = False) -> sm.Expr | int:
        """Return terms corresponding to purely inductive branches in the circuit.

        Parameters
        ----------
        substitute_params:
            when ``True``, all symbolic branch parameters are replaced by their
            numeric default values (default: ``False``)
        """
        inverse_inductance_mat = self._inductance_matrix_branch_vars(
            substitute_params, return_inverse=True
        )
        branch_fluxes = []
        for l_branch in [branch for branch in self.branches if branch.type == "L"]:
            # adding external flux
            phi_ext = self.branch_flux_allocations[l_branch.index]

            if l_branch.nodes[0].index == 0:
                branch_flux = symbols(f"φ{l_branch.nodes[1].index}") + phi_ext
            elif l_branch.nodes[1].index == 0:
                branch_flux = -symbols(f"φ{l_branch.nodes[0].index}") + phi_ext
            else:
                branch_flux = (
                    symbols(f"φ{l_branch.nodes[1].index}")
                    - symbols(f"φ{l_branch.nodes[0].index}")
                    + phi_ext
                )
            branch_fluxes.append(branch_flux)
        branch_currents = inverse_inductance_mat @ np.array(branch_fluxes)
        terms = 0
        for idx, branch in enumerate(
            [branch for branch in self.branches if branch.type == "L"]
        ):
            terms += 0.5 * 1 / (branch.parameters["EL"]) * branch_currents[idx] ** 2
        # substitute params if necessary
        if substitute_params and terms != 0:
            for symbol in self.symbolic_params:
                terms = terms.subs(symbol.name, self.symbolic_params[symbol])  # type: ignore[attr-defined]
        return terms

    def _node_voltage_exprs(self, substitute_params: bool = True) -> list[sm.Expr]:
        """Return the node-voltage expressions in terms of the new variables.

        The transformed momenta are ordered by ``np.sort`` over the periodic,
        extended, and free variable indices so that they line up with the
        column ordering of the transformation matrix.

        Parameters
        ----------
        substitute_params:
            when ``True`` (default), all symbolic branch parameters are
            replaced by their numeric default values
        """
        transformation_matrix = self.transformation_matrix
        EC_mat_θ = self._EC_matrix(substitute_params=substitute_params)
        p_θ_vars = [
            (
                symbols(f"Q{i}")
                if i not in self.var_categories["free"]
                else symbols(f"Qf{i}")
            )
            for i in np.sort(
                self.var_categories["periodic"]
                + self.var_categories["extended"]
                + self.var_categories["free"]
            )
            # replacing the free charge with 0, as it would not affect the circuit
            # Lagrangian.
        ]
        voltages = list(EC_mat_θ * sympy.Matrix(p_θ_vars))

        # insert the voltages for frozen modes
        for index in self.var_categories["sigma"]:
            voltages.insert(index, 0)
        # substitute expressions for frozen variables
        for index in self.var_categories["frozen"]:
            frozen_var_expr = self.frozen_var_exprs[index]
            frozen_var_expr = frozen_var_expr.subs(
                [
                    (var_sym, f"Q{get_trailing_number(var_sym.name)}")
                    for var_sym in frozen_var_expr.free_symbols
                ]
            )
            voltages.insert(index, round_symbolic_expr(frozen_var_expr, 10))

        node_voltages = list(transformation_matrix * sympy.Matrix(voltages))

        if self.is_grounded:
            node_voltages = [0] + node_voltages
        return node_voltages

    def _branch_charge_expr(
        self, branch: Branch, substitute_params: bool = True
    ) -> sm.Expr:
        """Return the charge on the branch in terms of the charge operators.

        Parameters
        ----------
        branch:
            one of the branches of the circuit
        substitute_params:
            when ``True`` (default), substitute the symbolic branch parameters
            with their corresponding numeric values

        Returns
        -------
        Symbolic expression of charge on the branch.
        """
        node_voltages = self._node_voltage_exprs(substitute_params=substitute_params)
        node_id1, node_id2 = [
            node.index - (1 if not self.is_grounded else 0) for node in branch.nodes
        ]
        branch_voltage_expr = node_voltages[node_id1] - node_voltages[node_id2]
        # adding the offset charge variables
        for var_index in self.var_categories["periodic"]:
            branch_voltage_expr = branch_voltage_expr.subs(
                symbols(f"Q{var_index}"),
                symbols(f"n{var_index}") + symbols(f"ng{var_index}"),
            )
        return branch_voltage_expr * (
            1 / (8 * branch.parameters[_capacitance_variable_for_branch(branch.type)])
        )

    def _branch_flux_expr(
        self,
        branch: Branch,
    ) -> sm.Expr:
        """Return the branch flux expression in terms of the new variables.

        Parameters
        ----------
        branch:
            one of the branches of the circuit

        Returns
        -------
        Symbolic expression of flux across the branch.
        """
        transformation_matrix = self.transformation_matrix

        node_id1, node_id2 = [node.index for node in branch.nodes]
        expr_node_vars = symbols(f"φ{node_id2}") - symbols(f"φ{node_id1}")
        expr_node_vars = expr_node_vars.subs(
            "φ0", 0
        )  # substituting node flux of ground to zero
        num_vars = len(self.nodes) - self.is_grounded
        new_vars = [
            (
                symbols(f"θ{index}")
                if index not in self.var_categories["frozen"]
                else self.frozen_var_exprs[index]
            )
            for index in range(1, 1 + num_vars)
        ]
        # free variables do not show up in the branch flux expression for inductors, assuming capacitances do not depend on the flux, but charge expression
        old_vars = [symbols(f"φ{index}") for index in range(1, 1 + num_vars)]
        transformed_expr = transformation_matrix.dot(new_vars)
        # add external flux
        phi_ext = self.branch_flux_allocations[branch.index]
        for idx, var in enumerate(old_vars):
            expr_node_vars = expr_node_vars.subs(var, transformed_expr[idx])
        return round_symbolic_expr(expr_node_vars + phi_ext, 12)

    def generate_symbolic_lagrangian(
        self, substitute_params: bool = False
    ) -> tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
        r"""Build the symbolic Lagrangian and potential in node and new variables.

        The :math:`\theta` variables denote the transformed (new) variables and
        the :math:`\varphi` variables denote the node variables.

        Parameters
        ----------
        substitute_params:
            when ``True``, all symbolic branch parameters are replaced by
            their numeric default values (default: ``False``)

        Returns
        -------
        Tuple ``(lagrangian_θ, potential_θ, lagrangian_φ, potential_φ)``.
        """
        transformation_matrix = self.transformation_matrix

        # defining the φ variables
        φ_dot_vars = [
            symbols(f"vφ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]

        # defining the θ variables
        θ_vars = [
            symbols(f"θ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]
        # defining the θ dot variables
        θ_dot_vars = [
            symbols(f"vθ{i}") for i in range(1, len(self.nodes) - self.is_grounded + 1)
        ]
        # writing φ in terms of θ variables
        φ_vars_θ = transformation_matrix.dot(θ_vars)
        # writing φ dot vars in terms of θ variables
        φ_dot_vars_θ = transformation_matrix.dot(θ_dot_vars)

        # C_terms = self._C_terms()
        C_mat = self._capacitance_matrix()
        if not self._is_any_branch_parameter_symbolic():
            # in terms of node variables
            C_terms_φ = C_mat.dot(φ_dot_vars).dot(φ_dot_vars) * 0.5
            # in terms of new variables
            C_terms_θ = C_mat.dot(φ_dot_vars_θ).dot(φ_dot_vars_θ) * 0.5
        else:
            C_terms_φ = (sympy.Matrix(φ_dot_vars).T * C_mat * sympy.Matrix(φ_dot_vars))[
                0
            ] * 0.5  # in terms of node variables
            C_terms_θ = (
                sympy.Matrix(φ_dot_vars_θ).T * C_mat * sympy.Matrix(φ_dot_vars_θ)
            )[
                0
            ] * 0.5  # in terms of new variables

        inductor_terms_φ = self._inductor_terms(substitute_params=substitute_params)

        JJ_terms_φ = self._junction_terms() + self._JJs_terms()

        lagrangian_φ = C_terms_φ - inductor_terms_φ - JJ_terms_φ

        potential_φ = inductor_terms_φ + JJ_terms_φ
        potential_θ = (
            potential_φ.copy() if potential_φ != 0 else symbols("x") * 0
        )  # copying the potential in terms of the old variables to make substitutions

        for index in range(
            len(self.nodes) - self.is_grounded
        ):  # converting potential to new variables
            potential_θ = potential_θ.subs(symbols(f"φ{index + 1}"), φ_vars_θ[index])

        # eliminating the frozen variables
        if len(self.var_categories["frozen"]) > 0:
            frozen_eom_list = []
            for frozen_var_index in self.var_categories["frozen"]:
                frozen_eom_list.append(
                    potential_θ.diff(symbols(f"θ{frozen_var_index}"))
                )

            frozen_exprs = sympy.solve(
                frozen_eom_list,
                [
                    symbols(f"θ{frozen_var_index}")
                    for frozen_var_index in self.var_categories["frozen"]
                ],
            )

            for frozen_var, frozen_expr in frozen_exprs.items():
                potential_θ = potential_θ.replace(frozen_var, frozen_expr).expand()
            self.frozen_var_exprs = {
                get_trailing_number(frozen_var.name): frozen_expr
                for frozen_var, frozen_expr in frozen_exprs.items()
            }

        lagrangian_θ = C_terms_θ - potential_θ

        return lagrangian_θ, potential_θ, lagrangian_φ, potential_φ

    def generate_symbolic_hamiltonian(
        self, substitute_params: bool = False, reevaluate_lagrangian: bool = False
    ) -> sympy.Expr:
        r"""Return the Hamiltonian in terms of the new variables :math:`\theta_i`.

        The transformed momenta are ordered by ``np.sort`` over the periodic,
        extended, and free variable indices so that they line up with the
        column ordering of the transformation matrix.

        Parameters
        ----------
        substitute_params:
            when ``True``, the symbols defined for branch parameters are
            substituted with the numerical values stored on the instance
        reevaluate_lagrangian:
            when ``True``, the symbolic Lagrangian (and hence the potential)
            is regenerated by calling :meth:`generate_symbolic_lagrangian`
            instead of reusing the cached :attr:`potential_symbolic`
        """
        if reevaluate_lagrangian:
            _, potential_symbolic, _, _ = self.generate_symbolic_lagrangian(
                substitute_params=substitute_params
            )
        else:
            potential_symbolic = self.potential_symbolic

        EC_mat_θ = self._EC_matrix(substitute_params=substitute_params)

        p_φ_vars = [
            (
                symbols(f"Q{i}")
                if i not in self.var_categories["free"]
                else symbols(f"Qf{i}")
            )
            for i in np.sort(
                self.var_categories["periodic"]
                + self.var_categories["extended"]
                + self.var_categories["free"]
            )
            # replacing the free charge with 0, as it would not affect the circuit
            # Lagrangian.
        ]
        # NOTE: sorting the variable indices in the above step is important as the transformation
        # matrix already takes care of defining the appropriate momenta in the new variables. So, the above variables should be
        # in the node variable order.
        # generating the kinetic energy terms for the Hamiltonian
        if not self._is_any_branch_parameter_symbolic():
            C_terms_new = (
                EC_mat_θ.dot(p_φ_vars).dot(p_φ_vars) * 0.5
            )  # in terms of new variables
        else:
            C_terms_new = (
                sympy.Matrix(p_φ_vars).T * EC_mat_θ * sympy.Matrix(p_φ_vars)
            )[
                0
            ] * 0.5  # in terms of new variables

        hamiltonian_symbolic = C_terms_new + potential_symbolic

        # adding the offset charge variables
        for var_index in self.var_categories["periodic"]:
            hamiltonian_symbolic = hamiltonian_symbolic.subs(
                symbols(f"Q{var_index}"),
                symbols(f"n{var_index}") + symbols(f"ng{var_index}"),
            )

        return round_symbolic_expr(hamiltonian_symbolic.expand(), 12)

    def trans_cap_matrix(
        self,
        substitute_params: bool = False,
    ) -> sympy.Expr:
        """Calculate the transformed capacitance matrix ``C_mat_theta``.

        Parameters
        ----------
        substitute_params:
            when ``True``, substitute symbolic parameters with their numerical
            values (default: ``False``)

        Returns
        -------
        Transformed capacitance matrix ``C_mat_theta``, returned as a
        ``numpy.ndarray`` when the branch parameters are numeric (or
        `substitute_params` is ``True``); otherwise as a ``sympy.Matrix``.
        """
        # Calculate indices to be excluded from the matrix
        frozen_indices = [
            i - 1 for i in self.var_categories["frozen"] + self.var_categories["sigma"]
        ]

        # Generate the transformed capacitance matrix
        if self._is_any_branch_parameter_symbolic() and not substitute_params:
            C_mat_theta = (
                self.transformation_matrix.T
                * self._capacitance_matrix()
                * self.transformation_matrix
            )
            relevant_indices = [
                i for i in range(C_mat_theta.shape[0]) if i not in frozen_indices
            ]
            C_mat_theta = C_mat_theta[relevant_indices, relevant_indices]

        else:
            C_mat_theta = (
                self.transformation_matrix.T
                @ self._capacitance_matrix(substitute_params=substitute_params)
                @ self.transformation_matrix
            )
            C_mat_theta = np.delete(C_mat_theta, frozen_indices, 0)
            C_mat_theta = np.delete(C_mat_theta, frozen_indices, 1)

        return C_mat_theta

    def inv_trans_cap_matrix(
        self,
        substitute_params: bool = False,
    ) -> sympy.Expr:
        """Calculate the inverse of the transformed capacitance matrix.

        Parameters
        ----------
        substitute_params:
            when ``True``, all symbolic branch parameters are replaced by
            their numeric default values (default: ``False``)
        """
        C_mat_theta = self.trans_cap_matrix(substitute_params=substitute_params)

        # remove free indices
        free_indices = [i - 1 for i in self.var_categories["free"]]
        if self._is_any_branch_parameter_symbolic() and not substitute_params:
            inv_C_mat_theta = C_mat_theta.inv()

            relevant_indices = [
                i for i in range(C_mat_theta.shape[0]) if i not in free_indices
            ]
            inv_C_mat_theta = inv_C_mat_theta[relevant_indices, relevant_indices]

        else:
            inv_C_mat_theta = np.linalg.inv(C_mat_theta)
            inv_C_mat_theta = np.delete(inv_C_mat_theta, free_indices, 0)
            inv_C_mat_theta = np.delete(inv_C_mat_theta, free_indices, 1)

        return inv_C_mat_theta
