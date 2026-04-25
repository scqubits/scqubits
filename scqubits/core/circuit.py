# circuit.py
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

from __future__ import annotations

import re
import warnings

from collections.abc import Callable
from typing import Any, get_type_hints

import numpy as np
import sympy as sm

from numpy import ndarray
from sympy import latex

try:
    import IPython
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

import scqubits.core.central_dispatch as dispatch
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.circuit_noise import NoisyCircuit
from scqubits.core.circuit_plotting import CircuitPlot
from scqubits.core.circuit_routines import CircuitRoutines
from scqubits.core.circuit_sym_methods import CircuitSymMethods
from scqubits.core.circuit_utils import (
    get_trailing_number,
)
from scqubits.core.symbolic_circuit import Branch, SymbolicCircuit
from scqubits.utils.misc import (
    flatten_list,
    flatten_list_recursive,
    is_string_float,
    number_of_lists_in_list,
)


class ConfigureError(RuntimeError):
    """Raised when ``Circuit.configure`` fails; the previous configuration is restored."""


class CircuitABC(CircuitRoutines, CircuitSymMethods, CircuitPlot):
    """Abstract base aggregating circuit routines, symbolic methods, and plotting."""

    pass


class Subsystem(  # type: ignore[misc]
    CircuitABC,
    base.QubitBaseClass,
    serializers.Serializable,
    dispatch.DispatchClient,
    NoisyCircuit,
):
    """Defines a subsystem of a circuit, usable recursively to define nested subsystems.

    Parameters
    ----------
    parent:
        the instance under which the new subsystem is defined.
    ext_basis:
        The basis that should be used for extended variables
    hamiltonian_symbolic:
        The symbolic expression which defines the Hamiltonian for the new subsystem
    system_hierarchy:
        Defines the hierarchy of the new subsystem, is set to None when hierarchical
        diagonalization is not required. by default None
    subsystem_trunc_dims:
        Defines the truncated dimensions for the subsystems inside the current
        subsystem, is set to None when hierarchical diagonalization is not required,
        by default ``None``
    truncated_dim:
        sets the truncated dimension for the current subsystem, set to 10 by default.
    evals_method:
        optional override for the eigenvalue solver routine, by default ``None``
    evals_method_options:
        keyword options forwarded to ``evals_method``, by default ``None``
    esys_method:
        optional override for the eigensystem solver routine, by default ``None``
    esys_method_options:
        keyword options forwarded to ``esys_method``, by default ``None``

    Attributes
    ----------
    hierarchical_diagonalization: bool
        set to True when the circuit is defined hierarchically, by default ``False``
    hamiltonian_symbolic: Sympy.Expr
        the symbolic Hamiltonian for the circuit
    external_fluxes: list[Sympy.Symbol]
        list of external flux variables
    offset_charges: list[Sympy.Symbol]
        list of offset charge variables
    free_charges: list[Sympy.Symbol]
        list of free charge variables
    var_categories: dict[str, list[int]]
        dictionary with keys "periodic", "extended", "free", "frozen" and values as
        the indices of the respective variable types
    cutoff_names: list[str]
        list of cutoff names for the variables
    discretized_phi_range: dict[int, tuple[float, float]]
        dictionary with keys as the indices of the extended variables and values as
        the range of discretized phi variables
    type_of_matrices: str
        type of matrices used to construct the operators "dense" or "sparse",
        by default "sparse"
    system_hierarchy: list
        nested list of variable indices provided by the user to define subsystems
    subsystem_trunc_dims: list
        list of truncated dimensions for the subsystems inside the current subsystem
    hilbert_space: HilbertSpace
        HilbertSpace instance for the circuit, when hierarchical diagonalization is used
    truncated_dim: int
        truncated dimension for the current instance
    is_purely_harmonic: bool
        internally set to True when the instance is purely harmonic
    """

    def __init__(
        self,
        parent: "Subsystem",
        hamiltonian_symbolic: sm.Expr,
        ext_basis: str | list,
        system_hierarchy: list | None = None,
        subsystem_trunc_dims: list | None = None,
        truncated_dim: int = 10,
        evals_method: Callable | str | None = None,
        evals_method_options: dict | None = None,
        esys_method: Callable | str | None = None,
        esys_method_options: dict | None = None,
    ):
        # switch used in protecting the class from erroneous addition of new attributes
        object.__setattr__(self, "_frozen", False)

        base.QubitBaseClass.__init__(
            self,
            id_str=None,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )

        self.system_hierarchy = system_hierarchy if system_hierarchy is not None else []
        self.truncated_dim: int = truncated_dim
        self.subsystem_trunc_dims = (
            subsystem_trunc_dims if subsystem_trunc_dims is not None else []
        )

        self.is_child: bool = True
        self.parent = parent
        self.hamiltonian_symbolic = hamiltonian_symbolic
        self._default_grid_phi = self.parent._default_grid_phi

        self.junction_potential = None
        self._H_LC_str_harmonic = None
        self.ext_basis = ext_basis

        self.dynamic_var_indices: list[int] = flatten_list_recursive(
            [self.system_hierarchy]
        )

        self.var_categories: dict[str, list[int]] = {}
        for var_type in self.parent.var_categories:
            self.var_categories[var_type] = [
                var_index
                for var_index in self.parent.var_categories[var_type]
                if var_index in self.dynamic_var_indices
            ]

        self.cutoff_names: list[str] = []
        for var_type in self.var_categories.keys():
            if var_type == "periodic":
                for var_index in self.var_categories["periodic"]:
                    self.cutoff_names.append(f"cutoff_n_{var_index}")
            if var_type == "extended":
                for var_index in self.var_categories["extended"]:
                    self.cutoff_names.append(f"cutoff_ext_{var_index}")

        self.discretized_phi_range: dict[int, tuple[float]] = {
            idx: self.parent.discretized_phi_range[idx]
            for idx in self.parent.discretized_phi_range
            if idx in self.dynamic_var_indices
        }

        # storing the potential terms separately

        self.potential_symbolic = self._generate_sym_potential()

        self.hierarchical_diagonalization: bool = (
            self.system_hierarchy != []
            and number_of_lists_in_list(self.system_hierarchy) > 0
        )

        if len(self.dynamic_var_indices) == 1:
            self.type_of_matrices = "dense"
        else:
            self.type_of_matrices = "sparse"

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []

        # attributes for purely harmonic
        self.normal_mode_freqs: np.ndarray = np.array([], dtype=float)

        self._configure()
        self._frozen = True

    def _find_and_set_sym_attrs(self):
        """Find symbolic and other circuit params from the symbolic Hamiltonian.

        Sets the attributes ``external_fluxes``, ``offset_charges`` and
        ``symbolic_params``. Only works when ``_frozen`` is set to ``False``, or the
        above attributes are already set.
        """
        self.external_fluxes = [
            var
            for var in self.parent.external_fluxes
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.offset_charges = [
            var
            for var in self.parent.offset_charges
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.free_charges = [
            var
            for var in self.parent.free_charges
            if var in self.hamiltonian_symbolic.free_symbols
        ]
        self.symbolic_params = {
            var: self.parent.symbolic_params[var]
            for var in self.parent.symbolic_params
            if var in self.hamiltonian_symbolic.free_symbols
        }
        for idx, param in enumerate(self.symbolic_params):
            self._make_property(
                param.name, getattr(self.parent, param.name), "update_param_vars"
            )

        # getting attributes from parent
        for flux in self.external_fluxes:
            self._make_property(
                flux.name,
                getattr(self.parent, flux.name),
                "update_external_flux_or_charge",
            )
        for charge_var in self.offset_charges + self.free_charges:
            self._make_property(
                charge_var.name,
                getattr(self.parent, charge_var.name),
                "update_external_flux_or_charge",
            )
        for cutoff_str in self.cutoff_names:
            self._make_property(
                cutoff_str, getattr(self.parent, cutoff_str), "update_cutoffs"
            )

    def _configure(
        self,
        subsys_dict: dict[str, Any] | None = None,
    ) -> None:
        """Initiate the subsystem instance.

        Parameters
        ----------
        subsys_dict:
            optional dictionary with keys ``"systems_sym"`` and ``"interaction_sym"``
            defining the symbolic Hamiltonians and interactions for the subsystems;
            by default ``None``, in which case it is generated internally.
        """
        self._frozen = False

        self._find_and_set_sym_attrs()

        # if subsystem hamiltonian is purely harmonic
        if (
            self._is_expression_purely_harmonic(self.hamiltonian_symbolic)
            and self.ext_basis == "harmonic"
        ):
            self.is_purely_harmonic = True
            self._annihilation_operator_in_eigenbasis = None
        else:
            self.is_purely_harmonic = False

        if self.hierarchical_diagonalization:
            # attribute to note updated subsystem indices
            self._hamiltonian_sym_for_numerics = self.hamiltonian_symbolic.copy()
            self._generate_subsystems(subsys_dict=subsys_dict)
            self.ext_basis = self.get_ext_basis()
            self._update_interactions()
            self._check_truncation_indices()
            self.affected_subsystem_indices = list(range(len(self.subsystems)))
        else:
            self._generate_hamiltonian_sym_for_numerics()
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                self._diagonalize_purely_harmonic_hamiltonian()

        self._set_vars()
        self.operators_by_name = self._set_operators()

        self._out_of_sync_with_parent = False
        if self.hierarchical_diagonalization:
            self._out_of_sync = False  # for use with CentralDispatch
            dispatch.CENTRAL_DISPATCH.register("CIRCUIT_UPDATE", self)
        self._frozen = True

    def _is_diagonalization_necessary(self) -> bool:
        """Check if the subsystem needs to be diagonalized."""
        parent_subsys_idx = self.parent.subsystems.index(self)
        if parent_subsys_idx in self.parent.affected_subsystem_indices:
            return True
        return False


class Circuit(  # type: ignore[misc]
    CircuitABC,
    base.QubitBaseClass,
    serializers.Serializable,
    dispatch.DispatchClient,
    NoisyCircuit,
):
    """Class for analysis of custom superconducting circuits.

    Parameters
    ----------
    input_string:
        String describing the number of nodes and branches connecting then along
        with their parameters
    from_file:
        set to True by default, when a file name should be provided to
        `input_string`, else the circuit graph description in YAML should be
        provided as a string.
    basis_completion:
        either "heuristic" or "canonical", defines the matrix used for completing the
        transformation matrix. Sometimes used to change the variable transformation
        to result in a simpler symbolic Hamiltonian, by default "heuristic"
    ext_basis:
        can be "discretized" or "harmonic" which chooses whether to use discretized
        phi or harmonic oscillator basis for extended variables,
        by default "discretized"
    use_dynamic_flux_grouping:
        set to False by default. Indicates if the flux allocation is done by assuming
        that flux is time dependent. When set to True, it disables the option to change
        the closure branches.
    generate_noise_methods:
        when ``True``, generate per-channel noise methods on the instance after
        construction; by default ``False``.
    initiate_sym_calc:
        parameter to initiate Circuit instance, by default ``True``
    truncated_dim:
        truncated dimension if the user wants to use this circuit instance in
        HilbertSpace, by default ``None``
    symbolic_param_dict:
        mapping from symbolic-parameter name (string) to numerical value, used when
        the instance is initialized from a ``symbolic_hamiltonian`` rather than a
        YAML input string; by default ``{}``.
    symbolic_hamiltonian:
        optional pre-built symbolic Hamiltonian. If provided, ``input_string`` must
        be ``None``; by default ``None``.
    evals_method:
        optional override for the eigenvalue solver routine, by default ``None``
    evals_method_options:
        keyword options forwarded to ``evals_method``, by default ``None``
    esys_method:
        optional override for the eigensystem solver routine, by default ``None``
    esys_method_options:
        keyword options forwarded to ``esys_method``, by default ``None``

    Attributes
    ----------
    hierarchical_diagonalization: bool
        set to True when the circuit is defined hierarchically, by default ``False``
    hamiltonian_symbolic: sm.Expr
        the symbolic Hamiltonian for the circuit
    external_fluxes: list[sm.Symbol]
        list of external flux variables
    offset_charges: list[sm.Symbol]
        list of offset charge variables
    free_charges: list[sm.Symbol]
        list of free charge variables
    var_categories: dict[str, list[int]]
        dictionary with keys "periodic", "extended", "free", "frozen" and values as
        the indices of the respective variable types
    cutoff_names: list[str]
        list of cutoff names for the variables
    discretized_phi_range: dict[int, tuple[float, float]]
        dictionary with keys as the indices of the extended variables and values as
        the range of discretized phi variables
    type_of_matrices: str
        type of matrices used to construct the operators "dense" or "sparse",
        by default "sparse"
    truncated_dim: int
        truncated dimension for the current instance
    is_purely_harmonic: bool
        internally set to True when the instance is purely harmonic
    dynamic_var_indices: list[int]
        list of dynamic variable indices, showing the degrees of freedom of the circuit.
    hilbert_space: HilbertSpace
        HilbertSpace instance for the instance, when hierarchical diagonalization is used
    system_hierarchy: list
        nested list of variable indices provided by the user to define subsystems
    subsystem_trunc_dims: list
        list of truncated dimensions for the subsystems inside the current subsystem
    """

    def __init__(
        self,
        input_string: str | None = None,
        from_file: bool = True,
        basis_completion: str = "heuristic",
        ext_basis: str = "discretized",
        use_dynamic_flux_grouping: bool = False,
        generate_noise_methods: bool = False,
        initiate_sym_calc: bool = True,
        truncated_dim: int = 10,
        symbolic_param_dict: dict[str, float] = {},
        symbolic_hamiltonian: sm.Expr | None = None,
        evals_method: Callable | str | None = None,
        evals_method_options: dict | None = None,
        esys_method: Callable | str | None = None,
        esys_method_options: dict | None = None,
    ):
        # switch used in protecting the class from erroneous addition of new attributes
        object.__setattr__(self, "_frozen", False)
        base.QubitBaseClass.__init__(
            self,
            id_str=None,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        if symbolic_hamiltonian and input_string:
            raise Exception(
                "Circuit instance cannot be initialized with both input_string and a symbolic hamiltonian."
            )
        if not symbolic_hamiltonian and not input_string:
            raise Exception(
                "Circuit instance must be initialized with either input_string or a symbolic hamiltonian."
            )
        if input_string:
            self.from_yaml(
                input_string=input_string,
                from_file=from_file,
                basis_completion=basis_completion,
                ext_basis=ext_basis,
                use_dynamic_flux_grouping=use_dynamic_flux_grouping,
                generate_noise_methods=generate_noise_methods,
                initiate_sym_calc=initiate_sym_calc,
                truncated_dim=truncated_dim,
            )

        elif symbolic_hamiltonian:
            if use_dynamic_flux_grouping or generate_noise_methods:
                raise Exception(
                    "Circuit instance initialized using symbolic Hamiltonian cannot be configured with closure_branches, use_dynamic_flux_grouping, transformation_matrix or generate_noise_methods."
                )
            self._from_symbolic_hamiltonian(
                symbolic_hamiltonian=symbolic_hamiltonian,
                symbolic_param_dict=symbolic_param_dict,
                initiate_sym_calc=initiate_sym_calc,
                truncated_dim=truncated_dim,
                ext_basis=ext_basis,
            )

    def _from_symbolic_hamiltonian(
        self,
        symbolic_hamiltonian: sm.Expr,
        symbolic_param_dict: dict[str, float],
        initiate_sym_calc: bool,
        truncated_dim: int,
        ext_basis: str,
    ):
        """Initialize the :class:`Circuit` instance from a symbolic Hamiltonian.

        Parameters
        ----------
        symbolic_hamiltonian:
            the symbolic Hamiltonian expression that defines the circuit.
        symbolic_param_dict:
            mapping from symbolic-parameter names to numerical default values.
            Entries whose name contains ``"ng"`` or ``"Φ"`` are skipped (these are
            treated as offset charges or external fluxes, respectively).
        initiate_sym_calc:
            if ``True``, run :meth:`configure` after the attributes are set.
        truncated_dim:
            truncated dimension for the resulting Circuit instance.
        ext_basis:
            ``"discretized"`` or ``"harmonic"``; the basis used for extended
            variables.
        """
        self.hamiltonian_symbolic = symbolic_hamiltonian

        self.symbolic_params = {}
        for param_str in symbolic_param_dict:
            if "ng" in param_str or "Φ" in param_str:
                continue
            self.symbolic_params[sm.symbols(param_str)] = symbolic_param_dict[param_str]

        sm.init_printing(pretty_print=False, order="none")
        self.is_child = False

        self.ext_basis = ext_basis
        self.truncated_dim: int = truncated_dim
        self.system_hierarchy: list = []
        self.subsystem_trunc_dims: list = []
        self.operators_by_name = None

        self.discretized_phi_range: dict[int, tuple[float, float]] = {}
        self.cutoff_names: list[str] = []

        # setting default grids for plotting
        self._default_grid_phi: discretization.Grid1d = discretization.Grid1d(
            -6 * np.pi, 6 * np.pi, 200
        )

        self.type_of_matrices: str = (
            "sparse"  # type of matrices used to construct the operators
        )

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []
        self._out_of_sync = False  # for use with CentralDispatch
        if initiate_sym_calc:
            self.configure()
        self._frozen = True
        dispatch.CENTRAL_DISPATCH.register("CIRCUIT_UPDATE", self)

    def from_yaml(
        self,
        input_string: str,
        from_file: bool = True,
        basis_completion: str = "heuristic",
        ext_basis: str = "discretized",
        use_dynamic_flux_grouping: bool = False,
        generate_noise_methods: bool = False,
        initiate_sym_calc: bool = True,
        truncated_dim: int = 10,
    ):
        """Wrapper to :meth:`Circuit.__init__` to create a class instance.

        This method is deprecated and will not be supported in future releases.

        Parameters
        ----------
        input_string:
            String describing the number of nodes and branches connecting then along
            with their parameters
        from_file:
            set to True by default, when a file name should be provided to
            `input_string`, else the circuit graph description in YAML should be
            provided as a string.
        basis_completion:
            either "heuristic" or "canonical", defines the matrix used for completing
            the transformation matrix. Sometimes used to change the variable
            transformation to result in a simpler symbolic Hamiltonian, by default
            "heuristic"
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use discretized
            phi or harmonic oscillator basis for extended variables,
            by default "discretized"
        initiate_sym_calc:
            attribute to initiate Circuit instance, by default ``True``
        truncated_dim:
            truncated dimension if the user wants to use this circuit instance in
            HilbertSpace, by default ``None``
        use_dynamic_flux_grouping:
            set to False by default. Indicates if the flux allocation is done by
            assuming that flux is time dependent. When set to True, it disables the
            option to change the closure branches.
        generate_noise_methods:
            when ``True``, generate per-channel noise methods on the instance after
            initialization; by default ``False``.
        """
        if basis_completion not in ["heuristic", "canonical"]:
            raise Exception(
                "Invalid choice for basis_completion: must be 'heuristic' or "
                "'canonical'."
            )

        symbolic_circuit = SymbolicCircuit.from_yaml(
            input_string,
            from_file=from_file,
            basis_completion=basis_completion,
            initiate_sym_calc=True,
            use_dynamic_flux_grouping=use_dynamic_flux_grouping,
        )
        sm.init_printing(pretty_print=False, order="none")
        self.is_child = False
        self.symbolic_circuit: SymbolicCircuit = symbolic_circuit

        self.ext_basis = ext_basis
        self.hierarchical_diagonalization = False
        self.truncated_dim = truncated_dim
        self.system_hierarchy = []
        self.subsystem_trunc_dims = []
        self.operators_by_name = None

        self.discretized_phi_range = {}
        self.cutoff_names = []

        # setting default grids for plotting
        self._default_grid_phi = discretization.Grid1d(-6 * np.pi, 6 * np.pi, 200)

        self.type_of_matrices = (
            "sparse"  # type of matrices used to construct the operators
        )
        self._import_from_symbolic_circuit()

        # needs to be included to make sure that plot_evals_vs_paramvals works
        self._init_params = []
        self._out_of_sync = False  # for use with CentralDispatch

        if initiate_sym_calc:
            self.configure()
        if generate_noise_methods:
            self.generate_noise_methods()
        self._frozen = True
        dispatch.CENTRAL_DISPATCH.register("CIRCUIT_UPDATE", self)

    SYMBOLIC_CIRCUIT_ATTRIBUTES = (
        "branches",
        "closure_branches",
        "external_fluxes",
        "ground_node",
        "hamiltonian_symbolic",
        "input_string",
        "is_grounded",
        "is_purely_harmonic",
        "lagrangian_node_vars",
        "lagrangian_symbolic",
        "nodes",
        "offset_charges",
        "free_charges",
        "potential_symbolic",
        "potential_node_vars",
        "symbolic_params",
        "transformation_matrix",
        "var_categories",
    )

    def _import_from_symbolic_circuit(self) -> None:
        """Copy ``SYMBOLIC_CIRCUIT_ATTRIBUTES`` from ``self.symbolic_circuit``."""
        for attr in self.SYMBOLIC_CIRCUIT_ATTRIBUTES:
            setattr(self, attr, getattr(self.symbolic_circuit, attr))

    def _find_branch(
        self, node_id_1: int, node_id_2: int, branch_type: str, branch_params: dict
    ):
        """Locate a branch in :attr:`symbolic_circuit` matching the given description.

        Parameters
        ----------
        node_id_1:
            integer index of the first endpoint node of the sought branch.
        node_id_2:
            integer index of the second endpoint node of the sought branch.
        branch_type:
            branch type identifier (e.g. ``"L"``, ``"C"``, ``"JJ"``).
        branch_params:
            dictionary of branch parameters; symbolic parameters are compared by
            their symbol name.

        Returns
        -------
        The matching :class:`Branch` instance, or ``None`` if no branch matches.
        """
        for branch in self.symbolic_circuit.branches:
            branch_node_ids = [node.index for node in branch.nodes]
            branch_params_circ = branch.parameters.copy()
            for param in branch_params_circ:
                val = branch_params_circ[param]
                if isinstance(val, sm.Symbol):
                    branch_params_circ[param] = val.name
            if node_id_1 not in branch_node_ids or node_id_2 not in branch_node_ids:
                continue
            if branch.type != branch_type:
                continue
            if branch_params != branch_params_circ:
                continue
            return branch
        return None

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return an empty dict of default parameters for :class:`Circuit`."""
        return {}

    def _clear_unnecessary_attribs(self):
        """Clear all the attributes which are not part of the circuit description."""
        necessary_attrib_names = (
            self.cutoff_names
            + [flux_symbol.name for flux_symbol in self.external_fluxes]
            + [
                charge_symbol.name
                for charge_symbol in self.offset_charges + self.free_charges
            ]
            + ["cutoff_names"]
        )
        attrib_keys = list(self.__dict__.keys()).copy()
        for attrib in attrib_keys:
            if attrib[1:] not in necessary_attrib_names:
                if (
                    "cutoff_n_" in attrib
                    or "Φ" in attrib
                    or "cutoff_ext_" in attrib
                    or attrib[1:3] == "ng"
                ):
                    delattr(self, attrib)

    def configure(
        self,
        transformation_matrix: ndarray | None = None,
        system_hierarchy: list | None = None,
        subsystem_trunc_dims: list | None = None,
        closure_branches: list[Branch | dict[Branch, float]] | None = None,
        ext_basis: str | list[str] | None = None,
        use_dynamic_flux_grouping: bool | None = None,
        generate_noise_methods: bool = False,
        subsys_dict: dict[str, Any] | None = None,
    ):
        """Re-initialize a circuit instance to update hierarchical settings.

        Re-initializes the instance to update hierarchical diagonalization
        parameters, closure branches, or the variable transformation used to
        describe the circuit.

        Parameters
        ----------
        transformation_matrix:
            A user defined variable transformation which has the dimensions of the
            number nodes (not counting the ground node), by default ``None``
        system_hierarchy:
            A list of lists which is provided by the user to define subsystems,
            by default ``None``
        subsystem_trunc_dims:
            dict object which can be generated for a specific system_hierarchy
            using the method ``truncation_template``, by default ``None``
        closure_branches:
            Each element of the list corresponds to one external flux variable.
            If the element is a branch the external flux will be associated with
            that branch. If the element is a dictionary, the external flux
            variable will be distributed across the branches according to the
            dictionary with the factor given as a key value.
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use
            discretized phi or harmonic oscillator basis for extended variables,
            by default ``None``
        use_dynamic_flux_grouping:
            set to False by default. Indicates if the flux allocation is done by
            assuming that flux is time dependent. When set to True, it disables
            the option to change the closure branches.
        generate_noise_methods:
            set to False by default. Indicates if the noise methods should be
            generated for the circuit instance.
        subsys_dict:
            User provided dictionary with two keys ``"systems_sym"`` and
            ``"interaction_sym"`` defining the symbolic Hamiltonians and
            interactions for the subsystems. By default set to None, and is
            internally generated.

        Raises
        ------
        ConfigureError
            When configuration fails — for example when ``system_hierarchy``
            is set without ``subsystem_trunc_dims``, or when
            ``closure_branches`` is set on an instance initialized with
            ``use_dynamic_flux_grouping=True``. The triggering exception is
            preserved as ``__cause__``; prior configuration is restored
            before the error is raised.

        Notes
        -----
        Mutates instance state by reassigning attributes such as
        ``system_hierarchy``, ``subsystem_trunc_dims``, ``transformation_matrix``,
        and ``closure_branches``.
        """

        old_system_hierarchy = self.system_hierarchy
        old_subsystem_trunc_dims = self.subsystem_trunc_dims
        old_ext_basis = self.ext_basis
        old_subsys_dict = subsys_dict
        if hasattr(self, "symbolic_circuit"):
            old_transformation_matrix = self.transformation_matrix
            old_use_dynamic_flux_grouping = (
                self.symbolic_circuit.use_dynamic_flux_grouping
            )
            old_closure_branches = (
                self.closure_branches if not old_use_dynamic_flux_grouping else None
            )
            old_generate_noise_methods = hasattr(self, "_noise_methods_generated")
        try:
            if hasattr(self, "symbolic_circuit"):
                self._configure(
                    transformation_matrix=transformation_matrix,
                    system_hierarchy=system_hierarchy,
                    subsystem_trunc_dims=subsystem_trunc_dims,
                    closure_branches=closure_branches,
                    ext_basis=ext_basis,
                    use_dynamic_flux_grouping=use_dynamic_flux_grouping,
                    generate_noise_methods=generate_noise_methods,
                    subsys_dict=subsys_dict,
                )
            else:
                if (
                    closure_branches is not None
                    or use_dynamic_flux_grouping
                    or generate_noise_methods
                ):
                    raise Exception(
                        "Circuit instance initialized using symbolic Hamiltonian cannot be configured with closure_branches, use_dynamic_flux_grouping, transformation_matrix or generate_noise_methods."
                    )
                self._configure_sym_hamiltonian(
                    system_hierarchy=system_hierarchy,
                    subsystem_trunc_dims=subsystem_trunc_dims,
                    ext_basis=ext_basis,
                    subsys_dict=subsys_dict,
                )
        except Exception as exc:
            # resetting the necessary attributes
            self.system_hierarchy = old_system_hierarchy
            self.subsystem_trunc_dims = old_subsystem_trunc_dims
            if hasattr(self, "symbolic_circuit"):
                self.transformation_matrix = old_transformation_matrix
                self.closure_branches = old_closure_branches  # type: ignore[assignment]
            # Calling configure
            if hasattr(self, "symbolic_circuit"):
                self._configure(
                    transformation_matrix=old_transformation_matrix,
                    system_hierarchy=old_system_hierarchy,
                    subsystem_trunc_dims=old_subsystem_trunc_dims,
                    closure_branches=old_closure_branches,
                    ext_basis=old_ext_basis,
                    use_dynamic_flux_grouping=old_use_dynamic_flux_grouping,
                    generate_noise_methods=old_generate_noise_methods,
                    subsys_dict=old_subsys_dict,
                )
            else:
                self._configure_sym_hamiltonian(
                    system_hierarchy=old_system_hierarchy,
                    subsystem_trunc_dims=old_subsystem_trunc_dims,
                    ext_basis=old_ext_basis,
                    subsys_dict=old_subsys_dict,
                )
            raise ConfigureError(
                "Configure failed; previous configuration restored."
            ) from exc

    def _read_symbolic_hamiltonian(
        self, symbolic_hamiltonian: sm.Expr
    ) -> tuple[list[sm.Expr], list[sm.Expr], list[sm.Expr], dict[str, list[int]]]:
        """Extract flux, charge, and variable-category metadata.

        Extracts the metadata from the supplied symbolic Hamiltonian.

        Parameters
        ----------
        symbolic_hamiltonian:
            symbolic expression for the Hamiltonian whose free symbols are inspected
            to identify external fluxes, offset charges, free charges, and variable
            categories

        Returns
        -------
        Tuple ``(external_fluxes, offset_charges, free_charges, var_categories)``
        where the first three entries are lists of sympy symbols and
        ``var_categories`` is a dictionary mapping the keys ``"periodic"``,
        ``"extended"``, ``"free"``, ``"frozen"`` to sorted lists of variable
        indices.
        """
        free_symbols = symbolic_hamiltonian.free_symbols
        external_fluxes = []
        offset_charges = []
        free_charges = []
        var_categories: dict[str, list[int]] = {
            "periodic": [],
            "extended": [],
            "free": [],
            "frozen": [],
        }
        for var_sym in free_symbols:
            if re.match(r"^ng\d+$", var_sym.name):
                offset_charges.append(var_sym)
            elif re.match(r"^Qf\d+$", var_sym.name):
                free_charges.append(var_sym)
            elif re.match(r"^Φ\d+$", var_sym.name):
                external_fluxes.append(var_sym)
            elif re.match(r"^n\d+$", var_sym.name):
                var_index = get_trailing_number(var_sym.name)
                var_categories["periodic"].append(var_index)
            elif re.match(r"^Q\d+$", var_sym.name):
                var_index = get_trailing_number(var_sym.name)
                var_categories["extended"].append(var_index)
        var_categories = {
            category: sorted(var_categories[category]) for category in var_categories
        }
        return external_fluxes, offset_charges, free_charges, var_categories

    def _configure_sym_hamiltonian(
        self,
        system_hierarchy: list | None = None,
        subsystem_trunc_dims: list | None = None,
        subsys_dict: dict[str, Any] | None = None,
        ext_basis: str | list[str] | None = None,
    ):
        """Re-initialize a symbolic-Hamiltonian circuit's hierarchical settings.

        Re-initializes a circuit instance (built from a symbolic Hamiltonian) to
        update hierarchical diagonalization parameters or basis choices.

        Parameters
        ----------
        system_hierarchy:
            A list of lists which is provided by the user to define subsystems,
            by default ``None``
        subsystem_trunc_dims:
            dict object which can be generated for a specific system_hierarchy
            using the method ``truncation_template``, by default ``None``
        subsys_dict:
            User provided dictionary with two keys ``"systems_sym"`` and
            ``"interaction_sym"`` defining the symbolic Hamiltonians and
            interactions for the subsystems. By default set to None, and is
            internally generated.
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use
            discretized phi or harmonic oscillator basis for extended variables,
            by default ``None``

        Raises
        ------
        Exception
            when system_hierarchy is set and subsystem_trunc_dims is not set.

        Notes
        -----
        Mutates instance state by reassigning many attributes (e.g.
        ``ext_basis``, ``hierarchical_diagonalization``, ``cutoff_names``,
        ``operators_by_name``).
        """
        self._frozen = False
        system_hierarchy = system_hierarchy or self.system_hierarchy
        subsystem_trunc_dims = subsystem_trunc_dims or self.subsystem_trunc_dims
        self.ext_basis = ext_basis or self.ext_basis

        self.hierarchical_diagonalization = (
            True if system_hierarchy is not None else False
        )

        self.is_purely_harmonic = self._is_expression_purely_harmonic(
            self.hamiltonian_symbolic
        )

        (
            self.external_fluxes,
            self.offset_charges,
            self.free_charges,
            self.var_categories,
        ) = self._read_symbolic_hamiltonian(self.hamiltonian_symbolic)

        # initiating the class properties
        self.cutoff_names = []
        for var_type in self.var_categories.keys():
            if var_type == "periodic":
                for idx, var_index in enumerate(self.var_categories["periodic"]):
                    if not hasattr(self, f"_cutoff_n_{var_index}"):
                        self._make_property(
                            f"cutoff_n_{var_index}", 5, "update_cutoffs"
                        )
                    self.cutoff_names.append(f"cutoff_n_{var_index}")
            if var_type == "extended":
                for idx, var_index in enumerate(self.var_categories["extended"]):
                    if not hasattr(self, f"_cutoff_ext_{var_index}"):
                        self._make_property(
                            f"cutoff_ext_{var_index}", 30, "update_cutoffs"
                        )
                    self.cutoff_names.append(f"cutoff_ext_{var_index}")

        self.dynamic_var_indices = (
            self.var_categories["periodic"] + self.var_categories["extended"]
        )

        # default values for the parameters
        for idx, param in enumerate(self.symbolic_params):
            if not hasattr(self, param.name):
                self._make_property(
                    param.name, self.symbolic_params[param], "update_param_vars"
                )
        # setting the ranges for flux ranges used for discrete phi vars
        for var_index in self.var_categories["extended"]:
            if var_index not in self.discretized_phi_range:
                self.discretized_phi_range[var_index] = (-6 * np.pi, 6 * np.pi)
        # external flux vars
        for flux in self.external_fluxes:
            # setting the default to zero external flux
            if not hasattr(self, flux.name):
                self._make_property(flux.name, 0.0, "update_external_flux_or_charge")
        # offset charges
        for charge_var in self.offset_charges + self.free_charges:
            # default to zero offset charge
            if not hasattr(self, charge_var.name):
                self._make_property(
                    charge_var.name, 0.0, "update_external_flux_or_charge"
                )

        self.potential_symbolic = self._generate_sym_potential()

        # changing the matrix type if necessary
        if len(flatten_list(self.var_categories.values())) == 1:
            self.type_of_matrices = "dense"

        if system_hierarchy is not None:
            self.hierarchical_diagonalization = (
                system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
            )

        if not self.hierarchical_diagonalization:
            if self.is_purely_harmonic and not ext_basis:
                if self.ext_basis != "harmonic":
                    warnings.warn(
                        "Purely harmonic circuits need ext_basis to be set to 'harmonic'"
                    )
                    self.ext_basis = "harmonic"
            self.ext_basis = ext_basis or self.ext_basis
            self._generate_hamiltonian_sym_for_numerics()
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                # using the default methods
                self.evals_method = None
                self.evals_method_options = None
                self._annihilation_operator_in_eigenbasis = None
                self._diagonalize_purely_harmonic_hamiltonian()
            self._set_vars()  # setting the attribute vars to store operator symbols
            self.operators_by_name = self._set_operators()
        else:
            # list for updating necessary subsystems when calling build hilbertspace
            self.operators_by_name = None
            self.system_hierarchy = system_hierarchy
            if subsystem_trunc_dims is None:
                raise Exception(
                    "The truncated dimensions attribute for hierarchical "
                    "diagonalization is not set."
                )

            self.subsystem_trunc_dims = subsystem_trunc_dims
            if ext_basis:
                self.ext_basis = ext_basis
            self._generate_hamiltonian_sym_for_numerics()
            self._generate_subsystems(subsys_dict=subsys_dict)
            self.ext_basis = (
                self.get_ext_basis()
            )  # update the ext_basis after generating subsystems
            self._set_vars()  # setting the attribute vars to store operator symbols
            self._check_truncation_indices()
            self.operators_by_name = self._set_operators()
            self.affected_subsystem_indices = list(range(len(self.subsystems)))
            self._update_interactions()

        # clear unnecessary attribs
        self._clear_unnecessary_attribs()
        self._frozen = True
        self.update()

    def _configure(
        self,
        transformation_matrix: ndarray | None = None,
        system_hierarchy: list | None = None,
        subsystem_trunc_dims: list | None = None,
        closure_branches: list[Branch | dict[Branch, float]] | None = None,
        ext_basis: str | list[str] | None = None,
        use_dynamic_flux_grouping: bool | None = None,
        subsys_dict: dict[str, Any] | None = None,
        generate_noise_methods: bool = False,
    ):
        """Re-initialize a symbolic-circuit instance's hierarchical settings.

        Re-initializes a circuit instance (built from a symbolic circuit) to
        update hierarchical diagonalization parameters, closure branches, or the
        variable transformation used to describe the circuit.

        Parameters
        ----------
        transformation_matrix:
            A user defined variable transformation which has the dimensions of
            the number nodes (not counting the ground node), by default ``None``
        system_hierarchy:
            A list of lists which is provided by the user to define subsystems,
            by default ``None``
        subsystem_trunc_dims:
            dict object which can be generated for a specific system_hierarchy
            using the method ``truncation_template``, by default ``None``
        closure_branches:
            Each element of the list corresponds to one external flux variable.
            If the element is a branch the external flux will be associated with
            that branch. If the element is a dictionary, the external flux
            variable will be distributed across the branches according to the
            dictionary with the factor given as a key value.
        ext_basis:
            can be "discretized" or "harmonic" which chooses whether to use
            discretized, or can be a list of lists of lists, when hierarchical
            diagonalization is used.
        use_dynamic_flux_grouping:
            set to False by default. Indicates if the flux allocation is done by
            assuming that flux is time dependent. When set to True, it disables
            the option to change the closure branches.
        subsys_dict:
            User provided dictionary with two keys ``"systems_sym"`` and
            ``"interaction_sym"`` defining the symbolic Hamiltonians and
            interactions for the subsystems. By default set to None, and is
            internally generated.
        generate_noise_methods:
            set to False by default. Indicates if the noise methods should be
            generated for the circuit instance.

        Raises
        ------
        Exception
            when system_hierarchy is set and subsystem_trunc_dims is not set.

        Notes
        -----
        Mutates instance state by reassigning many attributes (e.g.
        ``hierarchical_diagonalization``, ``hamiltonian_symbolic``,
        ``ext_basis``, ``operators_by_name``).
        """
        self._frozen = False

        # reinitiate the symbolic circuit when the transformation matrix and closure branches are provided
        if (
            transformation_matrix is not None
            or closure_branches is not None
            or use_dynamic_flux_grouping is not None
        ):
            self.symbolic_circuit.configure(
                transformation_matrix=transformation_matrix,
                closure_branches=closure_branches,  # type: ignore[arg-type]
                use_dynamic_flux_grouping=use_dynamic_flux_grouping,
            )

        system_hierarchy = system_hierarchy or self.system_hierarchy
        subsystem_trunc_dims = subsystem_trunc_dims or self.subsystem_trunc_dims
        closure_branches = closure_branches or self.closure_branches

        if transformation_matrix is None:
            if hasattr(
                self, "transformation_matrix"
            ):  # checking to see if configure is being called outside of init
                transformation_matrix = self.transformation_matrix

        self.hierarchical_diagonalization = (
            True if system_hierarchy is not None else False
        )

        self._import_from_symbolic_circuit()

        # initiating the class properties
        self.cutoff_names = []
        for var_type in self.var_categories.keys():
            if var_type == "periodic":
                for idx, var_index in enumerate(self.var_categories["periodic"]):
                    if not hasattr(self, f"_cutoff_n_{var_index}"):
                        self._make_property(
                            f"cutoff_n_{var_index}", 5, "update_cutoffs"
                        )
                    self.cutoff_names.append(f"cutoff_n_{var_index}")
            if var_type == "extended":
                for idx, var_index in enumerate(self.var_categories["extended"]):
                    if not hasattr(self, f"_cutoff_ext_{var_index}"):
                        self._make_property(
                            f"cutoff_ext_{var_index}", 30, "update_cutoffs"
                        )
                    self.cutoff_names.append(f"cutoff_ext_{var_index}")

        self.dynamic_var_indices = (
            self.var_categories["periodic"] + self.var_categories["extended"]
        )

        # default values for the parameters
        for idx, param in enumerate(self.symbolic_params):
            if not hasattr(self, param.name):
                self._make_property(
                    param.name, self.symbolic_params[param], "update_param_vars"
                )
        # setting the ranges for flux ranges used for discrete phi vars
        for var_index in self.var_categories["extended"]:
            if var_index not in self.discretized_phi_range:
                self.discretized_phi_range[var_index] = (-6 * np.pi, 6 * np.pi)
        # external flux vars
        for flux in self.external_fluxes:
            # setting the default to zero external flux
            if not hasattr(self, flux.name):
                self._make_property(flux.name, 0.0, "update_external_flux_or_charge")
        # offset and free charges
        for charge_var in self.offset_charges + self.free_charges:
            # default to zero offset charge
            if not hasattr(self, charge_var.name):
                self._make_property(
                    charge_var.name, 0.0, "update_external_flux_or_charge"
                )

        # changing the matrix type if necessary
        if (
            len((self.var_categories["extended"] + self.var_categories["periodic"]))
            == 1
        ):
            self.type_of_matrices = "dense"

        self.hamiltonian_symbolic = self.symbolic_circuit.hamiltonian_symbolic
        # if the flux is static, remove the linear terms from the potential
        if not self.symbolic_circuit.use_dynamic_flux_grouping:
            self.hamiltonian_symbolic = self._shift_harmonic_oscillator_potential(
                self.hamiltonian_symbolic
            )
            self.potential_symbolic = self._shift_harmonic_oscillator_potential(
                self.potential_symbolic.expand()
            )

        if system_hierarchy is not None:
            self.hierarchical_diagonalization = (
                system_hierarchy != [] and number_of_lists_in_list(system_hierarchy) > 0
            )

        if not self.hierarchical_diagonalization:
            if self.is_purely_harmonic and not ext_basis:
                self.normal_mode_freqs = self.symbolic_circuit.normal_mode_freqs
                if self.ext_basis != "harmonic":
                    warnings.warn(
                        "Purely harmonic circuits need ext_basis to be set to 'harmonic'"
                    )
                    self.ext_basis = "harmonic"
            self._generate_hamiltonian_sym_for_numerics()
            self.ext_basis = ext_basis or self.ext_basis
            if self.is_purely_harmonic and self.ext_basis == "harmonic":
                # using the default methods
                self.evals_method = None
                self.evals_method_options = None
                self._annihilation_operator_in_eigenbasis = None
                self._diagonalize_purely_harmonic_hamiltonian()
        else:
            # list for updating necessary subsystems when calling build hilbertspace
            self.operators_by_name = None
            self.system_hierarchy = system_hierarchy
            if subsystem_trunc_dims is None:
                raise Exception(
                    "The truncated dimensions attribute for hierarchical "
                    "diagonalization is not set."
                )

            self.subsystem_trunc_dims = subsystem_trunc_dims
            self._generate_hamiltonian_sym_for_numerics()
            self.ext_basis = ext_basis or self.ext_basis
            self._generate_subsystems(subsys_dict=subsys_dict)
            self.ext_basis = self.get_ext_basis()
            self._update_interactions()
            self._check_truncation_indices()
            self.affected_subsystem_indices = list(range(len(self.subsystems)))

        self._set_vars()  # setting the attribute vars to store operator symbols
        self.operators_by_name = self._set_operators()
        # clear unnecessary attribs
        self._clear_unnecessary_attribs()
        if generate_noise_methods:
            self.generate_noise_methods()
        self._frozen = True
        self.update()

    def supported_noise_channels(self) -> list[str]:  # type: ignore[override]
        """Return a list of supported noise channels."""
        if not hasattr(self, "_noise_methods_generated"):
            raise Exception(
                "Noise methods are not generated, please use configure() with generate_noise_methods=True to generate them."
            )
        return [
            method_name
            for method_name in self.__dict__
            if "tphi_1_over_f" in method_name or "t1_" in method_name
        ]

    def effective_noise_channels(self):
        """Return the names of effective (combined) noise-channel methods.

        Returns
        -------
        List of supported noise-channel method names that correspond to combined
        (effective) noise channels rather than per-branch contributions.
        """
        if not hasattr(self, "_noise_methods_generated"):
            raise Exception(
                "Noise methods are not generated, please use configure() with generate_noise_methods=True to generate them."
            )
        return [
            method_name
            for method_name in self.supported_noise_channels()
            if not is_string_float(method_name[-1])
        ]

    def variable_transformation(self, new_vars_to_node_vars: bool = True) -> None:
        """Print the variable transformation used in this circuit.

        Parameters
        ----------
        new_vars_to_node_vars:
            if ``True`` (default), display the new variables expressed in terms
            of node variables; if ``False``, display node variables in terms of
            the new variables.
        """
        trans_mat = self.transformation_matrix
        if new_vars_to_node_vars:
            trans_mat = np.linalg.inv(trans_mat)
        theta_vars = [
            sm.symbols(f"θ{index}")
            for index in range(
                1, len(self.symbolic_circuit.nodes) - self.is_grounded + 1
            )
        ]
        node_vars = [
            sm.symbols(f"φ{index}")
            for index in range(
                1, len(self.symbolic_circuit.nodes) - self.is_grounded + 1
            )
        ]
        var_eqns = []
        for idx, node_var in enumerate(node_vars):
            if not new_vars_to_node_vars:
                var_eqns.append(
                    sm.Eq(node_vars[idx], np.sum(trans_mat[idx, :] * theta_vars))
                )
            else:
                var_eqns.append(
                    sm.Eq(theta_vars[idx], np.sum(trans_mat[idx, :] * node_vars))
                )
        if _HAS_IPYTHON:
            self.print_expr_in_latex(var_eqns)
        else:
            print(var_eqns)

    def sym_lagrangian(
        self,
        vars_type: str = "node",
        print_latex: bool = False,
        return_expr: bool = False,
    ) -> sm.Expr | None:
        """Return or print a user-readable symbolic Lagrangian.

        Provides a Lagrangian for the current instance, in either node-variable
        or new-variable form.

        Parameters
        ----------
        vars_type:
            "node" or "new", fixes the kind of lagrangian requested, by default "node"
        print_latex:
            if set to True, the expression is additionally printed as LaTeX code
        return_expr:
            if set to True, all printing is suppressed and the function will silently
            return the sympy expression
        """
        if vars_type == "node":
            lagrangian = self.lagrangian_node_vars  # type: ignore[attr-defined]
            # replace v\theta with \theta_dot
            for var_index in range(
                1, 1 + len(self.symbolic_circuit.nodes) - self.is_grounded
            ):
                lagrangian = lagrangian.replace(
                    sm.symbols(f"vφ{var_index}"),
                    sm.symbols("\\dot{φ_" + str(var_index) + "}"),
                )
            # break down the lagrangian into kinetic and potential part, and rejoin
            # with evaluate=False to force the kinetic terms together and appear first
            sym_lagrangian_PE_node_vars = self.potential_node_vars  # type: ignore[attr-defined]
            for external_flux in self.external_fluxes:
                sym_lagrangian_PE_node_vars = sym_lagrangian_PE_node_vars.replace(
                    external_flux,
                    sm.symbols(
                        "(2π"
                        + "Φ_{"
                        + str(get_trailing_number(str(external_flux)))
                        + "})"
                    ),
                )
            lagrangian = sm.Add(
                (self._make_expr_human_readable(lagrangian + self.potential_node_vars)),  # type: ignore[attr-defined]
                (self._make_expr_human_readable(-sym_lagrangian_PE_node_vars)),
                evaluate=False,
            )

        elif vars_type == "new":
            lagrangian = self.lagrangian_symbolic  # type: ignore[attr-defined]
            # replace v\theta with \theta_dot
            for var_index in self.dynamic_var_indices:
                lagrangian = lagrangian.replace(
                    sm.symbols(f"vθ{var_index}"),
                    sm.symbols("\\dot{θ_" + str(var_index) + "}"),
                )
            # break down the lagrangian into kinetic and potential part, and rejoin
            # with evaluate=False to force the kinetic terms together and appear first
            sym_lagrangian_PE_new = self.potential_symbolic.expand()
            for external_flux in self.external_fluxes:
                sym_lagrangian_PE_new = sym_lagrangian_PE_new.replace(
                    external_flux,
                    sm.symbols(
                        "(2π"
                        + "Φ_{"
                        + str(get_trailing_number(str(external_flux)))
                        + "})"
                    ),
                )
            lagrangian = sm.Add(
                (
                    self._make_expr_human_readable(
                        lagrangian + self.potential_symbolic.expand()
                    )
                ),
                (self._make_expr_human_readable(-sym_lagrangian_PE_new)),
                evaluate=False,
            )
        if return_expr:
            return lagrangian
        if print_latex:
            print(latex(lagrangian))
        if _HAS_IPYTHON:
            self.print_expr_in_latex(lagrangian)
        else:
            print(lagrangian)
        return None

    def sym_external_fluxes(self) -> dict[sm.Expr, tuple["Branch", list["Branch"]]]:
        """Return a dictionary of external fluxes with associated branches/loops.

        Returns human-readable external fluxes mapped to their associated branches
        and loops (represented as lists of branches) for the current instance.

        Returns
        -------
        A dictionary of Human readable external fluxes with their associated
        branches and loops
        """
        if not self.closure_branches:
            return {}
        list_closure_branches = {
            element: idx
            for idx, element in enumerate(self.closure_branches)
            if isinstance(element, Branch)
        }
        return {
            self._make_expr_human_readable(
                self.external_fluxes[list_closure_branches[branch]]
            ): (
                branch,
                self.symbolic_circuit._find_loop(branch),
            )
            for branch in list_closure_branches
        }

    def oscillator_list(self, osc_index_list: list[int]):
        """Mark subsystems as single-mode oscillators (hierarchical diag only).

        If hierarchical diagonalization is used, the attributes
        ``_osc_subsys_list`` and ``osc_subsys_list`` of the :attr:`hilbert_space`
        attribute of the Circuit instance will be assigned accordingly, enabling
        the correct identification of harmonic modes for the dispersive regime
        analysis in :class:`ParameterSweep`.

        Parameters
        ----------
        osc_index_list:
            a list of indices of subsystems that are single-mode harmonic oscillators
        """
        # identify if each nominated subsystem indeed have a single harmonic oscillator
        osc_subsys_list = []
        for subsystem_index in osc_index_list:
            subsystem = self.subsystems[subsystem_index]
            if not subsystem.is_purely_harmonic:
                raise Exception(
                    f"the subsystem {subsystem_index} is not purely harmonic"
                )
            elif len(subsystem.var_categories["extended"]) != 1:
                raise Exception(
                    "the subsystem has more than one harmonic oscillator mode"
                )
            else:
                osc_subsys_list.append(subsystem)
        self.hilbert_space._osc_subsys_list = osc_subsys_list

    def qubit_list(self, qbt_index_list: list[int]):
        """Mark subsystems as qubits (hierarchical diagonalization only).

        The attribute ``_qbt_subsys_list`` of the :attr:`hilbert_space`
        attribute of the Circuit instance will be assigned accordingly, enabling
        the correct identification of qubit subsystems for the dispersive regime
        analysis in :class:`ParameterSweep`.

        Parameters
        ----------
        qbt_index_list:
            a list of indices of subsystems that are single-mode harmonic oscillators
        """
        # identify if each naminated subsystem indeed have a single harmonic oscillator
        qbt_subsys_list = []
        for subsystem_index in qbt_index_list:
            subsystem = self.subsystems[subsystem_index]
            qbt_subsys_list.append(subsystem)
        self.hilbert_space._qbt_subsys_list = qbt_subsys_list
