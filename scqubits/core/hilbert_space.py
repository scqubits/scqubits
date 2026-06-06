# hilbert_space.py
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

import functools
import importlib
import re
import warnings

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import qutip as qt

from numpy import ndarray
from scipy.sparse import csc_matrix, dia_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.diag as diag
import scqubits.core.oscillator as osc
import scqubits.core.spec_lookup as spec_lookup
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_qutip
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.ui.hspace_widget
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.namedslots_array import NamedSlotsNdarray, Parameters
from scqubits.core.storage import SpectrumData
from scqubits.io_utils.fileio_qutip import QutipEigenstates

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData

from scqubits.core.qubit_base import QubitBaseClass
from scqubits.utils.typedefs import OscillatorList, QuantumSys, QubitList


def has_duplicate_id_str(subsystem_list: list[QuantumSys]) -> bool:
    """Return whether any two subsystems share the same ``id_str``.

    Parameters
    ----------
    subsystem_list:
        list of subsystems whose ``id_str`` attributes are compared
    """
    id_str_list = [obj.id_str for obj in subsystem_list]
    id_str_set = set(obj.id_str for obj in subsystem_list)
    return len(id_str_set) != len(id_str_list)


class InteractionTerm(dispatch.DispatchClient, serializers.Serializable):
    """Specify a term in the interaction Hamiltonian of a composite Hilbert space.

    Constructs the Hamiltonian in :class:`qutip.Qobj` format. The interaction
    term takes one of two forms:

    1. ``V = g A B C ...``, where ``A, B, C, ...`` are Hermitian operators on
       subsystems in ``subsystem_list``;
    2. ``V = g A B C ... + h.c.``, where ``A, B, C, ...`` may be non-Hermitian.

    Parameters
    ----------
    g_strength:
        coefficient parametrizing the interaction strength.
    operator_list:
        list of ``(subsys_index, operator)`` tuples.
    add_hc:
        if ``True``, the interaction Hamiltonian is of type 2 and the Hermitian
        conjugate is added.
    """

    g_strength = descriptors.WatchedProperty(complex, "INTERACTIONTERM_UPDATE")

    operator_list = descriptors.WatchedProperty(
        list[tuple[int, ndarray | csc_matrix | Callable]], "INTERACTIONTERM_UPDATE"
    )  # Each item in the operator_list is a tuple (subsys_index, operator)
    add_hc = descriptors.WatchedProperty(bool, "INTERACTIONTERM_UPDATE")

    def __init__(
        self,
        g_strength: float | complex,
        operator_list: list[tuple[int, ndarray | csc_matrix | Callable]],
        add_hc: bool = False,
    ) -> None:
        self.g_strength = g_strength
        self.operator_list = operator_list
        self.add_hc = add_hc

    def __repr__(self) -> str:
        """Return an ``eval``-friendly developer-facing representation."""
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        """Return a human-readable summary of this interaction term."""
        indent_length = 25
        name_prepend = "InteractionTerm".ljust(indent_length, "-") + "|\n"

        output = ""
        for param_name in self._init_params:
            param_content = getattr(self, param_name).__repr__()
            param_content = param_content.strip("\n")
            if len(param_content) > 50:
                param_content = param_content[:50]
                param_content += " ..."

            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), param_content
            )
        return name_prepend + output

    def hamiltonian(
        self,
        subsystem_list: list[QuantumSys],
        bare_esys: dict[int, ndarray] | None = None,
    ) -> qt.Qobj:
        """Return the interaction-term Hamiltonian for the calling Hilbert space.

        Parameters
        ----------
        subsystem_list:
            list of all quantum systems in the :class:`HilbertSpace` calling
            :meth:`hamiltonian`; needed for identity wrapping.
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.

        Returns
        -------
        Hamiltonian in :class:`qutip.Qobj` format.
        """
        hamiltonian = cast(qt.Qobj, self.g_strength)
        id_wrapped_ops = self.id_wrap_all_ops(
            self.operator_list, subsystem_list, bare_esys=bare_esys
        )
        for op in id_wrapped_ops:
            hamiltonian *= op
        if self.add_hc:
            hamiltonian += hamiltonian.dag()
        return hamiltonian

    @staticmethod
    def id_wrap_all_ops(
        operator_list: list[tuple[int, ndarray | csc_matrix | Callable]],
        subsystem_list: list[QuantumSys],
        bare_esys: dict[int, ndarray] | None = None,
    ) -> list[qt.Qobj]:
        """Return identity-wrapped operators, one per entry in ``operator_list``.

        Any callable operator in ``operator_list`` is evaluated here.

        Parameters
        ----------
        operator_list:
            list of ``(subsys_index, operator)`` tuples.
        subsystem_list:
            list of all quantum systems in the :class:`HilbertSpace` calling
            :meth:`hamiltonian`; needed for identity wrapping.
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.

        Returns
        -------
        list of identity-wrapped operators.
        """
        id_wrapped_operators = []
        for subsys_index, operator in operator_list:
            if bare_esys is not None and subsys_index in bare_esys:
                esys = bare_esys[subsys_index]
                evecs = esys[1]
            else:
                esys = None
                evecs = None

            if callable(operator):
                try:
                    operator = operator(energy_esys=esys)
                except TypeError:
                    operator = operator()
                op_in_eigenbasis = bool(esys)
            else:
                op_in_eigenbasis = False

            id_wrapped_operators.append(
                spec_utils.identity_wrap(
                    operator,
                    subsystem_list[subsys_index],
                    subsystem_list,
                    evecs=evecs,
                    op_in_eigenbasis=op_in_eigenbasis,
                )
            )
        return id_wrapped_operators


class InteractionTermStr(dispatch.DispatchClient, serializers.Serializable):
    """Specify a term in the interaction Hamiltonian via a string expression.

    Constructs the Hamiltonian in :class:`qutip.Qobj` format. The interaction
    is defined by the ``expr`` string. Each operator must be Hermitian, unless
    ``add_hc=True``, in which case each operator may be non-Hermitian.
    Functions usable inside ``expr`` include ``cos()``, ``sin()``, ``dag()``,
    ``conj()``, ``exp()``, ``sqrt()``, ``trans()``, ``cosm()``, ``sinm()``,
    ``expm()``, and ``sqrtm()``, along with other operators allowed in Python
    expressions.

    Parameters
    ----------
    expr:
        string that defines the interaction.
    operator_list:
        list of ``(subsys_index, name, operator)`` tuples.
    id_wrapped_operator_list:
        optional list of ``(name, callable)`` pairs supplying operators that are
        already identity-wrapped on the full Hilbert space; the callable is invoked
        with ``bare_esys=...`` when the Hamiltonian is built.
    const:
        optional mapping of constant names to numerical or qubit-valued constants
        that may be referenced inside ``expr``.
    add_hc:
        if ``True``, the interaction Hamiltonian is of type 2 and the Hermitian
        conjugate is added.
    """

    expr = descriptors.WatchedProperty(str, "INTERACTIONTERM_UPDATE")
    operator_list = descriptors.WatchedProperty(
        list[tuple[int, str, ndarray | csc_matrix | dia_matrix]],
        "INTERACTIONTERM_UPDATE",
    )
    add_hc = descriptors.WatchedProperty(bool, "INTERACTIONTERM_UPDATE")

    def __init__(
        self,
        expr: str,
        operator_list: list[tuple[int, str, ndarray | csc_matrix | dia_matrix]],
        id_wrapped_operator_list: list[tuple[str, Callable[..., Any]]] | None = None,
        const: dict[str, float | complex | QubitBaseClass] | None = None,
        add_hc: bool = False,
    ) -> None:
        self.qutip_dict = {
            "cosm(": "Qobj.cosm(",
            "expm(": "Qobj.expm(",
            "sinm(": "Qobj.sinm(",
            "sqrtm(": "Qobj.sqrtm(",
            "cos(": "Qobj.cosm(",
            "exp(": "Qobj.expm(",
            "sin(": "Qobj.sinm(",
            "sqrt(": "Qobj.sqrtm(",
        }
        self.expr = expr
        self.operator_list = operator_list
        self.id_wrapped_operator_list = id_wrapped_operator_list or []
        self.const = const or {}
        self.add_hc = add_hc

    def __repr__(self) -> str:
        """Return an ``eval``-friendly developer-facing representation."""
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        """Return a human-readable summary of this string-based interaction term."""
        indent_length = 25
        name_prepend = "InteractionTermStr".ljust(indent_length, "-") + "|\n"

        output = ""
        for param_name in self._init_params:
            param_content = getattr(self, param_name).__repr__()
            param_content = param_content.strip("\n")
            if len(param_content) > 50:
                param_content = param_content[:50]
                param_content += " ..."

            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), param_content
            )
        return name_prepend + output

    def parse_qutip_functions(self, string: str) -> str:
        """Rewrite shorthand qutip function names to their ``Qobj`` equivalents.

        Parameters
        ----------
        string:
            interaction expression possibly containing the shorthand names.
        """
        for item, value in self.qutip_dict.items():
            if item in string:
                string = string.replace(item, value)
        return string

    def run_string_code(
        self, expression: str, idwrapped_ops_by_name: dict[str, qt.Qobj]
    ) -> qt.Qobj:
        """Evaluate ``expression`` against the wrapped operators and constants.

        Parameters
        ----------
        expression:
            Python expression (after shorthand substitution) defining the
            interaction Hamiltonian.
        idwrapped_ops_by_name:
            mapping from operator names used in ``expression`` to their
            identity-wrapped :class:`qutip.Qobj` instances.

        Returns
        -------
        the :class:`qutip.Qobj` produced by evaluating ``expression``.
        """
        expression = self.parse_qutip_functions(expression)
        idwrapped_ops_by_name["Qobj"] = qt.Qobj

        main = importlib.import_module("__main__")
        answer = eval(
            expression, {**main.__dict__, **idwrapped_ops_by_name, **self.const}
        )
        return answer

    def id_wrap_all_ops(
        self,
        subsys_list: list[QuantumSys],
        bare_esys: dict[int, ndarray] | None = None,
    ) -> dict[str, qt.Qobj]:
        """Return a mapping from operator names to identity-wrapped ``Qobj`` ops.

        Parameters
        ----------
        subsys_list:
            list of all quantum systems in the :class:`HilbertSpace`; needed for
            identity wrapping.
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.
        """
        idwrapped_ops_by_name = {}
        for subsys_index, name, op in self.operator_list:
            if bare_esys and subsys_index in bare_esys:
                evecs = bare_esys[subsys_index][1]
            else:
                evecs = None
            idwrapped_ops_by_name[name] = spec_utils.identity_wrap(
                op,
                subsys_list[subsys_index],
                subsys_list,
                evecs=evecs,
                op_in_eigenbasis=False,
            )
        return idwrapped_ops_by_name

    def hamiltonian(
        self,
        subsystem_list: list[QuantumSys],
        bare_esys: dict[int, ndarray] | None = None,
    ) -> qt.Qobj:
        """Return the Hamiltonian obtained by evaluating the stored expression.

        Parameters
        ----------
        subsystem_list:
            list of all quantum systems in the :class:`HilbertSpace` calling
            :meth:`hamiltonian`; needed for identity wrapping.
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.
        """
        idwrapped_ops_by_name = self.id_wrap_all_ops(
            subsystem_list, bare_esys=bare_esys
        )
        idwrapped_ops_by_name.update(
            {
                item[0]: item[1](bare_esys=bare_esys)
                for item in self.id_wrapped_operator_list
            }
        )
        hamiltonian = self.run_string_code(self.expr, idwrapped_ops_by_name)
        if not self.add_hc:
            return hamiltonian
        else:
            return hamiltonian + hamiltonian.dag()


def _auto_sparse_diag_method(dim: int, evals_count: int, kind: str) -> str | None:
    """Pick a default diagonalization method based on the problem size.

    Returns the name of a sparse ``diag`` method (scipy ``eigsh``) when only a small
    fraction of a large spectrum is requested -- the regime where sparse
    diagonalization is much faster than dense -- and ``None`` to use the dense QuTiP
    path otherwise. Controlled by ``settings.AUTO_SPARSE_DIAG`` and the
    ``SPARSE_DIAG_*`` thresholds.

    Parameters
    ----------
    dim:
        dimension of the Hamiltonian to be diagonalized.
    evals_count:
        number of eigenvalues/eigenstates requested.
    kind:
        ``'evals'`` or ``'esys'``, selecting the eigenvalues-only or
        eigenvalues-and-eigenvectors method.

    Returns
    -------
    A key into :data:`scqubits.core.diag.DIAG_METHODS`, or ``None`` for the dense
    default.
    """
    if (
        not getattr(settings, "AUTO_SPARSE_DIAG", False)
        or dim < settings.SPARSE_DIAG_MIN_DIM
        or evals_count > max(1, int(dim * settings.SPARSE_DIAG_MAX_EVALS_FRAC))
    ):
        return None
    return "{}_scipy_sparse".format(kind)


# Residual/sanity thresholds for the auto sparse-diagonalization guard. A converged
# `eigsh` eigenpair has a residual ||H v - lambda v|| of ~1e-10 (relative to the
# eigenvalue scale); a silently mis-converged one has an O(1) residual, so this
# generous relative tolerance cleanly separates the two without false positives on
# good results. Only a few eigenpairs are checked (a handful of sparse mat-vecs,
# negligible next to the solve).
_SPARSE_RESIDUAL_RTOL = 1e-6
_SPARSE_RESIDUAL_CHECK_STATES = 3


def _sparse_result_trustworthy(
    hamiltonian_mat: qt.Qobj, kind: str, result: Any, evals_count: int
) -> bool:
    """Cheap correctness check on an auto sparse-``eigsh`` result.

    Sparse ``eigsh`` raises on non-convergence (caught by the dense fallback in
    :func:`_diagonalize_default`), but in rare cases -- clustered spectra, too few
    Lanczos vectors -- it can return a wrong subspace *without* raising. This guard
    catches that silent case: it checks the eigenvalues are finite and the expected
    count, and -- when eigenvectors are available (``kind == 'esys'``) -- that the
    residual ``||H v - lambda v||`` of a few eigenpairs is small relative to the
    eigenvalue scale. (Convergence, not eigenvalue ordering, is what is verified;
    the sparse methods return eigenvalues ascending, like the dense path.)

    Parameters
    ----------
    hamiltonian_mat:
        the Hamiltonian that was diagonalized.
    kind:
        ``'evals'`` or ``'esys'``.
    result:
        the value returned by the sparse method (eigenvalues, or a
        ``(eigenvalues, eigenvectors)`` tuple).
    evals_count:
        number of eigenvalues/eigenstates that were requested.

    Returns
    -------
    ``True`` if the result passes the sanity/residual checks, ``False`` otherwise.
    """
    evals = np.asarray(result[0] if kind == "esys" else result, dtype=float)
    if evals.shape[0] != evals_count or not np.all(np.isfinite(evals)):
        return False
    if kind != "esys":
        return True  # eigenvalues only: no eigenvectors to form a residual from
    evecs = result[1]
    # check the lowest few states (which="SA" targets these) plus the highest
    # requested one (weakest convergence under SA)
    check_indices = sorted(
        set(range(min(evals_count, _SPARSE_RESIDUAL_CHECK_STATES))) | {evals_count - 1}
    )
    for i in check_indices:
        vec = evecs[i]
        residual = (hamiltonian_mat * vec - evals[i] * vec).norm()
        if residual > _SPARSE_RESIDUAL_RTOL * (1.0 + abs(float(evals[i]))):
            return False
    return True


def _diagonalize_default(
    hamiltonian_mat: qt.Qobj,
    evals_count: int,
    kind: str,
    dense_fallback: Callable[[], Any],
) -> Any:
    """Diagonalize via the size-based default method, sparse with dense fallback.

    Used by :meth:`HilbertSpace.eigenvals` / :meth:`HilbertSpace.eigensys` when no
    explicit ``evals_method`` / ``esys_method`` is set. Uses sparse ``eigsh`` when
    :func:`_auto_sparse_diag_method` selects it, otherwise the dense QuTiP path. Falls
    back to dense (with a warning) if the sparse solver raises *or* returns a result
    that fails :func:`_sparse_result_trustworthy`.

    Parameters
    ----------
    hamiltonian_mat:
        Hamiltonian to diagonalize.
    evals_count:
        number of eigenvalues/eigenstates requested.
    kind:
        ``'evals'`` or ``'esys'``.
    dense_fallback:
        zero-argument callable returning the dense result (``eigenenergies`` or
        ``eigenstates`` of ``hamiltonian_mat``).

    Returns
    -------
    The eigenvalues (``kind == 'evals'``) or ``(eigenvalues, eigenvectors)``
    (``kind == 'esys'``).
    """
    auto_method = _auto_sparse_diag_method(hamiltonian_mat.shape[0], evals_count, kind)
    if auto_method is None:
        return dense_fallback()
    try:
        result = diag.DIAG_METHODS[auto_method](
            hamiltonian_mat, evals_count=evals_count
        )
    except Exception:
        warnings.warn(
            "scqubits: automatic sparse diagonalization raised; falling back to "
            "dense. Set scqubits.settings.AUTO_SPARSE_DIAG = False to disable "
            "automatic sparse diagonalization.",
            RuntimeWarning,
        )
        return dense_fallback()
    if not _sparse_result_trustworthy(hamiltonian_mat, kind, result, evals_count):
        warnings.warn(
            "scqubits: automatic sparse diagonalization returned a result that "
            "failed a residual check; falling back to dense. Set "
            "scqubits.settings.AUTO_SPARSE_DIAG = False to disable automatic sparse "
            "diagonalization.",
            RuntimeWarning,
        )
        return dense_fallback()
    return result


class HilbertSpace(
    spec_lookup.SpectrumLookupMixin, dispatch.DispatchClient, serializers.Serializable
):
    """Composite Hilbert space assembled from one or more subsystems.

    Provides methods that lift subsystem operators to operators acting on the
    full Hilbert space and establishes the interface to qutip; returned
    operators are of the :class:`qutip.Qobj` type. Also provides methods for
    obtaining eigenvalues and absorption/emission spectra as a function of an
    external parameter.

    Parameters
    ----------
    subsystem_list:
        list of all quantum systems comprising the composite Hilbert space.
    interaction_list:
        optional list of interaction-term objects supplied at initialization.
        Typically, interaction terms are added one by one via
        :meth:`add_interaction` instead.
    ignore_low_overlap:
        if ``False`` (default), bare product states and dressed eigenstates are
        identified only when their overlap exceeds the threshold set in
        ``settings.OVERLAP_THRESHOLD``; if ``True``, identification always
        proceeds via the bare product state with the maximum overlap.
    esys_method:
        method for esys diagonalization; callable or string representation.
    esys_method_options:
        dictionary with esys diagonalization options.
    evals_method:
        method for evals diagonalization; callable or string representation.
    evals_method_options:
        dictionary with evals diagonalization options.
    """

    _lookup_exists = False
    osc_subsys_list = descriptors.ReadOnlyProperty(OscillatorList)
    qbt_subsys_list = descriptors.ReadOnlyProperty(QubitList)
    interaction_list = descriptors.WatchedProperty(
        list[InteractionTerm | InteractionTermStr], "INTERACTIONLIST_UPDATE"
    )

    def __init__(
        self,
        subsystem_list: list[QuantumSys],
        interaction_list: list[InteractionTerm | InteractionTermStr] | None = None,
        ignore_low_overlap: bool = False,
        evals_method: Callable | str | None = None,
        evals_method_options: dict | None = None,
        esys_method: Callable | str | None = None,
        esys_method_options: dict | None = None,
    ) -> None:
        if has_duplicate_id_str(subsystem_list):
            raise ValueError(
                "Subsystem list must not contain multiple objects with "
                "the same `id_str` name."
            )
        self._subsystems: list[QuantumSys] = subsystem_list
        self._subsys_by_id_str = {
            obj._id_str: self[index] for index, obj in enumerate(self)
        }
        if interaction_list:
            self.interaction_list = interaction_list
        else:
            self.interaction_list = []
        self._interaction_term_by_id_str = {
            "InteractionTerm_{}".format(index): interaction_term
            for index, interaction_term in enumerate(self.interaction_list)
        }

        self._osc_subsys_list = [
            subsys
            for subsys in self
            if isinstance(subsys, osc.Oscillator)
            or (hasattr(subsys, "is_purely_harmonic") and subsys.is_purely_harmonic)
        ]
        self._qbt_subsys_list = [
            subsys for subsys in self if subsys not in self._osc_subsys_list
        ]

        self.evals_method = evals_method
        self.evals_method_options = evals_method_options
        self.esys_method = esys_method
        self.esys_method_options = esys_method_options

        # The following attributes are for compatibility with SpectrumLookupMixin
        self._data: dict[str, Any] = {}
        self._parameters: Parameters = Parameters({"dummy_parameter": np.array([0])})
        self._ignore_low_overlap = ignore_low_overlap
        self._current_param_indices = 0
        self._evals_count: int = self.dimension
        self._out_of_sync = False
        self._out_of_sync_warning_issued = False
        # end attributes for compatibility with SpectrumLookupMixin

        dispatch.CENTRAL_DISPATCH.register("QUANTUMSYSTEM_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("INTERACTIONTERM_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("INTERACTIONLIST_UPDATE", self)

    @overload
    def __getitem__(self, key: int) -> QuantumSys: ...

    @overload
    def __getitem__(
        self, key: str
    ) -> QuantumSys | InteractionTerm | InteractionTermStr: ...

    def __getitem__(
        self, key: int | str
    ) -> QuantumSys | InteractionTerm | InteractionTermStr:
        """Return the subsystem, interaction term, or data entry for ``key``.

        Integer keys are interpreted as positional indices into the subsystem list;
        string keys are matched against subsystem ``id_str``s, interaction-term
        identifiers, and (last) keys in ``self._data``.

        Parameters
        ----------
        key:
            integer subsystem index, or string identifier of a subsystem,
            interaction term, or data entry.
        """
        if isinstance(key, int):
            return self._subsystems[key]
        if key in self._subsys_by_id_str:
            return self._subsys_by_id_str[key]
        if key in self._interaction_term_by_id_str:
            return self._interaction_term_by_id_str[key]
        if key in self._data.keys():
            return self._data[key]

        raise KeyError(
            "Unrecognized key: {}. Key must be an integer index or a "
            "string specifying a subsystem or interaction term part of "
            "HilbertSpace.".format(key)
        )

    def __iter__(self) -> Iterator[QuantumSys]:
        """Iterate over the subsystems making up this :class:`HilbertSpace`."""
        return iter(self._subsystems)

    def __repr__(self) -> str:
        """Return an ``eval``-friendly developer-facing representation."""
        init_dict = self.get_initdata()
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        """Return a human-readable summary of subsystems and interaction terms."""
        output = "HilbertSpace:  subsystems\n"
        output += "-------------------------\n"
        for subsystem in self:
            output += f"\n{subsystem}\n"
        if self.interaction_list:
            output += "\n\n"
            output += "HilbertSpace:  interaction terms\n"
            output += "--------------------------------\n"

            for id_str, interaction_term in self._interaction_term_by_id_str.items():
                indent_length = 25
                term_output = "InteractionTerm".ljust(indent_length, "-")
                term_output += f"| [{id_str}]\n"
                term_output += "\n".join(str(interaction_term).splitlines()[1:])
                term_output += "\n\n"
                output += term_output
        return output

    def __len__(self) -> int:
        """Return the number of subsystems composing this :class:`HilbertSpace`."""
        return len(self._subsystems)

    @property
    def hilbertspace(self) -> HilbertSpace:
        """[Legacy] Auxiliary reference to ``self`` for SpectrumLookupMixin."""
        return self

    @property
    @utils.DeprecationMessage(
        "`subsys_list` is deprecated. Use `subsystem_list` instead."
    )
    def subsys_list(self) -> list[QuantumSys]:
        """Deprecated alias for :attr:`subsystem_list`."""
        return list(self._subsystems)

    def subsys_by_id_str(self, id_str: str) -> QuantumSys:
        """Return the subsystem whose ``id_str`` matches the supplied identifier.

        Parameters
        ----------
        id_str:
            identifier string previously assigned to a subsystem.
        """
        return self._subsys_by_id_str[id_str]

    ###################################################################################
    # HilbertSpace: file IO methods
    ###################################################################################
    @classmethod
    def deserialize(cls, io_data: "IOData") -> HilbertSpace:
        """Return a new instance initialized from the supplied ``io_data`` payload.

        Parameters
        ----------
        io_data:
            :class:`IOData` payload produced by a previous :meth:`serialize` call.
        """
        alldata_dict = io_data.as_kwargs()
        alldata_dict["ignore_low_overlap"] = alldata_dict.pop("_ignore_low_overlap")
        data = alldata_dict.pop("_data", {})
        new_hilbertspace: HilbertSpace = cls(**alldata_dict)
        new_hilbertspace._data = data
        return new_hilbertspace

    def serialize(self) -> "IOData":
        """Convert the content of the current class instance into IOData format."""
        init_parameters = self._init_params.copy()
        init_parameters.remove("ignore_low_overlap")
        init_parameters.append("_ignore_low_overlap")
        initdata = {name: getattr(self, name) for name in init_parameters}
        if self._data:
            initdata = {**initdata, "_data": self._data}
        iodata = serializers.dict_serialize(initdata)
        iodata.typename = type(self).__name__
        return iodata

    def get_initdata(self) -> dict[str, Any]:
        """Return a dict suitable for initializing a new :class:`HilbertSpace`."""
        return {
            "subsystem_list": self._subsystems,
            "interaction_list": self.interaction_list,
        }

    ###################################################################################
    # HilbertSpace: creation via GUI
    ###################################################################################
    @classmethod
    def create(cls) -> HilbertSpace:
        """Launch the GUI widget that builds a :class:`HilbertSpace` interactively."""
        hilbertspace = cls([])
        scqubits.ui.hspace_widget.create_hilbertspace_widget(hilbertspace.__init__)  # type: ignore[misc]
        return hilbertspace

    ###################################################################################
    # HilbertSpace: methods for CentralDispatch
    ###################################################################################
    def receive(self, event: str, sender: Any, **kwargs: Any) -> None:
        """Handle central-dispatch events affecting this :class:`HilbertSpace`.

        Side effect: marks the lookup as out-of-sync (sets
        ``self._out_of_sync = True``) whenever a relevant update arrives and a
        lookup table already exists.

        Parameters
        ----------
        event:
            name of the event being broadcast.
        sender:
            the object originating the event.
        """
        if event == "QUANTUMSYSTEM_UPDATE" and sender in self:
            self.broadcast("HILBERTSPACE_UPDATE")
            if self.lookup_exists():
                self._out_of_sync = True
        elif event == "INTERACTIONTERM_UPDATE" and sender in self.interaction_list:
            self.broadcast("HILBERTSPACE_UPDATE")
            if self.lookup_exists():
                self._out_of_sync = True
        elif event == "INTERACTIONLIST_UPDATE" and sender is self:
            self.broadcast("HILBERTSPACE_UPDATE")
            if self.lookup_exists():
                self._out_of_sync = True

    ###################################################################################
    # HilbertSpace: subsystems, dimensions, etc.
    ###################################################################################
    def get_subsys_index(self, subsys: QuantumSys) -> int:
        """Return the index of the given subsystem in the :class:`HilbertSpace`.

        Parameters
        ----------
        subsys:
            the subsystem whose position in :attr:`subsystem_list` is requested.
        """
        return self._subsystems.index(subsys)

    @property
    def subsystem_list(self) -> list[QuantumSys]:
        """Return the list of subsystems composing the joint Hilbert space."""
        return self._subsystems

    @property
    def subsystem_dims(self) -> list[int]:
        """Return the list of Hilbert space dimensions of each subsystem."""
        return [subsystem.truncated_dim for subsystem in self]

    @property
    def dimension(self) -> int:
        """Return the total dimension of the joint Hilbert space."""
        return np.prod(np.asarray(self.subsystem_dims)).item()

    @property
    def subsystem_count(self) -> int:
        """Return the number of subsystems composing the joint Hilbert space."""
        return len(self._subsystems)

    ###################################################################################
    # HilbertSpace: generate SpectrumLookup
    ###################################################################################
    def generate_lookup(  # type: ignore[override]
        self,
        ordering: Literal["DE", "LX", "BE"] = "DE",
        subsys_priority: list[int] | None = None,
        BEs_count: int | None = None,
        update_subsystem_indices: list[int] | None = None,
    ) -> None:
        """Label the dressed states by bare labels and generate the lookup table.

        The labeling uses one of the following methods:

        - Dressed Energy (``ordering="DE"``): traverse the eigenstates in
          order of their dressed energy and find the corresponding bare state
          label by overlaps (default).
        - Lexical (``ordering="LX"``): traverse the bare states in lexical
          order (see
          https://en.wikipedia.org/wiki/Lexicographic_order#Cartesian_products)
          and perform the branch analysis generalized from Dumas et al. (2024).
        - Bare Energy (``ordering="BE"``): traverse the bare states in order
          of their energy before coupling and perform label assignment.
          Particularly useful when the Hilbert space is too large for every
          eigenstate to be labeled.

        Parameters
        ----------
        ordering:
            the ordering method for the labeling
            - "DE": Dressed Energy (default)
            - "LX": Lexical ordering
            - "BE": Bare Energy
        subsys_priority:
            a permutation of the subsystem indices and bare labels. If
            provided, lexical ordering is performed on the permuted labels. A
            "branch" is a series of eigenstates formed by putting excitations
            into the last subsystem in the list.
        BEs_count:
            number of eigenstates to be assigned (``"BE"`` scheme only). If
            ``None``, all eigenstates will be generated and labeled.
        update_subsystem_indices:
            optional list of subsystem indices whose bare eigensystems should be
            recomputed; subsystems not listed reuse cached bare data. If ``None``,
            all subsystems are refreshed.

        Returns
        -------
        a :class:`NamedSlotsNdarray` object containing the branch-analysis results
        organized by the parameter indices. For each parameter point, a flattened
        multi-dimensional array is stored, representing the dressed indices
        organized by the bare indices. E.g., if the subsystem dimensions are
        ``D0, D1, D2``, the returned array is ravelled from shape
        ``(D0, D1, D2)``.
        """
        self._lookup_exists = True
        bare_esys_dict = self.generate_bare_esys(
            update_subsystem_indices=update_subsystem_indices
        )
        dummy_params = self._parameters.paramvals_by_name

        if ordering == "DE" or ordering == "LX" or BEs_count is None:
            num_evals = self.dimension
        else:
            num_evals = BEs_count

        evals, evecs = self.eigensys(evals_count=num_evals, bare_esys=bare_esys_dict)
        # The following workaround ensures that eigenvectors maintain QutipEigenstates
        # view when getting placed inside an outer array
        evecs_wrapped = np.empty(shape=1, dtype=object)
        evecs_wrapped[0] = evecs

        self._data["evals"] = NamedSlotsNdarray(np.array([evals]), dummy_params)
        self._data["evecs"] = NamedSlotsNdarray(evecs_wrapped, dummy_params)
        self._data["dressed_indices"] = spec_lookup.SpectrumLookupMixin.generate_lookup(
            self,
            ordering=ordering,
            subsys_priority=subsys_priority,
            BEs_count=BEs_count,
        )

    def lookup_exists(self) -> bool:
        """Return whether a dressed-state lookup has already been generated."""
        return self._lookup_exists

    def generate_bare_esys(
        self, update_subsystem_indices: list[int] | None = None
    ) -> dict:
        """Compute and cache bare eigensystems for the requested subsystems.

        Parameters
        ----------
        update_subsystem_indices:
            indices of subsystems whose bare eigensystems should be recomputed; if
            ``None``, every subsystem is refreshed. Subsystems not listed reuse the
            cached bare data already stored on the instance.

        Returns
        -------
        a dict mapping subsystem index to the ``(evals, evecs)`` tuple produced for
        that subsystem during this call.
        """
        # update all the subsystems when update_subsystem_indices is set to None
        if update_subsystem_indices is None:
            update_subsystem_indices = list(range(self.subsystem_count))

        bare_evals = np.empty((self.subsystem_count,), dtype=object)
        bare_evecs = np.empty((self.subsystem_count,), dtype=object)
        bare_esys_dict = {}

        for subsys_index, subsys in enumerate(self):
            # generate bare_esys for the subsystem as well if necessary
            if (
                hasattr(subsys, "hierarchical_diagonalization")
                and subsys.hierarchical_diagonalization
            ):
                subsys.hilbert_space.generate_bare_esys(  # type: ignore[union-attr]
                    update_subsystem_indices=subsys.affected_subsystem_indices  # type: ignore[union-attr]
                )
                subsys.affected_subsystem_indices = []  # type: ignore[union-attr]
            # diagonalizing only those subsystems present in update_subsystem_indices
            if subsys_index in update_subsystem_indices:
                bare_esys = subsys.eigensys(evals_count=subsys.truncated_dim)
            else:
                bare_esys = (
                    self["bare_evals"][subsys_index][0],  # type: ignore[index]
                    self["bare_evecs"][subsys_index][0],  # type: ignore[index]
                )
            bare_esys_dict[subsys_index] = bare_esys
            bare_evals[subsys_index] = NamedSlotsNdarray(
                np.asarray([bare_esys[0].tolist()]),
                self._parameters.paramvals_by_name,
            )
            bare_evecs[subsys_index] = NamedSlotsNdarray(
                np.asarray([bare_esys[1].tolist()]),
                self._parameters.paramvals_by_name,
            )
        self._data["bare_evals"] = NamedSlotsNdarray(
            bare_evals, {"subsys": np.arange(self.subsystem_count)}
        )
        self._data["bare_evecs"] = NamedSlotsNdarray(
            bare_evecs, {"subsys": np.arange(self.subsystem_count)}
        )
        return bare_esys_dict

    ###################################################################################
    # HilbertSpace: energy spectrum
    ##################################################################################
    def eigenvals(
        self,
        evals_count: int = 6,
        bare_esys: dict[int, ndarray | list[ndarray]] | None = None,
    ) -> ndarray:
        """Calculate eigenvalues of the full Hamiltonian.

        By default (``evals_method`` is ``None``) a size-based heuristic selects the
        diagonalizer: sparse scipy ``eigsh`` for large Hamiltonians where only a
        small fraction of the spectrum is requested (see
        ``settings.AUTO_SPARSE_DIAG``), otherwise QuTiP's dense
        :meth:`qutip.Qobj.eigenenergies`. Setting :attr:`evals_method` to a method
        name or callable overrides this choice.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates.
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.
        """
        hamiltonian_mat = self.hamiltonian(bare_esys=bare_esys)  # type: ignore[arg-type]

        if not hasattr(self, "evals_method") or self.evals_method is None:
            evals = _diagonalize_default(
                hamiltonian_mat,
                evals_count,
                "evals",
                lambda: hamiltonian_mat.eigenenergies(eigvals=evals_count),
            )
        else:
            diagonalizer = (
                diag.DIAG_METHODS[self.evals_method]
                if isinstance(self.evals_method, str)
                else self.evals_method
            )
            evals = diagonalizer(
                hamiltonian_mat,
                evals_count=evals_count,
                **(
                    {}
                    if self.evals_method_options is None
                    else self.evals_method_options
                ),
            )
        return evals

    def eigensys(
        self,
        evals_count: int = 6,
        bare_esys: dict[int, ndarray | list[ndarray]] | None = None,
    ) -> tuple[ndarray, QutipEigenstates]:
        """Calculate eigenvalues and eigenvectors of the full Hamiltonian.

        By default (``esys_method`` is ``None``) a size-based heuristic selects the
        diagonalizer: sparse scipy ``eigsh`` for large Hamiltonians where only a
        small fraction of the spectrum is requested (see
        ``settings.AUTO_SPARSE_DIAG``), otherwise QuTiP's dense
        :meth:`qutip.Qobj.eigenstates`. Setting :attr:`esys_method` to a method name
        or callable overrides this choice.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates.
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.

        Returns
        -------
        eigenvalues and eigenvectors of the full Hamiltonian.

        Notes
        -----
        Eigenvalues and physical observables (matrix elements, lookup energies) are
        independent of the diagonalizer. Eigenvectors spanning a degenerate
        eigenspace are, however, only defined up to a basis choice within that
        space, so the integer dressed-state labels assigned to degenerate states by
        the overlap-based lookup may depend on the diagonalization method. Reference
        states by their bare-state labels rather than by hard-coded dressed indices.
        """
        hamiltonian_mat = self.hamiltonian(bare_esys=bare_esys)  # type: ignore[arg-type]

        if not hasattr(self, "esys_method") or self.esys_method is None:
            evals, evecs = _diagonalize_default(
                hamiltonian_mat,
                evals_count,
                "esys",
                lambda: hamiltonian_mat.eigenstates(eigvals=evals_count),
            )
        else:
            diagonalizer = (
                diag.DIAG_METHODS[self.esys_method]
                if isinstance(self.esys_method, str)
                else self.esys_method
            )
            evals, evecs = diagonalizer(
                hamiltonian_mat,
                evals_count=evals_count,
                **(
                    {} if self.esys_method_options is None else self.esys_method_options
                ),
            )

        evecs = evecs.view(scqubits.io_utils.fileio_qutip.QutipEigenstates)

        return evals, evecs

    def _esys_for_paramval(
        self,
        paramval: float,
        update_hilbertspace: Callable,
        evals_count: int,
        bare_esys: dict[int, ndarray | list[ndarray]] | None = None,
    ) -> tuple[ndarray, QutipEigenstates]:
        """Update the Hilbert space at ``paramval`` and return its eigensystem.

        Parameters
        ----------
        paramval:
            value of the external parameter at which to evaluate the eigensystem.
        update_hilbertspace:
            callback that mutates this :class:`HilbertSpace` to reflect ``paramval``.
        evals_count:
            number of eigenvalues/eigenstates requested.
        bare_esys:
            optional cached bare eigensystems forwarded to :meth:`eigensys`.
        """
        update_hilbertspace(paramval)
        return self.eigensys(evals_count, bare_esys=bare_esys)

    def _evals_for_paramval(
        self,
        paramval: float,
        update_hilbertspace: Callable,
        evals_count: int,
        bare_esys: dict[int, ndarray | list[ndarray]] | None = None,
    ) -> ndarray:
        """Update the Hilbert space at ``paramval`` and return its eigenvalues.

        Parameters
        ----------
        paramval:
            value of the external parameter at which to evaluate the spectrum.
        update_hilbertspace:
            callback that mutates this :class:`HilbertSpace` to reflect ``paramval``.
        evals_count:
            number of eigenvalues requested.
        bare_esys:
            optional cached bare eigensystems forwarded to :meth:`eigenvals`.
        """
        update_hilbertspace(paramval)
        return self.eigenvals(evals_count, bare_esys=bare_esys)

    ###################################################################################
    # HilbertSpace: Hamiltonian (bare, interaction, full)
    #######################################################

    def hamiltonian(
        self,
        bare_esys: dict[int, ndarray] | None = None,
    ) -> qt.Qobj:
        """Return the full composite Hamiltonian, including all interactions.

        Parameters
        ----------
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.

        Returns
        -------
        Hamiltonian of the composite system, including the interaction between
        components.
        """
        hamiltonian = self.bare_hamiltonian(bare_esys=bare_esys)
        hamiltonian += self.interaction_hamiltonian(bare_esys=bare_esys)
        return hamiltonian

    def bare_hamiltonian(self, bare_esys: dict[int, ndarray] | None = None) -> qt.Qobj:
        """Return the composite Hamiltonian assembled from bare subsystem terms.

        Parameters
        ----------
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.

        Returns
        -------
        composite Hamiltonian composed of bare Hamiltonians of subsystems
        independent of the external parameter.
        """
        # We create a dimension [1] system if no subsystems have been given
        bare_hamiltonian = qt.qzero(
            [1] if len(self.subsystem_dims) == 0 else self.subsystem_dims
        )

        for subsys_index, subsys in enumerate(self):
            if bare_esys is not None and subsys_index in bare_esys:
                evals = bare_esys[subsys_index][0]
            else:
                evals = subsys.eigenvals(evals_count=subsys.truncated_dim)
            bare_hamiltonian += self.diag_hamiltonian(subsys, evals)
        return bare_hamiltonian

    def interaction_hamiltonian(
        self, bare_esys: dict[int, ndarray] | None = None
    ) -> qt.Qobj:
        """Return the interaction Hamiltonian assembled from the registered terms.

        Parameters
        ----------
        bare_esys:
            optional precomputed bare eigensystems for each subsystem, supplied as
            a dict ``{subsys_index: esys}``; speeds up computation when available.

        Returns
        -------
        interaction Hamiltonian.
        """
        if not self.interaction_list:
            # We return a dimension [1] system if no subsystems have been given
            return qt.qzero(
                [1] if len(self.subsystem_dims) == 0 else self.subsystem_dims
            )

        operator_list = []
        for term in self.interaction_list:
            if isinstance(term, qt.Qobj):
                operator_list.append(term)
            elif isinstance(term, (InteractionTerm, InteractionTermStr)):
                operator_list.append(
                    term.hamiltonian(self.subsystem_list, bare_esys=bare_esys)
                )
            else:
                raise TypeError(
                    "Expected an instance of InteractionTerm, InteractionTermStr, "
                    "or Qobj; got {} instead.".format(type(term))
                )
        hamiltonian = sum(operator_list)
        return hamiltonian

    def diag_hamiltonian(
        self, subsystem: QuantumSys, evals: ndarray | None = None
    ) -> qt.Qobj:
        """Return a ``Qobj`` with the eigenenergies of ``subsystem`` on the diagonal.

        Parameters
        ----------
        subsystem:
            subsystem for which the Hamiltonian is to be provided.
        evals:
            precomputed eigenenergies; if ``None``, they are calculated.
        """
        evals_count = subsystem.truncated_dim

        if evals is None:
            evals = subsystem.eigenvals(evals_count=evals_count)
        diag_qt_op = qt.Qobj(np.diagflat(evals[0:evals_count]))
        return spec_utils.identity_wrap(
            diag_qt_op, subsystem, self.subsystem_list, op_in_eigenbasis=True
        )

    ###################################################################################
    # HilbertSpace: identity wrapping, operators
    ###################################################################################

    def diag_operator(self, diag_elements: ndarray, subsystem: QuantumSys) -> qt.Qobj:
        """Return the identity-wrapped diagonal operator for the full Hilbert space.

        Given the diagonal elements of a diagonal operator on ``subsystem``,
        returns the :class:`qutip.Qobj` operator on the full Hilbert space,
        wrapping in identities on the other subsystems.

        Parameters
        ----------
        diag_elements:
            diagonal elements of the subsystem diagonal operator.
        subsystem:
            subsystem on which the diagonal operator is defined.
        """
        dim = subsystem.truncated_dim
        index = range(dim)
        diag_matrix = np.zeros((dim, dim), dtype=np.float64)
        diag_matrix[index, index] = diag_elements
        return spec_utils.identity_wrap(
            diag_matrix, subsystem, self.subsystem_list, op_in_eigenbasis=True
        )

    def hubbard_operator(self, j: int, k: int, subsystem: QuantumSys) -> qt.Qobj:
        r"""Return the Hubbard operator :math:`|j\rangle\langle k|` for ``subsystem``.

        Parameters
        ----------
        j:
            row eigenstate index for the Hubbard operator.
        k:
            column eigenstate index for the Hubbard operator.
        subsystem:
            subsystem on which the Hubbard operator acts.
        """
        dim = subsystem.truncated_dim
        operator = qt.states.basis(dim, j) * qt.states.basis(dim, k).dag()
        return spec_utils.identity_wrap(
            operator, subsystem, self.subsystem_list, op_in_eigenbasis=True
        )

    def annihilate(self, subsystem: QuantumSys) -> qt.Qobj:
        r"""Return the annihilation operator :math:`a` for ``subsystem``.

        Parameters
        ----------
        subsystem:
            subsystem on which the annihilation operator acts.
        """
        dim = subsystem.truncated_dim
        operator = qt.destroy(dim)
        return spec_utils.identity_wrap(
            operator, subsystem, self.subsystem_list, op_in_eigenbasis=True
        )

    ###################################################################################
    # HilbertSpace: spectrum sweep
    ###################################################################################
    def get_spectrum_vs_paramvals(
        self,
        param_vals: ndarray,
        update_hilbertspace: Callable,
        evals_count: int = 10,
        get_eigenstates: bool = False,
        param_name: str = "external_parameter",
        num_cpus: int | str | None = None,
    ) -> SpectrumData:
        """Return the full-Hamiltonian spectrum as a function of an external param.

        Parameter values are specified as a list or array in ``param_vals``.
        The callback ``update_hilbertspace`` is invoked at each parameter value
        and is expected to set the subsystem parameters accordingly.

        Parameters
        ----------
        param_vals:
            array of parameter values.
        update_hilbertspace:
            ``update_hilbertspace(param_val)`` specifies how a change in the
            external parameter affects the Hilbert-space components.
        evals_count:
            number of desired energy levels (default: 10).
        get_eigenstates:
            if ``True``, eigenstates are returned alongside eigenvalues
            (default: ``False``).
        param_name:
            name for the parameter that is varied in ``param_vals``
            (default: ``"external_parameter"``).
        num_cpus:
            number of cores to use for computation, or ``"auto"`` to let scqubits
            choose the core count (and a per-worker BLAS-thread cap) from the
            workload via :func:`~scqubits.recommend_parallelization`. With
            ``settings.AUTO_PARALLEL = True`` an unspecified ``num_cpus`` is chosen
            the same way (default: ``settings.NUM_CPUS``).
        """
        if num_cpus == "auto" or (
            num_cpus is None and getattr(settings, "AUTO_PARALLEL", False)
        ):
            from scqubits.utils.parallel_tuning import _auto_config

            auto = _auto_config(self.dimension, len(param_vals), evals_count)
            num_cpus = auto.num_cpus
            blas_threads = auto.blas_threads
        else:
            num_cpus = (
                num_cpus
                if isinstance(num_cpus, int) and num_cpus
                else settings.NUM_CPUS
            )
            blas_threads = None
        # get_map_method returns a lazy, order-preserving map (built-in map when
        # serial, pool.imap when parallel); wrapping its output in tqdm gives a live
        # progress bar that advances as each grid point completes.
        target_map = cpu_switch.get_map_method(
            num_cpus, blas_threads, total=len(param_vals)
        )
        if get_eigenstates:
            func = functools.partial(
                self._esys_for_paramval,
                update_hilbertspace=update_hilbertspace,
                evals_count=evals_count,
            )
            eigensystem_mapdata = list(
                tqdm(
                    target_map(func, param_vals),
                    total=len(param_vals),
                    desc="Spectral data",
                    leave=False,
                    disable=settings.PROGRESSBAR_DISABLED,
                )
            )
            eigenvalue_table, eigenstate_table = spec_utils.recast_esys_mapdata(
                eigensystem_mapdata
            )
        else:
            func = functools.partial(  # type: ignore[assignment]
                self._evals_for_paramval,  # type: ignore[arg-type]
                update_hilbertspace=update_hilbertspace,
                evals_count=evals_count,
            )
            eigenvalue_table = np.asarray(
                list(
                    tqdm(
                        target_map(func, param_vals),
                        total=len(param_vals),
                        desc="Spectral data",
                        leave=False,
                        disable=settings.PROGRESSBAR_DISABLED,
                    )
                )
            )
            eigenstate_table = None

        return storage.SpectrumData(
            eigenvalue_table,
            self.get_initdata(),
            param_name,
            param_vals,
            state_table=eigenstate_table,
        )

    def standardize_eigenvector_phases(self) -> None:
        """Standardize the phases of the (dressed) eigenvectors."""
        for idx, evec in enumerate(self._data["evecs"][0]):
            array = utils.Qobj_to_scipy_csc_matrix(evec)
            phase = spec_utils.extract_phase(array)  # type: ignore[arg-type]
            self._data["evecs"][0][idx] = evec * np.exp(-1j * phase)

    @utils.check_lookup_exists
    def op_in_dressed_eigenbasis(
        self,
        op_callable_or_tuple: tuple[np.ndarray | csc_matrix, QuantumSys] | Callable,
        truncated_dim: int | None = None,
        **kwargs,
    ) -> qt.Qobj:
        """Express a subsystem operator in the dressed eigenbasis of the full system.

        (As opposed to either the "native basis" or "bare eigenbasis" of the
        subsystem.)

        The returned operator should not retain memory of the Hilbert-space sizes
        of the underlying subsystems, so the ``dims`` of the returned operator are
        flattened. ``truncated_dim`` sets the cutoff Hilbert-space size of the
        dressed system; if left at ``None`` (default), no cutoff is applied and
        the resulting :class:`qutip.Qobj` has ``dims=[[dimension], [dimension]]``.

        :meth:`op_in_dressed_eigenbasis` offers two different interfaces:

        1. subsystem operators may be expressed as Callables

            signature::

                .op_in_dressed_eigenbasis(op_callable_or_tuple=<Callable>,
                                          truncated_dim=<int>)

        2. subsystem operators may be passed as arrays, along with the
           corresponding subsystem. In this case the user must additionally
           specify if the operator is in the native, subsystem-internal
           basis or the subsystem bare eigenbasis::

                .op_in_dressed_eigenbasis(op_callable_or_tuple=(<ndarray>, <subsys>),
                                          truncated_dim=<int>,
                                          op_in_bare_eigenbasis=<Bool>)

        Parameters
        ----------
        op_callable_or_tuple:
            either a bound callable returning the subsystem operator, or a
            ``(operator_array, subsystem)`` tuple as described above.
        truncated_dim:
            optional cutoff Hilbert-space dimension of the dressed system; when
            ``None`` (default) no cutoff is applied.
        """
        if truncated_dim is None:
            truncated_dim = self.dimension
        if isinstance(op_callable_or_tuple, tuple):
            op, subsys = op_callable_or_tuple
            op_in_bare_eigenbasis = kwargs.pop("op_in_bare_eigenbasis", False)
            subsys_index = self.get_subsys_index(subsys)
        else:
            assert callable(op_callable_or_tuple)
            op = op_callable_or_tuple  # type: ignore[assignment]
            op_in_bare_eigenbasis = False
            subsys_index = self.get_subsys_index(op.__self__)  # type: ignore[union-attr]
        bare_evecs = self._data["bare_evecs"][subsys_index][0]
        id_wrapped_op = spec_utils.identity_wrap(
            op,
            self.subsystem_list[subsys_index],
            self.subsystem_list,
            op_in_eigenbasis=op_in_bare_eigenbasis,
            evecs=bare_evecs,
        )
        dressed_evecs = self._data["evecs"][0]
        dressed_op_data = utils.Qobj_to_scipy_csc_matrix(
            id_wrapped_op.transform(dressed_evecs)
        )
        dressed_op_truncated = qt.Qobj(
            dressed_op_data[0:truncated_dim, 0:truncated_dim],
            dims=[[truncated_dim], [truncated_dim]],
        )
        return dressed_op_truncated

    ###################################################################################
    # HilbertSpace: add interaction and parsing arguments to .add_interaction
    ###################################################################################
    def add_interaction(
        self,
        check_validity: bool = True,
        id_str: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Add an interaction term between subsystems of this :class:`HilbertSpace`.

        :meth:`add_interaction` offers three different interfaces:

        * Simple interface for operator products
        * String-based interface for more general interaction operator expressions
        * General :class:`qutip.Qobj` interface

        1. Simple interface for operator products
            Specify :class:`numpy.ndarray`, :class:`scipy.sparse.csc_matrix`, or
            :class:`scipy.sparse.dia_matrix` (subsystem operator in
            subsystem-internal basis) along with the corresponding subsystem

            signature::

                .add_interaction(g=<float>,
                                op1=(<ndarray>, <QuantumSystem>),
                                op2=(<csc_matrix>, <QuantumSystem>),
                                 ...,
                                add_hc=<bool>)

            Alternatively, specify subsystem operators via callable methods.

            signature::

                .add_interaction(g=<float>,
                                 op1=<Callable>,
                                 op2=<Callable>,
                                 ...,
                                 add_hc=<bool>)
        2. String-based interface for more general interaction operator expressions
                Specify a Python expression that generates the desired operator. The
                expression enables convenience use of basic qutip operations::

                    .add_interaction(expr=<str>,
                                     op1=(<str>, <ndarray>, <subsys>),
                                     op2=(<str>, <Callable>),
                                     ...)
        3. General Qobj operator
            Specify a fully identity-wrapped :class:`qutip.Qobj` operator.
            Signature::

                .add_interaction(qobj=<Qobj>)

        Parameters
        ----------
        check_validity:
            whether to check the validity of the interaction; switch off for
            speed if you are sure the interaction is valid.
        id_str:
            optional string by which this instance can be referred to in
            :class:`HilbertSpace` and :class:`scqubits.ParameterSweep`. If not
            provided, an id is auto-generated.
        """
        if "expr" in kwargs:
            interaction: InteractionTerm | InteractionTermStr = (
                self._parse_interactiontermstr(**kwargs)
            )
        elif "qobj" in kwargs:
            interaction = self._parse_qobj(**kwargs)
        elif "op1" in kwargs:
            interaction = self._parse_interactionterm(**kwargs)
        else:
            raise TypeError(
                "Invalid combination and/or types of arguments for `add_interaction`"
            )
        if self.lookup_exists():
            self._out_of_sync = True

        self.interaction_list.append(interaction)

        id_str = id_str or "Interaction_{}".format(len(self.interaction_list))
        self._interaction_term_by_id_str[id_str] = interaction

        if not check_validity:
            return None
        try:
            _ = self.interaction_hamiltonian()
        except Exception as inst:
            self.interaction_list.pop()
            del self._interaction_term_by_id_str[id_str]
            raise ValueError(f"Invalid Interaction Term. Exception: {inst}")

    def _parse_interactiontermstr(self, **kwargs: Any) -> InteractionTermStr:
        """Build an :class:`InteractionTermStr` from string-interface kwargs."""
        expr = kwargs.pop("expr")
        add_hc = kwargs.pop("add_hc", False)
        const = kwargs.pop("const", None)

        operator_list = []
        id_wrapped_operator_list = []
        for key in kwargs.keys():
            if callable(kwargs[key][1]) and not hasattr(kwargs[key][1], "__self__"):
                id_wrapped_operator_list.append(kwargs[key])
                continue
            if re.match(r"op\d+$", key) is None:
                raise TypeError("Unexpected keyword argument {}.".format(key))
            operator_list.append(self._parse_str_based_op(kwargs[key]))
        if id_wrapped_operator_list == []:
            id_wrapped_operator_list = None  # type: ignore[assignment]

        return InteractionTermStr(
            expr,
            operator_list,  # type: ignore[arg-type]
            id_wrapped_operator_list=id_wrapped_operator_list,
            const=const,
            add_hc=add_hc,
        )

    def _parse_interactionterm(self, **kwargs: Any) -> InteractionTerm:
        """Build an :class:`InteractionTerm` from operator-product kwargs."""
        g = kwargs.pop("g", None)
        if g is None:
            g = kwargs.pop("g_strength")
        add_hc = kwargs.pop("add_hc", False)

        operator_list = []
        for key in kwargs.keys():
            if re.match(r"op\d+$", key) is None:
                raise TypeError("Unexpected keyword argument {}.".format(key))
            subsys_index, op = self._parse_non_strbased_op(kwargs[key])
            operator_list.append((subsys_index, op))

        return InteractionTerm(g, operator_list, add_hc=add_hc)

    @staticmethod
    def _parse_qobj(**kwargs: Any) -> qt.Qobj:
        """Validate and return the ``qobj`` keyword from the Qobj interface."""
        op = kwargs["qobj"]
        if len(kwargs) > 1 or not isinstance(op, qt.Qobj):
            raise TypeError("Cannot interpret specified operator {}".format(op))
        return kwargs["qobj"]

    def _parse_str_based_op(
        self,
        keyword_value: tuple[str, ndarray, QuantumSys] | tuple[str, Callable],
    ) -> tuple[int, str, ndarray | csc_matrix | dia_matrix | Callable]:
        """Decompose a string-based descriptor into ``(subsys_index, name, op)``.

        Parameters
        ----------
        keyword_value:
            either ``(name, operator_array, subsystem)`` or ``(name, callable)``,
            where the callable is a bound method of the relevant subsystem.
        """
        if not isinstance(keyword_value, tuple):
            raise TypeError(
                "Cannot interpret specified operator {}".format(keyword_value)
            )
        if len(keyword_value) == 3:
            # format expected:  (<op name as str>, <op as array>, <subsys as QuantumSystem>)
            return (
                self.get_subsys_index(keyword_value[2]),
                keyword_value[0],
                keyword_value[1],
            )
        # format expected (<op name as str)>, <QuantumSystem.method callable>)
        return (
            self.get_subsys_index(keyword_value[1].__self__),  # type: ignore[attr-defined]
            keyword_value[0],
            keyword_value[1],
        )

    def _parse_non_strbased_op(
        self,
        op: Callable | tuple[ndarray | csc_matrix, QuantumSys],
    ) -> tuple[int, ndarray | csc_matrix | Callable]:
        """Decompose an operator descriptor into ``(subsys_index, operator)``.

        Parameters
        ----------
        op:
            either a bound callable returning the operator, or an
            ``(operator, subsystem)`` tuple.
        """
        if callable(op):
            return (
                self.get_subsys_index(op.__self__),  # type: ignore[attr-defined]
                op,
            )  # store op here, not op() [v3.2]
        if not isinstance(op, tuple):
            raise TypeError("Cannot interpret specified operator {}".format(op))
        if len(op) == 2:
            # format expected:  (<op as array>, <subsys as QuantumSystem>)
            return self.get_subsys_index(op[1]), op[0]
        raise TypeError("Cannot interpret specified operator {}".format(op))
