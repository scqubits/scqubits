# operator_factories.py
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
"""Closure factories that build the per-variable operator methods attached
dynamically to :class:`~scqubits.core.circuit.Subsystem` instances.

Each factory returns a callable suitable for binding as an instance
method via :class:`types.MethodType`. The wrapped callable wraps the
"bare" operator into the full Hilbert space using ``_kron_operator``
and optionally rotates into the energy eigenbasis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy import ndarray

from scqubits.utils.misc import Qobj_to_scipy_csc_matrix

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem


def make_grid_operator_method(inner_op: Callable, index: int) -> Callable:
    """Build an operator method for a discretized-grid variable.

    Wraps ``inner_op`` so that, when called as a method on a
    :class:`~scqubits.core.circuit.Subsystem`, it constructs the corresponding
    operator on the discretized grid for the variable identified by ``index``,
    embeds it via ``_kron_operator``, and applies optional energy-eigenbasis
    conversion.

    Parameters
    ----------
    inner_op:
        callable that returns the operator on a single discretized grid
    index:
        index of the variable on which the operator acts

    Returns
    -------
    Method to be attached to a :class:`~scqubits.core.circuit.Subsystem`
    instance.
    """

    def operator_func(
        self: "Subsystem", energy_esys: bool | tuple[ndarray, ndarray] = False
    ):
        native = self._kron_operator(
            inner_op(self.discretized_grids_dict_for_vars()[index]), index
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    return operator_func


def make_hierarchical_diag_method(symbol_name: str) -> Callable:
    """Build an operator method for a hierarchically diagonalized variable.

    The returned method retrieves the operator with name ``symbol_name`` from
    the parent subsystem (via :meth:`get_operator_by_name`), converts it to a
    SciPy CSC matrix, and applies optional energy-eigenbasis conversion.

    Parameters
    ----------
    symbol_name:
        name of the operator to be looked up via
        :meth:`~scqubits.core.circuit.Subsystem.get_operator_by_name`

    Returns
    -------
    Method to be attached to a :class:`~scqubits.core.circuit.Subsystem`
    instance.
    """

    def operator_func(
        self: "Subsystem", energy_esys: bool | tuple[ndarray, ndarray] = False
    ):
        """Returns the operator <op_name> (corresponds to the name of the method
        "<op_name>_operator") for the Circuit/Subsystem instance.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns charge operator n in the charge basis.
            If `True`, energy eigenspectrum is computed, returns charge operator n in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns charge operator n in the energy eigenbasis, and does not have to recalculate the
            eigenspectrum.

        Returns
        -------
            Returns the operator <op_name>(corresponds to the name of the method "<op_name>_operator").
            For `energy_esys=True`, n has dimensions of :attr:`truncated_dim` x :attr:`truncated_dim`.
            If an actual eigensystem is handed to `energy_sys`, then `n` has dimensions of m x m,
            where m is the number of given eigenvectors.
        """
        native = Qobj_to_scipy_csc_matrix(self.get_operator_by_name(symbol_name))
        return self.process_op(native_op=native, energy_esys=energy_esys)

    return operator_func


def make_basis_operator_method(
    inner_op: Callable, index: int, op_type: str | None = None
) -> Callable:
    """Build an operator method for periodic or harmonic-basis variables.

    Wraps ``inner_op`` so that, when called as a method on a
    :class:`~scqubits.core.circuit.Subsystem`, it constructs the corresponding
    operator on the relevant Hilbert subspace (using a prefactor derived from
    the oscillator length when ``ext_basis == "harmonic"`` and ``op_type`` is
    one of ``"position"``, ``"momentum"``, ``"sin"``, ``"cos"``), embeds it
    via ``_kron_operator``, and applies optional energy-eigenbasis conversion.

    Parameters
    ----------
    inner_op:
        callable that returns the bare operator on the variable's Hilbert space
    index:
        index of the variable on which the operator acts
    op_type:
        operator-type tag controlling the harmonic-basis prefactor; one of
        ``"position"``, ``"momentum"``, ``"sin"``, ``"cos"``, or ``None``

    Returns
    -------
    Method to be attached to a :class:`~scqubits.core.circuit.Subsystem`
    instance.
    """

    def operator_func(
        self: "Subsystem", energy_esys: bool | tuple[ndarray, ndarray] = False
    ):
        """Returns the operator <op_name> (corresponds to the name of the method
        "<op_name>_operator") for the Circuit/Subsystem instance.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns charge operator n in the charge basis.
            If `True`, energy eigenspectrum is computed, returns charge operator n in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns charge operator n in the energy eigenbasis, and does not have to recalculate the
            eigenspectrum.

        Returns
        -------
            Returns the operator <op_name>(corresponds to the name of the method "<op_name>_operator").
            For `energy_esys=True`, n has dimensions of :attr:`truncated_dim` x :attr:`truncated_dim`.
            If an actual eigensystem is handed to `energy_sys`, then `n` has dimensions of m x m,
            where m is the number of given eigenvectors.
        """
        prefactor = None
        if self.ext_basis == "harmonic":
            if op_type in ["position", "sin", "cos"]:
                prefactor = self.osc_lengths[index] / (2**0.5)
            elif op_type == "momentum":
                prefactor = 1 / (self.osc_lengths[index] * 2**0.5)
        if prefactor:
            native = self._kron_operator(
                inner_op(self.cutoffs_dict()[index], prefactor=prefactor), index
            )
        else:
            native = self._kron_operator(inner_op(self.cutoffs_dict()[index]), index)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    return operator_func
