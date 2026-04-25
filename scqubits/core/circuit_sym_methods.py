# circuit_sym_methods.py — backward-compatibility shim
#
# The real module now lives at ``scqubits.core.circuit_internals.sym_methods``. This file re-exports its
# public names so existing ``from scqubits.core.circuit_sym_methods import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``scqubits.core.circuit_internals.sym_methods``."""

from scqubits.core.circuit_internals.sym_methods import *  # noqa: F401, F403
