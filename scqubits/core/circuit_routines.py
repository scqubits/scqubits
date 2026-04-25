# circuit_routines.py — backward-compatibility shim
#
# The real module now lives at ``scqubits.core.circuit_internals.routines``. This file re-exports its
# public names so existing ``from scqubits.core.circuit_routines import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``scqubits.core.circuit_internals.routines``."""

from scqubits.core.circuit_internals.routines import *  # noqa: F401, F403
