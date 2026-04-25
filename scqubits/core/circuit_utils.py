# circuit_utils.py — backward-compatibility shim
#
# The real module now lives at ``scqubits.core.circuit_internals.utils``. This file re-exports its
# public names so existing ``from scqubits.core.circuit_utils import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``scqubits.core.circuit_internals.utils``."""

from scqubits.core.circuit_internals.utils import *  # noqa: F401, F403
