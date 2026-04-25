# circuit_plotting.py — backward-compatibility shim
#
# The real module now lives at ``scqubits.core.circuit_internals.plotting``. This file re-exports its
# public names so existing ``from scqubits.core.circuit_plotting import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``scqubits.core.circuit_internals.plotting``."""

from scqubits.core.circuit_internals.plotting import *  # noqa: F401, F403
