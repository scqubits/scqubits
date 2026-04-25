# circuit_noise.py — backward-compatibility shim
#
# The real module now lives at ``scqubits.core.circuit_internals.noise``. This file re-exports its
# public names so existing ``from scqubits.core.circuit_noise import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``scqubits.core.circuit_internals.noise``."""

from scqubits.core.circuit_internals.noise import *  # noqa: F401, F403
