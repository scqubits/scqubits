# circuit_input.py — backward-compatibility shim
#
# The real module now lives at ``scqubits.core.circuit_internals.input``. This file re-exports its
# public names so existing ``from scqubits.core.circuit_input import X``
# imports keep working. Will be removed in a future major release.
"""Backward-compatibility shim; see ``scqubits.core.circuit_internals.input``."""

import warnings

warnings.warn(
    "scqubits.core.circuit_input is deprecated; import from "
    "scqubits.core.circuit_internals.input instead.",
    DeprecationWarning,
    stacklevel=2,
)

from scqubits.core.circuit_internals.input import *  # noqa: E402, F401, F403
