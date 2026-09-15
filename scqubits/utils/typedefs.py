# typedefs.py
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

from typing import TYPE_CHECKING, Union

from numpy import ndarray
from scipy.sparse import csc_matrix, dia_matrix

if TYPE_CHECKING:
    from scqubits import GenericQubit, KerrOscillator, Oscillator
    from scqubits.core.qubit_base import QuantumSystem, QubitBaseClass

QuantumSys = Union["QubitBaseClass", "Oscillator", "KerrOscillator", "GenericQubit"]

OscillatorList = list["Oscillator"]
QubitList = list[Union["QubitBaseClass", "GenericQubit", "KerrOscillator"]]

OperatorSpecification = Union[str, ndarray, csc_matrix, dia_matrix]

NpIndex = Union[int, slice, tuple[int], list[int]]
NpIndexTuple = tuple[NpIndex, ...]
NpIndices = Union[NpIndex, NpIndexTuple]

GIndex = Union[int, float, complex, slice, tuple[int], list[int]]
GIndexTuple = tuple[GIndex, ...]
