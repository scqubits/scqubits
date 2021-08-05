# typedefs.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# arXiv:2107.08552 (2021). https://arxiv.org/abs/2107.08552
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from typing import TYPE_CHECKING, List, Tuple, Union

from numpy import ndarray
from scipy.sparse import csc_matrix, dia_matrix

if TYPE_CHECKING:
    from scqubits import GenericQubit, KerrOscillator, Oscillator
    from scqubits.core.qubit_base import QubitBaseClass

QuantumSys = Union["QubitBaseClass", "Oscillator", "KerrOscillator", "GenericQubit"]

OscillatorList = List["Oscillator"]
QubitList = List[Union["QubitBaseClass", "GenericQubit", "KerrOscillator"]]

OperatorSpecification = Union[str, ndarray, csc_matrix, dia_matrix]

NpIndex = Union[int, slice, Tuple[int], List[int]]
NpIndexTuple = Tuple[NpIndex, ...]
NpIndices = Union[NpIndex, NpIndexTuple]

GIndex = Union[int, float, complex, slice, Tuple[int], List[int]]
GIndexTuple = Tuple[GIndex, ...]
