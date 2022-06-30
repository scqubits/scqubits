# circuit_noise.py
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

from abc import ABC, abstractmethod

from scqubits.core.noise import NoisySystem
from scqubits.core.circuit import Circuit, Subsystem

from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy as sm

class NoisyCircuit(NoisySystem, ABC):

    def __init__(self, circuit: Subsystem) -> None:
        # initiating the methods required for noise estimation dynamically

        self.circuit = circuit # storing the Circuit class as an attribute

        # methods for 1/f flux noise
        for flux in circuit.external_fluxes:
            hamiltonian = 

    def hamiltonian_sym_for_numerics(self):
        hamiltonian = self.circuit.hamiltonian_symbolic
        hamiltonian = self.circuit._shift_harmonic_oscillator_potential(hamiltonian)

        # removing the constants from the Hamiltonian
        ordered_terms = hamiltonian.as_ordered_terms()
        constants = [
            term
            for term in ordered_terms
            if (
                set(
                    self.external_fluxes
                    + self.offset_charges
                    + list(self.symbolic_params.keys())
                    + [sm.symbols("I")]
                )
                & set(term.free_symbols)
            )
            == set(term.free_symbols)
        ]
        self._constant_terms_in_hamiltonian = constants
        for const in constants:
            hamiltonian -= const

        # associate an identity matrix with the external flux vars
        for ext_flux in self.external_fluxes:
            hamiltonian = hamiltonian.subs(
                ext_flux, ext_flux * sm.symbols("I") * 2 * np.pi
            )
        # associate an identity matrix with offset charge vars
        for offset_charge in self.offset_charges:
            hamiltonian = hamiltonian.subs(
                offset_charge, offset_charge * sm.symbols("I")
            )
        # finding the cosine terms
        cos_terms = sum(
            [term for term in hamiltonian.as_ordered_terms() if "cos" in str(term)]
        )
        setattr(self, "_hamiltonian_sym_for_numerics", hamiltonian)
        setattr(self, "junction_potential", cos_terms)



    def d_hamiltonian_d_param_function_factory(self, params: List):
        hamiltonian = self.circuit.hamiltonian_symbolic
