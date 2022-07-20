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

    def supported_noise_channels(self) -> List[str]:
        """Return a list of supported noise channels"""
        return ['tphi_1_over_f_cc', 
                'tphi_1_over_f_flux',
                't1_capacitive',
                't1_charge_impedance', 
                't1_flux_bias_line',
                't1_inductive',
                't1_quasiparticle_tunneling']

    def generate_methods_1_over_f_flux(self: Subsystem) -> List[Callable]:
        hamiltonian = self.hamiltonian_symbolic
        ext_flux_methods = []
        for ext_flux_sym in self.external_fluxes:
            diff_sym_expression = hamiltonian.diff(ext_flux_sym)
            

             
    def d_hamiltonian_d_param_function_factory(self, params: List):
        hamiltonian = self.circuit.hamiltonian_symbolic
