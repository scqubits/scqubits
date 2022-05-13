# test_circuit.py
# meant to be run with 'pytest'
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

import os
import numpy as np
import scqubits as scq
from scqubits import DataStore, SpectrumData
from scqubits import Circuit, SymbolicCircuit

TESTDIR, _ = os.path.split(scq.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")
DATADIR = os.path.join(TESTDIR, "data", "")


qubits_tested = ["fluxonium", "zeropi"]
reference_data = dict.fromkeys(qubits_tested)
qubits = dict.fromkeys(qubits_tested)
for qubit_name in qubits_tested:
    data_store = SpectrumData.create_from_file(
        DATADIR + "circuit_" + qubit_name + ".h5"
    )
    reference_data[qubit_name] = data_store.system_params

    qubits[qubit_name] = Circuit.from_yaml(
        reference_data[qubit_name]["input_string"],
        is_file=False,
        ext_basis=reference_data[qubit_name]["phi_basis"],
    )
    if "subsystem_indices" in reference_data[qubit_name]:
        qubits[qubit_name].set_system_hierarchy(
            system_hierarchy=reference_data[qubit_name]["subsystem_indices"],
            subsystem_trunc_dims=reference_data[qubit_name]["subsystem_trunc_dims"],
        )

    for attrib in reference_data[qubit_name]["extra_attribs"]:
        setattr(
            qubits[qubit_name],
            attrib,
            reference_data[qubit_name]["extra_attribs"][attrib],
        )

########## tests #######
def test_eigenvals():
    for qubit_name in qubits_tested:
        evals_ref = qubits[qubit_name].eigenvals()
        evals_test = reference_data[qubit_name]["eigenvals"]
        assert np.allclose(evals_ref, evals_test)


def test_eigenvecs():
    for qubit_name in qubits_tested:
        evecs_ref = reference_data[qubit_name]["eigensys"][1]
        evals_count = evecs_ref.shape[1]
        evecs_test = qubits[qubit_name].eigensys(evals_count=evals_count)[1]
        assert np.allclose(np.abs(evecs_ref), np.abs(evecs_test))
