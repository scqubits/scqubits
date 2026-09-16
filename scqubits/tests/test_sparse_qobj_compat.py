# test_sparse_qobj_compat.py
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

"""SciPy sparse-array handling in Qobj conversion and HilbertSpace."""

from typing import Any

import numpy as np
import pytest
import qutip as qt
import scipy.sparse as sparse

from scipy.sparse import csc_matrix

import scqubits as scq

from scqubits.utils.misc import Qobj_to_scipy_csc_matrix, as_csc_matrix, is_matrix_data
from scqubits.utils.spectrum_utils import convert_operator_to_qobj

# Optional: SciPy < 1.8 has no csc_array. Typed as Any so mypy accepts None.
csc_array: Any = getattr(sparse, "csc_array", None)


class TestSparseQobjCompat:
    def test_is_matrix_data_accepts_ndarray_and_sparse(self):
        dense = np.eye(2)
        matrix = csc_matrix(dense)
        assert is_matrix_data(dense)
        assert is_matrix_data(matrix)
        assert not is_matrix_data("n_operator")
        assert not is_matrix_data(qt.qeye(2))
        if csc_array is not None:
            assert is_matrix_data(csc_array(dense))
            coerced = as_csc_matrix(csc_array(dense))
            assert isinstance(coerced, csc_matrix)
            assert np.allclose(coerced.toarray(), dense)

    def test_qobj_to_scipy_csc_matrix_returns_csc_matrix(self):
        result = Qobj_to_scipy_csc_matrix(qt.qeye(3))
        assert isinstance(result, csc_matrix)
        assert not type(result).__name__.endswith("array")
        assert np.allclose(result.toarray(), np.eye(3))

    @pytest.mark.skipif(
        csc_array is None, reason="scipy.sparse.csc_array not available"
    )
    def test_convert_operator_to_qobj_accepts_csc_array(self):
        tmon = scq.Transmon(EJ=5.0, EC=1.0, ng=0.0, ncut=5, truncated_dim=3)
        diag = np.diag([0.0, 1.0, 2.0])
        qobj = convert_operator_to_qobj(
            csc_array(diag), tmon, op_in_eigenbasis=True, evecs=None
        )
        assert isinstance(qobj, qt.Qobj)
        assert np.allclose(qobj.full(), diag)

    def test_hilbertspace_generate_lookup_single_transmon(self):
        """HilbertSpace.generate_lookup on a single Transmon."""
        tmon = scq.Transmon(EJ=5.0, EC=1.0, ng=0.0, ncut=5, truncated_dim=3)
        hilbertspace = scq.HilbertSpace([tmon])
        hilbertspace.generate_lookup()
        assert hilbertspace.lookup_exists()
        evals = hilbertspace.eigenvals(evals_count=tmon.truncated_dim)
        assert evals.shape[0] == tmon.truncated_dim
