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

"""QuTiP 5.3.1+ / SciPy sparse-array compatibility.

QuTiP 5.3.1 switched ``Data.as_scipy()`` from ``spmatrix`` to ``sparray``.
These tests lock in both the conversion helper and the HilbertSpace path
that previously raised ``TypeError: Unsupported operator type: csc_array``.
"""

import numpy as np
import pytest
import qutip as qt

from scipy.sparse import csc_matrix

import scqubits as scq

from scqubits.utils.misc import Qobj_to_scipy_csc_matrix, as_csc_matrix, is_matrix_data
from scqubits.utils.spectrum_utils import convert_operator_to_qobj

try:
    from scipy.sparse import csc_array
except ImportError:  # SciPy < 1.8
    csc_array = None


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
        """Regression for qutip 5.3.1: HilbertSpace.generate_lookup used to crash."""
        tmon = scq.Transmon(EJ=5.0, EC=1.0, ng=0.0, ncut=5, truncated_dim=3)
        hilbertspace = scq.HilbertSpace([tmon])
        hilbertspace.generate_lookup()
        assert hilbertspace.lookup_exists()
        evals = hilbertspace.eigenvals(evals_count=tmon.truncated_dim)
        assert evals.shape[0] == tmon.truncated_dim
