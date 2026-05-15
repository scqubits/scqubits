# test_diag_internals.py
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

import importlib.util

import numpy as np
import pytest
import qutip as qt

from scipy.sparse import csc_matrix, issparse

from scqubits.core.diag import (
    _cast_matrix,
    _convert_evecs_to_qobjs,
    _dict_merge,
    esys_cupy_dense,
    esys_cupy_sparse,
    esys_jax_dense,
    esys_primme_sparse,
    evals_cupy_dense,
    evals_cupy_sparse,
    evals_jax_dense,
    evals_primme_sparse,
)
from scqubits.io_utils.fileio_qutip import QutipEigenstates


def _installed(library: str) -> bool:
    return importlib.util.find_spec(library) is not None


class TestDictMerge:
    def test_adds_new_keys_from_other(self):
        result = _dict_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_default_does_not_overwrite_existing_keys(self):
        result = _dict_merge({"a": 1}, {"a": 99, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_overwrite_true_replaces_existing_keys(self):
        result = _dict_merge({"a": 1}, {"a": 99, "b": 2}, overwrite=True)
        assert result == {"a": 99, "b": 2}

    def test_exclude_skips_listed_keys(self):
        result = _dict_merge({"a": 1}, {"b": 2, "c": 3}, exclude=["c"])
        assert result == {"a": 1, "b": 2}

    def test_original_dicts_not_mutated(self):
        d = {"a": 1}
        d_other = {"b": 2}
        _dict_merge(d, d_other, overwrite=True)
        assert d == {"a": 1}
        assert d_other == {"b": 2}

    def test_nested_values_are_deep_copied(self):
        d = {"nested": [1, 2, 3]}
        result = _dict_merge(d, {})
        result["nested"].append(4)
        assert d["nested"] == [1, 2, 3]

    def test_empty_other_returns_copy_of_first(self):
        d = {"a": 1}
        result = _dict_merge(d, {})
        assert result == d
        assert result is not d


class TestCastMatrix:
    def test_ndarray_to_dense_returns_ndarray(self):
        a = np.eye(3)
        result = _cast_matrix(a, "dense")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, a)

    def test_ndarray_to_sparse_returns_csc(self):
        a = np.eye(3)
        result = _cast_matrix(a, "sparse")
        assert isinstance(result, csc_matrix)
        assert np.allclose(result.toarray(), a)

    def test_csc_to_dense_returns_ndarray(self):
        m = csc_matrix(np.eye(3))
        result = _cast_matrix(m, "dense")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.eye(3))

    def test_csc_to_sparse_returns_csc(self):
        m = csc_matrix(np.eye(3))
        result = _cast_matrix(m, "sparse")
        assert isinstance(result, csc_matrix)

    def test_qobj_to_dense_returns_ndarray(self):
        q = qt.sigmax()
        result = _cast_matrix(q, "dense")
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, q.full())

    def test_qobj_to_sparse_returns_csc(self):
        q = qt.sigmax()
        result = _cast_matrix(q, "sparse")
        assert issparse(result)
        assert np.allclose(result.toarray(), q.full())

    def test_invalid_cast_target_raises(self):
        with pytest.raises(ValueError, match="'sparse' or 'dense'"):
            _cast_matrix(np.eye(2), "banana")

    def test_force_cast_false_passes_ndarray_through(self):
        a = np.eye(3)
        result = _cast_matrix(a, "sparse", force_cast=False)
        # With force_cast=False, a non-Qobj ndarray is not coerced to sparse.
        assert isinstance(result, np.ndarray)


class TestConvertEvecsToQobjs:
    @pytest.fixture
    def setup(self):
        matrix_qobj = qt.tensor(qt.sigmaz(), qt.qeye(2))
        evals, evecs = np.linalg.eigh(matrix_qobj.full())
        return matrix_qobj, evecs

    def test_returns_object_array_of_length_evecs_count(self, setup):
        matrix_qobj, evecs = setup
        result = _convert_evecs_to_qobjs(evecs, matrix_qobj)
        assert isinstance(result, np.ndarray)
        assert result.shape == (evecs.shape[1],)

    def test_each_entry_is_normalized_qobj(self, setup):
        matrix_qobj, evecs = setup
        result = _convert_evecs_to_qobjs(evecs, matrix_qobj)
        for v in result:
            assert isinstance(v, qt.Qobj)
            assert np.isclose(v.norm(), 1.0)

    def test_eigenvector_dims_match_matrix_tensor_structure(self, setup):
        matrix_qobj, evecs = setup
        result = _convert_evecs_to_qobjs(evecs, matrix_qobj)
        assert result[0].dims[0] == matrix_qobj.dims[0]

    def test_wrap_true_returns_qutip_eigenstates(self, setup):
        matrix_qobj, evecs = setup
        result = _convert_evecs_to_qobjs(evecs, matrix_qobj, wrap=True)
        assert isinstance(result, QutipEigenstates)


@pytest.mark.skipif(
    _installed("primme"),
    reason="primme installed; ImportError path only exists without it",
)
class TestPrimmeImportErrorPath:
    def test_evals_primme_sparse_raises(self):
        with pytest.raises(ImportError, match="primme is not installed"):
            evals_primme_sparse(np.eye(3), 2)

    def test_esys_primme_sparse_raises(self):
        with pytest.raises(ImportError, match="primme is not installed"):
            esys_primme_sparse(np.eye(3), 2)


@pytest.mark.skipif(
    _installed("cupy"), reason="cupy installed; ImportError path only exists without it"
)
class TestCupyImportErrorPath:
    def test_evals_cupy_dense_raises(self):
        with pytest.raises(ImportError, match="cupy is not installed"):
            evals_cupy_dense(np.eye(3), 2)

    def test_esys_cupy_dense_raises(self):
        with pytest.raises(ImportError, match="cupy is not installed"):
            esys_cupy_dense(np.eye(3), 2)

    def test_evals_cupy_sparse_raises(self):
        with pytest.raises(ImportError, match="cupyx"):
            evals_cupy_sparse(np.eye(3), 2)

    def test_esys_cupy_sparse_raises(self):
        with pytest.raises(ImportError, match="cupyx"):
            esys_cupy_sparse(np.eye(3), 2)


@pytest.mark.skipif(
    _installed("jax"), reason="jax installed; ImportError path only exists without it"
)
class TestJaxImportErrorPath:
    def test_evals_jax_dense_raises(self):
        with pytest.raises(ImportError, match="jax is not installed"):
            evals_jax_dense(np.eye(3), 2)

    def test_esys_jax_dense_raises(self):
        with pytest.raises(ImportError, match="jax is not installed"):
            esys_jax_dense(np.eye(3), 2)
