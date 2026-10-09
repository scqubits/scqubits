# test_cuquantum.py
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

import sys

import numpy as np
import pytest
import qutip as qt

import scqubits.settings as settings
import scqubits.utils.cuquantum_utils as cuquantum_utils
import scqubits.utils.spectrum_utils as spec_utils

from scqubits import HilbertSpace, Oscillator, Transmon
from scqubits.core import diag
from scqubits.io_utils.fileio_qutip import QutipEigenstates

EVALS_COUNT = 6


def _coupled_hilbertspace(**kwargs):
    """Transmon coupled to an oscillator: dimension 160, lowest levels well separated."""
    tmon = Transmon(EJ=30.02, EC=1.2, ng=0.3, ncut=40, truncated_dim=8)
    osc = Oscillator(E_osc=6.0, truncated_dim=20)
    hilbertspace = HilbertSpace([tmon, osc], **kwargs)
    hilbertspace.add_interaction(
        g=0.1, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True
    )
    return hilbertspace


@pytest.fixture(scope="module")
def workstream():
    """Shared cuQuantum workstream; skips the test where cuQuantum cannot run."""
    for package in ("cupy", "cuquantum.densitymat", "qutip_cuquantum"):
        pytest.importorskip(package)
    try:
        return cuquantum_utils.get_cuquantum_workstream()
    except Exception as exc:
        pytest.skip(f"cuQuantum workstream could not be created: {exc}")


# The tests in the following classes need neither a GPU nor cuQuantum.


class TestMaxEigvals:
    @pytest.mark.parametrize(
        "dimension, block, ratio, expected",
        [(1600, 1, 5, 159), (160, 1, 5, 15), (100, 2, 5, 4), (11, 1, 5, 1)],
    )
    def test_limit(self, monkeypatch, dimension, block, ratio, expected):
        monkeypatch.setattr(settings, "CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE", block)
        monkeypatch.setattr(settings, "CUQUANTUM_MAX_BUFFER_RATIO", ratio)
        assert cuquantum_utils.max_eigvals(dimension) == expected

    @pytest.mark.parametrize(
        "name, value",
        [
            ("CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE", 0),
            ("CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE", -1),
            ("CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE", 1.5),
            ("CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE", True),
            ("CUQUANTUM_MAX_BUFFER_RATIO", 0),
            ("CUQUANTUM_MAX_BUFFER_RATIO", 1),
            ("CUQUANTUM_MAX_BUFFER_RATIO", 2.5),
            ("CUQUANTUM_MAX_RESTARTS", -1),
            ("CUQUANTUM_MAX_RESTARTS", None),
        ],
    )
    def test_invalid_setting_raises(self, monkeypatch, name, value):
        monkeypatch.setattr(settings, name, value)
        with pytest.raises(ValueError, match=name):
            cuquantum_utils.max_eigvals(1600)

    def test_dimension_too_small_raises(self):
        with pytest.raises(ValueError, match="dimension is too small"):
            cuquantum_utils.max_eigvals(10)


class TestWorkstreamSingleton:
    def test_set_then_get_returns_same_object(self, monkeypatch):
        monkeypatch.setattr(cuquantum_utils, "_cuquantum_workstream", None)
        stand_in = object()
        cuquantum_utils.set_cuquantum_workstream(stand_in)
        assert cuquantum_utils.get_cuquantum_workstream() is stand_in

    def test_second_set_raises(self, monkeypatch):
        monkeypatch.setattr(cuquantum_utils, "_cuquantum_workstream", object())
        with pytest.raises(RuntimeError, match="already set"):
            cuquantum_utils.set_cuquantum_workstream(object())

    def test_get_without_cuquantum_raises(self, monkeypatch):
        monkeypatch.setattr(cuquantum_utils, "_cuquantum_workstream", None)
        monkeypatch.setattr(cuquantum_utils, "_HAS_CUQUANTUM", False)
        with pytest.raises(ImportError, match="cuDensityMat could not be imported"):
            cuquantum_utils.get_cuquantum_workstream()


class TestCuQuantumArgumentErrors:
    @pytest.mark.parametrize("method", [diag.evals_cuquantum, diag.esys_cuquantum])
    def test_unexpected_keyword_raises(self, method):
        with pytest.raises(TypeError, match="unexpected keyword arguments: 'tol'"):
            method(qt.qeye(12), 1, tol=1e-12)

    @pytest.mark.parametrize("method", [diag.evals_cuquantum, diag.esys_cuquantum])
    def test_missing_cupy_raises(self, monkeypatch, method):
        monkeypatch.setitem(sys.modules, "cupy", None)
        with pytest.raises(ImportError, match="cuQuantum eigensolvers require"):
            method(qt.qeye(12), 1)

    def test_identity_wrap_missing_qutip_cuquantum_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "qutip_cuquantum", None)
        osc = Oscillator(E_osc=6.0, truncated_dim=4)
        with pytest.raises(ImportError, match="qutip-cuquantum is required"):
            spec_utils.identity_wrap(
                osc.annihilation_operator(), osc, [osc], use_cuquantum=True
            )


class TestCuQuantumGuards:
    @pytest.mark.parametrize("method", ["evals_cuquantum", diag.evals_cuquantum])
    def test_qubit_eigenvals_raises(self, method):
        tmon = Transmon(EJ=30.02, EC=1.2, ng=0.3, ncut=40, evals_method=method)
        with pytest.raises(NotImplementedError, match="qubit eigenvals"):
            tmon.eigenvals()

    @pytest.mark.parametrize("method", ["esys_cuquantum", diag.esys_cuquantum])
    def test_qubit_eigensys_raises(self, method):
        tmon = Transmon(EJ=30.02, EC=1.2, ng=0.3, ncut=40, esys_method=method)
        with pytest.raises(NotImplementedError, match="qubit eigensys"):
            tmon.eigensys()

    @pytest.mark.parametrize("ordering", ["DE", "LX"])
    def test_generate_lookup_ordering_raises_and_leaves_no_lookup(self, ordering):
        hilbertspace = _coupled_hilbertspace(esys_method="esys_cuquantum")
        with pytest.raises(ValueError, match="DE or LX ordering"):
            hilbertspace.generate_lookup(ordering=ordering)
        assert not hilbertspace._lookup_exists


# The tests in the following class run the solver and need a GPU with cuQuantum.


class TestCuQuantumSolver:
    def test_eigenvals_match_dense(self, workstream):
        evals_dense = _coupled_hilbertspace().eigenvals(evals_count=EVALS_COUNT)
        hilbertspace = _coupled_hilbertspace(evals_method="evals_cuquantum")
        evals = hilbertspace.eigenvals(evals_count=EVALS_COUNT)
        assert np.allclose(evals, evals_dense)

    def test_eigensys_matches_dense(self, workstream):
        evals_dense, evecs_dense = _coupled_hilbertspace().eigensys(
            evals_count=EVALS_COUNT
        )
        hilbertspace = _coupled_hilbertspace(esys_method="esys_cuquantum")
        evals, evecs = hilbertspace.eigensys(evals_count=EVALS_COUNT)

        assert np.allclose(evals, evals_dense)
        assert isinstance(evecs, QutipEigenstates)
        assert len(evecs) == EVALS_COUNT
        # Eigenvectors are only defined up to a phase, so compare overlaps.
        overlaps = [
            abs(evec_dense.overlap(evec))
            for evec_dense, evec in zip(evecs_dense, evecs)
        ]
        assert np.allclose(overlaps, 1.0)

    def test_hamiltonian_wraps_cuoperator(self, workstream):
        import qutip_cuquantum as qcu

        hilbertspace = _coupled_hilbertspace()
        assert isinstance(
            hilbertspace.hamiltonian(use_cuquantum=True).data, qcu.CuOperator
        )
        assert not isinstance(hilbertspace.hamiltonian().data, qcu.CuOperator)

    def test_too_many_eigenvalues_raises(self, workstream):
        hilbertspace = _coupled_hilbertspace(evals_method="evals_cuquantum")
        limit = cuquantum_utils.max_eigvals(hilbertspace.dimension)
        with pytest.raises(ValueError, match="Too many eigenvalues requested"):
            hilbertspace.eigenvals(evals_count=limit + 1)

    def test_generate_lookup_bare_energy_ordering(self, workstream):
        reference = _coupled_hilbertspace()
        reference.generate_lookup(ordering="BE", BEs_count=EVALS_COUNT)
        hilbertspace = _coupled_hilbertspace(esys_method="esys_cuquantum")
        hilbertspace.generate_lookup(ordering="BE", BEs_count=EVALS_COUNT)
        for index in range(EVALS_COUNT):
            assert np.isclose(
                hilbertspace.energy_by_dressed_index(index),
                reference.energy_by_dressed_index(index),
            )

    @pytest.mark.parametrize("method", ["eigenvals", "eigensys", "generate_lookup"])
    def test_call_inside_cuquantum_backend_raises(self, workstream, method):
        import qutip_cuquantum as qcu

        hilbertspace = _coupled_hilbertspace()
        with qcu.CuQuantumBackend(workstream):
            with pytest.raises(RuntimeError, match="CuQuantumBackend is not supported"):
                getattr(hilbertspace, method)()
