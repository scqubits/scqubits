# test_circuit_characterization.py
# meant to be run with 'pytest'
#
# Pin the numerical and lifecycle behaviour of ``Circuit`` against committed
# golden ``.npy`` fixtures so behaviour-preserving refactors of the
# ``circuit_internals`` mixins (Tier 5b and beyond) can be verified without
# reasoning about every internal call path.
#
# To regenerate goldens after a deliberate numerical change, run::
#
#     SCQUBITS_REGENERATE_GOLDENS=1 python -m pytest \
#         --pyargs scqubits.tests.test_circuit_characterization
#
# and inspect the resulting diff before committing.
############################################################################
from __future__ import annotations

import json
import os
import warnings

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sparse

import scqubits as scq

GOLDENS_DIR = Path(__file__).parent / "characterization_goldens"
REGEN = os.environ.get("SCQUBITS_REGENERATE_GOLDENS") == "1"

# Tolerances calibrated for cross-platform numerical noise (Intel MKL vs.
# OpenBLAS). The pre-Tier-5b state on a single platform produces results
# tighter than these by 4-5 orders of magnitude, so any divergence at this
# tolerance signals a real algorithmic change, not BLAS noise.
RTOL_HAMILTONIAN = 1e-10
RTOL_EIGENVALS = 1e-10


# ----------------------------------------------------------------------
# Circuit fixtures
# ----------------------------------------------------------------------

TRANSMON_YAML = """branches:
- [JJ, 0, 1, EJ=15, ECJ=20]
- [C, 0, 1, EC=2]
"""

FLUXONIUM_YAML = """branches:
- [JJ, 0, 1, EJ=10, ECJ=20]
- [L, 0, 1, EL=0.04]
- [C, 0, 1, EC=2]
"""

ZERO_PI_YAML = """# zero-pi circuit
branches:
- ["JJ", 1, 2, 10, 20]
- ["JJ", 3, 4, 10, 20]
- ["L", 2, 3, 0.008]
- ["L", 4, 1, 0.008]
- ["C", 1, 3, 0.02]
- ["C", 2, 4, 0.02]
"""

COS2PHI_YAML = """branches:
- [JJ, 1, 2, EJ=10, ECJ=20]
- [JJ, 3, 4, EJ, ECJ]
- [L, 1, 4, EL=0.04]
- [L, 2, 3, EL]
- [C, 1, 3, EC=2]
"""

# Non-grounded LC + JJ between nodes 1 and 2 — exercises the sigma-mode
# centre-of-mass allocation that grounded fixtures never trigger.
UNGROUNDED_YAML = """branches:
- [JJ, 1, 2, EJ=12, ECJ=18]
- [L, 1, 2, EL=0.015]
- [C, 1, 2, EC=3]
"""

# Mutual-inductance ML coupler between two L branches — exercises the
# off-diagonal entry in the inductance matrix that no other fixture uses.
ML_COUPLED_YAML = """branches:
- [JJ, 0, 1, EJ=10, ECJ=20]
- [L, 0, 2, EL=0.01]
- [L, 1, 3, EL2=0.01]
- [ML, 1, 2, EML=0.003]
- [C, 0, 3, EC=2]
"""

# Higher-order Josephson junction (JJ2) — exercises the per-order EJ<k>
# parameter dict shape and the _junction_order machinery in
# branch_metadata.py.
JJ2_YAML = """branches:
- [JJ2, 0, 1, EJ=8, EJ2=0.5, ECJ=15]
- [C, 0, 1, EC=2.5]
"""


# Cutoffs deliberately minimised: characterization tests pin behaviour with
# small reproducible matrices, not production-scale spectra. Keeping the
# resulting golden ``.npy`` files small enough to commit (each < ~150 KB
# with these settings).


def _build_transmon():
    qubit = scq.Circuit.from_yaml_string(TRANSMON_YAML, ext_basis="discretized")
    qubit.cutoff_n_1 = 15
    return qubit


def _build_fluxonium():
    qubit = scq.Circuit.from_yaml_string(FLUXONIUM_YAML, ext_basis="discretized")
    qubit.cutoff_ext_1 = 40
    qubit.Φ1 = 0.3
    return qubit


def _build_zero_pi_hd():
    qubit = scq.Circuit.from_yaml_string(ZERO_PI_YAML, ext_basis="discretized")
    qubit.cutoff_n_1 = 12
    qubit.cutoff_ext_2 = 12
    qubit.cutoff_ext_3 = 60
    qubit.configure(system_hierarchy=[[1, 3], [2]], subsystem_trunc_dims=[12, 8])
    return qubit


def _build_cos2phi():
    qubit = scq.Circuit.from_yaml_string(COS2PHI_YAML, ext_basis="discretized")
    qubit.cutoff_n_1 = 2
    qubit.cutoff_ext_2 = 8
    qubit.cutoff_ext_3 = 8
    return qubit


def _build_ungrounded():
    qubit = scq.Circuit.from_yaml_string(UNGROUNDED_YAML, ext_basis="discretized")
    qubit.cutoff_ext_1 = 30
    return qubit


def _build_ml_coupled():
    qubit = scq.Circuit.from_yaml_string(ML_COUPLED_YAML, ext_basis="discretized")
    qubit.cutoff_n_1 = 5
    qubit.cutoff_ext_2 = 10
    return qubit


def _build_jj2():
    qubit = scq.Circuit.from_yaml_string(JJ2_YAML, ext_basis="discretized")
    qubit.cutoff_n_1 = 11
    return qubit


CIRCUIT_BUILDERS = {
    "transmon": _build_transmon,
    "fluxonium": _build_fluxonium,
    "zero_pi_hd": _build_zero_pi_hd,
    "cos2phi": _build_cos2phi,
    "ungrounded": _build_ungrounded,
    "ml_coupled": _build_ml_coupled,
    "jj2": _build_jj2,
}


# ----------------------------------------------------------------------
# Golden helpers
# ----------------------------------------------------------------------


def _golden_path(name: str, kind: str) -> Path:
    return GOLDENS_DIR / f"{name}__{kind}.npy"


def _hamiltonian_array(qubit) -> np.ndarray:
    """Return the dense array form of ``qubit.hamiltonian()``.

    The native return type can be a sparse matrix, a qutip Qobj, or an
    ndarray depending on basis configuration. Densify everything to a
    numpy ndarray so the goldens are version-stable across scipy / qutip
    upgrades.
    """
    H = qubit.hamiltonian()
    if sparse.issparse(H):
        return np.asarray(H.todense())
    if hasattr(H, "full"):  # qutip Qobj
        return H.full()
    return np.asarray(H)


def _summarize_for_sidecar(kind: str, value: np.ndarray) -> dict:
    """Produce a human-readable summary of ``value`` for JSON sidecar storage.

    The ``.npy`` files are the authoritative goldens; the ``.json`` sidecars
    exist so reviewers of numerical-change PRs can see what actually moved
    (e.g. "lowest eigenvalue 1.234 → 1.232") rather than reading a binary
    diff. Sidecars are NOT compared during tests — they are written
    alongside the ``.npy`` whenever the golden is regenerated.
    """
    if kind == "evals":
        evals = np.asarray(value).real
        return {
            "kind": "evals",
            "count": int(evals.size),
            "values": [round(float(x), 8) for x in evals],
            "gaps_from_ground": [
                round(float(x - evals[0]), 8) for x in evals
            ],
        }
    if kind == "hamiltonian":
        H = np.asarray(value)
        return {
            "kind": "hamiltonian",
            "shape": list(H.shape),
            "dtype": str(H.dtype),
            "trace": round(complex(np.trace(H)).real, 8),
            "frobenius_norm": round(float(np.linalg.norm(H)), 8),
            "max_abs": round(float(np.max(np.abs(H))), 8),
            "min_diag": round(float(np.min(np.diag(H).real)), 8),
            "max_diag": round(float(np.max(np.diag(H).real)), 8),
        }
    return {"kind": kind, "shape": list(np.shape(value))}


def _write_sidecar(npy_path: Path, kind: str, value: np.ndarray) -> None:
    sidecar_path = npy_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(_summarize_for_sidecar(kind, value), indent=2) + "\n",
        encoding="utf-8",
    )


def _check_or_save(name: str, kind: str, value: np.ndarray, *, rtol: float):
    path = _golden_path(name, kind)
    if REGEN or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        _write_sidecar(path, kind, value)
        if REGEN:
            pytest.skip(f"regenerated {path.name}")
        else:
            pytest.skip(f"created initial golden {path.name}; rerun to verify")
        return
    expected = np.load(path)
    # Backfill sidecar if a prior regeneration predates the sidecar feature.
    # The sidecar is informational only — it is NOT compared, just kept fresh
    # so reviewers always have a human-readable summary alongside the .npy.
    if not path.with_suffix(".json").exists():
        _write_sidecar(path, kind, expected)
    np.testing.assert_allclose(
        value,
        expected,
        rtol=rtol,
        atol=1e-15,
        err_msg=f"characterization mismatch for {name}/{kind} vs {path.name}",
    )


# ----------------------------------------------------------------------
# Numerical golden tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CIRCUIT_BUILDERS))
def test_hamiltonian_unchanged(name):
    """The numerical Hamiltonian matches the committed golden."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qubit = CIRCUIT_BUILDERS[name]()
    H = _hamiltonian_array(qubit)
    _check_or_save(name, "hamiltonian", H, rtol=RTOL_HAMILTONIAN)


@pytest.mark.parametrize("name", list(CIRCUIT_BUILDERS))
def test_eigenvals_unchanged(name):
    """Top-6 eigenvalues match the committed golden.

    Phase / sign of eigenvectors is not stable across runs for degenerate
    eigenvalues, so this only pins eigenvalues. The Hamiltonian-pin above
    plus the eigenvalue-pin together give strong coverage.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qubit = CIRCUIT_BUILDERS[name]()
    evals = qubit.eigenvals(evals_count=6)
    _check_or_save(name, "evals", evals, rtol=RTOL_EIGENVALS)


# ----------------------------------------------------------------------
# Lifecycle / dispatch tests
# ----------------------------------------------------------------------


class TestParameterDispatch:
    """Mutating a parameter must invalidate cached state and update eigenvals.

    This is the lifecycle counterpart to the numerical golden tests above.
    The goldens pin the matrix shape; these tests pin the dispatch chain
    that runs whenever a user mutates a watched property.
    """

    def test_external_flux_mutation_changes_eigenvals(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            qubit = _build_fluxonium()
        evals_at_03 = qubit.eigenvals(evals_count=4).copy()
        qubit.Φ1 = 0.5
        evals_at_05 = qubit.eigenvals(evals_count=4)
        assert not np.allclose(
            evals_at_03, evals_at_05
        ), "external-flux mutation did not invalidate cached eigenvals"

    def test_param_mutation_changes_hamiltonian(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            qubit = _build_transmon()
        H_at_15 = _hamiltonian_array(qubit).copy()
        qubit.EJ = 25.0
        H_at_25 = _hamiltonian_array(qubit)
        assert not np.allclose(
            H_at_15, H_at_25
        ), "EJ mutation did not invalidate cached Hamiltonian"

    def test_cutoff_mutation_changes_hilbertdim(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            qubit = _build_transmon()
        dim_before = qubit.hilbertdim()
        qubit.cutoff_n_1 = 30
        dim_after = qubit.hilbertdim()
        assert (
            dim_after != dim_before
        ), "cutoff mutation did not change Hilbert dimension"

    def test_hd_parent_cutoff_propagates_to_owning_subsystem(self):
        """Mutating a parent cutoff updates the cutoff on the subsystem that
        owns the corresponding variable. Pins the cutoff-mutation path of
        ``_set_property_and_update_cutoffs``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            qubit = _build_zero_pi_hd()
        # variable 2 lives on subsys 1 (per the [[1,3],[2]] hierarchy)
        owning = next(s for s in qubit.subsystems if 2 in s.dynamic_var_indices)
        assert owning.cutoff_ext_2 == qubit.cutoff_ext_2
        new_cutoff = qubit.cutoff_ext_2 * 2
        qubit.cutoff_ext_2 = new_cutoff
        assert (
            owning.cutoff_ext_2 == new_cutoff
        ), "parent cutoff_ext_2 mutation did not reach the owning subsystem"

    def test_hd_parent_flux_propagates_to_owning_subsystems(self):
        """Mutating a parent external flux updates the same flux attribute on
        every subsystem that carries that flux. Pins the dispatch path of
        ``_set_property_and_update_ext_flux_or_charge``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            qubit = _build_zero_pi_hd()
        if not qubit.external_fluxes:
            pytest.skip("fixture has no external flux to mutate")
        flux_name = qubit.external_fluxes[0].name
        # only subsystems whose own external_fluxes include this flux carry it
        carriers = [
            s
            for s in qubit.subsystems
            if any(f.name == flux_name for f in s.external_fluxes)
        ]
        assert (
            carriers
        ), "fixture invariant broken: no subsystem carries the parent flux"
        setattr(qubit, flux_name, 0.42)
        for subsys in carriers:
            assert (
                getattr(subsys, flux_name) == 0.42
            ), f"subsystem {subsys} did not receive parent's {flux_name} update"
