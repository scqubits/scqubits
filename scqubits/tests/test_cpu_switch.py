# test_cpu_switch.py
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
import warnings

import pytest

import scqubits.settings as settings
import scqubits.utils.cpu_switch as cpu_switch

from scqubits.utils.cpu_switch import (
    _BLAS_THREAD_ENV_VARS,
    _auto_blas_cap,
    _capped_blas_threads,
    _effective_blas_cap,
    _resolve_start_method,
    _validated_blas_thread_cap,
)


class TestValidatedBlasThreadCap:
    def test_none_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", None)
        assert _validated_blas_thread_cap() is None

    def test_auto_passes_through(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", "auto")
        assert _validated_blas_thread_cap() == "auto"

    def test_positive_int_passes_through(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", 3)
        assert _validated_blas_thread_cap() == 3

    @pytest.mark.parametrize("bad", [1.5, "2", "foo", [1], 2.0])
    def test_non_int_rejected(self, monkeypatch, bad):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", bad)
        with pytest.raises(TypeError):
            _validated_blas_thread_cap()

    def test_bool_rejected(self, monkeypatch):
        # bool is an int subclass; True must not be silently treated as 1
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", True)
        with pytest.raises(TypeError):
            _validated_blas_thread_cap()

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_rejected(self, monkeypatch, bad):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", bad)
        with pytest.raises(ValueError):
            _validated_blas_thread_cap()


class TestEffectiveBlasCap:
    def test_none_setting_disables_cap(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", None)
        assert _effective_blas_cap(num_cpus=4, blas_threads=None) is None

    def test_int_setting_used_as_is(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", 2)
        assert _effective_blas_cap(num_cpus=8, blas_threads=None) == 2

    def test_auto_setting_splits_cores_across_workers(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", "auto")
        monkeypatch.setattr(cpu_switch.os, "cpu_count", lambda: 8)
        assert _effective_blas_cap(num_cpus=4, blas_threads=None) == 2

    def test_auto_never_drops_below_one(self, monkeypatch):
        # more workers than cores must still leave each worker at least 1 thread
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", "auto")
        monkeypatch.setattr(cpu_switch.os, "cpu_count", lambda: 4)
        assert _effective_blas_cap(num_cpus=16, blas_threads=None) == 1

    def test_override_takes_precedence_over_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", "auto")
        assert _effective_blas_cap(num_cpus=4, blas_threads=3) == 3

    def test_override_bool_rejected(self, monkeypatch):
        # bool is an int subclass; True must not slip through as 1
        with pytest.raises(TypeError):
            _effective_blas_cap(num_cpus=4, blas_threads=True)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_override_non_positive_rejected(self, monkeypatch, bad):
        with pytest.raises(ValueError):
            _effective_blas_cap(num_cpus=4, blas_threads=bad)

    def test_auto_cap_helper(self, monkeypatch):
        monkeypatch.setattr(cpu_switch.os, "cpu_count", lambda: 12)
        assert _auto_blas_cap(num_cpus=4) == 3


class TestCappedBlasThreads:
    def test_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        with _capped_blas_threads(None):
            assert "OPENBLAS_NUM_THREADS" not in os.environ

    def test_sets_cap_inside_and_removes_after_when_unset_before(self, monkeypatch):
        for var in _BLAS_THREAD_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        with _capped_blas_threads(2):
            for var in _BLAS_THREAD_ENV_VARS:
                assert os.environ[var] == "2"
        # vars that were unset before the block must be removed again
        for var in _BLAS_THREAD_ENV_VARS:
            assert var not in os.environ

    def test_restores_preexisting_value(self, monkeypatch):
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
        with _capped_blas_threads(1):
            assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        # a value present before the block must be restored, not deleted
        assert os.environ["OPENBLAS_NUM_THREADS"] == "7"

    def test_restores_environment_on_exception(self, monkeypatch):
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        with pytest.raises(RuntimeError):
            with _capped_blas_threads(1):
                raise RuntimeError("boom")
        assert "OPENBLAS_NUM_THREADS" not in os.environ


class TestShutdownCachedPool:
    def test_noop_when_no_pool(self, monkeypatch):
        monkeypatch.setattr(settings, "POOL", None)
        cpu_switch._shutdown_cached_pool()  # must not raise
        assert settings.POOL is None

    def test_shuts_down_and_clears_cached_pool(self, monkeypatch):
        class FakePool:
            def __init__(self):
                self.terminated = False

            def terminate(self):
                self.terminated = True

            def close(self):
                pass

            def clear(self):
                pass

        pool = FakePool()
        monkeypatch.setattr(settings, "POOL", pool)
        cpu_switch._shutdown_cached_pool()
        assert pool.terminated is True
        assert settings.POOL is None


class TestShutdownPool:
    def test_cleanup_failure_is_logged_not_raised(self, caplog):
        class BadPool:
            def terminate(self):
                raise RuntimeError("boom")

            def close(self):
                pass

            def clear(self):
                pass

        with caplog.at_level("WARNING"):
            cpu_switch._shutdown_pool(BadPool())  # must not raise
        assert any("cleanup" in record.message for record in caplog.records)


class TestGetMapMethodSerial:
    def test_num_cpus_one_returns_builtin_map(self):
        assert cpu_switch.get_map_method(1) is map


class TestImapChunksize:
    class _FakePool:
        def imap(self, func, iterable, chunksize=1):
            return ("imap", chunksize)

    def test_total_none_uses_default_chunksize(self):
        mapper = cpu_switch._imap_with_chunksize(self._FakePool(), 4, None)
        assert mapper("f", "it")[1] == 1  # imap's own default

    def test_chunksize_matches_map_heuristic(self):
        # pool.map uses ceil(total / (4 * num_cpus)); 384 / 16 -> 24
        mapper = cpu_switch._imap_with_chunksize(self._FakePool(), 4, 384)
        assert mapper("f", "it")[1] == 24

    def test_chunksize_floored_at_one(self):
        mapper = cpu_switch._imap_with_chunksize(self._FakePool(), 4, 3)
        assert mapper("f", "it")[1] == 1


class TestPoolPickleReduction:
    # dill recurse can pull settings.POOL into a worker task (e.g. for circuits);
    # raw multiprocess pools must reduce to None rather than raise on pickle.
    def test_reduction_reconstructs_none(self):
        func, args = cpu_switch._pool_reduces_to_none(object())
        assert func(*args) is None

    def test_pool_classes_registered_in_copyreg(self):
        import copyreg

        from multiprocess.pool import Pool as MpPool

        cpu_switch._register_pool_pickle_reduction()
        assert copyreg.dispatch_table.get(MpPool) is cpu_switch._pool_reduces_to_none


class TestResolveStartMethod:
    def test_resolves_to_available_method(self):
        assert (
            _resolve_start_method()
            in cpu_switch._backend_module().get_all_start_methods()
        )

    def test_platform_default(self):
        import sys

        method = _resolve_start_method()
        if sys.platform.startswith("linux"):
            assert method == "fork"
        else:
            # macOS / Windows use spawn (fork-after-threads is unsafe on macOS)
            assert method == "spawn"


class TestSpawnGuardWarning:
    def test_warns_once_under_spawn_non_ipython(self, monkeypatch):
        monkeypatch.setattr(settings, "IN_IPYTHON", False)
        monkeypatch.setattr(cpu_switch, "_spawn_guard_warned", False)
        with pytest.warns(UserWarning, match="__main__"):
            cpu_switch._warn_spawn_guard("spawn")
        # second call is a no-op (no warning)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cpu_switch._warn_spawn_guard("spawn")

    def test_silent_under_ipython(self, monkeypatch):
        monkeypatch.setattr(settings, "IN_IPYTHON", True)
        monkeypatch.setattr(cpu_switch, "_spawn_guard_warned", False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cpu_switch._warn_spawn_guard("spawn")

    def test_silent_under_fork(self, monkeypatch):
        monkeypatch.setattr(settings, "IN_IPYTHON", False)
        monkeypatch.setattr(cpu_switch, "_spawn_guard_warned", False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cpu_switch._warn_spawn_guard("fork")


class TestPoolReuseSignature:
    def test_reusable_matches_signature(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC", "pathos")
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", None)
        method = _resolve_start_method()
        monkeypatch.setattr(cpu_switch, "_pool_signature", ("pathos", method, 4, None))
        assert cpu_switch._pool_is_reusable(object(), 4) is True
        assert cpu_switch._pool_is_reusable(object(), 2) is False  # different cpu count

    def test_not_reusable_with_different_blas_cap(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTIPROC", "pathos")
        monkeypatch.setattr(settings, "MULTIPROC_BLAS_THREADS", None)
        method = _resolve_start_method()
        monkeypatch.setattr(cpu_switch, "_pool_signature", ("pathos", method, 4, 1))
        # same num_cpus but the cached pool was built with a BLAS cap of 1
        assert cpu_switch._pool_is_reusable(object(), 4, blas_threads=1) is True
        assert cpu_switch._pool_is_reusable(object(), 4, blas_threads=2) is False
        assert cpu_switch._pool_is_reusable(object(), 4) is False  # cap None != 1

    def test_not_reusable_when_unset(self, monkeypatch):
        monkeypatch.setattr(cpu_switch, "_pool_signature", None)
        assert cpu_switch._pool_is_reusable(object(), 4) is False
