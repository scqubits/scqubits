"""Regression tests for artifact selection and installed-package CI guards."""

import os
import subprocess
import tarfile
import tempfile
import unittest

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ci_package


class PackageCITest(unittest.TestCase):
    def test_rejects_multiple_conda_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "noarch").mkdir()
            for version in ("4.3.1", "5.0.0"):
                (output / "noarch" / f"scqubits-{version}-py_0.conda").touch()
            with self.assertRaisesRegex(RuntimeError, "exactly one Conda artifact"):
                ci_package.conda_install(output, "3.14")

    def test_conda_solve_uses_exact_local_build_and_python(self):
        with tempfile.TemporaryDirectory(prefix="conda output ") as directory:
            output = Path(directory)
            (output / "noarch").mkdir()
            (output / "noarch" / "scqubits-5.0.0-py_0.conda").touch()
            with patch.object(ci_package, "run") as run:
                ci_package.conda_install(output, "3.14")
            command = run.call_args.args
            self.assertIn("python=3.14", command)
            self.assertIn(f"{output.resolve().as_uri()}::scqubits=5.0.0=py_0", command)
            self.assertIn("--strict-channel-priority", command)

    def test_rejects_python_downgrade(self):
        with patch.object(ci_package.sys, "version_info", (3, 12)):
            with self.assertRaisesRegex(RuntimeError, "Expected Python 3.14"):
                ci_package.test_installed("3.14", False, None)

    def test_conda_uses_activation_executable_without_path_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "noarch").mkdir()
            (output / "noarch" / "scqubits-5.0.0-py_0.conda").touch()
            executable = r"C:\Program Files\Miniforge3\Scripts\conda.exe"
            with patch.dict(os.environ, {"CONDA_EXE": executable, "PATH": ""}):
                with patch.object(ci_package, "run") as run:
                    ci_package.conda_install(output, "3.14")
            self.assertEqual(run.call_args.args[0], executable)

    def test_rejects_checkout_import(self):
        original_cwd = Path.cwd()
        try:
            spec = SimpleNamespace(origin=str(ci_package.ROOT / "scqubits/__init__.py"))
            with patch.object(
                ci_package.importlib.util, "find_spec", return_value=spec
            ):
                requested = ".".join(map(str, ci_package.sys.version_info[:2]))
                with self.assertRaisesRegex(RuntimeError, "import the checkout"):
                    ci_package.test_installed(requested, False, None)
            self.assertEqual(Path.cwd(), original_cwd)
        finally:
            os.chdir(original_cwd)

    def test_rejects_unsafe_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with tarfile.open(output / "scqubits-5.0.0.tar.gz", "w:gz") as archive:
                archive.addfile(tarfile.TarInfo("../../outside-checkout"))
            with patch.object(ci_package, "run"):
                with self.assertRaisesRegex(RuntimeError, "Unsafe sdist entry"):
                    ci_package.pip_build(output)

    def test_reports_survive_cleanup_and_have_distinct_paths(self):
        original_cwd = Path.cwd()
        handles = []
        working_directories = []
        with tempfile.TemporaryDirectory() as directory:
            # Windows temporary paths may use 8.3 names; macOS /var may be a
            # symlink. Match the helper's resolved paths rather than their aliases.
            base = Path(directory).resolve()
            spec = SimpleNamespace(origin=str(base / "installed/scqubits/__init__.py"))

            def record_run(*command):
                if "pytest" not in command:
                    return
                working_directories.append(Path.cwd())
                report = Path(
                    next(
                        x.split("=", 1)[1]
                        for x in command
                        if x.startswith("--junitxml=")
                    )
                )
                report.write_text("<testsuites />")
                # Keeping readers open reproduces the Windows sharing constraint.
                handles.append(report.open("rb"))
                if "--num_cpus=4" not in command:
                    self.assertIn(f"--cov-report=xml:{base / 'coverage.xml'}", command)

            try:
                os.chdir(base)
                with (
                    patch.object(
                        ci_package.importlib.util, "find_spec", return_value=spec
                    ),
                    patch.object(ci_package, "version", return_value="5.0.0"),
                    patch.object(ci_package, "run", side_effect=record_run),
                ):
                    requested = ".".join(map(str, ci_package.sys.version_info[:2]))
                    ci_package.test_installed(
                        requested, False, Path("coverage.xml"), Path("reports")
                    )
                self.assertEqual(Path.cwd(), base)
                self.assertEqual(len(working_directories), 2)
                self.assertTrue(all(not path.exists() for path in working_directories))
                self.assertEqual(
                    {p.name for p in (base / "reports").iterdir()},
                    {"serial.xml", "multiprocessing.xml"},
                )
                self.assertTrue(all(handle.read() for handle in handles))
            finally:
                for handle in handles:
                    handle.close()
                os.chdir(original_cwd)

    def test_test_failures_propagate_and_preserve_reports(self):
        original_cwd = Path.cwd()
        for fail_parallel in (False, True):
            with (
                self.subTest(fail_parallel=fail_parallel),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = Path(directory)
                spec = SimpleNamespace(
                    origin=str(base / "installed/scqubits/__init__.py")
                )
                working_directories = []

                def failing_run(*command):
                    if "pytest" not in command:
                        return
                    working_directories.append(Path.cwd())
                    report = Path(
                        next(
                            x.split("=", 1)[1]
                            for x in command
                            if x.startswith("--junitxml=")
                        )
                    )
                    report.write_text("<testsuites />")
                    if ("--num_cpus=4" in command) == fail_parallel:
                        raise subprocess.CalledProcessError(1, command)

                with (
                    patch.object(
                        ci_package.importlib.util, "find_spec", return_value=spec
                    ),
                    patch.object(ci_package, "version", return_value="5.0.0"),
                    patch.object(ci_package, "run", side_effect=failing_run),
                ):
                    requested = ".".join(map(str, ci_package.sys.version_info[:2]))
                    with self.assertRaises(subprocess.CalledProcessError):
                        ci_package.test_installed(
                            requested, False, None, base / "reports"
                        )
                self.assertEqual(Path.cwd(), original_cwd)
                self.assertTrue(all(not path.exists() for path in working_directories))
                self.assertTrue((base / "reports/serial.xml").is_file())
                self.assertEqual(
                    (base / "reports/multiprocessing.xml").exists(), fail_parallel
                )

    def test_reports_with_aliased_temporary_root(self):
        # A .. alias exercises path normalization on every OS without requiring
        # Windows symlink privileges. The original assertion fails in this case.
        with tempfile.TemporaryDirectory(prefix="ci path alias ") as directory:
            root = Path(directory).resolve()
            (root / "alias-parent").mkdir()
            (root / "real").mkdir()
            alias = root / "alias-parent" / ".." / "real"
            temporary_directory = tempfile.TemporaryDirectory

            @contextmanager
            def aliased_directory(*args, **kwargs):
                # Return the alias explicitly: newer Python versions normalize
                # tempfile.tempdir before returning a newly created directory.
                with temporary_directory(*args, dir=root / "real", **kwargs) as path:
                    yield str(alias / Path(path).name)

            with patch.object(tempfile, "TemporaryDirectory", aliased_directory):
                self.test_reports_survive_cleanup_and_have_distinct_paths()


if __name__ == "__main__":
    unittest.main()
