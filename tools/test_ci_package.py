"""Regression tests for artifact selection and installed-package CI guards."""

import os
import tarfile
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
