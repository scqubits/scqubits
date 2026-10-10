"""Build release artifacts and test installed scqubits outside the checkout.

Used by Azure's pip matrix and the manual Conda matrix. This helper deliberately
uses only the standard library so it can run in both kinds of test environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path | None = None) -> None:
    """Run a command, propagating failures to CI."""
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def one_artifact(directory: Path, pattern: str) -> Path:
    """Require exactly one artifact to avoid installing stale output."""
    artifacts = list(directory.glob(pattern))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}: {artifacts}")
    return artifacts[0].resolve()


def recipe_version() -> str:
    """Read the version declared by the local Conda recipe."""
    match = re.search(
        r'{% set version = "([^"]+)" %}', (ROOT / "meta.yaml").read_text()
    )
    if match is None:
        raise RuntimeError("Cannot read the version from meta.yaml")
    return match.group(1)


def pip_build(output: Path) -> None:
    """Build an sdist, build its wheel, verify package data, and install it."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "-I",
        "-m",
        "build",
        "--sdist",
        "--outdir",
        str(output),
        cwd=ROOT,
    )
    sdist = one_artifact(output, "scqubits-*.tar.gz")
    with tempfile.TemporaryDirectory(prefix="scqubits-sdist-") as directory:
        extracted = Path(directory)
        with tarfile.open(sdist) as archive:
            # The archive was produced by our own build, but still reject unsafe paths.
            for member in archive.getmembers():
                target = (extracted / member.name).resolve()
                if not target.is_relative_to(extracted.resolve()) or not (
                    member.isfile() or member.isdir()
                ):
                    raise RuntimeError(f"Unsafe sdist entry: {member.name}")
            archive.extractall(extracted)
        source = one_artifact(extracted, "scqubits-*")
        run(
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(output),
            str(source),
        )
    wheel = one_artifact(output, "scqubits-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        included = set(archive.namelist())
    required = []
    for relative in (
        "scqubits/tests",
        "scqubits/core/qubit_img",
        "scqubits/ui/icons",
    ):
        for path in (ROOT / relative).rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix
                in (".py", ".npy", ".json", ".h5", ".hdf5", ".png", ".jpg", ".svg")
            ):
                required.append(path.relative_to(ROOT).as_posix())
    missing = sorted(set(required) - included)
    if missing:
        raise RuntimeError(f"Wheel is missing package/test files: {missing}")
    run(sys.executable, "-I", "-m", "pip", "install", f"{wheel}[gui]")


def conda_install(output: Path, python_version: str) -> None:
    """Install the exact build artifact with a constrained test interpreter."""
    artifacts = list((output / "noarch").glob("scqubits-*.conda"))
    artifacts += list((output / "noarch").glob("scqubits-*.tar.bz2"))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected exactly one Conda artifact: {artifacts}")
    filename = artifacts[0].name
    stem = filename.removesuffix(".conda").removesuffix(".tar.bz2")
    name, artifact_version, build = stem.rsplit("-", 2)
    # A channel-qualified MatchSpec keeps dependency solving enabled. Passing an
    # archive filename directly would use Conda's explicit-install path instead.
    artifact_spec = f"{output.resolve().as_uri()}::{name}={artifact_version}={build}"
    run(
        "conda",
        "create",
        "-y",
        "-n",
        "scqubits-test",
        "--override-channels",
        "--strict-channel-priority",
        "-c",
        str(output.resolve()),
        "-c",
        "conda-forge",
        f"python={python_version}",
        artifact_spec,
        "pip",
        "pytest",
        "pytest-cov",
    )


def test_installed(
    python_version: str, check_recipe: bool, coverage: Path | None
) -> None:
    """Verify interpreter/import metadata and execute both installed test suites."""
    requested = tuple(map(int, python_version.split(".")))
    if sys.version_info[:2] != requested:
        raise RuntimeError(f"Expected Python {python_version}; got {sys.version}")
    with tempfile.TemporaryDirectory(prefix="scqubits-tests-") as directory:
        os.chdir(directory)
        spec = importlib.util.find_spec("scqubits")
        if spec is None or spec.origin is None:
            raise RuntimeError("scqubits is not installed")
        installed = Path(spec.origin).resolve()
        if installed.is_relative_to(ROOT):
            raise RuntimeError(f"Tests would import the checkout: {installed}")
        print(
            f"Python: {sys.version}\nscqubits: {version('scqubits')}\nImport: {installed}",
            flush=True,
        )
        if check_recipe and version("scqubits") != recipe_version():
            raise RuntimeError(
                "Installed Python version metadata does not match meta.yaml"
            )
        run(sys.executable, "-I", "-m", "pip", "check")
        run(sys.executable, "-I", "-m", "pip", "list")
        # An explicit installed path lets pytest load conftest.py before parsing
        # its custom --num_cpus option; --pyargs alone discovers it too late.
        tests = str(installed.parent / "tests")
        serial = [sys.executable, "-I", "-m", "pytest", "-v", tests]
        if coverage is not None:
            serial += ["--cov=scqubits", f"--cov-report=xml:{coverage.resolve()}"]
        run(*serial)
        run(
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-v",
            tests,
            "--num_cpus=4",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("pip-build")
    build.add_argument("--output", type=Path, required=True)
    conda = commands.add_parser("conda-install")
    conda.add_argument("--output", type=Path, required=True)
    conda.add_argument("--python-version", required=True)
    test = commands.add_parser("test")
    test.add_argument("--python-version", required=True)
    test.add_argument("--recipe-version", action="store_true")
    test.add_argument("--coverage", type=Path)
    args = parser.parse_args()
    if args.command == "pip-build":
        pip_build(args.output)
    elif args.command == "conda-install":
        conda_install(args.output, args.python_version)
    else:
        test_installed(args.python_version, args.recipe_version, args.coverage)


if __name__ == "__main__":
    main()
