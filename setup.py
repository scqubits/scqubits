"""
setup.py - a module to allow package installation
"""

from distutils.core import setup


NAME = "sc_qubits"
VERSION = "0.1alpha"
DEPENDENCIES = [
    "qutip",
    "matplotlib",
    "numpy",
    "scipy",
]
DESCRIPTION = "superconducting qubits in Python"
AUTHOR = "Jens Koch, Peter Groszkowski"
AUTHOR_EMAIL = "jens-koch@northwestern.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)