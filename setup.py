#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################
"""
setup.py - a module to allow package installation for sc_qubits
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