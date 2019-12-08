"""sc_qubits: superconducting qubits in Python
===========================================

sc_qubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within sc_qubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.
"""
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################



DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

# import statements
import os
import sys

try:
    import numpy as np
except:
    np = None


import setuptools
EXTRA_KWARGS = {}


# all information about sc_qubits goes here
MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.8)', 'scipy (>=1.0)', 'matplotlib(>=1.2.1)', 'qutip(>=4.3)']
EXTRAS_REQUIRE = {'graphics':['matplotlib-label-lines(>=0.3.6)']}
INSTALL_REQUIRES = ['numpy (>=1.8)', 'scipy (>=1.0)', 'matplotlib(>=1.2.1)', 'qutip(>=4.3)']
PACKAGES = ['sc_qubits', 'sc_qubits/core', 'sc_qubits/tests', 'sc_qubits/utils']

INCLUDE_DIRS = [np.get_include()] if np is not None else []

NAME = "sc_qubits"
AUTHOR = ("Jens Koch, Peter Groszkowski")
AUTHOR_EMAIL = ("jens-koch@northwestern.edu, piotrekg@gmail.com")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "superconducting qubits"

#TODO FILL IN URL
URL = "https://github.com/pypa/sampleproject"

CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        git_str = ""
    else:
        if git_str == '+': #fixes setuptools PEP issues with versioning
            git_str = ''
    return git_str

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'+str(MICRO)+git_short_hash()

# NumPy's distutils reads in versions differently than
# our fallback. To make sure that versions are added to
# egg-info correctly, we need to add FULLVERSION to
# EXTRA_KWARGS if NumPy wasn't imported correctly.
if np is None:
    EXTRA_KWARGS['version'] = FULLVERSION


def write_version_py(filename='sc_qubits/version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM SC_QUBITS SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'fullversion':
                FULLVERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, 'sc_qubits'))  # to retrive _version

# always rewrite _version
if os.path.exists('sc_qubits/version.py'):
    os.remove('sc_qubits/version.py')

write_version_py()

# Setup commands go here
setuptools.setup(name = NAME,
                 version = FULLVERSION,
                 packages = PACKAGES,
                 include_package_data=True,
                 include_dirs = INCLUDE_DIRS,
                 author = AUTHOR,
                 author_email = AUTHOR_EMAIL,
                 license = LICENSE,
                 description = DESCRIPTION,
                 long_description = LONG_DESCRIPTION,
                 keywords = KEYWORDS,
                 url = URL,
                 classifiers = CLASSIFIERS,
                 platforms = PLATFORMS,
                 requires = REQUIRES,
                 extras_require = EXTRAS_REQUIRE,
                 zip_safe = False,
                 install_requires=INSTALL_REQUIRES,
                 **EXTRA_KWARGS
)