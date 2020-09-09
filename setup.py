"""scqubits: superconducting qubits in Python
===========================================

scqubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within scqubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.
"""
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import os
import sys
import setuptools


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


EXTRA_KWARGS = {}

# version information about scqubits goes here
MAJOR = 1
MINOR = 3
MICRO = 1
ISRELEASED = True

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "requirements.txt")) as requirements:
    INSTALL_REQUIRES = requirements.read().splitlines()


EXTRAS_REQUIRE = {'graphics': ['matplotlib-label-lines (>=0.3.6)'],
                  'explorer': ['ipywidgets (>=7.5)'],
                  'h5-support': ['h5py (>=2.10)'],
                  'pathos': ['pathos', 'dill'],
                  'fitting': ['lmfit']}

TESTS_REQUIRE = ['h5py (>=2.7.1)',
                 'pathos',
                 'dill',
                 'ipywidgets',
                 'pytest',
                 'lmfit']

PACKAGES = ['scqubits',
            'scqubits/core',
            'scqubits/tests',
            'scqubits/utils',
            'scqubits/ui',
            'scqubits/io_utils']

PYTHON_VERSION = '>=3.6'


NAME = "scqubits"
AUTHOR = "Jens Koch, Peter Groszkowski"
AUTHOR_EMAIL = "jens-koch@northwestern.edu, piotrekg@gmail.com"
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "superconducting qubits"
URL = "https://scqubits.readthedocs.io"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except OSError:
        git_str = ""
    else:
        if git_str == '+':   # fixes setuptools PEP issues with versioning
            git_str = ''
    return git_str


FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'+str(MICRO)+git_short_hash()


def write_version_py(filename='scqubits/version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM scqubits SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    versionfile = open(filename, 'w')
    try:
        versionfile.write(cnt % {'version': VERSION, 'fullversion': FULLVERSION, 'isrelease': str(ISRELEASED)})
    finally:
        versionfile.close()


local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, 'scqubits'))  # to retrieve _version

# always rewrite _version
if os.path.exists('scqubits/version.py'):
    os.remove('scqubits/version.py')
write_version_py()

setuptools.setup(name=NAME,
                 version=FULLVERSION,
                 packages=PACKAGES,
                 author=AUTHOR,
                 author_email=AUTHOR_EMAIL,
                 license=LICENSE,
                 description=DESCRIPTION,
                 long_description=LONG_DESCRIPTION,
                 keywords=KEYWORDS,
                 url=URL,
                 classifiers=CLASSIFIERS,
                 platforms=PLATFORMS,
                 install_requires=INSTALL_REQUIRES,
                 extras_require=EXTRAS_REQUIRE,
                 tests_require=TESTS_REQUIRE,
                 zip_safe=False,
                 include_package_data=True,
                 python_requires=PYTHON_VERSION,
                 **EXTRA_KWARGS
                 )
