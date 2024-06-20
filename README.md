scqubits: superconducting qubits in Python
===========================================

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/scqubits/badges/downloads.svg)](https://anaconda.org/conda-forge/scqubits)
[![CodeFactor](https://www.codefactor.io/repository/github/scqubits/scqubits/badge)](https://www.codefactor.io/repository/github/scqubits/scqubits)
[![codecov](https://codecov.io/gh/scqubits/scqubits/branch/master/graph/badge.svg?token=PUBXSHF6HU)](https://codecov.io/gh/scqubits/scqubits)


[J. Koch](https://github.com/jkochNU), [P. Groszkowski](https://github.com/petergthatsme)

<br>

> **Join the scqubits mailing list!** Receive information about new releases
> and opportunities to contribute
> to new developments.
> |[SIGN UP](https://sites.northwestern.edu/koch/scqubits-news-sign-up/)|
> |---------------------------------------------------------------------|

<br>

scqubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within scqubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.

If scqubits is helpful to you in your research, please support its continued 
development and maintenance. Use of scqubits in research publications is 
appropriately acknowledged by citing:

&nbsp; &nbsp; Peter Groszkowski and Jens Koch,<br> 
&nbsp; &nbsp; *scqubits:  a Python package for superconducting qubits*,<br>
&nbsp; &nbsp; Quantum 5, 583 (2021).<br>
&nbsp; &nbsp; https://quantum-journal.org/papers/q-2021-11-17-583/

&nbsp; &nbsp; Sai Pavan Chitta, Tianpu Zhao, Ziwen Huang, Ian Mondragon-Shem, and Jens Koch,<br>
&nbsp; &nbsp; *Computer-aided quantization and numerical analysis of superconducting circuits*,<br>
&nbsp; &nbsp; New J. Phys. 24 103020 (2022).<br>
&nbsp; &nbsp; https://iopscience.iop.org/article/10.1088/1367-2630/ac94f2



Download and Installation
-------------------------

For Python 3.7, 3.8, 3.9, and 3.10: installation via conda is supported. 
```
conda install -c conda-forge scqubits
```

Alternatively, scqubits can be installed via pip (although it should be noted that installing via pip under a conda environment is strongly discouraged, and is not guaranteed to work - see conda documentation).
```
pip install scqubits
```



Documentation
-------------

The documentation for scqubits is available at: https://scqubits.readthedocs.io


Related Packages
----------------

There are two related packages on github:

documentation source code: https://github.com/scqubits/scqubits-doc   
example notebooks: https://github.com/scqubits/scqubits-examples  


Contribute
----------

You are welcome to contribute to scqubits development by forking this repository and sending pull requests, 
or filing bug reports at the
[issues page](https://github.com/scqubits/scqubits/issues).


All contributions are acknowledged in the
[contributors](https://scqubits.readthedocs.io/en/latest/contributors.html)
section in the documentation.

All contributions are expected to be consistent with [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).


License
-------
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE file are satisfied.
