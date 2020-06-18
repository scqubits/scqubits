scqubits: superconducting qubits in Python
===========================================

[J. Koch](https://github.com/jkochNU), [P. Groszkowski](https://github.com/petergthatsme)


scqubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within scqubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.

[![CodeFactor](https://www.codefactor.io/repository/github/scqubits/scqubits/badge)](https://www.codefactor.io/repository/github/scqubits/scqubits)

News
----

(2020/06/18) We're in the process of adding coherence calculations to scqubits. 
If you're interested in exploring and/or testing these additions, please install the `noise` branch.

Download and Installation
-------------------------

For Python 3.6, 3.7 and 3.8, installation via conda is supported. 
```
conda install -c conda-forge scqubits
```

Alternatively, you can install scqubits via pip.
```
pip install scqubits
```




Documentation
-------------

The documentation for scqubits is available at: https://scqubits.readthedocs.io



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
