sc_qubits: superconducting qubits in Python
===========================================

[J. Koch](https://github.com/jkochNU), [P. Groszkowski](https://github.com/petergthatsme)


sc_qubits is an open-source Python library for simulating superconducting qubits. It is meant to give the user
a convenient way to obtain energy spectra of common superconducting qubits, plot energy levels as a function of
external parameters, calculate matrix elements etc. The library further provides an interface to QuTiP, making it
easy to work with composite Hilbert spaces consisting of coupled superconducting qubits and harmonic modes.
Internally, numerics within sc_qubits is carried out with the help of Numpy and Scipy; plotting capabilities rely on
Matplotlib.




Download and Installation
-------------------------

You can install sc_qubits locally via pip.
```
git clone https://github.com/Northwestern-Koch-Group/sc_qubits.git
cd sc_qubits
pip install .
```

[//]: # (TODO: Decide whether we want to use PyPi)



Documentation
-------------

The documentation for sc_qubits is available at:

[//]: # (TODO Add link to documentation)


Contribute
----------

You are welcome to contribute to sc_qubits development by forking this repository and sending pull requests, 
or filing bug reports at the
[issues page](http://TODO).

[//]: # (TODO: fill in link)

All contributions are acknowledged in the
[contributors](http://TODO)
section in the documentation.

[//]: # (TODO: fill in link)

All contributions are expected to be consistent with [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).


License
-------
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE file are satisfied.
