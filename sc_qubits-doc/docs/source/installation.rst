.. sc_qubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

.. This file can be edited using retext 6.1 https://github.com/retext-project/retext

.. _install:

**************
Installation
**************

.. _install-requires:

General Requirements
=====================

sc_qubits depends on the following Python open-source libraries:

.. cssclass:: table-striped

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| **Python**     | 2.7+         | Version 3.5+ is recommended.                        |
+----------------+--------------+-----------------------------------------------------+
| **NumPy**      | 1.8+         | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **SciPy**      | 0.15+        | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **Matplotlib** | 1.2.1+       | Some plotting does not work on lower versions.      |
+----------------+--------------+-----------------------------------------------------+
| **Qutip**      | 0.21+        | Needed for compiling some time-dependent            |
|                |              | Hamiltonians.                                       |
+----------------+--------------+-----------------------------------------------------+


The following packages are optional:

+------------------------+--------------+-----------------------------------------------------+
| Package                | Version      | Details                                             |
+========================+==============+=====================================================+
| QuTiP                  | 4.3          | Needed for composite Hilbert spaces                 |
+------------------------+--------------+-----------------------------------------------------+
| nose                   | 1.1.2+       | For running the test suite.                         |
+------------------------+--------------+-----------------------------------------------------+
| matplotlib-label-lines | 0.3.6+       | For running the test suite.                         |
+------------------------+--------------+-----------------------------------------------------+




.. _install-via_pip:

Installing via pip
==================

Install sc_qubits using the Python package manager `pip <http://www.pip-installer.org/>`_.

.. code-block:: bash

   pip install sc_qubits


.. _install-verify:

Verifying the Installation
==========================

sc_qubits includes a set of nose tests that can be executed to verify that installation was successful.

.. code-block:: python

   import sc_qubits.testing as sctest
   sctest.run()
