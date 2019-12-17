.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _install:

**************
Installation
**************

.. _install-requires:

General Requirements
=====================

scqubits depends on the following Python open-source libraries:

.. cssclass:: table-striped

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| **Python**     | 3+           | Version 3.5+ is recommended.                        |
+----------------+--------------+-----------------------------------------------------+
| **NumPy**      | 1.14.2+      | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **SciPy**      | 1.1.0+       | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **Matplotlib** | 3.0.0+       | Some plotting does not work on lower versions.      |
+----------------+--------------+-----------------------------------------------------+
| **QuTiP**      | 4.3          |  Needed for composite Hilbert spaces.               |
+----------------+--------------+-----------------------------------------------------+
| **Cython**     | 0.28.5+      |  Required by QuTiP                                  |
+----------------+--------------+-----------------------------------------------------+
| **h5py**       | 2.7.1+       |  Needed for writing h5 data to file                 |
+----------------+--------------+-----------------------------------------------------+


The following packages are optional:

+------------------------+--------------+-----------------------------------------------------+
| Package                | Version      | Details                                             |
+========================+==============+=====================================================+
| nose                   | 1.3.7+       | For running the test suite.                         |
+------------------------+--------------+-----------------------------------------------------+
| matplotlib-label-lines | 0.3.6+       | For smart labelling of matrix element plots         |
+------------------------+--------------+-----------------------------------------------------+


.. _install-via_conda:

Installing via conda
====================

For Python 3.6 and 3.7, installation via conda is supported.

.. code-block:: bash

   conda install -c conda-forge scqubits



.. _install-via_pip:

Installing via pip
==================

scqubits can also be installed using the Python package manager `pip <http://www.pip-installer.org/>`_.

.. code-block:: bash

   pip install scqubits






.. _install-verify:

Verifying the Installation
==========================

scqubits includes a set of nose tests that can be executed to verify that installation was successful.

.. code-block:: python

   import scqubits.testing as sctest
   sctest.run()
