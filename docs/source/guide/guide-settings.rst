.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _settings:

*************************************
Modifying Internal scqubits Settings
*************************************

.. _settings-params:

User Accessible Parameters
==========================

scqubits has a few internal parameters that can be changed by the user:

.. tabularcolumns:: | p{3cm} | p{3cm} | p{3cm} |

.. cssclass:: table-striped

+------------------------+-------------------------------------------+-----------------------------+
| Setting                | Description                               | Options                     |
+========================+===========================================+=============================+
| `file_format`          | Switches between supported file formats   | FileType.h5, FileType.csv   |
|                        | for writing data to disk.                 |                             |
+------------------------+-------------------------------------------+-----------------------------+
| `progressbar_enabled`  | Switches display of progressbar on/off.   | True / False                |
+------------------------+-------------------------------------------+-----------------------------+

.. _settings-usage:

Example: Changing Settings
==========================

Modifying the settings is simple, for example::

   scqubits.settings.progressbar_enabled = False
   scqubits.settings.file_format = FileType.csv

