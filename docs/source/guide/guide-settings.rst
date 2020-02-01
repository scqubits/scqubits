.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide-settings:

*************************************
Modifying Internal scqubits Settings
*************************************

.. _settings-params:

User Accessible Parameters
==========================

scqubits has a few internal parameters that can be changed by the user:

.. tabularcolumns:: | p{3cm} | p{3cm} | p{3cm} |

.. cssclass:: table-striped

+------------------------+------------------------------+-------------------------------------------------------------------+
| Setting                |  Options                     | Description                                                       |
+========================+==============================+=============+=====================================================+
| `FILE_FORMAT`          | `FileType.h5`, `FileType.csv`| Switches between supported file formats for writing data to disk. |
+------------------------+------------------------------+-------------------------------------------------------------------+
| `PROGRESSBAR_DISABLED` |  True / False                | Switches display of progressbar on/off.                           |
+------------------------+------------------------------+-------------------------------------------------------------------+
| `DEFAULT_ENERGY_UNIT`  |  `str` (default: GHz)        | Used in axes labels with plotting                                 |
+------------------------+------------------------------+-------------------------------------------------------------------+
| `AUTORUN_SWEEP`        | True / False (default: True) | Whether to generate `ParameterSweep`                              |
|                        |                              | immediately upon initialization                                   |
+------------------------+------------------------------+-------------------------------------------------------------------+
| `DISPATCH_ENABLED`     | True / False (default: True) | Whether to use central dispatch system                            |
+------------------------+------------------------------+-------------------------------------------------------------------+

.. _settings-usage:

Example: Changing Settings
==========================

Modifying the settings is simple, for example::

   scqubits.settings.progressbar_enabled = False
   scqubits.settings.file_format = FileType.csv

