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

+-------------------------+------------------------------+-------------------------------------------------------------------+
| Setting                 |  Options                     | Description                                                       |
+=========================+==============================+=============+=====================================================+
| ``FILE_FORMAT``         | `FileType.h5`, `FileType.csv`| Switches between supported file formats for writing data to disk. |
+-------------------------+------------------------------+-------------------------------------------------------------------+
| ``PROGRESSBAR_DISABLED``|  True / False                | Switches display of progressbar on/off.                           |
+-------------------------+------------------------------+-------------------------------------------------------------------+
| ``AUTORUN_SWEEP``       | True / False (default: True) | Whether to generate ``ParameterSweep``                            |
|                         |                              | immediately upon initialization                                   |
+-------------------------+------------------------------+-------------------------------------------------------------------+
| ``DISPATCH_ENABLED``    | True / False (default: True) | Whether to use central dispatch system                            |
+-------------------------+------------------------------+-------------------------------------------------------------------+
| ``MULTIPROC``           | `str`                        | 'pathos' (default) or 'multiprocessing'                           |
+-------------------------+------------------------------+-------------------------------------------------------------------+
| ``NUM_CPUS``            | int                          | number of cores to be used in parallelization (default: 1)        |
+-------------------------+------------------------------+-------------------------------------------------------------------+

Users can also setup units of the energy scales. This is discussed in the :ref:`guide_units` section of the user guide. 


.. note:: The ``DEFAULT_ENERGY_UNIT`` setting is no longer used - see :ref:`guide_units` for information on how to set energy units. 


.. _settings-usage:

Example: Changing Settings
==========================

Modifying the settings is simple, for example::

   scqubits.settings.progressbar_enabled = False
   scqubits.settings.file_format = FileType.csv

