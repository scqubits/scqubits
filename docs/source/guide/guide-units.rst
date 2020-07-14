.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide_units:

***************
Units
***************

scqubits provides a means to set default units for the assumed energies of the quantum systems. These units play a key
role in :ref:`guide_noise` calculations. They are also used to set axes labels in some plot types.

The currently supported units are: ``GHz``, ``MHz``, ``kHz`` and ``Hz``, with ``GHz`` being the default. 
A list containing these possible choices can be shown with the ``show_supported_units`` function. 

The current units setting can be obtained with the ``get_units`` function. A new setting can be established with the
``set_units`` function::

    scq.get_units()
    scq.set_units('MHz')

scqubits also includes several helper functions for convenient conversion from the current system units to and
from `Hz`. This is accomplished with functions ``to_standard_units`` and ``from_standard_units``.

.. note:: The ``DEFAULT_ENERGY_UNIT`` :ref:`setting <guide-settings>` is no longer used. 

