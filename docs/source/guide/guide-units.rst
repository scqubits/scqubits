.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide_units:

***************
Units
***************

scqubits provides a means to set default units for the assumed energies of the quantum systems. These units are used to set labels in some of the plots, but also in :ref:`guide_noise` calculations. 

The currently supported units are: ``GHz``, ``MHz``, ``kHz`` and ``Hz``, with ``GHz`` being the default. 
A list containing these possible choices can be shown with the ``show_supported_units`` function. 

Currently set units can be obtained with ``get_units`` function, and new ones can be set with the ``set_units`` functions::

    scq.get_units()
    scq.set_units('MHz')

scqubits also include a couple of helper functions that help with conversion from currently set system units, to and from `Hz`. This is done with functions ``to_standard_units`` and ``from_standard_units``. 


Note that the ``DEFAULT_ENERGY_UNIT`` :ref:`setting <guide-settings>` is no longer used. 



