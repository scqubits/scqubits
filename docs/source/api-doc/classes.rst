.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

.. _classes:

***************
Qubit Classes
***************

.. _classes-transmon:

Transmon
--------------

.. autoclass:: scqubits.Transmon
    :members:
    :inherited-members:

----------------------------------

.. _classes-fluxonium:

Fluxonium
---------

.. autoclass:: scqubits.Fluxonium
    :members:
    :inherited-members:

----------------------------------

.. _classes-fluxqubit:

FluxQubit
---------

.. autoclass:: scqubits.FluxQubit
    :members:
    :inherited-members:

----------------------------------

.. _classes-zeropi:

ZeroPi
------

.. autoclass:: scqubits.ZeroPi
    :members:
    :inherited-members:

----------------------------------

.. _classes-fullzeropi:

FullZeroPi
--------------

.. autoclass:: scqubits.FullZeroPi
    :members:
    :inherited-members:

----------------------------------

.. _classes-oscillator:

Oscillator
--------------

.. autoclass:: scqubits.Oscillator
    :members:
    :inherited-members:


**********************************************************
Classes for Composite Hilbert Spaces, Interface with QuTiP
**********************************************************

.. _classes-hilbertspace:

HilbertSpace
--------------

.. autoclass:: scqubits.HilbertSpace
    :members:

.. _classes-paramsweep:

ParameterSweep
--------------

.. autoclass:: scqubits.ParameterSweep
    :members:

.. _classes-spectrumlookup:

SpectrumLookup
--------------

.. autoclass:: scqubits.core.spec_lookup.SpectrumLookup
    :members:

    .. automethod:: SpectrumLookup.dressed_index(self, bare_labels, param_index=0)

**********************************************************
Data Storage Classes
**********************************************************

.. _classes-datastore:

SpectrumData
--------------

.. autoclass:: scqubits.core.storage.SpectrumData
    :members:

DataStore
--------------

.. autoclass:: scqubits.core.storage.DataStore
    :members:

----------------------------------

.. _classes-wavefunction:

WaveFunction
--------------

.. autoclass:: scqubits.core.storage.WaveFunction
    :members:

----------------------------------

.. _classes-wavefunctionongrid:

WaveFunctionOnGrid
------------------

.. autoclass:: scqubits.core.storage.WaveFunctionOnGrid
    :members:


**********************************************************
Discretization Classes
**********************************************************

.. _classes-grid1d:

Grid1d
--------------

.. autoclass:: scqubits.Grid1d
    :members:

----------------------------------

.. _classes-gridspec:

GridSpec
--------------

.. autoclass:: scqubits.core.discretization.GridSpec
    :members:
