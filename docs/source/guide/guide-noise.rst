.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide_noise:

**************************
Estimating Coherence Times
**************************

scqubits implements coherence time calculations for estimating depolarization and pure-dephasing times for a majority of the
qubits implemented. For each qubit, a variety of noise channels can be considered, ranging from ubiquitous channels
affecting most circuits to more specialized ones applicable only to the qubit of interest.

.. note::

    The coherence calculations in scqubits use standard approximation techiniqes discussed in the listed references. 
    It is up to the user to ensure the assumptions are consistent with their parameter choices. 

A list of the channels supported by a given qubit can be obtained by calling the ``supported_noise_channels()`` method.
Each entry in the returned list represents a method that can be directly called on the qubit object.
For example, in the case of  the ``TunableTransmon`` qubit, this could take the form::

    tune_tmon = scq.TunableTransmon(
        EJmax=20.0,
        EC=0.5,
        d=0.00,
        flux=0.0,
        ng=0.3,
        ncut=150
    )
    transmon.supported_noise_channels()
    transmon.tphi_1_over_f_flux()
    

The last call to the method ``tphi_1_over_f_flux()`` returns an estimate of the pure dephasing time due
to 1/f flux noise.

By default, all noise methods return a decay time (:math:`T_1` in the case of a depolarizing channel or
:math:`T_\phi` in case of a dephasing channel). However, by supplying the option ``get_rate=True``,
one can obtain the corresponding rate instead.

The units of the returned decoherence times reflect the global units settings. For example, if the global units are set
to ``GHz`` (default), then the resulting decay and dephasing times will be given in ``1/GHz = ns``.
See the :ref:`guide_units` section for more information on how to set global units.

By default, all coherence time calculations assume that the qubit Hilbert space is reduced to the lowest two energy levels.
If needed, the user can provide different energy levels as arguments in order to extend the relevant subspace.

A set of examples that show how to perform a variety of cohrence-times estimates is presented in
this `jupyter notebook <https://nbviewer.jupyter.org/github/scqubits/scqubits/blob/master/examples/demo_noise.ipynb>`_ .

More detailed discussions of the individual supported noise channels can be found in the following subsections.

.. toctree::
   :maxdepth: 3

   noise/ipynb/visualization.ipynb
   noise/dephasing.rst
   noise/depolarization.rst
   noise/effective_noise.rst


