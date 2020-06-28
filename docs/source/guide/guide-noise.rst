.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide_noise:

******
Noise
******

scqubits implements noise calculations that allow one to estimate depolarization as well as pure dephasing rates and times for a majority of qubits. Each qubit may support a variety of noise channels, ranging from ones common to most superconducting circuits, but also specialized ones, only relevant to that particular qubit.

To list what channels are supported by a given qubit, one can call its  ``supported_noise_channels()``, method. Each of the elements of the list corresponds to a function that can be directly called on the qubit object. For example, in the case of  the ``TunableTransmon`` qubit one could do::

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
    
The last call to the method ``tphi_1_over_f_flux()`` returns an estimate of the pure dephasing time due to 1/f flux noise. 

By default, all the noise methods return a decay time (:math:`t_1` in the case of a depolarizing channel or :math:`t_\phi` in a case of a dephasing channel), however, by supplying an option ``get_rate=True``, one can obtain the corresponding rate instead. 

The units of of the returned decay (or dephasing) times reflect the global units settings. For example, if the global units are set to ``GHz``, then the resulting decay and dephasing times will be given in ``ns``. See the `Units` section of this guide for more information on how global units can be set. 

By default all the noise functions assume that the qubit Hilbert space consist of the lowest two energy levels. The user, however, can provide different energy levels as arguments in order to redefine the relevant subspace. 

A set of examples that show how many of the noise estimations can be calculated are presented in this `jupyter notebook <https://nbviewer.jupyter.org/github/scqubits/scqubits/blob/master/examples/demo_noise.ipynb>`_

A more detailed discussion of each of the supporting noise channels is shown below.  

.. toctree::
   :maxdepth: 3

   noise/dephasing.rst
   noise/depolarization.rst


