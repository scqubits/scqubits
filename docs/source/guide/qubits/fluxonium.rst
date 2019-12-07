.. sc_qubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski


Fluxonium Qubit
===============

.. figure:: ../../graphics/fluxonium.png
   :align: center
   :width: 4in

The Hamiltonian of the fluxonium qubit [Manucharyan2009]_ in phase basis representation is given by

.. math::

   H=-4E_\text{C}\partial_\phi^2-E_\text{J}\cos(\phi-\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2.

Here, :math:`E_C` is the charging energy, :math:`E_J` the Josephson energy, :math:`E_L` the inductive energy, and
:math:`\varphi_\text{ext}=2\pi \Phi_\text{ext}/\Phi_0` the external flux in dimensionless form. The ``Fluxonium`` class
internally uses the :math:`E_C`-:math:`E_L` harmonic-oscillator basis [Zhu2013]_ with truncation level specified by ``cutoff``.



An instance of the fluxonium created as follows::

   fluxonium = sc_qubits.Fluxonium(EJ = 8.9,
                                   EC = 2.5,
                                   EL = 0.5,
                                   flux = 0.33,
                                   cutoff = 110)

Here, ``flux`` is given in dimensionless units, in the form :math:`2\pi\Phi_\text{ext}/\Phi_0`.


Calculational methods related to Hamiltonian and energy spectra
---------------------------------------------------------------

.. autosummary::

    sc_qubits.Fluxonium.hamiltonian
    sc_qubits.Fluxonium.eigenvals
    sc_qubits.Fluxonium.eigensys
    sc_qubits.Fluxonium.get_spectrum_vs_paramvals


Wavefunctions and visualization of eigenstates
----------------------------------------------

.. autosummary::

    sc_qubits.Fluxonium.wavefunction
    sc_qubits.Fluxonium.plot_wavefunction


Implemented operators
---------------------

The following operators are implemented for use in matrix element calculations.

.. autosummary::
    sc_qubits.Fluxonium.n_operator
    sc_qubits.Fluxonium.phi_operator

.. todo:: implement cos(phi), sin(phi)


Computation and visualization of matrix elements
------------------------------------------------

.. autosummary::

    sc_qubits.Fluxonium.matrixelement_table
    sc_qubits.Fluxonium.plot_matrixelements
    sc_qubits.Fluxonium.get_matelements_vs_paramvals
    sc_qubits.Fluxonium.plot_matelem_vs_paramvals


