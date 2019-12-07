.. sc_qubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Transmon Qubit
==============

.. figure:: ../../graphics/transmon.png
   :align: center
   :width: 4in

The transmon qubit [Koch2007]_ is described by the Hamiltonian

.. math::

   H=4E_\text{C}(\hat{n}-n_g)^2+\frac{1}{2}E_\text{J}\sum_n(|n\rangle\langle n+1|+\text{h.c.}),

expressed in discrete charge basis. :math:`E_C` is the charging energy, :math:`E_J` the Josephson energy, and
:math:`n_g` the offset charge. Within the ``Transmon`` class, charge-basis representation is employed with a
charge-number cutoff specified by ``ncut``. This cutoff must be chosen sufficiently large for convergence.

An instance of the transmon qubit is initialized as follows::

   transmon = sc_qubits.Transmon(EJ=30.02,
                                 EC=1.2,
                                 ng=0.3,
                                 ncut=31)

Calculational methods related to Hamiltonian and energy spectra
---------------------------------------------------------------

.. autosummary::

    sc_qubits.Transmon.hamiltonian
    sc_qubits.Transmon.eigenvals
    sc_qubits.Transmon.eigensys
    sc_qubits.Transmon.get_spectrum_vs_paramvals


Wavefunctions and visualization of eigenstates
----------------------------------------------

.. autosummary::

    sc_qubits.Transmon.numberbasis_wavefunction
    sc_qubits.Transmon.phasebasis_wavefunction
    sc_qubits.Transmon.plot_n_wavefunction
    sc_qubits.Transmon.plot_phi_wavefunction


Implemented operators
---------------------

The following operators are implemented for use in matrix element calculations.

.. autosummary::
    sc_qubits.Transmon.n_operator

.. todo:: implement cos(phi), sin(phi), e^iphi


Computation and visualization of matrix elements
------------------------------------------------

.. autosummary::

    sc_qubits.Transmon.matrixelement_table
    sc_qubits.Transmon.plot_matrixelements
    sc_qubits.Transmon.get_matelements_vs_paramvals
    sc_qubits.Transmon.plot_matelem_vs_paramvals


