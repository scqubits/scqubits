.. sc_qubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Zero-Pi Qubit  (decoupled from zeta mode)
=========================================

.. figure:: ../../graphics/zeropi.png
   :align: center
   :width: 4in

The Zero-Pi qubit [Brooks2013]_ [Dempster2014]_, when decoupled from the zeta mode, is described by the Hamiltonian

.. math::

   H &= -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
    +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta\\
     &\qquad -2E_\text{J}\cos\theta\cos(\phi-\varphi_\text{ext}/2)+E_L\phi^2+2E_\text{J}
    + E_J dE_J \sin\theta\sin(\phi-\phi_\text{ext}/2)

expressed in phase basis. The definition of the relevant charging energies :math:`E_\text{CJ}`, :math:`E_{\text{C}\Sigma}`,
Josephson energies :math:`E_\text{J}`, inductive energies :math:`E_\text{L}`, and relative amounts of disorder
:math:`dC_\text{J}`, :math:`dE_\text{J}` follows [Groszkowski2018]_.

Internally, the ``ZeroPi`` class formulates the Hamiltonian matrix by discretizing the ``phi`` variable, and
using charge basis for the ``theta`` variable.

An instance of the Zero-Pi qubit is created as follows::

   phi_grid = sc_qubits.Grid1d(-6*np.pi, 6*np.pi, 200)

   zero_pi = sc_qubits.ZeroPi(grid = phi_grid,
                              EJ   = 0.25,
                              EL   = 10.0**(-2),
                              ECJ  = 0.5,
                              EC   = None,
                              ECS  = 10.0**(-3),
                              ng   = 0.1,
                              flux = 0.23,
                              ncut = 30)

Here, ``flux`` is given in dimensionless units, in the form :math:`2\pi\Phi_\text{ext}/\Phi_0`. In the above example,
the disorder parameters ``dEJ`` and ``dCJ`` are not specified, and hence take on the default value zero (no disorder).


Calculational methods related to Hamiltonian and energy spectra
---------------------------------------------------------------

.. autosummary::

    sc_qubits.ZeroPi.hamiltonian
    sc_qubits.ZeroPi.eigenvals
    sc_qubits.ZeroPi.eigensys
    sc_qubits.ZeroPi.get_spectrum_vs_paramvals


Wavefunctions and visualization of eigenstates
----------------------------------------------

.. autosummary::

    sc_qubits.ZeroPi.wavefunction
    sc_qubits.ZeroPi.plot_wavefunction


Implemented operators
---------------------

The following operators are implemented for use in matrix element calculations.

.. autosummary::
    sc_qubits.ZeroPi.i_d_dphi_operator
    sc_qubits.ZeroPi.phi_operator
    sc_qubits.ZeroPi.n_theta_operator
    sc_qubits.ZeroPi.cos_theta_operator
    sc_qubits.ZeroPi.sin_theta_operator

.. todo:: may want to implement additional ops


Computation and visualization of matrix elements
------------------------------------------------

.. autosummary::

    sc_qubits.ZeroPi.matrixelement_table
    sc_qubits.ZeroPi.plot_matrixelements
    sc_qubits.ZeroPi.get_matelements_vs_paramvals
    sc_qubits.ZeroPi.plot_matelem_vs_paramvals


Utility method for setting charging energies
--------------------------------------------

.. autosummary::

    sc_qubits.ZeroPi.set_EC_via_ECS

