.. sc_qubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Full Zero-Pi Qubit  (incl. coupling to zeta mode)
=================================================

.. figure:: ../../graphics/zeropifull.png
   :align: center
   :width: 4in

The full Zero-Pi qubit [Brooks2013]_ [Dempster2014]_, when including coupling to the zeta mode, is described by the
Hamiltonian :math:`H = H_{0-\pi} + H_\text{int} + H_\zeta`, where

.. math::
   &H_{0-\pi} = -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2 +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta\\
   &\qquad\qquad\qquad+2E_{C\Sigma}(\delta C_J/C_J)\partial_\phi\partial_\theta +2\,\delta E_J \sin\theta\sin(\phi-\phi_\text{ext}/2)\\
   &H_\text{int} = 2E_{C\Sigma}dC\,\partial_\theta\partial_\zeta + E_L dE_L \phi\,\zeta\\
   &H_\zeta = \omega_\zeta a^\dagger a

expressed in phase basis. The definition of the relevant charging energies :math:`E_\text{CJ}`, :math:`E_{\text{C}\Sigma}`,
Josephson energies :math:`E_\text{J}`, inductive energies :math:`E_\text{L}`, and relative amounts of disorder
:math:`dC_\text{J}`, :math:`dE_\text{J}`, :math:`dC`, :math:`dE_\text{L}` follows [Groszkowski2018]_.
Internally, the ``FullZeroPi`` class formulates the Hamiltonian matrix via the product basis of the decoupled Zero-Pi
qubit (realized by ``ZeroPi``)  on one hand, and the zeta LC oscillator on the other hand.

An instance of the full Zero-Pi qubit is created as follows::

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

    sc_qubits.FullZeroPi.hamiltonian
    sc_qubits.FullZeroPi.eigenvals
    sc_qubits.FullZeroPi.eigensys
    sc_qubits.FullZeroPi.get_spectrum_vs_paramvals


Wavefunctions and visualization of eigenstates
----------------------------------------------

.. todo:: consider adding this?


Implemented operators
---------------------

The following operators are implemented for use in matrix element calculations.

.. autosummary::

    sc_qubits.FullZeroPi.i_d_dphi_operator
    sc_qubits.FullZeroPi.phi_operator
    sc_qubits.FullZeroPi.n_theta_operator

.. todo:: may want to implement additional ops


Computation and visualization of matrix elements
------------------------------------------------

.. autosummary::

    sc_qubits.FullZeroPi.matrixelement_table
    sc_qubits.FullZeroPi.plot_matrixelements
    sc_qubits.FullZeroPi.get_matelements_vs_paramvals
    sc_qubits.FullZeroPi.plot_matelem_vs_paramvals
    sc_qubits.FullZeroPi.g_coupling_matrix
    sc_qubits.FullZeroPi.g_phi_coupling_matrix
    sc_qubits.FullZeroPi.g_theta_coupling_matrix

Utility method for setting charging energies
--------------------------------------------

.. autosummary::

    sc_qubits.ZeroPi.set_EC_via_ECS

