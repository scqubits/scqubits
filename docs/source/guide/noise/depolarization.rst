.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Depolarization
================

Noise may cause depolarization of the qubit by inducing spontaneous transitions among eigenstates. scqubits uses the
standard perturbative approach (Fermi's Golden Rule) to approximate the resulting transition rates due to different
noise channels.

The rate of a transition from state :math:`i` to state :math:`j` can be expressed as

.. math::

   \Gamma_{ij} = \frac{1}{\hbar^2} |\langle i| B_{\lambda} |j \rangle|^2 S(\omega_{ij}),

where :math:`B_\lambda` is the noise operator, and :math:`S(\omega_{ij})` the spectral density function evaluated at
the angular frequency associated with the transition frequeny, :math:`\omega_{ij} = \omega_{j} - \omega_{i}`.
:math:`\omega_{ij}` is positive in the case of  decay (the qubit emits energy to the bath), and negative in case of
excitations (the qubit absorbs energy from the bath).

Unless stated otherwise, it is assumed that the depolarizing noise channels satisfy detailed balanced. This implies

.. math::

    \frac{S(-\omega)}{S(\omega)} = \exp{\frac{\hbar \omega}{k_B T}},

where :math:`T` is the bath temperature, and :math:`k_B` Boltzmann's constant.


.. note::

    By default all :math:`t_1` methods estimate the coherence depolarization times from the sum of the upward and downard rates.  
    This behavior is controlled by the arugment `total`, which can be modified by the user. For example, setting `total=False` 
    will calculate only a single-directional transition rate from the state indexed `i` to the state indexed `j` (both of which
    cal also be changed by the user through providing their values as arguments) 


Capacitive noise
-----------------------

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1_capacitive_loss``                  |
+--------------------------------------------+-----------------------------------------+
| :math:`B_\lambda`                          | :math:`2e \hat{n}`                      |
+--------------------------------------------+-----------------------------------------+

Capacitive noise corresponds to noise coming from a lossy capacitance. The assumed spectral density reads

.. math::

    S(\omega) = \frac{\omega \hbar}{|\omega| C_J Q_{\rm cap}(\omega)} \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right)

where :math:`C_J` is the relevant capacitance, and :math:`Q_{\rm cap}` the corresponding capacitive quality factor.
The default value of the frequency-dependent quality factor is assumed to be

.. math::

    Q_{\rm cap}(\omega) =  10^{6}  \left( \frac{2 \pi \times 6 {\rm GHz} }{ |\omega|} \right)^{0.7}. 


Qubits that support this noise channel include: 
:ref:`TunableTransmon <qubit_tunable_transmon>`, 
:ref:`Fluxonium <qubit_fluxonium>`, 
:ref:`FullZeroPi <qubit_fullzeropi>`, 
:ref:`ZeroPi <qubit_zeropi>`.

Inductive noise
-----------------------

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1_inductive_loss``                   |
+--------------------------------------------+-----------------------------------------+
| :math:`B_\lambda`                          | :math:`\frac{\Phi_0}{2\pi} \hat{\phi}`  |
+--------------------------------------------+-----------------------------------------+

Inductive noise due to lossy inductance. The assumed spectral density reads

.. math::

    S(\omega) = \frac{\omega \hbar}{|\omega| L_{J} Q_{\rm ind}(\omega)} \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right)

where :math:`L_J` is the relevant inductance or superinductance, and :math:`Q_{\rm ind}` the corresponding inductive
quality factor. The default value of the frequency-dependent quality factor is assumed to be

.. math::

    Q_{\rm ind}(\omega) =  500 \times 10^{6} \frac{ K_{0} \left( \frac{h \times 0.5 {\rm GHz}}{2 k_B T} \right) 
    \sinh \left( \frac{\hbar |\omega| }{2 k_B T} \right)}{K_{0} \left( \frac{\hbar |\omega|}{2 k_B T} \right)\
    \sinh \left( \frac{\hbar |\omega| }{2 k_B T} \right)},

where :math:`K_0` is the Bessel function of the second kind. 

Qubits that support this noise channel include: 
:ref:`Fluxonium <qubit_fluxonium>`.


Charge-coupled impedance noise
------------------------------

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1_charge_impedance``                 |
+--------------------------------------------+-----------------------------------------+
| :math:`B_\lambda`                          | :math:`2e \hat{n}`                      |
+--------------------------------------------+-----------------------------------------+

Noise from a charge coupling to an impedance :math:`Z(\omega)`. The assumed spectral density reads

.. math::

    S(\omega) = \frac{\hbar \omega}{{\rm Re} Z(\omega)} \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right).

By default we assume the qubit couples to a infinite transmission line, which leads to 

.. math::

   {\rm Re} Z(\omega) = 50 \Omega.


Qubits that support this noise channel include: 
:ref:`TunableTransmon <qubit_tunable_transmon>`, 
:ref:`Fluxonium <qubit_fluxonium>`, 
:ref:`FullZeroPi <qubit_fullzeropi>`, 


Flux-bias line noise
-------------------------

+--------------------------------------------+--------------------------------------------------------------+
| Method name                                | ``t1_flux_bias_line``                                        |
+--------------------------------------------+--------------------------------------------------------------+
| :math:`B_\lambda`                          | :math:`\frac{\partial \hat{H}}{\partial \Phi_x}`             | 
+--------------------------------------------+--------------------------------------------------------------+

Noise due to current noisy biasing current coupled to the qubit via flux. The assumed spectral density reads

.. math::

    S(\omega) = \frac{M^{2} \omega \hbar}{R} \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right)

where :math:`M` is the mutual inductance between qubit and the flux line.

Qubits that support this noise channel include: 
:ref:`TunableTransmon <qubit_tunable_transmon>`, 
:ref:`Fluxonium <qubit_fluxonium>`, 
:ref:`FullZeroPi <qubit_fullzeropi>`, 
:ref:`ZeroPi <qubit_zeropi>`.

Quasiparticle-tunneling noise
----------------------------------

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1_quasiparticle_tunneling``          |
+--------------------------------------------+-----------------------------------------+
| :math:`B_\lambda`                          | :math:`\sin(\hat{\phi}/2)`              |
+--------------------------------------------+-----------------------------------------+

Noise due to quasiparticle tunelling. The assumed spectral density reads

.. math::

    S(\omega) = \hbar \omega {\rm Re} Y_{\rm qp}(\omega) \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right)

where :math:`L_J` (with :math:`E_J = \phi_0^2/L_J` ) is the relevant inductance or superinductance, and :math:`Q_{\rm ind}` the corresponding inductive
quality factor. The default value of the frequency-dependent quality factor is assumed to be

The default real part of admitance is assumed to be 

.. math::

    {\rm Re} Y_{\rm qp}(\omega) = \sqrt{\frac{2}{\pi}} \frac{8 E_J}{R_k \Delta} \
    \left(\frac{2 \Delta}{\hbar \omega} \right)^{3/2}  x_{\rm qp} \
    K_{0} \left( \frac{\hbar |\omega|}{2 k_B T} \right) \sinh \left( \frac{\hbar \omega }{2 k_B T} \right)


Qubits that support this noise channel include: 
:ref:`TunableTransmon <qubit_tunable_transmon>`, 
:ref:`Fluxonium <qubit_fluxonium>`, 
:ref:`FullZeroPi <qubit_fullzeropi>`, 
:ref:`ZeroPi <qubit_zeropi>`.

References: [Catelani2011]_, [Pop2014]_, [Smith2020]_

User-defined noise
-----------------------

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1``                                  |
+--------------------------------------------+-----------------------------------------+
| :math:`B_\lambda`                          | user defined                            |
+--------------------------------------------+-----------------------------------------+

All qubits support user defined noise, where both the noise operator as well as an arbitrary spectral density can be provided. 


Qubits that support this noise channel include: 
:ref:`Fluxonium <qubit_fluxonium>`, 
:ref:`FluxQubit <qubit_flux_qubit>`, 
:ref:`FullZeroPi <qubit_fullzeropi>`, 
:ref:`Transmon <qubit_tunable_transmon>`, 
:ref:`TunableTransmon <qubit_tunable_transmon>`, 
:ref:`ZeroPi <qubit_zeropi>`.

