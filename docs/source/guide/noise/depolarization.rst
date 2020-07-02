.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Depolarization
================

Depolarization noise captures transitions between eigenstates. scqubits uses standar Fermi's Golder Rule approach to approximate these transitions due to different noise channels. 

A transition rate from state :math:`i` to state :math:`j`, can be expressed as 

.. math::

   \Gamma_{ij} = \frac{1}{\hbar^2} |\langle i| A_{\lambda} |j \rangle|^2 S(\omega_{ij}),

where :math:`A_\lambda` is the noise operator, while S(\omega_ij) the spectral density function, evaluated at the angular frequency :math:`\omega_{ij} = \omega_{j} - \omega_{i}`. A positive :math:`\omega_{ij}` corrsponds to a decay where the system gives of energy to the bath, while a negative :math:`\omega_{ij}` to an excitation, where the system takes energy from the bath. 

Unless stated otherwise, each of the depolarizing noise channels assume that detailed balanced is satisfied, resulting in

.. math::

    \frac{S(-\omega)}{S(\omega)} = \exp{\frac{\hbar \omega}{k_B T}},

where :math:`T` is the bath temperature, and :math:`k_B` the Boltzmann constant.


Capacitive noise
-----------------------

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1_capacitive_loss``                  |
+--------------------------------------------+-----------------------------------------+
| :math:`A_\lambda`                          | :math:`n_g` (charge)                    |
+--------------------------------------------+-----------------------------------------+

The spectral density of this noise channel is [Smith2020]_:

.. math::

    S(\omega) = \frac{2 \hbar}{C_J Q_{\rm cap}(\omega)} \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right)

where :math:`C_J` is the relevant capacitance, and :math:`Q_{\rm cap}` the corresponding capacitive quality factor.
The default value of the frequency-dependent quality is assumed to be

.. math::

    Q_{\rm cap}(\omega) = 

Inductive noise
-----------------------


+--------------------------------------------+-----------------------------------------+
| Method name                                | ``t1_inductive_loss``                   |
+--------------------------------------------+-----------------------------------------+
| :math:`A_\lambda`                          | :math:`\phi` (phase)                    |
+--------------------------------------------+-----------------------------------------+

The spectral density of this noise channel is [Smith2020]_:

.. math::

    S(\omega) = \frac{2 \hbar}{L Q_{\rm ind}(\omega)} \left(1 + \coth \frac{\hbar |\omega|}{2 k_B T} \right)

where :math:`L_J` is the relevant superinductance, and :math:`Q_{\rm ind}` the corresponding inductive quality factor.
The default value of the frequency-dependent quality is assumed to be

.. math::

    Q_{\rm ind}(\omega) = 


Charge-coupled impedance noise
------------------------------

.. autosummary::

    scqubits.core.noise.NoisySystem.t1_charge_impedance

Flux-bias line noise
-------------------------


.. autosummary::

    scqubits.core.noise.NoisySystem.t1_flux_bias_line

Quasiparticle-tunneling noise
----------------------------------

.. autosummary::

    scqubits.core.noise.NoisySystem.t1_quasiparticle_tunneling



User-defined noise
-----------------------

.. autosummary::

    scqubits.core.noise.NoisySystem.t1

