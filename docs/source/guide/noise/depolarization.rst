.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Depolarization
================

Depolarization noise captures transitions between eigenstates. scqubits uses standar Fermi's Golder Rule approach to approximate these transitions due to different noise channels. 

A transition rate from state :math:`i` to state :math:`j`, can be expressed as 

.. math::

   \Gamma_{ij} = \frac{1}{\hbar^2} |\langle i| A_{\lambda} |j \rangle|^2 S(\omega_{ij}),

where :math:`A_\lambda` is the noise operator, while S(\omega_ij) the spectral density function, evaluated at the angular frequency :math:`\omega_{ij} = \omega_{j} - \omega_{i}`. A positive :math:`\omega_{ij}` corrsponds to a decay where the system gives of energy to the bath, while a negative :math:`\omega_{ij}` to an excitation, where the system takes energy from the bath. 



Capacitive noise
-----------------------

.. autosummary::

    scqubits.noise.NoisySystem.t1_capacitive_loss


Inductive noise
-----------------------

.. autosummary::

    scqubits.noise.NoisySystem.t1_inductive_loss


Charge-coupled impedance noise
------------------------------

.. autosummary::

    scqubits.noise.NoisySystem.t1_charge_impedance

Flux-bias line noise
-------------------------


.. autosummary::

    scqubits.noise.NoisySystem.t1_flux_bias_line

Quasiparticle-tunneling noise
----------------------------------

.. autosummary::

    scqubits.noise.NoisySystem.t1_quasiparticle_tunneling



User-defined noise
-----------------------

.. autosummary::

    scqubits.noise.NoisySystem.t1

