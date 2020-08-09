.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Depolarization
================

Noise may cause depolarization of the qubit by inducing spontaneous transitions among eigenstates. scqubits uses the
standard perturbative approach (Fermi's Golden Rule) to approximate the resulting transition rates due to different
noise channels.

The rate of a transition from state :math:`i` to state :math:`j` can be expressed as

.. math::

   \Gamma_{ij} = \frac{1}{\hbar^2} |\langle i| A_{\lambda} |j \rangle|^2 S(\omega_{ij}),

where :math:`A_\lambda` is the noise operator, and :math:`S(\omega_{ij})` the spectral density function evaluated at
the angular frequency associated with the transition frequeny, :math:`\omega_{ij} = \omega_{j} - \omega_{i}`.
:math:`\omega_{ij}` is positive in the case of  decay (the qubit emits energy to the bath), and negative in case of
excitations (the qubit absorbs energy from the bath).

Unless stated otherwise, it is assumed that the depolarizing noise channels satisfy detailed balanced. This implies

.. math::

    \frac{S(-\omega)}{S(\omega)} = \exp{\frac{\hbar \omega}{k_B T}},

where :math:`T` is the bath temperature, and :math:`k_B` Boltzmann's constant.


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

    Q_{\rm cap}(\omega) =   TODO: MISSING


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

where :math:`L_J` is the relevant inductance or superinductance, and :math:`Q_{\rm ind}` the corresponding inductive
quality factor. The default value of the frequency-dependent quality is assumed to be

.. math::

    Q_{\rm ind}(\omega) =    TODO: MISSING


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

