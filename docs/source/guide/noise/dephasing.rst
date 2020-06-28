.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Dephasing
==============

Dephasing noise leads to coherence loss between different levels of a quantum system. 


1/f noise
---------------

One of the most important noise channels that affects many superconducting qubits is the 1/f dephasing noise, with a spectral density function defined by 

.. math::

   S(\omega) = \frac{2 \pi A_{\lambda}^{2} }{|\omega|}.

It typically leads to slow fluctuations of quantum energy spacing, resulting in decoherence [Ithier2005]_. 
In the above expression, :math:`A_{\lambda}` corresponds to a strength of a particular noise channel :math:`\lambda`. scqubits provides sensible defaults, for this quantity, but it something that can also be set by the user. 

The resulting dephasing time is given by 

.. math::

   t_{\phi} = \sqrt{2} A_{\lambda} \frac{\partial \omega_{01}}{\partial \lambda}  \sqrt{| \ln \omega_{\rm low} t_{\rm exp} |}


with the following parameters:

+-----------------------------+---------------+---------------------------------+
| Parameter                   | Default Value | Description                     |
+-----------------------------+---------------+---------------------------------+
| :math:`\omega_{\rm low}`    |  `1 rad/s`    | Low frequency cutoff            |
+-----------------------------+---------------+---------------------------------+
| :math:`t_{\rm exp}`         |  `1 s`        | Experiment time                 |
+-----------------------------+---------------+---------------------------------+

The frequency derivatives in the above expressions are calculated from matrix elements of :math:`\partial_\lambda H`. 

The most general method that can be used to calculate 1/f depahsing noise from an arbitrary noise channel is called ``tphi_1_over_f()``. However, assuming a given qubit supports it, each of the common 1/f noise channels has its own predefined method, with appropriately set defaults (such as the noise strength :math:`A_{\lambda}` as well as the correct noise operator :math:`\partial_\lambda H`).


See the API for method signatures. 

1/f flux noise
^^^^^^^^^^^^^^^^^^^^^

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``tphi_1_over_f_flux``                  |
+--------------------------------------------+-----------------------------------------+
| Noise operator                             | :math:`\partial H/\partial \Phi_{x}`    |
+--------------------------------------------+-----------------------------------------+
| Default value of  :math:`A_{\lambda}`      |  :math:`10^{-6} \Phi_0`                 |
+--------------------------------------------+-----------------------------------------+


Qubits that support this noise channel include: `TunableTransmon`, `Fluxonium`, `ZeroPi`, `FullZeroPi`.

1/f charge noise
^^^^^^^^^^^^^^^^^^^^^


+--------------------------------------------+-----------------------------------------+
| Method name                                | ``tphi_1_over_ng``                      |
+--------------------------------------------+-----------------------------------------+
| Noise operator                             | :math:`\partial H/\partial /n_g`         |
+--------------------------------------------+-----------------------------------------+
| Default value of  :math:`A_{\lambda}`      |  :math:`10^{-4} e`                      |
+--------------------------------------------+-----------------------------------------+

Qubits that support this noise channel include: `Transmon`, `TunableTransmon`, `ZeroPi`, `FullZeroPi`.

1/f criticial current noise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The critical current noise is suspected to arise from trapping and de-trapping of charges at defect sites of Josephson junctions. These trapped charges then may drop the tunneling through some regions of the junction, leading to current fluctuations.   

+--------------------------------------------+-----------------------------------------+
| Method name                                | ``tphi_1_over_f_cc``                    |
+--------------------------------------------+-----------------------------------------+
| Noise operator                             | :math:`\partial H/\partial I_{c}`       |
+--------------------------------------------+-----------------------------------------+
| Default value of  :math:`A_{\lambda}`      |  :math:`10^{-7} I_{c}`                  |
+--------------------------------------------+-----------------------------------------+


Qubits that support this noise channel include: `Transmon`, `TunableTransmon`, `Fluxonium`, `ZeroPi`, `FullZeroPi`.

Shot noise
---------------

.. todo:: To be added for certain qubits


