.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Effective Noise
================

scqubits can approximate effective noise, which is calculated by considering total contributions from a variety of noise channels. In the case of a depolarization noise, the effective noise is obtained from 

.. math::

    \frac{1}{T_{1}^{\rm eff}} = \sum_k \frac{1}{T_{1}^{k}},


where the sum runs over all the noise channels that the user wants included. By default, those include ones returned by the ``effective_noise_channels`` method of each qubit. A different list of noise channels can be also provided by the user. 

Similarly, users can calculate effective depahsing, which includes contributions from both pure dephasing, as well as depolarization channels. Such a :math:`T_{2}` time is defined as

.. math::

    \frac{1}{T_{2}^{\rm eff}} = \sum_k \frac{1}{T_{\phi}^{k}} +  \frac{1}{2} \sum_j \frac{1}{T_{1}^{j}}, 

where :math:`k` (:math:`j`) run over the relevant pure dephasing (depolariztion) channels. 

For more information on the method signatures, see the 
:ref:`API documentation <apidoc>`
to see the complete list. 

