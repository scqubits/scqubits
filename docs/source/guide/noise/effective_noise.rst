.. scqubits
   Copyright (C) 2017 and later, Jens Koch & Peter Groszkowski

Effective coherence times due to multiple noise channels
========================================================

In the case of multiple noise channels inducing qubit depolarization, the effective (total) depolarization time follows
from the addition of the individual rates,

.. math::

    \frac{1}{T_{1}^{\rm eff}} = \sum_k \frac{1}{T_{1}^{k}},


where the sum runs over all included noise channels.

An analogous statement holds for the effective dephasing time, which includes contributions from both pure dephasing
as well as depolarization channels. This total :math:`T_{2}` time is defined as

.. math::

    \frac{1}{T_{2}^{\rm eff}} = \sum_k \frac{1}{T_{\phi}^{k}} +  \frac{1}{2} \sum_j \frac{1}{T_{1}^{j}},

where :math:`k` (:math:`j`) runs over all relevant pure-dephasing (depolarization) channels.

scqubits can enables the evaluation of these effective coherence times. By default, the included noise channels are
the ones returned by the ``effective_noise_channels`` method for each qubit. A different list of noise channels can
be selected by the user.

TODO: Name the methods for obtaining the effective coherence times.

For more information on the method signatures, see the 
:ref:`API documentation <apidoc>`
to see the complete list. 

