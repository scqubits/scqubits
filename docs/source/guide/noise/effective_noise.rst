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

scqubits enables the evaluation of these effective coherence times. By default, the noise channels that are included in the calculation,
can be shown using the ``effective_noise_channels`` method for each qubit. Note that this list may not include all the noise 
channels that can actually be calculated for any given qubit, to see a list of those, the ``supported_noise_channels`` method 
could be used. Users can also explicitly specify what noise processes should be included in effective noise calculations. 


Calculating :math:`T_1` and :math:`T_2` can be done via a methods ``t1_effective`` and ``t2_effective`` respectively. 

For more information on the method signatures, see the 
:ref:`API documentation <apidoc>`
to see the complete list. 

