.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide_noise:

******
Noise
******

scqubits implements calculations of dephasing and depolarization rates for each of the qubits. These calculations account for various noise channels that typical superconducting qubits are subject to. 


Dephasing Noise
=================

1/f noise
---------


abc [Manucharyan2009]_ 

.. math::

   H=-4E_\text{C}\partial_\phi^2-E_\text{J}\cos(\phi-\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2.

abc :math:`E_C` is the charging energy, :math:`E_J` the Josephson energy, :math:`E_L` the inductive energy, and
:math:`\varphi_\text{ext}=2\pi \Phi_\text{ext}/\Phi_0` the external flux in dimensionless form. The ``Fluxonium`` class
internally uses the :math:`E_C`-:math:`E_L` harmonic-oscillator basis [Zhu2013]_ with truncation level specified by ``cutoff``.



.. tabularcolumns:: | p{3cm} | p{3cm} | p{3cm} |

.. cssclass:: table-striped

+-------------------------+------------------------------+
| Noise Type              |  Noise Strength              |
+=========================+==============================+
| ``Flux``                |                              | 
+-------------------------+------------------------------+
| ``Charge``              |                              |
+-------------------------+------------------------------+
| ``Critical Current``    |                              |
+-------------------------+------------------------------+


Depolarization Noise
=================

Flux bias line noise
---------




