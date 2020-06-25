.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _changelog:

**********
Change Log
**********

Version 1.2.3
+++++++++++++

- **Bug fix**: the ``FullZeroPi`` Hamiltonian was incorrect in the case of nonzero ``dC``.
- improvement: thanks to adjusted ARPACK options, diagonalization should be noticeably faster for ``ZeroPi`` and ``FullZeroPi``.
- make ``pathos`` and ``dill`` the default for multiprocessing.


Version 1.2.2
+++++++++++++

- **Bug fix**: implementation of the ``add_hc=True`` flag in ``InteractionTerm`` involved a bug that could lead to incorrect results
- update to plotting routines now supports various extra plotting options such as ``linestyle=...`` etc.
- added ``TunableTransmon`` class for flux-tunable transmon, including junction asymmetry
- limit support to Python >= 3.6
- corrections to documentation of ``FullZeroPi``
- added missing jupyter notebook illustrating use of ``HilbertSpace`` and ``ParameterSweep``
- overhaul of file IO system now allows saving and loading various scqubit data via a custom h5 file format
- ipywidget support for creating qubits inside jupyter (try, for example, ``tmon = scqubits.Transmon.create()``)



Version 1.2.1
+++++++++++++
- update to the setup script to properly include testing data with the PyPi release.


Version 1.2
+++++++++++

**Major changes/additions**
   - scqubits now offers multiprocessing support for a number of methods.
   - Introduced checks ensuring that umbrella objects like ``HilbertSpace`` and ``ParameterSweep`` instances do not accidentally go "out-of-sync" with respect to their basic components. When needed, warnings are thrown for the user to re-run sweeps or spectrum lookups.

**Under the hood:**
   - Monitoring for changes of interdependent class instances is implemented through a central dispatch system. (disable: ``settings.DISPATCH_ENABLED``)
   - Removed ``HilbertSpace`` reference from within `InteractionTerm` (throws deprecation warning if still used)
   - Made ``HilbertSpace`` inherit from ``tuple`` rather than ``list``; composition changes to ``HilbertSpace`` warrant generating a new ``HilbertSpace`` instance
   - Shifted ``InteractionTerm.hamiltonian`` to ``HilbertSpace.interaction_hamiltonian``
   - Created ``DataStore`` as general purpose parent class to ``SpectrumData``
   - No longer store custom data inside ``ParameterSweep``, ``sweep_generators.py`` functions return ``DataStore`` objects


Version 1.1.1
+++++++++++++

   - fixed a bug in display of ``FluxQubit`` wavefunction
   - internal refactoring


Version 1.1.0
+++++++++++++

   - new class ``InteractionTerm`` works in tandem with ``HilbertSpace`` to ease setup of composite systems with pairwise interactions
   - new ``ParameterSweep`` class efficiently generates spectral data for performing a scan of a ``HilbertSpace`` object over an external parameters
   - new ``Explorer`` class introduces interactive plots (see docs and demo ipynb)
   - cleaned up implementation of file Serializable operations


Version 1.0.0 (first release)
++++++++++++++++++++++++++++++
