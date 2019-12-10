Source for scqubits documentation
==================================

This directory contains the source files for the scqubits documentation.

The built documentation is available online, see http://xxx

[//]: # (TODO: fill in above link once ReadTheDocs is set up)

Build requirements
------------------

* Sphinx: http://sphinx-doc.org/
* sphinx_rtd_theme
* numpydoc
* nbsphinx

In a conda environment use:
    
    $ conda install sphinx numpydoc sphinx_rtd_theme
    $ conda install -c conda-forge nbsphinx

Build
-----

    > make html
