.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide-io:

***************************************
Writing Data and Figures to File
***************************************

Much of the data computed with scqubits can easily be exported and written to files. Supported file formats for data
are:

+-----------------------------+--------------------------------------------------------------+
| h5                          | HDF5 file with `gzip` compression                            |
+-----------------------------+--------------------------------------------------------------+
| csv                         | comma-separated values                                       |
+-----------------------------+--------------------------------------------------------------+
| pdf, jpg, png, svg, ...     | graphics formats supported by Matplotlib for saving figures  |
+-----------------------------+--------------------------------------------------------------+

.. _guide-io-data:

Writing computed data to file
-----------------------------

The primary vehicles for writing data to disk is to store them in a ``SpectrumData`` or a ``DataStorage`` object.
Many of the computational routines for spectral data and parameter sweeps return data in this form. For example, ::


    import numpy as np
    import scqubits

    transmon = scqubits.Transmon(...init parameters...)

    ng_list = np.linspace(-2, 2, 220)
    specdata = transmon.get_spectrum_vs_paramvals('ng', ng_list, get_eigenstates=True)


The object ``specdata`` is of type ``SpectrumData`` and, in this case, contains information about the underlying system
parameters, the parameter used in the scan and its values, as well as the resulting eigenvalues and eigenvectors. The
data can be exported to a file by using::


    specdata.filewrite('output.h5')


The preferred output file format for data can changed by modifying scqubits, see :ref:`guide-settings`.

When using h5 files, data can also be read back from disk into a ``SpectrumData`` or ``DataStorage`` object::


   newspecdata = SpectrumData.create_from_file('output.h5')


.. _guide-io-figures:

Exporting figures to file
-----------------------------

All plotting routines in scqubits can be called with a keyword argument ``filename='output.pdf'``. This will generate
the plot, display it, and in addition write it to the specified file. The output format is determined by the file suffix
provided in the filename. A variety of output formats is supported via Matplotlib, including pdf, jpg, png, svg.