.. scqubits
   Copyright (C) 2019, Jens Koch & Peter Groszkowski

.. _guide_plotparameters:

***************
Plot Parameters
***************

scqubits implements a number of routines for plotting a variety of viewgraphs, such as one-dimensional and
two-dimensional qubit wavefunctions, plots of energy spectra as a function of an external parameter and more. These
plot functions and methods take specific mandatory arguments which differ from case to case. In addition,
**each routine accepts a set of keyword arguments** ``**kwargs`` which are used

- for setting graphics options affecting the format of the plot, and
- for prompting graphical output to be written to a file.


Supported options
-----------------

The options supported by scqubits plotting routines fall into two groups. First, any option that can be set on the
side of Matplotlib via ``Axes.set_xxx(...)`` is accessible through the keyword `xxx`. For example, to employ
``Axes.set_xlim(0,10)`` one provides the keyword argument ``xlim=(0,10)`` to the scqubits plot function of interest. The
second set of options is specific to scqubits. The most common Matplotlib options and the custom scqubits options are
summarized here:

+------------------------------+------------------------------------------------+
| keyword                      | Description                                    |
+==============================+================================================+
| ``xlim``: `(float,float)`    | Lower/upper bounds for x axis                  |
+------------------------------+------------------------------------------------+
| ``ylim``: `(float,float)`    | Lower/upper bounds for y axis                  |
+------------------------------+------------------------------------------------+
| ``ymax``: `float`            | Only set upper bound on y axis                 |
+------------------------------+------------------------------------------------+
| ``xlabel``: `str`            | Label text for x axis                          |
+------------------------------+------------------------------------------------+
| ``ylabel``: `str`            | Label text for y axis                          |
+------------------------------+------------------------------------------------+
| ``title``: `str`             | Title for plot                                 |
+------------------------------+------------------------------------------------+
| ``figsize``: `(float, float)`| Figure size in inches                          |
+------------------------------+------------------------------------------------+
| ``fig_ax``: `(Figure, Axes)` | If provided, plot is added to existing objects |
+------------------------------+------------------------------------------------+
| ``filename``: `str`          | If provided, plot is written to file           |
+------------------------------+------------------------------------------------+

A number of additional options falling in the ``Axes.set_xxx(...)`` category is available; consult the
``matplotlib.axes.Axes`` `API documentation`__ to see the complete list.

.. _API: https://matplotlib.org/api/axes_api.html#the-axes-class
__ API_






Returns of plot functions
-------------------------

Every scqubit routine for plotting returns a tuple ``(Figure, Axes)`` of Matplotlib objects. These can be used for
further processing by the user. In Jupyter, lines calling plot routines can be ended with a ``;`` to avoid the text
output indicating the returned objects.