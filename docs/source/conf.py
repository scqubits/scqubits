# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'sc_qubits'
copyright = '2019, Jens Koch, Peter Groszkowski'
author = 'Jens Koch, Peter Groszkowski'


# -- General configuration ---------------------------------------------------




# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'numpydoc',
              'sphinx.ext.extlinks',
              'sphinx.ext.viewcode',
              'sphinx.ext.ifconfig',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'nbsphinx',
              'sphinx.ext.mathjax']


html_show_sourcelink = False

autodoc_default_flags = ['members', 'inherited-members']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
full_logo= True

html_theme_options = {
}
# Add any paths that contain custom themes here, relative to this directory.

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'sc_qubits Documentation'

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = u'sc_qubits'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []
todo_include_todos = True

numpydoc_show_class_members = False
napoleon_numpy_docstring = True
napoleon_use_admonition_for_notes = True