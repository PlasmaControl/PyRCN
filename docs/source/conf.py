# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pyrcn import __version__
import doctest
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'PyRCN'
copyright = '2020-2024, Chair of Speech Technology and Cognitive Systems, ' \
            'TU Dresden; 2024-2026, Princeton University'
author = 'Peter Steiner, Azarakhsh Jalalvand, Simon Stone, Peter Birkholz'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
]

# Generate the per-object stub pages referenced by ``.. autosummary::``
# directives at build time (also regenerated on Read the Docs).
autosummary_generate = True
master_doc = 'index'

# Resolve cross-references to the scientific Python stack we build on.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

# Render numpydoc "Attributes" sections as info fields rather than separate
# object descriptions (avoids duplicate-object-description warnings for
# dataclass attributes and properties).
napoleon_use_ivar = True

# Make the runnable doctests robust to line wrapping in reprs (estimator reprs
# wrap at 79 columns) without weakening what they check.
doctest_default_flags = (
    doctest.ELLIPSIS
    | doctest.NORMALIZE_WHITESPACE
    | doctest.DONT_ACCEPT_TRUE_FOR_1
)

# Do not document scikit-learn's dynamically added metadata-routing methods
# (set_*_request); their docstrings reference sklearn-only glossary terms and
# labels that do not resolve in this documentation.
autodoc_default_options = {
    'exclude-members': (
        'set_fit_request,set_predict_request,set_partial_fit_request,'
        'set_score_request,set_transform_request,'
        'set_predict_proba_request,set_inverse_transform_request'
    ),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme tweaks: a compact, always-expanded left navigation.
html_theme_options = {
    'logo_only': True,
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
}

# Brand assets. The logo sits at the top of the sidebar; the SVG doubles as
# the browser-tab favicon (modern browsers render SVG favicons).
html_logo = '_static/img/pyrcn_logo.svg'
html_favicon = '_static/img/pyrcn_logo.svg'
html_title = 'PyRCN documentation'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Layer the brand accent on top of the Read the Docs theme.
html_css_files = ['custom.css']
