# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import datetime
import gnn_demo
import sphinx_rtd_theme
import doctest

# -- Project information -----------------------------------------------------

now = datetime.datetime.now()
project = 'gnn_demo'
author = 'hanhui'
copyright = '{}/{}, {}'.format(now.year, now.month, author)
version = gnn_demo.__version__
release = gnn_demo.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.todo', 
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]
autodoc_mock_imports = [
    "tensorlayer",
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
# html_logo = '_static/img/gnn_demo.png'
# html_static_path = ['_static']
# html_context = {'css_files': ['_static/css/custom.css']}
# rst_context = {'torch_geometric': gnn_demo}
doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}