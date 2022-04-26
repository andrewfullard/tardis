# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Astropy documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy.sphinx.conf for which values are set there.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath(".."))
# IMPORTANT: the above commented section was generated by sphinx-quickstart, but
# is *NOT* appropriate for astropy or Astropy affiliated packages. It is left
# commented out with this explanation to make it clear why this should not be
# done. If the sys.path entry above is added, when the astropy.sphinx.conf
# import occurs, it will import the *source* version of astropy instead of the
# version installed (if invoked as "make html" or directly with sphinx), or the
# version in the build directory (if "python setup.py build_sphinx" is used).
# Thus, any C-extensions that are needed to build the documentation will *not*
# be accessible, and the documentation will not build correctly.

import os
import sys
import datetime
from importlib import import_module

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print(
        "ERROR: the documentation requires the sphinx-astropy package to be installed"
    )
    sys.exit(1)

# Get configuration information from setup.cfg
from configparser import ConfigParser

conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__), "..", "setup.cfg")])
setup_cfg = dict(conf.items("metadata"))

# -- General configuration ----------------------------------------------------

# By default, code is not highlighted.
highlight_language = "none"

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = "1.2"

# To perform a Sphinx version check that needs to be more specific than
# major.minor, call `check_sphinx_version("X.Y.Z")` here.
# check_sphinx_version("1.2.1")

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")
exclude_patterns.append("_build")
exclude_patterns.append("**.ipynb_checkpoints")
exclude_patterns.append("resources/research_done_using_TARDIS/ads.ipynb")

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
"""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx-jsonschema",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.apidoc",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "numpydoc",
    "recommonmark",
]

bibtex_bibfiles = ["tardis.bib"]

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
    "astropy": ("http://docs.astropy.org/en/stable/", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev/", None),
}

apidoc_module_dir = "../tardis"
apidoc_output_dir = "api"
apidoc_excluded_paths = [
    "*tests*",
    "*setup_package*",
    "*conftest*",
    "*version*",
]
apidoc_separate_modules = True

## https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = False

# Force MathJax v2, see: https://github.com/spatialaudio/nbsphinx/issues/572#issuecomment-853389268
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
mathjax2_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
        "ignoreClass": "document",
        "processClass": "math|output_area",
    }
}

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. raw:: html
    
    <style>
        .launch-btn {
            background-color: #2980B9;
            border: none;
            border-radius: 4px;
            color: #fcfcfc;
            font-family: inherit;
            text-decoration: none;
            padding: 3px 8px;
            letter-spacing: 0.03em;
            display: inline-block;
            line-height: 1.5em;
        }

        .launch-btn:hover {
            background-color: #1b6391;
            color: #fcfcfc;
        }

        .launch-btn:visited {
            color: #fcfcfc;
        }

        .note-p {
            margin-bottom: 0.4em;
            line-height: 2em;
        }
    </style>
    
    <div class="admonition note">
    <p class="note-p">You can interact with this notebook online: <a href="https://mybinder.org/v2/gh/tardis-sn/tardis/HEAD?filepath={{ docname|e }}" class="launch-btn" target="_blank" rel="noopener noreferrer">Launch interactive version</a></p>
    </div>
"""

if os.getenv("DISABLE_NBSPHINX") == "1":
    nbsphinx_execute = "never"
else:
    nbsphinx_execute = "auto"


# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = setup_cfg["name"]
author = setup_cfg["author"]
copyright = "2013-{0}, {1}".format(datetime.datetime.now().year, author)

# The version info for the project you"re documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

import_module(setup_cfg["name"])
package = sys.modules[setup_cfg["name"]]

# The short X.Y version.
version = "latest"  # package.__version__.split("-", 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__


# -- Options for HTML output --------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, "bootstrap-astropy",
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some of the
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.


# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
import sphinx_rtd_theme

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
html_theme = "sphinx_rtd_theme"


html_theme_options = {
    #    "logotext1": "tardis",  # white,  semi-bold
    #    "logotext2": "",  # orange, light
    #    "logotext3": ":docs"   # white,  light
}


# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}
html_static_path = ["_static"]
templates_path = ["_templates"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = ""

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "tardis_logo.ico"

# If not "", a "Last updated on:" timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ""

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = project  # "{0} v{1}".format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# Prefixes that are ignored for sorting the Python module index
modindex_common_prefix = ["tardis."]


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual")
]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", project.lower(), project + " Documentation", [author], 1)
]


# -- Options for the edit_on_github extension ---------------------------------

if setup_cfg.get("edit_on_github").lower() == "true":

    extensions += ["sphinx_astropy.ext.edit_on_github"]

    edit_on_github_project = setup_cfg["github_project"]
    edit_on_github_branch = "main"

    edit_on_github_source_root = ""
    edit_on_github_doc_root = "docs"

# -- Resolving issue number to links in changelog -----------------------------
github_issues_url = "https://github.com/{0}/issues/".format(
    setup_cfg["github_project"]
)


# -- Options for linkcheck output -------------------------------------------
linkcheck_retry = 5
linkcheck_ignore = [
    r"https://github\.com/tardis-sn/tardis/(?:issues|pull)/\d+",
]
linkcheck_timeout = 180
linkcheck_anchors = False

# -- Turn on nitpicky mode for sphinx (to warn about references not found) ----
#
# nitpicky = True
# nitpick_ignore = []
#
# Some warnings are impossible to suppress, and you can list specific references
# that should be ignored in a nitpick-exceptions file which should be inside
# the docs/ directory. The format of the file should be:
#
# <type> <class>
#
# for example:
#
# py:class astropy.io.votable.tree.Element
# py:class astropy.io.votable.tree.SimpleElement
# py:class astropy.io.votable.tree.SimpleElementWithContent
#
# Uncomment the following lines to enable the exceptions:
#
# for line in open("nitpick-exceptions"):
#     if line.strip() == "" or line.startswith("#"):
#         continue
#     dtype, target = line.split(None, 1)
#     target = target.strip()
#     nitpick_ignore.append((dtype, six.u(target)))


# -- Creating redirects ------------------------------------------------------

# One entry per redirect. List of tuples: (old_fpath, new_fpath)
# Paths are relative to source dir i.e. "docs/" & must include file extension
# Only source files that convert to html like .rst, .ipynb, etc. are allowed

redirects = [
    ("using/gui/index.rst", "using/visualization/index.rst"),
]


# -- Sphinx hook-ins ---------------------------------------------------------

import re
import pathlib
import requests
import textwrap
import warnings
from shutil import copyfile


def generate_ZENODO(app):
    """Creating ZENODO.rst
    Adapted from: https://astrodata.nyc/posts/2021-04-23-zenodo-sphinx/"""
    CONCEPT_DOI = "592480"  # See: https://help.zenodo.org/#versioning
    zenodo_path = pathlib.Path("resources/ZENODO.rst")

    try:
        headers = {"accept": "application/x-bibtex"}
        response = requests.get(
            f"https://zenodo.org/api/records/{CONCEPT_DOI}", headers=headers
        )
        response.encoding = "utf-8"
        citation = re.findall("@software{(.*)\,", response.text)
        zenodo_record = (
            f".. |ZENODO| replace:: {citation[0]}\n\n"
            ".. code-block:: bibtex\n\n"
            + textwrap.indent(response.text, " " * 4)
        )

    except Exception as e:
        warnings.warn(
            "Failed to retrieve Zenodo record for TARDIS: " f"{str(e)}"
        )

        not_found_msg = """
                        Couldn"t retrieve the TARDIS software citation from Zenodo. Get it 
                        directly from `this link <https://zenodo.org/record/{CONCEPT_DOI}>`_    .
                        """

        zenodo_record = (
            ".. |ZENODO| replace:: <TARDIS SOFTWARE CITATION HERE> \n\n"
            ".. warning:: \n\n" + textwrap.indent(not_found_msg, " " * 4)
        )

    with open(zenodo_path, "w") as f:
        f.write(zenodo_record)

    print(zenodo_record)


def generate_tutorials_page(app):
    """Create tutorials.rst"""
    notebooks = ""

    for root, dirs, fnames in os.walk("io/"):
        for fname in fnames:
            if fname.endswith(".ipynb") and "checkpoint" not in fname:
                notebooks += f"\n* :doc:`{root}/{fname[:-6]}`"

    title = "Tutorials\n*********\n"
    description = "The following pages contain the TARDIS tutorials:"

    with open("tutorials.rst", mode="wt", encoding="utf-8") as f:
        f.write(f"{title}\n{description}\n{notebooks}")


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Exclude specific functions/methods from the documentation"""
    exclusions = ("yaml_constructors", "yaml_implicit_resolvers")
    exclude = name in exclusions
    return skip or exclude


def to_html_ext(path):
    """Convert extension in the file path to .html"""
    return os.path.splitext(path)[0] + ".html"


def create_redirect_files(app, docname):
    """Create redirect html files at old paths specified in `redirects` list."""
    template_html_path = os.path.join(
        app.srcdir, "_templates/redirect_file.html"
    )

    if app.builder.name == "html":
        for (old_fpath, new_fpath) in redirects:
            # Create a page redirection html file for old_fpath
            old_html_fpath = to_html_ext(os.path.join(app.outdir, old_fpath))
            os.makedirs(os.path.dirname(old_html_fpath), exist_ok=True)
            copyfile(template_html_path, old_html_fpath)

            # Replace url placeholders i.e. "#" in this file with the new url
            new_url = os.path.relpath(
                to_html_ext(new_fpath), os.path.dirname(old_fpath)
            )  # urls in a html file are relative to the dir containing it
            with open(old_html_fpath) as f:
                new_content = f.read().replace("#", new_url)
            with open(old_html_fpath, "w") as f:
                f.write(new_content)


def setup(app):
    app.connect("builder-inited", generate_ZENODO)
    app.connect("builder-inited", generate_tutorials_page)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("build-finished", create_redirect_files)
