# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # docs/source
REPO_ROOT = HERE.parents[2]  # repo root (docs/source -> docs -> repo)

# If your package is in repo_root/src/qemcmc:
SRC = REPO_ROOT / "src"
PKG = SRC / "qemcmc"

# If instead your package is in repo_root/QeMCMC/src/qemcmc, use:
# SRC = REPO_ROOT / "QeMCMC" / "src"
# PKG = SRC / "qemcmc"


project = "QeMCMC"
copyright = "2025, Feroz Hassan and Stuart Ferguson"
author = "Feroz Hassan"
release = "1.0.0"

print(HERE)
print(REPO_ROOT)
print(SRC)
print(PKG)


# sys.path.insert(0, os.path.abspath("../../QeMCMC/src"))
sys.path.insert(0, str(SRC))
autoapi_dirs = [str(PKG)]

# AutoAPI configuration
# autoapi_dirs = ["../../QeMCMC/src/qemcmc"]  # Path to your source code
autoapi_type = "python"
autoapi_template_dir = "_autoapi_templates"
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_keep_files = False


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "autoapi.extension",
    "sphinx_copybutton",
]


myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
    "smartquotes",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

myst_heading_anchors = 3

myst_url_schemes = ("http", "https", "mailto")


autosectionlabel_prefix_document = True

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "text": "Quantum Software Lab",
        "image_light": "_static/logo-qsl.jpeg",
        "image_dark": "_static/logo-qsl.jpeg",
    }
}
