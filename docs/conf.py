import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "Minerva-OPT"
copyright = "2026, Gabriel Gutierrez"
author = "Gabriel Gutierrez"
release = "1.2.0"
version = "1.2"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

# ---------------------------------------------------------------------------
# Napoleon — NumPy docstring style
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
add_module_names = False

# Heavy runtime deps are not installed in the docs build environment.
autodoc_mock_imports = [
    "ray",
    "lightning",
    "torch",
    "minerva",
    "hyperopt",
    "jsonargparse",
]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML — Furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "Minerva-OPT"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/gabrielbg0/Minerva-OPT",
    "source_branch": "main",
    "source_directory": "docs/",
}
