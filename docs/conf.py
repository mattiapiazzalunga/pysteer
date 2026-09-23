"""Sphinx configuration for pysteer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "pysteer"
author = "Mattia Piazzalunga"
copyright = "2026, Mattia Piazzalunga and pysteer contributors"
release = "0.1.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = [
    "accelerate",
    "pandas",
    "torch",
    "tqdm",
    "transformers",
]

html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "pysteer activation steering docs"
html_short_title = "pysteer"
html_baseurl = "https://mattiapiazzalunga.github.io/pysteer/"
html_meta = {
    "description": (
        "pysteer is a Python activation-steering library for LLMs, "
        "PyTorch transformer language models, representation engineering, "
        "and inference-time model steering."
    ),
    "keywords": (
        "pysteer, activation steering, Python activation steering, LLM steering, "
        "representation engineering, activation engineering, PyTorch transformers, "
        "mechanistic interpretability, AI safety, model steering"
    ),
}
html_show_sourcelink = True
