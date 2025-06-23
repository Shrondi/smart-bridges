# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SMART-BRIDGES'
copyright = '2025, Carlos Lucena Robles - Universidad de Córdoba'
author = 'Carlos Lucena Robles - Universidad de Córdoba'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.githubpages",
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,  # Desactiva el colapso automático de las secciones
    'navigation_depth': 0,          # 0 significa mostrar toda la profundidad
    'sticky_navigation': True,     # Navegación fija al hacer scroll
}
