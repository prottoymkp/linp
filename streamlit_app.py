"""Streamlit Cloud-friendly entrypoint.

This module simply imports the UI script so deployments can use the common
`streamlit_app.py` main file path without changing app behavior.
"""

from app import ui  # noqa: F401
