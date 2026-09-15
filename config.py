"""
config.py – Centralised configuration for the DataLens Streamlit app.

The Groq API key and model are loaded from the .env file.

Required environment variables:
    GROQ_API_KEY
    GROQ_MODEL
"""

from __future__ import annotations

import os
from dotenv import load_dotenv


# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

load_dotenv()


# =============================================================================
# GROQ CONFIGURATION
# =============================================================================

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

GROQ_MODEL: str | None = os.getenv("GROQ_MODEL")