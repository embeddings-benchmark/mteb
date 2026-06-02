"""FastAPI surface for the MTEB leaderboard.

Run locally with::

    pip install -e ".[api]"
    uvicorn mteb.api.app:app --reload --port 8000
"""

from __future__ import annotations

from mteb.api.app import create_app

__all__ = ["create_app"]
