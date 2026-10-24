"""MTEB Result Cache Module.

This module provides functionality for caching and managing MTEB evaluation results locally,
with support for syncing with the remote results repository on GitHub.
"""

from mteb.cache.result_cache import LoadExperimentEnum, ResultCache

__all__ = [
    "LoadExperimentEnum",
    "ResultCache",
]
