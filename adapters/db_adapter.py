"""
Adapter that replicates the db_controller interface used throughout the
original AppServer codebase.

Wire this to a real MongoDB instance by setting the MONGODB_URI environment
variable (see config.py).  All functions raise NotImplementedError by default
so missing configuration is surfaced immediately at runtime.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pymongo import MongoClient

import config

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI)
    return _client


# ---------------------------------------------------------------------------
# db_controller-compatible helpers
# ---------------------------------------------------------------------------

def get_db_data(
    db_name: str,
    collection_name: str,
    filter_term: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return all documents from *db_name*.*collection_name* matching *filter_term*."""
    flt = filter_term or {}
    col = _get_client()[db_name][collection_name]
    return list(col.find(flt))


def insert_docs(
    db_name: str,
    collection_name: str,
    docs: List[Dict[str, Any]],
) -> None:
    """Insert *docs* into *db_name*.*collection_name*."""
    if not docs:
        return
    col = _get_client()[db_name][collection_name]
    col.insert_many(docs)


def get_all_db_names() -> List[str]:
    """Return all database names visible to the configured MongoDB client."""
    return _get_client().list_database_names()
