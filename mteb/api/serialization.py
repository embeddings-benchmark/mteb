"""Pre-serialised JSON payload primitives.

Pulled out of :mod:`mteb.api.cache` so both ``cache`` and ``routes`` can depend
on it without forming a cycle. Stays free of intra-package imports so it sits
at the bottom of the import graph.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class Serialized:
    """Pre-serialised schema response — raw bytes + optional gzipped variant + ETag."""

    body: bytes
    body_gzip: bytes | None  # None when payload is too small to bother compressing
    etag: str


# Tiny payloads don't gain from gzip; framing alone is ~20 bytes of overhead.
_GZIP_MIN_BYTES = 1024


def serialize_bytes(body: bytes) -> Serialized:
    """Wrap raw JSON bytes with a gzipped variant (when worth it) + matching ETag.

    Synchronous — callers that hit this on a cold path should run it through
    :func:`asyncio.to_thread` so the gzip pass doesn't stall the event loop on
    multi-MB payloads.
    """
    body_gzip = (
        gzip.compress(body, compresslevel=6) if len(body) >= _GZIP_MIN_BYTES else None
    )
    etag = '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest() + '"'
    return Serialized(body=body, body_gzip=body_gzip, etag=etag)


def serialize_schema(schema: BaseModel) -> Serialized:
    """Serialise a pydantic model with by-alias JSON, then wrap with gzip + ETag."""
    return serialize_bytes(schema.model_dump_json(by_alias=True).encode())
