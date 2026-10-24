"""Pre-serialised JSON payload primitives."""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class Serialized:
    """Pre-serialised schema response — bytes + optional gzipped variant + ETag."""

    body: bytes
    body_gzip: bytes | None
    etag: str


# Tiny payloads don't gain from gzip; framing alone is ~20 bytes.
_GZIP_MIN_BYTES = 1024


def serialize_bytes(body: bytes) -> Serialized:
    """Wrap raw JSON bytes with a gzipped variant (when worth it) + matching ETag.

    Synchronous — callers on cold paths should ``asyncio.to_thread`` this so
    gzip on multi-MB payloads doesn't stall the event loop.
    """
    body_gzip = (
        gzip.compress(body, compresslevel=6) if len(body) >= _GZIP_MIN_BYTES else None
    )
    etag = '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest() + '"'
    return Serialized(body=body, body_gzip=body_gzip, etag=etag)


def serialize_schema(schema: BaseModel) -> Serialized:
    """Serialise a pydantic model (by-alias) then wrap with gzip + ETag."""
    return serialize_bytes(schema.model_dump_json(by_alias=True).encode())
