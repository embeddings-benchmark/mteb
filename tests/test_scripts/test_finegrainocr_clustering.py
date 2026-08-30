from __future__ import annotations

from dataclasses import replace

from scripts.data.finegrainocr_clustering.analyze_archive import ZipEntry
from scripts.data.finegrainocr_clustering.create_data import (
    plan_image_chunks,
    redact_barcodes,
)


def _entry(name: str, offset: int) -> ZipEntry:
    return ZipEntry(
        filename=name,
        compressed_size=50,
        uncompressed_size=100,
        local_header_offset=offset,
        compression_method=8,
        crc32=0,
    )


def test_redact_barcodes() -> None:
    assert redact_barcodes("UPC 123456789012 and lot 12345") == (
        "UPC [BARCODE] and lot 12345"
    )
    assert redact_barcodes("EAN 1 234-567 89012") == "EAN [BARCODE]"
    assert redact_barcodes("year 2026") == "year 2026"


def test_plan_image_chunks_merges_small_gaps() -> None:
    first = _entry("first.jpg", 100)
    filler = _entry("filler.jpg", 200)
    second = _entry("second.jpg", 300)
    final = _entry("final.txt", 400)
    all_entries = [first, filler, second, final]

    chunks = plan_image_chunks(
        [first, second],
        all_entries,
        max_gap_bytes=100,
        max_chunk_bytes=1_000,
    )

    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (100, 400)
    assert chunks[0].entries == (first, second)


def test_plan_image_chunks_respects_maximum_size() -> None:
    first = _entry("first.jpg", 100)
    second = _entry("second.jpg", 300)
    final = replace(second, filename="final.txt", local_header_offset=400)

    chunks = plan_image_chunks(
        [first, second],
        [first, second, final],
        max_gap_bytes=1_000,
        max_chunk_bytes=250,
    )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (100, 300),
        (300, 400),
    ]
