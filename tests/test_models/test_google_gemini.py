from __future__ import annotations

import pytest

from mteb.models.model_implementations.google_gemini import (
    _format_gemini_embedding_2_text,
)
from mteb.types import PromptType


@pytest.mark.parametrize(
    ("google_task_type", "expected"),
    [
        ("RETRIEVAL_QUERY", "task: search result | query: example text"),
        ("QUESTION_ANSWERING", "task: question answering | query: example text"),
        ("FACT_VERIFICATION", "task: fact checking | query: example text"),
        ("CLASSIFICATION", "task: classification | query: example text"),
        ("CLUSTERING", "task: clustering | query: example text"),
        ("SEMANTIC_SIMILARITY", "task: sentence similarity | query: example text"),
    ],
)
def test_gemini_embedding_2_formats_task_prefixes(
    google_task_type: str, expected: str
) -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", google_task_type, PromptType.query
        )
        == expected
    )


def test_gemini_embedding_2_formats_documents_with_title() -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", "FACT_VERIFICATION", PromptType.document, "Example"
        )
        == "title: Example | text: example text"
    )


def test_gemini_embedding_2_formats_documents_without_title() -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", "FACT_VERIFICATION", PromptType.document
        )
        == "title: none | text: example text"
    )


def test_gemini_embedding_2_leaves_unknown_task_type_unchanged() -> None:
    assert _format_gemini_embedding_2_text("example text", None, None) == "example text"
