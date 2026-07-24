from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from mteb.models.model_implementations.bge_structural_separator import (
    StructuralSeparatorSearch,
    _batch_size,
    _document_ids,
)


class _TokenizerStub:
    cls_token_id = 101
    sep_token_id = 102

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        self.encoded.append(text)
        return [len(self.encoded) + 10]


def _search_stub() -> StructuralSeparatorSearch:
    search = StructuralSeparatorSearch.__new__(StructuralSeparatorSearch)
    search.docids = ["d1", "d2", "d3"]
    search._docid_to_index = {"d1": 0, "d2": 1, "d3": 2}
    search.document_vectors = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
    )
    search.query_search_batch_size = 2
    search.document_search_block_size = 2
    search._index_path = None
    return search


def test_document_ids_preserve_title_body_boundary() -> None:
    tokenizer = _TokenizerStub()

    token_ids = _document_ids(
        tokenizer,
        {
            "title": "A title",
            "text": "A title Body sentence.",
            "body": "Body sentence.",
        },
        separator_token_id=3,
        max_length=512,
    )

    assert tokenizer.encoded == ["A title", "Body sentence."]
    assert token_ids == [101, 3, 11, 3, 12, 102]


@pytest.mark.parametrize(
    ("encode_kwargs", "expected"),
    [({}, 64), ({"batch_size": 7}, 7)],
)
def test_batch_size_comes_from_encode_kwargs(
    encode_kwargs: dict[str, Any], expected: int
) -> None:
    assert _batch_size(encode_kwargs) == expected


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _batch_size({"batch_size": 0})


def test_search_reranks_only_supplied_candidates() -> None:
    search = _search_stub()
    observed: dict[str, Any] = {}

    def encode_queries(queries: list[str], *, batch_size: int) -> np.ndarray:
        observed["queries"] = queries
        observed["batch_size"] = batch_size
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    search._encode_queries = encode_queries  # type: ignore[method-assign]
    queries = Dataset.from_list([{"id": "q1", "text": "query"}])

    result = search.search(
        queries,
        task_metadata=None,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
        top_k=2,
        encode_kwargs={"batch_size": 7},
        top_ranked={"q1": ["d2", "d3"]},
        num_proc=None,
    )

    assert observed == {"queries": ["query"], "batch_size": 7}
    assert list(result["q1"]) == ["d2", "d3"]
    assert "d1" not in result["q1"]


def test_index_uses_body_and_keeps_temp_file_open_for_writing() -> None:
    search = _search_stub()
    search.index_encoding_chunk_size = 2
    observed: list[tuple[list[dict[str, str]], int]] = []

    def encode_documents(
        documents: list[dict[str, str]], *, batch_size: int
    ) -> np.ndarray:
        observed.append(([document.copy() for document in documents], batch_size))
        return np.ones((len(documents), 2), dtype=np.float32)

    search._encode_documents = encode_documents  # type: ignore[method-assign]
    corpus = Dataset.from_list(
        [
            {
                "id": "d1",
                "title": "Title",
                "text": "Title Combined body",
                "body": "Combined body",
            },
            {"id": "d2", "title": "", "text": "Fallback body", "body": None},
        ]
    )

    search.index(
        corpus,
        task_metadata=None,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
        encode_kwargs={"batch_size": 7},
        num_proc=None,
    )

    try:
        assert observed == [
            (
                [
                    {"title": "Title", "body": "Combined body"},
                    {"title": "", "body": "Fallback body"},
                ],
                7,
            )
        ]
        assert isinstance(search.document_vectors, np.memmap)
        assert Path(search._index_path).exists()  # type: ignore[arg-type]
    finally:
        search.close()


def test_close_tolerates_numpy_interpreter_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = _search_stub()
    monkeypatch.setattr(np, "empty", None)

    search.close()
