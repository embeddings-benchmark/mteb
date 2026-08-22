from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from mteb.tasks.retrieval.eng.mm_bright_retrieval import (
    _document_candidates,
    _filter_evaluable_image_queries,
    _pair_id,
    _pair_reranking_data,
)


def _examples() -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": "query",
                "query": "Why?",
                "gold_ids": ["deadbeef_0001.txt"],
                "negative_ids": ["cafebabe_0001.txt", "deadbeef_0001.txt"],
                "positive_images": [
                    {
                        "image_path": "positive.png",
                        "source_passage_id": "deadbeef_0000.txt",
                    }
                ],
                "negative_images": [
                    {
                        "image_path": "negative.png",
                        "source_passage_id": "deadbeef_0000.txt",
                    },
                    {
                        "image_path": "negative-passage.png",
                        "source_passage_id": "cafebabe_0000.txt",
                    },
                ],
            }
        ]
    )


def test_document_candidates_include_hard_negatives_once() -> None:
    assert _document_candidates(_examples()) == {
        "query": ["deadbeef_0001.txt", "cafebabe_0001.txt"]
    }


def test_pair_reranking_uses_none_for_text_only_candidates() -> None:
    documents = Dataset.from_dict(
        {
            "id": ["deadbeef_0001.txt", "cafebabe_0001.txt"],
            "text": ["relevant", "hard negative"],
        }
    )
    images = Dataset.from_dict(
        {
            "id": ["positive.png", "negative.png", "negative-passage.png"],
            "image": [
                PILImage.new("RGB", (8, 8), "green"),
                PILImage.new("RGB", (8, 8), "red"),
                PILImage.new("RGB", (8, 8), "blue"),
            ],
        },
        features=Features({"id": Value("string"), "image": Image(mode="RGB")}),
    )

    corpus, qrels, top_ranked = _pair_reranking_data(documents, images, _examples())

    gold_text = "deadbeef_0001.txt"
    negative_text = "cafebabe_0001.txt"
    gold_pair = _pair_id("deadbeef_0001.txt", "positive.png")
    negative_pair = _pair_id("deadbeef_0001.txt", "negative.png")
    negative_passage_pair = _pair_id("cafebabe_0001.txt", "negative-passage.png")
    rows = {row["id"]: row for row in corpus}

    assert top_ranked == {
        "query": [
            gold_text,
            negative_text,
            gold_pair,
            negative_pair,
            negative_passage_pair,
        ]
    }
    assert qrels == {"query": {gold_text: 1, gold_pair: 2}}
    assert rows[gold_text]["image"] is None
    assert rows[negative_text]["image"] is None
    assert rows[gold_pair]["image"] is not None
    assert rows[negative_pair]["image"] is not None
    assert rows[negative_passage_pair]["image"] is not None


def test_image_reranking_drops_queries_without_positive_candidates() -> None:
    queries = Dataset.from_dict(
        {"id": ["evaluable", "negative-only"], "text": ["a", "b"]}
    )

    filtered_queries, top_ranked, qrels = _filter_evaluable_image_queries(
        queries,
        {
            "evaluable": ["positive", "negative"],
            "negative-only": ["negative"],
        },
        {"evaluable": {"positive": 1}, "negative-only": {}},
        domain="test",
    )

    assert filtered_queries["id"] == ["evaluable"]
    assert top_ranked == {"evaluable": ["positive", "negative"]}
    assert qrels == {"evaluable": {"positive": 1}}
