from datasets import Dataset

import mteb
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models import SearchCrossEncoderWrapper


def test_cross_encoder_search_ignores_top_ranked_without_query():
    """top_ranked can list query ids that are not in the queries (e.g. dropped for having no positives)."""
    task = MockRetrievalTask()
    model = SearchCrossEncoderWrapper(
        mteb.get_model("mteb/baseline-random-cross-encoder")
    )
    model.index(
        corpus=Dataset.from_list(
            [
                {"id": "doc1", "text": "document one"},
                {"id": "doc2", "text": "document two"},
            ]
        ),
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        encode_kwargs={},
    )

    res = model.search(
        queries=Dataset.from_list([{"id": "q1", "text": "query"}]),
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        encode_kwargs={},
        top_ranked={"q1": ["doc1", "doc2"], "q2": ["doc2"]},
    )
    assert list(res.keys()) == ["q1"]
    assert set(res["q1"]) == {"doc1", "doc2"}
