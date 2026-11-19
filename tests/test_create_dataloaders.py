from datasets import Dataset

from mteb._create_dataloaders import _corpus_to_dict, create_dataloader
from mteb.mocks import MockRetrievalTask
from mteb.types import PromptType


def test_corpus_with_nullable_titles() -> None:
    corpus = Dataset.from_dict(
        {
            "id": ["d1", "d2"],
            "title": [None, "Heading"],
            "text": ["body", "body2"],
        }
    )

    mapped = corpus.map(_corpus_to_dict)
    assert mapped["text"] == ["body", "Heading body2"]
    assert mapped["title"] == ["", "Heading"]

    dataloader = create_dataloader(
        corpus,
        task_metadata=MockRetrievalTask.metadata,
        prompt_type=PromptType.document,
        batch_size=2,
    )
    batches = list(dataloader)
    assert len(batches) == 1
    assert batches[0]["text"] == ["body", "Heading body2"]
    assert batches[0]["title"] == ["", "Heading"]
