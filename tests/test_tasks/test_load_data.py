from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

import mteb


@pytest.mark.parametrize(
    "task",
    [
        mteb.get_task("DiaBlaBitextMining", hf_subsets=["fr-en"]),
        mteb.get_task("AmazonCounterfactualClassification", hf_subsets=["en"]),
        mteb.get_task("WikiClusteringP2P", hf_subsets=["bs"]),
        mteb.get_task("MultiEURLEXMultilabelClassification", hf_subsets=["en"]),
        mteb.get_task("OpusparcusPC", hf_subsets=["en"]),
        mteb.get_task("STS17MultilingualVisualSTS", hf_subsets=["en-en"]),
    ],
)
def test_multilingual_load_data(task):
    dummy_dataset = DatasetDict({"test": Dataset.from_dict({"text": ["test"]})})

    with patch("mteb.abstasks.abstask.load_dataset") as mock_load:
        mock_load.return_value = dummy_dataset
        task.load_data()

    assert mock_load.called
    assert task.dataset is not None
    assert len(task.dataset) == 1


@pytest.mark.parametrize(
    "task",
    [
        mteb.get_task("MIRACLRetrievalHardNegatives", languages=["eng"]),
    ],
)
def test_multilingual_retrieval_load_data(task):
    dummy_split = {
        "corpus": Dataset.from_dict({"id": ["d1"], "text": ["doc"]}),
        "queries": Dataset.from_dict({"id": ["q1"], "text": ["query"]}),
        "relevant_docs": {"q1": {"d1": 1}},
        "top_ranked": None,
    }

    with patch("mteb.abstasks.retrieval.RetrievalDatasetLoader.load") as mock_load:
        mock_load.return_value = dummy_split
        task.load_data()

    assert mock_load.called
    assert task.dataset is not None
    assert len(task.dataset) == 1
