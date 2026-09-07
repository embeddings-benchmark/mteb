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


@pytest.mark.parametrize(
    ("language", "query", "positive_passages", "negative_passage"),
    [
        (
            "fr",
            "Pourquoi l’élève étudie-t-il à l’université de Montréal ?",
            [
                ("positive-1", "L’étudiante prépare un mémoire à Montréal."),
                ("positive-2", "Elle réussit grâce à sa persévérance."),
            ],
            ("negative-1", "Un passage sans rapport évoque l’été."),
        ),
        (
            "es",
            "¿Cómo cambió la situación después de la elección?",
            [
                ("positive-1", "La elección cambió la política económica."),
                ("positive-2", "También mejoró la cooperación pública."),
            ],
            ("negative-1", "Un texto sobre música clásica."),
        ),
        (
            "pt",
            "Qual é a relação entre educação e inovação?",
            [
                ("positive-1", "A educação pública estimula a inovação."),
                ("positive-2", "A pesquisa também fortalece a ciência."),
            ],
            ("negative-1", "Um texto sobre culinária açoriana."),
        ),
    ],
    ids=["french", "spanish", "portuguese"],
)
def test_mldr_loads_requested_split_and_all_qrels(
    language, query, positive_passages, negative_passage
):
    task = mteb.get_task(
        "MultiLongDocRetrieval",
        eval_splits=["test"],
        hf_subsets=[language],
    )
    configs = [f"{language}-{kind}" for kind in ("corpus", "qrels", "queries")]
    load_calls = []

    def mock_load_dataset(path, config, *, split, **kwargs: object):
        assert path == task.metadata.dataset["path"]
        assert kwargs["revision"] == task.metadata.dataset["revision"]
        load_calls.append((config, split))

        if config.endswith("-corpus"):
            passages = [*positive_passages, negative_passage]
            return Dataset.from_dict(
                {
                    "id": [passage_id for passage_id, _ in passages],
                    "text": [text for _, text in passages],
                }
            )
        if config.endswith("-queries"):
            return Dataset.from_dict({"id": ["query-1"], "text": [query]})
        return Dataset.from_dict(
            {
                "query-id": ["query-1"] * len(positive_passages),
                "corpus-id": [passage_id for passage_id, _ in positive_passages],
                "score": [1] * len(positive_passages),
            }
        )

    with (
        patch(
            "mteb.abstasks.retrieval_dataset_loaders.get_dataset_config_names",
            return_value=configs,
        ),
        patch(
            "mteb.abstasks.retrieval_dataset_loaders.get_dataset_split_names",
            return_value=["dev", "test"],
        ),
        patch(
            "mteb.abstasks.retrieval_dataset_loaders.load_dataset",
            side_effect=mock_load_dataset,
        ),
    ):
        task.load_data()

    assert task.dataset is not None
    assert set(task.dataset[language]) == {"test"}

    split_data = task.dataset[language]["test"]
    assert split_data["queries"]["text"] == [query]
    assert split_data["corpus"]["text"] == [
        text for _, text in [*positive_passages, negative_passage]
    ]
    assert split_data["relevant_docs"] == {
        "query-1": {"positive-1": 1, "positive-2": 1}
    }
    assert set(split_data["corpus"]["id"]) == {
        "positive-1",
        "positive-2",
        "negative-1",
    }
    assert load_calls == [
        (f"{language}-qrels", "test"),
        (f"{language}-corpus", "test"),
        (f"{language}-queries", "test"),
    ]
