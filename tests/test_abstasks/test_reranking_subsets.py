"""Upload/load and evaluation contracts for shared retriever candidate subsets."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from datasets import Dataset, DatasetDict, Features, Image, Value
from pydantic import ValidationError

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks.create_mock_samples import create_mock_images
from mteb.mocks.mock_tasks.retrieval import base_retrieval_datasplit
from mteb.results import TaskResult


class RetrieverSubsetsTask(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RetrieverSubsetsTest",
        description="Shared data with independent retriever candidates",
        dataset={"path": "test/reranking", "revision": "frozen-revision"},
        type="Reranking",
        category="t2t",
        eval_langs={name: ["eng-Latn"] for name in ("bm25", "qwen", "bge")},
        reranking_subsets=dict.fromkeys(("bm25", "qwen", "bge"), "default"),
        main_score="ndcg_at_10",
    )


def populated_task():
    task = RetrieverSubsetsTask()
    shared = base_retrieval_datasplit()
    candidates = {
        "bm25": {"q1": ["d1"], "q2": ["d1"]},
        "qwen": {"q1": ["d1"], "q2": ["d2"]},
        "bge": {"q1": ["d2"], "q2": ["d1"]},
    }
    task.dataset = {
        subset: {"test": {**shared, "top_ranked": ranked}}
        for subset, ranked in candidates.items()
    }
    task.data_loaded = True
    return task


@pytest.fixture
def hub(monkeypatch):
    """Keep uploads in memory and stub Hub dataset discovery and loading."""
    configs = {}
    reads = Counter()
    writes = []

    def push(data, repo_id, config_name, **kwargs: Any):
        assert repo_id == "test/reranking"
        assert kwargs["private"] is True
        writes.append(config_name)
        configs[config_name] = data

    def load(repo_id, config_name, *, split, revision, **kwargs: Any):
        assert repo_id == "test/reranking"
        assert revision == "frozen-revision"
        reads[(config_name, split)] += 1
        return configs[config_name][split]

    monkeypatch.setattr(DatasetDict, "push_to_hub", push)
    monkeypatch.setattr(TaskMetadata, "push_dataset_card_to_hub", lambda *a: None)
    module = "mteb.abstasks.retrieval_dataset_loaders"
    monkeypatch.setattr(f"{module}.get_dataset_config_names", lambda *a: list(configs))
    monkeypatch.setattr(
        f"{module}.get_dataset_split_names",
        lambda *a, config_name, **kw: list(configs[config_name]),
    )
    monkeypatch.setattr(f"{module}.load_dataset", load)
    return configs, reads, writes


@pytest.mark.parametrize("data_subset", ["default", "english"])
def test_parquet_round_trip(monkeypatch, tmp_path: Path, data_subset: str):
    """Load the uploaded layout through real Hugging Face config discovery and Parquet."""
    configs = []

    def push(data, repo_id, config_name, **kwargs: Any):
        assert repo_id == "test/reranking"
        directory = tmp_path / config_name
        directory.mkdir()
        data_files = []
        for split, dataset in data.items():
            path = directory / f"{split}.parquet"
            dataset.to_parquet(path)
            data_files.append(
                {"split": split, "path": path.relative_to(tmp_path).as_posix()}
            )
        configs.append({"config_name": config_name, "data_files": data_files})
        (tmp_path / "README.md").write_text(
            "---\n" + yaml.safe_dump({"configs": configs}) + "---\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(DatasetDict, "push_to_hub", push)
    monkeypatch.setattr(TaskMetadata, "push_dataset_card_to_hub", lambda *a: None)
    original = populated_task()
    original.metadata = TaskMetadata.model_validate(
        {
            **original.metadata.model_dump(),
            "reranking_subsets": dict.fromkeys(original.hf_subsets, data_subset),
        }
    )
    original.dataset["bm25"]["test"]["top_ranked"]["q1"] = ["d2", "d1"]
    original.push_dataset_to_hub("test/reranking", private=True)

    restored = RetrieverSubsetsTask()
    restored.metadata = original.metadata.model_copy(
        update={"dataset": {"path": str(tmp_path), "revision": "local"}}
    )
    restored.load_data()
    prefix = "" if data_subset == "default" else f"{data_subset}-"
    assert {config["config_name"] for config in configs} == {
        f"{prefix}corpus",
        f"{prefix}queries",
        f"{prefix}qrels",
        "bm25-top_ranked",
        "qwen-top_ranked",
        "bge-top_ranked",
    }
    for subset in original.hf_subsets:
        expected = original.dataset[subset]["test"]
        actual = restored.dataset[subset]["test"]
        assert actual["top_ranked"] == expected["top_ranked"]
        assert actual["relevant_docs"] == expected["relevant_docs"]
        for section in ("corpus", "queries"):
            assert actual[section].to_list() == expected[section].to_list()
            assert actual[section] is restored.dataset["bm25"]["test"][section]


def test_upload_load_shared_data_and_candidate_order(hub):
    configs, reads, writes = hub
    original = populated_task()
    original.dataset["bm25"]["test"]["top_ranked"]["q1"] = ["d2", "d1"]
    original.push_dataset_to_hub("test/reranking", private=True)
    assert set(configs) == {
        "corpus",
        "queries",
        "qrels",
        "bm25-top_ranked",
        "qwen-top_ranked",
        "bge-top_ranked",
    }
    assert len(writes) == len(configs)

    task = RetrieverSubsetsTask()
    task.load_data()
    task.load_data()  # Repeated loads are a no-op.
    assert set(task.dataset) == {"bm25", "qwen", "bge"}
    for subset in task.dataset:
        data = task.dataset[subset]["test"]
        assert data["top_ranked"] == original.dataset[subset]["test"]["top_ranked"]
        assert (
            data["relevant_docs"] == original.dataset[subset]["test"]["relevant_docs"]
        )
        for section in ("corpus", "queries", "relevant_docs"):
            assert data[section] is task.dataset["bm25"]["test"][section]
    assert all(count == 1 for count in reads.values())


def test_subset_selection_loads_only_requested_candidates(hub):
    _, reads, _ = hub
    populated_task().push_dataset_to_hub("test/reranking", private=True)
    task = RetrieverSubsetsTask()
    task.filter_languages(languages=None, hf_subsets=["qwen"])
    task.load_data()
    assert set(task.dataset) == {"qwen"}
    assert set(reads) == {
        ("corpus", "test"),
        ("queries", "test"),
        ("qrels", "test"),
        ("qwen-top_ranked", "test"),
    }


def test_image_and_text_corpus_is_shared(hub):
    pytest.importorskip("PIL")
    _, reads, writes = hub
    original = populated_task()
    corpus = Dataset.from_dict(
        {
            "id": ["d1", "d2"],
            "text": ["page one", "page two"],
            "image": create_mock_images(original.np_rng),
        },
        features=Features(
            {"id": Value("string"), "text": Value("string"), "image": Image()}
        ),
    )
    for subset in original.dataset.values():
        subset["test"]["corpus"] = corpus
    original.push_dataset_to_hub("test/reranking", private=True)
    task = RetrieverSubsetsTask()
    task.metadata = task.metadata.model_copy(
        update={"category": "t2it", "modalities": ["text", "image"]}
    )
    task.load_data()
    shared_corpus = task.dataset["bm25"]["test"]["corpus"]
    assert shared_corpus.features == corpus.features
    assert shared_corpus[0]["image"].size == (100, 100)
    assert task.dataset["qwen"]["test"]["corpus"] is shared_corpus
    assert writes.count("corpus") == 1
    assert reads[("corpus", "test")] == 1


def test_separate_language_data_and_splits_round_trip(hub):
    configs, reads, _ = hub
    task = populated_task()
    base = task.dataset["bm25"]["test"]
    task.metadata = TaskMetadata.model_validate(
        {
            **task.metadata.model_dump(),
            "eval_splits": ["test", "validation"],
            "eval_langs": {
                f"{retriever}-{lang}": [code]
                for retriever in ("bm25", "qwen")
                for lang, code in (("english", "eng-Latn"), ("french", "fra-Latn"))
            },
            "reranking_subsets": {
                f"{retriever}-{lang}": lang
                for retriever in ("bm25", "qwen")
                for lang in ("english", "french")
            },
        }
    )
    task.hf_subsets = task.metadata.hf_subsets
    task.dataset = {
        subset: {split: {**base} for split in task.eval_splits}
        for subset in task.hf_subsets
    }
    task.push_dataset_to_hub("test/reranking", private=True)
    restored = RetrieverSubsetsTask()
    restored.metadata = task.metadata
    restored.hf_subsets = task.hf_subsets
    restored.load_data()
    assert set(restored.dataset) == set(task.dataset)
    assert set(restored.dataset["qwen-french"]) == {"test", "validation"}
    assert "french-corpus" in configs and "english-corpus" in configs
    assert "bm25-french-corpus" not in configs
    assert all(count == 1 for count in reads.values())


@pytest.mark.parametrize("section", ["corpus", "queries", "relevant_docs", "splits"])
def test_conflicting_shared_data_rejected_before_upload(hub, section):
    _, _, writes = hub
    task = populated_task()
    data = task.dataset["qwen"]["test"]
    if section in {"corpus", "queries"}:
        data[section] = data[section].add_column(
            "extra", ["different"] * len(data[section])
        )
    elif section == "relevant_docs":
        data[section] = {"q1": {"d2": 1}, "q2": {"d1": 1}}
    else:
        task.dataset["qwen"]["validation"] = data.copy()
    with pytest.raises(ValueError, match="Shared"):
        task.push_dataset_to_hub("test/reranking", private=True)
    assert writes == []


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (None, "require top_ranked"),
        ({"q1": ["d1"]}, "exactly the evaluated query IDs"),
        ({"q1": ["d1"], "q2": ["d2"], "q3": ["d1"]}, "exactly the evaluated query IDs"),
        ({"q1": [], "q2": ["d2"]}, "Empty reranking"),
        ({"q1": ["d1", "d1"], "q2": ["d2"]}, "Duplicate reranking"),
        ({"q1": ["missing"], "q2": ["d2"]}, "Unknown corpus IDs"),
    ],
)
def test_invalid_candidates_rejected_before_upload(hub, candidates, message):
    _, _, writes = hub
    task = populated_task()
    task.dataset["qwen"]["test"]["top_ranked"] = candidates
    with pytest.raises(ValueError, match=message):
        task.push_dataset_to_hub("test/reranking", private=True)
    assert writes == []


@pytest.mark.parametrize(
    "corruption", ["missing-config", "duplicate-query", "missing-query", "unknown-doc"]
)
def test_invalid_uploaded_candidates_rejected_on_load(hub, corruption):
    configs, _, _ = hub
    populated_task().push_dataset_to_hub("test/reranking", private=True)
    if corruption == "missing-config":
        del configs["qwen-top_ranked"]
        message = "Missing reranking candidate configuration"
    else:
        rows = configs["qwen-top_ranked"]["test"].to_list()
        if corruption == "duplicate-query":
            rows.append(rows[0])
            message = "Duplicate query IDs"
        elif corruption == "missing-query":
            rows.pop()
            message = "exactly the evaluated query IDs"
        else:
            rows[0]["corpus-ids"] = ["missing"]
            message = "Unknown corpus IDs"
        configs["qwen-top_ranked"]["test"] = Dataset.from_list(rows)
    with pytest.raises(ValueError, match=message):
        RetrieverSubsetsTask().load_data()


@pytest.mark.parametrize(
    "mapping",
    [{}, {"bm25": "default"}, {"bm25": "default", "qwen": "default", "bge": ""}],
)
def test_invalid_subset_mapping(mapping):
    with pytest.raises(ValidationError):
        TaskMetadata.model_validate(
            {**RetrieverSubsetsTask.metadata.model_dump(), "reranking_subsets": mapping}
        )


def test_existing_layout_still_round_trips(hub):
    configs, _, _ = hub
    original = populated_task()
    original.metadata = original.metadata.model_copy(update={"reranking_subsets": None})
    original.push_dataset_to_hub("test/reranking", private=True)
    assert "bm25-corpus" in configs and "qwen-corpus" in configs
    task = RetrieverSubsetsTask()
    task.metadata = original.metadata
    task.load_data()
    assert task.dataset["bge"]["test"]["top_ranked"] == {"q1": ["d2"], "q2": ["d1"]}


def test_evaluation_keeps_retrievers_and_unretrieved_positives(hub, tmp_path: Path):
    populated_task().push_dataset_to_hub("test/reranking", private=True)
    result = mteb.evaluate(
        mteb.get_model("mteb/baseline-random-cross-encoder"),
        RetrieverSubsetsTask(),
        cache=mteb.ResultCache(tmp_path / "cache"),
        overwrite_strategy="always",
    )[0]
    # One candidate per query makes scores independent of the random cross encoder.
    # BM25 misses q2's relevant document; BGE misses both. Neither query is dropped.
    assert result.get_score(subsets=["bm25"]) == 0.5
    assert result.get_score(subsets=["qwen"]) == 1.0
    assert result.get_score(subsets=["bge"]) == 0.0
    assert result.get_score() == 0.5
    result.to_disk(tmp_path / "result.json")
    restored = TaskResult.from_disk(tmp_path / "result.json")
    assert restored.get_score(subsets=["bm25"]) == 0.5
    assert set(restored.hf_subsets) == {"bm25", "qwen", "bge"}
