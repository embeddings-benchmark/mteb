import json
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from datasets import Audio, Dataset

import mteb


def wav(sample: int) -> dict[str, bytes | None]:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setparams((1, 2, 24000, 1, "NONE", "not compressed"))
        recording.writeframes(sample.to_bytes(2, "little", signed=True))
    return {"path": None, "bytes": buffer.getvalue()}


@pytest.fixture
def expresso(tmp_path, monkeypatch):
    task = mteb.get_task("INSPIREExpressoAT2ARetrieval")
    source = {
        "document": Dataset.from_dict(
            {
                "audio_id": list("abcde"),
                "audio": [wav(i) for i in range(5)],
                "speaker": ["actor"] * 5,
            }
        ),
        "query": Dataset.from_dict(
            {
                "audio_id": ["ref", "ref"],
                "audio": [wav(12), wav(12)],
                "text": ["transcript", "transcript"],
                "instruction": ["same style", "same speaker whispering"],
                "positive_documents": [["c", "d"], ["a", "b"]],
                "excluded_ids": [["a", "b"], ["c", "d"]],
            }
        ),
    }
    files = {}
    for kind, data in source.items():
        files[f"{kind}/Expresso-00000-of-00001.parquet"] = str(
            tmp_path / f"{kind}.parquet"
        )
        data.cast_column("audio", Audio(decode=False)).to_parquet(
            tmp_path / f"{kind}.parquet"
        )

    def download(*, repo_id, revision, repo_type, filename):
        assert repo_id == "lca0503/INSPIRE" and repo_type == "dataset"
        assert revision == "4202b965891ca79897ce8cb639e08c591e993c06"
        return files[filename]

    monkeypatch.setattr(
        "mteb.tasks.retrieval.eng.inspire_expresso_retrieval.hf_hub_download", download
    )
    task.load_data()
    task.k_values = (1, 2)
    task.metadata = task.metadata.model_copy(update={"main_score": "ndcg_at_2"})
    task._top_k = 2
    return task


class FullGallerySearch:
    mteb_model_meta = mteb.get_model_meta("mteb/baseline-random-encoder")

    def index(self, corpus, **kwargs: Any):
        assert set(corpus.column_names) == {"id", "audio"}
        assert list(corpus["id"]) == list("abcde")

    def search(self, queries, *, top_ranked, top_k, **kwargs: Any):
        assert top_ranked is None
        assert top_k == 5
        assert {"excluded_ids", "relevant_docs", "candidates"}.isdisjoint(kwargs)
        assert set(queries.column_names) == {"id", "audio", "text"}
        assert list(queries.cast_column("audio", Audio(decode=False))["audio"]) == [
            wav(12),
            wav(12),
        ]
        assert list(queries["text"]) == ["same style", "same speaker whispering"]
        return {qid: dict(e=0.0, d=1.0, c=2.0, b=3.0, a=4.0) for qid in queries["id"]}


def test_official_exclusions_are_private_and_precede_top_k(expresso, tmp_path: Path):
    split = expresso.dataset["default"]["test"]
    assert split["relevant_docs"] == {
        "q00000": {"c": 1, "d": 1},
        "q00001": {"a": 1, "b": 1},
    }
    assert split["top_ranked"] == {"q00000": ["c", "d", "e"], "q00001": ["a", "b", "e"]}
    scores = expresso.evaluate(
        FullGallerySearch(), split="test", encode_kwargs={}, prediction_folder=tmp_path
    )
    assert scores["default"]["ndcg_at_2"] == 1.0
    predictions = json.loads(expresso._predictions_path(tmp_path).read_text())
    assert predictions["default"]["test"] == {
        "q00000": {"c": 2.0, "d": 1.0},
        "q00001": {"a": 4.0, "b": 3.0},
    }


@pytest.mark.parametrize(
    "invalid_scores",
    [
        {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0},
        {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0, "unknown": 0.0},
        {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0, "e": float("nan")},
    ],
)
def test_rejects_incomplete_gallery_scores(expresso, invalid_scores):
    class PartialSearch(FullGallerySearch):
        def search(self, queries, **kwargs: Any):
            result = super().search(queries, **kwargs)
            result["q00000"] = invalid_scores
            return result

    with pytest.raises(ValueError, match="every corpus document"):
        expresso.evaluate(PartialSearch(), split="test", encode_kwargs={})


def test_score_ties_at_cutoff_follow_trec_eval(expresso, tmp_path):
    class TiedSearch(FullGallerySearch):
        def search(self, queries, **kwargs: Any):
            return {qid: dict.fromkeys("abcde", 1.0) for qid in queries["id"]}

    expresso.evaluate(TiedSearch(), encode_kwargs={}, prediction_folder=tmp_path)
    predictions = json.loads(expresso._predictions_path(tmp_path).read_text())
    rankings = predictions["default"]["test"]
    assert list(rankings["q00000"]) == ["e", "d"]
    assert list(rankings["q00001"]) == ["e", "b"]


def test_external_shortlist_cannot_replace_official_candidates(expresso, tmp_path):
    with pytest.raises(ValueError, match="full corpus"):
        expresso.convert_to_reranking(tmp_path)
