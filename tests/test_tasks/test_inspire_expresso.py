import json
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
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
                "audio": [dict(wav(i), path=f"source-{i}.wav") for i in range(5)],
                "speaker": ["actor"] * 5,
            }
        ),
        "query": Dataset.from_dict(
            {
                "audio_id": ["ref", "ref"],
                "audio": [dict(wav(12), path="ref.wav")] * 2,
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


def document_ids(corpus):
    by_waveform = {wav(i)["bytes"]: letter for i, letter in enumerate("abcde")}
    return {
        by_waveform[row["audio"]["bytes"]]: row["id"]
        for row in corpus.cast_column("audio", Audio(decode=False))
    }


class FullGallerySearch:
    mteb_model_meta = mteb.get_model_meta("mteb/baseline-random-encoder")

    def index(self, corpus, **kwargs: Any):
        assert set(corpus.column_names) == {"id", "audio"}
        self.ids = document_ids(corpus)

    def search(self, queries, *, top_ranked, top_k, **kwargs: Any):
        assert top_ranked is None
        assert top_k == 5
        assert {"excluded_ids", "relevant_docs", "candidates"}.isdisjoint(kwargs)
        assert set(queries.column_names) == {"id", "audio", "text"}
        assert list(queries.cast_column("audio", Audio(decode=False))["audio"]) == [
            wav(12),
            wav(12),
        ]
        assert set(queries["text"]) == {"same style", "same speaker whispering"}
        return {
            qid: {self.ids[doc]: float(i) for i, doc in enumerate("edcba")}
            for qid in queries["id"]
        }


def test_model_inputs_hide_release_ids_file_paths_and_row_order(expresso, tmp_path):
    split = expresso.dataset["default"]["test"]
    corpus = split["corpus"].cast_column("audio", Audio(decode=False))
    queries = split["queries"].cast_column("audio", Audio(decode=False))
    assert not set(corpus["id"]).intersection("abcde")
    assert not set(queries["id"]).intersection(("q00000", "q00001", "ref"))
    assert all(item["path"] is None for item in [*corpus["audio"], *queries["audio"]])
    assert [item["bytes"] for item in corpus["audio"]] != [
        wav(i)["bytes"] for i in range(5)
    ]
    assert list(corpus["id"]) == sorted(corpus["id"])
    assert list(queries["id"]) == sorted(queries["id"])
    for kind in ("document", "query"):
        path = tmp_path / f"{kind}.parquet"
        table = pq.read_table(path)
        pq.write_table(table.take(list(reversed(range(len(table))))), path)
    reordered = mteb.get_task(expresso.metadata.name)
    reordered.load_data()
    other = reordered.dataset["default"]["test"]
    for side in ("corpus", "queries"):
        assert split[side].data.table.equals(other[side].data.table)
    assert split["relevant_docs"] == other["relevant_docs"]
    assert split["top_ranked"] == other["top_ranked"]


def test_official_exclusions_are_private_and_precede_top_k(expresso, tmp_path: Path):
    split = expresso.dataset["default"]["test"]
    docs = document_ids(split["corpus"])
    queries = split["queries"]
    qids = dict(zip(queries["text"], queries["id"], strict=True))
    style, whisper = qids["same style"], qids["same speaker whispering"]
    assert split["relevant_docs"] == {
        style: {docs["c"]: 1, docs["d"]: 1},
        whisper: {docs["a"]: 1, docs["b"]: 1},
    }
    assert split["top_ranked"] == {
        style: sorted(docs[letter] for letter in "cde"),
        whisper: sorted(docs[letter] for letter in "abe"),
    }
    scores = expresso.evaluate(
        FullGallerySearch(), split="test", encode_kwargs={}, prediction_folder=tmp_path
    )
    assert scores["default"]["ndcg_at_2"] == 1.0
    predictions = json.loads(expresso._predictions_path(tmp_path).read_text())
    assert predictions["default"]["test"] == {
        style: {docs["c"]: 2.0, docs["d"]: 1.0},
        whisper: {docs["a"]: 4.0, docs["b"]: 3.0},
    }


@pytest.mark.parametrize("error", ["missing", "unknown", "nonfinite"])
def test_rejects_incomplete_gallery_scores(expresso, error):
    class PartialSearch(FullGallerySearch):
        def search(self, queries, **kwargs: Any):
            result = super().search(queries, **kwargs)
            scores = next(iter(result.values()))
            if error == "nonfinite":
                scores[self.ids["e"]] = float("nan")
            else:
                value = scores.pop(self.ids["e"])
                if error == "unknown":
                    scores["unknown"] = value
            return result

    with pytest.raises(ValueError, match="every corpus document"):
        expresso.evaluate(PartialSearch(), split="test", encode_kwargs={})


@pytest.mark.parametrize(
    "use_position", [False, True], ids=["constant", "row-position"]
)
def test_input_free_rankings_follow_content_id_order(expresso, tmp_path, use_position):
    class NoInputSearch:
        mteb_model_meta = FullGallerySearch.mteb_model_meta

        def index(self, corpus, **kwargs: Any):
            self.scores = {
                doc: float(index if use_position else 1)
                for index, doc in enumerate(corpus["id"])
            }

        def search(self, queries, *, top_k, top_ranked, **kwargs: Any):
            assert top_ranked is None and top_k == len(self.scores)
            return dict.fromkeys(queries["id"], self.scores)

    expresso.evaluate(NoInputSearch(), encode_kwargs={}, prediction_folder=tmp_path)
    predictions = json.loads(expresso._predictions_path(tmp_path).read_text())
    rankings = predictions["default"]["test"]
    candidates = expresso.dataset["default"]["test"]["top_ranked"]
    for qid, eligible in candidates.items():
        assert list(rankings[qid]) == sorted(eligible, reverse=True)[:2]


def test_external_shortlist_cannot_replace_official_candidates(expresso, tmp_path):
    with pytest.raises(ValueError, match="full corpus"):
        expresso.convert_to_reranking(tmp_path)
