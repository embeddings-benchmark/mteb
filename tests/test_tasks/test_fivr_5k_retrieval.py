from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datasets import Dataset, Video

from mteb.tasks.retrieval.zxx import fivr_5k_retrieval as fivr


def test_fivr_regimes_use_full_ranking_map() -> None:
    tasks = (
        fivr.FIVR5KDSVRRetrieval,
        fivr.FIVR5KCSVRRetrieval,
        fivr.FIVR5KISVRRetrieval,
    )
    assert [task._regime for task in tasks] == ["dsvr", "csvr", "isvr"]
    for task in tasks:
        assert task.metadata.main_score == f"map_at_{fivr._CORPUS_SIZE}"
        assert task._top_k == fivr._CORPUS_SIZE
        assert fivr._CORPUS_SIZE in task.k_values


def test_load_fivr_uses_frozen_media_without_downloading(
    tmp_path: Path, monkeypatch: Any
) -> None:
    corpus = Dataset.from_list([{"id": "c1"}, {"id": "c2"}])
    queries = Dataset.from_list([{"id": "q1"}])
    qrels = Dataset.from_list([{"query-id": "q1", "corpus-id": "c2", "score": 1}])
    configs = {"corpus": corpus, "queries": queries, "dsvr-qrels": qrels}

    def fake_load_dataset(*, name: str, **kwargs: Any) -> Dataset:
        return configs[name]

    for video_id in ("c1", "c2", "q1"):
        (tmp_path / f"{video_id}.mp4").write_bytes(b"frozen-media")

    monkeypatch.setattr(fivr, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(fivr, "_CORPUS_SIZE", 2)
    monkeypatch.setattr(fivr, "_QUERY_COUNT", 1)
    task = SimpleNamespace(
        data_loaded=False,
        dataset=None,
        metadata=SimpleNamespace(dataset={"path": "test/fivr", "revision": "abc"}),
    )
    fivr._load_fivr(
        task,
        "dsvr",
        num_proc=None,
        metadata_dir=None,
        video_dir=tmp_path,
        download_workers=1,
    )

    split = task.dataset["default"]["test"]
    assert split["relevant_docs"] == {"q1": {"c2": 1}}
    assert split["corpus"]["id"] == ["c1", "c2"]
    assert split["queries"]["id"] == ["q1"]
    assert isinstance(split["corpus"].features["video"], Video)
