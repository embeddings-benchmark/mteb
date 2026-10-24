from __future__ import annotations

from typing import Any, Literal

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "Cerru02/Mars-VL-Pairs-MTEB"
_DATASET_REVISION = "f0084ab0ba2f584b15dc72a82502b38ee490f58d"
_FROZEN_PAIRS = 2_247
_REFERENCE = "https://arxiv.org/abs/2602.13961"
_BIBTEX = r"""
@article{wang2026marsretrieval,
  author = {Wang, Shuoyuan and Wang, Yiran and Wei, Hongxin},
  journal = {arXiv preprint arXiv:2602.13961},
  title = {MarsRetrieval: Benchmarking Vision-Language Models for Planetary-Scale Geospatial Retrieval on Mars},
  year = {2026},
}
"""
_DESCRIPTION = (
    "Mars-VL-Pairs is Task 1 of MarsRetrieval, a planetary-science benchmark "
    "covering Mars imagery from global orbital mosaics to rover-scale views. "
    "The source has 2,287 one-to-one image-caption pairs; this reproducibility "
    "release freezes 2,247 pairs after 38 unavailable images and two "
    "resize-equivalent image pairs were removed. It uses the expert-validated "
    "refined captions from the paper's main evaluation. "
)


def _load_mars_vl_pairs(
    task: AbsTaskRetrieval,
    direction: Literal["t2i", "i2t"],
    num_proc: int | None,
) -> None:
    if task.data_loaded:
        return

    pairs = load_dataset(
        task.metadata.dataset["path"],
        revision=task.metadata.dataset["revision"],
        split="test",
        num_proc=num_proc,
    )
    if len(pairs) != _FROZEN_PAIRS:
        raise ValueError(f"Expected {_FROZEN_PAIRS} frozen pairs, found {len(pairs)}")

    ids = [str(key) for key in pairs["key"]]

    text = (
        pairs.select_columns(["refined_caption"])
        .rename_column("refined_caption", "text")
        .add_column("id", ids)
        .select_columns(["id", "text"])
    )
    images = (
        pairs.select_columns(["image"])
        .add_column("id", ids)
        .select_columns(["id", "image"])
    )
    queries, corpus = (text, images) if direction == "t2i" else (images, text)
    qrels = {pair_id: {pair_id: 1} for pair_id in ids}

    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=qrels,
                top_ranked=None,
            )
        }
    }
    task.data_loaded = True


class MarsVLPairsT2IRetrieval(AbsTaskRetrieval):
    k_values = (1, 3, 5, 10, 20, 100, 1000, _FROZEN_PAIRS)

    metadata = TaskMetadata(
        name="MarsVLPairsT2IRetrieval",
        description=_DESCRIPTION
        + "Given a scientific caption, retrieve its paired Mars image from the "
        "full frozen gallery.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="mrr_at_2247",
        date=("2026-02-15", "2026-02-15"),
        domains=["Academic", "Nature", "Scene", "Web"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="multiple",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the Mars image that matches this scientific description."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_mars_vl_pairs(self, "t2i", num_proc)


class MarsVLPairsI2TRetrieval(AbsTaskRetrieval):
    k_values = (1, 3, 5, 10, 20, 100, 1000, _FROZEN_PAIRS)

    metadata = TaskMetadata(
        name="MarsVLPairsI2TRetrieval",
        description=_DESCRIPTION
        + "Given a Mars image, retrieve its paired scientific caption from the "
        "full frozen gallery.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="i2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="mrr_at_2247",
        date=("2026-02-15", "2026-02-15"),
        domains=["Academic", "Nature", "Scene", "Web"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="multiple",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the scientific caption that describes this Mars image."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_mars_vl_pairs(self, "i2t", num_proc)
