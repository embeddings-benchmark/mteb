from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "dukesun99/MomentSeeker-CVR"
_DATASET_REVISION = "e87fc71839cd14dfd09aa83b378cd51074318b3c"
_REFERENCE = "https://arxiv.org/abs/2502.12558"
_BIBTEX = r"""
@misc{yuan2025momentseeker,
  archiveprefix = {arXiv},
  author = {Huaying Yuan and Jian Ni and Zheng Liu and Yueze Wang and Junjie Zhou and Zhengyang Liang and Bo Zhao and Zhao Cao and Zhicheng Dou and Ji-Rong Wen},
  eprint = {2502.12558},
  primaryclass = {cs.CV},
  title = {MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval},
  url = {https://arxiv.org/abs/2502.12558},
  year = {2025},
}
"""
_DESCRIPTION = (
    "Composed video retrieval adapted from MomentSeeker, a long-video moment "
    "retrieval benchmark with videos averaging over 500 seconds (movies, "
    "cartoons, egocentric and open-domain footage). Source videos are split into "
    "30-second non-overlapping 360p chunks forming the corpus; a chunk is "
    "relevant to a query iff it overlaps an annotated answer interval by at "
    "least half of the shorter of chunk and interval. "
)


def _load_momentseeker(task: AbsTaskRetrieval, direction: str) -> None:
    """Load the shared chunk corpus plus the direction's queries and qrels."""
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    corpus = load_dataset(path, "corpus", split="test", revision=revision)
    queries = load_dataset(
        path, f"{direction}-queries", split="test", revision=revision
    )
    qrels_ds = load_dataset(path, f"{direction}-qrels", split="test", revision=revision)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                corpus=corpus, queries=queries, relevant_docs=qrels, top_ranked=None
            )
        }
    }
    task.data_loaded = True


class MomentSeekerTI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VRetrieval",
        description=_DESCRIPTION
        + "Queries combine a text question with a reference image, e.g. asking "
        "what happened around the pictured moment.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference image, retrieve the video segment that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "ti2v")


class MomentSeekerTV2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VRetrieval",
        description=_DESCRIPTION
        + "Queries combine a text question with a reference video clip, e.g. "
        "asking what happened after the shown event.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference clip, retrieve the video segment that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "tv2v")
