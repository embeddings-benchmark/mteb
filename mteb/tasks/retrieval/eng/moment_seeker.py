from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "dukesun99/MomentSeeker-Full"
_DATASET_REVISION = "e1be36f6a27a0da618704860cb50517634c01f3a"
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
    "Full-video moment retrieval adapted from MomentSeeker, a long-video moment "
    "retrieval benchmark with videos averaging over 500 seconds (movies, "
    "cartoons, egocentric and open-domain footage). The corpus is the 271 "
    "complete source videos (re-encoded to 360p, audio kept); a video is "
    "relevant to a query iff it contains the annotated answer moment. "
)

# Moment level -> the MomentSeeker task types it groups (paper Sec. 3.3).
_LEVEL_TASKS = {
    "global": "Causal Reasoning, Spatial Relation",
    "event": "Description Location, Action Recognition, Anomaly Detection",
    "object": "Object Recognition, Object Location, Attribute Recognition, OCR",
}
# Per direction: query modality, TaskCategory code, and the query's extra
# reference modality (None for the plain text->video direction).
_DIRECTIONS = {
    "ti2v": (["image", "text", "video"], "it2v", "a reference image"),
    "tv2v": (["video", "text"], "vt2v", "a reference video clip"),
    "t2v": (["text", "video"], "t2v", None),
}


def _load_momentseeker(task: AbsTaskRetrieval, subset: str) -> None:
    """Load the shared full-video corpus plus a `{direction}-{level}` slice."""
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    corpus = load_dataset(path, "corpus", split="test", revision=revision)
    queries = load_dataset(path, f"{subset}-queries", split="test", revision=revision)
    qrels_ds = load_dataset(path, f"{subset}-qrels", split="test", revision=revision)
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


def _meta(name: str, direction: str, level: str) -> TaskMetadata:
    modalities, category, qref = _DIRECTIONS[direction]
    if qref is None:  # plain text -> video
        qdesc = "Queries are text questions; retrieve the full video that contains the answer moment. "
        qprompt = "Given the question, retrieve the video that answers it."
    else:
        qdesc = (
            f"Queries combine a text question with {qref}; retrieve the full video "
            "that contains the answer moment. "
        )
        qprompt = f"Given the question and {qref}, retrieve the video that answers it."
    return TaskMetadata(
        name=name,
        description=_DESCRIPTION
        + qdesc
        + f"This subtask covers only {level}-level moments ({_LEVEL_TASKS[level]}).",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category=category,
        modalities=modalities,
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": qprompt},
        is_beta=True,
    )


class MomentSeekerTI2VGlobalLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerTI2VGlobalLevelRetrieval", "ti2v", "global")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "ti2v-global")


class MomentSeekerTI2VEventLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerTI2VEventLevelRetrieval", "ti2v", "event")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "ti2v-event")


class MomentSeekerTI2VObjectLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerTI2VObjectLevelRetrieval", "ti2v", "object")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "ti2v-object")


class MomentSeekerTV2VGlobalLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerTV2VGlobalLevelRetrieval", "tv2v", "global")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "tv2v-global")


class MomentSeekerTV2VEventLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerTV2VEventLevelRetrieval", "tv2v", "event")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "tv2v-event")


class MomentSeekerTV2VObjectLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerTV2VObjectLevelRetrieval", "tv2v", "object")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "tv2v-object")


class MomentSeekerT2VGlobalLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerT2VGlobalLevelRetrieval", "t2v", "global")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "t2v-global")


class MomentSeekerT2VEventLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerT2VEventLevelRetrieval", "t2v", "event")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "t2v-event")


class MomentSeekerT2VObjectLevelRetrieval(AbsTaskRetrieval):
    metadata = _meta("MomentSeekerT2VObjectLevelRetrieval", "t2v", "object")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_momentseeker(self, "t2v-object")
