from __future__ import annotations

from collections import defaultdict
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET = {
    "path": "mteb/IAW",
    "revision": "__REVISION__",
}
_REFERENCE = "https://arxiv.org/abs/2303.13800"
_BIBTEX = r"""
@inproceedings{zhang2023aligning,
  author = {Zhang, Jiahao and Cherian, Anoop and Liu, Yanbin and Ben-Shabat, Yizhak and Rodriguez, Cristian and Gould, Stephen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Aligning Step-by-Step Instructional Diagrams to Video Demonstrations},
  year = {2023},
}
"""
_DESCRIPTION = (
    "IKEA Assembly in the Wild (IAW): YouTube videos of people assembling IKEA furniture, "
    "matched to the step diagrams of the furniture's assembly manual, under the paper's "
    "test protocol. Videos are cut into the paper's 10 s test clips (silent), and step "
    "diagrams are cropped from the manual pages. Matching happens within one furniture "
    "item: a clip ranks the step diagrams of its own manual, and a diagram ranks the "
    "clips of its own video. Items whose manual or video is no longer available are "
    "left out. "
)

_COMMON: dict[str, Any] = dict(
    reference=_REFERENCE,
    dataset=_DATASET,
    type="Any2AnyReranking",
    eval_splits=["test"],
    eval_langs=["zxx-Zxxx"],
    main_score="hit_rate_at_1",
    date=("2014-11-10", "2022-04-22"),  # YouTube upload dates of the videos
    domains=["Instructional", "Web"],
    task_subtypes=["Cross-Modal Retrieval"],
    license="mit",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    is_beta=True,
    bibtex_citation=_BIBTEX,
)


class IAWV2IReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IAWV2IReranking",
        description=_DESCRIPTION
        + "This task is video-to-diagram retrieval: each clip has one relevant step "
        "diagram, and hit_rate_at_1 equals the paper's top-1 accuracy.",
        category="v2i",
        modalities=["video", "image"],
        prompt={
            "query": "Retrieve the assembly manual step diagram for the step carried out in this video clip."
        },
        **_COMMON,
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for split_data in subset.values():
                split_data["queries"] = split_data["queries"].select_columns(
                    ["id", "video"]
                )
                split_data["corpus"] = split_data["corpus"].select_columns(
                    ["id", "image"]
                )


class IAWI2VReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IAWI2VReranking",
        description=_DESCRIPTION
        + "This task is diagram-to-video retrieval: each step diagram that is carried "
        "out in the video is a query, and every clip of that step is relevant.",
        category="i2v",
        modalities=["image", "video"],
        prompt={
            "query": "Retrieve the video clip in which this assembly manual step is carried out."
        },
        **_COMMON,
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Reuse the video-to-diagram data with roles swapped: diagrams become queries."""
        for subset in self.dataset.values():
            for split_data in subset.values():
                relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
                for clip_id, docs in split_data["relevant_docs"].items():
                    for step_id, score in docs.items():
                        relevant_docs[step_id][clip_id] = score
                top_ranked: dict[str, list[str]] = defaultdict(list)
                for clip_id, step_ids in split_data["top_ranked"].items():
                    for step_id in step_ids:
                        if step_id in relevant_docs:
                            top_ranked[step_id].append(clip_id)
                # steps that are never carried out in the video have no relevant clip
                diagrams = split_data["corpus"]
                keep = [i for i, d in enumerate(diagrams["id"]) if d in relevant_docs]
                split_data["queries"], split_data["corpus"] = (
                    diagrams.select(keep).select_columns(["id", "image"]),
                    split_data["queries"].select_columns(["id", "video"]),
                )
                split_data["relevant_docs"] = dict(relevant_docs)
                split_data["top_ranked"] = dict(top_ranked)
