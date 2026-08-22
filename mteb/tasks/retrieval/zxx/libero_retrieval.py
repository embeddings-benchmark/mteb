from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LIBERO_BIBTEX = r"""
@inproceedings{liu2023libero,
  author = {Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  title = {{LIBERO}: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  year = {2023},
}
"""

_LIBERO_DESCRIPTION_TAIL = (
    "Built from the LIBERO lifelong robot learning benchmark "
    "(libero_spatial/object/goal/10 suites: 40 tabletop-manipulation tasks, "
    "1,693 human-teleoperated demonstrations rendered at 256x256, 10 fps). "
    "Per task, episodes are split into disjoint query and corpus pools, so a "
    "query's own source episode is never in the corpus and exact frame "
    "matching cannot solve the task. Relevance is task-level and "
    "multi-positive; distractors include same-scene episodes of other goals, "
    "so retrieval requires matching the achieved goal state rather than the "
    "scene."
)


class LIBEROI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LIBEROI2VRetrieval",
        description=(
            "Image-to-video retrieval over robot manipulation episodes: given "
            "a goal-state image (the final frame of a held-out episode), "
            "retrieve manipulation videos that accomplish the same task. "
            + _LIBERO_DESCRIPTION_TAIL
        ),
        reference="https://libero-project.github.io/",
        dataset={
            "path": "ZhixuLi/LIBERO-I2V",
            "revision": "1c9b26903c22ec4eeb1a8868073502d528498520",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2023-03-01", "2023-11-01"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="rendered",
        is_beta=True,
        bibtex_citation=_LIBERO_BIBTEX,
        prompt={
            "query": (
                "Retrieve robot manipulation videos that accomplish the task "
                "whose completed goal state is shown in the image."
            )
        },
    )


class LIBEROV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LIBEROV2IRetrieval",
        description=(
            "Video-to-image retrieval over robot manipulation episodes: given "
            "a manipulation video, retrieve goal-state images (final frames "
            "of held-out episodes) of the same task. " + _LIBERO_DESCRIPTION_TAIL
        ),
        reference="https://libero-project.github.io/",
        dataset={
            "path": "ZhixuLi/LIBERO-V2I",
            "revision": "5f27ced35197d6d8f7f7f55f49fbb3ff79751c08",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2023-03-01", "2023-11-01"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="rendered",
        is_beta=True,
        bibtex_citation=_LIBERO_BIBTEX,
        prompt={
            "query": (
                "Retrieve goal-state images showing the completed outcome of "
                "the task performed in the robot manipulation video."
            )
        },
    )
