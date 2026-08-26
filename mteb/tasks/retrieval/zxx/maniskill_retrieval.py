from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_MANISKILL_BIBTEX = r"""
@article{tao2025maniskill3,
  author = {Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and others},
  journal = {Robotics: Science and Systems},
  title = {ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  year = {2025},
}
"""

_MANISKILL_DESCRIPTION_TAIL = (
    "Built from the official ManiSkill3 motion-planning demonstrations: 8 "
    "tabletop manipulation tasks (PickCube, StackCube, StackPyramid, "
    "PegInsertionSide, PlugCharger, PushCube, PullCubeTool, DrawTriangle), "
    "150 successful episodes each, replayed via environment states and "
    "rendered at 256x256. Goal-state images come from the base sensor "
    "camera while videos come from the human render camera, so queries and "
    "documents never share a viewpoint and exact frame matching cannot "
    "solve the task. Per task, 10 episodes are held out as queries and the "
    "other 140 form the corpus, so the query's own episode never appears in "
    "the corpus. A corpus item is relevant if and only if it comes from an "
    "episode of the same task as the query (140 relevant items per query); "
    "items from the other seven tasks are non-relevant."
)


class ManiSkillI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ManiSkillI2VRetrieval",
        description=(
            "Image-to-video retrieval over simulated robot manipulation: "
            "given a goal-state image (final frame, base camera) of a "
            "held-out episode, retrieve videos of the same task from "
            "another viewpoint. " + _MANISKILL_DESCRIPTION_TAIL
        ),
        reference="https://github.com/haosulab/ManiSkill",
        dataset={
            "path": "ZhixuLi/ManiSkill-I2V",
            "revision": "6d8be5a8f7b0b77fae5c46edda399bd7ef739f2f",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-06-30"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="rendered",
        is_beta=True,
        bibtex_citation=_MANISKILL_BIBTEX,
        prompt={
            "query": (
                "Retrieve the robot manipulation video showing the episode "
                "whose final goal state is depicted in the image, seen from "
                "a different viewpoint."
            )
        },
    )


class ManiSkillV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ManiSkillV2IRetrieval",
        description=(
            "Video-to-image retrieval over simulated robot manipulation: "
            "given a manipulation episode video, retrieve goal-state images "
            "(final frames of held-out episodes) of the same task, captured "
            "from another viewpoint. " + _MANISKILL_DESCRIPTION_TAIL
        ),
        reference="https://github.com/haosulab/ManiSkill",
        dataset={
            "path": "ZhixuLi/ManiSkill-V2I",
            "revision": "879e9072f1820c13e3deecf30eefc5b9fc45f593",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-06-30"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="rendered",
        is_beta=True,
        bibtex_citation=_MANISKILL_BIBTEX,
        prompt={
            "query": (
                "Retrieve the goal-state image showing the final state of "
                "the manipulation episode in the video, seen from a "
                "different viewpoint."
            )
        },
    )
