from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{hong2025worldsense,
  author = {Hong, Jack and Yan, Shilin and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal = {arXiv preprint arXiv:2502.04326},
  title = {Worldsense: Evaluating real-world omnimodal understanding for multimodal llms},
  year = {2025},
}
"""


class WorldSenseAudioVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WorldSenseAudioVideoZeroShot",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This zero-shot classification task predicts the domain category of a video clip Filtered the test split to short (<1min) clips (~1,047 examples).",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSenseAudioVideoZeroShot",
            "revision": "02540a42ac70c9bd11c12d9e730fad034006d3bf",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-02-06", "2026-03-01"),
        domains=["AudioScene", "Scene", "Web"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class WorldSenseVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WorldSenseVideoZeroShot",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This zero-shot classification task predicts the domain category of a video clip Filtered the test split to short (<1min) clips (~1,047 examples).",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSenseVideoZeroShot",
            "revision": "da6d5ae9b19bfa22e4ea0b45280385243cf51438",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-02-06", "2026-03-01"),
        domains=["AudioScene", "Scene", "Web"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
