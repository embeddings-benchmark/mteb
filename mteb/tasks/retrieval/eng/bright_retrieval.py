from __future__ import annotations

import warnings
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

DOMAINS_LONG = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
]

DOMAINS = DOMAINS_LONG + [
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

DOMAINS_langs = {split: ["eng-Latn"] for split in DOMAINS}


def load_data(
    self: AbsTaskRetrieval, num_proc: int | None = None, **kwargs: Any
) -> None:
    if self.data_loaded:
        return

    warnings.warn(
        "This task contains wrong prompts in the metadata. "
        "Please use BRIGHT(v1.1) benchmark instead.",
        stacklevel=2,
        category=DeprecationWarning,
    )

    AbsTaskRetrieval.load_data(self, num_proc=num_proc, **kwargs)


class BrightRetrieval(AbsTaskRetrieval):
    is_long = False
    metadata = TaskMetadata(
        name="BrightRetrieval",
        dataset={
            "path": "mteb/BrightRetrieval",
            "revision": "50164c4651cc74ec3a6c8dc857821836bf2fe666",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval",
        type="Retrieval",
        category="t2t",
        eval_splits=["standard"],
        eval_langs=DOMAINS_langs,
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{su2024bright,
  archiveprefix = {arXiv},
  author = {Hongjin SU and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Liu Haisu and Quan Shi and Zachary S Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O Arik and Danqi Chen and Tao Yu},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.12883},
  title = {{BRIGHT}: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  url = {https://openreview.net/forum?id=ykuc5q381b},
  year = {2025},
}
""",
        superseded_by="BrightBiologyRetrieval",
    )
    load_data = load_data


long_metadata = BrightRetrieval.metadata.model_copy()
long_metadata.eval_splits = ["long"]
long_metadata.description = "Bright retrieval dataset with long documents."
long_metadata.name = "BrightLongRetrieval"
long_metadata.dataset = {
    "path": "mteb/BrightLongRetrieval",
    "revision": "d5fd7da60ea262b3d4c96633aae4f3194a2945b8",
}

dom_langs_long = {split: ["eng-Latn"] for split in DOMAINS_LONG}
long_metadata.eval_langs = dom_langs_long


class BrightLongRetrieval(AbsTaskRetrieval):
    is_long = True
    metadata = long_metadata

    load_data = load_data
