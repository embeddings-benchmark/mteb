from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from ._video_pair_helpers import build_pair_dataset, generate_pairs


class VinogroundPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VinogroundPairClassification",
        description=(
            "Pair classification on the Vinoground dataset: determining "
            "whether two video clips depict the same action category. "
            "Vinoground contains temporal counterfactual pairs — videos "
            "that share the same objects but differ in action ordering "
            "(e.g. 'dog chases cat' vs 'cat chases dog'). Tests whether "
            "an embedding model captures temporal/compositional differences."
        ),
        reference="https://arxiv.org/abs/2410.02763",
        dataset={
            "path": "zachz/Vinoground-PC",
            "revision": "ffaf8401703625840f3bf3bddc2670f1ff9d17e8",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-10-01", "2024-10-31"),
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{zhang2024vinoground,
  author = {Zhang, Jianrui and Cai, Mu and Lee, Yong Jae},
  journal = {arXiv preprint arXiv:2410.02763},
  title = {Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos},
  year = {2024},
}
""",
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        import random

        rng = random.Random(42)
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]

            # Map major categories to integer labels
            majors = ds["major"]
            unique_majors = sorted({m for m in majors if m is not None})
            major_to_label = {m: i for i, m in enumerate(unique_majors)}
            class_labels = [major_to_label.get(m, len(unique_majors)) for m in majors]

            pairs = generate_pairs(class_labels, rng)
            self.dataset[split] = build_pair_dataset(ds, pairs)
