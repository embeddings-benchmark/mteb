from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
            "revision": "5c011c8755517a299294686f3c87ad7ae4c93e4a",
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
