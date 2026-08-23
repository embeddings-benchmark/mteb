from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SciRepEvalBiomimicryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SciRepEvalBiomimicryClassification",
        description=(
            "Biomimicry relevance classification. Given the title and abstract "
            "of a scientific paper, predict whether the paper is relevant to "
            "biomimicry research (binary classification). Published as a part of "
            "SciRepEval."
        ),
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "name": "biomimicry",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["evaluation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2023-12-06"),
        domains=["Academic", "Engineering", "Written"],
        task_subtypes=["Topic classification"],
        license="odc-by",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{singh-etal-2023-scirepeval,
  address = {Singapore},
  author = {Singh, Amanpreet  and
D{'}Arcy, Mike  and
Cohan, Arman  and
Downey, Doug  and
Feldman, Sergey},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2023.emnlp-main.338},
  month = dec,
  pages = {5548--5566},
  publisher = {Association for Computational Linguistics},
  title = {{SciRepEval: A Multi-Format Benchmark for Scientific Document Representations}},
  url = {https://aclanthology.org/2023.emnlp-main.338/},
  year = {2023},
}
""",
    )

    is_cross_validation: bool = True
    train_split: str = "evaluation"

    def dataset_transform(self, num_proc: int | None = None) -> None:
        # SciRepEval exposes only a single "evaluation" split for `biomimicry`.
        # We keep it named "evaluation" and let MTEB's cross-validation
        # path (is_cross_validation) run KFold over the whole split.
        from datasets import DatasetDict

        ds = self.dataset["evaluation"]
        ds = ds.map(
            lambda x: {"text": f"{x['title'] or ''}\n\n{x['abstract'] or ''}".strip()},
            num_proc=num_proc,
        )
        keep = {"text", "label"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds = ds.class_encode_column("label")
        self.dataset = DatasetDict({"evaluation": ds})
